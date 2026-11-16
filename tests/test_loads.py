"""Rows into the cloud by the platform's load: the file each load uploads,
the table its first load makes, what the platform chose being read back,
rows cut into passages, and a table a load is writing."""

from __future__ import annotations

import io
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hermes_infino import backend, config, loads, mirror, recall, tables
from hermes_infino.client import InfinoClient, InfinoError
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore
from hermes_infino.mirror import Mirror
from hermes_infino.recall import Recaller, Source

from .helpers import TEAM_DB, sign_in

# Long enough for the stand-in platform to cut it into several passages.
LONG_TEXT = "the cache warm-up test flakes when the shard map is rebuilt under a reader " * 20
PASSAGES = 3


@pytest.fixture
def m(home, platform, tmp_path) -> Mirror:
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    out = Mirror(sign_in(tmp_path, platform), team, personal)
    out.ensure_cloud_tables()
    return out


def uploaded_tables(platform) -> list[pa.Table]:
    return [pq.read_table(io.BytesIO(data)) for data in platform.uploads.values()]


def test_a_sync_loads_rows_as_a_parquet_file_with_the_plugins_columns(m, platform):
    m.local.append(tables.CONVERSATIONS, [{"session_id": "s", "role": "assistant", "content": "looked it up",
                                           "tool_args": json.dumps({"path": "src/cache.rs"})}])
    m.sync(wait=True)
    [file] = uploaded_tables(platform)
    names = [f["name"] for f in tables.CONVERSATIONS_SCHEMA]
    assert file.column_names == names
    # Typed by the plugin, not by what one batch holds: an empty column keeps its type.
    assert file.schema.field("tool_result").type == pa.large_string()
    assert file.schema.field("seq").type == pa.int64()
    [load] = platform.loads
    assert load["table"] == tables.CONVERSATIONS and load["source"]["kind"] == "prefix"
    # JSON in a text column stays text; nothing is declared about indexes or embeddings.
    assert "tool_args" in load["text_columns"] and not {"fts", "embed"} & set(load)
    [row] = platform.rows(TEAM_DB, tables.CONVERSATIONS)
    assert json.loads(row["tool_args"]) == {"path": "src/cache.rs"}


def test_the_first_load_makes_the_table_and_later_ones_append(m, platform):
    assert tables.MEMORY not in platform.databases[TEAM_DB]
    m.local.append(tables.MEMORY, [{"content": "one"}])
    m.sync(wait=True)
    m.local.append(tables.MEMORY, [{"content": "two"}])
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["one", "two"]
    assert [load["if_exists"] for load in platform.loads] == ["append", "append"]


def test_a_sync_sends_each_tables_rows_in_one_load(m, platform):
    m.local.append(tables.CONVERSATIONS, [{"session_id": "s", "seq": i, "content": f"m{i}"} for i in range(250)])
    m.sync(wait=True)
    assert [load["table"] for load in platform.loads] == [tables.CONVERSATIONS]
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 250


def test_the_upload_session_never_gets_the_key(m, platform):
    m.local.append(tables.MEMORY, [{"content": "one"}])
    m.sync(wait=True)
    puts = [path for path, _ in platform.requests if path.startswith("/_storage/")]
    assert len(puts) == 1  # the fake refuses a put carrying a key


def test_a_failed_load_leaves_the_rows_here_and_the_mark_where_it_was(m, platform):
    m.local.append(tables.MEMORY, [{"content": "one"}])
    platform.load_states = ["failed"]
    with pytest.raises(InfinoError) as raised:
        m.sync(wait=True)
    assert raised.value.status == loads.REFUSED_LOAD_STATUS
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["one"]


def test_a_load_that_did_not_finish_is_checked_with_the_cloud_before_its_rows_go_again(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "r1", "content": "one"}])
    platform.load_states = ["stopped"]  # its rows went in, but the job was not seen to end loaded
    with pytest.raises(InfinoError) as raised:
        m.sync(wait=True)
    assert raised.value.status == loads.UNFINISHED_LOAD_STATUS
    m.sync(wait=True)
    assert [r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["r1"]  # not sent twice


def test_a_sync_waits_for_a_load_still_writing_the_table_before_it_checks(m, platform, monkeypatch):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "r1", "content": "one"}])
    platform.load_states = ["stopped"]
    with pytest.raises(InfinoError):
        m.sync(wait=True)
    looks = []

    def load_job(self, database, table):
        looks.append(table)
        if len(looks) == 1:
            return {"state": "running"}  # still writing: waited for
        return real(self, database, table)

    real = InfinoClient.load_job
    monkeypatch.setattr(InfinoClient, "load_job", load_job)
    monkeypatch.setattr(loads, "POLL_S", 0.0)
    m.sync(wait=True)
    assert looks[:2] == [tables.MEMORY, tables.MEMORY]
    assert [r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["r1"]


def test_a_forget_waits_while_a_load_is_writing_its_table(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "z", "content": "gone"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["z"])
    platform.loading = {(TEAM_DB, tables.MEMORY)}  # another machine's load
    with pytest.raises(InfinoError) as raised:
        m.sync(wait=True)
    assert raised.value.status == mirror.TABLE_BUSY_STATUS
    assert [r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["z"]
    assert "z" in (m.dir / mirror.FORGET_FILE).read_text()
    platform.loading = set()
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    assert (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_forget_removes_every_passage_of_a_row(m, platform):
    platform.passages = PASSAGES
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "z", "content": LONG_TEXT}])
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == PASSAGES
    m.forget(tables.MEMORY, TEAM_DB, ["z"])
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_rows_cut_into_passages_are_not_taken_for_missing_or_sent_twice(m, platform):
    platform.passages = PASSAGES
    m.local.append(tables.MEMORY, [{tables.ROW_ID: f"r{i}", "content": LONG_TEXT} for i in range(2)])
    m.sync(wait=True)
    m.ensure_cloud_tables()  # the lost-table check counts rows by id
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 2 * PASSAGES


def test_a_dropped_table_gets_this_machines_rows_again(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "r1", "content": "one"}])
    m.sync(wait=True)
    del platform.databases[TEAM_DB][tables.MEMORY]
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert [r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["r1"]


def test_a_row_read_back_whole_joins_its_passages(m, platform):
    platform.passages = PASSAGES
    m.local.append(tables.HISTORY, [{tables.ROW_ID: "h1", "content": LONG_TEXT, tables.SUPERSEDED: 1,
                                     tables.SUPERSEDED_BY: "", tables.PRIOR_ROW_ID: "x"}])
    m.sync(wait=True)
    [row] = m.matching_rows(tables.HISTORY, TEAM_DB, f"{tables.ROW_ID} = 'h1'")
    assert row["content"] == LONG_TEXT


def test_recall_searches_the_columns_the_platform_chose_and_takes_one_hit_per_row(m, platform, tmp_path):
    platform.passages = PASSAGES
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "r1", "content": LONG_TEXT}])
    m.sync(wait=True)
    cloud = InfinoClient("k", platform.url, retries=0)
    rows = Recaller([Source(recall.TEAM, m.local, TEAM_DB, lambda: (2 ** 62, ""))], cloud).search_cloud(
        TEAM_DB, "shard map rebuilt")
    assert [r[tables.ROW_ID] for r in rows] == ["r1"]
    searched = [json.loads(params) for path, params in platform.requests if path == f"/v1/hybrid_search/{TEAM_DB}"]
    assert searched  # hybrid, over the embedding the platform made


def test_recall_asks_the_platform_once_what_a_loaded_table_holds(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "r1", "content": "deploy freeze"}])
    m.sync(wait=True)
    cloud = InfinoClient("k", platform.url, retries=0)
    r = Recaller([Source(recall.TEAM, m.local, TEAM_DB, lambda: (2 ** 62, ""))], cloud)
    before = sum(1 for path, _ in platform.requests if path == f"/v1/hydrate/{TEAM_DB}")
    for _ in range(3):
        r.search_cloud(TEAM_DB, "deploy freeze")
    after = sum(1 for path, _ in platform.requests if path == f"/v1/hydrate/{TEAM_DB}")
    assert after - before == 1


def test_roles_read_from_the_load_record():
    class Client:
        def load_job(self, database, table):
            return {"state": "succeeded", "schema": {"fts": [{"column": "content"}, {"column": "chat"}],
                                                     "embed": {"column": "emb", "source": ["content"]}}}

    roles = loads.roles(Client(), TEAM_DB, tables.MEMORY)
    assert roles == loads.Roles(("content", "chat"), "emb")
    assert roles.text_column("content") == "content" and roles.text_column("body") == "content"


def test_one_hit_per_row_keeps_the_first_and_rows_without_an_id():
    rows = [{"row_id": "a", "chunk": 2}, {"row_id": "b"}, {"row_id": "a", "chunk": 0}, {"x": 1}, {"x": 1}]
    assert loads.one_per_row(rows) == [{"row_id": "a", "chunk": 2}, {"row_id": "b"}, {"x": 1}, {"x": 1}]


def test_background_syncs_start_at_least_the_interval_apart(home, platform, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SYNC_MIN_INTERVAL_S", 60.0)
    b = backend.for_settings(sign_in(tmp_path, platform))
    b._sync_started = time.monotonic()
    b.schedule_sync()
    try:
        assert b._timer.interval > 50
    finally:
        b._timer.cancel()

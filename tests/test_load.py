"""Many rows at once go to the cloud by one platform load job, as the Infino
client's index sends a large write; a few go by appends. A load that fails
leaves its rows to the next sync's appends, none sent twice, and a platform
that runs no load jobs takes appends."""

from __future__ import annotations

import pytest

from hermes_infino import chunks, client, config, mirror, tables
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore
from hermes_infino.mirror import Mirror

from .helpers import TEAM_DB, sign_in
from .test_chunks import fake_chunker, long_text

# Rows from which a sync loads, here, and a number of rows past it.
LOAD_FROM = 10
MANY = 12


@pytest.fixture
def m(home, platform, tmp_path, monkeypatch) -> Mirror:
    monkeypatch.setattr(mirror, "LOAD_JOB_MIN_ROWS", LOAD_FROM)
    monkeypatch.setattr(client, "LOAD_POLL_S", 0.0)
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    out = Mirror(sign_in(tmp_path, platform), team, personal)
    out.chunker = fake_chunker
    out.ensure_cloud_tables()
    return out


def messages(n: int) -> list[dict]:
    return [{tables.ROW_ID: f"t{i:02d}", "session_id": "s", "seq": i, "role": "user", "content": f"message {i}"}
            for i in range(n)]


def cloud_ids(platform) -> list[str]:
    return sorted(r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS))


def appends(platform) -> list[str]:
    return [path for path, _ in platform.requests if path.startswith("/v1/append/")]


def test_many_rows_go_by_one_load(m, platform):
    m.local.append(tables.CONVERSATIONS, messages(MANY))
    m.sync(wait=True)
    assert platform.loads == [(TEAM_DB, tables.CONVERSATIONS, MANY)]
    assert cloud_ids(platform) == [f"t{i:02d}" for i in range(MANY)]
    assert appends(platform) == []


def test_a_few_rows_go_by_appends(m, platform):
    m.local.append(tables.CONVERSATIONS, messages(LOAD_FROM - 1))
    m.sync(wait=True)
    assert platform.loads == []
    assert len(cloud_ids(platform)) == LOAD_FROM - 1


def test_a_long_message_goes_by_load_as_its_chunks(m, platform):
    rows = messages(MANY) + [{tables.ROW_ID: "long", "session_id": "s", "seq": 99, "role": "tool",
                              "tool_name": "read_file", "tool_result": long_text()}]
    m.local.append(tables.CONVERSATIONS, rows)
    m.sync(wait=True)
    assert platform.loads == [(TEAM_DB, tables.CONVERSATIONS, MANY + 3)]
    long = [r for r in platform.rows(TEAM_DB, tables.CONVERSATIONS) if r[tables.ROW_ID] == "long"]
    assert [r[chunks.CHUNK_COLUMN] for r in long] == [0, 1, 2]
    assert "\n".join(r["tool_result"] for r in long) == long_text()


def test_a_failed_load_leaves_its_rows_to_the_next_syncs_appends(m, platform):
    platform.failed_loads = 1
    m.local.append(tables.CONVERSATIONS, messages(MANY))
    with pytest.raises(client.InfinoError, match="ended failed"):
        m.sync(wait=True)
    assert cloud_ids(platform) == []
    m.sync(wait=True)
    assert platform.loads == []
    assert cloud_ids(platform) == [f"t{i:02d}" for i in range(MANY)]


def test_a_load_that_landed_unanswered_is_not_sent_again(m, platform, monkeypatch):
    real = m.client.load

    def lands_then_times_out(database, table, rows, **kw):
        real(database, table, rows, **kw)
        raise TimeoutError("the load had not ended")

    monkeypatch.setattr(m.client, "load", lands_then_times_out)
    m.local.append(tables.CONVERSATIONS, messages(MANY))
    with pytest.raises(TimeoutError):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "load", real)
    m.sync(wait=True)
    assert cloud_ids(platform) == [f"t{i:02d}" for i in range(MANY)]
    assert appends(platform) == []


def test_a_platform_that_runs_no_load_jobs_takes_appends(m, platform):
    platform.no_loads = True
    m.local.append(tables.CONVERSATIONS, messages(MANY))
    m.sync(wait=True)
    assert cloud_ids(platform) == [f"t{i:02d}" for i in range(MANY)]
    assert m.loads is False


def test_a_load_waits_past_an_earlier_loads_record(m, platform):
    platform.stale_load_reads = 2
    m.client.load(TEAM_DB, tables.CONVERSATIONS, messages(2))
    assert platform.stale_load_reads == 0
    assert platform.loads == [(TEAM_DB, tables.CONVERSATIONS, 2)]

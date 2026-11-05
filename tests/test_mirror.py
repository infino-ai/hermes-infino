"""Catching the cloud up: batches, marks, the settle window, the lock, and
the forget queue."""

from __future__ import annotations

import fcntl
import json

import pyarrow as pa
import pytest

from hermes_infino import config, mirror, tables
from hermes_infino.client import InfinoError
from hermes_infino.local import LATE_COMMITS_FILE, PERSONAL_DB_DIR, LocalStore, arrow_schema
from hermes_infino.mirror import Mirror

from .helpers import PERSONAL_DB, TEAM_DB, requests_to, sign_in

# A push batch small enough to see several batches in one sync.
SMALL_BATCH = 3
ROWS = 7
# A settle window no row in a test is old enough for.
LONG_SETTLE_MS = 3_600_000
ANOTHER_TEAM_DB = "team-other"


def make_mirror(tmp_path, settings) -> Mirror:
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    m = Mirror(settings, team, personal)
    m.ensure_cloud_tables()
    return m


@pytest.fixture
def m(home, platform, tmp_path) -> Mirror:
    return make_mirror(tmp_path, sign_in(tmp_path, platform))


def conversation_rows(n: int) -> list[dict]:
    return [{"session_id": "s", "seq": i, "role": "user", "content": f"m{i}"} for i in range(n)]


def test_the_sources_are_the_teams_tables(m):
    assert [(s.mark, s.table, s.database) for s in m.sources()] == [(f"{TEAM_DB}/{t}", t, TEAM_DB)
                                                                    for t in tables.ALL]


def test_a_personal_database_adds_the_personal_sources(home, platform, tmp_path):
    m = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    sources = [(s.mark, s.store, s.table, s.database) for s in m.sources()]
    for table in tables.PERSONAL:
        assert (f"{PERSONAL_DB}/personal_{table}", m.personal, table, PERSONAL_DB) in sources
    assert f"{PERSONAL_DB}/{mirror.PERSONAL_MEMORY}" in [s[0] for s in sources]  # the mark kept since before
    # The person's database holds their memory and the conversations and traces kept personal.
    assert set(platform.databases[PERSONAL_DB]) == set(tables.PERSONAL)
    # Nothing in the person's own tables has a source that goes to the team's database.
    assert all(s.database == PERSONAL_DB for s in m.sources() if s.store is m.personal)


def test_personal_transcripts_go_to_the_personal_database_only(home, platform, tmp_path):
    m = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    m.personal.append(tables.CONVERSATIONS, [{"session_id": "mine", "seq": 0, "role": "user", "content": "private"}])
    m.personal.append(tables.TRACES, [{"session_id": "mine", "kind": "model", "result": "private"}])
    m.local.append(tables.CONVERSATIONS, conversation_rows(1))
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)] == ["m0"]
    assert platform.rows(TEAM_DB, tables.TRACES) == []
    assert [r["content"] for r in platform.rows(PERSONAL_DB, tables.CONVERSATIONS)] == ["private"]
    assert [r["result"] for r in platform.rows(PERSONAL_DB, tables.TRACES)] == ["private"]


def test_without_a_personal_database_personal_transcripts_go_nowhere(m, platform):
    m.personal.append(tables.TRACES, [{"session_id": "mine", "kind": "model", "result": "private"}])
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.TRACES) == [] and set(platform.databases) == {TEAM_DB}


def test_rows_go_in_batches_and_the_mark_moves_with_each(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_BATCH", SMALL_BATCH)
    m.local.append(tables.CONVERSATIONS, conversation_rows(ROWS))
    seen = []
    pushed = m.sync(wait=True, progress=lambda mark, n: seen.append((mark, n)))
    mark = m.mark_of(tables.CONVERSATIONS)
    assert pushed[mark] == ROWS
    assert requests_to(platform, "append") == 3  # 3 + 3 + 1
    assert seen == [(mark, 3), (mark, 6), (mark, 7)]
    last = max(m.local.query_sql("SELECT stored_ms, row_id FROM conversations"),
               key=lambda r: (r[tables.STORED], r[tables.ROW_ID]))
    assert m.marks()[mark] == (last[tables.STORED], last[tables.ROW_ID])


def test_a_second_sync_pushes_nothing(m, platform):
    m.local.append(tables.CONVERSATIONS, conversation_rows(2))
    m.sync(wait=True)
    assert sum(m.sync(wait=True).values()) == 0
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 2


def test_rows_inside_the_settle_window_wait(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "SETTLE_MS", LONG_SETTLE_MS)
    m.local.append(tables.MEMORY, [{"content": "just written"}])
    assert sum(m.sync(wait=True).values()) == 0
    monkeypatch.setattr(mirror, "SETTLE_MS", 0)
    assert m.sync(wait=True)[m.mark_of(tables.MEMORY)] == 1


def test_a_sync_skips_while_another_holds_the_lock(m, platform):
    m.local.append(tables.MEMORY, [{"content": "x"}])
    with open(config.state_dir(m.settings.home) / mirror.LOCK_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        assert m.sync() == {}
        fcntl.flock(f, fcntl.LOCK_UN)
    assert m.sync()[m.mark_of(tables.MEMORY)] == 1


def test_an_unreadable_state_file_starts_over(m, platform):
    (m.dir / mirror.STATE_FILE).write_text("not json")
    assert m.marks() == {}
    assert m.last_error() == ""


def test_marks_an_older_version_kept_by_table_carry_over(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "old", "content": "pushed by the older version"}])
    [row] = m.local.query_sql("SELECT stored_ms, row_id FROM memory")
    (m.dir / mirror.STATE_FILE).write_text(json.dumps(
        {"marks": {tables.MEMORY: [row[tables.STORED], "old"], mirror.PERSONAL_MEMORY: [1, "x"]}}))
    assert m.marks() == {m.mark_of(tables.MEMORY): (row[tables.STORED], "old")}  # no personal database here
    assert m.sync(wait=True)[m.mark_of(tables.MEMORY)] == 0
    assert set(json.loads((m.dir / mirror.STATE_FILE).read_text())["marks"]) == {m.mark_of(tables.MEMORY)}


def test_a_big_batch_is_cut_to_fit_one_append(m, platform, monkeypatch):
    # Each row's JSON is a little over 1,000 bytes: two fit, three do not.
    monkeypatch.setattr(mirror, "PUSH_MAX_BYTES", 3_000)
    m.local.append(tables.CONVERSATIONS, [{"seq": i, "content": "x" * 1_000} for i in range(5)])
    m.sync(wait=True)
    assert requests_to(platform, "append") == 3  # 2 + 2 + 1
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 5


def test_one_row_larger_than_the_cap_still_goes(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_MAX_BYTES", 10)
    m.local.append(tables.MEMORY, [{"content": "longer than ten bytes"}])
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


def test_an_append_that_reached_the_cloud_before_failing_is_not_sent_again(m, platform, monkeypatch):
    m.local.append(tables.CONVERSATIONS, conversation_rows(3))
    real = m.client.append

    def times_out_after_storing(db, table, rows):
        real(db, table, rows)
        raise TimeoutError("the answer never came")

    monkeypatch.setattr(m.client, "append", times_out_after_storing)
    with pytest.raises(TimeoutError):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "append", real)
    m.local.append(tables.CONVERSATIONS, conversation_rows(1))
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)) == ["m0", "m0", "m1", "m2"]
    assert len({r["row_id"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)}) == 4


def test_a_row_the_cloud_refuses_is_set_aside_and_the_rest_go(m, platform):
    m.local.append(tables.CONVERSATIONS, [{"seq": i, "content": "REFUSE-ME" if i == 2 else f"m{i}"} for i in range(5)])
    platform.refused_marker = "REFUSE-ME"
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)) == ["m0", "m1", "m3", "m4"]
    assert m.refused() == 1 and m.last_error() == ""
    m.local.append(tables.CONVERSATIONS, [{"seq": 9, "content": "after it"}])
    m.sync(wait=True)  # never stuck on the refused row
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 5


def test_a_row_too_large_alone_is_set_aside(m, platform):
    m.local.append(tables.MEMORY, [{"content": "TOO-LARGE"}])
    platform.refused_marker, platform.refused_status = "TOO-LARGE", 413
    m.sync(wait=True)
    assert m.refused() == 1 and platform.rows(TEAM_DB, tables.MEMORY) == []


def test_a_plain_400_on_every_row_is_the_tables_trouble_not_a_rows(m, platform):
    m.local.append(tables.CONVERSATIONS, [{"seq": i, "content": f"BAD {i}"} for i in range(3)])
    platform.refused_marker = "BAD"
    with pytest.raises(InfinoError):
        m.sync(wait=True)
    assert m.refused() == 0  # nothing set aside: the next sync tries them again
    platform.refused_marker = None
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 3


def test_a_failing_source_holds_up_neither_the_others_nor_the_forgets(m, platform, monkeypatch):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "f", "content": "to forget"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["f"])
    m.local.append(tables.MEMORY, [{"content": "a new memory"}])
    m.local.append(tables.CONVERSATIONS, conversation_rows(2))
    real = m.client.append

    def conversations_fail(db, table, rows):
        if table == tables.CONVERSATIONS:
            raise RuntimeError("conversations are down")
        real(db, table, rows)

    monkeypatch.setattr(m.client, "append", conversations_fail)
    with pytest.raises(RuntimeError):
        m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["a new memory"]  # forget applied too
    assert platform.rows(TEAM_DB, tables.CONVERSATIONS) == []


def test_a_process_killed_mid_append_does_not_send_the_batch_twice(m, platform, monkeypatch):
    m.local.append(tables.CONVERSATIONS, conversation_rows(3))
    real = m.client.append

    def killed_after_the_cloud_stored_it(db, table, rows):
        real(db, table, rows)
        raise KeyboardInterrupt  # not an Exception: nothing records an error

    monkeypatch.setattr(m.client, "append", killed_after_the_cloud_stored_it)
    with pytest.raises(KeyboardInterrupt):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "append", real)
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 3


def test_a_slow_local_commit_is_pushed_after_all(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "a", "content": "pushed"}])
    m.sync(wait=True)
    # A row committed after the mark had passed its stamp: the sync never saw it.
    [stored] = {r[tables.STORED] for r in m.local.query_sql("SELECT stored_ms FROM memory")}
    late = {tables.ROW_ID: "0-late", "content": "committed slowly", tables.STORED: stored}
    m.local._connect().open_table(tables.MEMORY).append(
        pa.Table.from_pylist([{f.name: late.get(f.name) for f in arrow_schema(tables.MEMORY)}],
                             schema=arrow_schema(tables.MEMORY)))
    assert sum(m.sync(wait=True).values()) == 0
    with open(m.dir / LATE_COMMITS_FILE, "a") as f:  # what local.py notes for a slow commit
        f.write(json.dumps({"store": m.local.path.name, "table": tables.MEMORY, "stored_ms": stored}) + "\n")
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == ["committed slowly", "pushed"]


def test_a_cloud_slow_to_show_its_newest_row_is_not_a_lost_table(m, platform):
    m.local.append(tables.MEMORY, [{"content": "old"}])
    m.sync(wait=True)
    m.local.append(tables.MEMORY, [{"content": "new"}])
    m.sync(wait=True)
    platform.databases[TEAM_DB][tables.MEMORY] = [r for r in platform.rows(TEAM_DB, tables.MEMORY)
                                                  if r["content"] == "old"]  # the newest not visible yet
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == ["old"]  # nothing re-sent


def test_a_mark_whose_row_was_forgotten_is_kept(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "a", "content": "kept"}, {tables.ROW_ID: "z", "content": "gone"}])
    m.sync(wait=True)
    gone = m.local.delete(tables.MEMORY, "row_id = 'z'")
    m.forget(tables.MEMORY, TEAM_DB, [r[tables.ROW_ID] for r in gone])
    m.sync(wait=True)
    m.ensure_cloud_tables()  # the mark's row is gone from both: not a lost table
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept"]


def test_a_table_another_machine_made_again_gets_this_machines_rows(m, platform):
    m.local.append(tables.MEMORY, [{"content": "this machine's"}])
    m.sync(wait=True)
    platform.databases[TEAM_DB][tables.MEMORY] = [{"row_id": "theirs", "content": "another machine's"}]
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == ["another machine's",
                                                                                  "this machine's"]


def test_the_state_file_is_replaced_whole(m, platform):
    m.local.append(tables.MEMORY, [{"content": "x"}])
    m.sync(wait=True)
    state = json.loads((m.dir / mirror.STATE_FILE).read_text())
    assert set(state) == {"marks", "last_error", "in_flight", "recheck", "at_ms"}
    assert state["in_flight"] == [] and state["last_error"] == ""
    assert not (m.dir / "mirror.tmp").exists()


def test_a_failed_batch_keeps_the_ones_before_it(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_BATCH", SMALL_BATCH)
    m.local.append(tables.CONVERSATIONS, conversation_rows(ROWS))
    calls = []
    real = m.client.append

    def second_fails(db, table, rows):
        calls.append(len(rows))
        if len(calls) == 2:
            raise RuntimeError("network")
        real(db, table, rows)

    monkeypatch.setattr(m.client, "append", second_fails)
    with pytest.raises(RuntimeError):
        m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == SMALL_BATCH
    assert "network" in m.last_error()
    m.sync(wait=True)
    assert len({r["row_id"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)}) == ROWS


# -- forgets -------------------------------------------------------------------

def test_a_forget_with_no_rows_queues_nothing(m):
    m.forget(tables.MEMORY, TEAM_DB, [])
    assert not (m.dir / mirror.FORGET_FILE).exists()


def test_forgets_are_applied_in_batches_and_the_queue_empties(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "FORGET_BATCH", 2)
    m.local.append(tables.MEMORY, [{tables.ROW_ID: str(i), "content": f"m{i}"} for i in range(5)])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["0", "1", "2"])
    m.forget(tables.MEMORY, TEAM_DB, ["3"])
    m.sync(wait=True)
    assert [r["row_id"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["4"]
    assert requests_to(platform, "delete") == 3  # [0, 1], [2], [3]
    assert (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_failed_forget_stays_queued(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "m"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["1"])
    platform.failed_deletes = 1
    with pytest.raises(Exception):
        m.sync(wait=True)
    assert "1" in (m.dir / mirror.FORGET_FILE).read_text()
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_a_forget_goes_to_the_database_it_names(home, platform, tmp_path):
    m = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    m.personal.append(tables.MEMORY, [{tables.ROW_ID: "p", "content": "mine", "scope": "personal"}])
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "p", "content": "team row with the same id"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, PERSONAL_DB, ["p"])
    m.sync(wait=True)
    assert platform.rows(PERSONAL_DB, tables.MEMORY) == []
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


# -- where the marks point ---------------------------------------------------------

def test_a_new_team_database_gets_every_row(m, platform, tmp_path):
    m.local.append(tables.MEMORY, [{"content": "kept before the move"}])
    m.sync(wait=True)
    moved = make_mirror(tmp_path, sign_in(tmp_path, platform, team=ANOTHER_TEAM_DB))
    moved.sync(wait=True)
    assert [r["content"] for r in platform.rows(ANOTHER_TEAM_DB, tables.MEMORY)] == ["kept before the move"]
    # Each database keeps its own progress: moving back sends nothing twice.
    back = make_mirror(tmp_path, sign_in(tmp_path, platform))
    back.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


def test_a_dropped_and_recreated_cloud_table_is_filled_again(m, platform, tmp_path):
    m.local.append(tables.MEMORY, [{"content": "kept on this machine"}])
    m.sync(wait=True)
    del platform.databases[TEAM_DB][tables.MEMORY]
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept on this machine"]

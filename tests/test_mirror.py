"""Catching the cloud up: batches, marks, the settle window, the lock, and
the forget queue."""

from __future__ import annotations

import fcntl
import json

import pyarrow as pa
import pytest

from hermes_infino import config, local, mirror, tables
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
# A personal database other than the one an older version pushed to.
OTHER_PERSONAL_DB = "other-personal"
# A table of the team's database dropped since a forget for it was queued.
GONE_TABLE = "dropped-since"
# A database the settings named when a forget for it was queued, and no longer do.
EARLIER_DB = "named-by-earlier-settings"
# What the cloud answers a delete in that database: the key may not reach it
# (403), or it is not found, as the platform answers for another account's
# database (404), or it is being purged (410).
OUT_OF_REACH_STATUSES = (403, 404, 410)
# How far ahead this machine's clock was before it was put right.
CLOCK_AHEAD_MS = 3_600_000
# Row ids in a queued forget whose line is far longer than another forget's.
LONG_FORGET_IDS = 200


def local_mirror(tmp_path, settings) -> Mirror:
    """A mirror over the profile's local tables, before any cloud table is made."""
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    return Mirror(settings, team, personal)


def make_mirror(tmp_path, settings) -> Mirror:
    m = local_mirror(tmp_path, settings)
    m.ensure_cloud_tables()
    return m


@pytest.fixture
def m(home, platform, tmp_path) -> Mirror:
    return make_mirror(tmp_path, sign_in(tmp_path, platform))


def conversation_rows(n: int) -> list[dict]:
    return [{"session_id": "s", "seq": i, "role": "user", "content": f"m{i}"} for i in range(n)]


def commit_late(m: Mirror, table: str, row: dict) -> None:
    """A row committed slowly: with the stamp it took before a sync's mark
    passed it (its own `stored_ms`), not one stamped as it lands."""
    schema = arrow_schema(table)
    m.local._connect().open_table(table).append(
        pa.Table.from_pylist([{f.name: row.get(f.name) for f in schema}], schema=schema))


def note_late(m: Mirror, table: str, stored: int) -> None:
    """The note local.py leaves for a slow commit, for the next sync."""
    with open(m.dir / LATE_COMMITS_FILE, "a") as f:
        f.write(json.dumps({"store": m.local.path.name, "table": table, "stored_ms": stored}) + "\n")


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
    assert set(platform.databases[PERSONAL_DB]) == {*tables.PERSONAL, *tables.CLOUD_ONLY}
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


def test_a_personal_database_that_cannot_be_made_holds_up_only_its_own_rows(home, platform, tmp_path):
    platform.refused_databases = {PERSONAL_DB}  # a key limited to other databases
    m = local_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    tables.ensure_all(m.client, TEAM_DB)
    m.client.append(TEAM_DB, tables.MEMORY, [{tables.ROW_ID: "old", "content": "forgotten"}])
    m.forget(tables.MEMORY, TEAM_DB, ["old"])
    m.forget(tables.MEMORY, PERSONAL_DB, ["mine-old"])
    m.local.append(tables.CONVERSATIONS, conversation_rows(1))
    m.personal.append(tables.MEMORY, [{"content": "mine", "scope": "personal"}])
    with pytest.raises(InfinoError):
        m.sync(wait=True)
    # The team's rows and forgets go; the personal ones wait, and the sync says why.
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)] == ["m0"]
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    assert PERSONAL_DB not in platform.databases and "403" in m.last_error()
    assert PERSONAL_DB in (m.dir / mirror.FORGET_FILE).read_text()
    # Once the personal database can be made, its rows and forgets go too.
    platform.refused_databases = set()
    m._tables_failed_at = 0.0  # try again now, not after ENSURE_RETRY_S
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(PERSONAL_DB, tables.MEMORY)] == ["mine"]
    assert m.last_error() == "" and (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_failed_push_is_raised_before_a_database_out_of_reach(home, platform, tmp_path):
    # Raising the database out of reach would hide that the team's rows did not go either.
    platform.refused_databases = {PERSONAL_DB}
    m = local_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    m.local.append(tables.MEMORY, [{"content": "for the team"}])
    platform.failed_appends = 1
    with pytest.raises(InfinoError) as raised:
        m.sync(wait=True)
    assert raised.value.status == 500 and "HTTP 500" in m.last_error()
    assert m.waiting()[PERSONAL_DB].status == 403
    assert list(m.waiting()) == [PERSONAL_DB]


def test_a_platform_with_no_embedding_server_gets_a_keyword_only_memory(home, platform, tmp_path):
    platform.no_embedding_server = True
    m = make_mirror(tmp_path, sign_in(tmp_path, platform))
    assert tables.EMBEDDING_COLUMN not in m.client.column_names(TEAM_DB, tables.MEMORY)
    m.local.append(tables.MEMORY, [{"content": "found by keyword"}])
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["found by keyword"]


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


def test_a_damaged_state_file_sends_no_row_twice(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_BATCH", SMALL_BATCH)
    m.local.append(tables.CONVERSATIONS, conversation_rows(ROWS))
    m.sync(wait=True)
    (m.dir / mirror.STATE_FILE).write_text("")  # a power loss before the state reached the disk
    m.local.append(tables.CONVERSATIONS, conversation_rows(1))
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == ROWS + 1
    assert len({r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)}) == ROWS + 1


def test_an_older_versions_personal_mark_outlasts_a_time_without_a_personal_database(home, platform, tmp_path):
    m = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    m.personal.append(tables.MEMORY, [{tables.ROW_ID: "p", "content": "mine", "scope": "personal"}])
    [row] = m.personal.query_sql("SELECT stored_ms, row_id FROM memory")
    m.client.append(PERSONAL_DB, tables.MEMORY, [{tables.ROW_ID: "p", "content": "mine", "scope": "personal"}])
    (m.dir / mirror.STATE_FILE).write_text(json.dumps({"marks": {mirror.PERSONAL_MEMORY: [row[tables.STORED], "p"]}}))
    # A while without the personal database: its syncs save their state.
    make_mirror(tmp_path, sign_in(tmp_path, platform)).sync(wait=True)
    back = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    back.sync(wait=True)
    assert [r["content"] for r in platform.rows(PERSONAL_DB, tables.MEMORY)] == ["mine"]


def test_an_older_versions_personal_mark_is_not_taken_for_another_personal_database(home, platform, tmp_path):
    m = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    early = {tables.ROW_ID: "early", "content": "pushed while the other one was named", "scope": "personal"}
    late = {tables.ROW_ID: "late", "content": "pushed while this one was named", "scope": "personal"}
    m.personal.append(tables.MEMORY, [dict(early)])
    m.personal.append(tables.MEMORY, [dict(late)])
    [row] = m.personal.query_sql("SELECT stored_ms FROM memory WHERE row_id = 'late'")
    # An older version kept one personal mark whichever database it named: it pushed the
    # early memory to the other personal database, then the late one to this one.
    tables.ensure_all(m.client, OTHER_PERSONAL_DB, personal=True)
    m.client.append(OTHER_PERSONAL_DB, tables.MEMORY, [dict(early)])
    m.client.append(PERSONAL_DB, tables.MEMORY, [dict(late)])
    (m.dir / mirror.STATE_FILE).write_text(
        json.dumps({"marks": {mirror.PERSONAL_MEMORY: [row[tables.STORED], "late"]}}))
    # The person names the other personal database now: the late memory never went there.
    other = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=OTHER_PERSONAL_DB))
    other.sync(wait=True)
    assert sorted(r[tables.ROW_ID] for r in platform.rows(OTHER_PERSONAL_DB, tables.MEMORY)) == ["early", "late"]


def test_the_teams_memory_never_pushes_a_personal_memory(m, platform):
    # Personal memory in the team's own table: an older version's, not moved out yet
    # (Backend.ensure_local), or written there by an older version still running.
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "p1", "content": "my salary", "scope": "personal"},
                                   {"content": "the deploy freeze starts December 20", "scope": "team"}])
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["the deploy freeze starts December 20"]


def test_marks_an_older_version_kept_by_table_carry_over(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "old", "content": "pushed by the older version"}])
    [row] = m.local.query_sql("SELECT stored_ms, row_id FROM memory")
    (m.dir / mirror.STATE_FILE).write_text(json.dumps(
        {"marks": {tables.MEMORY: [row[tables.STORED], "old"], mirror.PERSONAL_MEMORY: [1, "x"]}}))
    # No personal database here: its mark is kept by its old name, for when there is one.
    assert m.marks() == {m.mark_of(tables.MEMORY): (row[tables.STORED], "old"), mirror.PERSONAL_MEMORY: (1, "x")}
    assert m.sync(wait=True)[m.mark_of(tables.MEMORY)] == 0
    assert set(json.loads((m.dir / mirror.STATE_FILE).read_text())["marks"]) == {m.mark_of(tables.MEMORY),
                                                                                mirror.PERSONAL_MEMORY}


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


def test_a_busy_worker_refusing_an_append_leaves_its_rows_for_the_next_sync(m, platform):
    m.local.append(tables.CONVERSATIONS, conversation_rows(3))
    platform.over_budget = True  # other queries fill the worker's memory budget
    with pytest.raises(InfinoError):  # the sync's failure, tried again: no row set aside
        m.sync(wait=True)
    platform.over_budget = False
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 3 and m.refused() == 0


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


def test_an_unanswered_append_is_still_checked_after_another_sources_append_is_cut_short(m, platform, monkeypatch):
    real = m.client.append

    def stores_then_raises(table, error):
        def append(db, t, rows):
            real(db, t, rows)
            if t == table:
                raise error
        return append

    m.local.append(tables.CONVERSATIONS, conversation_rows(3))
    monkeypatch.setattr(m.client, "append", stores_then_raises(tables.CONVERSATIONS, TimeoutError("no answer")))
    with pytest.raises(TimeoutError):
        m.sync(wait=True)
    # The next sync pushes memory first, and its process is killed mid-append.
    m.local.append(tables.MEMORY, [{"content": "saved after"}])
    monkeypatch.setattr(m.client, "append", stores_then_raises(tables.MEMORY, KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "append", real)
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 3
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


def test_a_slow_local_commit_is_pushed_after_all(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "a", "content": "pushed"}])
    m.sync(wait=True)
    # A row committed after the mark had passed its stamp: the sync never saw it.
    [stored] = {r[tables.STORED] for r in m.local.query_sql("SELECT stored_ms FROM memory")}
    commit_late(m, tables.MEMORY, {tables.ROW_ID: "0-late", "content": "committed slowly", tables.STORED: stored})
    assert sum(m.sync(wait=True).values()) == 0
    note_late(m, tables.MEMORY, stored)
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == ["committed slowly", "pushed"]


def test_an_unanswered_batch_is_not_sent_again_when_a_late_commit_moves_the_mark_back(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_BATCH", SMALL_BATCH)
    m.local.append(tables.CONVERSATIONS, conversation_rows(1))
    m.sync(wait=True)
    [first] = m.local.rows_after(tables.CONVERSATIONS, None, 2 ** 62, 1)
    m.local.append(tables.CONVERSATIONS, conversation_rows(SMALL_BATCH))
    real = m.client.append

    def times_out_after_storing(db, table, rows):
        real(db, table, rows)
        raise TimeoutError("the answer never came")

    monkeypatch.setattr(m.client, "append", times_out_after_storing)
    with pytest.raises(TimeoutError):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "append", real)
    # A row committed slowly behind the first one moves the mark back before that unanswered batch.
    commit_late(m, tables.CONVERSATIONS, {tables.ROW_ID: "0-late", "session_id": "s", "role": "user",
                                          "content": "committed slowly", tables.STORED: first[tables.STORED]})
    note_late(m, tables.CONVERSATIONS, first[tables.STORED])
    m.sync(wait=True)
    ids = [r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)]
    assert len(ids) == len(set(ids)) == SMALL_BATCH + 2


def test_a_sync_killed_after_taking_the_late_commit_notes_keeps_them(m, platform, monkeypatch):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "a", "content": "pushed"}])
    m.sync(wait=True)
    [stored] = {r[tables.STORED] for r in m.local.query_sql("SELECT stored_ms FROM memory")}
    commit_late(m, tables.MEMORY, {tables.ROW_ID: "0-late", "content": "committed slowly", tables.STORED: stored})
    note_late(m, tables.MEMORY, stored)
    real = m.client.query_sql

    def killed(*args, **kwargs):
        raise KeyboardInterrupt  # the process dies while the re-check asks the cloud

    monkeypatch.setattr(m.client, "query_sql", killed)
    with pytest.raises(KeyboardInterrupt):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "query_sql", real)
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == ["committed slowly", "pushed"]


def test_a_damaged_late_commit_note_stops_no_sync(m, platform):
    (m.dir / LATE_COMMITS_FILE).write_text('{"store": "db", "ta\n'  # cut short
                                           + json.dumps({"store": "db", "table": tables.MEMORY}) + "\n")
    m.local.append(tables.MEMORY, [{"content": "pushed all the same"}])
    m.sync(wait=True)
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["pushed all the same"]
    assert m.last_error() == ""


def test_rows_written_after_a_clock_ahead_is_put_right_do_not_wait_for_it(m, platform, monkeypatch):
    real = local._now_ms
    monkeypatch.setattr(local, "_now_ms", lambda: real() + CLOCK_AHEAD_MS)
    m.local.append(tables.MEMORY, [{"content": "stamped while the clock was ahead"}])
    monkeypatch.setattr(local, "_now_ms", real)
    m.local.append(tables.MEMORY, [{"content": "written after it was put right"}])
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["written after it was put right"]


def test_a_mark_a_clock_ahead_moved_on_passes_no_row_written_after_it_is_put_right(m, platform, monkeypatch):
    real_local, real_mirror = local._now_ms, mirror._now_ms
    monkeypatch.setattr(local, "_now_ms", lambda: real_local() + CLOCK_AHEAD_MS)
    monkeypatch.setattr(mirror, "_now_ms", lambda: real_mirror() + CLOCK_AHEAD_MS)
    m.local.append(tables.MEMORY, [{"content": "pushed while the clock was ahead"}])
    m.sync(wait=True)  # the mark is an hour ahead now
    monkeypatch.setattr(local, "_now_ms", real_local)
    monkeypatch.setattr(mirror, "_now_ms", real_mirror)
    m.local.append(tables.MEMORY, [{"content": "written after it was put right"}])
    m.sync(wait=True)
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == [
        "pushed while the clock was ahead", "written after it was put right"]


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


def test_a_teammates_forget_of_this_machines_rows_never_looks_like_a_lost_table(m, platform):
    # Another machine's infino_forget deleted the mark's row and the oldest from the cloud.
    m.local.append(tables.MEMORY, [{tables.ROW_ID: f"r{i}", "content": f"m{i}"} for i in range(5)])
    m.sync(wait=True)
    [first] = m.local.rows_after(tables.MEMORY, None, 2 ** 62, 1)
    mark = m.marks()[m.mark_of(tables.MEMORY)]
    platform.databases[TEAM_DB][tables.MEMORY] = [r for r in platform.rows(TEAM_DB, tables.MEMORY)
                                                  if r["row_id"] not in (mark[1], first[tables.ROW_ID])]
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 3  # nothing pushed again


@pytest.mark.parametrize("queued_as", ["ids", "match"])
def test_a_teammates_forget_of_every_row_this_machine_pushed_is_not_pushed_back(m, platform, tmp_path, queued_as):
    # A teammate's infino_forget matched every memory this machine pushed, by
    # their ids (the cloud answered the lookup) or as the match (it did not).
    m.local.append(tables.MEMORY, [{tables.ROW_ID: f"r{i}", "content": f"secret {i}"} for i in range(5)])
    m.sync(wait=True)
    teammate = make_mirror(tmp_path / "teammate", sign_in(tmp_path / "teammate", platform))
    if queued_as == "ids":
        ids = teammate.matching_ids(tables.MEMORY, TEAM_DB, "content LIKE 'secret%'")
        teammate.forget(tables.MEMORY, TEAM_DB, sorted(ids))
    else:
        teammate.forget_matching(tables.MEMORY, TEAM_DB, "content LIKE 'secret%'")
    teammate.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    mark = m.marks()[m.mark_of(tables.MEMORY)]
    m = make_mirror(tmp_path, sign_in(tmp_path, platform))  # this machine's next process
    assert m.marks()[m.mark_of(tables.MEMORY)] == mark  # a forget, not a lost table
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    # A row saved after the forget still goes.
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "later", "content": "saved after"}])
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["saved after"]


def test_a_forget_writes_the_ids_it_deletes_before_deleting_them(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "a", "content": "kept"}, {tables.ROW_ID: "z", "content": "gone"}])
    m.sync(wait=True)
    seen = []
    real = m.client.delete
    m.client.delete = lambda *a: (seen.append(list(platform.rows(TEAM_DB, tables.FORGOTTEN))), real(*a))[1]
    m.forget(tables.MEMORY, TEAM_DB, ["z"])
    m.sync(wait=True)
    [before] = seen
    assert [(r[tables.FORGOTTEN_FROM], r[tables.ROW_ID]) for r in before] == [(tables.MEMORY, "z")]
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept"]


def test_a_forget_where_forgotten_cannot_be_made_still_deletes(home, platform, tmp_path):
    platform.refused_tables = {tables.FORGOTTEN}
    m = make_mirror(tmp_path, sign_in(tmp_path, platform))
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "z", "content": "gone"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["z"])
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    assert (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_forget_by_match_whose_table_is_gone_is_dropped(m, platform):
    m.forget_matching("dropped_since", TEAM_DB, "content = 'x'")
    m.sync(wait=True)
    assert (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_table_dropped_after_a_forget_gets_the_rest_again_never_the_forgotten(m, platform, tmp_path):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: f"r{i}", "content": f"m{i}"} for i in range(3)])
    m.sync(wait=True)
    teammate = make_mirror(tmp_path / "teammate", sign_in(tmp_path / "teammate", platform))
    teammate.forget(tables.MEMORY, TEAM_DB, ["r1"])
    teammate.sync(wait=True)
    del platform.databases[TEAM_DB][tables.MEMORY]  # dropped; `forgotten` still names r1
    m = make_mirror(tmp_path, sign_in(tmp_path, platform))
    m.sync(wait=True)
    # Its rows go again, but not the one forgotten before it was dropped.
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)) == ["m0", "m2"]


def test_nothing_is_pushed_while_a_cloud_table_lacks_a_column_it_cannot_get(home, platform, tmp_path):
    client_rows = [f for f in tables.MEMORY_SCHEMA if f["name"] != tables.CHAT]
    settings = sign_in(tmp_path, platform)
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    m = Mirror(settings, team, personal)
    m.client.ensure_database(TEAM_DB)
    m.client.ensure_table(TEAM_DB, tables.MEMORY, client_rows, tables.MEMORY_INDEXES)
    platform.schema_patch_refused = True
    m.local.append(tables.MEMORY, [{"content": "kept for a chat", tables.CHAT: "telegram:-100"}])
    with pytest.raises(tables.OutdatedTable):
        m.sync(wait=True)
    assert requests_to(platform, "append") == 0
    assert "has no chat" in m.last_error()
    platform.schema_patch_refused = False
    m._tables_failed_at = 0.0  # try again now, not after ENSURE_RETRY_S
    m.sync(wait=True)
    assert [r[tables.CHAT] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["telegram:-100"]


def test_a_forget_queued_as_its_match_keeps_a_row_saved_later(m, platform):
    m.local.append(tables.MEMORY, [{"content": "use pnpm"}])
    m.sync(wait=True)
    m.forget_matching(tables.MEMORY, TEAM_DB, f"content = 'use pnpm' AND {tables.STORED} <= 1")
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1  # stored after the bound: kept
    m.forget_matching(tables.MEMORY, TEAM_DB, f"content = 'use pnpm' AND {tables.STORED} <= {2 ** 62}")
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    assert (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_forget_queued_while_the_queue_is_applied_neither_waits_nor_is_lost(m, platform, monkeypatch):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "a"}, {tables.ROW_ID: "2", "content": "b"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["1"])
    real = m.client.delete
    queued = []

    def delete_and_queue_another(db, table, predicate):
        if not queued:  # a tool call's forget arrives mid-apply; the queue's lock is free
            queued.append(True)
            m.forget(tables.MEMORY, TEAM_DB, ["2"])
        return real(db, table, predicate)

    monkeypatch.setattr(m.client, "delete", delete_and_queue_another)
    m.sync(wait=True)
    assert '"2"' in (m.dir / mirror.FORGET_FILE).read_text()
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


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


def test_a_forget_the_cloud_keeps_failing_holds_up_none_after_it(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "a"}, {tables.ROW_ID: "2", "content": "b"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["1"])
    m.forget(tables.MEMORY, TEAM_DB, ["2"])
    platform.failed_deletes = 1  # the first forget's delete fails
    with pytest.raises(InfinoError):
        m.sync(wait=True)
    assert [r[tables.ROW_ID] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["1"]
    queued = (m.dir / mirror.FORGET_FILE).read_text()
    assert '"1"' in queued and '"2"' not in queued
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_a_forget_whose_table_is_gone_is_dropped_and_holds_up_none_after_it(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "m"}])
    m.sync(wait=True)
    m.forget(GONE_TABLE, TEAM_DB, ["x"])  # the cloud answers 404: nothing is left there to delete
    m.forget(tables.MEMORY, TEAM_DB, ["1"])
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    assert (m.dir / mirror.FORGET_FILE).read_text() == ""


def test_a_forget_queued_as_the_queue_is_rewritten_is_kept_whole(m, platform, monkeypatch):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "a", "content": "a"}, {tables.ROW_ID: "b", "content": "b"}])
    m.sync(wait=True)
    m.forget(tables.MEMORY, TEAM_DB, ["a"] + [f"long-{i}" for i in range(LONG_FORGET_IDS)])
    m.forget(tables.MEMORY, TEAM_DB, ["b"])
    platform.failed_deletes = 1  # the long one stays queued; the other is taken off: the queue is rewritten
    real = fcntl.flock
    queued = []

    def flock(f, op):
        real(f, op)
        rewriting = getattr(f, "mode", "") == "r+" and str(getattr(f, "name", "")).endswith(mirror.FORGET_FILE)
        if op == fcntl.LOCK_UN and rewriting and not queued:
            queued.append(True)  # another process queues a forget the moment the lock is free
            m.forget(tables.MEMORY, TEAM_DB, ["late"])

    monkeypatch.setattr(mirror.fcntl, "flock", flock)
    with pytest.raises(InfinoError):
        m.sync(wait=True)
    lines = (m.dir / mirror.FORGET_FILE).read_text().splitlines()
    assert any('"late"' in line for line in lines)
    assert all(json.loads(line) for line in lines)


def test_a_damaged_line_in_the_forget_queue_stops_no_sync(m, platform):
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "m"}])
    m.sync(wait=True)
    (m.dir / mirror.FORGET_FILE).write_text('{"table": "memory", "datab\n')  # cut short by a crash
    m.forget(tables.MEMORY, TEAM_DB, ["1"])
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.MEMORY) == [] and m.last_error() == ""


@pytest.mark.parametrize("status", OUT_OF_REACH_STATUSES)
def test_a_forget_for_a_database_these_settings_do_not_name_fails_no_sync(m, platform, monkeypatch, status):
    m.forget(tables.MEMORY, EARLIER_DB, ["x"])  # queued while the settings named it
    real = m.client.delete

    def out_of_reach(db, table, predicate):
        if db == EARLIER_DB:
            raise InfinoError(status, "the API key may not address this database")
        return real(db, table, predicate)

    monkeypatch.setattr(m.client, "delete", out_of_reach)
    m.local.append(tables.MEMORY, [{"content": "x"}])
    m.sync(wait=True)
    m.sync(wait=True)
    assert m.last_error() == ""
    assert EARLIER_DB in (m.dir / mirror.FORGET_FILE).read_text()  # kept for when they name it again


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


def test_a_teammates_forget_of_every_earlier_version_this_machine_pushed_is_not_pushed_back(m, platform):
    # Every history row this machine pushed was forgotten in the cloud; its memory rows were not.
    m.local.append(tables.MEMORY, [{tables.ROW_ID: f"r{i}", "content": f"m{i}"} for i in range(3)])
    m.local.append(tables.HISTORY, [{tables.ROW_ID: "h1", "content": "standup is at nine", tables.SUPERSEDED: 1,
                                     tables.SUPERSEDED_BY: "m0", tables.PRIOR_ROW_ID: "x"}])
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.HISTORY)) == 1
    platform.databases[TEAM_DB][tables.HISTORY] = []
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert platform.rows(TEAM_DB, tables.HISTORY) == []  # a forget, not a lost table
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 3


def test_a_dropped_database_gets_its_history_back_with_its_memory(m, platform):
    m.local.append(tables.MEMORY, [{"content": "standup is at ten"}])
    m.local.append(tables.HISTORY, [{"content": "standup is at nine", tables.SUPERSEDED: 1,
                                     tables.SUPERSEDED_BY: "standup is at ten", tables.PRIOR_ROW_ID: "x"}])
    m.sync(wait=True)
    del platform.databases[TEAM_DB][tables.MEMORY]
    del platform.databases[TEAM_DB][tables.HISTORY]
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["standup is at ten"]
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.HISTORY)] == ["standup is at nine"]


def test_a_refused_history_table_is_asked_for_once_per_process(home, platform, tmp_path, monkeypatch):
    platform.refused_tables = {tables.HISTORY}
    asked = []
    real = tables.ensure_optional
    monkeypatch.setattr(tables, "ensure_optional",
                        lambda client, db, table: (asked.append(table), real(client, db, table)))
    monkeypatch.setattr(mirror, "ENSURE_RETRY_S", 0.0)
    m = make_mirror(tmp_path, sign_in(tmp_path, platform))
    m.local.append(tables.HISTORY, [{"content": "standup is at nine", tables.SUPERSEDED: 1,
                                     tables.SUPERSEDED_BY: "", tables.PRIOR_ROW_ID: "x"}])
    for _ in range(3):
        m.sync(wait=True)
    assert [t for t in asked if t == tables.HISTORY] == [tables.HISTORY]
    assert m.refused_optional(TEAM_DB, tables.HISTORY)
    platform.refused_tables = set()



def test_history_pushed_again_after_a_lost_memory_mark_is_checked_with_the_cloud_first(m, platform):
    # A teammate forgot the memory rows the lost-table check looks at; the history is still there.
    m.local.append(tables.MEMORY, [{tables.ROW_ID: "r1", "content": "standup is at ten"}])
    m.local.append(tables.HISTORY, [{tables.ROW_ID: "h1", "content": "standup is at nine", tables.SUPERSEDED: 1,
                                     tables.SUPERSEDED_BY: "standup is at ten", tables.PRIOR_ROW_ID: "x"}])
    m.sync(wait=True)
    platform.databases[TEAM_DB][tables.MEMORY] = []
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert [r["row_id"] for r in platform.rows(TEAM_DB, tables.HISTORY)] == ["h1"]  # not sent twice


def test_a_history_table_refused_with_a_400_is_asked_for_again(home, platform, tmp_path, monkeypatch):
    asked = []
    real = tables.ensure_optional

    def ensure(client, db, table):
        asked.append(table)
        if len(asked) == 1:
            raise InfinoError(400, "bad request")
        return real(client, db, table)

    monkeypatch.setattr(tables, "ensure_optional", ensure)
    monkeypatch.setattr(mirror, "ENSURE_RETRY_S", 0.0)
    m = make_mirror(tmp_path, sign_in(tmp_path, platform))
    m.local.append(tables.HISTORY, [{"content": "standup is at nine", tables.SUPERSEDED: 1,
                                     tables.SUPERSEDED_BY: "", tables.PRIOR_ROW_ID: "x"}])
    m.sync(wait=True)
    assert len(asked) >= 2 and not m.refused_optional(TEAM_DB, tables.HISTORY)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.HISTORY)] == ["standup is at nine"]

"""Catching the cloud up: batches, marks, the settle window, the lock, and
the forget queue."""

from __future__ import annotations

import fcntl
import json

import pytest

from hermes_infino import config, mirror, tables
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore
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
    personal.ensure_tables(with_vectors=False, only=(tables.MEMORY,))
    m = Mirror(settings, team, personal)
    m.ensure_cloud_tables()
    return m


@pytest.fixture
def m(home, platform, tmp_path) -> Mirror:
    return make_mirror(tmp_path, sign_in(tmp_path, platform))


def conversation_rows(n: int) -> list[dict]:
    return [{"session_id": "s", "seq": i, "role": "user", "content": f"m{i}"} for i in range(n)]


def test_the_sources_are_the_teams_tables(m):
    assert [(s.mark, s.table, s.database) for s in m.sources()] == [(t, t, TEAM_DB) for t in tables.ALL]


def test_a_personal_database_adds_the_personal_source(home, platform, tmp_path):
    m = make_mirror(tmp_path, sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    assert (mirror.PERSONAL_MEMORY, tables.MEMORY, PERSONAL_DB) in [(s.mark, s.table, s.database) for s in m.sources()]
    assert set(platform.databases[PERSONAL_DB]) == {tables.MEMORY}


def test_rows_go_in_batches_and_the_mark_moves_with_each(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_BATCH", SMALL_BATCH)
    m.local.append(tables.CONVERSATIONS, conversation_rows(ROWS))
    seen = []
    pushed = m.sync(wait=True, progress=lambda mark, n: seen.append((mark, n)))
    assert pushed[tables.CONVERSATIONS] == ROWS
    assert requests_to(platform, "append") == 3  # 3 + 3 + 1
    assert seen == [(tables.CONVERSATIONS, 3), (tables.CONVERSATIONS, 6), (tables.CONVERSATIONS, 7)]
    last = max(m.local.query_sql("SELECT stored_ms, row_id FROM conversations"),
               key=lambda r: (r[tables.STORED], r[tables.ROW_ID]))
    assert m.marks()[tables.CONVERSATIONS] == (last[tables.STORED], last[tables.ROW_ID])


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
    assert m.sync(wait=True)[tables.MEMORY] == 1


def test_a_sync_skips_while_another_holds_the_lock(m, platform):
    m.local.append(tables.MEMORY, [{"content": "x"}])
    with open(config.state_dir(m.settings.home) / mirror.LOCK_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        assert m.sync() == {}
        fcntl.flock(f, fcntl.LOCK_UN)
    assert m.sync()[tables.MEMORY] == 1


def test_an_unreadable_state_file_starts_over(m, platform):
    (m.dir / mirror.STATE_FILE).write_text("not json")
    assert set(m.marks().values()) == {None}
    assert m.last_error() == ""


def test_the_state_file_is_replaced_whole(m, platform):
    m.local.append(tables.MEMORY, [{"content": "x"}])
    m.sync(wait=True)
    state = json.loads((m.dir / mirror.STATE_FILE).read_text())
    assert set(state) == {"marks", "last_error", "at_ms"}
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

@pytest.mark.xfail(strict=True, reason="marks are kept per table, not per (database, table): a profile moved to "
                                       "another team database never pushes its earlier rows there")
def test_a_new_team_database_gets_every_row(m, platform, tmp_path):
    m.local.append(tables.MEMORY, [{"content": "kept before the move"}])
    m.sync(wait=True)
    moved = make_mirror(tmp_path, sign_in(tmp_path, platform, team=ANOTHER_TEAM_DB))
    moved.sync(wait=True)
    assert [r["content"] for r in platform.rows(ANOTHER_TEAM_DB, tables.MEMORY)] == ["kept before the move"]


@pytest.mark.xfail(strict=True, reason="tables.OutdatedTable says to drop the cloud table and its rows are 'pushed "
                                       "again', but the marks stay past them, so a recreated table stays empty")
def test_a_dropped_and_recreated_cloud_table_is_filled_again(m, platform, tmp_path):
    m.local.append(tables.MEMORY, [{"content": "kept on this machine"}])
    m.sync(wait=True)
    del platform.databases[TEAM_DB][tables.MEMORY]
    m.ensure_cloud_tables()
    m.sync(wait=True)
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept on this machine"]

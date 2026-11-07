"""The backend: one per profile, where writes go, and when they reach the cloud."""

from __future__ import annotations

import fcntl
import threading
import time

import pytest

from hermes_infino import backend, config, embed, mirror, recall, tables
from hermes_infino.backend import Backend
from hermes_infino.client import InfinoError

from .helpers import PERSONAL_DB, TEAM_DB, fake_embed, personal_rows, sign_in, wait_for

# How long the stand-in sync below takes after it has read its rows.
SLOW_SYNC_S = 0.5
# The first wait before a failed sync is tried again, short enough for a test.
QUICK_RETRY_S = 0.1
# Appends the platform fails in a row before it accepts one.
FAILURES_IN_A_ROW = 2
# How long after a write its sync waits, in the test of a replaced backend:
# long enough that the settings change while the old one's sync is pending.
PENDING_SYNC_S = 0.3
OTHER_TEAM_DB = "team-other"
# A settle window a write is still inside when its session ends at once.
SHORT_SETTLE_MS = 1_000
# How long one start's move of personal memory waits for another start begun during it.
OTHER_START_WAIT_S = 1.0
# A session end's bound on waiting for the cloud, short enough for a test.
QUICK_FLUSH_S = 0.2
# Session ends in a row while another process's sync holds the profile's sync.
SESSION_ENDS = 3
# A personal memory as an older version kept it: in the team's local table.
OLD_PERSONAL_MEMORY = {tables.ROW_ID: "p1", "content": "my salary", "scope": recall.PERSONAL}


@pytest.fixture
def local_backend(home, tmp_path) -> Backend:
    b = backend.for_settings(config.load(tmp_path))
    b.ensure()
    return b


@pytest.fixture
def cloud_backend(home, platform, tmp_path) -> Backend:
    b = backend.for_settings(sign_in(tmp_path, platform))
    b.ensure()
    # The catch-up sync that ensure() starts holds the sync lock while it runs.
    assert wait_for(lambda: not (b._timer and b._timer.is_alive()))
    return b


def test_one_backend_per_profile_and_sign_in(home, platform, tmp_path):
    local_settings = config.load(tmp_path)
    assert backend.for_settings(local_settings) is backend.for_settings(config.load(tmp_path))
    signed = sign_in(tmp_path, platform)
    assert backend.for_settings(signed) is not backend.for_settings(local_settings)
    assert backend.for_settings(signed).signed_in and not backend.for_settings(local_settings).signed_in


def test_local_mode_reads_and_lists_the_local_tables(local_backend):
    local_backend.append(tables.SKILLS, [{"name": "a", "content": "x"}])
    assert set(tables.ALL) <= set(local_backend.table_names())
    assert local_backend.query("SELECT name FROM skills") == [{"name": "a"}]


def test_personal_rows_are_split_from_the_teams(local_backend, tmp_path):
    local_backend.append(tables.MEMORY, [{"content": "mine", "scope": recall.PERSONAL},
                                         {"content": "ours", "scope": recall.TEAM}])
    assert [r["content"] for r in personal_rows(tmp_path)] == ["mine"]
    assert [r["content"] for r in local_backend.local.query_sql("SELECT content FROM memory")] == ["ours"]


def test_in_local_mode_nothing_is_scheduled(local_backend):
    local_backend.append(tables.MEMORY, [{"content": "x"}])
    assert local_backend._timer is None
    local_backend.flush()  # nothing to wait for


def test_signing_in_drops_the_local_vectors_and_makes_the_cloud_tables(cloud_backend, platform):
    assert not cloud_backend.local.has_vectors() and not cloud_backend.personal.has_vectors()
    assert set(platform.databases[TEAM_DB]) == set(tables.ALL)


def an_older_versions_personal_memory(b: Backend, database: str, personal: bool = False) -> None:
    """A personal memory an older version kept in the team's local table and
    pushed to `database` (the team's, when there was no personal database)."""
    b.local.ensure_tables(with_vectors=False)
    b.local.append(tables.MEMORY, [dict(OLD_PERSONAL_MEMORY)])
    tables.ensure_all(b.mirror.client, database, personal=personal)
    b.mirror.client.append(database, tables.MEMORY, [dict(OLD_PERSONAL_MEMORY)])


def test_personal_memory_an_older_version_pushed_to_the_team_leaves_the_teams_cloud(home, platform, tmp_path):
    b = backend.for_settings(sign_in(tmp_path, platform))
    an_older_versions_personal_memory(b, TEAM_DB)
    b.ensure()
    b.flush()
    assert [r for r in platform.rows(TEAM_DB, tables.MEMORY) if r[tables.ROW_ID] == "p1"] == []
    assert [r["content"] for r in personal_rows(tmp_path)] == ["my salary"]


def test_personal_memory_an_older_version_pushed_is_not_pushed_again(home, platform, tmp_path):
    b = backend.for_settings(sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    an_older_versions_personal_memory(b, PERSONAL_DB, personal=True)
    b.ensure()
    b.flush()
    assert [r["content"] for r in platform.rows(PERSONAL_DB, tables.MEMORY)] == ["my salary"]


def test_two_starts_move_an_older_versions_personal_memory_once(home, tmp_path, monkeypatch):
    settings = config.load(tmp_path)
    first, second = Backend(settings), Backend(settings)
    first.local.ensure_tables(with_vectors=False)
    first.local.append(tables.MEMORY, [dict(OLD_PERSONAL_MEMORY)])
    real = first.personal.append
    others = []

    def another_start_meanwhile(table, rows):
        other = threading.Thread(target=second.ensure_local)  # another process of the profile starting
        other.start()
        others.append(other)
        other.join(OTHER_START_WAIT_S)
        real(table, rows)

    monkeypatch.setattr(first.personal, "append", another_start_meanwhile)
    first.ensure_local()
    others[0].join()
    assert [r["content"] for r in personal_rows(tmp_path)] == ["my salary"]


def test_a_write_reaches_the_cloud_by_itself(cloud_backend, platform):
    cloud_backend.append(tables.MEMORY, [{"content": "pushed in the background"}])
    assert wait_for(lambda: len(platform.rows(TEAM_DB, tables.MEMORY)) == 1)


def test_a_session_end_pushes_before_it_returns(cloud_backend, platform, monkeypatch):
    monkeypatch.setattr(backend, "SYNC_DELAY_S", 3600)  # no background sync
    cloud_backend.append(tables.MEMORY, [{"content": "pushed at the end"}])
    cloud_backend.flush()
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


def test_a_failed_background_sync_leaves_the_rows_for_the_next(cloud_backend, platform, monkeypatch):
    monkeypatch.setattr(backend, "SYNC_DELAY_S", 3600)
    cloud_backend.append(tables.MEMORY, [{"content": "x"}])
    platform.failed_appends = 1
    cloud_backend._sync_quietly()  # logs, never raises
    assert platform.rows(TEAM_DB, tables.MEMORY) == []
    cloud_backend.flush()
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


def test_a_failed_background_sync_is_tried_again_without_a_write(cloud_backend, platform, monkeypatch):
    monkeypatch.setattr(backend, "SYNC_RETRY_FIRST_S", QUICK_RETRY_S, raising=False)
    platform.failed_appends = FAILURES_IN_A_ROW
    cloud_backend.append(tables.MEMORY, [{"content": "goes once the cloud takes it"}])
    assert wait_for(lambda: len(platform.rows(TEAM_DB, tables.MEMORY)) == 1)
    assert wait_for(lambda: cloud_backend._retry_s == 0)  # the success starts the wait over


def test_a_start_that_could_not_make_the_cloud_tables_is_tried_again(home, platform, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SYNC_RETRY_FIRST_S", QUICK_RETRY_S, raising=False)
    monkeypatch.setattr(mirror, "ENSURE_RETRY_S", 0)
    b = backend.for_settings(sign_in(tmp_path, platform))
    b.ensure_local()
    b.local.append(tables.MEMORY, [{"content": "kept before the cloud was reached"}])  # no write to come
    platform.database_limit = 0  # the team database cannot be made yet
    with pytest.raises(InfinoError):
        b.catch_up()
    platform.database_limit = None
    assert wait_for(lambda: len(platform.rows(TEAM_DB, tables.MEMORY)) == 1)


def test_a_backend_replaced_by_new_settings_never_syncs_again(home, platform, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SYNC_DELAY_S", PENDING_SYNC_S)
    old = backend.for_settings(sign_in(tmp_path, platform))
    old.ensure_local()
    old.append(tables.MEMORY, [{"content": "written before the change"}])
    new = backend.for_settings(sign_in(tmp_path, platform, team=OTHER_TEAM_DB))
    new.append(tables.MEMORY, [{"content": "written after it"}])
    old.append(tables.MEMORY, [{"content": "written through the old one"}])  # a caller not moved yet
    assert wait_for(lambda: len(platform.rows(OTHER_TEAM_DB, tables.MEMORY)) == 3)
    time.sleep(PENDING_SYNC_S * 2)  # past when the old one's syncs were due
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_a_session_end_through_a_replaced_backend_waits_for_its_rows_to_settle(home, platform, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SETTLE_MS", SHORT_SETTLE_MS)
    monkeypatch.setattr(mirror, "SETTLE_MS", SHORT_SETTLE_MS)
    old = backend.for_settings(sign_in(tmp_path, platform))
    new = backend.for_settings(sign_in(tmp_path, platform, personal_database=PERSONAL_DB))
    new.ensure()
    assert wait_for(lambda: not (new._timer and new._timer.is_alive()))
    old.append(tables.MEMORY, [{"content": "written through the old one"}])  # a caller not moved yet
    old.flush()
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["written through the old one"]


def test_session_ends_while_another_process_syncs_leave_one_sync_waiting(cloud_backend, platform, tmp_path,
                                                                         monkeypatch):
    monkeypatch.setattr(backend, "FLUSH_TIMEOUT_S", QUICK_FLUSH_S)
    cloud_backend.append(tables.MEMORY, [{"content": "written before the sessions end"}])
    before = set(threading.enumerate())
    with open(config.state_dir(tmp_path) / mirror.LOCK_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # another process's sync, held up by a slow cloud
        for _ in range(SESSION_ENDS):
            cloud_backend.flush()
        waiting = [t for t in threading.enumerate() if t not in before and t.name == "infino-flush" and t.is_alive()]
        fcntl.flock(f, fcntl.LOCK_UN)
    assert len(waiting) == 1
    assert wait_for(lambda: len(platform.rows(TEAM_DB, tables.MEMORY)) == 1)


def fail_the_first_look(store, monkeypatch):
    """The store's next read of its tables fails; the ones after it do not."""
    real = store.list_tables
    failed = []

    def fails_once():
        if not failed:
            failed.append(True)
            raise OSError("the local database could not be read")
        return real()

    monkeypatch.setattr(store, "list_tables", fails_once)


def test_a_look_at_the_personal_tables_that_failed_keeps_the_turn_personal(local_backend, monkeypatch):
    local_backend.append(tables.CONVERSATIONS, [{"session_id": "kept", "seq": 0, "role": "user", "content": "mine"}],
                         personal=True)
    fail_the_first_look(local_backend.personal, monkeypatch)
    assert local_backend.transcript_personal("kept") is True  # could not tell: the safe side
    assert local_backend.transcript_personal("kept") is True  # read this time


def test_a_session_whose_look_failed_is_asked_again_and_is_the_teams(local_backend, monkeypatch):
    fail_the_first_look(local_backend.personal, monkeypatch)
    assert local_backend.transcript_personal("team-chat") is True  # could not tell this time
    assert local_backend.transcript_personal("team-chat") is False  # nothing of it was kept personal
    assert local_backend.transcript_personal("team-chat") is False  # and it is not asked again


def test_a_forget_is_queued_for_the_cloud(cloud_backend, platform):
    cloud_backend.append(tables.MEMORY, [{"content": "to forget", "scope": "team"}])
    cloud_backend.flush()
    assert cloud_backend.forget_memory("to forget") == 1
    cloud_backend.flush()
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_signed_in_queries_go_to_the_cloud(cloud_backend, platform):
    platform.databases[TEAM_DB][tables.SKILLS].append({"name": "from-the-cloud"})
    assert cloud_backend.query("SELECT name FROM skills") == [{"name": "from-the-cloud"}]
    assert set(cloud_backend.table_names()) == set(tables.ALL)


def test_a_sync_another_process_held_off_runs_again(cloud_backend, platform, tmp_path):
    with open(config.state_dir(tmp_path) / mirror.LOCK_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # another process is syncing
        cloud_backend.append(tables.MEMORY, [{"content": "written meanwhile"}])
        time.sleep(SLOW_SYNC_S)
        assert platform.rows(TEAM_DB, tables.MEMORY) == []
        fcntl.flock(f, fcntl.LOCK_UN)
    assert wait_for(lambda: len(platform.rows(TEAM_DB, tables.MEMORY)) == 1)


def test_a_local_start_embeds_memories_saved_without_a_vector(home, tmp_path):
    b = backend.for_settings(config.load(tmp_path))
    b.ensure()

    def broken(texts):
        raise OSError("offline")

    embed.use(broken)
    b.append(tables.MEMORY, [{"content": "kept without a vector"}])
    embed.use(fake_embed)
    b.ensure()
    assert b.local.query_sql("SELECT count(*) AS n FROM memory_vectors")[0]["n"] == 1


def test_a_write_during_a_long_sync_is_pushed_after_it(cloud_backend, platform, monkeypatch):
    real = cloud_backend.mirror.sync
    calls = []

    def slow(*args, **kwargs):
        out = real(*args, **kwargs)
        if not calls:
            calls.append(1)
            time.sleep(SLOW_SYNC_S)  # a big push still going after it read its rows
        return out

    monkeypatch.setattr(cloud_backend.mirror, "sync", slow)
    cloud_backend.append(tables.MEMORY, [{"content": "first"}])
    assert wait_for(lambda: calls)
    cloud_backend.append(tables.MEMORY, [{"content": "written during the sync"}])
    assert wait_for(lambda: len(platform.rows(TEAM_DB, tables.MEMORY)) == 2, timeout=SLOW_SYNC_S * 6)

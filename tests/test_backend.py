"""The backend: one per profile, where writes go, and when they reach the cloud."""

from __future__ import annotations

import time

import pytest

from hermes_infino import backend, config, recall, tables
from hermes_infino.backend import Backend

from .helpers import TEAM_DB, personal_rows, sign_in, wait_for

# How long the stand-in sync below takes after it has read its rows.
SLOW_SYNC_S = 0.5


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


@pytest.mark.xfail(strict=True, reason="a write made while a sync is running is pushed only if it is newer than "
                                       "SYNC_DELAY_S when that sync ends; after a long sync it waits for the next "
                                       "write or the session's end")
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

"""The platform client: routes, retries, and how refusals surface."""

from __future__ import annotations

import pytest

from hermes_infino import client as client_mod, tables
from hermes_infino.client import (DEFAULT_RETRY_AFTER_S, MAX_RETRIES, MAX_RETRY_WAIT_S, InfinoClient, InfinoError,
                                  request_trial)

from .helpers import TEAM_DB, requests_to

# A database name that must be escaped in a URL path.
AWKWARD_DB = "team a/b"
# Text whose SQL literal needs its quotes doubled.
QUOTED = "it's Bob's"


@pytest.fixture
def no_sleep(monkeypatch):
    """The client's waits, recorded instead of slept."""
    waits = []
    monkeypatch.setattr(client_mod.time, "sleep", waits.append)
    return waits


@pytest.fixture
def team(platform):
    """A client with the team database and its memory table made."""
    c = InfinoClient("k", platform.url)
    c.ensure_database(TEAM_DB)
    c.ensure_table(TEAM_DB, tables.MEMORY, tables.MEMORY_SCHEMA, tables.MEMORY_INDEXES)
    return c


@pytest.mark.parametrize("header, wait", [
    (None, DEFAULT_RETRY_AFTER_S),
    ("", DEFAULT_RETRY_AFTER_S),
    ("soon", DEFAULT_RETRY_AFTER_S),
    ("2.5", 2.5),
    ("-3", 0.0),
    ("86400", MAX_RETRY_WAIT_S),
])
def test_retry_after_is_bounded(header, wait):
    assert client_mod._retry_after(header) == wait


@pytest.mark.parametrize("out, rows", [
    ({"data": [{"a": 1}]}, [{"a": 1}]),
    ([{"a": 1}], [{"a": 1}]),
    ({}, []),
    (None, []),
    ({"data": None}, []),
])
def test_search_responses_in_either_shape_become_rows(out, rows):
    assert client_mod._rows(out) == rows


def test_a_database_name_is_escaped_in_the_path(platform):
    c = InfinoClient("k", platform.url)
    c.ensure_database(AWKWARD_DB)
    c.ensure_table(AWKWARD_DB, tables.SKILLS, tables.SKILLS_SCHEMA, tables.SKILLS_INDEXES)
    assert tables.SKILLS in platform.databases[AWKWARD_DB]
    assert any(path == "/v1/create_table/team%20a%2Fb" for path, _ in platform.requests)


def test_every_request_carries_the_key(platform):
    InfinoClient("the-key", platform.url).list_databases()
    assert platform.keys_seen == ["the-key"]


def test_a_trailing_slash_on_the_url_is_dropped(platform):
    InfinoClient("k", platform.url + "/").ensure_database(TEAM_DB)
    assert all(not path.startswith("//") for path, _ in platform.requests)


def test_an_existing_database_is_not_created_again(platform):
    platform.databases[TEAM_DB] = {}
    InfinoClient("k", platform.url).ensure_database(TEAM_DB)
    assert requests_to(platform, "databases") == 1  # the list, no create


def test_an_existing_table_is_not_created_again(team, platform):
    team.ensure_table(TEAM_DB, tables.MEMORY, tables.MEMORY_SCHEMA, tables.MEMORY_INDEXES)
    assert requests_to(platform, "create_table") == 1


def test_a_table_made_by_someone_else_meanwhile_counts_as_made(team, platform, monkeypatch):
    # The list says no, the create says it exists: another writer made it in between.
    lists = iter([[], [tables.TRACES]])
    monkeypatch.setattr(team, "list_tables", lambda db: next(lists))
    platform.databases[TEAM_DB][tables.TRACES] = []
    team.ensure_table(TEAM_DB, tables.TRACES, tables.TRACES_SCHEMA, tables.TRACES_INDEXES)


def test_a_create_refused_for_another_reason_raises(team, platform):
    bad = [{"name": "content", "type": "utf8"}]  # FTS on utf8: the platform refuses it
    with pytest.raises(InfinoError) as e:
        team.ensure_table(TEAM_DB, "notes", bad, {"fts": ["content"]})
    assert e.value.status == 400 and "LargeUtf8" in e.value.message


def test_a_busy_table_past_every_retry_raises_409(team, platform, no_sleep):
    platform.busy_appends = MAX_RETRIES + 1
    with pytest.raises(InfinoError) as e:
        team.append(TEAM_DB, tables.MEMORY, [{"content": "x"}])
    assert e.value.status == 409
    assert requests_to(platform, "append") == MAX_RETRIES + 1
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_a_client_with_no_retries_gives_up_at_once(team, platform, no_sleep):
    platform.busy_appends = 1
    quick = InfinoClient("k", platform.url, retries=0)
    with pytest.raises(InfinoError):
        quick.append(TEAM_DB, tables.MEMORY, [{"content": "x"}])
    assert no_sleep == []


def test_a_server_error_is_not_retried(team, platform, no_sleep):
    platform.failed_appends = 1
    with pytest.raises(InfinoError) as e:
        team.append(TEAM_DB, tables.MEMORY, [{"content": "x"}])
    assert e.value.status == 500 and requests_to(platform, "append") == 1


def test_an_empty_append_sends_nothing(team, platform):
    before = len(platform.requests)
    team.append(TEAM_DB, tables.MEMORY, [])
    assert len(platform.requests) == before


def test_a_delete_reports_what_it_matched(team, platform):
    team.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": QUOTED}, {"row_id": "b", "content": "x"}])
    out = team.delete(TEAM_DB, tables.MEMORY, f"content = {tables.sql_string(QUOTED)}")
    assert out["matched"] == 1
    assert [r["row_id"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["b"]


def test_an_unknown_key_is_refused_with_its_status(platform):
    platform.valid_keys = {"good"}
    with pytest.raises(InfinoError) as e:
        InfinoClient("bad", platform.url).list_databases()
    assert e.value.status == 401


def test_list_responses_in_either_shape(platform, monkeypatch):
    c = InfinoClient("k", platform.url)
    monkeypatch.setattr(c, "_request", lambda *a, **k: {"databases": ["a", {"name": "b"}]})
    assert c.list_databases() == ["a", "b"]
    monkeypatch.setattr(c, "_request", lambda *a, **k: {"tables": [{"name": "memory"}]})
    assert c.list_tables("a") == ["memory"]
    monkeypatch.setattr(c, "_request", lambda *a, **k: {"fields": [{"name": "content"}, "junk"]})
    assert c.column_names("a", "memory") == ["content"]


def test_a_trial_returns_its_key(platform):
    out = request_trial(platform.url, TEAM_DB)
    assert out["api_key"] == "trial-key-1" and out["database"] == TEAM_DB


@pytest.mark.parametrize("status", [409, 501])
def test_a_refused_trial_raises_with_its_status(platform, status):
    platform.trial_status = status
    with pytest.raises(InfinoError) as e:
        request_trial(platform.url, TEAM_DB)
    assert e.value.status == status


def test_a_trial_answer_without_a_key_is_refused(platform):
    platform.trial_without_key = True
    with pytest.raises(InfinoError, match="no api_key"):
        request_trial(platform.url, TEAM_DB)

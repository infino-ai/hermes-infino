"""`infino_personal_search`: the person's own data, searched only where the
turn is the owner's, and a session that searches it kept personal."""

from __future__ import annotations

import json

import pytest

from hermes_infino import privacy, tables
from hermes_infino.client import InfinoClient
from hermes_infino.provider import PERSONAL_SEARCH_TOOL, PERSONAL_RESULT_CELL_CHARS

from .helpers import DM, GROUP, PERSONAL_DB, hook_ctx, in_session, local_rows, make_manager, wait_for
from .test_personal import personal, texts

PRIVATE = "Dr. Lee sent the test results"


def offered(p) -> bool:
    return PERSONAL_SEARCH_TOOL in {s["name"] for s in p.get_tool_schemas()}


def search(p, **args) -> dict:
    return json.loads(p.handle_tool_call(PERSONAL_SEARCH_TOOL, args))


def conversation(session: str, text: str) -> dict:
    return {"session_id": session, "seq": 0, "role": "user", "content": text, "created_ms": 1}


# -- where it is offered ---------------------------------------------------------------

def test_offered_in_your_own_sessions(home, tmp_path):
    _, p = make_manager(tmp_path)
    assert offered(p)


@pytest.mark.parametrize("platform, context", [("telegram", "primary"), ("cron", "cron"), ("cli", "subagent")])
def test_not_offered_on_a_shared_gateway_or_to_a_job_or_subagent(home, tmp_path, platform, context):
    _, p = make_manager(tmp_path, platform=platform, agent_context=context)
    assert not offered(p)
    assert search(p, query="Dr. Lee")["ok"] is False


def test_a_personal_profile_offers_it_on_its_gateway_but_never_in_a_group(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    _, p = make_manager(tmp_path, platform="telegram")
    assert offered(p)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    in_session(monkeypatch, DM)
    assert search(p, query="Dr. Lee")["ok"] is True
    in_session(monkeypatch, GROUP)
    assert search(p, query="Dr. Lee") == {"ok": False, "error": "your personal data is not searched here: only in "
                                          "your own sessions, or on a profile that is one person's, and never in "
                                          "a group chat"}


# -- what it finds ------------------------------------------------------------------

def test_finds_your_personal_rows_and_none_of_the_teams(home, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    p._backend.append(tables.CONVERSATIONS, [conversation("standup", "Dr. Lee joins standup")])  # the team's
    out = search(p, query="Dr. Lee")
    assert out["ok"] and out["table"] == tables.CONVERSATIONS
    assert [r["session_id"] for r in out["rows"]] == ["digest"]


def test_searches_tool_results_and_traces_too(home, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [{**conversation("digest", ""), "role": "tool", "tool_name": "terminal",
                                              "tool_result": f"From: {PRIVATE}"}], personal=True)
    p._backend.append(tables.TRACES, [{"session_id": "digest", "kind": "model", "result": f"Triage: {PRIVATE}"}],
                      personal=True)
    assert [r["tool_result"] for r in search(p, query="Dr. Lee")["rows"]] == [f"From: {PRIVATE}"]
    assert [r["result"] for r in search(p, query="Dr. Lee", table=tables.TRACES)["rows"]] == [f"Triage: {PRIVATE}"]


def test_memory_shows_the_owners_own_only(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "dana")
    _, p = make_manager(tmp_path)
    p._backend.append(tables.MEMORY, [{"content": "dentist on Tuesday", "scope": "personal", "author": "dana"},
                                      {"content": "dentist on Friday", "scope": "personal", "author": "mallory"}])
    assert [r["content"] for r in search(p, query="dentist", table=tables.MEMORY)["rows"]] == ["dentist on Tuesday"]


def test_a_long_cell_is_cut_and_the_count_is_bounded(home, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation(f"s{i}", "lee " + "x" * 5_000) for i in range(3)],
                      personal=True)
    rows = search(p, query="lee", k=2)["rows"]
    assert len(rows) == 2 and all(len(r["content"]) == PERSONAL_RESULT_CELL_CHARS for r in rows)


def test_an_unknown_table_is_refused_by_name(home, tmp_path):
    _, p = make_manager(tmp_path)
    out = search(p, query="x", table="skills")
    assert out == {"ok": False, "error": "table must be one of memory, conversations, traces, not 'skills'"}


def test_signed_in_it_searches_the_personal_cloud_database(env, platform, tmp_path):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    _, p = make_manager(tmp_path)
    # A row only the cloud has: this person's work on another machine.
    platform.databases[PERSONAL_DB][tables.CONVERSATIONS].append(
        {tables.ROW_ID: "elsewhere", **conversation("laptop", PRIVATE)})
    assert [r["session_id"] for r in search(p, query="Dr. Lee")["rows"]] == ["laptop"]


def test_a_cloud_that_does_not_answer_leaves_this_machines_copy(env, platform, tmp_path, monkeypatch):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)

    def down(*args, **kwargs):
        raise TimeoutError("no answer")

    monkeypatch.setattr(InfinoClient, "bm25_search", down)
    assert [r["session_id"] for r in search(p, query="Dr. Lee")["rows"]] == ["digest"]


# -- what a search does to its session ---------------------------------------------------

def test_a_session_that_searches_is_personal_from_then_on(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="s1")
    hooks = hook_ctx()
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    search(p, query="Dr. Lee")
    p.sync_turn("What did Dr. Lee send?", "The test results.", session_id="s1")
    hooks["post_api_request"](session_id="s1", turn_id="t1", model="m", assistant_message="The test results.")
    hooks["on_session_end"](session_id="s1")
    p._drain()
    assert wait_for(lambda: len(texts(personal(tmp_path), tables.TRACES)) == 1)
    assert [r for r in local_rows(tmp_path, tables.CONVERSATIONS) if r["session_id"] == "s1"] == []
    assert local_rows(tmp_path, tables.TRACES) == []
    assert any("Dr. Lee" in t for t in texts(personal(tmp_path), tables.CONVERSATIONS))


def test_a_search_call_is_withheld_from_any_row_it_lands_in():
    args = {"query": "Dr. Lee", "table": "conversations"}
    assert privacy.is_personal_call(PERSONAL_SEARCH_TOOL, args)
    assert privacy.withhold(PERSONAL_SEARCH_TOOL, args)["query"] == privacy.WITHHELD
    assert privacy.withhold(PERSONAL_SEARCH_TOOL, "not json") == privacy.WITHHELD

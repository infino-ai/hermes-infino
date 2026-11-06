"""`infino_personal_search`: the person's own data, searched only where the
turn is the owner's, and a session that searches it kept personal."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_infino import backend, privacy, provider as provider_mod, tables
from hermes_infino.client import InfinoClient
from hermes_infino.provider import (MAX_PERSONAL_SEARCH_K, PERSONAL_RESULT_CELL_CHARS, PERSONAL_SEARCH_TOOL,
                                    InfinoMemoryProvider)

from .helpers import DM, GROUP, PERSONAL_DB, hook_ctx, in_session, local_rows, make_manager, requests_to, wait_for
from .test_personal import personal, texts

PRIVATE = "Dr. Lee sent the test results"
# More rows than a search may ever return.
MANY_ROWS = MAX_PERSONAL_SEARCH_K + 10
# A sync delay no test waits out: rows written meanwhile stay on this machine.
NEVER_S = 3600


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


def tool_names(mgr) -> set[str]:
    """The tools Hermes shows the model (memory_manager_tools.inject_memory_provider_tools)."""
    return {s["name"] for s in mgr.get_all_tool_schemas()}


def test_hermes_routes_the_call_and_shows_the_tool_only_where_offered(home, tmp_path):
    mgr, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    assert PERSONAL_SEARCH_TOOL in tool_names(mgr)
    out = json.loads(mgr.handle_tool_call(PERSONAL_SEARCH_TOOL, {"query": "Dr. Lee"}))
    assert out["ok"] and [r["session_id"] for r in out["rows"]] == ["digest"]
    shared, _ = make_manager(tmp_path, platform="telegram")
    assert PERSONAL_SEARCH_TOOL not in tool_names(shared)
    assert json.loads(shared.handle_tool_call(PERSONAL_SEARCH_TOOL, {"query": "Dr. Lee"}))["ok"] is False


def test_the_search_call_itself_never_reaches_the_team(home, tmp_path):
    mgr, p = make_manager(tmp_path, session_id="s1")
    hooks = hook_ctx()
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    args = {"query": "Dr. Lee"}
    result = mgr.handle_tool_call(PERSONAL_SEARCH_TOOL, args)
    assert PRIVATE in result
    hooks["post_tool_call"](session_id="s1", turn_id="t1", tool_name=PERSONAL_SEARCH_TOOL, tool_call_id="c1",
                            args=args, result=result, status="ok")
    hooks["on_session_end"](session_id="s1")
    messages = [{"role": "user", "content": "What did my doctor send?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": PERSONAL_SEARCH_TOOL, "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": "c1", "content": result},
                {"role": "assistant", "content": "The test results are ready."}]
    p.sync_turn("What did my doctor send?", "The test results are ready.", session_id="s1", messages=messages)
    p._drain()
    assert wait_for(lambda: len(texts(personal(tmp_path), tables.TRACES)) == 1)
    # Nothing of the session in the team's tables: not the question, the call, or what it found.
    assert [r for r in local_rows(tmp_path, tables.CONVERSATIONS) if r["session_id"] == "s1"] == []
    assert local_rows(tmp_path, tables.TRACES) == []
    # In the person's own, the call is kept without what it asked or found.
    rows = [r for r in personal(tmp_path).query_sql("SELECT * FROM conversations ORDER BY seq")
            if r["session_id"] == "s1"]
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert "Dr. Lee" not in rows[1]["tool_args"] and privacy.WITHHELD in rows[1]["tool_args"]
    assert rows[2]["tool_result"] == privacy.WITHHELD
    [trace] = personal(tmp_path).query_sql("SELECT * FROM traces")
    assert "Dr. Lee" not in trace["args"] and trace["result"] == privacy.WITHHELD


# -- corner cases: where it runs -------------------------------------------------------

def team_rows_of(tmp_path, session: str) -> list[dict]:
    return [r for r in local_rows(tmp_path, tables.CONVERSATIONS) if r["session_id"] == session]


def personal_rows_of(tmp_path, session: str) -> list[dict]:
    return [r for r in personal(tmp_path).query_sql("SELECT * FROM conversations") if r["session_id"] == session]


def test_a_cli_session_of_a_profile_that_also_serves_a_gateway_is_offered_it(home, tmp_path):
    make_manager(tmp_path, platform="telegram")  # the profile answers a gateway too
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    assert offered(p) and search(p, query="Dr. Lee")["ok"] is True


def test_a_group_chat_is_refused_even_where_the_owner_may_write_their_memory(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    _, p = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP, binding=SimpleNamespace(owner_writes=True))
    assert search(p, query="Dr. Lee")["ok"] is False


def test_a_flush_run_never_searches(home, tmp_path):
    _, p = make_manager(tmp_path, agent_context="flush")
    assert not offered(p) and search(p, query="Dr. Lee")["ok"] is False


def test_a_provider_that_has_not_started_refuses_it():
    out = json.loads(InfinoMemoryProvider().handle_tool_call(PERSONAL_SEARCH_TOOL, {"query": "Dr. Lee"}))
    assert out == {"ok": False, "error": "not configured"}


def test_turning_the_personal_profile_off_stops_it_on_the_gateway(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    _, p = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, DM)
    assert search(p, query="Dr. Lee")["ok"] is True
    home.setenv("INFINO_PERSONAL_PROFILE", "off")
    p._checked_at -= provider_mod.SETTINGS_RECHECK_S  # half a minute later, the next turn re-reads the settings
    p.prefetch("anything")
    assert search(p, query="Dr. Lee")["ok"] is False


# -- corner cases: what it is asked ---------------------------------------------------

@pytest.mark.parametrize("args", [{}, {"query": ""}, {"query": "   "}, {"query": None}])
def test_an_empty_query_is_refused(home, tmp_path, args):
    _, p = make_manager(tmp_path)
    assert search(p, **args) == {"ok": False, "error": "query is empty: give the words to search for"}


@pytest.mark.parametrize("k, expected", [(0, 1), (-5, 1), (1, 1), ("3", 3), (10_000, MAX_PERSONAL_SEARCH_K)])
def test_the_number_of_rows_is_kept_in_bounds(home, tmp_path, k, expected):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation(f"s{i}", "lee") for i in range(MANY_ROWS)], personal=True)
    assert len(search(p, query="lee", k=k)["rows"]) == expected


def test_a_count_that_is_not_a_number_is_refused(home, tmp_path):
    _, p = make_manager(tmp_path)
    assert search(p, query="lee", k="lots")["ok"] is False


@pytest.mark.parametrize("table", ["Conversations", "memory_vectors", "skills", " traces"])
def test_only_the_personal_tables_by_their_exact_names(home, tmp_path, table):
    _, p = make_manager(tmp_path)
    out = search(p, query="lee", table=table)
    assert out["ok"] is False and out["error"].startswith("table must be one of memory, conversations, traces")


def test_no_table_means_conversations(home, tmp_path):
    _, p = make_manager(tmp_path)
    assert search(p, query="lee", table=None)["table"] == tables.CONVERSATIONS


@pytest.mark.parametrize("query", ["O'Brien's 100% café 🚀", "x'; DROP TABLE conversations; --", '"quoted" (and) [brackets]'])
def test_any_text_is_a_query(home, tmp_path, query):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", "O'Brien's café")], personal=True)
    out = search(p, query=query)
    assert out["ok"] is True
    assert personal(tmp_path).query_sql("SELECT count(*) AS n FROM conversations")[0]["n"] == 1


def test_nothing_found_is_an_empty_answer(home, tmp_path):
    _, p = make_manager(tmp_path)
    assert search(p, query="nothing like this") == {"ok": True, "table": tables.CONVERSATIONS, "rows": []}


# -- corner cases: what it answers --------------------------------------------------

def test_a_row_found_in_two_columns_is_answered_once(home, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [{**conversation("digest", "Lee called"), "tool_result": "Lee's results"}],
                      personal=True)
    assert len(search(p, query="Lee")["rows"]) == 1


def test_the_better_match_comes_first(home, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("weak", "Lee"), conversation("strong", "Dr Lee sent results")],
                      personal=True)
    assert [r["session_id"] for r in search(p, query="Dr Lee results")["rows"]] == ["strong", "weak"]


def test_empty_cells_are_left_out_and_times_kept(home, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    [row] = search(p, query="Dr. Lee")["rows"]
    assert "tool_name" not in row and "tool_result" not in row and row["created_ms"] == 1


def test_a_personal_profiles_gateway_sees_all_of_its_memory(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    _, p = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, DM)
    p._backend.append(tables.MEMORY, [{"content": "dentist on Tuesday", "scope": "personal", "author": "dana"},
                                      {"content": "dentist on Friday", "scope": "personal", "author": "@dana_phone"}])
    assert len(search(p, query="dentist", table=tables.MEMORY)["rows"]) == 2


def test_signed_in_without_a_personal_database_it_searches_this_machine(env, platform, tmp_path):
    _, p = make_manager(tmp_path)
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    assert [r["session_id"] for r in search(p, query="Dr. Lee")["rows"]] == ["digest"]
    assert requests_to(platform, "bm25_search") == 0


def test_a_row_written_here_since_the_last_sync_is_found_with_the_clouds(env, platform, tmp_path, monkeypatch):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    _, p = make_manager(tmp_path)
    monkeypatch.setattr(backend, "SYNC_DELAY_S", NEVER_S)
    platform.databases[PERSONAL_DB][tables.CONVERSATIONS].append(
        {tables.ROW_ID: "elsewhere", **conversation("laptop", PRIVATE)})
    p._backend.append(tables.CONVERSATIONS, [conversation("here", PRIVATE)], personal=True)
    assert sorted(r["session_id"] for r in search(p, query="Dr. Lee")["rows"]) == ["here", "laptop"]
    p._backend.mirror.sync(wait=True)  # now the cloud has it too: still once
    assert sorted(r["session_id"] for r in search(p, query="Dr. Lee")["rows"]) == ["here", "laptop"]


# -- corner cases: what a search does to its session ---------------------------------------

def test_a_refused_search_leaves_the_session_the_teams(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path, platform="telegram", session_id="g1")
    in_session(monkeypatch, DM)
    assert search(p, query="Dr. Lee")["ok"] is False
    p.sync_turn("hi", "hello", session_id="g1")
    p._drain()
    assert len(team_rows_of(tmp_path, "g1")) == 2 and personal_rows_of(tmp_path, "g1") == []


@pytest.mark.parametrize("args", [{"query": ""}, {"query": "lee", "table": "skills"}])
def test_a_search_refused_for_its_arguments_leaves_the_session_the_teams(home, tmp_path, args):
    _, p = make_manager(tmp_path, session_id="s1")
    assert search(p, **args)["ok"] is False
    p.sync_turn("hi", "hello", session_id="s1")
    p._drain()
    assert len(team_rows_of(tmp_path, "s1")) == 2


def test_a_search_that_finds_nothing_still_makes_the_session_personal(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="s1")
    assert search(p, query="nothing like this")["rows"] == []
    p.sync_turn("did my doctor write?", "No.", session_id="s1")
    p._drain()
    assert team_rows_of(tmp_path, "s1") == [] and len(personal_rows_of(tmp_path, "s1")) == 2


def test_a_search_that_fails_still_makes_the_session_personal(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path, session_id="s1")

    def broken(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(p._backend, "search_personal", broken)
    assert search(p, query="Dr. Lee")["ok"] is False
    p.sync_turn("did my doctor write?", "I could not look.", session_id="s1")
    p._drain()
    assert team_rows_of(tmp_path, "s1") == [] and len(personal_rows_of(tmp_path, "s1")) == 2


def test_a_search_in_one_session_leaves_another_the_teams(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="s1")
    search(p, query="Dr. Lee")
    p.sync_turn("standup at ten?", "Yes.", session_id="s2")
    p._drain()
    assert len(team_rows_of(tmp_path, "s2")) == 2


def test_after_a_session_switch_the_search_marks_the_new_session(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="s1")
    p.on_session_switch("s2")
    search(p, query="Dr. Lee")
    p.sync_turn("did my doctor write?", "No.", session_id="s2")
    p.sync_turn("standup at ten?", "Yes.", session_id="s1")
    p._drain()
    assert team_rows_of(tmp_path, "s2") == [] and len(team_rows_of(tmp_path, "s1")) == 2


def test_the_session_stays_personal_after_a_restart(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="s1")
    search(p, query="Dr. Lee")
    p.sync_turn("did my doctor write?", "No.", session_id="s1")
    p._drain()
    p.shutdown()
    backend._backends.clear()  # a new process: nothing remembered about any session
    privacy._sessions.clear()
    _, again = make_manager(tmp_path, session_id="s1")
    again.sync_turn("and the bank?", "Nothing from the bank.", session_id="s1")
    again._drain()
    assert team_rows_of(tmp_path, "s1") == [] and len(personal_rows_of(tmp_path, "s1")) == 4


def test_even_if_the_session_were_not_marked_the_call_reaches_no_team_row(home, tmp_path):
    mgr, p = make_manager(tmp_path, session_id="s1")
    hooks = hook_ctx()
    p._backend.append(tables.CONVERSATIONS, [conversation("digest", PRIVATE)], personal=True)
    args = {"query": "Dr. Lee"}
    result = mgr.handle_tool_call(PERSONAL_SEARCH_TOOL, args)
    privacy._sessions.clear()  # the mark lost (another process, or evicted): the session reads as the team's
    hooks["post_tool_call"](session_id="s1", turn_id="t1", tool_name=PERSONAL_SEARCH_TOOL, tool_call_id="c1",
                            args=args, result=result, status="ok")
    hooks["on_session_end"](session_id="s1")
    messages = [{"role": "user", "content": "What did my doctor send?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": PERSONAL_SEARCH_TOOL, "arguments": json.dumps(args)}}]},
                {"role": "tool", "tool_call_id": "c1", "content": result}]
    p.sync_turn("What did my doctor send?", "", session_id="s1", messages=messages)
    p._drain()
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    team = team_rows_of(tmp_path, "s1") + local_rows(tmp_path, tables.TRACES)
    assert team and not [r for r in team if "Lee" in json.dumps(r, default=str)]


def test_a_search_call_is_withheld_from_any_row_it_lands_in():
    args = {"query": "Dr. Lee", "table": "conversations"}
    assert privacy.is_personal_call(PERSONAL_SEARCH_TOOL, args)
    assert privacy.withhold(PERSONAL_SEARCH_TOOL, args)["query"] == privacy.WITHHELD
    assert privacy.withhold(PERSONAL_SEARCH_TOOL, "not json") == privacy.WITHHELD

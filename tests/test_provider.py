"""The memory provider on this machine: its tools, the rows a turn becomes,
and its lifecycle."""

from __future__ import annotations

import asyncio
import fcntl
import json
import sys
import threading
import time

import pytest

from agent import memory_provider
from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.turn_context import append_notes_to_multimodal_content, compose_multimodal_context_part
from cron.scheduler import _teardown_cron_agent
from gateway.owner_identity import host_owner_subject
from gateway.platforms.api_server_memory_sessions import ApiServerMemorySessions
from gateway.run import GatewayRunner
from hermes_cli import cli_shutdown, oneshot

from hermes_infino import backend, config, embed, mirror, privacy, provider as provider_mod, skills, tables
from hermes_infino.local import VECTORS
from hermes_infino.provider import CRON_PLATFORM, PERSONAL_SEARCH_TOOL, InfinoMemoryProvider

from .helpers import (DM, GROUP, SKILL_MD, TEAM_DB, WRITE_WAIT_S, agent_tool_call, end_session, fake_embed,
                      hermes_agent, hermes_dispatch, hook_ctx, in_session, local, local_rows, make_manager,
                      current, personal_rows, superseded, wait_for, webhook_delivery)

# What a provider declares before it starts, which is what Hermes routes by:
# every tool, the personal search included (it is shown only where offered).
TOOLS = ("infino_remember", "infino_forget", "infino_publish_skill", "infino_personal_search")
# A scheduled job's run, as Hermes's scheduler names its session.
CRON_SESSION = "cron_a1b2c3_20261105_070000"
# What a recall put in front of the model: the provider's own block, a personal line in it.
RECALLED = "## Relevant memory (Infino)\n- [personal] dentist on Tuesday"
# The author Hermes names for a turn another agent wrote into a profile's Bot
# Chat (agent/turn_author.py): a peer's agent on another machine (`hermes peer
# dm`), as on_turn_start is handed it.
PEER_AGENT = {"author_id": "bot:spark/researcher", "author_name": "researcher", "author_is_bot": True}
# How a Desktop session names this machine's own user to its provider (gateway/owner_identity.py).
DESKTOP_USER = "uid:1000"
# A session end's bound on waiting for the cloud, short enough for a test.
QUICK_FLUSH_S = 3.0
# The most a soft release may take: well under a settle window and a sync.
SOFT_RELEASE_S = 1.0
# A background sync's delay no test waits out: the process (a one-shot run, a
# gateway stopping) exits through os._exit before it runs.
NO_BACKGROUND_SYNC_S = 3600.0
# The most a test waits for one of Hermes's teardown threads to end.
TEARDOWN_JOIN_S = 4 * QUICK_FLUSH_S
# A cached gateway agent's key, and the api_server's bounds on its parked sessions.
GATEWAY_KEY = "agent:main:telegram:dm:1"
API_SESSIONS = 8
API_IDLE_S = 3600.0
# The job of a scheduled run's cleanup.
CRON_JOB = "a1b2c3"
# A turn, as Hermes hands it to its memory manager at the turn's end.
LAST_TURN = ("run the tests", "2 fail")


def call(mgr, tool: str, **args) -> dict:
    return json.loads(mgr.handle_tool_call(tool, args))


def turn_rows(tmp_path) -> list[dict]:
    return sorted(local_rows(tmp_path, tables.CONVERSATIONS), key=lambda r: (r["session_id"], r["seq"]))


@pytest.fixture
def mgr(home, tmp_path):
    return make_manager(tmp_path)


# -- what the provider offers ------------------------------------------------------

def test_the_tools_and_their_required_arguments():
    schemas = {s["name"]: s for s in InfinoMemoryProvider().get_tool_schemas()}
    assert tuple(schemas) == TOOLS
    assert schemas["infino_remember"]["parameters"]["required"] == ["content"]
    assert schemas["infino_forget"]["parameters"]["properties"]["scope"]["enum"] == ["team", "personal"]
    assert schemas["infino_publish_skill"]["parameters"]["required"] == ["name"]


def test_the_setup_fields_are_the_settings():
    assert InfinoMemoryProvider().get_config_schema() == config.config_schema()


def test_available_with_every_library_and_says_which_is_missing(monkeypatch):
    p = InfinoMemoryProvider()
    assert p.is_available()
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # an import of it now fails
    assert not p.is_available() and p.unavailable_reason() == "missing Python packages: onnxruntime"


def test_before_initialize_it_offers_no_prompt_and_refuses_tools():
    p = InfinoMemoryProvider()
    assert p.system_prompt_block() == ""
    assert p.prefetch("anything") == ""
    assert json.loads(p.handle_tool_call("infino_remember", {"content": "x"})) == {"ok": False, "error": "not configured"}


def test_an_unknown_tool_is_refused(mgr):
    _, p = mgr
    assert json.loads(p.handle_tool_call("infino_teleport", {})) == {"ok": False, "error": "unknown tool infino_teleport"}


def test_a_tool_call_missing_its_argument_reports_it(mgr):
    m, _ = mgr
    out = call(m, "infino_remember")
    assert out["ok"] is False and "KeyError" in out["error"]


# -- memory ----------------------------------------------------------------------

def test_a_remembered_row_says_who_where_and_how(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "priya")
    m, _ = make_manager(tmp_path, agent_identity="ops", session_id="s42")
    assert call(m, "infino_remember", content="the VPN is needed for staging") == {"ok": True, "scope": "team"}
    [row] = local_rows(tmp_path, tables.MEMORY)
    assert (row["author"], row["profile"], row["session_id"], row["origin"], row["scope"], row[tables.CHAT]) == (
        "priya", "ops", "s42", "remember", "team", "")


def test_personal_memory_goes_to_the_personal_database_only(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="my dentist is on Tuesday", scope="personal")
    assert local_rows(tmp_path, tables.MEMORY) == []
    assert [r["content"] for r in personal_rows(tmp_path)] == ["my dentist is on Tuesday"]


def test_a_forget_scoped_to_the_team_leaves_personal_memory(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="same words", scope="personal")
    call(m, "infino_remember", content="same words", scope="team")
    out = call(m, "infino_forget", content="same words", scope="team")
    assert out == {"ok": True, "scope": "team", "matched": 1}
    assert [r["content"] for r in personal_rows(tmp_path)] == ["same words"]


def test_a_forget_with_no_scope_reaches_both(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="same words", scope="personal")
    call(m, "infino_remember", content="same words", scope="team")
    assert call(m, "infino_forget", content="same words")["matched"] == 2


def test_a_forget_needs_the_exact_text(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="use pnpm 9")
    assert call(m, "infino_forget", content="use pnpm")["matched"] == 0
    assert len(local_rows(tmp_path, tables.MEMORY)) == 1


def test_a_removed_builtin_memory_is_kept_superseded(mgr, tmp_path):
    _, p = mgr
    p.on_memory_write("add", "memory", "the build uses make")
    p.on_memory_write("remove", "memory", "", {"previous_content": "the build uses make"})
    p.on_memory_write("remove", "memory", "", None)  # nothing named: nothing to do
    p._drain()
    rows = local_rows(tmp_path, tables.MEMORY)
    assert current(rows) == []
    # Removed, so nothing replaced it.
    assert [(r["content"], r[tables.SUPERSEDED_BY]) for r in superseded(rows)] == [("the build uses make", "")]


def test_a_chat_of_no_known_type_writes_none_of_the_owners_personal_memory_to_the_team(home, tmp_path, monkeypatch):
    # A profile that is one person's, behind a relay: the owner's own DM may come with no chat type.
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    hermes_dispatch(monkeypatch)
    in_session(monkeypatch, {**DM, "HERMES_SESSION_CHAT_TYPE": ""})
    m, p = make_manager(tmp_path, platform="telegram", session_id="relay-1")
    m.on_turn_start(1, "Remember my PIN, it's personal.")
    assert call(m, "infino_remember", content="Dana's PIN is 4321", scope="personal") == {
        "ok": False, "error": provider_mod.NO_PERSONAL_IN_GROUP}
    # Hermes's own memory tool, as the agent calls it: it tells the provider.
    added = agent_tool_call("relay-1", "memory", {"action": "add", "target": "user", "content": "PIN is 4321"}, m,
                            platform="telegram")
    assert added["success"]
    p._drain()
    assert not [r for r in local_rows(tmp_path, tables.MEMORY) + personal_rows(tmp_path) if "4321" in r["content"]]
    # The profile is not marked as one others talked to for good, and the
    # owner's next session still does not take the entry for theirs.
    assert not (config.state_dir(tmp_path) / provider_mod.OTHERS_MARK).exists()
    in_session(monkeypatch, {})
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    at_the_terminal._drain()
    assert not [r for r in personal_rows(tmp_path) if "4321" in r["content"]]


def test_a_users_entry_without_a_name_is_kept_as_written(mgr, tmp_path):
    _, p = mgr
    p.on_memory_write("add", "user", "Prefers metric units")
    p._drain()
    assert [r["content"] for r in personal_rows(tmp_path)] == ["Prefers metric units"]


def test_many_background_writes_all_land(mgr, tmp_path):
    _, p = mgr
    for i in range(12):
        p.on_memory_write("add", "memory", f"fact {i}")
    p._drain()
    assert sorted(r["content"] for r in local_rows(tmp_path, tables.MEMORY)) == sorted(f"fact {i}" for i in range(12))


def test_recall_status_is_per_question(mgr):
    m, p = mgr
    call(m, "infino_remember", content="the deploy freeze starts December 20")
    p.prefetch("deploy freeze")
    assert p.recall_status().count == 1 and p.recall_status().provider_label == "Infino"
    assert p.prefetch("   ") == "" and p.recall_status() is None


# -- conversations -----------------------------------------------------------------

def test_a_turn_without_its_messages_is_its_two_texts(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("what time is standup?", "Ten.", session_id="s1")
    p._drain()
    assert [(r["seq"], r["role"], r["content"]) for r in turn_rows(tmp_path)] == [
        (0, "user", "what time is standup?"), (1, "assistant", "Ten.")]


def test_rows_from_every_message_shape(mgr, tmp_path):
    _, p = mgr
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": [{"type": "text", "text": "look at"}, {"type": "image_url", "image_url": {}},
                                     {"type": "text", "text": "this"}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "function": {"name": "terminal", "arguments": "{\"cmd\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "file text"},
        {"role": "tool", "tool_call_id": "b", "name": "terminal", "content": "a.txt"},
        {"role": "assistant", "content": ""},  # nothing said and nothing called: no row
        {"role": "assistant", "content": "done"},
    ]
    p.sync_turn("look at this", "done", session_id="s1", messages=messages)
    p._drain()
    rows = turn_rows(tmp_path)
    # The turn starts at its user message; the system prompt is no part of it.
    assert [(r["seq"], r["role"]) for r in rows] == [(0, "user"), (1, "assistant"), (2, "tool"), (3, "tool"),
                                                    (5, "assistant")]
    assert rows[0]["content"] == "look at\n\nthis"  # an image part leaves an empty line
    assert rows[1]["tool_name"] == "read_file,terminal"
    assert [r["tool_name"] for r in rows[2:4]] == ["read_file", "terminal"]
    assert rows[2]["tool_result"] == "file text" and rows[2]["content"] == ""
    assert rows[1]["platform"] == "cli" and rows[1]["author"] == ""  # only the person's message names them


def test_a_huge_cell_is_cut(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("x" * (provider_mod.MAX_CELL_CHARS + 10), "ok", session_id="s1")
    p._drain()
    assert len(turn_rows(tmp_path)[0]["content"]) == provider_mod.MAX_CELL_CHARS


def test_each_session_counts_its_own_messages(mgr, tmp_path):
    _, p = mgr
    a = [{"role": "user", "content": "a1"}, {"role": "assistant", "content": "a2"}]
    b = [{"role": "user", "content": "b1"}, {"role": "assistant", "content": "b2"}]
    p.sync_turn("a1", "a2", session_id="A", messages=a)
    p.sync_turn("b1", "b2", session_id="B", messages=b)
    p.sync_turn("a3", "a4", session_id="A", messages=a + [{"role": "user", "content": "a3"},
                                                          {"role": "assistant", "content": "a4"}])
    p._drain()
    assert [(r["session_id"], r["seq"]) for r in turn_rows(tmp_path)] == [("A", 0), ("A", 1), ("A", 2), ("A", 3),
                                                                         ("B", 0), ("B", 1)]


def test_a_new_session_id_carries_into_memory_rows(mgr, tmp_path):
    m, p = mgr
    p.on_session_switch("s2")
    call(m, "infino_remember", content="after the switch")
    assert local_rows(tmp_path, tables.MEMORY)[0]["session_id"] == "s2"


def test_session_end_and_shutdown_in_local_mode(mgr, tmp_path):
    _, p = mgr
    p.on_memory_write("add", "memory", "kept")
    p.on_session_end([])
    end_session(p)
    assert len(local_rows(tmp_path, tables.MEMORY)) == 1


def test_a_published_skill_is_kept_locally_without_an_account(mgr, tmp_path):
    m, _ = mgr
    skill = tmp_path / "skills" / "release-checklist" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL_MD)
    assert call(m, "infino_publish_skill", name="release-checklist")["ok"]
    [row] = local_rows(tmp_path, tables.SKILLS)
    assert row["content"] == SKILL_MD and row["description"] == "Steps before a release."


# -- scope, resumed sessions, and a model that cannot run ------------------------------

@pytest.mark.parametrize("scope", ["Personal", "private", "everyone"])
def test_a_scope_outside_the_enum_never_lands_in_the_teams_database(mgr, tmp_path, scope):
    m, _ = mgr
    out = call(m, "infino_remember", content="my home address is 1 Main St", scope=scope)
    assert out["ok"] is False or local_rows(tmp_path, tables.MEMORY) == []


def test_a_scope_in_another_case_is_understood(mgr, tmp_path):
    m, _ = mgr
    assert call(m, "infino_remember", content="my dentist is Tuesday", scope=" Personal ") == {
        "ok": True, "scope": "personal"}
    assert [r["content"] for r in personal_rows(tmp_path)] == ["my dentist is Tuesday"]


def test_an_unknown_scope_is_refused_by_name(mgr, tmp_path):
    m, _ = mgr
    out = call(m, "infino_forget", content="x", scope="private")
    assert out == {"ok": False, "error": "scope must be 'team' or 'personal', not 'private'"}


def test_a_turn_without_its_messages_continues_the_sessions_count(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("one", "two", session_id="s1")
    p.sync_turn("three", "four", session_id="s1")
    p._drain()
    assert [(r["seq"], r["content"]) for r in turn_rows(tmp_path)] == [(0, "one"), (1, "two"), (2, "three"),
                                                                     (3, "four")]


def turn(user: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def test_a_session_compacted_in_place_keeps_being_stored(mgr, tmp_path):
    _, p = mgr
    history = turn("q1", "a1") + turn("q2", "a2") + turn("q3", "a3")
    for i in range(3):
        p.sync_turn(f"q{i + 1}", f"a{i + 1}", session_id="s1", messages=history[:2 * (i + 1)])
    # Hermes compacts in place: same session id, a summary and the tail.
    compacted = [{"role": "user", "content": "[summary of q1-q2]"}] + turn("q3", "a3") + turn("q4", "a4")
    p.on_session_switch("s1", reset=False, reason="compression")
    p.sync_turn("q4", "a4", session_id="s1", messages=compacted)
    p._drain()
    assert [(r["seq"], r["content"]) for r in turn_rows(tmp_path)] == [
        (0, "q1"), (1, "a1"), (2, "q2"), (3, "a2"), (4, "q3"), (5, "a3"), (6, "q4"), (7, "a4")]


def test_a_turn_after_undo_is_stored(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("q1", "a1", session_id="s1", messages=turn("q1", "a1"))
    p.sync_turn("q2", "a2", session_id="s1", messages=turn("q1", "a1") + turn("q2", "a2"))
    p.on_session_switch("s1", rewound=True)  # /undo took q2 back
    p.sync_turn("q2 again", "a2 again", session_id="s1", messages=turn("q1", "a1") + turn("q2 again", "a2 again"))
    p._drain()
    assert [r["content"] for r in turn_rows(tmp_path)] == ["q1", "a1", "q2", "a2", "q2 again", "a2 again"]


def test_a_branched_session_stores_only_its_own_turns(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("q1", "a1", session_id="s1", messages=turn("q1", "a1"))
    p.on_session_switch("s1-branch")  # /branch: a new id, the history copied
    p.sync_turn("q2", "a2", session_id="s1-branch", messages=turn("q1", "a1") + turn("q2", "a2"))
    p._drain()
    assert [(r["session_id"], r["seq"], r["content"]) for r in turn_rows(tmp_path)] == [
        ("s1", 0, "q1"), ("s1", 1, "a1"), ("s1-branch", 0, "q2"), ("s1-branch", 1, "a2")]


def test_the_turn_starts_at_its_own_user_message(mgr, tmp_path):
    _, p = mgr
    # A message the turn added in the user's role (a steer) after the user's own.
    messages = turn("q1", "a1") + [{"role": "user", "content": "fix the build"},
                                   {"role": "user", "content": "(also: run the linter)"},
                                   {"role": "assistant", "content": "done"}]
    p.sync_turn("q1", "a1", session_id="s1", messages=turn("q1", "a1"))
    p.sync_turn("fix the build", "done", session_id="s1", messages=messages)
    p._drain()
    assert [r["content"] for r in turn_rows(tmp_path)][2:] == ["fix the build", "(also: run the linter)", "done"]


def test_a_turn_corrected_while_it_ran_is_stored_from_its_start(mgr, tmp_path):
    # A correction sent mid-turn is a user message of its own, and Hermes adds it
    # to the turn's text it hands sync_turn (agent/turn_iteration_prep.py).
    _, p = mgr
    messages = [{"role": "user", "content": "fix bug"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "function": {"name": "terminal", "arguments": "{\"cmd\": \"git status\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "on branch dev"},
                {"role": "user", "content": "use main"},
                {"role": "assistant", "content": "done"}]
    p.sync_turn("fix bug\n\nUser correction during the turn: use main", "done", session_id="s1", messages=messages)
    p._drain()
    rows = turn_rows(tmp_path)
    assert rows[0]["content"] == "fix bug"
    assert [r["role"] for r in rows] == ["user", "assistant", "tool", "user", "assistant"]


def test_an_image_turn_stores_none_of_its_recalled_memory(mgr, tmp_path):
    # Hermes keeps a multimodal turn's recall as a text part of the user message
    # (agent/turn_context.py), where a string turn keeps it out of the message.
    _, p = mgr
    parts = [{"type": "text", "text": "what is in this picture?"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    assert append_notes_to_multimodal_content(parts, compose_multimodal_context_part(RECALLED, ""))
    messages = [{"role": "user", "content": parts}, {"role": "assistant", "content": "A cat."}]
    p.sync_turn(_summarize_user_message_for_log(parts, sep="\n"), "A cat.", session_id="s1", messages=messages)
    p._drain()
    rows = turn_rows(tmp_path)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"].startswith("what is in this picture?")
    assert "dentist" not in rows[0]["content"] and "memory-context" not in rows[0]["content"]


def test_a_personal_memory_call_is_logged_without_its_text(mgr, tmp_path):
    _, p = mgr
    messages = [{"role": "user", "content": "remember my dentist is Tuesday"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {
                    "name": "infino_remember",
                    "arguments": json.dumps({"content": "dentist is Tuesday", "scope": "personal"})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true, \"scope\": \"personal\"}"},
                {"role": "assistant", "content": "Saved."}]
    p.sync_turn("remember my dentist is Tuesday", "Saved.", session_id="s1", messages=messages)
    p._drain()
    rows = turn_rows(tmp_path)
    assert "dentist is Tuesday" not in rows[1]["tool_args"] and privacy.WITHHELD in rows[1]["tool_args"]
    assert rows[2]["tool_result"] == privacy.WITHHELD
    assert rows[0]["content"] == "remember my dentist is Tuesday"  # what was said is the conversation


def test_a_memory_is_capped(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="x" * (provider_mod.MAX_CELL_CHARS + 5))
    assert len(local_rows(tmp_path, tables.MEMORY)[0]["content"]) == provider_mod.MAX_CELL_CHARS


def test_a_skill_too_large_to_share_is_refused(mgr, tmp_path, monkeypatch):
    m, _ = mgr
    monkeypatch.setattr(skills, "MAX_SKILL_CHARS", 10)
    skill = tmp_path / "skills" / "big" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL_MD)
    out = call(m, "infino_publish_skill", name="big")
    assert out["ok"] is False and "the most a team skill may be is 10" in out["error"]
    assert local_rows(tmp_path, tables.SKILLS) == []


def test_session_end_only_starts_a_sync(env, platform, tmp_path, monkeypatch):
    m, p = make_manager(tmp_path)
    monkeypatch.setattr(p._backend, "flush", lambda: pytest.fail("session end waited for the cloud"))
    started = []
    monkeypatch.setattr(p._backend, "schedule_sync", lambda *a, **k: started.append(1))
    p.on_session_end([])
    assert started == [1]


def test_a_running_process_picks_up_a_sign_in(home, platform, tmp_path, monkeypatch):
    m, p = make_manager(tmp_path)
    assert not p._backend.signed_in
    config.write_credentials({"url": platform.url, "api_key": "k", "team_database": "team-test"}, tmp_path)
    p._checked_at -= provider_mod.SETTINGS_RECHECK_S  # half a minute later
    p.prefetch("anything")
    assert p._backend.signed_in
    assert wait_for(lambda: tables.MEMORY in platform.databases.get("team-test", {}))
    call(m, "infino_remember", content="written after the sign-in")
    end_session(p)
    assert [r["content"] for r in platform.rows("team-test", tables.MEMORY)] == ["written after the sign-in"]


def test_on_a_gateway_the_prompt_names_no_one(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "alice")
    _, p = make_manager(tmp_path, platform="telegram", user_name="whoever-built-the-agent")
    block = p.system_prompt_block()
    assert "You are working with" not in block and "Several people may talk to you here" in block


# -- the owner's own surfaces, and where several parties share a session -------------

@pytest.mark.parametrize("platform", ["desktop", "acp"])
def test_the_desktop_chat_and_an_editor_are_the_owners_own_sessions(home, tmp_path, platform):
    # Hermes's Desktop chat panel runs as "desktop", an editor's session (ACP) as "acp".
    home.setenv("INFINO_AUTHOR", "dana")
    m, p = make_manager(tmp_path, platform=platform)
    assert call(m, "infino_remember", content="my dentist is on Tuesday", scope="personal") == {
        "ok": True, "scope": "personal"}
    assert "[personal] my dentist is on Tuesday" in p.prefetch("dentist")
    assert PERSONAL_SEARCH_TOOL in {s["name"] for s in p.get_tool_schemas()}
    assert call(m, PERSONAL_SEARCH_TOOL, query="dentist")["ok"] is True
    assert "You are working with dana." in p.system_prompt_block()
    assert not (config.state_dir(tmp_path) / provider_mod.GATEWAY_MARK).exists()


def test_a_scheduled_jobs_rows_are_its_owners(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "dana")
    m, _ = make_manager(tmp_path, platform="cron", agent_context="cron", session_id=CRON_SESSION)
    assert call(m, "infino_remember", content="the weekly report ran")["ok"]
    [row] = local_rows(tmp_path, tables.MEMORY)
    assert row["author"] == "dana"
    # A job is the owner's own: it never marks the profile as answering a gateway.
    assert not (config.state_dir(tmp_path) / provider_mod.GATEWAY_MARK).exists()


def test_a_gateway_marks_the_profile_with_its_platform(home, tmp_path):
    make_manager(tmp_path, platform="telegram")
    make_manager(tmp_path, platform="slack", session_id="s2")
    make_manager(tmp_path, platform="telegram", session_id="s3")
    assert (config.state_dir(tmp_path) / provider_mod.GATEWAY_MARK).read_text().split() == ["telegram", "slack"]


@pytest.mark.parametrize("platform", ["bot_room", "a2a"])
def test_a_group_chat_room_or_another_profiles_turn_keeps_out_personal_memory(home, tmp_path, platform):
    # A hosted Group Chat room ("bot_room") and a turn another profile forwards
    # ("a2a") have several parties, even on a profile that is one person's.
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    m, p = make_manager(tmp_path, platform=platform)
    p._backend.append(tables.MEMORY, [{"content": "dentist on Tuesday", "scope": "personal", "author": "dana"}])
    assert "dentist" not in p.prefetch("dentist")
    assert PERSONAL_SEARCH_TOOL not in {s["name"] for s in p.get_tool_schemas()}
    assert call(m, PERSONAL_SEARCH_TOOL, query="dentist")["ok"] is False
    assert call(m, "infino_remember", content="my bank PIN is 1234", scope="personal")["ok"] is False
    p.on_memory_write("add", "user", "Prefers short answers")  # Hermes's USER.md, written in the room
    p._drain()
    assert [r["content"] for r in personal_rows(tmp_path)] == ["dentist on Tuesday"]
    assert local_rows(tmp_path, tables.MEMORY) == []
    assert "no personal memory is kept here" in p.system_prompt_block()


@pytest.mark.parametrize("platform", ["desktop", "local", "cli"])
def test_a_turn_another_agent_wrote_keeps_out_personal_memory(home, tmp_path, platform):
    # Bot Mode's message_agent and `hermes peer dm` run another agent's message
    # as a turn of this profile's Bot Chat, on the owner's own surface.
    m, p = make_manager(tmp_path, platform=platform)
    assert call(m, "infino_remember", content="Dana is interviewing at Acme", scope="personal")["ok"]
    m.on_turn_start(2, "What are Dana's plans?", **PEER_AGENT)
    assert "Acme" not in p.prefetch("Dana plans Acme")
    assert call(m, PERSONAL_SEARCH_TOOL, query="Acme")["ok"] is False
    assert call(m, "infino_remember", content="researcher's quota is 10k", scope="personal")["ok"] is False
    assert [r["content"] for r in personal_rows(tmp_path)] == ["Dana is interviewing at Acme"]
    m.on_turn_start(3, "What are Dana's plans?")  # the owner's next turn is the owner's again
    assert "Acme" in p.prefetch("Dana plans Acme")


def test_personal_memory_saved_in_the_desktop_is_recalled_at_the_terminal(home, tmp_path):
    m, p = make_manager(tmp_path, platform="desktop", agent_identity="default", user_id=DESKTOP_USER)
    assert call(m, "infino_remember", content="my dentist is on Tuesday", scope="personal")["ok"]
    _, at_the_terminal = make_manager(tmp_path, platform="cli", agent_identity="default", session_id="s2")
    assert "dentist" in at_the_terminal.prefetch("dentist")
    assert DESKTOP_USER not in p.system_prompt_block()  # "You are working with default."


def test_personal_memory_kept_under_this_machines_user_is_still_the_owners(home, tmp_path):
    # What the Desktop saved before its rows were credited to the owner by name.
    _, at_the_terminal = make_manager(tmp_path, platform="cli", agent_identity="default")
    at_the_terminal._backend.append(tables.MEMORY, [{"content": "my locker is 12", "scope": "personal",
                                                     "author": host_owner_subject()}])
    assert "locker" in at_the_terminal.prefetch("locker")


def write_skill(tmp_path) -> None:
    skill = tmp_path / "skills" / "release-checklist" / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(SKILL_MD)


@pytest.mark.parametrize("platform, session", [("telegram", {**DM, "HERMES_SESSION_USER_NAME": "mallory"}),
                                               ("bot_room", {})])
def test_only_the_owner_publishes_a_skill_to_the_team(home, tmp_path, monkeypatch, platform, session):
    # A shared profile's gateway answers whoever writes in; a room has several parties.
    write_skill(tmp_path)
    m, _ = make_manager(tmp_path, platform=platform)
    in_session(monkeypatch, session)
    out = call(m, "infino_publish_skill", name="release-checklist")
    assert out["ok"] is False and "owner" in out["error"]
    assert local_rows(tmp_path, tables.SKILLS) == []


def test_another_agents_turn_does_not_publish_a_skill(home, tmp_path):
    write_skill(tmp_path)
    m, _ = make_manager(tmp_path, platform="desktop")
    m.on_turn_start(1, "Share your release checklist.", **PEER_AGENT)
    assert call(m, "infino_publish_skill", name="release-checklist")["ok"] is False
    m.on_turn_start(2, "Share your release checklist.")  # the owner may
    assert call(m, "infino_publish_skill", name="release-checklist")["ok"] is True


def test_a_switched_sessions_traces_carry_the_owner(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "dana")
    m, _ = make_manager(tmp_path, agent_identity="ops", session_id="s1")
    hooks = hook_ctx()
    m.on_session_switch("s1-branch", parent_session_id="s1", reset=False, reason="branch")
    hooks["post_api_request"](session_id="s1-branch", turn_id="t1", model="m")
    hooks["on_session_end"](session_id="s1-branch")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert (row["author"], row["profile"]) == ("dana", "ops")


# -- one profile answering several people over DMs ---------------------------------

def dm_from(monkeypatch, name: str) -> None:
    in_session(monkeypatch, {**DM, "HERMES_SESSION_CHAT_ID": f"dm-{name}", "HERMES_SESSION_USER_NAME": name})


def test_a_shared_profiles_dms_keep_no_personal_memory(home, tmp_path, monkeypatch):
    # A personal database is one person's; a shared profile's DMs are anyone's.
    m, p = make_manager(tmp_path, platform="telegram")
    dm_from(monkeypatch, "ana")
    out = call(m, "infino_remember", content="ana's dentist appointment is Tuesday", scope="personal")
    assert out["ok"] is False and "nothing was saved" in out["error"]
    p.on_memory_write("add", "user", "Prefers short answers")  # Hermes's USER.md, written in the DM
    p._drain()
    assert personal_rows(tmp_path) == [] and local_rows(tmp_path, tables.MEMORY) == []
    assert "no personal memory is kept here" in p.system_prompt_block()


def test_a_shared_profiles_dms_read_no_personal_memory_an_earlier_version_kept(home, tmp_path, monkeypatch):
    m, p = make_manager(tmp_path, platform="telegram")
    p._backend.personal.append(tables.MEMORY, [{"content": "ana's dentist appointment is Tuesday",
                                                "scope": "personal", "author": "ana"}])
    dm_from(monkeypatch, "ana")  # a display name anyone could take
    assert "dentist" not in p.prefetch("dentist appointment")
    assert call(m, "infino_forget", content="ana's dentist appointment is Tuesday")["matched"] == 0


def test_a_personal_profiles_dms_keep_its_persons_memory(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")  # one person's profile: their own bot
    m, p = make_manager(tmp_path, platform="telegram")
    dm_from(monkeypatch, "ana")
    assert call(m, "infino_remember", content="ana's dentist appointment is Tuesday", scope="personal")["ok"]
    p.on_memory_write("add", "user", "Prefers short answers")
    p._drain()
    assert sorted((r["content"], r["author"]) for r in personal_rows(tmp_path)) == [
        ("About ana: Prefers short answers", "ana"), ("ana's dentist appointment is Tuesday", "ana")]
    assert "dentist" in p.prefetch("dentist appointment")
    assert "no personal memory is kept here" not in p.system_prompt_block()


def test_a_group_forgets_an_entry_another_member_wrote(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP)  # mallory says it
    p.on_memory_write("add", "user", "Is on call this week")
    p.on_memory_write("add", "user", "Is not on call this week")  # ends differently: kept
    p._drain()
    in_session(monkeypatch, {**GROUP, "HERMES_SESSION_USER_NAME": "zed"})  # zed takes it back
    p.on_memory_write("remove", "user", "", {"previous_content": "Is on call this week"})
    p._drain()
    rows = local_rows(tmp_path, tables.MEMORY)
    assert [r["content"] for r in current(rows)] == ["About mallory: Is not on call this week"]
    # Kept for the group it was written in, superseded: still never anyone's personal memory.
    [kept] = superseded(rows)
    assert kept["content"] == "About mallory: Is on call this week" and kept["scope"] == "team"
    assert kept[tables.CHAT] == current(rows)[0][tables.CHAT] != ""
    assert personal_rows(tmp_path) == []


def test_a_resumed_session_stores_each_message_once(home, tmp_path):
    history = [{"role": "user", "content": "run the tests"}, {"role": "assistant", "content": "2 fail"}]
    _, first = make_manager(tmp_path, session_id="s1")
    first.sync_turn("run the tests", "2 fail", session_id="s1", messages=history)
    first.on_session_end(history)
    # `hermes chat --resume s1`: a new process, the same session, its history first.
    _, resumed = make_manager(tmp_path, session_id="s1")
    messages = history + [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "fixed"}]
    resumed.sync_turn("fix it", "fixed", session_id="s1", messages=messages)
    resumed._drain()
    assert [(r["seq"], r["content"]) for r in turn_rows(tmp_path)] == [
        (0, "run the tests"), (1, "2 fail"), (2, "fix it"), (3, "fixed")]


def test_a_remember_that_reports_failure_saved_nothing(mgr, tmp_path):
    m, _ = mgr

    def broken(texts):
        raise OSError("no network to download the model")

    embed.use(broken)
    out = call(m, "infino_remember", content="the deploy freeze starts December 20")
    assert out["ok"] or local_rows(tmp_path, tables.MEMORY) == []


def test_a_memory_saved_while_the_model_could_not_run_is_embedded_at_the_next_start(home, tmp_path):
    def broken(texts):
        raise OSError("no network to download the model")

    m, _ = make_manager(tmp_path)
    embed.use(broken)
    assert call(m, "infino_remember", content="the deploy freeze starts December 20")["ok"]
    store = local(tmp_path)
    assert store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == 0
    embed.use(fake_embed)
    backend._backends.clear()
    make_manager(tmp_path)  # the next start
    assert wait_for(lambda: store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == 1)


def test_a_start_without_the_model_still_starts(home, tmp_path):
    def broken(texts):
        raise OSError("no network to download the model")

    embed.use(broken)
    m, p = make_manager(tmp_path)
    assert call(m, "infino_remember", content="the deploy freeze starts December 20")["ok"]
    embed.use(fake_embed)
    assert "deploy freeze" in p.prefetch("deploy freeze")


# -- a soft release and a session's end ------------------------------------------

def test_a_soft_release_never_waits_on_the_cloud(env, platform, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "FLUSH_TIMEOUT_S", QUICK_FLUSH_S)
    mgr, _ = make_manager(tmp_path)
    mgr.handle_tool_call("infino_remember", {"content": "written before the agent is rebuilt"})
    with open(config.state_dir(tmp_path) / mirror.LOCK_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # a sync held up by a slow cloud
        started = time.monotonic()
        mgr.release_soft()  # Hermes rebuilding the CLI's agent (/reasoning, a route change)
        took = time.monotonic() - started
        fcntl.flock(f, fcntl.LOCK_UN)
    assert took < SOFT_RELEASE_S
    # The background sync still takes the row.
    assert wait_for(lambda: [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)]
                    == ["written before the agent is rebuilt"])


@pytest.mark.parametrize("steps, waits", [
    ((), False),                                  # a soft release: shutdown alone
    (("end",), True),                             # a session's end: on_session_end, then shutdown
    (("end", "sync"), True),                      # ... with the last turn's sync drained after the end
    (("end", "turn"), False),                     # /new or compaction ended a session, and a turn started
    (("end", "switch"), False),                   # ... into a new session id
])
def test_only_a_sessions_end_waits_for_the_cloud(env, tmp_path, monkeypatch, steps, waits):
    _, p = make_manager(tmp_path)
    flushed = []
    monkeypatch.setattr(p._backend, "flush", lambda *a, **k: flushed.append(True))
    for step in steps:
        if step == "end":
            p.on_session_end([])
        elif step == "sync":
            p.sync_turn("q", "a", session_id="s1")
        elif step == "turn":
            p.on_turn_start(2, "q")
        else:
            p.on_session_switch("s2", parent_session_id="s1", reset=True)
    p.shutdown()
    assert flushed == ([True] if waits else [])


def pushed_turn(platform) -> list[str]:
    """What the team's cloud `conversations` holds, sorted: a turn's rows share
    their stamp, so the cloud has them in `row_id` order."""
    return sorted(r["content"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS))


def test_a_one_shot_runs_end_pushes_the_last_turn_still_queued_at_its_end(env, platform, tmp_path, monkeypatch):
    mgr, p = make_manager(tmp_path)
    monkeypatch.setattr(backend, "SYNC_DELAY_S", NO_BACKGROUND_SYNC_S)
    # Hermes's worker is still on the turn's sync when the run ends: on_session_end runs inline, and
    # shutdown_all drains the sync after it (hermes_cli/oneshot.py, AIAgent.shutdown_memory_provider).
    ended = threading.Event()
    real_end, real_sync = p.on_session_end, p.sync_turn
    monkeypatch.setattr(p, "on_session_end", lambda messages: (real_end(messages), ended.set()))
    monkeypatch.setattr(p, "sync_turn",
                        lambda *args, **kwargs: (ended.wait(WRITE_WAIT_S), real_sync(*args, **kwargs)))
    mgr.sync_all(*LAST_TURN, session_id="s1")
    oneshot._close_agent(hermes_agent(mgr), None)
    assert pushed_turn(platform) == sorted(LAST_TURN)


def cli_exit(agent) -> None:
    """The CLI's exit (hermes_cli/cli_shutdown.py), on its main thread."""
    cli_shutdown._shutdown_agent_memory_provider(agent)


def gateway_cleanup(context: str):
    """The gateway's end of a cached agent (gateway/run_shutdown.py), on its own
    housekeeping workers: at its stop, and at a session's expiry."""
    def end(agent) -> None:
        runner = GatewayRunner.__new__(GatewayRunner)
        try:
            asyncio.run(runner._cleanup_agent_resources_off_loop(agent, context=context, session_key=GATEWAY_KEY))
        finally:
            runner._housekeeping_executor.shutdown(wait=True)
    return end


def api_server_close(agent) -> None:
    """The api_server adapter's disconnect at the gateway's stop: its parked
    sessions closed on a worker thread (gateway/platforms/api_server.py), the
    gateway then exiting through os._exit."""
    sessions = ApiServerMemorySessions(max_size=API_SESSIONS, idle_ttl_secs=API_IDLE_S)
    sessions.checkin(agent)
    asyncio.run(asyncio.to_thread(sessions.close_all))


@pytest.mark.parametrize("surface, teardown", [
    ("cli", cli_exit),
    ("telegram", gateway_cleanup("shutdown idle-cache")),
    ("telegram", gateway_cleanup("session expiry")),
    ("api_server", api_server_close),
], ids=["cli-exit", "gateway-stop", "gateway-session-expiry", "api-server-at-gateway-stop"])
def test_a_teardown_hermes_waits_for_pushes_the_rows_before_it_returns(env, platform, tmp_path, monkeypatch,
                                                                       surface, teardown):
    mgr, _ = make_manager(tmp_path, platform=surface)
    monkeypatch.setattr(backend, "SYNC_DELAY_S", NO_BACKGROUND_SYNC_S)
    mgr.sync_all(*LAST_TURN, session_id="s1")  # the last turn, queued on Hermes's worker
    teardown(hermes_agent(mgr))
    assert pushed_turn(platform) == sorted(LAST_TURN)


def gateway_pressure_eviction(agent, monkeypatch) -> None:
    """The gateway's memory valve (gateway/run_agent_cache.py): the agents it
    evicts released one after another on a daemon thread."""
    runner = GatewayRunner.__new__(GatewayRunner)
    valve = threading.Thread(target=runner._release_pressure_batch, args=([(GATEWAY_KEY, agent)],), daemon=True,
                             name="agent-cache-pressure")
    valve.start()
    valve.join(TEARDOWN_JOIN_S)


def api_server_idle_eviction(agent, monkeypatch) -> None:
    """An api_server session parked past its idle time, shut down on a thread of
    its own (gateway/platforms/api_server_memory_sessions.py)."""
    spawned = []
    real = memory_provider.spawn_context_thread

    def recorded(*args, **kwargs):
        spawned.append(real(*args, **kwargs))
        return spawned[-1]

    monkeypatch.setattr(memory_provider, "spawn_context_thread", recorded)
    sessions = ApiServerMemorySessions(max_size=API_SESSIONS, idle_ttl_secs=API_IDLE_S)
    sessions._shutdown_async(agent._memory_manager, None)
    for thread in spawned:
        thread.join(TEARDOWN_JOIN_S)


def cron_cleanup(agent, monkeypatch) -> None:
    """A scheduled job's run torn down after it ran (cron/scheduler.py), bounded
    by the scheduler on a daemon thread."""
    _teardown_cron_agent(agent, CRON_JOB, timeout_seconds=TEARDOWN_JOIN_S)


@pytest.mark.parametrize("surface, context, session, teardown", [
    ("telegram", "primary", "s1", gateway_pressure_eviction),
    ("api_server", "primary", "s1", api_server_idle_eviction),
    (CRON_PLATFORM, "cron", CRON_SESSION, cron_cleanup),
], ids=["gateway-eviction-under-memory-pressure", "api-server-idle-eviction", "scheduled-job-cleanup"])
def test_a_teardown_in_a_process_that_goes_on_never_waits_on_the_cloud(env, platform, tmp_path, monkeypatch,
                                                                         surface, context, session, teardown):
    monkeypatch.setattr(backend, "FLUSH_TIMEOUT_S", QUICK_FLUSH_S)
    mgr, _ = make_manager(tmp_path, platform=surface, agent_context=context, session_id=session)
    mgr.handle_tool_call("infino_remember", {"content": "written before the agent is let go"})
    with open(config.state_dir(tmp_path) / mirror.LOCK_FILE, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # a sync held up by a slow cloud
        started = time.monotonic()
        teardown(hermes_agent(mgr, session), monkeypatch)
        took = time.monotonic() - started
        fcntl.flock(f, fcntl.LOCK_UN)
    assert took < QUICK_FLUSH_S
    # The background sync still takes the row.
    assert wait_for(lambda: [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)]
                    == ["written before the agent is let go"])


# -- webhook turns -----------------------------------------------------------------

def test_what_a_webhook_turn_saves_is_recalled_in_the_routes_next_delivery(home, tmp_path):
    # Hermes gives each delivery its own chat: a note kept for that one would never be recalled.
    m, p = make_manager(tmp_path, platform="webhook", session_id="hook-1")
    with webhook_delivery("ci-failure", "delivery-1"):
        assert call(m, "infino_remember", content="CI flakes come from the cache warm-up")["ok"]
    p._drain()
    with webhook_delivery("ci-failure", "delivery-2"):
        assert "cache warm-up" in p.prefetch("why does CI flake")
    with webhook_delivery("email-triage", "delivery-3"):  # another route's turns don't see it
        assert "cache warm-up" not in p.prefetch("why does CI flake")

"""A running turn in Infino as it happens, and a subagent pointed at it: each
message written as it comes, the turn's end writing only what came after, and
a subagent told where its parent's work is."""

from __future__ import annotations

import pytest

from hermes_infino import live, tables

from .helpers import local_rows, make_manager

TOOL_CALL = {"content": "", "tool_calls": [{"id": "a", "function": {"name": "read_file", "arguments": "{}"}}]}


def turn_rows(tmp_path) -> list[tuple]:
    rows = sorted(local_rows(tmp_path, tables.CONVERSATIONS), key=lambda r: (r["session_id"], r["seq"]))
    return [(r["session_id"], r["seq"], r["role"], r["content"] or r["tool_name"], r["tool_result"]) for r in rows]


@pytest.fixture
def p(home, tmp_path):
    _, provider = make_manager(tmp_path)
    return provider


def run_until_the_tool(session: str = "s1", parent: str = "") -> dict | None:
    """A turn as Hermes runs it, up to a tool's result: the prompt, a reply
    that calls a tool, the tool's result."""
    note = live.on_turn_start(session_id=session, turn_id="t1", user_message="review it",
                              parent_session_id=parent)
    live.on_reply(session_id=session, turn_id="t1", assistant_message=TOOL_CALL)
    live.on_tool_result(session_id=session, turn_id="t1", tool_name="read_file", tool_call_id="a",
                        result="file text")
    return note


def test_a_turns_messages_are_in_conversations_as_they_happen(p, tmp_path):
    run_until_the_tool()
    p._drain()
    assert turn_rows(tmp_path) == [("s1", 0, "user", "review it", ""), ("s1", 1, "assistant", "read_file", ""),
                                   ("s1", 2, "tool", "read_file", "file text")]


def test_the_turns_end_writes_only_what_came_after(p, tmp_path):
    run_until_the_tool()
    messages = [{"role": "user", "content": "review it"},
                {"role": "assistant", **TOOL_CALL},
                {"role": "tool", "tool_call_id": "a", "content": "file text"},
                {"role": "assistant", "content": "no findings"}]
    p.sync_turn("review it", "no findings", session_id="s1", messages=messages)
    p._drain()
    assert [r[1:4] for r in turn_rows(tmp_path)] == [(0, "user", "review it"), (1, "assistant", "read_file"),
                                                     (2, "tool", "read_file"), (3, "assistant", "no findings")]
    # The next turn numbers on from it.
    p.sync_turn("and now?", "done", session_id="s1")
    p._drain()
    assert [r[1] for r in turn_rows(tmp_path)][-2:] == [4, 5]


def test_a_subagent_is_told_where_its_parents_work_is(p, tmp_path):
    run_until_the_tool()
    note = run_until_the_tool(session="child", parent="s1")
    assert note is not None and "session_id = 's1'" in note["context"]
    assert "conversations" in note["context"]
    p._drain()
    # The subagent's own messages are kept too, under its own session.
    assert [r[:3] for r in turn_rows(tmp_path) if r[0] == "child"] == [
        ("child", 0, "user"), ("child", 1, "assistant"), ("child", 2, "tool")]


def test_an_agents_own_turn_gets_no_note():
    assert live.on_turn_start(session_id="s1", turn_id="t1", user_message="hi") is None


def test_a_gateway_session_writes_each_turn_whole(home, tmp_path):
    _, gateway = make_manager(tmp_path, platform="telegram")
    run_until_the_tool()
    gateway._drain()
    assert turn_rows(tmp_path) == []


def test_a_subagents_start_sends_the_parents_rows_to_the_cloud(p, monkeypatch):
    delays = []
    monkeypatch.setattr(p._backend, "schedule_sync", lambda delay_s=None: delays.append(delay_s))
    live.on_subagent_start(parent_session_id="s1", child_session_id="child")
    p._drain()
    assert delays == [0.0]


def test_a_reply_from_hermes_objects_becomes_a_message():
    class Fn:
        name, arguments = "terminal", "{\"cmd\": \"ls\"}"

    class Call:
        id, function = "b", Fn()

    class Reply:
        content, tool_calls = "looking", [Call()]

    assert live.reply_message(Reply()) == {"role": "assistant", "content": "looking", "tool_calls": [
        {"id": "b", "function": {"name": "terminal", "arguments": "{\"cmd\": \"ls\"}"}}]}

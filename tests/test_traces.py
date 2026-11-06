"""Trace hooks: token accounting, what one row holds, and the per-turn flush."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.memory_manager import _redact_for_provider

from hermes_infino import privacy, tables, traces

from .helpers import GROUP, hook_ctx, in_session, local, local_rows, wait_for

# A credential shaped the way Hermes's redaction recognizes one.
SECRET = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 4


@pytest.fixture
def hooks(home, tmp_path):
    # The tables the provider makes at session start.
    local(tmp_path).ensure_tables(with_vectors=True)
    return hook_ctx()


@pytest.mark.parametrize("usage, expected", [
    (None, (0, 0, 0)),
    ({}, (0, 0, 0)),
    ({"prompt_tokens": 120, "completion_tokens": 7}, (120, 7, 0)),
    ({"input_tokens": 10, "cache_creation_input_tokens": 90, "output_tokens": 3}, (100, 3, 90)),
    ({"input_tokens": 10, "cache_read_tokens": 5, "cache_write_tokens": 5, "output_tokens": 1}, (20, 1, 10)),
    ({"input_tokens": None, "output_tokens": None}, (0, 0, 0)),
])
def test_tokens_from_every_usage_shape(usage, expected):
    assert traces._tokens(usage) == expected


@pytest.mark.parametrize("value, expected", [
    (None, ""),
    ("text", "text"),
    ({"cmd": "ls"}, '{"cmd": "ls"}'),
    (["a", 1], '["a", 1]'),
])
def test_a_cell_is_text(value, expected):
    assert traces._clip(value) == expected


def test_a_long_cell_is_cut():
    assert len(traces._clip("x" * (traces.MAX_TRACE_CELL_CHARS + 50))) == traces.MAX_TRACE_CELL_CHARS


def test_a_model_call_row(hooks, tmp_path):
    hooks["post_api_request"](session_id="s9", turn_id="t1", api_request_id="r1", model="requested",
                              response_model="served", provider="openrouter", api_duration=0.25,
                              finish_reason="tool_calls", usage={"prompt_tokens": 50, "completion_tokens": 5},
                              assistant_message="calling a tool")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert (row["kind"], row["name"], row["model"], row["provider"]) == ("model", "served", "served", "openrouter")
    assert (row["status"], row["duration_ms"], row["input_tokens"], row["output_tokens"]) == ("tool_calls", 250.0, 50, 5)
    assert row["result"] == "calling a tool" and row["request_id"] == "r1"


def test_a_model_call_without_a_duration_or_reason(hooks, tmp_path):
    hooks["post_api_request"](session_id="s9", turn_id="t1", model="m", api_duration="n/a")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert row["duration_ms"] == 0.0 and row["status"] == "ok"


def test_a_failed_tool_call_row(hooks, tmp_path):
    hooks["post_tool_call"](session_id="s9", turn_id="t1", tool_name="terminal", tool_call_id="c1",
                            args={"cmd": "make"}, result=None, duration_ms=40, status="error",
                            error_message="exit 2")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert (row["kind"], row["tool"], row["status"], row["error"]) == ("tool", "terminal", "error", "exit 2")
    assert row["result"] == "" and row["input_tokens"] == 0


def test_a_turn_with_no_calls_writes_nothing(hooks, tmp_path, monkeypatch):
    flushed = []
    monkeypatch.setattr(traces, "spawn_context_thread", lambda fn, name: flushed.append(name))
    hooks["on_session_end"](session_id="nothing-here")
    assert flushed == [] and local_rows(tmp_path, tables.TRACES) == []


def test_each_session_flushes_only_its_own_rows(hooks, tmp_path):
    hooks["post_api_request"](session_id="a", turn_id="ta", model="m")
    hooks["post_api_request"](session_id="b", turn_id="tb", model="m")
    hooks["on_session_end"](session_id="a")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    assert local_rows(tmp_path, tables.TRACES)[0]["session_id"] == "a"
    hooks["on_session_end"](session_id="b")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 2)


def test_a_turn_is_noted_when_it_starts(hooks):
    hooks["pre_llm_call"](session_id="s7", turn_id="t1", user_message="check the build")
    assert traces.turn_id_for("s7", "check the build") == "t1"
    assert traces.turn_id_for("never-seen", "check the build") == ""


def test_a_turns_start_reaches_its_sync_from_hermess_hook_thread(hooks):
    # Hermes runs pre_llm_call on a thread of its own with a copy of the turn's
    # context; the sync runs on another thread again.
    hooks["pre_llm_call"](session_id="s7", turn_id="t1", user_message="q")
    found = []
    worker = threading.Thread(target=lambda: found.append(traces.turn_id_for("s7", "q")))
    worker.start()
    worker.join()
    assert found == ["t1"]


def test_a_sync_takes_its_own_turn_when_the_next_has_started(hooks):
    # Hermes syncs a turn on a background worker: the person's next message may
    # start the next turn first.
    hooks["pre_llm_call"](session_id="s7", turn_id="t1", user_message="first")
    hooks["pre_llm_call"](session_id="s7", turn_id="t2", user_message="second")
    assert traces.turn_id_for("s7", "first") == "t1"
    assert traces.turn_id_for("s7", "second") == "t2"


def test_a_turn_that_is_never_synced_is_dropped_by_a_later_one(hooks):
    # An interrupted turn, or one with no reply, is never synced.
    hooks["pre_llm_call"](session_id="s7", turn_id="t1", user_message="stop that")
    hooks["pre_llm_call"](session_id="s7", turn_id="t2", user_message="try again")
    assert traces.turn_id_for("s7", "try again") == "t2"
    assert traces.turn_id_for("s7", "stop that") == ""


def test_a_message_sent_again_belongs_to_the_later_turn(hooks):
    # Interrupted, then sent again: the turn that is synced is the later one.
    hooks["pre_llm_call"](session_id="s7", turn_id="t1", user_message="deploy")
    hooks["pre_llm_call"](session_id="s7", turn_id="t2", user_message="deploy")
    assert traces.turn_id_for("s7", "deploy") == "t2"


def test_model_and_tool_calls_never_move_a_turn(hooks):
    # Hermes's background review shares the session id and makes model and tool
    # calls of its own, but starts no turn (Hermes fires no pre_llm_call for it).
    hooks["pre_llm_call"](session_id="s8", turn_id="the-turn", user_message="q")
    hooks["post_api_request"](session_id="s8", turn_id="the-review", model="m")
    hooks["post_tool_call"](session_id="s8", turn_id="the-review", tool_name="memory")
    assert traces.turn_id_for("s8", "q") == "the-turn"


def test_a_message_hermes_scrubs_or_flattens_still_finds_its_turn(hooks):
    # sync_turn is handed the user message flattened to text and scrubbed of
    # secrets, by Hermes's own functions.
    with_secret = f"deploy with {SECRET}"
    hooks["pre_llm_call"](session_id="s9", turn_id="t1", user_message=with_secret)
    assert traces.turn_id_for("s9", _redact_for_provider(with_secret)) == "t1"
    parts = [{"type": "text", "text": "what is in this picture?"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
    hooks["pre_llm_call"](session_id="s9", turn_id="t2", user_message=parts)
    assert traces.turn_id_for("s9", _summarize_user_message_for_log(parts, sep="\n")) == "t2"


def test_a_sync_whose_message_matches_no_turn_takes_the_latest(hooks):
    hooks["pre_llm_call"](session_id="s9", turn_id="t1", user_message="q")
    assert traces.turn_id_for("s9", "a message no turn started with") == "t1"


def test_a_gateway_turns_rows_are_the_senders(hooks, tmp_path, monkeypatch):
    in_session(monkeypatch, GROUP)
    traces.note_identity("g1", "the-profiles-owner", "ops")
    hooks["post_api_request"](session_id="g1", turn_id="t1", model="m")
    hooks["on_session_end"](session_id="g1")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert (row["author"], row["profile"]) == ("mallory", "ops")


def test_a_model_calls_row_keeps_its_text_not_its_tool_calls(hooks, tmp_path):
    message = SimpleNamespace(content="Saving that.", tool_calls=[{"function": {"arguments": "private words"}}])
    hooks["post_api_request"](session_id="s9", turn_id="t1", model="m", assistant_message=message)
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    assert local_rows(tmp_path, tables.TRACES)[0]["result"] == "Saving that."


@pytest.mark.parametrize("tool, args", [
    ("infino_remember", {"content": "my dentist is Tuesday", "scope": "Personal"}),
    ("memory", {"target": "user", "operations": [{"action": "add", "content": "my dentist is Tuesday"}]}),
    ("memory", json.dumps({"target": "user", "action": "add", "content": "my dentist is Tuesday"})),
])
def test_a_personal_memory_call_is_traced_without_its_text(hooks, tmp_path, tool, args):
    hooks["post_tool_call"](session_id="s9", turn_id="t1", tool_name=tool, args=args,
                            result="entries: my dentist is Tuesday")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert "dentist" not in row["args"] and "dentist" not in row["result"]
    assert privacy.WITHHELD in row["args"] and row["tool"] == tool


def test_a_team_memory_call_is_traced_whole(hooks, tmp_path):
    hooks["post_tool_call"](session_id="s9", turn_id="t1", tool_name="infino_remember",
                            args={"content": "deploys freeze Dec 20"}, result="ok")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    assert "deploys freeze Dec 20" in local_rows(tmp_path, tables.TRACES)[0]["args"]


def test_a_session_nobody_identified_has_no_author(hooks, tmp_path):
    hooks["post_api_request"](session_id="anon", turn_id="t1", model="m")
    hooks["on_session_end"](session_id="anon")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    assert local_rows(tmp_path, tables.TRACES)[0]["author"] == ""


def test_hermes_masks_the_secret_the_next_test_uses():
    assert SECRET not in _redact_for_provider(f"key={SECRET}")


def test_a_secret_in_a_tool_call_is_scrubbed_as_hermes_scrubs_provider_egress(hooks, tmp_path):
    hooks["post_tool_call"](session_id="s9", turn_id="t1", tool_name="terminal", tool_call_id="c1",
                            args={"cmd": f"export ANTHROPIC_API_KEY={SECRET}"}, result=f"key={SECRET}",
                            status="error", error_message=f"bad key {SECRET}")
    hooks["post_api_request"](session_id="s9", turn_id="t1", model="m", assistant_message=f"your key is {SECRET}")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 2)
    for row in local_rows(tmp_path, tables.TRACES):
        assert all(SECRET not in str(row[c]) for c in ("args", "result", "error"))
    assert any("ANTHROPIC_API_KEY" in r["args"] for r in local_rows(tmp_path, tables.TRACES))  # the rest is kept


def test_a_secret_is_scrubbed_before_a_cell_is_cut(monkeypatch):
    monkeypatch.setattr(traces, "MAX_TRACE_CELL_CHARS", 30)
    assert SECRET[:20] not in traces._clip("x" * 20 + SECRET)


def test_only_the_latest_sessions_are_remembered(monkeypatch):
    monkeypatch.setattr(traces, "MAX_SESSIONS", 3)
    for i in range(5):
        traces.note_identity(f"cap-{i}", f"person-{i}", "p")
    assert traces._identity_of("cap-0") == ("", "") and traces._identity_of("cap-4") == ("person-4", "p")
    assert len([k for k in traces._identity if k.startswith("cap-")]) <= 3

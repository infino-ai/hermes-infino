"""Trace hooks: token accounting, what one row holds, and the per-turn flush."""

from __future__ import annotations

import pytest

from agent.memory_manager import _redact_for_provider

from hermes_infino import tables, traces

from .helpers import hook_ctx, local, local_rows, wait_for

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


def test_the_last_turn_id_ignores_calls_that_name_no_turn(hooks):
    hooks["post_api_request"](session_id="s7", turn_id="t1", model="m")
    hooks["post_tool_call"](session_id="s7", turn_id="", tool_name="x")
    assert traces.last_turn_id("s7") == "t1"
    assert traces.last_turn_id("never-seen") == ""


def test_a_session_nobody_identified_has_no_author(hooks, tmp_path):
    hooks["post_api_request"](session_id="anon", turn_id="t1", model="m")
    hooks["on_session_end"](session_id="anon")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    assert local_rows(tmp_path, tables.TRACES)[0]["author"] == ""


def test_hermes_masks_the_secret_the_next_test_uses():
    assert SECRET not in _redact_for_provider(f"key={SECRET}")


@pytest.mark.xfail(strict=True, reason="Hermes scrubs secrets from everything it hands a memory provider (#115104), "
                                       "but the trace hooks get raw tool args and results, and the plugin stores "
                                       "them unscrubbed, so a key a tool printed goes to the team's database")
def test_a_secret_in_a_tool_call_is_scrubbed_as_hermes_scrubs_provider_egress(hooks, tmp_path):
    hooks["post_tool_call"](session_id="s9", turn_id="t1", tool_name="terminal", tool_call_id="c1",
                            args={"cmd": f"export ANTHROPIC_API_KEY={SECRET}"}, result=f"key={SECRET}")
    hooks["on_session_end"](session_id="s9")
    assert wait_for(lambda: len(local_rows(tmp_path, tables.TRACES)) == 1)
    [row] = local_rows(tmp_path, tables.TRACES)
    assert SECRET not in row["args"] and SECRET not in row["result"]

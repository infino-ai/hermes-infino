"""Trace hooks: every model call and tool call, as rows in the `traces`
table (locally, and in the team's cloud database once signed in), next to
the conversation they belong to.

Rows are buffered per session and written once per turn (Hermes's plugin
`on_session_end` hook fires at each turn's end), so a turn with ten tool
calls is one append, not eleven. No OpenTelemetry collector is involved.

Hermes scrubs secrets from everything it hands a memory provider, but these
hooks get tool arguments and results as the tool saw them. So every text
cell is put through the same scrub (`redact_for_egress`) before it is kept:
a key a tool printed never reaches the team's database.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from agent.memory_provider import spawn_context_thread

from . import backend, config, tables

try:  # Hermes's scrub for anything leaving the agent.
    from agent.redact import redact_for_egress as _scrub
except ImportError:  # pragma: no cover - a Hermes without it scrubs nothing here either
    def _scrub(text: str) -> str:
        return text

logger = logging.getLogger(__name__)

# Longest args/result text kept per trace row.
MAX_TRACE_CELL_CHARS = 4_000
# Sessions whose turn and person are remembered; a long-running gateway sees
# many, and only the latest few are still running.
MAX_SESSIONS = 1_024


def _clip(value: Any) -> str:
    """A cell's text: scrubbed of secrets, then cut (so no secret is cut in half and kept)."""
    if value is None or value == "":
        return ""
    s = _scrub(value if isinstance(value, str) else json.dumps(value, default=str))
    return s if len(s) <= MAX_TRACE_CELL_CHARS else s[:MAX_TRACE_CELL_CHARS]


def _remember(table: OrderedDict, session_id: str, value) -> None:
    """Note a session's latest value, forgetting the oldest sessions past MAX_SESSIONS."""
    table[session_id] = value
    table.move_to_end(session_id)
    while len(table) > MAX_SESSIONS:
        table.popitem(last=False)


def _tokens(usage: Any) -> tuple[int, int, int]:
    """Prompt, output and cached-prompt tokens.

    Hermes hands hooks its `CanonicalUsage`, whose `input_tokens` leaves out the
    prompt-cache reads and writes; the whole prompt is its `prompt_tokens`
    (input + cache read + cache write). A plain dict is read the same way.
    """
    if usage is None:
        return 0, 0, 0
    get = usage.get if isinstance(usage, dict) else (lambda k, d=0: getattr(usage, k, d))
    cache_read = int(get("cache_read_tokens", 0) or get("cache_read_input_tokens", 0) or 0)
    cache_write = int(get("cache_write_tokens", 0) or get("cache_creation_input_tokens", 0) or 0)
    prompt = get("prompt_tokens", None)
    if not prompt:
        prompt = int(get("input_tokens", 0) or 0) + cache_read + cache_write
    out = int(get("output_tokens", 0) or get("completion_tokens", 0) or 0)
    return int(prompt), out, cache_read + cache_write


# The turn each session's latest model or tool call belonged to, so the
# provider's `conversations` rows carry the same `turn_id` as the turn's
# `traces` rows (Hermes hands the id to these hooks, not to `sync_turn`).
_last_turn: OrderedDict[str, str] = OrderedDict()
_last_turn_lock = threading.Lock()


def last_turn_id(session_id: str) -> str:
    """The turn id of the session's latest recorded call; empty for none."""
    with _last_turn_lock:
        return _last_turn.get(session_id or "", "")


# Who each session's turns are from, and the profile running it. Hermes hands
# the trace hooks no user identity; the memory provider learns it (at
# `initialize`, and per turn in `sync_turn`, which Hermes calls before the
# turn-end hook that flushes a turn's traces) and notes it here.
_identity: OrderedDict[str, tuple[str, str]] = OrderedDict()


def note_identity(session_id: str, author: str, profile: str) -> None:
    with _last_turn_lock:
        _remember(_identity, session_id or "", (author, profile))


def _identity_of(session_id: str) -> tuple[str, str]:
    with _last_turn_lock:
        return _identity.get(session_id or "", ("", ""))


class TraceRecorder:
    def __init__(self) -> None:
        self._buffers: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def _add(self, session_id: str, row: dict) -> None:
        with self._lock:
            self._buffers.setdefault(session_id or "", []).append(row)
        if row["turn_id"]:
            with _last_turn_lock:
                _remember(_last_turn, session_id or "", row["turn_id"])

    def on_api(self, **kw: Any) -> None:
        inp, out, cached = _tokens(kw.get("usage"))
        duration = kw.get("api_duration")
        model = str(kw.get("response_model") or kw.get("model") or "")
        self._add(kw.get("session_id", ""), {
            "session_id": str(kw.get("session_id") or ""),
            "turn_id": str(kw.get("turn_id") or ""),
            "request_id": str(kw.get("api_request_id") or ""),
            "kind": "model",
            "name": model,
            "model": model,
            "tool": "",
            "provider": str(kw.get("provider") or ""),
            "status": str(kw.get("finish_reason") or "ok"),
            "error": "",
            "duration_ms": float(duration) * 1000 if isinstance(duration, (int, float)) else 0.0,
            "input_tokens": inp,
            "output_tokens": out,
            "cached_tokens": cached,
            "args": "",
            "result": _clip(kw.get("assistant_message") or ""),
            "created_ms": int(time.time() * 1000),
        })

    def on_tool(self, **kw: Any) -> None:
        self._add(kw.get("session_id", ""), {
            "session_id": str(kw.get("session_id") or ""),
            "turn_id": str(kw.get("turn_id") or ""),
            "request_id": str(kw.get("tool_call_id") or ""),
            "kind": "tool",
            "name": str(kw.get("tool_name") or ""),
            "model": "",
            "tool": str(kw.get("tool_name") or ""),
            "provider": "",
            "status": str(kw.get("status") or ""),
            "error": _clip(kw.get("error_message") or ""),
            "duration_ms": float(kw.get("duration_ms") or 0),
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "args": _clip(kw.get("args")),
            "result": _clip(kw.get("result")),
            "created_ms": int(time.time() * 1000),
        })

    def on_turn_end(self, **kw: Any) -> None:
        session = str(kw.get("session_id") or "")
        with self._lock:
            rows = self._buffers.pop(session, [])
        if not rows:
            return
        settings = config.load()  # read in the hook's own profile scope
        author, profile = _identity_of(session)
        for row in rows:
            row["author"] = author
            row["profile"] = profile

        def flush() -> None:
            try:
                backend.for_settings(settings).append(tables.TRACES, rows)
            except Exception as e:
                logger.warning("infino-traces: %d trace rows not written: %s", len(rows), e)

        spawn_context_thread(flush, name="infino-traces-flush").start()


def register_hooks(ctx) -> None:
    recorder = TraceRecorder()
    ctx.register_hook("post_api_request", recorder.on_api)
    ctx.register_hook("post_tool_call", recorder.on_tool)
    ctx.register_hook("on_session_end", recorder.on_turn_end)

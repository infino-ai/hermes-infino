"""Trace hooks: every model call and tool call, as rows in the `traces`
table (locally, and in the team's cloud database once signed in), next to
the conversation they belong to.

Rows are buffered per session and written once per turn (Hermes's plugin
`on_session_end` hook fires at each turn's end), so a turn with ten tool
calls is one append, not eleven. No OpenTelemetry collector is involved.

Hermes scrubs secrets from everything it hands a memory provider, but these
hooks get tool arguments and results as the tool saw them. So every text
cell is put through the same scrub (`redact_for_egress`) before it is kept:
a key a tool printed never reaches the team's database. The text of a
personal memory written by a tool call is withheld (privacy.py). A model
call's row keeps the text the model wrote; its tool calls are rows of their
own.

Which turn a call belongs to, and who the turn is for, are read from the
turn itself, never from "the latest call": Hermes hands `sync_turn` no turn
id, but runs it in the context of the turn that ended (its background worker
carries the caller's context variables). So each turn's own model and tool
calls note their turn id in a context variable, and `sync_turn` reads that
one; Hermes's background review, which shares the session id, runs in a
context of its own and cannot change it. The person is the gateway's sender
for the turn (the turn's session values), or this machine's owner.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any

from agent.memory_provider import spawn_context_thread

from . import backend, chats, config, privacy, tables

try:  # Hermes's scrub for anything leaving the agent.
    from agent.redact import redact_for_egress as _scrub
except ImportError:  # pragma: no cover - a Hermes without it scrubs nothing here either
    def _scrub(text: str) -> str:
        return text

logger = logging.getLogger(__name__)

# Longest args/result text kept per trace row.
MAX_TRACE_CELL_CHARS = 4_000
# How much of a cell is scrubbed: enough that the kept part is scrubbed whole
# (a secret straddling the cut is dropped with what follows it), not a
# multi-megabyte tool output scrubbed on the turn's thread.
SCRUB_CHARS = 2 * MAX_TRACE_CELL_CHARS
# Sessions whose person is remembered; a long-running gateway sees many, and
# only the latest few are still running.
MAX_SESSIONS = 1_024


def _clip(value: Any) -> str:
    """A cell's text: cut to what can be kept, scrubbed of secrets, then cut."""
    if value is None or value == "":
        return ""
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    s = _scrub(s[:SCRUB_CHARS])
    return s if len(s) <= MAX_TRACE_CELL_CHARS else s[:MAX_TRACE_CELL_CHARS]


def _message_text(message: Any) -> Any:
    """What the model wrote: an assistant message's text, not its tool calls."""
    if message is None or isinstance(message, (str, dict)):
        return (message or {}).get("content") if isinstance(message, dict) else message
    return getattr(message, "content", None)


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


# The turn running in this context: (session id, turn id), noted by its own
# model and tool calls.
_turn: ContextVar[tuple[str, str]] = ContextVar("infino_turn", default=("", ""))


def current_turn_id(session_id: str) -> str:
    """The id of this context's turn of `session_id`; empty when none of its
    calls named one (or the context is another session's)."""
    session, turn = _turn.get()
    return turn if session == (session_id or "") else ""


# This machine's person for each session the provider started (`initialize`).
# A gateway turn's person is its sender instead, read from the turn.
_identity: OrderedDict[str, tuple[str, str]] = OrderedDict()
_identity_lock = threading.Lock()


def note_identity(session_id: str, author: str, profile: str) -> None:
    with _identity_lock:
        _identity[session_id or ""] = (author, profile)
        _identity.move_to_end(session_id or "")
        while len(_identity) > MAX_SESSIONS:
            _identity.popitem(last=False)


def _identity_of(session_id: str) -> tuple[str, str]:
    with _identity_lock:
        return _identity.get(session_id or "", ("", ""))


def turn_person(session_id: str) -> str:
    """Who the running turn is for: the gateway's sender, or this machine's person."""
    if chats.session("HERMES_SESSION_PLATFORM").lower() not in chats.LOCAL_PLATFORMS:
        return chats.session("HERMES_SESSION_USER_NAME") or chats.session("HERMES_SESSION_USER_ID")
    return _identity_of(session_id)[0]


class TraceRecorder:
    def __init__(self) -> None:
        self._buffers: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def _add(self, session_id: str, row: dict) -> None:
        with self._lock:
            self._buffers.setdefault(session_id or "", []).append(row)
        if row["turn_id"]:
            _turn.set((session_id or "", row["turn_id"]))

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
            "result": _clip(_message_text(kw.get("assistant_message"))),
            "created_ms": int(time.time() * 1000),
        })

    def on_tool(self, **kw: Any) -> None:
        tool = str(kw.get("tool_name") or "")
        personal = privacy.is_personal_call(tool, kw.get("args"))
        self._add(kw.get("session_id", ""), {
            "session_id": str(kw.get("session_id") or ""),
            "turn_id": str(kw.get("turn_id") or ""),
            "request_id": str(kw.get("tool_call_id") or ""),
            "kind": "tool",
            "name": tool,
            "model": "",
            "tool": tool,
            "provider": "",
            "status": str(kw.get("status") or ""),
            "error": _clip(kw.get("error_message") or ""),
            "duration_ms": float(kw.get("duration_ms") or 0),
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "args": _clip(privacy.withhold(tool, kw.get("args"))),
            "result": privacy.WITHHELD if personal else _clip(kw.get("result")),
            "created_ms": int(time.time() * 1000),
        })

    def on_turn_end(self, **kw: Any) -> None:
        session = str(kw.get("session_id") or "")
        with self._lock:
            rows = self._buffers.pop(session, [])
        if not rows:
            return
        settings = config.load()  # read in the hook's own profile scope
        author, profile = turn_person(session), _identity_of(session)[1]
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

"""Trace hooks: every model call and tool call, as rows in the `traces`
table (locally, and in the team's cloud database once signed in), next to
the conversation they belong to. A personal session's traces (privacy.py) go
to the person's own database instead, as its conversation does: which
scheduled job a run belongs to is noted from each of its calls, a chat a job's
output was delivered into from the turn's history before its first model call
(`pre_llm_call`), and a subagent follows the session that started it
(`subagent_start`).

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

Hermes hands `sync_turn` no turn id, so the conversation rows take theirs
from the turn's start: `pre_llm_call` fires once as each turn starts, with
the turn's id and its user message, and never for Hermes's background review
(a fork that shares the session id and makes model and tool calls of its
own). Each start is noted where every thread can read it: Hermes runs hooks
on threads of their own, each with a copy of the turn's context, so nothing a
hook sets in a context variable reaches `sync_turn`. `sync_turn` takes the
turn whose user message it was handed, since the person's next message may
start the next turn before the last one's sync runs. The person is the
gateway's sender for the turn (the turn's session values), or this machine's
owner.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
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
# How much of a turn's user message ties the turn's start to its sync.
TURN_KEY_CHARS = 200
# Turns remembered per session: those started and not synced yet. A turn
# Hermes never syncs (interrupted, or with no reply) is dropped once a later
# turn of the session is synced.
TURNS_KEPT = 16
# How Hermes marks a message's images when it flattens the message for a
# memory provider: "[1 image] what is this?".
IMAGES_MARK = re.compile(r"^\[\d+ images?\]\s*")


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


# The turns each session started and has not synced yet, oldest first:
# (the user message's key, turn id).
_started: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
_started_lock = threading.Lock()


def _turn_key(text: str) -> str:
    """A user message as a turn's key: its text without Hermes's image mark."""
    return IMAGES_MARK.sub("", (text or "").strip()).strip()[:TURN_KEY_CHARS]


def _message_key(user_message: Any) -> str:
    """The key of a user message as `pre_llm_call` hands it, made the way
    `sync_turn` will be handed the same message: its text parts joined, then
    scrubbed of secrets as Hermes scrubs what it hands a memory provider."""
    if isinstance(user_message, list):
        text = "\n".join(p["text"] for p in user_message if isinstance(p, dict) and isinstance(p.get("text"), str))
    else:
        text = user_message if isinstance(user_message, str) else ""
    return _turn_key(_scrub(text))


def note_turn_start(session_id: str, turn_id: str, user_message: Any) -> None:
    """A turn of `session_id` started: remember its id under its user message."""
    if not turn_id:
        return
    key = _message_key(user_message)
    with _started_lock:
        turns = _started.setdefault(session_id or "", [])
        turns.append((key, turn_id))
        del turns[:-TURNS_KEPT]
        _started.move_to_end(session_id or "")
        while len(_started) > MAX_SESSIONS:
            _started.popitem(last=False)


def turn_id_for(session_id: str, user_content: str) -> str:
    """The id of the turn of `session_id` whose user message `sync_turn` was
    handed as `user_content`: the latest started with that message (a message
    sent again after an interrupted turn belongs to the later turn). That turn
    and those started before it are forgotten: Hermes syncs a session's turns
    in order. With no turn of that message, the latest started (it is kept);
    with none at all, empty."""
    key = _turn_key(user_content)
    with _started_lock:
        turns = _started.get(session_id or "")
        if not turns:
            return ""
        for i in range(len(turns) - 1, -1, -1):
            if turns[i][0] == key:
                turn = turns[i][1]
                del turns[:i + 1]
                return turn
        return turns[-1][1]


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
        # Read in the call's own context: a scheduled job's run says which job it is.
        privacy.note_cron_job(session_id or "")
        with self._lock:
            self._buffers.setdefault(session_id or "", []).append(row)

    def on_llm_call(self, **kw: Any) -> None:
        """As a turn starts, before its first model call: note the turn's id
        for its conversation rows, and any job output delivered into this
        chat, which the turn's history holds."""
        note_turn_start(str(kw.get("session_id") or ""), str(kw.get("turn_id") or ""), kw.get("user_message"))
        history = list(kw.get("conversation_history") or [])
        history.append({"role": "user", "content": kw.get("user_message")})
        privacy.note_deliveries(str(kw.get("session_id") or ""), history)

    def on_subagent(self, **kw: Any) -> None:
        """A subagent's session is as personal as the session that started it."""
        privacy.note_parent(str(kw.get("child_session_id") or ""), str(kw.get("parent_session_id") or ""))

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
                b = backend.for_settings(settings)
                b.append(tables.TRACES, rows, b.transcript_personal(session))
            except Exception as e:
                logger.warning("infino-traces: %d trace rows not written: %s", len(rows), e)

        spawn_context_thread(flush, name="infino-traces-flush").start()


def register_hooks(ctx) -> None:
    recorder = TraceRecorder()
    ctx.register_hook("pre_llm_call", recorder.on_llm_call)
    ctx.register_hook("post_api_request", recorder.on_api)
    ctx.register_hook("post_tool_call", recorder.on_tool)
    ctx.register_hook("subagent_start", recorder.on_subagent)
    ctx.register_hook("on_session_end", recorder.on_turn_end)

"""Trace hooks: every model call and tool call, as rows in the `traces`
table (locally, and in the team's cloud database once signed in), next to
the conversation they belong to. A personal session's traces (privacy.py) go
to the person's own database instead, as its conversation does: which
scheduled job a run belongs to is noted from each of its calls, a chat a job's
output was delivered into from the turn's history before its first model call
(`pre_llm_call`), a session that read the person's own database from that
call (under the session Hermes names for it, which may be one Hermes moved the
agent to without telling the provider), and a subagent follows the session
that started it (`subagent_start`), whose standing is read there, in its own
context and profile: Hermes may run the subagent as another profile.

Rows are buffered per session and written once per turn (Hermes's plugin
`on_session_end` hook fires at each turn's end), so a turn with ten tool
calls is one append, not eleven. No OpenTelemetry collector is involved. The
buffers belong to the process, not to one registration of the hooks: Hermes
registers them anew each time it builds an agent (a gateway builds one per
message), and a turn still running in one chat keeps its rows when another
chat's message arrives. Hermes's background review and `/btw` are forks that
share the session's id and have no turn end of their own: their rows wait
for the session's next turn end, are written when the provider shuts down,
and, once more sessions wait than are remembered, the oldest session's rows
are written then, never dropped. A session's rows are written in the profile
scope they were made in, whichever turn's call sets them going.

Hermes scrubs secrets from everything it hands a memory provider, but these
hooks get tool arguments and results as the tool saw them. So every text
cell is put through the same scrub (`redact_for_egress`) before it is kept:
a key a tool printed never reaches the team's database. The text of a call
that may write personal memory, or that read the person's own database, is
withheld (privacy.py): from its arguments, its result and its error. A model
call's row keeps the text the model wrote; its tool calls are rows of their
own.

Two tools' calls are read for whose they are, besides being traced. Hermes's
memory tool, which its background review also calls (a copy of the agent,
under the id of the session whose turn it follows), is noted before it writes
the memory files (`pre_tool_call`): an entry a turn that was not the owner's
own writes is noted with that turn's group chat (builtin.py), from what the
provider noted of the session (privacy.writer_of), and a write whose note
cannot be made is refused. A job Hermes's cron tool makes in such a turn is
noted too (privacy.note_others_job): Hermes keeps no origin for a job made
where nothing can be delivered back, and its runs must not be the owner's.

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
owner; a subagent's is the session's that started it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agent.memory_provider import ctx_bound, spawn_context_thread

from . import backend, builtin, chats, config, privacy, tables

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
# Sessions whose person, started turns and unwritten rows are remembered; a
# long-running gateway sees many, and only the latest few are still running.
MAX_SESSIONS = 1_024
# Seconds a shutdown waits for the rows still buffered to be written: local
# appends, quick, but a stuck disk must not hold up Hermes's exit.
SHUTDOWN_WRITE_WAIT_S = 5.0
# How much of a turn's user message ties the turn's start to its sync.
TURN_KEY_CHARS = 200
# Turns remembered per session: those started and not synced yet. A turn
# Hermes never syncs (interrupted, or with no reply) is dropped once a later
# turn of the session is synced.
TURNS_KEPT = 16
# How Hermes marks a message's images when it flattens the message for a
# memory provider: "[1 image] what is this?".
IMAGES_MARK = re.compile(r"^\[\d+ images?\]\s*")
# Hermes's built-in memory tool and its cron tool, as its dispatch names them
# to the tool hooks (an older name of a tool is mapped to its name first), and
# the cron tool's action that makes a job.
MEMORY_TOOL = "memory"
CRON_TOOL = "cronjob_manage"
CRON_CREATE = "create"
# What a memory tool call is told when whose its entries are cannot be noted.
NOT_NOTED = ("not saved: Infino could not note whose this memory is (its note was busy or could not be written); "
             "nothing was changed, so try again")


def scrub(text: str) -> str:
    """`text` scrubbed of secrets, as Hermes scrubs everything leaving the agent."""
    return _scrub(text)


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


# This machine's person and the profile for each session the provider started
# (`initialize`) or moved to (`on_session_switch`), and for each subagent's
# session (the session's that started it). A gateway turn's person is its
# sender instead, read from the turn.
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
    if chats.session("HERMES_SESSION_PLATFORM").lower() not in chats.OWNER_PLATFORMS:
        return chats.session("HERMES_SESSION_USER_NAME") or chats.session("HERMES_SESSION_USER_ID")
    return _identity_of(session_id)[0]


def _write(session_id: str, rows: list[dict], author: Optional[str] = None) -> None:
    """Write a session's rows, in the profile scope this runs in: to the
    person's own tables when the session is personal (privacy.py). `author`
    is the person when the caller read it in the turn's own context; else
    whoever this context's turn is for."""
    author = turn_person(session_id) if author is None else author
    profile = _identity_of(session_id)[1]
    for row in rows:
        row["author"] = author
        row["profile"] = profile
    try:
        b = backend.for_settings(config.load())
        b.append(tables.TRACES, rows, b.transcript_personal(session_id))
    except Exception as e:
        logger.warning("infino-traces: %d trace rows not written: %s", len(rows), e)


def _personal(session_id: str) -> bool:
    """Whether a session is personal (privacy.py), read in the profile scope
    and context this runs in; personal when that cannot be read."""
    try:
        return backend.for_settings(config.load()).transcript_personal(session_id)
    except Exception as e:
        logger.warning("infino-traces: could not read whether session %s is personal (kept personal): %s",
                       session_id, e)
        return True


def _note_entries(session_id: str, target: str, entries: list[str]) -> None:
    """Note whose the entries a call of Hermes's memory tool writes are
    (builtin.note_writers), in the profile scope this runs in: unless the
    session's latest turn was the owner's own, whose entries are the owner's
    with no note. A session nothing here noted is no one's: its entries are
    kept for no chat's recall, and never taken for the owner's. Raises when
    the note cannot be written."""
    writer = privacy.writer_of(session_id)
    if not entries or (writer is not None and writer.owners_entries):
        return
    home = config.home()
    builtin.note_writers(config.state_dir(home), home, target, entries,
                         chats.UNATTRIBUTED if writer is None else writer.chat)


def _note_job(session_id: str, args: Any, result: Any) -> None:
    """Note a job Hermes's cron tool made in a turn that was not the owner's
    own (privacy.note_others_job): Hermes keeps no origin for a job made where
    nothing can be delivered back (an API server session, a gateway's
    `/background` run, a Kanban worker, the Desktop's Bot Chat), and the job's
    runs must keep out the owner's personal memory. A session nothing here
    noted is not the owner's."""
    call, answer = privacy.json_value(args), privacy.json_value(result)
    if not isinstance(call, dict) or str(call.get("action") or "").strip().lower() != CRON_CREATE:
        return
    job_id = answer.get("job_id") if isinstance(answer, dict) and answer.get("success") else None
    writer = privacy.writer_of(session_id)
    if job_id and not (writer is not None and writer.owners_jobs):
        privacy.note_others_job(config.state_dir(config.home()), str(job_id))


@dataclass
class _Waiting:
    """A session's rows not written yet."""
    home: Path                                         # the profile they were made in
    write: Callable[..., None]                         # `_write`, bound to the scope they were made in
    rows: list[dict] = field(default_factory=list)


# Each session's rows not written yet, oldest session first, shared by every
# registration of the hooks.
_buffers: OrderedDict[str, _Waiting] = OrderedDict()
_buffers_lock = threading.Lock()


def _write_later(session_id: str, waiting: _Waiting) -> None:
    """Write rows taken out of the buffers on a thread of their own, in the
    scope they were made in."""
    spawn_context_thread(lambda: waiting.write(session_id, waiting.rows), name="infino-traces-flush").start()


def write_buffered(home: Path, wait_s: float = SHUTDOWN_WRITE_WAIT_S) -> None:
    """Write now every row still buffered for the profile at `home` - those a
    fork added after its session's last turn ended - and wait for them, at
    most `wait_s`. The provider calls this as it shuts down; other profiles'
    rows wait for their own."""
    profile = Path(home).resolve()
    with _buffers_lock:
        mine = [(s, _buffers.pop(s)) for s in [s for s, w in _buffers.items() if w.home == profile]]
    if not mine:
        return

    def write_all() -> None:
        for session_id, waiting in mine:
            waiting.write(session_id, waiting.rows)

    writer = spawn_context_thread(write_all, name="infino-traces-shutdown")
    writer.start()
    writer.join(wait_s)


class TraceRecorder:
    def _add(self, session_id: str, row: dict) -> None:
        # Read in the call's own context: a scheduled job's run says which job it is.
        privacy.note_cron_job(session_id or "")
        session = session_id or ""
        with _buffers_lock:
            waiting = _buffers.get(session)
            if waiting is None:
                waiting = _buffers[session] = _Waiting(Path(config.home()).resolve(), ctx_bound(_write))
            waiting.rows.append(row)
            _buffers.move_to_end(session)
            evicted = []
            while len(_buffers) > MAX_SESSIONS:
                evicted.append(_buffers.popitem(last=False))
        for old_session, old in evicted:
            _write_later(old_session, old)

    def on_llm_call(self, **kw: Any) -> None:
        """As a turn starts, before its first model call: note the turn's id
        for its conversation rows, and any job output delivered into this
        chat, which the turn's history holds."""
        note_turn_start(str(kw.get("session_id") or ""), str(kw.get("turn_id") or ""), kw.get("user_message"))
        history = list(kw.get("conversation_history") or [])
        history.append({"role": "user", "content": kw.get("user_message")})
        privacy.note_deliveries(str(kw.get("session_id") or ""), history)

    def on_subagent(self, **kw: Any) -> None:
        """A subagent's session is as personal as the session that started it,
        and its rows are that session's person's and profile's: Hermes starts
        no provider for a subagent, which would note them.

        Hermes fires this as the parent, before the subagent runs, and may run
        the subagent as another profile (a delegation route): there the
        parent's profile settings are not read, and that profile may load its
        own copy of this plugin, which knows nothing noted here. So the
        parent's standing is read now, in its own context and profile, and a
        personal parent's subagent is noted personal."""
        child, parent = str(kw.get("child_session_id") or ""), str(kw.get("parent_session_id") or "")
        privacy.note_parent(child, parent)
        if not child or child == parent:
            return
        identity = _identity_of(parent)
        if any(identity) and not any(_identity_of(child)):
            note_identity(child, *identity)
        if parent:
            privacy.note_cron_job(parent)  # the job a run belongs to is said in the run's own context
            if _personal(parent):
                privacy.note_personal(child)

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

    def on_tool_start(self, **kw: Any) -> Optional[dict]:
        """Before a call of Hermes's memory tool writes a memory file: note
        whose the entries it writes whole are (_note_entries), so no comparison
        of the files, in any process, finds one before its note. A write whose
        note cannot be made is refused, as Hermes refuses any call this hook
        fails or hangs on (`pre_tool_call` fails closed)."""
        if kw.get("tool_name") != MEMORY_TOOL:
            return None
        # A batch that removes some entries and adds others: its removals are
        # earlier versions of what it adds (provider.on_memory_write).
        builtin.note_consolidation(str(kw.get("tool_call_id") or ""), privacy.json_value(kw.get("args")))
        try:
            _note_entries(str(kw.get("session_id") or ""), *builtin.call_entries(privacy.json_value(kw.get("args"))))
        except Exception as e:
            logger.warning("infino: a memory write was refused, since whose it is could not be noted: %s", e)
            return {"action": "block", "message": NOT_NOTED}
        return None

    def _note_call(self, tool: str, session: str, kw: dict) -> None:
        """After a call: note whose the entries a memory tool call edited in
        place are (a patch's, known only now), and a job the cron tool made."""
        try:
            if tool == MEMORY_TOOL:
                target, _ = builtin.call_entries(privacy.json_value(kw.get("args")))
                _note_entries(session, target, builtin.patched_entries(privacy.json_value(kw.get("result"))))
            elif tool == CRON_TOOL:
                _note_job(session, kw.get("args"), kw.get("result"))
        except Exception as e:
            logger.warning("infino: could not note whose a %s call's write is: %s", tool, e)

    def on_tool(self, **kw: Any) -> None:
        tool = str(kw.get("tool_name") or "")
        session = str(kw.get("session_id") or "")
        if tool == MEMORY_TOOL:
            builtin.end_consolidation(str(kw.get("tool_call_id") or ""))
        self._note_call(tool, session, kw)
        if session and privacy.read_personal(tool, kw.get("result")):
            # The session the agent runs, as Hermes names it for the call: Hermes may have moved
            # the agent to another session without telling the provider, which notes the one it knows.
            privacy.note_personal(session)
        personal = privacy.is_personal_call(tool, kw.get("args"))
        # A failed call's error can quote its arguments ("No entry matched '<text>'").
        error = kw.get("error_message") or ""
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
            "error": privacy.WITHHELD if personal and error else _clip(error),
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
        with _buffers_lock:
            waiting = _buffers.pop(session, None)
        if waiting is None or not waiting.rows:
            return
        # The person as the turn that ends names them, read in the hook's own context.
        author = turn_person(session)
        spawn_context_thread(lambda: waiting.write(session, waiting.rows, author), name="infino-traces-flush").start()


def register_hooks(ctx) -> None:
    recorder = TraceRecorder()
    ctx.register_hook("pre_llm_call", recorder.on_llm_call)
    ctx.register_hook("pre_tool_call", recorder.on_tool_start)
    ctx.register_hook("post_api_request", recorder.on_api)
    ctx.register_hook("post_tool_call", recorder.on_tool)
    ctx.register_hook("subagent_start", recorder.on_subagent)
    ctx.register_hook("on_session_end", recorder.on_turn_end)

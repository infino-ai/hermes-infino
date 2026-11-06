"""What is kept out of the team's tables: personal memory, and personal
transcripts.

Personal memory. `conversations` and `traces` record every tool call with
its arguments, and both go to the team's database. A call that writes
personal memory carries the private text in its arguments - `infino_remember`
with scope "personal", or Hermes's own `memory` tool on its `user` target
(USER.md) - so the text is withheld from those rows: the call is kept, with
WITHHELD in place of the text, and so is its result, which can echo the
entries. A call that reads the person's own database (`infino_personal_search`)
is withheld the same way: what it asked, and all it answered.

What a person says in a conversation is the conversation, and is kept as
said; this covers the memory written from it.

Personal transcripts. A session's whole conversation and traces are
personal, and go to the person's own database (or stay on this machine)
instead of the team's, when:
- the profile is personal (`INFINO_PERSONAL_PROFILE`): everything it does;
- the session is a run of a personal scheduled job (`INFINO_PERSONAL_JOBS`):
  what it read and wrote on its way to its output, and the sessions it
  delegated to;
- a personal job's output was delivered into the session's chat: Hermes
  mirrors a delivered job's output into the chat's transcript, labelled with
  the job's name, and what follows in that chat is about it. The session
  stays personal from then on, even once compaction has dropped the label
  (the backend sees that its earlier turns were kept personal);
- a personal job's output was posted into a group chat (`deliver: room:<id>`):
  Hermes shows the post in a member's turn, labelled with the job's name. The
  member's own profile has to list the job;
- the session read the person's own database (`infino_personal_search`): what
  it read is in the session now, so the session is personal from then on.

Everything else stays the team's, as before. Which job a run belongs to is
Hermes's to say (`cron.execution_identity`); a job is named by its id or its
name, the way Hermes's own `hermes cron` commands name one.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# What stands in for personal text: a personal memory's, or a personal search's.
WITHHELD = "[personal, kept out of the team's tables]"
# The text fields of the memory tools' arguments, and of the personal search's.
TEXT_FIELDS = ("content", "old_text", "new_text", "query")
# The tools that read the person's own database: every call of one is personal.
PERSONAL_READ_TOOLS = ("infino_personal_search",)

# How Hermes labels a scheduled job's output when it mirrors it into a chat's
# transcript: a user message whose first line is "[Cron delivery: <job>]", the
# job named by its name, or its id when it has none.
DELIVERY_PREFIX = "[Cron delivery: "
DELIVERY_SUFFIX = "]"
# How Hermes shows a scheduled job's output posted into a group chat
# (`deliver: room:<id>`) in a member's turn: a line of its own,
# `Scheduled job "<job>" (automated post, not the user): <output>`, the job
# named by its name, or its id when it has none. A person's or a member's
# message keeps its own newlines, so a later line of one can begin with the
# label too; that only ever makes the session personal, never the team's.
ROOM_DELIVERY_PREFIX = 'Scheduled job "'
ROOM_DELIVERY_MARK = '" (automated post, not the user):'
# A cron run's session id when Hermes's scheduler names it: cron_<job id>_<date>_<time>.
CRON_SESSION = re.compile(r"^cron_(.+)_\d{8}_\d{6}$")
# Hermes's record of the scheduled job the running code belongs to. Read only
# where Hermes has loaded it: a process that has run no job has no cron run.
CRON_IDENTITY_MODULE = "cron.execution_identity"
# Sessions whose standing is remembered; a long-running gateway sees many, and
# only the latest few are still running.
MAX_SESSIONS = 1_024


def _args(arguments: Any) -> Any:
    """Arguments as a dict when they are JSON text; otherwise as given."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def is_personal_call(tool_name: str, arguments: Any) -> bool:
    """Whether a tool call writes, changes or deletes personal memory, or
    reads the person's own database."""
    if tool_name in PERSONAL_READ_TOOLS:
        return True
    args = _args(arguments)
    if not isinstance(args, dict):
        return False
    if tool_name in ("infino_remember", "infino_forget"):
        return str(args.get("scope") or "").strip().lower() == "personal"
    if tool_name == "memory":
        return str(args.get("target") or "").strip().lower() == "user"
    return False


def withhold(tool_name: str, arguments: Any) -> Any:
    """The arguments with a personal call's text withheld, in the form they
    came in (JSON text or a dict); others unchanged."""
    if not is_personal_call(tool_name, arguments):
        return arguments
    if not isinstance(_args(arguments), dict):  # nothing to tell apart: all of it
        return WITHHELD
    args = dict(_args(arguments))
    for field_name in TEXT_FIELDS:
        if field_name in args:
            args[field_name] = WITHHELD
    if isinstance(args.get("operations"), list):
        args["operations"] = [{k: (WITHHELD if k in TEXT_FIELDS else v) for k, v in op.items()}
                              if isinstance(op, dict) else op for op in args["operations"]]
    return json.dumps(args) if isinstance(arguments, str) else args


def message_text(content: Any) -> str:
    """A message's text: the string, or its text parts joined."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


# -- personal transcripts ---------------------------------------------------------

def is_personal_job(jobs: Iterable[str], job_id: str, job_name: str) -> bool:
    """Whether a job is one of `jobs`: named by its exact id, or by its name in any case."""
    return any(j == job_id or j.lower() == job_name.lower() for j in jobs)


def cron_job(session_id: str = "") -> Optional[tuple[str, str]]:
    """The (id, name) of the scheduled job the running code belongs to, from
    Hermes's record of the run (a delegated child reads none: it is not the
    run); else from a cron session's id; None outside a cron run."""
    identity = sys.modules.get(CRON_IDENTITY_MODULE)
    try:
        execution = identity.current_cron_execution() if identity is not None else None
    except Exception:  # a read of Hermes's record never fails a call
        execution = None
    if execution is not None:
        return str(execution.job_id), str(execution.job_name)
    found = CRON_SESSION.match(session_id or "")
    return _job_ref(found.group(1)) if found else None


def _job_ref(ref: str) -> tuple[str, str]:
    """A job's (id, name) from a reference to it (its id, or its name), through
    Hermes's job store; the reference as both when the store cannot say."""
    try:
        from cron.jobs import resolve_job_ref  # Hermes's own resolution: exact id, then name
        job = resolve_job_ref(ref)
    except Exception:  # no job store in this process, or a name several jobs share
        job = None
    if not job:
        return ref, ref
    return str(job.get("id") or ref), str(job.get("name") or job.get("id") or ref)


def delivered_jobs(messages: Iterable[dict]) -> list[str]:
    """The jobs (as their labels name them) whose output Hermes delivered into
    these messages: mirrored into a chat, or posted into a group chat. A label
    may begin any line: Hermes merges consecutive user messages, so a delivery
    can share one with what the person wrote next, and a group chat member's
    turn lists the room's new messages one per line."""
    found = []
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        for line in message_text(m.get("content")).splitlines():
            line = line.strip()
            if line.startswith(DELIVERY_PREFIX) and line.endswith(DELIVERY_SUFFIX):
                found.append(line[len(DELIVERY_PREFIX):-len(DELIVERY_SUFFIX)].strip())
            elif line.startswith(ROOM_DELIVERY_PREFIX) and ROOM_DELIVERY_MARK in line:
                found.append(line[len(ROOM_DELIVERY_PREFIX):line.index(ROOM_DELIVERY_MARK)].strip())
    return found


@dataclass
class _Session:
    jobs: set = field(default_factory=set)  # (id, name): the job it runs, the jobs delivered into it
    runs_job: bool = False                  # the job it runs is noted
    parent: str = ""                        # the session that delegated to it
    personal: bool = False                  # its earlier turns were kept personal
    looked: bool = False                    # this machine's personal tables were asked about it


_sessions: OrderedDict[str, _Session] = OrderedDict()
_sessions_lock = threading.Lock()


def _session(session_id: str) -> _Session:
    """The session's record, made when new (call with the lock held)."""
    s = _sessions.get(session_id)
    if s is None:
        s = _sessions[session_id] = _Session()
    _sessions.move_to_end(session_id)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return s


def note_cron_job(session_id: str) -> None:
    """Note the scheduled job a session runs, read in its own context (a
    model or tool call of the run); once per session."""
    with _sessions_lock:
        known = _sessions.get(session_id or "")
        if known is not None and known.runs_job:
            return
    job = cron_job(session_id)
    if job is not None:
        with _sessions_lock:
            s = _session(session_id or "")
            s.jobs.add(job)
            s.runs_job = True


def note_deliveries(session_id: str, messages: Iterable[dict]) -> None:
    """Note the jobs whose output was delivered into a session's messages."""
    labels = set(delivered_jobs(messages))
    if not labels:
        return
    jobs = [_job_ref(label) for label in labels]
    with _sessions_lock:
        _session(session_id or "").jobs.update(jobs)


def note_parent(child_session_id: str, parent_session_id: str) -> None:
    """Note that a session was delegated to by another (a subagent)."""
    if child_session_id and parent_session_id and child_session_id != parent_session_id:
        with _sessions_lock:
            _session(child_session_id).parent = parent_session_id


def note_personal(session_id: str) -> None:
    """Note a session whose turns are kept personal."""
    with _sessions_lock:
        _session(session_id or "").personal = True


def first_look(session_id: str) -> bool:
    """True once per session (while it is remembered): whether this machine's
    personal tables still have to be asked about it."""
    with _sessions_lock:
        s = _session(session_id or "")
        looked, s.looked = s.looked, True
        return not looked


def transcript_personal(settings, session_id: str, messages: Optional[Iterable[dict]] = None) -> bool:
    """Whether a session's conversation and traces are personal (see the
    module's docstring). `messages`, when given, are the session's messages,
    searched for a delivered job's output unless the session is known to be
    personal already."""
    if settings.personal_profile:
        return True
    if _noted_personal(settings, session_id):
        return True
    if messages is None:
        return False
    note_deliveries(session_id, messages)
    return _noted_personal(settings, session_id)


def _noted_personal(settings, session_id: str) -> bool:
    """Whether what is noted about a session makes it personal. A delegated
    session is as personal as the session that started it."""
    session, seen = session_id or "", set()
    while session not in seen:
        seen.add(session)
        with _sessions_lock:
            s = _sessions.get(session)
            if s is None:
                return False
            personal, jobs, parent = s.personal, list(s.jobs), s.parent
        if personal or any(is_personal_job(settings.personal_jobs, *job) for job in jobs):
            return True
        if not parent:
            return False
        session = parent
    return False

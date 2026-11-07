"""What is kept out of the team's tables: personal memory, and personal
transcripts.

Personal memory. `conversations` and `traces` record every tool call with
its arguments, and both go to the team's database. A call that writes,
changes or deletes personal memory, or may, carries the private text in its
arguments - a call of `infino_remember` or `infino_forget` that does not keep
to the team's memory, or Hermes's own `memory` tool on its `user` target
(USER.md) - so the text is withheld from those rows: the call is kept, with
WITHHELD in place of the text, and so are its result, which can echo the
entries, and its error, which can quote them. A call that reads the person's
own database (`infino_personal_search`) is withheld the same way: what it
asked, and all it answered.

What a person says in a conversation is the conversation, and is kept as
said; this covers the memory written from it. What memory recalled for a
turn is not said by anyone: Hermes keeps a multimodal turn's recall in the
user message itself, and a message's text leaves it out (`message_text`).

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
- a personal job's output was noted into a local chat (`deliver: local:<chat>`,
  under its "Cronjob Response" header) or handed to a profile's Bot Chat as a
  turn (`deliver: bot-chat`). The receiving profile has to list the job;
- the session read the person's own database (`infino_personal_search`): what
  it read is in the session now, so the session is personal from then on;
- the session is a subagent's (Hermes's `delegate_task`) that no
  `subagent_start` this copy of the plugin heard of started: Hermes ran it as
  another profile, whose hooks reach that profile's own copy, so whose work it
  does cannot be told here.

Everything else stays the team's, as before. Which job a run belongs to is
Hermes's to say (`cron.execution_identity`); a job is named by its id or its
name, the way Hermes's own `hermes cron` commands name one. Hermes labels a
job's delivered output with its name: when several of the profile's jobs share
that name, the output is personal if any of them is listed. A job of another
profile is known here only by that name.

Whose writes a session makes. Hermes tells a provider of its sessions' turns,
but not of two writes that follow from them: its background review, a copy of
the agent, saves to the memory files under the id of the session whose turn it
follows, and its cron tool makes a scheduled job, keeping where it was made
only when that chat can be delivered to. So the provider notes, per session,
whose its latest turn was (`Writer`), and the tool hooks read it (traces.py):
an entry a turn that was not the owner's own wrote stays that chat's
(builtin.py), and a job made in such a turn never runs as the owner's
(`note_others_job`).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config
from .recall import TEAM

logger = logging.getLogger(__name__)

try:  # Hermes's mark on the code a delegated child (`delegate_task`) runs.
    from agent.delegation_context import is_delegated_child_context as _is_delegated_child
except ImportError:  # pragma: no cover - a Hermes without it marks none
    _is_delegated_child = None

# What stands in for personal text: a personal memory's, or a personal search's.
WITHHELD = "[personal, kept out of the team's tables]"
# The text fields of the memory tools' arguments, and of the personal search's.
TEXT_FIELDS = ("content", "old_text", "new_text", "query")
# The tools that read the person's own database: every call of one is personal.
PERSONAL_READ_TOOLS = ("infino_personal_search",)
# The plugin's own memory tools. A call of one is the team's only when it says
# so: a scope of "team", or, for a remember, none at all (it saves the team's).
# A forget with no scope deletes from personal memory too, and a scope the
# provider refuses names neither: either may carry personal memory's text.
REMEMBER_TOOL = "infino_remember"
FORGET_TOOL = "infino_forget"

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
# How Hermes notes a scheduled job's output into a local chat's history
# (`deliver: local:<chat>`, cron/scheduler_delivery.py): its output wrapped
# under a line "Cronjob Response: <job>", the job named by its name (or its id
# when it has none), and the job's id on the next line, "(job_id: <id>)".
# Without the wrapper (`cron.wrap_response: false`) nothing names the job.
LOCAL_DELIVERY_PREFIX = "Cronjob Response: "
LOCAL_DELIVERY_ID = re.compile(r"^\(job_id: (.+)\)$")
# How Hermes hands a scheduled job's output to a profile's Bot Chat
# (`deliver: bot-chat`) as an inbound turn: its first line is
# `[Cronjob "<job>" output — scheduled job, not the user. Review it, ...]`,
# the job named by its name, or its id when it has none.
BOT_CHAT_DELIVERY_PREFIX = '[Cronjob "'
BOT_CHAT_DELIVERY_MARK = '" output — scheduled job, not the user.'
# How Hermes fences the memory it recalled for a turn (agent/memory_context.py).
# A multimodal (list) user message carries the turn's injected context - the
# fenced recall, then any plugin's context - as a text part of its own, kept
# with the message (agent/turn_context.py); a string message keeps it in a
# sidecar a provider is never handed.
RECALL_FENCE = "<memory-context>"
# A cron run's session id when Hermes's scheduler names it: cron_<job id>_<date>_<time>.
CRON_SESSION = re.compile(r"^cron_(.+)_\d{8}_\d{6}$")
# Hermes's record of the scheduled job the running code belongs to. Read only
# where Hermes has loaded it: a process that has run no job has no cron run.
CRON_IDENTITY_MODULE = "cron.execution_identity"
# Sessions whose standing is remembered; a long-running gateway sees many, and
# only the latest few are still running.
MAX_SESSIONS = 1_024
# Beside the plugin's databases: the scheduled jobs made in a turn that was not
# the owner's own, one id a line (`note_others_job`).
OTHERS_JOBS = "others_jobs"


def json_value(arguments: Any) -> Any:
    """A tool call's arguments (or answer) as a dict when they are JSON text;
    otherwise as given."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def is_personal_call(tool_name: str, arguments: Any) -> bool:
    """Whether a tool call writes, changes or deletes personal memory, or may
    (a call of the plugin's memory tools that does not keep to the team's),
    or reads the person's own database."""
    if tool_name in PERSONAL_READ_TOOLS:
        return True
    args = json_value(arguments)
    if tool_name in (REMEMBER_TOOL, FORGET_TOOL):
        return not _keeps_to_team(tool_name, args)
    if not isinstance(args, dict):
        return False
    if tool_name == "memory":
        return str(args.get("target") or "").strip().lower() == "user"
    return False


def _keeps_to_team(tool_name: str, args: Any) -> bool:
    """Whether a call of the plugin's memory tools keeps to the team's memory:
    its scope is "team" (in any case or spacing, as the provider reads it), or
    it is a remember that names none. Arguments that cannot be read say
    neither."""
    if not isinstance(args, dict):
        return False
    scope = args.get("scope")
    if tool_name == REMEMBER_TOOL and scope in (None, ""):
        return True
    return str(scope or "").strip().lower() == TEAM


def read_personal(tool_name: str, result: Any) -> bool:
    """Whether a call read the person's own database: a call of a personal
    read tool whose answer does not say it was refused or failed (an
    `error`, as the provider and Hermes answer one). An answer that cannot be
    read counts as read."""
    if tool_name not in PERSONAL_READ_TOOLS:
        return False
    answer = json_value(result)
    return not (isinstance(answer, dict) and answer.get("error"))


def withhold(tool_name: str, arguments: Any) -> Any:
    """The arguments with a personal call's text withheld, in the form they
    came in (JSON text or a dict); others unchanged."""
    if not is_personal_call(tool_name, arguments):
        return arguments
    if not isinstance(json_value(arguments), dict):  # nothing to tell apart: all of it
        return WITHHELD
    args = dict(json_value(arguments))
    for field_name in TEXT_FIELDS:
        if field_name in args:
            args[field_name] = WITHHELD
    if isinstance(args.get("operations"), list):
        args["operations"] = [{k: (WITHHELD if k in TEXT_FIELDS else v) for k, v in op.items()}
                              if isinstance(op, dict) else op for op in args["operations"]]
    return json.dumps(args) if isinstance(arguments, str) else args


def _recalled(part: dict) -> bool:
    """Whether a part of a message is the context Hermes added for the turn,
    which starts with the recall's fence (RECALL_FENCE)."""
    text = part.get("text")
    return part.get("type") == "text" and isinstance(text, str) and text.startswith(RECALL_FENCE)


def message_text(content: Any) -> str:
    """A message's text: the string, or its text parts joined, without the part
    Hermes adds to a multimodal turn for its recall: that is what memory -
    personal memory included - showed the model, not what anyone said."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and not _recalled(p))
    return content if isinstance(content, str) else ""


# -- personal transcripts ---------------------------------------------------------

def is_personal_job(jobs: Iterable[str], job_id: str, job_name: str) -> bool:
    """Whether a job is one of `jobs`: named by its exact id, or by its name in any case."""
    return any(j == job_id or j.lower() == job_name.lower() for j in jobs)


def cron_job(session_id: str = "") -> list[tuple[str, str]]:
    """The (id, name) of the scheduled job the running code belongs to, from
    Hermes's record of the run (a delegated child reads none: it is not the
    run); else of each job a cron session's id may name; [] outside a cron run."""
    identity = sys.modules.get(CRON_IDENTITY_MODULE)
    try:
        execution = identity.current_cron_execution() if identity is not None else None
    except Exception:  # a read of Hermes's record never fails a call
        execution = None
    if execution is not None:
        return [(str(execution.job_id), str(execution.job_name))]
    found = CRON_SESSION.match(session_id or "")
    return _job_refs(found.group(1)) if found else []


def _job_refs(ref: str) -> list[tuple[str, str]]:
    """The (id, name) of each job a reference (an id, or a name) may name,
    through Hermes's job store: the job it resolves to, or every job of a name
    several share (any of them may be the one meant); the reference as both
    when the store cannot say (no job store in this process, or another
    profile's job)."""
    try:
        from cron.jobs import AmbiguousJobReference, resolve_job_ref  # Hermes's own resolution: exact id, then name
    except Exception:  # no job store in this process
        return [(ref, ref)]
    try:
        found = [resolve_job_ref(ref)]
    except AmbiguousJobReference as e:
        found = list(e.matches)
    except Exception:  # an unreadable job store says nothing
        found = []
    jobs = [(str(j.get("id") or ref), str(j.get("name") or j.get("id") or ref)) for j in found if j]
    return jobs or [(ref, ref)]


def delivered_jobs(messages: Iterable[dict]) -> list[str]:
    """The jobs (as their labels name them) whose output Hermes delivered into
    these messages: mirrored into a chat, posted into a group chat, noted into
    a local chat (named by its name and by its id), or handed to a Bot Chat. A
    label may begin any line: Hermes merges consecutive user messages, so a
    delivery can share one with what the person wrote next, and a group chat
    member's turn lists the room's new messages one per line."""
    found = []
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        lines = [line.strip() for line in message_text(m.get("content")).splitlines()]
        for i, line in enumerate(lines):
            if line.startswith(DELIVERY_PREFIX) and line.endswith(DELIVERY_SUFFIX):
                found.append(line[len(DELIVERY_PREFIX):-len(DELIVERY_SUFFIX)].strip())
            elif line.startswith(ROOM_DELIVERY_PREFIX) and ROOM_DELIVERY_MARK in line:
                found.append(line[len(ROOM_DELIVERY_PREFIX):line.index(ROOM_DELIVERY_MARK)].strip())
            elif line.startswith(BOT_CHAT_DELIVERY_PREFIX) and BOT_CHAT_DELIVERY_MARK in line:
                found.append(line[len(BOT_CHAT_DELIVERY_PREFIX):line.index(BOT_CHAT_DELIVERY_MARK)].strip())
            elif line.startswith(LOCAL_DELIVERY_PREFIX):
                found.append(line[len(LOCAL_DELIVERY_PREFIX):].strip())
                job_id = LOCAL_DELIVERY_ID.match(lines[i + 1]) if i + 1 < len(lines) else None
                if job_id:
                    found.append(job_id.group(1).strip())
    return [label for label in found if label]


@dataclass(frozen=True)
class Writer:
    """Whose a session's writes are, as the provider saw its latest turn."""
    chat: str = ""               # the group chat a MEMORY.md entry it writes is kept for ("" for none)
    owners_entries: bool = False  # a USER.md entry it writes may be the owner's personal memory
    owners_jobs: bool = False     # a scheduled job it makes is the owner's own


@dataclass
class _Session:
    jobs: set = field(default_factory=set)  # (id, name): the job it runs, the jobs delivered into it
    runs_job: bool = False                  # the job it runs is noted
    parent: str = ""                        # the session that delegated to it
    personal: bool = False                  # its earlier turns were kept personal
    looked: bool = False                    # this machine's personal tables were asked about it
    look_again: bool = False                # that look could not read them: they are asked again
    writer: Optional[Writer] = None         # whose its writes are (note_writer)


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


def _touch(session_id: str) -> Optional[_Session]:
    """A session's record, kept among the latest with the sessions that
    started it (call with the lock held): a parent waiting on its subagent
    makes no calls of its own, and its standing decides the subagent's. None
    when the session has no record."""
    s = _sessions.get(session_id)
    session, seen = session_id, set()
    while session in _sessions and session not in seen:
        seen.add(session)
        _sessions.move_to_end(session)
        session = _sessions[session].parent
    return s


def note_cron_job(session_id: str) -> None:
    """Note the scheduled job a session runs, read in its own context (a
    model or tool call of the run); once per session. Every call keeps the
    session's record among the latest, so a session still running is never
    the one forgotten."""
    with _sessions_lock:
        known = _touch(session_id or "")
        if known is not None and known.runs_job:
            return
    jobs = cron_job(session_id)
    if jobs:
        with _sessions_lock:
            s = _session(session_id or "")
            s.jobs.update(jobs)
            s.runs_job = True


def note_deliveries(session_id: str, messages: Iterable[dict]) -> None:
    """Note the jobs whose output was delivered into a session's messages."""
    labels = set(delivered_jobs(messages))
    if not labels:
        return
    jobs = [job for label in labels for job in _job_refs(label)]
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
    """Whether this machine's personal tables still have to be asked about a
    session: once per session (while it is remembered), and again after a look
    that could not read them (`look_again`). Kept with the session, not with a
    backend, so the backend a change of settings makes asks again too."""
    with _sessions_lock:
        s = _session(session_id or "")
        owed = not s.looked or s.look_again
        s.looked, s.look_again = True, False
        return owed


def look_again(session_id: str) -> None:
    """A look at this machine's personal tables could not read them: the
    session's next look asks them again (`first_look`)."""
    with _sessions_lock:
        _session(session_id or "").look_again = True


# -- whose writes a session makes ---------------------------------------------------

def note_writer(session_id: str, writer: Writer) -> None:
    """Note whose a session's writes are, as its latest turn says."""
    with _sessions_lock:
        _session(session_id or "").writer = writer


def writer_of(session_id: str) -> Optional[Writer]:
    """Whose a session's writes are: as its provider noted (the background
    review's are the session's whose turn it follows: it runs under that id),
    or, for a subagent, the session's that started it. None when nothing here
    noted it."""
    session, seen = session_id or "", set()
    with _sessions_lock:
        while session in _sessions and session not in seen:
            seen.add(session)
            s = _sessions[session]
            if s.writer is not None:
                return s.writer
            session = s.parent
    return None


def note_others_job(state_dir: Path, job_id: str) -> None:
    """Note that a scheduled job was made in a turn that was not the owner's
    own: its runs are not the owner's (`others_job`). Raises when the note
    cannot be written."""
    config.make_private_dir(Path(state_dir))
    with open(Path(state_dir) / OTHERS_JOBS, "a") as f:
        f.write(job_id + "\n")


def others_job(state_dir: Path, job_id: str) -> bool:
    """Whether a scheduled job was noted as made in a turn that was not the
    owner's own; True when the note cannot be read."""
    path = Path(state_dir) / OTHERS_JOBS
    try:
        return path.exists() and job_id in path.read_text().split()
    except (OSError, UnicodeDecodeError) as e:
        logger.info("infino: could not read which jobs were made for others (job %s taken as one): %s", job_id, e)
        return True


def transcript_personal(settings, session_id: str, messages: Optional[Iterable[dict]] = None) -> bool:
    """Whether a session's conversation and traces are personal (see the
    module's docstring). `messages`, when given, are the session's messages,
    searched for a delivered job's output unless the session is known to be
    personal already. Read in the context of the code the session runs."""
    if settings.personal_profile or _unclaimed_child(session_id):
        return True
    if _noted_personal(settings, session_id):
        return True
    if messages is None:
        return False
    note_deliveries(session_id, messages)
    return _noted_personal(settings, session_id)


def _unclaimed_child(session_id: str) -> bool:
    """Whether the running code is a delegated child's (Hermes's
    `delegate_task`) whose session no `subagent_start` here named a parent of:
    the session that started it ran as another profile, whose hooks reach
    that profile's own copy of this plugin, so whose work it does cannot be
    told here."""
    if _is_delegated_child is None or not _is_delegated_child():
        return False
    with _sessions_lock:
        s = _sessions.get(session_id or "")
        return s is None or not s.parent


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

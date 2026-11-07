"""Hermes's built-in memory files, MEMORY.md and USER.md, read so the
`memory` table keeps up with them whoever writes them.

Hermes tells a memory provider about a built-in memory write only when the
agent's own memory tool makes it (`on_memory_write`). Its background review -
a copy of the agent Hermes starts every few turns (`memory.nudge_interval`)
to save what a conversation taught it - writes the same files with every
memory provider switched off, and a person may edit them by hand. So the
provider compares the files with the rows it keeps from them (`memory` rows
of origin `builtin:memory` / `builtin:user`, on this machine) at its start
and after each turn: an entry no row holds is added, credited to the profile's
owner, and a row whose entry has left the file is deleted, here and in the
cloud.

Who wrote an entry is noted as it is written (`note_writers`): the review is
a copy of the agent, and Hermes runs the plugins' tool hooks for its memory
tool calls under the id of the session whose turn it follows (traces.py). An
entry a turn that was not the owner's own wrote is noted with the group chat
that turn was in, if any, before Hermes writes it to the file, so no
comparison, in any process, finds it first: a MEMORY.md entry is then kept
for that chat alone, and a USER.md entry is never taken for the owner's. An
entry no turn noted - a hand edit, a write in the owner's own session - is
the owner's.

A file that is missing or cannot be read says nothing: no row is deleted for
it. Entries are compared scrubbed of secrets, as Hermes scrubs what it hands
a provider, and a row of a USER.md entry may carry the "About <name>: "
prefix the provider writes it under.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config
from .chats import UNATTRIBUTED

try:  # Hermes's own separator of a memory file's entries.
    from tools.memory_tool_store import ENTRY_DELIMITER
except ImportError:  # pragma: no cover - only outside a Hermes install
    ENTRY_DELIMITER = "\n§\n"

logger = logging.getLogger(__name__)

# Each built-in memory target and its file under the profile's `memories/`.
FILES = {"memory": "MEMORY.md", "user": "USER.md"}
MEMORIES_DIR = "memories"
# Beside the plugin's databases: held while the files and the rows are compared
# and while a built-in memory row is added, so two processes of one profile
# never add the same entry twice.
LOCK_FILE = "builtin.lock"
# The start of the prefix a USER.md entry's row is written under.
ABOUT = "About "
ABOUT_END = ": "
# Beside the plugin's databases: the entries of the memory files that a turn
# which was not the owner's own wrote, with the group chats such turns were in
# (JSON: target -> entry -> {"chats": [chat key...], "at": milliseconds}), and
# the lock its writers take.
WRITERS = "builtin_writers.json"
WRITERS_LOCK = "builtin_writers.lock"
# The most a reader or writer of that note waits for another process holding
# it: the tool hook that notes an entry runs before Hermes's memory tool
# writes, and Hermes refuses every tool call while a pre_tool_call hook hangs.
WRITERS_LOCK_WAIT_S = 5.0
# How often a bounded wait for one of the profile's locks tries it again.
LOCK_RETRY_S = 0.05
# What an earlier version kept instead: when the group chats last had a turn.
# Found with no note of writers, it means an entry not compared yet may be a
# group's, which nothing now says (_read_writers).
EARLIER_WRITERS = "review_chats.json"
# How long a note of an entry no file holds is kept: Hermes may hold a write for
# approval (`/memory pending`) before it reaches the file.
WRITER_NOTE_KEEP_MS = 30 * 24 * 60 * 60 * 1000
# The memory tool's operations that write their `content` as an entry whole: an
# add, and a replace (one Hermes held for approval before it took replace away).
WHOLE_ENTRY_ACTIONS = ("add", "replace")


def read_entries(home: Path, target: str) -> Optional[list[str]]:
    """The entries of a target's file, as Hermes parses them (split on its
    separator, stripped, empty ones dropped, first of each kept); None when
    the file is missing or cannot be read."""
    path = Path(home) / MEMORIES_DIR / FILES[target]
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    return list(dict.fromkeys(e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e))


def entry_of(text: str, entries: set[str]) -> Optional[str]:
    """The entry a row's text holds: the text itself, or what follows an
    "About <name>: " prefix; None when it holds none of `entries`."""
    if text in entries:
        return text
    if text.startswith(ABOUT):
        at = text.find(ABOUT_END)
        while at != -1:
            rest = text[at + len(ABOUT_END):]
            if rest in entries:
                return rest
            at = text.find(ABOUT_END, at + 1)
    return None


def compare(entries: list[str], rows: list) -> tuple[list[str], list]:
    """The entries no row holds, in file order, and the rows whose entry is
    gone. `rows` are (personal, row) pairs, as `Backend.builtin_rows` gives."""
    present = set(entries)
    held = set()
    stale = []
    for flagged in rows:
        entry = entry_of(str(flagged[1].get("content") or ""), present)
        if entry is None:
            stale.append(flagged)
        else:
            held.add(entry)
    return [e for e in entries if e not in held], stale


@contextmanager
def lock(state_dir: Path, name: str = LOCK_FILE, wait_s: Optional[float] = None) -> Iterator[None]:
    """One of the profile's locks beside its databases, across processes: by
    default the one for built-in memory rows (LOCK_FILE). With `wait_s`, it
    waits at most that long for another holder (TimeoutError past it)."""
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(state_dir) / name, "a") as f:
        if wait_s is None:
            fcntl.flock(f, fcntl.LOCK_EX)
        else:
            deadline = time.monotonic() + wait_s
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name} is held by another process") from None
                    time.sleep(LOCK_RETRY_S)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# -- who wrote an entry ---------------------------------------------------------------

def call_entries(args: Any) -> tuple[str, list[str]]:
    """The file a call of Hermes's memory tool writes to (its target) and the
    entries it writes whole, as the file will hold them (stripped): an add's
    content, or each such operation's in a batch. A patch's edited entry is
    known once the call ran (`patched_entries`)."""
    call = args if isinstance(args, dict) else {}
    target = str(call.get("target") or "")
    if target not in FILES:  # Hermes writes nothing without a target it knows
        return target, []
    operations = call.get("operations")
    found = []
    for op in operations if isinstance(operations, list) and operations else [call]:
        if not isinstance(op, dict) or str(op.get("action") or "") not in WHOLE_ENTRY_ACTIONS:
            continue
        text = op.get("content") or op.get("new_text")
        if isinstance(text, str) and text.strip():
            found.append(text.strip())
    return target, list(dict.fromkeys(found))


def patched_entries(answer: Any) -> list[str]:
    """The entries a call of Hermes's memory tool edited in place (a patch),
    as its answer reports them: a single patch's, or each of a batch's."""
    if not isinstance(answer, dict):
        return []
    many = answer.get("patched_entries_after")
    found = [answer.get("patched_entry_after"), *(many.values() if isinstance(many, dict) else ())]
    return list(dict.fromkeys(e.strip() for e in found if isinstance(e, str) and e.strip()))


def note_writers(state_dir: Path, home: Path, target: str, entries: list[str], chat: str) -> None:
    """Note that a turn which was not the owner's own wrote `entries` to a
    target's file, in `chat` (a group chat's key, UNATTRIBUTED when nothing
    says which turn it was, "" for no group chat). An entry several such turns
    wrote keeps each of their chats. Raises when the note cannot be written."""
    if not entries or target not in FILES:
        return
    config.make_private_dir(Path(state_dir))
    with lock(state_dir, WRITERS_LOCK, WRITERS_LOCK_WAIT_S):
        notes, _ = _read_writers(state_dir, home)
        now = _now_ms()
        for entry in entries:
            note = notes.setdefault(target, {}).setdefault(entry, {"chats": []})
            if chat and chat not in note["chats"]:
                note["chats"].append(chat)
            note["at"] = now
        _write_writers(state_dir, notes)


def writers(state_dir: Path, home: Path) -> dict[str, dict[str, list[str]]]:
    """The noted writers of the memory files' entries: target -> entry -> the
    group chats of the turns that wrote it ([] for none). Notes of entries no
    file holds any longer are dropped once WRITER_NOTE_KEEP_MS old. Raises
    when the note cannot be read or set right."""
    with lock(state_dir, WRITERS_LOCK, WRITERS_LOCK_WAIT_S):
        notes, changed = _read_writers(state_dir, home)
        held = {target: set(read_entries(home, target) or ()) for target in FILES}
        oldest = _now_ms() - WRITER_NOTE_KEEP_MS
        for target, noted in notes.items():
            for entry in [e for e, note in noted.items() if e not in held.get(target, ()) and note["at"] < oldest]:
                del noted[entry]
                changed = True
        if changed:
            _write_writers(state_dir, notes)
    return {target: {entry: list(note["chats"]) for entry, note in noted.items()} for target, noted in notes.items()}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_writers(state_dir: Path, home: Path) -> tuple[dict, bool]:
    """The note of writers (call with WRITERS_LOCK held), and whether it had to
    be set right. A note that is damaged is begun again, with every entry the
    files hold now noted UNATTRIBUTED: one of them may be a group's whose note
    is lost, and an entry already kept is not added again. So is one an
    earlier version left none of (EARLIER_WRITERS), whose own note it removes.
    Raises when the note cannot be read or written at all."""
    path = Path(state_dir) / WRITERS
    earlier = Path(state_dir) / EARLIER_WRITERS
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            return _parsed(text), False
        except (ValueError, AttributeError, TypeError, KeyError) as e:
            logger.warning("infino: could not read who wrote the memory files' entries; every entry they hold now "
                           "is taken as no one's: %s", e)
    elif not earlier.exists():
        return {}, False
    now = _now_ms()
    notes = {target: {entry: {"chats": [UNATTRIBUTED], "at": now} for entry in read_entries(home, target) or ()}
             for target in FILES}
    _write_writers(state_dir, notes)
    earlier.unlink(missing_ok=True)
    return notes, True


def _parsed(text: str) -> dict:
    """A note of writers, checked to be what `note_writers` writes; raises
    ValueError when it is not."""
    notes = json.loads(text)
    if not isinstance(notes, dict):
        raise ValueError("not a note of writers")
    for target, noted in notes.items():
        if target not in FILES or not isinstance(noted, dict):
            raise ValueError(f"not a memory file: {target!r}")
        for note in noted.values():
            if not isinstance(note.get("chats"), list) or not isinstance(note.get("at"), int):
                raise ValueError("not a writer's note")
    return notes


def _write_writers(state_dir: Path, notes: dict) -> None:
    fresh = Path(state_dir) / (WRITERS + ".tmp")
    fresh.write_text(json.dumps(notes))
    os.replace(fresh, Path(state_dir) / WRITERS)

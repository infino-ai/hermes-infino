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
owner - nothing says who wrote it, nor after which conversation, so the
provider keeps it where it can do no harm (a group chat's note for that chat
alone; USER.md as the owner's only where no one else's turn has run) - and a
row whose entry has left the file is deleted, here and in the cloud.

A file that is missing or cannot be read says nothing: no row is deleted for
it. Entries are compared scrubbed of secrets, as Hermes scrubs what it hands
a provider, and a row of a USER.md entry may carry the "About <name>: "
prefix the provider writes it under.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

try:  # Hermes's own separator of a memory file's entries.
    from tools.memory_tool_store import ENTRY_DELIMITER
except ImportError:  # pragma: no cover - only outside a Hermes install
    ENTRY_DELIMITER = "\n§\n"

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
def lock(state_dir: Path, name: str = LOCK_FILE) -> Iterator[None]:
    """One of the profile's locks beside its databases, across processes: by
    default the one for built-in memory rows (LOCK_FILE)."""
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(state_dir) / name, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

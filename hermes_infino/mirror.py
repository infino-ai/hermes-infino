"""Catching the cloud up with the local database, once signed in.

Every row is written to a local database first (`local.py`). A sync pushes
to the cloud the rows of each source past a mark - the (stored_ms, row_id) of
the last row the cloud accepted - in that order, and moves the mark only
after each batch is accepted. A source is one local table and the one cloud
database it goes to: each of the team's tables to the team's database, and
the person's own memory to their personal database - or nowhere, when they
have none: personal memory never goes to the team's database. The same sync is the push of everything kept
before sign-up, live mirroring after it, and catching up after an outage, and
it misses no row whichever process wrote it: a Hermes process that started
before the sign-up keeps writing the local database, and the next sync from
any process pushes its rows.

`stored_ms` is stamped just before a row commits locally, so rows commit in
nearly `stored_ms` order; a sync leaves the last `SETTLE_MS` alone, so a row
still committing in another process is not passed by the mark.

A memory deleted locally (`infino_forget`, or Hermes replacing a built-in
memory) is deleted in the cloud by its `row_id`s, from a small queue applied
after each sync's pushes: by id, so a memory saved again later is never
deleted with it, and after the pushes, so a row pushed while it was being
deleted is deleted too.

One sync runs at a time per profile (a file lock); the others skip.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from . import config, tables
from .client import InfinoClient
from .local import LocalStore

logger = logging.getLogger(__name__)

STATE_FILE = "mirror.json"
FORGET_FILE = "forget.jsonl"
LOCK_FILE = "mirror.lock"
# Rows stored this recently are left for the next sync (another process may
# still be committing a row stamped just before them).
SETTLE_MS = 2_000
# Rows per cloud append.
PUSH_BATCH = 200
# Row ids per cloud delete.
FORGET_BATCH = 100
# The mark of the personal memory source (the team's are their tables' names).
PERSONAL_MEMORY = "personal_memory"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class Source:
    mark: str            # the key its progress is kept under
    store: LocalStore
    table: str
    database: str        # the cloud database it goes to


class Mirror:
    """The cloud side of one signed-in profile."""

    def __init__(self, settings: config.Settings, local: LocalStore, personal: LocalStore,
                 client: Optional[InfinoClient] = None):
        self.settings = settings
        self.local = local
        self.personal = personal
        self.client = client or InfinoClient(settings.api_key, settings.url)
        self.dir = config.state_dir(settings.home)

    # -- where rows go -------------------------------------------------------

    def sources(self) -> list[Source]:
        out = [Source(t, self.local, t, self.settings.team_database) for t in tables.ALL]
        if self.settings.personal_database:
            out.append(Source(PERSONAL_MEMORY, self.personal, tables.MEMORY, self.settings.personal_database))
        return out

    def ensure_cloud_tables(self) -> None:
        tables.ensure_all(self.client, self.settings.team_database)
        if self.settings.personal_database:
            tables.ensure_all(self.client, self.settings.personal_database, personal=True)

    # -- state ---------------------------------------------------------------

    def _state(self) -> dict:
        try:
            state = json.loads((self.dir / STATE_FILE).read_text())
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def marks(self) -> dict[str, Optional[tuple[int, str]]]:
        raw = self._state().get("marks", {})
        return {k: (tuple(raw[k]) if raw.get(k) else None) for k in (*tables.ALL, PERSONAL_MEMORY)}

    def last_error(self) -> str:
        """Why the last sync stopped, or empty when it finished."""
        return str(self._state().get("last_error") or "")

    def _save_marks(self, marks: dict, last_error: str = "") -> None:
        path = self.dir / STATE_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"marks": {t: list(m) if m else None for t, m in marks.items()},
                                   "last_error": last_error, "at_ms": _now_ms()}))
        os.replace(tmp, path)

    @contextmanager
    def _lock(self, wait: bool) -> Iterator[bool]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / LOCK_FILE, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # -- sync ----------------------------------------------------------------

    def sync(self, wait: bool = False, progress=None) -> dict[str, int]:
        """Push every settled local row past the marks, then apply queued
        forgets. Rows pushed per source; {} when another sync holds the lock.
        Raises on a cloud failure, with the marks left at the last batch
        the cloud accepted."""
        with self._lock(wait) as held:
            if not held:
                return {}
            marks = self.marks()
            until = _now_ms() - SETTLE_MS
            pushed = {}
            try:
                for src in self.sources():
                    pushed[src.mark] = 0
                    while True:
                        rows = src.store.rows_after(src.table, marks[src.mark], until, PUSH_BATCH)
                        if not rows:
                            break
                        # The mark moves after each accepted batch, so a refused
                        # append never sends the rows before it again.
                        self.client.append(src.database, src.table, rows)
                        marks[src.mark] = (int(rows[-1][tables.STORED]), str(rows[-1][tables.ROW_ID]))
                        self._save_marks(marks)
                        pushed[src.mark] += len(rows)
                        if progress:
                            progress(src.mark, pushed[src.mark])
                self._apply_forgets()
            except Exception as e:
                # The rows stay local and the mark where the cloud stopped
                # accepting: nothing is skipped, and the next sync starts there.
                self._save_marks(marks, last_error=f"{type(e).__name__}: {e}")
                raise
            self._save_marks(marks)
            return pushed

    def unsynced_after(self, mark: str) -> Optional[tuple[int, str]]:
        """The mark a source's local rows must be past to be missing from the cloud."""
        return self.marks()[mark]

    # -- forgets -------------------------------------------------------------

    def forget(self, table: str, database: str, row_ids: list[str]) -> None:
        """Queue the cloud delete of rows deleted locally."""
        if not row_ids:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"table": table, "database": database, "row_ids": row_ids}) + "\n"
        with open(self.dir / FORGET_FILE, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)

    def _apply_forgets(self) -> None:
        path = self.dir / FORGET_FILE
        if not path.exists():
            return
        with open(path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                entries = [json.loads(line) for line in f if line.strip()]
                remaining = list(entries)
                for e in entries:
                    ids = list(e["row_ids"])
                    for start in range(0, len(ids), FORGET_BATCH):
                        chunk = ", ".join(tables.sql_string(i) for i in ids[start:start + FORGET_BATCH])
                        self.client.delete(e["database"], e["table"], f"{tables.ROW_ID} IN ({chunk})")
                    remaining.remove(e)
            finally:
                f.seek(0)
                f.truncate()
                f.writelines(json.dumps(e) + "\n" for e in remaining)
                fcntl.flock(f, fcntl.LOCK_UN)

"""Catching the cloud up with the local database, once signed in.

Every row is written to the local database first (`local.py`). A sync pushes
to the cloud the rows of each table past a mark - the (stored_ms, row_id) of
the last row the cloud accepted - in that order, and moves the mark only
after each batch is accepted. The same sync is the push of everything kept
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
from pathlib import Path
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


def _now_ms() -> int:
    return int(time.time() * 1000)


def _runs(rows: list[dict], key) -> list[tuple[str, list[dict]]]:
    """`rows` split into consecutive runs with the same key, in order."""
    out: list[tuple[str, list[dict]]] = []
    for r in rows:
        k = key(r)
        if out and out[-1][0] == k:
            out[-1][1].append(r)
        else:
            out.append((k, [r]))
    return out


class Mirror:
    """The cloud side of one signed-in profile."""

    def __init__(self, settings: config.Settings, local: LocalStore, client: Optional[InfinoClient] = None):
        self.settings = settings
        self.local = local
        self.client = client or InfinoClient(settings.api_key, settings.url)
        self.dir = config.state_dir(settings.home)

    # -- where rows go -------------------------------------------------------

    def database_for(self, table: str, scope: str = "") -> str:
        if table == tables.MEMORY and scope == "personal" and self.settings.personal_database:
            return self.settings.personal_database
        return self.settings.team_database

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
        return {t: (tuple(raw[t]) if raw.get(t) else None) for t in tables.ALL}

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
        forgets. Rows pushed per table; {} when another sync holds the lock.
        Raises on a cloud failure, with the marks left at the last batch
        the cloud accepted."""
        with self._lock(wait) as held:
            if not held:
                return {}
            marks = self.marks()
            until = _now_ms() - SETTLE_MS
            pushed = {}
            try:
                for table in tables.ALL:
                    pushed[table] = 0
                    while True:
                        rows = self.local.rows_after(table, marks[table], until, PUSH_BATCH)
                        if not rows:
                            break
                        # Consecutive rows bound for one database go in one append,
                        # and the mark moves after each, so a refused append never
                        # sends the rows before it again.
                        for database, run in _runs(rows, lambda r: self.database_for(
                                table, str(r.get("scope") or ""))):
                            self.client.append(database, table, run)
                            marks[table] = (int(run[-1][tables.STORED]), str(run[-1][tables.ROW_ID]))
                            self._save_marks(marks)
                            pushed[table] += len(run)
                        if progress:
                            progress(table, pushed[table])
                self._apply_forgets()
            except Exception as e:
                # The rows stay local and the mark where the cloud stopped
                # accepting: nothing is skipped, and the next sync starts there.
                self._save_marks(marks, last_error=f"{type(e).__name__}: {e}")
                raise
            self._save_marks(marks)
            return pushed

    def unsynced_after(self, table: str) -> Optional[tuple[int, str]]:
        """The mark local rows must be past to be missing from the cloud."""
        return self.marks()[table]

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

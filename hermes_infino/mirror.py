"""Catching the cloud up with the local database, once signed in.

Every row is written to a local database first (`local.py`). A sync pushes
to the cloud the rows of each source past a mark - the (stored_ms, row_id) of
the last row the cloud accepted - in that order, and moves the mark only
after each batch is accepted. A source is one local table and the one cloud
database it goes to: each of the team's tables to the team's database, and
each of the person's own tables (their memory, and the conversations and
traces kept personal) to their personal database - or nowhere, when they
have none: nothing personal ever goes to the team's database. The same sync is the push of everything kept
before sign-up, live mirroring after it, and catching up after an outage, and
it misses no row whichever process wrote it: a Hermes process that started
before the sign-up keeps writing the local database, and the next sync from
any process pushes its rows.

`stored_ms` is stamped just before a row commits locally, so rows commit in
nearly `stored_ms` order; a sync leaves the last `SETTLE_MS` alone, so a row
still committing in another process is not passed by the mark.

A mark belongs to one cloud database and table (`<database>/<table>`): a
profile moved to another team database pushes everything there. And a mark
is dropped when the cloud has lost the row it points at while this machine
still has it - the cloud table was dropped and made again - so the new table
gets this machine's rows. (A forgotten memory is deleted here too, so its
mark never looks lost.)

A batch whose append was never answered may still have reached the cloud (a
timeout after the platform committed it, or a process killed mid-append). A
note is written before each append and cleared after it, so the next sync of
that source first asks the cloud which of its first batch's rows it has, and
sends only the rest. (The platform's append is not idempotent by `row_id`;
this check is how the plugin keeps from sending a batch twice.)

A row the cloud refuses outright (too large, malformed) is set aside in
REFUSED_FILE and the source goes on; a source the cloud is failing on stops
without stopping the others, or the forgets.

A local commit slower than the settle window may have been passed by a mark
before it was visible (local.LATE_COMMITS_FILE notes it): the next sync
moves the mark back before it and re-sends what the cloud lacks.

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
from .client import InfinoClient, InfinoError
from .local import LATE_COMMITS_FILE, LocalStore

logger = logging.getLogger(__name__)

STATE_FILE = "mirror.json"
FORGET_FILE = "forget.jsonl"
LOCK_FILE = "mirror.lock"
# Rows the cloud refused outright: kept on this machine, listed here.
REFUSED_FILE = "refused.jsonl"
# Statuses that refuse the rows sent, not the request: bad request, too
# large, unprocessable. Anything else (auth, busy, down) is the sync's failure.
BAD_REQUEST = 400
REFUSED_STATUSES = (BAD_REQUEST, 413, 422)
# Rows stored this recently are left for the next sync (another process may
# still be committing a row stamped just before them).
SETTLE_MS = 2_000
# Rows per cloud append.
PUSH_BATCH = 200
# The most JSON one append carries: a conversation row can hold three 20,000
# character cells, and a request that large risks the client's timeout.
PUSH_MAX_BYTES = 2_000_000
# Row ids per cloud delete.
FORGET_BATCH = 100
# The person's own sources, one per personal table: the team's are named by
# their tables, these by the table after this prefix.
PERSONAL_SOURCE_PREFIX = "personal_"
PERSONAL_SOURCES = {PERSONAL_SOURCE_PREFIX + t: t for t in tables.PERSONAL}
PERSONAL_MEMORY = PERSONAL_SOURCE_PREFIX + tables.MEMORY
# What each team table's mark was called before marks named their database.
LEGACY_MARKS = (*tables.ALL, PERSONAL_MEMORY)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fit(rows: list[dict]) -> list[dict]:
    """The leading rows whose JSON fits one append (PUSH_MAX_BYTES), at least one."""
    size = 0
    for i, r in enumerate(rows):
        size += len(json.dumps(r, default=str))
        if size > PUSH_MAX_BYTES and i:
            return rows[:i]
    return rows


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

    def mark_of(self, source: str) -> str:
        """The mark of a source: a team table's name, or one of PERSONAL_SOURCES."""
        if source in PERSONAL_SOURCES:
            return f"{self.settings.personal_database}/{source}"
        return f"{self.settings.team_database}/{source}"

    def sources(self) -> list[Source]:
        out = [Source(self.mark_of(t), self.local, t, self.settings.team_database) for t in tables.ALL]
        if self.settings.personal_database:
            out += [Source(self.mark_of(name), self.personal, table, self.settings.personal_database)
                    for name, table in PERSONAL_SOURCES.items()]
        return out

    def ensure_cloud_tables(self) -> None:
        tables.ensure_all(self.client, self.settings.team_database)
        if self.settings.personal_database:
            tables.ensure_all(self.client, self.settings.personal_database, personal=True)
        self._drop_lost_marks()

    def _drop_lost_marks(self) -> None:
        """Forget each mark whose cloud table has lost this machine's rows:
        neither the row the mark points at nor the oldest row pushed is in
        the cloud, while this machine still has both. Asking for two rows
        pushed at different times keeps a cloud slow to show the newest one
        from looking like a dropped table."""
        with self._lock(wait=True):
            state = self._state()
            marks = self.marks()
            lost = []
            for src in self.sources():
                mark = marks.get(src.mark)
                if mark is None or not src.store.has_row(src.table, mark[1]):
                    continue
                oldest = src.store.rows_after(src.table, None, mark[0], 1)
                ids = ", ".join(tables.sql_string(i) for i in {mark[1], *(str(r[tables.ROW_ID]) for r in oldest)})
                found = self.client.query_sql(
                    src.database, f"SELECT count(*) AS n FROM {src.table} WHERE {tables.ROW_ID} IN ({ids})")
                if not (found and found[0].get("n")):
                    lost.append(src.mark)
            if lost:
                logger.info("infino: the cloud lost %s; pushing this machine's rows again", ", ".join(lost))
                self._save_state({k: v for k, v in marks.items() if k not in lost},
                                 str(state.get("last_error") or ""), state.get("in_flight") or [],
                                 {k: v for k, v in self._rechecks(state).items() if k not in lost})

    # -- state ---------------------------------------------------------------

    def _state(self) -> dict:
        try:
            state = json.loads((self.dir / STATE_FILE).read_text())
        except (OSError, ValueError):
            return {}
        return state if isinstance(state, dict) else {}

    def marks(self) -> dict[str, tuple[int, str]]:
        """Every mark kept, by `<database>/<source>`. Marks an older version kept
        by table name alone were for the team database it had then, which is
        taken to be today's."""
        raw = self._state().get("marks", {})
        raw = raw if isinstance(raw, dict) else {}
        out = {k: (int(v[0]), str(v[1])) for k, v in raw.items() if v and k not in LEGACY_MARKS}
        for legacy in LEGACY_MARKS:
            if raw.get(legacy) and (legacy != PERSONAL_MEMORY or self.settings.personal_database):
                out.setdefault(self.mark_of(legacy), (int(raw[legacy][0]), str(raw[legacy][1])))
        return out

    def last_error(self) -> str:
        """Why the last sync stopped, or empty when it finished."""
        return str(self._state().get("last_error") or "")

    def refused(self) -> int:
        """How many rows the cloud refused outright (set aside in REFUSED_FILE)."""
        try:
            return sum(1 for line in (self.dir / REFUSED_FILE).read_text().splitlines() if line.strip())
        except OSError:
            return 0

    def _save_state(self, marks: dict, last_error: str = "", in_flight=(), recheck: Optional[dict] = None) -> None:
        """The marks, why the last sync stopped, the sources whose append had
        been sent but not answered (`in_flight`: written before every append,
        so a process killed mid-append leaves the next sync a note to check
        what reached the cloud), and the re-checks still to finish (see
        `_take_late_commits`)."""
        path = self.dir / STATE_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"marks": {t: list(m) if m else None for t, m in marks.items()},
                                   "last_error": last_error, "in_flight": list(in_flight),
                                   "recheck": {t: list(m) for t, m in (recheck or {}).items()}, "at_ms": _now_ms()}))
        os.replace(tmp, path)

    def _rechecks(self, state: dict) -> dict[str, tuple[int, str]]:
        raw = state.get("recheck")
        return {k: (int(v[0]), str(v[1])) for k, v in raw.items() if v} if isinstance(raw, dict) else {}

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

        A source the cloud fails on stops at the last batch it accepted, and
        the others go on; so do the forgets. The first failure is raised
        after them, recorded as `last_error`."""
        with self._lock(wait) as held:
            if not held:
                return {}
            state = self._state()
            marks = self.marks()
            # Sources whose last append was never answered: that batch may have
            # reached the cloud. A state from before `in_flight` was kept only
            # says the last sync stopped, so every source is checked.
            unsure = set(state.get("in_flight") or ([s.mark for s in self.sources()] if state.get("last_error") else []))
            recheck = self._rechecks(state)
            for mark, upto in self._take_late_commits(marks).items():
                recheck[mark] = max(recheck.get(mark, upto), upto)
            until = _now_ms() - SETTLE_MS
            pushed: dict[str, int] = {}
            failure: Optional[Exception] = None
            for src in self.sources():
                pushed[src.mark] = 0
                check = src.mark in unsure
                try:
                    while True:
                        rows = _fit(src.store.rows_after(src.table, marks.get(src.mark), until, PUSH_BATCH))
                        if not rows:
                            break
                        first = (int(rows[0][tables.STORED]), str(rows[0][tables.ROW_ID]))
                        upto = recheck.get(src.mark)
                        send = self._not_in_cloud(src, rows) if check or (upto and first <= upto) else rows
                        check = False
                        self._save_state(marks, in_flight=[src.mark], recheck=recheck)
                        self._push(src, send)
                        # The mark moves after each accepted batch, so a failed
                        # append never sends the rows before it again.
                        marks[src.mark] = (int(rows[-1][tables.STORED]), str(rows[-1][tables.ROW_ID]))
                        if upto and marks[src.mark] >= upto:
                            recheck.pop(src.mark, None)  # past every row that might be in the cloud
                        self._save_state(marks, recheck=recheck)
                        pushed[src.mark] += len(rows)
                        if progress:
                            progress(src.mark, pushed[src.mark])
                except Exception as e:
                    # The rows stay local and the mark where the cloud stopped
                    # accepting: nothing is skipped, and the next sync starts there.
                    logger.info("infino: %s stopped at its last accepted batch: %s", src.mark, e)
                    unsure = unsure | {src.mark}
                    failure = failure or e
                    continue
                unsure.discard(src.mark)
            try:
                self._apply_forgets()
            except Exception as e:
                failure = failure or e
            if failure is not None:
                self._save_state(marks, f"{type(failure).__name__}: {failure}", sorted(unsure), recheck)
                raise failure
            self._save_state(marks, recheck=recheck)
            return pushed

    def _push(self, src: Source, rows: list[dict]) -> None:
        """Append rows. A row the cloud refuses outright is set aside
        (REFUSED_FILE) rather than refused again on every sync, stalling the
        source: always for "too large" or "unprocessable"; for a plain 400
        only when other rows of the batch were accepted, since a 400 on every
        row says the table, not a row, is the trouble."""
        if not rows:
            return
        refused: list[tuple[dict, InfinoError]] = []
        accepted = self._push_finding_refused(src, rows, refused)
        for row, error in refused:
            if error.status != BAD_REQUEST or accepted:
                self._set_aside(src, row, error)
            else:
                raise error

    def _push_finding_refused(self, src: Source, rows: list[dict], refused: list) -> int:
        """Append rows, halving a refused batch down to the rows refused; how
        many were accepted. Failures other than refusals are raised."""
        try:
            self.client.append(src.database, src.table, rows)
            return len(rows)
        except InfinoError as e:
            if e.status not in REFUSED_STATUSES:
                raise
            if len(rows) == 1:
                refused.append((rows[0], e))
                return 0
            mid = len(rows) // 2
            return (self._push_finding_refused(src, rows[:mid], refused)
                    + self._push_finding_refused(src, rows[mid:], refused))

    def _set_aside(self, src: Source, row: dict, error: InfinoError) -> None:
        logger.warning("infino: the cloud refused %s row %s; it stays on this machine: %s",
                       src.table, row.get(tables.ROW_ID), error)
        line = json.dumps({"table": src.table, "database": src.database, "row_id": row.get(tables.ROW_ID),
                           "status": error.status, "error": error.message[:500], "at_ms": _now_ms()}) + "\n"
        with open(self.dir / REFUSED_FILE, "a") as f:
            f.write(line)

    def _take_late_commits(self, marks: dict) -> dict[str, tuple[int, str]]:
        """Local commits slower than the settle window (local.LATE_COMMITS_FILE):
        each source's mark goes back before them, and the rows up to where it
        was are re-sent only if the cloud lacks them. Returns, per source, the
        mark the re-check runs up to."""
        path = self.dir / LATE_COMMITS_FILE
        if not path.exists():
            return {}
        with open(path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            entries = [json.loads(line) for line in f if line.strip()]
            f.seek(0)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
        recheck: dict[str, tuple[int, str]] = {}
        for e in entries:
            for src in self.sources():
                mark = marks.get(src.mark)
                if (src.store.path.name, src.table) != (e.get("store"), e.get("table")) or mark is None:
                    continue
                stored = int(e["stored_ms"])
                if mark[0] >= stored:
                    recheck[src.mark] = max(recheck.get(src.mark, mark), mark)
                    marks[src.mark] = (stored - 1, "")
        return recheck

    def _not_in_cloud(self, src: Source, rows: list[dict]) -> list[dict]:
        """The rows of a batch the cloud does not have."""
        ids = ", ".join(tables.sql_string(str(r[tables.ROW_ID])) for r in rows)
        have = {str(r.get(tables.ROW_ID)) for r in self.client.query_sql(
            src.database, f"SELECT {tables.ROW_ID} FROM {src.table} WHERE {tables.ROW_ID} IN ({ids})")}
        return [r for r in rows if str(r[tables.ROW_ID]) not in have]

    def unsynced_after(self, source: str) -> Optional[tuple[int, str]]:
        """The mark a source's local rows must be past to be missing from the
        cloud; `source` is a team table's name or one of PERSONAL_SOURCES."""
        return self.marks().get(self.mark_of(source))

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

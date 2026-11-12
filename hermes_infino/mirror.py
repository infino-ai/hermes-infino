"""Catching the cloud up with the local database, once signed in.

Every row is written to a local database first (`local.py`). A sync pushes
to the cloud the rows of each source past a mark - the (stored_ms, row_id) of
the last row the cloud accepted - in that order, and moves the mark only
after each batch is accepted. A source is one local table and the one cloud
database it goes to: each of the team's tables to the team's database, and
each of the person's own tables (their memory, and the conversations and
traces kept personal) to their personal database - or nowhere, when they
have none: nothing personal ever goes to the team's database, not even a
personal memory still in the team's own table (an older version's, before it
moves out: Backend.ensure_local). The same sync is the push of everything kept
before sign-up, live mirroring after it, and catching up after an outage, and
it misses no row whichever process wrote it: a Hermes process that started
before the sign-up keeps writing the local database, and the next sync from
any process pushes its rows.

`stored_ms` is stamped just before a row commits locally, so rows commit in
nearly `stored_ms` order; a sync leaves the last `SETTLE_MS` alone, so a row
still committing in another process is not passed by the mark.

A mark belongs to one cloud database and table (`<database>/<table>`): a
profile moved to another team database pushes everything there. And a mark
is dropped when the cloud has lost this machine's rows while this machine
still has them - the cloud table was dropped and made again - so the new
table gets this machine's rows. A teammate's `infino_forget` deletes rows of
this machine from the cloud too, so a table counts as lost only when none of
the last LOST_CHECK_ROWS rows pushed, nor the first, is in the cloud.

Nothing is pushed to a cloud database until its tables are right
(`tables.ensure_all`): a database whose tables lack a column this version
writes, and cannot be given it, or that cannot be made or reached at all (a
key limited to other databases, a personal database out of reach), gets no
rows, and the sync says why. Each database is
apart: one that is not right holds up only its own sources, never the other
database's, nor the forgets to it.

A batch whose append was never answered may still have reached the cloud (a
timeout after the platform committed it, or a process killed mid-append). A
note is written before each append, naming its source with every other one
still unsure, and that source is cleared once a batch of it is accepted, so
the next sync of each first asks the cloud which of its first batch's rows it
has, and sends only the rest. The note also re-checks the source up to the
batch's last row (see below), so every row of that batch is asked about
wherever the next sync's mark starts. (The platform's append is not
idempotent by `row_id`; this check is how the plugin keeps from sending a
batch twice.) The marks and notes are kept in STATE_FILE, written whole; one
that is there but cannot be read (a power loss) has lost where the marks
were, so every row stored by then is checked with the cloud the same way
before it is sent.

A row the cloud refuses outright (too large, malformed) is set aside in
REFUSED_FILE and the source goes on; a source the cloud is failing on stops
without stopping the others, or the forgets. A worker too busy to take a
batch (413, its memory over budget while other queries fill it) refuses no
row: that is the sync's failure, and the rows go with a later one.

A local commit slower than the settle window, or stamped behind its table's
latest row (a clock put back), may have been passed by a mark before it was
visible (local.LATE_COMMITS_FILE notes it): the next sync moves the mark back
before it and re-sends what the cloud lacks.

A memory deleted locally (`infino_forget`, or Hermes replacing a built-in
memory) is deleted in the cloud by its `row_id`s, from a small queue applied
after each sync's pushes: by id, so a memory saved again later is never
deleted with it, and after the pushes, so a row pushed while it was being
deleted is deleted too. A forget the cloud could not be asked about at the
time (no answer) is queued as its match instead, bounded by `stored_ms` so a
memory saved again later is still kept. Each queued forget is applied on its
own: one the cloud fails stays queued for the next sync without holding up
the ones after it, and one whose database or table no longer exists is
dropped, there being nothing left there to delete. One for a database these
settings no longer name stays queued whatever the cloud answers, for when
they name it again, and fails no sync: a database the key does not reach is
"not found" too (the platform answers so for another account's). The queue's
lock is held only to read and rewrite the file, never across a request to the
cloud; a line that cannot be read (cut short by a crash) is taken off, since
it can never be applied.

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
from pathlib import Path
from typing import Iterator, Optional

from . import config, recall, tables
from .client import InfinoClient, InfinoError
from .local import LATE_COMMITS_FILE, LocalStore, append_line

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
# The engine's words, in a worker's 413, for a write its memory budget could
# not take while other queries filled it: the worker was busy, the rows not
# too large (a plugin batch is far under the platform's body limit), so it is
# the sync's failure, and the rows go with a later one.
OVER_BUDGET = "over budget"
# What `_state` holds for a state file that is there but cannot be read: where
# the marks were is lost.
DAMAGED = "damaged"
# How much of a queue's unreadable line the log shows.
LOGGED_LINE_CHARS = 200
# A delete's answer when its database or table no longer exists (never made,
# dropped, or being purged): nothing is left there to delete.
GONE_STATUSES = (404, 410)
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
# The latest rows pushed that a table's loss is checked against (with the
# first): a table is lost only when the cloud holds none of them.
LOST_CHECK_ROWS = 32
# Seconds a forget waits for the cloud to say which rows match before it is
# queued as its match instead.
FORGET_LOOKUP_TIMEOUT_S = 5.0
# Seconds between attempts to make the cloud tables right after one failed.
ENSURE_RETRY_S = 60.0
# A refusal to make an optional table that asking again will not change: the
# key may not make tables there. Any other failure (a bad key being fixed, a
# request the platform could not take) is tried again in ENSURE_RETRY_S.
OPTIONAL_REFUSED_STATUSES = (403,)
# The person's own sources, one per personal table: the team's are named by
# their tables, these by the table after this prefix.
PERSONAL_SOURCE_PREFIX = "personal_"
PERSONAL_SOURCES = {PERSONAL_SOURCE_PREFIX + t: t for t in tables.PERSONAL}
PERSONAL_MEMORY = PERSONAL_SOURCE_PREFIX + tables.MEMORY
# What each team table's mark was called before marks named their database.
LEGACY_MARKS = (tables.MEMORY, tables.CONVERSATIONS, tables.TRACES, tables.SKILLS, PERSONAL_MEMORY)


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


def _raw_marks(state: dict) -> dict:
    """The marks a state holds, as written."""
    raw = state.get("marks", {})
    return raw if isinstance(raw, dict) else {}


def _forget_entry(line: str) -> Optional[dict]:
    """A queued forget, or None for a line that cannot be read (cut short by a crash)."""
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    if not (isinstance(entry, dict) and isinstance(entry.get("database"), str) and isinstance(entry.get("table"), str)
            and (entry.get("predicate") or entry.get("row_ids"))):
        return None
    return entry


@contextmanager
def sync_lock(state_dir: Path, wait: bool = True) -> Iterator[bool]:
    """The profile's sync lock (LOCK_FILE), held by a sync and by whatever must
    not run beside one; whether it was taken (without `wait`, False while
    another holds it)."""
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(state_dir) / LOCK_FILE, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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
        # The cloud databases whose tables are right (ensure_cloud_tables);
        # why each of the others failed its last attempt, and when the last
        # failed attempt was.
        self._tables_ready: set[str] = set()
        self._tables_errors: dict[str, Exception] = {}
        self._tables_failed_at = 0.0
        # The optional tables (tables.OPTIONAL) made in each cloud database,
        # and when each of the others last failed to be: a table not made
        # there keeps its own rows on this machine, holding up nothing else.
        self._optional_ready: set[tuple[str, str]] = set()
        self._optional_failed_at: dict[tuple[str, str], float] = {}
        # Optional tables a database refused to make (a key that may not make
        # tables there): not asked again by this process.
        self._optional_refused: set[tuple[str, str]] = set()
        # Optional tables that are there but lack a column this version writes,
        # which the key may not add: their rows stay here; the cloud's are read.
        self._optional_outdated: set[tuple[str, str]] = set()

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

    def databases(self) -> list[tuple[str, bool]]:
        """Each cloud database rows go to, with whether it is the person's own."""
        out = [(self.settings.team_database, False)]
        if self.settings.personal_database:
            out.append((self.settings.personal_database, True))
        return out

    def ensure_cloud_tables(self) -> None:
        """Make each cloud database's tables right, each on its own (see
        `_ensure`); the first failure is raised once every one was tried."""
        failures = self._ensure(self.databases())
        if failures:
            raise next(iter(failures.values()))

    def _ensure(self, databases: list[tuple[str, bool]]) -> dict[str, Exception]:
        """Make the tables of each of `databases` right (`tables.ensure_all`),
        and drop the marks of its tables that lost this machine's rows; the
        databases that could not be made right, with why. Each is apart: one
        that cannot be (a key limited to other databases, a personal
        database out of reach) holds up only its own sources."""
        failures: dict[str, Exception] = {}
        for database, personal in databases:
            try:
                tables.ensure_all(self.client, database, personal=personal)
                self._drop_lost_marks(database)
            except Exception as e:
                logger.warning("infino: nothing goes to %s until its tables are right (tried again in %d s): %s",
                               database, ENSURE_RETRY_S, e)
                failures[database] = e
                continue
            self._tables_ready.add(database)
            self._tables_errors.pop(database, None)
            # The optional tables too, as best it can: one not made now is tried
            # when its rows are next pushed, and holds up nothing meanwhile.
            for table in tables.OPTIONAL:
                if personal and table not in tables.PERSONAL:
                    continue
                self._optional_made(database, table)
        if failures:
            self._tables_ready -= set(failures)
            self._tables_errors.update(failures)
            self._tables_failed_at = time.monotonic()
        return failures

    def _ensure_before_push(self) -> dict[str, Exception]:
        """Make the cloud tables right before a sync pushes anything; the
        databases whose tables are not, each with why. Their sources wait,
        and the other database's go on; one that failed is tried again once
        ENSURE_RETRY_S has passed."""
        retry = time.monotonic() - self._tables_failed_at >= ENSURE_RETRY_S
        due = [(database, personal) for database, personal in self.databases()
               if database not in self._tables_ready and (retry or database not in self._tables_errors)]
        if due:
            self._ensure(due)
        return self.waiting()

    def _optional_made(self, database: str, table: str) -> bool:
        """Whether optional `table` is made in cloud `database`, making it when
        it is not (at most once each ENSURE_RETRY_S after a failure). A table
        that cannot be made holds up only its own rows, which stay here."""
        key = (database, table)
        if key in self._optional_ready:
            return True
        if key in self._optional_refused or key in self._optional_outdated:
            return False
        failed = self._optional_failed_at.get(key)
        if failed is not None and time.monotonic() - failed < ENSURE_RETRY_S:
            return False
        try:
            tables.ensure_optional(self.client, database, table)
        except Exception as e:
            if isinstance(e, tables.OutdatedTable):
                # There, but without a column this version writes, which this key
                # may not add: its rows stay here, and the cloud's are still
                # searched. Asked again by the next process, not every minute.
                logger.warning("infino: %s.%s lacks a column this version writes and it cannot be added, so its "
                               "rows stay on this machine (asked again when Hermes next starts); everything else "
                               "syncs: %s", database, table, e)
                self._optional_outdated.add(key)
                return False
            if isinstance(e, InfinoError) and e.status in OPTIONAL_REFUSED_STATUSES:
                # Refused, not failing: asked again by the next process, not every minute.
                logger.warning("infino: %s refuses to make %s, so its rows stay on this machine (asked again when "
                               "Hermes next starts); everything else syncs: %s", database, table, e)
                self._optional_refused.add(key)
                return False
            logger.warning("infino: %s.%s could not be made, so its rows stay on this machine (tried again in %d s); "
                           "everything else syncs: %s", database, table, ENSURE_RETRY_S, e)
            self._optional_failed_at[key] = time.monotonic()
            return False
        self._optional_ready.add(key)
        self._optional_failed_at.pop(key, None)
        return True

    def refused_optional(self, database: Optional[str], table: str) -> bool:
        """Whether cloud `database` refused to make optional `table` (asked
        once per process): it is not there, and its rows are on this machine
        alone. A table that is there without a column this version writes is
        not refused: the cloud's rows are still read."""
        return (str(database), table) in self._optional_refused

    def waiting(self) -> dict[str, Exception]:
        """The cloud databases whose tables are not right, each with why its
        last attempt failed: nothing goes to them until an attempt succeeds."""
        return {database: self._tables_errors[database] for database, _ in self.databases()
                if database not in self._tables_ready and database in self._tables_errors}

    def _drop_lost_marks(self, database: str) -> None:
        """Forget each mark of `database` whose cloud table has lost this
        machine's rows: none of the latest LOST_CHECK_ROWS rows pushed, nor
        the oldest, is in the cloud, while this machine still has the row the
        mark points at. Many rows, pushed at different times, keep a
        teammate's forget of a few of them, or a cloud slow to show the
        newest, from looking like a dropped table. The history of memories
        is lost when its memory table is, and never on its own."""
        with sync_lock(self.dir):
            state = self._state()
            marks = self.marks()
            lost = []
            for src in self.sources():
                if src.database != database:
                    continue
                if src.table == tables.HISTORY:
                    continue  # lost with the memory table, never on its own (below)
                if src.table in tables.OPTIONAL and (database, src.table) not in self._optional_ready:
                    continue  # not made there yet (or not known to be): nothing of it to have lost
                mark = marks.get(src.mark)
                if mark is None or not src.store.has_row(src.table, mark[1]):
                    continue
                oldest = src.store.rows_after(src.table, None, mark[0], 1)
                latest = src.store.pushed_ids(src.table, mark[0], LOST_CHECK_ROWS)
                ids = ", ".join(tables.sql_string(i) for i in
                                {mark[1], *latest, *(str(r[tables.ROW_ID]) for r in oldest)})
                found = self.client.query_sql(
                    src.database, f"SELECT count(*) AS n FROM {src.table} WHERE {tables.ROW_ID} IN ({ids})")
                if not (found and found[0].get("n")):
                    lost.append(src.mark)
            # The history is lost exactly when the memory table beside it is:
            # a database dropped and made again. Its rows missing on their own
            # are a teammate's forget of earlier versions, and pushing them
            # again would bring the forgotten text back.
            mark_of_table = {src.table: src.mark for src in self.sources() if src.database == database}
            history = mark_of_table.get(tables.HISTORY)
            if mark_of_table.get(tables.MEMORY) in lost and history in marks:
                lost.append(history)
                self._optional_ready.discard((database, tables.HISTORY))  # made again before its rows go
            if lost:
                logger.info("infino: the cloud lost %s; pushing this machine's rows again", ", ".join(lost))
                # Each row up to the dropped mark is asked about before it is
                # sent again (the re-check): the cloud may still have some of
                # them, and a row it has is not sent twice.
                rechecks = self._rechecks(state)
                for mark in lost:
                    rechecks[mark] = max(rechecks.get(mark, marks[mark]), marks[mark])
                self._save_state({k: v for k, v in marks.items() if k not in lost},
                                 str(state.get("last_error") or ""), self._in_flight(state), rechecks)

    # -- state ---------------------------------------------------------------

    def _state(self) -> dict:
        """What the last sync left: {} when nothing yet, and {DAMAGED: why}
        for a file that is there but cannot be read (a power loss, a disk
        fault), whose marks are lost."""
        try:
            state = json.loads((self.dir / STATE_FILE).read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            return {DAMAGED: f"{type(e).__name__}: {e}"}
        return state if isinstance(state, dict) else {DAMAGED: "not a JSON object"}

    def marks(self) -> dict[str, tuple[int, str]]:
        """Every mark kept, by `<database>/<source>`. Marks an older version kept
        by table name alone were for the team database it had then, which is
        taken to be today's. Its personal mark was for whichever personal
        database it pushed to, which may not be today's: with one named it is
        a re-check, never a mark (`_rechecks`); while there is none it is kept
        by its old name, for when there is."""
        raw = _raw_marks(self._state())
        out = {k: (int(v[0]), str(v[1])) for k, v in raw.items() if v and k not in LEGACY_MARKS}
        for legacy in LEGACY_MARKS:
            if not raw.get(legacy):
                continue
            if legacy != PERSONAL_MEMORY:
                out.setdefault(self.mark_of(legacy), (int(raw[legacy][0]), str(raw[legacy][1])))
            elif not self.settings.personal_database:
                out.setdefault(legacy, (int(raw[legacy][0]), str(raw[legacy][1])))
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
        `_take_late_commits`). On disk before it replaces the last state, so a
        power loss leaves one or the other whole."""
        path = self.dir / STATE_FILE
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"marks": {t: list(m) if m else None for t, m in marks.items()},
                                "last_error": last_error, "in_flight": list(in_flight),
                                "recheck": {t: list(m) for t, m in (recheck or {}).items()}, "at_ms": _now_ms()}))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _in_flight(self, state: dict) -> list[str]:
        """The sources whose last append was never answered: that batch may
        have reached the cloud. A state from before `in_flight` was kept only
        says whether the last sync stopped, and then every source is."""
        in_flight = state.get("in_flight")
        if in_flight is None:
            return [s.mark for s in self.sources()] if state.get("last_error") else []
        return list(in_flight)

    def _rechecks(self, state: dict) -> dict[str, tuple[int, str]]:
        """The re-checks still to finish, per source: up to which mark its rows
        are sent only if the cloud lacks them. For a state that could not be
        read, every source's, up to now: any row stored by then may be there.
        An older version's personal mark (`marks`) is the personal memory's,
        with a personal database named: the rows up to it went to whichever
        personal database that version named, maybe not this one."""
        if state.get(DAMAGED):
            return {src.mark: (_now_ms(), "") for src in self.sources()}
        raw = state.get("recheck")
        out = {k: (int(v[0]), str(v[1])) for k, v in raw.items() if v} if isinstance(raw, dict) else {}
        legacy = _raw_marks(state).get(PERSONAL_MEMORY)
        if legacy and self.settings.personal_database:
            mark, upto = self.mark_of(PERSONAL_MEMORY), (int(legacy[0]), str(legacy[1]))
            out[mark] = max(out.get(mark, upto), upto)
        return out

    def recheck(self, source: str, upto: tuple[int, str]) -> None:
        """Have the syncs to come ask the cloud which of a source's rows up to
        `upto` it has, and send only the rest: rows that may be there already
        (personal memory an older version pushed, moved since). The caller
        holds the sync lock (`sync_lock`)."""
        state = self._state()
        rechecks = self._rechecks(state)
        mark = self.mark_of(source)
        rechecks[mark] = max(rechecks.get(mark, upto), upto)
        self._save_state(self.marks(), str(state.get("last_error") or ""), self._in_flight(state), rechecks)

    # -- sync ----------------------------------------------------------------

    def sync(self, wait: bool = False, progress=None) -> dict[str, int]:
        """Push every settled local row past the marks, then apply queued
        forgets. Rows pushed per source; {} when another sync holds the lock.

        A source the cloud fails on stops at the last batch it accepted, and
        the others go on; so do the forgets. The first failure is raised
        after them, recorded as `last_error`. Nothing is pushed to a database
        whose tables are not right (`_ensure_before_push`): its sources, and
        the forgets queued for it, wait, and why is raised when nothing else
        failed. A failed push or forget comes first, so a database out of
        reach never hides that the other's rows did not go either."""
        waiting = self._ensure_before_push()
        with sync_lock(self.dir, wait) as held:
            if not held:
                return {}
            state = self._state()
            if state.get(DAMAGED):
                logger.warning("infino: %s could not be read (%s); every row is checked with the cloud before it "
                               "is sent again", STATE_FILE, state[DAMAGED])
            marks = self.marks()
            unsure = set(self._in_flight(state))
            recheck = self._rechecks(state)
            self._take_late_commits(state, marks, unsure, recheck)
            until = _now_ms() - SETTLE_MS
            pushed: dict[str, int] = {}
            failure: Optional[Exception] = None
            for src in self.sources():
                pushed[src.mark] = 0
                if src.database in waiting:
                    continue  # its rows stay here, and its mark where it is, until its tables are right
                if src.table in tables.OPTIONAL and not self._optional_made(src.database, src.table):
                    continue  # this table's rows stay here until it is made; the others go on
                check = src.mark in unsure
                try:
                    while True:
                        rows = _fit(src.store.rows_after(src.table, marks.get(src.mark), until, PUSH_BATCH))
                        if not rows:
                            break
                        first = (int(rows[0][tables.STORED]), str(rows[0][tables.ROW_ID]))
                        last = (int(rows[-1][tables.STORED]), str(rows[-1][tables.ROW_ID]))
                        upto = recheck.get(src.mark)
                        kept = self._sendable(src, rows)
                        send = self._not_in_cloud(src, kept) if kept and (check or (upto and first <= upto)) else kept
                        check = False
                        # The note names every source still unsure, not this one
                        # alone: a sync killed before the answer leaves them all.
                        # And it re-checks this one up to the batch's last row: a
                        # late commit's note can move the next sync's mark back
                        # before the batch, whose rows are then each asked about,
                        # not only those in that sync's first batch.
                        recheck[src.mark] = max(upto or last, last)
                        self._save_state(marks, in_flight=sorted(unsure | {src.mark}), recheck=recheck)
                        self._push(src, send)
                        # Its unanswered batch was checked with this one, and this one went.
                        unsure.discard(src.mark)
                        # The mark moves after each accepted batch, so a failed
                        # append never sends the rows before it again.
                        marks[src.mark] = last
                        if marks[src.mark] >= recheck[src.mark]:
                            recheck.pop(src.mark)  # past every row that might be in the cloud
                        self._save_state(marks, in_flight=sorted(unsure), recheck=recheck)
                        pushed[src.mark] += len(kept)
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
                self._apply_forgets(frozenset(waiting))
            except Exception as e:
                failure = failure or e
            failure = failure or next(iter(waiting.values()), None)
            if failure is not None:
                self._save_state(marks, f"{type(failure).__name__}: {failure}", sorted(unsure), recheck)
                raise failure
            self._save_state(marks, in_flight=sorted(unsure), recheck=recheck)
            return pushed

    def _sendable(self, src: Source, rows: list[dict]) -> list[dict]:
        """The rows of a batch its cloud database may have: never a personal
        memory from the team's own table (an older version's, not moved out
        yet: Backend.ensure_local, or one an older version still running
        writes there), which goes to the person's database or nowhere. The
        mark moves past it all the same."""
        if src.store is not self.local or src.table != tables.MEMORY:
            return rows
        return [r for r in rows if r.get("scope") != recall.PERSONAL]

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
        many were accepted. Failures other than refusals are raised, a busy
        worker's 413 (OVER_BUDGET) among them."""
        try:
            self.client.append(src.database, src.table, rows)
            return len(rows)
        except InfinoError as e:
            if e.status not in REFUSED_STATUSES or OVER_BUDGET in e.message:
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

    def _take_late_commits(self, state: dict, marks: dict, unsure: set, recheck: dict) -> None:
        """Rows a mark may have passed before they were visible
        (local.LATE_COMMITS_FILE): each source's mark (in `marks`) goes back
        before them, and the rows up to where it was are re-sent only if the
        cloud lacks them (`recheck` gets, per source, the mark the re-check
        runs up to). Saved before the notes are taken off the file, so a sync
        killed after it still goes back for them; a line that cannot be read
        is skipped."""
        path = self.dir / LATE_COMMITS_FILE
        if not path.exists():
            return
        with open(path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            lines = [line for line in f if line.strip()]
            if not lines:
                return
            for line in lines:
                try:
                    e = json.loads(line)
                    store, table, stored = e["store"], e["table"], int(e["stored_ms"])
                except (ValueError, KeyError, TypeError) as err:
                    logger.warning("infino: skipped a line of %s that cannot be read (%s): %r",
                                   LATE_COMMITS_FILE, err, line[:LOGGED_LINE_CHARS])
                    continue
                for src in self.sources():
                    mark = marks.get(src.mark)
                    if (src.store.path.name, src.table) != (store, table) or mark is None:
                        continue
                    if mark[0] >= stored:
                        recheck[src.mark] = max(recheck.get(src.mark, mark), mark)
                        marks[src.mark] = (stored - 1, "")
            self._save_state(marks, str(state.get("last_error") or ""), sorted(unsure), recheck)
            f.seek(0)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)

    def _not_in_cloud(self, src: Source, rows: list[dict]) -> list[dict]:
        """The rows of a batch the cloud does not have."""
        ids = ", ".join(tables.sql_string(str(r[tables.ROW_ID])) for r in rows)
        have = {str(r.get(tables.ROW_ID)) for r in self.client.query_sql(
            src.database, f"SELECT {tables.ROW_ID} FROM {src.table} WHERE {tables.ROW_ID} IN ({ids})")}
        return [r for r in rows if str(r[tables.ROW_ID]) not in have]

    def unpushed(self) -> bool:
        """Whether a sync has anything to send: a local row past its source's
        mark, or a queued forget."""
        marks, now = self.marks(), _now_ms()
        if any(src.store.count_after(src.table, marks.get(src.mark), now) for src in self.sources()):
            return True
        try:
            return bool((self.dir / FORGET_FILE).read_text().strip())
        except FileNotFoundError:
            return False

    def unsynced_after(self, source: str) -> Optional[tuple[int, str]]:
        """The mark a source's local rows must be past to be missing from the
        cloud; `source` is a team table's name or one of PERSONAL_SOURCES."""
        return self.marks().get(self.mark_of(source))

    # -- forgets -------------------------------------------------------------

    def matching_ids(self, table: str, database: str, predicate: str) -> set[str]:
        """The `row_id`s of the cloud rows matching `predicate`, whichever
        machine pushed them. Asked while a tool call waits: a short timeout,
        no retries."""
        s = self.settings
        client = InfinoClient(s.api_key, s.url, timeout=FORGET_LOOKUP_TIMEOUT_S, retries=0)
        rows = client.query_sql(database, f"SELECT {tables.ROW_ID} FROM {table} WHERE {predicate}")
        return {str(r.get(tables.ROW_ID)) for r in rows if r.get(tables.ROW_ID)}

    def matching_rows(self, table: str, database: str, predicate: str) -> list[dict]:
        """The cloud rows matching `predicate`, whole, whichever machine pushed
        them. Asked while a tool call waits, as `matching_ids` is."""
        s = self.settings
        client = InfinoClient(s.api_key, s.url, timeout=FORGET_LOOKUP_TIMEOUT_S, retries=0)
        return client.query_sql(database, f"SELECT * FROM {table} WHERE {predicate}")

    def forget(self, table: str, database: str, row_ids: list[str]) -> None:
        """Queue the cloud delete of rows by their ids."""
        if row_ids:
            self._queue({"table": table, "database": database, "row_ids": list(row_ids)})

    def forget_matching(self, table: str, database: str, predicate: str) -> None:
        """Queue the cloud delete of the rows matching `predicate` (which the
        caller bounds by `stored_ms`, so rows saved later are kept)."""
        self._queue({"table": table, "database": database, "predicate": predicate})

    def _queue(self, entry: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        append_line(self.dir / FORGET_FILE, json.dumps(entry) + "\n")

    def _apply_forgets(self, waiting: frozenset[str] = frozenset()) -> None:
        """Apply the queued deletes, each on its own, then take the applied
        ones off the queue. One the cloud fails stays queued for the next
        sync, and holds up none after it (the first failure is raised after
        them all); one whose database or table no longer exists is taken off
        too, there being nothing left there to delete. One for a database in
        `waiting` (its tables are not right) stays queued, unasked. One for a
        database these settings do not name (they did when it was queued)
        stays queued whatever the cloud answers - the key may not reach that
        database now, and the platform answers "not found" for a database of
        another account - for when they name it again, and fails no sync. A
        line that cannot be read is taken off: it can never be applied.

        The lock is held only to read and to rewrite the file: a forget made
        meanwhile queues without waiting on the cloud, and stays queued."""
        path = self.dir / FORGET_FILE
        if not path.exists():
            return
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            lines = [line for line in f if line.strip()]
            fcntl.flock(f, fcntl.LOCK_UN)
        named = {database for database, _ in self.databases()}
        applied: list[str] = []
        elsewhere: dict[str, Exception] = {}  # unnamed databases whose forgets failed, with why
        failure: Optional[Exception] = None
        try:
            for line in lines:
                entry = _forget_entry(line)
                if entry is None:
                    logger.warning("infino: a line of %s cannot be read, and is taken off: %r",
                                   FORGET_FILE, line[:LOGGED_LINE_CHARS])
                    applied.append(line)
                    continue
                if entry["database"] in waiting:
                    continue
                try:
                    self._delete(entry)
                except Exception as e:
                    if entry["database"] not in named:
                        elsewhere.setdefault(entry["database"], e)
                        continue
                    if not (isinstance(e, InfinoError) and e.status in GONE_STATUSES):
                        failure = failure or e  # stays queued for the next sync
                        continue
                    logger.info("infino: a queued forget's %s.%s no longer exists; nothing is left there to delete: "
                                "%s", entry["database"], entry["table"], e)
                applied.append(line)
            if elsewhere:
                logger.info("infino: forgets queued for %s, which these settings do not name, stay queued until they "
                            "do: %s", ", ".join(sorted(elsewhere)), next(iter(elsewhere.values())))
        finally:
            if applied:
                self._take_off_queue(path, applied)
        if failure is not None:
            raise failure

    @staticmethod
    def _take_off_queue(path: Path, applied: list[str]) -> None:
        """Rewrite the forget queue without `applied` (one occurrence each: the
        same forget queued again stays), under its lock. The rest is written
        over the start of the file before the file is cut to its length, so a
        sync killed between the two leaves the old tail after it (forgets
        applied again, to no effect, or queued twice), never an empty queue;
        and it is on disk before the lock is released, so a forget queued next
        lands after it, not under it."""
        with open(path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            current = [line for line in f if line.strip()]
            for line in applied:
                if line in current:
                    current.remove(line)
            f.seek(0)
            f.write("".join(current))
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def _delete(self, entry: dict) -> None:
        """Delete in the cloud what one queued forget names: the rows matching
        its predicate, or its rows by id, FORGET_BATCH ids per request."""
        if entry.get("predicate"):
            self.client.delete(entry["database"], entry["table"], entry["predicate"])
            return
        ids = list(entry["row_ids"])
        for start in range(0, len(ids), FORGET_BATCH):
            chunk = ", ".join(tables.sql_string(i) for i in ids[start:start + FORGET_BATCH])
            self.client.delete(entry["database"], entry["table"], f"{tables.ROW_ID} IN ({chunk})")

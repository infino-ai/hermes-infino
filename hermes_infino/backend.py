"""Where a profile's rows go and come from: the local databases always, and
the cloud databases once signed in, caught up by the mirror.

Two local databases: the team's tables (`infino/db`), which the Infino tools
read, and the person's own (`infino/personal`): their memory, and the
conversations and traces kept personal (privacy.py). Nothing personal is ever
in the team's database: locally it is in its own, and in the cloud in the
personal database, or nowhere when the person has none. The person's own
database is read by recall and by `infino_personal_search`, both only where
the turn is the owner's (provider.py).

One backend per profile home, shared by the memory provider, the trace hooks
and the skills loader; when the home's settings change (another team, a
sign-in), the backend made for the new ones takes over and the old one is
retired: its syncs stop, so no row goes where the settings no longer say.
Writes land locally and return; when signed in they also schedule a sync,
which runs on a background thread once the rows have settled
(`mirror.SETTLE_MS`), at most one at a time. A write made while a sync runs,
or a sync that found another process syncing, schedules one more; a sync
that failed is tried again by itself (SYNC_RETRY_FIRST_S, doubling up to
SYNC_RETRY_MAX_S), so an idle gateway's rows do not wait for its next write.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config, embed, privacy, recall, tables
from .client import InfinoClient
from .local import PERSONAL_DB_DIR, LocalStore
from .mirror import PERSONAL_MEMORY, PERSONAL_SOURCE_PREFIX, SETTLE_MS, Mirror

logger = logging.getLogger(__name__)

# How long after a write its sync runs: past the settle window, so the write
# is pushed by the sync it scheduled.
SYNC_DELAY_S = SETTLE_MS / 1000 + 0.5
# The most a session's end waits for its rows to reach the cloud.
FLUSH_TIMEOUT_S = 30.0
# Seconds after a failed background sync that the next one runs, with no
# write needed to start it; each failure in a row doubles the wait, up to
# SYNC_RETRY_MAX_S, and a sync that succeeds starts it over.
SYNC_RETRY_FIRST_S = 30.0
SYNC_RETRY_MAX_S = 15 * 60.0
# The tables whose rows follow their session: personal sessions' go to the
# person's own database.
TRANSCRIPT_TABLES = (tables.CONVERSATIONS, tables.TRACES)
# Seconds a search of the personal cloud database may take before this
# machine's copy answers: a turn is waiting on it.
PERSONAL_SEARCH_TIMEOUT_S = recall.RECALL_TIMEOUT_S


def _best(results: list[list[dict]], k: int) -> list[dict]:
    """The `k` best rows of several searches, each row once (at its best
    score), best first."""
    best: dict[str, dict] = {}
    for rows in results:
        for r in rows:
            key = str(r.get(tables.ROW_ID))
            if key not in best or float(r.get("score") or 0.0) > float(best[key].get("score") or 0.0):
                best[key] = r
    return sorted(best.values(), key=lambda r: -float(r.get("score") or 0.0))[:k]


class Backend:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        state = config.state_dir(settings.home)
        try:
            config.make_private_dir(state)
        except OSError as e:  # a home this process may not change keeps its mode
            logger.warning("infino: could not make %s owner-only: %s", state, e)
        self.local = LocalStore.under(state)
        self.personal = LocalStore.under(state, PERSONAL_DB_DIR)
        self.mirror = Mirror(settings, self.local, self.personal) if settings.signed_in else None
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._last_write = 0.0
        # A sync was asked for while one was pending or running.
        self._again = False
        # The sync that tries again after a failed one, and how long it waited
        # (0 after a sync that succeeded).
        self._retry_timer: Optional[threading.Timer] = None
        self._retry_s = 0.0
        # The backend made for this home's new settings, once this one is
        # retired (`for_settings`): the syncs asked of this one are its.
        self._successor: Optional[Backend] = None
        # Sessions whose look at the personal tables could not read them:
        # looked at again, since privacy.first_look counts a look only once.
        # Few: a session leaves once a read succeeds.
        self._look_again: set[str] = set()

    @property
    def signed_in(self) -> bool:
        return self.mirror is not None

    def ensure(self) -> None:
        self.ensure_local()
        self.catch_up()

    def ensure_local(self) -> None:
        """Create the local tables that are missing (with local vectors only in
        local mode): all a write needs, and quick."""
        self.local.ensure_tables(with_vectors=not self.signed_in)
        self.personal.ensure_tables(with_vectors=not self.signed_in, only=tables.PERSONAL)
        self._move_personal_memory()

    def catch_up(self) -> None:
        """What may take a while, after the local tables exist: signed in, the
        cloud tables and a sync; in local mode, the model's load (in the
        background) and then the memories saved without a vector. The sync
        starts even when a cloud database could not be made right: the other
        database's rows go, and a failed sync is tried again by itself."""
        if self.mirror is not None:
            self.local.drop_vectors()
            self.personal.drop_vectors()
            try:
                self.mirror.ensure_cloud_tables()
            finally:
                self.schedule_sync(delay_s=0)
            return
        embed.load_in_background(self._backfill_vectors)

    def _backfill_vectors(self) -> None:
        n = self.local.backfill_vectors() + self.personal.backfill_vectors()
        if n:
            logger.info("infino: embedded %d memories saved without vectors", n)

    def _move_personal_memory(self) -> None:
        """Personal memories an older version kept in the team's local
        database (where the tools read) move to the personal one."""
        found = self.local.query_sql(f"SELECT * FROM {tables.MEMORY} WHERE scope = 'personal'")
        if not found:
            return
        keep = {f["name"] for f in tables.MEMORY_SCHEMA if f["type"] != "embedding"}
        self.personal.append(tables.MEMORY, [{k: v for k, v in r.items() if k in keep} for r in found])
        self.local.delete(tables.MEMORY, "scope = 'personal'")
        logger.info("infino: moved %d personal memories out of the team's local database", len(found))

    # -- writes --------------------------------------------------------------

    def append(self, table: str, rows: list[dict[str, Any]], personal: bool = False) -> None:
        """Write rows locally. A memory goes where its scope says; `personal`
        sends a personal session's conversation or trace rows to the person's
        own database instead of the team's."""
        if table == tables.MEMORY:
            mine = [r for r in rows if r.get("scope") == recall.PERSONAL]
            rows = [r for r in rows if r.get("scope") != recall.PERSONAL]
            self.personal.append(tables.MEMORY, mine)
        if personal and table in TRANSCRIPT_TABLES:
            self.personal.append(table, rows)
        else:
            self.local.append(table, rows)
        self._last_write = time.monotonic()
        self.schedule_sync()

    def transcript_personal(self, session_id: str, messages: Optional[list[dict]] = None) -> bool:
        """Whether a session's conversation and traces are personal
        (privacy.transcript_personal), or this machine kept its earlier turns
        personal: a chat stays personal after compaction drops the delivery
        that made it so, and after a restart. This machine's tables are asked
        once per session, counted only when they could be read: after a read
        that failed, the session's next call asks again."""
        if privacy.transcript_personal(self.settings, session_id, messages):
            return True
        if not (privacy.first_look(session_id) or session_id in self._look_again):
            return False
        kept = self._kept_personal(session_id)
        if kept is None:
            self._look_again.add(session_id)
            return False
        self._look_again.discard(session_id)
        if kept:
            privacy.note_personal(session_id)
        return kept

    def _kept_personal(self, session_id: str) -> Optional[bool]:
        """Whether the person's own tables hold any of a session's turns;
        None when they could not be read."""
        session = tables.sql_string(session_id)
        try:
            made = set(self.personal.list_tables())
            for table in TRANSCRIPT_TABLES:
                if table not in made:  # a table not made yet holds nothing
                    continue
                found = self.personal.query_sql(f"SELECT count(*) AS n FROM {table} WHERE session_id = {session}")
                if found and found[0]["n"]:
                    return True
        except Exception as e:
            logger.info("infino: could not read whether session %s was kept personal (asked again on its next "
                        "turn): %s", session_id, e)
            return None
        return False

    def forget_memory(self, content: str, scope: Optional[str] = None, visible: tuple[str, ...] = ("",),
                      people: Optional[tuple[str, ...]] = None, about_prefix: str = "",
                      everywhere: bool = False) -> int:
        """Delete memories with exactly this text, of the chats in `visible`
        ("" is no chat), in `scope` when given: a chat forgets only what it
        can recall. `people` limits personal memory to the rows of those
        authors (none: no personal memory; None: all of it). `about_prefix`
        also matches the text after any "About <name>: " prefix of that shape
        (a group's entries about its members, whoever wrote them).

        This machine's rows are deleted here and, signed in, in the cloud by
        their ids. With `everywhere` (`infino_forget`) so is every matching
        row in the cloud, whichever machine saved it: the cloud is asked
        which rows match, and when it does not answer the forget is queued as
        its match, bounded to rows stored by now (one saved later is kept).
        Returns how many rows were found."""
        now = int(time.time() * 1000)
        text = tables.sql_string(content)
        match = f"content = {text}"
        if about_prefix:
            match = (f"({match} OR (content LIKE {tables.sql_string(about_prefix + '%')} "
                     f"AND ends_with(content, {tables.sql_string(': ' + content)})))")
        predicate = f"{match} AND {tables.chat_predicate(visible)}"
        if scope:
            predicate += f" AND scope = {tables.sql_string(scope)}"
        team = self.local.delete(tables.MEMORY, predicate) if scope != recall.PERSONAL else []
        # A group chat never reaches the person's own memory (`visible` holds a chat).
        reaches_personal = scope in (None, recall.PERSONAL) and visible == ("",) and people != ()
        personal_predicate = predicate
        if people:
            personal_predicate += f" AND author IN ({', '.join(tables.sql_string(p) for p in people)})"
        personal = self.personal.delete(tables.MEMORY, personal_predicate) if reaches_personal else []
        found = {str(r[tables.ROW_ID]) for r in team} | {str(r[tables.ROW_ID]) for r in personal}
        if self.mirror is None:
            return len(found)
        # Each row is deleted from the cloud database its source went to.
        s = self.settings
        targets = [(s.team_database, predicate, team)] if scope != recall.PERSONAL else []
        if reaches_personal and s.personal_database:
            targets.append((s.personal_database, personal_predicate, personal))
        for database, where, rows in targets:
            ids = {str(r[tables.ROW_ID]) for r in rows}
            if everywhere:
                try:
                    ids |= self.mirror.matching_ids(tables.MEMORY, database, where)
                except Exception as e:  # unanswered: the match goes to the queue instead
                    logger.info("infino: the cloud did not say which memories match; the forget is queued: %s", e)
                    self.mirror.forget_matching(tables.MEMORY, database, f"{where} AND {tables.STORED} <= {now}")
            self.mirror.forget(tables.MEMORY, database, sorted(ids))
            found |= ids
        self.schedule_sync()
        return len(found)

    # -- Hermes's built-in memory files (builtin.py) --------------------------

    def builtin_rows(self, target: str) -> list[tuple[bool, dict]]:
        """This machine's `memory` rows kept from one of Hermes's built-in
        memory files, each with whether it is in the person's own database."""
        sql = (f"SELECT {tables.ROW_ID}, content FROM {tables.MEMORY} "
               f"WHERE origin = {tables.sql_string('builtin:' + target)}")
        return ([(False, r) for r in self.local.query_sql(sql)]
                + [(True, r) for r in self.personal.query_sql(sql)])

    def has_memory(self, row: dict) -> bool:
        """Whether this machine already keeps a memory row like `row` (its text,
        origin, chat and, for personal memory, author)."""
        where = (f"content = {tables.sql_string(row['content'])} AND origin = {tables.sql_string(row['origin'])} "
                 f"AND {tables.chat_predicate((str(row.get(tables.CHAT) or ''),))}")
        store = self.local
        if row.get("scope") == recall.PERSONAL:
            store = self.personal
            where += f" AND author = {tables.sql_string(str(row.get('author') or ''))}"
        found = store.query_sql(f"SELECT count(*) AS n FROM {tables.MEMORY} WHERE {where}")
        return bool(found and found[0]["n"])

    def forget_rows(self, rows: list[tuple[bool, dict]]) -> None:
        """Delete this machine's memory rows by id (each flagged personal or
        not, as `builtin_rows` gives them), here and in the cloud."""
        for personal, store in ((False, self.local), (True, self.personal)):
            ids = [str(r[tables.ROW_ID]) for p, r in rows if p == personal]
            if not ids:
                continue
            store.delete(tables.MEMORY, f"{tables.ROW_ID} IN ({', '.join(tables.sql_string(i) for i in ids)})")
            if self.mirror is not None:
                database = self.settings.personal_database if personal else self.settings.team_database
                if database:
                    self.mirror.forget(tables.MEMORY, database, ids)
        if self.mirror is not None:
            self.schedule_sync()

    # -- sync ----------------------------------------------------------------

    def schedule_sync(self, delay_s: float = SYNC_DELAY_S) -> None:
        with self._timer_lock:
            successor = self._successor
            if successor is None:
                if self.mirror is None:
                    return
                if self._timer is not None and self._timer.is_alive():
                    # The pending or running sync may have read its rows already: one more after it.
                    self._again = True
                    return
                self._timer = threading.Timer(delay_s, self._sync_quietly)
                self._timer.daemon = True
                self._timer.start()
                return
        successor.schedule_sync(delay_s)

    def _sync_quietly(self) -> None:
        with self._timer_lock:
            if self._successor is not None:  # retired while this waited to run
                return
        skipped = failed = False
        try:
            pushed = self.mirror.sync()
            skipped = pushed == {}  # another process holds the sync: it may not see these rows
            if any(pushed.values()):
                logger.info("infino: pushed %s", {t: n for t, n in pushed.items() if n})
        except Exception as e:
            failed = True
            logger.warning("infino: sync to the cloud stopped (the rows stay local and go next time): %s", e)
        with self._timer_lock:
            again, self._again = self._again or skipped, False
            self._timer = None
            if failed:
                self._retry_later()
            elif not skipped:
                self._retry_s = 0.0
                if self._retry_timer is not None:
                    self._retry_timer.cancel()
                    self._retry_timer = None
        if again:
            self.schedule_sync()

    def _retry_later(self) -> None:
        """After a failed sync (the timer lock held): one more by itself, so
        the rows go without waiting for a write, after a wait that doubles
        with each failure in a row. One retry pending at a time."""
        if self._successor is not None or (self._retry_timer is not None and self._retry_timer.is_alive()):
            return
        self._retry_s = min(self._retry_s * 2, SYNC_RETRY_MAX_S) if self._retry_s else SYNC_RETRY_FIRST_S
        self._retry_timer = threading.Timer(self._retry_s, self._retry_now)
        self._retry_timer.daemon = True
        self._retry_timer.start()

    def _retry_now(self) -> None:
        with self._timer_lock:
            self._retry_timer = None
        self.schedule_sync(delay_s=0)

    def retire(self, successor: Backend) -> None:
        """Hand this home's syncs to the backend made for its new settings:
        this one's timers stop and it never syncs again (one already running
        finishes), and every sync asked of it from now on - a write made
        through it, a session's end - is the successor's, which pushes each
        row to where the settings now say."""
        with self._timer_lock:
            self._successor = successor
            pending = [t for t in (self._timer, self._retry_timer) if t is not None]
            for timer in pending:
                timer.cancel()
            self._timer = self._retry_timer = None
            waiting = bool(pending) or self._again
            self._again = False
        if waiting:  # rows written through this one are waiting for a sync
            successor.schedule_sync()

    def flush(self) -> None:
        """At a session's end: wait for this process's rows to settle, then
        push them, so a one-shot session's work reaches the cloud before it
        exits. Bounded; whatever is left goes with the next sync."""
        with self._timer_lock:
            successor = self._successor
        if successor is not None:
            successor.flush()
            return
        if self.mirror is None:
            return
        settle = SETTLE_MS / 1000 - (time.monotonic() - self._last_write)
        if settle > 0:
            time.sleep(min(settle + 0.1, FLUSH_TIMEOUT_S))
        done = threading.Event()

        def run():
            try:
                self.mirror.sync(wait=True)
            except Exception as e:
                logger.warning("infino: sync at session end stopped (the rows stay local): %s", e)
            finally:
                done.set()

        threading.Thread(target=run, name="infino-flush", daemon=True).start()
        done.wait(FLUSH_TIMEOUT_S)

    # -- reads ---------------------------------------------------------------

    def recaller(self) -> recall.Recaller:
        if self.mirror is None:
            return recall.Recaller([recall.Source(recall.TEAM, self.local), recall.Source(recall.PERSONAL, self.personal)])
        s = self.settings
        mirror = self.mirror
        sources = [recall.Source(recall.TEAM, self.local, s.team_database,
                                 lambda: mirror.unsynced_after(tables.MEMORY))]
        # Without a personal database, personal memory stays on this machine:
        # its local copy is all there is, so it is searched whole.
        sources.append(recall.Source(recall.PERSONAL, self.personal, s.personal_database or None,
                                     (lambda: mirror.unsynced_after(PERSONAL_MEMORY)) if s.personal_database else None))
        # Recall is waited on by the turn: a short timeout and no retries.
        cloud = InfinoClient(s.api_key, s.url, timeout=recall.RECALL_TIMEOUT_S, retries=0)
        return recall.Recaller(sources, cloud)

    def search_personal(self, table: str, query: str, k: int) -> list[dict]:
        """The `k` best rows of one of the person's own tables for `query`, by
        keyword over each of its full-text columns, best first: from the
        personal cloud database when there is one (every machine's rows), with
        this machine's rows the cloud does not have yet (written since the
        last sync), as recall reads memory; from this machine's copy alone
        when there is no personal database or it does not answer."""
        schema, indexes = tables.PERSONAL_TABLES[table]
        columns = indexes["fts"]
        projection = [f["name"] for f in schema if f["type"] != "embedding"]
        here = _best([self.personal.keyword_search(table, c, query, k, projection) for c in columns], k)
        s = self.settings
        if self.mirror is None or not s.personal_database:
            return here
        client = InfinoClient(s.api_key, s.url, timeout=PERSONAL_SEARCH_TIMEOUT_S, retries=0)
        try:
            cloud = _best([client.bm25_search(s.personal_database, table, query, field=c, k=k,
                                              projection=projection + ["score"]) for c in columns], k)
        except Exception as e:  # this machine's copy answers instead
            logger.info("infino: the personal database did not answer a search: %s", e)
            return here
        mark = self.mirror.unsynced_after(PERSONAL_SOURCE_PREFIX + table)
        return _best([cloud, [r for r in here if recall.past(r, mark)]], k)

    def query(self, sql: str, timeout_s: Optional[float] = None) -> list[dict]:
        """SQL over the team's tables: the cloud's when signed in, else local."""
        if self.mirror is None:
            return self.local.query_sql(sql)
        s = self.settings
        client = InfinoClient(s.api_key, s.url, timeout=timeout_s or recall.RECALL_TIMEOUT_S, retries=0)
        return client.query_sql(s.team_database, sql)

    def table_names(self) -> list[str]:
        if self.mirror is None:
            return self.local.list_tables()
        s = self.settings
        return InfinoClient(s.api_key, s.url, timeout=recall.RECALL_TIMEOUT_S, retries=0).list_tables(s.team_database)


_backends: dict[tuple, Backend] = {}
_backends_lock = threading.Lock()


def for_settings(settings: config.Settings) -> Backend:
    """The shared backend of a profile, rebuilt when its sign-in changes; the
    one it replaces for the same home is retired (`Backend.retire`), so it
    never pushes the home's rows to where the old settings said."""
    home = str(Path(settings.home).resolve())
    key = (home, settings.api_key, settings.team_database, settings.personal_database, settings.url,
           settings.personal_profile, settings.personal_jobs)
    with _backends_lock:
        backend = _backends.get(key)
        if backend is not None:
            return backend
        backend = _backends[key] = Backend(settings)
        replaced = [_backends.pop(k) for k in list(_backends) if k[0] == home and k != key]
    for old in replaced:
        old.retire(backend)
    return backend

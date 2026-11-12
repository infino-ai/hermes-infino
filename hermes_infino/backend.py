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

A session's end the process may not outlive waits for its rows to reach the
cloud, bounded (`flush`, provider.shutdown says where). A process that exits
before a background sync is due pushes what is left as it exits
(`flush_at_exit`), unless it ends through os._exit, as Hermes's gateway and
one-shot runs do: their teardowns wait instead.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config, embed, privacy, recall, tables
from .builtin import ENTRY_DELIMITER
from .chats import CHATS_JOIN
from .client import InfinoClient
from .local import PERSONAL_DB_DIR, LocalStore
from .mirror import PERSONAL_MEMORY, PERSONAL_SOURCE_PREFIX, SETTLE_MS, Mirror, sync_lock

logger = logging.getLogger(__name__)

# How long after a write its sync runs: past the settle window, so the write
# is pushed by the sync it scheduled.
SYNC_DELAY_S = SETTLE_MS / 1000 + 0.5
# The most a session's end waits for its rows to reach the cloud, once they
# have settled. Within Hermes's own bounds: a CLI's exit gets 30 s before its
# watchdog ends the process (HERMES_EXIT_WATCHDOG_S), and a gateway agent's
# teardown 30 s (its _CLEANUP_TIMEOUT_S); before this wait each drains the
# agent's queued writes for up to 15 s.
FLUSH_TIMEOUT_S = 10.0
# Past the settle window, so the last write is in the sync that follows the wait.
SETTLE_MARGIN_S = 0.1
# The most a process's exit waits for the exit flush (flush_at_exit): the
# settle window, then a sync, as a session's end waits.
EXIT_FLUSH_WAIT_S = SETTLE_MS / 1000 + SETTLE_MARGIN_S + FLUSH_TIMEOUT_S
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
        # The sync of the latest session end, which may still be waiting on the
        # cloud after that session stopped waiting for it (`flush`), and when
        # the session end that started it stops waiting.
        self._flushing: Optional[threading.Thread] = None
        self._flush_deadline = 0.0

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
        database (where the tools read) move to the personal one, keeping
        their ids. Signed in, the copies it pushed to the team's cloud
        database (when there was no personal database) are forgotten there,
        and the personal database's syncs ask the cloud which of them it has
        before sending them (it pushed them there when there was one).

        Under the sync lock, so no sync runs meanwhile and no other process
        of the profile moves them too; a move cut short is finished by the
        next start, which copies only the rows not moved yet."""
        select = f"SELECT * FROM {tables.MEMORY} WHERE scope = 'personal'"
        if not self.local.query_sql(select):  # nearly always: only an upgrade leaves any
            return
        with sync_lock(config.state_dir(self.settings.home)):
            found = self.local.query_sql(select)
            if not found:  # another process moved them first
                return
            ids = f"{tables.ROW_ID} IN ({', '.join(tables.sql_string(str(r[tables.ROW_ID])) for r in found)})"
            moved = {str(r[tables.ROW_ID]) for r in
                     self.personal.query_sql(f"SELECT {tables.ROW_ID} FROM {tables.MEMORY} WHERE {ids}")}
            keep = {f["name"] for f in tables.MEMORY_SCHEMA if f["type"] != "embedding"}
            self.personal.append(tables.MEMORY, [{k: v for k, v in r.items() if k in keep}
                                                 for r in found if str(r[tables.ROW_ID]) not in moved])
            if self.mirror is not None:
                self.mirror.forget(tables.MEMORY, self.settings.team_database,
                                   [str(r[tables.ROW_ID]) for r in found])
                if self.settings.personal_database:
                    [last] = self.personal.query_sql(f"SELECT {tables.STORED}, {tables.ROW_ID} FROM {tables.MEMORY} "
                                                     f"WHERE {ids} ORDER BY {tables.STORED} DESC, {tables.ROW_ID} DESC "
                                                     f"LIMIT 1")
                    self.mirror.recheck(PERSONAL_MEMORY, (int(last[tables.STORED]), str(last[tables.ROW_ID])))
            self.local.delete(tables.MEMORY, ids)
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
        once per session, counted only when they could be read. A read that
        failed answers personal for that call, since a turn written to the
        team's tables cannot be taken back, and the session's next call asks
        again (finding that turn, it stays personal: the safe side), through
        whichever backend serves it then (privacy.first_look)."""
        if privacy.transcript_personal(self.settings, session_id, messages):
            return True
        if not privacy.first_look(session_id):
            return False
        kept = self._kept_personal(session_id)
        if kept is None:
            privacy.look_again(session_id)
            return True
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
            logger.info("infino: could not read whether session %s was kept personal (kept personal this time, "
                        "asked again on its next turn): %s", session_id, e)
            return None
        return False

    def forget_memory(self, content: str, scope: Optional[str] = None, visible: tuple[str, ...] = ("",),
                      people: Optional[tuple[str, ...]] = None, about_prefix: str = "",
                      everywhere: bool = False) -> int:
        """Delete memories with exactly this text, of the chats in `visible`
        ("" is no chat), in `scope` when given: a chat forgets only what it
        can recall, and what is kept for several chats together that it is
        one of (chats.CHATS_JOIN), which no chat's recall shows and any of
        them may forget. `people` limits personal memory to the rows of those
        authors (none: no personal memory; None: all of it). `about_prefix`
        also matches the text after any "About <name>: " prefix of that shape
        (a group's entries about its members, whoever wrote them).

        Forgetting is total: every earlier version of the memory goes too,
        the history rows that hold its text or were superseded by it, and
        theirs in turn (`_forget_history`).

        This machine's rows are deleted here and, signed in, in the cloud by
        their ids. With `everywhere` (`infino_forget`) so is every matching
        row in the cloud, whichever machine saved it: the cloud is asked
        which rows match, and when it does not answer the forget is queued as
        its match, bounded to rows stored by now (one saved later is kept).
        Returns how many rows were found, earlier versions included."""
        now = int(time.time() * 1000)
        match = _text_match(content, about_prefix)
        team_vis, personal_vis, reaches_personal = _reach(scope, visible, people)
        predicate, personal_predicate = f"{match} AND {team_vis}", f"{match} AND {personal_vis}"
        team = self.local.delete(tables.MEMORY, predicate) if scope != recall.PERSONAL else []
        personal = self.personal.delete(tables.MEMORY, personal_predicate) if reaches_personal else []
        found = {str(r[tables.ROW_ID]) for r in team} | {str(r[tables.ROW_ID]) for r in personal}
        s = self.settings
        history = 0
        if scope != recall.PERSONAL:
            history += self._forget_history(self.local, content, about_prefix, team_vis,
                                            s.team_database, everywhere, now)
        if reaches_personal:
            history += self._forget_history(self.personal, content, about_prefix, personal_vis,
                                            s.personal_database, everywhere, now)
        if self.mirror is None:
            return len(found) + history
        # Each row is deleted from the cloud database its source went to.
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
        return len(found) + history

    def _forget_history(self, store: LocalStore, content: str, about_prefix: str, visible: str,
                        database: Optional[str], everywhere: bool, now: int) -> int:
        """Delete the history of a memory being forgotten, from `store` and
        its cloud `database`: the rows holding its text, the rows it
        superseded (its earlier versions), and theirs in turn, followed back
        through `superseded_by` until no new text turns up. `visible` is the
        chat and scope filter the forget runs under. With `everywhere`, the
        cloud's rows are followed too, whichever machine pushed them; when the
        cloud does not answer, the first step's match is queued there as a
        delete, bounded to rows stored by now. Returns how many rows went."""
        rows: dict[str, dict] = {}
        texts: set[str] = set()
        predicate = (f"({_text_match(content, about_prefix)} OR "
                     f"{_text_match(content, about_prefix, tables.SUPERSEDED_BY)} OR "
                     f"{_named_among([content])}) AND {visible}")
        cloud = self.mirror is not None and bool(database) and everywhere
        for _ in range(MAX_HISTORY_CHAIN):
            found = store.query_sql(f"SELECT {tables.ROW_ID}, content FROM {tables.HISTORY} WHERE {predicate}")
            if cloud:
                try:
                    found += self.mirror.matching_rows(tables.HISTORY, database, predicate)
                except Exception as e:  # unanswered: what can be named now is queued there
                    logger.info("infino: the cloud did not say which earlier versions to forget; queued: %s", e)
                    self.mirror.forget_matching(tables.HISTORY, database, f"({predicate}) AND {tables.STORED} <= {now}")
                    cloud = False
            fresh = [r for r in found if str(r.get(tables.ROW_ID)) not in rows]
            for r in fresh:
                rows[str(r[tables.ROW_ID])] = r
            more = {str(r.get("content") or "") for r in fresh} - texts
            if not more:
                break
            texts |= more
            listed = ", ".join(tables.sql_string(t) for t in sorted(more))
            predicate = (f"(content IN ({listed}) OR {tables.SUPERSEDED_BY} IN ({listed}) OR "
                         f"{_named_among(sorted(more))}) AND {visible}")
        if not rows:
            return 0
        ids = sorted(rows)
        store.delete(tables.HISTORY, f"{tables.ROW_ID} IN ({', '.join(tables.sql_string(i) for i in ids)})")
        if self.mirror is not None and database:
            self.mirror.forget(tables.HISTORY, database, ids)
        return len(ids)

    def supersede_memory(self, content: str, superseded_by: str, scope: Optional[str] = None,
                         visible: tuple[str, ...] = ("",), people: Optional[tuple[str, ...]] = None,
                         about_prefix: str = "", replacement_chat: Optional[str] = None) -> int:
        """What a replacement of a built-in memory does to its row: the
        memories with exactly this text, matched as `forget_memory` matches
        them on this machine, go to the history as earlier versions of
        `superseded_by` (`supersede_rows`). Returns how many were found."""
        match = _text_match(content, about_prefix)
        team_vis, personal_vis, reaches_personal = _reach(scope, visible, people)
        rows: list[tuple[bool, dict]] = []
        if scope != recall.PERSONAL:
            rows += [(False, r) for r in self.local.query_sql(
                f"SELECT * FROM {tables.MEMORY} WHERE {match} AND {team_vis}")]
        if reaches_personal:
            rows += [(True, r) for r in self.personal.query_sql(
                f"SELECT * FROM {tables.MEMORY} WHERE {match} AND {personal_vis}")]
        self.supersede_rows(rows, superseded_by, replacement_chat)
        return len(rows)

    def supersede_rows(self, rows: list[tuple[bool, dict]], superseded_by: str = "",
                       replacement_chat: Optional[str] = None) -> None:
        """Keep memory rows that were replaced: each is copied into the
        history table of the database it was in (`tables.HISTORY`), with a row
        id of its own, when it was superseded, what superseded it and the id
        of the row it was; then the row itself is deleted by id, here and in
        the cloud, as `forget_rows` deletes it.

        `superseded_by` is written only on a copy of a row in the same chat
        as the replacement (`replacement_chat`; None: every row): a group's
        note never reaches the team's or another chat's history by naming.
        No copy is made of a row this machine pushed that the cloud no longer
        has, nor of one it cannot ask about: someone forgot it, and a copy
        would bring its text back (`_gone_from_cloud`). A text already kept
        (same origin, chat, scope and author) is kept once. The copy is
        written before the delete, so a crash between them leaves both and
        loses neither; the row is superseded again next time, and its copy
        is not made twice. `rows` are (personal, row) pairs, as
        `builtin_rows` gives them; a row without its other columns is read
        back first."""
        if not rows:
            return
        now = int(time.time() * 1000)
        for personal, store in ((False, self.local), (True, self.personal)):
            mine = [r for p, r in rows if p == personal]
            if not mine:
                continue
            ids = [str(r[tables.ROW_ID]) for r in mine]
            listed = ", ".join(tables.sql_string(i) for i in ids)
            whole = store.query_sql(f"SELECT * FROM {tables.MEMORY} WHERE {tables.ROW_ID} IN ({listed})")
            kept = {str(r[tables.PRIOR_ROW_ID]) for r in store.query_sql(
                f"SELECT {tables.PRIOR_ROW_ID} FROM {tables.HISTORY} WHERE {tables.PRIOR_ROW_ID} IN ({listed})")}
            gone = self._gone_from_cloud(personal, whole)

            def named(row: dict) -> str:
                same_chat = replacement_chat is None or str(row.get(tables.CHAT) or "") == replacement_chat
                return superseded_by if same_chat else ""

            copies = [_superseded_copy(r, now, named(r)) for r in whole
                      if str(r[tables.ROW_ID]) not in kept | gone and not self._has_history(store, r, named(r))]
            if copies:
                store.append(tables.HISTORY, copies)
                self._last_write = time.monotonic()
        self.forget_rows(rows)

    def _gone_from_cloud(self, personal: bool, rows: list[dict]) -> set[str]:
        """The ids among `rows` this machine pushed to the cloud that the cloud
        no longer has (a teammate forgot them), or every pushed one when the
        cloud cannot be asked: no history is made of those. Rows the cloud
        never had, or with no cloud, are none of them."""
        if self.mirror is None:
            return set()
        database = self.settings.personal_database if personal else self.settings.team_database
        if not database:
            return set()
        mark = self.mirror.unsynced_after(PERSONAL_MEMORY if personal else tables.MEMORY)
        pushed = [str(r[tables.ROW_ID]) for r in rows if not recall.past(r, mark)]
        if not pushed:
            return set()
        try:
            have = self.mirror.matching_ids(
                tables.MEMORY, database, f"{tables.ROW_ID} IN ({', '.join(tables.sql_string(i) for i in pushed)})")
        except Exception as e:
            logger.info("infino: could not ask the cloud whether replaced memories were forgotten there; "
                        "no history is kept of them: %s", e)
            return set(pushed)
        return set(pushed) - have

    @staticmethod
    def _has_history(store: LocalStore, row: dict, superseded_by: str) -> bool:
        """Whether `store`'s history already holds this text (same origin,
        chat, scope and author) as an earlier version of `superseded_by`: a
        text replaced again by the same text is kept once. One replaced by
        something else is kept again, so a forget of either replacement finds
        it (`_forget_history` follows `superseded_by`)."""
        def same(column: str, value) -> str:
            return f"{column} IS NULL" if value is None else f"{column} = {tables.sql_string(str(value))}"
        where = " AND ".join([f"content = {tables.sql_string(str(row.get('content') or ''))}",
                              *(same(c, row.get(c)) for c in ("origin", tables.CHAT, "scope", "author")),
                              same(tables.SUPERSEDED_BY, superseded_by)])
        found = store.query_sql(f"SELECT count(*) AS n FROM {tables.HISTORY} WHERE {where}")
        return bool(found and found[0]["n"])

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

    def schedule_sync(self, delay_s: Optional[float] = None) -> None:
        """A sync in the background, after `delay_s` (SYNC_DELAY_S when not
        given), or one more after the one pending or running."""
        with self._timer_lock:
            successor = self._successor
            if successor is None:
                if self.mirror is None:
                    return
                if self._timer is not None and self._timer.is_alive():
                    # The pending or running sync may have read its rows already: one more after it.
                    self._again = True
                    return
                self._timer = threading.Timer(SYNC_DELAY_S if delay_s is None else delay_s, self._sync_quietly)
                self._timer.daemon = True
                self._timer.start()
                return
        # Its syncs push this one's writes: a session's end or the process's
        # exit through it waits for them to settle too.
        successor._last_write = max(successor._last_write, self._last_write)
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
            successor._last_write = max(successor._last_write, self._last_write)
            pending = [t for t in (self._timer, self._retry_timer) if t is not None]
            for timer in pending:
                timer.cancel()
            self._timer = self._retry_timer = None
            waiting = bool(pending) or self._again
            self._again = False
        if waiting:  # rows written through this one are waiting for a sync
            successor.schedule_sync()

    def flush(self, wrote_at: float = 0.0) -> None:
        """At a session's end: wait for this process's rows to settle, then
        push them, so a one-shot session's work reaches the cloud before it
        exits. Bounded (FLUSH_TIMEOUT_S); whatever is left goes with the next
        sync. A retired backend's flush is its successor's, which waits for
        the rows written through either (`wrote_at`: the last write made
        through one it replaced).

        One session end's sync at a time: while an earlier session end's sync
        still waits (another process's sync holding the profile's lock, a slow
        cloud), a session end waits for it, within the same bound, rather than
        queueing another behind it. Once that sync has outlived its own bound,
        a session end does not wait on it again: a gateway's stop ends its
        cached agents one after another, within its own watchdog, and a cloud
        that did not answer the first would hold up each."""
        with self._timer_lock:
            successor = self._successor
        wrote_at = max(wrote_at, self._last_write)
        if successor is not None:
            successor.flush(wrote_at)
            return
        if self.mirror is None:
            return
        with self._timer_lock:
            stuck = self._flushing is not None and self._flushing.is_alive() and time.monotonic() > self._flush_deadline
        if stuck:
            self._leave_to_next_sync()
            return
        self._wait_to_settle(wrote_at)
        deadline = time.monotonic() + FLUSH_TIMEOUT_S
        while True:
            with self._timer_lock:
                earlier = self._flushing
                if earlier is None or not earlier.is_alive():
                    mine = self._flushing = threading.Thread(target=self._sync_at_end, name="infino-flush",
                                                             daemon=True)
                    self._flush_deadline = deadline
                    mine.start()
                    break
            earlier.join(max(0.0, deadline - time.monotonic()))
            if earlier.is_alive():
                self._leave_to_next_sync()
                return
        mine.join(max(0.0, deadline - time.monotonic()))

    def _leave_to_next_sync(self) -> None:
        """A session's end that does not wait on an earlier one's sync, still
        waiting on the cloud: its rows go with that sync, or the one after."""
        logger.info("infino: an earlier session end's sync is still waiting on the cloud; this one's rows go with it "
                    "or the next sync")
        self.schedule_sync()

    @staticmethod
    def _wait_to_settle(wrote_at: float) -> None:
        """Wait until the last write (`wrote_at`, on the monotonic clock) is
        past the settle window, so the sync that follows takes it."""
        settle = SETTLE_MS / 1000 - (time.monotonic() - wrote_at)
        if settle > 0:
            time.sleep(min(settle + SETTLE_MARGIN_S, FLUSH_TIMEOUT_S))

    def _push_before_exit(self) -> None:
        """This backend's part of the process's exit flush (flush_at_exit), on
        that flush's own thread: when a sync has anything to send, wait for
        this process's rows to settle, then sync."""
        if self.mirror is None or not self.mirror.unpushed():
            return
        self._wait_to_settle(self._last_write)
        self._sync_at_end()

    def _sync_at_end(self) -> None:
        try:
            self.mirror.sync(wait=True)
        except Exception as e:
            logger.warning("infino: sync at session end stopped (the rows stay local): %s", e)

    # -- reads ---------------------------------------------------------------

    def recaller(self) -> recall.Recaller:
        if self.mirror is None:
            return recall.Recaller([recall.Source(recall.TEAM, self.local), recall.Source(recall.PERSONAL, self.personal)])
        s = self.settings
        mirror = self.mirror
        sources = [recall.Source(recall.TEAM, self.local, s.team_database,
                                 lambda: mirror.unsynced_after(tables.MEMORY),
                                 lambda: mirror.unsynced_after(tables.HISTORY),
                                 lambda: mirror.refused_optional(s.team_database, tables.HISTORY))]
        # Without a personal database, personal memory stays on this machine:
        # its local copy is all there is, so it is searched whole.
        personal_history = PERSONAL_SOURCE_PREFIX + tables.HISTORY
        sources.append(recall.Source(recall.PERSONAL, self.personal, s.personal_database or None,
                                     (lambda: mirror.unsynced_after(PERSONAL_MEMORY)) if s.personal_database else None,
                                     (lambda: mirror.unsynced_after(personal_history)) if s.personal_database else None,
                                     lambda: mirror.refused_optional(s.personal_database, tables.HISTORY)))
        # Recall is waited on by the turn: a short timeout and no retries.
        cloud = InfinoClient(s.api_key, s.url, timeout=recall.RECALL_TIMEOUT_S, retries=0)
        return recall.Recaller(sources, cloud, rank=s.rank_recall)

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
# Set as the process exits (flush_at_exit), for the exit flush's thread.
_exiting = threading.Event()
# The exit flush's thread, started with the first signed-in backend and
# waiting: a Python that is exiting starts no new thread (3.12 and later
# refuse one from an atexit handler).
_exit_flusher: Optional[threading.Thread] = None


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
        if backend.signed_in:
            _watch_for_exit()
        replaced = [_backends.pop(k) for k in list(_backends) if k[0] == home and k != key]
    for old in replaced:
        old.retire(backend)
    return backend


def _watch_for_exit() -> None:
    """Start the exit flush's thread and register flush_at_exit, once per
    process (the caller holds _backends_lock)."""
    global _exit_flusher
    if _exit_flusher is not None:
        return
    _exit_flusher = threading.Thread(target=_flush_when_exiting, name="infino-exit-flush", daemon=True)
    _exit_flusher.start()
    atexit.register(flush_at_exit)


def _flush_when_exiting() -> None:
    _exiting.wait()
    with _backends_lock:
        live = list(_backends.values())
    for b in live:
        try:
            b._push_before_exit()
        except Exception as e:  # never raise into the exit: the rows stay local, for the next start's sync
            logger.warning("infino: could not push at exit (the rows stay local and go next time): %s", e)


def flush_at_exit() -> None:
    """As the process exits (atexit): push what the live backends hold that
    the cloud does not, within EXIT_FLUSH_WAIT_S. These are rows a teardown
    left to the background sync (one Hermes does not wait for: an evicted
    agent, a scheduled job's cleanup) in a process that exits before that sync
    is due. A process that ends through os._exit runs no atexit (Hermes's
    gateway, a one-shot run): there the teardown itself waits
    (provider.shutdown)."""
    _exiting.set()
    if _exit_flusher is not None:
        _exit_flusher.join(EXIT_FLUSH_WAIT_S)


def _text_match(content: str, about_prefix: str = "", column: str = tables.MEMORY_TEXT_COLUMN) -> str:
    """Rows whose `column` holds exactly this text, or, with `about_prefix`,
    holds it after any "About <name>: " prefix of that shape."""
    match = f"{column} = {tables.sql_string(content)}"
    if about_prefix:
        match = (f"({match} OR ({column} LIKE {tables.sql_string(about_prefix + '%')} "
                 f"AND ends_with({column}, {tables.sql_string(': ' + content)})))")
    return match


def _named_among(texts: list[str]) -> str:
    """History rows whose `superseded_by` names one of `texts` among several
    (a consolidation names every entry that carries the removed one, joined
    by REPLACED_JOIN): fenced by the joiner at both ends, it holds the text
    fenced the same way. No entry holds the joiner, so a text is matched only
    as a whole name."""
    joiner = tables.sql_string(REPLACED_JOIN)
    parts = [f"strpos(concat({joiner}, {tables.SUPERSEDED_BY}, {joiner}), "
             f"{tables.sql_string(REPLACED_JOIN + t + REPLACED_JOIN)}) > 0" for t in texts]
    return f"({' OR '.join(parts)})" if parts else "FALSE"


def _reach(scope: Optional[str], visible: tuple[str, ...],
           people: Optional[tuple[str, ...]]) -> tuple[str, str, bool]:
    """The rows a chat in `visible` may reach (see `Backend.forget_memory`):
    the filter on the team's tables, the filter on the person's own, and
    whether the person's own are reached at all."""
    # A chat's key is among a joined value's when the value, fenced by the
    # joiner at both ends, holds the key fenced the same way.
    joiner = tables.sql_string(CHATS_JOIN)
    among = [f"strpos(concat({joiner}, {tables.CHAT}, {joiner}), "
             f"{tables.sql_string(CHATS_JOIN + key + CHATS_JOIN)}) > 0" for key in visible if key]
    team = f"({' OR '.join([tables.chat_predicate(visible), *among])})"
    if scope:
        team += f" AND scope = {tables.sql_string(scope)}"
    # A group chat never reaches the person's own memory (`visible` holds a chat).
    reaches_personal = scope in (None, recall.PERSONAL) and visible == ("",) and people != ()
    personal = team
    if people:
        personal += f" AND author IN ({', '.join(tables.sql_string(p) for p in people)})"
    return team, personal, reaches_personal


# Steps a forget follows a memory's history back through `superseded_by`:
# each step is one query, and Hermes's files hold a few dozen entries.
MAX_HISTORY_CHAIN = 100
# What joins the entries a consolidated entry was folded into, in its
# history row's `superseded_by` (provider.on_memory_write): Hermes's own
# separator of a memory file's entries, which no entry can hold, so a name is
# never split inside an entry's text. Recall shows it as " § ".
REPLACED_JOIN = ENTRY_DELIMITER

# Columns a superseded copy does not carry over from the row it copies: its
# own id and commit stamp are its own, and a search's score is not a column.
_NOT_COPIED = (tables.ROW_ID, tables.STORED, tables.EMBEDDING_COLUMN, "score")


def _superseded_copy(row: dict, superseded_ms: int, superseded_by: str) -> dict:
    """A kept copy of a memory row Hermes no longer holds."""
    copy = {k: v for k, v in row.items() if k not in _NOT_COPIED}
    copy[tables.SUPERSEDED] = superseded_ms
    copy[tables.SUPERSEDED_BY] = superseded_by
    copy[tables.PRIOR_ROW_ID] = str(row[tables.ROW_ID])
    return copy

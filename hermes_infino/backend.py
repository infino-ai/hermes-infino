"""Where a profile's rows go and come from: the local databases always, and
the cloud databases once signed in, caught up by the mirror.

Two local databases: the team's tables (`infino/db`), which the Infino tools
read, and the person's own (`infino/personal`): their memory, and the
conversations and traces kept personal (privacy.py), which only the plugin
opens. Nothing personal is ever in a database a tool or a group chat can
reach: locally it is in its own, and in the cloud in the personal database,
or nowhere when the person has none.

One backend per profile home, shared by the memory provider, the trace hooks
and the skills loader. Writes land locally and return; when signed in they
also schedule a sync, which runs on a background thread once the rows have
settled (`mirror.SETTLE_MS`), at most one at a time. A write made while a
sync runs, or a sync that found another process syncing, schedules one more.
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
from .mirror import PERSONAL_MEMORY, SETTLE_MS, Mirror

logger = logging.getLogger(__name__)

# How long after a write its sync runs: past the settle window, so the write
# is pushed by the sync it scheduled.
SYNC_DELAY_S = SETTLE_MS / 1000 + 0.5
# The most a session's end waits for its rows to reach the cloud.
FLUSH_TIMEOUT_S = 30.0
# The tables whose rows follow their session: personal sessions' go to the
# person's own database.
TRANSCRIPT_TABLES = (tables.CONVERSATIONS, tables.TRACES)


class Backend:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        state = config.state_dir(settings.home)
        self.local = LocalStore.under(state)
        self.personal = LocalStore.under(state, PERSONAL_DB_DIR)
        self.mirror = Mirror(settings, self.local, self.personal) if settings.signed_in else None
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._last_write = 0.0
        # A sync was asked for while one was pending or running.
        self._again = False

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
        background) and then the memories saved without a vector."""
        if self.mirror is not None:
            self.local.drop_vectors()
            self.personal.drop_vectors()
            self.mirror.ensure_cloud_tables()
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
        that made it so, and after a restart."""
        if privacy.transcript_personal(self.settings, session_id, messages):
            return True
        if privacy.first_look(session_id) and self._kept_personal(session_id):
            privacy.note_personal(session_id)
            return True
        return False

    def _kept_personal(self, session_id: str) -> bool:
        """Whether the person's own tables hold any of a session's turns."""
        session = tables.sql_string(session_id)
        for table in TRANSCRIPT_TABLES:
            try:
                found = self.personal.query_sql(f"SELECT count(*) AS n FROM {table} WHERE session_id = {session}")
            except Exception:  # a table not made yet holds nothing
                continue
            if found and found[0]["n"]:
                return True
        return False

    def forget_memory(self, content: str, scope: Optional[str] = None, visible: tuple[str, ...] = ("",),
                      person: Optional[str] = None, about_prefix: str = "") -> int:
        """Delete memories with exactly this text, of the chats in `visible`
        ("" is no chat), in `scope` when given: a chat forgets only what it
        can recall. `person` (a gateway turn's sender) limits personal
        memory to theirs. `about_prefix` also matches the text after any
        "About <name>: " prefix of that shape (a group's entries about its
        members, whoever wrote them)."""
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
        reaches_personal = scope in (None, recall.PERSONAL) and visible == ("",) and person != ""
        if person:
            predicate += f" AND author = {tables.sql_string(person)}"
        personal = self.personal.delete(tables.MEMORY, predicate) if reaches_personal else []
        if self.mirror is not None:
            # Each row is deleted from the cloud database its source went to.
            s = self.settings
            self.mirror.forget(tables.MEMORY, s.team_database, [str(r[tables.ROW_ID]) for r in team])
            if s.personal_database:
                self.mirror.forget(tables.MEMORY, s.personal_database, [str(r[tables.ROW_ID]) for r in personal])
            self.schedule_sync()
        return len(team) + len(personal)

    # -- sync ----------------------------------------------------------------

    def schedule_sync(self, delay_s: float = SYNC_DELAY_S) -> None:
        if self.mirror is None:
            return
        with self._timer_lock:
            if self._timer is not None and self._timer.is_alive():
                # The pending or running sync may have read its rows already: one more after it.
                self._again = True
                return
            self._timer = threading.Timer(delay_s, self._sync_quietly)
            self._timer.daemon = True
            self._timer.start()

    def _sync_quietly(self) -> None:
        skipped = False
        try:
            pushed = self.mirror.sync()
            skipped = pushed == {}  # another process holds the sync: it may not see these rows
            if any(pushed.values()):
                logger.info("infino: pushed %s", {t: n for t, n in pushed.items() if n})
        except Exception as e:
            logger.warning("infino: sync to the cloud stopped (the rows stay local and go next time): %s", e)
        with self._timer_lock:
            again, self._again = self._again or skipped, False
            self._timer = None
        if again:
            self.schedule_sync()

    def flush(self) -> None:
        """At a session's end: wait for this process's rows to settle, then
        push them, so a one-shot session's work reaches the cloud before it
        exits. Bounded; whatever is left goes with the next sync."""
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
    """The shared backend of a profile, rebuilt when its sign-in changes."""
    key = (str(Path(settings.home).resolve()), settings.api_key, settings.team_database,
           settings.personal_database, settings.url, settings.personal_profile, settings.personal_jobs)
    with _backends_lock:
        backend = _backends.get(key)
        if backend is None:
            backend = _backends[key] = Backend(settings)
        return backend

"""Where a profile's rows go and come from: the local database always, and the
cloud database once signed in, caught up by the mirror.

One backend per profile home, shared by the memory provider, the trace hooks
and the skills loader. Writes land locally and return; when signed in they
also schedule a sync, which runs on a background thread once the rows have
settled (`mirror.SETTLE_MS`), at most one at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config, recall, tables
from .client import InfinoClient
from .local import LocalStore
from .mirror import SETTLE_MS, Mirror

logger = logging.getLogger(__name__)

# How long after a write its sync runs: past the settle window, so the write
# is pushed by the sync it scheduled.
SYNC_DELAY_S = SETTLE_MS / 1000 + 0.5
# The most a session's end waits for its rows to reach the cloud.
FLUSH_TIMEOUT_S = 30.0


class Backend:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        self.local = LocalStore.under(config.state_dir(settings.home))
        self.mirror = Mirror(settings, self.local) if settings.signed_in else None
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._last_write = 0.0

    @property
    def signed_in(self) -> bool:
        return self.mirror is not None

    def ensure(self) -> None:
        """Create what is missing: the local tables (with local vectors only
        in local mode) and, signed in, the cloud tables; then catch up."""
        self.local.ensure_tables(with_vectors=not self.signed_in)
        if self.mirror is not None:
            self.local.drop_vectors()
            self.mirror.ensure_cloud_tables()
            self.schedule_sync(delay_s=0)

    # -- writes --------------------------------------------------------------

    def append(self, table: str, rows: list[dict[str, Any]]) -> None:
        self.local.append(table, rows)
        self._last_write = time.monotonic()
        self.schedule_sync()

    def forget_memory(self, content: str, scope: Optional[str] = None) -> int:
        """Delete memories with exactly this text (in `scope` when given)."""
        predicate = f"content = {tables.sql_string(content)}"
        if scope:
            predicate += f" AND scope = {tables.sql_string(scope)}"
        deleted = self.local.delete(tables.MEMORY, predicate, (tables.ROW_ID, "scope"))
        if self.mirror is not None and deleted:
            # Each row is deleted from the database its scope put it in.
            by_db: dict[str, list[str]] = {}
            for r in deleted:
                by_db.setdefault(self.mirror.database_for(tables.MEMORY, str(r.get("scope") or "")), []).append(
                    str(r[tables.ROW_ID]))
            for database, ids in by_db.items():
                self.mirror.forget(tables.MEMORY, database, ids)
            self.schedule_sync()
        return len(deleted)

    # -- sync ----------------------------------------------------------------

    def schedule_sync(self, delay_s: float = SYNC_DELAY_S) -> None:
        if self.mirror is None:
            return
        with self._timer_lock:
            if self._timer is not None and self._timer.is_alive():
                return  # the pending sync will see these rows too, or reschedule
            self._timer = threading.Timer(delay_s, self._sync_quietly)
            self._timer.daemon = True
            self._timer.start()

    def _sync_quietly(self) -> None:
        try:
            pushed = self.mirror.sync()
            if any(pushed.values()):
                logger.info("infino: pushed %s", {t: n for t, n in pushed.items() if n})
        except Exception as e:
            logger.warning("infino: sync to the cloud stopped (the rows stay local and go next time): %s", e)
        # Rows written while this sync ran were too new for it.
        if time.monotonic() - self._last_write < SYNC_DELAY_S:
            with self._timer_lock:
                self._timer = None
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
            return recall.Recaller(self.local)
        s = self.settings
        databases = [(recall.TEAM, s.team_database)]
        if s.personal_database:
            databases.append((recall.PERSONAL, s.personal_database))
        # Recall is waited on by the turn: a short timeout and no retries.
        cloud = InfinoClient(s.api_key, s.url, timeout=recall.RECALL_TIMEOUT_S, retries=0)
        return recall.Recaller(self.local, cloud, databases, mark=lambda: self.mirror.unsynced_after(tables.MEMORY))

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
           settings.personal_database, settings.url)
    with _backends_lock:
        backend = _backends.get(key)
        if backend is None:
            backend = _backends[key] = Backend(settings)
        return backend

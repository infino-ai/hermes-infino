"""The local database: the plugin's four tables on this machine, through the
Infino engine's Python binding, under the profile's home.

It is always written, signed in or not: without an account it is the only
store; with one it is what the cloud is caught up from (`mirror.py`), and
what recall and the tools fall back on while the cloud cannot be reached.

The local tables have the cloud's columns, less the cloud's embedding
column. In local mode each memory's 384-dimension vector, computed here
(`embed.py`), goes in a local-only table beside them, `memory_vectors`
(`row_id`, `embedding`), and recall fuses a keyword search over `memory`
with a vector search over `memory_vectors` by reciprocal rank - what the
engine's own hybrid search does within one table. The vectors are kept
apart because signing in ends them: the cloud embeds with its own model, so
sign-up drops `memory_vectors` (the engine drops a table, never the column
behind a vector index, and allows no null vectors), and every memory write
checks for the table, so a process that started before the sign-up stops
embedding too. Local recall is then keyword-only, over this machine's own
rows, which is all the local copy holds.

A memory is saved before its vector: while the model is not loaded (it loads
in the background; offline on first use it cannot) the memory is kept and
found by keyword, and every memory without a vector is embedded once the
model is ready. Recall is keyword-only until then.

Every call opens its own connection: an open connection keeps the snapshot
it opened with and never sees a commit another process made after it (the
Hermes gateway and a CLI session can share a profile), and opening one on a
local folder is cheap.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import infino
import pyarrow as pa

from . import config, embed, tables

logger = logging.getLogger(__name__)

# The local databases under the profile's state directory: the team's tables,
# which the Infino tools read, and the person's own memory, which nothing but
# the plugin opens - so no chat, and no tool call, can reach it.
DB_DIR = "db"
PERSONAL_DB_DIR = "personal"
# Times a write is reissued when another process won the commit race.
CONFLICT_RETRIES = 5
# Seconds before the first reissue; doubled each time.
CONFLICT_BACKOFF_S = 0.05
# Rows per append when a caller hands over many.
APPEND_BATCH = 512
# A commit slower than this, from its stamp to done, may have been passed by a
# sync's mark: it must stay well under the mirror's settle window (SETTLE_MS).
SLOW_COMMIT_MS = 1_000
# The furthest past this machine's clock a row is stamped to keep commit order.
# A table whose latest row is stamped further ahead (written while the clock
# was ahead, and since put right) would hold every new row back from the cloud
# until the clock caught up: those are stamped by the clock instead, and noted
# like a slow commit.
MAX_STAMP_AHEAD_MS = SLOW_COMMIT_MS
# Beside the databases: the commits a sync's mark may have passed (slower than
# that, or stamped behind a table's latest row), for the next sync.
LATE_COMMITS_FILE = "late_commits.jsonl"
# The local-only table of memory vectors (local mode).
VECTORS = "memory_vectors"
# Reciprocal-rank fusion's damping constant (the usual 60): a row's fused
# score is the sum of 1 / (RRF_K + rank) over the searches that found it.
RRF_K = 60
# How many more candidates each side of a fused search fetches than it returns.
FUSION_DEPTH = 3
# The farthest a vector-only hit may be from the question (cosine distance, 1 -
# similarity). A nearest-neighbour search always returns its k nearest, related
# or not; measured with this model, questions about a memory scored 0.22-0.88
# similarity and unrelated or mismatched ones 0.0-0.25, so 0.2 similarity keeps
# every related pair seen and drops most of the rest.
MAX_VECTOR_DISTANCE = 0.8

_ARROW_TYPES = {
    "utf8": pa.utf8(),
    "large_utf8": pa.large_utf8(),
    "i64": pa.int64(),
    "f64": pa.float64(),
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def append_line(path: Path, line: str) -> None:
    """Add a line to a file the profile's processes share (a queue the mirror
    reads), under the file's lock and on disk before the lock is released:
    whoever takes the lock next sees the line whole, and a rewrite of the file
    under the lock cannot write over it."""
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def arrow_schema(table: str) -> pa.Schema:
    """The local schema: the cloud's columns, less the cloud's embedding column."""
    if table == VECTORS:
        return pa.schema([pa.field(tables.ROW_ID, pa.utf8()),
                          pa.field(tables.EMBEDDING_COLUMN, pa.list_(pa.float32(), embed.DIM), nullable=False)])
    schema, _ = tables.TEAM_TABLES[table]
    return pa.schema([pa.field(f["name"], _ARROW_TYPES[f["type"]]) for f in schema if f["type"] != "embedding"])


def index_spec(table: str) -> "infino.IndexSpec":
    spec = infino.IndexSpec()
    if table == VECTORS:
        return spec.vector(tables.EMBEDDING_COLUMN, embed.DIM, "cosine")
    _, indexes = tables.TEAM_TABLES[table]
    for column in indexes.get("fts", []):
        spec = spec.fts(column)
    return spec


class LocalStore:
    """The local database of one profile."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def under(cls, state_dir: Path, folder: str = DB_DIR) -> "LocalStore":
        return cls(Path(state_dir) / folder)

    def _connect(self):
        if not self.path.is_dir():
            config.make_private_dir(self.path.parent)
            self.path.mkdir(mode=config.PRIVATE_DIR_MODE, exist_ok=True)
        return infino.connect(str(self.path))

    def ensure_tables(self, with_vectors: bool, only: tuple[str, ...] = tables.ALL) -> None:
        """Create the tables of `only` that are missing, and `memory_vectors`
        when `with_vectors` (local mode)."""
        db = self._connect()
        existing = set(db.list_tables())
        for table in only + ((VECTORS,) if with_vectors else ()):
            if table in existing:
                self._add_missing_columns(db, table)
                continue
            try:
                db.create_table(table, arrow_schema(table), index_spec(table))
            except infino.InfinoError:
                if table not in self._connect().list_tables():  # another process made it first
                    raise

    def _add_missing_columns(self, db, table: str) -> None:
        """Columns a newer version writes, added to a table an older one made
        (its rows read them as null)."""
        if table == VECTORS:
            return
        schema, _ = tables.TEAM_TABLES[table]
        have = {f.get("name") for f in db.schema(table).get("fields", [])}
        missing = [{"name": f["name"], "type": f["type"], "nullable": True}
                   for f in schema if f["type"] != "embedding" and f["name"] not in have]
        if missing:
            db.schema(table, {"fields": missing})

    def list_tables(self) -> list[str]:
        return list(self._connect().list_tables())

    def has_vectors(self) -> bool:
        """Whether memories are embedded here (local mode)."""
        return VECTORS in self.list_tables()

    def drop_vectors(self) -> None:
        """Stop embedding the local copy (at sign-up)."""
        if self.has_vectors():
            try:
                self._connect().drop_table(VECTORS)
            except infino.InfinoError:
                if self.has_vectors():  # not another process dropping it first
                    raise

    # -- writes --------------------------------------------------------------

    def append(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Append rows, each given a `row_id` unless it has one and stamped
        `stored_ms` as it commits. `memory` rows get a vector from `content`
        while the table has its vector column."""
        if not rows:
            return
        for r in rows:
            r.setdefault(tables.ROW_ID, str(uuid.uuid4()))
        for start in range(0, len(rows), APPEND_BATCH):
            self._append_batch(table, rows[start:start + APPEND_BATCH])

    def _append_batch(self, table: str, batch: list[dict[str, Any]]) -> None:
        schema = arrow_schema(table)

        others = [f for f in schema if f.name != tables.STORED]
        values = {f.name: pa.array([r.get(f.name) for r in batch], f.type) for f in others}

        def attempt():
            # Stamped per attempt, as late as it can be: commit order is what the
            # mirror follows. Past every row already committed, so a clock
            # stepped back never stamps a row behind the mirror's mark - unless
            # that is more than MAX_STAMP_AHEAD_MS past the clock: then by the
            # clock, and noted, so the next sync goes back for the row if a mark
            # moved on by the clock that was ahead has passed it.
            now, after = _now_ms(), self._latest_stored(table) + 1
            clock_behind = after - now > MAX_STAMP_AHEAD_MS
            stored = now if clock_behind else max(now, after)
            columns = {f.name: values[f.name] if f.name != tables.STORED else pa.array([stored] * len(batch), f.type)
                       for f in schema}
            self._connect().open_table(table).append(pa.table(columns, schema=schema))
            if clock_behind:
                logger.info("infino: the latest %s row is stamped ahead of this machine's clock (set back since?); "
                            "this one is stamped by the clock, and the next sync re-checks it", table)
                self._note_late_commit(table, stored)
            elif _now_ms() - stored > SLOW_COMMIT_MS:
                logger.warning("infino: a %s commit took over %d ms; the next sync re-checks it", table, SLOW_COMMIT_MS)
                self._note_late_commit(table, stored)

        self._retry(attempt)
        if table == tables.MEMORY and self.has_vectors():
            try:
                self._append_vectors(batch)
            except Exception as e:  # the memories are saved; they get vectors when the model can run
                logger.warning("infino: %d memories saved without vectors (found by keyword until they are "
                               "embedded): %s", len(batch), e)

    def _note_late_commit(self, table: str, stored: int) -> None:
        """Rows a sync may have moved its mark past before they were visible (a
        commit slower than the mirror's settle window, or rows stamped behind
        the table's latest): the note makes the next sync go back for them
        (mirror.py)."""
        line = json.dumps({"store": self.path.name, "table": table, "stored_ms": stored}) + "\n"
        append_line(self.path.parent / LATE_COMMITS_FILE, line)

    def _latest_stored(self, table: str) -> int:
        found = self.query_sql(f"SELECT max({tables.STORED}) AS top FROM {table}")
        return int(found[0]["top"] or 0) if found else 0

    def backfill_vectors(self) -> int:
        """Embed the memories saved without a vector (the model was not ready
        then); how many. Local mode, once the model is loaded."""
        if not self.has_vectors() or embed.ready() is None:
            return 0
        missing = self.query_sql(f"SELECT {tables.ROW_ID}, {tables.MEMORY_TEXT_COLUMN} FROM {tables.MEMORY} "
                                 f"WHERE {tables.ROW_ID} NOT IN (SELECT {tables.ROW_ID} FROM {VECTORS})")
        for start in range(0, len(missing), APPEND_BATCH):
            self._append_vectors(missing[start:start + APPEND_BATCH])
        return len(missing)

    def _append_vectors(self, batch: list[dict[str, Any]]) -> None:
        """The batch's vectors, after its rows: a memory whose vector never
        lands (a crash between the two, or no model yet) is still found by
        keyword, and embedded by `backfill_vectors`."""
        model = embed.ready()
        if model is None:
            return
        vectors = model([str(r.get(tables.MEMORY_TEXT_COLUMN) or "") for r in batch])
        schema = arrow_schema(VECTORS)
        data = pa.table({tables.ROW_ID: pa.array([r[tables.ROW_ID] for r in batch], pa.utf8()),
                         tables.EMBEDDING_COLUMN: pa.array(vectors, schema.field(tables.EMBEDDING_COLUMN).type)},
                        schema=schema)
        try:
            self._retry(lambda: self._connect().open_table(VECTORS).append(data))
        except infino.InfinoError:
            if self.has_vectors():  # not the table dropped by a sign-up meanwhile
                raise

    def delete(self, table: str, predicate: str, columns: tuple[str, ...] = (tables.ROW_ID,)) -> list[dict]:
        """Delete matching rows (and, for `memory`, their vectors); `columns`
        of the rows deleted (the mirror deletes the same rows, by `row_id`)."""
        found = self.query_sql(f"SELECT {', '.join(dict.fromkeys((tables.ROW_ID,) + columns))} "
                               f"FROM {table} WHERE {predicate}")
        if found:
            # By id, not by the predicate again: a matching row another process
            # commits in between is not deleted here without being reported.
            by_id = f"{tables.ROW_ID} IN ({', '.join(tables.sql_string(str(r[tables.ROW_ID])) for r in found)})"
            self._retry(lambda: self._connect().open_table(table).delete(by_id))
            if table == tables.MEMORY and self.has_vectors():
                self._retry(lambda: self._connect().open_table(VECTORS).delete(by_id))
        return found

    def _retry(self, op):
        wait = CONFLICT_BACKOFF_S
        for attempt in range(CONFLICT_RETRIES + 1):
            try:
                return op()
            except infino.ConflictError:
                if attempt == CONFLICT_RETRIES:
                    raise
                time.sleep(wait)
                wait *= 2

    # -- reads ---------------------------------------------------------------

    def search_memory(self, query: str, k: int, projection: list[str]) -> list[dict]:
        """The `k` best memories: keyword and vector search (the query
        embedded here) fused by reciprocal rank in local mode, keyword alone
        once the local copy carries no vectors or while the model is not
        ready. `score` is the fused score (or the BM25 score), higher better."""
        db = self._connect()
        cols = [c for c in projection if c != "score"]
        keyword = self.keyword_search(tables.MEMORY, tables.MEMORY_TEXT_COLUMN, query, k * FUSION_DEPTH, cols, db)
        model = embed.ready()
        if model is None or not self.has_vectors():
            return keyword[:k]
        try:
            [vector] = model([query])
            nearest = [r for r in db.open_table(VECTORS).vector_search(
                tables.EMBEDDING_COLUMN, vector, k * FUSION_DEPTH, None, None, None, [tables.ROW_ID, "score"]
            ).to_pylist() if r["score"] <= MAX_VECTOR_DISTANCE]
        except Exception as e:  # the keyword hits still answer
            logger.info("infino: vector recall failed, keyword only: %s", e)
            return keyword[:k]
        fused: dict[str, float] = {}
        for ranked in (keyword, nearest):
            for rank, r in enumerate(ranked):
                fused[r[tables.ROW_ID]] = fused.get(r[tables.ROW_ID], 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        rows = {r[tables.ROW_ID]: r for r in keyword}
        missing = [i for i in best if i not in rows]
        if missing:
            ids = ", ".join(tables.sql_string(i) for i in missing)
            select = ", ".join(f'"{c}"' for c in dict.fromkeys(cols + [tables.ROW_ID]))
            for r in self.query_sql(f"SELECT {select} FROM {tables.MEMORY} WHERE {tables.ROW_ID} IN ({ids})"):
                rows[r[tables.ROW_ID]] = r
        return [{**rows[i], "score": fused[i]} for i in best if i in rows]

    def keyword_search(self, table: str, column: str, query: str, k: int, projection: list[str], db=None) -> list[dict]:
        """The `k` best rows of `table` for `query` by BM25 over `column`,
        with the columns of `projection` and their `row_id` and `score`
        (higher better). `db` is a connection already open, if any."""
        cols = [c for c in projection if c != "score"]
        return (db or self._connect()).open_table(table).bm25_search(
            column, query, k, "or", list(dict.fromkeys(cols + [tables.ROW_ID, "score"]))).to_pylist()

    def query_sql(self, sql: str) -> list[dict]:
        return self._connect().query_sql(sql).to_pylist()

    def rows_after(self, table: str, mark: Optional[tuple[int, str]], until_ms: int, limit: int) -> list[dict]:
        """Up to `limit` rows stored at or before `until_ms` and after `mark`,
        in (stored_ms, row_id) order, without the local-only vector."""
        schema, _ = tables.TEAM_TABLES[table]
        cols = ", ".join(f'"{f["name"]}"' for f in schema if f["type"] != "embedding")
        return self.query_sql(f"SELECT {cols} FROM {table} WHERE {_after(mark, until_ms)} "
                              f"ORDER BY {tables.STORED}, {tables.ROW_ID} LIMIT {int(limit)}")

    def count_after(self, table: str, mark: Optional[tuple[int, str]], until_ms: int) -> int:
        """How many rows `rows_after` would give with no limit."""
        found = self.query_sql(f"SELECT count(*) AS n FROM {table} WHERE {_after(mark, until_ms)}")
        return int(found[0]["n"]) if found else 0

    def pushed_ids(self, table: str, upto_ms: int, limit: int) -> list[str]:
        """The `row_id`s of the latest `limit` rows stored at or before `upto_ms`."""
        found = self.query_sql(f"SELECT {tables.ROW_ID} FROM {table} WHERE {tables.STORED} <= {int(upto_ms)} "
                               f"ORDER BY {tables.STORED} DESC, {tables.ROW_ID} DESC LIMIT {int(limit)}")
        return [str(r[tables.ROW_ID]) for r in found]

    def has_row(self, table: str, row_id: str) -> bool:
        found = self.query_sql(f"SELECT count(*) AS n FROM {table} WHERE {tables.ROW_ID} = {tables.sql_string(row_id)}")
        return bool(found and found[0]["n"])


def _after(mark: Optional[tuple[int, str]], until_ms: int) -> str:
    """Rows stored at or before `until_ms` and after `mark`, in (stored_ms, row_id) order."""
    where = f"{tables.STORED} <= {int(until_ms)}"
    if mark is not None:
        ms, row_id = int(mark[0]), tables.sql_string(mark[1])
        where += (f" AND ({tables.STORED} > {ms} OR "
                  f"({tables.STORED} = {ms} AND {tables.ROW_ID} > {row_id}))")
    return where

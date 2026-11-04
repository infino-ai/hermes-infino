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

Every call opens its own connection: an open connection keeps the snapshot
it opened with and never sees a commit another process made after it (the
Hermes gateway and a CLI session can share a profile), and opening one on a
local folder is cheap.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import infino
import pyarrow as pa

from . import embed, tables

logger = logging.getLogger(__name__)

# The local folder under the profile's state directory.
DB_DIR = "db"
# Times a write is reissued when another process won the commit race.
CONFLICT_RETRIES = 5
# Seconds before the first reissue; doubled each time.
CONFLICT_BACKOFF_S = 0.05
# Rows per append when a caller hands over many.
APPEND_BATCH = 512
# The local-only table of memory vectors (local mode).
VECTORS = "memory_vectors"
# Reciprocal-rank fusion's damping constant (the usual 60): a row's fused
# score is the sum of 1 / (RRF_K + rank) over the searches that found it.
RRF_K = 60
# How many more candidates each side of a fused search fetches than it returns.
FUSION_DEPTH = 3

_ARROW_TYPES = {
    "utf8": pa.utf8(),
    "large_utf8": pa.large_utf8(),
    "i64": pa.int64(),
    "f64": pa.float64(),
}


def _now_ms() -> int:
    return int(time.time() * 1000)


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
    def under(cls, state_dir: Path) -> "LocalStore":
        return cls(Path(state_dir) / DB_DIR)

    def _connect(self):
        self.path.mkdir(parents=True, exist_ok=True)
        return infino.connect(str(self.path))

    def ensure_tables(self, with_vectors: bool) -> None:
        """Create the tables that are missing, and `memory_vectors` when
        `with_vectors` (local mode)."""
        db = self._connect()
        existing = set(db.list_tables())
        for table in tables.ALL + ((VECTORS,) if with_vectors else ()):
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

        def attempt():
            stored = _now_ms()  # stamped per attempt: commit order is what the mirror follows
            columns = {
                f.name: pa.array([stored] * len(batch) if f.name == tables.STORED else [r.get(f.name) for r in batch],
                                 f.type)
                for f in schema
            }
            self._connect().open_table(table).append(pa.table(columns, schema=schema))

        self._retry(attempt)
        if table == tables.MEMORY and self.has_vectors():
            self._append_vectors(batch)

    def _append_vectors(self, batch: list[dict[str, Any]]) -> None:
        """The batch's vectors, after its rows: a memory whose vector never
        lands (a crash between the two) is still found by keyword."""
        vectors = embed.embedder()([str(r.get(tables.MEMORY_TEXT_COLUMN) or "") for r in batch])
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
            self._retry(lambda: self._connect().open_table(table).delete(predicate))
            if table == tables.MEMORY and self.has_vectors():
                ids = ", ".join(tables.sql_string(str(r[tables.ROW_ID])) for r in found)
                self._retry(lambda: self._connect().open_table(VECTORS).delete(f"{tables.ROW_ID} IN ({ids})"))
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
        once the local copy carries no vectors. `score` is the fused score
        (or the BM25 score), higher better."""
        db = self._connect()
        cols = [c for c in projection if c != "score"]
        keyword = db.open_table(tables.MEMORY).bm25_search(
            tables.MEMORY_TEXT_COLUMN, query, k * FUSION_DEPTH, "or", list(dict.fromkeys(cols + [tables.ROW_ID, "score"]))
        ).to_pylist()
        if not self.has_vectors():
            return keyword[:k]
        [vector] = embed.embedder()([query])
        nearest = db.open_table(VECTORS).vector_search(
            tables.EMBEDDING_COLUMN, vector, k * FUSION_DEPTH, None, None, None, [tables.ROW_ID]).to_pylist()
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

    def query_sql(self, sql: str) -> list[dict]:
        return self._connect().query_sql(sql).to_pylist()

    def rows_after(self, table: str, mark: Optional[tuple[int, str]], until_ms: int, limit: int) -> list[dict]:
        """Up to `limit` rows stored at or before `until_ms` and after `mark`,
        in (stored_ms, row_id) order, without the local-only vector."""
        schema, _ = tables.TEAM_TABLES[table]
        cols = ", ".join(f'"{f["name"]}"' for f in schema if f["type"] != "embedding")
        where = f"{tables.STORED} <= {int(until_ms)}"
        if mark is not None:
            ms, row_id = int(mark[0]), tables.sql_string(mark[1])
            where += (f" AND ({tables.STORED} > {ms} OR "
                      f"({tables.STORED} = {ms} AND {tables.ROW_ID} > {row_id}))")
        return self.query_sql(f"SELECT {cols} FROM {table} WHERE {where} "
                              f"ORDER BY {tables.STORED}, {tables.ROW_ID} LIMIT {int(limit)}")

"""Rows into a cloud table by the platform's load: the rows are written to a
Parquet file on this machine, uploaded to the database's storage, and a load
job appends them to the table, making it on its first load.

The platform chooses what a table indexes and embeds, from the rows of its
first load, and every later load keeps those choices. So nothing here
declares a text index or an embedding column: readers ask the platform what
the table has (`roles`). The file carries this plugin's own column types
(`tables`), so a column that is empty in one batch keeps its type, and every
load of a table has the same columns; every text column is named as text, so
a column that holds JSON (a tool call's arguments) stays the text it is.

The platform may cut a long text value into passages, one row each, every
passage keeping the row's other columns, `row_id` included: one row of this
plugin can be several rows in the cloud. Readers take one hit per `row_id`
(`one_per_row`), and a row read back whole joins its passages (`whole_rows`).

A load job is its table's only writer while it runs: the platform refuses
another load of the table meanwhile, and a delete must wait for it to end
(`busy`). A load is waited for until it ends; a load that failed refused its
rows (`LoadFailed`), one that stopped or could not be followed may still be
taken in later (`LoadUnfinished`).
"""

from __future__ import annotations

import io
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Optional

from . import tables
from .client import InfinoClient, InfinoError

# A job's states while it has not ended, and the states of one whose rows are in.
LIVE_STATES = ("admitting", "pending", "running", "cancelling")
LOADED_STATES = ("succeeded", "synced")
FAILED_STATE = "failed"
# Seconds between two looks at a running load.
POLL_S = 2.0
# The most a load is waited for before it is left to finish on its own.
LOAD_WAIT_S = 30 * 60.0
# The column the platform numbers a row's passages with.
PASSAGE_COLUMN = "chunk"
# What a failed load answers as: a refusal, which the rows or the table may
# have caused (the mirror tells the two apart, as for any 400).
REFUSED_LOAD_STATUS = 400
# What an unfinished load answers as: try again later.
UNFINISHED_LOAD_STATUS = 503

_ARROW_TYPES = {"utf8": "string", "large_utf8": "large_string", "i64": "int64", "f64": "float64"}


class LoadFailed(InfinoError):
    """The load job failed for good: the platform refused its rows."""

    def __init__(self, message: str):
        super().__init__(REFUSED_LOAD_STATUS, message)


class LoadUnfinished(InfinoError):
    """The load did not end loaded: stopped, still running past the wait, or
    not to be found. Its rows may be taken in yet, so whoever sent them asks
    the cloud which it has before sending them again."""

    def __init__(self, message: str):
        super().__init__(UNFINISHED_LOAD_STATUS, message)


def schema_of(table: str) -> list[dict]:
    """The columns this plugin writes to `table` in the cloud: its schema,
    less the local embedding column (the platform makes its own)."""
    schema = tables.TEAM_TABLES[table][0] if table in tables.TEAM_TABLES else tables.CLOUD_ONLY[table][0]
    return [f for f in schema if f["type"] != "embedding"]


def text_columns(table: str) -> list[str]:
    """`table`'s text columns, each kept as its text by the load."""
    return [f["name"] for f in schema_of(table) if f["type"] in ("utf8", "large_utf8")]


def parquet(table: str, rows: list[dict]) -> bytes:
    """`rows` as a Parquet file with `table`'s columns and types."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = schema_of(table)
    schema = pa.schema([pa.field(f["name"], getattr(pa, _ARROW_TYPES[f["type"]])()) for f in fields])
    names = [f["name"] for f in fields]
    out = io.BytesIO()
    pq.write_table(pa.Table.from_pylist([{n: r.get(n) for n in names} for r in rows], schema=schema), out)
    return out.getvalue()


def origin(client_url: str) -> str:
    """The web origin uploads are opened for: the platform API's own."""
    parsed = urllib.parse.urlsplit(client_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def load(client: InfinoClient, url: str, database: str, table: str, rows: list[dict],
         wait_s: float = LOAD_WAIT_S) -> None:
    """Append `rows` to `database`.`table` by a load job (making the table on
    its first load), and wait for the job to end. Raises the platform's
    refusal, LoadFailed when the job failed, LoadUnfinished when it did not
    end loaded."""
    if not rows:
        return
    data = parquet(table, rows)
    name = f"{table}-{uuid.uuid4().hex}.parquet"
    opened = client.open_upload(database, origin(url), [{"name": name, "size": len(data)}])
    files = opened.get("files") or []
    if not files or not files[0].get("url") or not opened.get("prefix"):
        raise InfinoError(0, f"the upload for {database}.{table} was opened without a place to put the file")
    client.put_upload(files[0]["url"], data)
    client.submit_load(database, {
        "table": table,
        "source": {"kind": "prefix", "prefix": opened["prefix"]},
        "format": "parquet",
        # An append to a table that is not there yet makes it.
        "if_exists": "append",
        "text_columns": text_columns(table),
    })
    wait(client, database, table, wait_s)


def wait(client: InfinoClient, database: str, table: str, wait_s: float = LOAD_WAIT_S) -> None:
    """Wait for `table`'s load job to end. Returns when it loaded its rows;
    raises LoadFailed or LoadUnfinished otherwise."""
    deadline = time.monotonic() + wait_s
    while True:
        record = client.load_job(database, table)
        state = str((record or {}).get("state") or "")
        if record is None:
            raise LoadUnfinished(f"no load job of {database}.{table} to follow")
        if state in LOADED_STATES:
            return
        if state == FAILED_STATE:
            raise LoadFailed(f"the load of {database}.{table} failed: {record.get('error') or 'no reason given'}")
        if state not in LIVE_STATES:
            raise LoadUnfinished(f"the load of {database}.{table} ended {state or 'unknown'}: "
                                 f"{record.get('error') or 'no reason given'}")
        if time.monotonic() >= deadline:
            raise LoadUnfinished(f"the load of {database}.{table} is still {state} after {int(wait_s)} s")
        time.sleep(POLL_S)


def busy(client: InfinoClient, database: str, table: str) -> bool:
    """Whether a load job is writing `table` now."""
    record = client.load_job(database, table)
    return str((record or {}).get("state") or "") in LIVE_STATES


def settle(client: InfinoClient, database: str, table: str, wait_s: float = LOAD_WAIT_S) -> None:
    """Wait while a load job is writing `table`, however it then ends: what a
    reader asks the table afterwards is what that load left. Raises
    LoadUnfinished when it is still writing after `wait_s`."""
    deadline = time.monotonic() + wait_s
    while busy(client, database, table):
        if time.monotonic() >= deadline:
            raise LoadUnfinished(f"a load of {database}.{table} is still running after {int(wait_s)} s")
        time.sleep(POLL_S)


@dataclass(frozen=True)
class Roles:
    """What the platform chose for a table at its first load: the columns
    with a text index, and the column it embeds into (None when none)."""

    text: tuple[str, ...]
    embedding: Optional[str]

    def text_column(self, preferred: str) -> Optional[str]:
        """`preferred` when it is text-indexed, else the first that is."""
        if preferred in self.text:
            return preferred
        return self.text[0] if self.text else None


def roles(client: InfinoClient, database: str, table: str) -> Optional[Roles]:
    """What the platform indexes and embeds in `table`, from the record of
    the load that made it; None for a table no load made (one made before
    tables were loaded, or none yet)."""
    record = client.load_job(database, table)
    if record is None:
        return None
    settled = record.get("schema") or {}
    text = tuple(str(f.get("column")) for f in settled.get("fts") or [] if isinstance(f, dict) and f.get("column"))
    embed = settled.get("embed") or {}
    return Roles(text, str(embed["column"]) if isinstance(embed, dict) and embed.get("column") else None)


def one_per_row(rows: list[dict]) -> list[dict]:
    """The first hit of each `row_id`, in order: a row cut into passages is
    one hit. Rows without a `row_id` are kept as they are."""
    seen: set[str] = set()
    out = []
    for r in rows:
        key = r.get(tables.ROW_ID)
        if key is not None:
            if str(key) in seen:
                continue
            seen.add(str(key))
        out.append(r)
    return out


def whole_rows(rows: list[dict], column: str = "content") -> list[dict]:
    """One row per `row_id`, its `column` the row's passages joined in
    order: the row as this plugin wrote it, read back from the cloud, but
    for the spaces the platform trimmed where it cut the passages (6 of
    12,120 characters, measured on the platform). Exact for a row the
    platform did not cut, which is any memory of one passage's length."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    loose: list[dict] = []
    for r in rows:
        key = r.get(tables.ROW_ID)
        if key is None:
            loose.append(r)
            continue
        if str(key) not in groups:
            order.append(str(key))
        groups.setdefault(str(key), []).append(r)
    out = []
    for key in order:
        passages = sorted(groups[key], key=lambda r: int(r.get(PASSAGE_COLUMN) or 0))
        first = dict(passages[0])
        if len(passages) > 1:
            first[column] = "".join(str(p.get(column) or "") for p in passages)
        out.append(first)
    return out + loose

"""Long conversation text kept as chunks in the cloud, never cut short.

A long message or tool result (a file read, a log, a test run) goes to the
cloud as chunks: the rows of one message share its `row_id`, each holds one
chunk of the long text in its own column, and `chunk` numbers them from 0.
The chunks are cut by a client's `chunk` command (INFINO_CHUNKER, else
INFINO_CLIENT) - SuperGrep's chunker, the one that cuts a repository's files
for its index - so a tool
result is cut where SuperGrep would cut the same text: markdown at its
headings, logs at their records, other text in overlapping line windows.
Each chunk is well inside what the platform's model embeds, so every chunk
is embedded whole and found by its words and by its meaning.

The local database keeps every message whole; only what goes to the cloud is
cut. A message short enough is sent as it is. Where no client is set or it
has no `chunk` command, text is sent whole, and the cloud embeds as much of
it as its model reads.

Rows that share a `row_id` are one row to the plugin: a forget by `row_id`
deletes them all, and the sync's checks of what the cloud has ask by
`row_id`.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Callable, Optional

from . import tables

logger = logging.getLogger(__name__)

# The column numbering a row's chunks.
CHUNK_COLUMN = "chunk"
# The columns of each table whose long values go as chunks, and the file name
# that has the chunker cut each as that kind of text: an agent's message as
# markdown (at its headings), a tool's result as plain text (line windows).
CHUNKED = {tables.CONVERSATIONS: {"content": "message.md", "tool_result": "result.txt"}}
# Text shorter than this goes whole: about a thousand tokens, far inside the
# platform model's window, and shorter than the chunks the chunker cuts.
LONG_CHARS = 4_000
# Seconds the chunker may take for one sync's batch of rows.
CHUNK_TIMEOUT_S = 60.0

# Runs the chunker: requests in, chunks out (see `client_chunker`).
Chunker = Callable[[list[dict]], list[dict]]

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def client_chunker(node: str, client_js: str) -> Optional[Chunker]:
    """The Infino client's `chunk` command as a chunker: JSON lines
    `{id, path, text}` in, one JSON line per chunk out. None without a client."""
    if not client_js:
        return None

    def run(requests: list[dict]) -> list[dict]:
        stdin = "".join(json.dumps(r) + "\n" for r in requests)
        done = subprocess.run([node, client_js, "chunk"], input=stdin, capture_output=True, text=True,
                              timeout=CHUNK_TIMEOUT_S)
        if done.returncode != 0:
            raise RuntimeError(f"`chunk` exited {done.returncode}: {done.stderr.strip()[:300]}")
        return [json.loads(line) for line in done.stdout.splitlines() if line.strip()]

    return run


def expand(table: str, rows: list[dict], chunker: Optional[Chunker]) -> list[dict]:
    """`rows` as they go to the cloud: each row whose chunked columns are all
    short as it is, and each row with a long one as several, one
    per chunk, sharing its `row_id`. The first holds the row with each long
    column's first chunk; each after it holds one more chunk of one long
    column, its other chunked columns empty and every other column the row's.
    Without a chunker, or when it fails, every row goes whole."""
    columns = CHUNKED.get(table)
    if not columns:
        return rows
    requests = [{"id": f"{i}:{column}", "path": path, "text": r[column]}
                for i, r in enumerate(rows) for column, path in columns.items()
                if isinstance(r.get(column), str) and len(r[column]) >= LONG_CHARS]
    cut: dict[str, list[str]] = {}
    if requests and chunker is None:
        _warn_once("no-client", "infino: neither INFINO_CHUNKER nor INFINO_CLIENT is set, so long conversation text "
                                "goes to the cloud whole (the cloud embeds as much of it as its model reads)")
    elif requests:
        try:
            for c in chunker(requests):
                cut.setdefault(str(c["id"]), []).append(str(c.get("content") or ""))
        except Exception as e:
            _warn_once(f"failed:{type(e).__name__}", "infino: the client could not cut long text into chunks (set "
                                                      "INFINO_CHUNKER to SuperGrep's dist/cli.js, which has a `chunk` "
                                                      "command), so it goes whole: %s", e)
            cut = {}
    out: list[dict] = []
    for i, r in enumerate(rows):
        pieces = {column: cut[f"{i}:{column}"] for column in columns if len(cut.get(f"{i}:{column}", ())) > 1}
        if not pieces:
            out.append(r)  # one row: its `chunk` stays empty
            continue
        out.append({**r, **{column: chunks[0] for column, chunks in pieces.items()}, CHUNK_COLUMN: 0})
        n = 1
        for column, chunks in pieces.items():
            for text in chunks[1:]:
                out.append({**r, **{other: None for other in columns}, column: text, CHUNK_COLUMN: n})
                n += 1
    return out

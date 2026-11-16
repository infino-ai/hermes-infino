"""The tables hermes-infino keeps in each team (and personal) database:
everything the team's agents remember, say, do and share.

- `memory`: what the team (or a person) should know. Recall searches it by
  keyword and, where the platform embeds it, by meaning, with the query
  embedded server-side.
- `conversations`: every message of every session, with the tool calls an
  assistant message made and each tool's result in their own columns, so
  the facts read from `content` come from what people and agents said, not
  from tool plumbing, while the results stay searchable.
- `traces`: every model call and tool call.
- `skills`: the team's shared skills, one row per published version.
- `forgotten`: the ids of the rows a forget deleted from the cloud, so a
  machine whose rows a teammate forgot, every one, can tell that from a
  dropped table (`mirror.py`). Cloud only: nothing on this machine.

`conversations` and `traces` share `session_id` and `turn_id`, the grain a
question joins them on: one turn is one user message, what answered it, and
the model and tool calls that produced it.

Times are milliseconds since the epoch in an `i64`, which the JSON append
path takes as a plain number. Keyword-indexed columns are `large_utf8`, as
the platform requires.

Every row carries a `row_id` (a UUID), and `stored_ms`, stamped just before
the row commits to the local database. (`stored_ms`, `row_id`) is the order
the cloud is caught up in (`mirror.py`): commit order, whichever process
wrote the row, where `created_ms` (when the thing happened) is not.

The same tables exist in two places: the local database (the engine's Python
binding, under the profile's home), always; and the team's cloud database,
once signed in. Locally, `memory.embedding` is a 384-dimension vector this
machine computes (`embed.py`), and the indexes are the ones below. In the
cloud a table is made by the first load of its rows (`loads.py`), and the
platform chooses what it indexes and embeds from those rows: readers ask it
(`loads.roles`) rather than assume the names here. A cloud table made before
tables were loaded keeps the columns and indexes it was made with.

A person's own database, on this machine and in the cloud, has `PERSONAL`
of them: their memory, and the conversations and traces kept personal.
"""

import logging

from .client import InfinoClient, InfinoError

logger = logging.getLogger(__name__)

MEMORY = "memory"
CONVERSATIONS = "conversations"
TRACES = "traces"
SKILLS = "skills"
# Earlier versions of memories (see SUPERSEDED below).
HISTORY = "memory_history"
ALL = (MEMORY, CONVERSATIONS, TRACES, SKILLS, HISTORY)

# The local vector column, and the column it embeds; also what a cloud table
# made before tables were loaded calls its embedding.
EMBEDDING_COLUMN = "embedding"
MEMORY_TEXT_COLUMN = "content"
ROW_ID = "row_id"
STORED = "stored_ms"
# A memory's group chat (chats.py): kept for that chat and recalled only there.
CHAT = "chat"
# The earlier versions of memories, kept in a table of their own beside
# `memory` (backend.supersede_rows): a memory Hermes replaced, or consolidated
# away in one batch with what replaced it, is copied here with when it was
# superseded, what superseded it (empty when nothing names it, or when the
# replacement was written for another chat) and the row it was. `memory`
# itself holds current memories only, exactly as before, so nothing that
# reads it changes, and a version that does not know this table never sees
# or deletes it.
SUPERSEDED = "superseded_ms"
SUPERSEDED_BY = "superseded_by"
PRIOR_ROW_ID = "prior_row_id"

MEMORY_SCHEMA = [
    {"name": ROW_ID, "type": "utf8"},
    {"name": STORED, "type": "i64"},
    {"name": "content", "type": "large_utf8"},
    {"name": "scope", "type": "utf8"},          # team | personal
    {"name": "author", "type": "utf8"},         # who the memory came from
    {"name": "profile", "type": "utf8"},        # the Hermes profile that wrote it
    {"name": "session_id", "type": "utf8"},
    {"name": "origin", "type": "utf8"},         # remember | builtin:memory | builtin:user
    {"name": CHAT, "type": "utf8"},             # the group chat it is kept for ("" or null: the team's or the person's)
    {"name": "created_ms", "type": "i64"},
    {"name": EMBEDDING_COLUMN, "type": "embedding", "source": [MEMORY_TEXT_COLUMN], "nullable": False},
]
# No vector index is declared: the platform indexes an embedding column
# (cosine) by itself and refuses a declaration for one.
MEMORY_INDEXES = {"fts": [MEMORY_TEXT_COLUMN]}

# `memory`'s columns, less its embedding (an earlier version is found by its
# words, and is ranked for the question when ranking is on), with when it was
# superseded, what superseded it, and the row it was.
HISTORY_SCHEMA = [f for f in MEMORY_SCHEMA if f["type"] != "embedding"] + [
    {"name": SUPERSEDED, "type": "i64"},
    {"name": SUPERSEDED_BY, "type": "large_utf8"},
    {"name": PRIOR_ROW_ID, "type": "utf8"},
]
HISTORY_INDEXES = {"fts": [MEMORY_TEXT_COLUMN]}

# The rows forgets deleted from the cloud, one row each, written before the
# delete: the table it was in, its `row_id`, and when. A table dropped and
# made again loses this machine's rows and has none of them here; a forget
# of them has. It is in the same database as the rows, so a database dropped
# and made again loses both. Ids only: never what was forgotten.
FORGOTTEN = "forgotten"
FORGOTTEN_FROM = "forgotten_from"
FORGOTTEN_MS = "forgotten_ms"
FORGOTTEN_SCHEMA = [
    {"name": FORGOTTEN_FROM, "type": "utf8"},
    {"name": ROW_ID, "type": "utf8"},
    {"name": FORGOTTEN_MS, "type": "i64"},
]
FORGOTTEN_INDEXES: dict = {}

CONVERSATIONS_SCHEMA = [
    {"name": ROW_ID, "type": "utf8"},
    {"name": STORED, "type": "i64"},
    {"name": "session_id", "type": "utf8"},
    {"name": "turn_id", "type": "utf8"},       # the turn the message belongs to, as in `traces`
    {"name": "seq", "type": "i64"},             # position in the session's message list
    {"name": "role", "type": "utf8"},           # user | assistant | tool
    {"name": "content", "type": "large_utf8"},  # what the person or the agent said
    {"name": "tool_name", "type": "utf8"},      # assistant: the tools it called; tool: the tool that answered
    {"name": "tool_args", "type": "large_utf8"},    # assistant: the calls' arguments, as JSON
    {"name": "tool_result", "type": "large_utf8"},  # tool: what the tool returned
    {"name": "author", "type": "utf8"},
    {"name": "profile", "type": "utf8"},
    {"name": "platform", "type": "utf8"},
    {"name": "created_ms", "type": "i64"},
]
CONVERSATIONS_INDEXES = {"fts": ["content", "tool_result"]}

TRACES_SCHEMA = [
    {"name": ROW_ID, "type": "utf8"},
    {"name": STORED, "type": "i64"},
    {"name": "session_id", "type": "utf8"},
    {"name": "turn_id", "type": "utf8"},
    {"name": "request_id", "type": "utf8"},
    {"name": "kind", "type": "utf8"},           # model | tool
    {"name": "name", "type": "large_utf8"},     # model id, or tool name (keyword-indexed: large_utf8)
    {"name": "model", "type": "utf8"},          # the model id, on model calls only
    {"name": "tool", "type": "utf8"},           # the tool name, on tool calls only
    {"name": "provider", "type": "utf8"},
    {"name": "status", "type": "utf8"},
    {"name": "error", "type": "large_utf8"},
    {"name": "duration_ms", "type": "f64"},
    {"name": "input_tokens", "type": "i64"},    # the whole prompt, cached tokens included
    {"name": "output_tokens", "type": "i64"},
    {"name": "cached_tokens", "type": "i64"},   # of input_tokens, read from or written to the prompt cache
    {"name": "args", "type": "large_utf8"},
    {"name": "result", "type": "large_utf8"},
    {"name": "author", "type": "utf8"},         # who the turn was for, as in `conversations`
    {"name": "profile", "type": "utf8"},
    {"name": "created_ms", "type": "i64"},
]
TRACES_INDEXES = {"fts": ["name", "result"]}

SKILLS_SCHEMA = [
    {"name": ROW_ID, "type": "utf8"},
    {"name": STORED, "type": "i64"},
    {"name": "name", "type": "utf8"},           # the skill's name, as Hermes names it
    {"name": "description", "type": "large_utf8"},
    {"name": "content", "type": "large_utf8"},  # the whole SKILL.md
    {"name": "author", "type": "utf8"},
    {"name": "profile", "type": "utf8"},
    {"name": "created_ms", "type": "i64"},      # the newest row of a name is its current version
]
SKILLS_INDEXES = {"fts": ["description", "content"]}

TEAM_TABLES = {
    MEMORY: (MEMORY_SCHEMA, MEMORY_INDEXES),
    CONVERSATIONS: (CONVERSATIONS_SCHEMA, CONVERSATIONS_INDEXES),
    TRACES: (TRACES_SCHEMA, TRACES_INDEXES),
    SKILLS: (SKILLS_SCHEMA, SKILLS_INDEXES),
    HISTORY: (HISTORY_SCHEMA, HISTORY_INDEXES),
}
# Tables a cloud database may lack without holding up the others: one that is
# there without a column this version writes, which the key may not add,
# keeps its own rows on this machine, and every other table syncs as before
# (ensure_optional).
OPTIONAL = (HISTORY,)
# Tables the cloud databases have and this machine does not (no rows of
# them are kept or pushed here): optional too, made as OPTIONAL are.
CLOUD_ONLY = {FORGOTTEN: (FORGOTTEN_SCHEMA, FORGOTTEN_INDEXES)}
# A person's own database holds their personal memory and its history, and
# the conversations and traces kept personal (privacy.py): a personal
# profile's, a personal scheduled job's, a chat a personal job's output went to.
PERSONAL = (MEMORY, CONVERSATIONS, TRACES, HISTORY)
PERSONAL_TABLES = {t: TEAM_TABLES[t] for t in PERSONAL}


def sql_string(value: str) -> str:
    """A SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def chat_predicate(visible: tuple[str, ...]) -> str:
    """Rows of the chats in `visible`, "" standing for no chat (null or empty)."""
    terms = [f"{CHAT} IS NULL OR {CHAT} = ''" if c == "" else f"{CHAT} = {sql_string(c)}" for c in visible]
    return "(" + " OR ".join(terms) + ")"


class OutdatedTable(Exception):
    """A cloud table lacks a column this version writes, and it could not be added."""


def ensure_all(client: InfinoClient, database: str, personal: bool = False) -> None:
    """Create the database unless it exists, and make its tables that exist
    right. A table that is not there yet is made by the first load of its
    rows (`loads.py`), with what the platform chooses to index and embed.

    A table that exists must have every column this version writes: a row
    loaded into a `memory` without its `chat` would read as the whole team's
    memory, and a load whose columns the table lacks is refused. A missing
    column is added (nullable: the rows already there read it as null). When
    the platform refuses that - a key that may not manage the database - the
    table is refused by name, and the mirror loads nothing into the database
    until it is fixed: every row stays on this machine meanwhile."""
    client.ensure_database(database)
    have_tables = set(client.list_tables(database))
    for table, (schema, _indexes) in (PERSONAL_TABLES if personal else TEAM_TABLES).items():
        if table in OPTIONAL or table not in have_tables:
            continue  # optional ones on their own (ensure_optional); a missing one is made by its first load
        missing = _missing_columns(client, database, table, schema)
        if not missing:
            continue
        try:
            client.add_columns(database, table, [{"name": f["name"], "type": f["type"], "nullable": True}
                                                 for f in missing])
        except InfinoError as e:
            names = ", ".join(f["name"] for f in missing)
            raise OutdatedTable(f"{database}.{table} has no {names}, and adding it was refused ({e}). Add it as a "
                                f"nullable column with a key that may manage {database}; until then nothing is "
                                f"loaded into {database}, and every row stays on this machine") from None


def ensure_optional(client: InfinoClient, database: str, table: str) -> None:
    """Make one of OPTIONAL (or CLOUD_ONLY) in `database` right when it is
    there: every column this version writes. One that is not there yet is
    made by its first load. Raises OutdatedTable when a missing column cannot
    be added: the caller keeps that table's rows on this machine, and the
    database's other tables are not held up."""
    schema = TEAM_TABLES[table][0] if table in TEAM_TABLES else CLOUD_ONLY[table][0]
    if table not in client.list_tables(database):
        return
    missing = _missing_columns(client, database, table, schema)
    if missing:
        try:
            client.add_columns(database, table, [{"name": f["name"], "type": f["type"], "nullable": True}
                                                 for f in missing])
        except InfinoError as e:
            if e.status != 403:
                raise
            raise OutdatedTable(f"{database}.{table} has no {', '.join(f['name'] for f in missing)}, and this key "
                                f"may not add it: {e}") from e


def _missing_columns(client: InfinoClient, database: str, table: str, schema: list[dict]) -> list[dict]:
    """The columns of `schema` the cloud table lacks, the local embedding
    aside (the platform makes its own)."""
    have = set(client.column_names(database, table))
    return [f for f in schema if f["type"] != "embedding" and f["name"] not in have]

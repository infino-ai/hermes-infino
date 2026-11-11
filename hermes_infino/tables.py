"""The tables hermes-infino keeps in each team (and personal) database:
everything the team's agents remember, say, do and share.

- `memory`: what the team (or a person) should know. Its embedding column is
  filled by the platform on append from `content`, so recall runs hybrid
  (keyword + vector) search with the query embedded server-side.
- `conversations`: every message of every session, with the tool calls an
  assistant message made and each tool's result in their own columns, so
  the facts read from `content` come from what people and agents said, not
  from tool plumbing, while the results stay searchable.
- `traces`: every model call and tool call.
- `skills`: the team's shared skills, one row per published version.

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
once signed in. They differ in one column: locally, `memory.embedding` is a
384-dimension vector this machine computes (`embed.py`); in the cloud it is
the platform's embedding column, which the platform fills from `content`.

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
ALL = (MEMORY, CONVERSATIONS, TRACES, SKILLS)

# The column the platform embeds, and the column it embeds from.
EMBEDDING_COLUMN = "embedding"
MEMORY_TEXT_COLUMN = "content"
ROW_ID = "row_id"
STORED = "stored_ms"
# A memory's group chat (chats.py): kept for that chat and recalled only there.
CHAT = "chat"
# A memory Hermes replaced or removed is kept, not deleted (provider.py): a
# copy of its row, stamped when it was superseded and with what (empty when
# nothing names the replacement), pointing at the row it copies. A current
# memory has no `superseded_ms`. Only `infino_forget` deletes a memory.
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
    {"name": SUPERSEDED, "type": "i64"},        # when Hermes replaced or removed it; null while current
    {"name": SUPERSEDED_BY, "type": "utf8"},    # the entry that replaced it ("" when nothing names one)
    {"name": PRIOR_ROW_ID, "type": "utf8"},     # the row this superseded copy was made from
    {"name": EMBEDDING_COLUMN, "type": "embedding", "source": [MEMORY_TEXT_COLUMN], "nullable": False},
]
# No vector index is declared: the platform indexes an embedding column
# (cosine) by itself and refuses a declaration for one.
MEMORY_INDEXES = {"fts": [MEMORY_TEXT_COLUMN]}

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

# `memory` without the embedding column, where the platform cannot fill one:
# a deployment with no embedding server (create_table answers 400), or an
# account it gives none (403). Recall is then keyword-only.
MEMORY_SCHEMA_KEYWORD_ONLY = [f for f in MEMORY_SCHEMA if f["name"] != EMBEDDING_COLUMN]
# The statuses create_table refuses an embedding column with, its message
# saying "embedding": no embedding server on the deployment, or a worker
# built without embedding columns (400); none for this account (403).
NO_EMBEDDING_STATUSES = (400, 403)

TEAM_TABLES = {
    MEMORY: (MEMORY_SCHEMA, MEMORY_INDEXES),
    CONVERSATIONS: (CONVERSATIONS_SCHEMA, CONVERSATIONS_INDEXES),
    TRACES: (TRACES_SCHEMA, TRACES_INDEXES),
    SKILLS: (SKILLS_SCHEMA, SKILLS_INDEXES),
}
# A person's own database holds their personal memory, and the conversations
# and traces kept personal (privacy.py): a personal profile's, a personal
# scheduled job's, a chat a personal job's output went to.
PERSONAL = (MEMORY, CONVERSATIONS, TRACES)
PERSONAL_TABLES = {t: TEAM_TABLES[t] for t in PERSONAL}


def sql_string(value: str) -> str:
    """A SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def chat_predicate(visible: tuple[str, ...]) -> str:
    """Rows of the chats in `visible`, "" standing for no chat (null or empty)."""
    terms = [f"{CHAT} IS NULL OR {CHAT} = ''" if c == "" else f"{CHAT} = {sql_string(c)}" for c in visible]
    return "(" + " OR ".join(terms) + ")"


def current_predicate() -> str:
    """Memories still in force: not superseded (rows written before the
    column existed read it as null, and are current)."""
    return f"{SUPERSEDED} IS NULL"


class OutdatedTable(Exception):
    """A cloud table lacks a column this version writes, and it could not be added."""


def ensure_all(client: InfinoClient, database: str, personal: bool = False) -> None:
    """Create the database and its tables unless they exist. Where the
    platform cannot fill an embedding column (no embedding server on the
    deployment, or none for the account), `memory` is created keyword-only.

    A table that exists must have every column this version writes: a row
    pushed into a `memory` without its `chat` would read as the whole team's
    memory. A missing column is added (nullable: the rows already there read
    it as null). When the platform refuses that - a key that may not manage
    the database - the table is refused by name, and the mirror pushes
    nothing to the database until it is fixed: every row stays on this
    machine meanwhile."""
    client.ensure_database(database)
    for table, (schema, indexes) in (PERSONAL_TABLES if personal else TEAM_TABLES).items():
        try:
            client.ensure_table(database, table, schema, indexes)
        except InfinoError as e:
            if table != MEMORY or e.status not in NO_EMBEDDING_STATUSES or "embedding" not in e.message:
                raise
            logger.warning("infino: %s.%s is made without its embedding column, so recall there is "
                           "keyword-only: %s", database, MEMORY, e)
            client.ensure_table(database, MEMORY, MEMORY_SCHEMA_KEYWORD_ONLY, MEMORY_INDEXES)
        have = set(client.column_names(database, table))
        missing = [f for f in schema if f["type"] != "embedding" and f["name"] not in have]
        if not missing:
            continue
        try:
            client.add_columns(database, table, [{"name": f["name"], "type": f["type"], "nullable": True}
                                                 for f in missing])
        except InfinoError as e:
            names = ", ".join(f["name"] for f in missing)
            raise OutdatedTable(f"{database}.{table} has no {names}, and adding it was refused ({e}). Add it as a "
                                f"nullable column with a key that may manage {database}; until then nothing is "
                                f"pushed to {database}, and every row stays on this machine") from None

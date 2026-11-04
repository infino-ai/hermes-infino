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
"""

from .client import InfinoClient, InfinoError

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

MEMORY_SCHEMA = [
    {"name": ROW_ID, "type": "utf8"},
    {"name": STORED, "type": "i64"},
    {"name": "content", "type": "large_utf8"},
    {"name": "scope", "type": "utf8"},          # team | personal
    {"name": "author", "type": "utf8"},         # who the memory came from
    {"name": "profile", "type": "utf8"},        # the Hermes profile that wrote it
    {"name": "session_id", "type": "utf8"},
    {"name": "origin", "type": "utf8"},         # remember | builtin:memory | builtin:user
    {"name": "created_ms", "type": "i64"},
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

# `memory` without the embedding column, for an account the platform gives no
# embedding server (create_table answers 403). Recall is then keyword-only.
MEMORY_SCHEMA_KEYWORD_ONLY = [f for f in MEMORY_SCHEMA if f["name"] != EMBEDDING_COLUMN]

TEAM_TABLES = {
    MEMORY: (MEMORY_SCHEMA, MEMORY_INDEXES),
    CONVERSATIONS: (CONVERSATIONS_SCHEMA, CONVERSATIONS_INDEXES),
    TRACES: (TRACES_SCHEMA, TRACES_INDEXES),
    SKILLS: (SKILLS_SCHEMA, SKILLS_INDEXES),
}
# A person's own database holds their personal memory only.
PERSONAL_TABLES = {MEMORY: (MEMORY_SCHEMA, MEMORY_INDEXES)}


def sql_string(value: str) -> str:
    """A SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def ensure_all(client: InfinoClient, database: str, personal: bool = False) -> None:
    """Create the database and its tables unless they exist. Where the
    account has no embedding server, `memory` is created keyword-only."""
    client.ensure_database(database)
    for table, (schema, indexes) in (PERSONAL_TABLES if personal else TEAM_TABLES).items():
        try:
            client.ensure_table(database, table, schema, indexes)
        except InfinoError as e:
            if table != MEMORY or e.status != 403 or "embedding" not in e.message:
                raise
            client.ensure_table(database, MEMORY, MEMORY_SCHEMA_KEYWORD_ONLY, MEMORY_INDEXES)

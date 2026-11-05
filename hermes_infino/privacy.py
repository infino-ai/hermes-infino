"""Personal memory kept out of the team's tables.

`conversations` and `traces` record every tool call with its arguments, and
both go to the team's database. A call that writes personal memory carries
the private text in its arguments - `infino_remember` with scope "personal",
or Hermes's own `memory` tool on its `user` target (USER.md) - so the text is
withheld from those rows: the call is kept, with WITHHELD in place of the
text, and so is its result, which can echo the entries.

What a person says in a conversation is the conversation, and is kept as
said; this covers the memory written from it.
"""

from __future__ import annotations

import json
from typing import Any

# What stands in for the text of a personal memory.
WITHHELD = "[personal memory, kept out of the team's tables]"
# The text fields of the memory tools' arguments.
TEXT_FIELDS = ("content", "old_text", "new_text")


def _args(arguments: Any) -> Any:
    """Arguments as a dict when they are JSON text; otherwise as given."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def is_personal_call(tool_name: str, arguments: Any) -> bool:
    """Whether a tool call writes, changes or deletes personal memory."""
    args = _args(arguments)
    if not isinstance(args, dict):
        return False
    if tool_name in ("infino_remember", "infino_forget"):
        return str(args.get("scope") or "").strip().lower() == "personal"
    if tool_name == "memory":
        return str(args.get("target") or "").strip().lower() == "user"
    return False


def withhold(tool_name: str, arguments: Any) -> Any:
    """The arguments with a personal call's text withheld, in the form they
    came in (JSON text or a dict); others unchanged."""
    if not is_personal_call(tool_name, arguments):
        return arguments
    args = dict(_args(arguments))
    for field in TEXT_FIELDS:
        if field in args:
            args[field] = WITHHELD
    if isinstance(args.get("operations"), list):
        args["operations"] = [{k: (WITHHELD if k in TEXT_FIELDS else v) for k, v in op.items()}
                              if isinstance(op, dict) else op for op in args["operations"]]
    return json.dumps(args) if isinstance(arguments, str) else args

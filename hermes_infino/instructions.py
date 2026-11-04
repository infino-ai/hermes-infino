"""The Infino MCP server's instructions, for Hermes's system prompt.

The client's MCP server says how to use its tools - which one for which
question, the tables it reads, the SQL dialect - in the `instructions` of
its handshake, as the MCP spec places them, and Claude Code puts them in
front of the model. Hermes's MCP client keeps the handshake result but
reads only its capabilities, so without this the model gets the bare tool
list. The text is taken from Hermes's own connection to the server, never
restated here, so it is always the running client's.
"""

from __future__ import annotations

import logging

try:  # Hermes's MCP client: its live connections, by server name (`_servers`).
    from tools import mcp_tool as _mcp_tool
except ImportError:  # pragma: no cover - only outside a Hermes install
    _mcp_tool = None

logger = logging.getLogger(__name__)

# The heading the instructions go under in the system prompt.
HEADING = "## Infino tools"


def server_instructions(server_name: str) -> str:
    """The instructions the named MCP server sent on connecting; empty when
    Hermes has no live connection to it (not configured, not started yet,
    or failed)."""
    if _mcp_tool is None:
        return ""
    server = getattr(_mcp_tool, "_servers", {}).get(server_name)
    result = getattr(server, "initialize_result", None)
    text = getattr(result, "instructions", None)
    return text.strip() if isinstance(text, str) else ""


def prompt_block(server_name: str) -> str:
    text = server_instructions(server_name)
    if not text:
        logger.info("infino: no instructions from MCP server %r (is it under mcp_servers and connected?)", server_name)
        return ""
    return f"{HEADING}\n{text}"

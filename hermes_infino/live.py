"""A running turn in Infino as it happens, and a subagent started from it.

Hermes hands its memory provider a turn when the turn ends (`sync_turn`), so
an agent's work in progress was nowhere but in its own process: a long
one-shot turn (a review, a fix) reached `conversations` only when it finished,
and another agent reading it saw nothing until then. These hooks write each
message as it happens instead (`InfinoMemoryProvider.live_message`): the turn's
prompt as it starts (`pre_llm_call`), each model reply (`post_api_request`),
each tool's result (`post_tool_call`). `sync_turn` then writes only what came
after them.

A subagent Hermes starts (`delegate_task`) gets its parent's tools but none of
its conversation: only the brief its parent writes. With the parent's turn in
`conversations` as it runs, the subagent can read all of it, so its first
message says where (`pre_llm_call`, which Hermes hands the parent's session),
and the parent's rows are sent to the cloud as the subagent starts
(`subagent_start`). The subagent's own messages go to `conversations` too,
under its own session: Hermes runs no memory provider for it, so they were
never kept before.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from . import tables
from .provider import live_provider

logger = logging.getLogger(__name__)


def parent_note(parent: str) -> str:
    """What a subagent is told about the session that started it."""
    return (
        f"You were started by another agent, in session {parent}, to do part of its work. Everything it "
        "has done so far is in Infino: its prompt, every message, and every tool call with its result, in "
        "the `conversations` table (full text; a long message is several rows of one `row_id`, numbered "
        "by `chunk`). Read it before you start, with the Infino tools' `sql`:\n"
        "SELECT seq, chunk, role, content, tool_name, tool_args, tool_result FROM conversations "
        f"WHERE session_id = {tables.sql_string(parent)} ORDER BY seq, chunk\n"
        "Work from what it already knows: do not read again what it has read, or run again what it has "
        "run, unless you need to."
    )


def _function(call: Any) -> dict[str, Any]:
    """A tool call's function as a dict, from Hermes's object or a dict."""
    fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
    if isinstance(fn, dict):
        return {"name": fn.get("name"), "arguments": fn.get("arguments")}
    return {"name": getattr(fn, "name", None), "arguments": getattr(fn, "arguments", None)}


def _call_id(call: Any) -> str:
    return str((call.get("id") if isinstance(call, dict) else getattr(call, "id", None)) or "")


def reply_message(assistant: Any) -> Optional[dict[str, Any]]:
    """A model reply as a conversation message: its text and its tool calls."""
    if assistant is None:
        return None
    get = assistant.get if isinstance(assistant, dict) else (lambda k, d=None: getattr(assistant, k, d))
    calls = get("tool_calls") or []
    return {"role": "assistant", "content": get("content") or "",
            "tool_calls": [{"id": _call_id(c), "function": _function(c)} for c in calls]}


def tool_message(kw: dict[str, Any]) -> dict[str, Any]:
    """A tool's result as a conversation message."""
    result = kw.get("result")
    if not isinstance(result, str):
        result = json.dumps(result, default=str)
    return {"role": "tool", "tool_call_id": str(kw.get("tool_call_id") or ""),
            "name": str(kw.get("tool_name") or ""), "content": result}


def on_turn_start(**kw: Any) -> Optional[dict[str, str]]:
    """The turn's prompt, written as it starts; for a subagent, where its
    parent's work is."""
    session = str(kw.get("session_id") or "")
    provider = live_provider()
    if provider is not None:
        try:
            provider.live_message(session, str(kw.get("turn_id") or ""),
                                  {"role": "user", "content": kw.get("user_message")})
        except Exception as e:  # a write never fails the turn
            logger.warning("infino: the turn's prompt was not written as it started: %s", e)
    parent = str(kw.get("parent_session_id") or "")
    if parent and parent != session:
        return {"context": parent_note(parent)}
    return None


def on_reply(**kw: Any) -> None:
    """A model reply, written as it comes."""
    provider = live_provider()
    message = reply_message(kw.get("assistant_message"))
    if provider is None or message is None:
        return
    try:
        provider.live_message(str(kw.get("session_id") or ""), str(kw.get("turn_id") or ""), message)
    except Exception as e:
        logger.warning("infino: a reply was not written as it came: %s", e)


def on_tool_result(**kw: Any) -> None:
    """A tool's result, written as it comes."""
    provider = live_provider()
    if provider is None:
        return
    try:
        provider.live_message(str(kw.get("session_id") or ""), str(kw.get("turn_id") or ""), tool_message(kw))
    except Exception as e:
        logger.warning("infino: a tool result was not written as it came: %s", e)


def on_subagent_start(**kw: Any) -> None:
    """The parent's rows to the cloud now, where the subagent reads them."""
    provider = live_provider()
    if provider is not None:
        provider.push_now()


def register_hooks(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_turn_start)
    ctx.register_hook("post_api_request", on_reply)
    ctx.register_hook("post_tool_call", on_tool_result)
    ctx.register_hook("subagent_start", on_subagent_start)

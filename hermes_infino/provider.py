"""The Infino memory provider for Hermes.

Captures what the team's agents remember, say and do: into the local
database on this machine always, and into the team's cloud database (and the
person's, in a personal profile) once signed in (`backend.py`):

- recall: before each turn, the best `memory` rows for the message;
- memory: Hermes's built-in MEMORY.md / USER.md writes and the
  `infino_remember` tool append to `memory`; `infino_forget` deletes;
- conversations: every message of each completed turn, tool calls and
  results included, to `conversations`;
- skills: `infino_publish_skill` shares one of the profile's skills with the
  team (`skills`).

In a group chat (chats.py) memory is the chat's: what its members ask the
agent to remember is kept for that chat and the team, never in one person's
memory, and recalled in that chat only - unless Hermes says the sender is the
profile's owner writing their own memory.

The agent's own retrieval over all of it - and, signed in, the facts, cards
and joins the platform builds from it - is the Infino client, which Hermes
runs as an MCP server. Hermes's MCP client does not pass that server's
instructions to the model, so this provider puts them in the system prompt.

The provider never blocks Hermes: writes run on background threads, one at a
time, and a failed recall returns nothing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread

from . import backend, chats, config, instructions, recall, skills, tables
from .traces import last_turn_id, note_identity

logger = logging.getLogger(__name__)

PROVIDER_NAME = "infino"
TEAM = recall.TEAM
PERSONAL = recall.PERSONAL
# Seconds a write waits for the tables to exist after startup.
READY_WAIT_S = 30.0
# Longest single conversation cell kept, so one huge message or tool result
# cannot make an append too large.
MAX_CELL_CHARS = 20_000
# Platforms where the person is whoever runs this machine's profile; anywhere
# else (a gateway channel) the turn itself must say who wrote it.
LOCAL_PLATFORMS = ("cli", "tui", "local", "")

SYSTEM_PROMPT = (
    "Team memory is kept in Infino. Relevant memories are added to your context before each turn. "
    "Use infino_remember to save what is worth keeping. Memory is the team's unless it is about one person: "
    "work knowledge (decisions, conventions, fixes, owners, how things are set up) is team memory, saved "
    "without asking; a person's preferences, credentials, schedule or private matters are personal "
    "(scope 'personal'). infino_forget deletes a memory by its exact text. infino_publish_skill shares one of "
    "your skills with the team. Everything the team's agents remembered, said and did is in the team's "
    "Infino database, read with the Infino tools below."
)
# Teammates read every memory, so it names people: "the user" would mean whoever reads it.
SYSTEM_PROMPT_PERSON = (
    "You are working with {author}. Write each memory so a teammate understands it on its own: "
    "name people ({author}, or whoever it is about), never \"the user\", \"I\" or \"me\"."
)
# Hermes's built-in USER.md entries are about the person a profile works with
# ("User prefers ..."); in the team's table they are kept under that person's name.
ABOUT_PERSON = "About {author}: "


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clip(text: Any) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS]


def _text(content: Any) -> str:
    """A message's text: the string, or its text parts joined."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def dependencies_missing() -> list[str]:
    """The local database's libraries this Python cannot import."""
    missing = []
    for module in ("infino", "pyarrow", "numpy", "onnxruntime", "tokenizers", "huggingface_hub"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


class InfinoMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._settings: Optional[config.Settings] = None
        self._backend: Optional[backend.Backend] = None
        self._recaller: Optional[recall.Recaller] = None
        self._ready = threading.Event()
        self._writes_enabled = False
        self._profile = ""
        self._platform = ""
        self._author = ""
        self._session_id = ""
        self._seen: dict[str, int] = {}        # session_id -> messages already appended
        self._writer: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_recall: Optional[RecallStatus] = None

    # -- identity and setup --------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return not dependencies_missing()

    def unavailable_reason(self) -> str:
        return "missing Python packages: " + ", ".join(dependencies_missing())

    def get_config_schema(self) -> list[dict[str, Any]]:
        return config.config_schema()

    def initialize(self, session_id: str, **kwargs) -> None:
        home = kwargs.get("hermes_home")
        self._settings = config.load(Path(home) if home else None)
        self._backend = backend.for_settings(self._settings)
        self._recaller = self._backend.recaller()
        self._session_id = session_id
        self._profile = str(kwargs.get("agent_identity") or "")
        self._platform = str(kwargs.get("platform") or "")
        # Who this profile's turns and memories are from when a turn names nobody:
        # on this machine, the configured name, then Hermes's user, then the
        # profile. The configured name comes first because a CLI or desktop
        # session passes the machine's user id ("uid:1000"), which names no one.
        # On a gateway channel, never the configured name: that is the profile's
        # owner, and a channel message is from whoever sent it.
        local = self._platform in LOCAL_PLATFORMS
        self._author = str((self._settings.author if local else "") or kwargs.get("user_name")
                           or kwargs.get("user_id") or (self._profile if local else ""))
        note_identity(session_id, self._author, self._profile)
        # Subagents, cron and flush runs read memory but never write it.
        self._writes_enabled = kwargs.get("agent_context", "primary") == "primary"
        spawn_context_thread(self._ensure_tables, name="infino-ensure-tables").start()

    def _ensure_tables(self) -> None:
        assert self._backend is not None
        try:
            self._backend.ensure()
        except Exception as e:  # never raise into Hermes
            logger.warning("infino: could not prepare tables: %s", e)
        finally:
            self._ready.set()

    def system_prompt_block(self) -> str:
        if not self._settings:
            return ""
        tools = instructions.prompt_block(self._settings.mcp_server)
        person = " " + SYSTEM_PROMPT_PERSON.format(author=self._author) if self._author else ""
        return SYSTEM_PROMPT + person + ("\n\n" + tools if tools else "")

    # -- recall --------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = None
        if not self._recaller or not query.strip():
            return ""
        block, count = self._recaller.recall(query, chats.current())
        if count:
            self._last_recall = RecallStatus(provider_label="Infino", count=count)
        return block

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_recall

    # -- writes --------------------------------------------------------------

    def _run_write(self, fn, *args) -> None:
        """Run a write on a background thread, one at a time, in order."""
        if not self._writes_enabled or not self._settings:
            return

        def job():
            self._ready.wait(READY_WAIT_S)
            try:
                fn(*args)
            except Exception as e:
                logger.warning("infino: write failed: %s", e)

        with self._lock:
            prev = self._writer

            def chained():
                if prev is not None:
                    prev.join()
                job()

            self._writer = spawn_context_thread(chained, name="infino-write")
            self._writer.start()

    @staticmethod
    def _default_scope() -> str:
        """Memory is the team's unless it is about one person: team memory that
        waits for someone to say "team" stays empty."""
        return TEAM

    def _memory_row(self, content: str, scope: str, origin: str, chat: chats.Chat = chats.NO_CHAT,
                    author: str = "") -> dict:
        return {
            "content": content,
            "scope": scope,
            "author": author or self._author,
            "profile": self._profile,
            "session_id": self._session_id,
            "origin": origin,
            tables.CHAT: chat.key,
            "created_ms": _now_ms(),
        }

    def _append(self, table: str, rows: list[dict]) -> None:
        assert self._backend is not None
        self._backend.append(table, rows)

    def _forget(self, content: str, scope: Optional[str], chat: chats.Chat = chats.NO_CHAT) -> int:
        assert self._backend is not None
        return self._backend.forget_memory(content, scope, chat.visible)

    @staticmethod
    def _sender() -> str:
        """Who wrote the turn running now, as the gateway names them."""
        return chats.session("HERMES_SESSION_USER_NAME") or chats.session("HERMES_SESSION_USER_ID")

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict] = None) -> None:
        assert self._settings is not None
        chat = chats.current()
        if chat.group and not chat.owner_writes:
            # A group member's note (Hermes's own memory tool, or its background
            # review): the chat's and the team's, credited to whoever said it.
            # Hermes's channel memory keeps these out of the owner's files; this
            # keeps them out of the owner's personal memory too.
            sender = self._sender()
            about = ABOUT_PERSON.format(author=sender) if target == "user" and sender else ""
            previous = (metadata or {}).get("previous_content")
            if action in ("replace", "remove") and previous:
                self._run_write(self._forget, about + previous, TEAM, chat)
            if action in ("add", "replace") and content:
                self._run_write(self._append, tables.MEMORY,
                                [self._memory_row(about + content, TEAM, f"builtin:{target}", chat, sender)])
            return
        # Built-in USER.md is about this person: personal (kept on this machine
        # when there is no personal database). MEMORY.md is what the agent learns
        # about the work: the team's.
        scope = PERSONAL if target == "user" else self._default_scope()
        origin = f"builtin:{target}"
        about = ABOUT_PERSON.format(author=self._author) if target == "user" and self._author else ""
        previous = (metadata or {}).get("previous_content")
        if action in ("replace", "remove") and previous:
            self._run_write(self._forget, about + previous, scope)
        if action in ("add", "replace") and content:
            self._run_write(self._append, tables.MEMORY, [self._memory_row(about + content, scope, origin)])

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[list[dict[str, Any]]] = None,
        turn_author: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._writes_enabled or not self._settings:
            return
        session = session_id or self._session_id
        named = (turn_author or {}).get("name")
        if self._platform in LOCAL_PLATFORMS:
            # A CLI turn carries only the machine's user id ("uid:1000"), which
            # the configured name beats.
            author = named or self._settings.author or (turn_author or {}).get("id") or self._author
        else:
            # A channel message is from whoever the gateway says sent it; with
            # nobody named the turn is kept unattributed, never put on the owner.
            author = named or (turn_author or {}).get("id") or ""
        note_identity(session, author, self._profile)
        if messages:
            start = self._seen.get(session, 0)
            new = messages[start:]
            self._seen[session] = len(messages)
        else:
            start, new = 0, [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        turn = last_turn_id(session)
        # A tool message names its call, not always its tool: the call's tool is
        # on the assistant message that made it.
        called = {str(c.get("id")): str((c.get("function") or {}).get("name", ""))
                  for m in (messages or new) for c in (m.get("tool_calls") or [])}
        rows = [self._conversation_row(session, turn, start + i, m, author, called)
                for i, m in enumerate(new) if m.get("role") in ("user", "assistant", "tool")]
        rows = [r for r in rows if r["content"] or r["tool_name"]]
        if rows:
            self._run_write(self._append, tables.CONVERSATIONS, rows)

    def _conversation_row(self, session: str, turn: str, seq: int, m: dict, author: str,
                          called: dict[str, str]) -> dict:
        role = str(m.get("role") or "")
        tool_name = tool_args = tool_result = ""
        content = _text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            calls = [c.get("function") or {} for c in m["tool_calls"]]
            tool_name = ",".join(str(c.get("name", "")) for c in calls)
            tool_args = json.dumps([{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls], default=str)
        elif role == "tool":
            tool_name = str(m.get("name") or m.get("tool_name") or called.get(str(m.get("tool_call_id")), ""))
            tool_result, content = content, ""
        return {
            "session_id": session,
            "turn_id": turn,
            "seq": seq,
            "role": role,
            "content": _clip(content),
            "tool_name": tool_name,
            "tool_args": _clip(tool_args),
            "tool_result": _clip(tool_result),
            "author": author if role == "user" else "",
            "profile": self._profile,
            "platform": self._platform,
            "created_ms": _now_ms(),
        }

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        scope = {"type": "string", "enum": [TEAM, PERSONAL],
                 "description": "team (the default: work knowledge) or personal (about one person)"}
        return [
            {
                "name": "infino_remember",
                "description": "Save a memory to Infino. It is the team's unless it is about one person: "
                               "decisions, conventions, fixes, owners and how things work are team memory; "
                               "a person's preferences, credentials, schedule or private matters take scope "
                               "'personal'. Teammates read it as written: name people, never 'the user', 'I' or 'me'.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "scope": scope},
                    "required": ["content"],
                },
            },
            {
                "name": "infino_forget",
                "description": "Delete a memory from Infino by its exact text (as recalled).",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "scope": scope},
                    "required": ["content"],
                },
            },
            {
                "name": "infino_publish_skill",
                "description": "Share one of your skills with the team: every profile on the team loads it as "
                               "infino:<name> from its next session.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "the skill's name, as skills_list shows it"}},
                    "required": ["name"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._settings:
            return json.dumps({"ok": False, "error": "not configured"})
        if not self._writes_enabled:
            return json.dumps({"ok": False, "error": "writes are off in this agent context"})
        try:
            chat = chats.current()  # read here, on the turn's thread
            self._ready.wait(READY_WAIT_S)
            if tool_name == "infino_remember":
                if chat.group:
                    # In a group chat a memory is the chat's, whatever scope was asked for.
                    sender = self._sender()
                    self._append(tables.MEMORY, [self._memory_row(str(args["content"]), TEAM, "remember", chat,
                                                                  "" if chat.owner_writes else sender)])
                    return json.dumps({"ok": True, "scope": "this group chat (and the team)"})
                scope = args.get("scope") or self._default_scope()
                self._append(tables.MEMORY, [self._memory_row(str(args["content"]), scope, "remember")])
                return json.dumps({"ok": True, "scope": scope})
            if tool_name == "infino_forget":
                scope = None if chat.group else args.get("scope")
                matched = self._forget(str(args["content"]), scope, chat)
                return json.dumps({"ok": True, "scope": scope or "any", "matched": matched})
            if tool_name == "infino_publish_skill":
                name = str(args["name"])
                path = skills.find_local_skill(name, self._settings.home)
                if path is None:
                    return json.dumps({"ok": False, "error": f"no skill named {name} in this profile"})
                self._append(tables.SKILLS, [skills.skill_row(name, path.read_text(), self._author, self._profile)])
                return json.dumps({"ok": True, "published": name})
        except Exception as e:  # a tool reports its failure; it never raises into Hermes
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return json.dumps({"ok": False, "error": f"unknown tool {tool_name}"})

    # -- lifecycle -----------------------------------------------------------

    def _drain(self) -> None:
        with self._lock:
            writer = self._writer
        if writer is not None:
            writer.join(timeout=READY_WAIT_S)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._drain()
        if self._backend is not None:
            self._backend.flush()

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._drain()
        self._session_id = new_session_id

    def shutdown(self) -> None:
        self._drain()
        if self._backend is not None:
            self._backend.flush()

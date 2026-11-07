"""The Infino memory provider for Hermes.

Captures what the team's agents remember, say and do: into the local
database on this machine always, and into the team's cloud database (and the
person's, in a personal profile) once signed in (`backend.py`):

- recall: before each turn, the best `memory` rows for the message;
- memory: Hermes's built-in MEMORY.md / USER.md writes and the
  `infino_remember` tool append to `memory` - the files' entries whoever
  wrote them, Hermes's background review included (builtin.py), an entry
  found by comparing the files kept as the owner's, never the running turn's;
  `infino_forget` deletes, every machine's copy;
- conversations: the messages of each completed turn, tool calls and
  results included, to `conversations` - the person's own, not the team's,
  for a personal session (privacy.py);
- skills: `infino_publish_skill` shares one of the profile's skills with the
  team (`skills`);
- personal search: `infino_personal_search` searches the person's own
  memory, conversations and traces, offered only where the turn is the
  owner's: one of the owner's own sessions (chats.OWNER_PLATFORMS), or a
  profile that is one person's, and never in a group chat or a session
  several parties share. A session that uses it is personal from then on
  (privacy.py), and so is a session that carries its history on (`/branch`,
  compaction into a new session): what it read never reaches the team's
  tables through it.

A turn's messages are the ones from its user message on: Hermes hands
`sync_turn` the whole list, which compaction and `/undo` shorten in place
and a resumed or branched session starts with its history, so a count of
messages already seen would lose or repeat turns. `conversations` is the log
of what was said: a turn `/undo` takes back stays in it. Each session's
`seq` keeps counting up across all of that.

In a group chat (chats.py) memory is the chat's: what its members ask the
agent to remember is kept for that chat and the team, never in one person's
memory, and recalled in that chat only - unless Hermes says the sender is the
profile's owner writing their own memory.

Personal memory is one person's: the profile owner's, in a database of its
own (nothing tells one person's rows from another's inside a database). So
it is read and written in the owner's own sessions - at the terminal, in
Hermes's Desktop chat or an editor, and the owner's scheduled jobs - and on a
gateway only when the profile is one person's (INFINO_PERSONAL_PROFILE). A
shared profile's gateway, which answers whoever writes in, keeps no personal
memory: the agent is told so, and recall there shows none. Neither does a
session several parties share with no gateway chat (a hosted Group Chat
room, a turn another profile forwards), whoever's profile it is.

The agent's own retrieval over all of it - and, signed in, the facts, cards
and joins the platform builds from it - is the Infino client, which Hermes
runs as an MCP server. Hermes's MCP client does not pass that server's
instructions to the model, so this provider puts them in the system prompt.

The provider never blocks Hermes: writes wait only for the local tables, run
on background threads one at a time, and reach the cloud from a background
sync; a recall the cloud is slow to answer is answered from this machine.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread

from . import backend, builtin, chats, config, instructions, privacy, recall, skills, tables
from .traces import note_identity, scrub, turn_id_for, write_buffered

logger = logging.getLogger(__name__)

PROVIDER_NAME = "infino"
TEAM = recall.TEAM
PERSONAL = recall.PERSONAL
# Seconds a write waits for the local tables to exist after startup.
READY_WAIT_S = 30.0
# Longest single conversation cell, or memory, kept, so one huge message or
# tool result cannot make an append too large.
MAX_CELL_CHARS = 20_000
# Seconds between looks at the profile's sign-in, so a running process picks
# up `hermes infino login` (or a key set in config) without a restart.
SETTINGS_RECHECK_S = 30.0
# How much of the user's message finds the turn's start in the message list.
TURN_KEY_CHARS = 200
# How Hermes adds a correction sent while a turn runs to the turn's user text
# it hands `sync_turn` (agent/turn_iteration_prep.py): the text the turn
# started with comes before the first of these.
REDIRECT_MARK = "\n\nUser correction during the turn: "
# Why Hermes switches a session (`on_session_switch`) to one that is another
# conversation's own, carrying none of the session left: `/resume`. A switch
# that resets (`/new`) carries none either; the others (`/branch`, compaction
# into a new session) carry the session's history on.
RESUME_REASON = "resume"
# The run Hermes starts a provider in: a person's own session. The others are
# "subagent", "cron" and "flush".
PRIMARY_CONTEXT = "primary"
# The runs the memory tools write from. The agent calls a tool on purpose, so a
# subagent or a scheduled job saving what it found writes like a session does.
# A flush run is Hermes saving its own memory before compaction: not here.
TOOL_WRITE_CONTEXTS = (PRIMARY_CONTEXT, "subagent", "cron")
# The tool that searches the person's own database.
PERSONAL_SEARCH_TOOL = privacy.PERSONAL_READ_TOOLS[0]
# Rows a personal search returns when the call names no number, and the most it may ask for.
PERSONAL_SEARCH_K = 10
MAX_PERSONAL_SEARCH_K = 50
# Longest cell of a personal search's rows, so one long message or tool result
# cannot crowd out the rest of the answer.
PERSONAL_RESULT_CELL_CHARS = 1_000
# What a personal search answers where the turn is not the owner's own.
NO_PERSONAL_SEARCH_HERE = ("your personal data is not searched here: only in your own sessions, or on a "
                           "profile that is one person's, and never in a group chat")
# Beside the plugin's databases once the profile has answered a gateway: its
# USER.md may then hold entries about anyone who wrote in. It names each such
# platform, one a line; an earlier version left it empty, which counts the same.
GATEWAY_MARK = "served_a_gateway"
# What a shared profile's gateway chat is told when asked to keep something personal.
NO_PERSONAL_HERE = ("nothing was saved: this profile is shared, so it keeps no personal memory for the people "
                    "it talks to here (only its owner's, in the owner's own sessions). Save it as team memory "
                    "only if it is not private.")
# What a session several parties share (chats.MULTI_PARTY_PLATFORMS) is told instead.
NO_PERSONAL_IN_GROUP = ("nothing was saved: several parties share this conversation, so no personal memory is "
                        "kept here (only in its owner's own sessions). Save it as team memory only if it is "
                        "not private.")

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
# On a gateway the prompt is built once and serves everyone who writes in, so it names no one.
SYSTEM_PROMPT_GATEWAY = (
    "Several people may talk to you here. Write each memory so a teammate understands it on its own: "
    "name the person it is about or who said it, never \"the user\", \"I\" or \"me\"."
)
# A shared profile's gateway keeps no personal memory (_keeps_personal).
SYSTEM_PROMPT_NO_PERSONAL = (
    "This profile is shared, so no personal memory is kept here: do not save a person's private matters."
)
# Nor does a session several parties share, whoever's profile it is.
SYSTEM_PROMPT_NO_PERSONAL_GROUP = (
    "Several parties share this conversation, so no personal memory is kept here: do not save a person's "
    "private matters."
)
# Hermes's built-in USER.md entries are about the person a profile works with
# ("User prefers ..."); in the team's table they are kept under that person's name.
ABOUT_PERSON = "About {author}: "
# The start every such entry shares, whoever it is about.
ABOUT_ANYONE = "About "


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clip(text: Any) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS]


def _scope(value: Any) -> Optional[str]:
    """A tool's `scope`: "" when none was given, the scope in its canonical
    spelling, or None when it names neither (a private memory must never land
    in the team's database because the model spelled its scope its own way)."""
    if value is None or value == "":
        return ""
    s = str(value).strip().lower()
    return s if s in (TEAM, PERSONAL) else None


def _started_with(user_content: str) -> str:
    """The user text a turn started with, from the text `sync_turn` is handed:
    without the corrections Hermes added to it while the turn ran."""
    return (user_content or "").split(REDIRECT_MARK, 1)[0]


def _turn_messages(messages: list[dict[str, Any]], user_content: str) -> list[dict[str, Any]]:
    """The messages of the turn that just ended: from its user message (the
    latest that carries the start of the text the turn started with) on. When
    no message carries it, from the latest user message; with no user
    message, all."""
    key = _started_with(user_content).strip()[:TURN_KEY_CHARS]
    latest_user = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "user":
            continue
        if latest_user is None:
            latest_user = i
        if key and key in privacy.message_text(messages[i].get("content")):
            return messages[i:]
    return messages[latest_user:] if latest_user is not None else messages


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
        self._home: Optional[Path] = None
        self._settings: Optional[config.Settings] = None
        self._backend: Optional[backend.Backend] = None
        self._recaller: Optional[recall.Recaller] = None
        self._checked_at = 0.0
        self._ready = threading.Event()        # the local tables exist: writes may go
        self._caught_up = threading.Event()    # the start's cloud (or model) work is done
        self._writes_enabled = False
        self._agent_context = PRIMARY_CONTEXT
        self._profile = ""
        self._platform = ""
        self._author = ""
        self._owner_names: tuple[str, ...] = ()
        self._session_id = ""
        # session_id -> the next `seq` of its messages; a session this process
        # has not stored yet continues from what the local database holds.
        self._next_seq: dict[str, int] = {}
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
        self._home = Path(home) if home else None
        self._bind(config.load(self._home))
        self._session_id = session_id
        self._profile = str(kwargs.get("agent_identity") or "")
        self._platform = str(kwargs.get("platform") or "")
        # Who this profile's turns and memories are from when a turn names nobody:
        # in the owner's own sessions, the configured name, then Hermes's user,
        # then the profile. The configured name comes first because a CLI or
        # desktop session passes the machine's user id ("uid:1000"), which names
        # no one. On a gateway channel, never the configured name: that is the
        # profile's owner, and a channel message is from whoever sent it.
        local = not self._gateway()
        assert self._settings is not None
        if not local:
            self._mark_gateway()
        self._author = str((self._settings.author if local else "") or kwargs.get("user_name")
                           or kwargs.get("user_id") or (self._profile if local else ""))
        # Every name this machine's person writes rows under, now or after a
        # change of settings: the configured name, Hermes's user, the profile
        # (and the one in use, even when it is empty). The personal memory of
        # these is the owner's; a gateway sender's is not.
        self._owner_names = tuple(dict.fromkeys([self._author, *(
            str(n) for n in (self._settings.author, kwargs.get("user_name"), kwargs.get("user_id"), self._profile)
            if n)]))
        note_identity(session_id, self._author, self._profile)
        # Turns and built-in memory are captured in a person's own session only,
        # as Hermes asks of providers; the memory tools follow TOOL_WRITE_CONTEXTS.
        self._agent_context = str(kwargs.get("agent_context") or PRIMARY_CONTEXT)
        self._writes_enabled = self._agent_context == PRIMARY_CONTEXT
        spawn_context_thread(self._ensure_tables, name="infino-ensure-tables").start()
        # What Hermes's background review wrote since this profile last ran.
        self._run_write(self._mirror_builtin)

    def _mark_gateway(self) -> None:
        """Note that this profile answers a gateway, naming its platform: from
        now on its USER.md may be about anyone who wrote in
        (_shared_gateway_profile)."""
        assert self._settings is not None
        mark = config.state_dir(self._settings.home) / GATEWAY_MARK
        try:
            config.make_private_dir(mark.parent)
            named = mark.read_text().split() if mark.exists() else []
            if self._platform not in named:
                with open(mark, "a") as f:
                    f.write(self._platform + "\n")
        except (OSError, ValueError) as e:  # unwritable, or a file that is not text: it still counts
            logger.info("infino: could not note that this profile answers a gateway: %s", e)

    def _bind(self, settings: config.Settings) -> None:
        self._settings = settings
        self._backend = backend.for_settings(settings)
        self._recaller = self._backend.recaller()
        self._checked_at = time.monotonic()

    def _ensure_tables(self) -> None:
        """The local tables first - all a write waits for - then the rest
        (the cloud, or the local model), which can take a while."""
        b = self._backend
        assert b is not None
        try:
            b.ensure_local()
        except Exception as e:  # never raise into Hermes
            logger.warning("infino: could not prepare the local tables: %s", e)
        finally:
            self._ready.set()
        try:
            b.catch_up()
        except Exception as e:
            logger.warning("infino: could not catch up with the cloud: %s", e)
        finally:
            self._caught_up.set()

    def _refresh_settings(self) -> None:
        """Pick up a sign-in made since this process started (at most every
        SETTINGS_RECHECK_S): recall, writes and syncs move to the new backend."""
        if self._settings is None or time.monotonic() - self._checked_at < SETTINGS_RECHECK_S:
            return
        self._checked_at = time.monotonic()
        try:
            new = config.load(self._home)
        except Exception as e:
            logger.info("infino: could not re-read the settings: %s", e)
            return
        old = self._settings
        if (new.api_key, new.team_database, new.personal_database, new.url, new.personal_profile,
                new.personal_jobs) == (old.api_key, old.team_database, old.personal_database, old.url,
                                       old.personal_profile, old.personal_jobs):
            return
        logger.info("infino: the sign-in changed; using %s", new.team_database or "this machine alone")
        self._caught_up.clear()
        self._bind(new)
        spawn_context_thread(self._ensure_tables, name="infino-ensure-tables").start()

    def _gateway(self) -> bool:
        """Whether the session is anywhere but the owner's own surfaces
        (chats.OWNER_PLATFORMS): there, the turn itself says who wrote it."""
        return self._platform not in chats.OWNER_PLATFORMS

    def _multi_party(self) -> bool:
        """Whether several parties share the session with no gateway chat to say
        so (chats.MULTI_PARTY_PLATFORMS): a group for every personal decision,
        on a profile that is one person's too."""
        return self._platform in chats.MULTI_PARTY_PLATFORMS

    @staticmethod
    def _sender() -> str:
        """Who wrote the turn running now, as the gateway names them."""
        return chats.session("HERMES_SESSION_USER_NAME") or chats.session("HERMES_SESSION_USER_ID")

    def _person(self) -> str:
        """Who the running turn is for: the gateway's sender, or this machine's person."""
        return self._sender() if self._gateway() else self._author

    def _owner(self) -> str:
        """The profile's owner, as a row names them when it is not the running
        turn's: the configured name, else the profile (Hermes names one for
        every session), else - in the owner's own sessions - this machine's
        person, so the owner's recall still finds it."""
        named = (self._settings.author if self._settings else "") or self._profile
        return named or ("" if self._gateway() else self._author)

    def _people(self) -> Optional[tuple[str, ...]]:
        """Whose personal memory the running turn may read or change.

        A personal database is one person's, this profile's owner's: nothing
        tells one person's rows from another's inside it. So the owner's own
        sessions read the owner's (the names the owner's rows are written
        under, which keeps out what an earlier version saved for a gateway's
        senders); a gateway reads all of it when the profile is one person's
        (INFINO_PERSONAL_PROFILE), and none of it when the profile is shared,
        where whoever writes in may be anyone, or several parties share the
        session."""
        if not self._gateway():
            return self._owner_names
        if self._multi_party() or not (self._settings and self._settings.personal_profile):
            return ()
        return None

    def _keeps_personal(self, chat: chats.Chat) -> bool:
        """Whether the running turn may write the owner's personal memory: in
        the owner's own sessions; on a gateway, only when the profile is one
        person's, or Hermes says the sender may write the owner's memory (a
        group chat's channel-memory binding); never where several parties
        share the session."""
        if not self._gateway():
            return True
        if self._multi_party():
            return False
        return bool((self._settings and self._settings.personal_profile) or (chat.group and chat.owner_writes))

    def _offers_personal_search(self) -> bool:
        """Whether this session may search the person's own database: a
        person's own session (never a scheduled job's or a subagent's run, whose
        rows may be the team's), on the owner's own surfaces or on a profile
        that is one person's, never where several parties share the session.
        A group chat is refused per call (handle_tool_call)."""
        if not self._settings or self._agent_context != PRIMARY_CONTEXT or self._multi_party():
            return False
        return not self._gateway() or self._settings.personal_profile

    def _shared_gateway_profile(self) -> bool:
        """Whether this profile answers a gateway's people without being one
        person's: Hermes's USER.md then holds entries about any of them."""
        if self._settings and self._settings.personal_profile:
            return False
        return self._gateway() or (config.state_dir(self._settings.home if self._settings else None)
                                   / GATEWAY_MARK).exists()

    def system_prompt_block(self) -> str:
        if not self._settings:
            return ""
        tools = instructions.prompt_block(self._settings.mcp_server)
        if self._gateway():
            person = " " + SYSTEM_PROMPT_GATEWAY
            if self._multi_party():
                person += " " + SYSTEM_PROMPT_NO_PERSONAL_GROUP
            elif not self._settings.personal_profile:
                person += " " + SYSTEM_PROMPT_NO_PERSONAL
        else:
            person = " " + SYSTEM_PROMPT_PERSON.format(author=self._author) if self._author else ""
        return SYSTEM_PROMPT + person + ("\n\n" + tools if tools else "")

    # -- recall --------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall = None
        self._refresh_settings()
        if not self._recaller or not query.strip():
            return ""
        block, count = self._recaller.recall(query, chats.current(), self._people())
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
            "content": _clip(content),
            "scope": scope,
            "author": author,
            "profile": self._profile,
            "session_id": self._session_id,
            "origin": origin,
            tables.CHAT: chat.key,
            "created_ms": _now_ms(),
        }

    def _append(self, table: str, rows: list[dict], personal: bool = False) -> None:
        assert self._backend is not None
        self._backend.append(table, rows, personal)

    def _forget(self, content: str, scope: Optional[str], chat: chats.Chat = chats.NO_CHAT,
                people: Optional[tuple[str, ...]] = None, about_prefix: str = "", everywhere: bool = False) -> int:
        assert self._backend is not None
        return self._backend.forget_memory(content, scope, chat.visible, people, about_prefix, everywhere)

    def _builtin_row(self, target: str, content: str, chat: chats.Chat) -> Optional[dict]:
        """The `memory` row of a built-in memory entry written in the running
        turn (Hermes's own memory tool); None when it is not kept.

        A group member's note is the chat's and the team's, credited to
        whoever said it: Hermes's channel memory keeps these out of the
        owner's files, and this keeps them out of the owner's personal memory
        too. Otherwise USER.md
        is about this person: personal (kept on this machine when there is no
        personal database), and not kept at all where the turn may not write
        the owner's personal memory (`_keeps_personal`); MEMORY.md is what the
        agent learns about the work: the team's."""
        content = content.strip()
        if chat.group and not chat.owner_writes:
            sender = self._sender()
            about = ABOUT_PERSON.format(author=sender) if target == "user" and sender else ""
            return self._memory_row(about + content, TEAM, f"builtin:{target}", chat, sender)
        if target == "user" and not self._keeps_personal(chat):
            return None
        person = self._person()
        scope = PERSONAL if target == "user" else self._default_scope()
        about = ABOUT_PERSON.format(author=person) if target == "user" and person else ""
        return self._memory_row(about + content, scope, f"builtin:{target}", author=person)

    def _add_builtin(self, row: dict) -> None:
        """Keep a built-in memory row unless this machine keeps one like it
        (the file comparison may have added it first)."""
        assert self._backend is not None and self._settings is not None
        with builtin.lock(config.state_dir(self._settings.home)):
            if not self._backend.has_memory(row):
                self._append(tables.MEMORY, [row])

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict] = None) -> None:
        assert self._settings is not None
        chat = chats.current()
        previous = (metadata or {}).get("previous_content")
        if action in ("replace", "remove") and previous:
            if chat.group and not chat.owner_writes:
                # The entry may be about another member, who wrote it: any "About <name>: ".
                self._run_write(self._forget, previous, TEAM, chat, None, ABOUT_ANYONE if target == "user" else "")
            elif target != "user" or self._keeps_personal(chat):
                person = self._person()
                about = ABOUT_PERSON.format(author=person) if target == "user" and person else ""
                scope = PERSONAL if target == "user" else self._default_scope()
                self._run_write(self._forget, about + previous, scope, chats.NO_CHAT, self._people())
        if action in ("add", "replace") and content:
            row = self._builtin_row(target, content, chat)
            if row is not None:
                self._run_write(self._add_builtin, row)

    def _mirrored_row(self, target: str, entry: str) -> dict:
        """The `memory` row of an entry found by comparing a built-in memory
        file with the table. Nothing says who wrote it, nor in which chat:
        Hermes's background review may have written it after any
        conversation, in any chat, and a person may edit the file by hand. So
        it is never the running turn's chat's or sender's: a MEMORY.md entry
        is the team's, in no chat; a USER.md entry the owner's personal
        memory; both credited to the profile's owner."""
        owner = self._owner()
        if target == "user":
            about = ABOUT_PERSON.format(author=owner) if owner else ""
            return self._memory_row(about + entry, PERSONAL, f"builtin:{target}", author=owner)
        return self._memory_row(entry, self._default_scope(), f"builtin:{target}", author=owner)

    def _mirror_builtin(self) -> None:
        """Keep the `memory` table in step with Hermes's built-in memory
        files, whoever wrote them (builtin.py): an entry no row holds is added
        as the owner's (_mirrored_row); a row whose entry left its file is
        deleted, here and in the cloud.

        USER.md is the owner's only where the profile is the owner's: not a
        shared gateway's, whose USER.md may be about anyone who wrote in, and
        not while a group chat's member or several parties take the turn,
        when it is left for the owner's next turn."""
        assert self._backend is not None and self._settings is not None
        home = self._home or config.home()
        owners_turn = not chats.current().group and not self._multi_party()
        with builtin.lock(config.state_dir(self._settings.home)):
            for target in builtin.FILES:
                if target == "user" and (self._shared_gateway_profile() or not owners_turn):
                    continue
                entries = builtin.read_entries(home, target)
                if entries is None:  # missing or unreadable: says nothing
                    continue
                entries = list(dict.fromkeys(_clip(scrub(e)) for e in entries))
                missing, stale = builtin.compare(entries, self._backend.builtin_rows(target))
                for entry in missing:
                    row = self._mirrored_row(target, entry)
                    if not self._backend.has_memory(row):
                        self._append(tables.MEMORY, [row])
                if stale:
                    self._backend.forget_rows(stale)

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
        self._refresh_settings()
        session = session_id or self._session_id
        named = (turn_author or {}).get("name")
        if not self._gateway():
            # A CLI turn carries only the machine's user id ("uid:1000"), which
            # the configured name beats.
            author = named or self._settings.author or (turn_author or {}).get("id") or self._author
            note_identity(session, author, self._profile)
        else:
            # A channel message is from whoever the gateway says sent it; with
            # nobody named the turn is kept unattributed, never put on the owner.
            author = named or (turn_author or {}).get("id") or ""
        if messages:
            new = _turn_messages(messages, user_content)
        else:
            new = [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        start = self._next_seq[session] if session in self._next_seq else self._stored_next_seq(session)
        self._next_seq[session] = start + len(new)
        # The turn's id, as Hermes announced it when the turn started (traces.py).
        turn = turn_id_for(session, _started_with(user_content))
        # A tool message names its call, not always its tool: the call's tool is
        # on the assistant message that made it.
        called = {str(c.get("id")): c.get("function") or {}
                  for m in (messages or new) for c in (m.get("tool_calls") or [])}
        rows = [self._conversation_row(session, turn, start + i, m, author, called)
                for i, m in enumerate(new) if m.get("role") in ("user", "assistant", "tool")]
        rows = [r for r in rows if r["content"] or r["tool_name"]]
        if rows:
            # Decided from the whole session: a personal job's output delivered
            # into the chat comes before the turn that answers it.
            self._ready.wait(READY_WAIT_S)
            personal = self._backend.transcript_personal(session, messages or new)
            self._run_write(self._append, tables.CONVERSATIONS, rows, personal)
        # What Hermes's background review wrote after the last turn.
        self._run_write(self._mirror_builtin)

    def _stored_next_seq(self, session: str) -> int:
        """The next `seq` of a session this process has not stored yet: one
        past the highest this machine kept for it, in the team's tables or
        the person's own (0 for a new session)."""
        assert self._backend is not None
        self._ready.wait(READY_WAIT_S)
        top = None
        for store in (self._backend.local, self._backend.personal):
            try:
                found = store.query_sql(f"SELECT max(seq) AS top FROM {tables.CONVERSATIONS} "
                                        f"WHERE session_id = {tables.sql_string(session)}")
            except Exception as e:  # unreadable: count from what the other holds, or the start
                logger.warning("infino: could not read session %s's stored messages: %s", session, e)
                continue
            if found and found[0]["top"] is not None:
                highest = int(found[0]["top"])
                top = highest if top is None else max(top, highest)
        return 0 if top is None else top + 1

    def _conversation_row(self, session: str, turn: str, seq: int, m: dict, author: str,
                          called: dict[str, dict]) -> dict:
        role = str(m.get("role") or "")
        tool_name = tool_args = tool_result = ""
        content = privacy.message_text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            calls = [c.get("function") or {} for c in m["tool_calls"]]
            tool_name = ",".join(str(c.get("name", "")) for c in calls)
            tool_args = json.dumps([{"name": c.get("name"),
                                     "arguments": privacy.withhold(str(c.get("name", "")), c.get("arguments"))}
                                    for c in calls], default=str)
        elif role == "tool":
            call = called.get(str(m.get("tool_call_id")), {})
            tool_name = str(m.get("name") or m.get("tool_name") or call.get("name", ""))
            personal = privacy.is_personal_call(tool_name, call.get("arguments"))
            tool_result, content = (privacy.WITHHELD if personal else content), ""
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
        # Hermes routes a provider's tools by the schemas it gives when the
        # provider is added, before `initialize`, and shows the model the ones it
        # gives after. So the personal search is always routed, and shown only
        # where it is offered; a call from anywhere else is refused (_personal_search).
        shown = self._settings is None or self._offers_personal_search()
        personal_search = [{
            "name": PERSONAL_SEARCH_TOOL,
            "description": "Search your own personal data, which the Infino tools do not: your personal memory, "
                           "and the conversations and traces of your personal work (personal scheduled jobs, the "
                           "chats their output went to, a personal profile). Keyword search, best rows first, "
                           "with when each was written. Once you use it, this session's conversation and traces "
                           "are kept personal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the words to search for"},
                    "table": {"type": "string", "enum": list(tables.PERSONAL),
                              "description": f"which of your tables to search (default {tables.CONVERSATIONS})"},
                    "k": {"type": "integer", "description": f"rows to return (default {PERSONAL_SEARCH_K}, "
                                                            f"at most {MAX_PERSONAL_SEARCH_K})"},
                },
                "required": ["query"],
            },
        }] if shown else []
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
                "description": "Delete a memory from Infino by its exact text (as recalled), whichever "
                               "teammate's agent saved it.",
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
        ] + personal_search

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._settings:
            return json.dumps({"ok": False, "error": "not configured"})
        if self._agent_context not in TOOL_WRITE_CONTEXTS:
            return json.dumps({"ok": False, "error": f"Infino's memory tools do not write from a Hermes "
                                                     f"'{self._agent_context}' run; they write from a session, "
                                                     "a subagent or a cron job"})
        try:
            chat = chats.current()  # read here, on the turn's thread
            if tool_name in ("infino_remember", "infino_forget"):
                asked = _scope(args.get("scope"))
                if asked is None:
                    return json.dumps({"ok": False, "error": f"scope must be '{TEAM}' or '{PERSONAL}', "
                                                             f"not {args.get('scope')!r}"})
            self._ready.wait(READY_WAIT_S)
            if tool_name == "infino_remember":
                content = str(args["content"])
                if chat.group:
                    # In a group chat a memory is the chat's, whatever scope was asked for.
                    sender = self._sender()
                    self._append(tables.MEMORY, [self._memory_row(content, TEAM, "remember", chat,
                                                                  self._author if chat.owner_writes else sender)])
                    return json.dumps({"ok": True, "scope": "this group chat (and the team)"})
                scope = asked or self._default_scope()
                if scope == PERSONAL and not self._keeps_personal(chat):
                    return json.dumps({"ok": False,
                                       "error": NO_PERSONAL_IN_GROUP if self._multi_party() else NO_PERSONAL_HERE})
                self._append(tables.MEMORY, [self._memory_row(content, scope, "remember", author=self._person())])
                return json.dumps({"ok": True, "scope": scope})
            if tool_name == "infino_forget":
                # Every machine's copy: a memory recalled from a teammate's is forgotten too.
                scope = None if chat.group else (asked or None)
                matched = self._forget(str(args["content"]), scope, chat, self._people(), "", everywhere=True)
                return json.dumps({"ok": True, "scope": scope or "any", "matched": matched})
            if tool_name == "infino_publish_skill":
                if chat.group and not chat.owner_writes:
                    # Every profile on the team loads a published skill: publishing is
                    # the profile owner's, as writing the owner's memory is.
                    return json.dumps({"ok": False, "error": "a group chat's members do not publish this "
                                                             "profile's skills to the team; its owner does"})
                name = str(args["name"])
                path = skills.find_local_skill(name, self._settings.home)
                if path is None:
                    return json.dumps({"ok": False, "error": f"no skill named {name} in this profile"})
                # Scrubbed of secrets as everything else leaving this machine is.
                text = scrub(path.read_text())
                if len(text) > skills.MAX_SKILL_CHARS:
                    return json.dumps({"ok": False, "error": f"{name}'s SKILL.md is {len(text)} characters; "
                                                             f"the most a team skill may be is {skills.MAX_SKILL_CHARS}"})
                self._append(tables.SKILLS, [skills.skill_row(name, text, self._person(), self._profile)])
                return json.dumps({"ok": True, "published": name})
            if tool_name == PERSONAL_SEARCH_TOOL:
                return self._personal_search(args, chat)
        except Exception as e:  # a tool reports its failure; it never raises into Hermes
            return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return json.dumps({"ok": False, "error": f"unknown tool {tool_name}"})

    def _personal_search(self, args: dict[str, Any], chat: chats.Chat) -> str:
        """`infino_personal_search`: the best rows of one of the person's own
        tables. Refused where the turn is not the owner's own; a session that
        searches is personal from then on, so nothing it read is written to
        the team's tables."""
        assert self._backend is not None
        if not self._offers_personal_search() or chat.group:
            return json.dumps({"ok": False, "error": NO_PERSONAL_SEARCH_HERE})
        query = str(args.get("query") or "").strip()
        if not query:
            return json.dumps({"ok": False, "error": "query is empty: give the words to search for"})
        table = str(args.get("table") or tables.CONVERSATIONS)
        if table not in tables.PERSONAL:
            return json.dumps({"ok": False, "error": f"table must be one of {', '.join(tables.PERSONAL)}, "
                                                     f"not {table!r}"})
        asked_k = args.get("k")
        k = PERSONAL_SEARCH_K if asked_k is None or asked_k == "" else int(asked_k)
        k = max(1, min(k, MAX_PERSONAL_SEARCH_K))
        # Before the search, so a search that fails partway still keeps the session personal.
        privacy.note_personal(self._session_id)
        rows = self._backend.search_personal(table, query, k)
        people = self._people()
        if table == tables.MEMORY and people is not None:
            # As recall shows it: the owner's own, not what an earlier version
            # kept for a gateway's senders.
            rows = [r for r in rows if r.get("author") in people]
        return json.dumps({"ok": True, "table": table, "rows": [
            {c: (v[:PERSONAL_RESULT_CELL_CHARS] if isinstance(v, str) else v)
             for c, v in r.items() if v not in (None, "")} for r in rows]}, default=str)

    # -- lifecycle -----------------------------------------------------------

    def _drain(self) -> None:
        with self._lock:
            writer = self._writer
        if writer is not None:
            writer.join(timeout=READY_WAIT_S)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Hermes calls this inline (at compaction, /new) as well as at exit:
        it only starts a sync. `shutdown`, at exit, waits for the rows to go."""
        if self._backend is not None:
            self._backend.schedule_sync()

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          **kwargs) -> None:
        """Hermes moved the agent to another session id. Writes carry the
        session they were made in; nothing to wait for. A new session's trace
        rows are this profile's person's, as its first session's were. A
        session that carries its parent's history on (`/branch`, compaction
        into a new session) holds whatever the parent read, so it is as
        personal as the parent; `/resume` and `/new` (a reset) carry nothing
        of the session left, which Hermes names as their parent too."""
        previous, self._session_id = self._session_id, new_session_id
        if new_session_id == previous:  # compacted in place, or /undo: the same session
            return
        note_identity(new_session_id, self._author, self._profile)
        carried_on = not reset and kwargs.get("reason") != RESUME_REASON
        if parent_session_id and parent_session_id != new_session_id and carried_on and self._backend is not None:
            self._ready.wait(READY_WAIT_S)
            if self._backend.transcript_personal(parent_session_id):
                privacy.note_personal(new_session_id)

    def shutdown(self) -> None:
        # Trace rows a fork of a session (Hermes's background review, /btw)
        # added after the session's last turn ended have no turn end to wait for.
        if self._settings is not None:
            write_buffered(self._settings.home)
        self._drain()
        if self._backend is not None:
            self._backend.flush()

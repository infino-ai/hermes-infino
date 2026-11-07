"""Which chat a turn comes from, and whether it is a group chat.

A group chat's memory is its own, as Hermes keeps it (channel memory, #144822):
what a group's members ask the agent to remember is kept for that chat and the
team, never in one person's memory, and a group's notes are recalled in that
chat only. This module says which chat that is.

The chat is read from the turn's gateway session (`gateway.session_context`,
task-local per turn): its platform, chat, thread, parent chat and workspace.
The key is built the way Hermes keys a chat's channel memory: a thread shares
its parent chat's notes, and a platform that reports a workspace (a Slack team)
prefixes it. Whether the chat is a group is Hermes's to decide:
- with channel memory in Hermes, its binding for the turn (`CHANNEL_MEMORY`)
  decides, including whether this sender may write the owner's own memory from
  the group (a sender the gateway would also admit in a DM);
- without it, the chat type decides, as Hermes itself then does: anything but
  a DM is a group. A chat of no known type (blank) may be either, and is a
  group, as Hermes's own access check settles it.

The owner's own surfaces (OWNER_PLATFORMS) have no gateway chat: a session
there is never a group. Hermes's hosted Group Chat rooms and turns another
profile forwards (MULTI_PARTY_PLATFORMS) have none either, though several
parties share them: the provider treats them as a group from their platform.

A webhook delivery (Hermes's webhook adapter) is a turn with no person behind
it: an event from GitHub, an email, an alert. Hermes gives each delivery a
chat of its own, which no later turn shares, so a route's turns are keyed by
the route instead: what one delivery's turn keeps is recalled in the route's
later deliveries. Its payload is outside content (a stranger's pull request
or email), so a route is a group for personal memory, unless the profile's
owner calls the route their own (`owner_routes`: INFINO_OWNER_ROUTES, on a
profile that is one person's): its turns are then the owner's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # Hermes's per-turn gateway session values.
    from gateway.session_context import get_session_env as _get_session_env
except ImportError:  # pragma: no cover - outside Hermes, or a Hermes without them
    _get_session_env = None

try:  # Hermes's channel memory binding (when Hermes has channel memory).
    from tools.memory_tool_channel import CHANNEL_MEMORY as _CHANNEL_MEMORY
except ImportError:
    _CHANNEL_MEMORY = None

try:  # The chat types Hermes itself treats as a one-to-one chat. The set lists
    # "" too, but Hermes reads a blank type as ambiguous - a relay frame sends ""
    # or null for it, and a restored session keeps a stored empty one - and its
    # access check settles that as a group (gateway/slash_access.py): so does `current`.
    from gateway.slash_access import _DM_CHAT_TYPES as DM_TYPES
except ImportError:  # the same set, for a Hermes that does not name it
    DM_TYPES = frozenset({"dm", "direct", "private", ""})
# The owner's own surfaces, the one place they are named: this machine's person
# at a terminal ("cli", "tui", "local", or none named), in Hermes's Desktop chat
# panel ("desktop") or an editor ("acp"), and the owner's scheduled jobs
# ("cron"). Never a gateway chat: on any other platform the turn itself says
# who wrote it.
OWNER_PLATFORMS = frozenset({"", "local", "cli", "tui", "desktop", "acp", "cron"})
# Where several parties share one conversation with no gateway chat to say so:
# Hermes's hosted Group Chat rooms ("bot_room") and turns another profile
# forwards (A2A, "a2a"). A group for every personal decision, on a profile that
# is one person's too.
MULTI_PARTY_PLATFORMS = frozenset({"bot_room", "a2a"})
# Hermes's webhook adapter, and how it names the route a delivery came in on:
# the turn's user id is `webhook:<route>` (gateway/platforms/webhook.py,
# `_admission_source`), while its chat id is the delivery's own.
WEBHOOK_PLATFORM = "webhook"
WEBHOOK_USER_PREFIX = "webhook:"


@dataclass(frozen=True)
class Chat:
    key: str = ""             # the chat's key ("" outside a group chat)
    group: bool = False       # a group chat: its memory is the chat's and the team's
    owner_writes: bool = True  # the sender may write the profile owner's own memory

    @property
    def visible(self) -> tuple[str, ...]:
        """The `chat` values a recall here may show: the team's ("") and this chat's."""
        return ("", self.key) if self.group else ("",)


NO_CHAT = Chat()


def session(name: str) -> str:
    """One of the turn's gateway session values ("" outside a gateway turn)."""
    if _get_session_env is None:
        return ""
    try:
        return str(_get_session_env(name, "") or "")
    except Exception:  # a session read never fails a turn
        return ""


def chat_key(platform: str, chat_id: str, thread_id: str = "", parent_chat_id: str = "", workspace: str = "") -> str:
    """A chat's key: `<platform>:<workspace>_<chat>`, a thread keyed by its parent chat."""
    chat = (parent_chat_id if thread_id and parent_chat_id else "") or chat_id
    if not chat:
        return ""
    return f"{platform}:{workspace}_{chat}" if workspace else f"{platform}:{chat}"


def webhook_route() -> str:
    """The webhook route the running turn came in on; "" for any other turn."""
    if session("HERMES_SESSION_PLATFORM").lower() != WEBHOOK_PLATFORM:
        return ""
    user = session("HERMES_SESSION_USER_ID")
    return user[len(WEBHOOK_USER_PREFIX):] if user.startswith(WEBHOOK_USER_PREFIX) else ""


def current(owner_routes: frozenset[str] = frozenset()) -> Chat:
    """The chat of the turn running now (call it on the turn's own thread).
    `owner_routes`: the webhook routes whose turns are the owner's own."""
    platform = session("HERMES_SESSION_PLATFORM").lower()
    if platform in OWNER_PLATFORMS:
        return NO_CHAT
    route = webhook_route()
    if route:
        if route in owner_routes:
            return NO_CHAT
        return Chat(key=chat_key(WEBHOOK_PLATFORM, route), group=True, owner_writes=False)
    key = chat_key(platform, session("HERMES_SESSION_CHAT_ID"), session("HERMES_SESSION_THREAD_ID"),
                   session("HERMES_SESSION_PARENT_CHAT_ID"), session("HERMES_SESSION_SCOPE_ID"))
    if not key:
        return NO_CHAT
    if _CHANNEL_MEMORY is not None:
        # Hermes binds every gateway turn: None means a confirmed one-to-one chat
        # (or channel memory turned off, which Hermes then treats the same way).
        binding = _binding()
        if binding is None:
            return NO_CHAT
        return Chat(key=key, group=True, owner_writes=bool(getattr(binding, "owner_writes", False)))
    chat_type = session("HERMES_SESSION_CHAT_TYPE").strip().lower()
    if chat_type and chat_type in DM_TYPES:  # a blank type may be a group's: a group
        return NO_CHAT
    # Without channel memory Hermes cannot say whether the sender may write the
    # owner's memory from a group, so no one does from here.
    return Chat(key=key, group=True, owner_writes=False)


def _binding() -> Optional[object]:
    try:
        return _CHANNEL_MEMORY.get(None)
    except Exception:
        return None

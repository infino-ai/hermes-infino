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
  a DM is a group.

A CLI, TUI or desktop session has no gateway chat: it is never a group.
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

try:  # The chat types Hermes itself treats as a one-to-one chat ("" included).
    from gateway.slash_access import _DM_CHAT_TYPES as DM_TYPES
except ImportError:  # the same set, for a Hermes that does not name it
    DM_TYPES = frozenset({"dm", "direct", "private", ""})
# Platforms whose sessions are this machine's own, never a gateway chat.
LOCAL_PLATFORMS = frozenset({"", "local", "cli", "tui"})


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


def current() -> Chat:
    """The chat of the turn running now (call it on the turn's own thread)."""
    platform = session("HERMES_SESSION_PLATFORM").lower()
    if platform in LOCAL_PLATFORMS:
        return NO_CHAT
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
    if session("HERMES_SESSION_CHAT_TYPE").lower() in DM_TYPES:
        return NO_CHAT
    # Without channel memory Hermes cannot say whether the sender may write the
    # owner's memory from a group, so no one does from here.
    return Chat(key=key, group=True, owner_writes=False)


def _binding() -> Optional[object]:
    try:
        return _CHANNEL_MEMORY.get(None)
    except Exception:
        return None

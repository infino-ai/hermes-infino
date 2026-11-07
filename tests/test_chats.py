"""Which chat a turn is from, and whether it is a group."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.slash_access import _DM_CHAT_TYPES

from hermes_infino import chats

from .helpers import DM, GROUP, in_session, webhook_delivery

# Two webhook routes, as `hermes webhook subscribe` names them.
ROUTE = "ci-failure"
OTHER_ROUTE = "email-triage"


@pytest.mark.parametrize("args, key", [
    (("telegram", ""), ""),
    (("telegram", "", "t1", "C1"), "telegram:C1"),
    (("discord", "thread-7", "thread-7", ""), "discord:thread-7"),
    (("discord", "chan-3", "", "parent-9"), "discord:chan-3"),
    (("slack", "C1", "", "", "T9"), "slack:T9_C1"),
])
def test_chat_keys(args, key):
    assert chats.chat_key(*args) == key


def test_outside_a_gateway_there_is_no_chat(monkeypatch):
    monkeypatch.setattr(chats, "_get_session_env", None)
    assert chats.session("HERMES_SESSION_PLATFORM") == ""
    assert chats.current() == chats.NO_CHAT


@pytest.mark.parametrize("platform", ["", "cli", "TUI", "local", "desktop", "acp", "cron"])
def test_the_owners_own_sessions_are_never_a_group(monkeypatch, platform):
    in_session(monkeypatch, {**GROUP, "HERMES_SESSION_PLATFORM": platform})
    assert chats.current() == chats.NO_CHAT


@pytest.mark.parametrize("chat_type", ["dm", "private", "DM", "direct", " dm "])
def test_a_one_to_one_chat_is_not_a_group(monkeypatch, chat_type):
    in_session(monkeypatch, {**DM, "HERMES_SESSION_CHAT_TYPE": chat_type})
    assert chats.current() == chats.NO_CHAT


@pytest.mark.parametrize("chat_type", ["", "  "])
def test_a_chat_of_no_known_type_is_a_group(monkeypatch, chat_type):
    # Hermes's own access check reads a blank type as ambiguous (a relay frame's "" or null, a
    # restored session's stored empty value) and settles it as a group: the safe side here too.
    in_session(monkeypatch, {**DM, "HERMES_SESSION_CHAT_TYPE": chat_type})
    chat = chats.current()
    assert chat == chats.Chat(key="telegram:111", group=True, owner_writes=False, untyped=True)
    # It may be the owner's own one-to-one chat: a personal memory asked for there is refused, not the chat's.
    assert chat.refuses_personal


def test_a_dm_is_what_hermes_calls_a_dm():
    assert chats.DM_TYPES == _DM_CHAT_TYPES


def test_without_channel_memory_a_group_writes_nobodys_own_memory(monkeypatch):
    in_session(monkeypatch, GROUP)
    chat = chats.current()
    assert chat == chats.Chat(key="telegram:-100", group=True, owner_writes=False)
    assert chat.visible == ("", "telegram:-100")
    assert not chat.refuses_personal  # its members' memory is the chat's


def test_a_chat_with_no_id_is_no_chat(monkeypatch):
    in_session(monkeypatch, {**GROUP, "HERMES_SESSION_CHAT_ID": ""})
    assert chats.current() == chats.NO_CHAT


def test_channel_memory_binding_decides(monkeypatch):
    in_session(monkeypatch, DM, binding=SimpleNamespace(owner_writes=True))
    assert chats.current() == chats.Chat(key="telegram:111", group=True, owner_writes=True)
    in_session(monkeypatch, GROUP, binding=None)
    assert chats.current() == chats.NO_CHAT


def test_a_binding_that_cannot_be_read_counts_as_none(monkeypatch):
    in_session(monkeypatch, GROUP, binding=None)

    def broken(default=None):
        raise LookupError("no binding in this context")

    monkeypatch.setattr(chats, "_CHANNEL_MEMORY", SimpleNamespace(get=broken))
    assert chats.current() == chats.NO_CHAT


def test_a_session_read_that_fails_is_empty(monkeypatch):
    def broken(name, default=""):
        raise RuntimeError("context gone")

    monkeypatch.setattr(chats, "_get_session_env", broken)
    assert chats.session("HERMES_SESSION_CHAT_ID") == ""


def test_outside_a_group_only_the_teams_memory_is_visible():
    assert chats.NO_CHAT.visible == ("",)


# -- webhook turns -----------------------------------------------------------------

def test_a_webhook_turn_is_its_routes_chat_whatever_the_delivery():
    # Hermes gives each delivery its own chat id; the notes a route's turns keep must outlive it.
    with webhook_delivery(ROUTE, "delivery-1"):
        first = chats.current()
    with webhook_delivery(ROUTE, "delivery-2"):
        second = chats.current()
    with webhook_delivery(OTHER_ROUTE, "delivery-3"):
        other = chats.current()
    assert first == second == chats.Chat(key=f"webhook:{ROUTE}", group=True, owner_writes=False, route=True)
    assert other == chats.Chat(key=f"webhook:{OTHER_ROUTE}", group=True, owner_writes=False, route=True)
    assert first.refuses_personal  # outside content, from no member


def test_a_route_the_owner_calls_their_own_is_no_group_chat():
    with webhook_delivery(ROUTE, "delivery-1"):
        assert chats.current(owner_routes=frozenset({ROUTE})) == chats.NO_CHAT
        assert chats.current(owner_routes=frozenset({OTHER_ROUTE})).group

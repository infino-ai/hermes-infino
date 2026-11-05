"""The Infino tools' instructions, taken from Hermes's live connection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_infino import instructions


def connection(text):
    return SimpleNamespace(initialize_result=SimpleNamespace(instructions=text))


@pytest.fixture
def servers(monkeypatch):
    """Hermes's live connections, keyed as given, with no profile-scope resolver."""
    def use(table: dict):
        monkeypatch.setattr(instructions, "_mcp_tool", SimpleNamespace(_servers=table))
        monkeypatch.setattr(instructions, "_mcp_scope", None)
    return use


def test_without_hermes_mcp_client_there_are_none(monkeypatch):
    monkeypatch.setattr(instructions, "_mcp_tool", None)
    assert instructions.server_instructions("infino") == ""
    assert instructions.prompt_block("infino") == ""


def test_the_text_is_trimmed_and_put_under_its_heading(servers):
    servers({"infino": connection("\n  Begin with ask.  \n")})
    assert instructions.prompt_block("infino") == f"{instructions.HEADING}\nBegin with ask."


@pytest.mark.parametrize("server", [
    connection(None),
    connection(42),
    SimpleNamespace(initialize_result=None),
    SimpleNamespace(),
])
def test_a_connection_without_text_instructions_gives_none(servers, server):
    servers({"infino": server})
    assert instructions.server_instructions("infino") == ""


def test_another_servers_instructions_are_never_used(servers):
    servers({"github": connection("Use gh."), ("profile-a", "github"): connection("Use gh.")})
    assert instructions.server_instructions("infino") == ""


def test_the_configured_server_name_is_the_one_read(servers):
    servers({"team-memory": connection("Begin with ask."), "infino": connection("Not this one.")})
    assert instructions.server_instructions("team-memory") == "Begin with ask."


def test_a_resolver_that_fails_falls_back_to_the_name(monkeypatch):
    def broken(name):
        raise RuntimeError("no scope here")

    monkeypatch.setattr(instructions, "_mcp_tool", SimpleNamespace(_servers={"infino": connection("Begin with ask.")}))
    monkeypatch.setattr(instructions, "_mcp_scope", SimpleNamespace(_resolve_server_key=broken))
    assert instructions.server_instructions("infino") == "Begin with ask."


def test_the_resolvers_key_wins(monkeypatch):
    table = {("profile-a", "infino"): connection("Profile A's."), ("profile-b", "infino"): connection("Profile B's.")}
    monkeypatch.setattr(instructions, "_mcp_tool", SimpleNamespace(_servers=table))
    monkeypatch.setattr(instructions, "_mcp_scope", SimpleNamespace(_resolve_server_key=lambda n: ("profile-b", n)))
    assert instructions.server_instructions("infino") == "Profile B's."

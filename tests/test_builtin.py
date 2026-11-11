"""Hermes's built-in memory files kept in step with the `memory` table,
whoever writes them: the agent's memory tool, Hermes's background review (which
runs with every memory provider switched off), or a person by hand."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager

import pytest

from hermes_infino import builtin, chats, config, provider as provider_mod, tables, traces
from hermes_infino.provider import GATEWAY_MARK

from .helpers import (DM, GROUP, OTHER_GROUP, PERSONAL_DB, TEAM_DB, agent_tool_call, end_session, hermes_dispatch,
                      current, in_session, local_rows, make_manager, personal_rows, superseded, wait_for)

# A credential shaped the way Hermes's redaction recognizes one.
SECRET = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 4
# What Hermes's background review writes after a group's turn: the group's business.
GROUP_NOTE = "Acme deal is $2M, keep it from eng"
# The keys of GROUP's and OTHER_GROUP's chats, as the provider keeps a group's memory.
GROUP_CHAT = chats.chat_key(GROUP["HERMES_SESSION_PLATFORM"], GROUP["HERMES_SESSION_CHAT_ID"])
OTHER_CHAT = chats.chat_key(OTHER_GROUP["HERMES_SESSION_PLATFORM"], OTHER_GROUP["HERMES_SESSION_CHAT_ID"])
# A group chat whose key begins as GROUP's does, and which a note kept for GROUP is not for.
PREFIX_GROUP = {**GROUP, "HERMES_SESSION_CHAT_ID": GROUP["HERMES_SESSION_CHAT_ID"][:-1]}
# The author Hermes names for a turn another agent wrote into a profile's Bot Chat.
PEER_AGENT = {"id": "bot:spark/researcher", "name": "researcher", "is_bot": True}
# Long after any turn: a background review that waited for the machine to be
# idle (agent/review_idle_queue.py), or a comparison that runs a day later.
MUCH_LATER_MS = 24 * 60 * 60 * 1000
# A chat of no known type: a relay's frame for the owner's own DM may carry none.
UNTYPED = {**DM, "HERMES_SESSION_CHAT_TYPE": ""}
# How long these tests let a note's reader or writer wait for another process holding it.
QUICK_LOCK_WAIT_S = 0.2


def write_file(home, target: str, entries: list[str]) -> None:
    """A memory file as Hermes writes one, or a person edits it by hand."""
    path = home / builtin.MEMORIES_DIR / builtin.FILES[target]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(builtin.ENTRY_DELIMITER.join(entries), encoding="utf-8")


def a_review_saves(session_id: str, target: str, entry: str) -> dict:
    """Hermes's background review, following a turn of `session_id`, saves an
    entry: its fork's memory tool call, under that session's id, through
    Hermes's own tool dispatch (hermes_dispatch first)."""
    return agent_tool_call(session_id, "memory", {"action": "add", "target": target, "content": entry})


def a_group_turn(tmp_path, monkeypatch, values: dict, session_id: str):
    """A gateway chat's agent, as Hermes builds one, and its turn."""
    in_session(monkeypatch, values)
    mgr, bot = make_manager(tmp_path, platform="telegram", session_id=session_id)
    mgr.on_turn_start(1, "How big is the Acme deal?")
    turn(bot)
    return mgr, bot


def turn(provider, n: int = 1) -> None:
    """A turn ends (the file comparison runs after it), and its writes land."""
    provider.sync_turn(f"q{n}", f"a{n}", session_id="s1")
    provider._drain()


def contents(rows) -> list[str]:
    return sorted(r["content"] for r in rows)


def test_entries_are_read_as_hermes_parses_them(tmp_path):
    write_file(tmp_path, "memory", ["  use pnpm  ", "", "deploys are on Tuesdays", "use pnpm"])
    assert builtin.read_entries(tmp_path, "memory") == ["use pnpm", "deploys are on Tuesdays"]
    assert builtin.read_entries(tmp_path, "user") is None  # missing: says nothing


def test_a_row_holds_its_entry_with_or_without_an_about_prefix():
    entries = {"prefers short answers", "works from Lisbon: mostly"}
    assert builtin.entry_of("prefers short answers", entries) == "prefers short answers"
    assert builtin.entry_of("About dana: prefers short answers", entries) == "prefers short answers"
    assert builtin.entry_of("About dana: works from Lisbon: mostly", entries) == "works from Lisbon: mostly"
    assert builtin.entry_of("About dana: gone", entries) is None


def test_the_background_reviews_writes_reach_the_memory_table(env, platform, tmp_path):
    # The review writes the files with every memory provider off: no on_memory_write.
    env.setenv("INFINO_AUTHOR", "dana")
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", ["deploys are on Tuesdays"])
    write_file(tmp_path, "user", ["prefers short answers"])
    turn(provider)
    end_session(provider)
    team = [r for r in platform.rows(TEAM_DB, tables.MEMORY)]
    assert [(r["content"], r["origin"], r["scope"]) for r in team] == [
        ("deploys are on Tuesdays", "builtin:memory", "team")]
    assert [(r["content"], r["author"]) for r in personal_rows(tmp_path)] == [
        ("About dana: prefers short answers", "dana")]


def test_an_entry_the_review_removes_is_kept_superseded_here_and_in_the_cloud(env, platform, tmp_path):
    # Hermes's review compacting the file drops an entry; the table keeps it,
    # marked superseded with nothing named as its replacement, beside its row's id.
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "use pnpm"])
    turn(provider, 1)
    end_session(provider)
    [original] = [r for r in local_rows(tmp_path, tables.MEMORY) if r["content"] == "deploys are on Tuesdays"]
    write_file(tmp_path, "memory", ["use pnpm"])
    turn(provider, 2)
    end_session(provider)
    for rows in (local_rows(tmp_path, tables.MEMORY), platform.rows(TEAM_DB, tables.MEMORY)):
        assert contents(current(rows)) == ["use pnpm"]
        [kept] = superseded(rows)
        assert kept["content"] == "deploys are on Tuesdays" and kept["origin"] == "builtin:memory"
        assert kept[tables.SUPERSEDED_BY] == "" and kept[tables.PRIOR_ROW_ID] == original["row_id"]
        assert kept["row_id"] != original["row_id"] and kept["created_ms"] == original["created_ms"]


def test_a_memory_tool_write_and_the_comparison_keep_one_row(env, platform, tmp_path):
    # Hermes writes the file, then tells the provider: both must not add it.
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", ["use pnpm"])
    provider.on_memory_write("add", "memory", "use pnpm")
    turn(provider)
    provider.on_memory_write("add", "memory", "use pnpm")
    turn(provider, 2)
    end_session(provider)
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["use pnpm"]


def test_a_missing_or_unreadable_file_deletes_nothing(env, platform, tmp_path):
    mgr, provider = make_manager(tmp_path)
    provider.on_memory_write("add", "memory", "use pnpm")  # no file: memory disabled, say
    turn(provider)
    (tmp_path / builtin.MEMORIES_DIR).mkdir(parents=True, exist_ok=True)
    (tmp_path / builtin.MEMORIES_DIR / "MEMORY.md").write_bytes(b"\xff\xfe not text")
    turn(provider, 2)
    end_session(provider)
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["use pnpm"]


def test_a_secret_in_a_memory_file_never_leaves_this_machine(env, platform, tmp_path):
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", [f"the deploy key is {SECRET}"])
    turn(provider)
    turn(provider, 2)  # compared again: the scrubbed row still holds the entry
    end_session(provider)
    [row] = platform.rows(TEAM_DB, tables.MEMORY)
    assert SECRET not in row["content"] and row["content"].startswith("the deploy key is")


def test_what_the_review_wrote_before_a_start_is_caught_up_at_the_start(env, platform, tmp_path):
    write_file(tmp_path, "memory", ["written while no session ran"])
    mgr, provider = make_manager(tmp_path)
    provider._drain()
    end_session(provider)
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["written while no session ran"]


def test_a_review_entry_of_a_group_turn_is_that_chats_and_not_its_senders(env, platform, tmp_path, monkeypatch):
    # The review learned it from the group: it is kept for that chat, credited
    # to the profile's owner (nothing says which member said it).
    env.setenv("INFINO_AUTHOR", "owner-name")
    hermes_dispatch(monkeypatch)
    _, provider = a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    assert a_review_saves("g1", "memory", "the build is flaky on Fridays")["success"]
    turn(provider, 2)
    end_session(provider)
    [row] = platform.rows(TEAM_DB, tables.MEMORY)
    assert (row["content"], row["chat"], row["author"], row["scope"]) == (
        "the build is flaky on Fridays", GROUP_CHAT, "owner-name", "team")


def test_without_a_configured_name_a_review_entry_is_credited_to_the_profile(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    in_session(monkeypatch, GROUP)
    mgr, provider = make_manager(tmp_path, platform="telegram", agent_identity="ops", session_id="g1")
    turn(provider)
    a_review_saves("g1", "memory", "the build is flaky on Fridays")
    turn(provider, 2)
    [row] = local_rows(tmp_path, tables.MEMORY)
    assert (row["chat"], row["author"]) == (GROUP_CHAT, "ops")


def test_a_personal_profiles_user_entry_found_during_a_group_turn_is_never_its_owners(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    home.setenv("INFINO_AUTHOR", "dana")
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP)
    write_file(tmp_path, "user", ["works from Lisbon"])
    turn(provider)
    # Nothing of it in the team's tables, nor in the owner's, even at the owner's next turn.
    assert [r for r in local_rows(tmp_path, tables.MEMORY) if "Lisbon" in r["content"]] == []
    assert personal_rows(tmp_path) == []
    in_session(monkeypatch, DM)  # the owner's own chat
    turn(provider, 2)
    assert personal_rows(tmp_path) == []


def test_a_review_entry_of_a_group_turn_shows_in_no_other_chat(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    _, provider = a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    a_review_saves("g1", "memory", GROUP_NOTE)
    turn(provider, 2)
    in_session(monkeypatch, OTHER_GROUP)
    assert "Acme" not in provider.prefetch("Acme deal")
    in_session(monkeypatch, GROUP)
    assert "Acme" in provider.prefetch("Acme deal")


def test_a_review_entry_another_session_finds_first_stays_the_groups(home, tmp_path, monkeypatch):
    # Each chat has an agent of its own; the group's review writes after its
    # turn, and the owner's session at the terminal compares the files first.
    hermes_dispatch(monkeypatch)
    a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    a_review_saves("g1", "memory", GROUP_NOTE)
    in_session(monkeypatch, {})
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    assert [(r["content"], r["chat"]) for r in local_rows(tmp_path, tables.MEMORY)] == [(GROUP_NOTE, GROUP_CHAT)]
    assert "Acme" not in at_the_terminal.prefetch("Acme deal")


@pytest.mark.parametrize("values, surface", [(OTHER_GROUP, "telegram"), (DM, "telegram"), ({}, "cli")])
def test_a_review_entry_stays_the_groups_however_late_another_session_compares(home, tmp_path, monkeypatch, values,
                                                                              surface):
    # The review may write long after the group's turn (Hermes defers it until
    # the machine is idle), and the file comparison that finds it may run in any
    # other chat's session, or the owner's, later still.
    hermes_dispatch(monkeypatch)
    a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    later = provider_mod._now_ms() + MUCH_LATER_MS
    monkeypatch.setattr(provider_mod, "_now_ms", lambda: later)
    a_review_saves("g1", "memory", GROUP_NOTE)
    in_session(monkeypatch, values)
    _, other = make_manager(tmp_path, platform=surface, session_id="other")
    turn(other)
    assert [(r["content"], r["chat"]) for r in local_rows(tmp_path, tables.MEMORY)] == [(GROUP_NOTE, GROUP_CHAT)]
    for chat in (OTHER_GROUP, DM, {}):
        in_session(monkeypatch, chat)
        assert "Acme" not in other.prefetch("Acme deal")
    in_session(monkeypatch, GROUP)
    assert "Acme" in other.prefetch("Acme deal")


def test_an_entry_no_turn_noted_is_the_owners_whatever_chat_ran_last(home, tmp_path, monkeypatch):
    # A hand edit, or an entry the owner's own session saved: the team's, in no chat.
    hermes_dispatch(monkeypatch)
    _, provider = a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    write_file(tmp_path, "memory", ["deploys are on Tuesdays"])
    turn(provider, 2)
    assert [(r["content"], r["chat"]) for r in local_rows(tmp_path, tables.MEMORY)] == [("deploys are on Tuesdays", "")]


def test_a_review_entry_two_groups_reviews_saved_shows_in_neither(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    bots = {session: a_group_turn(tmp_path, monkeypatch, chat, session)[1]
            for session, chat in (("g1", GROUP), ("g2", OTHER_GROUP))}
    for session in bots:  # each review saves it before any comparison finds it
        assert a_review_saves(session, "memory", GROUP_NOTE)["success"]
    turn(bots["g2"], 2)  # the second group's next turn compares the files
    assert [r["content"] for r in local_rows(tmp_path, tables.MEMORY)] == [GROUP_NOTE]
    for chat in (GROUP, OTHER_GROUP, DM):
        in_session(monkeypatch, chat)
        assert "Acme" not in bots["g2"].prefetch("Acme deal")


def test_a_note_kept_for_several_chats_can_be_forgotten_from_any_of_them(home, tmp_path, monkeypatch):
    in_session(monkeypatch, GROUP)
    mgr, provider = make_manager(tmp_path, platform="telegram", session_id="g1")
    provider._backend.append(tables.MEMORY, [{"content": GROUP_NOTE, "scope": "team", "origin": "builtin:memory",
                                             tables.CHAT: provider_mod.CHATS_JOIN.join([GROUP_CHAT, OTHER_CHAT])}])
    in_session(monkeypatch, PREFIX_GROUP)  # a chat it was not kept for
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": GROUP_NOTE}))["matched"] == 0
    in_session(monkeypatch, OTHER_GROUP)
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": GROUP_NOTE}))["matched"] == 1
    assert local_rows(tmp_path, tables.MEMORY) == []


def test_an_entry_a_session_nothing_noted_saved_is_no_ones(home, tmp_path, monkeypatch):
    # Whose turn it followed cannot be told: kept for no chat's recall, never the owner's.
    hermes_dispatch(monkeypatch)
    a_review_saves("unseen", "memory", GROUP_NOTE)
    a_review_saves("unseen", "user", "Bob is job hunting")
    in_session(monkeypatch, {})
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    assert [(r["content"], r["chat"]) for r in local_rows(tmp_path, tables.MEMORY)] == [(GROUP_NOTE,
                                                                                         chats.UNATTRIBUTED)]
    assert personal_rows(tmp_path) == []
    assert "Acme" not in at_the_terminal.prefetch("Acme deal")


def test_a_memory_write_whose_writer_cannot_be_noted_is_refused(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    (config.state_dir(tmp_path) / builtin.WRITERS).mkdir()  # nothing can be written there
    answer = a_review_saves("g1", "memory", GROUP_NOTE)
    assert answer.get("success") is not True and traces.NOT_NOTED in answer["error"]
    assert builtin.read_entries(tmp_path, "memory") in (None, [])


@contextmanager
def writers_held_elsewhere(tmp_path, monkeypatch):
    """Another process holding the note of who wrote the memory files'
    entries, past the short wait these tests allow it."""
    monkeypatch.setattr(builtin, "WRITERS_LOCK_WAIT_S", QUICK_LOCK_WAIT_S)
    state = config.state_dir(tmp_path)
    state.mkdir(parents=True, exist_ok=True)
    with open(state / builtin.WRITERS_LOCK, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def test_a_memory_write_waits_for_its_note_only_so_long(home, tmp_path, monkeypatch):
    # Hermes refuses every tool call while a pre_tool_call hook hangs: the wait is bounded, and the write refused.
    hermes_dispatch(monkeypatch)
    a_group_turn(tmp_path, monkeypatch, GROUP, "g1")
    with writers_held_elsewhere(tmp_path, monkeypatch):
        answer = a_review_saves("g1", "memory", GROUP_NOTE)
    assert answer.get("success") is not True and traces.NOT_NOTED in answer["error"]
    assert a_review_saves("g1", "memory", GROUP_NOTE)["success"]  # once it is free


def test_a_comparison_that_cannot_read_the_note_adds_its_entries_next_time(home, tmp_path, monkeypatch):
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "use pnpm"])
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "the VPN is needed for staging"])
    with writers_held_elsewhere(tmp_path, monkeypatch):
        turn(at_the_terminal, 2)
    rows = local_rows(tmp_path, tables.MEMORY)
    assert contents(current(rows)) == ["deploys are on Tuesdays"]  # what left is superseded
    assert contents(superseded(rows)) == ["use pnpm"]
    turn(at_the_terminal, 3)
    assert contents(current(local_rows(tmp_path, tables.MEMORY))) == ["deploys are on Tuesdays",
                                                                      "the VPN is needed for staging"]


def test_a_note_of_writers_that_cannot_be_read_never_stops_the_comparison(home, tmp_path, monkeypatch):
    # Every entry the files hold then is taken as no one's; what is written after is kept as before.
    home.setenv("INFINO_AUTHOR", "dana")
    write_file(tmp_path, "memory", ["deploys are on Tuesdays"])
    note = config.state_dir(tmp_path) / builtin.WRITERS
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("{not json")
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    assert [(r["content"], r["chat"]) for r in local_rows(tmp_path, tables.MEMORY)] == [
        ("deploys are on Tuesdays", chats.UNATTRIBUTED)]
    write_file(tmp_path, "user", ["prefers short answers"])
    turn(at_the_terminal, 2)
    assert [(r["content"], r["author"]) for r in personal_rows(tmp_path)] == [
        ("About dana: prefers short answers", "dana")]


def test_a_user_entry_saved_after_a_turn_in_a_chat_of_no_known_type_is_never_the_owners(home, tmp_path,
                                                                                       monkeypatch):
    # The chat may be the owner's own DM behind a relay, so the profile is not
    # marked for good; but what its review saves to USER.md is never theirs.
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    home.setenv("INFINO_AUTHOR", "dana")
    hermes_dispatch(monkeypatch)
    a_group_turn(tmp_path, monkeypatch, UNTYPED, "relay-1")
    a_review_saves("relay-1", "user", "Bob is job hunting")
    in_session(monkeypatch, DM)
    _, in_the_owners_dm = make_manager(tmp_path, platform="telegram", session_id="dm-1")
    turn(in_the_owners_dm)
    assert personal_rows(tmp_path) == []
    write_file(tmp_path, "user", ["Bob is job hunting", "prefers short answers"])  # the owner's own edit
    turn(in_the_owners_dm, 2)
    assert [r["content"] for r in personal_rows(tmp_path)] == ["About dana: prefers short answers"]


def test_a_personal_profiles_user_entry_found_after_a_group_turn_is_never_its_owners(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    home.setenv("INFINO_AUTHOR", "dana")
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP)
    turn(provider)  # the group's turn; the review follows it
    write_file(tmp_path, "user", ["Bob is job hunting"])
    in_session(monkeypatch, DM)  # the owner's own chat
    turn(provider, 2)
    assert personal_rows(tmp_path) == []


def test_a_user_entry_found_after_another_agents_turn_is_never_the_owners(home, tmp_path):
    # Bot Mode and `hermes peer dm` run another agent's message in the Bot Chat.
    mgr, provider = make_manager(tmp_path, platform="desktop")
    mgr.on_turn_start(1, "What does Dana prefer?", author_id=PEER_AGENT["id"], author_name=PEER_AGENT["name"],
                      author_is_bot=True)
    provider.sync_turn("What does Dana prefer?", "Short answers.", session_id="s1", turn_author=PEER_AGENT)
    provider._drain()
    write_file(tmp_path, "user", ["researcher prefers terse replies"])
    mgr.on_turn_start(2, "hi")  # the owner's next turn
    turn(provider, 2)
    assert personal_rows(tmp_path) == []


def test_a_user_entry_is_the_owners_after_a_scheduled_job_ran(home, tmp_path):
    # A cron run is the owner's own: it never makes the profile a shared gateway's.
    home.setenv("INFINO_AUTHOR", "dana")
    make_manager(tmp_path, platform="cron", agent_context="cron", session_id="cron_a1b2c3_20261105_070000")
    write_file(tmp_path, "user", ["prefers short answers"])
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    assert [(r["content"], r["author"]) for r in personal_rows(tmp_path)] == [
        ("About dana: prefers short answers", "dana")]


def test_a_mark_an_earlier_version_left_empty_is_still_honored(home, tmp_path):
    mark = config.state_dir(tmp_path) / GATEWAY_MARK
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.touch()
    write_file(tmp_path, "user", ["works from Lisbon"])
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    assert personal_rows(tmp_path) == []


def test_a_shared_profiles_user_file_is_never_taken_for_the_owners_memory(env, platform, tmp_path, monkeypatch):
    # Hermes's USER.md of a profile a gateway answers may be about anyone who wrote in.
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, DM)
    write_file(tmp_path, "user", ["works from Lisbon"])
    write_file(tmp_path, "memory", ["deploys are on Tuesdays"])
    turn(provider)
    end_session(provider)
    # The owner's own session on the same profile does not take it either.
    in_session(monkeypatch, {})
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal, 2)
    end_session(at_the_terminal)
    assert platform.rows(PERSONAL_DB, tables.MEMORY) == [] and personal_rows(tmp_path) == []
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["deploys are on Tuesdays"]


def test_a_personal_profiles_dm_user_entry_is_its_owners(env, platform, tmp_path, monkeypatch):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    env.setenv("INFINO_PERSONAL_PROFILE", "on")
    env.setenv("INFINO_AUTHOR", "dana")
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, DM)  # whoever sent the turn, the entry is the owner's
    write_file(tmp_path, "user", ["works from Lisbon"])
    turn(provider)
    end_session(provider)
    assert wait_for(lambda: len(platform.rows(PERSONAL_DB, tables.MEMORY)) == 1)
    [row] = platform.rows(PERSONAL_DB, tables.MEMORY)
    assert (row["content"], row["author"]) == ("About dana: works from Lisbon", "dana")

"""Hermes's built-in memory files kept in step with the `memory` table,
whoever writes them: the agent's memory tool, Hermes's background review (which
runs with every memory provider switched off), or a person by hand."""

from __future__ import annotations

from hermes_infino import builtin, chats, config, tables
from hermes_infino.provider import GATEWAY_MARK

from .helpers import (DM, GROUP, OTHER_GROUP, PERSONAL_DB, TEAM_DB, end_session, in_session, local_rows, make_manager,
                      personal_rows, wait_for)

# A credential shaped the way Hermes's redaction recognizes one.
SECRET = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 4
# What Hermes's background review writes after a group's turn: the group's business.
GROUP_NOTE = "Acme deal is $2M, keep it from eng"
# The key of GROUP's chat, as the provider keeps a group's memory.
GROUP_CHAT = chats.chat_key(GROUP["HERMES_SESSION_PLATFORM"], GROUP["HERMES_SESSION_CHAT_ID"])
# The author Hermes names for a turn another agent wrote into a profile's Bot Chat.
PEER_AGENT = {"id": "bot:spark/researcher", "name": "researcher", "is_bot": True}


def write_file(home, target: str, entries: list[str]) -> None:
    """A memory file as Hermes writes one."""
    path = home / builtin.MEMORIES_DIR / builtin.FILES[target]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(builtin.ENTRY_DELIMITER.join(entries), encoding="utf-8")


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


def test_an_entry_the_review_removes_leaves_the_table_here_and_in_the_cloud(env, platform, tmp_path):
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "use pnpm"])
    turn(provider, 1)
    end_session(provider)
    write_file(tmp_path, "memory", ["use pnpm"])
    turn(provider, 2)
    end_session(provider)
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["use pnpm"]
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["use pnpm"]


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


def test_a_review_entry_found_during_a_group_turn_is_that_chats_and_not_its_senders(env, platform, tmp_path,
                                                                                  monkeypatch):
    # The review may have learned it from the group: it is kept for that chat,
    # credited to the profile's owner (nothing says which member said it).
    env.setenv("INFINO_AUTHOR", "owner-name")
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP)
    write_file(tmp_path, "memory", ["the build is flaky on Fridays"])
    turn(provider)
    end_session(provider)
    [row] = platform.rows(TEAM_DB, tables.MEMORY)
    assert (row["content"], row["chat"], row["author"], row["scope"]) == (
        "the build is flaky on Fridays", GROUP_CHAT, "owner-name", "team")


def test_without_a_configured_name_a_review_entry_is_credited_to_the_profile(home, tmp_path, monkeypatch):
    mgr, provider = make_manager(tmp_path, platform="telegram", agent_identity="ops")
    in_session(monkeypatch, GROUP)
    write_file(tmp_path, "memory", ["the build is flaky on Fridays"])
    turn(provider)
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


def test_a_review_entry_found_after_a_group_turn_shows_in_no_other_chat(home, tmp_path, monkeypatch):
    in_session(monkeypatch, GROUP)
    mgr, provider = make_manager(tmp_path, platform="telegram")
    write_file(tmp_path, "memory", [GROUP_NOTE])
    turn(provider)
    in_session(monkeypatch, OTHER_GROUP)
    assert "Acme" not in provider.prefetch("Acme deal")
    in_session(monkeypatch, GROUP)
    assert "Acme" in provider.prefetch("Acme deal")


def test_a_review_entry_another_session_finds_first_stays_the_groups(home, tmp_path, monkeypatch):
    # Each chat has an agent of its own; the group's review writes after its
    # turn, and the owner's session at the terminal compares the files first.
    in_session(monkeypatch, GROUP)
    _, group_bot = make_manager(tmp_path, platform="telegram", session_id="g1")
    turn(group_bot)
    write_file(tmp_path, "memory", [GROUP_NOTE])
    in_session(monkeypatch, {})
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal)
    assert [(r["content"], r["chat"]) for r in local_rows(tmp_path, tables.MEMORY)] == [(GROUP_NOTE, GROUP_CHAT)]
    assert "Acme" not in at_the_terminal.prefetch("Acme deal")


def test_a_review_entry_two_groups_reviews_may_have_written_shows_in_neither(home, tmp_path, monkeypatch):
    for session, chat in (("g1", GROUP), ("g2", OTHER_GROUP)):
        in_session(monkeypatch, chat)
        _, bot = make_manager(tmp_path, platform="telegram", session_id=session)
        turn(bot)
    write_file(tmp_path, "memory", [GROUP_NOTE])
    turn(bot, 2)  # the second group's next turn compares the files
    assert [r["content"] for r in local_rows(tmp_path, tables.MEMORY)] == [GROUP_NOTE]
    for chat in (GROUP, OTHER_GROUP, DM):
        in_session(monkeypatch, chat)
        assert "Acme" not in bot.prefetch("Acme deal")


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

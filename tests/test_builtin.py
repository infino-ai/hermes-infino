"""Hermes's built-in memory files kept in step with the `memory` table,
whoever writes them: the agent's memory tool, Hermes's background review (which
runs with every memory provider switched off), or a person by hand."""

from __future__ import annotations

from hermes_infino import builtin, config, tables
from hermes_infino.provider import GATEWAY_MARK

from .helpers import (DM, GROUP, PERSONAL_DB, TEAM_DB, in_session, local_rows, make_manager, personal_rows,
                      wait_for)

# A credential shaped the way Hermes's redaction recognizes one.
SECRET = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 4


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
    provider.shutdown()
    team = [r for r in platform.rows(TEAM_DB, tables.MEMORY)]
    assert [(r["content"], r["origin"], r["scope"]) for r in team] == [
        ("deploys are on Tuesdays", "builtin:memory", "team")]
    assert [(r["content"], r["author"]) for r in personal_rows(tmp_path)] == [
        ("About dana: prefers short answers", "dana")]


def test_an_entry_the_review_removes_leaves_the_table_here_and_in_the_cloud(env, platform, tmp_path):
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "use pnpm"])
    turn(provider, 1)
    provider.shutdown()
    write_file(tmp_path, "memory", ["use pnpm"])
    turn(provider, 2)
    provider.shutdown()
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
    provider.shutdown()
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["use pnpm"]


def test_a_missing_or_unreadable_file_deletes_nothing(env, platform, tmp_path):
    mgr, provider = make_manager(tmp_path)
    provider.on_memory_write("add", "memory", "use pnpm")  # no file: memory disabled, say
    turn(provider)
    (tmp_path / builtin.MEMORIES_DIR).mkdir(parents=True, exist_ok=True)
    (tmp_path / builtin.MEMORIES_DIR / "MEMORY.md").write_bytes(b"\xff\xfe not text")
    turn(provider, 2)
    provider.shutdown()
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["use pnpm"]


def test_a_secret_in_a_memory_file_never_leaves_this_machine(env, platform, tmp_path):
    mgr, provider = make_manager(tmp_path)
    write_file(tmp_path, "memory", [f"the deploy key is {SECRET}"])
    turn(provider)
    turn(provider, 2)  # compared again: the scrubbed row still holds the entry
    provider.shutdown()
    [row] = platform.rows(TEAM_DB, tables.MEMORY)
    assert SECRET not in row["content"] and row["content"].startswith("the deploy key is")


def test_what_the_review_wrote_before_a_start_is_caught_up_at_the_start(env, platform, tmp_path):
    write_file(tmp_path, "memory", ["written while no session ran"])
    mgr, provider = make_manager(tmp_path)
    provider._drain()
    provider.shutdown()
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["written while no session ran"]


def test_a_review_entry_found_during_a_group_turn_is_neither_that_chats_nor_its_senders(env, platform, tmp_path,
                                                                                      monkeypatch):
    # The review may have written it after another chat's conversation: the
    # comparison cannot say which, nor who said it.
    env.setenv("INFINO_AUTHOR", "owner-name")
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP)
    write_file(tmp_path, "memory", ["the build is flaky on Fridays"])
    turn(provider)
    provider.shutdown()
    [row] = platform.rows(TEAM_DB, tables.MEMORY)
    assert (row["content"], row["chat"], row["author"], row["scope"]) == (
        "the build is flaky on Fridays", "", "owner-name", "team")


def test_without_a_configured_name_a_review_entry_is_credited_to_the_profile(home, tmp_path, monkeypatch):
    mgr, provider = make_manager(tmp_path, platform="telegram", agent_identity="ops")
    in_session(monkeypatch, GROUP)
    write_file(tmp_path, "memory", ["the build is flaky on Fridays"])
    turn(provider)
    [row] = local_rows(tmp_path, tables.MEMORY)
    assert (row["chat"], row["author"]) == ("", "ops")


def test_a_personal_profiles_user_entry_found_during_a_group_turn_waits_for_the_owner(home, tmp_path, monkeypatch):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    home.setenv("INFINO_AUTHOR", "dana")
    mgr, provider = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, GROUP)
    write_file(tmp_path, "user", ["works from Lisbon"])
    turn(provider)
    # Nothing of it in the team's tables, nor (yet) in the owner's.
    assert [r for r in local_rows(tmp_path, tables.MEMORY) if "Lisbon" in r["content"]] == []
    assert personal_rows(tmp_path) == []
    in_session(monkeypatch, DM)  # the owner's own chat
    turn(provider, 2)
    assert [(r["content"], r["author"]) for r in personal_rows(tmp_path)] == [("About dana: works from Lisbon", "dana")]


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
    provider.shutdown()
    # The owner's own session on the same profile does not take it either.
    in_session(monkeypatch, {})
    _, at_the_terminal = make_manager(tmp_path, session_id="cli")
    turn(at_the_terminal, 2)
    at_the_terminal.shutdown()
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
    provider.shutdown()
    assert wait_for(lambda: len(platform.rows(PERSONAL_DB, tables.MEMORY)) == 1)
    [row] = platform.rows(PERSONAL_DB, tables.MEMORY)
    assert (row["content"], row["author"]) == ("About dana: works from Lisbon", "dana")

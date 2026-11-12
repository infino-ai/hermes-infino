"""Earlier versions of memories, kept in `memory_history`: a replacement (a
patch, as Hermes reports one) and the removals of a batch that adds what
replaces them go there; any other removal deletes. Forgetting is total.
Recall shows earlier versions after every current memory."""

from __future__ import annotations

import json

import pytest

from hermes_infino import builtin, config, tables
from hermes_infino.client import InfinoClient
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore
from hermes_infino.recall import Recaller

from .helpers import (DM, GROUP, PERSONAL_DB, TEAM_DB, agent_tool_call, end_session, hermes_dispatch, in_session,
                      local_rows, make_manager)
from .test_builtin import contents, turn, write_file


def history_rows(tmp_path, personal: bool = False) -> list[dict]:
    store = LocalStore.under(config.state_dir(tmp_path), PERSONAL_DB_DIR) if personal else None
    if store is not None:
        return store.query_sql(f"SELECT * FROM {tables.HISTORY} ORDER BY stored_ms")
    return local_rows(tmp_path, tables.HISTORY)


def by(rows: list[dict]) -> list[tuple[str, str]]:
    """Each earlier version's text and what superseded it, oldest first."""
    return [(r["content"], r[tables.SUPERSEDED_BY]) for r in sorted(rows, key=lambda r: r["stored_ms"])]


# -- what is kept -----------------------------------------------------------------

def test_a_replacement_moves_the_old_entry_to_the_history(home, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "deploys are on Monday")
    p.on_memory_write("replace", "memory", "deploys are on Tuesday", {"previous_content": "deploys are on Monday"})
    p.on_memory_write("replace", "memory", "deploys are on Thursday", {"previous_content": "deploys are on Tuesday"})
    p._drain()
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["deploys are on Thursday"]  # memory: current only
    old = history_rows(tmp_path)
    assert by(old) == [("deploys are on Monday", "deploys are on Tuesday"),
                       ("deploys are on Tuesday", "deploys are on Thursday")]
    assert all(r[tables.SUPERSEDED] >= r["created_ms"] and r[tables.PRIOR_ROW_ID] for r in old)


def test_a_patch_through_hermess_own_tool_is_kept(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    mgr, p = make_manager(tmp_path, session_id="cli")
    agent_tool_call("cli", "memory", {"action": "add", "target": "memory", "content": "standup is at ten"}, mgr)
    agent_tool_call("cli", "memory", {"action": "patch", "target": "memory", "old_text": "ten", "new_text": "eleven"},
                    mgr)
    p._drain()
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["standup is at eleven"]
    assert by(history_rows(tmp_path)) == [("standup is at ten", "standup is at eleven")]


def test_a_batch_that_removes_and_adds_keeps_what_it_consolidated(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)  # the plugin's tool hook notes the batch before Hermes runs it
    mgr, p = make_manager(tmp_path, session_id="cli")
    agent_tool_call("cli", "memory", {"target": "memory", "operations": [
        {"action": "add", "content": "use pnpm"}, {"action": "add", "content": "pnpm 9 in the web repo"}]}, mgr)
    agent_tool_call("cli", "memory", {"target": "memory", "operations": [
        {"action": "remove", "old_text": "use pnpm"}, {"action": "remove", "old_text": "pnpm 9 in the web repo"},
        {"action": "add", "content": "use pnpm 9 in the web repo"}]}, mgr)
    p._drain()
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["use pnpm 9 in the web repo"]
    assert sorted(by(history_rows(tmp_path))) == [("pnpm 9 in the web repo", "use pnpm 9 in the web repo"),
                                                  ("use pnpm", "use pnpm 9 in the web repo")]


def test_a_removal_on_its_own_forgets_and_keeps_nothing(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    mgr, p = make_manager(tmp_path, session_id="cli")
    agent_tool_call("cli", "memory", {"action": "add", "target": "memory", "content": "Dana's flight lands at 6"}, mgr)
    agent_tool_call("cli", "memory", {"action": "remove", "target": "memory", "old_text": "Dana's flight"}, mgr)
    p._drain()
    assert local_rows(tmp_path, tables.MEMORY) == [] and history_rows(tmp_path) == []
    assert "flight" not in p.prefetch("when does Dana's flight land")


def test_what_leaves_the_file_otherwise_is_deleted_not_kept(home, tmp_path):
    # An approved removal, a hand edit: a person's removal, reported to no provider.
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "use pnpm"])
    turn(p, 1)
    write_file(tmp_path, "memory", ["use pnpm"])
    turn(p, 2)
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["use pnpm"]
    assert history_rows(tmp_path) == []


def test_a_text_replaced_again_is_kept_once(home, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "use pnpm")
    for n in range(3):
        p.on_memory_write("replace", "memory", "use pnpm 9", {"previous_content": "use pnpm"})
        p.on_memory_write("replace", "memory", "use pnpm", {"previous_content": "use pnpm 9"})
    p._drain()
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["use pnpm"]
    assert sorted(by(history_rows(tmp_path))) == [("use pnpm", "use pnpm 9"), ("use pnpm 9", "use pnpm")]


def test_every_earlier_version_is_kept(home, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "release 1")
    for n in range(2, 41):
        p.on_memory_write("replace", "memory", f"release {n}", {"previous_content": f"release {n - 1}"})
    p._drain()
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["release 40"]
    assert len(history_rows(tmp_path)) == 39


def test_a_crash_between_the_copy_and_the_delete_loses_nothing_and_copies_once(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "the build uses make")
    p._drain()
    b = p._backend
    real = b.forget_rows
    monkeypatch.setattr(b, "forget_rows", lambda rows: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        b.supersede_memory("the build uses make", "the build uses just")
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["the build uses make"] and len(history_rows(tmp_path)) == 1
    monkeypatch.setattr(b, "forget_rows", real)
    assert b.supersede_memory("the build uses make", "the build uses just") == 1
    assert local_rows(tmp_path, tables.MEMORY) == []
    assert by(history_rows(tmp_path)) == [("the build uses make", "the build uses just")]


# -- privacy ----------------------------------------------------------------------

def test_a_groups_replacement_never_names_its_text_on_a_team_wide_row(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path, platform="telegram")
    in_session(monkeypatch, DM)
    p.on_memory_write("add", "memory", "standup is at ten")
    p._drain()
    in_session(monkeypatch, GROUP)  # a member replaces the team-wide entry with the group's own note
    p.on_memory_write("replace", "memory", "standup moves to the Acme war room",
                      {"previous_content": "standup is at ten"})
    p._drain()
    [old] = history_rows(tmp_path)
    assert old["content"] == "standup is at ten" and not old[tables.CHAT]
    assert old[tables.SUPERSEDED_BY] == ""  # the group's text stays with the group's row


def test_a_persons_earlier_entry_goes_to_their_own_database_never_the_teams(env, platform, tmp_path):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    env.setenv("INFINO_AUTHOR", "priya")
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "user", "Prefers short answers")
    p.on_memory_write("replace", "user", "Prefers bullet points", {"previous_content": "Prefers short answers"})
    end_session(p)
    assert not [r for r in platform.rows(TEAM_DB, tables.HISTORY) if "Prefers" in r["content"]]
    assert by(platform.rows(PERSONAL_DB, tables.HISTORY)) == [
        ("About priya: Prefers short answers", "About priya: Prefers bullet points")]


# -- forgetting is total --------------------------------------------------------------

def test_forgetting_a_memory_forgets_every_earlier_version_of_it(env, platform, tmp_path):
    m, p = make_manager(tmp_path)
    p.on_memory_write("add", "user", "Lives at 12 Elm St")
    p.on_memory_write("replace", "user", "Lives at 99 Oak Ave", {"previous_content": "Lives at 12 Elm St"})
    p.on_memory_write("add", "memory", "the office is at 1 Main St")
    end_session(p)
    m2, p2 = make_manager(tmp_path)
    out = json.loads(m2.handle_tool_call("infino_forget", {"content": "Lives at 99 Oak Ave"}))
    assert out["ok"] and out["matched"] >= 2
    end_session(p2)
    everywhere = (local_rows(tmp_path, tables.MEMORY) + local_rows(tmp_path, tables.HISTORY)
                  + history_rows(tmp_path, personal=True)
                  + LocalStore.under(config.state_dir(tmp_path), PERSONAL_DB_DIR).query_sql("SELECT * FROM memory")
                  + platform.rows(TEAM_DB, tables.MEMORY) + platform.rows(TEAM_DB, tables.HISTORY))
    text = json.dumps(everywhere, default=str)
    assert "Elm" not in text and "Oak" not in text
    assert "1 Main St" in text  # nothing else


def test_infino_forget_follows_the_chain_in_the_cloud_whichever_machine_pushed_it(env, platform, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "standup is at ten")
    end_session(p)
    # A teammate's machine kept two earlier versions: nine, superseded by nine-thirty, superseded by ten.
    tables.ensure_optional(InfinoClient("k", platform.url), TEAM_DB, tables.HISTORY)
    platform.databases[TEAM_DB][tables.HISTORY] += [
        {"row_id": "t1", "stored_ms": 1, "content": "standup is at nine", "scope": "team", "chat": "",
         tables.SUPERSEDED: 1, tables.SUPERSEDED_BY: "standup is at nine-thirty", tables.PRIOR_ROW_ID: "x"},
        {"row_id": "t2", "stored_ms": 2, "content": "standup is at nine-thirty", "scope": "team", "chat": "",
         tables.SUPERSEDED: 2, tables.SUPERSEDED_BY: "standup is at ten", tables.PRIOR_ROW_ID: "y"}]
    m2, p2 = make_manager(tmp_path)
    assert json.loads(m2.handle_tool_call("infino_forget", {"content": "standup is at ten"}))["ok"]
    end_session(p2)
    assert platform.rows(TEAM_DB, tables.MEMORY) == [] and platform.rows(TEAM_DB, tables.HISTORY) == []


def test_infino_forget_takes_the_entry_out_of_hermess_file(home, tmp_path):
    mgr, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["the deploy key is in vault", "use pnpm"])
    turn(p, 1)
    out = json.loads(mgr.handle_tool_call("infino_forget", {"content": "the deploy key is in vault"}))
    assert out["ok"] and out["matched"] == 2  # the row, and the file's entry
    assert builtin.read_entries(tmp_path, "memory") == ["use pnpm"]
    turn(p, 2)  # the next comparison does not find it again
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["use pnpm"]


def test_no_history_is_made_of_a_memory_a_teammate_forgot(env, platform, tmp_path):
    # This machine still holds the row it pushed; a teammate's forget deleted it in the cloud.
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "standup is at ten")
    p.on_memory_write("add", "memory", "Dana's PIN is 4321")
    end_session(p)
    platform.databases[TEAM_DB][tables.MEMORY] = [r for r in platform.rows(TEAM_DB, tables.MEMORY)
                                                  if "4321" not in r["content"]]
    _, p2 = make_manager(tmp_path)
    p2.on_memory_write("replace", "memory", "Dana changed her PIN", {"previous_content": "Dana's PIN is 4321"})
    end_session(p2)
    assert not [r for r in local_rows(tmp_path, tables.HISTORY) + platform.rows(TEAM_DB, tables.HISTORY)
                if "4321" in r["content"]]


def test_no_history_is_made_of_a_pushed_memory_the_cloud_cannot_be_asked_about(env, platform, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "Dana's PIN is 4321")
    end_session(p)
    _, p2 = make_manager(tmp_path)
    b = p2._backend
    monkeypatch.setattr(b.mirror, "matching_ids", lambda *a, **k: (_ for _ in ()).throw(OSError("unreachable")))
    p2.on_memory_write("replace", "memory", "Dana changed her PIN", {"previous_content": "Dana's PIN is 4321"})
    p2._drain()
    assert local_rows(tmp_path, tables.HISTORY) == []


# -- upgrading -------------------------------------------------------------------------

def test_memory_keeps_its_schema_so_an_upgrade_adds_no_columns():
    names = {f["name"] for f in tables.MEMORY_SCHEMA}
    assert not names & {tables.SUPERSEDED, tables.SUPERSEDED_BY, tables.PRIOR_ROW_ID}


def test_a_database_that_refuses_the_history_table_still_syncs_everything_else(env, platform, tmp_path):
    platform.refused_tables = {tables.HISTORY}
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "standup is at ten")
    p.on_memory_write("replace", "memory", "standup is at eleven", {"previous_content": "standup is at ten"})
    end_session(p)
    assert contents(platform.rows(TEAM_DB, tables.MEMORY)) == ["standup is at eleven"]  # memory synced
    assert tables.HISTORY not in platform.databases[TEAM_DB]
    assert by(local_rows(tmp_path, tables.HISTORY)) == [("standup is at ten", "standup is at eleven")]  # kept here
    platform.refused_tables = set()


# -- review follow-ups ------------------------------------------------------------------

def test_forgetting_a_memory_md_entry_leaves_a_user_md_entry_that_shares_its_end(home, tmp_path):
    mgr, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["About the release: ship Friday", "use pnpm"])
    write_file(tmp_path, "user", ["ship Friday"])
    turn(p, 1)
    out = json.loads(mgr.handle_tool_call("infino_forget", {"content": "About the release: ship Friday"}))
    assert out["ok"]
    assert builtin.read_entries(tmp_path, "memory") == ["use pnpm"]
    assert builtin.read_entries(tmp_path, "user") == ["ship Friday"]


def test_a_team_forget_leaves_the_persons_own_file_alone(home, tmp_path):
    mgr, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["prefers tea"])
    write_file(tmp_path, "user", ["prefers tea"])
    turn(p, 1)
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": "prefers tea", "scope": "team"}))["ok"]
    assert builtin.read_entries(tmp_path, "memory") == []
    assert builtin.read_entries(tmp_path, "user") == ["prefers tea"]


def test_a_personal_forget_of_a_recalled_user_entry_takes_it_out_of_user_md(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "priya")
    mgr, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "user", ["Prefers tea", "Works from Lisbon"])
    turn(p, 1)
    out = json.loads(mgr.handle_tool_call("infino_forget", {"content": "About priya: Prefers tea",
                                                            "scope": "personal"}))
    assert out["ok"]
    assert builtin.read_entries(tmp_path, "user") == ["Works from Lisbon"]


def test_a_batch_that_forgets_one_entry_and_adds_an_unrelated_one_keeps_no_history(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    mgr, p = make_manager(tmp_path, session_id="cli")
    agent_tool_call("cli", "memory", {"action": "add", "target": "memory", "content": "Dana's PIN is 4321"}, mgr)
    agent_tool_call("cli", "memory", {"target": "memory", "operations": [
        {"action": "remove", "old_text": "Dana's PIN"},
        {"action": "add", "content": "the office moved to 99 Oak Ave"}]}, mgr)
    p._drain()
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["the office moved to 99 Oak Ave"]
    assert history_rows(tmp_path) == []
    assert "4321" not in p.prefetch("what is Dana's PIN")


def test_forgetting_one_of_two_entries_a_consolidation_split_into_forgets_the_earlier_version(home, tmp_path,
                                                                                               monkeypatch):
    hermes_dispatch(monkeypatch)
    mgr, p = make_manager(tmp_path, session_id="cli")
    agent_tool_call("cli", "memory", {"action": "add", "target": "memory",
                                      "content": "use pnpm 9 in the web repo and the docs repo"}, mgr)
    agent_tool_call("cli", "memory", {"target": "memory", "operations": [
        {"action": "remove", "old_text": "use pnpm 9 in the web repo and"},
        {"action": "add", "content": "use pnpm 9 in the web repo"},
        {"action": "add", "content": "use pnpm 9 in the docs repo"}]}, mgr)
    p._drain()
    assert by(history_rows(tmp_path)) == [("use pnpm 9 in the web repo and the docs repo",
                                           builtin.ENTRY_DELIMITER.join(["use pnpm 9 in the web repo",
                                                                         "use pnpm 9 in the docs repo"]))]
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": "use pnpm 9 in the docs repo"}))["ok"]
    p._drain()
    assert history_rows(tmp_path) == []
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["use pnpm 9 in the web repo"]


def test_a_text_that_comes_back_and_is_replaced_again_is_forgotten_with_its_newest_replacement(home, tmp_path):
    mgr, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "Lives at 12 Elm St")
    p.on_memory_write("replace", "memory", "Lives at 5 Pine Rd", {"previous_content": "Lives at 12 Elm St"})
    p.on_memory_write("replace", "memory", "Lives at 12 Elm St", {"previous_content": "Lives at 5 Pine Rd"})
    p.on_memory_write("replace", "memory", "Lives at 99 Oak Ave", {"previous_content": "Lives at 12 Elm St"})
    p._drain()
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": "Lives at 99 Oak Ave"}))["ok"]
    p._drain()
    text = json.dumps(local_rows(tmp_path, tables.MEMORY) + history_rows(tmp_path))
    assert "Elm" not in text and "Oak" not in text and "Pine" not in text


def test_a_comparison_while_a_replacement_is_in_flight_leaves_its_row_for_the_history(home, tmp_path):
    # Another process of the profile compares the files after Hermes wrote the
    # patch and before this one's provider kept the earlier entry.
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["standup is at ten"])
    turn(p, 1)
    state = config.state_dir(tmp_path)
    builtin.note_replacing(state, "call-1", {"target": "memory", "operations": [
        {"action": "patch", "old_text": "ten", "content": "eleven"}]})
    write_file(tmp_path, "memory", ["standup is at eleven"])
    turn(p, 2)  # the other process's comparison
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["standup is at eleven", "standup is at ten"]
    p.on_memory_write("replace", "memory", "standup is at eleven",
                      {"previous_content": "standup is at ten", "tool_call_id": "call-1"})
    p._drain()
    assert by(history_rows(tmp_path)) == [("standup is at ten", "standup is at eleven")]
    assert builtin.replacing(state) == {}  # the note is done with
    turn(p, 3)
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["standup is at eleven"]


def test_a_note_of_a_replacement_never_reported_holds_up_deletions_for_a_while_only(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["standup is at ten"])
    turn(p, 1)
    builtin.note_replacing(config.state_dir(tmp_path), "call-1", {"target": "memory", "operations": [
        {"action": "patch", "old_text": "ten", "content": "eleven"}]})
    write_file(tmp_path, "memory", ["standup is at eleven"])
    monkeypatch.setattr(builtin, "REPLACING_KEEP_MS", -1)  # the note is past its time
    turn(p, 2)
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["standup is at eleven"]


def test_recall_does_not_search_a_history_table_its_database_refused(env, platform, tmp_path, monkeypatch):
    platform.refused_tables = {tables.HISTORY}
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "standup is at ten")
    p.on_memory_write("replace", "memory", "standup is at eleven", {"previous_content": "standup is at ten"})
    end_session(p)
    _, p2 = make_manager(tmp_path)
    p2._backend.mirror.sync(wait=True)  # this process learns the table is refused
    searched = []
    real = Recaller._cloud_rows
    monkeypatch.setattr(Recaller, "_cloud_rows", lambda self, src, q, rerank=False, table=tables.MEMORY: (
        searched.append(table), real(self, src, q, rerank, table))[1])
    block = p2.prefetch("when is standup")
    assert tables.HISTORY not in searched
    assert "standup is at ten" in block  # this machine's history answers
    platform.refused_tables = set()



# -- second review --------------------------------------------------------------------

@pytest.mark.parametrize("removed, added", [
    ("Dana's PIN is 4321", "Dana's badge number is 77"),
    ("User is vegan", "User is based in Berlin"),
    ("Dana's PIN is 4321", "Dana's PIN is now 5678"),
])
def test_a_removal_sharing_only_a_subject_or_common_words_with_an_add_is_a_forget(removed, added):
    assert builtin.carried_into(removed, [added]) == []


def test_a_consolidation_still_ties_each_entry_it_folds_in():
    assert builtin.carried_into("use pnpm", ["use pnpm 9 in the web repo"]) == ["use pnpm 9 in the web repo"]
    assert builtin.carried_into("Dana prefers tea in the morning", ["Dana prefers tea in the morning, coffee after"])


def test_forget_x_remember_y_about_the_same_person_keeps_no_history(home, tmp_path, monkeypatch):
    hermes_dispatch(monkeypatch)
    mgr, p = make_manager(tmp_path, session_id="cli")
    agent_tool_call("cli", "memory", {"action": "add", "target": "memory", "content": "Dana's PIN is 4321"}, mgr)
    agent_tool_call("cli", "memory", {"target": "memory", "operations": [
        {"action": "remove", "old_text": "Dana's PIN"}, {"action": "add", "content": "Dana's badge number is 77"}]},
        mgr)
    p._drain()
    assert history_rows(tmp_path) == []
    assert "4321" not in p.prefetch("what is Dana's PIN")


def test_a_replacement_holding_the_old_joiner_is_never_split_by_a_forget(home, tmp_path):
    mgr, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "cache: redis")
    p.on_memory_write("replace", "memory", "cache: redis | memcached | none", {"previous_content": "cache: redis"})
    p.on_memory_write("add", "memory", "memcached")
    p._drain()
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": "memcached"}))["ok"]
    p._drain()
    assert by(history_rows(tmp_path)) == [("cache: redis", "cache: redis | memcached | none")]


def test_a_forget_takes_out_a_user_md_entry_written_under_an_earlier_name(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "priya")
    mgr, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "user", ["Prefers tea", "Works from Lisbon"])
    turn(p, 1)
    p._owner_names = (*p._owner_names, "priya-old")  # a name an earlier version wrote rows under
    out = json.loads(mgr.handle_tool_call("infino_forget", {"content": "About priya-old: Prefers tea",
                                                            "scope": "personal"}))
    assert out["ok"]
    assert builtin.read_entries(tmp_path, "user") == ["Works from Lisbon"]


def test_a_shared_gateways_sender_never_takes_an_entry_out_of_the_owners_user_md(home, tmp_path, monkeypatch):
    mgr, p = make_manager(tmp_path, platform="telegram")
    write_file(tmp_path, "user", ["Prefers tea"])
    in_session(monkeypatch, DM)  # a one-to-one chat with anyone who writes in
    assert json.loads(mgr.handle_tool_call("infino_forget", {"content": "Prefers tea"}))["ok"]
    assert builtin.read_entries(tmp_path, "user") == ["Prefers tea"]


def test_a_replacement_in_flight_holds_back_only_the_rows_it_replaces(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["standup is at ten", "use pnpm"])
    turn(p, 1)
    builtin.note_replacing(config.state_dir(tmp_path), "call-1", {"target": "memory", "operations": [
        {"action": "patch", "old_text": "ten", "content": "eleven"}]})
    write_file(tmp_path, "memory", ["standup is at eleven"])  # and "use pnpm" removed by hand
    turn(p, 2)
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["standup is at eleven", "standup is at ten"]


def test_a_note_dated_after_now_holds_nothing_back(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["standup is at ten"])
    turn(p, 1)
    state = config.state_dir(tmp_path)
    builtin.note_replacing(state, "call-1", {"target": "memory", "operations": [
        {"action": "patch", "old_text": "ten", "content": "eleven"}]})
    notes = json.loads((state / builtin.REPLACING).read_text())
    notes["call-1"]["at"] += 10 * 60 * 1000  # written before the clock was put back
    (state / builtin.REPLACING).write_text(json.dumps(notes))
    write_file(tmp_path, "memory", ["standup is at eleven"])
    turn(p, 2)
    assert contents(local_rows(tmp_path, tables.MEMORY)) == ["standup is at eleven"]


def test_a_history_table_lacking_a_new_column_is_still_searched(env, platform, tmp_path, monkeypatch):
    client = InfinoClient("k", platform.url)
    client.ensure_database(TEAM_DB)
    older = [f for f in tables.TEAM_TABLES[tables.HISTORY][0] if f["name"] != tables.PRIOR_ROW_ID]
    client.ensure_table(TEAM_DB, tables.HISTORY, older, tables.TEAM_TABLES[tables.HISTORY][1])
    platform.databases[TEAM_DB][tables.HISTORY].append(
        {"row_id": "t1", "stored_ms": 1, "content": "standup was at nine", "scope": "team", "chat": "",
         tables.SUPERSEDED: 1, tables.SUPERSEDED_BY: "standup is at ten"})
    platform.schema_patch_refused = True
    _, p = make_manager(tmp_path)
    p._backend.mirror.sync(wait=True)
    assert not p._backend.mirror.refused_optional(TEAM_DB, tables.HISTORY)
    assert "standup was at nine" in p.prefetch("when was standup")  # a teammate's history, from the cloud
    platform.schema_patch_refused = False

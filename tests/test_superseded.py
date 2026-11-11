"""Memory Hermes replaces, removes or rewrites away is kept, superseded:
copied with when and by what, the row itself gone from what is current.
Only `infino_forget` deletes. Recall shows superseded memories marked."""

from __future__ import annotations

import json

import pytest

from hermes_infino import tables
from hermes_infino.client import InfinoClient
from hermes_infino.local import LocalStore

from .helpers import PERSONAL_DB, TEAM_DB, current, end_session, local_rows, make_manager, superseded
from .test_builtin import contents, turn, write_file

# The memory table's columns before superseded memories were kept.
OLD_MEMORY_SCHEMA = [f for f in tables.MEMORY_SCHEMA
                     if f["name"] not in (tables.SUPERSEDED, tables.SUPERSEDED_BY, tables.PRIOR_ROW_ID)]


def by(rows: list[dict]) -> list[tuple[str, str]]:
    """Each superseded copy's text and what replaced it, oldest first."""
    return [(r["content"], r[tables.SUPERSEDED_BY]) for r in sorted(superseded(rows), key=lambda r: r["stored_ms"])]


def test_a_chain_of_replacements_keeps_every_earlier_entry_with_what_replaced_it(home, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "deploys are on Monday")
    p.on_memory_write("replace", "memory", "deploys are on Tuesday", {"previous_content": "deploys are on Monday"})
    p.on_memory_write("replace", "memory", "deploys are on Thursday", {"previous_content": "deploys are on Tuesday"})
    p._drain()
    rows = local_rows(tmp_path, tables.MEMORY)
    assert contents(current(rows)) == ["deploys are on Thursday"]
    assert by(rows) == [("deploys are on Monday", "deploys are on Tuesday"),
                        ("deploys are on Tuesday", "deploys are on Thursday")]
    assert all(r[tables.SUPERSEDED] >= r["created_ms"] for r in superseded(rows))


def test_an_entry_written_again_is_current_again_beside_its_superseded_copy(home, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "use pnpm")
    p.on_memory_write("remove", "memory", "", {"previous_content": "use pnpm"})
    p.on_memory_write("add", "memory", "use pnpm")
    p._drain()
    rows = local_rows(tmp_path, tables.MEMORY)
    assert contents(current(rows)) == ["use pnpm"]
    assert by(rows) == [("use pnpm", "")]


def test_a_dropped_entry_is_superseded_once_however_many_turns_follow(home, tmp_path):
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["deploys are on Tuesdays", "use pnpm"])
    turn(p, 1)
    write_file(tmp_path, "memory", ["use pnpm"])
    for n in range(2, 6):
        turn(p, n)
    rows = local_rows(tmp_path, tables.MEMORY)
    assert contents(current(rows)) == ["use pnpm"]
    assert by(rows) == [("deploys are on Tuesdays", "")]


def test_an_entry_the_agent_replaced_is_not_superseded_again_by_the_file_comparison(home, tmp_path):
    # The memory tool tells the provider, then the next comparison reads the file Hermes wrote.
    _, p = make_manager(tmp_path, session_id="cli")
    write_file(tmp_path, "memory", ["deploys are on Tuesdays"])
    turn(p, 1)
    p.on_memory_write("replace", "memory", "deploys are on Thursdays", {"previous_content": "deploys are on Tuesdays"})
    write_file(tmp_path, "memory", ["deploys are on Thursdays"])
    turn(p, 2)
    turn(p, 3)
    rows = local_rows(tmp_path, tables.MEMORY)
    assert contents(current(rows)) == ["deploys are on Thursdays"]
    assert by(rows) == [("deploys are on Tuesdays", "deploys are on Thursdays")]


def test_a_crash_between_the_copy_and_the_delete_loses_nothing_and_copies_once(home, tmp_path, monkeypatch):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "the build uses make")
    p._drain()
    b = p._backend
    real = b.forget_rows
    monkeypatch.setattr(b, "forget_rows", lambda rows: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        b.supersede_memory("the build uses make", "")
    rows = local_rows(tmp_path, tables.MEMORY)
    assert contents(current(rows)) == ["the build uses make"] and len(superseded(rows)) == 1  # both kept
    monkeypatch.setattr(b, "forget_rows", real)
    assert b.supersede_memory("the build uses make", "") == 1
    rows = local_rows(tmp_path, tables.MEMORY)
    assert current(rows) == [] and by(rows) == [("the build uses make", "")]


def test_infino_forget_deletes_the_memory_and_every_superseded_copy_here_and_in_the_cloud(env, platform, tmp_path):
    m, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "the deploy key is in vault")
    p.on_memory_write("remove", "memory", "", {"previous_content": "the deploy key is in vault"})
    p.on_memory_write("add", "memory", "the deploy key is in vault")
    end_session(p)
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 2  # current and superseded
    m2, p2 = make_manager(tmp_path)
    out = json.loads(m2.handle_tool_call("infino_forget", {"content": "the deploy key is in vault"}))
    assert out["ok"] and out["matched"] == 2
    end_session(p2)
    assert local_rows(tmp_path, tables.MEMORY) == []
    assert platform.rows(TEAM_DB, tables.MEMORY) == []


def test_a_persons_superseded_entry_goes_to_their_own_database_never_the_teams(env, platform, tmp_path):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    env.setenv("INFINO_AUTHOR", "priya")
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "user", "Prefers short answers")
    p.on_memory_write("replace", "user", "Prefers bullet points", {"previous_content": "Prefers short answers"})
    end_session(p)
    assert not [r for r in platform.rows(TEAM_DB, tables.MEMORY) if "Prefers" in r["content"]]
    mine = platform.rows(PERSONAL_DB, tables.MEMORY)
    assert contents(current(mine)) == ["About priya: Prefers bullet points"]
    assert by(mine) == [("About priya: Prefers short answers", "About priya: Prefers bullet points")]
    assert all(r["scope"] == "personal" and r["author"] == "priya" for r in mine)


def test_the_cloud_gets_the_copy_and_loses_the_current_row(env, platform, tmp_path):
    _, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "standup is at ten")
    end_session(p)
    [original] = platform.rows(TEAM_DB, tables.MEMORY)
    _, p2 = make_manager(tmp_path)
    p2.on_memory_write("replace", "memory", "standup is at eleven", {"previous_content": "standup is at ten"})
    end_session(p2)
    rows = platform.rows(TEAM_DB, tables.MEMORY)
    assert contents(current(rows)) == ["standup is at eleven"]
    [kept] = superseded(rows)
    assert kept[tables.PRIOR_ROW_ID] == original["row_id"] and kept["row_id"] != original["row_id"]
    assert kept[tables.SUPERSEDED_BY] == "standup is at eleven"


def test_recall_through_the_provider_shows_a_superseded_memory_marked(home, tmp_path):
    m, p = make_manager(tmp_path)
    p.on_memory_write("add", "memory", "the release train leaves on Friday")
    p.on_memory_write("replace", "memory", "the release train leaves on Wednesday",
                      {"previous_content": "the release train leaves on Friday"})
    p._drain()
    block = p.prefetch("when does the release train leave")
    assert "- [team] the release train leaves on Wednesday" in block
    assert "the release train leaves on Friday [superseded " in block
    assert "replaced by: the release train leaves on Wednesday]" in block
    # Current before superseded, with no grader to say otherwise.
    assert block.index("on Wednesday\n") < block.index("on Friday [superseded")


def test_a_table_an_older_version_made_gains_the_columns_and_its_rows_are_current(home, tmp_path, monkeypatch):
    old = {**tables.TEAM_TABLES, tables.MEMORY: (OLD_MEMORY_SCHEMA, tables.MEMORY_INDEXES)}
    monkeypatch.setattr(tables, "TEAM_TABLES", old)
    store = LocalStore(tmp_path / "db")
    store.ensure_tables(with_vectors=False)
    store.append(tables.MEMORY, [{"content": "written by the older version", "origin": "builtin:memory"}])
    monkeypatch.undo()
    store.ensure_tables(with_vectors=False)
    [row] = store.query_sql(f"SELECT * FROM {tables.MEMORY}")
    assert row[tables.SUPERSEDED] is None and row[tables.PRIOR_ROW_ID] is None
    assert store.query_sql(f"SELECT content FROM {tables.MEMORY} WHERE {tables.current_predicate()}") == [
        {"content": "written by the older version"}]


def test_a_cloud_table_an_older_version_made_gains_the_columns(home, platform):
    c = InfinoClient("k", platform.url, retries=0)
    c.ensure_database(TEAM_DB)
    c.ensure_table(TEAM_DB, tables.MEMORY, OLD_MEMORY_SCHEMA, tables.MEMORY_INDEXES)
    tables.ensure_all(c, TEAM_DB)
    assert {tables.SUPERSEDED, tables.SUPERSEDED_BY, tables.PRIOR_ROW_ID} <= set(c.column_names(TEAM_DB, tables.MEMORY))

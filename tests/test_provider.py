"""The memory provider on this machine: its tools, the rows a turn becomes,
and its lifecycle."""

from __future__ import annotations

import json
import sys

import pytest

from hermes_infino import backend, config, embed, provider as provider_mod, tables
from hermes_infino.local import VECTORS
from hermes_infino.provider import InfinoMemoryProvider

from .helpers import SKILL_MD, fake_embed, local, local_rows, make_manager, personal_rows

TOOLS = ("infino_remember", "infino_forget", "infino_publish_skill")


def call(mgr, tool: str, **args) -> dict:
    return json.loads(mgr.handle_tool_call(tool, args))


def turn_rows(tmp_path) -> list[dict]:
    return sorted(local_rows(tmp_path, tables.CONVERSATIONS), key=lambda r: (r["session_id"], r["seq"]))


@pytest.fixture
def mgr(home, tmp_path):
    return make_manager(tmp_path)


# -- what the provider offers ------------------------------------------------------

def test_the_tools_and_their_required_arguments():
    schemas = {s["name"]: s for s in InfinoMemoryProvider().get_tool_schemas()}
    assert tuple(schemas) == TOOLS
    assert schemas["infino_remember"]["parameters"]["required"] == ["content"]
    assert schemas["infino_forget"]["parameters"]["properties"]["scope"]["enum"] == ["team", "personal"]
    assert schemas["infino_publish_skill"]["parameters"]["required"] == ["name"]


def test_the_setup_fields_are_the_settings():
    assert InfinoMemoryProvider().get_config_schema() == config.config_schema()


def test_available_with_every_library_and_says_which_is_missing(monkeypatch):
    p = InfinoMemoryProvider()
    assert p.is_available()
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # an import of it now fails
    assert not p.is_available() and p.unavailable_reason() == "missing Python packages: onnxruntime"


def test_before_initialize_it_offers_no_prompt_and_refuses_tools():
    p = InfinoMemoryProvider()
    assert p.system_prompt_block() == ""
    assert p.prefetch("anything") == ""
    assert json.loads(p.handle_tool_call("infino_remember", {"content": "x"})) == {"ok": False, "error": "not configured"}


def test_an_unknown_tool_is_refused(mgr):
    _, p = mgr
    assert json.loads(p.handle_tool_call("infino_teleport", {})) == {"ok": False, "error": "unknown tool infino_teleport"}


def test_a_tool_call_missing_its_argument_reports_it(mgr):
    m, _ = mgr
    out = call(m, "infino_remember")
    assert out["ok"] is False and "KeyError" in out["error"]


# -- memory ----------------------------------------------------------------------

def test_a_remembered_row_says_who_where_and_how(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "priya")
    m, _ = make_manager(tmp_path, agent_identity="ops", session_id="s42")
    assert call(m, "infino_remember", content="the VPN is needed for staging") == {"ok": True, "scope": "team"}
    [row] = local_rows(tmp_path, tables.MEMORY)
    assert (row["author"], row["profile"], row["session_id"], row["origin"], row["scope"], row[tables.CHAT]) == (
        "priya", "ops", "s42", "remember", "team", "")


def test_personal_memory_goes_to_the_personal_database_only(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="my dentist is on Tuesday", scope="personal")
    assert local_rows(tmp_path, tables.MEMORY) == []
    assert [r["content"] for r in personal_rows(tmp_path)] == ["my dentist is on Tuesday"]


def test_a_forget_scoped_to_the_team_leaves_personal_memory(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="same words", scope="personal")
    call(m, "infino_remember", content="same words", scope="team")
    out = call(m, "infino_forget", content="same words", scope="team")
    assert out == {"ok": True, "scope": "team", "matched": 1}
    assert [r["content"] for r in personal_rows(tmp_path)] == ["same words"]


def test_a_forget_with_no_scope_reaches_both(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="same words", scope="personal")
    call(m, "infino_remember", content="same words", scope="team")
    assert call(m, "infino_forget", content="same words")["matched"] == 2


def test_a_forget_needs_the_exact_text(mgr, tmp_path):
    m, _ = mgr
    call(m, "infino_remember", content="use pnpm 9")
    assert call(m, "infino_forget", content="use pnpm")["matched"] == 0
    assert len(local_rows(tmp_path, tables.MEMORY)) == 1


def test_a_removed_builtin_memory_is_forgotten(mgr, tmp_path):
    _, p = mgr
    p.on_memory_write("add", "memory", "the build uses make")
    p.on_memory_write("remove", "memory", "", {"previous_content": "the build uses make"})
    p.on_memory_write("remove", "memory", "", None)  # nothing named: nothing to do
    p._drain()
    assert local_rows(tmp_path, tables.MEMORY) == []


def test_a_users_entry_without_a_name_is_kept_as_written(mgr, tmp_path):
    _, p = mgr
    p.on_memory_write("add", "user", "Prefers metric units")
    p._drain()
    assert [r["content"] for r in personal_rows(tmp_path)] == ["Prefers metric units"]


def test_many_background_writes_all_land(mgr, tmp_path):
    _, p = mgr
    for i in range(12):
        p.on_memory_write("add", "memory", f"fact {i}")
    p._drain()
    assert sorted(r["content"] for r in local_rows(tmp_path, tables.MEMORY)) == sorted(f"fact {i}" for i in range(12))


def test_recall_status_is_per_question(mgr):
    m, p = mgr
    call(m, "infino_remember", content="the deploy freeze starts December 20")
    p.prefetch("deploy freeze")
    assert p.recall_status().count == 1 and p.recall_status().provider_label == "Infino"
    assert p.prefetch("   ") == "" and p.recall_status() is None


# -- conversations -----------------------------------------------------------------

def test_a_turn_without_its_messages_is_its_two_texts(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("what time is standup?", "Ten.", session_id="s1")
    p._drain()
    assert [(r["seq"], r["role"], r["content"]) for r in turn_rows(tmp_path)] == [
        (0, "user", "what time is standup?"), (1, "assistant", "Ten.")]


def test_rows_from_every_message_shape(mgr, tmp_path):
    _, p = mgr
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": [{"type": "text", "text": "look at"}, {"type": "image_url", "image_url": {}},
                                     {"type": "text", "text": "this"}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "function": {"name": "terminal", "arguments": "{\"cmd\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "file text"},
        {"role": "tool", "tool_call_id": "b", "name": "terminal", "content": "a.txt"},
        {"role": "assistant", "content": ""},  # nothing said and nothing called: no row
        {"role": "assistant", "content": "done"},
    ]
    p.sync_turn("look at this", "done", session_id="s1", messages=messages)
    p._drain()
    rows = turn_rows(tmp_path)
    assert [(r["seq"], r["role"]) for r in rows] == [(1, "user"), (2, "assistant"), (3, "tool"), (4, "tool"),
                                                    (6, "assistant")]
    assert rows[0]["content"] == "look at\n\nthis"  # an image part leaves an empty line
    assert rows[1]["tool_name"] == "read_file,terminal"
    assert [r["tool_name"] for r in rows[2:4]] == ["read_file", "terminal"]
    assert rows[2]["tool_result"] == "file text" and rows[2]["content"] == ""
    assert rows[1]["platform"] == "cli" and rows[1]["author"] == ""  # only the person's message names them


def test_a_huge_cell_is_cut(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("x" * (provider_mod.MAX_CELL_CHARS + 10), "ok", session_id="s1")
    p._drain()
    assert len(turn_rows(tmp_path)[0]["content"]) == provider_mod.MAX_CELL_CHARS


def test_each_session_counts_its_own_messages(mgr, tmp_path):
    _, p = mgr
    a = [{"role": "user", "content": "a1"}, {"role": "assistant", "content": "a2"}]
    b = [{"role": "user", "content": "b1"}, {"role": "assistant", "content": "b2"}]
    p.sync_turn("a1", "a2", session_id="A", messages=a)
    p.sync_turn("b1", "b2", session_id="B", messages=b)
    p.sync_turn("a3", "a4", session_id="A", messages=a + [{"role": "user", "content": "a3"},
                                                          {"role": "assistant", "content": "a4"}])
    p._drain()
    assert [(r["session_id"], r["seq"]) for r in turn_rows(tmp_path)] == [("A", 0), ("A", 1), ("A", 2), ("A", 3),
                                                                         ("B", 0), ("B", 1)]


def test_a_new_session_id_carries_into_memory_rows(mgr, tmp_path):
    m, p = mgr
    p.on_session_switch("s2")
    call(m, "infino_remember", content="after the switch")
    assert local_rows(tmp_path, tables.MEMORY)[0]["session_id"] == "s2"


def test_session_end_and_shutdown_in_local_mode(mgr, tmp_path):
    _, p = mgr
    p.on_memory_write("add", "memory", "kept")
    p.on_session_end([])
    p.shutdown()
    assert len(local_rows(tmp_path, tables.MEMORY)) == 1


def test_a_published_skill_is_kept_locally_without_an_account(mgr, tmp_path):
    m, _ = mgr
    skill = tmp_path / "skills" / "release-checklist" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL_MD)
    assert call(m, "infino_publish_skill", name="release-checklist")["ok"]
    [row] = local_rows(tmp_path, tables.SKILLS)
    assert row["content"] == SKILL_MD and row["description"] == "Steps before a release."


# -- scope, resumed sessions, and a model that cannot run ------------------------------

@pytest.mark.parametrize("scope", ["Personal", "private", "everyone"])
def test_a_scope_outside_the_enum_never_lands_in_the_teams_database(mgr, tmp_path, scope):
    m, _ = mgr
    out = call(m, "infino_remember", content="my home address is 1 Main St", scope=scope)
    assert out["ok"] is False or local_rows(tmp_path, tables.MEMORY) == []


def test_a_scope_in_another_case_is_understood(mgr, tmp_path):
    m, _ = mgr
    assert call(m, "infino_remember", content="my dentist is Tuesday", scope=" Personal ") == {
        "ok": True, "scope": "personal"}
    assert [r["content"] for r in personal_rows(tmp_path)] == ["my dentist is Tuesday"]


def test_an_unknown_scope_is_refused_by_name(mgr, tmp_path):
    m, _ = mgr
    out = call(m, "infino_forget", content="x", scope="private")
    assert out == {"ok": False, "error": "scope must be 'team' or 'personal', not 'private'"}


def test_a_turn_without_its_messages_continues_the_sessions_count(mgr, tmp_path):
    _, p = mgr
    p.sync_turn("one", "two", session_id="s1")
    p.sync_turn("three", "four", session_id="s1")
    p._drain()
    assert [(r["seq"], r["content"]) for r in turn_rows(tmp_path)] == [(0, "one"), (1, "two"), (2, "three"),
                                                                     (3, "four")]


def test_a_resumed_session_stores_each_message_once(home, tmp_path):
    history = [{"role": "user", "content": "run the tests"}, {"role": "assistant", "content": "2 fail"}]
    _, first = make_manager(tmp_path, session_id="s1")
    first.sync_turn("run the tests", "2 fail", session_id="s1", messages=history)
    first.on_session_end(history)
    # `hermes chat --resume s1`: a new process, the same session, its history first.
    _, resumed = make_manager(tmp_path, session_id="s1")
    messages = history + [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "fixed"}]
    resumed.sync_turn("fix it", "fixed", session_id="s1", messages=messages)
    resumed._drain()
    assert [(r["seq"], r["content"]) for r in turn_rows(tmp_path)] == [
        (0, "run the tests"), (1, "2 fail"), (2, "fix it"), (3, "fixed")]


def test_a_remember_that_reports_failure_saved_nothing(mgr, tmp_path):
    m, _ = mgr

    def broken(texts):
        raise OSError("no network to download the model")

    embed.use(broken)
    out = call(m, "infino_remember", content="the deploy freeze starts December 20")
    assert out["ok"] or local_rows(tmp_path, tables.MEMORY) == []


def test_a_memory_saved_while_the_model_could_not_run_is_embedded_at_the_next_start(home, tmp_path):
    def broken(texts):
        raise OSError("no network to download the model")

    m, _ = make_manager(tmp_path)
    embed.use(broken)
    assert call(m, "infino_remember", content="the deploy freeze starts December 20")["ok"]
    store = local(tmp_path)
    assert store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == 0
    embed.use(fake_embed)
    backend._backends.clear()
    make_manager(tmp_path)  # the next start
    assert store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == 1


def test_a_start_without_the_model_still_starts(home, tmp_path):
    def broken(texts):
        raise OSError("no network to download the model")

    embed.use(broken)
    m, p = make_manager(tmp_path)
    assert call(m, "infino_remember", content="the deploy freeze starts December 20")["ok"]
    embed.use(fake_embed)
    assert "deploy freeze" in p.prefetch("deploy freeze")

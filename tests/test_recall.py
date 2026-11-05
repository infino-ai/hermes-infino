"""Recall: which source answers, what the block holds, and how it is packed."""

from __future__ import annotations

import pytest

from hermes_infino import chats, recall, tables
from hermes_infino.client import InfinoClient
from hermes_infino.local import LocalStore
from hermes_infino.recall import Recaller, Source

from .helpers import TEAM_DB, requests_to

# Long enough that a few of them fill the block's character budget.
LONG = 900
GROUP = chats.Chat(key="telegram:-100", group=True, owner_writes=False)


@pytest.fixture
def team_local(home, tmp_path) -> LocalStore:
    s = LocalStore(tmp_path / "db")
    s.ensure_tables(with_vectors=False)
    return s


@pytest.fixture
def personal_local(home, tmp_path) -> LocalStore:
    s = LocalStore(tmp_path / "personal")
    s.ensure_tables(with_vectors=False, only=(tables.MEMORY,))
    return s


@pytest.fixture
def cloud(platform) -> InfinoClient:
    c = InfinoClient("k", platform.url, retries=0)
    tables.ensure_all(c, TEAM_DB)
    return c


def test_a_question_with_nothing_recalled_is_empty(team_local):
    assert Recaller([Source(recall.TEAM, team_local)]).recall("anything") == ("", 0)


def test_the_block_names_its_scope_and_author(team_local, personal_local):
    team_local.append(tables.MEMORY, [{"content": "deploys freeze Dec 20", "author": "alice", "scope": "team"}])
    personal_local.append(tables.MEMORY, [{"content": "deploys make me nervous", "scope": "personal"}])
    block, n = Recaller([Source(recall.TEAM, team_local), Source(recall.PERSONAL, personal_local)]).recall("deploys")
    assert n == 2
    assert block.splitlines()[:2] == [recall.HEADING, recall.PREAMBLE]
    assert "- [team] deploys freeze Dec 20 (from alice)" in block
    assert "- [personal] deploys make me nervous" in block and "nervous (from" not in block


def test_a_chats_own_memory_is_labelled_this_chat(team_local):
    team_local.append(tables.MEMORY, [{"content": "standup moved", tables.CHAT: GROUP.key}])
    block, _ = Recaller([Source(recall.TEAM, team_local)]).recall("standup", GROUP)
    assert "- [this chat] standup moved" in block


def test_the_block_fits_its_character_budget(team_local):
    team_local.append(tables.MEMORY, [{"content": f"deploy {i} " + "x" * LONG} for i in range(recall.RECALL_K)])
    block, n = Recaller([Source(recall.TEAM, team_local)]).recall("deploy")
    body = block.split("\n", 2)[2]
    assert 0 < n < recall.RECALL_K and len(body) <= recall.RECALL_MAX_CHARS + n


def test_each_source_gives_at_most_k(team_local):
    team_local.append(tables.MEMORY, [{"content": f"deploy note {i}"} for i in range(recall.RECALL_K + 4)])
    assert Recaller([Source(recall.TEAM, team_local)]).recall("deploy note")[1] == recall.RECALL_K


def test_a_group_chat_never_reads_the_personal_source(team_local, personal_local):
    personal_local.append(tables.MEMORY, [{"content": "my deploy password hint", "scope": "personal"}])
    block, n = Recaller([Source(recall.TEAM, team_local), Source(recall.PERSONAL, personal_local)]).recall(
        "deploy password", GROUP)
    assert (block, n) == ("", 0)


def test_a_broken_local_database_recalls_nothing_and_never_raises(home, tmp_path):
    broken = LocalStore(tmp_path / "never-made")  # no tables
    assert Recaller([Source(recall.TEAM, broken)]).recall("anything") == ("", 0)


# -- signed in ----------------------------------------------------------------------

def cloud_source(local: LocalStore, mark=None) -> Source:
    return Source(recall.TEAM, local, TEAM_DB, (lambda: mark))


def test_the_cloud_answers_and_this_machine_adds_what_it_lacks(team_local, cloud, platform):
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "c1", "stored_ms": 1, "content": "deploy freeze from the cloud"}])
    team_local.append(tables.MEMORY, [{"row_id": "c1", "content": "deploy freeze from the cloud"},
                                      {"row_id": "n1", "content": "deploy freeze written just now"}])
    block, n = Recaller([cloud_source(team_local)], cloud).recall("deploy freeze")
    assert n == 2 and block.count("from the cloud") == 1 and "written just now" in block


def test_local_rows_the_cloud_already_has_are_left_out(team_local, cloud):
    team_local.append(tables.MEMORY, [{"content": "deploy freeze, synced long ago"}])
    synced_past = (2 ** 62, "")
    assert Recaller([cloud_source(team_local, synced_past)], cloud).recall("deploy freeze") == ("", 0)


def test_the_cloud_down_leaves_this_machine_to_answer(team_local, cloud, platform):
    team_local.append(tables.MEMORY, [{"content": "release 2.3 rolled back"}])
    platform.close()
    synced_past = (2 ** 62, "")
    block, n = Recaller([cloud_source(team_local, synced_past)], cloud).recall("release 2.3")
    assert n == 1 and "rolled back" in block


def test_a_transient_refusal_falls_back_once_and_tries_hybrid_again(team_local, cloud, platform):
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploy freeze"}])
    r = Recaller([cloud_source(team_local)], cloud)
    platform.hybrid_statuses = [503]
    assert "deploy freeze" in r.recall("deploy freeze")[0]
    before = requests_to(platform, "hybrid_search")
    r.recall("deploy freeze")
    assert requests_to(platform, "hybrid_search") == before + 1


def test_a_lasting_refusal_turns_the_database_keyword_only(team_local, cloud, platform):
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploy freeze"}])
    r = Recaller([cloud_source(team_local)], cloud)
    platform.hybrid_statuses = [403]
    r.recall("deploy freeze")
    before = requests_to(platform, "hybrid_search")
    assert "deploy freeze" in r.recall("deploy freeze")[0]
    assert requests_to(platform, "hybrid_search") == before


def test_an_unexpected_cloud_error_falls_back_to_this_machine(team_local, cloud, platform):
    team_local.append(tables.MEMORY, [{"content": "deploy freeze kept here"}])
    platform.hybrid_statuses = [500]
    block, _ = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud).recall("deploy freeze")
    assert "kept here" in block


def test_another_chats_cloud_rows_are_dropped(team_local, cloud):
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "standup moved", tables.CHAT: "telegram:-200"},
                                          {"row_id": "b", "content": "standup is at ten", tables.CHAT: ""}])
    block, n = Recaller([cloud_source(team_local)], cloud).recall("standup", GROUP)
    assert n == 1 and "at ten" in block and "moved" not in block


@pytest.mark.parametrize("row, mark, past", [
    ({tables.STORED: 5, tables.ROW_ID: "b"}, None, True),
    ({tables.STORED: 5, tables.ROW_ID: "b"}, (5, "a"), True),
    ({tables.STORED: 5, tables.ROW_ID: "b"}, (5, "b"), False),
    ({tables.STORED: 4, tables.ROW_ID: "z"}, (5, "a"), False),
    ({}, (0, ""), False),
])
def test_a_row_is_past_a_mark_in_commit_order(row, mark, past):
    assert recall._past(row, mark) is past

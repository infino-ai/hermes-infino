"""Recall: which source answers, what the block holds, and how it is packed."""

from __future__ import annotations

import time

import pytest

from hermes_infino import chats, recall, tables
from hermes_infino.client import InfinoClient, InfinoError, _rows, rerank_unavailable
from hermes_infino.local import LocalStore
from hermes_infino.recall import Recaller, Source

from .helpers import TEAM_DB, requests_to

# Long enough that a few of them fill the block's character budget.
LONG = 900
# A recall budget, and a cloud slower than it.
SHORT_BUDGET_S = 0.3
SLOW_CLOUD_S = 1.5
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


def test_a_slow_cloud_leaves_this_machine_to_answer_in_time(team_local, personal_local, cloud, monkeypatch):
    team_local.append(tables.MEMORY, [{"content": "deploy freeze kept here"}])
    monkeypatch.setattr(recall, "RECALL_BUDGET_S", SHORT_BUDGET_S)
    monkeypatch.setattr(Recaller, "_cloud_rows", lambda self, src, query: time.sleep(SLOW_CLOUD_S) or [])
    sources = [cloud_source(team_local, (2 ** 62, "")), Source(recall.PERSONAL, personal_local, "person-db", lambda: None)]
    started = time.monotonic()
    block, _ = Recaller(sources, cloud).recall("deploy freeze")
    assert "kept here" in block
    assert time.monotonic() - started < SLOW_CLOUD_S  # the two sources' clouds were waited on together, briefly


def test_a_400_about_the_query_does_not_turn_hybrid_off(team_local, cloud, platform):
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploy freeze"}])
    r = Recaller([cloud_source(team_local)], cloud)
    platform.hybrid_statuses = [400]  # "scripted": nothing about vectors
    r.recall("deploy freeze")
    before = requests_to(platform, "hybrid_search")
    r.recall("deploy freeze")
    assert requests_to(platform, "hybrid_search") == before + 1


def test_personal_memory_shows_only_to_its_person(team_local, personal_local):
    personal_local.append(tables.MEMORY, [{"content": "dentist on Tuesday", "author": "ana", "scope": "personal"},
                                          {"content": "dentist on Friday", "author": "ben", "scope": "personal"}])
    r = Recaller([Source(recall.TEAM, team_local), Source(recall.PERSONAL, personal_local)])
    ana, _ = r.recall("dentist", people=("ana",))
    assert "Tuesday" in ana and "Friday" not in ana
    assert r.recall("dentist", people=())[1] == 0  # a sender nobody named reads no one's
    assert r.recall("dentist", people=("ana", "uid:1000"))[1] == 1  # an owner known by several names
    assert r.recall("dentist")[1] == 2  # no one named: everyone's


@pytest.mark.parametrize("row, mark, past", [
    ({tables.STORED: 5, tables.ROW_ID: "b"}, None, True),
    ({tables.STORED: 5, tables.ROW_ID: "b"}, (5, "a"), True),
    ({tables.STORED: 5, tables.ROW_ID: "b"}, (5, "b"), False),
    ({tables.STORED: 4, tables.ROW_ID: "z"}, (5, "a"), False),
    ({}, (0, ""), False),
])
def test_a_row_is_past_a_mark_in_commit_order(row, mark, past):
    assert recall.past(row, mark) is past


# -- ranked at recall, superseded kept ---------------------------------------------

# A day after the epoch, in milliseconds: the date a superseded memory's line names.
A_DAY_MS = 86_400_000


def by_words(*best_first: str):
    """A grader that scores rows by which of these words their text holds, the
    first word highest: it orders rows against their search score."""
    def grade(question, row):
        text = str(row.get("content", ""))
        return next((len(best_first) - i for i, w in enumerate(best_first) if w in text), 0)
    return grade


def lines(block: str) -> list[str]:
    return block.splitlines()[2:]


def test_the_cloud_ranks_recall_for_the_question_and_its_order_stands(team_local, cloud, platform):
    platform.rerank_supported = True
    platform.grade = by_words("Thursday", "Tuesday")
    # The plain search ranks "deploys deploys" first; the grader prefers Thursday.
    cloud.append(TEAM_DB, tables.MEMORY, [
        {"row_id": "a", "content": "deploys deploys on Tuesday"},
        {"row_id": "b", "content": "deploys moved to Thursday"}])
    block, n = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud).recall("when are deploys")
    assert n == 2
    assert lines(block) == ["- [team] deploys moved to Thursday", "- [team] deploys deploys on Tuesday"]
    [asked] = platform.reranked
    assert asked["rerank"] == {"question": "when are deploys", "k": recall.RECALL_K * recall.CHAT_FILTER_DEPTH}
    assert {tables.SUPERSEDED, tables.SUPERSEDED_BY} <= set(asked["projection"])


def test_a_deployment_without_the_triage_pass_answers_from_the_plain_search_and_is_not_asked_again(
        team_local, cloud, platform):
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploy freeze"}])
    r = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud)
    assert "deploy freeze" in r.recall("deploy freeze")[0]
    assert len(platform.reranked) == 1  # refused with 501
    assert "deploy freeze" in r.recall("deploy freeze")[0]
    assert len(platform.reranked) == 1  # remembered: no second ask


def test_a_worker_built_without_the_triage_pass_is_remembered_too(team_local, cloud, platform):
    platform.rerank_supported = True
    platform.rerank_statuses = [400]
    # The fake's scripted 400 says nothing of rerank: only the worker's own words turn it off.
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploy freeze"}])
    r = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud)
    r.recall("deploy freeze")
    asked = len(platform.reranked)
    r.recall("deploy freeze")
    assert len(platform.reranked) > asked  # a 400 about this request is not "unavailable"
    assert rerank_unavailable(InfinoError(400, "rerank is not available on this deployment; send the query without"))
    assert rerank_unavailable(InfinoError(501, "anything"))
    assert not rerank_unavailable(InfinoError(400, "scripted"))
    assert not rerank_unavailable(InfinoError(503, "rerank is not available"))


def test_a_ranking_that_fails_once_falls_back_and_is_asked_again(team_local, cloud, platform):
    platform.rerank_supported = True
    platform.rerank_statuses = [500]
    platform.grade = by_words("Thursday")
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploys deploys on Tuesday"},
                                          {"row_id": "b", "content": "deploys moved to Thursday"}])
    r = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud)
    first, _ = r.recall("deploys")
    assert lines(first)[0] == "- [team] deploys deploys on Tuesday"  # the plain search's order
    second, _ = r.recall("deploys")
    assert lines(second)[0] == "- [team] deploys moved to Thursday"  # ranked again
    assert len(platform.reranked) == 2


def test_a_ranking_slower_than_the_budget_leaves_the_plain_search_to_answer_in_time(
        team_local, cloud, platform, monkeypatch):
    monkeypatch.setattr(recall, "RECALL_BUDGET_S", SHORT_BUDGET_S)
    platform.rerank_supported = True
    platform.rerank_delay_s = SLOW_CLOUD_S
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploy freeze"}])
    started = time.monotonic()
    block, n = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud).recall("deploy freeze")
    assert n == 1 and "deploy freeze" in block
    assert time.monotonic() - started < SLOW_CLOUD_S


def test_a_ranked_recall_still_shows_only_what_the_chat_may_see(team_local, cloud, platform):
    platform.rerank_supported = True
    platform.grade = by_words("moved")
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "standup moved", tables.CHAT: "telegram:-200"},
                                          {"row_id": "b", "content": "standup is at ten", tables.CHAT: ""}])
    block, n = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud).recall("standup", GROUP)
    assert n == 1 and lines(block) == ["- [team] standup is at ten"]


def test_a_ranked_recall_of_personal_memory_shows_only_its_person(team_local, personal_local, cloud, platform):
    platform.rerank_supported = True
    tables.ensure_all(cloud, "dana-own", personal=True)
    cloud.append("dana-own", tables.MEMORY, [
        {"row_id": "p1", "content": "deploy password hint is blue", "author": "dana", "scope": "personal"},
        {"row_id": "p2", "content": "deploy password hint is red", "author": "sam", "scope": "personal"}])
    source = Source(recall.PERSONAL, personal_local, "dana-own", lambda: (2 ** 62, ""))
    block, n = Recaller([source], cloud).recall("deploy password hint", people=("dana",))
    assert n == 1 and "blue" in block and "red" not in block


def test_a_keyword_only_database_is_ranked_too(team_local, cloud, platform):
    platform.rerank_supported = True
    platform.grade = by_words("Thursday")
    cloud.append(TEAM_DB, tables.MEMORY, [{"row_id": "a", "content": "deploys deploys on Tuesday"},
                                          {"row_id": "b", "content": "deploys moved to Thursday"}])
    r = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud)
    r._keyword_only.add(TEAM_DB)
    block, _ = r.recall("deploys")
    assert lines(block)[0] == "- [team] deploys moved to Thursday"
    assert all("field_name" in body for body in platform.reranked)  # bm25_search's body


def test_a_superseded_memory_is_recalled_marked_with_when_and_what_replaced_it(team_local):
    team_local.append(tables.MEMORY, [
        {"content": "deploys are on Tuesday", tables.SUPERSEDED: A_DAY_MS, tables.SUPERSEDED_BY: "deploys are on Thursday"},
        {"content": "deploys are on Thursday"},
        {"content": "deploys need a ticket", tables.SUPERSEDED: A_DAY_MS, tables.SUPERSEDED_BY: ""}])
    block, n = Recaller([Source(recall.TEAM, team_local)]).recall("deploys")
    assert n == 3
    assert lines(block)[0] == "- [team] deploys are on Thursday"  # current first when nothing graded them
    assert "[superseded 1970-01-02; replaced by: deploys are on Thursday]" in block
    assert "deploys need a ticket [superseded 1970-01-02; no longer in memory]" in block
    assert "superseded" in recall.PREAMBLE


def test_superseded_memories_only_fill_what_current_ones_leave(team_local):
    team_local.append(tables.MEMORY, [{"content": f"deploy note {i} old", tables.SUPERSEDED: A_DAY_MS}
                                      for i in range(recall.RECALL_K)])
    team_local.append(tables.MEMORY, [{"content": f"deploy note {i}"} for i in range(recall.RECALL_K)])
    block, n = Recaller([Source(recall.TEAM, team_local)]).recall("deploy note")
    assert n == recall.RECALL_K and "superseded" not in "\n".join(lines(block))


def test_the_grader_may_put_a_superseded_memory_first(team_local, cloud, platform):
    platform.rerank_supported = True
    platform.grade = by_words("Tuesday")
    cloud.append(TEAM_DB, tables.MEMORY, [
        {"row_id": "a", "content": "deploys on Tuesday, staging first", tables.SUPERSEDED: A_DAY_MS,
         tables.SUPERSEDED_BY: "deploys on Thursday"},
        {"row_id": "b", "content": "deploys on Thursday"}])
    block, _ = Recaller([cloud_source(team_local, (2 ** 62, ""))], cloud).recall("what did we say about staging")
    assert lines(block)[0].startswith("- [team] deploys on Tuesday, staging first [superseded")


def test_a_long_replacement_is_cut_in_the_note(team_local):
    team_local.append(tables.MEMORY, [{"content": "deploys are on Tuesday", tables.SUPERSEDED: A_DAY_MS,
                                       tables.SUPERSEDED_BY: "deploys " + "y" * 500}])
    block, _ = Recaller([Source(recall.TEAM, team_local)]).recall("deploys")
    note = block.split("replaced by: ", 1)[1]
    assert note.endswith("…]") and len(note) <= recall.REPLACED_BY_CHARS + len("…]")


def test_a_reranked_response_is_read_under_rows():
    assert _rows({"rows": [{"a": 1}], "rerank": {"graded": 1}}) == [{"a": 1}]
    assert _rows({"data": [{"a": 2}]}) == [{"a": 2}]
    assert _rows([{"a": 3}]) == [{"a": 3}]

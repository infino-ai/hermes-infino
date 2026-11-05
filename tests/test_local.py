"""The local database, on the real Infino engine: writes, the order the
mirror reads in, deletes, and local recall."""

from __future__ import annotations

import json
import time

import infino
import pyarrow as pa
import pytest

from hermes_infino import embed, local as local_mod, mirror, tables
from hermes_infino.local import VECTORS, LocalStore, arrow_schema

from .helpers import fake_embed, wait_for

# Words the stand-in model below treats as one meaning, with no word in common.
SYNONYMS = {"car": "automobile", "automobile": "automobile"}
# A row count past one append batch, once the batch is made small.
SMALL_BATCH = 2
MANY_ROWS = 5
# Text whose SQL literal needs its quotes doubled.
QUOTED = "it's Bob's"
# The mirror's settle window as shipped (the tests' fixture turns it off).
SETTLE_MS = mirror.SETTLE_MS


@pytest.fixture
def store(home, tmp_path) -> LocalStore:
    s = LocalStore(tmp_path / "db")
    s.ensure_tables(with_vectors=True)
    return s


@pytest.fixture
def keyword_store(home, tmp_path) -> LocalStore:
    s = LocalStore(tmp_path / "db")
    s.ensure_tables(with_vectors=False)
    return s


def synonym_embed(texts: list[str]) -> list[list[float]]:
    """The stand-in model, with SYNONYMS mapped to one word first."""
    return fake_embed([" ".join(SYNONYMS.get(w, w) for w in t.lower().split()) for t in texts])


def ids(rows) -> list[str]:
    return [r[tables.ROW_ID] for r in rows]


def test_the_local_tables_are_the_clouds_without_the_embedding():
    names = arrow_schema(tables.MEMORY).names
    assert tables.EMBEDDING_COLUMN not in names
    assert names == [f["name"] for f in tables.MEMORY_SCHEMA if f["type"] != "embedding"]
    vector = arrow_schema(VECTORS).field(tables.EMBEDDING_COLUMN)
    assert vector.type == pa.list_(pa.float32(), embed.DIM) and not vector.nullable


def test_tables_are_made_once(store):
    store.ensure_tables(with_vectors=True)
    assert sorted(store.list_tables()) == sorted((*tables.ALL, VECTORS))


def test_only_the_tables_asked_for_are_made(home, tmp_path):
    s = LocalStore(tmp_path / "personal")
    s.ensure_tables(with_vectors=False, only=(tables.MEMORY,))
    assert s.list_tables() == [tables.MEMORY] and not s.has_vectors()


def test_rows_get_an_id_unless_they_have_one(store):
    store.append(tables.MEMORY, [{"content": "a"}, {"content": "b", tables.ROW_ID: "mine"}])
    rows = store.query_sql("SELECT row_id, content FROM memory ORDER BY content")
    assert rows[1][tables.ROW_ID] == "mine"
    assert len(rows[0][tables.ROW_ID]) == 36  # a UUID


def test_a_batch_is_stamped_once_as_it_commits(store):
    store.append(tables.CONVERSATIONS, [{"content": str(i)} for i in range(3)])
    stamps = {r[tables.STORED] for r in store.query_sql("SELECT stored_ms FROM conversations")}
    assert len(stamps) == 1 and stamps.pop() > 0


def test_many_rows_go_in_several_batches(store, monkeypatch):
    monkeypatch.setattr(local_mod, "APPEND_BATCH", SMALL_BATCH)
    store.append(tables.MEMORY, [{"content": f"memory {i}"} for i in range(MANY_ROWS)])
    assert store.query_sql("SELECT count(*) AS n FROM memory")[0]["n"] == MANY_ROWS
    assert store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == MANY_ROWS


def test_nothing_to_append_opens_nothing(home, tmp_path):
    LocalStore(tmp_path / "never").append(tables.MEMORY, [])
    assert not (tmp_path / "never").exists()


def test_text_round_trips_whole(store):
    text = "naïve café ✓ — it's \"quoted\"\nnext line\t tab"
    store.append(tables.MEMORY, [{"content": text}])
    assert store.query_sql("SELECT content FROM memory")[0]["content"] == text


def test_rows_after_follow_commit_order_then_id(store):
    store.append(tables.CONVERSATIONS, [{tables.ROW_ID: i, "content": i} for i in ("c", "a", "b")])
    [stored] = {r[tables.STORED] for r in store.query_sql("SELECT stored_ms FROM conversations")}
    assert ids(store.rows_after(tables.CONVERSATIONS, None, stored, 10)) == ["a", "b", "c"]
    assert ids(store.rows_after(tables.CONVERSATIONS, (stored, "a"), stored, 10)) == ["b", "c"]
    assert ids(store.rows_after(tables.CONVERSATIONS, (stored, "c"), stored, 10)) == []
    assert ids(store.rows_after(tables.CONVERSATIONS, None, stored, 2)) == ["a", "b"]
    assert store.rows_after(tables.CONVERSATIONS, None, stored - 1, 10) == []  # too new for this sync


def test_rows_after_leave_out_the_local_vectors(store):
    store.append(tables.MEMORY, [{"content": "x"}])
    [row] = store.rows_after(tables.MEMORY, None, 2 ** 62, 10)
    assert tables.EMBEDDING_COLUMN not in row and row["content"] == "x"


def test_a_delete_returns_the_rows_it_deleted(store):
    store.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": QUOTED}, {tables.ROW_ID: "2", "content": "y"}])
    gone = store.delete(tables.MEMORY, f"content = {tables.sql_string(QUOTED)}")
    assert ids(gone) == ["1"]
    assert ids(store.query_sql("SELECT row_id FROM memory")) == ["2"]
    assert store.delete(tables.MEMORY, "content = 'nothing like it'") == []


def test_a_deleted_memory_takes_its_vector_with_it(store):
    store.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "a"}, {tables.ROW_ID: "2", "content": "b"}])
    store.delete(tables.MEMORY, "row_id = '1'")
    assert ids(store.query_sql(f"SELECT row_id FROM {VECTORS}")) == ["2"]


def test_vectors_can_be_dropped_twice(store):
    store.drop_vectors()
    store.drop_vectors()
    assert not store.has_vectors()
    store.append(tables.MEMORY, [{"content": "no vector now"}])  # no embedding once dropped
    assert store.query_sql("SELECT count(*) AS n FROM memory")[0]["n"] == 1


def test_another_connection_sees_a_commit(store, tmp_path):
    other = LocalStore(tmp_path / "db")  # as another process on the same profile
    store.append(tables.MEMORY, [{"content": "seen elsewhere"}])
    assert other.query_sql("SELECT content FROM memory")[0]["content"] == "seen elsewhere"


# -- recall --------------------------------------------------------------------

PROJECTION = [tables.ROW_ID, "content", "score"]


def test_keyword_recall_without_vectors(keyword_store):
    keyword_store.append(tables.MEMORY, [{"content": "the deploy freeze starts December 20"},
                                         {"content": "lunch is at noon"}])
    hits = keyword_store.search_memory("deploy freeze", 5, PROJECTION)
    assert [h["content"] for h in hits] == ["the deploy freeze starts December 20"]


def test_the_vector_leg_finds_a_memory_with_no_word_in_common(store):
    embed.use(synonym_embed)
    store.append(tables.MEMORY, [{"content": "automobile"}, {"content": "lunch is at noon"}])
    hits = store.search_memory("car", 1, PROJECTION)
    assert [h["content"] for h in hits] == ["automobile"]
    assert hits[0]["score"] > 0


def test_a_memory_found_both_ways_ranks_first(store):
    store.append(tables.MEMORY, [{"content": "rollback rule two percent"}, {"content": "rollback owner is priya"},
                                 {"content": "lunch"}])
    hits = store.search_memory("rollback rule", 3, PROJECTION)
    assert hits[0]["content"] == "rollback rule two percent"
    assert hits == sorted(hits, key=lambda h: -h["score"])


def test_recall_returns_at_most_k(store):
    store.append(tables.MEMORY, [{"content": f"deploy note {i}"} for i in range(6)])
    assert len(store.search_memory("deploy note", 4, PROJECTION)) == 4


def test_a_vector_whose_memory_is_gone_is_not_recalled(store):
    store.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "deploy freeze"}])
    store._connect().open_table(tables.MEMORY).delete("row_id = '1'")  # the vector is left behind
    assert store.search_memory("deploy freeze", 5, PROJECTION) == []


def test_a_memory_saved_while_the_model_cannot_run_is_kept_and_found_by_keyword(store):
    def broken(texts):
        raise RuntimeError("model unavailable")

    embed.use(broken)
    store.append(tables.MEMORY, [{"content": "deploy freeze starts December 20"}])
    embed.use(fake_embed)
    assert [h["content"] for h in store.search_memory("deploy freeze", 5, PROJECTION)] == [
        "deploy freeze starts December 20"]
    assert store.backfill_vectors() == 1
    assert store.backfill_vectors() == 0  # each memory once
    assert store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == 1


def test_before_the_model_loads_recall_is_keyword_and_writes_skip_vectors(store):
    embed.use(None)  # not loaded yet
    store.append(tables.MEMORY, [{"content": "the deploy freeze starts December 20"}])
    assert [h["content"] for h in store.search_memory("deploy freeze", 5, PROJECTION)] == [
        "the deploy freeze starts December 20"]
    assert store.query_sql(f"SELECT count(*) AS n FROM {VECTORS}")[0]["n"] == 0
    assert store.backfill_vectors() == 0  # nothing to embed with yet
    embed.use(fake_embed)
    assert store.backfill_vectors() == 1


def test_a_model_that_fails_on_a_question_leaves_the_keyword_hits(store):
    store.append(tables.MEMORY, [{"content": "the deploy freeze starts December 20"}])

    def broken(texts):
        raise RuntimeError("model crashed")

    embed.use(broken)
    assert [h["content"] for h in store.search_memory("deploy freeze", 5, PROJECTION)] == [
        "the deploy freeze starts December 20"]


def test_a_failed_model_load_is_not_retried_at_once(monkeypatch):
    loads = []

    def offline():
        loads.append(1)
        raise OSError("no network")

    embed.use(None)
    monkeypatch.setattr(embed, "_load", offline)
    embed.load_in_background()
    assert wait_for(lambda: embed._failed_at is not None)
    embed.load_in_background()
    time.sleep(0.2)
    assert loads == [1] and embed.ready() is None


def test_a_model_load_calls_back_once_ready(monkeypatch):
    called = []
    embed.use(None)
    monkeypatch.setattr(embed, "_load", lambda: fake_embed)
    embed.load_in_background(lambda: called.append(1))
    assert wait_for(lambda: called == [1]) and embed.ready() is fake_embed


def test_a_slow_commit_leaves_a_note_for_the_next_sync(store, monkeypatch):
    clock = iter([1_000, 1_000 + local_mod.SLOW_COMMIT_MS + 1])
    monkeypatch.setattr(local_mod, "_now_ms", lambda: next(clock))
    store.append(tables.CONVERSATIONS, [{"content": "slow"}])
    [note] = [json.loads(line) for line in (store.path.parent / local_mod.LATE_COMMITS_FILE).read_text().splitlines()]
    assert note == {"store": "db", "table": tables.CONVERSATIONS, "stored_ms": 1_000}


def test_a_slow_commit_is_still_well_inside_the_settle_window():
    assert local_mod.SLOW_COMMIT_MS < SETTLE_MS


def test_without_vectors_there_is_nothing_to_backfill(keyword_store):
    keyword_store.append(tables.MEMORY, [{"content": "x"}])
    assert keyword_store.backfill_vectors() == 0


def test_an_unrelated_question_recalls_nothing(store):
    store.append(tables.MEMORY, [{"content": "the deploy freeze starts December 20"}])
    assert store.search_memory("lunch menu tomorrow", 5, PROJECTION) == []


def test_a_clock_stepped_back_never_stamps_behind_a_committed_row(store, monkeypatch):
    store.append(tables.CONVERSATIONS, [{"content": "first"}])
    [first] = store.query_sql("SELECT stored_ms FROM conversations")
    monkeypatch.setattr(local_mod, "_now_ms", lambda: first[tables.STORED] - 60_000)
    store.append(tables.CONVERSATIONS, [{"content": "after the clock moved back"}])
    later = store.query_sql("SELECT stored_ms FROM conversations WHERE content = 'after the clock moved back'")
    assert later[0][tables.STORED] > first[tables.STORED]


def test_a_delete_takes_only_the_rows_it_found(store, monkeypatch):
    store.append(tables.MEMORY, [{tables.ROW_ID: "1", "content": "same words"}])
    real = store.query_sql

    def another_process_commits_meanwhile(sql):
        out = real(sql)
        if sql.startswith("SELECT row_id FROM memory WHERE content"):
            LocalStore(store.path).append(tables.MEMORY, [{tables.ROW_ID: "2", "content": "same words"}])
        return out

    monkeypatch.setattr(store, "query_sql", another_process_commits_meanwhile)
    assert ids(store.delete(tables.MEMORY, "content = 'same words'")) == ["1"]
    assert ids(real("SELECT row_id FROM memory")) == ["2"]  # still here, and still to be pushed


def test_rows_after_and_their_count_agree(store):
    store.append(tables.CONVERSATIONS, [{tables.ROW_ID: i, "content": i} for i in ("a", "b", "c")])
    [stored] = {r[tables.STORED] for r in store.query_sql("SELECT stored_ms FROM conversations")}
    for mark in (None, (stored, "a"), (stored, "c")):
        assert store.count_after(tables.CONVERSATIONS, mark, stored) == len(
            store.rows_after(tables.CONVERSATIONS, mark, stored, 100))
    assert store.has_row(tables.CONVERSATIONS, "b") and not store.has_row(tables.CONVERSATIONS, "nope")


# -- commit races ----------------------------------------------------------------

def test_a_lost_commit_race_is_retried(monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr(local_mod.time, "sleep", waits.append)

    def op():
        calls.append(1)
        if len(calls) < 3:
            raise infino.ConflictError("another writer committed first")
        return "done"

    assert LocalStore("unused")._retry(op) == "done"
    assert waits == [local_mod.CONFLICT_BACKOFF_S, local_mod.CONFLICT_BACKOFF_S * 2]


def test_a_commit_race_lost_every_time_raises(monkeypatch):
    monkeypatch.setattr(local_mod.time, "sleep", lambda s: None)

    def op():
        raise infino.ConflictError("always busy")

    with pytest.raises(infino.ConflictError):
        LocalStore("unused")._retry(op)


def test_other_engine_errors_are_not_retried(monkeypatch):
    calls = []

    def op():
        calls.append(1)
        raise ValueError("bad")

    with pytest.raises(ValueError):
        LocalStore("unused")._retry(op)
    assert len(calls) == 1

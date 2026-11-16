"""Long conversation text goes to the cloud as chunks, cut by the Infino
client's chunker, never cut short: the rows of one message, how they are
sent, and what happens without a client."""

from __future__ import annotations

import pytest

from hermes_infino import chunks, config, mirror, tables
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore
from hermes_infino.mirror import Mirror

from .helpers import TEAM_DB, sign_in

# Lines in a long tool result, and how many lines the stand-in chunker puts in a chunk.
LINES = 300
PER_CHUNK = 100


def long_text(tag: str = "line") -> str:
    return "\n".join(f"{tag} {i}: the shard map is rebuilt while a reader holds it" for i in range(LINES))


def fake_chunker(requests: list[dict]) -> list[dict]:
    """The client's `chunk` command, as it answers: one line per chunk, in order."""
    out = []
    for r in requests:
        lines = r["text"].split("\n")
        for n, start in enumerate(range(0, len(lines), PER_CHUNK)):
            out.append({"id": r["id"], "chunk": n, "start_line": start + 1,
                        "end_line": min(start + PER_CHUNK, len(lines)),
                        "content": "\n".join(lines[start:start + PER_CHUNK]), "path": r["path"]})
    return out


@pytest.fixture
def m(home, platform, tmp_path) -> Mirror:
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    out = Mirror(sign_in(tmp_path, platform), team, personal)
    out.chunker = fake_chunker
    out.ensure_cloud_tables()
    return out


def test_a_short_message_goes_as_one_row():
    rows = [{"row_id": "r", "content": "short", "tool_result": None}]
    assert chunks.expand(tables.CONVERSATIONS, rows, fake_chunker) == rows


def test_a_long_tool_result_goes_as_its_chunks_sharing_the_row_id():
    row = {"row_id": "r", "role": "tool", "content": "", "tool_name": "read_file", "tool_result": long_text()}
    out = chunks.expand(tables.CONVERSATIONS, [row], fake_chunker)
    assert [r[chunks.CHUNK_COLUMN] for r in out] == [0, 1, 2]
    assert {r["row_id"] for r in out} == {"r"} and {r["tool_name"] for r in out} == {"read_file"}
    assert "\n".join(r["tool_result"] for r in out) == row["tool_result"]  # nothing dropped


def test_both_long_columns_of_one_message_go_as_chunks():
    row = {"row_id": "r", "content": long_text("said"), "tool_result": long_text("got")}
    out = chunks.expand(tables.CONVERSATIONS, [row], fake_chunker)
    assert [r[chunks.CHUNK_COLUMN] for r in out] == list(range(5))
    assert out[0]["content"].startswith("said 0") and out[0]["tool_result"].startswith("got 0")
    assert "\n".join(r["content"] for r in out if r["content"]) == row["content"]
    assert "\n".join(r["tool_result"] for r in out if r["tool_result"]) == row["tool_result"]


def test_without_a_chunker_or_when_it_fails_long_text_goes_whole():
    row = {"row_id": "r", "tool_result": long_text()}
    assert chunks.expand(tables.CONVERSATIONS, [row], None) == [row]

    def broken(requests):
        raise RuntimeError("unknown command 'chunk'")

    assert chunks.expand(tables.CONVERSATIONS, [row], broken) == [row]


def test_other_tables_are_never_chunked():
    row = {"row_id": "r", "content": long_text()}
    assert chunks.expand(tables.MEMORY, [row], fake_chunker) == [row]


def test_a_sync_sends_a_long_tool_result_as_chunks_and_never_twice(m, platform):
    m.local.append(tables.CONVERSATIONS, [{tables.ROW_ID: "t1", "session_id": "s", "seq": 0, "role": "tool",
                                           "tool_name": "read_file", "tool_result": long_text()}])
    m.sync(wait=True)
    cloud = platform.rows(TEAM_DB, tables.CONVERSATIONS)
    assert [(r[tables.ROW_ID], r[chunks.CHUNK_COLUMN]) for r in cloud] == [("t1", 0), ("t1", 1), ("t1", 2)]
    m.ensure_cloud_tables()  # the lost-table check asks by row_id
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 3


def test_an_append_never_parts_one_messages_chunks(m, platform, monkeypatch):
    monkeypatch.setattr(mirror, "PUSH_MAX_BYTES", 500)  # smaller than one message's chunks
    m.local.append(tables.CONVERSATIONS, [{tables.ROW_ID: f"t{i}", "session_id": "s", "seq": i, "role": "tool",
                                           "tool_result": long_text()} for i in range(2)])
    sent = []
    real = m.client.append
    monkeypatch.setattr(m.client, "append", lambda db, t, rows: (sent.append([r[tables.ROW_ID] for r in rows]),
                                                                 real(db, t, rows))[1])
    m.sync(wait=True)
    assert sent == [["t0", "t0", "t0"], ["t1", "t1", "t1"]]


def test_a_message_whose_chunks_reached_the_cloud_before_the_answer_is_not_sent_again(m, platform, monkeypatch):
    m.local.append(tables.CONVERSATIONS, [{tables.ROW_ID: "t1", "session_id": "s", "seq": 0, "role": "tool",
                                           "tool_result": long_text()}])
    real = m.client.append

    def times_out_after_storing(db, table, rows):
        real(db, table, rows)
        raise TimeoutError("the answer never came")

    monkeypatch.setattr(m.client, "append", times_out_after_storing)
    with pytest.raises(TimeoutError):
        m.sync(wait=True)
    monkeypatch.setattr(m.client, "append", real)
    m.sync(wait=True)
    assert len(platform.rows(TEAM_DB, tables.CONVERSATIONS)) == 3


def test_conversations_and_traces_are_embedded_in_the_cloud(m, platform):
    for table, source in ((tables.CONVERSATIONS, ["content", "tool_result"]), (tables.TRACES, ["args", "result"])):
        [embedding] = [f for f in platform.schemas[(TEAM_DB, table)] if f["type"] == "embedding"]
        assert embedding["source"] == source


def test_a_platform_that_cannot_embed_gets_keyword_only_conversations(home, platform, tmp_path):
    platform.no_embedding_server = True
    state = config.state_dir(tmp_path)
    team, personal = LocalStore.under(state), LocalStore.under(state, PERSONAL_DB_DIR)
    team.ensure_tables(with_vectors=False)
    personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    m = Mirror(sign_in(tmp_path, platform), team, personal)
    m.ensure_cloud_tables()
    for table in (tables.MEMORY, tables.CONVERSATIONS, tables.TRACES):
        assert all(f["type"] != "embedding" for f in platform.schemas[(TEAM_DB, table)])


def test_the_client_chunker_runs_the_clients_chunk_command(tmp_path):
    client = tmp_path / "cli.js"
    client.write_text("")  # the stand-in node below ignores it
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\n"
                    "test \"$2\" = chunk || exit 3\n"
                    "while IFS= read -r line; do echo \"$line\" | sed 's/\"text\"/\"content\"/'; done\n")
    node.chmod(0o755)
    run = chunks.client_chunker(str(node), str(client))
    out = run([{"id": "0:content", "path": "message.md", "text": "hello"}])
    assert out == [{"id": "0:content", "path": "message.md", "content": "hello"}]
    assert chunks.client_chunker(str(node), "") is None


def test_a_client_without_the_chunk_command_fails_the_chunker(tmp_path):
    node = tmp_path / "node"
    node.write_text("#!/bin/sh\necho \"error: unknown command '$2'\" >&2\nexit 1\n")
    node.chmod(0o755)
    with pytest.raises(RuntimeError, match="unknown command"):
        chunks.client_chunker(str(node), str(tmp_path / "cli.js"))([{"id": "x", "path": "a.txt", "text": "t"}])


def test_the_chunker_setting_defaults_to_the_tools_client(home, tmp_path):
    home.setenv("INFINO_CLIENT", "/opt/client/dist/cli.js")
    assert config.load(tmp_path).chunker == "/opt/client/dist/cli.js"
    home.setenv("INFINO_CHUNKER", "/opt/supergrep/dist/cli.js")
    assert config.load(tmp_path).chunker == "/opt/supergrep/dist/cli.js"

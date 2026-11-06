"""Personal transcripts: a personal scheduled job's run, the chat its output
is delivered into, and a personal profile keep their conversations and
traces out of the team's database; everything else still goes there."""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest

from cron.execution_identity import enter_cron_execution, exit_cron_execution

from hermes_infino import backend, config, privacy, tables
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore

from .helpers import DM, PERSONAL_DB, TEAM_DB, hook_ctx, in_session, local, make_manager, wait_for

# A person's email digest (personal) and the team's weekly report (not).
DIGEST = {"id": "a1b2c3", "name": "email-digest"}
REPORT = {"id": "d4e5f6", "name": "weekly-report"}
# What the digest's run reads, writes and delivers: words found nowhere else.
MAIL = "From: Dr. Lee - your test results are ready"
TRIAGE = "Triage: Dr. Lee sent the test results; reply today."
DELIVERED = f"[Cron delivery: {DIGEST['name']}]\n{TRIAGE}"
PRIVATE = "Dr. Lee"
# A cron run's session, as Hermes's scheduler names one.
DIGEST_SESSION = f"cron_{DIGEST['id']}_20261105_070000"
REPORT_SESSION = f"cron_{REPORT['id']}_20261105_070000"
# Far enough ahead that every row stored is "settled" for a read of the sources.
LATER_MS = 60_000


def personal(tmp_path) -> LocalStore:
    return LocalStore.under(config.state_dir(tmp_path), PERSONAL_DB_DIR)


def texts(store: LocalStore, table: str) -> list[str]:
    """Every text cell of a local table's rows, one string per row ([] when
    the store has no such table)."""
    if table not in store.list_tables():
        return []
    cells = ("content", "tool_args", "tool_result") if table == tables.CONVERSATIONS else ("args", "result")
    return [" ".join(str(r.get(c) or "") for c in cells) for r in store.query_sql(f"SELECT * FROM {table}")]


@contextmanager
def running(job: dict):
    """Code running inside one of the job's runs, as Hermes's scheduler binds it."""
    token = enter_cron_execution(job, "execution-1", {"source": "builtin"})
    try:
        yield
    finally:
        exit_cron_execution(token)


def a_run(hooks, session: str) -> None:
    """A job's run: it reads mail, writes the triage, and its turn ends."""
    hooks["post_tool_call"](session_id=session, turn_id="t1", tool_name="terminal", tool_call_id="c1",
                            args={"command": "himalaya envelope list"}, result=MAIL, status="ok")
    hooks["post_api_request"](session_id=session, turn_id="t1", model="m", assistant_message=TRIAGE)
    hooks["on_session_end"](session_id=session)


def turn(user: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def restart() -> None:
    """What a new process starts with: no backend, nothing known about any session."""
    backend._backends.clear()
    privacy._sessions.clear()


@pytest.fixture
def digest_is_personal(home):
    home.setenv("INFINO_PERSONAL_JOBS", f"{DIGEST['name']}, some-other-job")
    return home


@pytest.fixture
def cron_run(digest_is_personal, tmp_path):
    """A cron run's provider, as Hermes starts one (it makes the local tables)."""
    return make_manager(tmp_path, platform="cron", agent_context="cron", session_id=DIGEST_SESSION)


# -- a personal job's own run --------------------------------------------------------

@pytest.mark.parametrize("named", [DIGEST["name"], DIGEST["id"], DIGEST["name"].upper()])
def test_a_personal_jobs_run_is_traced_to_the_personal_tables_only(home, tmp_path, named):
    home.setenv("INFINO_PERSONAL_JOBS", named)
    make_manager(tmp_path, platform="cron", agent_context="cron", session_id=DIGEST_SESSION)
    hooks = hook_ctx()
    with running(DIGEST):
        a_run(hooks, DIGEST_SESSION)
    assert wait_for(lambda: len(texts(local(tmp_path), tables.TRACES) + texts(personal(tmp_path), tables.TRACES)) == 2)
    assert texts(local(tmp_path), tables.TRACES) == []
    assert all(PRIVATE in t for t in texts(personal(tmp_path), tables.TRACES))


def test_a_personal_jobs_run_writes_no_conversation_to_the_team(cron_run, tmp_path):
    _, p = cron_run
    with running(DIGEST):
        p.sync_turn("Triage my inbox.", TRIAGE, session_id=DIGEST_SESSION,
                    messages=turn("Triage my inbox.", TRIAGE))
    p._drain()
    assert texts(local(tmp_path), tables.CONVERSATIONS) == []


def test_a_subagent_of_a_personal_run_is_personal_too(cron_run, tmp_path):
    hooks = hook_ctx()
    with running(DIGEST):
        hooks["post_api_request"](session_id=DIGEST_SESSION, turn_id="t1", model="m", assistant_message="delegating")
        hooks["subagent_start"](parent_session_id=DIGEST_SESSION, child_session_id="child-1")
    # Hermes's record of the run reads nothing in a delegated child: it is not the run.
    hooks["post_tool_call"](session_id="child-1", turn_id="c1", tool_name="terminal", args={}, result=MAIL)
    hooks["on_session_end"](session_id="child-1")
    with running(DIGEST):
        hooks["on_session_end"](session_id=DIGEST_SESSION)
    assert wait_for(lambda: len(texts(local(tmp_path), tables.TRACES) + texts(personal(tmp_path), tables.TRACES)) == 2)
    assert texts(local(tmp_path), tables.TRACES) == []


def test_a_cron_session_names_its_job_when_hermes_has_no_record_of_the_run(home, tmp_path):
    # The session id carries the job's id; with no job store to give its name, the id is listed.
    home.setenv("INFINO_PERSONAL_JOBS", DIGEST["id"])
    make_manager(tmp_path, platform="cron", agent_context="cron", session_id=DIGEST_SESSION)
    hooks = hook_ctx()
    a_run(hooks, DIGEST_SESSION)  # no record bound: the session id still names the job
    assert wait_for(lambda: len(texts(personal(tmp_path), tables.TRACES)) == 2)
    assert texts(local(tmp_path), tables.TRACES) == []


def test_a_team_jobs_run_is_traced_to_the_team_as_before(digest_is_personal, tmp_path):
    make_manager(tmp_path, platform="cron", agent_context="cron", session_id=REPORT_SESSION)
    hooks = hook_ctx()
    with running(REPORT):
        a_run(hooks, REPORT_SESSION)
    assert wait_for(lambda: len(texts(local(tmp_path), tables.TRACES)) == 2)
    assert texts(personal(tmp_path), tables.TRACES) == []


# -- the chat a personal job's output is delivered into -----------------------------------

@pytest.fixture
def chat(digest_is_personal, tmp_path, monkeypatch):
    """The person's DM with their agent on a gateway."""
    in_session(monkeypatch, DM)
    return make_manager(tmp_path, platform="telegram", session_id="dm-1")


@pytest.mark.parametrize("merged", [False, True])
def test_a_chat_a_personal_jobs_output_was_delivered_into_is_personal(chat, tmp_path, merged):
    _, p = chat
    ask = "Draft a reply to Dr. Lee."
    answer = "Here is a draft for Dr. Lee: thank you, I will call today."
    if merged:  # Hermes merged the delivery and the person's next message into one
        messages = [{"role": "user", "content": DELIVERED + "\n\n" + ask}, {"role": "assistant", "content": answer}]
    else:
        messages = [{"role": "user", "content": DELIVERED}] + turn(ask, answer)
    p.sync_turn(ask, answer, session_id="dm-1", messages=messages)
    p._drain()
    assert texts(local(tmp_path), tables.CONVERSATIONS) == []
    assert len(texts(personal(tmp_path), tables.CONVERSATIONS)) == 2


def test_the_chat_stays_personal_after_compaction_and_a_restart(chat, tmp_path):
    _, p = chat
    first = [{"role": "user", "content": DELIVERED}] + turn("Draft a reply to Dr. Lee.", "Drafted.")
    p.sync_turn("Draft a reply to Dr. Lee.", "Drafted.", session_id="dm-1", messages=first)
    p._drain()
    p.shutdown()
    # A new process; Hermes compacted the chat in place: the delivery is gone from its messages.
    restart()
    _, again = make_manager(tmp_path, platform="telegram", session_id="dm-1")
    compacted = [{"role": "user", "content": "[summary: a reply was drafted]"}] + turn("Send it.", "Sent to Dr. Lee.")
    again.sync_turn("Send it.", "Sent to Dr. Lee.", session_id="dm-1", messages=compacted)
    again._drain()
    assert texts(local(tmp_path), tables.CONVERSATIONS) == []
    assert len(texts(personal(tmp_path), tables.CONVERSATIONS)) == 4
    # The session's messages keep counting across the two.
    assert sorted(r["seq"] for r in personal(tmp_path).query_sql("SELECT seq FROM conversations")) == [0, 1, 2, 3]


def test_a_delivered_chats_traces_are_personal(chat, tmp_path):
    hooks = hook_ctx()
    hooks["pre_llm_call"](session_id="dm-1", user_message="Draft a reply to Dr. Lee.",
                          conversation_history=[{"role": "user", "content": DELIVERED}], platform="telegram")
    hooks["post_api_request"](session_id="dm-1", turn_id="t2", model="m", assistant_message="Drafted for Dr. Lee.")
    hooks["on_session_end"](session_id="dm-1")
    assert wait_for(lambda: len(texts(personal(tmp_path), tables.TRACES)) == 1)
    assert texts(local(tmp_path), tables.TRACES) == []


def room_turn(job_name: str, output: str) -> str:
    """A group chat member's turn prompt, as Hermes builds one when a scheduled
    job posts its output into the room (`deliver: room:<id>`)."""
    return "\n".join([
        '[Discussion: "Family"] You are @helper, one participant with @planner and the user.', "",
        "New messages in this thread since your last turn (oldest first):",
        f'  Scheduled job "{job_name}" (automated post, not the user): {output}',
        "  User (user): Can you draft the reply?", "",
        "Rules for this Discussion:",
        "- Reply with one conversational message only when you have something new worth adding."])


def test_a_group_chat_a_personal_jobs_output_was_posted_into_is_personal(chat, tmp_path):
    _, p = chat
    prompt = room_turn(DIGEST["name"], TRIAGE)
    p.sync_turn(prompt, "Drafted a reply to Dr. Lee.", session_id="room-1",
                messages=turn(prompt, "Drafted a reply to Dr. Lee."))
    p._drain()
    assert texts(local(tmp_path), tables.CONVERSATIONS) == []
    assert len(texts(personal(tmp_path), tables.CONVERSATIONS)) == 2


def test_a_group_chat_a_team_jobs_output_was_posted_into_stays_the_teams(chat, tmp_path):
    _, p = chat
    prompt = room_turn(REPORT["name"], "12 deploys this week.")
    p.sync_turn(prompt, "The Tuesday deploy failed.", session_id="room-1",
                messages=turn(prompt, "The Tuesday deploy failed."))
    p._drain()
    assert len(texts(local(tmp_path), tables.CONVERSATIONS)) == 2
    assert texts(personal(tmp_path), tables.CONVERSATIONS) == []


def hermes_room_turn(tmp_path, job: dict, output: str) -> str:
    """The turn prompt Hermes itself builds for a group chat member that a
    scheduled job's post @mentions: the room store, the job's post and the
    Discussion planner are Hermes's own (in Hermes 0.23 and later)."""
    hosted_rooms = pytest.importorskip("gateway.hosted_rooms")
    room_context = pytest.importorskip("gateway.hosted_room_context")
    discussion = pytest.importorskip("gateway.hosted_room_discussion")
    members = [{"member_id": "member-helper", "profile": "helper", "handle": "helper"},
               {"member_id": "member-planner", "profile": "planner", "handle": "planner"}]
    db = tmp_path / "rooms.db"
    hosted_rooms.create_room(db, room_id="room-family", name="Family", members=members,
                             authority_gateway_id=hosted_rooms.local_authority_gateway_id(), now=1)
    room_context.post_scheduled_message(db, room_id="room-family", job_id=job["id"], job_name=job["name"],
                                        run_key="run-1", text=f"@helper {output}")
    room = hosted_rooms.room_state(db, room_id="room-family")
    room["members"] = members
    events = hosted_rooms.read_events(db, room_id="room-family", limit=hosted_rooms.MAX_LOG_LIMIT)["events"]
    task = discussion.plan_next_task(room, events, local_profiles=("helper", "planner")).task
    return task.payload["prompt"]


@pytest.mark.parametrize("job, personal_rows, team_rows", [(DIGEST, 2, 0), (REPORT, 0, 2)])
def test_a_group_chat_post_as_hermes_builds_it_follows_the_job(chat, tmp_path, job, personal_rows, team_rows):
    _, p = chat
    prompt = hermes_room_turn(tmp_path, job, TRIAGE)
    p.sync_turn(prompt, "Drafted a reply to Dr. Lee.", session_id="room-1",
                messages=turn(prompt, "Drafted a reply to Dr. Lee."))
    p._drain()
    assert len(texts(personal(tmp_path), tables.CONVERSATIONS)) == personal_rows
    assert len(texts(local(tmp_path), tables.CONVERSATIONS)) == team_rows


def test_a_group_chat_member_whose_profile_does_not_list_the_job_writes_to_the_team(home, tmp_path, monkeypatch):
    """The setting is read per profile: a member's turn runs as its own profile,
    and this one does not list the job."""
    in_session(monkeypatch, DM)
    _, p = make_manager(tmp_path, platform="telegram", session_id="room-1")
    prompt = hermes_room_turn(tmp_path, DIGEST, TRIAGE)
    p.sync_turn(prompt, "Drafted a reply to Dr. Lee.", session_id="room-1",
                messages=turn(prompt, "Drafted a reply to Dr. Lee."))
    p._drain()
    assert texts(personal(tmp_path), tables.CONVERSATIONS) == []
    assert len(texts(local(tmp_path), tables.CONVERSATIONS)) == 2


def test_a_person_quoting_a_room_label_mid_line_names_no_job():
    said = 'User (user): it printed Scheduled job "email-digest" (automated post, not the user): hi'
    assert privacy.delivered_jobs([{"role": "user", "content": said}]) == []


def test_a_chat_a_team_jobs_output_was_delivered_into_stays_the_teams(chat, tmp_path):
    _, p = chat
    messages = [{"role": "user", "content": "[Cron delivery: weekly-report]\n12 deploys this week."}] + turn(
        "Which one failed?", "The Tuesday deploy.")
    p.sync_turn("Which one failed?", "The Tuesday deploy.", session_id="dm-1", messages=messages)
    p._drain()
    assert len(texts(local(tmp_path), tables.CONVERSATIONS)) == 2
    assert texts(personal(tmp_path), tables.CONVERSATIONS) == []


# -- a personal profile --------------------------------------------------------------

def test_a_personal_profile_keeps_its_conversations_and_traces_personal(home, tmp_path):
    home.setenv("INFINO_PERSONAL_PROFILE", "on")
    _, p = make_manager(tmp_path, session_id="errands")
    hooks = hook_ctx()
    p.sync_turn("Book Dr. Lee for Tuesday.", "Booked.", session_id="errands")
    hooks["post_api_request"](session_id="errands", turn_id="t1", model="m", assistant_message="Booked Dr. Lee.")
    hooks["on_session_end"](session_id="errands")
    p._drain()
    assert wait_for(lambda: len(texts(personal(tmp_path), tables.TRACES)) == 1)
    assert texts(local(tmp_path), tables.CONVERSATIONS) == [] and texts(local(tmp_path), tables.TRACES) == []
    assert len(texts(personal(tmp_path), tables.CONVERSATIONS)) == 2


def test_a_profile_that_is_not_personal_shares_as_before(home, tmp_path):
    home.setenv("INFINO_PERSONAL_PROFILE", "off")
    _, p = make_manager(tmp_path, session_id="work")
    p.sync_turn("What time is standup?", "Ten.", session_id="work")
    p._drain()
    assert len(texts(local(tmp_path), tables.CONVERSATIONS)) == 2


# -- what reaches the cloud ----------------------------------------------------------

def team_source_texts(p) -> list[str]:
    """Every text the team database's sources would push, settled or not."""
    out = []
    for src in p._backend.mirror.sources():
        if src.database != TEAM_DB:
            continue
        for r in src.store.rows_after(src.table, None, int(time.time() * 1000) + LATER_MS, 10_000):
            out.append(" ".join(str(v) for v in r.values()))
    return out


def cloud_texts(platform, db: str, table: str) -> list[str]:
    return [" ".join(str(v) for v in r.values()) for r in platform.rows(db, table)]


@pytest.fixture
def signed_in(env, tmp_path, monkeypatch):
    env.setenv("INFINO_PERSONAL_DATABASE", PERSONAL_DB)
    env.setenv("INFINO_PERSONAL_JOBS", DIGEST["name"])
    in_session(monkeypatch, DM)
    return env


def test_personal_rows_never_reach_the_team_databases_sources_and_team_rows_still_do(signed_in, platform, tmp_path):
    _, p = make_manager(tmp_path, platform="telegram", session_id="dm-1")
    hooks = hook_ctx()
    # The team's work: a turn and its trace.
    p.sync_turn("What time is standup?", "Ten.", session_id="standup")
    hooks["post_api_request"](session_id="standup", turn_id="s1", model="m", assistant_message="Ten.")
    hooks["on_session_end"](session_id="standup")
    # The digest's run, and the chat its output was delivered into.
    with running(DIGEST):
        a_run(hooks, DIGEST_SESSION)
    messages = [{"role": "user", "content": DELIVERED}] + turn("Draft a reply to Dr. Lee.", "Drafted for Dr. Lee.")
    p.sync_turn("Draft a reply to Dr. Lee.", "Drafted for Dr. Lee.", session_id="dm-1", messages=messages)
    p._drain()
    assert wait_for(lambda: len(texts(local(tmp_path), tables.TRACES) + texts(personal(tmp_path), tables.TRACES)) == 3)
    p.shutdown()  # pushes what is left before it returns
    assert not [t for t in team_source_texts(p) if PRIVATE in t]
    for table in (tables.CONVERSATIONS, tables.TRACES):
        assert not [t for t in cloud_texts(platform, TEAM_DB, table) if PRIVATE in t]
    # The team's rows go to the team, as before; the personal ones to the person's database.
    assert sorted(r["content"] for r in platform.rows(TEAM_DB, tables.CONVERSATIONS)) == ["Ten.", "What time is standup?"]
    assert [r["session_id"] for r in platform.rows(TEAM_DB, tables.TRACES)] == ["standup"]
    assert len(platform.rows(PERSONAL_DB, tables.CONVERSATIONS)) == 2
    assert sorted(r["session_id"] for r in platform.rows(PERSONAL_DB, tables.TRACES)) == [DIGEST_SESSION] * 2


def test_without_a_personal_database_personal_rows_stay_on_this_machine(env, platform, tmp_path):
    env.setenv("INFINO_PERSONAL_JOBS", DIGEST["name"])
    make_manager(tmp_path, platform="cron", agent_context="cron", session_id=DIGEST_SESSION)
    hooks = hook_ctx()
    with running(DIGEST):
        a_run(hooks, DIGEST_SESSION)
    assert wait_for(lambda: len(texts(local(tmp_path), tables.TRACES) + texts(personal(tmp_path), tables.TRACES)) == 2)
    backend.for_settings(config.load(tmp_path)).flush()
    assert platform.rows(TEAM_DB, tables.TRACES) == []
    assert len(texts(personal(tmp_path), tables.TRACES)) == 2
    assert set(platform.databases) == {TEAM_DB}

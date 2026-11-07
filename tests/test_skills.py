"""The team's skills: reading a SKILL.md, picking versions, and loading
them into a profile."""

from __future__ import annotations

import json

import pytest

from hermes_infino import config, skills, tables
from hermes_infino.client import InfinoClient

from .helpers import SKILL_MD, TEAM_DB, local_rows, make_manager, requests_to, sign_in


class Ctx:
    """Hermes's plugin context, as far as the skills loader uses it."""

    def __init__(self, refuse: tuple[str, ...] = ()):
        self.registered: list[tuple[str, str, str]] = []
        self._refuse = refuse

    def register_skill(self, name, path, description=""):
        if name in self._refuse:
            raise ValueError(f"{name} clashes with a built-in skill")
        self.registered.append((name, path.read_text(), description))


def publish(platform, *rows: dict, team: str = TEAM_DB) -> None:
    c = InfinoClient("k", platform.url)
    tables.ensure_all(c, team)
    c.append(team, tables.SKILLS, list(rows))


def write_skill(path, text: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(text)


@pytest.mark.parametrize("md, description", [
    (SKILL_MD, "Steps before a release."),
    ("---\nname: x\ndescription: \"Quoted: with a colon\"\n---\nbody", "Quoted: with a colon"),
    ("---\nname: x\n---\nbody", ""),
    ("no frontmatter at all\ndescription: not this", ""),
    ("", ""),
])
def test_the_description_comes_from_the_frontmatter(md, description):
    assert skills.description_of(md) == description


def test_a_published_row_carries_who_and_when(home):
    row = skills.skill_row("release-checklist", SKILL_MD, "alice", "ops")
    assert (row["name"], row["author"], row["profile"], row["content"]) == ("release-checklist", "alice", "ops", SKILL_MD)
    assert row["description"] == "Steps before a release." and row["created_ms"] > 0


def test_names_hermes_would_refuse_and_empty_skills_are_skipped():
    rows = [{"name": n, "content": "x", "created_ms": 1} for n in ("../escape", "a/b", "has space", "", "ok_name-2")]
    rows.append({"name": "empty", "content": "", "created_ms": 1})
    assert list(skills.latest(rows)) == ["ok_name-2"]


def test_a_tie_keeps_the_first_row_read():
    rows = [{"name": "a", "content": "first", "created_ms": 5}, {"name": "a", "content": "second", "created_ms": 5}]
    assert skills.latest(rows)["a"]["content"] == "first"


def test_a_local_skill_is_found_at_any_depth(home, tmp_path):
    deep = tmp_path / "skills" / "team" / "ops" / "release-checklist" / "SKILL.md"
    deep.parent.mkdir(parents=True)
    deep.write_text(SKILL_MD)
    assert skills.find_local_skill("release-checklist", tmp_path) == deep
    assert skills.find_local_skill("release", tmp_path) is None


def test_a_profile_without_skills_has_none(home, tmp_path):
    assert skills.find_local_skill("anything", tmp_path) is None


def test_a_copy_the_curator_archived_is_never_the_skill_found(home, tmp_path):
    # Hermes's curator moves a skill it retires to skills/.archive/<name>; Hermes skips that folder.
    write_skill(tmp_path / "skills" / ".archive" / "release-checklist" / "SKILL.md", "old")
    write_skill(tmp_path / "skills" / "devops" / "release-checklist" / "SKILL.md", SKILL_MD)
    assert skills.find_local_skill("release-checklist", tmp_path).read_text() == SKILL_MD


def test_a_skill_is_found_by_the_name_skills_list_shows(home, tmp_path):
    # skills_list names a skill by its frontmatter; its folder may be named otherwise.
    skill = tmp_path / "skills" / "release" / "SKILL.md"
    write_skill(skill, SKILL_MD)  # name: release-checklist
    assert skills.find_local_skill("release-checklist", tmp_path) == skill
    assert skills.find_local_skill("release", tmp_path) == skill  # the folder's name still finds it


def test_a_skill_named_as_hermes_never_loads_a_team_skill_is_refused_at_publish(home, tmp_path):
    # Hermes's own skill tool takes dots in a name; a plugin's skills may have none.
    write_skill(tmp_path / "skills" / "deploy.v2" / "SKILL.md", "---\nname: deploy.v2\ndescription: Ship.\n---\nbody\n")
    m, _ = make_manager(tmp_path)
    out = json.loads(m.handle_tool_call("infino_publish_skill", {"name": "deploy.v2"}))
    assert out["ok"] is False and "'deploy.v2'" in out["error"] and "letters" in out["error"]
    assert local_rows(tmp_path, tables.SKILLS) == []


def test_without_an_account_nothing_is_loaded_or_asked(home, platform, tmp_path):
    assert skills.register_team_skills(Ctx(), config.load(tmp_path), home=tmp_path) == []
    assert platform.requests == []


def test_a_signed_in_team_with_no_skills_table_loads_none(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    platform.databases[TEAM_DB] = {}
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == []
    assert requests_to(platform, "query_sql") == 1  # one request, no list of tables first


def test_the_newest_version_of_each_skill_is_written_and_registered(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "v1", "created_ms": 1},
            {"name": "deploy", "content": "v2", "description": "Ship it.", "author": "alice", "created_ms": 2},
            {"name": "triage", "content": "t1", "created_ms": 1})
    ctx = Ctx()
    assert skills.register_team_skills(ctx, settings, home=tmp_path) == ["deploy", "triage"]
    assert ctx.registered == [("deploy", "v2", "Ship it. (team skill published by alice)"),
                              ("triage", "t1", "(team skill)")]
    assert (skills.cache_root(settings, tmp_path) / "deploy" / "SKILL.md").read_text() == "v2"


def test_a_platform_that_does_not_answer_loads_the_last_copies(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "v2", "description": "Ship it.", "author": "alice",
                       "created_ms": 2})
    skills.register_team_skills(Ctx(), settings, home=tmp_path)
    platform.close()
    skills._loaded_at.clear()  # a new process
    ctx = Ctx()
    assert skills.register_team_skills(ctx, settings, home=tmp_path) == ["deploy"]
    assert ctx.registered == [("deploy", "v2", "Ship it. (team skill published by alice)")]


def test_a_copy_without_its_note_reads_the_description_from_the_skill(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    copy = skills.cache_root(settings, tmp_path) / "release-checklist" / "SKILL.md"
    copy.parent.mkdir(parents=True)
    copy.write_text(SKILL_MD)
    platform.close()
    ctx = Ctx()
    skills.register_team_skills(ctx, settings, home=tmp_path)
    assert ctx.registered == [("release-checklist", SKILL_MD, "Steps before a release. (team skill)")]


def test_team_skills_turned_off_load_nothing_and_ask_nothing(home, platform, tmp_path):
    home.setenv("INFINO_TEAM_SKILLS", "off")
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "v2", "created_ms": 2})
    before = requests_to(platform, "query_sql")
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == []
    assert requests_to(platform, "query_sql") == before


def test_agents_built_soon_after_reuse_the_load(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "v2", "created_ms": 2})
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy"]
    before = requests_to(platform, "query_sql")
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy"]
    assert requests_to(platform, "query_sql") == before


def test_the_load_waits_briefly(home, platform, tmp_path, monkeypatch):
    waits = []
    settings = sign_in(tmp_path, platform)

    def query(sql, timeout_s=None):
        waits.append(timeout_s)
        raise TimeoutError("slow platform")

    monkeypatch.setattr(skills.backend.Backend, "query", lambda self, sql, timeout_s=None: query(sql, timeout_s))
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == []
    assert waits == [skills.LOAD_TIMEOUT_S] and skills.LOAD_TIMEOUT_S <= 2.0


def test_one_refused_skill_does_not_stop_the_rest(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "d", "created_ms": 1},
            {"name": "triage", "content": "t", "created_ms": 1})
    assert skills.register_team_skills(Ctx(refuse=("deploy",)), settings, home=tmp_path) == ["triage"]


def test_a_skill_name_never_writes_outside_the_cache(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "../../escape", "content": "owned", "created_ms": 1})
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == []
    assert not (tmp_path / "escape").exists() and not (tmp_path.parent / "escape").exists()


def test_after_a_team_switch_the_last_teams_skills_never_load(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform, team="team-a")
    publish(platform, {"name": "deploy", "content": "team a's", "created_ms": 1}, team="team-a")
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy"]
    settings = sign_in(tmp_path, platform, team="team-b")  # no skills table there yet
    skills._loaded_at.clear()  # a new process
    ctx = Ctx()
    assert skills.register_team_skills(ctx, settings, home=tmp_path) == []
    assert ctx.registered == []


def test_a_skill_gone_from_the_teams_table_never_loads_from_its_copy(home, platform, tmp_path):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "d", "created_ms": 1},
            {"name": "triage", "content": "t", "created_ms": 1})
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy", "triage"]
    platform.databases[TEAM_DB][tables.SKILLS] = [r for r in platform.rows(TEAM_DB, tables.SKILLS)
                                                  if r["name"] == "deploy"]  # triage deleted from the table
    skills._loaded_at.clear()
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy"]
    platform.close()
    skills._loaded_at.clear()
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy"]


def test_a_copy_another_load_wrote_after_this_ones_query_is_kept(home, platform, tmp_path, monkeypatch):
    settings = sign_in(tmp_path, platform)
    publish(platform, {"name": "deploy", "content": "d", "created_ms": 1})
    real_query = skills.backend.Backend.query
    other = Ctx()
    queries = []

    def query(self, sql, timeout_s=None):
        queries.append(sql)
        rows = real_query(self, sql, timeout_s=timeout_s)  # this load's answer: deploy alone
        if len(queries) == 1:
            # Meanwhile a skill is published, and another process's load (the
            # gateway's, say, with this profile's CLI) reads it, writes its
            # copy and registers it.
            publish(platform, {"name": "fresh", "content": "f", "created_ms": 2})
            skills._loaded_at.clear()  # that process has no load of its own yet
            assert skills.register_team_skills(other, settings, home=tmp_path) == ["deploy", "fresh"]
        return rows

    monkeypatch.setattr(skills.backend.Backend, "query", query)
    assert skills.register_team_skills(Ctx(), settings, home=tmp_path) == ["deploy"]
    # The copy the other process registered is still there for Hermes to load.
    assert (skills.cache_root(settings, tmp_path) / "fresh" / "SKILL.md").read_text() == "f"

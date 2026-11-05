"""`hermes infino status | login | sync`, as a person at a terminal runs them."""

from __future__ import annotations

import argparse
import io
from types import SimpleNamespace

import pytest

from hermes_infino import cli, config, tables

from .helpers import TEAM_DB, make_manager, sign_in

# A port nothing listens on.
DEAD_URL = "http://127.0.0.1:9"


def run(*argv: str) -> int:
    parser = argparse.ArgumentParser()
    cli.register_cli(parser)
    return cli.infino_command(parser.parse_args(list(argv)))


@pytest.fixture
def terminal(monkeypatch):
    """A terminal whose person answers `answer` to the question."""
    def at(answer: str):
        monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda _q: answer)
    return at


@pytest.fixture
def no_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "team.key"
    path.write_text("team-key\n")
    return path


def test_no_action_is_status(home, capsys):
    assert run() == 0
    assert "signed in        no" in capsys.readouterr().out


def test_status_counts_what_is_kept_here(home, tmp_path, capsys):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "the deploy freeze starts December 20"})
    m.handle_tool_call("infino_remember", {"content": "my dentist is Tuesday", "scope": "personal"})
    assert run("status") == 0
    out = capsys.readouterr().out
    team, personal = out.split("\npersonal ", 1)  # the person's own tables come second
    assert "memory         1 rows" in team
    assert "memory         1 rows" in personal and "conversations  0 rows" in personal
    assert "memory vectors on (local model)" in out


def test_status_signed_in_says_how_far_behind_the_cloud_is(home, platform, tmp_path, capsys):
    sign_in(tmp_path, platform)
    m, _ = make_manager(tmp_path)
    capsys.readouterr()
    assert run("status") == 0
    out = capsys.readouterr().out
    assert f"signed in        {platform.url} / {TEAM_DB}, from the sign-in" in out
    assert f"rows not in {TEAM_DB} yet" in out
    assert "personal memory stays on this machine" in out


def test_status_shows_why_the_last_sync_stopped(home, platform, tmp_path, capsys):
    sign_in(tmp_path, platform)
    _, p = make_manager(tmp_path)
    p._backend.local.append(tables.MEMORY, [{"content": "x"}])
    platform.failed_appends = 1
    with pytest.raises(Exception):
        p._backend.mirror.sync(wait=True)
    run("status")
    assert "last sync        stopped: InfinoError: HTTP 500" in capsys.readouterr().out


def test_sync_without_an_account_says_so(home, capsys):
    assert run("sync") == 0
    assert "Not signed in" in capsys.readouterr().out


def test_sync_pushes_and_reports(home, platform, tmp_path, capsys):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "kept before sign-in"})
    sign_in(tmp_path, platform)
    assert run("sync") == 0
    assert "(1 rows pushed)" in capsys.readouterr().out
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept before sign-in"]


def test_sync_reports_a_cloud_that_stops_accepting(home, platform, tmp_path, capsys):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "x"})
    sign_in(tmp_path, platform)
    platform.failed_appends = 1
    assert run("sync") == 1
    assert "`hermes infino sync` resumes" in capsys.readouterr().err


def test_login_when_already_signed_in_changes_nothing(home, platform, tmp_path, capsys):
    sign_in(tmp_path, platform)
    assert run("login") == 0
    assert "Already signed in" in capsys.readouterr().out and platform.trials == 0


def test_login_with_a_given_key_and_yes_needs_no_terminal(home, platform, tmp_path, no_terminal, key_file):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "kept before sign-in"})
    assert run("login", "--url", platform.url, "--api-key-file", str(key_file), "--team-database", TEAM_DB, "--yes") == 0
    creds = config.read_credentials(tmp_path)
    assert (creds["api_key"], creds["team_database"], creds["url"]) == ("team-key", TEAM_DB, platform.url)
    assert platform.trials == 0 and "team-key" in platform.keys_seen
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept before sign-in"]


def test_login_with_a_given_key_asks_without_yes(home, platform, tmp_path, no_terminal, key_file):
    assert run("login", "--url", platform.url, "--api-key-file", str(key_file), "--team-database", TEAM_DB) == 1
    assert config.read_credentials(tmp_path) == {}


def test_login_with_a_key_the_platform_refuses_stores_nothing(home, platform, tmp_path, no_terminal, key_file, capsys):
    platform.valid_keys = {"some-other-key"}
    assert run("login", "--url", platform.url, "--api-key-file", str(key_file), "--team-database", TEAM_DB, "--yes") == 1
    assert "HTTP 401" in capsys.readouterr().err
    assert config.read_credentials(tmp_path) == {}


def test_login_where_no_free_account_is_offered(home, platform, tmp_path, terminal, capsys):
    terminal("y")
    platform.trial_status = 501
    assert run("login", "--url", platform.url, "--team-database", TEAM_DB) == 1
    assert "HTTP 501" in capsys.readouterr().err and config.read_credentials(tmp_path) == {}


def test_login_keeps_a_personal_database(home, platform, tmp_path, terminal):
    terminal("yes")
    assert run("login", "--url", platform.url, "--team-database", TEAM_DB, "--personal-database", "me") == 0
    assert config.read_credentials(tmp_path)["personal_database"] == "me"
    assert set(platform.databases["me"]) == set(tables.PERSONAL)


def test_the_upload_notice_says_what_goes_and_what_stays():
    notice = cli._upload_notice("https://x", TEAM_DB, {"memory": 3, "traces": 9}, new_account=True)
    assert "creates a free Infino account on https://x" in notice
    assert f"database '{TEAM_DB}'" in notice and "3 memory, 9 traces" in notice
    assert "personal memory" in notice and "It stays on this machine too." in notice
    assert "nothing yet" in cli._upload_notice("https://x", TEAM_DB, {}, new_account=False)


def test_login_to_an_unreachable_platform_reports_it(home, tmp_path, no_terminal, key_file, capsys):
    assert run("login", "--url", DEAD_URL, "--api-key-file", str(key_file), "--team-database", TEAM_DB, "--yes") == 1
    assert "could not reach the platform" in capsys.readouterr().err
    assert config.read_credentials(tmp_path) == {}


def test_login_with_a_key_file_that_is_not_there(home, platform, tmp_path, no_terminal, capsys):
    assert run("login", "--url", platform.url, "--api-key-file", str(tmp_path / "missing.key"),
               "--team-database", TEAM_DB, "--yes") == 1
    assert "cannot read the key file" in capsys.readouterr().err


def test_sync_to_an_unreachable_platform_reports_it(home, tmp_path, capsys):
    config.write_credentials({"url": DEAD_URL, "api_key": "k", "team_database": TEAM_DB}, tmp_path)
    assert run("sync") == 1
    assert "could not reach the platform" in capsys.readouterr().err

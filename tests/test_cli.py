"""`hermes infino status | login | sync`, as a person at a terminal runs them."""

from __future__ import annotations

import argparse
import io
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_infino import cli, config, tables

from .helpers import PERSONAL_DB, TEAM_DB, make_manager, sign_in

# A port nothing listens on.
DEAD_URL = "http://127.0.0.1:9"
# A row carrying this text is refused by the fake platform, when told to.
REFUSED_TEXT = "refuse-this-row"
# "Too large": a refusal that sets the row aside rather than stopping the sync.
TOO_LARGE = 413


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


@pytest.mark.parametrize("status, why", [(501, "gives out no free accounts"),
                                         (409, "already gave this network its free account")])
def test_login_where_no_free_account_is_given_says_how_to_sign_in(home, platform, tmp_path, terminal, capsys,
                                                                  status, why):
    terminal("y")
    platform.trial_status = status
    assert run("login", "--url", platform.url, "--team-database", TEAM_DB) == 1
    err = capsys.readouterr().err
    assert why in err and "--api-key-file" in err and "nothing was sent" in err
    assert config.read_credentials(tmp_path) == {}


def test_login_keeps_a_personal_database(home, platform, tmp_path, terminal):
    terminal("yes")
    assert run("login", "--url", platform.url, "--team-database", TEAM_DB, "--personal-database", "me") == 0
    assert config.read_credentials(tmp_path)["personal_database"] == "me"
    assert set(platform.databases["me"]) == set(tables.PERSONAL)


def personal_database_of(home_dir) -> str:
    """The personal database a sign-in saved ("" for none)."""
    return config.read_credentials(home_dir).get("personal_database", "")


def login_with_key(platform, key_file, *extra: str) -> int:
    return run("login", "--url", platform.url, "--api-key-file", str(key_file), "--team-database", TEAM_DB, "--yes",
               *extra)


def test_login_makes_a_personal_database_by_default(home, platform, tmp_path, terminal, capsys):
    terminal("yes")
    assert run("login", "--url", platform.url, "--team-database", TEAM_DB) == 0
    personal = personal_database_of(tmp_path)
    assert personal.startswith(TEAM_DB + cli.PERSONAL_PART)
    assert set(platform.databases[personal]) == set(tables.PERSONAL)
    out = capsys.readouterr().out
    assert cli.NEW_PERSONAL_DATABASE in out and f"Your personal database is '{personal}'" in out


def test_a_key_someone_gave_gets_a_personal_database_too(home, platform, tmp_path, no_terminal, key_file):
    assert login_with_key(platform, key_file) == 0
    assert set(platform.databases) == {TEAM_DB, personal_database_of(tmp_path)}


def test_each_sign_in_gets_a_new_personal_database(home, platform, tmp_path, no_terminal, key_file):
    assert login_with_key(platform, key_file) == 0
    first = personal_database_of(tmp_path)
    (config.state_dir(tmp_path) / config.CREDENTIALS_FILE).unlink()  # signed out; someone signs in again
    assert login_with_key(platform, key_file) == 0
    assert personal_database_of(tmp_path) not in ("", first)


def test_a_new_personal_database_is_never_one_that_exists(home, platform, tmp_path, no_terminal, key_file,
                                                         monkeypatch, capsys):
    monkeypatch.setattr(cli.secrets, "token_hex", lambda n: "0" * 2 * n)
    taken = TEAM_DB + cli.PERSONAL_PART + "0" * 2 * cli.PERSONAL_RANDOM_BYTES
    platform.databases[taken] = {}  # someone else's
    assert login_with_key(platform, key_file) == 0
    assert personal_database_of(tmp_path) == "" and platform.databases[taken] == {}
    assert "No personal database" in capsys.readouterr().err


def test_a_long_team_name_still_makes_a_name_the_platform_takes():
    name = cli._new_personal_name("t" * cli.MAX_DATABASE_NAME)
    assert len(name) == cli.MAX_DATABASE_NAME and cli.PERSONAL_PART in name


def test_no_personal_database_keeps_what_is_personal_here(home, platform, tmp_path, no_terminal, key_file, capsys):
    assert login_with_key(platform, key_file, "--no-personal-database") == 0
    assert personal_database_of(tmp_path) == "" and set(platform.databases) == {TEAM_DB}
    assert "nowhere: it stays on this machine" in capsys.readouterr().out


@pytest.mark.parametrize("named", [[], ["--personal-database", "mine"]])
def test_a_personal_database_the_platform_refuses_leaves_the_sign_in_working(home, platform, tmp_path, terminal,
                                                                             capsys, named):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "kept before sign-in"})
    platform.database_limit = 1  # a key that may make no database past the team's
    terminal("yes")
    assert run("login", "--url", platform.url, "--team-database", TEAM_DB, *named) == 0
    assert personal_database_of(tmp_path) == "" and set(platform.databases) == {TEAM_DB}
    assert "No personal database" in capsys.readouterr().err
    # Without a personal database saved, nothing stops the team's sync.
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["kept before sign-in"]


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


@pytest.mark.parametrize("failure", [urllib.error.URLError("connection reset"), TimeoutError("timed out")])
def test_a_new_accounts_key_is_kept_whatever_stops_the_personal_database(home, platform, tmp_path, terminal,
                                                                         monkeypatch, capsys, failure):
    # The platform gives one free account per network, and its key only once.
    def fail(client, name):
        raise failure

    monkeypatch.setattr(cli.InfinoClient, "create_database", fail)
    terminal("yes")
    code = run("login", "--url", platform.url, "--team-database", TEAM_DB)
    creds = config.read_credentials(tmp_path)
    assert (creds.get("api_key"), creds.get("team_database"), creds.get("url")) == ("trial-key-1", TEAM_DB,
                                                                                    platform.url)
    assert "personal_database" not in creds and "No personal database" in capsys.readouterr().err
    assert code == 0


def test_login_refuses_a_personal_database_that_is_the_team_database(home, platform, tmp_path, no_terminal,
                                                                     key_file, capsys):
    assert login_with_key(platform, key_file, "--personal-database", TEAM_DB) == 1
    assert config.read_credentials(tmp_path) == {} and platform.databases == {}
    assert f"'{TEAM_DB}' is the team database" in capsys.readouterr().err


def test_login_names_the_team_database_already_set(home, platform, tmp_path, no_terminal, key_file, capsys):
    home.setenv("INFINO_TEAM_DATABASE", "ops")
    assert run("login", "--url", platform.url, "--api-key-file", str(key_file), "--yes") == 0
    assert "database 'ops'" in capsys.readouterr().out
    assert config.read_credentials(tmp_path)["team_database"] == "ops"
    assert cli.DEFAULT_DATABASE not in platform.databases


def test_already_signed_in_says_how_to_sign_in_again(home, platform, tmp_path, capsys):
    sign_in(tmp_path, platform)
    assert run("login") == 0
    out = capsys.readouterr().out
    assert str(config.state_dir(tmp_path) / config.CREDENTIALS_FILE) in out
    assert "environment" in out and "mcp_servers.infino" in out


def test_status_before_hermes_has_run_says_no_vectors_yet(home, capsys):
    assert run("status") == 0
    out = capsys.readouterr().out
    assert "the cloud embeds" not in out and "memory vectors not made yet" in out


def test_sync_without_a_personal_database_says_what_stays_here(home, platform, tmp_path, capsys):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "the deploy freeze starts December 20"})
    m.handle_tool_call("infino_remember", {"content": "my dentist is Tuesday", "scope": "personal"})
    sign_in(tmp_path, platform)
    assert run("sync") == 0
    out = capsys.readouterr().out
    assert "The cloud has everything kept here (" not in out
    assert "what is personal" in out and "no personal database" in out


def test_sync_counts_the_rows_the_cloud_refused(home, platform, tmp_path, capsys):
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": f"too large to take: {REFUSED_TEXT}"})
    sign_in(tmp_path, platform, personal_database=PERSONAL_DB)
    platform.refused_marker, platform.refused_status = REFUSED_TEXT, TOO_LARGE
    assert run("sync") == 0
    out = capsys.readouterr().out
    assert "The cloud has everything kept here (" not in out
    assert "1 rows the cloud would not take" in out


def remember_team_and_personal(tmp_path) -> None:
    m, _ = make_manager(tmp_path)
    m.handle_tool_call("infino_remember", {"content": "the deploy freeze starts December 20"})
    m.handle_tool_call("infino_remember", {"content": "my dentist is Tuesday", "scope": "personal"})


def test_sync_pushes_the_teams_rows_past_a_personal_database_out_of_reach(home, platform, tmp_path, capsys):
    remember_team_and_personal(tmp_path)
    sign_in(tmp_path, platform, personal_database=PERSONAL_DB)
    platform.refused_databases = {PERSONAL_DB}  # a key limited to other databases
    assert run("sync") == 1
    out = capsys.readouterr().out
    assert [r["content"] for r in platform.rows(TEAM_DB, tables.MEMORY)] == ["the deploy freeze starts December 20"]
    assert PERSONAL_DB not in platform.databases
    assert f"what is personal: '{PERSONAL_DB}' could not be made ready" in out and "403" in out
    assert "The cloud has everything kept here (" not in out
    # Once it can be made, the next sync sends what is personal.
    platform.refused_databases = set()
    assert run("sync") == 0
    assert [r["content"] for r in platform.rows(PERSONAL_DB, tables.MEMORY)] == ["my dentist is Tuesday"]
    assert len(platform.rows(TEAM_DB, tables.MEMORY)) == 1


def test_sync_says_the_team_push_failed_while_the_personal_database_is_out_of_reach(home, platform, tmp_path,
                                                                                    capsys):
    remember_team_and_personal(tmp_path)
    sign_in(tmp_path, platform, personal_database=PERSONAL_DB)
    platform.refused_databases = {PERSONAL_DB}
    platform.failed_appends = 1
    assert run("sync") == 1
    err = capsys.readouterr().err
    assert "The cloud stopped accepting rows: " in err and "HTTP 500" in err


def test_sync_where_no_database_can_be_made_ready_says_why(home, platform, tmp_path, capsys):
    remember_team_and_personal(tmp_path)
    sign_in(tmp_path, platform, personal_database=PERSONAL_DB)
    platform.refused_databases = {TEAM_DB, PERSONAL_DB}
    assert run("sync") == 1
    assert "the platform refused" in capsys.readouterr().err
    assert platform.databases == {}

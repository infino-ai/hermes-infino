"""Settings: where each one comes from, the sign-in file, and the manifests
that must agree with the code."""

from __future__ import annotations

import json
import os
import stat
import tomllib
from pathlib import Path

import hermes_constants
import pytest
import yaml

from hermes_infino import config
from hermes_infino.client import DEFAULT_URL

from .helpers import TEAM_DB

REPO = Path(__file__).resolve().parent.parent
PLUGIN_YAML = REPO / "hermes_infino" / "plugin.yaml"
PYPROJECT = REPO / "pyproject.toml"


def write_config(home_dir: Path, doc: dict) -> None:
    (home_dir / "config.yaml").write_text(json.dumps(doc))  # JSON is YAML


def test_nothing_set_is_local_mode_on_the_default_url(home, tmp_path):
    s = config.load(tmp_path)
    assert not s.signed_in and s.source == ""
    assert (s.api_key, s.team_database, s.personal_database, s.url) == ("", "", None, DEFAULT_URL)
    assert s.mcp_server == config.DEFAULT_MCP_SERVER and s.home == tmp_path


def test_a_key_without_a_team_database_is_not_signed_in(home, tmp_path):
    home.setenv("INFINO_API_KEY", "k")
    assert not config.load(tmp_path).signed_in


def test_the_environment_wins_over_the_sign_in(home, tmp_path):
    config.write_credentials({"api_key": "file-key", "team_database": "file-db", "url": "http://file"}, tmp_path)
    home.setenv("INFINO_API_KEY", "env-key")
    home.setenv("INFINO_TEAM_DATABASE", "env-db")
    s = config.load(tmp_path)
    assert (s.api_key, s.team_database, s.url, s.source) == ("env-key", "env-db", "http://file", "environment")


def test_a_split_source_is_named_by_both(home, tmp_path):
    config.write_credentials({"api_key": "file-key", "team_database": "file-db"}, tmp_path)
    home.setenv("INFINO_TEAM_DATABASE", "env-db")
    assert config.load(tmp_path).source == "environment+sign-in"


def test_the_sign_in_alone(home, tmp_path):
    config.write_credentials({"api_key": "k", "team_database": TEAM_DB, "personal_database": "me"}, tmp_path)
    s = config.load(tmp_path)
    assert s.signed_in and s.source == "sign-in" and s.personal_database == "me"


@pytest.mark.parametrize("value", ["", "   ", "${INFINO_API_KEY}", "${}"])
def test_an_empty_or_unresolved_value_is_unset(home, tmp_path, value):
    home.setenv("INFINO_API_KEY", value)
    home.setenv("INFINO_TEAM_DATABASE", TEAM_DB)
    assert config.load(tmp_path).api_key == ""


def test_values_are_trimmed(home, tmp_path):
    home.setenv("INFINO_TEAM_DATABASE", f"  {TEAM_DB}\n")
    assert config.load(tmp_path).team_database == TEAM_DB


def test_the_author_comes_from_the_environment(home, tmp_path):
    home.setenv("INFINO_AUTHOR", "priya")
    assert config.load(tmp_path).author == "priya"


def test_the_tools_block_is_read_under_the_configured_server_name(home, tmp_path):
    write_config(tmp_path, {"mcp_servers": {"team-memory": {"command": "x", "env": {"INFINO_TEAM_DATABASE": TEAM_DB}}}})
    assert config.load(tmp_path).team_database == ""  # not under the default name
    home.setenv("INFINO_MCP_SERVER", "team-memory")
    s = config.load(tmp_path)
    assert s.team_database == TEAM_DB and s.mcp_server == "team-memory"


def test_a_tools_block_without_env_reads_nothing(home, tmp_path):
    write_config(tmp_path, {"mcp_servers": {"infino": {"command": "hermes", "args": ["infino", "mcp"]}}})
    assert config.load(tmp_path).team_database == ""


def test_the_tools_block_names_the_author_too(home, tmp_path):
    write_config(tmp_path, {"mcp_servers": {"infino": {"command": "x", "env": {"INFINO_AUTHOR": "deploy-bot"}}}})
    home.setenv("INFINO_AUTHOR", "someone-else")
    assert config.load(tmp_path).author == "deploy-bot"


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "\"a string\""])
def test_an_unreadable_sign_in_is_none(home, tmp_path, text):
    path = config.state_dir(tmp_path) / config.CREDENTIALS_FILE
    path.parent.mkdir(parents=True)
    path.write_text(text)
    assert config.read_credentials(tmp_path) == {}


def test_the_sign_in_is_written_whole_and_private(home, tmp_path):
    config.write_credentials({"api_key": "one"}, tmp_path)
    path = config.write_credentials({"api_key": "two"}, tmp_path)
    assert config.read_credentials(tmp_path) == {"api_key": "two"}
    assert stat.S_IMODE(os.stat(path).st_mode) == config.CREDENTIALS_MODE
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.xfail(strict=True, reason="a .tmp left by a crash keeps its mode: os.open sets the mode only on create")
def test_a_left_over_temporary_file_never_widens_the_sign_ins_mode(home, tmp_path):
    tmp = config.state_dir(tmp_path) / (Path(config.CREDENTIALS_FILE).stem + ".tmp")
    tmp.parent.mkdir(parents=True)
    tmp.write_text("{}")
    os.chmod(tmp, 0o644)
    path = config.write_credentials({"api_key": "k"}, tmp_path)
    assert stat.S_IMODE(os.stat(path).st_mode) == config.CREDENTIALS_MODE


def test_the_home_is_hermes_home(home, tmp_path):
    assert config.home() == tmp_path
    assert config.state_dir() == tmp_path / "infino"


def test_without_hermes_home_the_home_is_the_users(home, monkeypatch):
    home.delenv("HERMES_HOME")
    assert config.home() == Path.home() / ".hermes"


@pytest.mark.xfail(strict=True, reason="config.home() reads HERMES_HOME from os.environ; Hermes sets a profile's "
                                       "home context-locally (hermes_constants.get_hermes_home), so the trace hooks "
                                       "and the skills loader of a multiplexed gateway use the wrong profile")
def test_the_home_follows_hermes_context_local_profile(home, tmp_path):
    other = tmp_path / "profile-b"
    other.mkdir()
    token = hermes_constants._HERMES_HOME_OVERRIDE.set(str(other))
    try:
        assert hermes_constants.get_hermes_home() == other
        assert config.home() == other
    finally:
        hermes_constants._HERMES_HOME_OVERRIDE.reset(token)


# -- the manifests agree with the code ----------------------------------------

def test_every_setting_is_declared_to_hermes():
    declared = {e["name"] for e in yaml.safe_load(PLUGIN_YAML.read_text())["optional_env"]}
    assert declared == {f["env_var"] for f in config.config_schema()}


def test_the_key_is_the_only_secret():
    secrets = {e["name"] for e in yaml.safe_load(PLUGIN_YAML.read_text())["optional_env"] if e.get("password")}
    assert secrets == {config.KEY_ENV}
    assert [f["env_var"] for f in config.config_schema() if f.get("secret")] == [config.KEY_ENV]


def test_hermes_and_pip_install_the_same_dependencies():
    hermes = yaml.safe_load(PLUGIN_YAML.read_text())["python_dependencies"]
    pip = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    assert sorted(hermes) == sorted(pip)


@pytest.mark.xfail(strict=True, reason="plugin.yaml says 0.4.0, pyproject.toml says 0.1.0")
def test_the_two_manifests_carry_one_version():
    assert yaml.safe_load(PLUGIN_YAML.read_text())["version"] == tomllib.loads(PYPROJECT.read_text())["project"]["version"]


def test_the_default_url_is_declared_where_people_read_it():
    assert DEFAULT_URL in PLUGIN_YAML.read_text()
    assert [f["default"] for f in config.config_schema() if f["key"] == "url"] == [DEFAULT_URL]

"""The tools' launcher: what it starts, with which settings, in each mode."""

from __future__ import annotations

import os
import sys

import pytest
from hermes_cli.env_loader import load_hermes_dotenv
from plugins.config_backends.remote.values import secret_literal_path
from tools.mcp_tool_config import _build_safe_env, _load_mcp_config

from hermes_infino import config, local, mcp_launch, tables

from .helpers import TEAM_DB, sign_in, write_tools_setting

CLIENT_JS = "/opt/infino/client/dist/cli.js"
NODE = "/opt/node/bin/node"
# A profile's .env left by an earlier setup: the key, as Remote Config asks it
# kept, and a team database from before the fleet's.
LEFT_IN_ENV_FILE = {"INFINO_API_KEY": "test-key", "INFINO_TEAM_DATABASE": "hermes-team"}


def fleet_entry(platform) -> dict:
    """The Infino tools' entry as the README's "Running a fleet" writes it."""
    return {"command": "hermes", "args": ["infino", "mcp"],
            "env": {"INFINO_CLIENT": CLIENT_JS, "INFINO_NODE": NODE, "INFINO_FLEET_TEAM_DATABASE": TEAM_DB,
                    "INFINO_FLEET_URL": platform.url, "INFINO_FLEET_API_KEY": "${INFINO_API_KEY}"}}


def become_the_tools(monkeypatch, home_dir) -> None:
    """Turn this process into the Infino tools as Hermes starts them: the
    environment Hermes hands a stdio server (`_build_safe_env` over the entry's
    `env`, read from the config Hermes reads), with the profile's .env loaded
    over it, as `hermes infino mcp` loads it as it starts."""
    started_with = _build_safe_env(_load_mcp_config()["infino"]["env"])
    for name in list(os.environ):
        if name not in started_with:
            monkeypatch.delenv(name)
    for name, value in started_with.items():
        monkeypatch.setenv(name, str(value))
    load_hermes_dotenv(hermes_home=home_dir)


@pytest.fixture
def started(home, monkeypatch):
    """The command the launcher would exec, captured instead."""
    out = {}

    def execvpe(path, argv, env):
        out.update(path=path, argv=argv, env=env)

    monkeypatch.setattr(mcp_launch.os, "execvpe", execvpe)
    home.setenv("INFINO_CLIENT", CLIENT_JS)
    home.setenv("INFINO_NODE", NODE)
    return out


def test_local_mode_starts_the_client_on_this_machines_database(started, tmp_path):
    mcp_launch.launch(config, local)
    assert started["argv"] == [NODE, CLIENT_JS, "mcp"]
    env = started["env"]
    assert env["CX_LOCAL_DB"] == str(tmp_path / "infino" / "db") and env["CX_TABLE"] == mcp_launch.PRIMARY_TABLE
    assert "CX_REMOTE_SEARCH" not in env
    store = local.LocalStore.under(config.state_dir(tmp_path))
    assert set(tables.ALL) <= set(store.list_tables()) and store.has_vectors()
    assert (config.state_dir(tmp_path) / "client-root").is_dir()


def test_signed_in_through_the_environment_starts_it_on_the_cloud(started, home, platform, tmp_path):
    # Hermes starts the launcher with the config's `env` block as its environment.
    home.setenv("INFINO_API_KEY", "the-key")
    home.setenv("INFINO_TEAM_DATABASE", TEAM_DB)
    home.setenv("INFINO_URL", platform.url + "/")
    mcp_launch.launch(config, local)
    assert started["argv"][2:5] == ["mcp", "--db", f"{platform.url}/{TEAM_DB}"]
    assert started["argv"][-2:] == ["--embed-provider", "platform"]
    assert {k: started["env"][k] for k in mcp_launch.CLOUD_ENV} == mcp_launch.CLOUD_ENV
    assert started["env"]["INFINO_API_KEY"] == "the-key" and "CX_LOCAL_DB" not in started["env"]
    assert not local.LocalStore.under(config.state_dir(tmp_path)).has_vectors()


def test_the_config_block_wins_over_the_profiles_env_file_for_the_tools_too(started, home, tmp_path):
    """`hermes infino mcp` is Hermes's own command line: as it starts, it loads
    the profile's .env over the environment Hermes started it with (the config
    block). The tools still read the team the block names, as the plugin does."""
    write_tools_setting(tmp_path, f"INFINO_TEAM_DATABASE: {TEAM_DB}")
    home.setenv("INFINO_TEAM_DATABASE", TEAM_DB)  # the block, as Hermes passes it to the tools
    home.setenv("INFINO_API_KEY", "k")
    (tmp_path / ".env").write_text("INFINO_TEAM_DATABASE=hermes-team\n")  # left by an earlier setup
    load_hermes_dotenv(hermes_home=tmp_path)  # what `hermes` does as it starts
    mcp_launch.launch(config, local)
    argv = started["argv"]
    assert argv[argv.index("--db") + 1].endswith(f"/{TEAM_DB}")
    assert config.load(tmp_path).team_database == TEAM_DB


def test_on_remote_config_the_tools_read_the_settings_the_plugin_reads(started, remote, home, platform, tmp_path,
                                                                       monkeypatch):
    """Hermes on Remote Config: the tools' entry comes from the config service,
    and the tools, started without the service's settings, read none of it
    themselves. They still search the team the plugin writes to."""
    remote.upper = {"memory": {"provider": "infino"}, "mcp_servers": {"infino": fleet_entry(platform)}}
    key_path = ("mcp_servers", "infino", "env", config.FLEET_NAMES[config.KEY_ENV])
    assert secret_literal_path(key_path, "test-key") is not None  # the service takes the key as a reference only
    assert secret_literal_path(key_path, "${INFINO_API_KEY}") is None
    (tmp_path / ".env").write_text("".join(f"{k}={v}\n" for k, v in LEFT_IN_ENV_FILE.items()))
    for name, value in LEFT_IN_ENV_FILE.items():  # Hermes loaded that .env as it started
        home.setenv(name, value)
    plugin = config.load(tmp_path)
    assert (plugin.team_database, plugin.url) == (TEAM_DB, platform.url)
    become_the_tools(monkeypatch, tmp_path)
    assert "HERMES_CONFIG_BACKEND" not in os.environ  # the tools cannot read Remote Config
    mcp_launch.launch(config, local)
    argv = started["argv"]
    assert argv[argv.index("--db") + 1] == f"{plugin.url}/{plugin.team_database}"
    assert started["env"]["INFINO_API_KEY"] == plugin.api_key


def test_signed_in_through_the_sign_in_file(started, platform, tmp_path):
    sign_in(tmp_path, platform)
    mcp_launch.launch(config, local)
    assert "--db" in started["argv"] and started["env"]["INFINO_API_KEY"] == "k"


def test_without_the_clients_path_it_stops_with_the_reason(started, monkeypatch):
    monkeypatch.delenv("INFINO_CLIENT")
    with pytest.raises(SystemExit, match="INFINO_CLIENT is not set"):
        mcp_launch.launch(config, local)
    assert started == {}


def test_as_a_script_its_modules_load_under_their_own_name(monkeypatch):
    for name in [n for n in sys.modules if n.startswith(mcp_launch.PACKAGE)]:
        monkeypatch.delitem(sys.modules, name)
    cfg, loc = mcp_launch._load_plugin()
    assert cfg.__name__ == f"{mcp_launch.PACKAGE}.config" and loc.__name__ == f"{mcp_launch.PACKAGE}.local"
    assert cfg is not config  # never the package the test imported
    for name in [n for n in sys.modules if n.startswith(mcp_launch.PACKAGE)]:
        monkeypatch.delitem(sys.modules, name)

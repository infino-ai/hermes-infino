"""The tools' launcher: what it starts, with which settings, in each mode."""

from __future__ import annotations

import sys

import pytest

from hermes_infino import config, local, mcp_launch, tables

from .helpers import TEAM_DB, sign_in

CLIENT_JS = "/opt/infino/client/dist/cli.js"
NODE = "/opt/node/bin/node"


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

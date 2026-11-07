"""The Infino client as Hermes's MCP server, reading what this profile reads.

Hermes starts one command for `mcp_servers.infino`:

    mcp_servers:
      infino:
        command: hermes
        args: [infino, mcp]
        env:
          INFINO_CLIENT: <path to the client's dist/cli.js>

`hermes infino mcp` (cli.py) runs `launch` inside Hermes's own environment,
where the plugin's dependencies are installed; a standard Hermes install keeps
that environment at a path that changes with its dependencies, so no fixed
Python path could name it. This file also runs as a script, under a Python
that has the dependencies (a development checkout's venv).

At each start it reads the profile's sign-in and runs the client over the
local database (no account) or the team's cloud database (signed in), so
signing in moves the agent's tools to the cloud at the next start with no
config to edit. The key reaches the client in its environment
(INFINO_API_KEY), never on its command line, where any process could read it.
Hermes hands an MCP server only an allowlist of its environment plus the
entry's `env`, so a deployment that signs in through config names the team
database and the key in that block, as INFINO_FLEET_TEAM_DATABASE and
`INFINO_FLEET_API_KEY: ${INFINO_API_KEY}` (config.FLEET_NAMES): names the
profile's `.env` never sets, so they reach this launcher as the block has them
on every deployment, Remote Config included, and the plugin reads them too.

Before the client starts, the local tables are created: the client opens the
local database's tables once, at its start, and a first session's plugin may
not have made them yet.

Run as a script by Hermes, so the plugin's modules are imported under a name
of their own (`hermes_infino_launch`): the folder is named `infino`, and the
Infino engine the local database needs is a package of that name too.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

# The client's settings for each mode: the doors over the tables the plugin
# keeps, conversations the table a call reads when it names none; never the
# client's own code index, and no answer tool (the agent answers).
PRIMARY_TABLE = "conversations"
LOCAL_ENV = {"CX_TABLE": PRIMARY_TABLE}
CLOUD_ENV = {"CX_TABLE": PRIMARY_TABLE, "CX_REMOTE_SEARCH": "1", "CX_AUTO_INDEX": "0", "CX_ANSWER_TOOL": "0"}
PACKAGE = "hermes_infino_launch"


def _load_plugin():
    """The plugin's package, imported from this file's folder under PACKAGE."""
    here = Path(__file__).resolve().parent
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, here / "__init__.py",
                                                      submodule_search_locations=[str(here)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{PACKAGE}.config"), importlib.import_module(f"{PACKAGE}.local")


def client_command(settings, client_js: str, node: str, empty_root: str,
                   local_db: str) -> tuple[list[str], dict[str, str]]:
    """The client's command line and the environment it adds, for `settings`;
    `local_db` is the team tables' local database, which it reads in local mode."""
    if settings.signed_in:
        argv = [node, client_js, "mcp", "--db", f"{settings.url.rstrip('/')}/{settings.team_database}",
                "--embed-provider", "platform"]
        env = {**CLOUD_ENV, "INFINO_API_KEY": settings.api_key, "CX_ROOT": empty_root}
    else:
        argv = [node, client_js, "mcp"]
        env = {**LOCAL_ENV, "CX_LOCAL_DB": local_db, "CX_ROOT": empty_root}
    return argv, env


def node_command() -> str:
    """INFINO_NODE, else the Node Hermes's installer keeps, else `node` on PATH."""
    if os.environ.get("INFINO_NODE"):
        return os.environ["INFINO_NODE"]
    try:  # inside Hermes only; a script run outside it has no hermes_constants
        from hermes_constants import with_hermes_node_path
    except ImportError:
        return "node"
    return shutil.which("node", path=with_hermes_node_path({"PATH": ""})["PATH"]) or "node"


def launch(config, local, *, from_config: bool = True) -> None:
    """Create the local tables, then become the client (never returns).

    Hermes starts this process with the config's `env` block as its
    environment. As `hermes infino mcp`, it is Hermes's own command line, which
    loads the profile's `.env` over that environment as it starts: the block's
    fleet names (config.FLEET_NAMES), which `.env` never sets, still hold what
    the block says, and its usual names are read from the config, as the
    plugin reads them, where this process can read it (a file; it starts
    without Remote Config's settings). Run as a script (`from_config=False`),
    nothing loads `.env`, and the block is read from the environment Hermes
    gave it."""
    settings = config.load(from_config=from_config)
    store = local.LocalStore.under(config.state_dir(settings.home))
    store.ensure_tables(with_vectors=not settings.signed_in)
    client_js = os.environ.get("INFINO_CLIENT") or ""
    if not client_js:
        sys.exit("INFINO_CLIENT is not set: the path to the Infino client's dist/cli.js")
    # A folder the client may take as its root; nothing is indexed in it.
    empty_root = str(config.state_dir(settings.home) / "client-root")
    os.makedirs(empty_root, exist_ok=True)
    argv, env = client_command(settings, client_js, node_command(), empty_root, str(store.path))
    sys.stdout.flush()
    os.execvpe(argv[0], argv, {**os.environ, **env})


def main() -> None:
    # Run as a script, this folder is first on the import path, where its
    # modules (config, client, local...) would shadow any library's of the
    # same name; the plugin's are reached through PACKAGE instead.
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(here)]
    launch(*_load_plugin(), from_config=False)


if __name__ == "__main__":
    main()

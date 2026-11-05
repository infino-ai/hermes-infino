"""Settings, read from three places, first found wins:

1. the `env` block of the Infino tools' entry in the profile's config
   (`mcp_servers.infino.env`), through Hermes's own config loader, so a fleet's
   Remote Config sets them the way it sets everything else. The key is a
   `${INFINO_API_KEY}` reference there (Remote Config refuses a secret
   written out), which the loader resolves from the profile's environment;
2. the active profile's environment (`.env`) through Hermes's secret scope, so
   each profile on a multiplexed gateway reads its own;
3. the profile's sign-in (`credentials.json`, written by `hermes infino login`).

The config block comes first because it is also the tools' environment: Hermes
starts the tools with that block, so the plugin and its tools always read the
same team. A `${VAR}` reference that resolves to nothing counts as unset.

Without a key and a team database the plugin runs on the local database
alone. With both it is signed in: the cloud database is written too, and read
first.

- INFINO_API_KEY             the key (a secret; minted per person or bot)
- INFINO_TEAM_DATABASE       the team's database
- INFINO_PERSONAL_DATABASE   this person's database (optional; personal scope)
- INFINO_URL                 the platform URL (defaults to Infino Cloud)
- INFINO_AUTHOR              who this profile's turns and memories are from, when
                             Hermes does not say (a CLI or desktop profile has no
                             gateway user); defaults to the profile name
- INFINO_MCP_SERVER          the Infino client's name under `mcp_servers`
                             (default `infino`), whose instructions this plugin
                             hands the model; read from the environment only,
                             since it names the config block
- INFINO_TEAM_SKILLS         `off` stops this profile loading the skills its
                             teammates publish (default on)

A variable that is set wins over the sign-in, so a deployment's `.env` is
never overridden by a file on the machine.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .client import DEFAULT_URL

try:  # Hermes's profile-scoped secret lookup; plain env outside Hermes.
    from agent.secret_scope import get_secret as _get_secret
except ImportError:  # pragma: no cover - only outside a Hermes install
    _get_secret = None
try:  # Hermes's home for the profile running now: a gateway serving several
    # profiles sets it per turn, not in the process environment.
    from hermes_constants import get_hermes_home as _get_hermes_home
except ImportError:  # pragma: no cover - only outside a Hermes install
    _get_hermes_home = None

logger = logging.getLogger(__name__)

KEY_ENV = "INFINO_API_KEY"
TEAM_DB_ENV = "INFINO_TEAM_DATABASE"
PERSONAL_DB_ENV = "INFINO_PERSONAL_DATABASE"
URL_ENV = "INFINO_URL"
AUTHOR_ENV = "INFINO_AUTHOR"
MCP_SERVER_ENV = "INFINO_MCP_SERVER"
TEAM_SKILLS_ENV = "INFINO_TEAM_SKILLS"

DEFAULT_MCP_SERVER = "infino"
# Values of INFINO_TEAM_SKILLS that turn the team's skills off.
OFF_VALUES = ("off", "false", "no", "0")
# A `${VAR}` reference Hermes left as written: its variable is not set.
UNRESOLVED_REF = re.compile(r"^\$\{[^}]*\}$")

# The plugin's state under the profile's home: the local database, the
# sign-in, the queue of cloud writes not yet made, and the push's progress.
STATE_DIR = Path("infino")
CREDENTIALS_FILE = "credentials.json"
# Owner read/write only: the file holds an API key.
CREDENTIALS_MODE = 0o600


def home() -> Path:
    """The home of the profile running now, as Hermes resolves it."""
    if _get_hermes_home is not None:
        return Path(_get_hermes_home())
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def state_dir(home_dir: Optional[Path] = None) -> Path:
    return (home_dir or home()) / STATE_DIR


def _set(value) -> str:
    """`value` as a setting, or "" when it is empty or an unresolved `${VAR}`."""
    value = str(value or "").strip()
    return "" if UNRESOLVED_REF.match(value) else value


def _read(name: str) -> str:
    value = _get_secret(name) if _get_secret else None
    if value is None:
        value = os.environ.get(name)
    return _set(value)


def tools_env(server: str, home_dir: Path) -> dict:
    """The `env` block of `mcp_servers.<server>` in the profile's config, with
    `${VAR}` references resolved; {} outside Hermes, or for a home that is not
    the profile Hermes has loaded (its config is not this one's).

    Hermes's loader is imported here, not at the top: inside Hermes it is
    loaded already, and a process outside it never pays for it (or starts
    Remote Config by importing it)."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
    except ImportError:
        return {}
    try:
        if _get_hermes_home is None or home().resolve() != Path(home_dir).resolve():
            return {}
        env = cfg_get(load_config_readonly(), "mcp_servers", server, "env")
    except Exception as e:  # an unreadable config must not take memory down with it
        logger.warning("infino: could not read mcp_servers.%s.env from the config: %s", server, e)
        return {}
    return env if isinstance(env, dict) else {}


def read_credentials(home_dir: Optional[Path] = None) -> dict:
    """The profile's sign-in, or {} when it has none (or it is unreadable)."""
    try:
        data = json.loads((state_dir(home_dir) / CREDENTIALS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_credentials(creds: dict, home_dir: Optional[Path] = None) -> Path:
    """Store the sign-in with owner-only permissions; returns its path."""
    path = state_dir(home_dir) / CREDENTIALS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # A file left by a crash keeps its own mode through os.open: start a new one.
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, CREDENTIALS_MODE)
    with os.fdopen(fd, "w") as f:
        json.dump(creds, f, indent=2)
    os.replace(tmp, path)
    return path


@dataclass(frozen=True)
class Settings:
    api_key: str
    team_database: str
    personal_database: Optional[str]
    url: str
    author: str = ""
    mcp_server: str = DEFAULT_MCP_SERVER
    home: Path = Path(".")
    # Where the key and team database came from: "config", "environment",
    # "sign-in", or "" when there are none.
    source: str = ""
    # Whether the skills teammates publish load in this profile.
    team_skills: bool = True

    @property
    def signed_in(self) -> bool:
        """Whether the cloud database is written and read, not the local one alone."""
        return bool(self.api_key and self.team_database)


def load(home_dir: Optional[Path] = None, *, from_config: bool = True) -> Settings:
    """This profile's settings. `from_config=False` skips the config block: the
    tools' launcher is started with that block as its environment already, and
    a process Hermes did not start must not open Remote Config itself."""
    home_dir = home_dir or home()
    creds = read_credentials(home_dir)
    server = _read(MCP_SERVER_ENV) or DEFAULT_MCP_SERVER
    block = tools_env(server, home_dir) if from_config else {}

    def pick(env_name: str, cred_key: str = "") -> tuple[str, str]:
        for value, where in ((_set(block.get(env_name)), "config"), (_read(env_name), "environment"),
                             (_set(creds.get(cred_key)) if cred_key else "", "sign-in")):
            if value:
                return value, where
        return "", ""

    api_key, key_from = pick(KEY_ENV, "api_key")
    team_database, team_from = pick(TEAM_DB_ENV, "team_database")
    return Settings(
        api_key=api_key,
        team_database=team_database,
        personal_database=pick(PERSONAL_DB_ENV, "personal_database")[0] or None,
        url=pick(URL_ENV, "url")[0] or DEFAULT_URL,
        author=pick(AUTHOR_ENV)[0],
        mcp_server=server,
        home=home_dir,
        source=team_from if team_from == key_from else "+".join(w for w in (team_from, key_from) if w),
        team_skills=pick(TEAM_SKILLS_ENV)[0].lower() not in OFF_VALUES,
    )


def config_schema() -> list[dict]:
    """Fields for `hermes memory setup`; every one is an env var, so the
    provider needs no `save_config`. None is required: without a key the
    plugin keeps everything on this machine (`hermes infino login` signs in)."""
    return [
        {
            "key": "api_key",
            "description": "Infino API key for this person or bot (empty: keep everything on this machine)",
            "secret": True,
            "env_var": KEY_ENV,
        },
        {
            "key": "team_database",
            "description": "The team's Infino database",
            "env_var": TEAM_DB_ENV,
        },
        {
            "key": "personal_database",
            "description": "Your personal Infino database (leave empty for bots and shared profiles)",
            "env_var": PERSONAL_DB_ENV,
        },
        {
            "key": "author",
            "description": "Your name, as teammates should see it on your memories (bots: the bot's name)",
            "env_var": AUTHOR_ENV,
        },
        {
            "key": "mcp_server",
            "description": "The Infino client's name under mcp_servers",
            "default": DEFAULT_MCP_SERVER,
            "env_var": MCP_SERVER_ENV,
        },
        {
            "key": "url",
            "description": "Infino platform URL",
            "default": DEFAULT_URL,
            "env_var": URL_ENV,
        },
        {
            "key": "team_skills",
            "description": "Load the skills teammates publish (off: only this profile's own)",
            "default": "on",
            "env_var": TEAM_SKILLS_ENV,
        },
    ]

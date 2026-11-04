"""Settings, read from the active profile's environment (`.env`) through
Hermes's secret scope, so each profile on a multiplexed gateway reads its own,
and from the profile's sign-in (`credentials.json`, written by
`hermes infino login`).

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
                             hands the model

A variable that is set wins over the sign-in, so a deployment's `.env` is
never overridden by a file on the machine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .client import DEFAULT_URL

try:  # Hermes's profile-scoped secret lookup; plain env outside Hermes.
    from agent.secret_scope import get_secret as _get_secret
except ImportError:  # pragma: no cover - only outside a Hermes install
    _get_secret = None

KEY_ENV = "INFINO_API_KEY"
TEAM_DB_ENV = "INFINO_TEAM_DATABASE"
PERSONAL_DB_ENV = "INFINO_PERSONAL_DATABASE"
URL_ENV = "INFINO_URL"
AUTHOR_ENV = "INFINO_AUTHOR"
MCP_SERVER_ENV = "INFINO_MCP_SERVER"

DEFAULT_MCP_SERVER = "infino"

# The plugin's state under the profile's home: the local database, the
# sign-in, the queue of cloud writes not yet made, and the push's progress.
STATE_DIR = Path("infino")
CREDENTIALS_FILE = "credentials.json"
# Owner read/write only: the file holds an API key.
CREDENTIALS_MODE = 0o600


def home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def state_dir(home_dir: Optional[Path] = None) -> Path:
    return (home_dir or home()) / STATE_DIR


def _read(name: str) -> str:
    value = _get_secret(name) if _get_secret else None
    if value is None:
        value = os.environ.get(name)
    return (value or "").strip()


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
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, CREDENTIALS_MODE)
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

    @property
    def signed_in(self) -> bool:
        """Whether the cloud database is written and read, not the local one alone."""
        return bool(self.api_key and self.team_database)


def load(home_dir: Optional[Path] = None) -> Settings:
    home_dir = home_dir or home()
    creds = read_credentials(home_dir)
    return Settings(
        api_key=_read(KEY_ENV) or str(creds.get("api_key") or ""),
        team_database=_read(TEAM_DB_ENV) or str(creds.get("team_database") or ""),
        personal_database=_read(PERSONAL_DB_ENV) or creds.get("personal_database") or None,
        url=_read(URL_ENV) or str(creds.get("url") or "") or DEFAULT_URL,
        author=_read(AUTHOR_ENV),
        mcp_server=_read(MCP_SERVER_ENV) or DEFAULT_MCP_SERVER,
        home=home_dir,
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
    ]

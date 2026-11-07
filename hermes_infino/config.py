"""Settings, read from three places, first found wins:

1. the `env` block of the Infino tools' entry in the profile's config
   (`mcp_servers.infino.env`), through Hermes's own config loader, so a fleet's
   Remote Config sets them the way it sets everything else. The key is a
   `${INFINO_API_KEY}` reference there (Remote Config refuses a secret
   written out), which the loader resolves from the profile's environment;
2. the active profile's environment (`.env`), through Hermes's secret scope
   alone: on a gateway serving several profiles each reads its own, and a name
   its own does not set is unset, never the launch profile's (Hermes reads its
   own `${VAR}` the same way);
3. the profile's sign-in (`credentials.json`, written by `hermes infino login`).

Hermes starts the tools with that block as their environment, and `hermes
infino mcp` then loads the profile's `.env` over it, as every `hermes` command
does. So the three settings the tools read (the key, team database and URL)
have names of their own for the block, which `.env` never sets (FLEET_NAMES):
written under those, a setting reaches the tools as the block has it, on
every deployment, and the plugin reads it first too. Under its usual name the
tools read it from the config themselves; on Remote Config they cannot (Hermes
starts them without its settings), and `.env` setting the same name wins
there. A `${VAR}` reference that resolves to nothing counts as unset. The
block holds only text in the tools' environment: a value YAML reads as
something else (a bare `on`, a number, a list) is warned about, since the
tools would not start with it.

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
- INFINO_PERSONAL_PROFILE    `on` makes this profile's conversations and traces
                             personal: kept in the personal database, or on
                             this machine, never the team's (default off). Any
                             value but an off one counts as on: a spelling read
                             wrongly must not put personal work in the team's
                             tables
- INFINO_PERSONAL_JOBS       the scheduled jobs (ids or names, comma-separated,
                             or a list) whose runs, and the chats their output
                             is delivered into, are personal the same way
- INFINO_OWNER_ROUTES        on a personal profile, the webhook routes
                             (comma-separated, or a list) whose turns are the
                             owner's own: their events are the owner's
                             business, so they read and write the owner's
                             personal memory (chats.py)
- INFINO_FLEET_API_KEY, INFINO_FLEET_TEAM_DATABASE, INFINO_FLEET_URL
                             the key, team database and URL, written in the
                             config block (FLEET_NAMES)

A variable that is set wins over the sign-in, so a deployment's `.env` is
never overridden by a file on the machine (and `login` refuses a sign-in one
of them would override). The sign-in's URL and personal database go with its
key, so they apply only when the key in use is the sign-in's: another key,
from the config block or the environment, is never sent to the sign-in's
platform, and never writes into its personal database.

A personal database that is the team database counts as unset, wherever it is
named (the next place's applies): personal rows would land in the team's
tables, where everyone reads them.
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
PERSONAL_PROFILE_ENV = "INFINO_PERSONAL_PROFILE"
PERSONAL_JOBS_ENV = "INFINO_PERSONAL_JOBS"
OWNER_ROUTES_ENV = "INFINO_OWNER_ROUTES"

DEFAULT_MCP_SERVER = "infino"
# The settings a sign-in holds. The config block or the environment may set
# each instead, and then that value is the one in effect.
SIGN_IN_SETTINGS = (KEY_ENV, TEAM_DB_ENV, PERSONAL_DB_ENV, URL_ENV)
# The settings the Infino tools read (mcp_launch.client_command), with the name
# each has in the config block. Hermes starts the tools with the block as their
# environment, and `hermes infino mcp` loads the profile's `.env` over it as it
# starts, so a name `.env` sets too would reach the tools as `.env` has it.
# Nothing writes these names into `.env` (`hermes memory setup` writes only the
# declared settings). The plugin reads them first, from the environment (where
# the tools find the block) then from the block, as the tools do: both read the
# same.
FLEET_NAMES = {KEY_ENV: "INFINO_FLEET_API_KEY", TEAM_DB_ENV: "INFINO_FLEET_TEAM_DATABASE",
               URL_ENV: "INFINO_FLEET_URL"}
# Values that turn a switch off (INFINO_TEAM_SKILLS, INFINO_PERSONAL_PROFILE).
OFF_VALUES = ("off", "false", "no", "0")
# Values that turn a switch on; any other spelling of INFINO_PERSONAL_PROFILE
# turns it on too, with a warning.
ON_VALUES = ("on", "true", "yes", "1")
# What separates the names of a list setting (INFINO_PERSONAL_JOBS,
# INFINO_OWNER_ROUTES); a YAML list's items are joined with it.
JOBS_SEPARATOR = ","
# A `${VAR}` reference Hermes left as written: its variable is not set.
UNRESOLVED_REF = re.compile(r"^\$\{[^}]*\}$")
# Hermes reads its config as YAML 1.1, where a bare on, off, yes, no, true or
# false is a boolean: each reads as the setting it says.
BOOLEAN_SETTINGS = {True: "on", False: "off"}
# Where a setting came from, as `hermes infino status` names it.
FROM_CONFIG = "config"
FROM_ENVIRONMENT = "environment"
FROM_SIGN_IN = "sign-in"

# The plugin's state under the profile's home: the local database, the
# sign-in, the queue of cloud writes not yet made, and the push's progress.
STATE_DIR = Path("infino")
CREDENTIALS_FILE = "credentials.json"
# Owner read/write only: the file holds an API key.
CREDENTIALS_MODE = 0o600
# Owner only: the state directory holds every conversation, and personal
# memory. Hermes keeps its home owner-only too, but not in a container
# (unless HERMES_HOME_MODE says so), where a data directory is often shared.
PRIVATE_DIR_MODE = 0o700

# What was warned about already, each with the profile home it was found in:
# settings load on every turn, and one warning says it.
_warned: set[tuple[str, ...]] = set()


def make_private_dir(path: Path) -> None:
    """Create `path` (and its parents) and make it owner-only."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, PRIVATE_DIR_MODE)


def home() -> Path:
    """The home of the profile running now, as Hermes resolves it."""
    if _get_hermes_home is not None:
        return Path(_get_hermes_home())
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def state_dir(home_dir: Optional[Path] = None) -> Path:
    return (home_dir or home()) / STATE_DIR


def _set(value) -> str:
    """`value` as a setting, or "" when it is missing, empty or an unresolved
    `${VAR}`. A YAML boolean or number from the config block is the setting
    it was written as (`off` is "off", `0` is "0"), never unset; a YAML list
    is its items, as INFINO_PERSONAL_JOBS lists them."""
    if value is None:
        return ""
    if isinstance(value, bool):
        value = BOOLEAN_SETTINGS[value]
    elif isinstance(value, list):
        value = JOBS_SEPARATOR.join(item for item in map(_set, value) if item)
    value = str(value).strip()
    return "" if UNRESOLVED_REF.match(value) else value


def _read(name: str) -> str:
    """`name` from the profile's environment. Inside Hermes that is its secret
    scope's answer alone (agent/secret_scope.get_secret): for a profile served
    beside others (by a gateway serving several, or a desktop serving another)
    its own `.env`, where a name it does not set is unset, never the value the
    process environment holds for the profile Hermes started with; otherwise
    the process environment."""
    if _get_secret is not None:
        return _set(_get_secret(name))
    return _set(os.environ.get(name))


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


def credentials_path(home_dir: Optional[Path] = None) -> Path:
    """Where the profile's sign-in is kept."""
    return state_dir(home_dir) / CREDENTIALS_FILE


def read_credentials(home_dir: Optional[Path] = None) -> dict:
    """The profile's sign-in, or {} when it has none (or it is unreadable)."""
    try:
        data = json.loads(credentials_path(home_dir).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def sign_in_key(home_dir: Optional[Path] = None) -> str:
    """The key the profile's sign-in holds; "" without one."""
    return _set(read_credentials(home_dir).get("api_key"))


def write_credentials(creds: dict, home_dir: Optional[Path] = None) -> Path:
    """Store the sign-in with owner-only permissions; returns its path."""
    path = credentials_path(home_dir)
    make_private_dir(path.parent)
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
    # Whether every conversation and trace of this profile is personal.
    personal_profile: bool = False
    # The scheduled jobs (ids or names) whose runs and delivered output are personal.
    personal_jobs: tuple[str, ...] = ()
    # The webhook routes whose turns are the owner's own (on a personal profile only).
    owner_routes: tuple[str, ...] = ()

    @property
    def signed_in(self) -> bool:
        """Whether the cloud database is written and read, not the local one alone."""
        return bool(self.api_key and self.team_database)


def _warn_once(seen: tuple[str, ...], message: str, *args) -> None:
    """Log `message` the first time `seen` comes up in this process."""
    if seen in _warned:
        return
    _warned.add(seen)
    logger.warning(message, *args)


def _names_the_team(personal: str, team: str) -> bool:
    """Whether `personal` is the team database: as a personal database, it
    counts as unset (personal rows would land in the team's tables)."""
    return bool(personal) and personal == team


def _warn_personal_is_team(home_dir: Path, name: str, where: str) -> None:
    """Say once per profile and place that the personal database named there
    is the team's, so it is not used."""
    _warn_once(("personal is team", str(home_dir), name, where),
               "infino: %s names the team database '%s' as the personal database, so it is not used: personal "
               "rows would land in the team's tables, where everyone reads them. Name another there.", where, name)


def _warn_not_text(home_dir: Path, server: str, block: dict) -> None:
    """Say once per profile and setting that a value in the config block is
    not text (YAML reads a bare on, off, yes, no, true or false, a number or a
    list as something else). Hermes starts the tools with the block as their
    environment, which holds only text, so they would not start; written in
    quotes, it is text."""
    for name, value in block.items():
        if not isinstance(value, str):
            _warn_once(("not text", str(home_dir), str(name)),
                       "infino: mcp_servers.%s.env.%s is %r, not text: Hermes starts the Infino tools with that "
                       "block as their environment, which takes only text, so they will not start. Quote it in "
                       "the config: %s: \"%s\"", server, name, value, name, _set(value))


def _personal_profile(home_dir: Path, value: str) -> bool:
    """INFINO_PERSONAL_PROFILE as a switch. Only an off value (or none) turns
    it off: a spelling this plugin does not know (`y`, `enabled`) turns it on,
    with a warning, since reading it as off would put the profile's
    conversations and traces in the team's tables."""
    setting = value.lower()
    if not setting or setting in OFF_VALUES:
        return False
    if setting not in ON_VALUES:
        _warn_once(("personal profile", str(home_dir), value),
                   "infino: %s is '%s', which is neither on nor off, so this profile is personal: its "
                   "conversations and traces stay out of the team database. Set it to on or off.",
                   PERSONAL_PROFILE_ENV, value)
    return True


def _server() -> str:
    """The Infino tools' name under `mcp_servers`: whose `env` block is read."""
    return _read(MCP_SERVER_ENV) or DEFAULT_MCP_SERVER


def _outside_sign_in(name: str, block: dict) -> tuple[str, str, str]:
    """Setting `name` from the config block or the environment: (value, where
    it came from, the variable it is set under); ("", "", "") when neither sets
    it. A setting the tools read comes under its fleet name first
    (FLEET_NAMES), then under its own: from the block, else the environment."""
    fleet = FLEET_NAMES.get(name)
    places = ((_read(fleet), FROM_ENVIRONMENT, fleet), (_set(block.get(fleet)), FROM_CONFIG, fleet)) if fleet else ()
    for value, where, variable in (*places, (_set(block.get(name)), FROM_CONFIG, name),
                                   (_read(name), FROM_ENVIRONMENT, name)):
        if value:
            return value, where, variable
    return "", "", ""


def overrides(home_dir: Optional[Path] = None, *, team_database: str = "") -> dict[str, tuple[str, str, str]]:
    """The settings a sign-in holds that the config block or the environment
    sets, each taking precedence over any sign-in: {setting: (value, where,
    the variable it is set under)}. `team_database` is the one a sign-in
    would hold: a personal database that is the team database in effect
    counts as unset, as `load` counts it."""
    home_dir = home_dir or home()
    block = tools_env(_server(), home_dir)
    found = {name: _outside_sign_in(name, block) for name in SIGN_IN_SETTINGS}
    if _names_the_team(found[PERSONAL_DB_ENV][0], found[TEAM_DB_ENV][0] or team_database):
        found[PERSONAL_DB_ENV] = ("", "", "")
    return {name: found[name] for name in SIGN_IN_SETTINGS if found[name][0]}


def load(home_dir: Optional[Path] = None, *, from_config: bool = True) -> Settings:
    """This profile's settings. `from_config=False` skips the config block: for
    the tools' launcher run as a script, which Hermes starts with that block as
    its environment already, and which must not open Remote Config itself."""
    home_dir = home_dir or home()
    creds = read_credentials(home_dir)
    server = _server()
    block = tools_env(server, home_dir) if from_config else {}
    _warn_not_text(home_dir, server, block)

    def pick(env_name: str, cred_key: str = "") -> tuple[str, str]:
        value, where, _ = _outside_sign_in(env_name, block)
        if value or not cred_key:
            return value, where
        value = _set(creds.get(cred_key))
        return (value, FROM_SIGN_IN) if value else ("", "")

    api_key, key_from = pick(KEY_ENV, "api_key")
    team_database, team_from = pick(TEAM_DB_ENV, "team_database")
    # The sign-in's URL and personal database belong to its key. With another
    # key (a fleet's), they would send that key to the sign-in's platform, or
    # put that account's personal rows in the sign-in's database.
    own_key = bool(api_key) and api_key == _set(creds.get("api_key"))
    # The first personal database named that is not the team database.
    outside, _, variable = _outside_sign_in(PERSONAL_DB_ENV, block)
    signed = _set(creds.get("personal_database")) if own_key else ""
    personal_database = ""
    for named, where in ((outside, variable), (signed, "the sign-in")):
        if _names_the_team(named, team_database):
            _warn_personal_is_team(home_dir, named, where)
        elif named:
            personal_database = named
            break
    return Settings(
        api_key=api_key,
        team_database=team_database,
        personal_database=personal_database or None,
        url=pick(URL_ENV, "url" if own_key else "")[0] or DEFAULT_URL,
        author=pick(AUTHOR_ENV)[0],
        mcp_server=server,
        home=home_dir,
        source=team_from if team_from == key_from else "+".join(w for w in (team_from, key_from) if w),
        team_skills=pick(TEAM_SKILLS_ENV)[0].lower() not in OFF_VALUES,
        personal_profile=_personal_profile(home_dir, pick(PERSONAL_PROFILE_ENV)[0]),
        personal_jobs=_names(pick(PERSONAL_JOBS_ENV)[0]),
        owner_routes=_names(pick(OWNER_ROUTES_ENV)[0]),
    )


def _names(value: str) -> tuple[str, ...]:
    """The names a list setting holds (comma-separated, as `_set` writes a YAML list)."""
    return tuple(name.strip() for name in value.split(JOBS_SEPARATOR) if name.strip())


def config_schema() -> list[dict]:
    """Fields for `hermes memory setup`; every one is an env var, so the
    provider needs no `save_config`. None is required: without a key the
    plugin keeps everything on this machine (`hermes infino login` signs in).

    The setup writes a field's default into `.env` when the person just
    presses Enter, and `.env` takes precedence over a sign-in: so no setting a
    sign-in holds has a default here (the URL's is in its description), or a
    later `login --url` would be overridden by a value nobody chose."""
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
            "description": f"Infino platform URL (empty: {DEFAULT_URL})",
            "env_var": URL_ENV,
        },
        {
            "key": "team_skills",
            "description": "Load the skills teammates publish (off: only this profile's own)",
            "default": "on",
            "env_var": TEAM_SKILLS_ENV,
        },
        {
            "key": "personal_profile",
            "description": "on: this profile's conversations and traces are personal (your personal database, "
                           "or this machine), never the team's",
            "default": "off",
            "env_var": PERSONAL_PROFILE_ENV,
        },
        {
            "key": "personal_jobs",
            "description": "Scheduled jobs (names, or this profile's job ids, comma-separated) whose runs and "
                           "delivered output are personal, never the team's",
            "env_var": PERSONAL_JOBS_ENV,
        },
        {
            "key": "owner_routes",
            "description": "On a personal profile: webhook routes (comma-separated) whose events are your own, so "
                           "their turns read and write your personal memory",
            "env_var": OWNER_ROUTES_ENV,
        },
    ]

"""`hermes infino ...`: sign in, see where things stand, catch the cloud up.

    hermes infino status    local or signed in; rows here; how far the cloud is
    hermes infino login     create a free account, or use a key you were given,
                            then push everything kept on this machine
    hermes infino sync      push what the cloud does not have yet, now
    hermes infino mcp       the Infino tools' server; Hermes starts it from
                            `mcp_servers.infino` (not a command for people)

Signing in uploads this machine's memories, conversations (tool results
included), traces and skills, and from then on everything new. So `login`
asks first, at the terminal, and only the person at the terminal can answer:
there is no flag that agrees to creating an account, and without a terminal
it refuses. Using a key someone gave you (`--api-key-file`) may be agreed to
with `--yes`, since that person chose to hand it over.

What is personal (personal memory, and the conversations and traces of
personal profiles and scheduled jobs) goes to a personal database, which
`login` makes for this sign-in unless one is named (`--personal-database`) or
none is wanted (`--no-personal-database`). A database made this way is always
a new one, named after the team with a random part, so no two people ever
share one. When it cannot be made (a key that may not make that database, or
a platform that does not answer), the sign-in goes on without one and what is
personal stays on this machine. The sign-in is saved before that step: the
platform gives a new account's key only once.

The config block and the environment take precedence over a sign-in
(config.py). So `login` names the team and personal database they set, and
refuses, before anything is created or sent, a URL, team database, personal
database or key they set otherwise, naming the variable and where it is set:
the notice would name a place the rows never go. A personal database they set
to the team database counts as unset, as it does everywhere.

Hermes imports this module to build its command line for every command, so
the engine and the model are imported only inside the commands that use them.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
import urllib.error
from http.client import HTTPException
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

from . import config
from .client import DEFAULT_URL, InfinoClient, InfinoError, request_trial
from .tables import OutdatedTable

# The team database a new account registers when none is named.
DEFAULT_DATABASE = "hermes-team"
# The platform's answers to a free-account request that refuse it: it gives
# out none, or this network already had one.
NO_FREE_ACCOUNTS = 501
FREE_ACCOUNT_TAKEN = 409
# Seconds `login` and `sync` keep catching up while rows are still arriving.
CATCH_UP_S = 60.0
# A personal database `login` makes: `<team><PERSONAL_PART><random hex>`.
PERSONAL_PART = "-personal-"
# Random bytes in that name (hex: twice as many characters).
PERSONAL_RANDOM_BYTES = 4
# The longest database name the platform takes.
MAX_DATABASE_NAME = 128
# What a platform URL starts with (any case).
URL_SCHEMES = ("https://", "http://")
# What each setting a sign-in holds is, as a refusal names it.
SETTING_NAMES = {config.URL_ENV: "the platform", config.TEAM_DB_ENV: "the team database",
                 config.PERSONAL_DB_ENV: "the personal database"}
# How to have a personal database without signing in again: name one, and the
# sync makes it (tables.ensure_all), or takes one of that name as it is. A
# running Hermes read its environment when it started.
ADD_PERSONAL_DATABASE = (f"To have a personal database without signing in again, name one that is yours alone: "
                         f"`hermes config set {config.PERSONAL_DB_ENV} <name>`. The next sync makes it (a database "
                         f"of that name that exists already is used as it is), and a running Hermes uses it once it "
                         f"restarts.")


def register_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="infino_action")
    sub.add_parser("status", help="local or signed in, and how far the cloud is")
    login = sub.add_parser("login", help="sign in to Infino and push this machine's memory, conversations, traces and skills")
    login.add_argument("--url", default="", help=f"platform URL (default {DEFAULT_URL})")
    login.add_argument("--api-key-file", default="", help="use this key (a team's) instead of creating an account")
    login.add_argument("--team-database", default="", help=f"the team's database (default {DEFAULT_DATABASE})")
    login.add_argument("--personal-database", default="",
                       help="your personal database, made if it is missing (default: a new one for this sign-in)")
    login.add_argument("--no-personal-database", action="store_true",
                       help="make no personal database: what is personal stays on this machine")
    login.add_argument("--yes", action="store_true", help="agree to the upload when using --api-key-file")
    sub.add_parser("sync", help="push what the cloud does not have yet")
    sub.add_parser("mcp", help="the Infino tools' MCP server, which Hermes starts (mcp_servers.infino)")


def infino_command(args: argparse.Namespace) -> int:
    action = getattr(args, "infino_action", None) or "status"
    try:
        return {"status": _status, "login": _login, "sync": _sync, "mcp": _mcp}[action](args)
    except InfinoError as e:
        print(f"infino: the platform refused: {e}", file=sys.stderr)
    except OutdatedTable as e:
        print(f"infino: {e}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"infino: could not reach the platform: {e.reason}", file=sys.stderr)
    except TimeoutError:
        print("infino: the platform did not answer in time", file=sys.stderr)
    except (ConnectionError, HTTPException) as e:  # a connection dropped before any answer
        print(f"infino: could not reach the platform: {e}", file=sys.stderr)
    except JSONDecodeError:  # a proxy's page, say
        print("infino: the URL did not answer as the Infino platform does (its answer is not JSON): check the "
              "URL, and any proxy on the way", file=sys.stderr)
    return 1


# -- status -------------------------------------------------------------------

def _status(args) -> int:
    from .backend import Backend
    from . import tables

    settings = config.load()
    b = Backend(settings)
    for label, store, names in (("local database ", b.local, tables.ALL),
                                ("personal       ", b.personal, tables.PERSONAL)):
        print(f"{label}  {store.path}")
        try:
            have = set(store.list_tables())
            for t in names:
                n = store.query_sql(f"SELECT count(*) AS n FROM {t}")[0]["n"] if t in have else 0
                print(f"   {t:<14} {n} rows")
        except Exception as e:
            print(f"   unreadable: {e}")
    try:
        if b.local.has_vectors():
            vectors = "on (local model)"
        elif settings.signed_in:
            vectors = "off (the cloud embeds)"
        else:  # local mode: the provider makes them when Hermes starts
            vectors = "not made yet (Hermes makes them when it starts)"
        print(f"   memory vectors {vectors}")
    except Exception:
        pass
    if not settings.signed_in:
        print("signed in        no - everything stays on this machine (`hermes infino login` to sign in)")
        return 0
    print(f"signed in        {settings.url} / {settings.team_database}"
          + (f" (personal: {settings.personal_database})" if settings.personal_database else "")
          + f", from the {settings.source}")
    marks = b.mirror.marks()
    for src in b.mirror.sources():
        try:
            behind = src.store.count_after(src.table, marks.get(src.mark), int(time.time() * 1000))
        except Exception:
            behind = "?"
        print(f"   {src.table:<14} {behind} rows not in {src.database} yet")
    if not settings.personal_database:
        print("   personal memory stays on this machine, as do personal conversations and traces "
              "(no personal database)")
    if b.mirror.last_error():
        print(f"last sync        stopped: {b.mirror.last_error()}")
    refused = _refused(b)
    if refused:
        print(f"refused          {refused}; they stay here")
    return 0


def _refused(b) -> str:
    """The rows the cloud refused outright, as `status` and `sync` say it;
    "" when there are none."""
    from .mirror import REFUSED_FILE

    n = b.mirror.refused()
    if not n:
        return ""
    return f"{n} rows the cloud would not take (listed in {config.state_dir(b.settings.home) / REFUSED_FILE})"


# -- login --------------------------------------------------------------------

def _ask(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


# Where what is personal goes, as the upload notice says it, when login makes
# the personal database.
NEW_PERSONAL_DATABASE = "a personal database of your own, made now"


def _personal_destination(personal: Optional[str]) -> str:
    """Where what is personal goes, in the upload notice's words: `personal`
    is the database named, "" for a new one, None for none."""
    if personal is None:
        return "nowhere: it stays on this machine (--no-personal-database)"
    if personal:
        return f"your personal database '{personal}'"
    return NEW_PERSONAL_DATABASE


def _upload_notice(url: str, database: str, counts: dict, new_account: bool,
                   personal: str = NEW_PERSONAL_DATABASE) -> str:
    kept = ", ".join(f"{n} {t}" for t, n in counts.items()) or "nothing yet"
    return "\n".join([
        f"Signing in {'creates a free Infino account on' if new_account else 'connects this profile to'} {url}",
        f"and uploads to the database '{database}':",
        f"  - everything kept on this machine so far ({kept}), except personal memory and the",
        "    conversations and traces of personal profiles and scheduled jobs, which go only to",
        f"    {personal};",
        "  - and from then on, every memory, conversation (tool calls and their results included),",
        "    trace and published skill this profile writes.",
        "It stays on this machine too.",
    ])


def _new_personal_name(team: str) -> str:
    """A new personal database's name: the team's, then a random part, within
    the platform's length."""
    token = secrets.token_hex(PERSONAL_RANDOM_BYTES)
    return team[:MAX_DATABASE_NAME - len(PERSONAL_PART) - len(token)] + PERSONAL_PART + token


def _personal_database(client: InfinoClient, personal: Optional[str], team: str) -> Optional[str]:
    """The personal database this sign-in uses, ready in the cloud: the one
    named (`personal`, made if missing), else ("") a new one, never one that
    exists already (it could be anyone's). None with none wanted (`personal`
    None), or when it cannot be made, for any reason (refused, unreachable,
    timed out): what is personal then stays on this machine, and the sign-in,
    saved already, goes on, rather than naming a personal database whose rows
    would never go."""
    if personal is None:
        return None
    name = personal or _new_personal_name(team)
    try:
        if personal:
            client.ensure_database(name)
        else:
            client.create_database(name)
    except Exception as e:
        print(f"No personal database: '{name}' could not be made ({e}). Personal memory, and the "
              f"conversations and traces of personal profiles and scheduled jobs, stay on this machine. "
              f"{ADD_PERSONAL_DATABASE}", file=sys.stderr)
        return None
    return name


def _overridden(outside: dict, signing_in: dict, server: str) -> str:
    """Why the sign-in would not be the settings in effect: the first of
    `signing_in` ({setting: what the sign-in would hold}) that `outside`
    (config.overrides: set in the config block or the environment, which takes
    precedence over a sign-in) holds otherwise; "" when there is none. A
    personal database is None when none is wanted, "" for a new one, and a key
    "" when a new account is to make it. The refusal names the variable and
    where it is set: that is where it is changed."""
    for name, (value, where, variable) in outside.items():
        wanted = signing_in[name]
        if wanted == value or (name == config.URL_ENV and wanted == value.rstrip("/")):
            continue
        place = (f"the `env` block of mcp_servers.{server} in the config" if where == config.FROM_CONFIG
                 else "this profile's environment (its .env)")
        if name == config.KEY_ENV:  # never printed: it is a secret
            return (f"{variable} is set in {place}, and a key set there is used instead of a sign-in's, so the key "
                    f"this sign-in would save would never be used. Remove {variable} there, or sign in with that "
                    "key.")
        instead = "none" if wanted is None else "a new one" if not wanted else f"'{wanted}'"
        return (f"{variable} is '{value}' in {place}, and a value set there is used instead of a sign-in's, so "
                f"{SETTING_NAMES[name]} would be '{value}', not {instead}. Change or remove {variable} there, or sign "
                "in with that value.")
    return ""


def _local_counts() -> dict:
    """Rows kept on this machine that a sign-in uploads to the team (what is
    personal goes only to a personal database, so it is not counted here)."""
    from .local import LocalStore
    from . import tables

    store = LocalStore.under(config.state_dir())
    try:
        names = set(store.list_tables())
        return {t: store.query_sql(f"SELECT count(*) AS n FROM {t}")[0]["n"] for t in tables.ALL if t in names}
    except Exception:
        return {}


def _login(args) -> int:
    current = config.load()
    if current.signed_in:
        print(f"Already signed in: {current.url} / {current.team_database}, from the {current.source}. "
              "`hermes infino status` shows how far the cloud is.")
        if current.api_key == config.sign_in_key():
            # The file may hold the only copy of a free account's key. Signing
            # in again names everything this sign-in has, its personal
            # database (or none) too: another would split personal memory in two.
            path = config.credentials_path()
            personal = (f"--personal-database {current.personal_database}" if current.personal_database
                        else "--no-personal-database")
            print(f"To sign in again, keep the key: it is the `api_key` in {path}, and the platform gives a free "
                  f"account's key only once. Save it to a file, remove {path}, and run `hermes infino login "
                  f"--url {current.url} --api-key-file <that file> --team-database {current.team_database} "
                  f"{personal}`.")
        print(f"The `env` block of mcp_servers.{current.mcp_server} in the config, and the environment, take "
              "precedence over a sign-in: change a setting there to change it.")
        if not current.personal_database:
            print(ADD_PERSONAL_DATABASE)
        return 0
    new_account = not args.api_key_file
    api_key = ""  # a new account's comes from the platform
    if not new_account:
        try:
            api_key = Path(args.api_key_file).expanduser().read_text().strip()
        except OSError as e:
            print(f"infino: cannot read the key file: {e}", file=sys.stderr)
            return 1
    url = (args.url or current.url or DEFAULT_URL).rstrip("/")
    if not url.lower().startswith(URL_SCHEMES):
        print(f"infino: the platform URL '{url}' does not start with https:// (or http://). Nothing was created or "
              "sent.", file=sys.stderr)
        return 1
    # The team and personal database the profile already names, so the
    # notice, the account and the rows all name the same ones.
    database = args.team_database or current.team_database or DEFAULT_DATABASE
    outside = config.overrides(team_database=database)
    named = outside.get(config.PERSONAL_DB_ENV, ("",))[0]
    personal = None if args.no_personal_database else (args.personal_database or named or "")
    overridden = _overridden(outside, {config.KEY_ENV: api_key, config.URL_ENV: url, config.TEAM_DB_ENV: database,
                                       config.PERSONAL_DB_ENV: personal}, current.mcp_server)
    if overridden:
        print(f"infino: {overridden} Nothing was created or sent.", file=sys.stderr)
        return 1
    if personal and personal == database:  # from --personal-database: a set one is never the team's (overrides)
        print(f"infino: '{database}' is the team database, so --personal-database cannot name it: what is personal "
              "would go into the team's tables. Name another, or leave it out and one is made. Nothing was created "
              "or sent.", file=sys.stderr)
        return 1
    counts = _local_counts()
    print(_upload_notice(url, database, counts, new_account, _personal_destination(personal)))
    if new_account or not args.yes:
        # Creating an account is a question only the person at the terminal answers.
        if not sys.stdin.isatty():
            print("There is no terminal to ask on, so nothing was created or sent. Run `hermes infino login` "
                  "from a terminal.", file=sys.stderr)
            return 1
        if not _ask("Sign in and upload? [y/N] "):
            print("Nothing was created or sent. Everything stays on this machine.")
            return 0

    if new_account:
        try:
            trial = request_trial(url, database)
        except InfinoError as e:
            if e.status not in (NO_FREE_ACCOUNTS, FREE_ACCOUNT_TAKEN):
                raise
            why = ("gives out no free accounts" if e.status == NO_FREE_ACCOUNTS
                   else "already gave this network its free account")
            print(f"{url} {why}, so no account was created and nothing was sent. Sign in with a key someone "
                  f"gave you: hermes infino login --api-key-file KEY --team-database TEAM", file=sys.stderr)
            return 1
        api_key, database = str(trial["api_key"]), str(trial.get("database") or database)
    else:
        client = InfinoClient(api_key, url)
        client.list_databases()  # the key works before anything is stored
        client.ensure_database(database)

    # Saved before anything else can fail: the platform gives a new account's
    # key only once, and one free account per network.
    creds = {"url": url, "api_key": api_key, "team_database": database,
             "consented_at_ms": int(time.time() * 1000)}
    path = config.write_credentials(creds)
    if new_account:
        credit = trial.get("credit_cents")
        print(f"Created a free account" + (f" with ${int(credit) / 100:.2f} of credit" if credit else "") + ".")
        if trial.get("console_url"):
            print(f"   manage it at {trial['console_url']}")
    print(f"Signed in; the key is in {path} (owner-only).")
    made = _personal_database(InfinoClient(api_key, url), personal, database)
    if made:
        creds["personal_database"] = made
        config.write_credentials(creds)
        print(f"Your personal database is '{made}'. To use it from another machine too, sign in there "
              f"with --personal-database {made}.")
    status = _catch_up(config.load())
    # The launcher picks the database when Hermes starts the tool server, and a
    # running Hermes (its gateway) keeps the one it started.
    print("A running Hermes moves recall and writes to the cloud within half a minute. Its Infino tools read "
          "the cloud once Hermes restarts; until then they read this machine's copy, which has everything too.")
    return status


# -- mcp ----------------------------------------------------------------------

def _mcp(args) -> int:
    """Become the Infino client, in Hermes's own environment (mcp_launch.py)."""
    from . import local, mcp_launch

    mcp_launch.launch(config, local)
    return 0  # not reached: launch execs the client


# -- sync ---------------------------------------------------------------------

def _sync(args) -> int:
    settings = config.load()
    if not settings.signed_in:
        print("Not signed in: everything stays on this machine. `hermes infino login` signs in.")
        return 0
    return _catch_up(settings)


def _catch_up(settings: config.Settings) -> int:
    """Create the cloud tables, then push until nothing is left to push. A
    database whose tables cannot be made right (a personal database the key
    may not reach, say) holds up only its own rows: the other's are pushed,
    and the end says what was not, and why. Personal memory an older version
    kept in the team's local table moves to the person's own first
    (Backend.ensure_local), so none of it is pushed to the team."""
    from .backend import Backend
    from .mirror import SETTLE_MS

    b = Backend(settings)
    b.ensure_local()
    b.local.drop_vectors()
    b.personal.drop_vectors()
    try:
        b.mirror.ensure_cloud_tables()
    except Exception:
        if len(b.mirror.waiting()) == len(b.mirror.databases()):
            raise  # nothing can go anywhere: the platform's answer says why
        # The others go on; each database that failed is reported at the end.

    pushed_now: dict[str, int] = {}

    def show(mark, n):
        pushed_now[mark] = n
        print(f"\r   {mark:<14} {n} rows pushed", end="", flush=True)

    deadline = time.monotonic() + CATCH_UP_S
    total = 0
    last_pass = False
    while True:
        pushed_now.clear()
        try:
            b.mirror.sync(wait=True, progress=show)
        except Exception as e:
            # A database out of reach was skipped, and the others' rows went
            # (sync raises a failed push before it); anything else stops here.
            if not any(e is why for why in b.mirror.waiting().values()):
                print(f"\nThe cloud stopped accepting rows: {e}\nThey are kept here; `hermes infino sync` resumes.",
                      file=sys.stderr)
                return 1
        pushed = sum(pushed_now.values())
        total += pushed
        if last_pass:
            break
        # Rows from the last few seconds are left for the next pass; one more,
        # once they have settled, takes them.
        last_pass = not pushed or time.monotonic() > deadline
        time.sleep(SETTLE_MS / 1000)
    # What this sync did not send: the rows of a database out of reach; what
    # is personal, without a personal database; and the rows the cloud
    # refused outright.
    kept_here = []
    out_of_reach = b.mirror.waiting()
    for database, personal in b.mirror.databases():
        if database in out_of_reach:
            whose = "what is personal" if personal else "the team's rows"
            kept_here.append(f"{whose}: '{database}' could not be made ready ({out_of_reach[database]}); "
                             "`hermes infino sync` tries again")
    if not settings.personal_database:
        kept_here.append("what is personal: there is no personal database")
    refused = _refused(b)
    if refused:
        kept_here.append(refused)
    if not kept_here:
        print(f"\nThe cloud has everything kept here ({total} rows pushed).")
        return 0
    print(f"\nThe cloud has everything kept here except what stays on this machine ({total} rows pushed):")
    for what in kept_here:
        print(f"   {what}")
    return 1 if out_of_reach else 0

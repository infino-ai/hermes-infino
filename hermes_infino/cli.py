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

Hermes imports this module to build its command line for every command, so
the engine and the model are imported only inside the commands that use them.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
from pathlib import Path

from . import config
from .client import DEFAULT_URL, InfinoClient, InfinoError, request_trial

# The team database a new account registers when none is named.
DEFAULT_DATABASE = "hermes-team"
# Seconds `login` and `sync` keep catching up while rows are still arriving.
CATCH_UP_S = 60.0


def register_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="infino_action")
    sub.add_parser("status", help="local or signed in, and how far the cloud is")
    login = sub.add_parser("login", help="sign in to Infino and push this machine's memory, conversations, traces and skills")
    login.add_argument("--url", default="", help=f"platform URL (default {DEFAULT_URL})")
    login.add_argument("--api-key-file", default="", help="use this key (a team's) instead of creating an account")
    login.add_argument("--team-database", default="", help=f"the team's database (default {DEFAULT_DATABASE})")
    login.add_argument("--personal-database", default="", help="your personal database (optional)")
    login.add_argument("--yes", action="store_true", help="agree to the upload when using --api-key-file")
    sub.add_parser("sync", help="push what the cloud does not have yet")
    sub.add_parser("mcp", help="the Infino tools' MCP server, which Hermes starts (mcp_servers.infino)")


def infino_command(args: argparse.Namespace) -> int:
    action = getattr(args, "infino_action", None) or "status"
    try:
        return {"status": _status, "login": _login, "sync": _sync, "mcp": _mcp}[action](args)
    except InfinoError as e:
        print(f"infino: the platform refused: {e}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"infino: could not reach the platform: {e.reason}", file=sys.stderr)
    except TimeoutError:
        print("infino: the platform did not answer in time", file=sys.stderr)
    return 1


# -- status -------------------------------------------------------------------

def _status(args) -> int:
    from .backend import Backend
    from .mirror import REFUSED_FILE
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
        print(f"   memory vectors {'on (local model)' if b.local.has_vectors() else 'off (the cloud embeds)'}")
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
    if b.mirror.refused():
        print(f"refused          {b.mirror.refused()} rows the cloud would not take; they stay here "
              f"(listed in {config.state_dir(settings.home) / REFUSED_FILE})")
    return 0


# -- login --------------------------------------------------------------------

def _ask(question: str) -> bool:
    try:
        return input(question).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _upload_notice(url: str, database: str, counts: dict, new_account: bool) -> str:
    kept = ", ".join(f"{n} {t}" for t, n in counts.items()) or "nothing yet"
    return "\n".join([
        f"Signing in {'creates a free Infino account on' if new_account else 'connects this profile to'} {url}",
        f"and uploads to the database '{database}':",
        f"  - everything kept on this machine so far ({kept}), except personal memory and the",
        "    conversations and traces of personal profiles and scheduled jobs, which go only to a",
        "    personal database (--personal-database) and otherwise stay here;",
        "  - and from then on, every memory, conversation (tool calls and their results included),",
        "    trace and published skill this profile writes.",
        "It stays on this machine too.",
    ])


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
        print(f"Already signed in: {current.url} / {current.team_database}. `hermes infino status` shows how far the cloud is.")
        return 0
    url = (args.url or current.url or DEFAULT_URL).rstrip("/")
    database = args.team_database or DEFAULT_DATABASE
    new_account = not args.api_key_file
    counts = _local_counts()
    print(_upload_notice(url, database, counts, new_account))
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
        trial = request_trial(url, database)
        api_key, database = str(trial["api_key"]), str(trial.get("database") or database)
        credit = trial.get("credit_cents")
        print(f"Created a free account" + (f" with ${int(credit) / 100:.2f} of credit" if credit else "") + ".")
        if trial.get("console_url"):
            print(f"   manage it at {trial['console_url']}")
    else:
        try:
            api_key = Path(args.api_key_file).expanduser().read_text().strip()
        except OSError as e:
            print(f"infino: cannot read the key file: {e}", file=sys.stderr)
            return 1
        client = InfinoClient(api_key, url)
        client.list_databases()  # the key works before anything is stored
        client.ensure_database(database)

    creds = {"url": url, "api_key": api_key, "team_database": database,
             "consented_at_ms": int(time.time() * 1000)}
    if args.personal_database:
        creds["personal_database"] = args.personal_database
    path = config.write_credentials(creds)
    print(f"Signed in; the key is in {path} (owner-only).")
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
    """Create the cloud tables, then push until nothing is left to push."""
    from . import tables
    from .backend import Backend
    from .mirror import SETTLE_MS

    b = Backend(settings)
    b.local.ensure_tables(with_vectors=False)
    b.personal.ensure_tables(with_vectors=False, only=tables.PERSONAL)
    b.local.drop_vectors()
    b.personal.drop_vectors()
    b.mirror.ensure_cloud_tables()

    def show(table, n):
        print(f"\r   {table:<14} {n} rows pushed", end="", flush=True)

    deadline = time.monotonic() + CATCH_UP_S
    total = 0
    last_pass = False
    while True:
        try:
            pushed = sum(b.mirror.sync(wait=True, progress=show).values())
        except Exception as e:
            print(f"\nThe cloud stopped accepting rows: {e}\nThey are kept here; `hermes infino sync` resumes.",
                  file=sys.stderr)
            return 1
        total += pushed
        if last_pass:
            break
        # Rows from the last few seconds are left for the next pass; one more,
        # once they have settled, takes them.
        last_pass = not pushed or time.monotonic() > deadline
        time.sleep(SETTLE_MS / 1000)
    print(f"\nThe cloud has everything kept here ({total} rows pushed).")
    return 0

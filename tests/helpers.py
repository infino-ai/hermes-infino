"""What the test files share: names, a stand-in embedding model, and ways to
drive the provider and read the local databases."""

from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import hermes_cli

from agent.memory_manager import MemoryManager

from hermes_infino import chats, config, embed, traces
from hermes_infino.local import PERSONAL_DB_DIR, LocalStore
from hermes_infino.provider import InfinoMemoryProvider

TEAM_DB = "team-test"
PERSONAL_DB = "person-test"
# Hermes's own stand-in for the Remote Config service, which its tests ship.
REMOTE_CONFIG_STUB = Path(hermes_cli.__file__).resolve().parent.parent / "tests" / "plugins" / "config_backends" / "stub_plane.py"
# Seconds a test waits for a background write to land.
WRITE_WAIT_S = 5.0
# How long a scheduled sync waits in these tests (the settle window is off).
TEST_SYNC_DELAY_S = 0.05

# A gateway turn's session values: a group chat, another group, and a one-to-one chat.
GROUP = {"HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "-100", "HERMES_SESSION_CHAT_TYPE": "group",
         "HERMES_SESSION_USER_ID": "222", "HERMES_SESSION_USER_NAME": "mallory"}
OTHER_GROUP = {**GROUP, "HERMES_SESSION_CHAT_ID": "-200"}
DM = {**GROUP, "HERMES_SESSION_CHAT_ID": "111", "HERMES_SESSION_CHAT_TYPE": "dm", "HERMES_SESSION_USER_NAME": "owner"}

SKILL_MD = "---\nname: release-checklist\ndescription: Steps before a release.\n---\n\n1. Freeze.\n"


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for the model: words hashed into the 384 dimensions."""
    out = []
    for t in texts:
        v = [0.0] * embed.DIM
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % embed.DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n if any(v) else 1.0 / math.sqrt(embed.DIM) for x in v])
    return out


def make_manager(tmp_path, **kwargs) -> tuple[MemoryManager, InfinoMemoryProvider]:
    provider = InfinoMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)
    kwargs.setdefault("platform", "cli")
    mgr.initialize_all(session_id=kwargs.pop("session_id", "s1"), hermes_home=str(tmp_path), **kwargs)
    provider._ready.wait(5)
    return mgr, provider


def wait_for(predicate, timeout: float = WRITE_WAIT_S):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.05)
    return predicate()


def hook_ctx():
    hooks = {}

    class Ctx:
        def register_hook(self, name, fn):
            hooks[name] = fn

    traces.register_hooks(Ctx())
    return hooks


def local(tmp_path) -> LocalStore:
    return LocalStore.under(config.state_dir(tmp_path))


def local_rows(tmp_path, table: str) -> list[dict]:
    return local(tmp_path).query_sql(f"SELECT * FROM {table} ORDER BY stored_ms, row_id")


def personal_rows(tmp_path) -> list[dict]:
    return LocalStore.under(config.state_dir(tmp_path), PERSONAL_DB_DIR).query_sql("SELECT * FROM memory")


def in_session(monkeypatch, values: dict, binding="absent"):
    """The turn's gateway session values; `binding` stands in for Hermes's
    channel memory binding ("absent": a Hermes without channel memory)."""
    monkeypatch.setattr(chats, "_get_session_env", lambda name, default="": values.get(name, default))
    if binding == "absent":
        monkeypatch.setattr(chats, "_CHANNEL_MEMORY", None)
    else:
        monkeypatch.setattr(chats, "_CHANNEL_MEMORY", SimpleNamespace(get=lambda default=None: binding))


def sign_in(tmp_path, platform, team=TEAM_DB, **extra) -> config.Settings:
    """Sign the profile at `tmp_path` in, as `hermes infino login` leaves it."""
    config.write_credentials({"url": platform.url, "api_key": "k", "team_database": team, **extra}, tmp_path)
    return config.load(tmp_path)


def requests_to(platform, route: str) -> int:
    """How many requests the platform got on a route (`append`, `hybrid_search`...)."""
    return sum(1 for path, _ in platform.requests if path.startswith(f"/v1/{route}"))

"""What the test files share: names, a stand-in embedding model, and ways to
drive the provider and read the local databases."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import hermes_cli

from agent.agent_runtime_helpers import invoke_tool
from agent.background_review import build_memory_write_metadata
from agent.memory_manager import MemoryManager
from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from gateway.session_context import reset_session_vars, set_session_vars
from hermes_cli import plugins as hermes_plugins
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent
from tools.memory_tool_store import MemoryStore
from tools.thread_context import propagate_context_to_thread

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
# The task and call ids of a tool call these tests make through Hermes's dispatch.
TOOL_TASK = "task-1"
TOOL_CALL = "call-1"

# A gateway turn's session values: a group chat, another group, and a one-to-one chat.
GROUP = {"HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "-100", "HERMES_SESSION_CHAT_TYPE": "group",
         "HERMES_SESSION_USER_ID": "222", "HERMES_SESSION_USER_NAME": "mallory"}
OTHER_GROUP = {**GROUP, "HERMES_SESSION_CHAT_ID": "-200"}
DM = {**GROUP, "HERMES_SESSION_CHAT_ID": "111", "HERMES_SESSION_CHAT_TYPE": "dm", "HERMES_SESSION_USER_NAME": "owner"}

SKILL_MD = "---\nname: release-checklist\ndescription: Steps before a release.\n---\n\n1. Freeze.\n"


@contextmanager
def webhook_delivery(route: str, delivery_id: str) -> Iterator[None]:
    """The turn of one webhook delivery on `route`: its session bound from the
    source Hermes's webhook adapter builds for it (`_admission_source`), the way
    the gateway binds a turn (`set_session_vars`), and read back through Hermes's
    own `get_session_env`."""
    source = WebhookAdapter(PlatformConfig(enabled=True, extra={}))._admission_source(route, None, delivery_id)
    set_session_vars(platform=source.platform.value, chat_id=source.chat_id or "", chat_type=source.chat_type or "",
                     chat_name=source.chat_name or "", user_id=source.user_id or "", user_name=source.user_name or "")
    try:
        yield
    finally:
        reset_session_vars()


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
    settle(provider)
    return mgr, provider


def settle(provider: InfinoMemoryProvider) -> None:
    """Wait for the provider's start: the local tables, the catch-up, and the sync it began."""
    provider._ready.wait(WRITE_WAIT_S)
    provider._caught_up.wait(WRITE_WAIT_S)
    wait_for(lambda: provider._backend._timer is None or not provider._backend._timer.is_alive())


def wait_for(predicate, timeout: float = WRITE_WAIT_S):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.05)
    return predicate()


def plugin_manager() -> PluginManager:
    """Hermes's own plugin manager, with the plugin's trace hooks registered
    as Hermes registers them."""
    manager = PluginManager()

    class Ctx:
        def register_hook(self, name, fn):
            manager._hooks.setdefault(name, []).append(fn)

    traces.register_hooks(Ctx())
    return manager


def hook_ctx():
    """The plugin's trace hooks, each called the way Hermes calls a plugin's
    hooks: through Hermes's own plugin manager, which runs most of them on a
    thread of their own with a copy of the caller's context."""
    manager = plugin_manager()

    def caller(name):
        return lambda **kw: manager.invoke_hook(name, **kw)

    return {name: caller(name) for name in manager._hooks}


def hermes_dispatch(monkeypatch) -> None:
    """Hermes's own tool dispatch reaching the plugin's hooks: the profile's
    plugin manager is one with the plugin loaded (and nothing else to load),
    so a tool call runs its pre_tool_call hooks, the tool, then its
    post_tool_call hooks, as in a running Hermes."""
    manager = plugin_manager()
    manager._discovered = True
    monkeypatch.setattr(hermes_plugins, "_plugin_managers_by_home", {})
    monkeypatch.setattr(hermes_plugins, "_plugin_manager", manager)


def agent_tool_call(session_id: str, name: str, args: dict, manager: Optional[MemoryManager] = None,
                    platform: str = "cli") -> dict:
    """A tool call an agent of `session_id` makes, through Hermes's own
    dispatch (`invoke_tool`: the plugins' pre_tool_call hooks, the tool, their
    post_tool_call hooks; hermes_dispatch first), on a thread of its own
    carrying this one's context. Without `manager`, as Hermes's background
    review calls its memory tool: the review's fork shares the id of the
    session whose turn it follows, and has no memory manager. With it, as the
    agent itself does, which tells the session's memory providers of a write."""
    store = MemoryStore()
    store.load_from_disk()
    agent = SimpleNamespace(session_id=session_id, _parent_session_id="", platform=platform, _memory_store=store,
                            _memory_manager=manager, valid_tool_names=set())
    agent._build_memory_write_metadata = partial(build_memory_write_metadata, agent)
    answer = {}

    def call():
        answer["result"] = invoke_tool(agent, name, args, TOOL_TASK, tool_call_id=TOOL_CALL)

    worker = threading.Thread(target=propagate_context_to_thread(call))
    worker.start()
    worker.join()
    return json.loads(answer["result"])


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


def hermes_agent(mgr: MemoryManager, session_id: str = "s1") -> AIAgent:
    """Hermes's agent holding `mgr`, as a CLI session, a gateway chat, a
    scheduled job's run or a one-shot run holds its memory manager, for
    Hermes's own teardown functions to end: its memory methods
    (shutdown_memory_provider, commit_memory_session, release_clients, close)
    are Hermes's; its model clients and tools are not built."""
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = mgr
    agent.session_id = session_id
    agent._session_messages = []
    return agent


def end_session(provider) -> None:
    """The provider's part of a session's end, as Hermes runs it
    (AIAgent.shutdown_memory_provider): on_session_end, then shutdown, which
    then waits for the session's rows to reach the cloud. A shutdown alone is
    a soft release, which does not."""
    provider.on_session_end([])
    provider.shutdown()


def sign_in(tmp_path, platform, team=TEAM_DB, **extra) -> config.Settings:
    """Sign the profile at `tmp_path` in, as `hermes infino login` leaves it."""
    config.write_credentials({"url": platform.url, "api_key": "k", "team_database": team, **extra}, tmp_path)
    return config.load(tmp_path)


def write_tools_setting(home_dir: Path, line: str) -> None:
    """A config whose tools block holds one setting, written as a person
    writes YAML (a bare `off` is a boolean to Hermes's loader)."""
    (home_dir / "config.yaml").write_text(f"mcp_servers:\n  infino:\n    command: hermes\n    env:\n      {line}\n")


def requests_to(platform, route: str) -> int:
    """How many requests the platform got on a route (`append`, `hybrid_search`...)."""
    return sum(1 for path, _ in platform.requests if path.startswith(f"/v1/{route}"))

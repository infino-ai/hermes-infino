"""Fixtures every test file uses: a quick settle window and stand-in model,
a fake platform, a profile home, and Hermes on Remote Config."""

from __future__ import annotations

import importlib.util

import pytest

from plugins.config_backends import remote as remote_config
from plugins.config_backends.remote import backend as remote_backend, credentials as remote_credentials

from hermes_infino import backend, embed, mirror, privacy, skills, traces

from .fake_platform import FakePlatform
from .helpers import REMOTE_CONFIG_STUB, TEAM_DB, TEST_SYNC_DELAY_S, fake_embed

# Every setting the plugin reads from the environment.
INFINO_ENV = ("INFINO_API_KEY", "INFINO_TEAM_DATABASE", "INFINO_URL", "INFINO_PERSONAL_DATABASE",
              "INFINO_AUTHOR", "INFINO_MCP_SERVER", "INFINO_NODE", "INFINO_CLIENT",
              "INFINO_PERSONAL_PROFILE", "INFINO_PERSONAL_JOBS", "INFINO_OWNER_ROUTES")


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    embed.use(fake_embed)
    monkeypatch.setattr(mirror, "SETTLE_MS", 0)
    monkeypatch.setattr(backend, "SETTLE_MS", 0)
    monkeypatch.setattr(backend, "SYNC_DELAY_S", TEST_SYNC_DELAY_S)
    backend._backends.clear()
    skills._loaded_at.clear()
    privacy._sessions.clear()  # what one test noted about its sessions is not another's
    traces._started.clear()
    traces._buffers.clear()
    yield
    embed.use(None)
    backend._backends.clear()
    privacy._sessions.clear()
    traces._started.clear()
    traces._buffers.clear()


@pytest.fixture
def platform():
    fake = FakePlatform()
    yield fake
    fake.close()


@pytest.fixture
def home(monkeypatch, tmp_path):
    # Hermes refuses memory calls from a profile other than the one the provider was
    # initialized for, so the test's caller profile is the temporary home too.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in INFINO_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def env(home, platform):
    """A profile signed in through its environment, as a deployment configures one."""
    home.setenv("INFINO_API_KEY", "test-key")
    home.setenv("INFINO_URL", platform.url)
    home.setenv("INFINO_TEAM_DATABASE", TEAM_DB)
    return home


@pytest.fixture
def remote(home, tmp_path):
    """Hermes on Remote Config, served by Hermes's own stand-in service."""
    spec = importlib.util.spec_from_file_location("hermes_remote_config_stub", REMOTE_CONFIG_STUB)
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    with stub.StubPlane() as plane:
        for k, v in stub.remote_env(plane).items():
            home.setenv(k, v)
        home.setattr(remote_backend, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_config._reset_for_tests()
        remote_credentials._IDP_CACHE.clear()
        plane.home = tmp_path
        yield plane
        remote_config._reset_for_tests()
        remote_credentials._IDP_CACHE.clear()

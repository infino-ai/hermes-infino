"""The team's shared skills: published to the team's `skills` table, and
loaded into every profile on the team as read-only `infino:<skill>` skills.

Hermes registers a plugin's skill from a SKILL.md on disk, at plugin load
(`ctx.register_skill(name, path, description)`). So at load the newest
version of each skill in the table is written under the profile's home
(`cache/infino/team-skills/<team>/<name>/SKILL.md`, a folder per platform and
team database) and registered from there; a skill gone from the table loses
its copy. A skill published later loads in the next session. Hermes loads a
plugin's skill only under a name of letters, digits, `-` and `_`, so no other
name is published.

Every Hermes start waits on this, so the platform gets LOAD_TIMEOUT_S; when
it does not answer, the copies written by the last load of the same team are
registered.

A team skill is text a teammate's agent wrote, loaded into every profile: its
description says who published it, so the model and the person see where it
came from, and INFINO_TEAM_SKILLS=off keeps a profile to its own skills.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from . import backend, config, tables

try:  # Hermes's rule for the folders its skill scanners skip (archive, hub, backups)
    from agent.skill_utils import is_excluded_skill_path as _is_excluded_skill_path
except ImportError:  # pragma: no cover - only outside a Hermes install
    _is_excluded_skill_path = None

logger = logging.getLogger(__name__)

# Where the team's skills are written under the profile's home.
CACHE_DIR = Path("cache") / "infino" / "team-skills"
# Hex characters of the hash naming each team's folder under CACHE_DIR.
TEAM_KEY_CHARS = 16
# Beside each cached SKILL.md: its description and who published it.
ABOUT_FILE = "infino.json"
# Hermes's skill names: what `register_skill` accepts.
SKILL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# Seconds the load waits on the platform before the session starts with the
# copies the last load wrote.
LOAD_TIMEOUT_S = 2.0
# The most skill rows read at load.
MAX_SKILL_ROWS = 2_000
# Seconds a load's copies stand for the next one in the same process: a
# gateway loads the plugin for every agent it builds, and the team's skills
# do not change that fast.
REFRESH_S = 300.0
# The largest SKILL.md that may be published (characters).
MAX_SKILL_CHARS = 200_000

# When each profile home last read the team's skills, in this process.
_loaded_at: dict[str, float] = {}


def latest(rows: list[dict]) -> dict[str, dict]:
    """The newest row of each valid skill name."""
    out: dict[str, dict] = {}
    for r in rows:
        name = str(r.get("name") or "")
        if not SKILL_NAME.match(name) or not r.get("content"):
            continue
        if name not in out or int(r.get("created_ms") or 0) > int(out[name].get("created_ms") or 0):
            out[name] = r
    return out


def described(description: str, author: str) -> str:
    """A team skill's description, saying who published it."""
    by = f"(team skill published by {author})" if author else "(team skill)"
    return f"{description} {by}".strip()


def cache_root(settings: config.Settings, home: Optional[Path] = None) -> Path:
    """Where the copies of the team's skills are written: a folder for each
    platform and team database, so a profile that moves to another team never
    loads the last one's skills from their copies."""
    team = f"{settings.url.rstrip('/')}/{settings.team_database}"
    return (home or settings.home) / CACHE_DIR / hashlib.sha256(team.encode()).hexdigest()[:TEAM_KEY_CHARS]


def register_team_skills(ctx, settings: Optional[config.Settings] = None, home: Optional[Path] = None) -> list[str]:
    """Write the team's skills under the profile's home and register each; the
    names registered. Never raises: a platform that does not answer in time
    starts the session with the last load's copies. Without an account there
    is no team: the only published skills are this profile's own, which
    Hermes already loads."""
    settings = settings or config.load(home)
    if not settings.signed_in or not settings.team_skills:
        return []
    root = cache_root(settings, home)
    key = str(root.resolve())
    if time.monotonic() - _loaded_at.get(key, -REFRESH_S) < REFRESH_S:
        return _register_cached(ctx, root)
    try:
        rows = backend.for_settings(settings).query(
            f"SELECT name, description, content, author, created_ms FROM {tables.SKILLS} "
            f"ORDER BY created_ms DESC LIMIT {MAX_SKILL_ROWS}", timeout_s=LOAD_TIMEOUT_S)
    except Exception as e:  # no answer in time, or no skills table yet
        logger.info("infino: team skills read from the last load's copies: %s", e)
        return _register_cached(ctx, root)
    _loaded_at[key] = time.monotonic()
    newest = latest(rows)
    # A skill the table no longer has must not load from its copy next time.
    for copy in root.iterdir() if root.is_dir() else []:
        if copy.name not in newest:
            shutil.rmtree(copy, ignore_errors=True)
    registered = []
    for name, row in sorted(newest.items()):
        try:
            path = root / name / "SKILL.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(row["content"]))
            about = {"description": str(row.get("description") or ""), "author": str(row.get("author") or "")}
            (path.parent / ABOUT_FILE).write_text(json.dumps(about))
            ctx.register_skill(name, path, described(about["description"], about["author"]))
            registered.append(name)
        except Exception as e:
            logger.info("infino: team skill %s not registered: %s", name, e)
    return registered


def _register_cached(ctx, root: Path) -> list[str]:
    """Register the copies a previous load wrote; the names registered."""
    registered = []
    for path in sorted(root.glob("*/SKILL.md")) if root.is_dir() else []:
        name = path.parent.name
        if not SKILL_NAME.match(name):
            continue
        try:
            about = json.loads((path.parent / ABOUT_FILE).read_text())
        except (OSError, ValueError):
            about = {"description": description_of(path.read_text()), "author": ""}
        try:
            ctx.register_skill(name, path, described(str(about.get("description") or ""), str(about.get("author") or "")))
            registered.append(name)
        except Exception as e:
            logger.info("infino: team skill %s not registered: %s", name, e)
    return registered


def _frontmatter_field(skill_md: str, field: str) -> str:
    """The `field:` of a SKILL.md's frontmatter, or empty."""
    m = re.search(r"^---\s*\n(.*?)\n---", skill_md, re.S)
    if not m:
        return ""
    d = re.search(rf"^{field}:\s*(.+)$", m.group(1), re.M)
    return d.group(1).strip().strip("\"'") if d else ""


def description_of(skill_md: str) -> str:
    """The `description:` of a SKILL.md's frontmatter, or empty."""
    return _frontmatter_field(skill_md, "description")


def skill_row(name: str, skill_md: str, author: str, profile: str) -> dict:
    """The row that publishes a skill to the team. A name Hermes would not
    load a plugin's skill under (its own skill tool takes dots too) raises
    ValueError: published, it would load in no profile."""
    if not SKILL_NAME.match(name):
        raise ValueError(f"'{name}' cannot be a team skill's name: Hermes loads a plugin's skill only under a "
                         "name of letters, digits, '-' and '_'. Rename the skill to publish it.")
    return {
        "name": name,
        "description": description_of(skill_md),
        "content": skill_md,
        "author": author,
        "profile": profile,
        "created_ms": int(time.time() * 1000),
    }


def find_local_skill(name: str, home: Optional[Path] = None) -> Optional[Path]:
    """The profile's own SKILL.md for `name`, wherever under `skills/` it sits:
    the skill skills_list shows by that name (its frontmatter's `name`, else
    its folder's), else the one in a folder of that name. A SKILL.md in a
    folder Hermes's own scanners skip (the curator's `.archive`, a hub's
    cache, backups) is not the skill, and is never found."""
    root = (home or config.home()) / "skills"
    found = [p for p in sorted(root.rglob("SKILL.md")) if not _skipped(p, root)] if root.is_dir() else []
    for path in found:
        if (_frontmatter_field(_read_skill(path), "name") or path.parent.name) == name:
            return path
    return next((path for path in found if path.parent.name == name), None)


def _skipped(path: Path, root: Path) -> bool:
    """Whether Hermes's skill scanners skip `path`, a SKILL.md under `root`."""
    return _is_excluded_skill_path is not None and _is_excluded_skill_path(path.relative_to(root), root=root)


def _read_skill(path: Path) -> str:
    """A SKILL.md's text; empty when it cannot be read (it is then known by its folder)."""
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return ""

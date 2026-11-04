"""The team's shared skills: published to the team's `skills` table, and
loaded into every profile on the team as read-only `infino:<skill>` skills.

Hermes registers a plugin's skill from a SKILL.md on disk, at plugin load
(`ctx.register_skill(name, path, description)`). So at load the newest
version of each skill in the table is written under the profile's home
(`cache/infino/team-skills/<name>/SKILL.md`) and registered from there.
A skill published later loads in the next session.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from . import backend, config, tables

logger = logging.getLogger(__name__)

# Where the team's skills are written under the profile's home.
CACHE_DIR = Path("cache") / "infino" / "team-skills"
# Hermes's skill names: what `register_skill` accepts.
SKILL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# Seconds the load waits on the platform before the session starts without
# the team's skills.
LOAD_TIMEOUT_S = 5.0
# The most skill rows read at load.
MAX_SKILL_ROWS = 2_000


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


def register_team_skills(ctx, settings: Optional[config.Settings] = None, home: Optional[Path] = None) -> list[str]:
    """Write the team's skills under the profile's home and register each; the
    names registered. Never raises: a team with no skills, or a platform that
    does not answer in time, starts the session without them. Without an
    account there is no team: the only published skills are this profile's
    own, which Hermes already loads."""
    settings = settings or config.load(home)
    if not settings.signed_in:
        return []
    try:
        b = backend.for_settings(settings)
        if tables.SKILLS not in b.table_names():
            return []
        rows = b.query(f"SELECT name, description, content, created_ms FROM {tables.SKILLS} "
                       f"ORDER BY created_ms DESC LIMIT {MAX_SKILL_ROWS}", timeout_s=LOAD_TIMEOUT_S)
    except Exception as e:
        logger.info("infino: team skills not loaded: %s", e)
        return []
    root = (home or settings.home) / CACHE_DIR
    registered = []
    for name, row in sorted(latest(rows).items()):
        try:
            path = root / name / "SKILL.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(row["content"]))
            ctx.register_skill(name, path, str(row.get("description") or ""))
            registered.append(name)
        except Exception as e:
            logger.info("infino: team skill %s not registered: %s", name, e)
    return registered


def description_of(skill_md: str) -> str:
    """The `description:` of a SKILL.md's frontmatter, or empty."""
    m = re.search(r"^---\s*\n(.*?)\n---", skill_md, re.S)
    if not m:
        return ""
    d = re.search(r"^description:\s*(.+)$", m.group(1), re.M)
    return d.group(1).strip().strip("\"'") if d else ""


def skill_row(name: str, skill_md: str, author: str, profile: str) -> dict:
    return {
        "name": name,
        "description": description_of(skill_md),
        "content": skill_md,
        "author": author,
        "profile": profile,
        "created_ms": int(time.time() * 1000),
    }


def find_local_skill(name: str, home: Optional[Path] = None) -> Optional[Path]:
    """The profile's own SKILL.md for `name`, wherever under `skills/` it sits."""
    root = (home or config.home()) / "skills"
    for path in sorted(root.rglob("SKILL.md")) if root.is_dir() else []:
        if path.parent.name == name:
            return path
    return None

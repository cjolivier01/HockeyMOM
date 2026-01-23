from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional

from .orm import _orm_modules


# Schedules may include "<division> Seed <n>" placeholders, which are not real teams.
_SEED_PLACEHOLDER_RE = re.compile(r"(?i)^\s*(?P<div>.+?)\s+seed\s*#?\s*(?P<seed>[1-9]\d*)\s*$")

# Generic team row used to satisfy non-null FK constraints for placeholder games.
SEED_PLACEHOLDER_TEAM_NAME = "Playoff Seed"


@dataclass(frozen=True)
class SeedPlaceholder:
    division_token: str
    seed: int
    raw: str


def parse_seed_placeholder_name(name: str) -> Optional[SeedPlaceholder]:
    s = str(name or "").replace("\xa0", " ").strip()
    if not s:
        return None
    m = _SEED_PLACEHOLDER_RE.match(s)
    if not m:
        return None
    div = str(m.group("div") or "").strip()
    if not div:
        return None
    try:
        seed = int(m.group("seed"))
    except Exception:
        return None
    if seed <= 0:
        return None
    return SeedPlaceholder(division_token=div, seed=seed, raw=s)


def is_seed_placeholder_name(name: str) -> bool:
    return parse_seed_placeholder_name(name) is not None


def ensure_seed_placeholder_team_for_import(owner_user_id: int, *, commit: bool = True) -> int:
    del commit
    _django_orm, m = _orm_modules()
    existing = (
        m.Team.objects.filter(user_id=int(owner_user_id), name=SEED_PLACEHOLDER_TEAM_NAME)
        .values_list("id", flat=True)
        .first()
    )
    if existing is not None:
        return int(existing)
    t = m.Team.objects.create(
        user_id=int(owner_user_id),
        name=SEED_PLACEHOLDER_TEAM_NAME,
        is_external=True,
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(t.id)

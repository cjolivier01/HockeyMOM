"""Direct (no-sqlite) accessors for TimeToScore sites.

This module provides helpers to fetch seasons/divisions/schedules and per-game stats
without using the sqlite cache in `hmlib.time2score.database`.

The goal is to keep all website-specific knowledge under `hmlib/time2score/` so
other code (e.g. webapp import scripts) can be thin adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class Division:
    season_id: int
    division_id: int
    conference_id: int
    name: str
    teams: list[dict[str, Any]]  # at least: {"id": int, "name": str}


def _module_for_source(source: str):
    src = str(source or "").strip().lower()
    if src == "caha":
        from . import caha_lib

        return caha_lib
    if src in ("sharksice", "sharks_ice", "siahl"):
        from . import sharks_ice_lib

        return sharks_ice_lib
    raise ValueError(f"Unknown time2score source {source!r} (expected 'caha' or 'sharksice')")


def list_seasons(source: str) -> dict[str, int]:
    mod = _module_for_source(source)
    seasons = mod.scrape_seasons()
    out: dict[str, int] = {}
    for k, v in (seasons or {}).items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def pick_current_season_id(source: str) -> int:
    seasons = list_seasons(source)
    if "Current" in seasons and int(seasons["Current"]) > 0:
        return int(seasons["Current"])
    nonzero = [int(v) for v in seasons.values() if int(v) > 0]
    return max(nonzero) if nonzero else 0


def list_divisions(source: str, *, season_id: int) -> list[Division]:
    mod = _module_for_source(source)
    divs = mod.scrape_season_divisions(season_id=int(season_id))
    out: list[Division] = []
    for d in divs or []:
        try:
            out.append(
                Division(
                    season_id=int(season_id),
                    division_id=int(d.get("id") or 0),
                    conference_id=int(d.get("conferenceId") or 0),
                    name=str(d.get("name") or ""),
                    teams=list(d.get("teams") or []),
                )
            )
        except Exception:
            continue
    return out


def iter_season_games(
    source: str,
    *,
    season_id: int,
    divisions: Optional[Iterable[tuple[int, int]]] = None,
    team_name_substrings: Optional[Iterable[str]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    progress_every_teams: int = 25,
    heartbeat_seconds: float = 60.0,
) -> dict[int, dict[str, Any]]:
    """Return a mapping {game_id -> best-effort schedule info}.

    The returned values are schedule-derived and are useful as fallbacks when
    per-game stats are incomplete. Keys may include:
      - home, away (team names)
      - start_time (datetime or string, depending on site parser)
      - rink
      - division_id, conference_id, division_name
    """
    mod = _module_for_source(source)
    logger = logging.getLogger(__name__)
    allowed_divs = set(divisions) if divisions is not None else None

    def _norm(s: str) -> str:
        return str(s or "").lower().replace(" ", "")

    team_filters = [_norm(x) for x in (team_name_substrings or []) if str(x or "").strip()]

    games_by_id: dict[int, dict[str, Any]] = {}
    divs_list = list_divisions(source, season_id=int(season_id))
    total_teams = 0
    for div in divs_list:
        key = (int(div.division_id), int(div.conference_id))
        if allowed_divs is not None and key not in allowed_divs:
            continue
        total_teams += len(div.teams)

    processed_teams = 0
    last_beat = time.monotonic()
    if progress_cb:
        progress_cb(f"Discovering schedules for {total_teams} teams...")
    elif total_teams:
        logger.info("Discovering schedules for %d teams...", total_teams)

    for div in divs_list:
        key = (int(div.division_id), int(div.conference_id))
        if allowed_divs is not None and key not in allowed_divs:
            continue
        for t in div.teams:
            tid = t.get("id")
            if tid is None:
                continue
            try:
                tid_i = int(tid)
            except Exception:
                continue
            processed_teams += 1
            now = time.monotonic()
            if progress_cb:
                if progress_every_teams > 0 and (processed_teams % int(progress_every_teams) == 0):
                    progress_cb(f"Discovered schedules: {processed_teams}/{total_teams} teams...")
                    last_beat = now
                elif heartbeat_seconds > 0 and (now - last_beat) >= float(heartbeat_seconds):
                    progress_cb(f"Discovered schedules: {processed_teams}/{total_teams} teams...")
                    last_beat = now
            info = mod.get_team(season_id=int(season_id), team_id=tid_i)
            for g in list((info or {}).get("games") or []):
                try:
                    gid = int(g.get("id"))
                except Exception:
                    continue
                home = str(g.get("home") or "").strip()
                away = str(g.get("away") or "").strip()
                if team_filters:
                    hn = _norm(home)
                    an = _norm(away)
                    if not any(tf in hn or tf in an for tf in team_filters):
                        continue
                existing = games_by_id.get(gid) or {}
                # Prefer non-empty names / start_time / rink.
                merged = dict(existing)
                for k2 in ("home", "away", "start_time", "rink", "league", "level", "type"):
                    v2 = g.get(k2)
                    if merged.get(k2) in ("", None) and v2 not in ("", None):
                        merged[k2] = v2
                merged.setdefault("division_id", int(div.division_id))
                merged.setdefault("conference_id", int(div.conference_id))
                merged.setdefault("division_name", div.name)
                games_by_id[gid] = merged
    return games_by_id


def scrape_game_stats(source: str, *, game_id: int, season_id: Optional[int] = None) -> dict[str, Any]:
    mod = _module_for_source(source)
    if source.strip().lower() in ("sharksice", "sharks_ice", "siahl"):
        return mod.scrape_game_stats(int(game_id), season_id=int(season_id) if season_id is not None else None)
    return mod.scrape_game_stats(int(game_id))

"""Direct (no-sqlite) accessors for TimeToScore sites.

This module provides helpers to fetch seasons/divisions/schedules and per-game stats
without using the sqlite cache in `hmlib.time2score.database`.

The goal is to keep all website-specific knowledge under `hmlib/time2score/` so
other code (e.g. webapp import scripts) can be thin adapters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urljoin


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


def list_seasons(source: str, *, league_id: Optional[int] = None) -> dict[str, int]:
    mod = _module_for_source(source)
    src = str(source or "").strip().lower()
    if src == "caha":
        league_id_i = (
            int(league_id) if league_id is not None and int(league_id) > 0 else int(mod.CAHA_LEAGUE)
        )
        seasons = mod.scrape_seasons(league_id=league_id_i)
    else:
        seasons = mod.scrape_seasons()
    out: dict[str, int] = {}
    for k, v in (seasons or {}).items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


def pick_current_season_id(source: str, *, league_id: Optional[int] = None) -> int:
    seasons = list_seasons(source, league_id=league_id)
    if "Current" in seasons and int(seasons["Current"]) > 0:
        return int(seasons["Current"])
    nonzero = [int(v) for v in seasons.values() if int(v) > 0]
    return max(nonzero) if nonzero else 0


def list_divisions(
    source: str, *, season_id: int, league_id: Optional[int] = None
) -> list[Division]:
    mod = _module_for_source(source)
    src = str(source or "").strip().lower()
    if src == "caha":
        league_id_i = (
            int(league_id) if league_id is not None and int(league_id) > 0 else int(mod.CAHA_LEAGUE)
        )
        divs = mod.scrape_season_divisions(season_id=int(season_id), league_id=league_id_i)
    else:
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
    league_id: Optional[int] = None,
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
    divs_list = list_divisions(source, season_id=int(season_id), league_id=league_id)

    # CAHA youth exposes a league-wide schedule page that includes scores and game ids.
    # Prefer it to avoid slow per-team traversal and to capture scores even when boxscore pages are missing.
    if str(source or "").strip().lower() == "caha":
        name_to_ids: dict[str, tuple[int, int, str]] = {}
        for div in divs_list:
            dn = str(div.name or "").strip()
            key = dn.lower().replace(" ", "")
            name_to_ids[key] = (int(div.division_id), int(div.conference_id), dn)

        def _norm_div(s: str) -> str:
            t = str(s or "").replace("\xa0", " ").strip()
            # "10U A" -> "10 A"
            t = t.replace("U ", " ")
            t = t.replace("U\t", " ")
            t = t.replace("U", " ")
            t = " ".join(t.split())
            return t.lower().replace(" ", "")

        total_rows = 0
        try:
            if league_id is not None:
                schedule_rows = mod.scrape_league_schedule(season_id=int(season_id), league_id=int(league_id))  # type: ignore[attr-defined]
            else:
                schedule_rows = mod.scrape_league_schedule(season_id=int(season_id))  # type: ignore[attr-defined]
        except Exception:
            schedule_rows = []
        total_rows = len(schedule_rows)
        if progress_cb:
            progress_cb(f"Loading league schedule rows: {total_rows}...")
        elif total_rows:
            logger.info("Loading league schedule rows: %d...", total_rows)

        # Map rows by game id
        for row in schedule_rows:
            raw_gid = str((row or {}).get("id") or "").strip()
            gid_s = "".join([c for c in raw_gid if c.isdigit()])
            if not gid_s:
                continue
            gid = int(gid_s)
            home = str((row or {}).get("home") or "").strip()
            away = str((row or {}).get("away") or "").strip()
            if not home or not away:
                continue

            if team_filters:
                hn = _norm(home)
                an = _norm(away)
                if not any(tf in hn or tf in an for tf in team_filters):
                    continue

            date_s = str((row or {}).get("date") or "").strip()
            time_s = str((row or {}).get("time") or "").strip()
            start_time = None
            if date_s and time_s:
                try:
                    from . import util as tutil

                    start_time = tutil.parse_game_time(date_s, time_s, year=None)
                except Exception:
                    start_time = None

            level = str((row or {}).get("level") or "").strip()
            div_key = _norm_div(level)
            div_id = None
            conf_id = None
            div_name = level.replace("U ", " ").replace("U", " ").strip() if level else ""
            if div_key in name_to_ids:
                div_id, conf_id, div_name = name_to_ids[div_key]

            # We can't apply allowed_divs accurately without ids; if ids are known, enforce.
            if allowed_divs is not None and div_id is not None and conf_id is not None:
                if (int(div_id), int(conf_id)) not in allowed_divs:
                    continue

            def _parse_goal(v: Any) -> Optional[int]:
                s = str(v).strip()
                if not s:
                    return None
                s2 = "".join([c for c in s if c.isdigit()])
                return int(s2) if s2.isdigit() else None

            info = {
                "home": home,
                "away": away,
                "start_time": start_time,
                "rink": str((row or {}).get("rink") or "").strip() or None,
                "league": str((row or {}).get("league") or "").strip() or None,
                "level": level,
                "type": str((row or {}).get("type") or "").strip() or None,
                "division_id": int(div_id) if div_id is not None else None,
                "conference_id": int(conf_id) if conf_id is not None else None,
                "division_name": div_name or None,
                "homeGoals": _parse_goal((row or {}).get("homeGoals")),
                "awayGoals": _parse_goal((row or {}).get("awayGoals")),
            }
            games_by_id[gid] = info

        return games_by_id

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
            if str(source or "").strip().lower() == "caha":
                league_id_i = (
                    int(league_id)
                    if league_id is not None and int(league_id) > 0
                    else int(mod.CAHA_LEAGUE)
                )
                info = mod.get_team(season_id=int(season_id), team_id=tid_i, league_id=league_id_i)
            else:
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
                for k2 in (
                    "home",
                    "away",
                    "start_time",
                    "rink",
                    "league",
                    "level",
                    "type",
                    "homeGoals",
                    "awayGoals",
                ):
                    v2 = g.get(k2)
                    if merged.get(k2) in ("", None) and v2 not in ("", None):
                        merged[k2] = v2
                merged.setdefault("division_id", int(div.division_id))
                merged.setdefault("conference_id", int(div.conference_id))
                merged.setdefault("division_name", div.name)
                games_by_id[gid] = merged
    return games_by_id


def scrape_game_stats(
    source: str, *, game_id: int, season_id: Optional[int] = None
) -> dict[str, Any]:
    mod = _module_for_source(source)
    if source.strip().lower() in ("sharksice", "sharks_ice", "siahl"):
        return mod.scrape_game_stats(
            int(game_id), season_id=int(season_id) if season_id is not None else None
        )
    return mod.scrape_game_stats(int(game_id))


def scrape_team_logo_url(
    source: str, *, season_id: int, team_id: int, league_id: Optional[int] = None
) -> Optional[str]:
    """Best-effort scrape of the team logo URL from the team schedule page."""
    mod = _module_for_source(source)
    # Import here to avoid cycles; util already depends on requests/bs4.
    from . import util

    params: dict[str, str] = {}
    src = str(source or "").strip().lower()
    if src == "caha":
        league_id_i = (
            int(league_id) if league_id is not None and int(league_id) > 0 else int(mod.CAHA_LEAGUE)
        )
        params = {
            "team": str(int(team_id)),
            "league": str(league_id_i),
            "stat_class": str(mod.STAT_CLASS),
        }
        if int(season_id) > 0:
            params["season"] = str(int(season_id))
    else:
        params = {"team": str(int(team_id)), "season": str(int(season_id))}

    soup = util.get_html(str(mod.TEAM_URL), params=params)

    candidates: list[str] = []
    for img in soup.find_all("img"):
        src_attr = str(img.get("src") or "").strip()
        if not src_attr:
            continue
        # Some sharksice teams have a placeholder URL with no filename.
        if src_attr.endswith("/"):
            continue
        full = src_attr
        if not (full.startswith("http://") or full.startswith("https://")):
            base = getattr(mod, "TIMETOSCORE_URL", "")
            full = urljoin(str(base), full)
        candidates.append(full)

    if not candidates:
        return None

    def _score(u: str) -> tuple[int, int]:
        ul = u.lower()
        has_logo = 1 if ("logo" in ul or "/logos/" in ul or "/logo/" in ul) else 0
        looks_img = (
            1
            if any(ul.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"))
            else 0
        )
        return (has_logo, looks_img)

    # Prefer explicit logo-like URLs; otherwise if there's a single image on the page (common on CAHA),
    # accept it as the logo.
    best = sorted(candidates, key=_score, reverse=True)[0]
    if _score(best) != (0, 0):
        return best
    if len(candidates) == 1:
        return candidates[0]
    return None

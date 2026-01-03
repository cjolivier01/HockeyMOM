#!/usr/bin/env python3
"""Import the current TimeToScore season into the shared "Norcal" league via the webapp REST API.

By default this targets the SharksIce TimeToScore site (NorCal) and posts upserts to a locally
running webapp (`http://127.0.0.1:8008`). Use `--url` to target a remote deployment.

Existing stats are not overwritten unless `--replace` is passed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional


@contextlib.contextmanager
def chdir(path: Path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def _pick_current_season_id(seasons: list[dict[str, Any]]) -> int:
    nonzero = []
    for s in seasons:
        try:
            sid = int(s.get("season_id"))
        except Exception:
            continue
        if sid > 0:
            nonzero.append(sid)
    return max(nonzero) if nonzero else 0


def _discover_sharksice_current_season_id() -> int:
    import requests

    url = "https://stats.sharksice.timetoscore.com/display-stats.php"
    resp = requests.get(url, params={"league": "1"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    seasons = sorted({int(m.group(1)) for m in re.finditer(r"season=(\d+)", resp.text)})
    if not seasons:
        raise RuntimeError("Could not discover current SharksIce season id from display-stats.php")
    return int(max(seasons))


def _parse_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s.split()[0])
    except Exception:
        return None


def _aggregate_player_stats(game_stats: dict[str, Any]) -> list[dict[str, Any]]:
    # SharksIce game-center API provides per-player totals directly.
    for key_home, key_away in (("homeSkaters", "awaySkaters"), ("home_skaters", "away_skaters")):
        if isinstance(game_stats.get(key_home), list) or isinstance(game_stats.get(key_away), list):
            out = []
            for row in list(game_stats.get(key_home) or []) + list(game_stats.get(key_away) or []):
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    gval = int(row.get("goals") or 0)
                except Exception:
                    gval = 0
                try:
                    aval = int(row.get("assists") or 0)
                except Exception:
                    aval = 0
                if gval == 0 and aval == 0:
                    continue
                out.append({"name": name, "goals": gval, "assists": aval})
            return out

    def incr(d: Dict[str, Dict[str, int]], who: str, key: str) -> None:
        rec = d.setdefault(who, {"goals": 0, "assists": 0})
        rec[key] = rec.get(key, 0) + 1

    stats_by_player: Dict[str, Dict[str, int]] = {}
    for side in ("home", "away"):
        scoring = game_stats.get(f"{side}Scoring") or []
        if not isinstance(scoring, list):
            continue
        for srow in scoring:
            if not isinstance(srow, dict):
                continue
            gname = str(srow.get("goal") or "").strip()
            a1 = str(srow.get("assist1") or "").strip()
            a2 = str(srow.get("assist2") or "").strip()
            if gname:
                incr(stats_by_player, gname, "goals")
            if a1:
                incr(stats_by_player, a1, "assists")
            if a2:
                incr(stats_by_player, a2, "assists")

    out = []
    for name, rec in stats_by_player.items():
        out.append({"name": name, "goals": int(rec.get("goals", 0)), "assists": int(rec.get("assists", 0))})
    return out


def _extract_roster(game_stats: dict[str, Any], side: str) -> list[dict[str, Any]]:
    rows = game_stats.get(f"{side}Players") or []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "number": str(row.get("number") or "").strip() or None,
                "position": str(row.get("position") or "").strip() or None,
            }
        )
    return out


def _build_game_payload(
    *,
    game_id: int,
    season_id: int,
    source: str,
    lib,
    util_mod,
    fallback: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    try:
        if source == "sharksice":
            stats = lib.scrape_game_stats(int(game_id), season_id=int(season_id))
        else:
            stats = lib.scrape_game_stats(int(game_id))
    except Exception:
        return None
    home_name = str(stats.get("home") or "").strip()
    away_name = str(stats.get("away") or "").strip()
    date_s = str(stats.get("date") or "").strip()
    time_s = str(stats.get("time") or "").strip()

    if not home_name and isinstance((fallback or {}).get("home"), str):
        home_name = str((fallback or {}).get("home") or "").strip()
    if not away_name and isinstance((fallback or {}).get("away"), str):
        away_name = str((fallback or {}).get("away") or "").strip()
    if not home_name or not away_name:
        return None

    loc = str(stats.get("location") or "").strip() or None
    if not loc and isinstance((fallback or {}).get("rink"), str):
        loc = str((fallback or {}).get("rink") or "").strip() or None

    starts_at = None
    if date_s and time_s:
        try:
            dt_val = util_mod.parse_game_time(date_s, time_s, year=None)
            starts_at = dt_val.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # SharksIce API returns full date strings like "Wednesday, September 10, 2025".
            try:
                d = dt.datetime.strptime(date_s, "%A, %B %d, %Y").date()
                t = dt.datetime.strptime(time_s, "%I:%M %p").time()
                starts_at = dt.datetime.combine(d, t).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                starts_at = None
    if not starts_at:
        st = (fallback or {}).get("start_time")
        if isinstance(st, str) and st.strip():
            starts_at = st.strip()

    return {
        "timetoscore_game_id": int(game_id),
        "season_id": int(season_id),
        "home_name": home_name,
        "away_name": away_name,
        "starts_at": starts_at,
        "location": loc,
        "home_score": _parse_int(stats.get("homeGoals")),
        "away_score": _parse_int(stats.get("awayGoals")),
        "home_roster": _extract_roster(stats, "home"),
        "away_roster": _extract_roster(stats, "away"),
        "player_stats": _aggregate_player_stats(stats),
        "source": source,
    }


def _post_json(url: str, payload: dict[str, Any], token: Optional[str]) -> dict[str, Any]:
    import requests

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:500]!r}")
    data = resp.json()
    if not isinstance(data, dict) or not data.get("ok", False):
        raise RuntimeError(f"API call failed: {data!r}")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Import TimeToScore current season into shared Norcal league")
    ap.add_argument("--url", default="http://127.0.0.1:8008", help="Base webapp URL (default: localhost)")
    ap.add_argument(
        "--token",
        default=os.environ.get("HM_WEBAPP_IMPORT_TOKEN"),
        help="Import token (env: HM_WEBAPP_IMPORT_TOKEN). Required for non-localhost deployments.",
    )
    ap.add_argument("--replace", action="store_true", help="Overwrite existing stats (player_stats and scores)")
    ap.add_argument("--season", type=int, default=0, help="Season id (0 = current/latest)")
    ap.add_argument(
        "--source",
        choices=("sharksice", "caha"),
        default="sharksice",
        help="TimeToScore site to scrape (default: sharksice)",
    )
    ap.add_argument("--league-name", default="Norcal", help="Webapp league name to upsert into")
    ap.add_argument("--shared", dest="shared", action="store_true", default=True, help="Mark league as shared")
    ap.add_argument("--private", dest="shared", action="store_false", help="Make league private")
    ap.add_argument("--owner-email", default="norcal-import@hockeymom.local", help="Owner email for imported data")
    ap.add_argument("--owner-name", default="Norcal Import", help="Owner display name")
    ap.add_argument(
        "--tts-db-dir",
        default=str(base_dir / "instance" / "time2score_db_norcal"),
        help="Directory to hold the sqlite hockey_league.db cache",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max number of games to import")
    args = ap.parse_args(argv)

    base_url = str(args.url).rstrip("/")
    ensure_url = f"{base_url}/api/import/hockey/ensure_league"
    game_url = f"{base_url}/api/import/hockey/game"

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)
    # SharksIce scraper imports `database` and `util` as top-level modules; add the package dir too.
    tts_pkg_dir = repo_root / "hmlib" / "time2score"
    tts_pkg_s = str(tts_pkg_dir)
    if tts_pkg_s not in sys.path:
        sys.path.insert(0, tts_pkg_s)

    import database as tdb_mod  # type: ignore
    import util as tutil_mod  # type: ignore

    if args.source == "sharksice":
        from hmlib.time2score import sharks_ice_lib as tts_lib
    else:
        from hmlib.time2score import caha_lib as tts_lib

    tts_dir = Path(args.tts_db_dir)
    with chdir(tts_dir):
        tdb = tdb_mod.Database()
        tdb.create_tables()

        season_id = int(args.season)
        if args.source == "sharksice":
            if season_id == 0:
                season_id = _discover_sharksice_current_season_id()
            # Ensure the season exists in the sqlite DB even if the upstream page no longer lists seasons.
            try:
                tdb.add_season(int(season_id), f"Season {season_id}")
            except Exception:
                pass
            try:
                tdb.add_season(0, "Current")
            except Exception:
                pass
        else:
            tts_lib.sync_seasons(tdb)
            seasons = tdb.list_seasons()
            if season_id == 0:
                season_id = _pick_current_season_id(seasons)

        tts_lib.sync_divisions(tdb, season_id)
        tts_lib.sync_season_teams(tdb, season_id)

        games_info = tdb.list_games_info(f"g.season_id = {int(season_id)}")
        games_info_by_id: dict[int, dict[str, Any]] = {}
        for g in games_info:
            try:
                gid = int(g.get("game_id"))
            except Exception:
                continue
            games_info_by_id[gid] = g
        game_ids = sorted(games_info_by_id.keys())

    season_key = f"{args.source}:{season_id}"
    _post_json(
        ensure_url,
        {
            "league_name": args.league_name,
            "shared": bool(args.shared),
            "owner_email": args.owner_email,
            "owner_name": args.owner_name,
            "source": "timetoscore",
            "external_key": season_key,
        },
        args.token,
    )

    count = 0
    skipped = 0
    for gid in game_ids:
        if args.limit is not None and count >= int(args.limit):
            break
        with chdir(Path(args.tts_db_dir)):
            payload_game = _build_game_payload(
                game_id=int(gid),
                season_id=int(season_id),
                source=str(args.source),
                lib=tts_lib,
                util_mod=tutil_mod,
                fallback=games_info_by_id.get(int(gid)),
            )
        if payload_game is None:
            skipped += 1
            continue
        _post_json(
            game_url,
            {
                "league_name": args.league_name,
                "shared": bool(args.shared),
                "replace": bool(args.replace),
                "owner_email": args.owner_email,
                "owner_name": args.owner_name,
                "source": "timetoscore",
                "external_key": season_key,
                "game": payload_game,
            },
            args.token,
        )
        count += 1
        if count % 25 == 0:
            ts = dt.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Imported {count}/{len(game_ids)} games...", flush=True)

    print(
        f"Imported {count} games into league {args.league_name!r} (season {season_id}, source {args.source}). Skipped {skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

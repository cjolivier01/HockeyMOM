"""Lightweight API for CAHA TimeToScore data.

Provides a function to retrieve aggregated game details by game id.
If the local database is empty or the game is missing, it can perform
on-demand syncing for the specified season.
"""

from __future__ import annotations

from typing import Any, Optional
import json

try:  # pragma: no cover
  from . import caha_lib  # type: ignore
  from . import database  # type: ignore
except Exception:  # noqa: BLE001
  # Fallback for direct execution without package context.
  import sys as _sys
  import os as _os
  _here = _os.path.dirname(__file__)
  if _here not in _sys.path:
    _sys.path.insert(0, _here)
  import caha_lib  # type: ignore  # noqa: E402
  import database  # type: ignore  # noqa: E402


def _choose_season(db: database.Database, preferred: Optional[int]) -> int:
  if preferred is not None:
    return preferred
  # Use known seasons, prefer non-zero max id, else 0 (Current)
  seasons = db.list_seasons()
  if not seasons:
    # scrape and store seasons if none exist
    caha_lib.sync_seasons(db)
    seasons = db.list_seasons()
  # Prefer highest numeric id > 0
  nonzero = [s["season_id"] for s in seasons if s["season_id"] > 0]
  if nonzero:
    return max(nonzero)
  return 0


def _ensure_synced_for_season(db: database.Database, season: int) -> None:
  # Ensure core tables exist
  db.create_tables()
  # Ensure seasons list contains this season
  has_season = db.get_season(season) is not None
  if not has_season:
    # load seasons from remote
    caha_lib.sync_seasons(db)
  # Ensure we have divisions+teams and schedules for this season
  if not db.list_teams(f"season_id = {season}"):
    caha_lib.sync_divisions(db, season)
  # Populate schedules/games if not present for this season
  if not db.list_games(f"season_id = {season}"):
    caha_lib.sync_season_teams(db, season)


def _json_load_stats_if_needed(team_row: dict[str, Any]) -> dict[str, Any]:
  # Teams.stats is stored as JSON string in DB layer; convert to dict.
  stats = team_row.get("stats")
  if isinstance(stats, str):
    try:
      return json.loads(stats)
    except Exception:
      return {}
  return stats or {}


def get_game_details(
    game_id: int,
    *,
    season: Optional[int] = None,
    sync_if_missing: bool = True,
    fetch_stats_if_missing: bool = True,
) -> dict[str, Any]:
  """Return aggregated info for a game id.

  - If DB is empty or the game is not found, sync for the selected season.
  - If stats are missing, optionally scrape and store them.

  Returns a dict with keys: game, home, away, stats.
  """
  db = database.Database()
  db.create_tables()

  # Try find the game as-is
  row = db.get_game(game_id)
  the_season = season

  if row is None and sync_if_missing:
    # Decide which season to sync
    the_season = _choose_season(db, season)
    _ensure_synced_for_season(db, the_season)
    row = db.get_game(game_id)

  # If still missing, try scraping stats-only (no DB insert of schedule)
  scraped_stats = None
  if row is None and fetch_stats_if_missing:
    try:
      scraped_stats = caha_lib.scrape_game_stats(game_id)
    except Exception:
      scraped_stats = None

  if row is None:
    # Return best-effort info from scraped stats
    return {
        "game": {"game_id": game_id},
        "home": {"team": None, "score": None},
        "away": {"team": None, "score": None},
        "stats": scraped_stats,
    }

  # Row found. Ensure stats are present if requested.
  if fetch_stats_if_missing and not row.get("stats"):
    try:
      stats = caha_lib.scrape_game_stats(game_id)
      db.add_game_stats(game_id, stats)
      row = db.get_game(game_id)
    except Exception:
      pass

  # Collect team rows
  home_team = db.get_team(row["season_id"], row["division_id"], row["conference_id"], row["homeId"]) if row.get("homeId") is not None else None
  away_team = db.get_team(row["season_id"], row["division_id"], row["conference_id"], row["awayId"]) if row.get("awayId") is not None else None

  # Build response
  info = row.get("info") or {}
  stats = row.get("stats") or scraped_stats
  result = {
      "game": {
          "game_id": row["game_id"],
          "season_id": row["season_id"],
          "division_id": row["division_id"],
          "conference_id": row["conference_id"],
          "start_time": row.get("start_time"),
          "rink": row.get("rink"),
          "league": info.get("league"),
          "level": info.get("level"),
          "type": info.get("type"),
      },
      "home": {
          "team": home_team and {
              "team_id": home_team["team_id"],
              "name": home_team["name"],
              "season_id": home_team["season_id"],
              "division_id": home_team["division_id"],
              "conference_id": home_team["conference_id"],
              "stats": _json_load_stats_if_needed(home_team),
          },
          "score": info.get("homeGoals"),
      },
      "away": {
          "team": away_team and {
              "team_id": away_team["team_id"],
              "name": away_team["name"],
              "season_id": away_team["season_id"],
              "division_id": away_team["division_id"],
              "conference_id": away_team["conference_id"],
              "stats": _json_load_stats_if_needed(away_team),
          },
          "score": info.get("awayGoals"),
      },
      "stats": stats,
  }
  return result

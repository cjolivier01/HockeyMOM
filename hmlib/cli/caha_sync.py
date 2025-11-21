import argparse
import os
import sys
from typing import Optional

# Try normal package imports first; if the broader hmlib package pulls in
# optional native deps (e.g., hockeymom), fall back to importing the time2score
# modules directly from source so this CLI can run standalone.
try:  # pragma: no cover - import ergonomics
  from hmlib.time2score import caha_lib  # type: ignore
  from hmlib.time2score import database  # type: ignore
except Exception as _e:  # noqa: BLE001
  # Fallback path-local import
  this_dir = os.path.dirname(__file__)
  t2s_path = os.path.normpath(os.path.join(this_dir, "..", "time2score"))
  if t2s_path not in sys.path:
    sys.path.insert(0, t2s_path)
  import caha_lib  # type: ignore  # noqa: E402
  import database  # type: ignore  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
  parser = argparse.ArgumentParser(description="Sync CAHA youth data from TimeToScore")
  parser.add_argument(
      "--season",
      type=int,
      default=0,
      help="Season id to sync (e.g., 30). 0 uses 'Current' if available.",
  )
  parser.add_argument(
      "--update-stats",
      action="store_true",
      help="Fetch oss-scoresheet stats for games without stats.",
  )
  parser.add_argument(
      "--limit-divisions",
      type=int,
      default=0,
      help="If >0, only sync this many divisions (useful for testing).",
  )
  parser.add_argument(
      "--limit-teams",
      type=int,
      default=0,
      help="If >0, only sync this many teams per division (useful for testing).",
  )
  parser.add_argument(
      "--print-sample",
      action="store_true",
      help="Print a small sample of the scraped data.",
  )
  args = parser.parse_args(argv)

  db = database.Database()
  db.create_tables()

  # Ensure seasons list includes Current
  caha_lib.sync_seasons(db)
  seasons = {s["season_id"]: s for s in db.list_seasons()}
  if args.season == 0:
    # Use Current if present
    if any(s["name"] == "Current" for s in seasons.values()):
      # Take the season id of Current entry
      # If stored as 0, that's fine; our scraper handles missing season id
      season = next((sid for sid, s in seasons.items() if s["name"] == "Current"), 0)
    else:
      season = 0
  else:
    season = args.season

  # Scrape divisions/teams for the requested season
  divs = caha_lib.scrape_season_divisions(season)
  if args.limit_divisions > 0:
    divs = divs[: args.limit_divisions]
  print(f"Found {len(divs)} divisions in season {season}…")

  for div in divs:
    db.add_division(div["id"], div["conferenceId"], div["name"])
    teams = div["teams"]
    if args.limit_teams > 0:
      teams = teams[: args.limit_teams]
    print(f"  Division {div['name']} (level={div['id']} conf={div['conferenceId']}) teams={len(teams)}")
    for t in teams:
      team_id = t.pop("id")
      team_name = t.pop("name")
      db.add_team(
          season_id=season,
          division_id=div["id"],
          conference_id=div["conferenceId"],
          team_id=team_id,
          name=team_name,
          stats=t,
      )

  # Fetch team schedules and store Games
  game_count = 0
  teams = db.list_teams(f"season_id = {season}")
  if args.limit_divisions > 0:
    # further narrow teams to those in the selected divisions
    allowed = {(d["id"], d["conferenceId"]) for d in divs}
    teams = [t for t in teams if (t["division_id"], t["conference_id"]) in allowed]
  if args.limit_teams > 0:
    # keep first N per division
    temp = []
    per_div = {}
    for t in teams:
      key = (t["division_id"], t["conference_id"])
      per_div.setdefault(key, 0)
      if per_div[key] < args.limit_teams:
        temp.append(t)
        per_div[key] += 1
    teams = temp

  seen_games = set()
  for t in teams:
    team_info = caha_lib.get_team(season_id=season, team_id=t["team_id"])  # type: ignore
    for g in team_info.get("games", []):
      gid = g.get("id")
      if not gid or gid in seen_games:
        continue
      seen_games.add(gid)
      caha_lib.add_game(db, season, t, dict(g))
      game_count += 1
  print(f"Stored {game_count} games.")

  if args.update_stats:
    caha_lib.sync_game_stats(db)

  if args.print_sample:
    # Show a few games with joined info
    sample = db.list_games_info(f"season_id = {season}")[:5]
    print("Sample games:")
    for s in sample:
      print({
          "home": s["home"],
          "away": s["away"],
          "game_id": s["game_id"],
          "start_time": s["start_time"],
          "rink": s["rink"],
      })

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

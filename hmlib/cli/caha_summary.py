import argparse
import os
import re
import sys
from collections import Counter
from typing import Optional

# Import only the lightweight DB layer; avoid heavy top-level hmlib imports.
try:  # pragma: no cover
  from hmlib.time2score import database  # type: ignore
except Exception:  # noqa: BLE001
  this_dir = os.path.dirname(__file__)
  t2s_path = os.path.normpath(os.path.join(this_dir, "..", "time2score"))
  if t2s_path not in sys.path:
    sys.path.insert(0, t2s_path)
  import database  # type: ignore  # noqa: E402


def _to_int(val):
  if val is None:
    return None
  s = str(val)
  m = re.search(r"\d+", s)
  return int(m.group(0)) if m else None


def _select_team(db: database.Database, season: int, name: str, team_id: Optional[int]):
  if team_id is not None:
    # Find by team_id + season (division unknown across seasons)
    # Find any team row matching this team_id and season
    rows = db.list_teams(f"season_id = {season} AND team_id = {team_id}")
    return rows[0] if rows else None

  # Prefer exact match, otherwise fallback to LIKE
  exact = db.list_teams(f'season_id = {season} AND name = "{name}"')
  if exact:
    return exact[0]
  like = db.list_teams(f'season_id = {season} AND name LIKE "%{name}%"')
  if len(like) == 1:
    return like[0]
  return like  # return list for disambiguation


def main(argv: Optional[list[str]] = None) -> int:
  p = argparse.ArgumentParser(description="Print season summary for a CAHA team from DB")
  p.add_argument("--season", type=int, required=True, help="Season id (e.g., 30)")
  p.add_argument("--name", type=str, help="Team name (full or partial)")
  p.add_argument("--team-id", type=int, default=None, help="Team id if you know it")
  p.add_argument("--json", action="store_true", help="Print JSON summary")
  p.add_argument("--only-regular", action="store_true", help="Only include 'Regular' games in outputs")
  p.add_argument("--only-playoffs", action="store_true", help="Only include playoff games (excludes Regular/Preseason/Exhibition)")
  p.add_argument("--csv-out", type=str, default="", help="Write full game-by-game CSV to this path (use '-' for stdout)")
  p.add_argument("--csv-sep", type=str, default=",", help="CSV separator (default ',')")
  p.add_argument("--export-json", type=str, default="", help="Write full game-by-game JSON to this path (use '-' for stdout)")
  p.add_argument("--opponent", type=str, default="", help="Filter to games vs opponent (substring, case-insensitive)")
  p.add_argument("--since", type=str, default="", help="Filter to games on/after YYYY-MM-DD (by start_time)")
  p.add_argument("--until", type=str, default="", help="Filter to games on/before YYYY-MM-DD (by start_time)")
  p.add_argument("--levels", type=str, default="", help="Comma-separated level filters (case-insensitive substring match), e.g. '12U AA,10U A'")
  args = p.parse_args(argv)

  db = database.Database()
  db.create_tables()

  sel = _select_team(db, args.season, args.name or "", args.team_id)
  if sel is None:
    print("No matching team found. Consider syncing or refining filters.")
    return 1
  if isinstance(sel, list):
    print("Multiple teams matched; be more specific or pass --team-id:")
    for t in sel:
      print({
          "division_id": t["division_id"],
          "conference_id": t["conference_id"],
          "team_id": t["team_id"],
          "name": t["name"],
      })
    return 2

  team = sel
  team_id = team["team_id"]
  name = team["name"]

  rows = db.list_games(
      f"(homeId = {team_id} OR awayId = {team_id}) AND season_id = {args.season}"
  )

  w = l = t = 0
  gf = ga = 0
  by_type = Counter()
  examples = []
  # Helper to decide inclusion by type
  def _include_by_type(info):
    typ = (info.get("type") or "Unknown").split()[0]
    if args.only_regular and args.only_playoffs:
      # If both flags are set, prefer Regular (explicit over implicit)
      return typ == "Regular"
    if args.only_regular:
      return typ == "Regular"
    if args.only_playoffs:
      # Consider anything not Regular/Preseason/Exhibition as playoffs-type
      return typ not in {"Regular", "Preseason", "Exhibition"}
    return True

  def _parse_date(s: str | None):
    if not s:
      return None
    try:
      from datetime import datetime
      return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").date()
    except Exception:
      return None

  since_date = None
  until_date = None
  if args.since:
    from datetime import datetime
    since_date = datetime.strptime(args.since, "%Y-%m-%d").date()
  if args.until:
    from datetime import datetime
    until_date = datetime.strptime(args.until, "%Y-%m-%d").date()

  def _include_by_date(start_time: str | None):
    d = _parse_date(start_time)
    if since_date and (not d or d < since_date):
      return False
    if until_date and (not d or d > until_date):
      return False
    return True

  opp_filter = (args.opponent or "").lower()
  def _include_by_opponent(home_name: str, away_name: str, info_home: str | None, info_away: str | None):
    if not opp_filter:
      return True
    # Prefer DB names, fall back to scraped info
    hn = (home_name or info_home or "").lower()
    an = (away_name or info_away or "").lower()
    return (opp_filter in hn) or (opp_filter in an)

  levels_filter = [s.strip().lower() for s in (args.levels or "").split(",") if s.strip()]
  def _include_by_level(info):
    if not levels_filter:
      return True
    lvl = (info.get("level") or "").lower()
    return any(token in lvl for token in levels_filter)

  for r in rows:
    info = r.get("info") or {}
    if not _include_by_type(info):
      continue
    if not _include_by_date(r.get("start_time")):
      continue
    # For opponent filtering, resolve names once per row
    home_id = r["homeId"]; away_id = r["awayId"]
    home_team = db.get_team(r["season_id"], r["division_id"], r["conference_id"], home_id) if home_id is not None else None
    away_team = db.get_team(r["season_id"], r["division_id"], r["conference_id"], away_id) if away_id is not None else None
    home_name = (home_team and home_team.get("name")) or info.get("home") or ""
    away_name = (away_team and away_team.get("name")) or info.get("away") or ""
    if not _include_by_opponent(home_name, away_name, info.get("home"), info.get("away")):
      continue
    if not _include_by_level(info):
      continue
    hg = _to_int(info.get("homeGoals"))
    ag = _to_int(info.get("awayGoals"))
    if hg is None or ag is None:
      continue
    if r["homeId"] == team_id:
      gfor, gagainst = hg, ag
    else:
      gfor, gagainst = ag, hg
    gf += gfor
    ga += gagainst
    typ = (info.get("type") or "Unknown").split()[0]
    by_type[typ] += 1
    if gfor > gagainst:
      w += 1
    elif gfor < gagainst:
      l += 1
    else:
      t += 1
    if len(examples) < 5:
      opp_id = r["awayId"] if r["homeId"] == team_id else r["homeId"]
      opp = db.get_team(r["season_id"], r["division_id"], r["conference_id"], opp_id)
      examples.append({
          "game_id": r["game_id"],
          "vs": opp["name"] if opp else "UNKNOWN",
          "start_time": r["start_time"],
          "rink": r["rink"],
          "gfor": gfor,
          "gagainst": gagainst,
          "type": info.get("type"),
      })

  summary = {
      "team": name,
      "team_id": team_id,
      "season": args.season,
      "record": {"wins": w, "losses": l, "ties": t},
      "goals": {"for": gf, "against": ga, "diff": gf - ga},
      "games_scored": w + l + t,
      "by_type": dict(by_type),
      "examples": examples,
  }

  if args.json:
    import json

    print(json.dumps(summary, indent=2, sort_keys=True))
  else:
    print(f"Summary for {summary['team']} (season {summary['season']})")
    print(
        f"Record (W-L-T): {w}-{l}-{t}\nGoals For/Against: {gf}/{ga} (diff={gf-ga})\n"
        f"Games with scores: {summary['games_scored']}\nBy type: {summary['by_type']}"
    )
    print("Sample games:")
    for ex in examples:
      print(ex)

  # CSV export of full game-by-game details
  if args.csv_out or args.export_json:
    import csv
    import json
    fields = [
        "game_id",
        "date",
        "time",
        "start_time",
        "rink",
        "league",
        "level",
        "type",
        "home",
        "away",
        "homeGoals",
        "awayGoals",
        "team_role",
        "goals_for",
        "goals_against",
        "result",
    ]
    # Build rows honoring type filter
    out_rows = []
    for r in rows:
      info = r.get("info") or {}
      if not _include_by_type(info):
        continue
      if not _include_by_date(r.get("start_time")):
        continue
      home_id = r["homeId"]; away_id = r["awayId"]
      # Resolve team names
      home_team = db.get_team(r["season_id"], r["division_id"], r["conference_id"], home_id) if home_id is not None else None
      away_team = db.get_team(r["season_id"], r["division_id"], r["conference_id"], away_id) if away_id is not None else None
      home_name = (home_team and home_team.get("name")) or info.get("home") or ""
      away_name = (away_team and away_team.get("name")) or info.get("away") or ""
      if not _include_by_opponent(home_name, away_name, info.get("home"), info.get("away")):
        continue
      if not _include_by_level(info):
        continue

      hg = _to_int(info.get("homeGoals"))
      ag = _to_int(info.get("awayGoals"))
      if team_id == home_id:
        role = "home"; gfor = hg; gagainst = ag
      elif team_id == away_id:
        role = "away"; gfor = ag; gagainst = hg
      else:
        role = "unknown"; gfor = None; gagainst = None
      res = ""
      if gfor is not None and gagainst is not None:
        res = "W" if gfor > gagainst else ("L" if gfor < gagainst else "T")
      out_rows.append({
          "game_id": r["game_id"],
          "date": info.get("date"),
          "time": info.get("time"),
          "start_time": r["start_time"],
          "rink": r["rink"],
          "league": info.get("league"),
          "level": info.get("level"),
          "type": info.get("type"),
          "home": home_name,
          "away": away_name,
          "homeGoals": info.get("homeGoals"),
          "awayGoals": info.get("awayGoals"),
          "team_role": role,
          "goals_for": gfor,
          "goals_against": gagainst,
          "result": res,
      })

    if args.csv_out:
      if args.csv_out == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fields, delimiter=args.csv_sep)
        writer.writeheader()
        writer.writerows(out_rows)
      else:
        with open(args.csv_out, "w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=fields, delimiter=args.csv_sep)
          writer.writeheader()
          writer.writerows(out_rows)
        print(f"Wrote CSV: {args.csv_out} ({len(out_rows)} rows)")

    if args.export_json:
      # JSON payload: metadata + rows
      payload = {
          "team": name,
          "team_id": team_id,
          "season": args.season,
          "filters": {
              "only_regular": args.only_regular,
              "only_playoffs": args.only_playoffs,
          },
          "rows": out_rows,
      }
      if args.export_json == "-":
        print(json.dumps(payload, indent=2, sort_keys=True))
      else:
        with open(args.export_json, "w") as f:
          json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote JSON: {args.export_json} ({len(out_rows)} rows)")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

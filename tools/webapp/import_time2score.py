#!/usr/bin/env python3
"""Import teams and games from hmlib.time2score into the HockeyMOM WebApp database.

This script bridges the sqlite-backed time2score scraper output to the
MySQL/MariaDB schema used by tools/webapp (tables: teams, hky_games, game_types).

Key behavior:
- Connects to the webapp DB using the same JSON config as tools/webapp/app.py
- Optionally scrapes/syncs time2score data for a given season
- Upserts teams as external teams for a specified user
- Upserts hockey games with mapped teams, start time, location, scores, and game type

Usage example:
  python3 tools/webapp/import_time2score.py \
    --config /opt/hm-webapp/app/config.json \
    --season 2024 \
    --user-email demo@example.com \
    --sync --stats

Notes:
- time2score stores its sqlite DB at the current working directory as
  hockey_league.db; to avoid polluting the repo root, this script uses a
  dedicated directory (see --tts-db-dir).
- Teams are imported as is_external=1, associated to the provided user.
  The web UI can include these via the "all=1" query param.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_db_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg.get("db", {})


def connect_pymysql(db_cfg: dict):
    import pymysql

    return pymysql.connect(
        host=db_cfg.get("host", "127.0.0.1"),
        port=int(db_cfg.get("port", 3306)),
        user=db_cfg.get("user", "hmapp"),
        password=db_cfg.get("pass", ""),
        database=db_cfg.get("name", "hm_app_db"),
        autocommit=False,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )


def ensure_league_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leagues (
              id INT AUTO_INCREMENT PRIMARY KEY,
              name VARCHAR(255) UNIQUE NOT NULL,
              owner_user_id INT NOT NULL,
              is_shared TINYINT(1) NOT NULL DEFAULT 0,
              source VARCHAR(64) NULL,
              external_key VARCHAR(255) NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(owner_user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_members (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              user_id INT NOT NULL,
              role VARCHAR(32) NOT NULL DEFAULT 'viewer',
              created_at DATETIME NOT NULL,
              UNIQUE KEY uniq_member (league_id, user_id),
              INDEX(league_id), INDEX(user_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_teams (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              team_id INT NOT NULL,
              UNIQUE KEY uniq_league_team (league_id, team_id),
              INDEX(league_id), INDEX(team_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              game_id INT NOT NULL,
              UNIQUE KEY uniq_league_game (league_id, game_id),
              INDEX(league_id), INDEX(game_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
    conn.commit()


def ensure_league(
    conn,
    name: str,
    owner_user_id: int,
    is_shared: bool,
    source: Optional[str],
    external_key: Optional[str],
) -> int:
    ensure_league_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM leagues WHERE name=%s", (name,))
        r = cur.fetchone()
        if r:
            return int(r[0])
        now = dt.datetime.now().isoformat()
        cur.execute(
            "INSERT INTO leagues(name, owner_user_id, is_shared, source, external_key, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
            (name, owner_user_id, 1 if is_shared else 0, source, external_key, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def ensure_league_member(conn, league_id: int, user_id: int, role: str = "viewer") -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT IGNORE INTO league_members(league_id, user_id, role, created_at) VALUES(%s,%s,%s,%s)",
            (league_id, user_id, role, dt.datetime.now().isoformat()),
        )
    conn.commit()


def map_team_to_league(conn, league_id: int, team_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT IGNORE INTO league_teams(league_id, team_id) VALUES(%s,%s)",
            (league_id, team_id),
        )
    conn.commit()


def map_game_to_league(conn, league_id: int, game_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT IGNORE INTO league_games(league_id, game_id) VALUES(%s,%s)",
            (league_id, game_id),
        )
    conn.commit()


def ensure_defaults(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM game_types")
        count = (cur.fetchone() or [0])[0]
        if int(count) == 0:
            for name in ("Preseason", "Regular Season", "Tournament", "Exhibition"):
                cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 1))
        conn.commit()


def ensure_user(conn, email: str, name: str | None = None, password_hash: str | None = None) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        # Create a user if requested data provided; otherwise, fail clearly
        if not password_hash:
            raise RuntimeError(
                f"User {email!r} does not exist. Provide --create-user or a --password-hash to create it."
            )
        now = dt.datetime.now().isoformat()
        cur.execute(
            "INSERT INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
            (email, password_hash, name or email, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def ensure_game_type(conn, name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM game_types WHERE name=%s", (name,))
        r = cur.fetchone()
        if r:
            return int(r[0])
        cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 0))
        conn.commit()
        return int(cur.lastrowid)


def ensure_team(conn, user_id: int, name: str, is_external: int = 1) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM teams WHERE user_id=%s AND name=%s", (user_id, name))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (user_id, name, is_external, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def _fetch_and_store_team_logo(
    conn, team_mysql_id: int, tts_team_id: int, season_id: int, logo_dir: Optional[str]
) -> None:
    if not logo_dir:
        return
    # Fetch first reasonable image from the team schedule page
    try:
        from pathlib import Path as _Path

        import requests as _req

        from hmlib.time2score import util as tutil

        base = "https://stats.caha.timetoscore.com/display-schedule"
        params = {"team": int(tts_team_id), "league": "3", "stat_class": "1"}
        if season_id and season_id > 0:
            params["season"] = int(season_id)
        soup = tutil.get_html(base, params=params)
        # Heuristics: prefer images with 'logo' or within header; else first non-tracker image
        imgs = soup.find_all("img")
        if not imgs:
            return

        def _score_img(img) -> int:
            src = (img.get("src") or "").lower()
            score = 0
            if "logo" in src:
                score += 2
            if any(ext in src for ext in (".png", ".jpg", ".jpeg", ".gif")):
                score += 1
            return score

        best = max(imgs, key=_score_img)
        src = best.get("src")
        if not src:
            return
        if not src.startswith("http"):
            # Make absolute
            if src.startswith("/"):
                src = "https://stats.caha.timetoscore.com" + src
            else:
                src = "https://stats.caha.timetoscore.com/" + src
        resp = _req.get(src, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if resp.status_code != 200 or not resp.content:
            return
        # Write to logo_dir
        d = _Path(logo_dir)
        d.mkdir(parents=True, exist_ok=True)
        ext = ".png"
        ctype = resp.headers.get("Content-Type", "").lower()
        if "jpeg" in ctype or src.lower().endswith(".jpg") or src.lower().endswith(".jpeg"):
            ext = ".jpg"
        elif "gif" in ctype or src.lower().endswith(".gif"):
            ext = ".gif"
        dest = d / f"team{team_mysql_id}_{dt.datetime.now():%Y%m%d%H%M%S}{ext}"
        dest.write_bytes(resp.content)
        # Update DB path
        with conn.cursor() as cur:
            cur.execute("UPDATE teams SET logo_path=%s WHERE id=%s", (str(dest), team_mysql_id))
        conn.commit()
    except Exception:
        return


def upsert_hky_game(
    conn,
    *,
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[str],
    location: Optional[str],
    team1_score: Optional[int],
    team2_score: Optional[int],
    notes: Optional[str],
) -> int:
    # Try to find an existing entry by a stable tuple (user, teams, starts_at)
    with conn.cursor() as cur:
        if starts_at:
            cur.execute(
                "SELECT id FROM hky_games WHERE user_id=%s AND team1_id=%s AND team2_id=%s AND starts_at=%s",
                (user_id, team1_id, team2_id, starts_at),
            )
            r = cur.fetchone()
        else:
            r = None
        if r:
            gid = int(r[0])
            # Update scores/type/location/notes if provided
            cur.execute(
                """
                UPDATE hky_games
                SET game_type_id=COALESCE(%s, game_type_id),
                    location=COALESCE(%s, location),
                    team1_score=COALESCE(%s, team1_score),
                    team2_score=COALESCE(%s, team2_score),
                    is_final=CASE WHEN %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                    updated_at=%s,
                    notes=COALESCE(%s, notes)
                WHERE id=%s
                """,
                (
                    game_type_id,
                    location,
                    team1_score,
                    team2_score,
                    team1_score,
                    team2_score,
                    dt.datetime.now().isoformat(),
                    notes,
                    gid,
                ),
            )
            conn.commit()
            return gid
        # Insert new
        cur.execute(
            """
            INSERT INTO hky_games(user_id, team1_id, team2_id, game_type_id, starts_at, location,
                                  team1_score, team2_score, is_final, notes, created_at)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                user_id,
                team1_id,
                team2_id,
                game_type_id,
                starts_at,
                location,
                team1_score,
                team2_score,
                1 if (team1_score is not None and team2_score is not None) else 0,
                notes,
                dt.datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


@contextlib.contextmanager
def chdir(path: Path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


def parse_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int,)):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    # Strip shootout/OT markers like "3 SO" -> 3
    try:
        return int(s.split()[0])
    except Exception:
        return None


def _ensure_player(
    conn, user_id: int, team_id: int, name: str, jersey: Optional[str], position: Optional[str]
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
            (user_id, team_id, name),
        )
        r = cur.fetchone()
        if r:
            pid = int(r[0])
            # Opportunistically update jersey/position if provided
            if jersey or position:
                cur.execute(
                    "UPDATE players SET jersey_number=COALESCE(%s, jersey_number), position=COALESCE(%s, position) WHERE id=%s",
                    (jersey, position, pid),
                )
                conn.commit()
            return pid
        cur.execute(
            "INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
            (user_id, team_id, name, jersey, position, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def _import_game_by_id(
    conn,
    tts_dir: Path,
    game_id: int,
    user_id: int,
    league_id: Optional[int],
    team_filters: Optional[list[str]] = None,
) -> Optional[int]:
    """Import a single game by its time2score game id.

    Scrapes per-game stats to derive teams, scores, start time, location,
    rosters, and minimal player stats. Returns the hky_games.id or None.
    """
    # Scope scraper imports & cache
    with chdir(tts_dir):
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[2]
        tts_pkg_dir = repo_root / "hmlib" / "time2score"
        if str(tts_pkg_dir) not in _sys.path:
            _sys.path.insert(0, str(tts_pkg_dir))
        from hmlib.time2score import caha_lib
        from hmlib.time2score import util as tutil

        try:
            data = caha_lib.scrape_game_stats(int(game_id))
        except Exception:
            return None

        # Team names
        home_name = str(data.get("home", "")).strip()
        away_name = str(data.get("away", "")).strip()
        if team_filters:

            def norm(value: str) -> str:
                return value.lower().replace(" ", "")

            hn = norm(home_name)
            an = norm(away_name)
            if not any(norm(tf) in hn or norm(tf) in an for tf in team_filters if tf):
                return None
        # Scores
        t1_score = parse_int_or_none(data.get("homeGoals"))
        t2_score = parse_int_or_none(data.get("awayGoals"))
        # Time and location
        date_s = data.get("date") or ""
        time_s = data.get("time") or ""
        loc = (data.get("location") or "").strip() or None
        # Parse datetime; util.parse_game_time expects strings like 'Fri Oct 18' and '08:20 PM'
        starts_at = None
        try:
            starts_dt = tutil.parse_game_time(str(date_s), str(time_s), year=None)
            starts_at = starts_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            starts_at = None

        # Ensure teams
        team1_id = ensure_team(conn, user_id=user_id, name=home_name or "UNKNOWN", is_external=1)
        team2_id = ensure_team(conn, user_id=user_id, name=away_name or "UNKNOWN", is_external=1)

        # Upsert game
        gid = upsert_hky_game(
            conn,
            user_id=user_id,
            team1_id=team1_id,
            team2_id=team2_id,
            game_type_id=None,
            starts_at=starts_at,
            location=loc,
            team1_score=t1_score,
            team2_score=t2_score,
            notes=None,
        )

        # Map to league if set
        if league_id is not None:
            map_team_to_league(conn, league_id, team1_id)
            map_team_to_league(conn, league_id, team2_id)
            map_game_to_league(conn, league_id, gid)

        # Import rosters and minimal player stats
        def load_roster(prefix: str, team_id: int):
            rows = data.get(f"{prefix}Players") or []
            for row in rows:
                name = str(row.get("name") or "").strip()
                num = str(row.get("number") or "").strip() or None
                pos = str(row.get("position") or "").strip() or None
                if not name:
                    continue
                _ensure_player(conn, user_id, team_id, name, num, pos)

        load_roster("home", team1_id)
        load_roster("away", team2_id)

        # Aggregate goals/assists from scoring tables
        def incr(d: Dict[str, Dict[str, int]], who: str, key: str):
            rec = d.setdefault(who, {"goals": 0, "assists": 0, "pim": 0})
            rec[key] = rec.get(key, 0) + 1

        stats_by_player: Dict[str, Dict[str, int]] = {}
        for side in ("home", "away"):
            scoring = data.get(f"{side}Scoring") or []
            for srow in scoring:
                gname = str(srow.get("goal") or "").strip()
                a1 = str(srow.get("assist1") or "").strip()
                a2 = str(srow.get("assist2") or "").strip()
                if gname:
                    incr(stats_by_player, gname, "goals")
                if a1:
                    incr(stats_by_player, a1, "assists")
                if a2:
                    incr(stats_by_player, a2, "assists")
            # penalties minutes
            pens = data.get(f"{side}Penalties") or []
            for prow in pens:
                # Some sites list number not name in penalties; skip to avoid incorrect attribution
                continue

        # Write player_stats by resolving players by name within both teams
        def player_id_by_name(team_id: int, name: str) -> Optional[int]:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                    (user_id, team_id, name),
                )
                r = cur.fetchone()
                return int(r[0]) if r else None

        for pname, agg in stats_by_player.items():
            pid = player_id_by_name(team1_id, pname)
            team_ref = team1_id
            if pid is None:
                pid = player_id_by_name(team2_id, pname)
                team_ref = team2_id if pid is not None else team1_id
            if pid is None:
                # create a player placeholder under home team to store stats
                pid = _ensure_player(conn, user_id, team_ref, pname, None, None)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                    VALUES(%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE goals=VALUES(goals), assists=VALUES(assists)
                    """,
                    (
                        user_id,
                        team_ref,
                        gid,
                        pid,
                        int(agg.get("goals", 0)),
                        int(agg.get("assists", 0)),
                    ),
                )
            conn.commit()

        return gid


def main(argv: Optional[list[str]] = None) -> int:
    def log(msg: str) -> None:
        try:
            ts = dt.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {msg}", flush=True)
        except Exception:
            # Best-effort logging
            print(msg, flush=True)

    base_dir = Path(__file__).resolve().parent
    default_cfg = os.environ.get("HM_DB_CONFIG") or str(base_dir / "config.json")
    ap = argparse.ArgumentParser(description="Import time2score data into HockeyMOM WebApp DB")
    ap.add_argument("--config", default=default_cfg, help="Path to webapp DB config.json")
    ap.add_argument("--season", type=int, default=0, help="Season id to import (0=Current)")
    ap.add_argument(
        "--user-email", required=True, help="Webapp user email that will own imported data"
    )
    ap.add_argument("--user-name", default=None, help="Name for user creation if missing")
    ap.add_argument(
        "--password-hash", default=None, help="Password hash for creating user if missing"
    )
    ap.add_argument(
        "--create-user",
        action="store_true",
        help="Create user if missing (requires --password-hash)",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="Fetch/sync time2score data (seasons/divisions/teams/games)",
    )
    ap.add_argument(
        "--stats", action="store_true", help="Fetch game stats to enrich scores if missing"
    )
    ap.add_argument(
        "--division",
        dest="divisions",
        action="append",
        default=[],
        help=(
            "Only import divisions that match. Accepts multiple. "
            "Match is case-insensitive substring on division name (e.g., '12AA'), "
            "or numeric 'LEVEL' (e.g., '12') to include all conferences for that level, "
            "or 'LEVEL:CONF' to match exact level and conference (e.g., '12:1')."
        ),
    )
    ap.add_argument(
        "--tts-db-dir",
        default=str(base_dir / "instance" / "time2score_db"),
        help="Directory to hold the sqlite hockey_league.db for time2score",
    )
    ap.add_argument(
        "--list-seasons",
        action="store_true",
        help="List available seasons discovered via time2score and exit",
    )
    ap.add_argument(
        "--list-divisions",
        action="store_true",
        help="List divisions for the chosen season and exit",
    )
    # League grouping/sharing
    ap.add_argument(
        "--league-name",
        default=None,
        help="Name of the league grouping to attach imported data to (default: CAHA-<season>)",
    )
    ap.add_argument(
        "--league-owner-email", default=None, help="Owner of the league (defaults to --user-email)"
    )
    ap.add_argument(
        "--share-with",
        action="append",
        default=[],
        help="Emails to add as league viewers (repeatable)",
    )
    ap.add_argument(
        "--shared", action="store_true", help="Mark the league as shared (default: private)"
    )
    # Import by explicit game ids
    ap.add_argument(
        "--game-id",
        dest="game_ids",
        action="append",
        default=[],
        help="Import specific game id (repeatable)",
    )
    ap.add_argument(
        "--games-file", default=None, help="Path to file containing one game id per line"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Maximum number of games to import (for testing)"
    )
    ap.add_argument(
        "--team",
        dest="teams",
        action="append",
        default=[],
        help="Filter to games involving team name substring (repeatable)",
    )
    ap.add_argument(
        "--logo-dir",
        default=None,
        help="Directory to save team logos (optional). If set, importer fetches logos from team pages and updates teams.logo_path",
    )
    args = ap.parse_args(argv)

    # Startup summary
    summary = {
        "season": args.season,
        "divisions": args.divisions or [],
        "teams_filter": args.teams or [],
        "game_ids": len(args.game_ids or []),
        "games_file": bool(args.games_file),
        "league": args.league_name,
        "shared": args.shared,
        "logo_dir": args.logo_dir or "",
    }
    log(f"Starting import (summary: {summary})")

    db_cfg = load_db_cfg(args.config)
    log("Connecting to MySQL...")
    try:
        conn = connect_pymysql(db_cfg)
    except Exception:
        print(
            "[!] Failed to connect to DB. Ensure the webapp is installed and DB configured.",
            file=sys.stderr,
        )
        raise
    log("Connected.")

    ensure_defaults(conn)
    log("Ensured default game types.")
    ensure_league_schema(conn)
    log("Ensured league schema tables.")

    # Ensure user exists or create if requested
    try:
        user_id = ensure_user(
            conn,
            args.user_email,
            name=args.user_name or args.user_email,
            password_hash=(args.password_hash if args.create_user else None),
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    # Prepare time2score DB in an isolated working directory
    tts_dir = Path(args.tts_db_dir)
    with chdir(tts_dir):
        # Import scraper modules lazily to scope their working dir.
        # caha_lib imports `database` as a top-level module, so ensure the
        # package directory is on sys.path for that import to resolve.
        import sys as _sys

        repo_root = Path(__file__).resolve().parents[2]
        tts_pkg_dir = repo_root / "hmlib" / "time2score"
        if str(tts_pkg_dir) not in _sys.path:
            _sys.path.insert(0, str(tts_pkg_dir))
        from hmlib.time2score import caha_lib
        from hmlib.time2score import database as tdb_mod

        tdb = tdb_mod.Database()
        tdb.create_tables()

        season_id = int(args.season)
        # Ensure seasons table populated and normalize season=0 to latest numeric
        log("Loading seasons (may fetch remote)...")
        seasons_list = tdb.list_seasons()
        if args.sync or not seasons_list or args.list_seasons or args.list_divisions:
            caha_lib.sync_seasons(tdb)
            seasons_list = tdb.list_seasons()
        log(f"Seasons known: {len(seasons_list)}")
        if args.list_seasons:
            print("Seasons:")
            for s in sorted(seasons_list, key=lambda x: int(x["season_id"])):
                print(f"  {s['season_id']}: {s['name']}")
            return 0
        if season_id == 0 and seasons_list:
            nonzero = [s["season_id"] for s in seasons_list if int(s["season_id"]) > 0]
            if nonzero:
                season_id = max(nonzero)
        log(f"Using season: {season_id}")

        if args.list_divisions:
            # Determine target season for listing
            target_season = season_id
            if target_season == 0 and seasons_list:
                nonzero = [s["season_id"] for s in seasons_list if int(s["season_id"]) > 0]
                if nonzero:
                    target_season = max(nonzero)
            # Parse the main stats page to list division Schedules (not "Division Player Stats")
            from hmlib.time2score import util as tutil

            MAIN_STATS_URL = "https://stats.caha.timetoscore.com/display-stats"
            params = {"league": "3", "stat_class": "1"}
            if target_season and target_season > 0:
                params["season"] = str(target_season)
            log("Fetching schedule list from main stats page...")
            soup = tutil.get_html(MAIN_STATS_URL, params=params)
            print(f"Schedules for season {target_season}:")
            for a in soup.find_all("a", href=True):
                text = a.get_text(strip=True) or ""
                if "Schedule" not in text:
                    continue
                # Present the name without the trailing "Schedule"
                name = text.replace("Schedule", "").strip()
                print(f"  {name}")
            return 0

        if args.sync:
            # If importing specific divisions or game ids, avoid Division Player Stats pages
            if not args.divisions and not args.game_ids and not args.games_file:
                # Full season sync when no specific filters
                caha_lib.sync_divisions(tdb, season_id)
                caha_lib.sync_season_teams(tdb, season_id)

        # Build division filter set from args.divisions
        allowed_div_keys: set[Tuple[int, int]] | None = None
        if args.divisions:
            allowed_div_keys = set()
            all_divs = tdb.list_divisions()
            # Build helpers
            name_map: Dict[Tuple[int, int], str] = {}
            for d in all_divs:
                key = (int(d["division_id"]), int(d["conference_id"]))
                name_map[key] = str(d["name"] or "")
            for tok in args.divisions:
                s = str(tok).strip()
                if not s:
                    continue
                if ":" in s:
                    a, b = s.split(":", 1)
                    if a.isdigit() and b.isdigit():
                        allowed_div_keys.add((int(a), int(b)))
                        continue
                if s.isdigit():
                    level = int(s)
                    for (lvl, conf), _nm in name_map.items():
                        if lvl == level:
                            allowed_div_keys.add((lvl, conf))
                    continue
                # Substring match on name (case-insensitive)
                s_low = s.lower()
                for key, nm in name_map.items():
                    if s_low in nm.lower():
                        allowed_div_keys.add(key)

        # Build team map for the season; apply division filter if any
        teams = tdb.list_teams(f"season_id = {season_id}")
        if allowed_div_keys is not None:
            teams = [
                t
                for t in teams
                if (int(t["division_id"]), int(t["conference_id"])) in allowed_div_keys
            ]
        log(
            f"Teams in season {season_id}: total={len(tdb.list_teams(f'season_id = {season_id}'))}, filtered={len(teams)}"
        )

        # Upsert teams in webapp DB (and optionally fetch logos)
        ext_team_map: Dict[int, int] = {}
        for t in teams:
            mysql_tid = ensure_team(conn, user_id=user_id, name=t["name"], is_external=1)
            ext_team_map[int(t["team_id"])] = mysql_tid
            if args.logo_dir:
                try:
                    _fetch_and_store_team_logo(
                        conn, mysql_tid, int(t["team_id"]), season_id, args.logo_dir
                    )
                except Exception:
                    pass

        # Import games
        # Decide import source: explicit game ids, division schedules, or DB games
        explicit_game_ids: list[int] = []
        if args.games_file:
            try:
                for line in Path(args.games_file).read_text().splitlines():
                    s = line.strip()
                    if s and s.isdigit():
                        explicit_game_ids.append(int(s))
            except Exception:
                pass
        for s in args.game_ids:
            if s and str(s).isdigit():
                explicit_game_ids.append(int(s))

        # Determine league (optional)
        league_id: Optional[int] = None
        if args.league_name or args.shared or args.share_with:
            league_name = args.league_name or f"CAHA-{season_id or 'current'}"
            owner_email = args.league_owner_email or args.user_email
            owner_id = ensure_user(
                conn,
                owner_email,
                name=owner_email,
                password_hash=(args.password_hash if args.create_user else None),
            )
            league_id = ensure_league(
                conn,
                league_name,
                owner_id,
                args.shared,
                source="timetoscore",
                external_key=str(season_id),
            )
            # Add owner (admin) + primary user (editor)
            ensure_league_member(conn, league_id, owner_id, role="admin")
            ensure_league_member(conn, league_id, user_id, role="editor")
            # Add others
            for em in args.share_with:
                try:
                    uid = ensure_user(conn, em, name=em, password_hash=None)
                except RuntimeError:
                    # If the user doesn't exist and no password hash provided, skip adding
                    continue
                ensure_league_member(conn, league_id, uid, role="viewer")
            log(f"Using league '{league_name}' (id={league_id}) shared={bool(args.shared)}")

        # 1) Import by explicit game ids if provided
        if explicit_game_ids:
            count = 0
            log(f"Importing {len(explicit_game_ids)} explicit game ids...")
            for gid in explicit_game_ids:
                log(f"- Importing game_id={gid}")
                _import_game_by_id(conn, tts_dir, gid, user_id, league_id, team_filters=args.teams)
                count += 1
                if args.limit is not None and count >= args.limit:
                    break
        else:
            # 2) If divisions requested, discover games from division schedule links on main stats page
            if args.divisions:
                from hmlib.time2score import util as tutil

                MAIN_STATS_URL = "https://stats.caha.timetoscore.com/display-stats"
                params = {"league": "3", "stat_class": "1"}
                if season_id and season_id > 0:
                    params["season"] = str(season_id)
                log("Fetching main stats page for schedule discovery...")
                soup = tutil.get_html(MAIN_STATS_URL, params=params)
                # Find schedule links like "12U AA Schedule"
                sched_links = []

                def _norm_div_str(s: str) -> str:
                    s2 = (s or "").upper().replace("SCHEDULE", "")
                    s2 = s2.replace(" ", "").replace("U", "")
                    return s2.strip()

                for a in soup.find_all("a", href=True):
                    text = a.get_text(strip=True) or ""
                    if "Schedule" not in text:
                        continue
                    # Normalize e.g., "12U AA" so tokens like "12AA" match
                    norm = _norm_div_str(text)
                    want = False
                    for tok in args.divisions:
                        if not tok:
                            continue
                        tnorm = _norm_div_str(tok)
                        if not tnorm:
                            continue
                        # Require exact match after normalization so
                        # "12U A" (-> 12A) does not match "12U AA" (-> 12AA)
                        if tnorm == norm:
                            want = True
                            break
                    if want:
                        sched_links.append(a["href"])
                log(
                    f"Matched {len(sched_links)} division schedule links for filters {args.divisions}"
                )
                # For each schedule link, parse game ids from the table
                import io as _io

                import pandas as _pd

                discovered: set[int] = set()
                for href in sched_links:
                    log(f"Reading schedule: {href}")
                    try:
                        soup2 = tutil.get_html(
                            href
                            if href.startswith("http")
                            else ("https://stats.caha.timetoscore.com/" + href)
                        )
                        tables = soup2.find_all("table")
                        if not tables:
                            log("  (no tables found on schedule page)")
                            continue
                        import re as _re

                        def _val_to_text(v):
                            # pandas extract_links returns (text, link) for cells with <a>
                            if isinstance(v, tuple) and len(v) > 0:
                                return str(v[0])
                            return str(v)

                        def _parse_table_bs4(tbl_obj):
                            # Returns (headers, rows) where rows is list of list[str]
                            all_rows = []
                            for tr in tbl_obj.find_all("tr"):
                                tds = tr.find_all(["td", "th"])
                                if not tds:
                                    continue
                                row = []
                                for td in tds:
                                    a = td.find("a")
                                    if a and a.get("href"):
                                        row.append(a.get_text(strip=True))
                                    else:
                                        row.append(td.get_text(strip=True))
                                all_rows.append(row)

                            # Find the first row that has same number of columns (>1) as the next row
                            header_idx = None
                            for i in range(len(all_rows) - 1):
                                if len(all_rows[i]) > 1 and len(all_rows[i]) == len(
                                    all_rows[i + 1]
                                ):
                                    header_idx = i
                                    break

                            if header_idx is None:
                                # Default to no headers
                                headers = []
                                rows = all_rows
                            else:
                                headers = all_rows[header_idx]
                                rows = all_rows[header_idx + 1 :]
                                # sanity check
                                rows = [r for r in rows if len(r) == len(headers)]

                            return headers, rows

                        for tbl in tables:
                            try:
                                df = _pd.read_html(_io.StringIO(str(tbl)), extract_links="body")[0]
                            except Exception:
                                # Fallback: try BeautifulSoup parsing
                                try:
                                    headers, rows = _parse_table_bs4(tbl)
                                    if headers:
                                        log(
                                            f"Schedule table (bs4) headers: {headers} rows={len(rows)}"
                                        )
                                    # Map teams if possible (Away/Home/Team)
                                    if headers:
                                        lowers = [h.strip().lower() for h in headers]
                                        away_idx = lowers.index("away") if "away" in lowers else -1
                                        home_idx = lowers.index("home") if "home" in lowers else -1
                                        team_idx = lowers.index("team") if "team" in lowers else -1
                                        if away_idx >= 0 and home_idx >= 0:
                                            for r in rows:
                                                try:
                                                    away_cell = (
                                                        r[away_idx] if away_idx < len(r) else ""
                                                    )
                                                    home_cell = (
                                                        r[home_idx] if home_idx < len(r) else ""
                                                    )
                                                    away_name = (
                                                        away_cell[0]
                                                        if isinstance(away_cell, tuple)
                                                        else str(away_cell or "")
                                                    )
                                                    home_name = (
                                                        home_cell[0]
                                                        if isinstance(home_cell, tuple)
                                                        else str(home_cell or "")
                                                    )
                                                    if away_name:
                                                        tid = ensure_team(
                                                            conn,
                                                            user_id=user_id,
                                                            name=str(away_name).strip(),
                                                            is_external=1,
                                                        )
                                                        if league_id is not None:
                                                            map_team_to_league(conn, league_id, tid)
                                                    if home_name:
                                                        tid = ensure_team(
                                                            conn,
                                                            user_id=user_id,
                                                            name=str(home_name).strip(),
                                                            is_external=1,
                                                        )
                                                        if league_id is not None:
                                                            map_team_to_league(conn, league_id, tid)
                                                except Exception:
                                                    continue
                                        if team_idx >= 0:
                                            added = 0
                                            for r in rows:
                                                try:
                                                    cell = r[team_idx] if team_idx < len(r) else ""
                                                    if isinstance(cell, tuple):
                                                        name, href = cell[0], cell[1]
                                                    else:
                                                        name, href = str(cell or ""), None
                                                    name = (name or "").strip()
                                                    if not name:
                                                        continue
                                                    tid_mysql = ensure_team(
                                                        conn,
                                                        user_id=user_id,
                                                        name=name,
                                                        is_external=1,
                                                    )
                                                    if league_id is not None:
                                                        map_team_to_league(
                                                            conn, league_id, tid_mysql
                                                        )
                                                    if args.logo_dir and href:
                                                        t_id = tutil.get_value_from_link(
                                                            str(href), "team"
                                                        )
                                                        if t_id and str(t_id).isdigit():
                                                            try:
                                                                _fetch_and_store_team_logo(
                                                                    conn,
                                                                    tid_mysql,
                                                                    int(t_id),
                                                                    season_id,
                                                                    args.logo_dir,
                                                                )
                                                            except Exception:
                                                                pass
                                                    added += 1
                                                except Exception:
                                                    continue
                                            if added:
                                                log(
                                                    f"  + Imported {added} teams from 'Team' table (bs4)"
                                                )
                                    # Extract game ids from first column if header mentions Game or first col looks numeric
                                    game_idx = -1
                                    if headers:
                                        lowers = [h.strip().lower() for h in headers]
                                        if "game" in lowers:
                                            game_idx = lowers.index("game")
                                    add_cnt = 0
                                    for r in rows:
                                        try:
                                            cell = r[game_idx if game_idx >= 0 else 0]
                                            s = _re.sub(r"[^0-9]", "", str(cell or ""))
                                            if s.isdigit():
                                                discovered.add(int(s))
                                                add_cnt += 1
                                        except Exception:
                                            continue
                                    if add_cnt:
                                        log(f"  + Found {add_cnt} game ids from table (bs4)")
                                except Exception:
                                    continue
                                continue
                            try:
                                log(f"Schedule table columns: {list(df.columns)} rows={len(df)}")
                            except Exception:
                                pass
                            # Map teams from Away/Home or Team columns (create for user; map to league if present)
                            away_col = None
                            home_col = None
                            team_col = None
                            for col in list(df.columns):
                                lc = str(col).strip().lower()
                                if lc == "away":
                                    away_col = col
                                elif lc == "home":
                                    home_col = col
                                elif lc == "team":
                                    team_col = col
                            if away_col is not None and home_col is not None:
                                try:
                                    for _, row in df.iterrows():
                                        away_name = _val_to_text(row.get(away_col, "")).strip()
                                        home_name = _val_to_text(row.get(home_col, "")).strip()
                                        if away_name:
                                            tid = ensure_team(
                                                conn, user_id=user_id, name=away_name, is_external=1
                                            )
                                            if league_id is not None:
                                                map_team_to_league(conn, league_id, tid)
                                        if home_name:
                                            tid = ensure_team(
                                                conn, user_id=user_id, name=home_name, is_external=1
                                            )
                                            if league_id is not None:
                                                map_team_to_league(conn, league_id, tid)
                                except Exception:
                                    pass
                            if team_col is not None:
                                try:
                                    added = 0
                                    ser = df[team_col].apply(_pd.Series)
                                    names = ser[0]
                                    hrefs = ser[1] if 1 in ser.columns else None
                                    for idx, nm in names.items():
                                        name = str(nm or "").strip()
                                        if not name:
                                            continue
                                        tid_mysql = ensure_team(
                                            conn, user_id=user_id, name=name, is_external=1
                                        )
                                        if league_id is not None:
                                            map_team_to_league(conn, league_id, tid_mysql)
                                        if args.logo_dir and hrefs is not None:
                                            lnk = hrefs.get(idx)
                                            if lnk:
                                                t_id = tutil.get_value_from_link(str(lnk), "team")
                                                if t_id and str(t_id).isdigit():
                                                    try:
                                                        _fetch_and_store_team_logo(
                                                            conn,
                                                            tid_mysql,
                                                            int(t_id),
                                                            season_id,
                                                            args.logo_dir,
                                                        )
                                                    except Exception:
                                                        pass
                                        added += 1
                                    if added:
                                        log(
                                            f"  + Imported {added} teams from 'Team' table (pandas)"
                                        )
                                except Exception:
                                    pass
                            # Extract from "Game" column (preferred)
                            game_col = None
                            for col in list(df.columns):
                                lc = str(col).strip().lower()
                                if lc == "game":
                                    game_col = col
                                    break
                            # Fallback to first column only if we don't find 'Game'
                            if game_col is None and len(df.columns) >= 1:
                                game_col = df.columns[0]
                            # Pull display text; for extract_links pandas returns (text, link)
                            try:
                                cnt_before = len(discovered)
                                for _, row in df.iterrows():
                                    raw = row.get(game_col, "")
                                    txt = _val_to_text(raw)
                                    s = _re.sub(r"[^0-9]", "", str(txt or ""))
                                    if s.isdigit():
                                        discovered.add(int(s))
                                if len(discovered) > cnt_before:
                                    log(
                                        f"  + Found {len(discovered) - cnt_before} game ids from 'Game' column"
                                    )
                            except Exception:
                                pass
                            # Supplement from Scoresheet/Box Score links (secondary)
                            extra_before = len(discovered)
                            for col in list(df.columns):
                                if "Scoresheet" in str(col) or "Box Score" in str(col):
                                    try:
                                        ser = df[col].apply(_pd.Series)
                                        links = ser[1].dropna().tolist()
                                        for lnk in links:
                                            gid = tutil.get_value_from_link(lnk, "game_id")
                                            if gid and gid.isdigit():
                                                discovered.add(int(gid))
                                    except Exception:
                                        continue
                            if len(discovered) > extra_before:
                                log(
                                    f"  + Found {len(discovered) - extra_before} game ids from Scoresheet/Box Score links"
                                )
                    except Exception:
                        continue
                log(f"Discovered {len(discovered)} game ids from schedules")
                count = 0
                for gid in sorted(discovered):
                    if args.limit is not None and count >= args.limit:
                        break
                    log(f"- Importing game_id={gid}")
                    _import_game_by_id(
                        conn, tts_dir, gid, user_id, league_id, team_filters=args.teams
                    )
                    count += 1
            else:
                # 3) Fallback: use DB games previously scraped
                log("Falling back to DB games previously scraped...")
                games = tdb.list_games(f"season_id = {season_id}")
                if allowed_div_keys is not None:
                    games = [
                        g
                        for g in games
                        if (int(g["division_id"]), int(g["conference_id"])) in allowed_div_keys
                    ]
                count = 0
                for g in games:
                    if args.limit is not None and count >= args.limit:
                        break
                    gid = int(g["game_id"]) if g.get("game_id") is not None else None
                    if gid is None:
                        continue
                    log(f"- Importing game_id={gid}")
                    _import_game_by_id(
                        conn, tts_dir, gid, user_id, league_id, team_filters=args.teams
                    )
                    count += 1
        # Done
    log("Import complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

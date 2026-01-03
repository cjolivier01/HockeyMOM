#!/usr/bin/env python3
"""Import TimeToScore data into the HockeyMOM webapp DB (no local sqlite cache).

This script scrapes TimeToScore directly via `hmlib.time2score` and upserts:
- teams (as external teams owned by the specified user)
- games (hky_games)
- players + player_stats (goals/assists derived from TimeToScore game pages)

By default it targets CAHA youth (league=3). SharksIce (adult, league=1) is also supported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


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


def ensure_defaults(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM game_types")
        count = int((cur.fetchone() or [0])[0])
        if count == 0:
            for name in ("Preseason", "Regular Season", "Tournament", "Exhibition"):
                cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 1))
    conn.commit()


def ensure_league_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leagues (
              id INT AUTO_INCREMENT PRIMARY KEY,
              name VARCHAR(255) UNIQUE NOT NULL,
              owner_user_id INT NOT NULL,
              is_shared TINYINT(1) NOT NULL DEFAULT 0,
              is_public TINYINT(1) NOT NULL DEFAULT 0,
              source VARCHAR(64) NULL,
              external_key VARCHAR(255) NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(owner_user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Best-effort migration for older installs.
        for col_ddl in [
            "is_public TINYINT(1) NOT NULL DEFAULT 0",
        ]:
            col = col_ddl.split(" ", 1)[0]
            try:
                cur.execute("SHOW COLUMNS FROM leagues LIKE %s", (col,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(f"ALTER TABLE leagues ADD COLUMN {col_ddl}")
            except Exception:
                try:
                    cur.execute(f"ALTER TABLE leagues ADD COLUMN {col_ddl}")
                except Exception:
                    pass
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
              division_name VARCHAR(255) NULL,
              division_id INT NULL,
              conference_id INT NULL,
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
              division_name VARCHAR(255) NULL,
              division_id INT NULL,
              conference_id INT NULL,
              UNIQUE KEY uniq_league_game (league_id, game_id),
              INDEX(league_id), INDEX(game_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Best-effort migration for older installs.
        for table in ("league_teams", "league_games"):
            for col_ddl in [
                "division_name VARCHAR(255) NULL",
                "division_id INT NULL",
                "conference_id INT NULL",
            ]:
                col = col_ddl.split(" ", 1)[0]
                try:
                    cur.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (col,))
                    exists = cur.fetchone()
                    if not exists:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_ddl}")
                except Exception:
                    try:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_ddl}")
                    except Exception:
                        pass
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
            league_id = int(r[0])
            cur.execute(
                "UPDATE leagues SET is_shared=%s, source=%s, external_key=%s, updated_at=%s WHERE id=%s",
                (1 if is_shared else 0, source, external_key, dt.datetime.now().isoformat(), league_id),
            )
            conn.commit()
            return league_id
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
            """
            INSERT INTO league_teams(league_id, team_id, division_name, division_id, conference_id)
            VALUES(%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              division_name=COALESCE(VALUES(division_name), division_name),
              division_id=COALESCE(VALUES(division_id), division_id),
              conference_id=COALESCE(VALUES(conference_id), conference_id)
            """,
            (league_id, team_id, None, None, None),
        )
    conn.commit()


def map_team_to_league_with_division(
    conn,
    *,
    league_id: int,
    team_id: int,
    division_name: Optional[str],
    division_id: Optional[int],
    conference_id: Optional[int],
) -> None:
    dn = (division_name or "").strip() or None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO league_teams(league_id, team_id, division_name, division_id, conference_id)
            VALUES(%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              division_name=COALESCE(VALUES(division_name), division_name),
              division_id=COALESCE(VALUES(division_id), division_id),
              conference_id=COALESCE(VALUES(conference_id), conference_id)
            """,
            (league_id, team_id, dn, division_id, conference_id),
        )
    conn.commit()


def map_game_to_league(conn, league_id: int, game_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id)
            VALUES(%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              division_name=COALESCE(VALUES(division_name), division_name),
              division_id=COALESCE(VALUES(division_id), division_id),
              conference_id=COALESCE(VALUES(conference_id), conference_id)
            """,
            (league_id, game_id, None, None, None),
        )
    conn.commit()


def map_game_to_league_with_division(
    conn,
    *,
    league_id: int,
    game_id: int,
    division_name: Optional[str],
    division_id: Optional[int],
    conference_id: Optional[int],
) -> None:
    dn = (division_name or "").strip() or None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id)
            VALUES(%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              division_name=COALESCE(VALUES(division_name), division_name),
              division_id=COALESCE(VALUES(division_id), division_id),
              conference_id=COALESCE(VALUES(conference_id), conference_id)
            """,
            (league_id, game_id, dn, division_id, conference_id),
        )
    conn.commit()


def ensure_user(conn, email: str, name: str | None = None, password_hash: str | None = None) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm:
        raise RuntimeError("user email is required")
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (email_norm,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        if not password_hash:
            raise RuntimeError(
                f"User {email_norm!r} does not exist; pass --create-user and --password-hash to create it"
            )
        cur.execute(
            "INSERT INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
            (email_norm, password_hash, name or email_norm, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def ensure_team(conn, user_id: int, name: str, *, is_external: bool = True) -> int:
    nm = (name or "").strip()
    if not nm:
        nm = "UNKNOWN"
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM teams WHERE user_id=%s AND name=%s", (user_id, nm))
        row = cur.fetchone()
        if row:
            tid = int(row[0])
            cur.execute(
                "UPDATE teams SET is_external=%s, updated_at=%s WHERE id=%s",
                (1 if is_external else 0, dt.datetime.now().isoformat(), tid),
            )
            conn.commit()
            return tid
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (user_id, nm, 1 if is_external else 0, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def _download_logo_bytes(url: str) -> tuple[bytes, Optional[str]]:
    import requests

    resp = requests.get(
        url,
        timeout=float(os.environ.get("HM_T2S_HTTP_TIMEOUT", "30")),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type")


def _guess_ext(url: str, content_type: Optional[str]) -> str:
    path = urlparse(url).path
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
        if path.lower().endswith(ext):
            return ext
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "gif" in ct:
        return ".gif"
    if "webp" in ct:
        return ".webp"
    if "svg" in ct:
        return ".svg"
    return ".jpg"


def _ensure_team_logo(
    conn,
    *,
    team_db_id: int,
    team_owner_user_id: int,
    source: str,
    season_id: int,
    tts_team_id: int,
    logo_dir: Path,
    replace: bool,
    tts_direct,
) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT logo_path FROM teams WHERE id=%s", (team_db_id,))
            row = cur.fetchone()
        existing = str(row[0]) if row and row[0] else ""
    except Exception:
        existing = ""

    if existing and not replace:
        return

    try:
        url = tts_direct.scrape_team_logo_url(str(source), season_id=int(season_id), team_id=int(tts_team_id))
    except Exception:
        url = None
    if not url:
        return

    logo_dir.mkdir(parents=True, exist_ok=True)
    data, content_type = _download_logo_bytes(url)
    ext = _guess_ext(url, content_type)
    dest = logo_dir / f"t2s_{source}_season{int(season_id)}_team{int(tts_team_id)}{ext}"
    if not dest.exists() or replace:
        dest.write_bytes(data)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE teams SET logo_path=%s, updated_at=%s WHERE id=%s AND user_id=%s",
            (str(dest), dt.datetime.now().isoformat(), team_db_id, team_owner_user_id),
        )
    conn.commit()


def _cleanup_numeric_named_players(conn, *, user_id: int, team_id: int) -> int:
    """Fix bogus players created from numeric scorer ids (e.g. name='88').

    Migrate any player_stats rows from bogus numeric-name players to the real player
    matching jersey_number, then delete the bogus player records if unused.
    """
    moved = 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM players WHERE user_id=%s AND team_id=%s AND name REGEXP '^[0-9]+$'",
            (user_id, team_id),
        )
        bogus = [(int(r[0]), str(r[1])) for r in (cur.fetchall() or [])]
    for bogus_id, bogus_name in bogus:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM players
                WHERE user_id=%s AND team_id=%s AND jersey_number=%s AND name NOT REGEXP '^[0-9]+$'
                LIMIT 1
                """,
                (user_id, team_id, bogus_name),
            )
            row = cur.fetchone()
        if not row:
            continue
        real_id = int(row[0])
        with conn.cursor() as cur:
            cur.execute(
                "SELECT game_id FROM player_stats WHERE player_id=%s",
                (bogus_id,),
            )
            game_ids = [int(r[0]) for r in (cur.fetchall() or [])]
        for gid in game_ids:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM player_stats WHERE game_id=%s AND player_id=%s",
                    (gid, real_id),
                )
                exists = cur.fetchone()
                if exists:
                    cur.execute("DELETE FROM player_stats WHERE game_id=%s AND player_id=%s", (gid, bogus_id))
                else:
                    cur.execute(
                        "UPDATE player_stats SET player_id=%s WHERE game_id=%s AND player_id=%s",
                        (real_id, gid, bogus_id),
                    )
                    moved += 1
            conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM player_stats WHERE player_id=%s", (bogus_id,))
            cnt = int((cur.fetchone() or [0])[0])
            if cnt == 0:
                cur.execute("DELETE FROM players WHERE id=%s", (bogus_id,))
                conn.commit()
    return moved


def upsert_hky_game(
    conn,
    *,
    user_id: int,
    team1_id: int,
    team2_id: int,
    starts_at: Optional[str],
    location: Optional[str],
    team1_score: Optional[int],
    team2_score: Optional[int],
    replace: bool,
    notes: Optional[str] = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, team1_score, team2_score
            FROM hky_games
            WHERE user_id=%s AND team1_id=%s AND team2_id=%s AND starts_at <=> %s
            LIMIT 1
            """,
            (user_id, team1_id, team2_id, starts_at),
        )
        row = cur.fetchone()
        if row:
            gid, old1, old2 = int(row[0]), row[1], row[2]
            new1 = team1_score if (replace or old1 is None) else old1
            new2 = team2_score if (replace or old2 is None) else old2
            cur.execute(
                """
                UPDATE hky_games
                SET location=COALESCE(%s, location),
                    team1_score=%s,
                    team2_score=%s,
                    is_final=%s,
                    notes=COALESCE(%s, notes),
                    updated_at=%s
                WHERE id=%s
                """,
                (
                    location,
                    new1,
                    new2,
                    1 if (new1 is not None and new2 is not None) else 0,
                    notes,
                    dt.datetime.now().isoformat(),
                    gid,
                ),
            )
            conn.commit()
            return gid

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
                None,
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


def ensure_player(
    conn, *, user_id: int, team_id: int, name: str, jersey: Optional[str], position: Optional[str]
) -> int:
    nm = (name or "").strip()
    if not nm:
        nm = "UNKNOWN"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, jersey_number, position FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
            (user_id, team_id, nm),
        )
        row = cur.fetchone()
        if row:
            pid, old_num, old_pos = int(row[0]), row[1], row[2]
            if jersey and not old_num:
                cur.execute("UPDATE players SET jersey_number=%s WHERE id=%s", (jersey, pid))
            if position and not old_pos:
                cur.execute("UPDATE players SET position=%s WHERE id=%s", (position, pid))
            conn.commit()
            return pid
        cur.execute(
            "INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
            (user_id, team_id, nm, jersey, position, dt.datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)


def parse_starts_at(source: str, *, stats: dict[str, Any], fallback: Optional[dict[str, Any]]) -> Optional[str]:
    from hmlib.time2score import util as tutil

    date_s = str(stats.get("date") or "").strip()
    time_s = str(stats.get("time") or "").strip()
    if date_s and time_s:
        try:
            dt_val = tutil.parse_game_time(date_s, time_s, year=None)
            return dt_val.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            if source == "sharksice":
                try:
                    d = dt.datetime.strptime(date_s, "%A, %B %d, %Y").date()
                    t = dt.datetime.strptime(time_s, "%I:%M %p").time()
                    return dt.datetime.combine(d, t).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
    st = (fallback or {}).get("start_time")
    if isinstance(st, dt.datetime):
        return st.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(st, str) and st.strip():
        return st.strip()
    return None


def main(argv: Optional[list[str]] = None) -> int:
    base_dir = Path(__file__).resolve().parent
    default_cfg = os.environ.get("HM_DB_CONFIG") or str(base_dir / "config.json")
    ap = argparse.ArgumentParser(description="Import TimeToScore into HockeyMOM webapp DB (no sqlite cache)")
    ap.add_argument("--config", default=default_cfg, help="Path to webapp DB config.json")
    ap.add_argument("--source", choices=("caha", "sharksice"), default="caha", help="TimeToScore site")
    ap.add_argument("--season", type=int, default=0, help="Season id (0 = current/latest)")
    ap.add_argument("--list-seasons", action="store_true", help="List seasons and exit")
    ap.add_argument("--list-divisions", action="store_true", help="List divisions for season and exit")

    ap.add_argument("--user-email", required=True, help="Webapp user email that will own imported data")
    ap.add_argument("--user-name", default=None, help="Name for user creation if missing")
    ap.add_argument("--password-hash", default=None, help="Password hash for creating user if missing")
    ap.add_argument("--create-user", action="store_true", help="Create user if missing (requires --password-hash)")

    ap.add_argument("--replace", action="store_true", help="Overwrite existing scores/player_stats")
    ap.add_argument("--no-import-logos", action="store_true", help="Skip downloading and saving team logos")
    ap.add_argument(
        "--logo-dir",
        default=None,
        help="Directory to store downloaded logos (default: /opt/hm-webapp/app/instance/uploads/team_logos if present)",
    )
    ap.add_argument(
        "--no-cleanup-bogus-players",
        action="store_true",
        help="Skip cleaning bogus numeric-name players created from score sheets",
    )
    ap.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run DB cleanup (no schedule/game scraping).",
    )
    ap.add_argument("--division", dest="divisions", action="append", default=[], help="Division filter token (repeatable)")
    ap.add_argument("--team", dest="teams", action="append", default=[], help="Team substring filter (repeatable)")
    ap.add_argument("--game-id", dest="game_ids", action="append", default=[], help="Import specific game id (repeatable)")
    ap.add_argument("--games-file", default=None, help="File containing one game id per line")
    ap.add_argument("--limit", type=int, default=None, help="Max games to import (for testing)")

    ap.add_argument(
        "--league-name",
        default=None,
        help="League name to import into (default: same as --source; created if missing)",
    )
    ap.add_argument("--league-owner-email", default=None, help="Owner of the league (defaults to --user-email)")
    ap.add_argument("--shared", action="store_true", help="Mark the league as shared")
    ap.add_argument("--share-with", action="append", default=[], help="Emails to add as league viewers (repeatable)")

    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from hmlib.time2score import direct as tts_direct
    from hmlib.time2score import normalize as tts_norm

    log(f"Connecting to DB via config: {args.config}")
    conn = connect_pymysql(load_db_cfg(args.config))
    ensure_defaults(conn)
    ensure_league_schema(conn)

    logo_dir = None
    if not args.no_import_logos:
        if args.logo_dir:
            logo_dir = Path(str(args.logo_dir)).expanduser()
        else:
            preferred = Path("/opt/hm-webapp/app/instance/uploads/team_logos")
            logo_dir = preferred if preferred.exists() else (base_dir / "instance" / "uploads" / "team_logos")

    log(f"Resolving user: {args.user_email}")
    user_id = ensure_user(
        conn,
        args.user_email,
        name=args.user_name or args.user_email,
        password_hash=(args.password_hash if args.create_user else None),
    )

    seasons = tts_direct.list_seasons(args.source)
    if args.list_seasons:
        for name, sid in sorted(seasons.items(), key=lambda kv: int(kv[1])):
            print(f"{sid}\t{name}")
        return 0

    season_id = int(args.season) or tts_direct.pick_current_season_id(args.source)
    log(f"Using source={args.source} season_id={season_id}")
    divs = tts_direct.list_divisions(args.source, season_id=season_id)
    if args.list_divisions:
        for d in sorted(divs, key=lambda x: (int(x.division_id), int(x.conference_id), x.name)):
            print(f"{d.division_id}:{d.conference_id}\t{d.name}\tteams={len(d.teams)}")
        return 0

    # Used to disambiguate team names that appear in multiple divisions (common on CAHA).
    team_name_counts: dict[str, int] = {}
    for d in divs:
        for t in d.teams:
            nm = str((t or {}).get("name") or "").strip()
            if nm:
                k = nm.lower()
                team_name_counts[k] = team_name_counts.get(k, 0) + 1

    def canonical_team_name(name: str, division_name: Optional[str]) -> str:
        nm = str(name or "").strip() or "UNKNOWN"
        dn = (division_name or "").strip()
        if dn and team_name_counts.get(nm.lower(), 0) > 1:
            return f"{nm} ({dn})"
        return nm

    # Map (team_name_lower, division_name_lower) -> team_id for logo retrieval.
    tts_team_id_by_name_div: dict[tuple[str, str], int] = {}
    tts_team_ids_by_name: dict[str, list[int]] = {}
    for d in divs:
        dn = str(d.name or "").strip()
        for t in d.teams:
            nm = str((t or {}).get("name") or "").strip()
            tid = (t or {}).get("id")
            if not nm or tid is None:
                continue
            try:
                tid_i = int(tid)
            except Exception:
                continue
            tts_team_id_by_name_div[(nm.lower(), dn.lower())] = tid_i
            tts_team_ids_by_name.setdefault(nm.lower(), []).append(tid_i)

    def resolve_tts_team_id(name: str, division_name: Optional[str]) -> Optional[int]:
        nm = str(name or "").strip().lower()
        dn = str(division_name or "").strip().lower()
        if nm and dn and (nm, dn) in tts_team_id_by_name_div:
            return int(tts_team_id_by_name_div[(nm, dn)])
        ids = tts_team_ids_by_name.get(nm) or []
        if len(ids) == 1:
            return int(ids[0])
        return None

    # Division filter resolution
    allowed_divs: Optional[set[tuple[int, int]]] = None
    if args.divisions:
        allowed_divs = set()
        name_map = {(d.division_id, d.conference_id): d.name for d in divs}
        for tok in args.divisions:
            s = str(tok).strip()
            if not s:
                continue
            if ":" in s:
                a, b = s.split(":", 1)
                if a.isdigit() and b.isdigit():
                    allowed_divs.add((int(a), int(b)))
                    continue
            if s.isdigit():
                lvl = int(s)
                for (d_id, conf), _nm in name_map.items():
                    if int(d_id) == lvl:
                        allowed_divs.add((int(d_id), int(conf)))
                continue
            s_low = s.lower()
            for key, nm in name_map.items():
                if s_low in (nm or "").lower():
                    allowed_divs.add(key)

    # Always import into a league (create if needed).
    league_name = str(args.league_name or args.source).strip() or str(args.source)
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
        bool(args.shared),
        source="timetoscore",
        external_key=f"{args.source}:{season_id}",
    )
    ensure_league_member(conn, league_id, owner_id, role="admin")
    ensure_league_member(conn, league_id, user_id, role="editor")
    for em in args.share_with:
        try:
            uid = ensure_user(conn, em, name=em, password_hash=None)
        except RuntimeError:
            continue
        ensure_league_member(conn, league_id, uid, role="viewer")

    if args.cleanup_only:
        game_ids = []
        fallback_by_gid = {}
    else:
        # Build explicit game list if present
        explicit_game_ids: list[int] = []
        if args.games_file:
            for line in Path(args.games_file).read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.isdigit():
                    explicit_game_ids.append(int(s))
        for s in args.game_ids:
            if str(s).strip().isdigit():
                explicit_game_ids.append(int(s))

        if explicit_game_ids:
            game_ids = list(explicit_game_ids)
            fallback_by_gid = {}
        else:
            log("Discovering game ids from team schedules...")
            fallback_by_gid = tts_direct.iter_season_games(
                args.source,
                season_id=season_id,
                divisions=sorted(allowed_divs) if allowed_divs is not None else None,
                team_name_substrings=args.teams,
            )
            game_ids = sorted(fallback_by_gid.keys())

    total = len(game_ids)
    log(f"Importing games: total={total} replace={bool(args.replace)}")

    count = 0
    skipped = 0
    cleaned_team_ids: set[int] = set()
    started = time.time()
    for gid in game_ids:
        if args.limit is not None and count >= int(args.limit):
            break
        fb = fallback_by_gid.get(int(gid))
        try:
            stats = tts_direct.scrape_game_stats(args.source, game_id=int(gid), season_id=season_id)
        except Exception:
            skipped += 1
            continue

        home_name = str(stats.get("home") or "").strip() or str((fb or {}).get("home") or "").strip()
        away_name = str(stats.get("away") or "").strip() or str((fb or {}).get("away") or "").strip()
        if not home_name or not away_name:
            skipped += 1
            continue

        division_name = str((fb or {}).get("division_name") or "").strip() or None
        division_id = None
        conference_id = None
        try:
            division_id = int((fb or {}).get("division_id")) if (fb or {}).get("division_id") is not None else None
        except Exception:
            division_id = None
        try:
            conference_id = (
                int((fb or {}).get("conference_id")) if (fb or {}).get("conference_id") is not None else None
            )
        except Exception:
            conference_id = None

        home_name_raw = str(home_name or "").strip()
        away_name_raw = str(away_name or "").strip()
        home_name = canonical_team_name(home_name_raw, division_name)
        away_name = canonical_team_name(away_name_raw, division_name)

        starts_at = parse_starts_at(args.source, stats=stats, fallback=fb)
        location = str(stats.get("location") or "").strip() or str((fb or {}).get("rink") or "").strip() or None
        t1_score = tts_norm.parse_int_or_none(stats.get("homeGoals"))
        t2_score = tts_norm.parse_int_or_none(stats.get("awayGoals"))

        team1_id = ensure_team(conn, user_id, home_name, is_external=True)
        team2_id = ensure_team(conn, user_id, away_name, is_external=True)
        if logo_dir is not None:
            try:
                tts_home_id = resolve_tts_team_id(home_name_raw, division_name)
                tts_away_id = resolve_tts_team_id(away_name_raw, division_name)
                if tts_home_id is not None:
                    _ensure_team_logo(
                        conn,
                        team_db_id=team1_id,
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        tts_team_id=int(tts_home_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
                if tts_away_id is not None:
                    _ensure_team_logo(
                        conn,
                        team_db_id=team2_id,
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        tts_team_id=int(tts_away_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
            except Exception:
                pass

        notes = f"Imported from TimeToScore {args.source} game_id={gid}"
        game_db_id = upsert_hky_game(
            conn,
            user_id=user_id,
            team1_id=team1_id,
            team2_id=team2_id,
            starts_at=starts_at,
            location=location,
            team1_score=t1_score,
            team2_score=t2_score,
            replace=bool(args.replace),
            notes=notes,
        )

        map_team_to_league_with_division(
            conn,
            league_id=league_id,
            team_id=team1_id,
            division_name=division_name,
            division_id=division_id,
            conference_id=conference_id,
        )
        map_team_to_league_with_division(
            conn,
            league_id=league_id,
            team_id=team2_id,
            division_name=division_name,
            division_id=division_id,
            conference_id=conference_id,
        )
        map_game_to_league_with_division(
            conn,
            league_id=league_id,
            game_id=game_db_id,
            division_name=division_name,
            division_id=division_id,
            conference_id=conference_id,
        )

        # Rosters
        for row in tts_norm.extract_roster(stats, "home"):
            ensure_player(conn, user_id=user_id, team_id=team1_id, name=row["name"], jersey=row["number"], position=row["position"])
        for row in tts_norm.extract_roster(stats, "away"):
            ensure_player(conn, user_id=user_id, team_id=team2_id, name=row["name"], jersey=row["number"], position=row["position"])
        if not args.no_cleanup_bogus_players:
            if int(team1_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(conn, user_id=user_id, team_id=int(team1_id))
                cleaned_team_ids.add(int(team1_id))
            if int(team2_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(conn, user_id=user_id, team_id=int(team2_id))
                cleaned_team_ids.add(int(team2_id))

        # Minimal player stats (goals/assists)
        for agg in tts_norm.aggregate_goals_assists(stats):
            pname = str(agg.get("name") or "").strip()
            if not pname:
                continue
            goals = int(agg.get("goals") or 0)
            assists = int(agg.get("assists") or 0)
            if goals == 0 and assists == 0:
                continue

            # Prefer matching by name within each team roster.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                    (user_id, team1_id, pname),
                )
                r1 = cur.fetchone()
                cur.execute(
                    "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                    (user_id, team2_id, pname),
                )
                r2 = cur.fetchone()
            team_ref = team1_id
            pid = int(r1[0]) if r1 else (int(r2[0]) if r2 else None)
            if r2 and not r1:
                team_ref = team2_id
            if pid is None:
                pid = ensure_player(conn, user_id=user_id, team_id=team_ref, name=pname, jersey=None, position=None)

            with conn.cursor() as cur:
                if args.replace:
                    cur.execute(
                        """
                        INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                        VALUES(%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE goals=VALUES(goals), assists=VALUES(assists)
                        """,
                        (user_id, team_ref, game_db_id, pid, goals, assists),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                        VALUES(%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE goals=COALESCE(goals, VALUES(goals)), assists=COALESCE(assists, VALUES(assists))
                        """,
                        (user_id, team_ref, game_db_id, pid, goals, assists),
                    )
            conn.commit()

        count += 1
        if count % 25 == 0 or count == total:
            elapsed = max(0.001, time.time() - started)
            rate = count / elapsed
            pct = (count / total * 100.0) if total else 100.0
            log(f"Progress: {count}/{total} ({pct:.1f}%) games, skipped={skipped}, {rate:.2f} games/s")

    log(f"Import complete. Imported {count} games, skipped={skipped}.")
    if not args.no_cleanup_bogus_players:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT lt.team_id
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s AND t.user_id=%s
                    """,
                    (league_id, user_id),
                )
                all_team_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})
            moved_total = 0
            for tid in all_team_ids:
                moved_total += _cleanup_numeric_named_players(conn, user_id=user_id, team_id=int(tid))
            if moved_total:
                log(f"Cleaned bogus numeric-name players: migrated {moved_total} stat rows.")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

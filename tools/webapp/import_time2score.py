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
import base64
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
              mhr_div_rating DOUBLE NULL,
              mhr_rating DOUBLE NULL,
              mhr_agd DOUBLE NULL,
              mhr_sched DOUBLE NULL,
              mhr_games INT NULL,
              mhr_updated_at DATETIME NULL,
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
              sort_order INT NULL,
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
                "sort_order INT NULL",
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

        # Add rating columns (best-effort) so direct-DB imports work even if the webapp hasn't run yet.
        for col_ddl in [
            "mhr_div_rating DOUBLE NULL",
            "mhr_rating DOUBLE NULL",
            "mhr_agd DOUBLE NULL",
            "mhr_sched DOUBLE NULL",
            "mhr_games INT NULL",
            "mhr_updated_at DATETIME NULL",
        ]:
            col = col_ddl.split(" ", 1)[0]
            try:
                cur.execute("SHOW COLUMNS FROM league_teams LIKE %s", (col,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(f"ALTER TABLE league_teams ADD COLUMN {col_ddl}")
            except Exception:
                try:
                    cur.execute(f"ALTER TABLE league_teams ADD COLUMN {col_ddl}")
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
            INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id, sort_order)
            VALUES(%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              division_name=COALESCE(VALUES(division_name), division_name),
              division_id=COALESCE(VALUES(division_id), division_id),
              conference_id=COALESCE(VALUES(conference_id), conference_id),
              sort_order=COALESCE(VALUES(sort_order), sort_order)
            """,
            (league_id, game_id, None, None, None, None),
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
    sort_order: Optional[int] = None,
) -> None:
    dn = (division_name or "").strip() or None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id, sort_order)
            VALUES(%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
              division_name=COALESCE(VALUES(division_name), division_name),
              division_id=COALESCE(VALUES(division_id), division_id),
              conference_id=COALESCE(VALUES(conference_id), conference_id),
              sort_order=COALESCE(VALUES(sort_order), sort_order)
            """,
            (league_id, game_id, dn, division_id, conference_id, sort_order),
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
    def _norm_team_name(s: str) -> str:
        # Normalize whitespace and common unicode variants to avoid duplicate teams that render identically in HTML.
        t = str(s or "").replace("\xa0", " ").strip()
        t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
        t = " ".join(t.split())
        return t

    nm = _norm_team_name(name or "")
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
    game_type_id: Optional[int],
    starts_at: Optional[str],
    location: Optional[str],
    team1_score: Optional[int],
    team2_score: Optional[int],
    replace: bool,
    notes: Optional[str] = None,
    timetoscore_game_id: Optional[int] = None,
    timetoscore_season_id: Optional[int] = None,
    timetoscore_source: Optional[str] = None,
) -> int:
    # Standardize notes with JSON like the webapp import API, but keep backward
    # compatibility for matching older token formats.
    notes_json_fields: dict[str, Any] = {}
    if timetoscore_game_id is not None:
        notes_json_fields["timetoscore_game_id"] = int(timetoscore_game_id)
    if timetoscore_season_id is not None:
        notes_json_fields["timetoscore_season_id"] = int(timetoscore_season_id)
    if timetoscore_source:
        notes_json_fields["timetoscore_source"] = str(timetoscore_source)
    if notes and str(notes).strip():
        # If caller passed JSON already, merge it; otherwise preserve as raw string under "notes_raw".
        try:
            parsed = json.loads(str(notes))
            if isinstance(parsed, dict):
                notes_json_fields.update(parsed)
            else:
                notes_json_fields["notes_raw"] = str(notes)
        except Exception:
            notes_json_fields["notes_raw"] = str(notes)

    def _merge_notes(existing: Optional[str], new_fields: dict[str, Any]) -> str:
        if not existing:
            return json.dumps(new_fields, sort_keys=True)
        try:
            cur = json.loads(existing)
            if isinstance(cur, dict):
                cur.update(new_fields)
                return json.dumps(cur, sort_keys=True)
        except Exception:
            pass
        return existing

    with conn.cursor() as cur:
        gid: Optional[int] = None
        if starts_at:
            cur.execute(
                "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND team1_id=%s AND team2_id=%s AND starts_at=%s",
                (user_id, team1_id, team2_id, starts_at),
            )
            row = cur.fetchone()
            if row:
                gid = int(row[0])

        if gid is None and timetoscore_game_id is not None:
            tts_int = int(timetoscore_game_id)
            token_plain = f"game_id={tts_int}"
            token_json_nospace = f"\"timetoscore_game_id\":{tts_int}"
            token_json_space = f"\"timetoscore_game_id\": {tts_int}"
            for token in (token_json_nospace, token_json_space, token_plain):
                cur.execute(
                    "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND notes LIKE %s",
                    (user_id, f"%{token}%"),
                )
                row = cur.fetchone()
                if row:
                    gid = int(row[0])
                    break

        if gid is None:
            merged_notes = json.dumps(notes_json_fields, sort_keys=True) if notes_json_fields else (notes or None)
            cur.execute(
                """
                INSERT INTO hky_games(user_id, team1_id, team2_id, game_type_id, starts_at, location, team1_score, team2_score, is_final, notes, stats_imported_at, created_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
                    merged_notes,
                    dt.datetime.now().isoformat(),
                    dt.datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

        cur.execute("SELECT notes, team1_score, team2_score FROM hky_games WHERE id=%s", (gid,))
        row2 = cur.fetchone()
        existing_notes = row2[0] if row2 else None
        merged_notes = _merge_notes(existing_notes, notes_json_fields) if notes_json_fields else (existing_notes or "")

        if replace:
            cur.execute(
                """
                UPDATE hky_games
                SET game_type_id=COALESCE(%s, game_type_id),
                    location=COALESCE(%s, location),
                    team1_score=%s,
                    team2_score=%s,
                    is_final=CASE WHEN %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                    notes=%s,
                    stats_imported_at=%s,
                    updated_at=%s
                WHERE id=%s
                """,
                (
                    game_type_id,
                    location,
                    team1_score,
                    team2_score,
                    team1_score,
                    team2_score,
                    merged_notes,
                    dt.datetime.now().isoformat(),
                    dt.datetime.now().isoformat(),
                    gid,
                ),
            )
        else:
            cur.execute(
                """
                UPDATE hky_games
                SET game_type_id=COALESCE(%s, game_type_id),
                    location=COALESCE(%s, location),
                    team1_score=COALESCE(team1_score, %s),
                    team2_score=COALESCE(team2_score, %s),
                    is_final=CASE WHEN team1_score IS NULL AND team2_score IS NULL AND %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                    notes=%s,
                    stats_imported_at=%s,
                    updated_at=%s
                WHERE id=%s
                """,
                (
                    game_type_id,
                    location,
                    team1_score,
                    team2_score,
                    team1_score,
                    team2_score,
                    merged_notes,
                    dt.datetime.now().isoformat(),
                    dt.datetime.now().isoformat(),
                    gid,
                ),
            )
        conn.commit()
        return gid


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


def apply_games_batch_payload_to_db(conn, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the same payload shape used by the webapp REST endpoint
    `/api/import/hockey/games_batch` directly to a DB connection.

    This is primarily used for regression testing equivalence between direct DB
    import and REST import.
    """
    league_name = str(payload.get("league_name") or "").strip()
    if not league_name:
        raise ValueError("league_name is required")
    shared = bool(payload.get("shared", False))
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "").strip().lower()
    owner_name = str(payload.get("owner_name") or owner_email).strip()
    source = str(payload.get("source") or "").strip() or None
    external_key = str(payload.get("external_key") or "").strip() or None
    games = payload.get("games") or []
    if not isinstance(games, list):
        raise ValueError("games must be a list")

    # Best-effort: the deployed webapp DB already has these tables; in unit tests
    # we use a fake DB that doesn't implement DDL queries.
    try:
        ensure_defaults(conn)
    except Exception:
        pass
    try:
        ensure_league_schema(conn)
    except Exception:
        pass

    # Ensure owner user exists (mirror webapp import behavior; do not require create-user flags here).
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE email=%s", (owner_email,))
        r = cur.fetchone()
        if r:
            owner_user_id = int(r[0])
        else:
            # Deterministic placeholder hash; caller can update later.
            cur.execute(
                "INSERT INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
                (owner_email, "imported", owner_name or owner_email, dt.datetime.now().isoformat()),
            )
            conn.commit()
            owner_user_id = int(cur.lastrowid)

    # Ensure league exists.
    with conn.cursor() as cur:
        cur.execute("SELECT id, is_shared FROM leagues WHERE name=%s", (league_name,))
        row = cur.fetchone()
        if row:
            league_id = int(row[0])
            want_shared = 1 if shared else 0
            if int(row[1]) != want_shared:
                cur.execute(
                    "UPDATE leagues SET is_shared=%s, updated_at=%s WHERE id=%s",
                    (want_shared, dt.datetime.now().isoformat(), league_id),
                )
                conn.commit()
        else:
            cur.execute(
                "INSERT INTO leagues(name, owner_user_id, is_shared, source, external_key, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
                (league_name, owner_user_id, 1 if shared else 0, source, external_key, dt.datetime.now().isoformat()),
            )
            conn.commit()
            league_id = int(cur.lastrowid)

    ensure_league_member(conn, league_id, owner_user_id, role="admin")

    def _normalize_import_game_type_name(raw: Any) -> Optional[str]:
        s = str(raw or "").strip()
        if not s:
            return None
        sl = s.casefold()
        if sl.startswith("regular"):
            return "Regular Season"
        if sl.startswith("preseason"):
            return "Preseason"
        if sl.startswith("exhibition"):
            return "Exhibition"
        if sl.startswith("tournament"):
            return "Tournament"
        return s

    def _ensure_game_type_id(name_any: Any) -> Optional[int]:
        nm = _normalize_import_game_type_name(name_any)
        if not nm:
            return None
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM game_types WHERE name=%s", (nm,))
            r = cur.fetchone()
            if r:
                return int(r[0])
            cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (nm, 0))
            conn.commit()
            return int(cur.lastrowid)

    def _ensure_player_for_import(
        team_id: int, name: str, jersey_number: Optional[str], position: Optional[str]
    ) -> int:
        nm = (name or "").strip()
        if not nm:
            raise ValueError("player name is required")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                (owner_user_id, team_id, nm),
            )
            row = cur.fetchone()
            if row:
                pid = int(row[0])
                if jersey_number or position:
                    cur.execute(
                        "UPDATE players SET jersey_number=COALESCE(%s, jersey_number), position=COALESCE(%s, position), updated_at=%s WHERE id=%s",
                        (jersey_number, position, dt.datetime.now().isoformat(), pid),
                    )
                    conn.commit()
                return pid
            cur.execute(
                "INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
                (owner_user_id, team_id, nm, jersey_number, position, dt.datetime.now().isoformat()),
            )
            conn.commit()
            return int(cur.lastrowid)

    results: list[dict[str, Any]] = []

    def _clean_division_name(dn: Any) -> Optional[str]:
        s = str(dn or "").strip()
        if not s:
            return None
        if s.lower() == "external":
            return None
        return s

    def _league_team_div_meta(team_id: int) -> tuple[Optional[str], Optional[int], Optional[int]]:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT division_name, division_id, conference_id FROM league_teams WHERE league_id=%s AND team_id=%s",
                (int(league_id), int(team_id)),
            )
            r = cur.fetchone()
        if not r:
            return None, None, None
        dn = _clean_division_name(r[0])
        try:
            did = int(r[1]) if r[1] is not None else None
        except Exception:
            did = None
        try:
            cid = int(r[2]) if r[2] is not None else None
        except Exception:
            cid = None
        return dn, did, cid

    for game in games:
        if not isinstance(game, dict):
            continue
        home_name = str(game.get("home_name") or "").strip()
        away_name = str(game.get("away_name") or "").strip()
        if not home_name or not away_name:
            continue

        team1_id = ensure_team(conn, owner_user_id, home_name, is_external=True)
        team2_id = ensure_team(conn, owner_user_id, away_name, is_external=True)

        starts_at = str(game.get("starts_at") or "").strip() or None
        location = str(game.get("location") or "").strip() or None
        team1_score = int(game["home_score"]) if game.get("home_score") is not None else None
        team2_score = int(game["away_score"]) if game.get("away_score") is not None else None

        tts_game_id = int(game["timetoscore_game_id"]) if game.get("timetoscore_game_id") is not None else None
        season_id = int(game["season_id"]) if game.get("season_id") is not None else None
        game_type_id = _ensure_game_type_id(
            game.get("game_type_name") or game.get("game_type") or game.get("timetoscore_type") or game.get("type")
        )

        gid = upsert_hky_game(
            conn,
            user_id=owner_user_id,
            team1_id=team1_id,
            team2_id=team2_id,
            game_type_id=game_type_id,
            starts_at=starts_at,
            location=location,
            team1_score=team1_score,
            team2_score=team2_score,
            replace=replace,
            notes=None,
            timetoscore_game_id=tts_game_id,
            timetoscore_season_id=season_id,
            timetoscore_source=source,
        )

        # Map teams/games to league with division metadata (best-effort).
        def _int_or_none(x: Any) -> Optional[int]:
            try:
                return int(x) if x is not None else None
            except Exception:
                return None

        division_name = _clean_division_name(game.get("division_name"))
        division_id = _int_or_none(game.get("division_id"))
        conference_id = _int_or_none(game.get("conference_id"))
        sort_order = _int_or_none(game.get("sort_order"))

        home_div_name = _clean_division_name(game.get("home_division_name")) or division_name
        away_div_name = _clean_division_name(game.get("away_division_name")) or division_name

        map_team_to_league_with_division(
            conn,
            league_id=league_id,
            team_id=team1_id,
            division_name=home_div_name,
            division_id=_int_or_none(game.get("home_division_id")) or division_id,
            conference_id=_int_or_none(game.get("home_conference_id")) or conference_id,
        )
        map_team_to_league_with_division(
            conn,
            league_id=league_id,
            team_id=team2_id,
            division_name=away_div_name,
            division_id=_int_or_none(game.get("away_division_id")) or division_id,
            conference_id=_int_or_none(game.get("away_conference_id")) or conference_id,
        )

        effective_div_name = division_name or home_div_name or away_div_name
        effective_div_id = division_id or _int_or_none(game.get("home_division_id")) or _int_or_none(game.get("away_division_id"))
        effective_conf_id = conference_id or _int_or_none(game.get("home_conference_id")) or _int_or_none(game.get("away_conference_id"))
        if not effective_div_name:
            t1_dn, t1_did, t1_cid = _league_team_div_meta(int(team1_id))
            t2_dn, t2_did, t2_cid = _league_team_div_meta(int(team2_id))
            if t1_dn:
                effective_div_name = t1_dn
                effective_div_id = effective_div_id or t1_did
                effective_conf_id = effective_conf_id or t1_cid
            elif t2_dn:
                effective_div_name = t2_dn
                effective_div_id = effective_div_id or t2_did
                effective_conf_id = effective_conf_id or t2_cid

        map_game_to_league_with_division(
            conn,
            league_id=league_id,
            game_id=gid,
            division_name=effective_div_name,
            division_id=effective_div_id,
            conference_id=effective_conf_id,
            sort_order=sort_order,
        )

        # Rosters (optional).
        for side_key, tid in (("home", team1_id), ("away", team2_id)):
            roster = game.get(f"{side_key}_roster") or []
            if not isinstance(roster, list):
                continue
            for row in roster:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name") or "").strip()
                if not nm:
                    continue
                jersey = str(row.get("number") or "").strip() or None
                pos = str(row.get("position") or "").strip() or None
                _ensure_player_for_import(int(tid), nm, jersey, pos)

        # Minimal goals/assists stats.
        stats_rows = game.get("player_stats") or []
        if isinstance(stats_rows, list):
            for srow in stats_rows:
                if not isinstance(srow, dict):
                    continue
                pname = str(srow.get("name") or "").strip()
                if not pname:
                    continue
                goals = int(srow.get("goals") or 0)
                assists = int(srow.get("assists") or 0)
                if goals == 0 and assists == 0:
                    continue

                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                        (owner_user_id, team1_id, pname),
                    )
                    r1 = cur.fetchone()
                    cur.execute(
                        "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                        (owner_user_id, team2_id, pname),
                    )
                    r2 = cur.fetchone()
                team_ref = team1_id
                pid = int(r1[0]) if r1 else (int(r2[0]) if r2 else None)
                if r2 and not r1:
                    team_ref = team2_id
                if pid is None:
                    pid = _ensure_player_for_import(int(team_ref), pname, None, None)

                with conn.cursor() as cur:
                    if replace:
                        cur.execute(
                            """
                            INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                            VALUES(%s,%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE goals=VALUES(goals), assists=VALUES(assists)
                            """,
                            (owner_user_id, team_ref, gid, pid, goals, assists),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                            VALUES(%s,%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE goals=COALESCE(goals, VALUES(goals)), assists=COALESCE(assists, VALUES(assists))
                            """,
                            (owner_user_id, team_ref, gid, pid, goals, assists),
                        )
                conn.commit()

        results.append({"game_id": int(gid), "team1_id": int(team1_id), "team2_id": int(team2_id)})

    return {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id), "imported": len(results), "results": results}


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
    ap.add_argument(
        "--refresh-team-metadata",
        action="store_true",
        help="Refresh league team divisions/logos from TimeToScore division lists (no game scraping).",
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

    ap.add_argument(
        "--api-url",
        "--url",
        dest="api_url",
        default=None,
        help="If set, import via the webapp REST API at this base URL (e.g. http://127.0.0.1:8008).",
    )
    ap.add_argument(
        "--api-token",
        "--import-token",
        dest="api_token",
        default=None,
        help="Optional import token for REST API auth (sent as Authorization: Bearer ... and X-HM-Import-Token).",
    )
    ap.add_argument(
        "--api-batch-size",
        type=int,
        default=50,
        help="Games per REST batch request (only with --api-url).",
    )

    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from hmlib.time2score import direct as tts_direct
    from hmlib.time2score import normalize as tts_norm

    rest_mode = bool(args.api_url)
    if rest_mode and (args.cleanup_only or args.refresh_team_metadata or args.share_with):
        raise SystemExit(
            "--cleanup-only/--refresh-team-metadata/--share-with are only supported for direct DB imports (omit --api-url)"
        )

    conn = None
    if not rest_mode:
        log(f"Connecting to DB via config: {args.config}")
        conn = connect_pymysql(load_db_cfg(args.config))
        ensure_defaults(conn)
        ensure_league_schema(conn)

    logo_dir = None
    if (not args.no_import_logos) and (not rest_mode):
        if args.logo_dir:
            logo_dir = Path(str(args.logo_dir)).expanduser()
        else:
            preferred = Path("/opt/hm-webapp/app/instance/uploads/team_logos")
            logo_dir = preferred if preferred.exists() else (base_dir / "instance" / "uploads" / "team_logos")

    user_id = None
    if not rest_mode:
        assert conn is not None
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
    tts_team_div_meta_by_id: dict[int, tuple[str, int, int]] = {}
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
            tts_team_div_meta_by_id[tid_i] = (dn, int(d.division_id), int(d.conference_id))

    def resolve_tts_team_id(name: str, division_name: Optional[str]) -> Optional[int]:
        nm = str(name or "").strip().lower()
        dn = str(division_name or "").strip().lower()
        if nm and dn and (nm, dn) in tts_team_id_by_name_div:
            return int(tts_team_id_by_name_div[(nm, dn)])
        ids = tts_team_ids_by_name.get(nm) or []
        if len(ids) == 1:
            return int(ids[0])
        return None

    def resolve_team_division_meta(name: str, fallback_division_name: Optional[str]) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """Return (division_name, division_id, conference_id, team_tts_id)."""
        tts_id = resolve_tts_team_id(name, fallback_division_name)
        if tts_id is not None and int(tts_id) in tts_team_div_meta_by_id:
            dn, did, cid = tts_team_div_meta_by_id[int(tts_id)]
            return dn or None, int(did), int(cid), int(tts_id)
        # If the team name is unique across all divisions, assign its division without relying on the game.
        ids = tts_team_ids_by_name.get(str(name or "").strip().lower()) or []
        if len(ids) == 1 and int(ids[0]) in tts_team_div_meta_by_id:
            dn, did, cid = tts_team_div_meta_by_id[int(ids[0])]
            return dn or None, int(did), int(cid), int(ids[0])
        return (fallback_division_name, None, None, int(tts_id) if tts_id is not None else None)

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
    owner_id = None
    league_id = None
    if not rest_mode:
        assert conn is not None
        assert user_id is not None
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

    if args.cleanup_only or args.refresh_team_metadata:
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
                progress_cb=log,
                progress_every_teams=10,
                heartbeat_seconds=30.0,
            )
            game_ids = sorted(fallback_by_gid.keys())

    total = len(game_ids)
    log(f"Importing games: total={total} replace={bool(args.replace)}")

    count = 0
    skipped = 0
    posted = 0
    api_games_batch: list[dict[str, Any]] = []
    logo_url_cache: dict[int, Optional[str]] = {}
    logo_b64_cache: dict[int, Optional[str]] = {}
    logo_ct_cache: dict[int, Optional[str]] = {}
    sent_logo_ids: set[int] = set()
    cleaned_team_ids: set[int] = set()
    started = time.time()
    last_heartbeat = started

    api_base = str(args.api_url or "").rstrip("/")
    api_headers: dict[str, str] = {}
    if rest_mode and args.api_token:
        tok = str(args.api_token).strip()
        if tok:
            api_headers["Authorization"] = f"Bearer {tok}"
            api_headers["X-HM-Import-Token"] = tok
    api_batch_size = max(1, int(args.api_batch_size or 1))

    def _post_batch() -> None:
        nonlocal posted, api_games_batch
        if not rest_mode or not api_games_batch:
            return
        import requests

        payload = {
            "league_name": league_name,
            "shared": bool(args.shared),
            "replace": bool(args.replace),
            "owner_email": owner_email,
            "owner_name": owner_email,
            "source": "timetoscore",
            "external_key": f"{args.source}:{season_id}",
            "games": api_games_batch,
        }
        r = requests.post(f"{api_base}/api/import/hockey/games_batch", json=payload, headers=api_headers, timeout=180)
        r.raise_for_status()
        out = r.json()
        if not out.get("ok"):
            raise RuntimeError(str(out))
        posted += int(out.get("imported") or 0)
        api_games_batch = []
    for gid in game_ids:
        now = time.time()
        if (now - last_heartbeat) >= 30.0:
            pct = (count / total * 100.0) if total else 100.0
            log(f"Working... next game_id={gid} ({count}/{total}, {pct:.1f}%)")
            last_heartbeat = now
        if args.limit is not None and count >= int(args.limit):
            break
        fb = fallback_by_gid.get(int(gid))
        try:
            stats = tts_direct.scrape_game_stats(args.source, game_id=int(gid), season_id=season_id)
        except Exception:
            stats = {}
            if not fb:
                skipped += 1
                continue

        home_name = str(stats.get("home") or "").strip() or str((fb or {}).get("home") or "").strip()
        away_name = str(stats.get("away") or "").strip() or str((fb or {}).get("away") or "").strip()
        if not home_name or not away_name:
            skipped += 1
            continue

        fb_division_name = str((fb or {}).get("division_name") or "").strip() or None
        fb_division_id = None
        fb_conference_id = None
        try:
            fb_division_id = int((fb or {}).get("division_id")) if (fb or {}).get("division_id") is not None else None
        except Exception:
            fb_division_id = None
        try:
            fb_conference_id = int((fb or {}).get("conference_id")) if (fb or {}).get("conference_id") is not None else None
        except Exception:
            fb_conference_id = None

        home_name_raw = str(home_name or "").strip()
        away_name_raw = str(away_name or "").strip()
        home_div_name, home_div_id, home_conf_id, home_tts_id = resolve_team_division_meta(home_name_raw, fb_division_name)
        away_div_name, away_div_id, away_conf_id, away_tts_id = resolve_team_division_meta(away_name_raw, fb_division_name)

        # Use per-team division for disambiguation, not the game's division block.
        home_name = canonical_team_name(home_name_raw, home_div_name)
        away_name = canonical_team_name(away_name_raw, away_div_name)

        starts_at = parse_starts_at(args.source, stats=stats, fallback=fb)
        location = str(stats.get("location") or "").strip() or str((fb or {}).get("rink") or "").strip() or None
        t1_score = tts_norm.parse_int_or_none(stats.get("homeGoals"))
        t2_score = tts_norm.parse_int_or_none(stats.get("awayGoals"))
        if t1_score is None and (fb or {}).get("homeGoals") is not None:
            t1_score = tts_norm.parse_int_or_none((fb or {}).get("homeGoals"))
        if t2_score is None and (fb or {}).get("awayGoals") is not None:
            t2_score = tts_norm.parse_int_or_none((fb or {}).get("awayGoals"))

        if rest_mode:
            def _logo_url(tts_id: Optional[int]) -> Optional[str]:
                if args.no_import_logos or tts_id is None:
                    return None
                tid_i = int(tts_id)
                if tid_i in logo_url_cache:
                    return logo_url_cache[tid_i]
                try:
                    u = tts_direct.scrape_team_logo_url(str(args.source), season_id=int(season_id), team_id=tid_i)
                except Exception:
                    u = None
                logo_url_cache[tid_i] = u
                return u

            def _logo_b64_and_type(tts_id: Optional[int]) -> tuple[Optional[str], Optional[str]]:
                # When importing via REST, embed logo bytes because the deployed webapp venv may not include `requests`.
                if args.no_import_logos or tts_id is None:
                    return None, None
                tid_i = int(tts_id)
                if tid_i in logo_b64_cache:
                    return logo_b64_cache[tid_i], logo_ct_cache.get(tid_i)

                url = _logo_url(tid_i)
                if not url:
                    logo_b64_cache[tid_i] = None
                    logo_ct_cache[tid_i] = None
                    return None, None

                try:
                    data, content_type = _download_logo_bytes(url)
                except Exception:
                    logo_b64_cache[tid_i] = None
                    logo_ct_cache[tid_i] = None
                    return None, None

                try:
                    b64 = base64.b64encode(data).decode("ascii")
                except Exception:
                    b64 = None
                logo_b64_cache[tid_i] = b64
                logo_ct_cache[tid_i] = content_type
                return b64, content_type

            game_div_name = home_div_name or fb_division_name
            game_div_id = home_div_id if home_div_id is not None else fb_division_id
            game_conf_id = home_conf_id if home_conf_id is not None else fb_conference_id

            home_logo_url = _logo_url(home_tts_id)
            away_logo_url = _logo_url(away_tts_id)
            home_logo_b64 = None
            home_logo_ct = None
            away_logo_b64 = None
            away_logo_ct = None
            if not args.no_import_logos:
                if home_tts_id is not None:
                    tid_i = int(home_tts_id)
                    if bool(args.replace) or tid_i not in sent_logo_ids:
                        home_logo_b64, home_logo_ct = _logo_b64_and_type(tid_i)
                        if home_logo_b64:
                            sent_logo_ids.add(tid_i)
                if away_tts_id is not None:
                    tid_i = int(away_tts_id)
                    if bool(args.replace) or tid_i not in sent_logo_ids:
                        away_logo_b64, away_logo_ct = _logo_b64_and_type(tid_i)
                        if away_logo_b64:
                            sent_logo_ids.add(tid_i)

            api_games_batch.append(
                {
                    "home_name": home_name,
                    "away_name": away_name,
                    "game_type_name": str((fb or {}).get("type") or "").strip() or None,
                    "division_name": game_div_name,
                    "division_id": game_div_id,
                    "conference_id": game_conf_id,
                    "home_division_name": home_div_name,
                    "home_division_id": home_div_id,
                    "home_conference_id": home_conf_id,
                    "away_division_name": away_div_name,
                    "away_division_id": away_div_id,
                    "away_conference_id": away_conf_id,
                    "starts_at": starts_at,
                    "location": location,
                    "home_score": t1_score,
                    "away_score": t2_score,
                    "timetoscore_game_id": int(gid),
                    "season_id": int(season_id),
                    "home_logo_url": home_logo_url,
                    "away_logo_url": away_logo_url,
                    "home_logo_b64": home_logo_b64,
                    "home_logo_content_type": home_logo_ct,
                    "away_logo_b64": away_logo_b64,
                    "away_logo_content_type": away_logo_ct,
                    "home_roster": list(tts_norm.extract_roster(stats, "home")),
                    "away_roster": list(tts_norm.extract_roster(stats, "away")),
                    "player_stats": [
                        {
                            "name": str(agg.get("name") or "").strip(),
                            "goals": int(agg.get("goals") or 0),
                            "assists": int(agg.get("assists") or 0),
                        }
                        for agg in tts_norm.aggregate_goals_assists(stats)
                        if str(agg.get("name") or "").strip()
                    ],
                }
            )
            if len(api_games_batch) >= api_batch_size:
                _post_batch()

            count += 1
            if count % 25 == 0 or count == total:
                elapsed = max(0.001, time.time() - started)
                rate = count / elapsed
                pct = (count / total * 100.0) if total else 100.0
                log(
                    f"Progress: scraped {count}/{total} ({pct:.1f}%) games, posted={posted}, skipped={skipped}, {rate:.2f} games/s"
                )
            continue

        assert conn is not None
        assert user_id is not None
        assert league_id is not None

        team1_id = ensure_team(conn, user_id, home_name, is_external=True)
        team2_id = ensure_team(conn, user_id, away_name, is_external=True)
        if logo_dir is not None:
            try:
                if home_tts_id is not None:
                    _ensure_team_logo(
                        conn,
                        team_db_id=team1_id,
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        tts_team_id=int(home_tts_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
                if away_tts_id is not None:
                    _ensure_team_logo(
                        conn,
                        team_db_id=team2_id,
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        tts_team_id=int(away_tts_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
            except Exception:
                pass

        notes = f"Imported from TimeToScore {args.source} game_id={gid}"
        game_type_name = str((fb or {}).get("type") or "").strip() or None
        game_type_id = None
        if game_type_name:
            with conn.cursor() as cur:
                # Map TimeToScore schedule Type to our canonical game type names.
                sl = game_type_name.casefold()
                if sl.startswith("regular"):
                    nm = "Regular Season"
                elif sl.startswith("preseason"):
                    nm = "Preseason"
                elif sl.startswith("exhibition"):
                    nm = "Exhibition"
                elif sl.startswith("tournament"):
                    nm = "Tournament"
                else:
                    nm = game_type_name
                cur.execute("SELECT id FROM game_types WHERE name=%s", (nm,))
                r = cur.fetchone()
                if r:
                    game_type_id = int(r[0])
                else:
                    cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (nm, 0))
                    conn.commit()
                    game_type_id = int(cur.lastrowid)
        game_db_id = upsert_hky_game(
            conn,
            user_id=user_id,
            team1_id=team1_id,
            team2_id=team2_id,
            game_type_id=game_type_id,
            starts_at=starts_at,
            location=location,
            team1_score=t1_score,
            team2_score=t2_score,
            replace=bool(args.replace),
            notes=notes,
            timetoscore_game_id=int(gid),
        )

        map_team_to_league_with_division(
            conn,
            league_id=league_id,
            team_id=team1_id,
            division_name=home_div_name,
            division_id=home_div_id if home_div_id is not None else fb_division_id,
            conference_id=home_conf_id if home_conf_id is not None else fb_conference_id,
        )
        map_team_to_league_with_division(
            conn,
            league_id=league_id,
            team_id=team2_id,
            division_name=away_div_name,
            division_id=away_div_id if away_div_id is not None else fb_division_id,
            conference_id=away_conf_id if away_conf_id is not None else fb_conference_id,
        )
        # For a per-division schedule, attribute games to the home team's division when available.
        game_div_name = home_div_name or fb_division_name
        game_div_id = home_div_id if home_div_id is not None else fb_division_id
        game_conf_id = home_conf_id if home_conf_id is not None else fb_conference_id
        map_game_to_league_with_division(
            conn,
            league_id=league_id,
            game_id=game_db_id,
            division_name=game_div_name,
            division_id=game_div_id,
            conference_id=game_conf_id,
        )

        # Rosters (optional for schedule-only rows)
        for row in tts_norm.extract_roster(stats, "home"):
            ensure_player(
                conn,
                user_id=user_id,
                team_id=team1_id,
                name=row["name"],
                jersey=row["number"],
                position=row["position"],
            )
        for row in tts_norm.extract_roster(stats, "away"):
            ensure_player(
                conn,
                user_id=user_id,
                team_id=team2_id,
                name=row["name"],
                jersey=row["number"],
                position=row["position"],
            )
        if not args.no_cleanup_bogus_players:
            if int(team1_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(conn, user_id=user_id, team_id=int(team1_id))
                cleaned_team_ids.add(int(team1_id))
            if int(team2_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(conn, user_id=user_id, team_id=int(team2_id))
                cleaned_team_ids.add(int(team2_id))

        # Minimal player stats (goals/assists). If we couldn't load boxscore stats, this will be empty.
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

    if rest_mode:
        _post_batch()

    log(f"Import complete. Imported {count} games, skipped={skipped}.")
    if (not rest_mode) and (not args.no_cleanup_bogus_players):
        try:
            assert conn is not None
            assert league_id is not None
            assert user_id is not None
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
    if (not rest_mode) and args.refresh_team_metadata:
        log("Refreshing league team metadata (divisions/logos)...")
        # Update league_teams divisions based on TimeToScore team list, and fill missing logos.
        try:
            assert conn is not None
            assert league_id is not None
            assert user_id is not None
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lt.team_id, t.name, t.logo_path
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s AND t.user_id=%s
                    """,
                    (league_id, user_id),
                )
                league_team_rows = list(cur.fetchall() or [])
            updated_div = 0
            updated_logo = 0
            for team_db_id, team_name, logo_path in league_team_rows:
                nm = str(team_name or "").strip()
                # If canonical disambiguation is present, strip " (Division)" suffix for matching.
                base_name = nm
                if base_name.endswith(")") and "(" in base_name:
                    base_name = base_name.rsplit("(", 1)[0].strip()
                div_name, div_id, conf_id, tts_id = resolve_team_division_meta(base_name, None)
                if div_name:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE league_teams
                            SET division_name=%s,
                                division_id=COALESCE(%s, division_id),
                                conference_id=COALESCE(%s, conference_id)
                            WHERE league_id=%s AND team_id=%s
                            """,
                            (div_name, div_id, conf_id, league_id, int(team_db_id)),
                        )
                    conn.commit()
                    updated_div += 1
                if logo_dir is not None and (not logo_path) and tts_id is not None:
                    _ensure_team_logo(
                        conn,
                        team_db_id=int(team_db_id),
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        tts_team_id=int(tts_id),
                        logo_dir=Path(logo_dir),
                        replace=False,
                        tts_direct=tts_direct,
                    )
                    updated_logo += 1
            # Also align league_games division to the home team division when possible.
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE league_games lg
                      JOIN hky_games g ON lg.game_id=g.id
                      JOIN league_teams lt ON lt.league_id=lg.league_id AND lt.team_id=g.team1_id
                    SET lg.division_name=COALESCE(lt.division_name, lg.division_name),
                        lg.division_id=COALESCE(lt.division_id, lg.division_id),
                        lg.conference_id=COALESCE(lt.conference_id, lg.conference_id)
                    WHERE lg.league_id=%s
                    """,
                    (league_id,),
                )
            conn.commit()
            log(f"Refreshed teams: divisions_updated={updated_div} logos_checked={updated_logo}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

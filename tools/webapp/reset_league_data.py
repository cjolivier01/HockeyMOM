#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Optional
from urllib.parse import urljoin


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


def wipe_all(conn) -> dict:
    counts: dict = {}
    with conn.cursor() as cur:
        for table in (
            "player_stats",
            "league_games",
            "hky_games",
            "league_teams",
            "players",
            "teams",
        ):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = int((cur.fetchone() or [0])[0])
        # Delete in FK-safe order
        cur.execute("DELETE FROM player_stats")
        cur.execute("DELETE FROM league_games")
        cur.execute("DELETE FROM hky_games")
        cur.execute("DELETE FROM league_teams")
        cur.execute("DELETE FROM players")
        cur.execute("DELETE FROM teams")
    conn.commit()
    return counts


def wipe_league(conn, league_id: int) -> dict:
    # NOTE: This script historically used a more destructive implementation.
    # The REST-backed implementation uses the webapp's safer logic (only deletes
    # exclusive games/teams to avoid impacting other leagues).
    stats = {"player_stats": 0, "league_games": 0, "hky_games": 0, "league_teams": 0, "players": 0, "teams": 0}
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM league_games WHERE league_id=%s", (league_id,))
        stats["league_games"] = int((cur.fetchone() or [0])[0])
        cur.execute("SELECT COUNT(*) FROM league_teams WHERE league_id=%s", (league_id,))
        stats["league_teams"] = int((cur.fetchone() or [0])[0])

        cur.execute(
            """
            SELECT game_id
            FROM league_games
            WHERE league_id=%s
              AND game_id NOT IN (SELECT game_id FROM league_games WHERE league_id<>%s)
            """,
            (league_id, league_id),
        )
        exclusive_game_ids = sorted({int(r[0]) for r in cur.fetchall() or []})
        if exclusive_game_ids:
            q = ",".join(["%s"] * len(exclusive_game_ids))
            cur.execute(f"SELECT COUNT(*) FROM player_stats WHERE game_id IN ({q})", exclusive_game_ids)
            stats["player_stats"] = int((cur.fetchone() or [0])[0])
            cur.execute(f"SELECT COUNT(*) FROM hky_games WHERE id IN ({q})", exclusive_game_ids)
            stats["hky_games"] = int((cur.fetchone() or [0])[0])

        cur.execute(
            """
            SELECT team_id
            FROM league_teams
            WHERE league_id=%s
              AND team_id NOT IN (SELECT team_id FROM league_teams WHERE league_id<>%s)
            """,
            (league_id, league_id),
        )
        exclusive_team_ids = sorted({int(r[0]) for r in cur.fetchall() or []})

        # Remove mappings
        cur.execute("DELETE FROM league_games WHERE league_id=%s", (league_id,))
        cur.execute("DELETE FROM league_teams WHERE league_id=%s", (league_id,))

        # Delete exclusive games (cascades to player_stats and hky_game_*).
        if exclusive_game_ids:
            q = ",".join(["%s"] * len(exclusive_game_ids))
            cur.execute(f"DELETE FROM hky_games WHERE id IN ({q})", exclusive_game_ids)

        # Delete safe external teams (and their players) that are not referenced by remaining games.
        if exclusive_team_ids:
            q = ",".join(["%s"] * len(exclusive_team_ids))
            cur.execute(f"SELECT id, is_external FROM teams WHERE id IN ({q})", exclusive_team_ids)
            eligible = [int(tid) for (tid, is_ext) in (cur.fetchall() or []) if int(is_ext or 0) == 1]
            if eligible:
                q2 = ",".join(["%s"] * len(eligible))
                cur.execute(
                    f"""
                    SELECT DISTINCT team1_id AS tid FROM hky_games WHERE team1_id IN ({q2})
                    UNION
                    SELECT DISTINCT team2_id AS tid FROM hky_games WHERE team2_id IN ({q2})
                    """,
                    eligible * 2,
                )
                still_used = {int(r[0]) for r in (cur.fetchall() or [])}
                safe_team_ids = sorted([tid for tid in eligible if tid not in still_used])
                if safe_team_ids:
                    q3 = ",".join(["%s"] * len(safe_team_ids))
                    cur.execute(f"SELECT COUNT(*) FROM players WHERE team_id IN ({q3})", safe_team_ids)
                    stats["players"] = int((cur.fetchone() or [0])[0])
                    cur.execute(f"SELECT COUNT(*) FROM teams WHERE id IN ({q3})", safe_team_ids)
                    stats["teams"] = int((cur.fetchone() or [0])[0])
                    cur.execute(f"DELETE FROM teams WHERE id IN ({q3})", safe_team_ids)
    conn.commit()
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reset hockey data (teams/players/games/stats) without touching users or permissions"
    )
    ap.add_argument("--config", default="/opt/hm-webapp/app/config.json")
    ap.add_argument("--force", action="store_true", help="Do not prompt for confirmation")
    ap.add_argument("--yes", "-y", action="store_true", help="Alias for --force")
    ap.add_argument("--webapp-url", default=None, help="If set, reset via webapp REST API (e.g. http://127.0.0.1:8008)")
    ap.add_argument("--webapp-token", default=None, help="Optional import token for REST mode")
    ap.add_argument("--import-token", dest="webapp_token", default=None, help="Alias for --webapp-token")
    ap.add_argument("--webapp-owner-email", default=None, help="League owner email for REST mode")
    ap.add_argument(
        "--league-id", type=int, default=None, help="Only wipe data associated to this league id"
    )
    ap.add_argument(
        "--league-name", default=None, help="Only wipe data associated to this league name"
    )
    args = ap.parse_args(argv)
    force = bool(args.force or args.yes)

    if args.webapp_url:
        if not args.league_name:
            print("Error: --webapp-url requires --league-name.", file=sys.stderr)
            return 2
        if not args.webapp_owner_email:
            print("Error: --webapp-url requires --webapp-owner-email.", file=sys.stderr)
            return 2

        scope = f"league_name={args.league_name}"
        if not force:
            ans = input(f"This will reset hockey data for {scope} via REST. Type RESET to continue: ").strip()
            if ans != "RESET":
                print("Aborted.")
                return 1

        import requests

        base = str(args.webapp_url).rstrip("/") + "/"
        url = urljoin(base, "api/internal/reset_league_data")
        headers = {}
        if args.webapp_token:
            tok = str(args.webapp_token).strip()
            if tok:
                headers["Authorization"] = f"Bearer {tok}"
                headers["X-HM-Import-Token"] = tok
        r = requests.post(
            url,
            json={"owner_email": str(args.webapp_owner_email).strip(), "league_name": str(args.league_name).strip()},
            headers=headers,
            timeout=120,
        )
        if r.status_code != 200:
            print(f"[!] REST reset failed: {r.status_code} {r.text}", file=sys.stderr)
            if r.status_code == 404:
                print(
                    "[!] Hint: the webapp at --webapp-url does not have /api/internal/reset_league_data. "
                    "If you recently updated the webapp code, restart the running gunicorn/service "
                    "(and ensure nothing else is already listening on that port).",
                    file=sys.stderr,
                )
            return 3
        print(json.dumps(r.json(), indent=2, sort_keys=True))
        return 0

    db_cfg = load_db_cfg(args.config)
    conn = connect_pymysql(db_cfg)

    # Resolve league id if name provided
    league_id = args.league_id
    if args.league_name and not league_id:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM leagues WHERE name=%s", (args.league_name,))
            row = cur.fetchone()
            if not row:
                print(f"[!] League named '{args.league_name}' not found", file=sys.stderr)
                return 2
            league_id = int(row[0])

    scope = f"league_id={league_id}" if league_id else "ALL"
    if not force:
        ans = input(
            f"This will wipe teams/players/hky games/stats for {scope}. Type RESET to continue: "
        ).strip()
        if ans != "RESET":
            print("Aborted.")
            return 1

    if league_id:
        stats = wipe_league(conn, league_id)
        print(
            "Wiped league data:",
            json.dumps(stats, indent=2, sort_keys=True),
        )
    else:
        counts = wipe_all(conn)
        print("Existing rows:", json.dumps(counts, indent=2, sort_keys=True))
        print("Wipe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

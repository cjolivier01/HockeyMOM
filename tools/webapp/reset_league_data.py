#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Optional


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
    stats = {
        "player_stats": 0,
        "league_games": 0,
        "hky_games": 0,
        "league_teams": 0,
        "players": 0,
        "teams": 0,
    }
    with conn.cursor() as cur:
        # games in this league
        cur.execute("SELECT DISTINCT game_id FROM league_games WHERE league_id=%s", (league_id,))
        game_ids = [int(r[0]) for r in cur.fetchall() or []]
        if game_ids:
            # player stats for those games
            cur.execute(
                f"SELECT COUNT(*) FROM player_stats WHERE game_id IN ({','.join(['%s']*len(game_ids))})",
                game_ids,
            )
            stats["player_stats"] = int((cur.fetchone() or [0])[0])
            cur.execute(
                f"DELETE FROM player_stats WHERE game_id IN ({','.join(['%s']*len(game_ids))})",
                game_ids,
            )
            # delete league games mapping for this league
            cur.execute("SELECT COUNT(*) FROM league_games WHERE league_id=%s", (league_id,))
            stats["league_games"] = int((cur.fetchone() or [0])[0])
            cur.execute("DELETE FROM league_games WHERE league_id=%s", (league_id,))
            # only delete hky_games that are not mapped by other leagues
            cur.execute(
                f"SELECT DISTINCT game_id FROM league_games WHERE game_id IN ({','.join(['%s']*len(game_ids))}) AND league_id<>%s",
                game_ids + [league_id],
            )
            keep = {int(r[0]) for r in cur.fetchall() or []}
            to_delete = [gid for gid in game_ids if gid not in keep]
            if to_delete:
                cur.execute(
                    f"SELECT COUNT(*) FROM hky_games WHERE id IN ({','.join(['%s']*len(to_delete))})",
                    to_delete,
                )
                stats["hky_games"] = int((cur.fetchone() or [0])[0])
                cur.execute(
                    f"DELETE FROM hky_games WHERE id IN ({','.join(['%s']*len(to_delete))})",
                    to_delete,
                )
        # teams in this league
        cur.execute("SELECT DISTINCT team_id FROM league_teams WHERE league_id=%s", (league_id,))
        team_ids = [int(r[0]) for r in cur.fetchall() or []]
        if team_ids:
            # delete league team mappings for this league
            cur.execute("SELECT COUNT(*) FROM league_teams WHERE league_id=%s", (league_id,))
            stats["league_teams"] = int((cur.fetchone() or [0])[0])
            cur.execute("DELETE FROM league_teams WHERE league_id=%s", (league_id,))
            # delete players for teams in this league
            cur.execute(
                f"SELECT COUNT(*) FROM players WHERE team_id IN ({','.join(['%s']*len(team_ids))})",
                team_ids,
            )
            stats["players"] = int((cur.fetchone() or [0])[0])
            cur.execute(
                f"DELETE FROM players WHERE team_id IN ({','.join(['%s']*len(team_ids))})",
                team_ids,
            )
            # delete teams that are not used by other leagues and not referenced by remaining games
            cur.execute(
                f"SELECT DISTINCT team_id FROM league_teams WHERE team_id IN ({','.join(['%s']*len(team_ids))})",
                team_ids,
            )
            still_mapped = {int(r[0]) for r in cur.fetchall() or []}
            # teams used in games
            ref_counts = {}
            for tid in team_ids:
                cur.execute(
                    "SELECT COUNT(*) FROM hky_games WHERE team1_id=%s OR team2_id=%s", (tid, tid)
                )
                ref_counts[tid] = int((cur.fetchone() or [0])[0])
            to_delete_teams = [
                tid for tid in team_ids if tid not in still_mapped and ref_counts.get(tid, 0) == 0
            ]
            if to_delete_teams:
                cur.execute(
                    f"SELECT COUNT(*) FROM teams WHERE id IN ({','.join(['%s']*len(to_delete_teams))})",
                    to_delete_teams,
                )
                stats["teams"] = int((cur.fetchone() or [0])[0])
                cur.execute(
                    f"DELETE FROM teams WHERE id IN ({','.join(['%s']*len(to_delete_teams))})",
                    to_delete_teams,
                )
    conn.commit()
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reset hockey data (teams/players/games/stats) without touching users or permissions"
    )
    ap.add_argument("--config", default="/opt/hm-webapp/app/config.json")
    ap.add_argument("--force", action="store_true", help="Do not prompt for confirmation")
    ap.add_argument(
        "--league-id", type=int, default=None, help="Only wipe data associated to this league id"
    )
    ap.add_argument(
        "--league-name", default=None, help="Only wipe data associated to this league name"
    )
    args = ap.parse_args(argv)

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
    if not args.force:
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

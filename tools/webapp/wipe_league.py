#!/usr/bin/env python3
"""Wipe a league and its imported hockey data from the local webapp database.

This is a destructive, local DB maintenance tool meant for cleaning out an entire league
(league row + mappings + league-owned games + league-owned external teams/players).

Safety:
- Only games/teams that are *exclusively* mapped to the target league are deleted.
- Existing data referenced by other leagues is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _load_db_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg.get("db", {})


def _connect(db_cfg: dict):
    import pymysql

    return pymysql.connect(
        host=db_cfg.get("host", "127.0.0.1"),
        port=int(db_cfg.get("port", 3306)),
        user=db_cfg.get("user", "hmapp"),
        password=db_cfg.get("pass", ""),
        database=db_cfg.get("name", "hm_app_db"),
        autocommit=False,
        charset="utf8mb4",
    )


def _chunks(ids: list[int], n: int = 500) -> Iterable[list[int]]:
    for i in range(0, len(ids), n):
        yield ids[i : i + n]


def main(argv: list[str] | None = None) -> int:
    base_dir = Path(__file__).resolve().parent
    default_cfg = os.environ.get("HM_DB_CONFIG") or str(base_dir / "config.json")
    ap = argparse.ArgumentParser(description="Wipe an entire league from the local HockeyMOM webapp DB")
    ap.add_argument("--config", default=default_cfg, help="Path to webapp config.json (DB creds)")
    ap.add_argument("--league-name", required=True, help="League name to delete (exact match)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be deleted")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = ap.parse_args(argv)

    league_name = str(args.league_name).strip()
    if not league_name:
        raise SystemExit("--league-name is required")

    conn = _connect(_load_db_cfg(args.config))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, owner_user_id, is_shared FROM leagues WHERE name=%s", (league_name,))
            row = cur.fetchone()
            if not row:
                print(f"League {league_name!r} not found.", file=sys.stderr)
                return 2
            league_id = int(row[0])
            owner_user_id = int(row[1])
            is_shared = int(row[2])

            # Exclusive game/team ids (only mapped to this league, not to other leagues).
            cur.execute(
                """
                SELECT game_id
                FROM league_games
                WHERE league_id=%s
                  AND game_id NOT IN (SELECT game_id FROM league_games WHERE league_id<>%s)
                """,
                (league_id, league_id),
            )
            exclusive_game_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})

            cur.execute(
                """
                SELECT team_id
                FROM league_teams
                WHERE league_id=%s
                  AND team_id NOT IN (SELECT team_id FROM league_teams WHERE league_id<>%s)
                """,
                (league_id, league_id),
            )
            exclusive_team_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})

            # Count existing mappings
            cur.execute("SELECT COUNT(*) FROM league_games WHERE league_id=%s", (league_id,))
            mapped_games = int((cur.fetchone() or [0])[0])
            cur.execute("SELECT COUNT(*) FROM league_teams WHERE league_id=%s", (league_id,))
            mapped_teams = int((cur.fetchone() or [0])[0])
            cur.execute("SELECT COUNT(*) FROM league_members WHERE league_id=%s", (league_id,))
            members = int((cur.fetchone() or [0])[0])

            # How many of those teams are safe to delete (external and owned by league owner, and not referenced by remaining games)?
            eligible_team_ids: list[int] = []
            safe_team_ids: list[int] = []
            if exclusive_team_ids:
                q = ",".join(["%s"] * len(exclusive_team_ids))
                cur.execute(
                    f"SELECT id, user_id, is_external FROM teams WHERE id IN ({q})",
                    tuple(exclusive_team_ids),
                )
                team_rows = cur.fetchall() or []
                eligible_team_ids = [
                    int(tid)
                    for (tid, uid, is_ext) in team_rows
                    if int(uid) == owner_user_id and int(is_ext) == 1
                ]

            print(f"League: {league_name!r} (id={league_id}, owner_user_id={owner_user_id}, shared={is_shared})")
            print(f"- Mapped: games={mapped_games} teams={mapped_teams} members={members}")
            print(f"- Delete candidates: games={len(exclusive_game_ids)} teams~={len(eligible_team_ids)}")
            if args.dry_run:
                conn.rollback()
                return 0

            if not args.yes:
                ans = input(f"Type DELETE to permanently wipe league {league_name!r}: ").strip()
                if ans != "DELETE":
                    print("Aborted.")
                    conn.rollback()
                    return 1

            # Remove league row + mappings first; independent of whether we can delete any shared games/teams.
            cur.execute("DELETE FROM league_games WHERE league_id=%s", (league_id,))
            cur.execute("DELETE FROM league_teams WHERE league_id=%s", (league_id,))
            cur.execute("DELETE FROM league_members WHERE league_id=%s", (league_id,))
            cur.execute("DELETE FROM leagues WHERE id=%s", (league_id,))

            # Delete exclusive games (cascades to player_stats, hky_game_stats, player_period_stats).
            for chunk in _chunks(exclusive_game_ids, n=500):
                q = ",".join(["%s"] * len(chunk))
                cur.execute(f"DELETE FROM hky_games WHERE id IN ({q})", tuple(chunk))

            # After deleting the games, delete eligible external teams that are no longer referenced by any remaining games.
            safe_team_ids = []
            if eligible_team_ids:
                q2 = ",".join(["%s"] * len(eligible_team_ids))
                cur.execute(
                    f"""
                    SELECT DISTINCT team1_id AS tid FROM hky_games WHERE team1_id IN ({q2})
                    UNION
                    SELECT DISTINCT team2_id AS tid FROM hky_games WHERE team2_id IN ({q2})
                    """,
                    tuple(eligible_team_ids) * 2,
                )
                still_used = {int(r[0]) for r in (cur.fetchall() or [])}
                safe_team_ids = sorted([tid for tid in eligible_team_ids if tid not in still_used])

            # Delete safe external teams (cascades to players and player_stats via team_id FK).
            for chunk in _chunks(safe_team_ids, n=500):
                q = ",".join(["%s"] * len(chunk))
                cur.execute(f"DELETE FROM teams WHERE id IN ({q})", tuple(chunk))

        conn.commit()
        print(f"Deleted league {league_name!r}.")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Delete a league and its associated hockey data directly from the server DB.

This operates on the *database* used by the deployed webapp (default: /opt/hm-webapp/app/config.json).
It does not use the REST API.

Safety:
- Only deletes hky games/teams that are *not* referenced by other leagues.
- League membership and mapping rows are removed by deleting the league row (FK cascade).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Iterable, Optional


def load_db_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg.get("db", {})


def connect_pymysql(db_cfg: dict):
    try:
        import pymysql
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "pymysql is required; on a deployed server use `/opt/hm-webapp/venv/bin/python` to run this script"
        ) from e

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


def _chunks(seq: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@dataclass(frozen=True)
class PurgePlan:
    league_id: int
    league_name: str
    delete_game_ids: list[int]
    delete_team_ids: list[int]


def compute_purge_plan(
    *,
    league_id: int,
    league_name: str,
    league_game_ids: Iterable[int],
    shared_game_ids: Iterable[int],
    league_team_ids: Iterable[int],
    shared_team_ids: Iterable[int],
    team_ref_counts_after_game_delete: dict[int, int],
) -> PurgePlan:
    league_game_ids_s = {int(x) for x in league_game_ids}
    shared_game_ids_s = {int(x) for x in shared_game_ids}
    league_team_ids_s = {int(x) for x in league_team_ids}
    shared_team_ids_s = {int(x) for x in shared_team_ids}

    delete_game_ids = sorted(league_game_ids_s - shared_game_ids_s)

    delete_team_ids = []
    for tid in sorted(league_team_ids_s - shared_team_ids_s):
        if int(team_ref_counts_after_game_delete.get(int(tid), 0)) == 0:
            delete_team_ids.append(int(tid))

    return PurgePlan(
        league_id=int(league_id),
        league_name=str(league_name),
        delete_game_ids=delete_game_ids,
        delete_team_ids=delete_team_ids,
    )


def _resolve_league(conn, *, league_id: Optional[int], league_name: Optional[str]) -> tuple[int, str]:
    if league_id is None and not league_name:
        raise ValueError("Must pass --league-id or --league-name")

    with conn.cursor() as cur:
        if league_id is not None:
            cur.execute("SELECT id, name FROM leagues WHERE id=%s", (int(league_id),))
        else:
            cur.execute("SELECT id, name FROM leagues WHERE name=%s", (str(league_name),))
        row = cur.fetchone()
        if not row:
            raise ValueError("League not found")
        return int(row[0]), str(row[1])


def _fetch_ids(conn, sql: str, params: tuple) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [int(r[0]) for r in (cur.fetchall() or []) if r and r[0] is not None]


def _fetch_shared_ids(conn, table: str, id_col: str, league_id: int, ids: list[int]) -> list[int]:
    if not ids:
        return []
    shared: set[int] = set()
    with conn.cursor() as cur:
        for chunk in _chunks(ids, 900):
            q = ",".join(["%s"] * len(chunk))
            cur.execute(
                f"SELECT DISTINCT {id_col} FROM {table} WHERE {id_col} IN ({q}) AND league_id<>%s",
                tuple(chunk) + (int(league_id),),
            )
            shared.update(int(r[0]) for r in (cur.fetchall() or []) if r and r[0] is not None)
    return sorted(shared)


def _team_ref_counts_after_game_delete(conn, team_ids: list[int], delete_game_ids: list[int]) -> dict[int, int]:
    if not team_ids:
        return {}
    counts: dict[int, int] = {int(t): 0 for t in team_ids}
    delete_set = {int(g) for g in delete_game_ids}
    with conn.cursor() as cur:
        for chunk in _chunks([int(t) for t in team_ids], 800):
            q = ",".join(["%s"] * len(chunk))
            cur.execute(
                f"""
                SELECT id, team1_id, team2_id
                FROM hky_games
                WHERE team1_id IN ({q}) OR team2_id IN ({q})
                """,
                tuple(chunk) + tuple(chunk),
            )
            for row in cur.fetchall() or []:
                try:
                    gid, t1, t2 = row
                except Exception:
                    continue
                gid_i = int(gid)
                if gid_i in delete_set:
                    continue
                if t1 is not None and int(t1) in counts:
                    counts[int(t1)] += 1
                if t2 is not None and int(t2) in counts:
                    counts[int(t2)] += 1
    return counts


def plan_purge(conn, *, league_id: Optional[int], league_name: Optional[str]) -> PurgePlan:
    lid, lname = _resolve_league(conn, league_id=league_id, league_name=league_name)

    league_game_ids = _fetch_ids(conn, "SELECT DISTINCT game_id FROM league_games WHERE league_id=%s", (lid,))
    shared_game_ids = _fetch_shared_ids(conn, "league_games", "game_id", lid, league_game_ids)
    delete_game_ids = sorted(set(league_game_ids) - set(shared_game_ids))

    league_team_ids = _fetch_ids(conn, "SELECT DISTINCT team_id FROM league_teams WHERE league_id=%s", (lid,))
    shared_team_ids = _fetch_shared_ids(conn, "league_teams", "team_id", lid, league_team_ids)

    ref_counts = _team_ref_counts_after_game_delete(conn, league_team_ids, delete_game_ids)
    return compute_purge_plan(
        league_id=lid,
        league_name=lname,
        league_game_ids=league_game_ids,
        shared_game_ids=shared_game_ids,
        league_team_ids=league_team_ids,
        shared_team_ids=shared_team_ids,
        team_ref_counts_after_game_delete=ref_counts,
    )


def apply_purge(conn, plan: PurgePlan) -> dict:
    stats: dict[str, int] = {
        "delete_games": len(plan.delete_game_ids),
        "delete_teams": len(plan.delete_team_ids),
        "cleared_default_league": 0,
        "deleted_league": 0,
    }
    with conn.cursor() as cur:
        cur.execute("UPDATE users SET default_league_id=NULL WHERE default_league_id=%s", (int(plan.league_id),))
        stats["cleared_default_league"] = int(cur.rowcount or 0)

        # Delete games first (teams are referenced with ON DELETE RESTRICT).
        for chunk in _chunks(plan.delete_game_ids, 500):
            q = ",".join(["%s"] * len(chunk))
            cur.execute(f"DELETE FROM hky_games WHERE id IN ({q})", tuple(int(x) for x in chunk))

        # Delete teams (will cascade players + player_stats via FK).
        for chunk in _chunks(plan.delete_team_ids, 500):
            q = ",".join(["%s"] * len(chunk))
            cur.execute(f"DELETE FROM teams WHERE id IN ({q})", tuple(int(x) for x in chunk))

        # Delete league row last (FK cascades league_members/league_games/league_teams).
        cur.execute("DELETE FROM leagues WHERE id=%s", (int(plan.league_id),))
        stats["deleted_league"] = int(cur.rowcount or 0)

    conn.commit()
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Delete a league and associated hockey data from the server DB")
    ap.add_argument("--config", default="/opt/hm-webapp/app/config.json", help="Webapp config.json with DB cfg")
    ap.add_argument("--league-id", type=int, default=None, help="League id to delete")
    ap.add_argument("--league-name", default=None, help="League name to delete (e.g. Norcal)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be deleted, then exit")
    ap.add_argument("--force", action="store_true", help="Do not prompt for confirmation")
    args = ap.parse_args(argv)

    db_cfg = load_db_cfg(args.config)
    conn = connect_pymysql(db_cfg)

    try:
        plan = plan_purge(conn, league_id=args.league_id, league_name=args.league_name)
    except Exception as e:
        print(f"[!] {e}", file=sys.stderr)
        return 2

    summary = {
        "league_id": plan.league_id,
        "league_name": plan.league_name,
        "delete_games": len(plan.delete_game_ids),
        "delete_teams": len(plan.delete_team_ids),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.dry_run:
        return 0

    if not args.force:
        expected = f"DELETE {plan.league_name}"
        ans = input(f"Type '{expected}' to delete this league and its data: ").strip()
        if ans != expected:
            print("Aborted.")
            return 1

    stats = apply_purge(conn, plan)
    print("Delete complete:", json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

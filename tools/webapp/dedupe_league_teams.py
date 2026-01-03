#!/usr/bin/env python3
"""Merge duplicate-looking teams within a league in the local webapp DB.

TimeToScore sometimes emits team names with invisible differences (NBSP vs space,
unicode hyphen variants, etc). HTML renders those the same, so they look like
duplicates in the league UI. This tool merges such duplicates in-place.

This operates directly on the server DB (no REST API).
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


def normalize_team_name(name: str) -> str:
    t = str(name or "").replace("\xa0", " ").strip()
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2212"):
        t = t.replace(ch, "-")
    t = " ".join(t.split())
    return t.lower()


def clean_team_name(name: str) -> str:
    t = str(name or "").replace("\xa0", " ").strip()
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2212"):
        t = t.replace(ch, "-")
    return " ".join(t.split())


def _chunks(seq: list[int], n: int = 500) -> Iterable[list[int]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@dataclass(frozen=True)
class TeamRow:
    team_id: int
    name: str
    logo_path: Optional[str]
    division_name: Optional[str]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Deduplicate teams within a league by normalized name")
    ap.add_argument("--config", default="/opt/hm-webapp/app/config.json", help="Webapp config.json with DB cfg")
    ap.add_argument("--league-name", required=True, help="League name (exact match)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; do not modify DB")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = ap.parse_args(argv)

    conn = connect_pymysql(load_db_cfg(args.config))
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM leagues WHERE name=%s", (str(args.league_name),))
            row = cur.fetchone()
            if not row:
                print("League not found.", file=sys.stderr)
                return 2
            league_id = int(row[0])

            cur.execute(
                """
                SELECT t.id, t.name, t.logo_path, lt.division_name
                FROM league_teams lt
                JOIN teams t ON lt.team_id=t.id
                WHERE lt.league_id=%s
                """,
                (league_id,),
            )
            teams = [
                TeamRow(int(tid), str(name or ""), (str(lp) if lp else None), (str(dn) if dn else None))
                for (tid, name, lp, dn) in (cur.fetchall() or [])
            ]

        groups: dict[str, list[TeamRow]] = {}
        for t in teams:
            key = normalize_team_name(t.name)
            groups.setdefault(key, []).append(t)

        # Also compute name cleanups (collapse whitespace, normalize dashes) for better UI display.
        cleanups: list[tuple[int, str, str]] = []
        for t in teams:
            cleaned = clean_team_name(t.name)
            if cleaned and cleaned != t.name:
                cleanups.append((t.team_id, t.name, cleaned))

        dup_groups = [(k, v) for k, v in groups.items() if len(v) > 1]
        if not dup_groups and not cleanups:
            print("No duplicate-looking teams found and no names to clean.")
            conn.rollback()
            return 0

        # Build merge plan
        plan: list[tuple[int, list[int], list[str]]] = []
        with conn.cursor() as cur:
            for key, rows in sorted(dup_groups, key=lambda kv: kv[0]):
                team_ids = [r.team_id for r in rows]
                q = ",".join(["%s"] * len(team_ids))
                cur.execute(
                    f"""
                    SELECT t.id,
                           COALESCE(t.logo_path,'') AS logo_path,
                           (SELECT COUNT(*) FROM players p WHERE p.team_id=t.id) AS players_n,
                           (SELECT COUNT(*) FROM hky_games g WHERE g.team1_id=t.id OR g.team2_id=t.id) AS games_n
                    FROM teams t
                    WHERE t.id IN ({q})
                    """,
                    tuple(team_ids),
                )
                scored: list[tuple[int, int, int, int]] = []
                for tid, logo_path, players_n, games_n in (cur.fetchall() or []):
                    has_logo = 1 if str(logo_path or "").strip() else 0
                    scored.append((int(tid), int(has_logo), int(players_n or 0), int(games_n or 0)))
                # Prefer: has_logo, then most games, then most players, then lowest id (stable).
                scored.sort(key=lambda x: (-x[1], -x[3], -x[2], x[0]))
                keep_id = int(scored[0][0])
                merge_ids = sorted([int(t) for t in team_ids if int(t) != keep_id])
                names = sorted({r.name for r in rows})
                plan.append((keep_id, merge_ids, names))

        print(f"League {args.league_name!r} (id={league_id})")
        if plan:
            print(f"- Duplicate groups: {len(plan)}")
            for keep_id, merge_ids, names in plan:
                print(f"  - Keep team_id={keep_id}, merge={merge_ids} names={names}")
        if cleanups:
            print(f"- Name cleanups: {len(cleanups)}")
            for tid, old, new in cleanups[:25]:
                print(f"  - team_id={tid}: {old!r} -> {new!r}")
            if len(cleanups) > 25:
                print(f"  ... (+{len(cleanups) - 25} more)")

        if args.dry_run:
            conn.rollback()
            return 0

        if not args.yes:
            ans = input("Type MERGE to apply this dedupe plan: ").strip()
            if ans != "MERGE":
                print("Aborted.")
                conn.rollback()
                return 1

        with conn.cursor() as cur:
            # Clean names first (dedupe grouping already accounted for these variants).
            for tid, _old, new in cleanups:
                cur.execute("UPDATE teams SET name=%s WHERE id=%s", (new, int(tid)))

            for keep_id, merge_ids, _names in plan:
                for old_id in merge_ids:
                    # Update game team references
                    cur.execute("UPDATE hky_games SET team1_id=%s WHERE team1_id=%s", (keep_id, old_id))
                    cur.execute("UPDATE hky_games SET team2_id=%s WHERE team2_id=%s", (keep_id, old_id))
                    # Update player/team references
                    cur.execute("UPDATE players SET team_id=%s WHERE team_id=%s", (keep_id, old_id))
                    cur.execute("UPDATE player_stats SET team_id=%s WHERE team_id=%s", (keep_id, old_id))
                    # Remove league mapping for old team
                    cur.execute("DELETE FROM league_teams WHERE league_id=%s AND team_id=%s", (league_id, old_id))
                    # Finally delete the team (should cascade nothing critical after re-pointing)
                    cur.execute("DELETE FROM teams WHERE id=%s", (old_id,))

        conn.commit()
        print("Dedupe complete.")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

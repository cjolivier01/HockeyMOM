#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional


def _load_db_cfg(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return dict(cfg.get("db", {}) or {})


def _connect_pymysql(db_cfg: dict[str, Any]):
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


def _list_league_ids(conn) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM leagues ORDER BY id")
        return [int(r[0]) for r in (cur.fetchall() or []) if r and r[0] is not None]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute Ratings for leagues (weekly job).")
    ap.add_argument(
        "--config",
        default=os.environ.get("HM_DB_CONFIG") or "/opt/hm-webapp/app/config.json",
        help="Path to webapp config.json (default: HM_DB_CONFIG or /opt/hm-webapp/app/config.json).",
    )
    ap.add_argument("--league-id", type=int, default=0, help="If set, recompute only this league id.")
    args = ap.parse_args(argv)

    # Import from deployed app module (this is intentionally coupled to the webapp code).
    import app as webapp_app  # type: ignore

    db_cfg = _load_db_cfg(args.config)
    conn = _connect_pymysql(db_cfg)
    try:
        league_ids = [int(args.league_id)] if int(args.league_id) > 0 else _list_league_ids(conn)
        if not league_ids:
            print("[i] No leagues found; nothing to do.")
            return 0
        ok = 0
        fail = 0
        for lid in league_ids:
            try:
                webapp_app.recompute_league_mhr_ratings(conn, int(lid))
                ok += 1
                print(f"[ok] Recomputed Ratings for league_id={lid}")
            except Exception as e:  # noqa: BLE001
                try:
                    conn.rollback()
                except Exception:
                    pass
                fail += 1
                print(f"[!] Failed recompute for league_id={lid}: {e}")
        if fail:
            return 2
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

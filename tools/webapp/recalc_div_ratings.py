#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional


def _orm_modules(*, config_path: str):
    try:
        from tools.webapp import django_orm  # type: ignore
    except Exception:  # pragma: no cover
        import django_orm  # type: ignore

    django_orm.setup_django(config_path=config_path)
    django_orm.ensure_schema()
    try:
        from tools.webapp.django_app import models as m  # type: ignore
    except Exception:  # pragma: no cover
        from django_app import models as m  # type: ignore

    return django_orm, m


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

    _django_orm, m = _orm_modules(config_path=str(args.config))

    league_ids = [int(args.league_id)] if int(args.league_id) > 0 else list(m.League.objects.order_by("id").values_list("id", flat=True))
    if not league_ids:
        print("[i] No leagues found; nothing to do.")
        return 0

    ok = 0
    fail = 0
    for lid in league_ids:
        try:
            webapp_app.recompute_league_mhr_ratings(None, int(lid))
            ok += 1
            print(f"[ok] Recomputed Ratings for league_id={lid}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[!] Failed recompute for league_id={lid}: {e}")
    if fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

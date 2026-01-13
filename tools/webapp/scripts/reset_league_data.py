#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Optional
from urllib.parse import urljoin


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


def wipe_all(m) -> dict:
    counts: dict = {}
    counts["player_stats"] = int(m.PlayerStat.objects.count())
    counts["league_games"] = int(m.LeagueGame.objects.count())
    counts["hky_games"] = int(m.HkyGame.objects.count())
    counts["league_teams"] = int(m.LeagueTeam.objects.count())
    counts["players"] = int(m.Player.objects.count())
    counts["teams"] = int(m.Team.objects.count())

    from django.db import transaction

    with transaction.atomic():
        # Delete in FK-safe order (teams are referenced with ON DELETE RESTRICT from hky_games).
        m.PlayerStat.objects.all().delete()
        m.PlayerPeriodStat.objects.all().delete()
        m.HkyGameEventSuppression.objects.all().delete()
        m.HkyGamePlayer.objects.all().delete()
        m.HkyGameEventRow.objects.all().delete()
        m.LeagueGame.objects.all().delete()
        m.HkyGame.objects.all().delete()
        m.LeagueTeam.objects.all().delete()
        m.Player.objects.all().delete()
        m.Team.objects.all().delete()
    return counts


def wipe_league(m, league_id: int) -> dict:
    try:
        from tools.webapp import app as webapp_app  # type: ignore
    except Exception:  # pragma: no cover
        import app as webapp_app  # type: ignore

    # Use the webapp's safer logic (only deletes exclusive games/teams to avoid impacting other leagues).
    del m
    return webapp_app.reset_league_data(None, int(league_id))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reset imported hockey data without touching users or permissions"
    )
    ap.add_argument("--config", default="/opt/hm-webapp/app/config.json")
    ap.add_argument("--force", action="store_true", help="Do not prompt for confirmation")
    ap.add_argument("--yes", "-y", action="store_true", help="Alias for --force")
    ap.add_argument(
        "--webapp-url",
        default=None,
        help="If set, reset via webapp REST API (e.g. http://127.0.0.1:8008)",
    )
    ap.add_argument("--webapp-token", default=None, help="Optional import token for REST mode")
    ap.add_argument(
        "--import-token", dest="webapp_token", default=None, help="Alias for --webapp-token"
    )
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
            ans = input(
                f"This will reset hockey data for {scope} via REST. Type RESET to continue: "
            ).strip()
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
            json={
                "owner_email": str(args.webapp_owner_email).strip(),
                "league_name": str(args.league_name).strip(),
            },
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

    _django_orm, m = _orm_modules(config_path=str(args.config))

    # Resolve league id if name provided
    league_id = args.league_id
    if args.league_name and not league_id:
        row = (
            m.League.objects.filter(name=str(args.league_name)).values_list("id", flat=True).first()
        )
        if row is None:
            print(f"[!] League named '{args.league_name}' not found", file=sys.stderr)
            return 2
        league_id = int(row)

    scope = f"league_id={league_id}" if league_id else "ALL"
    if not force:
        ans = input(
            f"This will wipe teams/players/hky games/stats for {scope}. Type RESET to continue: "
        ).strip()
        if ans != "RESET":
            print("Aborted.")
            return 1

    if league_id:
        stats = wipe_league(m, int(league_id))
        print(
            "Wiped league data:",
            json.dumps(stats, indent=2, sort_keys=True),
        )
    else:
        counts = wipe_all(m)
        print("Existing rows:", json.dumps(counts, indent=2, sort_keys=True))
        print("Wipe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

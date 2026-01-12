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
import os
import sys
from pathlib import Path
from typing import Iterable


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


def _chunks(ids: list[int], n: int = 500) -> Iterable[list[int]]:
    for i in range(0, len(ids), n):
        yield ids[i : i + n]


def main(argv: list[str] | None = None) -> int:
    base_dir = Path(__file__).resolve().parents[1]
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

    _django_orm, m = _orm_modules(config_path=str(args.config))
    from django.db import transaction
    from django.db.models import Q

    league_row = m.League.objects.filter(name=league_name).values("id", "owner_user_id", "is_shared").first()
    if not league_row:
        print(f"League {league_name!r} not found.", file=sys.stderr)
        return 2
    league_id = int(league_row["id"])
    owner_user_id = int(league_row["owner_user_id"])
    is_shared = int(1 if league_row.get("is_shared") else 0)

    league_game_ids = list(m.LeagueGame.objects.filter(league_id=league_id).values_list("game_id", flat=True))
    other_game_ids = set()
    if league_game_ids:
        other_game_ids = set(
            m.LeagueGame.objects.exclude(league_id=league_id).filter(game_id__in=league_game_ids).values_list("game_id", flat=True)
        )
    exclusive_game_ids = sorted({int(gid) for gid in league_game_ids if gid is not None and gid not in other_game_ids})

    league_team_ids = list(m.LeagueTeam.objects.filter(league_id=league_id).values_list("team_id", flat=True))
    other_team_ids = set()
    if league_team_ids:
        other_team_ids = set(
            m.LeagueTeam.objects.exclude(league_id=league_id).filter(team_id__in=league_team_ids).values_list("team_id", flat=True)
        )
    exclusive_team_ids = sorted({int(tid) for tid in league_team_ids if tid is not None and tid not in other_team_ids})

    mapped_games = m.LeagueGame.objects.filter(league_id=league_id).count()
    mapped_teams = m.LeagueTeam.objects.filter(league_id=league_id).count()
    members = m.LeagueMember.objects.filter(league_id=league_id).count()

    eligible_team_ids: list[int] = []
    if exclusive_team_ids:
        eligible_team_ids = list(
            m.Team.objects.filter(id__in=exclusive_team_ids, user_id=owner_user_id, is_external=True).values_list("id", flat=True)
        )

    print(f"League: {league_name!r} (id={league_id}, owner_user_id={owner_user_id}, shared={is_shared})")
    print(f"- Mapped: games={mapped_games} teams={mapped_teams} members={members}")
    print(f"- Delete candidates: games={len(exclusive_game_ids)} teams~={len(eligible_team_ids)}")
    if args.dry_run:
        return 0

    if not args.yes:
        ans = input(f"Type DELETE to permanently wipe league {league_name!r}: ").strip()
        if ans != "DELETE":
            print("Aborted.")
            return 1

    with transaction.atomic():
        # Clear default league for any users (best-effort: DB constraints may not exist).
        m.User.objects.filter(default_league_id=league_id).update(default_league=None)

        # Remove league row + mappings (FK cascades league_members/league_games/league_teams).
        m.League.objects.filter(id=league_id).delete()

        # Delete exclusive games (cascades to player_stats and hky_game_*).
        if exclusive_game_ids:
            for chunk in _chunks(exclusive_game_ids, n=500):
                m.HkyGame.objects.filter(id__in=chunk).delete()

        # After deleting the games, delete eligible external teams that are no longer referenced by remaining games.
        safe_team_ids: list[int] = []
        if eligible_team_ids:
            still_used: set[int] = set()
            for team1_id, team2_id in m.HkyGame.objects.filter(
                Q(team1_id__in=eligible_team_ids) | Q(team2_id__in=eligible_team_ids)
            ).values_list("team1_id", "team2_id"):
                if team1_id in eligible_team_ids:
                    still_used.add(int(team1_id))
                if team2_id in eligible_team_ids:
                    still_used.add(int(team2_id))
            safe_team_ids = sorted([int(tid) for tid in eligible_team_ids if int(tid) not in still_used])

        if safe_team_ids:
            for chunk in _chunks(safe_team_ids, n=500):
                m.Team.objects.filter(id__in=chunk).delete()

    print(f"Deleted league {league_name!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

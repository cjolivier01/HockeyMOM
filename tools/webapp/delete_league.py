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


def _resolve_league(m, *, league_id: Optional[int], league_name: Optional[str]) -> tuple[int, str]:
    if league_id is None and not league_name:
        raise ValueError("Must pass --league-id or --league-name")

    if league_id is not None:
        row = m.League.objects.filter(id=int(league_id)).values_list("id", "name").first()
    else:
        row = m.League.objects.filter(name=str(league_name)).values_list("id", "name").first()
    if not row:
        raise ValueError("League not found")
    return int(row[0]), str(row[1])


def _team_ref_counts_after_game_delete(m, team_ids: list[int], delete_game_ids: list[int]) -> dict[int, int]:
    if not team_ids:
        return {}
    counts: dict[int, int] = {int(t): 0 for t in team_ids}
    delete_set = {int(g) for g in delete_game_ids}
    from django.db.models import Q

    for t1, t2, gid in m.HkyGame.objects.filter(
        Q(team1_id__in=team_ids) | Q(team2_id__in=team_ids)
    ).values_list("team1_id", "team2_id", "id"):
        gid_i = int(gid)
        if gid_i in delete_set:
            continue
        if t1 is not None and int(t1) in counts:
            counts[int(t1)] += 1
        if t2 is not None and int(t2) in counts:
            counts[int(t2)] += 1
    return counts


def plan_purge(m, *, league_id: Optional[int], league_name: Optional[str]) -> PurgePlan:
    lid, lname = _resolve_league(m, league_id=league_id, league_name=league_name)

    league_game_ids = list(m.LeagueGame.objects.filter(league_id=lid).values_list("game_id", flat=True))
    shared_game_ids = set(
        m.LeagueGame.objects.exclude(league_id=lid).filter(game_id__in=league_game_ids).values_list("game_id", flat=True)
    )
    delete_game_ids = sorted({int(g) for g in league_game_ids if g is not None and g not in shared_game_ids})

    league_team_ids = list(m.LeagueTeam.objects.filter(league_id=lid).values_list("team_id", flat=True))
    shared_team_ids = set(
        m.LeagueTeam.objects.exclude(league_id=lid).filter(team_id__in=league_team_ids).values_list("team_id", flat=True)
    )

    ref_counts = _team_ref_counts_after_game_delete(m, [int(t) for t in league_team_ids if t is not None], delete_game_ids)
    return compute_purge_plan(
        league_id=lid,
        league_name=lname,
        league_game_ids=league_game_ids,
        shared_game_ids=shared_game_ids,
        league_team_ids=league_team_ids,
        shared_team_ids=shared_team_ids,
        team_ref_counts_after_game_delete=ref_counts,
    )


def apply_purge(m, plan: PurgePlan) -> dict:
    stats: dict[str, int] = {
        "delete_games": len(plan.delete_game_ids),
        "delete_teams": len(plan.delete_team_ids),
        "cleared_default_league": 0,
        "deleted_league": 0,
    }
    from django.db import transaction

    with transaction.atomic():
        stats["cleared_default_league"] = int(
            m.User.objects.filter(default_league_id=int(plan.league_id)).update(default_league=None)
        )

        # Delete games first (teams are referenced with ON DELETE RESTRICT).
        for chunk in _chunks(plan.delete_game_ids, 500):
            m.HkyGame.objects.filter(id__in=[int(x) for x in chunk]).delete()

        # Delete teams (will cascade players + player_stats via FK).
        for chunk in _chunks(plan.delete_team_ids, 500):
            m.Team.objects.filter(id__in=[int(x) for x in chunk]).delete()

        # Delete league-associated rows even if DB constraints were created without FK cascades.
        m.LeagueMember.objects.filter(league_id=int(plan.league_id)).delete()
        m.LeaguePageView.objects.filter(league_id=int(plan.league_id)).delete()
        m.LeagueGame.objects.filter(league_id=int(plan.league_id)).delete()
        m.LeagueTeam.objects.filter(league_id=int(plan.league_id)).delete()

        deleted, _details = m.League.objects.filter(id=int(plan.league_id)).delete()
        stats["deleted_league"] = int(deleted or 0)
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Delete a league and associated hockey data from the server DB")
    ap.add_argument("--config", default="/opt/hm-webapp/app/config.json", help="Webapp config.json with DB cfg")
    ap.add_argument("--league-id", type=int, default=None, help="League id to delete")
    ap.add_argument("--league-name", default=None, help="League name to delete (e.g. Norcal)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be deleted, then exit")
    ap.add_argument("--force", action="store_true", help="Do not prompt for confirmation")
    args = ap.parse_args(argv)

    _django_orm, m = _orm_modules(config_path=str(args.config))

    try:
        plan = plan_purge(m, league_id=args.league_id, league_name=args.league_name)
    except Exception as e:  # noqa: BLE001
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

    stats = apply_purge(m, plan)
    print("Delete complete:", json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

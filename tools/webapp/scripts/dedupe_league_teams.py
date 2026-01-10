#!/usr/bin/env python3
"""Merge duplicate-looking teams within a league in the local webapp DB.

TimeToScore sometimes emits team names with invisible differences (NBSP vs space,
unicode hyphen variants, etc). HTML renders those the same, so they look like
duplicates in the league UI. This tool merges such duplicates in-place.

This operates directly on the webapp DB (no REST API).
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

    _django_orm, m = _orm_modules(config_path=str(args.config))
    from django.db import transaction
    from django.db.models import Count

    league_id = m.League.objects.filter(name=str(args.league_name)).values_list("id", flat=True).first()
    if league_id is None:
        print("League not found.", file=sys.stderr)
        return 2
    league_id = int(league_id)

    teams = [
        TeamRow(
            team_id=int(r["team_id"]),
            name=str(r.get("team__name") or ""),
            logo_path=(str(r.get("team__logo_path")) if r.get("team__logo_path") else None),
            division_name=(str(r.get("division_name")) if r.get("division_name") else None),
        )
        for r in m.LeagueTeam.objects.filter(league_id=league_id)
        .select_related("team")
        .values("team_id", "division_name", "team__name", "team__logo_path")
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
        return 0

    # Build merge plan
    plan: list[tuple[int, list[int], list[str]]] = []
    all_team_ids = sorted({int(t.team_id) for t in teams})
    players_n_by_team = dict(
        m.Player.objects.filter(team_id__in=all_team_ids).values("team_id").annotate(n=Count("id")).values_list("team_id", "n")
    )
    games_n_by_team: dict[int, int] = {int(tid): 0 for tid in all_team_ids}
    for tid, n in (
        m.HkyGame.objects.filter(team1_id__in=all_team_ids).values("team1_id").annotate(n=Count("id")).values_list("team1_id", "n")
    ):
        games_n_by_team[int(tid)] = games_n_by_team.get(int(tid), 0) + int(n or 0)
    for tid, n in (
        m.HkyGame.objects.filter(team2_id__in=all_team_ids).values("team2_id").annotate(n=Count("id")).values_list("team2_id", "n")
    ):
        games_n_by_team[int(tid)] = games_n_by_team.get(int(tid), 0) + int(n or 0)

    logo_by_team = {int(t.team_id): str(t.logo_path or "") for t in teams}

    for _key, rows in sorted(dup_groups, key=lambda kv: kv[0]):
        team_ids = [int(r.team_id) for r in rows]
        scored: list[tuple[int, int, int, int]] = []
        for tid in team_ids:
            has_logo = 1 if str(logo_by_team.get(int(tid), "")).strip() else 0
            scored.append(
                (int(tid), int(has_logo), int(players_n_by_team.get(int(tid), 0)), int(games_n_by_team.get(int(tid), 0)))
            )
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
        return 0

    if not args.yes:
        ans = input("Type MERGE to apply this dedupe plan: ").strip()
        if ans != "MERGE":
            print("Aborted.")
            return 1

    with transaction.atomic():
        # Clean names first (dedupe grouping already accounted for these variants).
        for tid, _old, new in cleanups:
            m.Team.objects.filter(id=int(tid)).update(name=str(new))

        for keep_id, merge_ids, _names in plan:
            for old_id in merge_ids:
                # Update game team references
                m.HkyGame.objects.filter(team1_id=int(old_id)).update(team1_id=int(keep_id))
                m.HkyGame.objects.filter(team2_id=int(old_id)).update(team2_id=int(keep_id))
                # Update player/team references
                m.Player.objects.filter(team_id=int(old_id)).update(team_id=int(keep_id))
                m.PlayerStat.objects.filter(team_id=int(old_id)).update(team_id=int(keep_id))
                # Remove league mapping for old team
                m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(old_id)).delete()
                # Finally delete the team (should cascade nothing critical after re-pointing)
                m.Team.objects.filter(id=int(old_id)).delete()

    print("Dedupe complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

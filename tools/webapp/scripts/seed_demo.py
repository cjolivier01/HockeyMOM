#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import random
import string
import sys
from pathlib import Path
from typing import Any, Optional


def _orm_modules(*, config_path: str):
    try:
        from tools.webapp import django_orm  # type: ignore
    except Exception:  # pragma: no cover
        import django_orm  # type: ignore

    django_orm.setup_django(config_path=config_path)
    django_orm.ensure_schema()
    django_orm.ensure_bootstrap_data()
    try:
        from tools.webapp.django_app import models as m  # type: ignore
    except Exception:  # pragma: no cover
        from django_app import models as m  # type: ignore

    return django_orm, m


def ensure_user(m: Any, *, email: str, name: str, password_hash: str) -> int:
    now = dt.datetime.now()
    user, _created = m.User.objects.get_or_create(
        email=str(email).strip().lower(),
        defaults={"password_hash": password_hash, "name": name, "created_at": now},
    )
    return int(user.id)


def ensure_game_type(m: Any, *, name: str) -> Optional[int]:
    if not name:
        return None
    gt, _created = m.GameType.objects.get_or_create(name=str(name), defaults={"is_default": False})
    return int(gt.id)


def create_team(m: Any, *, user_id: int, name: str, is_external: bool = False) -> int:
    now = dt.datetime.now()
    team, _created = m.Team.objects.get_or_create(
        user_id=int(user_id),
        name=str(name),
        defaults={"is_external": bool(is_external), "created_at": now},
    )
    if bool(is_external) and not bool(team.is_external):
        m.Team.objects.filter(id=team.id).update(is_external=True, updated_at=now)
    return int(team.id)


def create_player(m: Any, *, user_id: int, team_id: int, name: str, jersey: str, position: str) -> int:
    now = dt.datetime.now()
    p = m.Player.objects.create(
        user_id=int(user_id),
        team_id=int(team_id),
        name=str(name),
        jersey_number=str(jersey),
        position=str(position),
        created_at=now,
    )
    return int(p.id)


def create_game(
    m: Any,
    *,
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_name: str,
    starts_at: dt.datetime,
    location: str,
    score: tuple | None = None,
) -> int:
    now = dt.datetime.now()
    gt_id = ensure_game_type(m, name=game_type_name)
    g = m.HkyGame.objects.create(
        user_id=int(user_id),
        team1_id=int(team1_id),
        team2_id=int(team2_id),
        game_type_id=gt_id,
        starts_at=starts_at,
        location=location,
        team1_score=score[0] if score else None,
        team2_score=score[1] if score else None,
        is_final=bool(score),
        created_at=now,
    )
    return int(g.id)


def add_random_stats(m: Any, *, game_id: int, team_id: int, user_id: int) -> None:
    import random as _r

    pids = list(m.Player.objects.filter(team_id=int(team_id)).order_by("id").values_list("id", flat=True))
    for pid in pids:
        goals = _r.randint(0, 2)
        assists = _r.randint(0, 2)
        shots = goals + assists + _r.randint(0, 3)
        pim = _r.choice([0, 0, 2, 4])
        plus_minus = _r.randint(-1, 2)
        ps, created = m.PlayerStat.objects.get_or_create(
            game_id=int(game_id),
            player_id=int(pid),
            defaults={
                "user_id": int(user_id),
                "team_id": int(team_id),
                "goals": int(goals),
                "assists": int(assists),
                "shots": int(shots),
                "pim": int(pim),
                "plus_minus": int(plus_minus),
            },
        )
        if not created:
            m.PlayerStat.objects.filter(id=ps.id).update(
                goals=int(goals),
                assists=int(assists),
                shots=int(shots),
                pim=int(pim),
                plus_minus=int(plus_minus),
            )


def main():
    ap = argparse.ArgumentParser(description="Seed demo data for HockeyMOM WebApp hockey features")
    ap.add_argument(
        "--config",
        default=os.environ.get("HM_DB_CONFIG")
        or str((Path(__file__).resolve().parents[1] / "config.json")),
    )
    ap.add_argument("--email", default="demo@example.com")
    ap.add_argument("--name", default="Demo User")
    ap.add_argument(
        "--password-hash",
        default="pbkdf2:sha256:260000$demo$Yy6lWp5oSz5Ahh3yI9sRhW/9k5D5mZ0t8Xr6Z3YYc2U=",
    )
    ap.add_argument("--teams", nargs="*", default=["Thunderbirds 12U", "Falcons 12U"])
    args = ap.parse_args()

    try:
        _django_orm, m = _orm_modules(config_path=str(args.config))
    except Exception:
        print("Failed to initialize ORM. Ensure Django is installed and DB configured.", file=sys.stderr)
        raise

    user_id = ensure_user(m, email=args.email, name=args.name, password_hash=args.password_hash)

    # Teams
    team_ids = []
    for nm in args.teams:
        team_ids.append(create_team(m, user_id=user_id, name=nm, is_external=False))
    ext_id = create_team(m, user_id=user_id, name="Ice Wolves", is_external=True)

    # Players
    def gen_name():
        return "Player " + "".join(random.choice(string.ascii_uppercase) for _ in range(3))

    positions = ["F", "D", "G"]
    num = 1
    for tid in team_ids:
        for i in range(10):
            create_player(m, user_id=user_id, team_id=tid, name=gen_name(), jersey=str(num), position=random.choice(positions))
            num += 1

    # Games
    now = dt.datetime.now()
    g1 = create_game(
        m,
        user_id=user_id,
        team1_id=team_ids[0],
        team2_id=team_ids[1],
        game_type_name="Regular Season",
        starts_at=now - dt.timedelta(days=1),
        location="Main Rink",
        score=(3, 2),
    )
    create_game(
        m,
        user_id=user_id,
        team1_id=team_ids[0],
        team2_id=ext_id,
        game_type_name="Exhibition",
        starts_at=now + dt.timedelta(days=2),
        location="Community Ice",
        score=None,
    )

    # Stats for completed game
    add_random_stats(m, game_id=g1, team_id=team_ids[0], user_id=user_id)
    add_random_stats(m, game_id=g1, team_id=team_ids[1], user_id=user_id)

    print("Seeded demo data.")


if __name__ == "__main__":
    main()

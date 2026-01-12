from __future__ import annotations

import datetime as dt
import importlib.util
import os


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app_mhr_group", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_disqualify_team_without_any_games_in_its_age_group(webapp_db):
    _django_orm, m = webapp_db

    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    now = dt.datetime(2025, 1, 1, 0, 0, 0)
    user = m.User.objects.create(
        id=123,
        email="u123@example.com",
        password_hash="x",
        name="Test User",
        created_at=now,
    )
    league = m.League.objects.create(id=999, name="Test League", owner_user=user, created_at=now)

    t_aa = m.Team.objects.create(id=1, user=user, name="Team 12AA", created_at=now)
    t_aa_2 = m.Team.objects.create(id=2, user=user, name="Team 12AA-2", created_at=now)
    t_aaa = m.Team.objects.create(id=3, user=user, name="Team 12AAA", created_at=now)
    t_aaa_2 = m.Team.objects.create(id=4, user=user, name="Team 12AAA-2", created_at=now)

    m.LeagueTeam.objects.bulk_create(
        [
            m.LeagueTeam(league=league, team=t_aa, division_name="12 AA"),
            m.LeagueTeam(league=league, team=t_aa_2, division_name="12 AA"),
            m.LeagueTeam(league=league, team=t_aaa, division_name="12 AAA"),
            m.LeagueTeam(league=league, team=t_aaa_2, division_name="12 AAA"),
        ]
    )

    regular = m.GameType.objects.get(name="Regular Season")

    # Cross-level games (AA vs AAA) do not count as being in the team's age group.
    g1 = m.HkyGame.objects.create(
        user=user,
        team1=t_aa,
        team2=t_aaa,
        team1_score=1,
        team2_score=2,
        is_final=True,
        created_at=now,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league=league, game=g1, division_name="12 AAA")

    g2 = m.HkyGame.objects.create(
        user=user,
        team1=t_aaa,
        team2=t_aa,
        team1_score=3,
        team2_score=2,
        is_final=True,
        created_at=now,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league=league, game=g2, division_name="12 AAA")

    # Same-group games (AAA vs AAA) make AAA teams eligible for ratings.
    g3 = m.HkyGame.objects.create(
        user=user,
        team1=t_aaa,
        team2=t_aaa_2,
        team1_score=2,
        team2_score=1,
        is_final=True,
        created_at=now,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league=league, game=g3, division_name="12 AAA")

    g4 = m.HkyGame.objects.create(
        user=user,
        team1=t_aaa_2,
        team2=t_aaa,
        team1_score=1,
        team2_score=4,
        is_final=True,
        created_at=now,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league=league, game=g4, division_name="12 AAA")

    mod.recompute_league_mhr_ratings(None, league_id=int(league.id))

    lt_aa = m.LeagueTeam.objects.get(league_id=int(league.id), team_id=int(t_aa.id))
    assert lt_aa.mhr_rating is None
    assert int(lt_aa.mhr_games or 0) == 0

    lt_aaa = m.LeagueTeam.objects.get(league_id=int(league.id), team_id=int(t_aaa.id))
    assert lt_aaa.mhr_rating is not None

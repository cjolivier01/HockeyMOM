from __future__ import annotations

import importlib.util
import datetime as dt
import os


def _load_app_module():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_ignore_cross_division_non_external_games_in_league_player_totals(webapp_db):
    _django_orm, m = webapp_db
    mod = _load_app_module()

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    league = m.League.objects.create(
        id=99,
        name="L",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )

    team1 = m.Team.objects.create(
        id=1,
        user_id=int(owner.id),
        name="Self",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=2,
        user_id=int(owner.id),
        name="Opp",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team3 = m.Team.objects.create(
        id=3,
        user_id=int(owner.id),
        name="SameDiv",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team1.id), division_name="12AA"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team2.id), division_name="12A"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team3.id), division_name="12AA"
    )

    g10 = m.HkyGame.objects.create(
        id=10,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=1,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g11 = m.HkyGame.objects.create(
        id=11,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=3,
        team2_score=3,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g12 = m.HkyGame.objects.create(
        id=12,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team3.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=2,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g10.id), division_name="12AA", sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g11.id), division_name="External", sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g12.id), division_name="12AA", sort_order=None
    )

    player = m.Player.objects.create(
        id=100,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="P",
        jersey_number="9",
        position=None,
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.PlayerStat.objects.create(
        game_id=int(g10.id),
        player_id=int(player.id),
        user_id=int(owner.id),
        team_id=int(team1.id),
        goals=1,
        assists=0,
    )
    m.PlayerStat.objects.create(
        game_id=int(g11.id),
        player_id=int(player.id),
        user_id=int(owner.id),
        team_id=int(team1.id),
        goals=2,
        assists=1,
    )
    m.PlayerStat.objects.create(
        game_id=int(g12.id),
        player_id=int(player.id),
        user_id=int(owner.id),
        team_id=int(team1.id),
        goals=0,
        assists=2,
    )

    totals = mod.aggregate_players_totals_league(
        None, team_id=int(team1.id), league_id=int(league.id)
    )
    p100 = totals[int(player.id)]
    # Cross-division games (e.g. 12AA vs 12A) are included in league totals.
    assert p100["gp"] == 3
    assert p100["goals"] == 3  # game 10 (1) + game 11 (2)
    assert p100["assists"] == 3  # game 11 (1) + game 12 (2)

from __future__ import annotations

import datetime as dt

import pytest


@pytest.fixture()
def client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from django.test import Client

    now = dt.datetime.now()
    user = m.User.objects.create(email="u@example.com", password_hash="x", name="U", created_at=now)
    c = Client()
    sess = c.session
    sess["user_id"] = int(user.id)
    sess["user_email"] = "u@example.com"
    sess.save()
    return c


def should_gray_out_tournament_blowout_and_exclude_from_stats(client, webapp_db):
    _django_orm, m = webapp_db
    now = dt.datetime.now()
    user = m.User.objects.get(email="u@example.com")

    league = m.League.objects.create(
        id=42,
        name="L",
        owner_user_id=int(user.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    team_a = m.Team.objects.create(
        id=1,
        user_id=int(user.id),
        name="Team A",
        is_external=False,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=2,
        user_id=int(user.id),
        name="Team B",
        is_external=False,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_a.id), division_name="12AA"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_b.id), division_name="12AA"
    )

    regular = m.GameType.objects.get(name="Regular Season")
    tournament = m.GameType.objects.get(name="Tournament")

    g_regular = m.HkyGame.objects.create(
        id=101,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(regular.id),
        starts_at=None,
        location="",
        notes=None,
        team1_score=2,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_regular.id), division_name="12AA"
    )

    g_blowout = m.HkyGame.objects.create(
        id=102,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(tournament.id),
        starts_at=None,
        location="",
        notes=None,
        team1_score=10,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_blowout.id), division_name="12AA"
    )

    sess = client.session
    sess["league_id"] = int(league.id)
    sess.save()

    resp = client.get("/teams/1")
    assert resp.status_code == 200
    html = resp.content.decode()
    assert "Tournament blowout" in html
    assert html.count('class="game-excluded"') == 1
    assert "Record: <strong>1-0-0</strong>" in html
    assert "GF: 2" in html
    assert "GA: 1" in html


def should_gray_out_caha_preseason_relegation_game_and_exclude_from_stats(client, webapp_db):
    _django_orm, m = webapp_db
    now = dt.datetime.now()
    user = m.User.objects.get(email="u@example.com")

    league = m.League.objects.create(
        id=42,
        name="CAHA",
        owner_user_id=int(user.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    team_a = m.Team.objects.create(
        id=1,
        user_id=int(user.id),
        name="Team A",
        is_external=False,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=2,
        user_id=int(user.id),
        name="Team B",
        is_external=False,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    # Team A's final placement is 12A; preseason game played in 12AA should be excluded.
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_a.id), division_name="12A"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_b.id), division_name="12A"
    )

    preseason = m.GameType.objects.get(name="Preseason")
    regular = m.GameType.objects.get(name="Regular Season")

    g_pre_relegated = m.HkyGame.objects.create(
        id=201,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(preseason.id),
        starts_at=None,
        location="",
        notes=None,
        team1_score=3,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_pre_relegated.id), division_name="12AA"
    )

    g_pre_ok = m.HkyGame.objects.create(
        id=202,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(preseason.id),
        starts_at=None,
        location="",
        notes=None,
        team1_score=1,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_pre_ok.id), division_name="12A"
    )

    g_regular = m.HkyGame.objects.create(
        id=203,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(regular.id),
        starts_at=None,
        location="",
        notes=None,
        team1_score=2,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_regular.id), division_name="12A"
    )

    sess = client.session
    sess["league_id"] = int(league.id)
    sess.save()

    resp = client.get("/teams/1")
    assert resp.status_code == 200
    html = resp.content.decode()
    assert "CAHA preseason" in html
    assert html.count('class="game-excluded"') == 1
    # Included games: preseason tie + regular win.
    assert "Record: <strong>1-0-1</strong>" in html
    assert "GF: 3" in html
    assert "GA: 1" in html

from __future__ import annotations

import datetime as dt

import pytest


@pytest.fixture()
def client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from django.test import Client

    from tools.webapp import app as logic

    # Avoid needing to assert standings/league-wide player totals in this unit test.
    monkeypatch.setattr(
        logic,
        "compute_team_stats_league",
        lambda *_args, **_kwargs: {
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "gf": 0,
            "ga": 0,
            "points": 0,
        },
    )
    monkeypatch.setattr(logic, "aggregate_players_totals_league", lambda *_args, **_kwargs: {})

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=True,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.League.objects.create(
        id=2,
        name="Private League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )

    team_a = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Team A",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team B",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_c = m.Team.objects.create(
        id=103,
        user_id=int(owner.id),
        name="Team C",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team_a.id),
        division_name="10 B West",
        division_id=136,
        conference_id=0,
    )
    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team_b.id),
        division_name="10 B West",
        division_id=136,
        conference_id=0,
    )
    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team_c.id),
        division_name="12A",
        division_id=137,
        conference_id=0,
    )

    m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team_a.id),
        name="Player 1",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    g1 = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=1,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=dt.datetime(2026, 1, 1, 0, 0, 0),
        updated_at=None,
    )
    g2 = m.HkyGame.objects.create(
        id=1002,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=dt.datetime(2099, 1, 1, 10, 0, 0),
        location="Future Rink",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=dt.datetime(2026, 1, 1, 0, 0, 0),
        updated_at=None,
    )
    g3 = m.HkyGame.objects.create(
        id=1003,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_c.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 3, 10, 0, 0),
        location="Other Rink",
        notes=None,
        team1_score=3,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        created_at=dt.datetime(2026, 1, 1, 0, 0, 0),
        updated_at=None,
    )

    m.LeagueGame.objects.create(
        league_id=1, game_id=int(g1.id), division_name="10 B West", sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=1, game_id=int(g2.id), division_name="10 B West", sort_order=None
    )
    # Cross-division game (10 B West vs 12A): should still be visible in public league views.
    m.LeagueGame.objects.create(
        league_id=1, game_id=int(g3.id), division_name="10 B West", sort_order=None
    )

    ev_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(g1.id),
        event_type_id=int(ev_goal.id),
        import_key="g1-goal-1",
        team_id=int(team_a.id),
        player_id=501,
        team_side="Home",
        period=1,
        game_seconds=10,
        created_at=now,
        updated_at=None,
    )

    return Client()


def should_list_public_leagues_without_login(client):
    r = client.get("/public/leagues")
    assert r.status_code == 200
    html = r.content.decode()
    assert "Public League" in html
    assert "Private League" not in html


def should_allow_public_league_teams_schedule_and_game_pages_without_login(client):
    r1 = client.get("/public/leagues/1/teams")
    assert r1.status_code == 200
    html = r1.content.decode()
    assert "10 B West" in html
    assert "Team A" in html

    r2 = client.get("/public/leagues/1/schedule")
    assert r2.status_code == 200
    html2 = r2.content.decode()
    assert "Team A" in html2 and "Team B" in html2

    r3 = client.get("/public/leagues/1/hky/games/1001")
    assert r3.status_code == 200
    assert "Team A" in r3.content.decode()


def should_hide_future_unplayed_game_pages_in_public_schedule(client):
    html = client.get("/public/leagues/1/schedule").content.decode()
    assert "/public/leagues/1/hky/games/1002" not in html
    assert client.get("/public/leagues/1/hky/games/1002").status_code == 404


def should_show_cross_division_timetoscore_games_in_public_views(client):
    html = client.get("/public/leagues/1/schedule").content.decode()
    assert "/public/leagues/1/hky/games/1003" in html
    assert client.get("/public/leagues/1/hky/games/1003").status_code == 200


def should_return_to_entry_page_when_viewing_game_from_team_page(client):
    team_html = client.get("/public/leagues/1/teams/101").content.decode()
    assert "/public/leagues/1/hky/games/1001?return_to=/public/leagues/1/teams/101" in team_html
    assert "/public/leagues/1/hky/games/1003?return_to=/public/leagues/1/teams/101" in team_html

    game_html = client.get(
        "/public/leagues/1/hky/games/1001?return_to=/public/leagues/1/teams/101"
    ).content.decode()
    assert 'href="/public/leagues/1/teams/101"' in game_html


def should_reject_private_league_public_routes(client):
    r = client.get("/public/leagues/2/teams")
    assert r.status_code == 404

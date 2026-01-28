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
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=2,
        user_id=int(user.id),
        name="Team B",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_a.id), division_name="DivA"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_b.id), division_name="DivA"
    )

    g_future = m.HkyGame.objects.create(
        id=123,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=dt.datetime(2099, 1, 1, 10, 0, 0),
        location="",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g_unknown = m.HkyGame.objects.create(
        id=124,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=None,
        location="",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g_today_no_data = m.HkyGame.objects.create(
        id=125,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=now - dt.timedelta(hours=1),
        location="",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g_today_with_shifts = m.HkyGame.objects.create(
        id=126,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=now - dt.timedelta(hours=2),
        location="",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_future.id), division_name="DivA", sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id), game_id=int(g_unknown.id), division_name="DivA", sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(g_today_no_data.id),
        division_name="DivA",
        sort_order=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(g_today_with_shifts.id),
        division_name="DivA",
        sort_order=None,
    )

    m.HkyGameShiftRow.objects.create(
        game_id=int(g_today_with_shifts.id),
        import_key="g126-shift",
        created_at=now,
        updated_at=None,
    )

    c = Client()
    sess = c.session
    sess["user_id"] = int(user.id)
    sess["user_email"] = "u@example.com"
    sess["league_id"] = int(league.id)
    sess.save()
    return c


def should_hide_view_links_for_future_unplayed_games_but_allow_unknown_date_games(client):
    html = client.get("/schedule").content.decode()
    assert 'href="/hky/games/123"' not in html
    assert 'href="/hky/games/124?return_to=/schedule"' in html
    assert 'href="/hky/games/125"' not in html
    assert 'href="/hky/games/126?return_to=/schedule"' in html
    assert html.count('class="game-untracked-upcoming"') == 2


def should_sort_schedule_games_by_time_within_date(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from tools.webapp import app as mod

    games = [
        {
            "id": 1,
            "starts_at": "2026-01-01 15:00:00",
            "sort_order": 200,
            "created_at": "2026-01-01 00:00:00",
        },
        {
            "id": 2,
            "starts_at": "2026-01-01 09:00:00",
            "sort_order": 100,
            "created_at": "2026-01-01 00:00:00",
        },
    ]
    out = mod.sort_games_schedule_order(games)
    assert [g["id"] for g in out] == [2, 1]

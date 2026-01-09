from __future__ import annotations

import importlib.util
import datetime as dt

import pytest


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture()
def client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

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
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team_a.id), division_name="DivA")
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team_b.id), division_name="DivA")

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
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g_future.id), division_name="DivA", sort_order=None)
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g_unknown.id), division_name="DivA", sort_order=None)

    app = mod.create_app()
    app.testing = True
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = int(user.id)
        sess["user_email"] = "u@example.com"
        sess["league_id"] = int(league.id)
    return c


def should_hide_view_links_for_future_unplayed_games_but_allow_unknown_date_games(client):
    html = client.get("/schedule").get_data(as_text=True)
    assert 'href="/hky/games/123"' not in html
    assert 'href="/hky/games/124?return_to=/schedule"' in html


def should_sort_schedule_games_by_time_within_date(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    games = [
        {"id": 1, "starts_at": "2026-01-01 15:00:00", "sort_order": 200, "created_at": "2026-01-01 00:00:00"},
        {"id": 2, "starts_at": "2026-01-01 09:00:00", "sort_order": 100, "created_at": "2026-01-01 00:00:00"},
    ]
    out = mod.sort_games_schedule_order(games)
    assert [g["id"] for g in out] == [2, 1]

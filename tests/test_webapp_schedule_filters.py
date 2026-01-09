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
    m.LeagueMember.objects.create(league_id=int(league.id), user_id=int(user.id), role="admin", created_at=now)
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
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team_b.id), division_name="DivB")

    app = mod.create_app()
    app.testing = True
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = int(user.id)
        sess["user_email"] = "u@example.com"
        sess["league_id"] = int(league.id)
    return c


def should_filter_team_dropdown_by_division(client):
    html = client.get("/schedule?division=DivA").get_data(as_text=True)
    assert "Team A" in html
    assert "Team B" not in html

    html2 = client.get("/schedule").get_data(as_text=True)
    assert "Team A" in html2 and "Team B" in html2


def should_render_teams_in_league_view(client):
    html = client.get("/teams").get_data(as_text=True)
    assert "Team A" in html
    assert "Team B" in html

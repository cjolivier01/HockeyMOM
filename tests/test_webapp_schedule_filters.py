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

    c = Client()
    sess = c.session
    sess["user_id"] = int(user.id)
    sess["user_email"] = "u@example.com"
    sess["league_id"] = int(league.id)
    sess.save()
    return c


def should_filter_team_dropdown_by_division(client):
    html = client.get("/schedule?division=DivA").content.decode()
    assert "Team A" in html
    assert "Team B" not in html

    html2 = client.get("/schedule").content.decode()
    assert "Team A" in html2 and "Team B" in html2


def should_render_teams_in_league_view(client):
    html = client.get("/teams").content.decode()
    assert "Team A" in html
    assert "Team B" in html

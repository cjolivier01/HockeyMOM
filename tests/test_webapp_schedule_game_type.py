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
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=2,
        user_id=int(user.id),
        name="Team B",
        is_external=True,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team_a.id), division_name="External")
    g = m.HkyGame.objects.create(
        id=123,
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
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g.id), division_name="External", sort_order=None)

    c = Client()
    sess = c.session
    sess["user_id"] = int(user.id)
    sess["user_email"] = "u@example.com"
    sess["league_id"] = int(league.id)
    sess.save()
    return c


def should_default_external_division_game_type_to_tournament_in_schedule_table(client):
    html = client.get("/schedule").content.decode()
    assert ">Tournament<" in html


def should_default_external_division_game_type_to_tournament_in_team_schedule_table(client):
    html = client.get("/teams/1").content.decode()
    assert "<th>Type</th>" in html
    assert ">Tournament<" in html

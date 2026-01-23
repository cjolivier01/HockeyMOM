from __future__ import annotations

import datetime as dt
import json

import pytest


@pytest.fixture()
def client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")

    from django.test import Client

    now = dt.datetime(2025, 1, 1, 0, 0, 0)
    user = m.User.objects.create(
        email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    league = m.League.objects.create(
        id=42,
        name="L",
        owner_user_id=int(user.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueMember.objects.create(
        league_id=int(league.id), user_id=int(user.id), role="admin", created_at=now
    )

    team_a = m.Team.objects.create(
        id=1,
        user_id=int(user.id),
        name="Team A",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=2,
        user_id=int(user.id),
        name="Team B",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    team_c = m.Team.objects.create(
        id=3,
        user_id=int(user.id),
        name="Team C",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.bulk_create(
        [
            m.LeagueTeam(league_id=int(league.id), team_id=int(team_a.id), division_name="12AA"),
            m.LeagueTeam(league_id=int(league.id), team_id=int(team_b.id), division_name="12AA"),
            m.LeagueTeam(league_id=int(league.id), team_id=int(team_c.id), division_name="12AA"),
        ]
    )

    regular = m.GameType.objects.get(name="Regular Season")
    g1 = m.HkyGame.objects.create(
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        team1_score=4,
        team2_score=0,
        is_final=True,
        created_at=now,
        updated_at=None,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g1.id), division_name="12AA")
    g2 = m.HkyGame.objects.create(
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_c.id),
        team1_score=3,
        team2_score=0,
        is_final=True,
        created_at=now,
        updated_at=None,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g2.id), division_name="12AA")
    g3 = m.HkyGame.objects.create(
        user_id=int(user.id),
        team1_id=int(team_b.id),
        team2_id=int(team_c.id),
        team1_score=2,
        team2_score=0,
        is_final=True,
        created_at=now,
        updated_at=None,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g3.id), division_name="12AA")

    c = Client()
    sess = c.session
    sess["user_id"] = int(user.id)
    sess["user_email"] = "owner@example.com"
    sess["league_id"] = int(league.id)
    sess.save()
    return c


def should_not_create_seed_teams_and_should_resolve_seed_placeholders_in_schedule(
    client, webapp_db
):
    _django_orm, m = webapp_db

    payload = {
        "league_name": "L",
        "owner_email": "owner@example.com",
        "owner_name": "Owner",
        "source": "test",
        "external_key": "seed-test",
        "games": [
            {
                "home_name": "12AA Seed 2",
                "away_name": "12AA Seed 1",
                "division_name": "12AA",
                "game_type_name": "Tournament",
                "starts_at": "2025-01-10 10:00:00",
                "location": "SeedTest Rink",
            }
        ],
    }
    resp = client.post(
        "/api/import/hockey/games_batch",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    out = json.loads(resp.content.decode())
    assert out.get("ok") is True

    assert not m.Team.objects.filter(name="12AA Seed 1").exists()
    assert not m.Team.objects.filter(name="12AA Seed 2").exists()

    placeholder = m.Team.objects.filter(name="Playoff Seed").first()
    assert placeholder is not None
    assert not m.LeagueTeam.objects.filter(league_id=42, team_id=int(placeholder.id)).exists()

    # Team-filtered schedule should still include placeholder games after runtime resolution.
    team_a_id = m.Team.objects.get(name="Team A").id
    html = client.get(f"/schedule?division=12AA&team_id={int(team_a_id)}").content.decode()
    assert "SeedTest Rink" in html
    assert "12AA Seed 1" not in html
    assert "12AA Seed 2" not in html
    assert "Playoff Seed" not in html

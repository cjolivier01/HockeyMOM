from __future__ import annotations

import datetime as dt
import json


def should_infer_external_division_from_team_names_for_shift_package(webapp_db):
    _django_orm, m = webapp_db
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        email="owner@example.com", password_hash="x", name="O", created_at=now
    )
    league = m.League.objects.create(
        name="CAHA",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )

    away_team = m.Team.objects.create(
        user_id=int(owner.id),
        name="San Jose Jr Sharks 12AA-2",
        is_external=True,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(away_team.id), division_name="12 AA"
    )

    payload = {
        "external_game_key": "chicago-2",
        "owner_email": "owner@example.com",
        "league_name": "CAHA",
        "home_team_name": "Reston Raiders 12AA",
        "away_team_name": "San Jose Jr Sharks 12AA-2",
        "replace": False,
        # Keep stats payloads empty; this test focuses on league/team mappings.
    }

    client = Client()
    resp = client.post(
        "/api/import/hockey/shift_package",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out.get("ok") is True
    game_id = int(out.get("game_id") or 0)
    assert game_id > 0

    home_team_id = (
        m.Team.objects.filter(name="Reston Raiders 12AA").values_list("id", flat=True).first()
    )
    assert home_team_id is not None

    home_lt = (
        m.LeagueTeam.objects.filter(league_id=int(league.id), team_id=int(home_team_id))
        .values("division_name")
        .first()
    )
    assert home_lt is not None
    assert str(home_lt.get("division_name") or "") == "External 12 AA"

    away_lt = (
        m.LeagueTeam.objects.filter(league_id=int(league.id), team_id=int(away_team.id))
        .values("division_name")
        .first()
    )
    assert away_lt is not None
    assert str(away_lt.get("division_name") or "") == "12 AA"

    lg = (
        m.LeagueGame.objects.filter(league_id=int(league.id), game_id=int(game_id))
        .values("division_name")
        .first()
    )
    assert lg is not None
    assert str(lg.get("division_name") or "") == "External 12 AA"

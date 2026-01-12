from __future__ import annotations

import datetime as dt
import json

import pytest


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="cjolivier01@gmail.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    other_owner = m.User.objects.create(
        id=11,
        email="other@example.com",
        password_hash="x",
        name="Other",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="CAHA",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.League.objects.create(
        id=2,
        name="Other",
        owner_user_id=int(other_owner.id),
        is_shared=False,
        is_public=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )

    t201 = m.Team.objects.create(
        id=201,
        user_id=int(owner.id),
        name="T201",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    t202 = m.Team.objects.create(
        id=202,
        user_id=int(owner.id),
        name="T202",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    t999 = m.Team.objects.create(
        id=999,
        user_id=int(owner.id),
        name="T999",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )

    g1001 = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(t201.id),
        team2_id=int(t202.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g1002 = m.HkyGame.objects.create(
        id=1002,
        user_id=int(owner.id),
        team1_id=int(t202.id),
        team2_id=int(t999.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueGame.objects.create(
        league_id=1, game_id=int(g1001.id), division_name=None, sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=1, game_id=int(g1002.id), division_name=None, sort_order=None
    )
    m.LeagueGame.objects.create(
        league_id=2, game_id=int(g1002.id), division_name=None, sort_order=None
    )  # shared game

    m.LeagueTeam.objects.create(league_id=1, team_id=int(t201.id))
    m.LeagueTeam.objects.create(league_id=1, team_id=int(t202.id))
    m.LeagueTeam.objects.create(league_id=2, team_id=int(t202.id))  # shared team

    m.Player.objects.create(
        id=1,
        user_id=int(owner.id),
        team_id=int(t201.id),
        name="Skater A",
        jersey_number=None,
        position=None,
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=2,
        user_id=int(owner.id),
        team_id=int(t202.id),
        name="Skater B",
        jersey_number=None,
        position=None,
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.PlayerStat.objects.create(
        game_id=int(g1001.id), player_id=1, user_id=int(owner.id), team_id=int(t201.id)
    )
    m.PlayerStat.objects.create(
        game_id=int(g1002.id), player_id=2, user_id=int(owner.id), team_id=int(t202.id)
    )

    return Client(), m


def _post_json(client, path: str, payload: dict, **extra):
    return client.post(path, data=json.dumps(payload), content_type="application/json", **extra)


def should_require_import_auth_for_internal_reset_endpoint(client_and_models):
    client, _m = client_and_models
    r = _post_json(
        client,
        "/api/internal/reset_league_data",
        {"owner_email": "cjolivier01@gmail.com", "league_name": "CAHA"},
    )
    assert r.status_code == 401


def should_reset_league_data_via_hidden_api_and_preserve_shared_entities(client_and_models):
    client, m = client_and_models
    r = _post_json(
        client,
        "/api/internal/reset_league_data",
        {"owner_email": "cjolivier01@gmail.com", "league_name": "CAHA"},
        HTTP_X_HM_IMPORT_TOKEN="sekret",
    )
    assert r.status_code == 200
    out = json.loads(r.content.decode())
    assert out["ok"] is True
    assert int(out["league_id"]) == 1

    # League 1 mappings removed.
    assert not m.LeagueGame.objects.filter(league_id=1).exists()
    assert not m.LeagueTeam.objects.filter(league_id=1).exists()

    # Shared game remains (mapped to league 2).
    assert m.HkyGame.objects.filter(id=1002).exists()
    # Exclusive game was deleted.
    assert not m.HkyGame.objects.filter(id=1001).exists()
    # Exclusive team 201 was deleted; shared team 202 remains.
    assert not m.Team.objects.filter(id=201).exists()
    assert m.Team.objects.filter(id=202).exists()

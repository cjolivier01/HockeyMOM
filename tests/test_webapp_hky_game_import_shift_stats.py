from __future__ import annotations

import datetime as dt

import pytest


@pytest.fixture()
def client_and_db(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    t1 = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Smoke Team",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    t2 = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Opp Smoke",
        logo_path=None,
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(t1.id),
        name="Smoke Skater",
        jersey_number="77",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(t1.id),
        team2_id=int(t2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    return Client(), m


def should_import_shift_stats_player_csv_into_game(client_and_db):
    client, m = client_and_db
    sess = client.session
    sess["user_id"] = 10
    sess["user_email"] = "owner@example.com"
    sess.save()

    from django.core.files.uploadedfile import SimpleUploadedFile

    csv_text = (
        "Player,Goals,Assists,Shots,SOG,xG,Plus Minus,Shifts,TOI Total,TOI Total (Video),Period 1 TOI,Period 1 Shifts,Period 1 GF,Period 1 GA\n"
        "77 Smoke Skater,1,0,2,1,0,1,5,5:00,5:00,5:00,5,1,0\n"
    )
    f = SimpleUploadedFile("player_stats.csv", csv_text.encode("utf-8"), content_type="text/csv")

    r = client.post(
        "/hky/games/1001/import_shift_stats?return_to=/schedule", {"player_stats_csv": f}
    )
    assert r.status_code == 302

    ps = (
        m.PlayerStat.objects.filter(game_id=1001, player_id=501)
        .values("goals", "assists", "shots", "sog", "plus_minus")
        .first()
    )
    assert ps is not None
    assert ps["goals"] == 1
    assert ps["assists"] == 0
    assert ps["shots"] == 2
    assert ps["sog"] == 1
    assert ps["plus_minus"] == 1

    assert m.HkyGame.objects.filter(id=1001).exclude(stats_imported_at=None).exists()

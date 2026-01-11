from __future__ import annotations

import datetime as dt
import importlib.util

import pytest


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_normalize_events_video_fields_bidirectionally_for_display(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    # Seconds -> time
    headers = ["Event Type", "Video Seconds"]
    rows = [{"Event Type": "Goal", "Video Seconds": "83", "Video Time": ""}]
    out_headers, out_rows = mod.normalize_events_video_time_for_display(headers, rows)
    assert "Video Time" in out_headers
    assert out_rows[0]["Video Time"] == "1:23"

    # Time -> seconds (even when seconds column was missing)
    headers2 = ["Event Type", "Video Time"]
    rows2 = [{"Event Type": "Goal", "Video Time": "1:23"}]
    out_headers2, out_rows2 = mod.normalize_events_video_time_for_display(headers2, rows2)
    assert "Video Seconds" in out_headers2
    assert out_rows2[0]["Video Seconds"] == "83"


def should_normalize_video_time_and_seconds_pair(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    vt1, vs1 = mod.normalize_video_time_and_seconds("1:23", None)
    assert vt1 == "1:23"
    assert vs1 == 83

    vt2, vs2 = mod.normalize_video_time_and_seconds("", 83)
    assert vt2 == "1:23"
    assert vs2 == 83


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
        name="Team 1",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    t2 = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team 2",
        logo_path=None,
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    p1 = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(t1.id),
        name="Skater One",
        jersey_number="9",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    g1 = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(t1.id),
        team2_id=int(t2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=2,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    et_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    m.HkyGameEventRow.objects.create(
        id=9001,
        game_id=int(g1.id),
        event_type_id=int(et_goal.id),
        import_key="e1",
        team_id=int(t1.id),
        player_id=int(p1.id),
        period=1,
        game_time="0:30",
        game_seconds=30,
        video_time=None,
        video_seconds=83,
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        id=9002,
        game_id=int(g1.id),
        event_type_id=int(et_goal.id),
        import_key="e2",
        team_id=int(t1.id),
        player_id=int(p1.id),
        period=1,
        game_time="0:40",
        game_seconds=40,
        video_time="1:23",
        video_seconds=None,
        created_at=now,
        updated_at=None,
    )

    return Client(), m, {"game_id": int(g1.id), "team_id": int(t1.id), "player_id": int(p1.id)}


def should_api_hky_game_events_always_returns_time_and_seconds(client_and_db):
    client, _m, ids = client_and_db
    sess = client.session
    sess["user_id"] = 10
    sess["user_email"] = "owner@example.com"
    sess.save()

    r = client.get(f"/api/hky/games/{ids['game_id']}/events?limit=1000")
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    by_import_key = {e["import_key"]: e for e in out["events"]}
    assert by_import_key["e1"]["video_time"] == "1:23"
    assert by_import_key["e1"]["video_seconds"] == 83
    assert by_import_key["e2"]["video_time"] == "1:23"
    assert by_import_key["e2"]["video_seconds"] == 83


def should_api_hky_team_player_events_always_returns_time_and_seconds(client_and_db):
    client, _m, ids = client_and_db
    sess = client.session
    sess["user_id"] = 10
    sess["user_email"] = "owner@example.com"
    sess.save()

    r = client.get(f"/api/hky/teams/{ids['team_id']}/players/{ids['player_id']}/events?limit=1000")
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    # Both events are attributed to the player, so they should appear in the 'events' list.
    assert len(out["events"]) >= 2
    for e in out["events"]:
        if e.get("video_time") or e.get("video_seconds") is not None:
            assert e.get("video_time") == "1:23"
            assert e.get("video_seconds") == 83

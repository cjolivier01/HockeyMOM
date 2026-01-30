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


def should_propagate_video_and_on_ice_across_same_game_time_for_display(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    headers = [
        "Event Type",
        "Team Side",
        "Period",
        "Game Seconds",
        "Video Seconds",
        "On-Ice Players (Home)",
        "On-Ice Players (Away)",
    ]
    rows = [
        {
            "Event Type": "Goal",
            "Source": "timetoscore",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Video Time": "",
            "Video Seconds": "",
            "On-Ice Players (Home)": "",
            "On-Ice Players (Away)": "",
        },
        {
            "Event Type": "xG",
            "Source": "long",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Video Seconds": "83",
            "On-Ice Players (Home)": ("9 Alice,12 Bob,13 Carl,14 Dan,15 Ed,30 Goalie,31 Extra"),
            "On-Ice Players (Away)": "",
        },
    ]

    out_headers, out_rows = mod.normalize_events_video_time_for_display(headers, rows)
    assert "Video Time" in out_headers
    assert "Video Seconds" in out_headers

    goal = out_rows[0]
    assert goal["Video Seconds"] == "83"
    assert goal["Video Time"] == "1:23"
    home = str(goal.get("On-Ice Players (Home)") or "")
    assert home.count(",") == 5  # clamped to 6 players
    assert home.endswith(" …")


def should_enrich_timetoscore_goals_with_event_id(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    headers = ["Event Type", "Source", "Team Side", "Period", "Game Seconds", "Attributed Jerseys"]
    rows = [
        {
            "Event Type": "Goal",
            "Source": "timetoscore",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Attributed Jerseys": "9",
        },
        {
            "Event Type": "Goal",
            "Source": "goals",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Attributed Jerseys": "9",
            "Event ID": "54",
        },
    ]

    out_headers, out_rows = mod.enrich_timetoscore_goals_with_long_video_times(
        existing_headers=headers,
        existing_rows=rows,
        incoming_headers=headers,
        incoming_rows=rows,
    )
    assert "Event ID" in out_headers
    tts_goal = [r for r in out_rows if str(r.get("Source") or "").strip() == "timetoscore"][0]
    assert tts_goal["Event ID"] == "54"


def should_enrich_timetoscore_goals_with_long_video_times_within_one_second(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    # Some sources round fractional seconds differently (e.g., "13.9" remaining -> 13 vs 14).
    # When a TimeToScore goal is missing video timing, prefer a near-by long-sheet Goal row over
    # leaving the clip info blank.
    headers = [
        "Event Type",
        "Source",
        "Team Side",
        "Period",
        "Game Seconds",
        "Video Time",
        "Video Seconds",
    ]
    existing_rows = [
        {
            "Event Type": "Goal",
            "Source": "timetoscore",
            "Team Side": "Away",
            "Period": "2",
            "Game Seconds": "14",
            "Video Time": "",
            "Video Seconds": "",
        }
    ]
    incoming_rows = [
        {
            "Event Type": "Goal",
            "Source": "long",
            "Team Side": "Away",
            "Period": "2",
            "Game Seconds": "13",
            "Video Time": "1:23",
            "Video Seconds": "83",
        }
    ]

    out_headers, out_rows = mod.enrich_timetoscore_goals_with_long_video_times(
        existing_headers=headers,
        existing_rows=existing_rows,
        incoming_headers=headers,
        incoming_rows=incoming_rows,
    )
    assert "Video Time" in out_headers
    assert "Video Seconds" in out_headers
    tts_goal = out_rows[0]
    assert tts_goal["Video Time"] == "1:23"
    assert tts_goal["Video Seconds"] == "83"
    assert "long" in str(tts_goal.get("Source") or "").split(",")


def should_enrich_goal_video_times_from_long_events_for_uncorrected_goals(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    headers = [
        "Event Type",
        "Source",
        "Team Side",
        "Period",
        "Game Seconds",
        "Video Time",
        "Video Seconds",
    ]
    rows = [
        {
            "__hm_player_id": "501",
            "__hm_has_correction": "",
            "Event Type": "Goal",
            "Source": "goals",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Video Time": "9:99",
            "Video Seconds": "999",
        },
        {
            "__hm_player_id": "501",
            "Event Type": "xG",
            "Source": "long",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Video Time": "1:23",
            "Video Seconds": "83",
        },
    ]

    out_headers, out_rows = mod.enrich_goal_video_times_from_long_events(headers=headers, rows=rows)
    assert out_headers == headers
    goal = [r for r in out_rows if str(r.get("Event Type") or "").strip() == "Goal"][0]
    assert goal["Video Seconds"] == "83"
    assert goal["Video Time"] == "1:23"
    assert "long" in str(goal.get("Source") or "").split(",")


def should_not_override_goal_video_times_when_corrected(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    headers = [
        "Event Type",
        "Source",
        "Team Side",
        "Period",
        "Game Seconds",
        "Video Time",
        "Video Seconds",
    ]
    rows = [
        {
            "__hm_player_id": "501",
            "__hm_has_correction": "1",
            "Event Type": "Goal",
            "Source": "goals",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Video Time": "2:34",
            "Video Seconds": "154",
        },
        {
            "__hm_player_id": "501",
            "Event Type": "xG",
            "Source": "long",
            "Team Side": "Home",
            "Period": "1",
            "Game Seconds": "100",
            "Video Time": "1:23",
            "Video Seconds": "83",
        },
    ]

    _out_headers, out_rows = mod.enrich_goal_video_times_from_long_events(
        headers=headers, rows=rows
    )
    goal = [r for r in out_rows if str(r.get("Event Type") or "").strip() == "Goal"][0]
    assert goal["Video Seconds"] == "154"
    assert goal["Video Time"] == "2:34"


def should_merge_events_overlays_missing_video_and_on_ice_for_duplicates(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    existing_csv = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Video Seconds,On-Ice Players (Away)\n"
        "xG,Away,1,100,9,,\n"
    )
    incoming_csv = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Video Seconds,On-Ice Players (Away)\n"
        'xG,Away,1,100,9,83,"9 A,12 B,13 C,14 D,15 E,30 G"\n'
    )

    merged_csv, merged_source = mod.merge_events_csv_prefer_timetoscore(
        existing_csv=existing_csv,
        existing_source_label="unit-test",
        incoming_csv=incoming_csv,
        incoming_source_label="unit-test",
        protected_types={"goal", "assist", "penalty", "penalty expired", "goaliechange"},
    )
    assert merged_source == "unit-test"
    out_headers, out_rows = mod.parse_events_csv(merged_csv)
    assert len(out_rows) == 1
    assert out_rows[0]["Video Seconds"] == "83"
    assert str(out_rows[0]["On-Ice Players (Away)"] or "").strip()


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

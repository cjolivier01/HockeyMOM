from __future__ import annotations

import datetime as dt
import importlib.util


def _load_importer_module():
    spec = importlib.util.spec_from_file_location(
        "import_time2score_mod", "tools/webapp/scripts/import_time2score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_show_timetoscore_assist_events_in_player_events_api(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    importer_mod = _load_importer_module()

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    league = m.League.objects.create(
        id=1,
        name="Test League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    team1 = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Team 1",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team 2",
        logo_path=None,
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team1.id))
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team2.id))

    ethan = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Ethan L Olivier",
        jersey_number="1",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=502,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Bob Helper",
        jersey_number="2",
        position="F",
        shoots="L",
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=503,
        user_id=int(owner.id),
        team_id=int(team2.id),
        name="Opponent",
        jersey_number="9",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )

    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=2,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(game.id), sort_order=1)

    stats = {
        "homeScoring": [
            {"period": "1", "time": "0:30", "goal": "1", "assist1": "2", "assist2": ""},
            {"period": "1", "time": "1:00", "goal": "2", "assist1": "1", "assist2": ""},
        ],
        "awayScoring": [],
    }
    goal_events, assist_events = importer_mod.build_timetoscore_goal_and_assist_events(
        stats=stats,
        period_len_s=15 * 60,
        num_to_name_home={"1": "Ethan L Olivier", "2": "Bob Helper"},
        num_to_name_away={"9": "Opponent"},
    )
    assert len(goal_events) == 2
    assert len(assist_events) == 2

    events_headers = [
        "Event Type",
        "Source",
        "Team Raw",
        "Team Side",
        "For/Against",
        "Team Rel",
        "Period",
        "Game Time",
        "Game Seconds",
        "Game Seconds End",
        "Video Time",
        "Video Seconds",
        "Details",
        "Attributed Players",
        "Attributed Jerseys",
    ]
    # Provide video seconds via other event types at the same instants; goal/assist rows should inherit it.
    xg_rows = [
        {
            "Event Type": "xG",
            "Source": "long",
            "Team Side": "Home",
            "For/Against": "For",
            "Team Rel": "Home",
            "Team Raw": "Home",
            "Period": "1",
            "Game Time": "0:30",
            "Game Seconds": "30",
            "Video Seconds": "83",
            "Details": "xG",
        },
        {
            "Event Type": "xG",
            "Source": "long",
            "Team Side": "Home",
            "For/Against": "For",
            "Team Rel": "Home",
            "Team Raw": "Home",
            "Period": "1",
            "Game Time": "1:00",
            "Game Seconds": "60",
            "Video Seconds": "120",
            "Details": "xG",
        },
    ]
    events_csv = importer_mod._to_csv_text(  # type: ignore[attr-defined]
        events_headers, list(goal_events) + list(assist_events) + xg_rows
    )

    from tools.webapp.django_app import views as v

    up = v._upsert_game_event_rows_from_events_csv(  # type: ignore[attr-defined]
        game_id=int(game.id),
        events_csv=str(events_csv),
        replace=True,
        create_missing_players=False,
    )
    assert up["ok"] is True

    # Ethan should have exactly one goal and one assist attributed, and both should have video fields.
    from django.test import Client

    client = Client()
    sess = client.session
    sess["user_id"] = int(owner.id)
    sess["user_email"] = str(owner.email)
    sess.save()

    r = client.get(
        f"/api/hky/teams/{int(team1.id)}/players/{int(ethan.id)}/events?league_id={int(league.id)}&limit=1000"
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True

    events = list(out.get("events") or [])
    assert len([e for e in events if e.get("event_type_key") == "goal"]) == 1
    assert len([e for e in events if e.get("event_type_key") == "assist"]) == 1

    by_gs = {
        (int(e.get("game_seconds") or 0), str(e.get("event_type_key") or "")): e for e in events
    }
    goal_30 = by_gs[(30, "goal")]
    assert goal_30["video_seconds"] == 83
    assert goal_30["video_time"] == "1:23"

    assist_60 = by_gs[(60, "assist")]
    assert assist_60["video_seconds"] == 120
    assert assist_60["video_time"] == "2:00"
    assert assist_60["details"] == "Goal: #2 Bob Helper"

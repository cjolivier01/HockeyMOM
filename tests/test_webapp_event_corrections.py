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
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    league = m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team_home = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Home",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_away = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Away",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(team_home.id),
        division_name="10 A",
        division_id=None,
        conference_id=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(team_away.id),
        division_name="10 A",
        division_id=None,
        conference_id=None,
    )

    notes = json.dumps({"game_video": "https://youtu.be/abc123"}, sort_keys=True)
    game = m.HkyGame.objects.create(
        id=2001,
        user_id=int(owner.id),
        team1_id=int(team_home.id),
        team2_id=int(team_away.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 16, 10, 0, 0),
        location="Rink",
        notes=notes,
        team1_score=1,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key="utah-1",
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(game.id),
        division_name="10 A",
        division_id=None,
        conference_id=None,
        sort_order=1,
    )

    p3 = m.Player.objects.create(
        id=503,
        user_id=int(owner.id),
        team_id=int(team_away.id),
        name="Three",
        jersey_number="3",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    p13 = m.Player.objects.create(
        id=513,
        user_id=int(owner.id),
        team_id=int(team_away.id),
        name="Thirteen",
        jersey_number="13",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    return Client(), m, game, team_away, p3, p13


def should_apply_patch_corrections_and_expose_diffs_in_player_events_api(client_and_models):
    client, m, game, team, p3, p13 = client_and_models

    from tools.webapp.django_app import views

    # Seed an initial goal+assist at P2 03:14 for the Away side.
    events_csv = (
        "Event Type,Team Side,Period,Game Seconds,Video Time,Attributed Jerseys\n"
        "Goal,Away,2,194,10:20,3\n"
        "Assist,Away,2,194,10:20,13\n"
    )
    out = views._upsert_game_event_rows_from_events_csv(
        game_id=int(game.id),
        events_csv=events_csv,
        replace=True,
        create_missing_players=False,
        incoming_source_label="unit-test",
    )
    assert out["ok"] is True

    payload = {
        "corrections": [
            {
                "external_game_key": "utah-1",
                "owner_email": "owner@example.com",
                "reason": "Swap scorer/assist for goal at P2 03:14",
                "patch": [
                    {
                        "match": {
                            "event_type": "Goal",
                            "period": 2,
                            "game_time": "03:14",
                            "team_side": "Away",
                            "jersey": "3",
                        },
                        "set": {"jersey": "13", "video_time": "10:20"},
                        "note": "see video",
                    },
                    {
                        "match": {
                            "event_type": "Assist",
                            "period": 2,
                            "game_time": "03:14",
                            "team_side": "Away",
                            "jersey": "13",
                        },
                        "set": {"jersey": "3", "video_time": "10:20"},
                        "note": "see video",
                    },
                ],
            }
        ]
    }
    resp = client.post(
        "/api/internal/apply_event_corrections",
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_X_HM_IMPORT_TOKEN="sekret",
    )
    assert resp.status_code == 200
    body = json.loads(resp.content)
    assert body["ok"] is True

    goal = m.HkyGameEventRow.objects.filter(
        game_id=int(game.id),
        event_type__key="goal",
        team_side="Away",
        period=2,
        game_seconds=194,
    ).first()
    assert goal is not None
    assert str(goal.attributed_jerseys or "") == "13"
    assert int(goal.player_id or 0) == int(p13.id)
    assert goal.correction
    goal_corr = json.loads(str(goal.correction))
    assert goal_corr.get("note") == "see video"
    assert any(
        c.get("field") == "jersey" and c.get("from") == "3" and c.get("to") == "13"
        for c in (goal_corr.get("changes") or [])
    )

    assist = m.HkyGameEventRow.objects.filter(
        game_id=int(game.id),
        event_type__key="assist",
        team_side="Away",
        period=2,
        game_seconds=194,
        attributed_jerseys="3",
    ).first()
    assert assist is not None
    assert int(assist.player_id or 0) == int(p3.id)
    assert assist.correction
    assist_corr = json.loads(str(assist.correction))
    assert assist_corr.get("note") == "see video"
    assert any(
        c.get("field") == "jersey" and c.get("from") == "13" and c.get("to") == "3"
        for c in (assist_corr.get("changes") or [])
    )

    old_assist_key = views._compute_event_import_key(
        event_type_key="assist",
        period=2,
        game_seconds=194,
        team_side_norm="away",
        jersey_norm="13",
        event_id=None,
        details=None,
        game_seconds_end=None,
    )
    assert m.HkyGameEventSuppression.objects.filter(
        game_id=int(game.id), import_key=str(old_assist_key)
    ).exists()

    # Player Events API should surface correction metadata and formatted goal-assist detail.
    resp = client.get(f"/api/hky/teams/{int(team.id)}/players/{int(p13.id)}/events?league_id=1")
    assert resp.status_code == 200
    data = json.loads(resp.content)
    assert data["ok"] is True

    goal_rows = [r for r in (data.get("events") or []) if (r.get("event_type_key") == "goal")]
    assert goal_rows
    r0 = goal_rows[0]
    assert "correction" in r0 and r0["correction"]
    assert "A:" in str(r0.get("details") or "")
    assert "#13" not in str(r0.get("details") or "")

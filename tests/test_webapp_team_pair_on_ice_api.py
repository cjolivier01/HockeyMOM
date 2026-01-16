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
    m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=True,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team_a = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Team A",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team B",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team_a.id),
        division_name="10 A",
        division_id=None,
        conference_id=None,
    )
    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team_b.id),
        division_name="10 A",
        division_id=None,
        conference_id=None,
    )

    notes = json.dumps({"timetoscore_game_id": 123, "timetoscore_season_id": 31}, sort_keys=True)
    m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=notes,
        team1_score=1,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=dt.datetime(2026, 1, 1, 0, 0, 0),
        updated_at=None,
    )
    m.LeagueGame.objects.create(league_id=1, game_id=1001, division_name="10 A", sort_order=None)

    m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team_a.id),
        name="Alice",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=503,
        user_id=int(owner.id),
        team_id=int(team_a.id),
        name="Carol",
        jersey_number="10",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=502,
        user_id=int(owner.id),
        team_id=int(team_b.id),
        name="Bob",
        jersey_number="12",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    return Client(), m


def should_compute_pair_on_ice_from_shift_and_goal_rows(client_and_models):
    client, m = client_and_models
    now = dt.datetime.now()

    # Shifts: both skaters on ice together from 900->700 (200s overlap).
    m.HkyGameShiftRow.objects.create(
        game_id=1001,
        import_key="alice-1",
        source="unit-test",
        team_id=101,
        player_id=501,
        team_side="Home",
        period=1,
        game_seconds=900,
        game_seconds_end=600,
        video_seconds=None,
        video_seconds_end=None,
        created_at=now,
        updated_at=None,
    )
    m.HkyGameShiftRow.objects.create(
        game_id=1001,
        import_key="carol-1",
        source="unit-test",
        team_id=101,
        player_id=503,
        team_side="Home",
        period=1,
        game_seconds=900,
        game_seconds_end=700,
        video_seconds=None,
        video_seconds_end=None,
        created_at=now,
        updated_at=None,
    )

    from tools.webapp.django_app import views

    events_csv = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys\n"
        "Goal,Away,1,800,12\n"
        "Goal,Home,1,750,9\n"
        "Assist,Home,1,750,10\n"
    )
    out = views._upsert_game_event_rows_from_events_csv(
        game_id=1001,
        events_csv=events_csv,
        replace=True,
        create_missing_players=False,
        incoming_source_label="unit-test",
    )
    assert out["ok"] is True

    resp = client.get("/api/hky/teams/101/pair_on_ice?league_id=1")
    assert resp.status_code == 200
    payload = json.loads(resp.content)
    assert payload["ok"] is True
    assert int(payload["eligible_games"]) == 1
    assert int(payload["shift_games"]) == 1

    rows = payload.get("rows") or []
    assert rows

    def _find(player: str, teammate: str):
        for r in rows:
            if (
                str(r.get("player_name") or "") == player
                and str(r.get("teammate_name") or "") == teammate
            ):
                return r
        return None

    alice_carol = _find("Alice", "Carol")
    carol_alice = _find("Carol", "Alice")
    assert alice_carol is not None
    assert carol_alice is not None

    assert int(alice_carol.get("shift_games") or 0) == 1
    assert int(alice_carol.get("gf_together") or 0) == 1
    assert int(alice_carol.get("ga_together") or 0) == 1
    assert int(alice_carol.get("plus_minus_together") or 0) == 0
    assert float(alice_carol.get("overlap_pct") or 0.0) == pytest.approx(66.666, abs=0.2)
    assert int(alice_carol.get("player_goals_on_ice_together") or 0) == 1
    assert int(alice_carol.get("goals_collab_with_teammate") or 0) == 1
    assert int(alice_carol.get("assists_collab_with_teammate") or 0) == 0

    assert float(carol_alice.get("overlap_pct") or 0.0) == pytest.approx(100.0, abs=0.01)
    assert int(carol_alice.get("player_assists_on_ice_together") or 0) == 1
    assert int(carol_alice.get("assists_collab_with_teammate") or 0) == 1

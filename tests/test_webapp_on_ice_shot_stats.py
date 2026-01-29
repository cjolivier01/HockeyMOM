from __future__ import annotations

import datetime as dt


def should_compute_on_ice_sog_for_and_against_from_shift_and_event_rows(webapp_db, monkeypatch):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")

    now = dt.datetime.now()
    user = m.User.objects.create(email="u@example.com", password_hash="x", name="U", created_at=now)

    team_a = m.Team.objects.create(
        user_id=int(user.id),
        name="Team A",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        user_id=int(user.id),
        name="Team B",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )

    alice = m.Player.objects.create(
        user_id=int(user.id),
        team_id=int(team_a.id),
        name="Alice",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    carol = m.Player.objects.create(
        user_id=int(user.id),
        team_id=int(team_a.id),
        name="Carol",
        jersey_number="10",
        position="D",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        user_id=int(user.id),
        team_id=int(team_b.id),
        name="Bob",
        jersey_number="12",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    g = m.HkyGame.objects.create(
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=1,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key=None,
        created_at=now,
        updated_at=None,
    )

    et_shot = m.HkyEventType.objects.create(key="shot", name="Shot", created_at=now)
    et_sog = m.HkyEventType.objects.create(key="sog", name="SOG", created_at=now)
    et_goal = m.HkyEventType.objects.create(key="goal", name="Goal", created_at=now)

    # Shift intervals (Period 1, within-period seconds).
    # Alice: 0-60, Carol: 30-90.
    m.HkyGameShiftRow.objects.create(
        game_id=int(g.id),
        import_key="shift-alice-1",
        source="unit-test",
        team_id=int(team_a.id),
        player_id=int(alice.id),
        team_side="Home",
        period=1,
        game_seconds=0,
        game_seconds_end=60,
        video_seconds=None,
        video_seconds_end=None,
        created_at=now,
        updated_at=None,
    )
    m.HkyGameShiftRow.objects.create(
        game_id=int(g.id),
        import_key="shift-carol-1",
        source="unit-test",
        team_id=int(team_a.id),
        player_id=int(carol.id),
        team_side="Home",
        period=1,
        game_seconds=30,
        game_seconds_end=90,
        video_seconds=None,
        video_seconds_end=None,
        created_at=now,
        updated_at=None,
    )

    def _add_ev(*, import_key: str, et, team_id: int, team_side: str, t: int) -> None:
        m.HkyGameEventRow.objects.create(
            game_id=int(g.id),
            event_type_id=int(et.id),
            import_key=str(import_key),
            team_id=int(team_id),
            team_side=str(team_side),
            for_against=None,
            team_rel=None,
            team_raw=None,
            period=1,
            game_time=None,
            video_time=None,
            game_seconds=int(t),
            game_seconds_end=None,
            video_seconds=None,
            details=None,
            correction=None,
            player_id=None,
            attributed_players=None,
            attributed_jerseys=None,
            on_ice_players=None,
            on_ice_players_home=None,
            on_ice_players_away=None,
            created_at=now,
            updated_at=None,
        )

    # Events:
    #  - 15: Team A Shot (only Alice on)
    #  - 45: Team B SOG (Alice + Carol on)
    #  - 75: Team A Goal (only Carol on; counts as SOG evidence)
    _add_ev(import_key="ev-1", et=et_shot, team_id=int(team_a.id), team_side="Home", t=15)
    _add_ev(import_key="ev-2", et=et_sog, team_id=int(team_b.id), team_side="Away", t=45)
    _add_ev(import_key="ev-4", et=et_goal, team_id=int(team_a.id), team_side="Home", t=75)

    from tools.webapp.django_app import views

    schedule_games = [{"id": int(g.id), "team1_id": int(team_a.id), "team2_id": int(team_b.id)}]
    roster_players = [
        {
            "id": int(alice.id),
            "team_id": int(team_a.id),
            "name": "Alice",
            "jersey_number": "9",
            "position": "F",
        },
        {
            "id": int(carol.id),
            "team_id": int(team_a.id),
            "name": "Carol",
            "jersey_number": "10",
            "position": "D",
        },
    ]

    rows = views._player_stat_rows_from_event_tables_for_team_games(
        team_id=int(team_a.id),
        schedule_games=schedule_games,
        roster_players=roster_players,
        show_shift_data=True,
    )
    by_pid = {int(r["player_id"]): r for r in rows}

    assert by_pid[int(alice.id)]["sog_for_on_ice"] == 0
    assert by_pid[int(alice.id)]["sog_against_on_ice"] == 1

    assert by_pid[int(carol.id)]["sog_for_on_ice"] == 1
    assert by_pid[int(carol.id)]["sog_against_on_ice"] == 1

    # When shift data display is disabled, on-ice SOG stats should be treated as unknown.
    rows_hidden = views._player_stat_rows_from_event_tables_for_team_games(
        team_id=int(team_a.id),
        schedule_games=schedule_games,
        roster_players=roster_players,
        show_shift_data=False,
    )
    by_pid_hidden = {int(r["player_id"]): r for r in rows_hidden}
    assert by_pid_hidden[int(alice.id)]["sog_for_on_ice"] is None
    assert by_pid_hidden[int(alice.id)]["sog_against_on_ice"] is None

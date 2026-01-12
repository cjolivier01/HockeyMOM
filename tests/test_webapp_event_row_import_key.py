from __future__ import annotations

import datetime as dt


def should_not_duplicate_goal_rows_when_event_id_is_added_later(webapp_db):
    _django_orm, m = webapp_db
    now = dt.datetime.now()

    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    t_home = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Home",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    t_away = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Away",
        logo_path=None,
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    away_scorer = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(t_away.id),
        name="Scorer",
        jersey_number="21",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(t_home.id),
        team2_id=int(t_away.id),
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

    from tools.webapp.django_app import views

    events_csv_no_id = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys\n" "Goal,Away,1,100,21\n"
    )
    out1 = views._upsert_game_event_rows_from_events_csv(
        game_id=int(game.id),
        events_csv=events_csv_no_id,
        replace=True,
        create_missing_players=False,
    )
    assert out1["ok"] is True

    rows1 = list(
        m.HkyGameEventRow.objects.filter(game_id=int(game.id), event_type__key="goal").values(
            "id", "event_id", "player_id", "import_key"
        )
    )
    assert len(rows1) == 1
    assert rows1[0]["event_id"] is None
    assert int(rows1[0]["player_id"] or 0) == int(away_scorer.id)
    key1 = str(rows1[0]["import_key"])

    events_csv_with_id = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Event ID\n"
        "Goal,Away,1,100,21,12\n"
    )
    out2 = views._upsert_game_event_rows_from_events_csv(
        game_id=int(game.id),
        events_csv=events_csv_with_id,
        replace=False,
        create_missing_players=False,
    )
    assert out2["ok"] is True

    rows2 = list(
        m.HkyGameEventRow.objects.filter(game_id=int(game.id), event_type__key="goal").values(
            "id", "event_id", "import_key"
        )
    )
    assert len(rows2) == 1
    assert int(rows2[0]["event_id"] or 0) == 12
    assert str(rows2[0]["import_key"]) == key1


def should_enforce_single_goal_row_per_team_and_time(webapp_db):
    _django_orm, m = webapp_db
    now = dt.datetime.now()

    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    t_home = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Home",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    t_away = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Away",
        logo_path=None,
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    away_scorer_1 = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(t_away.id),
        name="Scorer 1",
        jersey_number="21",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    away_scorer_2 = m.Player.objects.create(
        id=502,
        user_id=int(owner.id),
        team_id=int(t_away.id),
        name="Scorer 2",
        jersey_number="22",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(t_home.id),
        team2_id=int(t_away.id),
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

    from tools.webapp.django_app import views

    events_csv_1 = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys\nGoal,Away,1,100,21\n"
    )
    out1 = views._upsert_game_event_rows_from_events_csv(
        game_id=int(game.id),
        events_csv=events_csv_1,
        replace=True,
        create_missing_players=False,
    )
    assert out1["ok"] is True

    rows1 = list(
        m.HkyGameEventRow.objects.filter(game_id=int(game.id), event_type__key="goal").values(
            "id", "player_id", "import_key"
        )
    )
    assert len(rows1) == 1
    assert int(rows1[0]["player_id"] or 0) == int(away_scorer_1.id)
    key1 = str(rows1[0]["import_key"])

    events_csv_2 = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys\nGoal,Away,1,100,22\n"
    )
    out2 = views._upsert_game_event_rows_from_events_csv(
        game_id=int(game.id),
        events_csv=events_csv_2,
        replace=False,
        create_missing_players=False,
    )
    assert out2["ok"] is True

    rows2 = list(
        m.HkyGameEventRow.objects.filter(game_id=int(game.id), event_type__key="goal").values(
            "id", "player_id", "import_key"
        )
    )
    assert len(rows2) == 1
    assert int(rows2[0]["player_id"] or 0) == int(away_scorer_2.id)
    assert str(rows2[0]["import_key"]) == key1

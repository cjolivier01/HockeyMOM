from __future__ import annotations

import datetime as dt

import pytest


def should_reject_goal_event_rows_without_resolvable_team_side(webapp_db):
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
    m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(t_home.id),
        name="Alice",
        jersey_number="9",
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
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    from tools.webapp.django_app import views

    # Missing Team Side and not inferable from jersey -> should hard error.
    events_csv = "Event Type,Period,Game Seconds,Attributed Jerseys\nGoal,1,100,99\n"
    with pytest.raises(ValueError, match=r"invalid_goal_event_row"):
        views._upsert_game_event_rows_from_events_csv(
            game_id=int(game.id),
            events_csv=events_csv,
            replace=True,
            create_missing_players=False,
        )


def should_infer_goal_team_side_from_unique_scorer_jersey_when_missing(webapp_db):
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
        id=502,
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
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    from tools.webapp.django_app import views

    events_csv = "Event Type,Period,Game Seconds,Attributed Jerseys\nGoal,1,100,21\n"
    out = views._upsert_game_event_rows_from_events_csv(
        game_id=int(game.id),
        events_csv=events_csv,
        replace=True,
        create_missing_players=False,
    )
    assert out["ok"] is True

    row = (
        m.HkyGameEventRow.objects.filter(game_id=int(game.id), event_type__key="goal")
        .values("team_id", "team_side", "player_id")
        .first()
    )
    assert row is not None
    assert int(row.get("team_id") or 0) == int(t_away.id)
    assert str(row.get("team_side") or "") == "Away"
    assert int(row.get("player_id") or 0) == int(away_scorer.id)

from __future__ import annotations

import datetime as dt


def should_synthesize_deterministic_event_id_for_penalties(webapp_db):
    _django_orm, m = webapp_db
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

    events_csv = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Details,Game Seconds End\n"
        "Penalty,Home,1,120,9,Tripping,150\n"
    )

    from tools.webapp.django_app import views

    out1 = views._upsert_game_event_rows_from_events_csv(
        game_id=int(g1.id),
        events_csv=events_csv,
        replace=True,
        create_missing_players=False,
    )
    assert out1["ok"] is True

    row1 = (
        m.HkyGameEventRow.objects.filter(game_id=int(g1.id), event_type__key="penalty")
        .values("event_id", "import_key")
        .first()
    )
    assert row1 is not None
    assert row1["event_id"] is not None
    eid1 = int(row1["event_id"])

    out2 = views._upsert_game_event_rows_from_events_csv(
        game_id=int(g1.id),
        events_csv=events_csv,
        replace=True,
        create_missing_players=False,
    )
    assert out2["ok"] is True

    row2 = (
        m.HkyGameEventRow.objects.filter(game_id=int(g1.id), event_type__key="penalty")
        .values("event_id", "import_key")
        .first()
    )
    assert row2 is not None
    assert int(row2["event_id"]) == eid1

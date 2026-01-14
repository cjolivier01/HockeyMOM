from __future__ import annotations

import datetime as dt


def should_include_goal_events_when_player_id_missing_but_jersey_matches(webapp_db):
    _django_orm, m = webapp_db

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
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team1.id))
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team2.id))

    player = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Adam Ro",
        jersey_number="8",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )

    game1 = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=5,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    game2 = m.HkyGame.objects.create(
        id=1002,
        user_id=int(owner.id),
        team1_id=int(team2.id),
        team2_id=int(team1.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 3, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=0,
        team2_score=4,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(game1.id), sort_order=1)
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(game2.id), sort_order=2)

    # Totals (what the team page shows): 5 goals across eligible games.
    m.PlayerStat.objects.create(
        user_id=int(owner.id),
        team_id=int(team1.id),
        game_id=int(game1.id),
        player_id=int(player.id),
        goals=3,
        assists=0,
    )
    m.PlayerStat.objects.create(
        user_id=int(owner.id),
        team_id=int(team1.id),
        game_id=int(game2.id),
        player_id=int(player.id),
        goals=2,
        assists=0,
    )

    et_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )

    # Game 1: 3 goal events with resolved player_id (these already show up).
    for i, gs in enumerate([30, 60, 90], start=1):
        m.HkyGameEventRow.objects.create(
            game_id=int(game1.id),
            event_type_id=int(et_goal.id),
            import_key=f"g1_goal_{i}",
            source="timetoscore",
            event_id=i,
            team_id=int(team1.id),
            player_id=int(player.id),
            team_side="Home",
            for_against="For",
            team_rel="Home",
            period=1,
            game_time=f"0:{gs:02d}",
            game_seconds=int(gs),
            attributed_jerseys="8",
            details="Adam Ro",
            created_at=now,
            updated_at=None,
        )

    # Game 2: 2 goal events that have the jersey but no resolved player_id (these should show up too).
    for i, gs in enumerate([45, 75], start=1):
        m.HkyGameEventRow.objects.create(
            game_id=int(game2.id),
            event_type_id=int(et_goal.id),
            import_key=f"g2_goal_{i}",
            source="goals",
            event_id=i,
            team_id=int(team1.id),
            player_id=None,
            team_side="Away",
            for_against="For",
            team_rel="Away",
            period=2,
            game_time=f"1:{gs:02d}",
            game_seconds=15 * 60 + int(gs),
            attributed_jerseys="8",
            details=None,
            created_at=now,
            updated_at=None,
        )

    from django.test import Client

    client = Client()
    sess = client.session
    sess["user_id"] = int(owner.id)
    sess["user_email"] = str(owner.email)
    sess.save()

    r = client.get(
        f"/api/hky/teams/{int(team1.id)}/players/{int(player.id)}/events?league_id={int(league.id)}&limit=1000"
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True

    # The API should include both the player_id-attributed goals and the jersey-attributed goals.
    goals = [e for e in (out.get("events") or []) if e.get("event_type_key") == "goal"]
    assert len(goals) == 5

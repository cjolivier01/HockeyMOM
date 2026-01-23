from __future__ import annotations

import datetime as dt


def should_compute_team_player_stats_plus_minus_from_shift_rows_and_gp_from_rosters(webapp_db):
    """
    Regression test:
      - Goal +/- coverage should include games where goal rows have no on-ice lists (e.g. TimeToScore),
        as long as shift rows exist for the team/game.
      - GP should reflect per-game participation (shift rows for skaters when present, otherwise roster links),
        not a full player×game grid.
    """

    _django_orm, m = webapp_db
    from tools.webapp import app as logic
    from tools.webapp.django_app import views

    now = dt.datetime.now()

    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    league = m.League.objects.create(
        id=99,
        name="L",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )

    team1 = m.Team.objects.create(
        id=1,
        user_id=int(owner.id),
        name="Self",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=2,
        user_id=int(owner.id),
        name="Opp",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team1.id), division_name="12AA"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team2.id), division_name="12AA"
    )

    g1 = m.HkyGame.objects.create(
        id=10,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=1,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    g2 = m.HkyGame.objects.create(
        id=11,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=0,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    # No shift rows for this game; participation comes from roster links.
    g3 = m.HkyGame.objects.create(
        id=12,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=0,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    for game in (g1, g2, g3):
        m.LeagueGame.objects.create(
            league_id=int(league.id),
            game_id=int(game.id),
            division_name="12AA",
            sort_order=None,
        )

    p1 = m.Player.objects.create(
        id=101,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Skater A",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    p2 = m.Player.objects.create(
        id=102,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Skater B",
        jersey_number="10",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    goalie = m.Player.objects.create(
        id=103,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Goalie",
        jersey_number="1",
        position="G",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    # Per-game rosters (e.g. TimeToScore rosters or goals.xlsx rosters).
    for gid, pids in (
        (int(g1.id), [int(p1.id), int(p2.id), int(goalie.id)]),
        (int(g2.id), [int(p1.id), int(p2.id), int(goalie.id)]),
        (int(g3.id), [int(p2.id), int(goalie.id)]),
    ):
        for pid in pids:
            m.HkyGamePlayer.objects.create(
                game_id=int(gid),
                player_id=int(pid),
                team_id=int(team1.id),
                created_at=now,
                updated_at=None,
            )

    # Shifts exist for games 1 and 2. In game 1, only p1 has a shift (p2 is on the roster but did not play).
    m.HkyGameShiftRow.objects.create(
        game_id=int(g1.id),
        import_key="g1-p1",
        source="primary",
        team_id=int(team1.id),
        player_id=int(p1.id),
        team_side="Home",
        period=1,
        game_seconds=150,
        game_seconds_end=0,
        video_seconds=None,
        video_seconds_end=None,
        created_at=now,
        updated_at=None,
    )
    for pid in (int(p1.id), int(p2.id)):
        m.HkyGameShiftRow.objects.create(
            game_id=int(g2.id),
            import_key=f"g2-{pid}",
            source="primary",
            team_id=int(team1.id),
            player_id=int(pid),
            team_side="Home",
            period=1,
            game_seconds=250,
            game_seconds_end=0,
            video_seconds=None,
            video_seconds_end=None,
            created_at=now,
            updated_at=None,
        )

    ev_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    # TimeToScore-like goal rows: no on-ice lists, but timestamped.
    m.HkyGameEventRow.objects.create(
        game_id=int(g1.id),
        event_type_id=int(ev_goal.id),
        import_key="g1-goal",
        source="timetoscore",
        team_id=int(team1.id),
        team_side="Home",
        period=1,
        game_seconds=100,
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(g2.id),
        event_type_id=int(ev_goal.id),
        import_key="g2-goal",
        source="timetoscore",
        team_id=int(team2.id),
        team_side="Away",
        period=1,
        game_seconds=120,
        created_at=now,
        updated_at=None,
    )

    schedule_games = [
        {"id": int(g1.id), "team1_id": int(team1.id), "team2_id": int(team2.id)},
        {"id": int(g2.id), "team1_id": int(team1.id), "team2_id": int(team2.id)},
        {"id": int(g3.id), "team1_id": int(team1.id), "team2_id": int(team2.id)},
    ]
    roster_players = list(
        m.Player.objects.filter(team_id=int(team1.id)).values(
            "id", "name", "position", "jersey_number"
        )
    )

    ps_rows = views._player_stat_rows_from_event_tables_for_team_games(
        team_id=int(team1.id),
        schedule_games=schedule_games,
        roster_players=roster_players,
        show_shift_data=False,
    )

    cov_counts, cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows, eligible_game_ids=[int(g1.id), int(g2.id), int(g3.id)]
    )
    assert cov_total == 3
    assert cov_counts["plus_minus"] == 2
    assert cov_counts["gf_counted"] == 2
    assert cov_counts["ga_counted"] == 2

    totals = logic._aggregate_player_totals_from_rows(
        player_stats_rows=ps_rows, allowed_game_ids={int(g1.id), int(g2.id), int(g3.id)}
    )
    assert totals[int(p1.id)]["gp"] == 2  # games 1-2 (shift-based)
    assert totals[int(p2.id)]["gp"] == 2  # game 2 (shift) + game 3 (roster-only)

from __future__ import annotations

import datetime as dt


def should_use_outlier_games_only_for_mhr_ratings(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")

    from tools.webapp import app as webapp_app

    now = dt.datetime(2026, 1, 1, 0, 0, 0)
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
        name="Test League",
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

    team_a = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Team A",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team B",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(team_a.id),
        division_name="12AA",
        division_id=None,
        conference_id=None,
        mhr_div_rating=None,
        mhr_rating=None,
        mhr_agd=None,
        mhr_sched=None,
        mhr_games=None,
        mhr_updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(team_b.id),
        division_name="12AA",
        division_id=None,
        conference_id=None,
        mhr_div_rating=None,
        mhr_rating=None,
        mhr_agd=None,
        mhr_sched=None,
        mhr_games=None,
        mhr_updated_at=None,
    )

    gt_regular = m.GameType.objects.filter(name="Regular Season").first()
    gt_tournament = m.GameType.objects.filter(name="Tournament").first()
    assert gt_regular is not None
    assert gt_tournament is not None

    g_regular = m.HkyGame.objects.create(
        id=201,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(gt_regular.id),
        starts_at=None,
        location=None,
        notes=None,
        team1_score=3,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key=None,
        created_at=now,
        updated_at=None,
    )
    g_blowout = m.HkyGame.objects.create(
        id=202,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(gt_tournament.id),
        starts_at=None,
        location=None,
        notes=None,
        team1_score=15,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key=None,
        created_at=now,
        updated_at=None,
    )

    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(g_regular.id),
        division_name="12AA",
        division_id=None,
        conference_id=None,
        sort_order=1,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(g_blowout.id),
        division_name="12AA",
        division_id=None,
        conference_id=None,
        sort_order=2,
    )

    # Outlier games are excluded from team totals.
    stats_user = webapp_app.compute_team_stats(None, int(team_a.id), int(owner.id))
    assert stats_user["wins"] == 1
    assert stats_user["losses"] == 0
    assert stats_user["ties"] == 0
    assert stats_user["gf"] == 3
    assert stats_user["ga"] == 2

    stats_league = webapp_app.compute_team_stats_league(None, int(team_a.id), int(league.id))
    assert stats_league["wins"] == 1
    assert stats_league["losses"] == 0
    assert stats_league["ties"] == 0
    assert stats_league["gf"] == 3
    assert stats_league["ga"] == 2

    # MHR-like ratings can use outlier games (goal-diff is capped internally).
    webapp_app.recompute_league_mhr_ratings(None, int(league.id), max_goal_diff=7, min_games=2)
    lt_a = (
        m.LeagueTeam.objects.filter(league_id=int(league.id), team_id=int(team_a.id))
        .values("mhr_games")
        .first()
    )
    assert lt_a is not None
    assert int(lt_a["mhr_games"]) == 2

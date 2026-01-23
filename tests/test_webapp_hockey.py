import datetime as dt
import importlib.util
import os


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_import_webapp_without_db_init():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()
    assert not hasattr(mod, "create_app")
    # Smoke-check that key helpers remain importable without DB init.
    assert callable(mod.parse_dt_or_none)
    assert callable(mod.compute_team_stats_league)
    assert callable(mod.normalize_game_events_csv)


def should_parse_date_formats():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()
    f = mod.parse_dt_or_none
    assert f("2025-01-02") == "2025-01-02 00:00:00"
    assert f("2025-01-02T12:34") == "2025-01-02 12:34:00"
    assert f("") is None
    assert f(None) is None


def should_split_roster_separates_coaches_and_goalies():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    players = [
        {"id": 1, "name": "Skater A", "jersey_number": "8", "position": "F"},
        {"id": 2, "name": "Goalie", "jersey_number": "1", "position": "G"},
        {"id": 3, "name": "Head Coach", "jersey_number": "", "position": "HC"},
        {"id": 4, "name": "Asst Coach", "jersey_number": "", "position": "AC"},
        {"id": 5, "name": "HC Jane Doe", "jersey_number": "", "position": ""},
        {"id": 6, "name": "AC John Doe", "jersey_number": "", "position": None},
        {"id": 7, "name": "Jane Coach", "jersey_number": "HC", "position": ""},
        {"id": 8, "name": "John Coach", "jersey_number": "AC", "position": ""},
    ]
    skaters, goalies, hcs, acs = mod.split_roster(players)
    assert [p["name"] for p in skaters] == ["Skater A"]
    assert [p["name"] for p in goalies] == ["Goalie"]
    assert [p["name"] for p in hcs] == ["Head Coach", "HC Jane Doe", "Jane Coach"]
    assert [p["name"] for p in acs] == ["Asst Coach", "AC John Doe", "John Coach"]


def should_hide_ot_and_blank_only_team_stat_columns():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    cols = (
        ("gp", "GP"),
        ("ot_goals", "OT Goals"),
        ("ot_assists", "OT Assists"),
        ("faceoff_pct", "Faceoff %"),
    )
    rows = [
        {"gp": 2, "ot_goals": 0, "ot_assists": 0, "faceoff_pct": None},
        {"gp": 1, "ot_goals": 0, "ot_assists": 0, "faceoff_pct": ""},
    ]
    kept = mod.filter_player_stats_display_columns_for_rows(cols, rows)
    assert kept == (("gp", "GP"),)


def should_sort_players_table_default_points_desc_with_stable_tiebreakers():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    rows = [
        {"name": "B", "goals": 2, "assists": 0, "points": 2},
        {"name": "A", "goals": 1, "assists": 1, "points": 2},
        {"name": "C", "goals": 0, "assists": 1, "points": 1},
        {"name": "D", "goals": 3, "assists": 0, "points": 3},
    ]
    out = mod.sort_players_table_default(rows)
    assert [r["name"] for r in out] == ["D", "B", "A", "C"]


def should_merge_imported_and_db_game_player_stats():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    players = [
        {"id": 1, "team_id": 10, "name": "Ethan L Olivier", "jersey_number": "1", "position": "F"},
        {"id": 2, "team_id": 20, "name": "Other Skater", "jersey_number": "5", "position": "F"},
    ]
    stats_by_pid = {
        1: {"goals": 0, "assists": None, "shots": None, "plus_minus": None},
        2: {"goals": 1, "assists": 0, "shots": 2, "plus_minus": -1},
    }
    imported_csv = (
        "Jersey #,Player,Goals,Assists,Shots,PIM,Plus Minus,SOG,xG\n"
        "1,Ethan L Olivier,2,1,4,0,0,3,1\n"
    )

    cols, cell_text_by_pid, cell_conf_by_pid, warn = mod.build_game_player_stats_table(
        players=players, stats_by_pid=stats_by_pid, imported_csv_text=imported_csv
    )
    assert warn is None
    col_ids = [c["id"] for c in cols]
    assert "goals" in col_ids
    assert "assists" in col_ids
    assert "shots" in col_ids
    assert cell_text_by_pid[1]["goals"] == "2"
    assert cell_conf_by_pid[1]["goals"] is False
    assert cell_text_by_pid[2]["goals"] == "1"

    # Real conflict (non-zero vs non-zero) shows "a/b" and marks conflict.
    stats_by_pid[1]["goals"] = 1
    _cols2, cell_text2, cell_conf2, _warn2 = mod.build_game_player_stats_table(
        players=players, stats_by_pid=stats_by_pid, imported_csv_text=imported_csv
    )
    assert cell_text2[1]["goals"] == "1/2"
    assert cell_conf2[1]["goals"] is True


def should_compute_game_points_and_hide_blank_columns_in_game_player_stats_table():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    players = [
        {"id": 1, "team_id": 10, "name": "Skater One", "jersey_number": "9", "position": "F"},
        {"id": 2, "team_id": 10, "name": "Skater Two", "jersey_number": "8", "position": "F"},
    ]
    stats_by_pid = {
        1: {"goals": 1, "assists": 2},
        2: {"goals": 0, "assists": 1},
    }
    imported_csv = (
        "Jersey #,Player,Goals,Assists,Unused Blank\n9,Skater One,1,2,\n8,Skater Two,0,1,\n"
    )

    cols, cell_text_by_pid, _cell_conf_by_pid, warn = mod.build_game_player_stats_table(
        players=players, stats_by_pid=stats_by_pid, imported_csv_text=imported_csv
    )
    assert warn is None
    col_labels = [str(c.get("label")) for c in cols]
    assert "P" in col_labels
    # Column with all blanks is removed.
    assert "Unused Blank" not in col_labels
    assert cell_text_by_pid[1]["points"] == "3"
    assert cell_text_by_pid[2]["points"] == "1"


def should_build_game_player_stats_table_preserves_unknown_imported_columns():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    players = [
        {"id": 1, "team_id": 10, "name": "Ethan L Olivier", "jersey_number": "1", "position": "F"},
    ]
    stats_by_pid = {1: {"goals": 0, "assists": 0}}
    imported_csv = "Jersey #,Player,Goals,Assists,TOI,Shifts\n" "1,Ethan L Olivier,1,0,12:34,18\n"
    cols, cell_text_by_pid, _conf, warn = mod.build_game_player_stats_table(
        players=players, stats_by_pid=stats_by_pid, imported_csv_text=imported_csv
    )
    assert warn is None
    col_ids = [c["id"] for c in cols]
    assert "toi" not in col_ids
    assert "shifts" not in col_ids
    assert all("toi" not in k for k in cell_text_by_pid[1].keys())
    assert all("shift" not in k for k in cell_text_by_pid[1].keys())


def should_compute_team_stats_from_rows(webapp_db):
    _django_orm, m = webapp_db

    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    now = dt.datetime(2025, 1, 1, 0, 0, 0)
    user = m.User.objects.create(
        id=123,
        email="u123@example.com",
        password_hash="x",
        name="Test User",
        created_at=now,
    )
    t1 = m.Team.objects.create(id=1, user=user, name="Team 1", created_at=now)
    t2 = m.Team.objects.create(id=2, user=user, name="Team 2", created_at=now)
    t3 = m.Team.objects.create(id=3, user=user, name="Team 3", created_at=now)

    m.HkyGame.objects.create(
        user=user,
        team1=t1,
        team2=t2,
        team1_score=3,
        team2_score=1,
        is_final=True,
        created_at=now,
    )  # win
    m.HkyGame.objects.create(
        user=user,
        team1=t2,
        team2=t1,
        team1_score=2,
        team2_score=2,
        is_final=True,
        created_at=now,
    )  # tie
    m.HkyGame.objects.create(
        user=user,
        team1=t3,
        team2=t1,
        team1_score=4,
        team2_score=2,
        is_final=True,
        created_at=now,
    )  # loss

    stats = mod.compute_team_stats(None, team_id=1, user_id=123)
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["ties"] == 1
    assert stats["gf"] == 3 + 2 + 2
    assert stats["ga"] == 1 + 2 + 4
    assert stats["points"] == 1 * 2 + 1 * 1


def should_compute_league_points_from_regular_only_but_keep_record_from_all_games(webapp_db):
    _django_orm, m = webapp_db

    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    now = dt.datetime(2025, 1, 1, 0, 0, 0)
    user = m.User.objects.create(
        id=123,
        email="u123@example.com",
        password_hash="x",
        name="Test User",
        created_at=now,
    )
    league = m.League.objects.create(
        id=999,
        name="Test League",
        owner_user=user,
        created_at=now,
    )

    t1 = m.Team.objects.create(id=1, user=user, name="Team 1", created_at=now)
    t2 = m.Team.objects.create(id=2, user=user, name="Team 2", created_at=now)
    t3 = m.Team.objects.create(id=3, user=user, name="Team 3", created_at=now, is_external=True)

    m.LeagueTeam.objects.bulk_create(
        [
            m.LeagueTeam(league=league, team=t1, division_name="12 AA"),
            m.LeagueTeam(league=league, team=t2, division_name="12 AA"),
            m.LeagueTeam(league=league, team=t3, division_name="External"),
        ]
    )

    regular = m.GameType.objects.get(name="Regular Season")
    preseason = m.GameType.objects.get(name="Preseason")

    g1 = m.HkyGame.objects.create(
        user=user,
        team1=t1,
        team2=t2,
        team1_score=3,
        team2_score=1,
        is_final=True,
        created_at=now,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league=league, game=g1, division_name="12 AA")

    g2 = m.HkyGame.objects.create(
        user=user,
        team1=t2,
        team2=t1,
        team1_score=2,
        team2_score=2,
        is_final=True,
        created_at=now,
        game_type=preseason,
    )
    m.LeagueGame.objects.create(league=league, game=g2, division_name="12 AA")

    g3 = m.HkyGame.objects.create(
        user=user,
        team1=t1,
        team2=t3,
        team1_score=4,
        team2_score=2,
        is_final=True,
        created_at=now,
        game_type=regular,
    )
    m.LeagueGame.objects.create(league=league, game=g3, division_name="External")

    stats = mod.compute_team_stats_league(None, team_id=1, league_id=999)
    # Record/GF/GA include all games
    assert stats["wins"] == 2
    assert stats["losses"] == 0
    assert stats["ties"] == 1
    assert stats["gf"] == 3 + 2 + 4
    assert stats["ga"] == 1 + 2 + 2
    # Points only count regular-season games that are not external.
    assert stats["points"] == 2
    # Total points include all games.
    assert stats["points_total"] == 2 * 2 + 1 * 1


def should_aggregate_player_totals_from_rows(webapp_db):
    _django_orm, m = webapp_db

    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    now = dt.datetime(2025, 1, 1, 0, 0, 0)
    user = m.User.objects.create(
        id=123,
        email="u123@example.com",
        password_hash="x",
        name="Test User",
        created_at=now,
    )
    team = m.Team.objects.create(id=1, user=user, name="Team 1", created_at=now)
    opponent = m.Team.objects.create(id=2, user=user, name="Team 2", created_at=now)
    game = m.HkyGame.objects.create(
        user=user,
        team1=team,
        team2=opponent,
        team1_score=1,
        team2_score=0,
        is_final=True,
        created_at=now,
    )

    p10 = m.Player.objects.create(id=10, user=user, team=team, name="Player 10", created_at=now)
    p11 = m.Player.objects.create(id=11, user=user, team=team, name="Player 11", created_at=now)

    ev_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    ev_assist, _created = m.HkyEventType.objects.get_or_create(
        key="assist", defaults={"name": "Assist", "created_at": now}
    )

    # Player 10: 2G, 1A
    for i in range(2):
        m.HkyGameEventRow.objects.create(
            game_id=int(game.id),
            event_type_id=int(ev_goal.id),
            import_key=f"p10-g-{i}",
            team_id=int(team.id),
            player_id=int(p10.id),
            period=1,
            game_seconds=10 + i,
            created_at=now,
            updated_at=None,
        )
    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(ev_assist.id),
        import_key="p10-a-0",
        team_id=int(team.id),
        player_id=int(p10.id),
        period=1,
        game_seconds=20,
        created_at=now,
        updated_at=None,
    )

    # Player 11: 0G, 2A
    for i in range(2):
        m.HkyGameEventRow.objects.create(
            game_id=int(game.id),
            event_type_id=int(ev_assist.id),
            import_key=f"p11-a-{i}",
            team_id=int(team.id),
            player_id=int(p11.id),
            period=1,
            game_seconds=30 + i,
            created_at=now,
            updated_at=None,
        )

    agg = mod.aggregate_players_totals(None, team_id=1, user_id=123)
    assert agg[10]["goals"] == 2 and agg[10]["assists"] == 1 and agg[10]["points"] == 3
    assert agg[11]["goals"] == 0 and agg[11]["assists"] == 2 and agg[11]["points"] == 2


def should_compute_scoring_by_period_when_tts_source_is_multi_valued():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    rows = [
        {"Event Type": "Goal", "Period": "1", "Team Side": "Home", "Source": "timetoscore,long"},
        {"Event Type": "Goal", "Period": "1", "Team Side": "Away", "Source": "timetoscore"},
        # When tts_linked=True, ignore non-TimeToScore goal rows.
        {"Event Type": "Goal", "Period": "1", "Team Side": "Home", "Source": "long"},
    ]
    out = mod.compute_team_scoring_by_period_from_events(rows, tts_linked=True)
    assert out[0]["period"] == 1
    assert out[0]["team1_gf"] == 1
    assert out[0]["team1_ga"] == 1
    assert out[0]["team2_gf"] == 1
    assert out[0]["team2_ga"] == 1


def should_sort_standings_by_points_then_tiebreakers():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()
    f = mod.sort_key_team_standings

    a = {"id": 1, "name": "A"}
    b = {"id": 2, "name": "B"}
    c = {"id": 3, "name": "C"}
    # A and B tied on points, A has better goal diff; C higher points.
    stats = {
        1: {"points": 10, "wins": 5, "gf": 20, "ga": 10},
        2: {"points": 10, "wins": 5, "gf": 18, "ga": 12},
        3: {"points": 12, "wins": 6, "gf": 15, "ga": 5},
    }
    ordered = sorted([a, b, c], key=lambda tr: f(tr, stats[tr["id"]]))
    assert [t["id"] for t in ordered] == [3, 1, 2]

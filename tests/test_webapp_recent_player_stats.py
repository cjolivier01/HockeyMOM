import importlib.util
import os


def _load_app_module():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_compute_recent_player_totals_per_player_by_schedule_order():
    mod = _load_app_module()

    schedule_games = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6}]
    # player 10 played games 1,3,4,6; player 11 played games 2,6
    player_stats_rows = [
        {"player_id": 10, "game_id": 1, "goals": 1, "assists": 0, "shots": 2},
        {"player_id": 10, "game_id": 3, "goals": 0, "assists": 1, "shots": 1},
        {"player_id": 10, "game_id": 4, "goals": 2, "assists": 0, "shots": 5},
        {"player_id": 10, "game_id": 6, "goals": 0, "assists": 2, "shots": 3},
        {"player_id": 11, "game_id": 2, "goals": 1, "assists": 0, "shots": 4},
        {"player_id": 11, "game_id": 6, "goals": 0, "assists": 1, "shots": 2},
    ]

    recent = mod.compute_recent_player_totals_from_rows(
        schedule_games=schedule_games, player_stats_rows=player_stats_rows, n=2
    )

    # Player 10: last 2 games are game 6 and 4.
    p10 = recent[10]
    assert p10["gp"] == 2
    assert p10["goals"] == 2  # (0 + 2)
    assert p10["assists"] == 2  # (2 + 0)
    assert p10["points"] == 4
    assert p10["shots"] == 8  # (3 + 5)
    assert p10["ppg"] == 2.0

    # Player 11: last 2 games are game 6 and 2.
    p11 = recent[11]
    assert p11["gp"] == 2
    assert p11["goals"] == 1
    assert p11["assists"] == 1
    assert p11["points"] == 2
    assert p11["shots"] == 6
    assert p11["ppg"] == 1.0


def should_compute_per_game_rates_using_stat_coverage_denominators():
    mod = _load_app_module()

    schedule_games = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    player_stats_rows = [
        # Player 10 has shots missing for game 2 (coverage=3 games, but GP=4).
        {"player_id": 10, "game_id": 1, "goals": 0, "assists": 0, "shots": 2},
        {"player_id": 10, "game_id": 2, "goals": 0, "assists": 0, "shots": None},
        {"player_id": 10, "game_id": 3, "goals": 0, "assists": 0, "shots": 4},
        {"player_id": 10, "game_id": 4, "goals": 0, "assists": 0, "shots": 4},
        # Player 11 has shots in all games (coverage=4 games).
        {"player_id": 11, "game_id": 1, "goals": 0, "assists": 0, "shots": 1},
        {"player_id": 11, "game_id": 2, "goals": 0, "assists": 0, "shots": 1},
        {"player_id": 11, "game_id": 3, "goals": 0, "assists": 0, "shots": 1},
        {"player_id": 11, "game_id": 4, "goals": 0, "assists": 0, "shots": 1},
    ]

    totals = mod._aggregate_player_totals_from_rows(
        player_stats_rows=player_stats_rows, allowed_game_ids={1, 2, 3, 4}
    )
    p10 = totals[10]
    assert p10["gp"] == 4
    assert p10["shots"] == 10
    assert abs(float(p10["shots_per_game"]) - (10.0 / 3.0)) < 1e-9

    p11 = totals[11]
    assert p11["gp"] == 4
    assert p11["shots"] == 4
    assert float(p11["shots_per_game"]) == 1.0

    recent = mod.compute_recent_player_totals_from_rows(
        schedule_games=schedule_games, player_stats_rows=player_stats_rows, n=4
    )
    r10 = recent[10]
    assert r10["gp"] == 4
    assert r10["shots"] == 10
    assert abs(float(r10["shots_per_game"]) - (10.0 / 3.0)) < 1e-9


def should_not_infer_shot_rates_from_goals_when_shots_missing():
    mod = _load_app_module()

    player_stats_rows = [
        {"player_id": 10, "game_id": 1, "goals": 1, "assists": 0, "shots": None},
        {"player_id": 10, "game_id": 2, "goals": 1, "assists": 0, "shots": None},
        {"player_id": 10, "game_id": 3, "goals": 1, "assists": 0, "shots": None},
    ]
    totals = mod._aggregate_player_totals_from_rows(
        player_stats_rows=player_stats_rows, allowed_game_ids={1, 2, 3}
    )
    p10 = totals[10]
    assert p10["gp"] == 3
    assert p10["goals"] == 3
    assert p10["shots"] == 0
    assert p10["shots_per_game"] is None


def should_order_pseudo_cf_pct_next_to_shots_on_game_page():
    mod = _load_app_module()

    cols = mod.build_game_player_stats_display_columns(
        rows=[
            {"shots": 2, "pseudo_cf_pct": 55.0},
            {"shots": 0, "pseudo_cf_pct": 0.0},
        ]
    )
    keys = [c.get("key") for c in cols]
    assert "shots" in keys
    assert "pseudo_cf_pct" in keys
    assert keys.index("pseudo_cf_pct") == keys.index("shots") + 1


def should_render_recent_player_stats_section_in_team_template(webapp_test_config_path):
    _load_app_module()
    from tools.webapp import django_orm

    django_orm.setup_django(config_path=str(webapp_test_config_path))

    from django.template.loader import render_to_string
    from django.test import RequestFactory

    team = {"id": 101, "name": "Team A", "logo_path": None, "is_external": 1, "user_id": 10}
    players = [{"id": 501, "name": "Player 1", "jersey_number": "9", "position": "F"}]
    cols = [
        {"key": "gp", "label": "GP", "n_games": 0, "total_games": 0, "show_count": False},
        {"key": "goals", "label": "Goals", "n_games": 0, "total_games": 0, "show_count": False},
        {"key": "assists", "label": "Assists", "n_games": 0, "total_games": 0, "show_count": False},
        {"key": "points", "label": "Points", "n_games": 0, "total_games": 0, "show_count": False},
        {"key": "ppg", "label": "PPG", "n_games": 0, "total_games": 0, "show_count": False},
        {
            "key": "shots_per_game",
            "label": "Shots/Game",
            "n_games": 5,
            "total_games": 6,
            "show_count": True,
        },
    ]
    row = {
        "player_id": 501,
        "jersey_number": "9",
        "name": "Player 1",
        "position": "F",
        "gp": 2,
        "goals": 1,
        "assists": 1,
        "points": 2,
        "ppg": 1.0,
        "shots_per_game": 2.0,
        "_per_game_denoms": {"shots_per_game": 3},
    }

    rf = RequestFactory()
    request = rf.get("/teams/101", {"gt": "All", "gid": "1"})
    request.session = {}
    html = render_to_string(
        "team_detail.html",
        {
            "team": team,
            "players": players,
            "player_stats_columns": cols,
            "player_stats_rows": [row],
            "game_type_filter_options": [{"label": "All", "checked": True}],
            "game_type_filter_label": "All",
            "select_games_options": [{"id": 1, "label": "Game 1", "checked": True}],
            "select_games_label": "All",
            "select_games_partial": False,
            "tstats": {"wins": 0, "losses": 0, "ties": 0, "gf": 0, "ga": 0, "points": 0},
            "schedule_games": [],
            "editable": False,
        },
        request=request,
    )

    assert "Player Stats (Skaters)" in html
    assert "Game Types:" in html
    assert "Select Games:" in html
    assert 'data-freeze-cols="2"' in html
    assert "table-nowrap" in html
    assert "(5 Games)" in html
    assert html.count("(3 games)") == 1


def should_split_coaches_out_of_player_lists():
    mod = _load_app_module()
    players = [
        {"id": 1, "name": "Head Coach A", "position": "HC"},
        {"id": 2, "name": "Assistant Coach B", "position": "AC"},
        {"id": 3, "name": "Skater C", "position": "F"},
    ]
    players_only, head_coaches, assistant_coaches = mod.split_players_and_coaches(players)
    assert [p["name"] for p in head_coaches] == ["Head Coach A"]
    assert [p["name"] for p in assistant_coaches] == ["Assistant Coach B"]
    assert [p["name"] for p in players_only] == ["Skater C"]


def should_default_sort_players_table_by_points_then_goals_assists_name():
    mod = _load_app_module()
    rows = [
        {"name": "Bob", "assists": 1, "goals": 0, "points": 1},
        {"name": "Alice", "assists": 1, "goals": 2, "points": 3},
        {"name": "Alice", "assists": 2, "goals": 0, "points": 2},
        {"name": "Alice", "assists": 2, "goals": 1, "points": 3},
        {"name": "Bob", "assists": 0, "goals": 5, "points": 5},
    ]
    out = mod.sort_players_table_default(rows)
    assert [r["name"] for r in out] == ["Bob", "Alice", "Alice", "Alice", "Bob"]
    # For same points: goals desc, then assists desc, then name asc.
    assert [(r["goals"], r["assists"], r["name"]) for r in out if r["points"] == 3] == [
        (2, 1, "Alice"),
        (1, 2, "Alice"),
    ]

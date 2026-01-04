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

    recent = mod.compute_recent_player_totals_from_rows(schedule_games=schedule_games, player_stats_rows=player_stats_rows, n=2)

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


def should_render_recent_player_stats_section_in_team_template():
    mod = _load_app_module()
    app = mod.create_app()
    app.testing = True

    from flask import render_template

    team = {"id": 101, "name": "Team A", "logo_path": None, "is_external": 1, "user_id": 10}
    players = [{"id": 501, "name": "Player 1", "jersey_number": "9", "position": "F"}]
    cols = mod.PLAYER_STATS_DISPLAY_COLUMNS
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
    }

    with app.test_request_context("/teams/101?recent_n=5&recent_sort=points&recent_dir=desc"):
        html = render_template(
            "team_detail.html",
            team=team,
            players=players,
            player_stats_columns=cols,
            player_stats_rows=[row],
            recent_player_stats_columns=cols,
            recent_player_stats_rows=[row],
            recent_n=5,
            recent_sort="points",
            recent_dir="desc",
            tstats={"wins": 0, "losses": 0, "ties": 0, "gf": 0, "ga": 0, "points": 0},
            schedule_games=[],
            editable=False,
        )

    assert "Recent Player Stats  -- Are They On a Roll?" in html
    assert "data-freeze-cols=\"2\"" in html
    assert "table-nowrap" in html
    assert "<select" in html and "recent_n" in html


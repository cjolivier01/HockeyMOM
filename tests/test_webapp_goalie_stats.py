from __future__ import annotations


def should_compute_goalie_stats_from_goaliechange_goal_and_sog_events():
    from tools.webapp import app as logic

    home_goalies = [
        {"id": 1, "name": "Home Goalie 1", "jersey_number": "30"},
        {"id": 3, "name": "Home Goalie 2", "jersey_number": "31"},
    ]
    away_goalies = [{"id": 2, "name": "Away Goalie", "jersey_number": "35"}]

    events = [
        # Starting goalies at 15:00 (900s remaining).
        {
            "event_type__key": "goaliechange",
            "team_side": "Home",
            "period": 1,
            "game_seconds": 900,
            "player_id": 1,
            "attributed_players": "Home Goalie 1",
            "details": "Home Goalie 1 Starting",
        },
        {
            "event_type__key": "goaliechange",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 900,
            "player_id": 2,
            "attributed_players": "Away Goalie",
            "details": "Away Goalie Starting",
        },
        # Mid-period home goalie change at 10:00 (600s remaining).
        {
            "event_type__key": "goaliechange",
            "team_side": "Home",
            "period": 1,
            "game_seconds": 600,
            "player_id": 3,
            "attributed_players": "Home Goalie 2",
            "details": "Home Goalie 2",
        },
        # Away SOG + Goal at 14:00 (840s remaining) against home goalie 1.
        {
            "event_type__key": "sog",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 840,
            "event_id": 1,
        },
        {
            "event_type__key": "goal",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 840,
            "event_id": 1,
        },
        # Away SOG at 14:10 (850s remaining) against home goalie 1.
        {
            "event_type__key": "sog",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 850,
            "event_id": 2,
        },
        # Away SOG + Goal at 7:30 (450s remaining) against home goalie 2.
        {
            "event_type__key": "sog",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 450,
            "event_id": 3,
        },
        {
            "event_type__key": "goal",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 450,
            "event_id": 3,
        },
        # Away SOG at 8:20 (500s remaining) against home goalie 2.
        {
            "event_type__key": "sog",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 500,
            "event_id": 4,
        },
        # Home SOG at 11:40 (700s remaining) against away goalie.
        {
            "event_type__key": "sog",
            "team_side": "Home",
            "period": 1,
            "game_seconds": 700,
            "event_id": 5,
        },
    ]

    out = logic.compute_goalie_stats_for_game(
        events, home_goalies=home_goalies, away_goalies=away_goalies
    )
    assert out["meta"]["has_sog"] is True
    assert out["meta"]["has_xg"] is False

    home_rows = {int(r["player_id"]): r for r in out["home"]}
    away_rows = {int(r["player_id"]): r for r in out["away"]}

    assert home_rows[1]["toi_seconds"] == 300
    assert home_rows[1]["ga"] == 1
    assert home_rows[1]["xga"] is None
    assert home_rows[1]["sa"] == 2
    assert home_rows[1]["saves"] == 1

    assert home_rows[3]["toi_seconds"] == 2400
    assert home_rows[3]["ga"] == 1
    assert home_rows[3]["xga"] is None
    assert home_rows[3]["sa"] == 2
    assert home_rows[3]["saves"] == 1

    assert away_rows[2]["toi_seconds"] == 2700
    assert away_rows[2]["ga"] == 0
    assert away_rows[2]["xga"] is None
    assert away_rows[2]["sa"] == 1
    assert away_rows[2]["saves"] == 1


def should_omit_shot_based_goalie_stats_when_no_shot_events_exist():
    from tools.webapp import app as logic

    home_goalies = [{"id": 1, "name": "Home Goalie 1", "jersey_number": "30"}]
    away_goalies = [{"id": 2, "name": "Away Goalie", "jersey_number": "35"}]

    events = [
        {
            "event_type__key": "goaliechange",
            "team_side": "Home",
            "period": 1,
            "game_seconds": 900,
            "player_id": 1,
            "attributed_players": "Home Goalie 1",
            "details": "Home Goalie 1 Starting",
        },
        {
            "event_type__key": "goaliechange",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 900,
            "player_id": 2,
            "attributed_players": "Away Goalie",
            "details": "Away Goalie Starting",
        },
        {
            "event_type__key": "penalty",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 840,
            "event_id": 1,
            "details": "Minor",
        },
    ]

    out = logic.compute_goalie_stats_for_game(
        events, home_goalies=home_goalies, away_goalies=away_goalies
    )
    assert out["meta"]["has_sog"] is False
    assert out["meta"]["has_xg"] is False

    home_rows = {int(r["player_id"]): r for r in out["home"]}
    assert home_rows[1]["ga"] == 0
    assert home_rows[1]["sa"] is None
    assert home_rows[1]["saves"] is None
    assert home_rows[1]["sv_pct"] is None
    assert home_rows[1]["xga"] is None
    assert home_rows[1]["xg_saves"] is None
    assert home_rows[1]["xg_sv_pct"] is None


def should_compute_xg_based_goalie_stats_when_expectedgoal_events_exist():
    from tools.webapp import app as logic

    home_goalies = [
        {"id": 1, "name": "Home Goalie 1", "jersey_number": "30"},
        {"id": 3, "name": "Home Goalie 2", "jersey_number": "31"},
    ]
    away_goalies = [{"id": 2, "name": "Away Goalie", "jersey_number": "35"}]

    events = [
        # Starting goalies at 15:00 (900s remaining).
        {
            "event_type__key": "goaliechange",
            "team_side": "Home",
            "period": 1,
            "game_seconds": 900,
            "player_id": 1,
            "attributed_players": "Home Goalie 1",
            "details": "Home Goalie 1 Starting",
        },
        {
            "event_type__key": "goaliechange",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 900,
            "player_id": 2,
            "attributed_players": "Away Goalie",
            "details": "Away Goalie Starting",
        },
        # Mid-period home goalie change at 10:00 (600s remaining).
        {
            "event_type__key": "goaliechange",
            "team_side": "Home",
            "period": 1,
            "game_seconds": 600,
            "player_id": 3,
            "attributed_players": "Home Goalie 2",
            "details": "Home Goalie 2",
        },
        # Away Goal at 14:00 (840s remaining) against home goalie 1.
        {
            "event_type__key": "goal",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 840,
            "event_id": 1,
        },
        # Away xG at 14:10 (850s remaining) against home goalie 1.
        {
            "event_type__key": "expectedgoal",
            "team_side": "Away",
            "period": 1,
            "game_seconds": 850,
            "event_id": 2,
        },
    ]

    out = logic.compute_goalie_stats_for_game(
        events, home_goalies=home_goalies, away_goalies=away_goalies
    )
    assert out["meta"]["has_sog"] is True
    assert out["meta"]["has_xg"] is True

    home_rows = {int(r["player_id"]): r for r in out["home"]}
    assert home_rows[1]["ga"] == 1
    assert home_rows[1]["sa"] == 2
    assert home_rows[1]["xga"] == 2
    assert home_rows[1]["xg_saves"] == 1
    assert home_rows[1]["xg_sv_pct"] == 0.5

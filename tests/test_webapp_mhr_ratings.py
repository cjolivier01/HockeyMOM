from __future__ import annotations

import importlib.util
import sys


def _load_rankings_module():
    spec = importlib.util.spec_from_file_location("hockey_rankings", "tools/webapp/hockey_rankings.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["hockey_rankings"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_cap_goal_differential_at_7():
    mod = _load_rankings_module()
    assert mod.clamp_goal_diff(10, cap=7) == 7
    assert mod.clamp_goal_diff(-10, cap=7) == -7
    assert mod.clamp_goal_diff(3, cap=7) == 3


def should_compute_mhr_like_ratings_two_team_case_centered():
    mod = _load_rankings_module()
    # Five games, Team 1 wins by 2 each time.
    games = [mod.GameScore(team1_id=1, team2_id=2, team1_score=4, team2_score=2) for _ in range(5)]
    out = mod.compute_mhr_like_ratings(games=games, max_goal_diff=7, min_games_for_rating=5)
    assert out[1]["games"] == 5
    assert out[2]["games"] == 5
    # The system is translation invariant; our solver centers to mean 0.
    # With a consistent +2 GD each game, ratings should differ by ~2.
    r1 = out[1]["rating"]
    r2 = out[2]["rating"]
    assert r1 is not None and r2 is not None
    assert abs((r1 - r2) - 2.0) < 1e-3


def should_require_min_games_for_rating():
    mod = _load_rankings_module()
    games = [mod.GameScore(team1_id=1, team2_id=2, team1_score=3, team2_score=2) for _ in range(4)]
    out = mod.compute_mhr_like_ratings(games=games, max_goal_diff=7, min_games_for_rating=5)
    assert out[1]["games"] == 4
    assert out[1]["rating"] is None
    assert out[1]["rating_raw"] is not None


def should_scale_ratings_to_0_99_9_top_is_99_9():
    mod = _load_rankings_module()
    out = {
        1: {"rating": -5.0, "games": 5},
        2: {"rating": 0.0, "games": 5},
        3: {"rating": 10.0, "games": 5},
        4: {"rating": None, "games": 4},
    }
    scaled = mod.scale_ratings_to_0_99_9(out)
    assert scaled[3]["rating"] == 99.9
    # Shift-only normalization preserves differences (up to rounding).
    assert scaled[2]["rating"] == 89.9
    assert scaled[1]["rating"] == 84.9
    assert round(scaled[3]["rating"] - scaled[2]["rating"], 2) == 10.0
    assert scaled[4]["rating"] is None


def should_scale_ratings_to_0_99_9_equal_values_all_99_9():
    mod = _load_rankings_module()
    out = {1: {"rating": 1.23, "games": 5}, 2: {"rating": 1.23, "games": 5}}
    scaled = mod.scale_ratings_to_0_99_9(out)
    assert scaled[1]["rating"] == 99.9
    assert scaled[2]["rating"] == 99.9

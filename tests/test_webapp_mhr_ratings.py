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


def should_scale_ratings_by_component_top_is_99_9_per_disconnected_component():
    mod = _load_rankings_module()
    # Two disconnected components: (1-2) and (10-11)
    games = [
        mod.GameScore(team1_id=1, team2_id=2, team1_score=3, team2_score=1),
        mod.GameScore(team1_id=10, team2_id=11, team1_score=2, team2_score=4),
    ]
    results = {
        1: {"rating": 5.0, "games": 5},
        2: {"rating": 2.0, "games": 5},
        10: {"rating": -1.0, "games": 5},
        11: {"rating": 4.0, "games": 5},
    }
    scaled = mod.scale_ratings_to_0_99_9_by_component(results, games=games, key="rating")
    assert scaled[1]["rating"] == 99.9
    assert scaled[11]["rating"] == 99.9
    # Differences within components preserved.
    assert round(scaled[1]["rating"] - scaled[2]["rating"], 2) == 3.0
    assert round(scaled[11]["rating"] - scaled[10]["rating"], 2) == 5.0


def should_normalize_ratings_age_aware_splits_weak_cross_age_edges():
    mod = _load_rankings_module()
    # Cross-age game exists (10 vs 12) but only 1 team per side -> weak, so it shouldn't anchor ages together.
    games = [mod.GameScore(team1_id=1, team2_id=10, team1_score=1, team2_score=2)]
    base = {1: {"rating": 3.0, "games": 5}, 2: {"rating": 2.0, "games": 5}, 10: {"rating": 4.0, "games": 5}}
    team_age = {1: 10, 2: 10, 10: 12}
    out = mod.normalize_ratings_to_0_99_9_age_aware(base, games=games, team_age=team_age, key="rating")
    # Team 1 is alone in its anchoring component (no same-age edges provided), so it gets 99.9.
    assert out[1]["rating"] == 99.9
    # Team 10 is alone in 12U component and gets 99.9 as well.
    assert out[10]["rating"] == 99.9


def should_normalize_ratings_age_aware_anchors_strong_cross_age_edges():
    mod = _load_rankings_module()
    # Strong cross-age: two 10U teams each play two 12U teams (2x2), so ages anchor together.
    games = [
        mod.GameScore(team1_id=1, team2_id=10, team1_score=1, team2_score=2),
        mod.GameScore(team1_id=2, team2_id=11, team1_score=1, team2_score=2),
        mod.GameScore(team1_id=1, team2_id=11, team1_score=1, team2_score=2),
        mod.GameScore(team1_id=2, team2_id=10, team1_score=1, team2_score=2),
    ]
    base = {
        1: {"rating": 3.0, "games": 5},
        2: {"rating": 2.0, "games": 5},
        10: {"rating": 4.0, "games": 5},
        11: {"rating": 1.0, "games": 5},
    }
    team_age = {1: 10, 2: 10, 10: 12, 11: 12}
    out = mod.normalize_ratings_to_0_99_9_age_aware(base, games=games, team_age=team_age, key="rating")
    # Single anchored component -> only the overall max becomes 99.9.
    assert out[10]["rating"] == 99.9
    assert out[1]["rating"] < 99.9


def should_parse_age_from_division_name_common_formats():
    mod = _load_rankings_module()
    assert mod.parse_age_from_division_name("10U B West") == 10
    assert mod.parse_age_from_division_name("10 B West") == 10
    assert mod.parse_age_from_division_name("12AA") == 12
    assert mod.parse_age_from_division_name("16A") == 16
    assert mod.parse_age_from_division_name("Mite B") == 8


def should_filter_games_ignore_cross_age():
    mod = _load_rankings_module()
    games = [
        mod.GameScore(team1_id=1, team2_id=2, team1_score=3, team2_score=2),  # same age
        mod.GameScore(team1_id=1, team2_id=10, team1_score=3, team2_score=2),  # cross age
        mod.GameScore(team1_id=99, team2_id=2, team1_score=3, team2_score=2),  # unknown age
    ]
    team_age = {1: 10, 2: 10, 10: 12}
    kept = mod.filter_games_ignore_cross_age(games, team_age=team_age)
    assert [(g.team1_id, g.team2_id) for g in kept] == [(1, 2)]

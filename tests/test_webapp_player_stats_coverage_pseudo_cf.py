from tools.webapp.core import player_stats as ps


def should_compute_pseudo_cf_coverage_from_on_ice_shots_keys():
    rows = [
        {"player_id": 10, "game_id": 1, "shots_for_on_ice": 5, "shots_against_on_ice": 3},
        {"player_id": 11, "game_id": 1, "shots_for_on_ice": 2, "shots_against_on_ice": 1},
        {"player_id": 10, "game_id": 2, "goals": 1},
    ]

    cov_counts, cov_total = ps._compute_team_player_stats_coverage(
        player_stats_rows=rows, eligible_game_ids=[1, 2, 3]
    )

    assert cov_total == 3
    assert cov_counts["gp"] == 2
    assert cov_counts["pseudo_cf_pct"] == 1

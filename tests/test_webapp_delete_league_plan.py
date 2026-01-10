from tools.webapp.scripts.delete_league import compute_purge_plan


def should_delete_all_unshared_games_and_teams():
    plan = compute_purge_plan(
        league_id=1,
        league_name="CAHA",
        league_game_ids=[1, 2, 3],
        shared_game_ids=[],
        league_team_ids=[10, 11],
        shared_team_ids=[],
        team_ref_counts_after_game_delete={10: 0, 11: 0},
    )
    assert plan.delete_game_ids == [1, 2, 3]
    assert plan.delete_team_ids == [10, 11]


def should_preserve_games_shared_with_other_leagues():
    plan = compute_purge_plan(
        league_id=1,
        league_name="L",
        league_game_ids=[1, 2, 3, 4],
        shared_game_ids=[2, 4],
        league_team_ids=[10],
        shared_team_ids=[],
        team_ref_counts_after_game_delete={10: 0},
    )
    assert plan.delete_game_ids == [1, 3]


def should_preserve_teams_shared_or_still_referenced():
    plan = compute_purge_plan(
        league_id=1,
        league_name="L",
        league_game_ids=[1],
        shared_game_ids=[],
        league_team_ids=[10, 11, 12],
        shared_team_ids=[11],
        team_ref_counts_after_game_delete={10: 0, 11: 0, 12: 2},
    )
    assert plan.delete_team_ids == [10]

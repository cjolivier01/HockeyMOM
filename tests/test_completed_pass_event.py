from __future__ import annotations

import importlib.util

import pandas as pd


def _load_parse_module():
    spec = importlib.util.spec_from_file_location(
        "parse_stats_inputs_mod_completed_pass", "scripts/parse_stats_inputs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod  # type: ignore


def should_parse_completed_pass_events_from_long_sheet():
    mod = _load_parse_module()

    df = pd.DataFrame(
        [
            ["1st Period", "Video Time", "Scoreboard", "Team", "Shots", "Shots on Goal", "Assist"],
            ["Completed Pass", "0:10", "0:10", "Blue", "12", "", ""],
        ]
    )
    events, goal_rows, jerseys = mod._parse_long_left_event_table(df)  # type: ignore[attr-defined]
    assert not goal_rows
    assert jerseys.get("Blue") == {12}
    completed = [ev for ev in events if ev.event_type == "CompletedPass"]
    assert completed, "Expected a CompletedPass event"
    ev = completed[0]
    assert ev.team == "Blue"
    assert ev.jerseys == (12,)
    assert ev.period == 1
    assert ev.game_s == 10

    ctx = mod._event_log_context_from_long_events(  # type: ignore[attr-defined]
        events,
        jersey_to_players={"12": ["Blue_12"]},
        focus_team="Blue",
        jerseys_by_team=jerseys,
    )
    counts = ctx.event_counts_by_player.get("Blue_12", {})
    assert counts.get("CompletedPass") == 1
    assert ctx.event_counts_by_type_team.get(("CompletedPass", "Blue")) == 1


def should_aggregate_completed_passes_into_all_games_rows():
    mod = _load_parse_module()
    rows = [{"player": "12_A", "completed_passes": "3", "gp": "1"}]
    agg_rows, periods, _denoms = mod._aggregate_stats_rows(  # type: ignore[attr-defined]
        [(rows, [])]
    )
    assert periods == []
    assert agg_rows and agg_rows[0]["player"] == "12_A"
    assert agg_rows[0]["completed_passes"] == "3"
    assert agg_rows[0]["completed_passes_per_game"] == "3.0"

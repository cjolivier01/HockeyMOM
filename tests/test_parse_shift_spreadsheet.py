import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.parse_shift_spreadsheet as pss  # noqa: E402


def should_parse_period_labels_and_tokens():
    assert pss.parse_period_label("Period 1") == 1
    assert pss.parse_period_label("1st Period") == 1
    assert pss.parse_period_label("1st Period (Blue team)") == 1
    assert pss.parse_period_label("period 2") == 2
    assert pss.parse_period_label("OT") == 4
    assert pss.parse_period_label("Overtime") == 4
    assert pss.parse_period_label("OT 3 on 3") == 4
    assert pss.parse_period_label("") is None

    assert pss.parse_period_token(1) == 1
    assert pss.parse_period_token("2") == 2
    assert pss.parse_period_token("Period 3") == 3
    assert pss.parse_period_token("OT") == 4
    assert pss.parse_period_token(True) is None


def should_parse_flex_times_and_format_seconds():
    assert pss.parse_flex_time_to_seconds("1:02") == 62
    assert pss.parse_flex_time_to_seconds("01:02") == 62
    assert pss.parse_flex_time_to_seconds("0:58.9") == 58
    assert pss.parse_flex_time_to_seconds("58.2") == 58
    assert pss.parse_flex_time_to_seconds("1:02:03.9") == 3723

    assert pss.seconds_to_mmss_or_hhmmss(62) == "1:02"
    assert pss.seconds_to_mmss_or_hhmmss(3723) == "1:02:03"
    assert pss.seconds_to_hhmmss(62) == "00:01:02"


def should_extract_pairs_from_row_formats_excel_time_cells():
    row = pd.Series(
        [
            datetime.time(0, 1),
            datetime.time(0, 2),
            pd.Timestamp("2024-01-01 00:03:00"),
            "nan",
            "0:04",
        ]
    )
    pairs = pss.extract_pairs_from_row(row, start_cols=[0, 3], end_cols=[1, 2, 4])
    # "nan" is treated as empty, and HH:MM strings are produced for Excel-like time cells.
    assert pairs == [("00:01", "00:02")]


def _basic_shift_sheet_df() -> pd.DataFrame:
    header = [
        "Jersey No",
        "Name",
        "Shift Start (Scoreboard Time)",
        None,
        "Shift End (Scoreboard Time)",
        None,
        "Shift Start (Video Time)",
        None,
        "Shift End (Video Time)",
        None,
    ]
    rows = [
        ["Period 1"] + [None] * 9,
        [None] * 10,
        header,
        [
            "12",
            "Alice",
            "15:00",
            "14:30",
            "14:00",
            "13:45",
            "0:10",
            "0:40",
            "0:50",
            "1:10",
        ],
        [
            "(G) 1",
            "Goalie",
            "15:00",
            "",
            "14:00",
            "",
            "0:00",
            "",
            "0:10",
            "",
        ],
        ["", ""] + [None] * 8,
    ]
    return pd.DataFrame(rows)


def should_parse_per_player_layout_and_skip_goalies():
    df = _basic_shift_sheet_df()
    video_pairs, sb_pairs, conv, validation_errors = pss._parse_per_player_layout(
        df, keep_goalies=False, skip_validation=True
    )
    assert validation_errors == 0

    assert list(video_pairs.keys()) == ["12_Alice"]
    assert video_pairs["12_Alice"] == [("0:10", "0:50"), ("0:40", "1:10")]

    assert list(sb_pairs.keys()) == ["12_Alice"]
    assert sb_pairs["12_Alice"] == [
        (1, "15:00", "14:00"),
        (1, "14:30", "13:45"),
    ]

    assert conv[1] == [
        (900, 840, 10, 50),
        (870, 825, 40, 70),
    ]


def should_parse_per_player_layout_reports_validation_errors():
    header = [
        "Jersey No",
        "Name",
        "Shift Start (Scoreboard Time)",
        None,
        "Shift End (Scoreboard Time)",
        None,
        "Shift Start (Video Time)",
        None,
        "Shift End (Video Time)",
        None,
    ]
    rows = [
        ["Period 1"] + [None] * 9,
        [None] * 10,
        header,
        [
            "12",
            "Alice",
            "15:00",
            "",
            "15:00",
            "",
            "0:50",
            "",
            "0:10",
            "",
        ],
        ["", ""] + [None] * 8,
    ]
    df = pd.DataFrame(rows)
    _video_pairs, _sb_pairs, _conv, validation_errors = pss._parse_per_player_layout(
        df, keep_goalies=True, skip_validation=False
    )
    assert validation_errors == 2


def should_compute_player_stats_plus_minus_skips_goal_on_shift_start():
    player_key = "12_Alice"
    sb_list = [(1, "15:00", "14:00")]
    goals_by_period = {
        1: [
            pss.GoalEvent("GF", 1, "15:00"),  # exactly shift start → should be skipped
            pss.GoalEvent("GF", 1, "14:30"),  # during shift → counted
            pss.GoalEvent("GA", 1, "14:00"),  # exactly shift end → counted
        ]
    }
    row_map, _per_counts, _unused, per_period_toi_map = pss._compute_player_stats(
        player_key,
        sb_list,
        video_pairs_by_player={},
        goals_by_period=goals_by_period,
    )
    assert row_map["plus_minus"] == "0"
    assert row_map["gf_counted"] == "1"
    assert row_map["ga_counted"] == "1"
    assert per_period_toi_map == {1: "1:00"}


def should_parse_long_sheet_times_mmss_and_excel_like_cells():
    assert pss._parse_long_mmss_time_to_seconds(datetime.time(23, 56)) == 23 * 60 + 56
    assert pss._parse_long_mmss_time_to_seconds("23:56:00") == 23 * 60 + 56
    assert pss._parse_long_mmss_time_to_seconds("24:50") == 24 * 60 + 50
    assert pss._parse_long_mmss_time_to_seconds("1:09:19") == 1 * 3600 + 9 * 60 + 19
    assert pss._parse_long_mmss_time_to_seconds("1:00:00") == 3600
    assert pss._parse_long_mmss_time_to_seconds("Video Time") is None


def should_parse_long_left_event_table_goal_xg_and_turnovers():
    rows = [
        ["1st Period", "Video Time", "Scoreboard", "Team", "Shots", "Shots on Goal", "Assist"],
        ["Goal", "0:10", "14:20", "Blue", "12", "Goal", "34, 56"],
        ["Turnover", "0:20", "14:00", "Blue", "12", "Caused by #91", "#14"],
        ["Unforced TO", "0:25", "13:50", "Blue", "12", "Caused by #91", ""],
        ["Expected Goal", "0:30", "13:40", "Blue", "12", "", ""],
        ["Controlled Entry", "0:35", "13:30", "Blue", "", "", ""],
        ["2nd Period", "Video Time", "Scoreboard", "Team", "Shots", "Shots on Goal", "Assist"],
        ["Goal", "0:40", "14:00", "White", "91", "Goal", ""],
    ]
    df = pd.DataFrame(rows)
    events, goal_rows, jerseys_by_team = pss._parse_long_left_event_table(df)

    assert goal_rows == [
        {"team": "Blue", "period": 1, "game_s": 14 * 60 + 20, "scorer": 12, "assists": [34, 56]},
        {"team": "White", "period": 2, "game_s": 14 * 60 + 0, "scorer": 91, "assists": []},
    ]
    assert jerseys_by_team == {"Blue": {12, 34, 56}, "White": {14, 91}}

    def _has(event_type: str, team: str, jerseys: tuple[int, ...]) -> bool:
        return any(e.event_type == event_type and e.team == team and e.jerseys == jerseys for e in events)

    assert _has("Goal", "Blue", (12,))
    assert _has("ExpectedGoal", "Blue", (12,))  # goals count as xG
    assert _has("SOG", "Blue", (12,))  # goal marker implies SOG
    assert _has("TurnoverForced", "Blue", (12,))
    assert _has("CreatedTurnover", "White", (91,))
    assert _has("Takeaway", "White", (14,))
    assert _has("Giveaway", "Blue", (12,))
    assert _has("Takeaway", "White", (91,))
    assert any(e.event_type == "ControlledEntry" and e.team == "Blue" and not e.jerseys for e in events)


def should_infer_focus_team_from_long_sheet_by_roster_overlap():
    jerseys_by_team = {"Blue": {12, 34, 56}, "White": {14, 91}}
    assert pss._infer_focus_team_from_long_sheet({"12", "34"}, jerseys_by_team) == "Blue"
    assert pss._infer_focus_team_from_long_sheet({"91"}, jerseys_by_team) == "White"
    assert pss._infer_focus_team_from_long_sheet({"12", "91"}, jerseys_by_team) is None


def should_map_long_events_to_player_keys_for_focus_team():
    long_events = [
        pss.LongEvent(event_type="Goal", team="Blue", period=1, video_s=10, game_s=860, jerseys=(12,)),
        pss.LongEvent(event_type="Goal", team="White", period=1, video_s=20, game_s=840, jerseys=(91,)),
    ]
    ctx = pss._event_log_context_from_long_events(
        long_events,
        jersey_to_players={"12": ["12_Alice"]},
        focus_team="Blue",
        jerseys_by_team={"Blue": {12}, "White": {91}},
    )
    assert ctx.event_counts_by_type_team[("Goal", "Blue")] == 1
    assert ctx.event_counts_by_type_team[("Goal", "White")] == 1
    assert ctx.event_counts_by_player["12_Alice"]["Goal"] == 1
    assert ctx.event_counts_by_player["White_91"]["Goal"] == 1


def should_expand_dir_input_to_game_sheets_and_ignore_goals_xlsx(tmp_path: Path):
    # Primary + companion long sheet.
    (tmp_path / "game-54111.xlsx").write_text("", encoding="utf-8")
    (tmp_path / "game-long-54111.xlsx").write_text("", encoding="utf-8")
    # Ignored files.
    (tmp_path / "goals.xlsx").write_text("", encoding="utf-8")
    (tmp_path / "~$temp.xlsx").write_text("", encoding="utf-8")
    (tmp_path / "player_stats-foo.xlsx").write_text("", encoding="utf-8")

    paths = pss._expand_dir_input_to_game_sheets(tmp_path)
    assert [p.name for p in paths] == ["game-54111.xlsx", "game-long-54111.xlsx"]

    # Multiple game labels in a dir should error.
    (tmp_path / "other.xlsx").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=r"multiple game labels"):
        _ = pss._expand_dir_input_to_game_sheets(tmp_path)


def should_select_tracking_output_video_prefers_highest_numbered(tmp_path: Path):
    (tmp_path / "tracking_output-with-audio-1.mp4").write_text("", encoding="utf-8")
    (tmp_path / "tracking_output-with-audio-10.mp4").write_text("", encoding="utf-8")
    (tmp_path / "tracking_output-with-audio.mp4").write_text("", encoding="utf-8")
    best = pss._select_tracking_output_video(tmp_path)
    assert best is not None
    assert best.name == "tracking_output-with-audio-10.mp4"


if __name__ == "__main__":
    # Make `bazel test //tests:test_parse_shift_spreadsheet` run pytest collection.
    raise SystemExit(
        pytest.main(
            [
                "-q",
                "-o",
                "python_functions=should_*",
                os.path.abspath(__file__),
            ]
        )
    )

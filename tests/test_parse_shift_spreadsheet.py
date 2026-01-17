import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.parse_stats_inputs as pss  # noqa: E402


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


def should_write_shift_rows_csv_for_webapp_upload_even_without_shifts_flag(tmp_path: Path):
    df = _basic_shift_sheet_df()
    xls_path = tmp_path / "game.xlsx"
    df.to_excel(xls_path, index=False, header=False)

    out_root = tmp_path / "player_focus"
    final_outdir, _rows, _periods, _events, _pair_rows = pss.process_sheet(
        xls_path=xls_path,
        sheet_name=None,
        outdir=out_root,
        keep_goalies=False,
        goals=[],
        roster_map=None,
        t2s_rosters_by_side=None,
        t2s_side="home",
        t2s_game_id=None,
        allow_remote=False,
        allow_full_sync=False,
        long_xls_paths=None,
        focus_team_override=None,
        include_shifts_in_stats=False,
        write_events_summary=True,
        skip_validation=True,
        create_scripts=False,
        write_opponent_stats_from_long_shifts=False,
        verbose=False,
    )
    shift_rows = Path(final_outdir) / "stats" / "shift_rows.csv"
    assert shift_rows.exists()
    text = shift_rows.read_text(encoding="utf-8")
    assert "Jersey #" in text.splitlines()[0]
    assert "Alice" in text


def should_write_shift_rows_csv_for_long_shift_team_when_enabled(tmp_path: Path):
    df = _basic_shift_sheet_df()
    xls_path = tmp_path / "game.xlsx"
    df.to_excel(xls_path, index=False, header=False)

    long_shift_tables_by_team = {
        "Team X": {
            "sb_pairs_by_player": {"12_Alice": [(1, "15:00", "14:00")]},
            "video_pairs_by_player": {"12_Alice": [("0:10", "0:50")]},
        }
    }
    outdir, _rows, _periods, _events, _pair_rows = pss._write_team_stats_from_long_shift_team(
        game_out_root=tmp_path,
        format_dir="per_player",
        team_side="home",
        team_name="Team X",
        long_shift_tables_by_team=long_shift_tables_by_team,
        goals=[],
        event_log_context=None,
        focus_team=None,
        include_shifts_in_stats=False,
        write_shift_rows_csv=True,
        xls_path=xls_path,
        t2s_rosters_by_side=None,
        create_scripts=False,
        skip_if_exists=False,
    )
    shift_rows = Path(outdir) / "stats" / "shift_rows.csv"
    assert shift_rows.exists()
    text = shift_rows.read_text(encoding="utf-8")
    assert "12" in text
    assert "15:00" not in text  # stored as seconds, not raw strings


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


def should_count_goal_role_flags_from_annotated_goals():
    # Simple game where the 2nd goal ties the game and the 3rd becomes the game-winner.
    goals = [
        pss.GoalEvent("GA", 1, "1:00", scorer=9),
        pss.GoalEvent("GF", 1, "2:00", scorer=12),
        pss.GoalEvent("GF", 1, "3:00", scorer=12),
    ]
    pss._annotate_goal_roles(goals)
    gt, gw = pss._count_goal_role_flags([g for g in goals if g.kind == "GF" and g.scorer == 12])
    assert gt == 1
    assert gw == 1


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
        return any(
            e.event_type == event_type and e.team == team and e.jerseys == jerseys for e in events
        )

    assert _has("Goal", "Blue", (12,))
    assert _has("ExpectedGoal", "Blue", (12,))  # goals count as xG
    assert _has("SOG", "Blue", (12,))  # goal marker implies SOG
    assert _has("TurnoverForced", "Blue", (12,))
    assert _has("CreatedTurnover", "White", (91,))
    assert _has("Takeaway", "White", (14,))
    assert _has("Giveaway", "Blue", (12,))
    assert _has("Takeaway", "White", (91,))
    assert any(
        e.event_type == "ControlledEntry" and e.team == "Blue" and not e.jerseys for e in events
    )


def should_parse_goals_xlsx_team_tables_with_video_time(monkeypatch, tmp_path: Path):
    goals_xlsx = tmp_path / "goals.xlsx"
    goals_xlsx.write_text("placeholder", encoding="utf-8")

    rows = [
        ["Utah", None, None, None, None, None, "Opp", None, None, None, None, None],
        [None] * 12,
        [
            "Period",
            "Time",
            "Video Time",
            "Goal",
            "Assist 1",
            "Assist 2",
            "Period",
            "Time",
            "Video Time",
            "Goal",
            "Assist 1",
            "Assist 2",
        ],
        [1, "14:20", "0:10", "#12", "#34", None, 1, "10:00", "0:50", "#91", None, None],
    ]
    df = pd.DataFrame(rows)

    def _fake_read_excel(path, *args, **kwargs):
        assert Path(path) == goals_xlsx
        return df

    monkeypatch.setattr(pss.pd, "read_excel", _fake_read_excel)

    goals = pss._goals_from_goals_xlsx(goals_xlsx, our_team_name="Utah")
    assert [str(g) for g in goals] == ["GA:1/10:00", "GF:1/14:20"]
    assert goals[0].video_t_sec == 50
    assert goals[1].video_t_sec == 10


def should_build_stats_dataframe_omits_per_game_columns_for_single_game():
    rows = [
        {
            "player": "12_Alice",
            "gp": "1",
            "goals": "1",
            "assists": "0",
            "points": "1",
            "ppg": "1.0",
            "shots": "2",
            "shots_per_game": "2.0",
            "sb_toi_total": "10:00",
            "sb_toi_per_game": "10:00",
            "shifts": "20",
            "shifts_per_game": "20.0",
        }
    ]
    df, cols = pss._build_stats_dataframe(
        rows,
        [1],
        include_shifts_in_stats=True,
        include_per_game_columns=False,
        include_gp_column=False,
    )
    assert "ppg" not in cols
    assert "sb_toi_per_game" not in cols
    assert "shifts_per_game" not in cols
    assert "gp" not in cols
    assert not any("_per_game" in c for c in cols)


def should_write_player_stats_csv_omits_per_game_columns_for_single_game(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "player": "12_Alice",
            "gp": "1",
            "goals": "1",
            "assists": "0",
            "points": "1",
            "ppg": "1.0",
            "shots": "2",
            "shots_per_game": "2.0",
            "sb_toi_total": "10:00",
            "sb_toi_per_game": "10:00",
            "shifts": "20",
            "shifts_per_game": "20.0",
        }
    ]
    pss._write_player_stats_text_and_csv(stats_dir, rows, [1], include_shifts_in_stats=True)
    out = pd.read_csv(stats_dir / "player_stats.csv")
    assert not any("per Game" in str(c) for c in out.columns)
    assert "PPG" not in out.columns
    assert "GP" not in out.columns


def should_infer_focus_team_from_long_sheet_by_roster_overlap():
    jerseys_by_team = {"Blue": {12, 34, 56}, "White": {14, 91}}
    assert pss._infer_focus_team_from_long_sheet({"12", "34"}, jerseys_by_team) == "Blue"
    assert pss._infer_focus_team_from_long_sheet({"91"}, jerseys_by_team) == "White"
    assert pss._infer_focus_team_from_long_sheet({"12", "91"}, jerseys_by_team) is None


def should_map_long_events_to_player_keys_for_focus_team():
    long_events = [
        pss.LongEvent(
            event_type="Goal", team="Blue", period=1, video_s=10, game_s=860, jerseys=(12,)
        ),
        pss.LongEvent(
            event_type="Goal", team="White", period=1, video_s=20, game_s=840, jerseys=(91,)
        ),
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


def should_invert_for_against_labels_for_turnovers_in_event_clips(tmp_path: Path):
    outdir = tmp_path / "out"
    stats_dir = outdir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    ctx = pss.EventLogContext(
        event_counts_by_player={},
        event_counts_by_type_team={
            ("TurnoverForced", "Blue"): 1,
            ("TurnoverForced", "White"): 1,
            ("Shot", "Blue"): 1,
            ("Shot", "White"): 1,
        },
        event_instances={
            ("TurnoverForced", "Blue"): [{"period": 1, "video_s": 100, "game_s": None}],
            ("TurnoverForced", "White"): [{"period": 1, "video_s": 200, "game_s": None}],
            ("Shot", "Blue"): [{"period": 1, "video_s": 300, "game_s": None}],
            ("Shot", "White"): [{"period": 1, "video_s": 400, "game_s": None}],
        },
        event_player_rows=[],
        team_roster={},
        team_excluded={},
    )

    pss._write_event_summaries_and_clips(
        outdir,
        stats_dir,
        ctx,
        conv_segments_by_period={},
        create_scripts=True,
        focus_team="Blue",
    )

    # Turnovers are recorded for the team that loses possession, but clips are labeled
    # relative to the focus team (opponent turnovers are "For").
    assert (outdir / "events_Turnovers_forced_For_video_times.txt").read_text(encoding="utf-8") == (
        "00:03:10 00:03:25 00:03:20\n"
    )
    assert (outdir / "events_Turnovers_forced_Against_video_times.txt").read_text(
        encoding="utf-8"
    ) == ("00:01:30 00:01:45 00:01:40\n")

    # Non-turnover events keep the normal labeling (our team == "For").
    assert (outdir / "events_Shot_For_video_times.txt").read_text(encoding="utf-8") == (
        "00:04:50 00:05:05 00:05:00\n"
    )
    assert (outdir / "events_Shot_Against_video_times.txt").read_text(encoding="utf-8") == (
        "00:06:30 00:06:45 00:06:40\n"
    )


def should_write_all_events_summary_without_shifts_when_requested(tmp_path: Path):
    shift_xlsx = tmp_path / "game.xlsx"
    long_xlsx = tmp_path / "game-long.xlsx"

    # Minimal per-player shift sheet with one skater.
    shift_df = _basic_shift_sheet_df()
    shift_df.to_excel(shift_xlsx, index=False, header=False)

    # Minimal TimeToScore-like long sheet with one goal event for jersey 12.
    long_rows = [
        ["1st Period", "Video Time", "Scoreboard", "Team", "Shots", "Shots on Goal", "Assist"],
        ["Goal", "0:10", "14:20", "Blue", "12", "Goal", ""],
    ]
    pd.DataFrame(long_rows).to_excel(long_xlsx, index=False, header=False)

    outdir = tmp_path / "out_sheet"
    final_outdir, *_rest = pss.process_sheet(
        xls_path=shift_xlsx,
        sheet_name=None,
        outdir=outdir,
        keep_goalies=False,
        goals=[],
        long_xls_paths=[long_xlsx],
        include_shifts_in_stats=False,
        write_events_summary=True,
        skip_validation=True,
        create_scripts=False,
    )
    events_path = final_outdir / "stats" / "all_events_summary.csv"
    assert events_path.exists()
    txt = events_path.read_text(encoding="utf-8")
    assert "Event Type" in txt
    assert "Event Type Raw" not in txt
    assert txt.splitlines()[0].split(",")[0].strip() == "Event Type"

    outdir2 = tmp_path / "out_sheet2"
    final_outdir2, *_rest2 = pss.process_sheet(
        xls_path=shift_xlsx,
        sheet_name=None,
        outdir=outdir2,
        keep_goalies=False,
        goals=[],
        long_xls_paths=[long_xlsx],
        include_shifts_in_stats=False,
        write_events_summary=False,
        skip_validation=True,
        create_scripts=False,
    )
    assert not (final_outdir2 / "stats" / "all_events_summary.csv").exists()


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


def should_expand_dir_input_to_game_sheets_accepts_goals_xlsx_only(tmp_path: Path):
    (tmp_path / "goals.xlsx").write_text("", encoding="utf-8")
    paths = pss._expand_dir_input_to_game_sheets(tmp_path)
    assert [p.name for p in paths] == ["goals.xlsx"]


def should_select_tracking_output_video_prefers_highest_numbered(tmp_path: Path):
    (tmp_path / "tracking_output-with-audio-1.mp4").write_text("", encoding="utf-8")
    (tmp_path / "tracking_output-with-audio-10.mp4").write_text("", encoding="utf-8")
    (tmp_path / "tracking_output-with-audio.mp4").write_text("", encoding="utf-8")
    best = pss._select_tracking_output_video(tmp_path)
    assert best is not None
    assert best.name == "tracking_output-with-audio-10.mp4"


def should_compute_pair_on_ice_overlap_pct_and_plus_minus_together():
    sb_pairs_by_player = {
        "12_Alice": [(1, "15:00", "14:00"), (1, "13:00", "12:00")],
        "34_Bob": [(1, "14:30", "13:30")],
        "56_Cara": [(1, "10:00", "9:00")],
        "1_Goalie": [],  # zero TOI -> should be excluded
    }
    goals_by_period = {
        1: [
            pss.GoalEvent("GF", 1, "14:15"),
            pss.GoalEvent("GA", 1, "14:30"),  # Bob shift start -> should be skipped
            pss.GoalEvent("GA", 1, "14:05"),
        ]
    }
    rows = pss._compute_pair_on_ice_rows(sb_pairs_by_player, goals_by_period)
    assert not any(r.get("player") == "1_Goalie" or r.get("teammate") == "1_Goalie" for r in rows)

    def _row(player: str, teammate: str) -> dict:
        for r in rows:
            if r.get("player") == player and r.get("teammate") == teammate:
                return r
        raise AssertionError(f"missing pair row {player} / {teammate}")

    a_b = _row("12_Alice", "34_Bob")
    b_a = _row("34_Bob", "12_Alice")

    assert a_b["overlap_seconds"] == 30
    assert round(a_b["overlap_pct"], 1) == 25.0
    assert round(b_a["overlap_pct"], 1) == 50.0

    assert a_b["gf_together"] == 1
    assert a_b["ga_together"] == 1
    assert a_b["plus_minus_together"] == 0
    assert a_b["player_total_plus_minus"] == -1
    assert a_b["teammate_total_plus_minus"] == 0

    a_c = _row("12_Alice", "56_Cara")
    assert a_c["overlap_seconds"] == 0
    assert a_c["gf_together"] == 0
    assert a_c["ga_together"] == 0


def should_count_player_total_plus_minus_even_when_no_teammates_on_ice():
    sb_pairs_by_player = {"12_Alice": [(1, "15:00", "14:00")], "34_Bob": [(1, "13:00", "12:00")]}
    goals_by_period = {1: [pss.GoalEvent("GA", 1, "14:30")]}
    rows = pss._compute_pair_on_ice_rows(sb_pairs_by_player, goals_by_period)
    a_b = next(r for r in rows if r["player"] == "12_Alice" and r["teammate"] == "34_Bob")
    assert a_b["player_total_plus_minus"] == -1


def should_compute_pair_on_ice_goal_at_shift_boundary_counts_like_player_stats():
    # Goal at 14:00 is the end of Alice's first shift and the start of her second shift.
    # Player stats count this goal (it matches the earlier shift interval and is not
    # at that shift's start), so pair-on-ice must count it too.
    sb_pairs_by_player = {
        "12_Alice": [(1, "15:00", "14:00"), (1, "14:00", "13:00")],
        "34_Bob": [(1, "15:00", "13:00")],
    }
    goals_by_period = {1: [pss.GoalEvent("GF", 1, "14:00")]}
    rows = pss._compute_pair_on_ice_rows(sb_pairs_by_player, goals_by_period)
    a_b = next(r for r in rows if r["player"] == "12_Alice" and r["teammate"] == "34_Bob")
    assert a_b["player_total_plus_minus"] == 1
    assert a_b["gf_together"] == 1
    assert a_b["plus_minus_together"] == 1


def should_compute_pair_on_ice_player_goal_assist_and_collaboration_counts():
    sb_pairs_by_player = {
        "12_Alice": [(1, "15:00", "14:00")],
        "34_Bob": [(1, "15:00", "14:00")],
    }
    goals_by_period = {1: [pss.GoalEvent("GF", 1, "14:30", scorer="12", assists=["34"])]}
    rows = pss._compute_pair_on_ice_rows(sb_pairs_by_player, goals_by_period)

    def _row(player: str, teammate: str) -> dict:
        for r in rows:
            if r.get("player") == player and r.get("teammate") == teammate:
                return r
        raise AssertionError(f"missing pair row {player} / {teammate}")

    a_b = _row("12_Alice", "34_Bob")
    b_a = _row("34_Bob", "12_Alice")

    assert a_b["player_goals_on_ice_together"] == 1
    assert a_b["player_assists_on_ice_together"] == 0
    assert a_b["goals_collab_with_teammate"] == 1  # Alice scored, Bob assisted
    assert a_b["assists_collab_with_teammate"] == 0

    assert b_a["player_goals_on_ice_together"] == 0
    assert b_a["player_assists_on_ice_together"] == 1
    assert b_a["goals_collab_with_teammate"] == 0
    assert b_a["assists_collab_with_teammate"] == 1  # Bob assisted Alice's goal


def should_parse_long_shift_tables_for_both_teams():
    # Minimal synthetic long-sheet layout: two team blocks, each with a 1st period shift table.
    import datetime

    import pandas as pd

    rows = []
    # Team A header + table header.
    rows.append(
        [None] * 10 + ["Team A (1st Period)", None, None, None, None, None, None, None, None, None]
    )
    header = [None] * 10 + [
        "Jersey Number",
        "Player Name",
        "Shift start (Scoreboard time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift Start (Video Time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift End (Scoreboard Time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift End (Video Time)",
    ]
    rows.append(header)
    rows.append(
        [None] * 10
        + [
            12,
            "Alice Example",
            datetime.time(15, 0),
            datetime.time(14, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            # start_sb group spans 8 columns total -> 2 times + 6 blanks
            datetime.time(0, 10),
            datetime.time(0, 30),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(14, 30),
            datetime.time(13, 30),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(0, 20),
            datetime.time(0, 40),
        ]
    )
    # Spacer then Team B.
    rows.append([None] * len(header))
    rows.append([None] * 10 + ["Team B (1st Period)"] + [None] * (len(header) - 11))
    rows.append(header)
    rows.append(
        [None] * 10
        + [
            34,
            "Bob Example",
            "15:00",
            "14:10",
            None,
            None,
            None,
            None,
            None,
            None,
            # start_sb group spans 8 columns total -> 2 times + 6 blanks
            "0:05",
            "0:25",
            None,
            None,
            None,
            None,
            None,
            None,
            "14:40",
            "13:50",
            None,
            None,
            None,
            None,
            None,
            None,
            "0:15",
            "0:35",
        ]
    )

    df = pd.DataFrame(rows)
    parsed = pss._parse_long_shift_tables(df)
    assert "Team A" in parsed
    assert "Team B" in parsed
    a_sb = parsed["Team A"]["sb_pairs_by_player"]
    b_sb = parsed["Team B"]["sb_pairs_by_player"]
    assert "12_Alice_Example" in a_sb
    assert a_sb["12_Alice_Example"][:2] == [(1, "15:00", "14:30"), (1, "14:00", "13:30")]
    assert "34_Bob_Example" in b_sb
    assert b_sb["34_Bob_Example"][:2] == [(1, "15:00", "14:40"), (1, "14:10", "13:50")]


def should_write_opponent_player_stats_from_long_shift_tables(tmp_path: Path):
    import datetime

    shift_xlsx = tmp_path / "game.xlsx"
    long_xlsx = tmp_path / "game-long.xlsx"

    # Minimal per-player shift sheet with one skater on our team (jersey 12).
    shift_df = _basic_shift_sheet_df()
    shift_df.to_excel(shift_xlsx, index=False, header=False)

    # Build a synthetic long sheet with:
    #   - leftmost per-period event table (records a SOG for opponent jersey 34 on White)
    #   - embedded shift tables for both teams (Team A has jersey 12, Team B has jersey 34)
    header = [None] * 10 + [
        "Jersey Number",
        "Player Name",
        "Shift start (Scoreboard time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift Start (Video Time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift End (Scoreboard Time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift End (Video Time)",
    ]
    width = len(header)

    rows = []
    rows.append(
        [
            "1st Period",
            "Video Time",
            "Scoreboard",
            "Team",
            "Shots",
            "Shots on Goal",
            "Assist",
        ]
        + [None] * (width - 7)
    )
    rows.append(["Shot", "0:10", "14:20", "White", "34", "SOG", ""] + [None] * (width - 7))

    # Team A shift table (jersey 12).
    rows.append([None] * 10 + ["Team A (1st Period)"] + [None] * (width - 11))
    rows.append(header)
    rows.append(
        [None] * 10
        + [
            12,
            "Alice Example",
            datetime.time(15, 0),
            datetime.time(14, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(0, 10),
            datetime.time(0, 30),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(14, 30),
            datetime.time(13, 30),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(0, 20),
            datetime.time(0, 40),
        ]
    )

    # Spacer then Team B shift table (jersey 34).
    rows.append([None] * width)
    rows.append([None] * 10 + ["Team B (1st Period)"] + [None] * (width - 11))
    rows.append(header)
    rows.append(
        [None] * 10
        + [
            34,
            "Bob Example",
            "15:00",
            "14:10",
            None,
            None,
            None,
            None,
            None,
            None,
            "0:05",
            "0:25",
            None,
            None,
            None,
            None,
            None,
            None,
            "14:40",
            "13:50",
            None,
            None,
            None,
            None,
            None,
            None,
            "0:15",
            "0:35",
        ]
    )

    import pandas as pd

    pd.DataFrame(rows).to_excel(long_xlsx, index=False, header=False)

    outdir = tmp_path / "out"
    final_outdir, *_rest = pss.process_sheet(
        xls_path=shift_xlsx,
        sheet_name=None,
        outdir=outdir,
        keep_goalies=False,
        goals=[pss.GoalEvent("GA", 1, "14:20", scorer="34", assists=[])],
        t2s_side="home",
        long_xls_paths=[long_xlsx],
        include_shifts_in_stats=False,
        write_events_summary=False,
        skip_validation=True,
        create_scripts=False,
    )
    assert final_outdir == outdir / "Home" / "per_player"

    opp_csv = outdir / "Away" / "per_player" / "stats" / "player_stats.csv"
    assert opp_csv.exists()
    df_opp = pd.read_csv(opp_csv)
    row = df_opp[(df_opp["Jersey #"] == 34) & (df_opp["Player"] == "Bob Example")].iloc[0]
    assert int(row["Goals"]) == 1
    assert int(row["SOG"]) == 1


def should_process_long_only_sheets_writes_home_and_away_stats(tmp_path: Path):
    import datetime

    long_xlsx = tmp_path / "game-long.xlsx"

    # Build a synthetic long sheet with:
    #   - leftmost per-period event table (Blue/White) for team-color inference
    #   - embedded shift tables for both teams
    header = [None] * 10 + [
        "Jersey Number",
        "Player Name",
        "Shift start (Scoreboard time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift Start (Video Time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift End (Scoreboard Time)",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "Shift End (Video Time)",
    ]
    width = len(header)
    rows = []
    rows.append(
        [
            "1st Period",
            "Video Time",
            "Scoreboard",
            "Team",
            "Shots",
            "Shots on Goal",
            "Assist",
        ]
        + [None] * (width - 7)
    )
    rows.append(["Shot", "0:05", "14:50", "Blue", "12", "SOG", ""] + [None] * (width - 7))
    rows.append(["Shot", "0:10", "14:20", "White", "34", "SOG", ""] + [None] * (width - 7))

    # Team A shift table (jersey 12).
    rows.append([None] * 10 + ["Team A (1st Period)"] + [None] * (width - 11))
    rows.append(header)
    rows.append(
        [None] * 10
        + [
            12,
            "Alice Example",
            datetime.time(15, 0),
            datetime.time(14, 0),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(0, 10),
            datetime.time(0, 30),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(14, 30),
            datetime.time(13, 30),
            None,
            None,
            None,
            None,
            None,
            None,
            datetime.time(0, 20),
            datetime.time(0, 40),
        ]
    )

    # Spacer then Team B shift table (jersey 34).
    rows.append([None] * width)
    rows.append([None] * 10 + ["Team B (1st Period)"] + [None] * (width - 11))
    rows.append(header)
    rows.append(
        [None] * 10
        + [
            34,
            "Bob Example",
            "15:00",
            "14:10",
            None,
            None,
            None,
            None,
            None,
            None,
            "0:05",
            "0:25",
            None,
            None,
            None,
            None,
            None,
            None,
            "14:40",
            "13:50",
            None,
            None,
            None,
            None,
            None,
            None,
            "0:15",
            "0:35",
        ]
    )

    pd.DataFrame(rows).to_excel(long_xlsx, index=False, header=False)

    outdir = tmp_path / "out"
    final_outdir, *_rest = pss.process_long_only_sheets(
        long_xls_paths=[long_xlsx],
        outdir=outdir,
        goals=[pss.GoalEvent("GA", 1, "14:20", scorer="34", assists=[])],
        roster_map={"12": "Alice Example"},
        t2s_side="home",
        t2s_game_id=None,
        include_shifts_in_stats=False,
        write_events_summary=False,
        create_scripts=False,
    )
    assert final_outdir == outdir / "Home" / "per_player"

    home_csv = outdir / "Home" / "per_player" / "stats" / "player_stats.csv"
    away_csv = outdir / "Away" / "per_player" / "stats" / "player_stats.csv"
    assert home_csv.exists()
    assert away_csv.exists()

    df_home = pd.read_csv(home_csv)
    row_home = df_home[(df_home["Jersey #"] == 12) & (df_home["Player"] == "Alice Example")].iloc[0]
    assert int(row_home["SOG"]) == 1

    df_away = pd.read_csv(away_csv)
    row_away = df_away[(df_away["Jersey #"] == 34) & (df_away["Player"] == "Bob Example")].iloc[0]
    assert int(row_away["Goals"]) == 1
    assert int(row_away["SOG"]) == 1


def should_compare_primary_vs_long_shift_summary_counts():
    # Primary has 2 shifts; long has 3, with one mismatch and one extra.
    primary = {
        "12_Alice": [(1, "15:00", "14:30"), (1, "14:00", "13:30")],
    }
    long_tables = {
        "Team A": {
            "sb_pairs_by_player": {
                "12_Alice_Long": [
                    (1, "15:00", "14:30"),
                    (1, "14:10", "13:30"),
                    (1, "13:00", "12:30"),
                ]
            },
            "video_pairs_by_player": {},
        }
    }
    summary = pss._compare_primary_shifts_to_long_shifts(
        primary_sb_pairs_by_player=primary,
        long_shift_tables_by_team=long_tables,
        threshold_seconds=5,
        warn_label="primary.xlsx",
        long_sheet_paths=[],
    )
    assert summary["matched_team"] == "Team A"
    assert summary["total_primary_shifts"] == 2
    assert summary["total_long_shifts"] == 3
    assert summary["missing_in_long"] == 0
    assert summary["extra_in_long"] == 1
    assert summary["mismatched"] == 1


def should_not_compute_per_shift_rates():
    shift_game_rows = [
        {
            "player": "1_A",
            "goals": "2",
            "assists": "1",
            "shots": "5",
            "sog": "4",
            "expected_goals": "3",
            "shifts": "10",
            "sb_toi_total": "10:00",
            "plus_minus": "1",
            "gf_counted": "2",
            "ga_counted": "1",
        }
    ]
    t2s_only_rows = [
        {
            "player": "1_A",
            "goals": "1",
            "assists": "0",
            "shifts": "",
            "sb_toi_total": "",
            "plus_minus": "",
            "gf_counted": "",
            "ga_counted": "",
        }
    ]
    agg_rows, _periods, _per_game_denoms = pss._aggregate_stats_rows(
        [(shift_game_rows, [1]), (t2s_only_rows, [1])]
    )
    row = next(r for r in agg_rows if r.get("player") == "1_A")

    # Totals still include T2S-only games.
    assert row["goals"] == "3"
    assert row["assists"] == "1"

    # Per-shift rates are intentionally not computed.
    assert not any("per_shift" in k for k in row.keys())


def should_not_include_per_shift_columns_even_when_shifts_enabled():
    rows = [
        {"player": "1_Ethan", "gp": "1", "goals": "1", "assists": "0", "points": "1", "ppg": "1.0"}
    ]
    _df0, cols0 = pss._build_stats_dataframe(rows, [1], include_shifts_in_stats=False)
    assert "goals_per_shift" not in cols0
    assert "shots_per_shift" not in cols0

    rows2 = [
        {
            "player": "1_Ethan",
            "gp": "1",
            "goals": "1",
            "assists": "0",
            "points": "1",
            "ppg": "1.0",
            "shifts": "10",
        }
    ]
    _df1, cols1 = pss._build_stats_dataframe(rows2, [1], include_shifts_in_stats=True)
    assert "goals_per_shift" not in cols1
    assert "shots_per_shift" not in cols1


def should_place_plus_minus_columns_right_after_ppg():
    rows = [
        {
            "player": "1_Ethan",
            "gp": "2",
            "goals": "1",
            "assists": "1",
            "points": "2",
            "ppg": "1.0",
            "plus_minus": "0",
            "plus_minus_per_game": "0.0",
            "gf_counted": "2",
            "gf_per_game": "1.0",
            "ga_counted": "2",
            "ga_per_game": "1.0",
            "shots": "1",
        }
    ]
    _df, cols = pss._build_stats_dataframe(rows, [1], include_shifts_in_stats=True)
    i = cols.index("ppg")
    assert cols[i + 1 : i + 7] == [
        "plus_minus",
        "plus_minus_per_game",
        "gf_counted",
        "gf_per_game",
        "ga_counted",
        "ga_per_game",
    ]


def should_write_pair_on_ice_csv_headers_human_readable(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pss._write_pair_on_ice_csv(
        stats_dir,
        [
            {
                "player": "12_Alice",
                "teammate": "34_Bob",
                "shift_games": 1,
                "player_toi_seconds": 120,
                "overlap_seconds": 30,
                "overlap_pct": 25.1234,
                "gf_together": 1,
                "ga_together": 0,
                "plus_minus_together": 1,
                "player_goals_on_ice_together": 0,
                "player_assists_on_ice_together": 0,
                "goals_collab_with_teammate": 0,
                "assists_collab_with_teammate": 0,
            }
        ],
        include_toi=True,
    )
    header = (stats_dir / "pair_on_ice.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "Player TOI" in header
    assert "Overlap" in header
    assert "Overlap %" in header
    assert "Player Total +/-" in header
    assert "Teammate Total +/-" in header
    assert "Player Goals (On Ice Together)" in header
    assert "Player Assists (On Ice Together)" in header
    assert "Goals Collaborated" in header
    assert "Assists Collaborated" in header
    # CSV stores full precision for Overlap % (XLSX applies display formatting).
    first_row = (stats_dir / "pair_on_ice.csv").read_text(encoding="utf-8").splitlines()[1]
    assert ",25.1234," in f",{first_row},"
    assert ",30," in f",{first_row},"


def should_write_pair_on_ice_csv_without_toi_when_disabled(tmp_path: Path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pss._write_pair_on_ice_csv(
        stats_dir,
        [
            {
                "player": "12_Alice",
                "teammate": "34_Bob",
                "shift_games": 1,
                "player_toi_seconds": 120,
                "overlap_seconds": 30,
                "overlap_pct": 25.0,
                "gf_together": 1,
                "ga_together": 0,
                "plus_minus_together": 1,
                "player_goals_on_ice_together": 0,
                "player_assists_on_ice_together": 0,
                "goals_collab_with_teammate": 0,
                "assists_collab_with_teammate": 0,
            }
        ],
        include_toi=False,
    )
    header = (stats_dir / "pair_on_ice.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "Player TOI" not in header
    assert "Overlap" not in header
    assert "Overlap %" in header
    assert "Player Total +/-" in header
    assert "Teammate Total +/-" in header


def should_consolidate_pair_on_ice_across_games(tmp_path: Path):
    ok = pss._write_pair_on_ice_consolidated_files(
        tmp_path,
        [
            {
                "pair_on_ice": [
                    {
                        "player": "12_Alice",
                        "teammate": "34_Bob",
                        "shift_games": 1,
                        "player_toi_seconds": 100,
                        "overlap_seconds": 25,
                        "overlap_pct": 25.0,
                        "gf_together": 1,
                        "ga_together": 0,
                        "plus_minus_together": 1,
                    }
                ]
            },
            {
                "pair_on_ice": [
                    {
                        "player": "12_Alice",
                        "teammate": "34_Bob",
                        "shift_games": 1,
                        "player_toi_seconds": 200,
                        "overlap_seconds": 50,
                        "overlap_pct": 25.0,
                        "gf_together": 0,
                        "ga_together": 1,
                        "plus_minus_together": -1,
                    }
                ]
            },
        ],
        include_toi=False,
    )
    assert ok
    csv_text = (tmp_path / "pair_on_ice_consolidated.csv").read_text(encoding="utf-8")
    assert "Games with Data" in csv_text.splitlines()[0]
    assert "Player Total +/-" in csv_text.splitlines()[0]
    assert "Teammate Total +/-" in csv_text.splitlines()[0]
    # Games with Data should sum to 2.
    assert ",2," in csv_text.replace("\r\n", "\n")


def should_write_empty_pair_on_ice_consolidated_files(tmp_path: Path):
    ok = pss._write_pair_on_ice_consolidated_files(
        tmp_path, [{"pair_on_ice": []}], include_toi=False
    )
    assert ok
    assert (tmp_path / "pair_on_ice_consolidated.csv").exists()
    assert (tmp_path / "pair_on_ice_consolidated.xlsx").exists()


def should_error_when_upload_webapp_missing_required_args(tmp_path: Path, monkeypatch):
    # Fail-fast CLI validation: upload mode requires owner + league args.
    xlsx_path = tmp_path / "game-123.xlsx"
    xlsx_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_stats_inputs.py",
            "--input",
            str(xlsx_path),
            "--upload-webapp",
        ],
    )
    with pytest.raises(SystemExit) as e:
        pss.main()
    assert int(getattr(e.value, "code", 0) or 0) == 2


def should_error_when_upload_webapp_external_game_missing_team_names(tmp_path: Path, monkeypatch):
    # External games (no T2S id) require home_team/away_team metadata (via file-list '|key=value').
    xlsx_path = tmp_path / "chicago-4.xlsx"
    xlsx_path.write_text("dummy", encoding="utf-8")

    file_list = tmp_path / "games.txt"
    file_list.write_text(str(xlsx_path) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_stats_inputs.py",
            "--file-list",
            str(file_list),
            "--upload-webapp",
            "--webapp-owner-email",
            "owner@example.com",
            "--webapp-league-name",
            "CAHA",
        ],
    )
    with pytest.raises(SystemExit) as e:
        pss.main()
    assert int(getattr(e.value, "code", 0) or 0) == 2


def should_error_when_upload_webapp_external_game_missing_date(tmp_path: Path, monkeypatch):
    # Upload mode requires a resolvable game date for external games.
    xlsx_path = tmp_path / "coyotes-1.xlsx"
    xlsx_path.write_text("dummy", encoding="utf-8")

    file_list = tmp_path / "games.txt"
    file_list.write_text(
        str(xlsx_path) + "|home_team=Arizona Coyotes|away_team=Jr Sharks 12AA-2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_stats_inputs.py",
            "--file-list",
            str(file_list),
            "--upload-webapp",
            "--webapp-owner-email",
            "owner@example.com",
            "--webapp-league-name",
            "CAHA",
        ],
    )
    with pytest.raises(SystemExit) as e:
        pss.main()
    assert int(getattr(e.value, "code", 0) or 0) == 2


def should_allow_file_list_home_away_suffix_on_directory_inputs(tmp_path: Path):
    # File-list inputs can be directories, and may include ':HOME' / ':AWAY' to
    # specify whether the "For" team is home or away (used for T2S mapping and
    # webapp upload semantics).
    stats_dir = tmp_path / "coyotes-1" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            ["1st Period", "", "", "", "", ""],
            [
                "Jersey No",
                "Name",
                pss.LABEL_START_SB,
                pss.LABEL_END_SB,
                pss.LABEL_START_V,
                pss.LABEL_END_V,
            ],
            ["8", "Adam Ro", "0:10", "0:40", "0:10", "0:40"],
        ]
    )
    xlsx_path = stats_dir / "coyotes-1.xlsx"
    df.to_excel(xlsx_path, index=False, header=False)

    p, side = pss._parse_input_token(f"{stats_dir}:AWAY")
    assert p == stats_dir
    assert side == "away"

    discovered = pss._expand_dir_input_to_game_sheets(p)
    assert discovered and discovered[0].name == "coyotes-1.xlsx"

    # Mimic the group-side selection logic from main(): side should carry through.
    groups_by_label: dict[str, dict] = {}
    for order_idx, fp in enumerate(discovered):
        label = pss._base_label_from_path(fp)
        g = groups_by_label.setdefault(
            label,
            {
                "label": label,
                "primary": None,
                "long_paths": [],
                "side": None,
                "order": order_idx,
                "meta": {},
            },
        )
        if g.get("side") is None:
            g["side"] = side

    assert groups_by_label["coyotes-1"]["side"] == "away"


def should_parse_file_list_inline_meta_after_side_suffix(tmp_path: Path):
    stats_dir = tmp_path / "chicago-2" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            ["1st Period", "", "", "", "", ""],
            [
                "Jersey No",
                "Name",
                pss.LABEL_START_SB,
                pss.LABEL_END_SB,
                pss.LABEL_START_V,
                pss.LABEL_END_V,
            ],
            ["8", "Adam Ro", "0:10", "0:40", "0:10", "0:40"],
        ]
    )
    xlsx_path = stats_dir / "chicago-2.xlsx"
    df.to_excel(xlsx_path, index=False, header=False)

    token = f"{stats_dir}:AWAY:home_team=Reston Renegades 12AA"
    p, side, meta = pss._parse_input_token_with_meta(token)
    assert p == stats_dir
    assert side == "away"
    assert meta.get("home_team") == "Reston Renegades 12AA"


def should_resolve_file_list_paths_relative_to_hockeymom_stats_base_dir(
    tmp_path: Path, monkeypatch
):
    base_dir = tmp_path / "Videos"
    stats_dir = base_dir / "utah-1" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            ["Utah", None, None, None, None, None, "Opp", None, None, None, None, None],
            [None] * 12,
            [
                "Period",
                "Time",
                "Video Time",
                "Goal",
                "Assist 1",
                "Assist 2",
                "Period",
                "Time",
                "Video Time",
                "Goal",
                "Assist 1",
                "Assist 2",
            ],
            [1, "14:20", "0:10", "#12", "#34", None, 1, "10:00", "0:50", "#91", None, None],
        ]
    )
    goals_xlsx = stats_dir / "goals.xlsx"
    df.to_excel(goals_xlsx, index=False, header=False)

    file_list_dir = tmp_path / "lists"
    file_list_dir.mkdir(parents=True, exist_ok=True)
    file_list = file_list_dir / "games.txt"
    file_list.write_text("utah-1/stats\n", encoding="utf-8")

    monkeypatch.setenv("HOCKEYMOM_STATS_BASE_DIR", str(base_dir))
    outdir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_stats_inputs.py",
            "--file-list",
            str(file_list),
            "--outdir",
            str(outdir),
            "--no-scripts",
            "--shifts",
        ],
    )
    pss.main()
    assert (outdir / "Home" / "per_player" / "stats" / "all_events_summary.csv").exists()
    assert (outdir / "Home" / "per_player" / "stats" / "game_stats.csv").exists()


def should_dump_events_for_goals_only_xlsx(tmp_path: Path, monkeypatch, capsys):
    stats_dir = tmp_path / "utah-1" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            ["Utah", None, None, None, None, None, "Opp", None, None, None, None, None],
            [None] * 12,
            [
                "Period",
                "Time",
                "Video Time",
                "Goal",
                "Assist 1",
                "Assist 2",
                "Period",
                "Time",
                "Video Time",
                "Goal",
                "Assist 1",
                "Assist 2",
            ],
            [1, "14:20", "0:10", "#12", "#34", None, 1, "10:00", "0:50", "#91", None, None],
        ]
    )
    goals_xlsx = stats_dir / "goals.xlsx"
    df.to_excel(goals_xlsx, index=False, header=False)

    outdir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_stats_inputs.py",
            "--input",
            f"{goals_xlsx}:HOME:home_team=Utah:away_team=Opp",
            "--outdir",
            str(outdir),
            "--no-scripts",
            "--dump-events",
        ],
    )
    pss.main()
    out = capsys.readouterr().out
    assert "[dump-events] utah-1" in out
    assert "Goal" in out
    assert "P1 14:20" in out


def should_build_webapp_logo_payload_from_meta_path_and_base64(tmp_path: Path, capsys):
    # Minimal valid PNG header + chunk (not a full image, but enough for sniffing).
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    logo_path = tmp_path / "logo.png"
    logo_path.write_bytes(png_bytes)

    meta = {"home_logo": str(logo_path)}
    fields = pss._load_logo_fields_from_meta(meta, base_dir=None, warn_label="test")
    assert "home_logo_b64" in fields
    assert fields.get("home_logo_content_type") == "image/png"
    assert "away_logo_b64" not in fields

    # Missing path should only warn, not error.
    meta2 = {"away_logo": str(tmp_path / "missing.png")}
    fields2 = pss._load_logo_fields_from_meta(meta2, base_dir=None, warn_label="test2")
    assert fields2 == {}
    err = capsys.readouterr().err
    assert "away_logo path does not exist" in err

    # Base64 path (with optional content type).
    b64 = fields["home_logo_b64"]
    meta3 = {"away_logo_base64": b64, "away_logo_content_type": "image/png"}
    fields3 = pss._load_logo_fields_from_meta(meta3, base_dir=None, warn_label="test3")
    assert fields3.get("away_logo_b64") == b64
    assert fields3.get("away_logo_content_type") == "image/png"


def should_parse_webapp_starts_at_from_meta_date_and_starts_at(capsys):
    assert pss._starts_at_from_meta({"date": "2026-01-04"}) == "2026-01-04 00:00:00"
    assert pss._starts_at_from_meta({"date": "1/4/2026"}) == "2026-01-04 00:00:00"
    assert (
        pss._starts_at_from_meta({"date": "2026-01-04", "time": "13:05"}) == "2026-01-04 13:05:00"
    )
    assert (
        pss._starts_at_from_meta({"date": "2026-01-04", "time": "13:05:09"})
        == "2026-01-04 13:05:09"
    )
    assert pss._starts_at_from_meta({"starts_at": "2026-01-04T13:05:09"}) == "2026-01-04 13:05:09"
    assert pss._starts_at_from_meta({"starts_at": "2026-01-04"}) == "2026-01-04 00:00:00"

    bad = pss._starts_at_from_meta({"date": "nope"}, warn_label="x")
    assert bad is None
    err = capsys.readouterr().err
    assert "could not parse date" in err


if __name__ == "__main__":
    # Make `bazel test //tests:test_parse_stats_inputs` run pytest collection.
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

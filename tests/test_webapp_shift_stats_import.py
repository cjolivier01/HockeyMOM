import importlib.util
import os


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_parse_shift_spreadsheet_player_stats_csv():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    csv_text = (
        "Player,GP,Goals,Assists,Shots,SOG,xG,Giveaways,Takeaways,"
        "Controlled Entry For (On-Ice),Controlled Entry Against (On-Ice),"
        "Controlled Exit For (On-Ice),Controlled Exit Against (On-Ice),"
        "Plus Minus,GF Counted,GA Counted,Shifts,TOI Total,Average Shift,Median Shift,Longest Shift,Shortest Shift,"
        "Period 1 TOI,Period 1 Shifts,Period 1 GF,Period 1 GA,TOI Total (Video)\n"
        '" 8 Adam Ro",1,1,2,5,4,3,0,1,2,3,4,5,1,1,0,10,12:34,0:45,0:42,1:10,0:10,4:10,3,1,0,12:00\n'
    )

    rows = mod.parse_shift_stats_player_stats_csv(csv_text)
    assert len(rows) == 1
    r = rows[0]
    assert r["jersey_number"] == "8"
    assert r["name_norm"] == "adamro"
    assert r["stats"]["goals"] == 1
    assert r["stats"]["assists"] == 2
    assert r["stats"]["shots"] == 5
    assert r["stats"]["sog"] == 4
    assert r["stats"]["expected_goals"] == 3
    assert r["stats"]["toi_seconds"] == 12 * 60 + 34
    assert r["stats"]["video_toi_seconds"] == 12 * 60
    assert r["stats"]["sb_avg_shift_seconds"] == 45
    assert r["stats"]["sb_median_shift_seconds"] == 42
    assert r["stats"]["sb_longest_shift_seconds"] == 70
    assert r["stats"]["sb_shortest_shift_seconds"] == 10

    p1 = r["period_stats"][1]
    assert p1["toi_seconds"] == 4 * 60 + 10
    assert p1["shifts"] == 3
    assert p1["gf"] == 1
    assert p1["ga"] == 0


def should_parse_shift_spreadsheet_game_stats_csv():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    csv_text = "Stat,sharks-12-1-r2\nGoals For,3\nShots For,22\nxG For,5\n"
    out = mod.parse_shift_stats_game_stats_csv(csv_text)
    assert out["_label"] == "sharks-12-1-r2"
    assert out["Goals For"] == "3"
    assert out["Shots For"] == "22"
    assert out["xG For"] == "5"


def should_format_seconds_to_toi_strings():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    assert mod.format_seconds_to_mmss_or_hhmmss(12 * 60 + 34) == "12:34"
    assert mod.format_seconds_to_mmss_or_hhmmss(1 * 3600 + 2 * 60 + 3) == "1:02:03"


def should_parse_shift_spreadsheet_otg_ota_columns():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    csv_text = 'Player,OT Goals,OT Assists\n" 8 Adam Ro",1,2\n'
    rows = mod.parse_shift_stats_player_stats_csv(csv_text)
    assert len(rows) == 1
    r = rows[0]
    assert r["stats"]["ot_goals"] == 1
    assert r["stats"]["ot_assists"] == 2


def should_not_include_empty_period_columns_in_consolidated_stats():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "parse_shift_spreadsheet_mod", "scripts/parse_shift_spreadsheet.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore

    rows = [
        {"player": "1_Ethan", "gp": "1", "goals": "1", "assists": "0", "P4_GF": "1"},
        {"player": "2_Other", "gp": "1", "goals": "0", "assists": "0"},
    ]
    df, cols = mod._build_stats_dataframe(rows, [1, 2, 3, 4], include_shifts_in_stats=False)  # type: ignore[attr-defined]
    assert "P4_GA" not in cols
    assert "P4_GF" in cols


def should_not_write_times_files_when_no_scripts():
    import importlib.util
    import tempfile
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "parse_shift_spreadsheet_mod", "scripts/parse_shift_spreadsheet.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)

        mod._write_video_times_and_scripts(  # type: ignore[attr-defined]
            outdir, {"1_Ethan": [("0:00", "0:10")]}, create_scripts=False
        )
        mod._write_scoreboard_times(  # type: ignore[attr-defined]
            outdir, {"1_Ethan": [(1, "0:00", "0:10")]}, create_scripts=False
        )

        stats_dir = outdir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        ctx = mod.EventLogContext(  # type: ignore[attr-defined]
            event_counts_by_player={},
            event_counts_by_type_team={("Goal", "For"): 1},
            event_instances={},
            event_player_rows=[
                {
                    "event_type": "Goal",
                    "team": "For",
                    "player": "1_Ethan",
                    "jersey": "1",
                    "period": 1,
                    "video_s": 100,
                    "game_s": 200,
                }
            ],
            team_roster={},
            team_excluded={},
        )

        mod._write_event_summaries_and_clips(  # type: ignore[attr-defined]
            outdir,
            stats_dir,
            ctx,
            conv_segments_by_period={},
            create_scripts=False,
        )
        mod._write_player_event_highlights(  # type: ignore[attr-defined]
            outdir,
            ctx,
            conv_segments_by_period={},
            player_keys=["1_Ethan"],
            create_scripts=False,
        )

        assert (stats_dir / "event_summary.csv").exists()
        assert not list(outdir.glob("*_times.txt"))
        assert not list(outdir.glob("clip_*.sh"))


def should_process_t2s_only_game_without_spreadsheets():
    import importlib.util
    import tempfile
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "parse_shift_spreadsheet_mod_t2s", "scripts/parse_shift_spreadsheet.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore

    class DummyT2SApi:
        def get_game_details(self, game_id: int):  # noqa: ARG002
            return {
                "stats": {
                    "homePlayers": [{"number": "1", "position": "F", "name": "Ethan Olivier"}],
                    "awayPlayers": [{"number": "99", "position": "F", "name": "Opponent Player"}],
                    "homeScoring": [
                        {
                            "period": "OT",
                            "time": "0:45",
                            "extra": "",
                            "goal": {"text": "#1 Ethan Olivier"},
                            "assist1": "",
                            "assist2": "",
                        }
                    ],
                    "awayScoring": [],
                }
            }

    # Install dummy T2S API to avoid network/DB.
    mod._t2s_api = DummyT2SApi()  # type: ignore[attr-defined]
    mod._t2s_api_loaded = True  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        outdir = base / "out"
        hockey_db_dir = base / "db"
        final_outdir, stats_rows, periods, per_player_events = mod.process_t2s_only_game(  # type: ignore[attr-defined]
            t2s_id=51602,
            side="home",
            outdir=outdir,
            label="t2s-51602",
            hockey_db_dir=hockey_db_dir,
            include_shifts_in_stats=False,
        )

        assert (final_outdir / "stats" / "player_stats.csv").exists()
        assert (final_outdir / "stats" / "game_stats.csv").exists()

        rows_by_player = {r["player"]: r for r in stats_rows}
        assert rows_by_player["1_Ethan_Olivier"]["goals"] == "1"
        assert rows_by_player["1_Ethan_Olivier"]["assists"] == "0"
        assert rows_by_player["1_Ethan_Olivier"]["ot_goals"] == "1"
        assert rows_by_player["1_Ethan_Olivier"]["ot_assists"] == "0"
        assert 4 in periods  # OT -> period 4

        assert len(per_player_events["1_Ethan_Olivier"]["goals"]) == 1

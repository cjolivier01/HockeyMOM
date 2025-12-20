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


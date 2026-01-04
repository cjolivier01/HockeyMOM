import importlib.util
import os


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_import_webapp_without_db_init():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()
    assert hasattr(mod, "create_app")
    app = mod.create_app()
    # Ensure key routes are registered
    paths = {str(r) for r in app.url_map.iter_rules()}
    expected = {
        "/",
        "/teams",
        "/teams/new",
        "/teams/<int:team_id>",
        "/teams/<int:team_id>/edit",
        "/teams/<int:team_id>/players/new",
        "/teams/<int:team_id>/players/<int:player_id>/edit",
        "/teams/<int:team_id>/players/<int:player_id>/delete",
        "/schedule",
        "/schedule/new",
        "/hky/games/<int:game_id>",
        "/game_types",
        "/media/team_logo/<int:team_id>",
    }
    assert expected.issubset(paths)


def should_parse_date_formats():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()
    f = mod.parse_dt_or_none
    assert f("2025-01-02") == "2025-01-02 00:00:00"
    assert f("2025-01-02T12:34") == "2025-01-02 12:34:00"
    assert f("") is None
    assert f(None) is None


def should_compute_team_stats_from_rows():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *_args, **_kwargs):
            return 1

        def fetchall(self):
            return list(self._rows)

    class _Conn:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self, *_args, **_kwargs):
            return _Cur(self._rows)

    # team_id=1: wins one, loses one, ties one
    rows = [
        {"team1_id": 1, "team2_id": 2, "team1_score": 3, "team2_score": 1},  # win
        {"team1_id": 2, "team2_id": 1, "team1_score": 2, "team2_score": 2},  # tie
        {"team1_id": 3, "team2_id": 1, "team1_score": 4, "team2_score": 2},  # loss
    ]
    stats = mod.compute_team_stats(_Conn(rows), team_id=1, user_id=123)
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["ties"] == 1
    assert stats["gf"] == 3 + 2 + 2
    assert stats["ga"] == 1 + 2 + 4
    assert stats["points"] == 1 * 2 + 1 * 1


def should_compute_league_points_from_regular_only_but_keep_record_from_all_games():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *_args, **_kwargs):
            return 1

        def fetchall(self):
            return list(self._rows)

    class _Conn:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self, *_args, **_kwargs):
            return _Cur(self._rows)

    # team_id=1:
    # - Regular Season win (counts for points)
    # - Preseason tie (does not count for points)
    # - Regular Season win in External division (does not count for points)
    rows = [
        {
            "team1_id": 1,
            "team2_id": 2,
            "team1_score": 3,
            "team2_score": 1,
            "game_type_name": "Regular Season",
            "league_division_name": "12 AA",
            "team1_league_division_name": "12 AA",
            "team2_league_division_name": "12 AA",
        },
        {
            "team1_id": 2,
            "team2_id": 1,
            "team1_score": 2,
            "team2_score": 2,
            "game_type_name": "Preseason",
            "league_division_name": "12 AA",
            "team1_league_division_name": "12 AA",
            "team2_league_division_name": "12 AA",
        },
        {
            "team1_id": 1,
            "team2_id": 3,
            "team1_score": 4,
            "team2_score": 2,
            "game_type_name": "Regular Season",
            "league_division_name": "External",
            "team1_league_division_name": "12 AA",
            "team2_league_division_name": "External",
        },
    ]
    stats = mod.compute_team_stats_league(_Conn(rows), team_id=1, league_id=999)
    # Record/GF/GA include all games
    assert stats["wins"] == 2
    assert stats["losses"] == 0
    assert stats["ties"] == 1
    assert stats["gf"] == 3 + 2 + 4
    assert stats["ga"] == 1 + 2 + 2
    # Points only count regular-season games that are not external.
    assert stats["points"] == 2
    # Total points include all games.
    assert stats["points_total"] == 2 * 2 + 1 * 1


def should_aggregate_player_totals_from_rows():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()

    class _Cur:
        def __init__(self, rows):
            self._rows = rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *_args, **_kwargs):
            return 1

        def fetchall(self):
            return list(self._rows)

    class _Conn:
        def __init__(self, rows):
            self._rows = rows

        def cursor(self, *_args, **_kwargs):
            return _Cur(self._rows)

    rows = [
        {"player_id": 10, "goals": 2, "assists": 1, "pim": 0, "shots": 5},
        {"player_id": 11, "goals": 0, "assists": 2, "pim": 2, "shots": 1},
    ]
    agg = mod.aggregate_players_totals(_Conn(rows), team_id=1, user_id=123)
    assert agg[10]["goals"] == 2 and agg[10]["assists"] == 1 and agg[10]["points"] == 3
    assert agg[11]["goals"] == 0 and agg[11]["assists"] == 2 and agg[11]["points"] == 2


def should_sort_standings_by_points_then_tiebreakers():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
    mod = _load_app_module()
    f = mod.sort_key_team_standings

    a = {"id": 1, "name": "A"}
    b = {"id": 2, "name": "B"}
    c = {"id": 3, "name": "C"}
    # A and B tied on points, A has better goal diff; C higher points.
    stats = {
        1: {"points": 10, "wins": 5, "gf": 20, "ga": 10},
        2: {"points": 10, "wins": 5, "gf": 18, "ga": 12},
        3: {"points": 12, "wins": 6, "gf": 15, "ga": 5},
    }
    ordered = sorted([a, b, c], key=lambda tr: f(tr, stats[tr["id"]]))
    assert [t["id"] for t in ordered] == [3, 1, 2]

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from flask import render_template


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class _DummyPyMySQL:
    class cursors:
        DictCursor = object


class FakeConn:
    def cursor(self, cursorclass: Any = None):
        return FakeCursor(dict_mode=cursorclass is not None)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeCursor:
    def __init__(self, *, dict_mode: bool) -> None:
        self._dict_mode = dict_mode
        self._rows: list[Any] = []
        self._idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query: str, params: Any = None) -> int:
        q = " ".join(str(query).split())
        p = tuple(params or ())
        self._rows = []
        self._idx = 0

        def d(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row)

        # open_db() league access validation
        if q.startswith("SELECT 1 FROM leagues l LEFT JOIN league_members m"):
            self._rows = [(1,)]
            return 1

        # schedule() filter options
        if q.startswith("SELECT DISTINCT division_name FROM league_teams"):
            self._rows = [d({"division_name": "External"})]
            return 1

        if "SELECT DISTINCT t.id, t.name FROM league_teams lt JOIN teams t" in q:
            self._rows = [d({"id": 1, "name": "Team A"})]
            return 1

        # schedule() games list query
        if "FROM league_games lg" in q and "JOIN hky_games g" in q:
            self._rows = [
                d(
                    {
                        "id": 123,
                        "user_id": 1,
                        "team1_id": 1,
                        "team2_id": 2,
                        "team1_name": "Team A",
                        "team2_name": "Team B",
                        "division_name": "External",
                        "game_type_name": None,
                        "starts_at": None,
                        "location": "",
                        "team1_score": None,
                        "team2_score": None,
                        "is_final": 0,
                    }
                )
            ]
            return 1

        # _is_league_admin()
        if q.startswith("SELECT 1 FROM leagues WHERE id=%s AND owner_user_id=%s"):
            self._rows = []
            return 1
        if q.startswith("SELECT 1 FROM league_members WHERE league_id=%s AND user_id=%s"):
            self._rows = []
            return 1

        raise AssertionError(f"Unhandled SQL: {q!r} params={p!r}")

    def fetchone(self) -> Any:
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def fetchall(self) -> list[Any]:
        if self._idx >= len(self._rows):
            return []
        out = self._rows[self._idx :]
        self._idx = len(self._rows)
        return out


@pytest.fixture()
def mod_and_app(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    monkeypatch.setattr(mod, "get_db", lambda: FakeConn())
    app = mod.create_app()
    app.testing = True
    return mod, app


def should_default_external_division_game_type_to_tournament_in_schedule_table(mod_and_app):
    _mod, app = mod_and_app
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_email"] = "u@example.com"
        sess["league_id"] = 42
    html = client.get("/schedule").get_data(as_text=True)
    assert ">Tournament<" in html


def should_default_external_division_game_type_to_tournament_in_team_schedule_table(mod_and_app):
    _mod, app = mod_and_app
    with app.test_request_context("/teams/1"):
        html = render_template(
            "team_detail.html",
            team={"id": 1, "name": "Team A", "logo_path": "", "is_external": 0},
            roster_players=[],
            players=[],
            head_coaches=[],
            assistant_coaches=[],
            player_stats_columns=[],
            player_stats_rows=[],
            recent_player_stats_columns=[],
            recent_player_stats_rows=[],
            recent_n=5,
            recent_sort="points",
            recent_dir="desc",
            tstats={"wins": 0, "losses": 0, "ties": 0, "gf": 0, "ga": 0, "points": 0},
            schedule_games=[
                {
                    "id": 123,
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_name": "Team A",
                    "team2_name": "Team B",
                    "division_name": "External",
                    "game_type_name": None,
                    "starts_at": None,
                    "location": "",
                    "team1_score": None,
                    "team2_score": None,
                }
            ],
            editable=False,
        )
    assert "<th>Type</th>" in html
    assert ">Tournament<" in html


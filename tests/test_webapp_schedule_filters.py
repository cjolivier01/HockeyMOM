from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest


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

        # schedule() filter options
        if q.startswith("SELECT DISTINCT division_name FROM league_teams"):
            self._rows = [d({"division_name": "DivA"}), d({"division_name": "DivB"})]
            return 1

        # open_db() league access validation
        if q.startswith("SELECT 1 FROM leagues l LEFT JOIN league_members m"):
            self._rows = [(1,)]
            return 1

        if "SELECT DISTINCT t.id, t.name FROM league_teams lt JOIN teams t" in q:
            league_id = int(p[0])
            assert league_id == 42
            if "lt.division_name=%s" in q:
                div = str(p[1])
                if div == "DivA":
                    self._rows = [d({"id": 1, "name": "Team A"})]
                else:
                    self._rows = [d({"id": 2, "name": "Team B"})]
            else:
                self._rows = [d({"id": 1, "name": "Team A"}), d({"id": 2, "name": "Team B"})]
            return 1

        # schedule() games list query
        if "FROM league_games lg" in q and "JOIN hky_games g" in q:
            self._rows = []
            return 1

        # Personal schedule query (should not be hit in this test, but keep harmless)
        if q.startswith("SELECT g.*, t1.name AS team1_name") and "FROM hky_games g" in q:
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
def client(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    monkeypatch.setattr(mod, "get_db", lambda: FakeConn())
    app = mod.create_app()
    app.testing = True
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_email"] = "u@example.com"
        sess["league_id"] = 42
    return c


def should_filter_team_dropdown_by_division(client):
    html = client.get("/schedule?division=DivA").get_data(as_text=True)
    assert "Team A" in html
    assert "Team B" not in html

    html2 = client.get("/schedule").get_data(as_text=True)
    assert "Team A" in html2 and "Team B" in html2

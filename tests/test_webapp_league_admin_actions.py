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
    def __init__(self) -> None:
        self.leagues = {1: {"id": 1, "owner_user_id": 10, "is_shared": 0, "is_public": 0}}
        self.league_members = {(1, 20): {"league_id": 1, "user_id": 20, "role": "admin"}}
        self.league_games: list[dict[str, Any]] = []
        self.league_teams: list[dict[str, Any]] = []
        self.teams: dict[int, dict[str, Any]] = {}
        self.hky_games: dict[int, dict[str, Any]] = {}

    def cursor(self, cursorclass: Any = None):
        return FakeCursor(self, dict_mode=cursorclass is not None)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeCursor:
    def __init__(self, conn: FakeConn, *, dict_mode: bool) -> None:
        self._conn = conn
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

        def t(*vals: Any) -> tuple[Any, ...]:
            return tuple(vals)

        if q == "SELECT 1 FROM leagues WHERE id=%s AND owner_user_id=%s":
            league_id, user_id = int(p[0]), int(p[1])
            league = self._conn.leagues.get(league_id)
            if league and int(league["owner_user_id"]) == user_id:
                self._rows = [t(1)]
            return 1

        if q == "SELECT 1 FROM league_members WHERE league_id=%s AND user_id=%s AND role IN ('admin','owner')":
            league_id, user_id = int(p[0]), int(p[1])
            m = self._conn.league_members.get((league_id, user_id))
            if m and str(m.get("role")) in ("admin", "owner"):
                self._rows = [t(1)]
            return 1

        if q == "UPDATE leagues SET is_shared=%s, is_public=%s, updated_at=%s WHERE id=%s":
            is_shared, is_public, _updated_at, league_id = p
            league_id = int(league_id)
            self._conn.leagues[league_id]["is_shared"] = int(is_shared)
            self._conn.leagues[league_id]["is_public"] = int(is_public)
            return 1

        if q == "SELECT owner_user_id FROM leagues WHERE id=%s":
            league_id = int(p[0])
            self._rows = [t(int(self._conn.leagues[league_id]["owner_user_id"]))]
            return 1

        if q.startswith("SELECT game_id FROM league_games"):
            self._rows = []
            return 1

        if q.startswith("SELECT team_id FROM league_teams"):
            self._rows = []
            return 1

        if q == "DELETE FROM league_games WHERE league_id=%s":
            return 1

        if q == "DELETE FROM league_teams WHERE league_id=%s":
            return 1

        if q == "DELETE FROM league_members WHERE league_id=%s":
            league_id = int(p[0])
            self._conn.league_members = {k: v for k, v in self._conn.league_members.items() if k[0] != league_id}
            return 1

        if q == "DELETE FROM leagues WHERE id=%s":
            league_id = int(p[0])
            self._conn.leagues.pop(league_id, None)
            return 1

        raise AssertionError(f"Unhandled SQL in FakeCursor.execute: {q!r} params={p!r}")

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
def client_and_db(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    fake_db = FakeConn()
    monkeypatch.setattr(mod, "get_db", lambda: fake_db)
    app = mod.create_app()
    app.testing = True
    return app.test_client(), fake_db


def should_allow_league_admin_to_update_shared_and_public(client_and_db):
    client, db = client_and_db
    with client.session_transaction() as sess:
        sess["user_id"] = 20
        sess["user_email"] = "admin@example.com"

    r = client.post("/leagues/1/update", data={"is_shared": "1", "is_public": "1"})
    assert r.status_code == 302
    assert db.leagues[1]["is_shared"] == 1
    assert db.leagues[1]["is_public"] == 1


def should_allow_league_admin_to_delete_league(client_and_db):
    client, db = client_and_db
    with client.session_transaction() as sess:
        sess["user_id"] = 20
        sess["user_email"] = "admin@example.com"

    r = client.post("/leagues/1/delete")
    assert r.status_code == 302
    assert 1 not in db.leagues


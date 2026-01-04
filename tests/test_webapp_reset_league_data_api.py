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
        self.users_by_email = {"cjolivier01@gmail.com": 10}
        self.leagues = {1: {"id": 1, "name": "Norcal", "owner_user_id": 10}, 2: {"id": 2, "name": "Other", "owner_user_id": 11}}
        self.league_games = [
            {"league_id": 1, "game_id": 1001},
            {"league_id": 1, "game_id": 1002},
            {"league_id": 2, "game_id": 1002},  # shared game
        ]
        self.league_teams = [
            {"league_id": 1, "team_id": 201},
            {"league_id": 1, "team_id": 202},
            {"league_id": 2, "team_id": 202},  # shared team
        ]
        self.hky_games = {
            1001: {"id": 1001, "team1_id": 201, "team2_id": 202},
            1002: {"id": 1002, "team1_id": 202, "team2_id": 999},
        }
        self.teams = {
            201: {"id": 201, "is_external": 1},
            202: {"id": 202, "is_external": 1},
            999: {"id": 999, "is_external": 0},
        }
        self.players = [
            {"id": 1, "team_id": 201, "name": "Skater A"},
            {"id": 2, "team_id": 202, "name": "Skater B"},
        ]
        self.player_stats = [
            {"game_id": 1001, "player_id": 1},
            {"game_id": 1002, "player_id": 2},
        ]

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

        if q == "SELECT id FROM users WHERE email=%s":
            email = str(p[0]).strip().lower()
            uid = self._conn.users_by_email.get(email)
            self._rows = [t(uid)] if uid is not None else []
            return 1

        if q == "SELECT id FROM leagues WHERE name=%s AND owner_user_id=%s":
            name = str(p[0]).strip()
            owner = int(p[1])
            for lid, row in self._conn.leagues.items():
                if str(row["name"]) == name and int(row["owner_user_id"]) == owner:
                    self._rows = [t(int(lid))]
                    break
            return 1

        if q == "SELECT COUNT(*) FROM league_games WHERE league_id=%s":
            lid = int(p[0])
            self._rows = [t(sum(1 for r in self._conn.league_games if int(r["league_id"]) == lid))]
            return 1

        if q == "SELECT COUNT(*) FROM league_teams WHERE league_id=%s":
            lid = int(p[0])
            self._rows = [t(sum(1 for r in self._conn.league_teams if int(r["league_id"]) == lid))]
            return 1

        if "SELECT game_id FROM league_games WHERE league_id=%s AND game_id NOT IN" in q:
            lid = int(p[0])
            other_lid = int(p[1])
            mine = {int(r["game_id"]) for r in self._conn.league_games if int(r["league_id"]) == lid}
            others = {int(r["game_id"]) for r in self._conn.league_games if int(r["league_id"]) != other_lid}
            ex = sorted([gid for gid in mine if gid not in others])
            self._rows = [t(x) for x in ex]
            return 1

        if "SELECT team_id FROM league_teams WHERE league_id=%s AND team_id NOT IN" in q:
            lid = int(p[0])
            other_lid = int(p[1])
            mine = {int(r["team_id"]) for r in self._conn.league_teams if int(r["league_id"]) == lid}
            others = {int(r["team_id"]) for r in self._conn.league_teams if int(r["league_id"]) != other_lid}
            ex = sorted([tid for tid in mine if tid not in others])
            self._rows = [t(x) for x in ex]
            return 1

        if q.startswith("SELECT COUNT(*) FROM player_stats WHERE game_id IN"):
            game_ids = {int(x) for x in p}
            self._rows = [t(sum(1 for r in self._conn.player_stats if int(r["game_id"]) in game_ids))]
            return 1

        if q.startswith("SELECT COUNT(*) FROM hky_games WHERE id IN"):
            gids = {int(x) for x in p}
            self._rows = [t(sum(1 for gid in self._conn.hky_games.keys() if int(gid) in gids))]
            return 1

        if q == "DELETE FROM league_games WHERE league_id=%s":
            lid = int(p[0])
            self._conn.league_games = [r for r in self._conn.league_games if int(r["league_id"]) != lid]
            return 1

        if q == "DELETE FROM league_teams WHERE league_id=%s":
            lid = int(p[0])
            self._conn.league_teams = [r for r in self._conn.league_teams if int(r["league_id"]) != lid]
            return 1

        if q.startswith("DELETE FROM hky_games WHERE id IN"):
            gids = {int(x) for x in p}
            for gid in list(self._conn.hky_games.keys()):
                if int(gid) in gids:
                    del self._conn.hky_games[int(gid)]
            # Cascade to player_stats by game_id
            self._conn.player_stats = [r for r in self._conn.player_stats if int(r["game_id"]) not in gids]
            return 1

        if q.startswith("SELECT id, user_id, is_external FROM teams WHERE id IN"):
            ids = [int(x) for x in p]
            rows = []
            for tid in ids:
                tr = self._conn.teams.get(tid)
                if not tr:
                    continue
                # Fake owner: team ids < 900 are owned by user 10
                rows.append(t(int(tid), 10, int(tr.get("is_external") or 0)))
            self._rows = rows
            return 1

        if q.startswith("SELECT id, is_external FROM teams WHERE id IN"):
            ids = [int(x) for x in p]
            rows = []
            for tid in ids:
                tr = self._conn.teams.get(tid)
                if not tr:
                    continue
                rows.append(t(int(tid), int(tr.get("is_external") or 0)))
            self._rows = rows
            return 1

        if "SELECT DISTINCT team1_id AS tid FROM hky_games WHERE team1_id IN" in q and "UNION" in q:
            # Params are eligible*2
            half = int(len(p) / 2)
            eligible = {int(x) for x in p[:half]}
            used = set()
            for g in self._conn.hky_games.values():
                if int(g["team1_id"]) in eligible:
                    used.add(int(g["team1_id"]))
                if int(g["team2_id"]) in eligible:
                    used.add(int(g["team2_id"]))
            self._rows = [t(x) for x in sorted(used)]
            return 1

        if q.startswith("SELECT COUNT(*) FROM players WHERE team_id IN"):
            tids = {int(x) for x in p}
            self._rows = [t(sum(1 for r in self._conn.players if int(r["team_id"]) in tids))]
            return 1

        if q.startswith("SELECT COUNT(*) FROM teams WHERE id IN"):
            tids = {int(x) for x in p}
            self._rows = [t(sum(1 for tid in self._conn.teams.keys() if int(tid) in tids))]
            return 1

        if q.startswith("DELETE FROM teams WHERE id IN"):
            tids = {int(x) for x in p}
            for tid in list(self._conn.teams.keys()):
                if int(tid) in tids:
                    del self._conn.teams[int(tid)]
            # Cascade to players by team_id
            self._conn.players = [r for r in self._conn.players if int(r["team_id"]) not in tids]
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
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    fake_db = FakeConn()
    monkeypatch.setattr(mod, "get_db", lambda: fake_db)
    app = mod.create_app()
    app.testing = True
    return app.test_client(), fake_db


def should_require_import_auth_for_internal_reset_endpoint(client_and_db):
    client, _db = client_and_db
    r = client.post("/api/internal/reset_league_data", json={"owner_email": "cjolivier01@gmail.com", "league_name": "Norcal"})
    assert r.status_code == 401


def should_reset_league_data_via_hidden_api_and_preserve_shared_entities(client_and_db):
    client, db = client_and_db
    r = client.post(
        "/api/internal/reset_league_data",
        json={"owner_email": "cjolivier01@gmail.com", "league_name": "Norcal"},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    out = r.get_json()
    assert out["ok"] is True
    assert int(out["league_id"]) == 1
    # League 1 mappings removed.
    assert not any(int(x["league_id"]) == 1 for x in db.league_games)
    assert not any(int(x["league_id"]) == 1 for x in db.league_teams)
    # Shared game remains (mapped to league 2).
    assert 1002 in db.hky_games
    # Exclusive game was deleted.
    assert 1001 not in db.hky_games
    # Exclusive team 201 was deleted; shared team 202 remains.
    assert 201 not in db.teams
    assert 202 in db.teams


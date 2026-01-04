from __future__ import annotations

import importlib.util
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
        self.leagues = {1: {"id": 1, "name": "Public League", "is_public": 1, "owner_user_id": 10, "is_shared": 0}}
        self.teams = {
            101: {"id": 101, "user_id": 10, "name": "Team A", "is_external": 1, "logo_path": None},
            102: {"id": 102, "user_id": 10, "name": "Team B", "is_external": 1, "logo_path": None},
        }
        self.hky_games = {
            1001: {
                "id": 1001,
                "user_id": 10,
                "team1_id": 101,
                "team2_id": 102,
                "starts_at": "2026-01-02 10:00:00",
                "location": "Rink",
                "team1_score": 1,
                "team2_score": 2,
                "is_final": 1,
                "notes": '{"timetoscore_game_id":123,"timetoscore_season_id":31}',
                "created_at": "2026-01-01 00:00:00",
                "updated_at": None,
                "stats_imported_at": None,
            }
        }
        self.league_games = [{"league_id": 1, "game_id": 1001, "division_name": "10 A"}]
        self.players = [
            {"id": 501, "team_id": 101, "name": "Alice", "jersey_number": "9"},
            {"id": 502, "team_id": 102, "name": "Bob", "jersey_number": "12"},
        ]
        self.player_stats: list[dict[str, Any]] = []
        self.player_period_stats: list[dict[str, Any]] = []
        self.hky_game_stats: dict[int, dict[str, Any]] = {}
        self.hky_game_events: dict[int, dict[str, Any]] = {}
        self.hky_game_player_stats_csv: dict[int, dict[str, Any]] = {}

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

        def d(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row)

        def t(*vals: Any) -> tuple[Any, ...]:
            return tuple(vals)

        if q == "SELECT id, name FROM leagues WHERE id=%s AND is_public=1":
            lid = int(p[0])
            league = self._conn.leagues.get(lid)
            if league and int(league.get("is_public") or 0) == 1:
                self._rows = [d({"id": league["id"], "name": league["name"]})]
            return 1

        if "SELECT id FROM hky_games WHERE notes LIKE %s LIMIT 1" in q:
            token = str(p[0])
            for gid, g in self._conn.hky_games.items():
                if token.strip("%") in str(g.get("notes") or ""):
                    self._rows = [t(int(gid))]
                    return 1
            self._rows = []
            return 1

        if q == "SELECT * FROM hky_games WHERE id=%s":
            gid = int(p[0])
            g = self._conn.hky_games.get(gid)
            self._rows = [d(g)] if g else []
            return 1

        if q == "SELECT id, team_id, name, jersey_number FROM players WHERE team_id IN (%s,%s)":
            t1, t2 = int(p[0]), int(p[1])
            rows = [pl for pl in self._conn.players if int(pl["team_id"]) in (t1, t2)]
            self._rows = [d(r) for r in rows]
            return 1

        if q == "SELECT events_csv FROM hky_game_events WHERE game_id=%s":
            gid = int(p[0])
            ev = self._conn.hky_game_events.get(gid)
            self._rows = [t(ev.get("events_csv"))] if ev else []
            return 1

        if q == "SELECT player_stats_csv FROM hky_game_player_stats_csv WHERE game_id=%s":
            gid = int(p[0])
            ps = self._conn.hky_game_player_stats_csv.get(gid)
            self._rows = [t(ps.get("player_stats_csv"))] if ps else []
            return 1

        if q.startswith("INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at) VALUES"):
            gid, events_csv, source_label, updated_at = p
            gid = int(gid)
            self._conn.hky_game_events[gid] = {
                "game_id": gid,
                "events_csv": str(events_csv),
                "source_label": source_label,
                "updated_at": updated_at,
            }
            return 1

        if q.startswith(
            "INSERT INTO hky_game_player_stats_csv(game_id, player_stats_csv, source_label, updated_at) VALUES"
        ):
            gid, csv_text, source_label, updated_at = p
            gid_i = int(gid)
            if "ON DUPLICATE KEY UPDATE" in q:
                self._conn.hky_game_player_stats_csv[gid_i] = {
                    "player_stats_csv": str(csv_text),
                    "source_label": source_label,
                    "updated_at": updated_at,
                }
            else:
                if gid_i not in self._conn.hky_game_player_stats_csv:
                    self._conn.hky_game_player_stats_csv[gid_i] = {
                        "player_stats_csv": str(csv_text),
                        "source_label": source_label,
                        "updated_at": updated_at,
                    }
            return 1

        if q.startswith("INSERT INTO player_stats(") and "ON DUPLICATE KEY UPDATE" in q:
            cols_part = q.split("INSERT INTO player_stats(", 1)[1].split(") VALUES", 1)[0]
            cols = [c.strip() for c in cols_part.split(",") if c.strip()]
            row = dict(zip(cols, p))
            gid = int(row["game_id"])
            pid = int(row["player_id"])
            existing = None
            for r in self._conn.player_stats:
                if int(r["game_id"]) == gid and int(r["player_id"]) == pid:
                    existing = r
                    break
            if existing is None:
                self._conn.player_stats.append(dict(row))
            else:
                for k, v in row.items():
                    if k in {"user_id", "team_id", "game_id", "player_id"}:
                        existing[k] = v
                    else:
                        if v is not None:
                            existing[k] = v
            return 1

        if q.startswith("INSERT INTO player_period_stats("):
            cols_part = q.split("INSERT INTO player_period_stats(", 1)[1].split(") VALUES", 1)[0]
            cols = [c.strip() for c in cols_part.split(",") if c.strip()]
            row = dict(zip(cols, p))
            gid = int(row["game_id"])
            pid = int(row["player_id"])
            per = int(row["period"])
            existing = None
            for r in self._conn.player_period_stats:
                if int(r["game_id"]) == gid and int(r["player_id"]) == pid and int(r["period"]) == per:
                    existing = r
                    break
            if existing is None:
                self._conn.player_period_stats.append(dict(row))
            else:
                for k, v in row.items():
                    if v is not None:
                        existing[k] = v
            return 1

        if q.startswith("INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at) VALUES") and "ON DUPLICATE KEY UPDATE" in q:
            gid, events_csv, source_label, updated_at = p
            gid = int(gid)
            self._conn.hky_game_events[gid] = {
                "game_id": gid,
                "events_csv": str(events_csv),
                "source_label": source_label,
                "updated_at": updated_at,
            }
            return 1

        if q == "UPDATE hky_games SET stats_imported_at=%s WHERE id=%s":
            _ts, gid = p
            gid = int(gid)
            if gid in self._conn.hky_games:
                self._conn.hky_games[gid]["stats_imported_at"] = _ts
            return 1

        if "FROM league_games lg JOIN hky_games g ON lg.game_id=g.id" in q and "WHERE g.id=%s AND lg.league_id=%s" in q:
            gid, league_id = int(p[0]), int(p[1])
            ok = any(int(lg["league_id"]) == league_id and int(lg["game_id"]) == gid for lg in self._conn.league_games)
            if ok:
                g = dict(self._conn.hky_games[gid])
                t1 = self._conn.teams[int(g["team1_id"])]
                t2 = self._conn.teams[int(g["team2_id"])]
                self._rows = [d(dict(g, team1_name=t1["name"], team2_name=t2["name"], team1_ext=1, team2_ext=1))]
            return 1

        if q.startswith("SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC"):
            team_id = int(p[0])
            rows = [pl for pl in self._conn.players if int(pl["team_id"]) == team_id]
            self._rows = [d(dict(r, user_id=10, position=None)) for r in rows]
            return 1

        if q == "SELECT * FROM player_stats WHERE game_id=%s":
            gid = int(p[0])
            rows = [r for r in self._conn.player_stats if int(r["game_id"]) == gid]
            self._rows = [d(r) for r in rows]
            return 1

        if q == "SELECT stats_json, updated_at FROM hky_game_stats WHERE game_id=%s":
            gid = int(p[0])
            row = self._conn.hky_game_stats.get(gid)
            self._rows = [d(row)] if row else []
            return 1

        if q.startswith("SELECT player_id, period, toi_seconds, shifts, gf, ga FROM player_period_stats"):
            gid = int(p[0])
            rows = [r for r in self._conn.player_period_stats if int(r["game_id"]) == gid]
            self._rows = [d(r) for r in rows]
            return 1

        if q == "SELECT events_csv, source_label, updated_at FROM hky_game_events WHERE game_id=%s":
            gid = int(p[0])
            ev = self._conn.hky_game_events.get(gid)
            self._rows = [d(ev)] if ev else []
            return 1

        if q == "SELECT player_stats_csv, source_label, updated_at FROM hky_game_player_stats_csv WHERE game_id=%s":
            gid = int(p[0])
            ps = self._conn.hky_game_player_stats_csv.get(gid)
            self._rows = [d(ps)] if ps else []
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


def should_store_events_via_shift_package_and_render_public_game_page(client_and_db):
    client, db = client_and_db
    events1 = "Period,Time,Team,Event,Player,On-Ice Players\n1,13:45,Blue,Shot,#9 Alice,\"Alice,Bob\"\n"
    assert "\n" in events1
    r = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events1, "source_label": "unit-test", "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert db.hky_game_events[1001]["events_csv"] == events1
    assert "\n" in db.hky_game_events[1001]["events_csv"]

    html = client.get("/public/leagues/1/hky/games/1001").get_data(as_text=True)
    assert "Game Events" in html
    assert "Shot" in html
    assert "#9 Alice" in html
    assert "Event Type" in html
    assert 'data-sortable="1"' in html
    assert 'class="cell-pre"' in html
    assert 'data-freeze-cols="1"' in html
    assert "table-scroll-y" in html


def should_not_overwrite_events_without_replace(client_and_db):
    client, db = client_and_db
    events1 = "Period,Time,Team,Event\n1,00:10,Blue,Goal\n"
    events2 = "Period,Time,Team,Event\n1,00:11,Blue,Goal\n"
    r1 = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events1, "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r1.status_code == 200
    assert db.hky_game_events[1001]["events_csv"] == events1

    r2 = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events2, "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r2.status_code == 200
    assert db.hky_game_events[1001]["events_csv"] == events1

    r3 = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events2, "replace": True},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r3.status_code == 200
    assert db.hky_game_events[1001]["events_csv"] == events2


def should_store_player_stats_csv_via_shift_package_and_render_public_game_page(client_and_db):
    client, db = client_and_db
    player_stats_csv = "Player,Goals,Assists,Average Shift,Shifts,TOI Total\n9 Alice,1,0,0:45,12,12:34\n"
    r = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "player_stats_csv": player_stats_csv, "source_label": "unit-test"},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    # Webapp sanitizes stored CSV to remove shift/ice-time fields.
    assert "Average Shift" not in db.hky_game_player_stats_csv[1001]["player_stats_csv"]
    assert "Shifts" not in db.hky_game_player_stats_csv[1001]["player_stats_csv"]
    assert "TOI Total" not in db.hky_game_player_stats_csv[1001]["player_stats_csv"]

    html = client.get("/public/leagues/1/hky/games/1001").get_data(as_text=True)
    assert "Imported Player Stats" in html
    assert "Average Shift" not in html
    assert "Shifts" not in html
    assert "TOI Total" not in html

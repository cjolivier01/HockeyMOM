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
        self.leagues = {
            1: {"id": 1, "name": "Public League", "is_public": 1, "owner_user_id": 10, "is_shared": 0},
            2: {"id": 2, "name": "Private League", "is_public": 0, "owner_user_id": 10, "is_shared": 0},
        }
        self.teams = {
            101: {"id": 101, "user_id": 10, "name": "Team A", "is_external": 1, "logo_path": None},
            102: {"id": 102, "user_id": 10, "name": "Team B", "is_external": 1, "logo_path": None},
            103: {"id": 103, "user_id": 10, "name": "Team C", "is_external": 1, "logo_path": None},
        }
        self.league_teams = [
            {"league_id": 1, "team_id": 101, "division_name": "10 B West", "division_id": 136, "conference_id": 0},
            {"league_id": 1, "team_id": 102, "division_name": "10 B West", "division_id": 136, "conference_id": 0},
            {"league_id": 1, "team_id": 103, "division_name": "12A", "division_id": 137, "conference_id": 0},
        ]
        self.players = [
            {"id": 501, "user_id": 10, "team_id": 101, "name": "Player 1", "jersey_number": "9", "position": "F"},
        ]
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
                "created_at": "2026-01-01 00:00:00",
                "updated_at": None,
            }
            ,
            1002: {
                "id": 1002,
                "user_id": 10,
                "team1_id": 101,
                "team2_id": 102,
                "starts_at": "2099-01-01 10:00:00",
                "location": "Future Rink",
                "team1_score": None,
                "team2_score": None,
                "is_final": 0,
                "created_at": "2026-01-01 00:00:00",
                "updated_at": None,
            },
            1003: {
                "id": 1003,
                "user_id": 10,
                "team1_id": 101,
                "team2_id": 103,
                "starts_at": "2026-01-03 10:00:00",
                "location": "Other Rink",
                "team1_score": 3,
                "team2_score": 1,
                "is_final": 1,
                "created_at": "2026-01-01 00:00:00",
                "updated_at": None,
            },
        }
        self.league_games = [
            {"league_id": 1, "game_id": 1001, "division_name": "10 B West"},
            {"league_id": 1, "game_id": 1002, "division_name": "10 B West"},
            # Cross-division game (10 B West vs 12A): should be ignored for league views.
            {"league_id": 1, "game_id": 1003, "division_name": "10 B West"},
        ]
        self.player_stats = [{"game_id": 1001, "player_id": 501, "goals": 1, "assists": 0}]

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

        if q == "SELECT id, name FROM leagues WHERE is_public=1 ORDER BY name":
            rows = [l for l in self._conn.leagues.values() if int(l["is_public"]) == 1]
            rows.sort(key=lambda r: str(r["name"]))
            self._rows = [d({"id": r["id"], "name": r["name"]}) for r in rows]
            return 1

        if q == "SELECT id, name FROM leagues WHERE id=%s AND is_public=1":
            lid = int(p[0])
            l = self._conn.leagues.get(lid)
            if l and int(l["is_public"]) == 1:
                self._rows = [d({"id": l["id"], "name": l["name"]})]
            return 1

        if "FROM league_teams lt JOIN teams t ON lt.team_id=t.id" in q and "WHERE lt.league_id=%s" in q:
            league_id = int(p[0])
            rows = []
            for lt in self._conn.league_teams:
                if int(lt["league_id"]) != league_id:
                    continue
                team = self._conn.teams[int(lt["team_id"])]
                merged = dict(team)
                merged.update(
                    {
                        "division_name": lt.get("division_name"),
                        "division_id": lt.get("division_id"),
                        "conference_id": lt.get("conference_id"),
                    }
                )
                rows.append(d(merged))
            self._rows = rows
            return 1

        if q.startswith("SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC"):
            team_id = int(p[0])
            rows = [pl for pl in self._conn.players if int(pl["team_id"]) == team_id]
            self._rows = [d(r) for r in rows]
            return 1

        if q.startswith("SELECT DISTINCT division_name FROM league_teams"):
            league_id = int(p[0])
            divs = sorted({lt["division_name"] for lt in self._conn.league_teams if int(lt["league_id"]) == league_id})
            self._rows = [d({"division_name": x}) for x in divs if x]
            return 1

        if q.startswith("SELECT t.id, t.name FROM league_teams lt JOIN teams t"):
            league_id = int(p[0])
            tids = sorted({int(lt["team_id"]) for lt in self._conn.league_teams if int(lt["league_id"]) == league_id})
            self._rows = [d({"id": tid, "name": self._conn.teams[tid]["name"]}) for tid in tids]
            return 1

        if "FROM league_teams lt JOIN teams t ON lt.team_id=t.id" in q and "WHERE lt.league_id=%s AND t.id=%s" in q:
            league_id, team_id = int(p[0]), int(p[1])
            ok = any(int(lt["league_id"]) == league_id and int(lt["team_id"]) == team_id for lt in self._conn.league_teams)
            if ok:
                self._rows = [d(self._conn.teams[team_id])]
            return 1

        if "FROM league_games lg JOIN hky_games g" in q and "WHERE g.id=%s AND lg.league_id=%s" in q:
            game_id, league_id = int(p[0]), int(p[1])
            ok = any(int(lg["league_id"]) == league_id and int(lg["game_id"]) == game_id for lg in self._conn.league_games)
            if ok:
                g = dict(self._conn.hky_games[game_id])
                t1 = self._conn.teams[int(g["team1_id"])]
                t2 = self._conn.teams[int(g["team2_id"])]
                lt1 = next(
                    (lt for lt in self._conn.league_teams if int(lt["league_id"]) == league_id and int(lt["team_id"]) == int(g["team1_id"])),
                    None,
                )
                lt2 = next(
                    (lt for lt in self._conn.league_teams if int(lt["league_id"]) == league_id and int(lt["team_id"]) == int(g["team2_id"])),
                    None,
                )
                lg_row = next(
                    (lg for lg in self._conn.league_games if int(lg["league_id"]) == league_id and int(lg["game_id"]) == game_id),
                    None,
                )
                self._rows = [
                    d(
                        dict(
                            g,
                            team1_name=t1["name"],
                            team2_name=t2["name"],
                            team1_ext=t1["is_external"],
                            team2_ext=t2["is_external"],
                            division_name=(lg_row or {}).get("division_name"),
                            team1_league_division_name=(lt1 or {}).get("division_name"),
                            team2_league_division_name=(lt2 or {}).get("division_name"),
                        )
                    )
                ]
            return 1

        if "FROM league_games lg" in q and "JOIN hky_games g" in q and "WHERE" in q:
            # Public schedule list
            league_id = int(p[0])
            rows = []
            for lg in self._conn.league_games:
                if int(lg["league_id"]) != league_id:
                    continue
                g = self._conn.hky_games[int(lg["game_id"])]
                team1 = self._conn.teams[int(g["team1_id"])]
                team2 = self._conn.teams[int(g["team2_id"])]
                lt1 = next(
                    (lt for lt in self._conn.league_teams if int(lt["league_id"]) == league_id and int(lt["team_id"]) == int(g["team1_id"])),
                    None,
                )
                lt2 = next(
                    (lt for lt in self._conn.league_teams if int(lt["league_id"]) == league_id and int(lt["team_id"]) == int(g["team2_id"])),
                    None,
                )
                rows.append(
                    d(
                        dict(
                            g,
                            team1_name=team1["name"],
                            team2_name=team2["name"],
                            game_type_name=None,
                            division_name=lg.get("division_name"),
                            team1_league_division_name=(lt1 or {}).get("division_name"),
                            team2_league_division_name=(lt2 or {}).get("division_name"),
                        )
                    )
                )
            self._rows = rows
            return 1

        if q == "SELECT * FROM player_stats WHERE game_id=%s":
            game_id = int(p[0])
            rows = [r for r in self._conn.player_stats if int(r["game_id"]) == game_id]
            self._rows = [d(r) for r in rows]
            return 1

        if q == "SELECT stats_json, updated_at FROM hky_game_stats WHERE game_id=%s":
            self._rows = []
            return 1

        if q.startswith("SELECT player_id, period, toi_seconds, shifts, gf, ga FROM player_period_stats"):
            self._rows = []
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
def client(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    fake_db = FakeConn()
    monkeypatch.setattr(mod, "get_db", lambda: fake_db)
    # Avoid needing to emulate league_games/hky_games for standings in this unit test.
    monkeypatch.setattr(mod, "compute_team_stats_league", lambda *_args, **_kwargs: {"wins": 0, "losses": 0, "ties": 0, "gf": 0, "ga": 0, "points": 0})
    monkeypatch.setattr(mod, "aggregate_players_totals_league", lambda *_args, **_kwargs: {})
    app = mod.create_app()
    app.testing = True
    return app.test_client()


def should_list_public_leagues_without_login(client):
    r = client.get("/public/leagues")
    assert r.status_code == 200
    assert "Public League" in r.get_data(as_text=True)
    assert "Private League" not in r.get_data(as_text=True)


def should_allow_public_league_teams_schedule_and_game_pages_without_login(client):
    r1 = client.get("/public/leagues/1/teams")
    assert r1.status_code == 200
    html = r1.get_data(as_text=True)
    assert "10 B West" in html
    assert "Team A" in html

    r2 = client.get("/public/leagues/1/schedule")
    assert r2.status_code == 200
    html2 = r2.get_data(as_text=True)
    assert "Team A" in html2 and "Team B" in html2

    r3 = client.get("/public/leagues/1/hky/games/1001")
    assert r3.status_code == 200
    assert "Team A" in r3.get_data(as_text=True)


def should_hide_future_unplayed_game_pages_in_public_schedule(client):
    html = client.get("/public/leagues/1/schedule").get_data(as_text=True)
    assert "/public/leagues/1/hky/games/1002" not in html
    assert client.get("/public/leagues/1/hky/games/1002").status_code == 404


def should_hide_cross_division_timetoscore_games_from_public_views(client):
    html = client.get("/public/leagues/1/schedule").get_data(as_text=True)
    assert "/public/leagues/1/hky/games/1003" not in html
    assert client.get("/public/leagues/1/hky/games/1003").status_code == 404

def should_reject_private_league_public_routes(client):
    r = client.get("/public/leagues/2/teams")
    assert r.status_code == 404

import importlib.util
import os
import sys
from typing import Any, Optional
from pathlib import Path

import pytest


def _load_app_module():
    os.environ.setdefault("HM_WEBAPP_SKIP_DB_INIT", "1")
    os.environ.setdefault("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
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
        self._next_id = {
            "users": 1,
            "leagues": 1,
            "teams": 1,
            "players": 1,
            "hky_games": 1,
        }
        self.users_by_id: dict[int, dict[str, Any]] = {}
        self.user_id_by_email: dict[str, int] = {}
        self.leagues_by_id: dict[int, dict[str, Any]] = {}
        self.league_id_by_name: dict[str, int] = {}
        self.league_members: dict[tuple[int, int], dict[str, Any]] = {}
        self.teams_by_id: dict[int, dict[str, Any]] = {}
        self.team_id_by_user_name: dict[tuple[int, str], int] = {}
        self.players_by_id: dict[int, dict[str, Any]] = {}
        self.player_id_by_user_team_name: dict[tuple[int, int, str], int] = {}
        self.hky_games_by_id: dict[int, dict[str, Any]] = {}
        self.league_teams: set[tuple[int, int]] = set()
        self.league_teams_meta: dict[tuple[int, int], dict[str, Any]] = {}
        self.league_games: set[tuple[int, int]] = set()
        self.league_games_meta: dict[tuple[int, int], dict[str, Any]] = {}
        self.player_stats: dict[tuple[int, int], dict[str, Any]] = {}

    def _alloc_id(self, table: str) -> int:
        nid = self._next_id[table]
        self._next_id[table] += 1
        return nid

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
        self.lastrowid: int = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query: str, params: Any = None) -> int:
        q = " ".join(str(query).split())
        p = tuple(params or ())
        self._rows = []
        self._idx = 0

        def as_row_dict(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row)

        def as_row_tuple(*vals: Any) -> tuple[Any, ...]:
            return tuple(vals)

        # Users
        if q == "SELECT id FROM users WHERE email=%s":
            email = str(p[0]).strip().lower()
            uid = self._conn.user_id_by_email.get(email)
            if uid is not None:
                self._rows = [as_row_tuple(uid)]
            return 1

        if q.startswith("INSERT INTO users(email, password_hash, name, created_at) VALUES"):
            email, password_hash, name, created_at = p
            email = str(email).strip().lower()
            if email in self._conn.user_id_by_email:
                self.lastrowid = self._conn.user_id_by_email[email]
                return 1
            uid = self._conn._alloc_id("users")
            self._conn.user_id_by_email[email] = uid
            self._conn.users_by_id[uid] = {
                "id": uid,
                "email": email,
                "password_hash": password_hash,
                "name": name,
                "created_at": created_at,
                "default_league_id": None,
            }
            self.lastrowid = uid
            return 1

        # Leagues
        if q == "SELECT id, is_shared FROM leagues WHERE name=%s":
            name = str(p[0]).strip()
            lid = self._conn.league_id_by_name.get(name)
            if lid is not None:
                self._rows = [as_row_tuple(lid, int(self._conn.leagues_by_id[lid]["is_shared"]))]
            return 1

        if q == "UPDATE leagues SET is_shared=%s, updated_at=%s WHERE id=%s":
            is_shared, updated_at, lid = p
            lid = int(lid)
            self._conn.leagues_by_id[lid]["is_shared"] = int(is_shared)
            self._conn.leagues_by_id[lid]["updated_at"] = updated_at
            return 1

        if q.startswith(
            "INSERT INTO leagues(name, owner_user_id, is_shared, source, external_key, created_at) VALUES"
        ):
            name, owner_user_id, is_shared, source, external_key, created_at = p
            name = str(name).strip()
            if name in self._conn.league_id_by_name:
                self.lastrowid = self._conn.league_id_by_name[name]
                return 1
            lid = self._conn._alloc_id("leagues")
            self._conn.league_id_by_name[name] = lid
            self._conn.leagues_by_id[lid] = {
                "id": lid,
                "name": name,
                "owner_user_id": int(owner_user_id),
                "is_shared": int(is_shared),
                "source": source,
                "external_key": external_key,
                "created_at": created_at,
                "updated_at": None,
            }
            self.lastrowid = lid
            return 1

        # League members / mappings
        if q.startswith("INSERT IGNORE INTO league_members(league_id, user_id, role, created_at) VALUES"):
            league_id, user_id, role, created_at = p
            key = (int(league_id), int(user_id))
            if key not in self._conn.league_members:
                self._conn.league_members[key] = {
                    "league_id": int(league_id),
                    "user_id": int(user_id),
                    "role": str(role),
                    "created_at": created_at,
                }
            return 1

        if q == "INSERT IGNORE INTO league_teams(league_id, team_id) VALUES(%s,%s)":
            league_id, team_id = p
            key = (int(league_id), int(team_id))
            self._conn.league_teams.add(key)
            self._conn.league_teams_meta.setdefault(
                key, {"division_name": None, "division_id": None, "conference_id": None}
            )
            return 1

        if q.startswith(
            "INSERT INTO league_teams(league_id, team_id, division_name, division_id, conference_id) VALUES"
        ):
            league_id, team_id, division_name, division_id, conference_id = p
            key = (int(league_id), int(team_id))
            self._conn.league_teams.add(key)
            prev = self._conn.league_teams_meta.get(key) or {
                "division_name": None,
                "division_id": None,
                "conference_id": None,
            }
            if division_name is not None and str(division_name).strip():
                prev["division_name"] = str(division_name).strip()
            if division_id is not None:
                prev["division_id"] = int(division_id)
            if conference_id is not None:
                prev["conference_id"] = int(conference_id)
            self._conn.league_teams_meta[key] = prev
            return 1

        if q == "INSERT IGNORE INTO league_games(league_id, game_id) VALUES(%s,%s)":
            league_id, game_id = p
            key = (int(league_id), int(game_id))
            self._conn.league_games.add(key)
            self._conn.league_games_meta.setdefault(
                key, {"division_name": None, "division_id": None, "conference_id": None, "sort_order": None}
            )
            return 1

        if q.startswith(
            "INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id, sort_order) VALUES"
        ):
            league_id, game_id, division_name, division_id, conference_id, sort_order = p
            key = (int(league_id), int(game_id))
            self._conn.league_games.add(key)
            prev = self._conn.league_games_meta.get(key) or {
                "division_name": None,
                "division_id": None,
                "conference_id": None,
                "sort_order": None,
            }
            if division_name is not None and str(division_name).strip():
                prev["division_name"] = str(division_name).strip()
            if division_id is not None:
                prev["division_id"] = int(division_id)
            if conference_id is not None:
                prev["conference_id"] = int(conference_id)
            if sort_order is not None:
                prev["sort_order"] = int(sort_order)
            self._conn.league_games_meta[key] = prev
            return 1

        # Team logos (REST import)
        if q == "SELECT logo_path FROM teams WHERE id=%s":
            team_id = int(p[0])
            team = self._conn.teams_by_id.get(team_id)
            if team:
                self._rows = [as_row_dict({"logo_path": team.get("logo_path")})] if self._dict_mode else [as_row_tuple(team.get("logo_path"))]
            return 1

        if q == "UPDATE teams SET logo_path=%s, updated_at=%s WHERE id=%s":
            logo_path, updated_at, team_id = p
            team_id = int(team_id)
            if team_id in self._conn.teams_by_id:
                self._conn.teams_by_id[team_id]["logo_path"] = str(logo_path)
                self._conn.teams_by_id[team_id]["updated_at"] = updated_at
            return 1

        # Teams
        if q == "SELECT id FROM teams WHERE user_id=%s AND name=%s":
            user_id, name = p
            key = (int(user_id), str(name).strip())
            tid = self._conn.team_id_by_user_name.get(key)
            if tid is not None:
                self._rows = [as_row_tuple(tid)]
            return 1

        if q == "SELECT id, name FROM teams WHERE user_id=%s":
            user_id = int(p[0])
            rows = []
            for tid, tr in self._conn.teams_by_id.items():
                if int(tr.get("user_id") or 0) != user_id:
                    continue
                rows.append(as_row_tuple(int(tid), str(tr.get("name") or "")))
            self._rows = rows
            return 1

        if q.startswith("INSERT INTO teams(user_id, name, is_external, created_at) VALUES"):
            user_id, name, is_external, created_at = p
            key = (int(user_id), str(name).strip())
            if key in self._conn.team_id_by_user_name:
                self.lastrowid = self._conn.team_id_by_user_name[key]
                return 1
            tid = self._conn._alloc_id("teams")
            self._conn.team_id_by_user_name[key] = tid
            self._conn.teams_by_id[tid] = {
                "id": tid,
                "user_id": int(user_id),
                "name": str(name).strip(),
                "is_external": int(is_external),
                "logo_path": None,
                "created_at": created_at,
                "updated_at": None,
            }
            self.lastrowid = tid
            return 1

        # Players
        if q == "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s":
            user_id, team_id, name = p
            key = (int(user_id), int(team_id), str(name).strip())
            pid = self._conn.player_id_by_user_team_name.get(key)
            if pid is not None:
                self._rows = [as_row_tuple(pid)]
            return 1

        if q.startswith(
            "UPDATE players SET jersey_number=COALESCE(%s, jersey_number), position=COALESCE(%s, position), updated_at=%s WHERE id=%s"
        ):
            jersey_number, position, updated_at, pid = p
            pid = int(pid)
            rec = self._conn.players_by_id[pid]
            if jersey_number is not None:
                rec["jersey_number"] = jersey_number
            if position is not None:
                rec["position"] = position
            rec["updated_at"] = updated_at
            return 1

        if q.startswith("INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES"):
            user_id, team_id, name, jersey_number, position, created_at = p
            key = (int(user_id), int(team_id), str(name).strip())
            if key in self._conn.player_id_by_user_team_name:
                self.lastrowid = self._conn.player_id_by_user_team_name[key]
                return 1
            pid = self._conn._alloc_id("players")
            self._conn.player_id_by_user_team_name[key] = pid
            self._conn.players_by_id[pid] = {
                "id": pid,
                "user_id": int(user_id),
                "team_id": int(team_id),
                "name": str(name).strip(),
                "jersey_number": jersey_number,
                "position": position,
                "created_at": created_at,
                "updated_at": None,
            }
            self.lastrowid = pid
            return 1

        # Games
        if q.startswith(
            "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND team1_id=%s AND team2_id=%s AND starts_at=%s"
        ):
            user_id, team1_id, team2_id, starts_at = p
            for g in self._conn.hky_games_by_id.values():
                if (
                    int(g["user_id"]) == int(user_id)
                    and int(g["team1_id"]) == int(team1_id)
                    and int(g["team2_id"]) == int(team2_id)
                    and g.get("starts_at") == starts_at
                ):
                    row = {
                        "id": g["id"],
                        "notes": g.get("notes"),
                        "team1_score": g.get("team1_score"),
                        "team2_score": g.get("team2_score"),
                    }
                    self._rows = [as_row_dict(row)]
                    break
            return 1

        if q.startswith(
            "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND notes LIKE %s"
        ):
            user_id, like_pat = p
            token = str(like_pat).strip("%")
            for g in self._conn.hky_games_by_id.values():
                if int(g["user_id"]) == int(user_id) and token in str(g.get("notes") or ""):
                    row = {
                        "id": g["id"],
                        "notes": g.get("notes"),
                        "team1_score": g.get("team1_score"),
                        "team2_score": g.get("team2_score"),
                    }
                    self._rows = [as_row_dict(row)]
                    break
            return 1

        if q.startswith(
            "INSERT INTO hky_games(user_id, team1_id, team2_id, starts_at, location, team1_score, team2_score, is_final, notes, stats_imported_at, created_at) VALUES"
        ):
            (
                user_id,
                team1_id,
                team2_id,
                starts_at,
                location,
                team1_score,
                team2_score,
                is_final,
                notes,
                stats_imported_at,
                created_at,
            ) = p
            gid = self._conn._alloc_id("hky_games")
            self._conn.hky_games_by_id[gid] = {
                "id": gid,
                "user_id": int(user_id),
                "team1_id": int(team1_id),
                "team2_id": int(team2_id),
                "starts_at": starts_at,
                "location": location,
                "team1_score": team1_score,
                "team2_score": team2_score,
                "is_final": int(is_final),
                "notes": notes,
                "stats_imported_at": stats_imported_at,
                "created_at": created_at,
                "updated_at": None,
            }
            self.lastrowid = gid
            return 1

        if q == "SELECT notes, team1_score, team2_score FROM hky_games WHERE id=%s":
            gid = int(p[0])
            g = self._conn.hky_games_by_id.get(gid)
            if g:
                self._rows = [
                    as_row_dict(
                        {"notes": g.get("notes"), "team1_score": g.get("team1_score"), "team2_score": g.get("team2_score")}
                    )
                ]
            return 1

        if q.startswith("UPDATE hky_games SET location=COALESCE(%s, location), team1_score=%s, team2_score=%s"):
            location, t1, t2, _t1, _t2, notes, stats_imported_at, updated_at, gid = p
            gid = int(gid)
            g = self._conn.hky_games_by_id[gid]
            if location is not None and not g.get("location"):
                g["location"] = location
            g["team1_score"] = t1
            g["team2_score"] = t2
            if t1 is not None and t2 is not None:
                g["is_final"] = 1
            g["notes"] = notes
            g["stats_imported_at"] = stats_imported_at
            g["updated_at"] = updated_at
            return 1

        if q.startswith(
            "UPDATE hky_games SET location=COALESCE(%s, location), team1_score=COALESCE(team1_score, %s), team2_score=COALESCE(team2_score, %s)"
        ):
            location, t1, t2, _t1, _t2, notes, stats_imported_at, updated_at, gid = p
            gid = int(gid)
            g = self._conn.hky_games_by_id[gid]
            if location is not None and not g.get("location"):
                g["location"] = location
            if g.get("team1_score") is None:
                g["team1_score"] = t1
            if g.get("team2_score") is None:
                g["team2_score"] = t2
            if g.get("team1_score") is not None and g.get("team2_score") is not None:
                g["is_final"] = 1
            g["notes"] = notes
            g["stats_imported_at"] = stats_imported_at
            g["updated_at"] = updated_at
            return 1

        # Player stats
        if q.startswith("INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists) VALUES"):
            user_id, team_id, game_id, player_id, goals, assists = p
            key = (int(game_id), int(player_id))
            is_replace = "goals=VALUES(goals)" in q
            if key not in self._conn.player_stats:
                self._conn.player_stats[key] = {
                    "user_id": int(user_id),
                    "team_id": int(team_id),
                    "game_id": int(game_id),
                    "player_id": int(player_id),
                    "goals": goals,
                    "assists": assists,
                }
                return 1
            rec = self._conn.player_stats[key]
            if is_replace:
                rec["goals"] = goals
                rec["assists"] = assists
            else:
                if rec.get("goals") is None:
                    rec["goals"] = goals
                if rec.get("assists") is None:
                    rec["assists"] = assists
            return 1

        # Shared league visibility queries (used by /leagues and context processor)
        if "FROM leagues l WHERE l.is_shared=1 OR l.owner_user_id=%s" in q:
            user_id = int(p[-2]) if len(p) >= 2 else 0
            rows = []
            for l in self._conn.leagues_by_id.values():
                if int(l["is_shared"]) == 1 or int(l["owner_user_id"]) == user_id:
                    rows.append(
                        {
                            "id": l["id"],
                            "name": l["name"],
                            "is_shared": int(l["is_shared"]),
                            "is_admin": 1 if int(l["owner_user_id"]) == user_id else 0,
                            "is_owner": 1 if int(l["owner_user_id"]) == user_id else 0,
                        }
                    )
            rows.sort(key=lambda r: str(r["name"]))
            self._rows = [as_row_dict(r) for r in rows]
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


def _post(client, path: str, payload: dict[str, Any], *, token: Optional[str] = None, environ: Optional[dict] = None):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(path, json=payload, headers=headers, environ_base=environ or {})


def should_require_token_when_configured(client_and_db, monkeypatch):
    client, _db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    r = _post(client, "/api/import/hockey/ensure_league", {"league_name": "Norcal"})
    assert r.status_code == 401
    r2 = _post(client, "/api/import/hockey/ensure_league", {"league_name": "Norcal"}, token="wrong")
    assert r2.status_code == 401
    r3 = _post(client, "/api/import/hockey/ensure_league", {"league_name": "Norcal"}, token="sekret")
    assert r3.status_code == 200
    assert r3.get_json()["ok"] is True


def should_deny_remote_import_without_token(client_and_db, monkeypatch):
    client, _db = client_and_db
    monkeypatch.delenv("HM_WEBAPP_IMPORT_TOKEN", raising=False)
    r = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal"},
        environ={"REMOTE_ADDR": "10.0.0.1"},
    )
    assert r.status_code == 403
    assert r.get_json()["error"] == "import_token_required"
    r2 = client.post(
        "/api/import/hockey/ensure_league",
        json={"league_name": "Norcal"},
        headers={"X-Forwarded-For": "1.2.3.4"},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert r2.status_code == 403


def should_create_and_update_shared_league(client_and_db, monkeypatch):
    client, db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    r = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    data = r.get_json()
    assert data["ok"] is True
    lid = int(data["league_id"])
    assert db.leagues_by_id[lid]["is_shared"] == 1
    assert (lid, int(data["owner_user_id"])) in db.league_members

    r2 = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": False, "owner_email": "owner@example.com"},
        token="sekret",
    )
    assert int(r2.get_json()["league_id"]) == lid
    assert db.leagues_by_id[lid]["is_shared"] == 0


def should_import_game_and_be_non_destructive_without_replace(client_and_db, monkeypatch):
    client, db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )

    payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "game": {
            "home_name": "Home",
            "away_name": "Away",
            "starts_at": "2026-01-01 10:00:00",
            "location": "Rink 1",
            "home_score": 1,
            "away_score": 2,
            "timetoscore_game_id": 123,
            "season_id": 77,
            "home_roster": [{"name": "Alice", "number": "9", "position": "F"}],
            "away_roster": [{"name": "Bob", "number": "", "position": "D"}],
            "player_stats": [{"name": "Alice", "goals": 1, "assists": 0}],
        },
        "source": "timetoscore",
        "external_key": "sharksice:77",
    }

    r = _post(client, "/api/import/hockey/game", payload, token="sekret")
    assert r.status_code == 200
    out = r.get_json()
    assert out["ok"] is True

    gid = int(out["game_id"])
    g = db.hky_games_by_id[gid]
    assert g["team1_score"] == 1 and g["team2_score"] == 2
    assert "timetoscore_game_id" in (g["notes"] or "")
    assert "timetoscore_season_id" in (g["notes"] or "")

    lid = db.league_id_by_name["Norcal"]
    assert (lid, int(out["team1_id"])) in db.league_teams
    assert (lid, int(out["team2_id"])) in db.league_teams
    assert (lid, gid) in db.league_games

    # Re-import with different scores/stats without replace should not overwrite existing scores/stats
    payload2 = dict(payload)
    payload2["game"] = dict(payload["game"])
    payload2["game"]["home_score"] = 9
    payload2["game"]["away_score"] = 9
    payload2["game"]["player_stats"] = [{"name": "Alice", "goals": 7, "assists": 7}]
    r2 = _post(client, "/api/import/hockey/game", payload2, token="sekret")
    assert r2.status_code == 200
    gid2 = int(r2.get_json()["game_id"])
    assert gid2 == gid
    g2 = db.hky_games_by_id[gid2]
    assert g2["team1_score"] == 1 and g2["team2_score"] == 2

    # Player stats remain original (non-null) without replace
    alice_pid = db.player_id_by_user_team_name[(out["owner_user_id"], out["team1_id"], "Alice")]
    ps = db.player_stats[(gid, alice_pid)]
    assert ps["goals"] == 1 and ps["assists"] == 0


def should_persist_division_metadata_on_import(client_and_db, monkeypatch):
    client, db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    r = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    assert r.status_code == 200
    lid = int(r.get_json()["league_id"])

    payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "game": {
            "home_name": "Alpha",
            "away_name": "Beta",
            "starts_at": "2026-01-02 10:00:00",
            "location": "Rink",
            "home_score": 1,
            "away_score": 2,
            "division_name": "10U A",
            "division_id": 55,
            "conference_id": 7,
            "home_roster": [{"name": "P1"}],
            "away_roster": [{"name": "P2"}],
            "player_stats": [{"name": "P1", "goals": 1, "assists": 0}],
        },
        "source": "timetoscore",
        "external_key": "caha:0",
    }
    r2 = _post(client, "/api/import/hockey/game", payload, token="sekret")
    assert r2.status_code == 200
    gid = int(r2.get_json()["game_id"])

    team1_id = int(r2.get_json()["team1_id"])
    team2_id = int(r2.get_json()["team2_id"])
    assert db.league_teams_meta[(lid, team1_id)]["division_name"] == "10U A"
    assert db.league_teams_meta[(lid, team2_id)]["division_id"] == 55
    assert db.league_games_meta[(lid, gid)]["conference_id"] == 7

    # Re-import without division fields should not erase existing metadata.
    payload2 = dict(payload)
    payload2["game"] = dict(payload["game"])
    payload2["game"].pop("division_name", None)
    payload2["game"].pop("division_id", None)
    payload2["game"].pop("conference_id", None)
    r3 = _post(client, "/api/import/hockey/game", payload2, token="sekret")
    assert r3.status_code == 200
    assert db.league_teams_meta[(lid, team1_id)]["division_name"] == "10U A"


def should_overwrite_with_replace(client_and_db, monkeypatch):
    client, db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": True,
        "owner_email": "owner@example.com",
        "game": {
            "home_name": "Home",
            "away_name": "Away",
            "starts_at": "2026-01-01 10:00:00",
            "location": "Rink 1",
            "home_score": 3,
            "away_score": 4,
            "timetoscore_game_id": 123,
            "season_id": 77,
            "home_roster": [{"name": "Alice", "number": None, "position": None}],
            "away_roster": [],
            "player_stats": [{"name": "Alice", "goals": 2, "assists": 1}],
        },
    }
    r = _post(client, "/api/import/hockey/game", payload, token="sekret")
    assert r.status_code == 200
    out = r.get_json()
    gid = int(out["game_id"])
    assert db.hky_games_by_id[gid]["team1_score"] == 3
    alice_pid = db.player_id_by_user_team_name[(out["owner_user_id"], out["team1_id"], "Alice")]
    ps = db.player_stats[(gid, alice_pid)]
    assert ps["goals"] == 2 and ps["assists"] == 1


def should_match_existing_game_by_timetoscore_id_when_no_starts_at(client_and_db, monkeypatch):
    client, db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    # Pre-create owner/user/league
    r = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    owner_user_id = int(r.get_json()["owner_user_id"])

    # Pre-create teams + game with notes containing timetoscore_game_id
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (owner_user_id, "Home", 1, "t"),
        )
        home_tid = int(cur.lastrowid)
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (owner_user_id, "Away", 1, "t"),
        )
        away_tid = int(cur.lastrowid)
    with db.cursor(_DummyPyMySQL.cursors.DictCursor) as curd:
        curd.execute(
            "INSERT INTO hky_games(user_id, team1_id, team2_id, starts_at, location, team1_score, team2_score, is_final, notes, stats_imported_at, created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                owner_user_id,
                home_tid,
                away_tid,
                None,
                None,
                None,
                None,
                0,
                '{"timetoscore_game_id":999}',
                "t",
                "t",
            ),
        )
        existing_gid = int(curd.lastrowid)

    payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "game": {
            "home_name": "Home",
            "away_name": "Away",
            "starts_at": None,
            "location": "Rink X",
            "home_score": 1,
            "away_score": 0,
            "timetoscore_game_id": 999,
            "season_id": 77,
            "home_roster": [{"name": "Skater", "number": "", "position": ""}],
            "away_roster": [],
            "player_stats": [],
        },
    }
    r2 = _post(client, "/api/import/hockey/game", payload, token="sekret")
    assert r2.status_code == 200
    gid2 = int(r2.get_json()["game_id"])
    assert gid2 == existing_gid
    assert db.hky_games_by_id[gid2]["team1_score"] == 1
    assert db.hky_games_by_id[gid2]["team2_score"] == 0


def should_update_player_roster_fields_without_replace(client_and_db, monkeypatch):
    client, db = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "game": {
            "home_name": "Home",
            "away_name": "Away",
            "starts_at": "2026-02-01 10:00:00",
            "location": None,
            "home_score": None,
            "away_score": None,
            "timetoscore_game_id": 555,
            "season_id": 77,
            "home_roster": [{"name": "Alice", "number": None, "position": None}],
            "away_roster": [],
            "player_stats": [],
        },
    }
    r1 = _post(client, "/api/import/hockey/game", payload, token="sekret")
    out = r1.get_json()
    alice_pid = db.player_id_by_user_team_name[(out["owner_user_id"], out["team1_id"], "Alice")]
    assert db.players_by_id[alice_pid]["jersey_number"] is None

    payload2 = dict(payload)
    payload2["game"] = dict(payload["game"])
    payload2["game"]["home_roster"] = [{"name": "Alice", "number": "12", "position": "F"}]
    _post(client, "/api/import/hockey/game", payload2, token="sekret")
    assert db.players_by_id[alice_pid]["jersey_number"] == "12"
    assert db.players_by_id[alice_pid]["position"] == "F"


def should_import_games_batch_and_match_individual_imports(monkeypatch):
    import copy

    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)

    current_db = [FakeConn()]
    monkeypatch.setattr(mod, "get_db", lambda: current_db[0])

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    game1 = {
        "home_name": "Home A",
        "away_name": "Away A",
        "starts_at": "2026-01-01 10:00:00",
        "location": "Rink 1",
        "home_score": 1,
        "away_score": 2,
        "timetoscore_game_id": 1001,
        "season_id": 77,
        "home_roster": [{"name": "Alice", "number": "9", "position": "F"}],
        "away_roster": [{"name": "Bob", "number": "4", "position": "D"}],
        "player_stats": [{"name": "Alice", "goals": 1, "assists": 0}],
        "home_division_name": "10 B West",
        "away_division_name": "10 B West",
    }
    game2 = {
        "home_name": "Home B",
        "away_name": "Away B",
        "starts_at": "2026-01-02 11:00:00",
        "location": "Rink 2",
        "home_score": 3,
        "away_score": 4,
        "timetoscore_game_id": 1002,
        "season_id": 77,
        "home_roster": [{"name": "Carol", "number": "12", "position": "F"}],
        "away_roster": [{"name": "Dan", "number": "2", "position": "D"}],
        "player_stats": [{"name": "Carol", "goals": 2, "assists": 1}],
        "home_division_name": "12U A",
        "away_division_name": "12U A",
    }

    # Reference: individual imports
    payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
    }
    _post(client, "/api/import/hockey/game", dict(payload, game=game1), token="sekret")
    _post(client, "/api/import/hockey/game", dict(payload, game=game2), token="sekret")
    db_individual = current_db[0]
    def _strip_time_fields(obj):  # noqa: ANN001
        if isinstance(obj, dict):
            return {
                k: _strip_time_fields(v)
                for k, v in obj.items()
                if k not in ("created_at", "updated_at", "stats_imported_at")
            }
        if isinstance(obj, list):
            return [_strip_time_fields(x) for x in obj]
        return obj

    snap_individual = _strip_time_fields(
        copy.deepcopy(
            {
                "leagues_by_id": db_individual.leagues_by_id,
                "league_members": db_individual.league_members,
                "teams_by_id": db_individual.teams_by_id,
                "players_by_id": db_individual.players_by_id,
                "hky_games_by_id": db_individual.hky_games_by_id,
                "league_teams_meta": db_individual.league_teams_meta,
                "league_games_meta": db_individual.league_games_meta,
                "player_stats": db_individual.player_stats,
            }
        )
    )

    # Batch imports into a fresh DB should yield identical final state.
    current_db[0] = FakeConn()
    batch_payload = dict(payload)
    batch_payload["games"] = [game1, game2]
    r = _post(client, "/api/import/hockey/games_batch", batch_payload, token="sekret")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert int(r.get_json()["imported"]) == 2

    db_batch = current_db[0]
    snap_batch = _strip_time_fields(
        copy.deepcopy(
            {
                "leagues_by_id": db_batch.leagues_by_id,
                "league_members": db_batch.league_members,
                "teams_by_id": db_batch.teams_by_id,
                "players_by_id": db_batch.players_by_id,
                "hky_games_by_id": db_batch.hky_games_by_id,
                "league_teams_meta": db_batch.league_teams_meta,
                "league_games_meta": db_batch.league_games_meta,
                "player_stats": db_batch.player_stats,
            }
        )
    )
    assert snap_batch == snap_individual


def should_import_team_logos_from_urls_in_games_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    monkeypatch.setattr(mod, "INSTANCE_DIR", tmp_path, raising=False)

    # Dummy requests module to ensure headers are passed and to provide bytes.
    class _Resp:
        def __init__(self, content: bytes, headers: dict[str, str]) -> None:
            self.content = content
            self.headers = headers

        def raise_for_status(self) -> None:
            return None

    got: dict[str, Any] = {}

    class _Requests:
        @staticmethod
        def get(url: str, timeout=None, headers=None):  # noqa: ANN001
            got["url"] = url
            got["timeout"] = timeout
            got["headers"] = dict(headers or {})
            return _Resp(b"PNGDATA", {"Content-Type": "image/png"})

    monkeypatch.setitem(sys.modules, "requests", _Requests)

    current_db = [FakeConn()]
    monkeypatch.setattr(mod, "get_db", lambda: current_db[0])

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    batch_payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
        "games": [
            {
                "home_name": "Home A",
                "away_name": "Away A",
                "starts_at": "2026-01-01 10:00:00",
                "location": "Rink 1",
                "home_score": 1,
                "away_score": 2,
                "timetoscore_game_id": 1001,
                "season_id": 77,
                "home_logo_url": "https://stats.caha.timetoscore.com/logo.png",
                "away_logo_url": "https://stats.caha.timetoscore.com/logo2.png",
                "home_roster": [],
                "away_roster": [],
                "player_stats": [],
                "home_division_name": "10 B West",
                "away_division_name": "10 B West",
            }
        ],
    }
    r = _post(client, "/api/import/hockey/games_batch", batch_payload, token="sekret")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    # REST importer should set a reasonable User-Agent for hotlink-protected hosts.
    assert got["headers"].get("User-Agent") == "Mozilla/5.0"
    assert "Referer" in got["headers"]

    db = current_db[0]
    # Teams were created; their logo_path should be set and file should exist.
    assert len(db.teams_by_id) == 2
    for team in db.teams_by_id.values():
        logo_path = str(team.get("logo_path") or "")
        assert logo_path
        assert tmp_path in Path(logo_path).resolve().parents
        assert Path(logo_path).exists()
        assert Path(logo_path).read_bytes() == b"PNGDATA"


def should_import_team_logos_from_b64_in_games_batch_without_requests(tmp_path, monkeypatch):
    import base64

    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    monkeypatch.setattr(mod, "INSTANCE_DIR", tmp_path, raising=False)

    # Ensure server doesn't need `requests` for this path.
    monkeypatch.setitem(sys.modules, "requests", None)

    current_db = [FakeConn()]
    monkeypatch.setattr(mod, "get_db", lambda: current_db[0])

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    png_bytes = b"\x89PNG\r\n\x1a\nFAKE"
    b64 = base64.b64encode(png_bytes).decode("ascii")

    batch_payload = {
        "league_name": "Norcal",
        "shared": True,
        "replace": True,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
        "games": [
            {
                "home_name": "Home A",
                "away_name": "Away A",
                "starts_at": "2026-01-01 10:00:00",
                "location": "Rink 1",
                "home_score": 1,
                "away_score": 2,
                "timetoscore_game_id": 1001,
                "season_id": 77,
                "home_logo_b64": b64,
                "home_logo_content_type": "image/png",
                "away_logo_b64": b64,
                "away_logo_content_type": "image/png",
                "home_roster": [],
                "away_roster": [],
                "player_stats": [],
                "home_division_name": "10 B West",
                "away_division_name": "10 B West",
            }
        ],
    }
    r = _post(client, "/api/import/hockey/games_batch", batch_payload, token="sekret")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    db = current_db[0]
    assert len(db.teams_by_id) == 2
    for team in db.teams_by_id.values():
        logo_path = str(team.get("logo_path") or "")
        assert logo_path
        p = Path(logo_path)
        assert p.exists()
        assert p.read_bytes() == png_bytes


def should_filter_single_game_player_stats_csv_drops_per_game_columns():
    mod = _load_app_module()
    headers = [
        "Jersey",
        "Player",
        "GP",
        "Goals",
        "Shots per Game",
        "TOI per Game",
        "PPG",
        "Shots per Shift",
        "Shifts",
        "TOI Total",
        "Average Shift",
    ]
    rows = [
        {
            "Jersey": "12",
            "Player": "Alice",
            "GP": "1",
            "Goals": "1",
            "Shots per Game": "2.0",
            "TOI per Game": "10:00",
            "PPG": "1.0",
            "Shots per Shift": "0.10",
            "Shifts": "12",
            "TOI Total": "12:34",
            "Average Shift": "0:45",
        }
    ]
    kept_headers, kept_rows = mod.filter_single_game_player_stats_csv(headers, rows)
    assert kept_headers == ["Jersey", "Player", "Goals"]
    assert kept_rows == [{"Jersey": "12", "Player": "Alice", "Goals": "1"}]


def should_normalize_game_events_csv_moves_event_type_first_and_drops_raw():
    mod = _load_app_module()
    headers = ["Event ID", "Source", "Event Type", "Event Type Raw", "Team Rel"]
    rows = [{"Event ID": "1", "Source": "long", "Event Type": "Shot", "Event Type Raw": "Shot", "Team Rel": "For"}]
    out_headers, out_rows = mod.normalize_game_events_csv(headers, rows)
    assert out_headers[0] == "Event Type"
    assert "Event Type Raw" not in out_headers
    assert out_rows == [{"Event Type": "Shot", "Event ID": "1", "Source": "long", "Team Rel": "For"}]


def should_normalize_game_events_csv_renames_event_to_event_type():
    mod = _load_app_module()
    headers = ["Period", "Time", "Team", "Event", "Player"]
    rows = [{"Period": "1", "Time": "13:45", "Team": "Blue", "Event": "Shot", "Player": "#9"}]
    out_headers, out_rows = mod.normalize_game_events_csv(headers, rows)
    assert out_headers[0] == "Event Type"
    assert "Event" not in out_headers
    assert out_rows == [{"Event Type": "Shot", "Period": "1", "Time": "13:45", "Team": "Blue", "Player": "#9"}]

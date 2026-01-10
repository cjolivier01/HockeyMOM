from __future__ import annotations

import datetime as dt
import importlib.util
import json

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
        self._next_id = {"users": 11, "leagues": 2, "teams": 103, "players": 503, "hky_games": 1002}
        self.users_by_email: dict[str, dict[str, Any]] = {}
        self.leagues = {1: {"id": 1, "name": "Public League", "is_public": 1, "owner_user_id": 10, "is_shared": 0}}
        self.league_id_by_name = {str(v["name"]): int(k) for k, v in self.leagues.items()}
        self.teams = {
            101: {"id": 101, "user_id": 10, "name": "Team A", "is_external": 1, "logo_path": None},
            102: {"id": 102, "user_id": 10, "name": "Team B", "is_external": 1, "logo_path": None},
        }
        self.team_id_by_user_name = {
            (int(v["user_id"]), str(v["name"])): int(k) for k, v in self.teams.items()
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
        self.league_games = [{"league_id": 1, "game_id": 1001, "division_name": "10 A", "sort_order": None}]
        self.league_teams: list[dict[str, Any]] = []
        self.players = [
            {"id": 501, "team_id": 101, "name": "Alice", "jersey_number": "9"},
            {"id": 502, "team_id": 102, "name": "Bob", "jersey_number": "12"},
        ]
        self.player_id_by_user_team_name: dict[tuple[int, int, str], int] = {}
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

        def d(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row)

        def t(*vals: Any) -> tuple[Any, ...]:
            return tuple(vals)

        if q.startswith("SELECT id, name") and "FROM leagues WHERE id=%s AND is_public=1" in q:
            lid = int(p[0])
            league = self._conn.leagues.get(lid)
            if league and int(league.get("is_public") or 0) == 1:
                row = {"id": league["id"], "name": league["name"]}
                if "owner_user_id" in q:
                    row["owner_user_id"] = league.get("owner_user_id")
                self._rows = [d(row)]
            return 1

        if q == "SELECT owner_user_id FROM leagues WHERE id=%s":
            lid = int(p[0])
            league = self._conn.leagues.get(lid)
            if league:
                self._rows = [t(int(league.get("owner_user_id") or 0))]
            return 1

        if q == "SELECT 1 FROM leagues WHERE id=%s AND owner_user_id=%s":
            lid, uid = int(p[0]), int(p[1])
            league = self._conn.leagues.get(lid)
            if league and int(league.get("owner_user_id") or 0) == uid:
                self._rows = [t(1)]
            return 1

        if q.startswith(
            "SELECT 1 FROM leagues l LEFT JOIN league_members m ON m.league_id=l.id AND m.user_id=%s WHERE l.id=%s AND (l.is_shared=1 OR l.owner_user_id=%s OR m.user_id=%s)"
        ):
            uid1, lid, uid2, uid3 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            assert uid1 == uid2 == uid3
            league = self._conn.leagues.get(lid)
            if league and (int(league.get("is_shared") or 0) == 1 or int(league.get("owner_user_id") or 0) == uid1):
                self._rows = [t(1)]
            return 1

        if q == "SELECT 1 FROM league_members WHERE league_id=%s AND user_id=%s AND role IN ('admin','owner')":
            self._rows = []
            return 1

        if q == "SELECT view_count FROM league_page_views WHERE league_id=%s AND page_kind=%s AND entity_id=%s":
            self._rows = []
            return 1

        if q == "SELECT video_clip_len_s FROM users WHERE id=%s":
            self._rows = []
            return 1

        if q.startswith(
            "SELECT DISTINCT division_name FROM league_teams WHERE league_id=%s AND division_name IS NOT NULL AND division_name<>''"
        ):
            # No league_teams in this unit test fixture.
            self._rows = []
            return 1

        if q.startswith("SELECT DISTINCT t.id, t.name FROM league_teams lt JOIN teams t ON lt.team_id=t.id"):
            # No league_teams in this unit test fixture.
            self._rows = []
            return 1

        if "FROM league_games lg" in q and "JOIN hky_games g ON lg.game_id=g.id" in q and "WHERE lg.league_id=%s" in q:
            league_id = int(p[0])
            rows: list[dict[str, Any]] = []
            for lg in self._conn.league_games:
                if int(lg.get("league_id") or 0) != league_id:
                    continue
                gid = int(lg.get("game_id") or 0)
                g = dict(self._conn.hky_games.get(gid) or {})
                if not g:
                    continue
                t1 = self._conn.teams[int(g["team1_id"])]
                t2 = self._conn.teams[int(g["team2_id"])]
                rows.append(
                    dict(
                        g,
                        team1_name=t1["name"],
                        team2_name=t2["name"],
                        game_type_name=None,
                        division_name=str(lg.get("division_name") or ""),
                        team1_league_division_name=None,
                        team2_league_division_name=None,
                    )
                )
            self._rows = [d(r) for r in rows]
            return 1

        if (
            "FROM hky_games g JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id" in q
            and "WHERE g.id=%s AND g.user_id=%s" in q
        ):
            gid, uid = int(p[0]), int(p[1])
            g = self._conn.hky_games.get(gid)
            if g and int(g.get("user_id") or 0) == uid:
                t1 = self._conn.teams[int(g["team1_id"])]
                t2 = self._conn.teams[int(g["team2_id"])]
                self._rows = [
                    d(
                        dict(
                            g,
                            team1_name=t1["name"],
                            team2_name=t2["name"],
                            team1_ext=int(t1.get("is_external") or 0),
                            team2_ext=int(t2.get("is_external") or 0),
                        )
                    )
                ]
            return 1

        if q == "SELECT id FROM users WHERE email=%s":
            email = str(p[0]).strip().lower()
            u = self._conn.users_by_email.get(email)
            self._rows = [t(int(u["id"]))] if u else []
            return 1

        if q.startswith("INSERT INTO users(email, password_hash, name, created_at) VALUES"):
            email, _pw, name, created_at = p
            email = str(email).strip().lower()
            existing = self._conn.users_by_email.get(email)
            if existing:
                self.lastrowid = int(existing["id"])
                return 1
            uid = int(self._conn._next_id["users"])
            self._conn._next_id["users"] += 1
            self._conn.users_by_email[email] = {"id": uid, "email": email, "name": name, "created_at": created_at}
            self.lastrowid = uid
            return 1

        if q == "SELECT id FROM leagues WHERE name=%s":
            name = str(p[0]).strip()
            lid = self._conn.league_id_by_name.get(name)
            self._rows = [t(int(lid))] if lid is not None else []
            return 1

        if q.startswith(
            "INSERT INTO leagues(name, owner_user_id, is_shared, is_public, source, external_key, created_at) VALUES"
        ):
            name, owner_user_id, is_shared, is_public, source, external_key, created_at = p
            name = str(name).strip()
            existing = self._conn.league_id_by_name.get(name)
            if existing is not None:
                self.lastrowid = int(existing)
                return 1
            lid = int(self._conn._next_id["leagues"])
            self._conn._next_id["leagues"] += 1
            self._conn.league_id_by_name[name] = lid
            self._conn.leagues[lid] = {
                "id": lid,
                "name": name,
                "owner_user_id": int(owner_user_id),
                "is_shared": int(is_shared),
                "is_public": int(is_public),
                "source": source,
                "external_key": external_key,
                "created_at": created_at,
            }
            self.lastrowid = lid
            return 1

        if q == "SELECT id FROM teams WHERE user_id=%s AND name=%s":
            user_id, name = int(p[0]), str(p[1])
            tid = self._conn.team_id_by_user_name.get((user_id, name))
            self._rows = [t(int(tid))] if tid is not None else []
            return 1

        if q == "SELECT id, name FROM teams WHERE user_id=%s":
            user_id = int(p[0])
            rows = []
            for tid, tr in self._conn.teams.items():
                if int(tr.get("user_id") or 0) != user_id:
                    continue
                rows.append(t(int(tid), str(tr.get("name") or "")))
            self._rows = rows
            return 1

        if q == "SELECT logo_path FROM teams WHERE id=%s":
            team_id = int(p[0])
            team = self._conn.teams.get(team_id)
            if team:
                self._rows = [d({"logo_path": team.get("logo_path")})] if self._dict_mode else [t(team.get("logo_path"))]
            return 1

        if q == "UPDATE teams SET logo_path=%s, updated_at=%s WHERE id=%s":
            logo_path, _updated_at, team_id = p
            team_id = int(team_id)
            if team_id in self._conn.teams:
                self._conn.teams[team_id]["logo_path"] = str(logo_path)
            return 1

        if q.startswith("INSERT INTO teams(user_id, name, is_external, created_at) VALUES"):
            user_id, name, is_external, created_at = p
            user_id = int(user_id)
            name = str(name)
            existing = self._conn.team_id_by_user_name.get((user_id, name))
            if existing is not None:
                self.lastrowid = int(existing)
                return 1
            tid = int(self._conn._next_id["teams"])
            self._conn._next_id["teams"] += 1
            self._conn.team_id_by_user_name[(user_id, name)] = tid
            self._conn.teams[tid] = {
                "id": tid,
                "user_id": user_id,
                "name": name,
                "is_external": int(is_external),
                "logo_path": None,
                "created_at": created_at,
            }
            self.lastrowid = tid
            return 1

        if q == "INSERT IGNORE INTO league_teams(league_id, team_id) VALUES(%s,%s)":
            league_id, team_id = int(p[0]), int(p[1])
            existing = next(
                (
                    r
                    for r in self._conn.league_teams
                    if int(r["league_id"]) == league_id and int(r["team_id"]) == team_id
                ),
                None,
            )
            if existing is None:
                self._conn.league_teams.append(
                    {"league_id": league_id, "team_id": team_id, "division_name": None, "division_id": None, "conference_id": None}
                )
            return 1

        if q.startswith(
            "INSERT INTO league_teams(league_id, team_id, division_name, division_id, conference_id) VALUES"
        ):
            league_id, team_id, division_name, division_id, conference_id = p
            existing = next(
                (
                    r
                    for r in self._conn.league_teams
                    if int(r["league_id"]) == int(league_id) and int(r["team_id"]) == int(team_id)
                ),
                None,
            )
            if existing is None:
                self._conn.league_teams.append(
                    {
                        "league_id": int(league_id),
                        "team_id": int(team_id),
                        "division_name": division_name,
                        "division_id": division_id,
                        "conference_id": conference_id,
                    }
                )
            else:
                incoming_dn = str(division_name).strip() if division_name is not None else None
                cur_dn = existing.get("division_name")
                cur_dn_s = str(cur_dn).strip() if cur_dn is not None else ""
                if incoming_dn and incoming_dn.lower() != "external":
                    existing["division_name"] = incoming_dn
                elif (not cur_dn_s) and incoming_dn:
                    existing["division_name"] = incoming_dn
                if division_id is not None and existing.get("division_id") is None:
                    existing["division_id"] = division_id
                if conference_id is not None and existing.get("conference_id") is None:
                    existing["conference_id"] = conference_id
            return 1

        if q.startswith(
            "INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id, sort_order) VALUES"
        ):
            league_id, game_id, division_name, division_id, conference_id, sort_order = p
            existing = next(
                (
                    r
                    for r in self._conn.league_games
                    if int(r["league_id"]) == int(league_id) and int(r["game_id"]) == int(game_id)
                ),
                None,
            )
            if existing is None:
                self._conn.league_games.append(
                    {
                        "league_id": int(league_id),
                        "game_id": int(game_id),
                        "division_name": division_name,
                        "division_id": division_id,
                        "conference_id": conference_id,
                        "sort_order": int(sort_order) if sort_order is not None else None,
                    }
                )
            else:
                incoming_dn = str(division_name).strip() if division_name is not None else None
                cur_dn = existing.get("division_name")
                cur_dn_s = str(cur_dn).strip() if cur_dn is not None else ""
                if incoming_dn and incoming_dn.lower() != "external":
                    existing["division_name"] = incoming_dn
                elif (not cur_dn_s) and incoming_dn:
                    existing["division_name"] = incoming_dn
                if division_id is not None and existing.get("division_id") is None:
                    existing["division_id"] = division_id
                if conference_id is not None and existing.get("conference_id") is None:
                    existing["conference_id"] = conference_id
                if sort_order is not None and existing.get("sort_order") is None:
                    existing["sort_order"] = int(sort_order)
            return 1

        if "FROM league_teams lt JOIN teams t ON lt.team_id=t.id" in q and "WHERE lt.league_id=%s" in q:
            league_id = int(p[0])
            out = []
            for lt in self._conn.league_teams:
                if int(lt["league_id"]) != league_id:
                    continue
                team = self._conn.teams.get(int(lt["team_id"]))
                if not team:
                    continue
                if self._dict_mode:
                    out.append(
                        d(
                            {
                                "team_id": int(team["id"]),
                                "team_name": str(team.get("name") or ""),
                                "division_name": lt.get("division_name"),
                                "division_id": lt.get("division_id"),
                                "conference_id": lt.get("conference_id"),
                            }
                        )
                    )
                else:
                    out.append(
                        t(
                            int(team["id"]),
                            str(team.get("name") or ""),
                            lt.get("division_name"),
                            lt.get("division_id"),
                            lt.get("conference_id"),
                        )
                    )
            self._rows = out
            return 1

        if "SELECT id FROM hky_games WHERE notes LIKE %s LIMIT 1" in q:
            token = str(p[0])
            for gid, g in self._conn.hky_games.items():
                if token.strip("%") in str(g.get("notes") or ""):
                    self._rows = [t(int(gid))]
                    return 1
            self._rows = []
            return 1

        if q == "SELECT id FROM hky_games WHERE user_id=%s AND notes LIKE %s LIMIT 1":
            user_id, token = int(p[0]), str(p[1])
            for gid, g in self._conn.hky_games.items():
                if int(g.get("user_id") or 0) != user_id:
                    continue
                if token.strip("%") in str(g.get("notes") or ""):
                    self._rows = [t(int(gid))]
                    return 1
            self._rows = []
            return 1

        if q == "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND notes LIKE %s":
            user_id, token = int(p[0]), str(p[1])
            for gid, g in self._conn.hky_games.items():
                if int(g.get("user_id") or 0) != user_id:
                    continue
                if token.strip("%") in str(g.get("notes") or ""):
                    row = {
                        "id": int(gid),
                        "notes": g.get("notes"),
                        "team1_score": g.get("team1_score"),
                        "team2_score": g.get("team2_score"),
                    }
                    self._rows = [d(row)]
                    return 1
            self._rows = []
            return 1

        if q.startswith("INSERT INTO hky_games(user_id, team1_id, team2_id,"):
            # Schema evolved to include game_type_id.
            if "game_type_id" in q:
                (
                    user_id,
                    team1_id,
                    team2_id,
                    game_type_id,
                    starts_at,
                    location,
                    team1_score,
                    team2_score,
                    is_final,
                    notes,
                    stats_imported_at,
                    created_at,
                ) = p
            else:
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
                game_type_id = None
            gid = int(self._conn._next_id["hky_games"])
            self._conn._next_id["hky_games"] += 1
            self._conn.hky_games[gid] = {
                "id": gid,
                "user_id": int(user_id),
                "team1_id": int(team1_id),
                "team2_id": int(team2_id),
                "game_type_id": game_type_id,
                "starts_at": starts_at,
                "location": location,
                "team1_score": team1_score,
                "team2_score": team2_score,
                "is_final": int(is_final),
                "notes": notes,
                "created_at": created_at,
                "updated_at": None,
                "stats_imported_at": stats_imported_at,
            }
            self.lastrowid = gid
            return 1

        if q == "SELECT notes, team1_score, team2_score FROM hky_games WHERE id=%s":
            gid = int(p[0])
            g = self._conn.hky_games.get(gid)
            if g:
                self._rows = [d({"notes": g.get("notes"), "team1_score": g.get("team1_score"), "team2_score": g.get("team2_score")})]
            return 1

        if q == "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s":
            user_id, team_id, name = int(p[0]), int(p[1]), str(p[2])
            pid = self._conn.player_id_by_user_team_name.get((user_id, team_id, name))
            self._rows = [t(int(pid))] if pid is not None else []
            return 1

        if q.startswith(
            "INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES"
        ):
            user_id, team_id, name, jersey_number, position, created_at = p
            user_id = int(user_id)
            team_id = int(team_id)
            name = str(name)
            existing = self._conn.player_id_by_user_team_name.get((user_id, team_id, name))
            if existing is not None:
                self.lastrowid = int(existing)
                return 1
            pid = int(self._conn._next_id["players"])
            self._conn._next_id["players"] += 1
            self._conn.player_id_by_user_team_name[(user_id, team_id, name)] = pid
            self._conn.players.append({"id": pid, "team_id": team_id, "name": name, "jersey_number": jersey_number})
            self.lastrowid = pid
            return 1

        if q.startswith("UPDATE players SET jersey_number=COALESCE(%s, jersey_number)"):
            jersey_number, _position, _updated_at, pid = p
            pid = int(pid)
            for pl in self._conn.players:
                if int(pl["id"]) == pid:
                    if jersey_number is not None:
                        pl["jersey_number"] = jersey_number
                    break
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

        if q.startswith("INSERT INTO hky_game_stats(game_id, stats_json, updated_at) VALUES"):
            gid, stats_json, updated_at = p
            gid = int(gid)
            self._conn.hky_game_stats[gid] = {
                "game_id": gid,
                "stats_json": str(stats_json),
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

        if q == "SELECT notes FROM hky_games WHERE id=%s":
            gid = int(p[0])
            g = self._conn.hky_games.get(gid)
            self._rows = [d({"notes": g.get("notes") if g else None})] if g else []
            return 1

        if q == "UPDATE hky_games SET notes=%s, updated_at=%s WHERE id=%s":
            notes, updated_at, gid = p
            gid = int(gid)
            if gid in self._conn.hky_games:
                self._conn.hky_games[gid]["notes"] = notes
                self._conn.hky_games[gid]["updated_at"] = updated_at
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


def _post_json(client, path: str, payload: dict, *, token: str = "sekret"):
    return client.post(
        path,
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_X_HM_IMPORT_TOKEN=token,
    )


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.User.objects.create(
        id=11,
        email="other@example.com",
        password_hash="x",
        name="Other",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=True,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team_a = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Team A",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team B",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team_a.id), division_name="10 A", division_id=None, conference_id=None)
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team_b.id), division_name="10 A", division_id=None, conference_id=None)

    notes = json.dumps({"timetoscore_game_id": 123, "timetoscore_season_id": 31}, sort_keys=True)
    m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=notes,
        team1_score=1,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=dt.datetime(2026, 1, 1, 0, 0, 0),
        updated_at=None,
    )
    m.LeagueGame.objects.create(league_id=1, game_id=1001, division_name="10 A", sort_order=None)

    m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team_a.id),
        name="Alice",
        jersey_number="9",
        position=None,
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=502,
        user_id=int(owner.id),
        team_id=int(team_b.id),
        name="Bob",
        jersey_number="12",
        position=None,
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    return Client(), m


def should_store_events_via_shift_package_and_render_public_game_page(client_and_models):
    client, m = client_and_models
    events1 = "Period,Time,Team,Event,Player,On-Ice Players\n1,13:45,Blue,Shot,#9 Alice,\"Alice,Bob\"\n"
    assert "\n" in events1
    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {"timetoscore_game_id": 123, "events_csv": events1, "source_label": "unit-test", "replace": False},
    )
    assert r.status_code == 200
    assert json.loads(r.content)["ok"] is True
    row = m.HkyGameEvent.objects.filter(game_id=1001).values("events_csv").first()
    assert row is not None
    assert row["events_csv"] == events1
    assert "\n" in str(row["events_csv"])

    html = client.get("/public/leagues/1/hky/games/1001").content.decode()
    assert "Game Events" in html
    assert "Shot" in html
    assert "#9 Alice" in html
    assert "Event Type" in html
    assert 'data-sortable="1"' in html
    assert 'class="cell-pre"' in html
    assert 'data-freeze-cols="1"' in html
    assert "table-scroll-y" in html


def should_find_existing_game_when_notes_are_legacy_game_id_token(client_and_models):
    client, m = client_and_models
    m.HkyGame.objects.filter(id=1001).update(notes="game_id=123")
    before_game_count = m.HkyGame.objects.count()

    # Non-Goal events are allowed to be stored even for TimeToScore-linked games.
    events1 = "Period,Time,Team,Event\n1,00:10,Blue,Shot\n"
    r = _post_json(client, "/api/import/hockey/shift_package", {"timetoscore_game_id": 123, "events_csv": events1, "replace": False})
    assert r.status_code == 200
    out = json.loads(r.content)
    assert out["ok"] is True
    assert int(out["game_id"]) == 1001
    assert m.HkyGame.objects.count() == before_game_count
    row = m.HkyGameEvent.objects.filter(game_id=1001).values("events_csv").first()
    assert row is not None
    assert row["events_csv"] == events1


def should_not_overwrite_events_without_replace(client_and_models):
    client, m = client_and_models
    events1 = "Period,Time,Team,Event\n1,00:10,Blue,Shot\n"
    events2 = "Period,Time,Team,Event\n1,00:11,Blue,Shot\n"
    r1 = _post_json(client, "/api/import/hockey/shift_package", {"timetoscore_game_id": 123, "events_csv": events1, "replace": False})
    assert r1.status_code == 200
    assert m.HkyGameEvent.objects.filter(game_id=1001).values_list("events_csv", flat=True).first() == events1

    r2 = _post_json(client, "/api/import/hockey/shift_package", {"timetoscore_game_id": 123, "events_csv": events2, "replace": False})
    assert r2.status_code == 200
    assert m.HkyGameEvent.objects.filter(game_id=1001).values_list("events_csv", flat=True).first() == events1

    r3 = _post_json(client, "/api/import/hockey/shift_package", {"timetoscore_game_id": 123, "events_csv": events2, "replace": True})
    assert r3.status_code == 200
    assert m.HkyGameEvent.objects.filter(game_id=1001).values_list("events_csv", flat=True).first() == events2


def should_store_player_stats_csv_via_shift_package_and_render_public_game_page(client_and_models):
    client, m = client_and_models
    player_stats_csv = "Player,Goals,Assists,Average Shift,Shifts,TOI Total\n9 Alice,1,0,0:45,12,12:34\n"
    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {"timetoscore_game_id": 123, "player_stats_csv": player_stats_csv, "source_label": "unit-test"},
    )
    assert r.status_code == 200
    assert json.loads(r.content)["ok"] is True
    stored = m.HkyGamePlayerStatsCsv.objects.filter(game_id=1001).values_list("player_stats_csv", flat=True).first()
    assert stored is not None
    assert "Average Shift" not in stored
    assert "Shifts" not in stored
    assert "TOI Total" not in stored

    html = client.get("/public/leagues/1/hky/games/1001").content.decode()
    assert "Imported Player Stats" in html
    assert "Average Shift" not in html
    assert "Shifts" not in html
    assert "TOI Total" not in html


def should_store_game_video_url_via_shift_package_and_show_link_in_schedule(client_and_models):
    client, m = client_and_models
    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {"timetoscore_game_id": 123, "game_video_url": "https://example.com/video", "source_label": "unit-test"},
    )
    assert r.status_code == 200
    assert json.loads(r.content)["ok"] is True
    notes = str(m.HkyGame.objects.filter(id=1001).values_list("notes", flat=True).first() or "")
    assert "game_video_url" in notes

    schedule_html = client.get("/public/leagues/1/schedule").content.decode()
    assert 'href="https://example.com/video"' in schedule_html
    assert 'target="_blank"' in schedule_html


def should_create_external_game_via_shift_package_and_map_to_league(client_and_models):
    client, m = client_and_models
    player_stats_csv = "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n"
    game_stats_csv = "Stat,chicago-4\nGoals For,2\nGoals Against,1\n"
    events_csv = "Period,Time,Team,Event Type\n1,12:00,Home,Shot\n"

    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "chicago-4",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "division_name": "External",
            "sort_order": 7,
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Opponent X",
            "player_stats_csv": player_stats_csv,
            "game_stats_csv": game_stats_csv,
            "events_csv": events_csv,
            "replace": False,
        },
    )
    assert r.status_code == 200
    out = json.loads(r.content)
    assert out["ok"] is True
    gid = int(out["game_id"])

    g = m.HkyGame.objects.filter(id=gid).values("notes", "team1_score", "team2_score").first()
    assert g is not None
    assert "external_game_key" in str(g.get("notes") or "")
    assert int(g["team1_score"]) == 2
    assert int(g["team2_score"]) == 1

    norcal = m.League.objects.filter(name="Norcal").values("id").first()
    assert norcal is not None
    assert int(norcal["id"]) >= 2
    assert m.LeagueGame.objects.filter(league_id=int(norcal["id"]), game_id=gid, sort_order=7).exists()

    assert m.Player.objects.filter(name="Charlie", jersey_number="13").exists()


def should_merge_external_game_key_into_tts_game_when_both_keys_provided(client_and_models):
    client, m = client_and_models
    events_csv = "Period,Time,Team,Event Type\n1,12:00,Home,Shot\n"

    r1 = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "game-123",
            "owner_email": "owner@example.com",
            "league_id": 1,
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Team B",
            "events_csv": events_csv,
            "replace": False,
        },
    )
    assert r1.status_code == 200
    out1 = json.loads(r1.content)
    assert out1["ok"] is True
    gid_ext = int(out1["game_id"])
    assert gid_ext != 1001
    assert m.HkyGame.objects.filter(id=gid_ext, external_game_key="game-123").exists()

    r2 = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {"timetoscore_game_id": 123, "external_game_key": "game-123", "events_csv": events_csv, "replace": False},
    )
    assert r2.status_code == 200
    out2 = json.loads(r2.content)
    assert out2["ok"] is True
    assert int(out2["game_id"]) == 1001
    assert not m.HkyGame.objects.filter(id=gid_ext).exists()

    g = m.HkyGame.objects.filter(id=1001).values("timetoscore_game_id", "external_game_key").first()
    assert g is not None
    assert int(g.get("timetoscore_game_id") or 0) == 123
    assert str(g.get("external_game_key") or "") == "game-123"


def should_render_private_game_page_as_league_owner_when_not_game_owner(client_and_models):
    client, m = client_and_models
    sess = client.session
    sess["user_id"] = 10
    sess["user_email"] = "owner@example.com"
    sess["league_id"] = 1
    sess.save()

    m.HkyGame.objects.filter(id=1001).update(user_id=11)
    r = client.get("/hky/games/1001?return_to=/teams/44")
    assert r.status_code == 200
    html = r.content.decode()
    assert "Game Summary" in html


def should_reuse_existing_league_team_by_name_and_preserve_division(client_and_models):
    client, m = client_and_models
    now = dt.datetime.now()
    m.League.objects.create(
        id=2,
        name="Norcal",
        owner_user_id=10,
        is_shared=False,
        is_public=True,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=2, team_id=101, division_name="10 B West", division_id=136, conference_id=0
    )

    before_team_count = m.Team.objects.count()
    before_div = m.LeagueTeam.objects.filter(league_id=2, team_id=101).values("division_name").first()
    assert before_div and before_div["division_name"] == "10 B West"

    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "tourny-1",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Opponent X",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
    )
    assert r.status_code == 200
    out = json.loads(r.content)
    assert out["ok"] is True
    gid = int(out["game_id"])

    assert m.Team.objects.count() == before_team_count + 1
    after_div = m.LeagueTeam.objects.filter(league_id=2, team_id=101).values("division_name").first()
    assert after_div and str(after_div["division_name"]) == "10 B West"

    lg = m.LeagueGame.objects.filter(league_id=2, game_id=gid).values("division_name").first()
    assert lg and str(lg.get("division_name") or "") == "External"

    opp_id = (
        m.LeagueTeam.objects.filter(league_id=2)
        .exclude(team_id=101)
        .values_list("team_id", flat=True)
        .first()
    )
    assert opp_id is not None
    opp = m.LeagueTeam.objects.filter(league_id=2, team_id=int(opp_id)).values("division_name").first()
    assert opp and str(opp.get("division_name") or "") == "External"


def should_match_league_team_names_case_and_punctuation_insensitive(client_and_models):
    client, m = client_and_models
    now = dt.datetime.now()
    m.League.objects.create(
        id=2,
        name="Norcal",
        owner_user_id=10,
        is_shared=False,
        is_public=True,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    sj = m.Team.objects.create(
        id=103,
        user_id=10,
        name="San Jose Jr Sharks 12AA-1",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=2, team_id=int(sj.id), division_name="12AA", division_id=0, conference_id=0)

    before_team_count = m.Team.objects.count()
    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "tourny-2",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "SAN JOSE JR. SHARKS 12AA–1",
            "away_team_name": "Opponent X",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
    )
    assert r.status_code == 200
    out = json.loads(r.content)
    assert out["ok"] is True

    assert m.Team.objects.count() == before_team_count + 1
    after_div = m.LeagueTeam.objects.filter(league_id=2, team_id=int(sj.id)).values("division_name").first()
    assert after_div and str(after_div["division_name"]) == "12AA"
    gid = int(out["game_id"])
    lg = m.LeagueGame.objects.filter(league_id=2, game_id=gid).values("division_name").first()
    assert lg and str(lg.get("division_name") or "") == "External"


def should_not_create_duplicate_external_teams_for_name_variants(client_and_models):
    client, m = client_and_models
    before_team_count = m.Team.objects.count()

    r1 = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "tourny-a",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Arizona Coyotes 12AA",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
    )
    assert r1.status_code == 200
    assert json.loads(r1.content)["ok"] is True
    assert m.Team.objects.count() == before_team_count + 1

    r2 = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "tourny-b",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "ARIZONA COYOTES 12AA",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
    )
    assert r2.status_code == 200
    assert json.loads(r2.content)["ok"] is True
    assert m.Team.objects.count() == before_team_count + 1


def should_match_team_names_even_when_db_has_division_suffix_parens(client_and_models):
    client, m = client_and_models
    now = dt.datetime.now()
    m.League.objects.create(
        id=2,
        name="Norcal",
        owner_user_id=10,
        is_shared=False,
        is_public=True,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    tid = m.Team.objects.create(
        id=103,
        user_id=10,
        name="Team A (12AA)",
        is_external=False,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=2, team_id=int(tid.id), division_name="12AA", division_id=0, conference_id=0)

    before_team_count = m.Team.objects.count()
    r = _post_json(
        client,
        "/api/import/hockey/shift_package",
        {
            "external_game_key": "tourny-parens",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Opponent X",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
    )
    assert r.status_code == 200
    out = json.loads(r.content)
    assert out["ok"] is True
    gid = int(out["game_id"])

    assert m.Team.objects.count() == before_team_count + 1
    opp_lt = (
        m.LeagueTeam.objects.filter(league_id=2)
        .exclude(team_id=int(tid.id))
        .values("division_name")
        .first()
    )
    assert opp_lt and str(opp_lt.get("division_name") or "") == "External"
    lg = m.LeagueGame.objects.filter(league_id=2, game_id=gid).values("division_name").first()
    assert lg and str(lg.get("division_name") or "") == "External"

from __future__ import annotations

import importlib.util
import os
from typing import Any


def _load_app_module():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    os.environ["HM_WATCH_ROOT"] = "/tmp/hm-incoming-test"
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
        self.league_games = [
            {"league_id": 99, "game_id": 10, "division_name": "12AA"},
            {"league_id": 99, "game_id": 11, "division_name": "External"},
            {"league_id": 99, "game_id": 12, "division_name": "12AA"},
        ]
        self.hky_games = {
            10: {"id": 10, "team1_id": 1, "team2_id": 2, "team1_score": 1, "team2_score": 2, "is_final": 1},
            11: {"id": 11, "team1_id": 1, "team2_id": 2, "team1_score": 3, "team2_score": 3, "is_final": 1},
            12: {"id": 12, "team1_id": 1, "team2_id": 3, "team1_score": 2, "team2_score": 1, "is_final": 1},
        }
        # Note: webapp uses teams.is_external for "not owned by user" and is often 1 for imported teams,
        # so cross-division filtering must not treat is_external=1 as "External division".
        self.teams = {1: {"id": 1, "is_external": 0}, 2: {"id": 2, "is_external": 1}, 3: {"id": 3, "is_external": 0}}
        # league_teams division mapping: opponent team 2 is in different division.
        self.league_teams = [
            {"league_id": 99, "team_id": 1, "division_name": "12AA"},
            {"league_id": 99, "team_id": 2, "division_name": "12A"},
            {"league_id": 99, "team_id": 3, "division_name": "12AA"},
        ]
        # player_stats: player 100 has stats in all games.
        self.player_stats = [
            {"game_id": 10, "team_id": 1, "player_id": 100, "goals": 1, "assists": 0},
            {"game_id": 11, "team_id": 1, "player_id": 100, "goals": 2, "assists": 1},
            {"game_id": 12, "team_id": 1, "player_id": 100, "goals": 0, "assists": 2},
        ]

    def cursor(self, cursorclass: Any = None):
        return FakeCursor(self, dict_mode=cursorclass is not None)


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

        if "FROM league_games lg" in q and "JOIN hky_games g" in q and "GROUP BY ps.player_id" in q:
            # Assert the query includes the cross-division filter joins.
            assert "LEFT JOIN league_teams lt_self" in q
            assert "LEFT JOIN league_teams lt_opp" in q
            assert "CASE WHEN g.team1_id=ps.team_id THEN g.team2_id ELSE g.team1_id END" in q
            assert "LOWER(COALESCE(lg.division_name,''))='external'" in q
            assert "lt_self.division_name=lt_opp.division_name" in q

            league_id, team_id = int(p[0]), int(p[1])
            div_by_team = {int(r["team_id"]): str(r.get("division_name") or "") for r in self._conn.league_teams if int(r["league_id"]) == league_id}
            div_by_game = {int(r["game_id"]): str(r.get("division_name") or "") for r in self._conn.league_games if int(r["league_id"]) == league_id}

            def include_game(game_id: int) -> bool:
                g = self._conn.hky_games[int(game_id)]
                opp_id = int(g["team2_id"]) if int(g["team1_id"]) == team_id else int(g["team1_id"])
                lg_div = str(div_by_game.get(int(game_id), "") or "")
                if lg_div.strip().lower() == "external":
                    return True
                opp_div = div_by_team.get(opp_id)
                if opp_div is None or str(opp_div).strip() == "":
                    return True
                if str(opp_div).strip().lower() == "external":
                    return True
                self_div = div_by_team.get(team_id)
                if self_div is None or str(self_div).strip() == "":
                    return True
                return str(self_div) == str(opp_div)

            totals: dict[int, dict[str, Any]] = {}
            for r in self._conn.player_stats:
                if int(r["team_id"]) != team_id:
                    continue
                gid = int(r["game_id"])
                if gid not in div_by_game:
                    continue
                if not include_game(gid):
                    continue
                pid = int(r["player_id"])
                rec = totals.setdefault(pid, {"player_id": pid, "gp": 0, "goals": 0, "assists": 0})
                rec["gp"] += 1
                rec["goals"] += int(r.get("goals") or 0)
                rec["assists"] += int(r.get("assists") or 0)

            self._rows = list(totals.values())
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


def should_ignore_cross_division_non_external_games_in_league_player_totals():
    mod = _load_app_module()
    mod.pymysql = _DummyPyMySQL()  # type: ignore[attr-defined]
    db = FakeConn()
    totals = mod.aggregate_players_totals_league(db, team_id=1, league_id=99)
    p100 = totals[100]
    # Game 10 is cross-division (12AA vs 12A) and not External => excluded.
    # Game 11 is in External division => included.
    # Game 12 is same-division => included.
    assert p100["gp"] == 2
    assert p100["goals"] == 2  # game 11 goals only
    assert p100["assists"] == 3  # game 11 (1) + game 12 (2)

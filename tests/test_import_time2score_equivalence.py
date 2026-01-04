from __future__ import annotations

import importlib.util
import json
import os
from typing import Any

import pytest


def _load_webapp_module():
    os.environ.setdefault("HM_WEBAPP_SKIP_DB_INIT", "1")
    os.environ.setdefault("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _load_importer_module():
    spec = importlib.util.spec_from_file_location("import_time2score_mod", "tools/webapp/import_time2score.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class _DummyPyMySQL:
    class cursors:
        DictCursor = object


def _canonical_snapshot(fake_db) -> dict[str, Any]:
    def _parse_notes(notes: Any) -> dict[str, Any]:
        try:
            v = json.loads(str(notes or ""))
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    leagues = {lid: dict(v) for lid, v in (getattr(fake_db, "leagues_by_id", {}) or {}).items()}
    teams = {tid: dict(v) for tid, v in (getattr(fake_db, "teams_by_id", {}) or {}).items()}
    players = {pid: dict(v) for pid, v in (getattr(fake_db, "players_by_id", {}) or {}).items()}
    games = {gid: dict(v) for gid, v in (getattr(fake_db, "hky_games_by_id", {}) or {}).items()}

    # Canonicalize by names / external ids, ignoring auto IDs and timestamps.
    league_by_name = {str(v.get("name")): {"is_shared": int(v.get("is_shared") or 0)} for v in leagues.values()}

    team_by_name = {}
    for t in teams.values():
        team_by_name[str(t.get("name"))] = {
            "is_external": int(t.get("is_external") or 0),
        }

    # League mappings
    league_teams = []
    for (lid, tid), meta in (getattr(fake_db, "league_teams_meta", {}) or {}).items():
        league_name = str(leagues[int(lid)]["name"])
        team_name = str(teams[int(tid)]["name"])
        league_teams.append((league_name, team_name, str((meta or {}).get("division_name") or "")))
    league_teams.sort()

    league_games = []
    for (lid, gid), meta in (getattr(fake_db, "league_games_meta", {}) or {}).items():
        league_name = str(leagues[int(lid)]["name"])
        g = games[int(gid)]
        notes = _parse_notes(g.get("notes"))
        league_games.append(
            (
                league_name,
                int(notes.get("timetoscore_game_id") or 0),
                str((meta or {}).get("division_name") or ""),
                str(g.get("starts_at") or ""),
                g.get("team1_score"),
                g.get("team2_score"),
            )
        )
    league_games.sort()

    # Players by (team_name, name, jersey)
    players_out = []
    for p in players.values():
        team_name = str(teams[int(p["team_id"])]["name"])
        players_out.append((team_name, str(p.get("name") or ""), str(p.get("jersey_number") or "")))
    players_out.sort()

    # Player stats by (t2s_game_id, team_name, player_name, goals, assists)
    ps_out = []
    for (_gid, _pid), row in (getattr(fake_db, "player_stats", {}) or {}).items():
        gid = int(row["game_id"])
        pid = int(row["player_id"])
        game = games[gid]
        notes = _parse_notes(game.get("notes"))
        t2s_id = int(notes.get("timetoscore_game_id") or 0)
        team_name = str(teams[int(row["team_id"])]["name"])
        player_name = str(players[pid]["name"])
        ps_out.append((t2s_id, team_name, player_name, int(row.get("goals") or 0), int(row.get("assists") or 0)))
    ps_out.sort()

    return {
        "leagues": league_by_name,
        "teams": team_by_name,
        "league_teams": league_teams,
        "league_games": league_games,
        "players": players_out,
        "player_stats": ps_out,
    }


@pytest.fixture()
def modules():
    return _load_webapp_module(), _load_importer_module()


@pytest.fixture()
def fake_db_class():
    # Reuse FakeConn from the existing webapp import API tests.
    spec = importlib.util.spec_from_file_location("webapp_import_api_tests", "tests/test_webapp_import_api.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.FakeConn


def should_import_time2score_direct_and_rest_end_in_equivalent_state(monkeypatch, modules, fake_db_class):
    webapp_mod, importer_mod = modules

    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    monkeypatch.setattr(webapp_mod, "pymysql", _DummyPyMySQL(), raising=False)

    payload = {
        "league_name": "Equiv League",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "owner_name": "Owner",
        "source": "timetoscore",
        "external_key": "caha:31",
        "games": [
            {
                "home_name": "Team A (10U A)",
                "away_name": "Team B (10U A)",
                "division_name": "10U A",
                "division_id": 10,
                "conference_id": 1,
                "home_division_name": "10U A",
                "home_division_id": 10,
                "home_conference_id": 1,
                "away_division_name": "10U A",
                "away_division_id": 10,
                "away_conference_id": 1,
                "starts_at": None,  # exercise match-by-notes behavior
                "location": "Rink 1",
                "home_score": 2,
                "away_score": 1,
                "timetoscore_game_id": 123,
                "season_id": 31,
                "home_roster": [{"name": "Alice", "number": "9", "position": "F"}],
                "away_roster": [{"name": "Bob", "number": "12", "position": "D"}],
                "player_stats": [{"name": "Alice", "goals": 1, "assists": 0}],
            }
        ],
    }

    # REST import path: hit the webapp endpoint.
    db_rest = fake_db_class()
    monkeypatch.setattr(webapp_mod, "get_db", lambda: db_rest)
    app = webapp_mod.create_app()
    app.testing = True
    client = app.test_client()
    r = client.post(
        "/api/import/hockey/games_batch",
        json=payload,
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    # Direct-DB import path: apply the same payload via the importer helper.
    db_direct = fake_db_class()
    importer_mod.apply_games_batch_payload_to_db(db_direct, payload)

    assert _canonical_snapshot(db_direct) == _canonical_snapshot(db_rest)


from __future__ import annotations

import datetime as dt
import importlib.util
import json

import pytest


def _load_webapp_module():
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


def _snapshot_state(m) -> dict[str, object]:
    def _tts_id(notes: object) -> int:
        s = str(notes or "").strip()
        if not s:
            return 0
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                return int(d.get("timetoscore_game_id") or 0)
        except Exception:
            return 0
        return 0

    def _fmt_dt(val: object) -> str:
        if val is None:
            return ""
        if isinstance(val, dt.datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return str(val)

    leagues = list(m.League.objects.values("name", "is_shared", "source", "external_key", "owner_user__email"))
    leagues.sort(key=lambda r: str(r["name"]))

    teams = list(m.Team.objects.values("user__email", "name", "is_external"))
    teams.sort(key=lambda r: (str(r["user__email"]), str(r["name"])))

    league_teams = list(
        m.LeagueTeam.objects.values("league__name", "team__name", "division_name", "division_id", "conference_id")
    )
    league_teams.sort(key=lambda r: (str(r["league__name"]), str(r["team__name"])))

    league_games = list(m.LeagueGame.objects.select_related("game").values("league__name", "game__notes", "division_name"))
    lg2 = []
    for r in league_games:
        lg2.append((str(r["league__name"]), _tts_id(r.get("game__notes")), str(r.get("division_name") or "")))
    lg2.sort()

    games = list(
        m.HkyGame.objects.select_related("team1", "team2").values(
            "notes",
            "team1__name",
            "team2__name",
            "starts_at",
            "team1_score",
            "team2_score",
        )
    )
    g2 = []
    for r in games:
        g2.append(
            (
                _tts_id(r.get("notes")),
                str(r.get("team1__name") or ""),
                str(r.get("team2__name") or ""),
                _fmt_dt(r.get("starts_at")),
                r.get("team1_score"),
                r.get("team2_score"),
            )
        )
    g2.sort()

    players = list(m.Player.objects.select_related("team").values("team__name", "name", "jersey_number", "position"))
    p2 = [(str(r.get("team__name") or ""), str(r.get("name") or ""), str(r.get("jersey_number") or "")) for r in players]
    p2.sort()

    pstats = list(
        m.PlayerStat.objects.select_related("player", "team", "game").values(
            "game__notes",
            "team__name",
            "player__name",
            "goals",
            "assists",
        )
    )
    ps2 = []
    for r in pstats:
        ps2.append(
            (
                _tts_id(r.get("game__notes")),
                str(r.get("team__name") or ""),
                str(r.get("player__name") or ""),
                int(r.get("goals") or 0),
                int(r.get("assists") or 0),
            )
        )
    ps2.sort()

    return {
        "leagues": leagues,
        "teams": teams,
        "league_teams": league_teams,
        "league_games": lg2,
        "games": g2,
        "players": p2,
        "player_stats": ps2,
    }


@pytest.fixture()
def modules(monkeypatch, webapp_db):
    _django_orm, _m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    return _load_webapp_module(), _load_importer_module()


def should_import_time2score_direct_and_rest_end_in_equivalent_state(monkeypatch, modules, webapp_db_reset, webapp_orm_modules):
    webapp_mod, importer_mod = modules
    _django_orm, m = webapp_orm_modules

    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

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
    snap_rest = _snapshot_state(m)

    # Direct-DB import path: apply the same payload via the importer helper.
    webapp_db_reset()
    importer_mod.apply_games_batch_payload_to_db(None, payload)
    snap_direct = _snapshot_state(m)

    assert snap_direct == snap_rest


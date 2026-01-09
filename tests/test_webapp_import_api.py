import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture()
def webapp_mod(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()
    return mod, m


def _post(
    client,
    path: str,
    payload: dict[str, Any],
    *,
    token: Optional[str] = None,
    environ: Optional[dict] = None,
):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(path, json=payload, headers=headers, environ_base=environ or {})


def should_require_token_when_configured(webapp_mod, monkeypatch):
    mod, _m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    r = _post(client, "/api/import/hockey/ensure_league", {"league_name": "Norcal"})
    assert r.status_code == 401
    r2 = _post(client, "/api/import/hockey/ensure_league", {"league_name": "Norcal"}, token="wrong")
    assert r2.status_code == 401
    r3 = _post(client, "/api/import/hockey/ensure_league", {"league_name": "Norcal"}, token="sekret")
    assert r3.status_code == 200
    assert r3.get_json()["ok"] is True


def should_deny_remote_import_without_token(webapp_mod, monkeypatch):
    mod, _m = webapp_mod
    monkeypatch.delenv("HM_WEBAPP_IMPORT_TOKEN", raising=False)

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

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


def should_create_and_update_shared_league(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    r = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    data = r.get_json()
    assert data["ok"] is True
    lid = int(data["league_id"])
    owner_user_id = int(data["owner_user_id"])

    league = m.League.objects.filter(id=lid).values("is_shared").first()
    assert league is not None
    assert bool(league["is_shared"]) is True
    member = m.LeagueMember.objects.filter(league_id=lid, user_id=owner_user_id).values("role").first()
    assert member is not None
    assert str(member["role"]) == "admin"

    r2 = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": False, "owner_email": "owner@example.com"},
        token="sekret",
    )
    assert int(r2.get_json()["league_id"]) == lid
    league2 = m.League.objects.filter(id=lid).values("is_shared").first()
    assert league2 is not None
    assert bool(league2["is_shared"]) is False


def should_import_game_and_be_non_destructive_without_replace(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

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
    owner_user_id = int(out["owner_user_id"])
    lid = int(out["league_id"])
    team1_id = int(out["team1_id"])
    team2_id = int(out["team2_id"])

    g = m.HkyGame.objects.filter(id=gid).values("team1_score", "team2_score", "notes").first()
    assert g is not None
    assert int(g["team1_score"]) == 1 and int(g["team2_score"]) == 2
    assert "timetoscore_game_id" in (g.get("notes") or "")
    assert "timetoscore_season_id" in (g.get("notes") or "")

    assert m.LeagueTeam.objects.filter(league_id=lid, team_id=team1_id).exists()
    assert m.LeagueTeam.objects.filter(league_id=lid, team_id=team2_id).exists()
    assert m.LeagueGame.objects.filter(league_id=lid, game_id=gid).exists()

    # Re-import with different scores/stats without replace should not overwrite existing scores,
    # but should still refresh TimeToScore-sourced goal/assist attribution.
    payload2 = dict(payload)
    payload2["game"] = dict(payload["game"])
    payload2["game"]["home_score"] = 9
    payload2["game"]["away_score"] = 9
    payload2["game"]["player_stats"] = [{"name": "Alice", "goals": 7, "assists": 7}]
    r2 = _post(client, "/api/import/hockey/game", payload2, token="sekret")
    assert r2.status_code == 200
    gid2 = int(r2.get_json()["game_id"])
    assert gid2 == gid

    g2 = m.HkyGame.objects.filter(id=gid2).values("team1_score", "team2_score").first()
    assert g2 is not None
    assert int(g2["team1_score"]) == 1 and int(g2["team2_score"]) == 2

    alice = m.Player.objects.filter(user_id=owner_user_id, team_id=team1_id, name="Alice").values("id").first()
    assert alice is not None
    ps = (
        m.PlayerStat.objects.filter(game_id=gid, player_id=int(alice["id"]))
        .values("goals", "assists")
        .first()
    )
    assert ps is not None
    assert int(ps["goals"]) == 7 and int(ps["assists"]) == 7


def should_persist_division_metadata_on_import(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

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
    lt1 = m.LeagueTeam.objects.filter(league_id=lid, team_id=team1_id).values("division_name").first()
    assert lt1 is not None
    assert str(lt1.get("division_name")) == "10U A"
    lt2 = m.LeagueTeam.objects.filter(league_id=lid, team_id=team2_id).values("division_id").first()
    assert lt2 is not None
    assert int(lt2["division_id"]) == 55
    lg = m.LeagueGame.objects.filter(league_id=lid, game_id=gid).values("conference_id").first()
    assert lg is not None
    assert int(lg["conference_id"]) == 7

    # Re-import without division fields should not erase existing metadata.
    payload2 = dict(payload)
    payload2["game"] = dict(payload["game"])
    payload2["game"].pop("division_name", None)
    payload2["game"].pop("division_id", None)
    payload2["game"].pop("conference_id", None)
    r3 = _post(client, "/api/import/hockey/game", payload2, token="sekret")
    assert r3.status_code == 200
    lt1b = m.LeagueTeam.objects.filter(league_id=lid, team_id=team1_id).values("division_name").first()
    assert lt1b is not None
    assert str(lt1b.get("division_name")) == "10U A"


def should_overwrite_with_replace(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

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

    g = m.HkyGame.objects.filter(id=gid).values("team1_score").first()
    assert g is not None
    assert int(g["team1_score"]) == 3

    alice = (
        m.Player.objects.filter(user_id=int(out["owner_user_id"]), team_id=int(out["team1_id"]), name="Alice")
        .values("id")
        .first()
    )
    assert alice is not None
    ps = (
        m.PlayerStat.objects.filter(game_id=gid, player_id=int(alice["id"]))
        .values("goals", "assists")
        .first()
    )
    assert ps is not None
    assert int(ps["goals"]) == 2 and int(ps["assists"]) == 1


def should_match_existing_game_by_timetoscore_id_when_no_starts_at(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    r = _post(
        client,
        "/api/import/hockey/ensure_league",
        {"league_name": "Norcal", "shared": True, "owner_email": "owner@example.com"},
        token="sekret",
    )
    owner_user_id = int(r.get_json()["owner_user_id"])

    now = dt.datetime.now()
    home = m.Team.objects.create(user_id=owner_user_id, name="Home", is_external=True, created_at=now, updated_at=None)
    away = m.Team.objects.create(user_id=owner_user_id, name="Away", is_external=True, created_at=now, updated_at=None)
    g0 = m.HkyGame.objects.create(
        user_id=owner_user_id,
        team1_id=int(home.id),
        team2_id=int(away.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=json.dumps({"timetoscore_game_id": 999}),
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=now,
        created_at=now,
        updated_at=None,
    )
    existing_gid = int(g0.id)

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
    g = m.HkyGame.objects.filter(id=gid2).values("team1_score", "team2_score").first()
    assert g is not None
    assert int(g["team1_score"]) == 1
    assert int(g["team2_score"]) == 0


def should_update_player_roster_fields_without_replace(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

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
    owner_user_id = int(out["owner_user_id"])
    team1_id = int(out["team1_id"])

    alice = (
        m.Player.objects.filter(user_id=owner_user_id, team_id=team1_id, name="Alice").values("id", "jersey_number").first()
    )
    assert alice is not None
    assert alice["jersey_number"] is None

    payload2 = dict(payload)
    payload2["game"] = dict(payload["game"])
    payload2["game"]["home_roster"] = [{"name": "Alice", "number": "12", "position": "F"}]
    _post(client, "/api/import/hockey/game", payload2, token="sekret")

    alice2 = m.Player.objects.filter(id=int(alice["id"])).values("jersey_number", "position").first()
    assert alice2 is not None
    assert alice2["jersey_number"] == "12"
    assert alice2["position"] == "F"


def should_not_map_games_or_teams_into_external_when_division_is_external_but_team_exists(webapp_mod, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    payload1 = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
        "games": [
            {
                "home_name": "San Jose Jr Sharks 12AA-1",
                "away_name": "Arizona Coyotes 12AA",
                "starts_at": "2026-01-01 10:00:00",
                "location": "Rink 1",
                "home_score": 1,
                "away_score": 2,
                "timetoscore_game_id": 2001,
                "season_id": 77,
                "division_name": "12AA",
                "home_division_name": "12AA",
                "away_division_name": "12AA",
            }
        ],
    }
    r1 = _post(client, "/api/import/hockey/games_batch", payload1, token="sekret")
    assert r1.status_code == 200
    assert r1.get_json()["ok"] is True
    league_id = int(r1.get_json()["league_id"])
    owner_user_id = int(r1.get_json()["owner_user_id"])

    assert m.Team.objects.filter(user_id=owner_user_id, is_external=True).count() == 2

    payload2 = {
        "league_name": "Norcal",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
        "games": [
            {
                "home_name": "San Jose Jr Sharks 12AA-1",
                "away_name": "Arizona Coyotes 12AA",
                "starts_at": "2026-01-02 10:00:00",
                "location": "Rink 1",
                "home_score": 3,
                "away_score": 4,
                "timetoscore_game_id": 2002,
                "season_id": 77,
                "division_name": "External",
                "home_division_name": "External",
                "away_division_name": "External",
            }
        ],
    }
    r2 = _post(client, "/api/import/hockey/games_batch", payload2, token="sekret")
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True

    assert m.Team.objects.filter(user_id=owner_user_id, is_external=True).count() == 2

    gid2 = int(r2.get_json()["results"][0]["game_id"])
    lg2 = m.LeagueGame.objects.filter(league_id=league_id, game_id=gid2).values("division_name").first()
    assert lg2 is not None
    assert str(lg2.get("division_name")) == "12AA"

    team_ids = list(
        m.Team.objects.filter(user_id=owner_user_id, is_external=True)
        .order_by("id")
        .values_list("id", flat=True)
    )
    assert len(team_ids) == 2
    lt_rows = list(
        m.LeagueTeam.objects.filter(league_id=league_id, team_id__in=team_ids)
        .order_by("team_id")
        .values("team_id", "division_name")
    )
    assert [str(r.get("division_name")) for r in lt_rows] == ["12AA", "12AA"]


def _snapshot_import_state(m) -> dict[str, Any]:
    def fmt_dt(v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, dt.datetime):
            return v.strftime("%Y-%m-%d %H:%M:%S")
        s = str(v).strip()
        return s or None

    def tts_id(notes: Any) -> Optional[int]:
        s = str(notes or "").strip()
        if not s:
            return None
        try:
            d = json.loads(s)
            if isinstance(d, dict) and d.get("timetoscore_game_id") is not None:
                return int(d["timetoscore_game_id"])
        except Exception:
            return None
        return None

    users = list(m.User.objects.exclude(email="admin").values("email", "name", "video_clip_len_s"))
    users.sort(key=lambda r: str(r["email"]))

    leagues = list(
        m.League.objects.values("name", "is_shared", "source", "external_key", "owner_user__email").order_by("name")
    )
    members = list(
        m.LeagueMember.objects.values("league__name", "user__email", "role")
        .order_by("league__name", "user__email", "role")
    )
    teams = list(m.Team.objects.values("user__email", "name", "is_external").order_by("user__email", "name"))
    league_teams = list(
        m.LeagueTeam.objects.values(
            "league__name",
            "team__name",
            "division_name",
            "division_id",
            "conference_id",
        ).order_by("league__name", "team__name")
    )

    games_rows = list(
        m.HkyGame.objects.select_related("team1", "team2")
        .values(
            "id",
            "notes",
            "team1__name",
            "team2__name",
            "starts_at",
            "location",
            "team1_score",
            "team2_score",
            "is_final",
        )
        .order_by("id")
    )
    games = []
    for r in games_rows:
        games.append(
            {
                "tts_game_id": tts_id(r.get("notes")),
                "team1_name": r.get("team1__name"),
                "team2_name": r.get("team2__name"),
                "starts_at": fmt_dt(r.get("starts_at")),
                "location": r.get("location"),
                "team1_score": r.get("team1_score"),
                "team2_score": r.get("team2_score"),
                "is_final": bool(r.get("is_final")),
            }
        )
    games.sort(key=lambda r: (r.get("tts_game_id") is None, int(r.get("tts_game_id") or 0)))

    league_games = list(
        m.LeagueGame.objects.select_related("game", "league")
        .values("league__name", "game__notes", "division_name", "division_id", "conference_id", "sort_order")
    )
    lg2 = []
    for r in league_games:
        lg2.append(
            {
                "league_name": r.get("league__name"),
                "tts_game_id": tts_id(r.get("game__notes")),
                "division_name": r.get("division_name"),
                "division_id": r.get("division_id"),
                "conference_id": r.get("conference_id"),
                "sort_order": r.get("sort_order"),
            }
        )
    lg2.sort(key=lambda r: (str(r.get("league_name") or ""), int(r.get("tts_game_id") or 0)))

    players = list(
        m.Player.objects.select_related("team")
        .values("team__name", "name", "jersey_number", "position")
        .order_by("team__name", "name")
    )
    pstats_rows = list(
        m.PlayerStat.objects.select_related("player", "team", "game")
        .values("game__notes", "player__name", "team__name", "goals", "assists")
    )
    pstats = []
    for r in pstats_rows:
        pstats.append(
            {
                "tts_game_id": tts_id(r.get("game__notes")),
                "player_name": r.get("player__name"),
                "team_name": r.get("team__name"),
                "goals": r.get("goals"),
                "assists": r.get("assists"),
            }
        )
    pstats.sort(
        key=lambda r: (
            int(r.get("tts_game_id") or 0),
            str(r.get("team_name") or ""),
            str(r.get("player_name") or ""),
        )
    )

    return {
        "users": users,
        "leagues": leagues,
        "members": members,
        "teams": teams,
        "league_teams": league_teams,
        "games": games,
        "league_games": lg2,
        "players": players,
        "player_stats": pstats,
    }


def should_import_games_batch_and_match_individual_imports(webapp_mod, webapp_db_reset, monkeypatch):
    mod, m = webapp_mod
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

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
    snap_individual = _snapshot_import_state(m)

    webapp_db_reset()
    app2 = mod.create_app()
    app2.testing = True
    client2 = app2.test_client()

    batch_payload = dict(payload)
    batch_payload["games"] = [game1, game2]
    r = _post(client2, "/api/import/hockey/games_batch", batch_payload, token="sekret")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert int(r.get_json()["imported"]) == 2
    snap_batch = _snapshot_import_state(m)

    assert snap_batch == snap_individual


def should_import_team_logos_from_urls_in_games_batch(tmp_path, webapp_db, monkeypatch):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    mod = _load_app_module()
    monkeypatch.setattr(mod, "INSTANCE_DIR", tmp_path, raising=False)

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

    assert got["headers"].get("User-Agent") == "Mozilla/5.0"
    assert "Referer" in got["headers"]

    teams = list(m.Team.objects.order_by("id").values("logo_path"))
    assert len(teams) == 2
    for team in teams:
        logo_path = str(team.get("logo_path") or "")
        assert logo_path
        assert tmp_path in Path(logo_path).resolve().parents
        assert Path(logo_path).exists()
        assert Path(logo_path).read_bytes() == b"PNGDATA"


def should_import_team_logos_from_b64_in_games_batch_without_requests(tmp_path, webapp_db, monkeypatch):
    import base64

    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    mod = _load_app_module()
    monkeypatch.setattr(mod, "INSTANCE_DIR", tmp_path, raising=False)
    monkeypatch.setitem(sys.modules, "requests", None)

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

    teams = list(m.Team.objects.order_by("id").values("logo_path"))
    assert len(teams) == 2
    for team in teams:
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


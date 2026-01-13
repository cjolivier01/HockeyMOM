from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Optional

import pytest


def _post_json(client, path: str, payload: dict[str, Any], *, token: str = "sekret"):
    return client.post(
        path,
        data=json.dumps(payload),
        content_type="application/json",
        HTTP_X_HM_IMPORT_TOKEN=token,
    )


def _set_session(client, *, user_id: int, email: str, league_id: Optional[int] = None) -> None:
    sess = client.session
    sess["user_id"] = int(user_id)
    sess["user_email"] = str(email)
    if league_id is not None:
        sess["league_id"] = int(league_id)
    sess.save()


def _extract_game_stats_counts(html: str, *, event_type: str) -> tuple[int, int]:
    m = re.search(
        r"<h3[^>]*>\s*Game Stats\s*</h3>.*?<table[^>]*>.*?<tbody>(?P<tbody>.*?)</tbody>",
        html,
        re.DOTALL,
    )
    assert m is not None, "Game Stats table not found"
    tbody = m.group("tbody")
    row_m = re.search(
        rf"<tr>\s*<td>\s*{re.escape(event_type)}\s*</td>\s*"
        r"<td>\s*(?P<home>[^<]*)\s*</td>\s*"
        r"<td>\s*(?P<away>[^<]*)\s*</td>",
        tbody,
        re.DOTALL,
    )
    assert row_m is not None, f"Event type row not found in Game Stats table: {event_type!r}"
    return int(str(row_m.group("home") or "0").strip() or 0), int(
        str(row_m.group("away") or "0").strip() or 0
    )


def _extract_events_table_event_id(html: str, *, event_type: str) -> str:
    m = re.search(
        r'<table[^>]*id="game-events-table"[^>]*>.*?<tbody>(?P<tbody>.*?)</tbody>',
        html,
        re.DOTALL,
    )
    assert m is not None, "Events table not found"
    tbody = m.group("tbody")
    row_m = re.search(
        rf"<tr[^>]*>\s*<td>\s*{re.escape(event_type)}\s*</td>\s*<td>\s*(?P<eid>[^<]*)\s*</td>",
        tbody,
        re.DOTALL,
    )
    assert row_m is not None, f"Events table row not found: {event_type!r}"
    return str(row_m.group("eid") or "").strip()


def _fmt_dt(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v).strip() or None


def _stable_state(m) -> dict[str, Any]:
    leagues = list(
        m.League.objects.values(
            "name",
            "is_shared",
            "is_public",
            "show_goalie_stats",
            "source",
            "external_key",
            "owner_user__email",
        ).order_by("name")
    )
    teams = list(
        m.Team.objects.values("user__email", "name", "is_external").order_by("user__email", "name")
    )
    players = list(
        m.Player.objects.select_related("team")
        .values("team__name", "name", "jersey_number", "position")
        .order_by("team__name", "name", "jersey_number")
    )
    games = []
    for r in (
        m.HkyGame.objects.select_related("team1", "team2")
        .values(
            "id",
            "timetoscore_game_id",
            "external_game_key",
            "team1__name",
            "team2__name",
            "starts_at",
            "location",
            "team1_score",
            "team2_score",
            "is_final",
        )
        .order_by("id")
    ):
        games.append(
            {
                "timetoscore_game_id": r.get("timetoscore_game_id"),
                "external_game_key": r.get("external_game_key"),
                "team1_name": r.get("team1__name"),
                "team2_name": r.get("team2__name"),
                "starts_at": _fmt_dt(r.get("starts_at")),
                "location": r.get("location"),
                "team1_score": r.get("team1_score"),
                "team2_score": r.get("team2_score"),
                "is_final": bool(r.get("is_final")),
            }
        )

    league_games = list(
        m.LeagueGame.objects.select_related("league", "game")
        .values("league__name", "game_id", "division_name", "sort_order")
        .order_by("league__name", "game_id")
    )
    league_teams = list(
        m.LeagueTeam.objects.select_related("league", "team")
        .values("league__name", "team__name", "division_name")
        .order_by("league__name", "team__name")
    )
    player_stats = list(
        m.PlayerStat.objects.select_related("player", "team", "game")
        .values("game_id", "player__name", "team__name", "goals", "assists", "shots")
        .order_by("game_id", "team__name", "player__name")
    )
    event_rows = list(
        m.HkyGameEventRow.objects.select_related("event_type")
        .values(
            "game_id",
            "event_type__key",
            "import_key",
            "event_id",
            "team_side",
            "period",
            "game_seconds",
            "game_seconds_end",
            "player_id",
            "details",
        )
        .order_by("game_id", "event_type__key", "import_key")
    )
    return {
        "leagues": leagues,
        "teams": teams,
        "league_teams": league_teams,
        "games": games,
        "league_games": league_games,
        "players": players,
        "player_stats": player_stats,
        "event_rows": event_rows,
    }


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    from django.test import Client

    return Client(), m


def should_render_game_stats_table_without_double_counting_duplicate_goals_and_be_idempotent(
    client_and_models,
):
    client, m = client_and_models

    events_csv = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Source,Details,Game Seconds End\n"
        "Goal,Away,1,100,21,timetoscore,,\n"
        "Goal,Away,1,100,22,timetoscore,,\n"
        "Goal,Home,1,200,9,timetoscore,,\n"
        "Goal,Away,1,300,21,timetoscore,,\n"
        "Penalty,Home,1,400,9,timetoscore,Tripping,430\n"
    )

    payload = {
        "league_name": "CAHA",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
        "games": [
            {
                "home_name": "Home A",
                "away_name": "Away A",
                "starts_at": "2026-01-02 10:00:00",
                "location": "Rink 1",
                "home_score": 1,
                "away_score": 2,
                "is_final": True,
                "timetoscore_game_id": 123,
                "season_id": 77,
                "division_name": "12AA",
                "home_division_name": "12AA",
                "away_division_name": "12AA",
                "home_roster": [{"name": "Alice", "number": "9", "position": "F"}],
                "away_roster": [
                    {"name": "Bob", "number": "21", "position": "F"},
                    {"name": "Carl", "number": "22", "position": "F"},
                ],
                "events_csv": events_csv,
            }
        ],
    }

    r1 = _post_json(client, "/api/import/hockey/games_batch", payload)
    assert r1.status_code == 200
    out1 = json.loads(r1.content)
    assert out1["ok"] is True
    gid = int(out1["results"][0]["game_id"])
    league_id = int(out1["league_id"])
    owner_user_id = int(out1["owner_user_id"])

    assert m.HkyGameEventRow.objects.filter(game_id=gid, event_type__key="goal").count() == 3
    assert m.HkyGameEventRow.objects.filter(game_id=gid, event_type__key="penalty").count() == 1

    _set_session(client, user_id=owner_user_id, email="owner@example.com", league_id=league_id)
    page1 = client.get(f"/hky/games/{gid}?return_to=/teams")
    assert page1.status_code == 200
    html1 = page1.content.decode()
    assert _extract_game_stats_counts(html1, event_type="Goal") == (1, 2)
    penalty_eid = _extract_events_table_event_id(html1, event_type="Penalty")
    assert penalty_eid.isdigit()

    # Re-import the same payload and ensure we did not create duplicates.
    before = m.HkyGameEventRow.objects.filter(game_id=gid).count()
    r2 = _post_json(client, "/api/import/hockey/games_batch", payload)
    assert r2.status_code == 200
    out2 = json.loads(r2.content)
    assert out2["ok"] is True
    assert int(out2["results"][0]["game_id"]) == gid
    assert m.HkyGameEventRow.objects.filter(game_id=gid).count() == before

    page2 = client.get(f"/hky/games/{gid}?return_to=/teams")
    assert page2.status_code == 200
    html2 = page2.content.decode()
    assert _extract_game_stats_counts(html2, event_type="Goal") == (1, 2)


def should_run_games_batch_plus_shift_package_twice_and_keep_stable_state(client_and_models):
    client, m = client_and_models

    events_csv_tts = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Source,Details,Game Seconds End\n"
        "Goal,Away,1,100,21,timetoscore,,\n"
        "Goal,Away,1,100,22,timetoscore,,\n"
        "Goal,Home,1,200,9,timetoscore,,\n"
        "Goal,Away,1,300,21,timetoscore,,\n"
        "Penalty,Home,1,400,9,timetoscore,Tripping,430\n"
    )
    events_csv_shift = (
        "Event Type,Team Side,Period,Game Seconds,Attributed Jerseys,Event ID,Source,Details,Game Seconds End\n"
        "Goal,Away,1,100,21,101,shift_package,,\n"
        "Goal,Away,1,100,22,102,shift_package,,\n"
        "Goal,Home,1,200,9,201,shift_package,,\n"
        "Goal,Away,1,300,21,103,shift_package,,\n"
        "Penalty,Home,1,400,9,,shift_package,Tripping,430\n"
    )

    games_payload = {
        "league_name": "CAHA",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:77",
        "games": [
            {
                "home_name": "Home A",
                "away_name": "Away A",
                "starts_at": "2026-01-02 10:00:00",
                "location": "Rink 1",
                "home_score": 1,
                "away_score": 2,
                "is_final": True,
                "timetoscore_game_id": 123,
                "season_id": 77,
                "division_name": "12AA",
                "home_division_name": "12AA",
                "away_division_name": "12AA",
                "home_roster": [{"name": "Alice", "number": "9", "position": "F"}],
                "away_roster": [
                    {"name": "Bob", "number": "21", "position": "F"},
                    {"name": "Carl", "number": "22", "position": "F"},
                ],
                "events_csv": events_csv_tts,
            }
        ],
    }

    shift_payload: dict[str, Any] = {
        "owner_email": "owner@example.com",
        "league_name": "CAHA",
        "team_side": "home",
        "replace": False,
        "events_csv": events_csv_shift,
        "player_stats_csv": (
            "Jersey #,Player,Goals,Assists,Shots\n" "9,Alice,1,0,2\n" "21,Bob,2,0,3\n"
        ),
        "game_stats_csv": "Stat,Game 1\nShots (Home),5\nShots (Away),7\n",
    }

    def _run_once() -> tuple[int, int, int]:
        r = _post_json(client, "/api/import/hockey/games_batch", games_payload)
        assert r.status_code == 200
        out = json.loads(r.content)
        assert out["ok"] is True
        gid = int(out["results"][0]["game_id"])
        lid = int(out["league_id"])
        uid = int(out["owner_user_id"])
        return gid, lid, uid

    gid1, lid1, uid1 = _run_once()
    shift_payload["game_id"] = int(gid1)
    r_shift1 = _post_json(client, "/api/import/hockey/shift_package", shift_payload)
    assert r_shift1.status_code == 200
    out_shift1 = json.loads(r_shift1.content)
    assert out_shift1["ok"] is True

    snap1 = _stable_state(m)

    gid2, lid2, uid2 = _run_once()
    assert gid2 == gid1
    assert lid2 == lid1
    assert uid2 == uid1
    r_shift2 = _post_json(client, "/api/import/hockey/shift_package", shift_payload)
    assert r_shift2.status_code == 200
    out_shift2 = json.loads(r_shift2.content)
    assert out_shift2["ok"] is True

    snap2 = _stable_state(m)
    assert snap2 == snap1

    _set_session(client, user_id=uid1, email="owner@example.com", league_id=lid1)
    page = client.get(f"/hky/games/{gid1}?return_to=/teams")
    assert page.status_code == 200
    html = page.content.decode()
    assert _extract_game_stats_counts(html, event_type="Goal") == (1, 2)
    assert _extract_events_table_event_id(html, event_type="Penalty").isdigit()

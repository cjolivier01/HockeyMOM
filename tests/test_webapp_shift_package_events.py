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


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")

    mod = _load_app_module()

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

    app = mod.create_app()
    app.testing = True
    return app.test_client(), m


def should_store_events_via_shift_package_and_render_public_game_page(client_and_models):
    client, m = client_and_models
    events1 = "Period,Time,Team,Event,Player,On-Ice Players\n1,13:45,Blue,Shot,#9 Alice,\"Alice,Bob\"\n"
    assert "\n" in events1
    r = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events1, "source_label": "unit-test", "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    row = m.HkyGameEvent.objects.filter(game_id=1001).values("events_csv").first()
    assert row is not None
    assert row["events_csv"] == events1
    assert "\n" in str(row["events_csv"])

    html = client.get("/public/leagues/1/hky/games/1001").get_data(as_text=True)
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

    events1 = "Period,Time,Team,Event\n1,00:10,Blue,Shot\n"
    r = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events1, "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    out = r.get_json()
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
    r1 = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events1, "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r1.status_code == 200
    assert m.HkyGameEvent.objects.filter(game_id=1001).values_list("events_csv", flat=True).first() == events1

    r2 = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events2, "replace": False},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r2.status_code == 200
    assert m.HkyGameEvent.objects.filter(game_id=1001).values_list("events_csv", flat=True).first() == events1

    r3 = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "events_csv": events2, "replace": True},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r3.status_code == 200
    assert m.HkyGameEvent.objects.filter(game_id=1001).values_list("events_csv", flat=True).first() == events2


def should_store_player_stats_csv_via_shift_package_and_render_public_game_page(client_and_models):
    client, m = client_and_models
    player_stats_csv = "Player,Goals,Assists,Average Shift,Shifts,TOI Total\n9 Alice,1,0,0:45,12,12:34\n"
    r = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "player_stats_csv": player_stats_csv, "source_label": "unit-test"},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    stored = m.HkyGamePlayerStatsCsv.objects.filter(game_id=1001).values_list("player_stats_csv", flat=True).first()
    assert stored is not None
    assert "Average Shift" not in stored
    assert "Shifts" not in stored
    assert "TOI Total" not in stored

    html = client.get("/public/leagues/1/hky/games/1001").get_data(as_text=True)
    assert "Imported Player Stats" in html
    assert "Average Shift" not in html
    assert "Shifts" not in html
    assert "TOI Total" not in html


def should_store_game_video_url_via_shift_package_and_show_link_in_schedule(client_and_models):
    client, m = client_and_models
    r = client.post(
        "/api/import/hockey/shift_package",
        json={"timetoscore_game_id": 123, "game_video_url": "https://example.com/video", "source_label": "unit-test"},
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    notes = str(m.HkyGame.objects.filter(id=1001).values_list("notes", flat=True).first() or "")
    assert "game_video_url" in notes

    schedule_html = client.get("/public/leagues/1/schedule").get_data(as_text=True)
    assert 'href="https://example.com/video"' in schedule_html
    assert 'target="_blank"' in schedule_html


def should_create_external_game_via_shift_package_and_map_to_league(client_and_models):
    client, m = client_and_models
    player_stats_csv = "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n"
    game_stats_csv = "Stat,chicago-4\nGoals For,2\nGoals Against,1\n"
    events_csv = "Period,Time,Team,Event Type\n1,12:00,Home,Shot\n"

    r = client.post(
        "/api/import/hockey/shift_package",
        json={
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
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    out = r.get_json()
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


def should_render_private_game_page_as_league_owner_when_not_game_owner(client_and_models):
    client, m = client_and_models
    with client.session_transaction() as sess:
        sess["user_id"] = 10
        sess["user_email"] = "owner@example.com"
        sess["league_id"] = 1

    m.HkyGame.objects.filter(id=1001).update(user_id=11)
    r = client.get("/hky/games/1001?return_to=/teams/44")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
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

    r = client.post(
        "/api/import/hockey/shift_package",
        json={
            "external_game_key": "tourny-1",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Opponent X",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    out = r.get_json()
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
    r = client.post(
        "/api/import/hockey/shift_package",
        json={
            "external_game_key": "tourny-2",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "SAN JOSE JR. SHARKS 12AA–1",
            "away_team_name": "Opponent X",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    out = r.get_json()
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

    r1 = client.post(
        "/api/import/hockey/shift_package",
        json={
            "external_game_key": "tourny-a",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Arizona Coyotes 12AA",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r1.status_code == 200
    assert r1.get_json()["ok"] is True
    assert m.Team.objects.count() == before_team_count + 1

    r2 = client.post(
        "/api/import/hockey/shift_package",
        json={
            "external_game_key": "tourny-b",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "ARIZONA COYOTES 12AA",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True
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
    r = client.post(
        "/api/import/hockey/shift_package",
        json={
            "external_game_key": "tourny-parens",
            "owner_email": "owner@example.com",
            "league_name": "Norcal",
            "team_side": "home",
            "home_team_name": "Team A",
            "away_team_name": "Opponent X",
            "player_stats_csv": "Jersey #,Player,Goals,Assists\n13,Charlie,1,0\n",
        },
        headers={"X-HM-Import-Token": "sekret"},
    )
    assert r.status_code == 200
    out = r.get_json()
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


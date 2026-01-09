from __future__ import annotations

import importlib.util
import datetime as dt
from typing import Any

import pytest
from flask import render_template


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture()
def mod_and_app(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    now = dt.datetime.now()
    user = m.User.objects.create(email="u@example.com", password_hash="x", name="U", created_at=now)
    league = m.League.objects.create(
        id=42,
        name="L",
        owner_user_id=int(user.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    team_a = m.Team.objects.create(
        id=1,
        user_id=int(user.id),
        name="Team A",
        is_external=True,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=2,
        user_id=int(user.id),
        name="Team B",
        is_external=True,
        logo_path="",
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team_a.id), division_name="External")
    g = m.HkyGame.objects.create(
        id=123,
        user_id=int(user.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=None,
        location="",
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(g.id), division_name="External", sort_order=None)

    app = mod.create_app()
    app.testing = True
    return mod, app, int(user.id), int(league.id)


def should_default_external_division_game_type_to_tournament_in_schedule_table(mod_and_app):
    _mod, app, user_id, league_id = mod_and_app
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = int(user_id)
        sess["user_email"] = "u@example.com"
        sess["league_id"] = int(league_id)
    html = client.get("/schedule").get_data(as_text=True)
    assert ">Tournament<" in html


def should_default_external_division_game_type_to_tournament_in_team_schedule_table(mod_and_app):
    _mod, app, _user_id, _league_id = mod_and_app
    with app.test_request_context("/teams/1"):
        html = render_template(
            "team_detail.html",
            team={"id": 1, "name": "Team A", "logo_path": "", "is_external": 0},
            roster_players=[],
            players=[],
            head_coaches=[],
            assistant_coaches=[],
            player_stats_columns=[],
            player_stats_rows=[],
            recent_player_stats_columns=[],
            recent_player_stats_rows=[],
            recent_n=5,
            recent_sort="points",
            recent_dir="desc",
            tstats={"wins": 0, "losses": 0, "ties": 0, "gf": 0, "ga": 0, "points": 0},
            schedule_games=[
                {
                    "id": 123,
                    "team1_id": 1,
                    "team2_id": 2,
                    "team1_name": "Team A",
                    "team2_name": "Team B",
                    "division_name": "External",
                    "game_type_name": None,
                    "starts_at": None,
                    "location": "",
                    "team1_score": None,
                    "team2_score": None,
                }
            ],
            editable=False,
        )
    assert "<th>Type</th>" in html
    assert ">Tournament<" in html

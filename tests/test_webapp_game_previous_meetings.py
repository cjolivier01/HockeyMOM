from __future__ import annotations

import json
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


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "sekret")
    from django.test import Client

    return Client(), m


def should_render_previous_meetings_summary_for_private_and_public_game_pages(client_and_models):
    client, m = client_and_models

    payload = {
        "league_name": "CAHA",
        "shared": True,
        "replace": False,
        "owner_email": "owner@example.com",
        "source": "timetoscore",
        "external_key": "caha:season31",
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
                "season_id": 31,
                "division_name": "12AA",
                "game_type_name": "Regular Season",
            },
            {
                "home_name": "Home A",
                "away_name": "Away A",
                "starts_at": "2026-02-03 10:00:00",
                "location": "Rink 1",
                "home_score": 3,
                "away_score": 4,
                "is_final": True,
                "timetoscore_game_id": 124,
                "season_id": 31,
                "division_name": "12AA",
                "game_type_name": "Playoffs",
            },
        ],
    }

    r = _post_json(client, "/api/import/hockey/games_batch", payload)
    assert r.status_code == 200
    out = json.loads(r.content)
    assert out["ok"] is True
    league_id = int(out["league_id"])
    owner_user_id = int(out["owner_user_id"])
    gid1 = int(out["results"][0]["game_id"])
    gid2 = int(out["results"][1]["game_id"])
    assert gid1 != gid2

    _set_session(client, user_id=owner_user_id, email="owner@example.com", league_id=league_id)
    html = client.get(f"/hky/games/{gid2}?return_to=/schedule").content.decode()
    assert "Previous Meetings" in html
    assert "Regular Season" in html
    assert "2 - 1" in html

    m.League.objects.filter(id=int(league_id)).update(is_public=True)
    public_html = client.get(f"/public/leagues/{league_id}/hky/games/{gid2}").content.decode()
    assert "Previous Meetings" in public_html
    assert "Regular Season" in public_html
    assert "2 - 1" in public_html

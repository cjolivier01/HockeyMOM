from __future__ import annotations

import datetime as dt
import json

import pytest


def _set_session(client, *, user_id: int, email: str) -> None:
    sess = client.session
    sess["user_id"] = int(user_id)
    sess["user_email"] = str(email)
    sess.save()


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    from django.test import Client

    return Client(), m


def should_mark_league_page_view_baseline_and_show_delta(client_and_models):
    client_owner, m = client_and_models
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
    m.League.objects.create(
        id=1,
        name="CAHA",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )

    client_anon = Client()
    for _ in range(3):
        r = client_anon.get("/public/leagues/1/teams")
        assert r.status_code == 200

    _set_session(client_owner, user_id=int(owner.id), email=str(owner.email))
    r = client_owner.get("/api/leagues/1/page_views?kind=teams&entity_id=0")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["count"] == 3
    assert j.get("baseline_count") is None
    assert j.get("delta_count") is None

    r = client_owner.post("/api/leagues/1/page_views/mark", data={"kind": "teams", "entity_id": 0})
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["count"] == 3
    assert j["baseline_count"] == 3
    assert j["delta_count"] == 0

    for _ in range(2):
        r = client_anon.get("/public/leagues/1/teams")
        assert r.status_code == 200

    r = client_owner.get("/api/leagues/1/page_views?kind=teams&entity_id=0")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["count"] == 5
    assert j["baseline_count"] == 3
    assert j["delta_count"] == 2

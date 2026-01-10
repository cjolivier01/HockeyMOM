from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest


@pytest.fixture()
def mod_and_client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from django.test import Client

    from tools.webapp import app as mod

    now = dt.datetime.now()
    owner = m.User.objects.create(id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now)
    m.League.objects.create(
        id=1,
        name="L1",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )

    client = Client()
    return mod, client


def should_recalc_div_ratings_via_internal_api(monkeypatch, mod_and_client):
    mod, client = mod_and_client
    called: list[int] = []

    def _fake_recompute(db_conn, league_id: int, *args: Any, **kwargs: Any):
        called.append(int(league_id))
        return {}

    monkeypatch.setattr(mod, "recompute_league_mhr_ratings", _fake_recompute, raising=True)

    r = client.post(
        "/api/internal/recalc_div_ratings",
        data=json.dumps({"league_name": "L1"}),
        content_type="application/json",
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["league_ids"] == [1]
    assert called == [1]


def should_require_import_token_for_internal_recalc_div_ratings_when_configured(monkeypatch, mod_and_client):
    _mod, client = mod_and_client
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "secret-token")

    r = client.post(
        "/api/internal/recalc_div_ratings",
        data=json.dumps({"league_name": "L1"}),
        content_type="application/json",
    )
    assert r.status_code == 401

    r2 = client.post(
        "/api/internal/recalc_div_ratings",
        data=json.dumps({"league_name": "L1"}),
        content_type="application/json",
        **{"HTTP_AUTHORIZATION": "Bearer secret-token"},
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True


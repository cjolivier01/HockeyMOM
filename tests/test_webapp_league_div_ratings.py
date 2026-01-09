from __future__ import annotations

import importlib.util
import datetime as dt
import os
from typing import Any

import pytest


def _load_app_module():
    os.environ.setdefault("HM_WEBAPP_SKIP_DB_INIT", "1")
    os.environ.setdefault("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture()
def mod_and_client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    mod = _load_app_module()

    now = dt.datetime.now()
    owner = m.User.objects.create(id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now)
    admin = m.User.objects.create(id=20, email="admin@example.com", password_hash="x", name="Admin", created_at=now)
    league = m.League.objects.create(
        id=1,
        name="L1",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueMember.objects.create(league_id=int(league.id), user_id=int(admin.id), role="admin", created_at=now)

    app = mod.create_app()
    app.testing = True
    client = app.test_client()
    return mod, client, m


def should_allow_league_admin_to_recalc_div_ratings_for_active_league(monkeypatch, mod_and_client):
    mod, client, _m = mod_and_client
    called: list[int] = []

    def _fake_recompute(db_conn, league_id: int, *args: Any, **kwargs: Any):
        called.append(int(league_id))
        return {}

    monkeypatch.setattr(mod, "recompute_league_mhr_ratings", _fake_recompute, raising=True)

    with client.session_transaction() as sess:
        sess["user_id"] = 20
        sess["user_email"] = "admin@example.com"
        sess["league_id"] = 1

    r = client.post("/leagues/recalc_div_ratings")
    assert r.status_code == 302
    assert called == [1]


def should_reject_recalc_when_not_admin(monkeypatch, mod_and_client):
    mod, _client, m = mod_and_client
    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    now = dt.datetime.now()
    m.User.objects.create(id=99, email="u@example.com", password_hash="x", name="U", created_at=now)

    monkeypatch.setattr(
        mod,
        "recompute_league_mhr_ratings",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
        raising=True,
    )

    with client.session_transaction() as sess:
        sess["user_id"] = 99
        sess["user_email"] = "u@example.com"
        sess["league_id"] = 1

    r = client.post("/leagues/recalc_div_ratings")
    assert r.status_code == 302

from __future__ import annotations

import importlib.util
import datetime as dt

import pytest


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture()
def client_and_db(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
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
    return app.test_client(), m


def should_allow_league_admin_to_update_shared_and_public(client_and_db):
    client, m = client_and_db
    with client.session_transaction() as sess:
        sess["user_id"] = 20
        sess["user_email"] = "admin@example.com"

    r = client.post("/leagues/1/update", data={"is_shared": "1", "is_public": "1"})
    assert r.status_code == 302
    row = m.League.objects.filter(id=1).values("is_shared", "is_public").first()
    assert row is not None
    assert bool(row["is_shared"]) is True
    assert bool(row["is_public"]) is True


def should_allow_league_admin_to_delete_league(client_and_db):
    client, m = client_and_db
    with client.session_transaction() as sess:
        sess["user_id"] = 20
        sess["user_email"] = "admin@example.com"

    r = client.post("/leagues/1/delete")
    assert r.status_code == 302
    assert not m.League.objects.filter(id=1).exists()

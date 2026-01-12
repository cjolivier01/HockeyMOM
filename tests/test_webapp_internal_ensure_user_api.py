from __future__ import annotations

import json

import pytest
from werkzeug.security import check_password_hash


@pytest.fixture()
def client_and_db(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from django.test import Client

    return Client(), m


def should_create_user_via_internal_ensure_user_api(client_and_db):
    client, m = client_and_db

    payload = {"email": "git@example.com", "name": "Git User", "password": "password"}
    r = client.post("/api/internal/ensure_user", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["created"] is True
    uid = int(out["user_id"])

    row = m.User.objects.filter(id=uid).values("email", "name", "password_hash").first()
    assert row is not None
    assert row["email"] == "git@example.com"
    assert row["name"] == "Git User"
    assert check_password_hash(str(row["password_hash"]), "password")

    # Idempotent: second call should not recreate.
    r2 = client.post("/api/internal/ensure_user", data=json.dumps(payload), content_type="application/json")
    assert r2.status_code == 200
    out2 = r2.json()
    assert out2["ok"] is True
    assert out2["created"] is False
    assert int(out2["user_id"]) == uid


def should_require_import_token_for_internal_ensure_user_when_configured(monkeypatch, client_and_db):
    client, _m = client_and_db
    monkeypatch.setenv("HM_WEBAPP_IMPORT_TOKEN", "secret-token")

    payload = {"email": "tok@example.com", "name": "Tok User", "password": "password"}
    r = client.post("/api/internal/ensure_user", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 401

    r2 = client.post(
        "/api/internal/ensure_user",
        data=json.dumps(payload),
        content_type="application/json",
        **{"HTTP_AUTHORIZATION": "Bearer secret-token"},
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True


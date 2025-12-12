import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hmwebapp.settings")
os.environ.setdefault("HM_WEBAPP_SKIP_DB_INIT", "1")

import django

django.setup()

from django.test import Client  # type: ignore


def should_return_health_json_and_db_flag():
    client = Client()
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("status") in {"ok", "degraded"}
    assert "db" in data


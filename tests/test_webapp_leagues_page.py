from __future__ import annotations

import datetime as dt

import pytest


@pytest.fixture()
def client(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    m.League.objects.create(
        id=1,
        name="L1",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    return Client()


def should_render_leagues_page_for_logged_in_user(client):
    sess = client.session
    sess["user_id"] = 10
    sess["user_email"] = "owner@example.com"
    sess["league_id"] = 1
    sess.save()

    r = client.get("/leagues")
    assert r.status_code == 200
    html = r.content.decode()
    assert "Leagues" in html
    assert "Your Leagues" in html
    assert "L1" in html
    assert 'name="show_goalie_stats"' in html

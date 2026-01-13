from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


_ensure_repo_root_on_path()


@pytest.fixture(scope="session")
def webapp_test_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Create a sqlite-backed config.json usable by tools/webapp's Django ORM layer.

    Django settings are global for the Python process, so tests should share a single config.
    """
    root = tmp_path_factory.mktemp("hm_webapp")
    db_path = root / "hm_webapp.sqlite3"
    cfg_path = root / "config.json"
    cfg_path.write_text(
        json.dumps({"db": {"engine": "sqlite3", "name": str(db_path)}}), encoding="utf-8"
    )

    os.environ.setdefault("HM_DB_CONFIG", str(cfg_path))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tools.webapp.django_settings")
    os.environ.setdefault("HM_WEBAPP_SECRET", "hm-webapp-test-secret")
    return cfg_path


@pytest.fixture(scope="session")
def webapp_orm_modules(webapp_test_config_path: Path):
    from werkzeug.security import generate_password_hash

    from tools.webapp import django_orm

    django_orm.setup_django(config_path=str(webapp_test_config_path))
    django_orm.ensure_schema()
    django_orm.ensure_bootstrap_data(default_admin_password_hash=generate_password_hash("admin"))

    from tools.webapp.django_app import models as m

    return django_orm, m


def _reset_webapp_db(django_orm, m) -> None:
    from django.contrib.sessions.models import Session
    from django.db import connection, transaction
    from werkzeug.security import generate_password_hash

    with transaction.atomic():
        m.LeagueGame.objects.all().delete()
        m.LeagueTeam.objects.all().delete()
        m.LeaguePageView.objects.all().delete()
        m.LeagueMember.objects.all().delete()
        m.PlayerPeriodStat.objects.all().delete()
        m.PlayerStat.objects.all().delete()
        m.HkyGameEventSuppression.objects.all().delete()
        m.HkyGamePlayer.objects.all().delete()
        m.HkyGameEventRow.objects.all().delete()
        m.HkyGame.objects.all().delete()
        m.Player.objects.all().delete()
        m.Team.objects.all().delete()
        m.League.objects.all().delete()
        m.Game.objects.all().delete()
        m.Job.objects.all().delete()
        m.Reset.objects.all().delete()
        m.GameType.objects.all().delete()
        m.User.objects.all().delete()
        Session.objects.all().delete()
        if connection.vendor == "sqlite":
            try:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM sqlite_sequence;")
            except Exception:
                pass

    django_orm.ensure_bootstrap_data(default_admin_password_hash=generate_password_hash("admin"))


@pytest.fixture()
def webapp_db(webapp_orm_modules):
    django_orm, m = webapp_orm_modules
    _reset_webapp_db(django_orm, m)
    return django_orm, m


@pytest.fixture()
def webapp_db_reset(webapp_orm_modules):
    django_orm, m = webapp_orm_modules

    def _reset() -> None:
        _reset_webapp_db(django_orm, m)

    return _reset

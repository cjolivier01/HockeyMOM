from __future__ import annotations

import datetime as dt
import os
from typing import Iterable, Optional, Type


def setup_django(*, config_path: Optional[str] = None) -> None:
    if config_path:
        os.environ["HM_DB_CONFIG"] = config_path

    settings_module = "tools.webapp.django_settings" if __name__.startswith("tools.webapp.") else "django_settings"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)

    try:
        import pymysql  # type: ignore

        pymysql.install_as_MySQLdb()
        version_info = getattr(pymysql, "version_info", (0, 0, 0))
        if tuple(version_info[:3]) < (2, 2, 1):
            pymysql.version_info = (2, 2, 1, "final", 0)
            pymysql.__version__ = "2.2.1"
    except Exception:
        pass

    import django
    from django.conf import settings

    if not settings.configured:
        django.setup()
        return

    # Idempotent: calling setup() multiple times is safe (it no-ops once configured).
    django.setup()


def _import_models():
    if __name__.startswith("tools.webapp."):
        try:
            from tools.webapp.django_app import models as m  # type: ignore

            return m
        except Exception:
            from django_app import models as m  # type: ignore

            return m

    try:
        from django_app import models as m  # type: ignore

        return m
    except Exception:
        from tools.webapp.django_app import models as m  # type: ignore

        return m


def _all_models() -> list[Type]:
    m = _import_models()

    return [
        m.User,
        m.League,
        m.LeagueMember,
        m.LeaguePageView,
        m.Game,
        m.Job,
        m.Reset,
        m.Team,
        m.Player,
        m.GameType,
        m.HkyGame,
        m.PlayerStat,
        m.PlayerPeriodStat,
        m.HkyGameStat,
        m.HkyGameEvent,
        m.HkyGamePlayerStatsCsv,
        m.LeagueTeam,
        m.LeagueGame,
    ]


def ensure_schema() -> None:
    setup_django()

    from django.db import connection

    existing_tables = set(connection.introspection.table_names())
    models_to_create = [m for m in _all_models() if m._meta.db_table not in existing_tables]

    if models_to_create:
        with connection.schema_editor() as schema_editor:
            for model in models_to_create:
                schema_editor.create_model(model)

    # Best-effort: add missing columns for older installs.
    for model in _all_models():
        table = model._meta.db_table
        if table not in existing_tables and table not in set(connection.introspection.table_names()):
            continue

        with connection.cursor() as cursor:
            desc = connection.introspection.get_table_description(cursor, table)
        existing_cols = {c.name for c in desc or []}
        missing_fields = [f for f in model._meta.local_fields if getattr(f, "column", None) not in existing_cols]
        if not missing_fields:
            continue

        with connection.schema_editor() as schema_editor:
            for field in missing_fields:
                schema_editor.add_field(model, field)


def ensure_bootstrap_data(*, default_admin_password_hash: Optional[str] = None) -> None:
    setup_django()

    m = _import_models()
    GameType = m.GameType
    User = m.User

    if default_admin_password_hash:
        User.objects.get_or_create(
            email="admin",
            defaults={
                "password_hash": default_admin_password_hash,
                "name": "admin",
                "created_at": dt.datetime.now(),
            },
        )

    if not GameType.objects.exists():
        GameType.objects.bulk_create(
            [
                GameType(name="Preseason", is_default=True),
                GameType(name="Regular Season", is_default=True),
                GameType(name="Tournament", is_default=True),
                GameType(name="Exhibition", is_default=True),
            ]
        )


def close_connections() -> None:
    try:
        from django.db import close_old_connections

        close_old_connections()
    except Exception:
        return


def iter_chunks(seq: Iterable[int], n: int) -> Iterable[list[int]]:
    chunk: list[int] = []
    for item in seq:
        chunk.append(int(item))
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

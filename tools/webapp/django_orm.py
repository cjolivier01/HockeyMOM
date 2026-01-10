from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Iterable, Optional, Type


def setup_django(*, config_path: Optional[str] = None) -> None:
    if config_path:
        os.environ["HM_DB_CONFIG"] = config_path

    if "DJANGO_SETTINGS_MODULE" not in os.environ:
        if __name__.startswith("tools.webapp."):
            settings_module = "tools.webapp.hm_webapp.settings"
            legacy_module = "tools.webapp.django_settings"
        else:
            settings_module = "hm_webapp.settings"
            legacy_module = "django_settings"

        try:
            __import__(settings_module)
        except Exception:
            settings_module = legacy_module

        os.environ["DJANGO_SETTINGS_MODULE"] = settings_module

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
    try:
        from django.contrib.sessions.models import Session
    except Exception:  # pragma: no cover
        Session = None  # type: ignore

    models: list[Type] = [
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
    if Session is not None:
        models.append(Session)
    return models


def _extract_timetoscore_game_id_from_notes(notes: Optional[str]) -> Optional[int]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            v = d.get("timetoscore_game_id")
            if v is not None:
                return int(v)
    except Exception:
        # If notes are not valid JSON (or not a dict), fall back to regex-based extraction below.
        pass
    m1 = re.search(r"(?:^|[\s,;|])game_id\s*=\s*(\d+)", s, flags=re.IGNORECASE)
    if m1:
        try:
            return int(m1.group(1))
        except Exception:
            return None
    m2 = re.search(r'"timetoscore_game_id"\s*:\s*(\d+)', s)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def _extract_external_game_key_from_notes(notes: Optional[str]) -> Optional[str]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            v = d.get("external_game_key")
            if v is not None:
                out = str(v).strip()
                return out or None
    except Exception:
        # If notes are not valid JSON (or not a dict), fall back to regex-based extraction below.
        pass
    m1 = re.search(r'"external_game_key"\s*:\s*"([^"]+)"', s)
    if m1:
        out = str(m1.group(1)).strip()
        return out or None
    return None


def _merge_notes_text(keep: Optional[str], drop: Optional[str]) -> Optional[str]:
    keep_s = str(keep or "").strip()
    drop_s = str(drop or "").strip()
    if not keep_s:
        return drop_s or None
    if not drop_s:
        return keep_s or None
    try:
        a = json.loads(keep_s)
        b = json.loads(drop_s)
        if isinstance(a, dict) and isinstance(b, dict):
            for k, v in b.items():
                if k not in a or a.get(k) in (None, ""):
                    a[k] = v
            return json.dumps(a, sort_keys=True)
    except Exception:
        # If JSON parsing/merging fails, fall back to simple string merging below.
        pass
    if drop_s in keep_s:
        return keep_s
    return (keep_s + "\n" + drop_s).strip() or None


def merge_hky_games(*, keep_id: int, drop_id: int) -> None:
    setup_django()
    m = _import_models()

    if int(keep_id) == int(drop_id):
        return

    from django.db import transaction

    with transaction.atomic():
        keep = (
            m.HkyGame.objects.filter(id=int(keep_id))
            .values(
                "id",
                "notes",
                "timetoscore_game_id",
                "external_game_key",
                "starts_at",
                "location",
                "team1_score",
                "team2_score",
                "is_final",
                "stats_imported_at",
                "game_type_id",
            )
            .first()
        )
        drop = (
            m.HkyGame.objects.filter(id=int(drop_id))
            .values(
                "id",
                "notes",
                "timetoscore_game_id",
                "external_game_key",
                "starts_at",
                "location",
                "team1_score",
                "team2_score",
                "is_final",
                "stats_imported_at",
                "game_type_id",
            )
            .first()
        )
        if not keep or not drop:
            return

        keep_has_ps = set(
            m.PlayerStat.objects.filter(game_id=int(keep_id)).values_list("player_id", flat=True)
        )
        m.PlayerStat.objects.filter(game_id=int(drop_id), player_id__in=keep_has_ps).delete()
        m.PlayerStat.objects.filter(game_id=int(drop_id)).update(game_id=int(keep_id))

        keep_has_pps = set(
            m.PlayerPeriodStat.objects.filter(game_id=int(keep_id)).values_list(
                "player_id", "period"
            )
        )
        for pid, period in keep_has_pps:
            m.PlayerPeriodStat.objects.filter(
                game_id=int(drop_id), player_id=int(pid), period=int(period)
            ).delete()
        m.PlayerPeriodStat.objects.filter(game_id=int(drop_id)).update(game_id=int(keep_id))

        if m.HkyGameStat.objects.filter(game_id=int(keep_id)).exists():
            m.HkyGameStat.objects.filter(game_id=int(drop_id)).delete()
        else:
            m.HkyGameStat.objects.filter(game_id=int(drop_id)).update(game_id=int(keep_id))

        if m.HkyGameEvent.objects.filter(game_id=int(keep_id)).exists():
            m.HkyGameEvent.objects.filter(game_id=int(drop_id)).delete()
        else:
            m.HkyGameEvent.objects.filter(game_id=int(drop_id)).update(game_id=int(keep_id))

        if m.HkyGamePlayerStatsCsv.objects.filter(game_id=int(keep_id)).exists():
            m.HkyGamePlayerStatsCsv.objects.filter(game_id=int(drop_id)).delete()
        else:
            m.HkyGamePlayerStatsCsv.objects.filter(game_id=int(drop_id)).update(
                game_id=int(keep_id)
            )

        keep_leagues = set(
            m.LeagueGame.objects.filter(game_id=int(keep_id)).values_list("league_id", flat=True)
        )
        m.LeagueGame.objects.filter(game_id=int(drop_id), league_id__in=keep_leagues).delete()
        m.LeagueGame.objects.filter(game_id=int(drop_id)).update(game_id=int(keep_id))

        updates: dict[str, object] = {}
        if keep.get("timetoscore_game_id") is None and drop.get("timetoscore_game_id") is not None:
            updates["timetoscore_game_id"] = drop.get("timetoscore_game_id")
        if not keep.get("external_game_key") and drop.get("external_game_key"):
            updates["external_game_key"] = drop.get("external_game_key")
        if keep.get("starts_at") is None and drop.get("starts_at") is not None:
            updates["starts_at"] = drop.get("starts_at")
        if keep.get("location") is None and drop.get("location") is not None:
            updates["location"] = drop.get("location")
        if keep.get("team1_score") is None and drop.get("team1_score") is not None:
            updates["team1_score"] = drop.get("team1_score")
        if keep.get("team2_score") is None and drop.get("team2_score") is not None:
            updates["team2_score"] = drop.get("team2_score")
        if not bool(keep.get("is_final")) and bool(drop.get("is_final")):
            updates["is_final"] = True
        if keep.get("game_type_id") is None and drop.get("game_type_id") is not None:
            updates["game_type_id"] = drop.get("game_type_id")
        if keep.get("stats_imported_at") is None and drop.get("stats_imported_at") is not None:
            updates["stats_imported_at"] = drop.get("stats_imported_at")
        if keep.get("stats_imported_at") is not None and drop.get("stats_imported_at") is not None:
            updates["stats_imported_at"] = max(keep["stats_imported_at"], drop["stats_imported_at"])  # type: ignore[assignment]

        merged = _merge_notes_text(keep.get("notes"), drop.get("notes"))
        if merged is not None and merged != str(keep.get("notes") or "").strip():
            updates["notes"] = merged

        # Delete the duplicate first so UNIQUE constraints on the key fields don't block updating the kept row.
        m.HkyGame.objects.filter(id=int(drop_id)).delete()
        if updates:
            m.HkyGame.objects.filter(id=int(keep_id)).update(
                **updates, updated_at=dt.datetime.now()
            )


def backfill_hky_game_keys() -> None:
    setup_django()
    m = _import_models()

    for row in list(
        m.HkyGame.objects.filter(timetoscore_game_id__isnull=True)
        .exclude(notes__isnull=True)
        .exclude(notes="")
        .values("id", "notes")
    ):
        gid = int(row["id"])
        tts_id = _extract_timetoscore_game_id_from_notes(row.get("notes"))
        if tts_id is not None:
            m.HkyGame.objects.filter(id=int(gid), timetoscore_game_id__isnull=True).update(
                timetoscore_game_id=int(tts_id)
            )

    for row in list(
        m.HkyGame.objects.filter(external_game_key__isnull=True)
        .exclude(notes__isnull=True)
        .exclude(notes="")
        .values("id", "notes")
    ):
        gid = int(row["id"])
        ext = _extract_external_game_key_from_notes(row.get("notes"))
        if ext:
            m.HkyGame.objects.filter(id=int(gid), external_game_key__isnull=True).update(
                external_game_key=str(ext)
            )


def dedupe_hky_games() -> None:
    setup_django()
    m = _import_models()

    from django.db import transaction
    from django.db.models import Count

    with transaction.atomic():
        tts_dups = (
            m.HkyGame.objects.filter(timetoscore_game_id__isnull=False)
            .values("timetoscore_game_id")
            .annotate(n=Count("id"))
            .filter(n__gt=1)
        )
        for row in list(tts_dups):
            tts_id = row.get("timetoscore_game_id")
            if tts_id is None:
                continue
            ids = list(
                m.HkyGame.objects.filter(timetoscore_game_id=int(tts_id))
                .values_list("id", flat=True)
                .order_by("id")
            )
            if len(ids) < 2:
                continue
            keep_id = int(ids[0])
            for drop_id in ids[1:]:
                merge_hky_games(keep_id=keep_id, drop_id=int(drop_id))

        ext_dups = (
            m.HkyGame.objects.filter(external_game_key__isnull=False)
            .exclude(external_game_key="")
            .values("user_id", "external_game_key")
            .annotate(n=Count("id"))
            .filter(n__gt=1)
        )
        for row in list(ext_dups):
            uid = row.get("user_id")
            ek = str(row.get("external_game_key") or "").strip()
            if uid is None or not ek:
                continue
            ids = list(
                m.HkyGame.objects.filter(user_id=int(uid), external_game_key=str(ek))
                .values_list("id", flat=True)
                .order_by("id")
            )
            if len(ids) < 2:
                continue
            keep_id = int(ids[0])
            for drop_id in ids[1:]:
                merge_hky_games(keep_id=keep_id, drop_id=int(drop_id))


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
        if table not in existing_tables and table not in set(
            connection.introspection.table_names()
        ):
            continue

        with connection.cursor() as cursor:
            desc = connection.introspection.get_table_description(cursor, table)
        existing_cols = {c.name for c in desc or []}
        missing_fields = [
            f for f in model._meta.local_fields if getattr(f, "column", None) not in existing_cols
        ]
        if not missing_fields:
            continue

        with connection.schema_editor() as schema_editor:
            for field in missing_fields:
                schema_editor.add_field(model, field)

    def _ensure_constraints(model: Type, *, only_names: Optional[set[str]] = None) -> None:
        table = model._meta.db_table
        with connection.cursor() as cursor:
            existing = connection.introspection.get_constraints(cursor, table)
        existing_names = set(existing.keys())
        wanted = []
        for c in getattr(model._meta, "constraints", []) or []:
            if only_names is not None and getattr(c, "name", None) not in only_names:
                continue
            if getattr(c, "name", None) and c.name not in existing_names:
                wanted.append(c)
        if not wanted:
            return
        with connection.schema_editor() as schema_editor:
            for c in wanted:
                schema_editor.add_constraint(model, c)

    # Backfill/dedupe before adding constraints so we can enforce uniqueness without breaking existing installs.
    try:
        backfill_hky_game_keys()
        dedupe_hky_games()
        m = _import_models()
        _ensure_constraints(m.HkyGame, only_names={"uniq_hky_tts_id", "uniq_hky_user_ext_key"})
    except Exception:
        raise


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

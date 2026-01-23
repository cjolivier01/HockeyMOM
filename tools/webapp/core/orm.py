import datetime as dt
from functools import lru_cache
from typing import Optional

from werkzeug.security import generate_password_hash

LEAGUE_PAGE_VIEW_KIND_TEAMS = "teams"
LEAGUE_PAGE_VIEW_KIND_SCHEDULE = "schedule"
LEAGUE_PAGE_VIEW_KIND_TEAM = "team"
LEAGUE_PAGE_VIEW_KIND_GAME = "game"

LEAGUE_PAGE_VIEW_KINDS: set[str] = {
    LEAGUE_PAGE_VIEW_KIND_TEAMS,
    LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
    LEAGUE_PAGE_VIEW_KIND_TEAM,
    LEAGUE_PAGE_VIEW_KIND_GAME,
}


@lru_cache(maxsize=1)
def _orm_modules():
    try:
        from tools.webapp import django_orm  # type: ignore
    except Exception:  # pragma: no cover
        import django_orm  # type: ignore

    django_orm.setup_django()

    try:
        from tools.webapp.django_app import models as m  # type: ignore
    except Exception:  # pragma: no cover
        from django_app import models as m  # type: ignore

    return django_orm, m


def _get_league_owner_user_id(db_conn, league_id: int) -> Optional[int]:
    del db_conn
    try:
        _django_orm, m = _orm_modules()
        owner_id = (
            m.League.objects.filter(id=int(league_id))
            .values_list("owner_user_id", flat=True)
            .first()
        )
        return int(owner_id) if owner_id is not None else None
    except Exception:
        return None


def _get_league_name(db_conn, league_id: int) -> Optional[str]:
    del db_conn
    try:
        _django_orm, m = _orm_modules()
        name = m.League.objects.filter(id=int(league_id)).values_list("name", flat=True).first()
        s = str(name or "").strip()
        return s or None
    except Exception:
        return None


def _get_league_page_view_count(db_conn, league_id: int, *, kind: str, entity_id: int = 0) -> int:
    kind_s = str(kind or "").strip()
    if kind_s not in LEAGUE_PAGE_VIEW_KINDS:
        raise ValueError(f"Unsupported league page view kind: {kind_s}")
    eid = int(entity_id or 0)
    if kind_s in {LEAGUE_PAGE_VIEW_KIND_TEAM, LEAGUE_PAGE_VIEW_KIND_GAME} and eid <= 0:
        raise ValueError(f"entity_id is required for kind={kind_s}")
    if kind_s in {LEAGUE_PAGE_VIEW_KIND_TEAMS, LEAGUE_PAGE_VIEW_KIND_SCHEDULE}:
        eid = 0
    del db_conn
    try:
        _django_orm, m = _orm_modules()
        v = (
            m.LeaguePageView.objects.filter(
                league_id=int(league_id), page_kind=kind_s, entity_id=int(eid)
            )
            .values_list("view_count", flat=True)
            .first()
        )
        return int(v or 0)
    except Exception:
        return 0


def _canon_league_page_view_kind_entity(*, kind: str, entity_id: int = 0) -> tuple[str, int]:
    kind_s = str(kind or "").strip()
    if kind_s not in LEAGUE_PAGE_VIEW_KINDS:
        raise ValueError(f"Unsupported league page view kind: {kind_s}")
    eid = int(entity_id or 0)
    if kind_s in {LEAGUE_PAGE_VIEW_KIND_TEAM, LEAGUE_PAGE_VIEW_KIND_GAME} and eid <= 0:
        raise ValueError(f"entity_id is required for kind={kind_s}")
    if kind_s in {LEAGUE_PAGE_VIEW_KIND_TEAMS, LEAGUE_PAGE_VIEW_KIND_SCHEDULE}:
        eid = 0
    return kind_s, eid


def _get_league_page_view_baseline_count(
    db_conn, league_id: int, *, kind: str, entity_id: int = 0
) -> Optional[int]:
    try:
        kind_s, eid = _canon_league_page_view_kind_entity(kind=kind, entity_id=int(entity_id or 0))
    except Exception:
        return None
    del db_conn
    try:
        _django_orm, m = _orm_modules()
        v = (
            m.LeaguePageViewBaseline.objects.filter(
                league_id=int(league_id), page_kind=kind_s, entity_id=int(eid)
            )
            .values_list("baseline_count", flat=True)
            .first()
        )
        return int(v) if v is not None else None
    except Exception:
        return None


def _set_league_page_view_baseline_count(
    db_conn, league_id: int, *, kind: str, entity_id: int = 0, baseline_count: int
) -> bool:
    kind_s, eid = _canon_league_page_view_kind_entity(kind=kind, entity_id=int(entity_id or 0))
    count_i = int(baseline_count or 0)
    del db_conn
    try:
        _django_orm, m = _orm_modules()
        from django.db import IntegrityError, transaction

        now = dt.datetime.now()
        with transaction.atomic():
            updated = m.LeaguePageViewBaseline.objects.filter(
                league_id=int(league_id), page_kind=kind_s, entity_id=int(eid)
            ).update(baseline_count=int(count_i), updated_at=now)
            if updated:
                return
            try:
                m.LeaguePageViewBaseline.objects.create(
                    league_id=int(league_id),
                    page_kind=kind_s,
                    entity_id=int(eid),
                    baseline_count=int(count_i),
                    created_at=now,
                    updated_at=now,
                )
            except IntegrityError:
                m.LeaguePageViewBaseline.objects.filter(
                    league_id=int(league_id), page_kind=kind_s, entity_id=int(eid)
                ).update(baseline_count=int(count_i), updated_at=now)
        return True
    except Exception:
        return False


def _record_league_page_view(
    db_conn,
    league_id: int,
    *,
    kind: str,
    entity_id: int = 0,
    viewer_user_id: Optional[int] = None,
    league_owner_user_id: Optional[int] = None,
) -> None:
    kind_s = str(kind or "").strip()
    if kind_s not in LEAGUE_PAGE_VIEW_KINDS:
        return
    eid = int(entity_id or 0)
    if kind_s in {LEAGUE_PAGE_VIEW_KIND_TEAM, LEAGUE_PAGE_VIEW_KIND_GAME} and eid <= 0:
        return
    if kind_s in {LEAGUE_PAGE_VIEW_KIND_TEAMS, LEAGUE_PAGE_VIEW_KIND_SCHEDULE}:
        eid = 0

    owner_id = league_owner_user_id
    if owner_id is None:
        owner_id = _get_league_owner_user_id(db_conn, int(league_id))
    if viewer_user_id is not None and owner_id is not None and int(viewer_user_id) == int(owner_id):
        return

    del db_conn
    try:
        _django_orm, m = _orm_modules()
        from django.db.models import F
        from django.db.utils import IntegrityError

        now = dt.datetime.now()
        updated = m.LeaguePageView.objects.filter(
            league_id=int(league_id), page_kind=kind_s, entity_id=int(eid)
        ).update(view_count=F("view_count") + 1, updated_at=now)
        if updated:
            return
        try:
            m.LeaguePageView.objects.create(
                league_id=int(league_id),
                page_kind=kind_s,
                entity_id=int(eid),
                view_count=1,
                created_at=now,
                updated_at=now,
            )
        except IntegrityError:
            m.LeaguePageView.objects.filter(
                league_id=int(league_id), page_kind=kind_s, entity_id=int(eid)
            ).update(view_count=F("view_count") + 1, updated_at=now)
    except Exception:
        return


def init_db():
    try:
        from tools.webapp import django_orm
    except Exception:  # pragma: no cover
        import django_orm  # type: ignore

    django_orm.ensure_schema()
    django_orm.ensure_bootstrap_data(default_admin_password_hash=generate_password_hash("admin"))

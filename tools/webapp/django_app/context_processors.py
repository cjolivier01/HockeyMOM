from __future__ import annotations

import os
from pathlib import Path
from typing import Any


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


def user_leagues(request) -> dict[str, Any]:
    leagues: list[dict[str, Any]] = []
    selected = request.session.get("league_id")

    user_id = request.session.get("user_id")
    if not user_id:
        return {"user_leagues": leagues, "selected_league_id": selected}

    try:
        _django_orm, m = _orm_modules()
        from django.db.models import Q

        uid = int(user_id)
        admin_ids = set(
            m.LeagueMember.objects.filter(user_id=uid, role__in=["admin", "owner"]).values_list(
                "league_id", flat=True
            )
        )
        for row in (
            m.League.objects.filter(
                Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid)
            )
            .distinct()
            .order_by("name")
            .values("id", "name", "is_shared", "is_public", "owner_user_id")
        ):
            lid = int(row["id"])
            is_owner = int(int(row["owner_user_id"]) == uid)
            is_admin = 1 if is_owner or lid in admin_ids else 0
            leagues.append(
                {
                    "id": lid,
                    "name": row["name"],
                    "is_shared": bool(row["is_shared"]),
                    "is_public": bool(row.get("is_public")),
                    "owner_user_id": int(row["owner_user_id"]),
                    "is_owner": is_owner,
                    "is_admin": is_admin,
                }
            )
    except Exception:
        leagues = []

    return {"user_leagues": leagues, "selected_league_id": selected}


def static_versions(_request) -> dict[str, Any]:
    """
    Cache-busting query params for static assets.
    """
    try:
        base = Path(__file__).resolve().parents[1]  # tools/webapp/
        css_path = base / "static" / "styles.css"
        v = int(os.path.getmtime(str(css_path)))
    except Exception:
        v = 0
    return {"styles_css_version": v}

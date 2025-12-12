from __future__ import annotations

from typing import Any, Dict

from django.db.models import Q

from .models import League, LeagueMember


def user_leagues(request) -> Dict[str, Any]:
    leagues: list[dict[str, Any]] = []
    selected = request.session.get("league_id")
    uid = request.session.get("user_id")
    if uid:
        qs = (
            League.objects.filter(
                Q(owner_user_id=uid) | Q(memberships__user_id=uid),
            )
            .distinct()
            .order_by("name")
        )
        for l in qs:
            is_owner = int(l.owner_user_id) == int(uid)
            is_admin = is_owner or LeagueMember.objects.filter(
                league=l, user_id=uid, role__in=["admin", "owner"]
            ).exists()
            leagues.append(
                {
                    "id": l.id,
                    "name": l.name,
                    "is_shared": l.is_shared,
                    "is_owner": is_owner,
                    "is_admin": is_admin,
                }
            )
    return {"user_leagues": leagues, "selected_league_id": selected, "session": request.session}


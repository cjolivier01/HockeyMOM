from __future__ import annotations

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


class LeagueSessionMiddleware:
    def __init__(self, get_response) -> None:
        self.get_response = get_response

    def __call__(self, request):
        self._close_old_connections()
        try:
            self._ensure_league_session(request)
        except Exception:
            # Non-fatal: never take down the request on session cleanup.
            pass
        return self.get_response(request)

    @staticmethod
    def _close_old_connections() -> None:
        try:
            from django.db import close_old_connections

            close_old_connections()
        except Exception:
            return

    @staticmethod
    def _ensure_league_session(request) -> None:
        user_id = request.session.get("user_id")
        if not user_id:
            return
        uid = int(user_id)

        _django_orm, m = _orm_modules()
        from django.db.models import Q

        def _has_access(league_id: int) -> bool:
            lid = int(league_id)
            return (
                m.League.objects.filter(id=lid)
                .filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
                .exists()
            )

        sid = request.session.get("league_id")
        if sid is not None:
            try:
                lid = int(sid)
            except Exception:
                request.session.pop("league_id", None)
                return

            if _has_access(lid):
                return

            request.session.pop("league_id", None)
            m.User.objects.filter(id=uid, default_league_id=lid).update(default_league=None)
            return

        pref = m.User.objects.filter(id=uid).values_list("default_league_id", flat=True).first()
        if pref is None:
            return
        try:
            pref_i = int(pref)
        except Exception:
            return
        if _has_access(pref_i):
            request.session["league_id"] = pref_i
            return
        m.User.objects.filter(id=uid, default_league_id=pref_i).update(default_league=None)


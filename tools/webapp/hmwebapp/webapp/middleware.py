from __future__ import annotations

from typing import Callable

from django.http import HttpRequest, HttpResponse

from .models import LeagueMember, User


class LeagueSelectionMiddleware:
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        uid = request.session.get("user_id")
        if uid:
            try:
                user = User.objects.get(pk=uid)
            except User.DoesNotExist:
                request.session.flush()
            else:
                league_id = request.session.get("league_id")

                def _has_access(lid: int) -> bool:
                    return LeagueMember.objects.filter(league_id=lid, user_id=uid).exists() or (
                        user.owned_leagues.filter(id=lid).exists()
                    )

                if league_id is not None:
                    try:
                        lid_int = int(league_id)
                    except Exception:
                        request.session.pop("league_id", None)
                    else:
                        if not _has_access(lid_int):
                            request.session.pop("league_id", None)
                            if user.default_league_id == lid_int:
                                user.default_league_id = None
                                user.save(update_fields=["default_league_id"])
                else:
                    pref = getattr(user, "default_league_id", None)
                    if pref:
                        try:
                            lid_int = int(pref)
                        except Exception:
                            pass
                        else:
                            if _has_access(lid_int):
                                request.session["league_id"] = lid_int
                            else:
                                user.default_league_id = None
                                user.save(update_fields=["default_league_id"])
        response = self.get_response(request)
        return response


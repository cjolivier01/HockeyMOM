from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Optional

from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from werkzeug.security import check_password_hash, generate_password_hash


def _import_logic():
    try:
        from tools.webapp import app as logic  # type: ignore

        return logic
    except Exception:  # pragma: no cover
        import app as logic  # type: ignore

        return logic


logic = _import_logic()


def _orm_modules():
    return logic._orm_modules()


def _session_user_id(request: HttpRequest) -> int:
    try:
        return int(request.session.get("user_id") or 0)
    except Exception:
        return 0


def _require_login(request: HttpRequest) -> Optional[HttpResponse]:
    if not _session_user_id(request):
        return redirect("/login")
    return None


def _is_league_admin(league_id: int, user_id: int) -> bool:
    _django_orm, m = _orm_modules()
    if m.League.objects.filter(id=int(league_id), owner_user_id=int(user_id)).exists():
        return True
    return m.LeagueMember.objects.filter(
        league_id=int(league_id),
        user_id=int(user_id),
        role__in=["admin", "owner"],
    ).exists()


def _is_public_league(league_id: int) -> Optional[dict[str, Any]]:
    _django_orm, m = _orm_modules()
    return (
        m.League.objects.filter(id=int(league_id), is_public=True)
        .values("id", "name", "owner_user_id")
        .first()
    )


def _safe_file_response(path: Path, *, as_attachment: bool = False) -> FileResponse:
    if not path.exists() or not path.is_file():
        raise Http404
    resp = FileResponse(path.open("rb"))
    if as_attachment:
        resp["Content-Disposition"] = f'attachment; filename="{path.name}"'
    return resp


def _json_body(request: HttpRequest) -> dict[str, Any]:
    try:
        raw = request.body or b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


# ----------------------------
# Landing/auth
# ----------------------------


def index(request: HttpRequest) -> HttpResponse:
    if _session_user_id(request):
        return redirect("/games")
    return render(request, "index.html")


def register(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        password = str(request.POST.get("password") or "")
        name = str(request.POST.get("name") or "").strip()
        if not email or not password:
            messages.error(request, "Email and password are required")
        elif logic.get_user_by_email(email):
            messages.error(request, "Email already registered")
        else:
            uid = logic.create_user(email, password, name)
            request.session["user_id"] = int(uid)
            request.session["user_email"] = email
            request.session["user_name"] = name
            try:
                logic.send_email(
                    to_addr=email,
                    subject="Welcome to HockeyMOM",
                    body=(
                        f"Hello {name or email},\n\n"
                        "Your account has been created successfully.\n"
                        "You can now create a game, upload files, and run jobs.\n\n"
                        "Regards,\nHockeyMOM"
                    ),
                )
            except Exception:
                pass
            return redirect("/games")
    return render(request, "register.html")


def login_view(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        password = str(request.POST.get("password") or "")
        u = logic.get_user_by_email(email)
        if not u or not check_password_hash(str(u.get("password_hash") or ""), password):
            messages.error(request, "Invalid credentials")
        else:
            request.session["user_id"] = int(u["id"])
            request.session["user_email"] = str(u["email"])
            request.session["user_name"] = str(u.get("name") or u["email"])
            return redirect("/games")
    return render(request, "login.html")


def forgot_password(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        try:
            _django_orm, m = _orm_modules()
            u = m.User.objects.filter(email=email).values("id").first()
            if u and u.get("id"):
                token = secrets.token_urlsafe(32)
                now = dt.datetime.now()
                m.Reset.objects.create(
                    user_id=int(u["id"]),
                    token=token,
                    expires_at=(now + dt.timedelta(hours=1)),
                    created_at=now,
                )
                link = request.build_absolute_uri(f"/reset/{token}")
                logic.send_email(
                    to_addr=email,
                    subject="Password reset",
                    body=(
                        "We received a request to reset your password.\n\n"
                        f"Use this link within 1 hour: {link}\n\n"
                        "If you did not request this, you can ignore this message."
                    ),
                )
        except Exception:
            pass
        messages.success(request, "If the account exists, a reset email has been sent.")
        return redirect("/login")
    return render(request, "forgot_password.html")


def reset_password(request: HttpRequest, token: str) -> HttpResponse:
    from django.contrib import messages

    _django_orm, m = _orm_modules()
    row = (
        m.Reset.objects.select_related("user")
        .filter(token=str(token))
        .values("id", "user_id", "token", "expires_at", "used_at", "user__email")
        .first()
    )
    if not row:
        messages.error(request, "Invalid or expired token")
        return redirect("/login")
    now = dt.datetime.now()
    expires_raw = row.get("expires_at")
    expires = (
        expires_raw
        if isinstance(expires_raw, dt.datetime)
        else dt.datetime.fromisoformat(str(expires_raw))
    )
    if row.get("used_at") or now > expires:
        messages.error(request, "Invalid or expired token")
        return redirect("/login")
    if request.method == "POST":
        pw1 = str(request.POST.get("password") or "")
        pw2 = str(request.POST.get("password2") or "")
        if not pw1 or pw1 != pw2:
            messages.error(request, "Passwords do not match")
            return render(request, "reset_password.html")
        newhash = generate_password_hash(pw1)
        from django.db import transaction

        now2 = dt.datetime.now()
        with transaction.atomic():
            m.User.objects.filter(id=int(row["user_id"])).update(password_hash=newhash)
            m.Reset.objects.filter(id=int(row["id"])).update(used_at=now2)
        messages.success(request, "Password updated. Please log in.")
        return redirect("/login")
    return render(request, "reset_password.html")


def logout_view(request: HttpRequest) -> HttpResponse:
    try:
        request.session.flush()
    except Exception:
        request.session.clear()
    return redirect("/")


def league_select(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    lid = str(request.POST.get("league_id") or "").strip()
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    uid = _session_user_id(request)
    if lid and lid.isdigit():
        lid_i = int(lid)
        ok = (
            m.League.objects.filter(id=lid_i)
            .filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
            .exists()
        )
        if ok:
            request.session["league_id"] = lid_i
            m.User.objects.filter(id=uid).update(default_league_id=lid_i)
    else:
        request.session.pop("league_id", None)
        m.User.objects.filter(id=uid).update(default_league=None)
    referer = str(request.META.get("HTTP_REFERER") or "").strip()
    return redirect(referer or "/")


# ----------------------------
# API
# ----------------------------


@csrf_exempt
def api_user_video_clip_len(request: HttpRequest) -> JsonResponse:
    if not _session_user_id(request):
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    payload = _json_body(request)
    raw = payload.get("clip_len_s")
    try:
        v = int(raw)
    except Exception:
        return JsonResponse(
            {"ok": False, "error": "clip_len_s must be one of: 15, 20, 30, 45, 60, 90"}, status=400
        )
    if v not in {15, 20, 30, 45, 60, 90}:
        return JsonResponse(
            {"ok": False, "error": "clip_len_s must be one of: 15, 20, 30, 45, 60, 90"}, status=400
        )
    try:
        _django_orm, m = _orm_modules()
        m.User.objects.filter(id=_session_user_id(request)).update(video_clip_len_s=int(v))
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "clip_len_s": int(v)})


def api_league_page_views(request: HttpRequest, league_id: int) -> JsonResponse:
    r = _require_login(request)
    if r:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    user_id = _session_user_id(request)
    _django_orm, m = _orm_modules()
    owner_id = (
        m.League.objects.filter(id=int(league_id)).values_list("owner_user_id", flat=True).first()
    )
    owner_id_i = int(owner_id) if owner_id is not None else None
    if owner_id_i is None:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    if int(owner_id_i) != int(user_id):
        return JsonResponse({"ok": False, "error": "not_authorized"}, status=403)

    kind = str(request.GET.get("kind") or "").strip()
    entity_id_raw = request.GET.get("entity_id")
    try:
        entity_id = int(str(entity_id_raw or "0").strip() or "0")
    except Exception:
        entity_id = 0
    try:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=kind, entity_id=entity_id
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "kind": kind,
            "entity_id": int(entity_id),
            "count": int(count),
        }
    )


# ----------------------------
# Legacy uploads/jobs UI
# ----------------------------


def games(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    rows = list(m.Game.objects.filter(user_id=uid).order_by("-created_at").values())
    dw_state = logic.read_dirwatch_state()
    for g in rows:
        try:
            st = (
                (dw_state.get("processed", {}) or {}).get(str(g.get("dir_path") or ""), {}) or {}
            ).get("status") or g.get("status")
            g["display_status"] = st
        except Exception:
            g["display_status"] = g.get("status")
    return render(request, "games.html", {"games": rows, "state": dw_state})


def new_game(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if request.method == "POST":
        name = (
            str(request.POST.get("name") or "").strip() or f"game-{dt.datetime.now():%Y%m%d-%H%M%S}"
        )
        gid, _dir_path = logic.create_game(
            _session_user_id(request), name, str(request.session.get("user_email") or "")
        )
        messages.success(request, "Game created")
        return redirect(f"/games/{gid}")
    return render(request, "new_game.html")


def game_detail(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        messages.error(request, "Not found")
        return redirect("/games")
    files: list[str] = []
    try:
        all_files = (
            os.listdir(str(game.get("dir_path") or ""))
            if os.path.isdir(str(game.get("dir_path") or ""))
            else []
        )

        def _is_user_file(fname: str) -> bool:
            if not fname:
                return False
            if fname.startswith(".") or fname.startswith("_") or fname.startswith("slurm-"):
                return False
            return True

        files = [f for f in sorted(all_files) if _is_user_file(f)]
    except Exception:
        files = []

    latest_status: Optional[str] = None
    latest_job = m.Job.objects.filter(game_id=int(gid)).order_by("-id").values("status").first()
    if latest_job and latest_job.get("status") is not None:
        latest_status = str(latest_job["status"])
    if not latest_status:
        dw_state = logic.read_dirwatch_state()
        latest_status = (dw_state.get("processed", {}) or {}).get(
            str(game.get("dir_path") or ""), {}
        ).get("status") or str(game.get("status") or "")

    is_locked = bool(latest_job)
    final_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
    if latest_status and str(latest_status).upper() in final_states:
        is_locked = True
    return render(
        request,
        "game_detail.html",
        {"game": game, "files": files, "status": latest_status, "is_locked": is_locked},
    )


def delete_game(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        messages.error(request, "Not found")
        return redirect("/games")

    latest = (
        m.Job.objects.filter(game_id=int(gid))
        .order_by("-id")
        .values("id", "slurm_job_id", "status")
        .first()
    )
    if request.method == "POST":
        token = str(request.POST.get("confirm") or "").strip().upper()
        if token != "DELETE":
            messages.error(request, "Type DELETE to confirm permanent deletion.")
            return render(request, "confirm_delete.html", {"game": game})

        try:
            active_states = ("SUBMITTED", "RUNNING", "PENDING")
            if latest and str(latest.get("status", "")).upper() in active_states:
                import subprocess as _sp
                import time as _time

                dir_leaf = Path(str(game["dir_path"])).name
                job_name = f"dirwatch-{dir_leaf}"
                job_ids: list[str] = []
                jid = latest.get("slurm_job_id")
                if jid:
                    job_ids.append(str(jid))
                else:
                    try:
                        out = _sp.check_output(["squeue", "-h", "-o", "%i %j"]).decode()
                        for line in out.splitlines():
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2 and parts[1] == job_name:
                                job_ids.append(parts[0])
                    except Exception:
                        pass
                for jid2 in job_ids:
                    try:
                        _sp.run(["scancel", str(jid2)], check=False)
                    except Exception:
                        pass
                _time.sleep(0.5)
        except Exception:
            pass

        from django.db import transaction

        with transaction.atomic():
            m.Job.objects.filter(game_id=int(gid)).delete()
            m.Game.objects.filter(id=int(gid)).delete()
        try:
            import shutil

            shutil.rmtree(str(game.get("dir_path") or ""), ignore_errors=True)
        except Exception:
            pass
        messages.success(request, "Game deleted")
        return redirect("/games")
    return render(request, "confirm_delete.html", {"game": game})


def upload_to_game(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        raise Http404
    files = request.FILES.getlist("files")
    if not files:
        messages.error(request, "No files selected")
        return redirect(f"/games/{gid}")

    try:
        watch_root = Path(os.environ.get("HM_WATCH_ROOT", logic.WATCH_ROOT)).resolve()
        gp = Path(str(game.get("dir_path") or "")).resolve()
        if watch_root not in gp.parents and gp != watch_root:
            messages.error(request, "Invalid upload path")
            return redirect(f"/games/{gid}")
    except Exception:
        pass

    base_dir = Path(str(game.get("dir_path") or ""))
    for f in files:
        if not f or not getattr(f, "name", ""):
            continue
        fname = Path(str(f.name)).name
        dest = base_dir / fname
        with dest.open("wb") as out:
            for chunk in f.chunks():
                out.write(chunk)
    messages.success(request, "Uploaded")
    return redirect(f"/games/{gid}")


def run_game(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        raise Http404
    ready = Path(str(game.get("dir_path") or "")) / "_READY"
    ready.write_text("ready\n", encoding="utf-8")
    now = dt.datetime.now()
    m.Job.objects.create(
        user_id=uid,
        game_id=int(gid),
        dir_path=str(game.get("dir_path") or ""),
        slurm_job_id=None,
        status="SUBMITTED",
        created_at=now,
        updated_at=now,
        finished_at=None,
        user_email=str(request.session.get("user_email") or ""),
    )
    messages.success(request, "Job submitted")
    return redirect(f"/games/{gid}")


def serve_upload(request: HttpRequest, gid: int, name: str) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values("dir_path").first()
    if not game:
        raise Http404
    base = Path(str(game.get("dir_path") or "")).resolve()
    target = (base / str(name)).resolve()
    if base not in target.parents and target != base:
        raise Http404
    return _safe_file_response(target, as_attachment=True)


def jobs(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    rows = list(m.Job.objects.filter(user_id=uid).order_by("-created_at").values())
    return render(request, "jobs.html", {"jobs": rows})


# ----------------------------
# Hockey UI (partial; ported in follow-up patch)
# ----------------------------


def media_team_logo(request: HttpRequest, team_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    row = (
        m.Team.objects.filter(id=int(team_id), user_id=uid)
        .values("id", "user_id", "logo_path")
        .first()
    )
    if not row and league_id:
        row = (
            m.Team.objects.filter(id=int(team_id), league_teams__league_id=int(league_id))
            .values("id", "user_id", "logo_path")
            .first()
        )
    if not row or not row.get("logo_path"):
        raise Http404
    return _safe_file_response(Path(str(row["logo_path"])).resolve(), as_attachment=False)


def teams(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    include_external = str(request.GET.get("all") or "0") == "1"
    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
            entity_id=0,
            viewer_user_id=_session_user_id(request),
            league_owner_user_id=league_owner_user_id,
        )
    is_league_admin = bool(
        league_id and _is_league_admin(int(league_id), _session_user_id(request))
    )

    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    if league_id:
        rows_raw = list(
            m.LeagueTeam.objects.filter(league_id=int(league_id))
            .select_related("team")
            .values(
                "team_id",
                "team__user_id",
                "team__name",
                "team__logo_path",
                "team__is_external",
                "team__created_at",
                "team__updated_at",
                "division_name",
                "division_id",
                "conference_id",
                "mhr_rating",
                "mhr_agd",
                "mhr_sched",
                "mhr_games",
                "mhr_updated_at",
            )
        )
        rows: list[dict[str, Any]] = []
        for r0 in rows_raw:
            rows.append(
                {
                    "id": int(r0["team_id"]),
                    "user_id": int(r0["team__user_id"]),
                    "name": r0.get("team__name"),
                    "logo_path": r0.get("team__logo_path"),
                    "is_external": bool(r0.get("team__is_external")),
                    "created_at": r0.get("team__created_at"),
                    "updated_at": r0.get("team__updated_at"),
                    "division_name": r0.get("division_name"),
                    "division_id": r0.get("division_id"),
                    "conference_id": r0.get("conference_id"),
                    "mhr_rating": r0.get("mhr_rating"),
                    "mhr_agd": r0.get("mhr_agd"),
                    "mhr_sched": r0.get("mhr_sched"),
                    "mhr_games": r0.get("mhr_games"),
                    "mhr_updated_at": r0.get("mhr_updated_at"),
                }
            )
    else:
        qs = m.Team.objects.filter(user_id=uid)
        if not include_external:
            qs = qs.filter(is_external=False)
        rows = list(qs.order_by("name").values())

    stats: dict[int, dict[str, Any]] = {}
    for t in rows:
        tid = int(t["id"])
        if league_id:
            stats[tid] = logic.compute_team_stats_league(None, tid, int(league_id))
        else:
            stats[tid] = logic.compute_team_stats(None, tid, uid)
        try:
            s = stats[tid]
            s["gp"] = (
                int(s.get("wins", 0) or 0)
                + int(s.get("losses", 0) or 0)
                + int(s.get("ties", 0) or 0)
            )
        except Exception:
            pass

    divisions = None
    if league_id:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for t in rows:
            dn = str(t.get("division_name") or "").strip() or "Unknown Division"
            grouped.setdefault(dn, []).append(t)
        divisions = []
        for dn in sorted(grouped.keys(), key=logic.division_sort_key):
            teams_sorted = sorted(
                grouped[dn],
                key=lambda tr: logic.sort_key_team_standings(tr, stats.get(int(tr["id"]), {})),
            )
            divisions.append({"name": dn, "teams": teams_sorted})

    league_page_views = None
    if league_id and is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
            "entity_id": 0,
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
            ),
        }
    return render(
        request,
        "teams.html",
        {
            "teams": rows,
            "divisions": divisions,
            "stats": stats,
            "include_external": include_external,
            "league_view": bool(league_id),
            "current_user_id": uid,
            "is_league_admin": is_league_admin,
            "league_page_views": league_page_views,
        },
    )


# ----------------------------
# Team/player management
# ----------------------------


def new_team(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        if not name:
            messages.error(request, "Team name is required")
            return render(request, "team_new.html")
        tid = logic.create_team(_session_user_id(request), name, is_external=False)
        f = request.FILES.get("logo")
        if f and getattr(f, "name", ""):
            try:
                p = logic.save_team_logo(f, tid)
                _django_orm, m = _orm_modules()
                m.Team.objects.filter(id=int(tid), user_id=_session_user_id(request)).update(
                    logo_path=str(p)
                )
            except Exception:
                messages.error(request, "Failed to save team logo")
        messages.success(request, "Team created")
        return redirect(f"/teams/{tid}")
    return render(request, "team_new.html")


def team_detail(request: HttpRequest, team_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r

    recent_n_raw = request.GET.get("recent_n")
    try:
        recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
    except Exception:
        recent_n = 5
    recent_sort = str(request.GET.get("recent_sort") or "points").strip() or "points"
    recent_dir = str(request.GET.get("recent_dir") or "desc").strip().lower() or "desc"

    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )

    is_league_admin = False
    if league_id:
        try:
            is_league_admin = bool(_is_league_admin(int(league_id), _session_user_id(request)))
        except Exception:
            is_league_admin = False

    team = logic.get_team(int(team_id), _session_user_id(request))
    editable = bool(team)
    _django_orm, m = _orm_modules()
    if not team and league_id:
        team = (
            m.Team.objects.filter(id=int(team_id), league_teams__league_id=int(league_id))
            .values(
                "id",
                "user_id",
                "name",
                "logo_path",
                "is_external",
                "created_at",
                "updated_at",
            )
            .first()
        )
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")

    if league_id:
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            entity_id=int(team_id),
            viewer_user_id=_session_user_id(request),
            league_owner_user_id=league_owner_user_id,
        )

    team_owner_id = int(team["user_id"])
    players_qs = m.Player.objects.filter(team_id=int(team_id))
    if editable:
        players_qs = players_qs.filter(user_id=_session_user_id(request))
    players = list(
        players_qs.order_by("jersey_number", "name").values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    skaters, goalies, head_coaches, assistant_coaches = logic.split_roster(players or [])
    roster_players = list(skaters) + list(goalies)

    from django.db.models import Q

    if league_id:
        tstats = logic.compute_team_stats_league(None, int(team_id), int(league_id))
        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                "team_id", "division_name"
            )
        }
        schedule_rows_raw = list(
            m.LeagueGame.objects.filter(league_id=int(league_id))
            .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
            .select_related("game", "game__team1", "game__team2", "game__game_type")
            .values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__stats_imported_at",
                "game__created_at",
                "game__updated_at",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        schedule_games: list[dict[str, Any]] = []
        for r0 in schedule_rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            schedule_games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "stats_imported_at": r0.get("game__stats_imported_at"),
                    "created_at": r0.get("game__created_at"),
                    "updated_at": r0.get("game__updated_at"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )

        schedule_games = [
            g2
            for g2 in (schedule_games or [])
            if not logic._league_game_is_cross_division_non_external(g2)
        ]
        now_dt = dt.datetime.now()
        for g2 in schedule_games:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = logic._sanitize_http_url(
                    logic._extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])
        schedule_game_ids = [
            int(g2.get("id")) for g2 in (schedule_games or []) if g2.get("id") is not None
        ]
        ps_rows = list(
            m.PlayerStat.objects.filter(team_id=int(team_id), game_id__in=schedule_game_ids).values(
                "player_id", "game_id", *logic.PLAYER_STATS_SUM_KEYS
            )
        )
    else:
        tstats = logic.compute_team_stats(None, int(team_id), team_owner_id)
        schedule_rows = list(
            m.HkyGame.objects.filter(user_id=int(team_owner_id))
            .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "stats_imported_at",
                "created_at",
                "updated_at",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        schedule_games = []
        for r0 in schedule_rows:
            schedule_games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "stats_imported_at": r0.get("stats_imported_at"),
                    "created_at": r0.get("created_at"),
                    "updated_at": r0.get("updated_at"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )
        now_dt = dt.datetime.now()
        for g2 in schedule_games:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = logic._sanitize_http_url(
                    logic._extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
        schedule_game_ids = [
            int(g2.get("id")) for g2 in (schedule_games or []) if g2.get("id") is not None
        ]
        ps_rows = list(
            m.PlayerStat.objects.filter(team_id=int(team_id), game_id__in=schedule_game_ids).values(
                "player_id", "game_id", *logic.PLAYER_STATS_SUM_KEYS
            )
        )

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"
    # Tournament-only players: show them on game pages, but not on team/division-level roster/stats.
    try:
        tournament_game_ids: set[int] = {
            int(g2.get("id"))
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "").strip().casefold() == "tournament"
            and g2.get("id") is not None
        }
        player_ids_with_any_stats: set[int] = set()
        player_ids_with_non_tournament_stats: set[int] = set()
        for r0 in ps_rows or []:
            try:
                pid_i = int(r0.get("player_id"))
                gid_i = int(r0.get("game_id"))
            except Exception:
                continue
            player_ids_with_any_stats.add(pid_i)
            if gid_i not in tournament_game_ids:
                player_ids_with_non_tournament_stats.add(pid_i)
        tournament_only_player_ids = (
            player_ids_with_any_stats - player_ids_with_non_tournament_stats
        )
    except Exception:
        tournament_only_player_ids = set()

    if tournament_only_player_ids:
        skaters = [p for p in skaters if int(p.get("id") or 0) not in tournament_only_player_ids]
        goalies = [p for p in goalies if int(p.get("id") or 0) not in tournament_only_player_ids]
        roster_players = list(skaters) + list(goalies)
        ps_rows = [
            r0 for r0 in ps_rows if int(r0.get("player_id") or 0) not in tournament_only_player_ids
        ]
    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [g2 for g2 in stats_schedule_games if logic._game_has_recorded_result(g2)]
    eligible_game_ids_in_order: list[int] = []
    for g2 in eligible_games:
        try:
            eligible_game_ids_in_order.append(int(g2.get("id")))
        except Exception:
            continue
    eligible_game_ids: set[int] = set(eligible_game_ids_in_order)
    ps_rows_filtered = []
    for r0 in ps_rows or []:
        try:
            if int(r0.get("game_id")) in eligible_game_ids:
                ps_rows_filtered.append(r0)
        except Exception:
            continue

    player_totals = logic._aggregate_player_totals_from_rows(
        player_stats_rows=ps_rows_filtered, allowed_game_ids=eligible_game_ids
    )
    player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(skaters, player_totals)
    )
    player_stats_columns = logic.filter_player_stats_display_columns_for_rows(
        logic.PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows
    )
    cov_counts, cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows_filtered, eligible_game_ids=eligible_game_ids_in_order
    )
    player_stats_columns = logic._player_stats_columns_with_coverage(
        columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
    )

    recent_scope_ids = (
        eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
    )
    recent_totals = logic.compute_recent_player_totals_from_rows(
        schedule_games=stats_schedule_games, player_stats_rows=ps_rows_filtered, n=recent_n
    )
    recent_player_stats_rows = logic.sort_player_stats_rows(
        logic.build_player_stats_table_rows(skaters, recent_totals),
        sort_key=recent_sort,
        sort_dir=recent_dir,
    )
    recent_player_stats_columns = logic.filter_player_stats_display_columns_for_rows(
        logic.PLAYER_STATS_DISPLAY_COLUMNS, recent_player_stats_rows
    )
    recent_cov_counts, recent_cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows_filtered, eligible_game_ids=recent_scope_ids
    )
    recent_player_stats_columns = logic._player_stats_columns_with_coverage(
        columns=recent_player_stats_columns,
        coverage_counts=recent_cov_counts,
        total_games=recent_cov_total,
    )

    player_stats_sources = logic._compute_team_player_stats_sources(
        None, eligible_game_ids=eligible_game_ids_in_order
    )
    selected_label = (
        "All"
        if selected_types is None
        else ", ".join(sorted(list(selected_types), key=lambda s: s.lower()))
    )
    game_type_filter_options = [
        {"label": gt, "checked": (selected_types is None) or (gt in selected_types)}
        for gt in game_type_options
    ]
    league_page_views = None
    if league_id and is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            "entity_id": int(team_id),
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM, entity_id=int(team_id)
            ),
        }
    return render(
        request,
        "team_detail.html",
        {
            "team": team,
            "roster_players": roster_players,
            "players": skaters,
            "head_coaches": head_coaches,
            "assistant_coaches": assistant_coaches,
            "player_stats_columns": player_stats_columns,
            "player_stats_rows": player_stats_rows,
            "recent_player_stats_columns": recent_player_stats_columns,
            "recent_player_stats_rows": recent_player_stats_rows,
            "recent_n": recent_n,
            "recent_sort": recent_sort,
            "recent_dir": recent_dir,
            "tstats": tstats,
            "schedule_games": schedule_games,
            "editable": editable,
            "is_league_admin": is_league_admin,
            "player_stats_sources": player_stats_sources,
            "player_stats_coverage_total_games": cov_total,
            "player_stats_recent_coverage_total_games": recent_cov_total,
            "game_type_filter_options": game_type_filter_options,
            "game_type_filter_label": selected_label,
            "league_page_views": league_page_views,
        },
    )


def team_edit(request: HttpRequest, team_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    team = logic.get_team(int(team_id), _session_user_id(request))
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        if name:
            _django_orm, m = _orm_modules()
            m.Team.objects.filter(id=int(team_id), user_id=_session_user_id(request)).update(
                name=name
            )
        f = request.FILES.get("logo")
        if f and getattr(f, "name", ""):
            p = logic.save_team_logo(f, int(team_id))
            _django_orm, m = _orm_modules()
            m.Team.objects.filter(id=int(team_id), user_id=_session_user_id(request)).update(
                logo_path=str(p)
            )
        messages.success(request, "Team updated")
        return redirect(f"/teams/{team_id}")
    return render(request, "team_edit.html", {"team": team})


def player_new(request: HttpRequest, team_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    team = logic.get_team(int(team_id), _session_user_id(request))
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        jersey = str(request.POST.get("jersey_number") or "").strip()
        position = str(request.POST.get("position") or "").strip()
        shoots = str(request.POST.get("shoots") or "").strip()
        if not name:
            messages.error(request, "Player name is required")
            return render(request, "player_edit.html", {"team": team})
        _django_orm, m = _orm_modules()
        m.Player.objects.create(
            user_id=_session_user_id(request),
            team_id=int(team_id),
            name=name,
            jersey_number=jersey or None,
            position=position or None,
            shoots=shoots or None,
            created_at=dt.datetime.now(),
            updated_at=None,
        )
        messages.success(request, "Player added")
        return redirect(f"/teams/{team_id}")
    return render(request, "player_edit.html", {"team": team})


def player_edit(request: HttpRequest, team_id: int, player_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    team = logic.get_team(int(team_id), _session_user_id(request))
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")
    _django_orm, m = _orm_modules()
    pl = (
        m.Player.objects.filter(
            id=int(player_id), team_id=int(team_id), user_id=_session_user_id(request)
        )
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
        .first()
    )
    if not pl:
        messages.error(request, "Not found")
        return redirect(f"/teams/{team_id}")
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        jersey = str(request.POST.get("jersey_number") or "").strip()
        position = str(request.POST.get("position") or "").strip()
        shoots = str(request.POST.get("shoots") or "").strip()
        m.Player.objects.filter(
            id=int(player_id), team_id=int(team_id), user_id=_session_user_id(request)
        ).update(
            name=name or pl["name"],
            jersey_number=jersey or None,
            position=position or None,
            shoots=shoots or None,
            updated_at=dt.datetime.now(),
        )
        messages.success(request, "Player updated")
        return redirect(f"/teams/{team_id}")
    return render(request, "player_edit.html", {"team": team, "player": pl})


def player_delete(request: HttpRequest, team_id: int, player_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    m.Player.objects.filter(
        id=int(player_id), team_id=int(team_id), user_id=_session_user_id(request)
    ).delete()
    messages.success(request, "Player deleted")
    return redirect(f"/teams/{team_id}")


def schedule(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            entity_id=0,
            viewer_user_id=_session_user_id(request),
            league_owner_user_id=league_owner_user_id,
        )

    selected_division = str(request.GET.get("division") or "").strip() or None
    selected_team_id = str(request.GET.get("team_id") or "")
    team_id_i: Optional[int] = None
    try:
        team_id_i = int(selected_team_id) if selected_team_id.strip() else None
    except Exception:
        team_id_i = None

    divisions: list[Any] = []
    league_teams: list[dict[str, Any]] = []
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    games: list[dict[str, Any]] = []
    if league_id:
        divisions = list(
            m.LeagueTeam.objects.filter(league_id=int(league_id))
            .exclude(division_name__isnull=True)
            .exclude(division_name="")
            .values_list("division_name", flat=True)
            .distinct()
        )
        divisions.sort(key=logic.division_sort_key)
        if selected_division:
            league_teams = list(
                m.Team.objects.filter(
                    league_teams__league_id=int(league_id),
                    league_teams__division_name=str(selected_division),
                )
                .distinct()
                .order_by("name")
                .values("id", "name")
            )
        else:
            league_teams = list(
                m.Team.objects.filter(league_teams__league_id=int(league_id))
                .distinct()
                .order_by("name")
                .values("id", "name")
            )
        if team_id_i is not None and not any(int(t["id"]) == int(team_id_i) for t in league_teams):
            team_id_i = None
            selected_team_id = ""

        lg_qs = m.LeagueGame.objects.filter(league_id=int(league_id))
        if selected_division:
            lg_qs = lg_qs.filter(division_name=str(selected_division))
        if team_id_i is not None:
            lg_qs = lg_qs.filter(
                Q(game__team1_id=int(team_id_i)) | Q(game__team2_id=int(team_id_i))
            )

        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                "team_id", "division_name"
            )
        }
        rows_raw = list(
            lg_qs.select_related("game", "game__team1", "game__team2", "game__game_type").values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__stats_imported_at",
                "game__created_at",
                "game__updated_at",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        for r0 in rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "stats_imported_at": r0.get("game__stats_imported_at"),
                    "created_at": r0.get("game__created_at"),
                    "updated_at": r0.get("game__updated_at"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
    else:
        rows = list(
            m.HkyGame.objects.filter(user_id=_session_user_id(request))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "stats_imported_at",
                "created_at",
                "updated_at",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        for r0 in rows:
            games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "stats_imported_at": r0.get("stats_imported_at"),
                    "created_at": r0.get("created_at"),
                    "updated_at": r0.get("updated_at"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )

    if league_id:
        games = [
            g2 for g2 in (games or []) if not logic._league_game_is_cross_division_non_external(g2)
        ]

    now_dt = dt.datetime.now()
    is_league_admin = bool(
        league_id and _is_league_admin(int(league_id), _session_user_id(request))
    )
    for g2 in games or []:
        sdt = g2.get("starts_at")
        started = False
        if sdt is not None:
            try:
                started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
            except Exception:
                started = False
        has_score = (
            (g2.get("team1_score") is not None)
            or (g2.get("team2_score") is not None)
            or bool(g2.get("is_final"))
        )
        g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None
        try:
            g2["can_edit"] = bool(
                int(g2.get("user_id") or 0) == _session_user_id(request) or is_league_admin
            )
        except Exception:
            g2["can_edit"] = bool(is_league_admin)

    games = logic.sort_games_schedule_order(games or [])
    league_page_views = None
    if league_id and is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            "entity_id": 0,
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
            ),
        }
    return render(
        request,
        "schedule.html",
        {
            "games": games,
            "league_view": bool(league_id),
            "divisions": divisions,
            "league_teams": league_teams,
            "selected_division": selected_division or "",
            "selected_team_id": str(team_id_i) if team_id_i is not None else "",
            "league_page_views": league_page_views,
        },
    )


def schedule_new(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    my_teams = list(
        m.Team.objects.filter(user_id=_session_user_id(request), is_external=False)
        .order_by("name")
        .values("id", "name")
    )
    gt = list(m.GameType.objects.order_by("name").values("id", "name"))
    if request.method == "POST":
        team1_id = int(request.POST.get("team1_id") or 0)
        team2_id = int(request.POST.get("team2_id") or 0)
        opp_name = str(request.POST.get("opponent_name") or "").strip()
        game_type_id = int(request.POST.get("game_type_id") or 0)
        starts_at = str(request.POST.get("starts_at") or "").strip()
        location = str(request.POST.get("location") or "").strip()

        if not team1_id and not team2_id:
            messages.error(request, "Select at least one of your teams")
            return render(request, "schedule_new.html", {"my_teams": my_teams, "game_types": gt})
        if team1_id and not team2_id:
            team2_id = logic.ensure_external_team(_session_user_id(request), opp_name or "Opponent")
        elif team2_id and not team1_id:
            team1_id = logic.ensure_external_team(_session_user_id(request), opp_name or "Opponent")

        gid = logic.create_hky_game(
            user_id=_session_user_id(request),
            team1_id=int(team1_id),
            team2_id=int(team2_id),
            game_type_id=int(game_type_id) if game_type_id else None,
            starts_at=logic.to_dt(starts_at),
            location=location or None,
        )

        league_id = request.session.get("league_id")
        if league_id:
            from django.db import transaction

            try:
                with transaction.atomic():
                    m.LeagueTeam.objects.get_or_create(
                        league_id=int(league_id), team_id=int(team1_id)
                    )
                    m.LeagueTeam.objects.get_or_create(
                        league_id=int(league_id), team_id=int(team2_id)
                    )
                    m.LeagueGame.objects.get_or_create(league_id=int(league_id), game_id=int(gid))
            except Exception:
                pass
        messages.success(request, "Game created")
        return redirect(f"/hky/games/{gid}")
    return render(request, "schedule_new.html", {"my_teams": my_teams, "game_types": gt})


def hky_game_detail(request: HttpRequest, game_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )

    _django_orm, m = _orm_modules()
    session_uid = _session_user_id(request)
    owned_row = (
        m.HkyGame.objects.filter(id=int(game_id), user_id=session_uid)
        .select_related("team1", "team2")
        .values(
            "id",
            "user_id",
            "team1_id",
            "team2_id",
            "game_type_id",
            "starts_at",
            "location",
            "notes",
            "team1_score",
            "team2_score",
            "is_final",
            "stats_imported_at",
            "created_at",
            "updated_at",
            "team1__name",
            "team2__name",
            "team1__is_external",
            "team2__is_external",
        )
        .first()
    )
    game: Optional[dict[str, Any]] = None
    if owned_row:
        game = {
            "id": int(owned_row["id"]),
            "user_id": int(owned_row["user_id"]),
            "team1_id": int(owned_row["team1_id"]),
            "team2_id": int(owned_row["team2_id"]),
            "game_type_id": owned_row.get("game_type_id"),
            "starts_at": owned_row.get("starts_at"),
            "location": owned_row.get("location"),
            "notes": owned_row.get("notes"),
            "team1_score": owned_row.get("team1_score"),
            "team2_score": owned_row.get("team2_score"),
            "is_final": owned_row.get("is_final"),
            "stats_imported_at": owned_row.get("stats_imported_at"),
            "created_at": owned_row.get("created_at"),
            "updated_at": owned_row.get("updated_at"),
            "team1_name": owned_row.get("team1__name"),
            "team2_name": owned_row.get("team2__name"),
            "team1_ext": owned_row.get("team1__is_external"),
            "team2_ext": owned_row.get("team2__is_external"),
        }
    elif league_id:
        league_row = (
            m.LeagueGame.objects.filter(league_id=int(league_id), game_id=int(game_id))
            .select_related("game", "game__team1", "game__team2")
            .values(
                "game_id",
                "division_name",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__stats_imported_at",
                "game__created_at",
                "game__updated_at",
                "game__team1__name",
                "game__team2__name",
                "game__team1__is_external",
                "game__team2__is_external",
            )
            .first()
        )
        if league_row:
            t1 = int(league_row["game__team1_id"])
            t2 = int(league_row["game__team2_id"])
            div_map = {
                int(tid): (str(dn).strip() if dn is not None else None)
                for tid, dn in m.LeagueTeam.objects.filter(
                    league_id=int(league_id),
                    team_id__in=[t1, t2],
                ).values_list("team_id", "division_name")
            }
            game = {
                "id": int(league_row["game_id"]),
                "user_id": int(league_row["game__user_id"]),
                "team1_id": t1,
                "team2_id": t2,
                "game_type_id": league_row.get("game__game_type_id"),
                "starts_at": league_row.get("game__starts_at"),
                "location": league_row.get("game__location"),
                "notes": league_row.get("game__notes"),
                "team1_score": league_row.get("game__team1_score"),
                "team2_score": league_row.get("game__team2_score"),
                "is_final": league_row.get("game__is_final"),
                "stats_imported_at": league_row.get("game__stats_imported_at"),
                "created_at": league_row.get("game__created_at"),
                "updated_at": league_row.get("game__updated_at"),
                "team1_name": league_row.get("game__team1__name"),
                "team2_name": league_row.get("game__team2__name"),
                "team1_ext": league_row.get("game__team1__is_external"),
                "team2_ext": league_row.get("game__team2__is_external"),
                "division_name": league_row.get("division_name"),
                "team1_league_division_name": div_map.get(t1),
                "team2_league_division_name": div_map.get(t2),
            }

    if not game:
        messages.error(request, "Not found")
        return redirect("/schedule")

    try:
        game["game_video_url"] = logic._sanitize_http_url(
            logic._extract_game_video_url_from_notes(game.get("notes"))
        )
    except Exception:
        game["game_video_url"] = None

    is_owner = int(game.get("user_id") or 0) == int(session_uid)
    if league_id and not is_owner and logic._league_game_is_cross_division_non_external(game):
        raise Http404

    now_dt = dt.datetime.now()
    sdt = game.get("starts_at")
    started = False
    if sdt is not None:
        try:
            started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
        except Exception:
            started = False
    has_score = (
        (game.get("team1_score") is not None)
        or (game.get("team2_score") is not None)
        or bool(game.get("is_final"))
    )
    can_view_summary = bool(has_score or (sdt is None) or started)
    if not can_view_summary:
        raise Http404

    if league_id:
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME,
            entity_id=int(game_id),
            viewer_user_id=session_uid,
            league_owner_user_id=league_owner_user_id,
        )

    tts_linked = logic._extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

    return_to = logic._safe_return_to_url(request.GET.get("return_to"), default="/schedule")

    can_edit = bool(is_owner)
    if league_id and not can_edit:
        try:
            can_edit = bool(_is_league_admin(int(league_id), session_uid))
        except Exception:
            can_edit = False
    edit_mode = bool(can_edit and (request.GET.get("edit") == "1" or request.method == "POST"))

    team1_players_qs = m.Player.objects.filter(team_id=int(game["team1_id"]))
    team2_players_qs = m.Player.objects.filter(team_id=int(game["team2_id"]))
    if is_owner:
        team1_players_qs = team1_players_qs.filter(user_id=session_uid)
        team2_players_qs = team2_players_qs.filter(user_id=session_uid)

    team1_players = list(
        team1_players_qs.order_by("jersey_number", "name").values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    team2_players = list(
        team2_players_qs.order_by("jersey_number", "name").values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    stats_rows = list(m.PlayerStat.objects.filter(game_id=int(game_id)).values())
    game_stats_row = (
        m.HkyGameStat.objects.filter(game_id=int(game_id))
        .values("stats_json", "updated_at")
        .first()
    )
    team1_skaters, team1_goalies, team1_hc, team1_ac = logic.split_roster(team1_players or [])
    team2_skaters, team2_goalies, team2_hc, team2_ac = logic.split_roster(team2_players or [])
    team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
    team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
    stats_by_pid = {r0["player_id"]: r0 for r0 in stats_rows}

    game_stats = None
    game_stats_updated_at = None
    try:
        if game_stats_row and game_stats_row.get("stats_json"):
            game_stats = json.loads(game_stats_row["stats_json"])
            game_stats_updated_at = game_stats_row.get("updated_at")
    except Exception:
        game_stats = None
    game_stats = logic.filter_game_stats_for_display(game_stats)
    period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}

    events_headers: list[str] = []
    events_rows: list[dict[str, str]] = []
    events_meta: Optional[dict[str, Any]] = None
    try:
        erow = (
            m.HkyGameEvent.objects.filter(game_id=int(game_id))
            .values("events_csv", "source_label", "updated_at")
            .first()
        )
        if erow and str(erow.get("events_csv") or "").strip():
            events_headers, events_rows = logic.parse_events_csv(str(erow.get("events_csv") or ""))
            events_meta = {
                "source_label": erow.get("source_label"),
                "updated_at": erow.get("updated_at"),
                "count": len(events_rows),
            }
    except Exception:
        events_headers, events_rows, events_meta = [], [], None

    try:
        events_headers, events_rows = logic.normalize_game_events_csv(events_headers, events_rows)
    except Exception:
        pass
    events_rows = logic.filter_events_rows_prefer_timetoscore_for_goal_assist(
        events_rows, tts_linked=tts_linked
    )
    try:
        events_headers, events_rows = logic.normalize_events_video_time_for_display(
            events_headers, events_rows
        )
        events_headers, events_rows = logic.filter_events_headers_drop_empty_on_ice_split(
            events_headers, events_rows
        )
        events_rows = logic.sort_events_rows_default(events_rows)
    except Exception:
        pass
    if events_meta is not None:
        try:
            events_meta["count"] = len(events_rows)
            events_meta["sources"] = logic.summarize_event_sources(
                events_rows, fallback_source_label=str(events_meta.get("source_label") or "")
            )
        except Exception:
            pass

    scoring_by_period_rows = logic.compute_team_scoring_by_period_from_events(
        events_rows, tts_linked=tts_linked
    )
    try:
        game_event_stats_rows = logic.compute_game_event_stats_by_side(events_rows)
    except Exception:
        game_event_stats_rows = []

    imported_player_stats_csv_text: Optional[str] = None
    player_stats_import_meta: Optional[dict[str, Any]] = None
    try:
        prow = (
            m.HkyGamePlayerStatsCsv.objects.filter(game_id=int(game_id))
            .values("player_stats_csv", "source_label", "updated_at")
            .first()
        )
        if prow and str(prow.get("player_stats_csv") or "").strip():
            imported_player_stats_csv_text = str(prow.get("player_stats_csv") or "")
            player_stats_import_meta = {
                "source_label": prow.get("source_label"),
                "updated_at": prow.get("updated_at"),
            }
    except Exception:
        imported_player_stats_csv_text, player_stats_import_meta = None, None

    (
        game_player_stats_columns,
        player_stats_cells_by_pid,
        player_stats_cell_conflicts_by_pid,
        player_stats_import_warning,
    ) = logic.build_game_player_stats_table(
        players=list(team1_skaters) + list(team2_skaters),
        stats_by_pid=stats_by_pid,
        imported_csv_text=imported_player_stats_csv_text,
        prefer_db_stats_for_keys={"goals", "assists"} if tts_linked else None,
    )
    team1_skaters_sorted = list(team1_skaters)
    team2_skaters_sorted = list(team2_skaters)
    try:

        def _sort_players_for_game(players_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows = []
            by_pid = {}
            for p in players_in:
                pid = int(p.get("id"))
                cells = (player_stats_cells_by_pid or {}).get(pid, {})
                g = logic._parse_int_from_cell_text(cells.get("goals", ""))
                a = logic._parse_int_from_cell_text(cells.get("assists", ""))
                rows.append(
                    {
                        "player_id": pid,
                        "name": str(p.get("name") or ""),
                        "goals": g,
                        "assists": a,
                        "points": g + a,
                    }
                )
                by_pid[pid] = p
            sorted_rows = logic.sort_players_table_default(rows)
            return [
                by_pid[int(r0["player_id"])] for r0 in sorted_rows if int(r0["player_id"]) in by_pid
            ]

        team1_skaters_sorted = _sort_players_for_game(team1_skaters_sorted)
        team2_skaters_sorted = _sort_players_for_game(team2_skaters_sorted)
    except Exception:
        team1_skaters_sorted = list(team1_skaters)
        team2_skaters_sorted = list(team2_skaters)

    if request.method == "POST" and not edit_mode:
        messages.error(
            request, "You do not have permission to edit this game in the selected league."
        )
        return redirect(f"/hky/games/{int(game_id)}?return_to={return_to}")

    if request.method == "POST" and edit_mode:
        loc = str(request.POST.get("location") or "").strip()
        starts_at = str(request.POST.get("starts_at") or "").strip()
        t1_score = request.POST.get("team1_score")
        t2_score = request.POST.get("team2_score")
        is_final = bool(request.POST.get("is_final"))
        from django.db import transaction

        starts_at_dt = logic.to_dt(starts_at)
        updates = {
            "location": loc or None,
            "starts_at": starts_at_dt,
            "team1_score": int(t1_score) if (t1_score or "").strip() else None,
            "team2_score": int(t2_score) if (t2_score or "").strip() else None,
            "is_final": bool(is_final),
            "updated_at": dt.datetime.now(),
        }

        def _collect(prefix: str, pid: int) -> dict[str, Optional[int]]:
            def _ival(name: str) -> Optional[int]:
                v = request.POST.get(f"{prefix}_{name}_{pid}")
                return int(v) if v and v.strip() else None

            return {
                "goals": _ival("goals"),
                "assists": _ival("assists"),
                "shots": _ival("shots"),
                "pim": _ival("pim"),
                "plus_minus": _ival("plusminus"),
                "hits": _ival("hits"),
                "blocks": _ival("blocks"),
                "faceoff_wins": _ival("fow"),
                "faceoff_attempts": _ival("foa"),
                "goalie_saves": _ival("saves"),
                "goalie_ga": _ival("ga"),
                "goalie_sa": _ival("sa"),
            }

        cols = [
            "goals",
            "assists",
            "shots",
            "pim",
            "plus_minus",
            "hits",
            "blocks",
            "faceoff_wins",
            "faceoff_attempts",
            "goalie_saves",
            "goalie_ga",
            "goalie_sa",
        ]
        game_owner_user_id = int(game.get("user_id") or session_uid)
        with transaction.atomic():
            if is_owner:
                m.HkyGame.objects.filter(id=int(game_id), user_id=session_uid).update(**updates)
            else:
                m.HkyGame.objects.filter(id=int(game_id)).update(**updates)
            for p in list(team1_skaters) + list(team2_skaters):
                pid = int(p["id"])
                vals = _collect("ps", pid)
                team_id_for_player = int(p["team_id"])
                defaults = {
                    "user_id": int(game_owner_user_id),
                    "team_id": int(team_id_for_player),
                    **{c: vals.get(c) for c in cols},
                }
                ps, created = m.PlayerStat.objects.get_or_create(
                    game_id=int(game_id),
                    player_id=int(pid),
                    defaults=defaults,
                )
                if not created:
                    m.PlayerStat.objects.filter(id=ps.id).update(**{c: vals.get(c) for c in cols})

        messages.success(request, "Game updated")
        return redirect(f"/hky/games/{int(game_id)}?return_to={return_to}")

    league_page_views = None
    if league_id and is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_GAME,
            "entity_id": int(game_id),
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME, entity_id=int(game_id)
            ),
        }

    return render(
        request,
        "hky_game_detail.html",
        {
            "game": game,
            "team1_roster": team1_roster,
            "team2_roster": team2_roster,
            "team1_players": team1_skaters_sorted,
            "team2_players": team2_skaters_sorted,
            "stats_by_pid": stats_by_pid,
            "period_stats_by_pid": period_stats_by_pid,
            "game_stats": game_stats,
            "game_stats_updated_at": game_stats_updated_at,
            "editable": bool(edit_mode),
            "can_edit": bool(can_edit),
            "edit_mode": bool(edit_mode),
            "back_url": return_to,
            "return_to": return_to,
            "events_headers": events_headers,
            "events_rows": events_rows,
            "events_meta": events_meta,
            "scoring_by_period_rows": scoring_by_period_rows,
            "game_event_stats_rows": game_event_stats_rows,
            "user_video_clip_len_s": logic.get_user_video_clip_len_s(
                None, int(request.session.get("user_id") or 0)
            ),
            "user_is_logged_in": True,
            "game_player_stats_columns": game_player_stats_columns,
            "player_stats_cells_by_pid": player_stats_cells_by_pid,
            "player_stats_cell_conflicts_by_pid": player_stats_cell_conflicts_by_pid,
            "player_stats_import_meta": player_stats_import_meta,
            "player_stats_import_warning": player_stats_import_warning,
            "league_page_views": league_page_views,
        },
    )


def hky_game_import_shift_stats(
    request: HttpRequest, game_id: int
) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages
    from urllib.parse import urlencode

    r = _require_login(request)
    if r:
        return r
    if request.method != "POST":
        raise Http404

    league_id = request.session.get("league_id")

    _django_orm, m = _orm_modules()
    game = (
        m.HkyGame.objects.filter(id=int(game_id))
        .values("id", "user_id", "team1_id", "team2_id", "notes")
        .first()
    )
    if not game:
        messages.error(request, "Not found")
        return redirect("/schedule")

    tts_linked = logic._extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

    session_uid = _session_user_id(request)
    is_owner = int(game.get("user_id") or 0) == int(session_uid)
    if not is_owner:
        if (
            not league_id
            or not m.LeagueGame.objects.filter(
                league_id=int(league_id), game_id=int(game_id)
            ).exists()
        ):
            messages.error(request, "Not found")
            return redirect("/schedule")

    can_edit = bool(is_owner)
    if league_id and not can_edit:
        can_edit = bool(_is_league_admin(int(league_id), int(session_uid)))
    if not can_edit:
        messages.error(request, "You do not have permission to import stats for this game.")
        qs = urlencode(
            {
                "return_to": logic._safe_return_to_url(
                    request.GET.get("return_to"), default="/schedule"
                )
            }
        )
        return redirect(f"/hky/games/{int(game_id)}?{qs}")

    ps_file = request.FILES.get("player_stats_csv")
    if not ps_file or not getattr(ps_file, "name", ""):
        messages.error(request, "Select a player_stats.csv file to import.")
        qs = urlencode(
            {
                "return_to": logic._safe_return_to_url(
                    request.GET.get("return_to"), default="/schedule"
                )
            }
        )
        return redirect(f"/hky/games/{int(game_id)}?{qs}")

    try:
        ps_text = ps_file.read().decode("utf-8", errors="replace")
    except Exception:
        messages.error(request, "Failed to read uploaded player_stats.csv")
        qs = urlencode(
            {
                "return_to": logic._safe_return_to_url(
                    request.GET.get("return_to"), default="/schedule"
                )
            }
        )
        return redirect(f"/hky/games/{int(game_id)}?{qs}")

    try:
        parsed_rows = logic.parse_shift_stats_player_stats_csv(ps_text)
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"Failed to parse player_stats.csv: {e}")
        qs = urlencode(
            {
                "return_to": logic._safe_return_to_url(
                    request.GET.get("return_to"), default="/schedule"
                )
            }
        )
        return redirect(f"/hky/games/{int(game_id)}?{qs}")

    # Optional game_stats.csv
    game_stats = None
    gs_file = request.FILES.get("game_stats_csv")
    if gs_file and getattr(gs_file, "name", ""):
        try:
            gs_text = gs_file.read().decode("utf-8", errors="replace")
            game_stats = logic.parse_shift_stats_game_stats_csv(gs_text)
        except Exception:
            game_stats = None

    owner_user_id = int(game.get("user_id") or 0)
    team1_id = int(game["team1_id"])
    team2_id = int(game["team2_id"])

    players = list(
        m.Player.objects.filter(
            user_id=int(owner_user_id), team_id__in=[team1_id, team2_id]
        ).values("id", "team_id", "name", "jersey_number")
    )

    players_by_team: dict[int, list[dict[str, Any]]] = {}
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    player_team_by_id: dict[int, int] = {}

    for p in players:
        tid = int(p["team_id"])
        pid = int(p["id"])
        player_team_by_id[pid] = tid
        players_by_team.setdefault(tid, []).append(p)
        j = logic.normalize_jersey_number(p.get("jersey_number"))
        if j:
            jersey_to_player_ids.setdefault((tid, j), []).append(pid)
        nm = logic.normalize_player_name(p.get("name") or "")
        if nm:
            name_to_player_ids.setdefault((tid, nm), []).append(pid)

    def _resolve_player_id(jersey_norm: Optional[str], name_norm: str) -> Optional[int]:
        candidates: list[int] = []
        for tid in (team1_id, team2_id):
            if jersey_norm:
                candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])

        candidates = []
        for tid in (team1_id, team2_id):
            candidates.extend(name_to_player_ids.get((tid, name_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])

        # Jersey match + fuzzy name tie-breaker
        if jersey_norm:
            fuzzy: list[int] = []
            for tid in (team1_id, team2_id):
                for pid0 in jersey_to_player_ids.get((tid, jersey_norm), []):
                    pl = next(
                        (
                            x
                            for x in players_by_team.get(tid, [])
                            if int(x.get("id") or 0) == int(pid0)
                        ),
                        None,
                    )
                    if not pl:
                        continue
                    n2 = logic.normalize_player_name(pl.get("name") or "")
                    if n2 and (n2 in name_norm or name_norm in n2):
                        fuzzy.append(int(pid0))
            if len(set(fuzzy)) == 1:
                return int(list(set(fuzzy))[0])

        return None

    imported = 0
    unmatched: list[str] = []

    from django.db import transaction

    cols = [
        "goals",
        "assists",
        "shots",
        "pim",
        "plus_minus",
        "hits",
        "blocks",
        "faceoff_wins",
        "faceoff_attempts",
        "goalie_saves",
        "goalie_ga",
        "goalie_sa",
        "sog",
        "expected_goals",
        "giveaways",
        "turnovers_forced",
        "created_turnovers",
        "takeaways",
        "controlled_entry_for",
        "controlled_entry_against",
        "controlled_exit_for",
        "controlled_exit_against",
        "gt_goals",
        "gw_goals",
        "ot_goals",
        "ot_assists",
        "gf_counted",
        "ga_counted",
    ]

    now = dt.datetime.now()
    with transaction.atomic():
        # Persist raw player_stats.csv for full-fidelity UI rendering.
        try:
            ps_text_sanitized = logic.sanitize_player_stats_csv_for_storage(ps_text)
            m.HkyGamePlayerStatsCsv.objects.update_or_create(
                game_id=int(game_id),
                defaults={
                    "player_stats_csv": ps_text_sanitized,
                    "source_label": "upload_form",
                    "updated_at": now,
                },
            )
        except Exception:
            pass

        for row in parsed_rows:
            jersey_norm = row.get("jersey_number")
            name_norm = row.get("name_norm") or ""
            pid = _resolve_player_id(jersey_norm, name_norm)
            if pid is None:
                unmatched.append(row.get("player_label") or "")
                continue

            team_id_for_player = player_team_by_id.get(int(pid))
            if team_id_for_player is None:
                unmatched.append(row.get("player_label") or "")
                continue

            stats = row.get("stats") or {}
            if tts_linked:
                stats = dict(stats)
                stats["goals"] = None
                stats["assists"] = None

            defaults = {
                "user_id": int(owner_user_id),
                "team_id": int(team_id_for_player),
                **{c: stats.get(c) for c in cols},
            }
            ps, created = m.PlayerStat.objects.get_or_create(
                game_id=int(game_id),
                player_id=int(pid),
                defaults=defaults,
            )
            if not created:
                updates = {c: stats.get(c) for c in cols if stats.get(c) is not None}
                if updates:
                    m.PlayerStat.objects.filter(id=ps.id).update(**updates)
            imported += 1

        if game_stats is not None:
            try:
                game_stats = logic.filter_game_stats_for_display(game_stats)
            except Exception:
                pass
            m.HkyGameStat.objects.update_or_create(
                game_id=int(game_id),
                defaults={
                    "stats_json": json.dumps(game_stats, ensure_ascii=False),
                    "updated_at": now,
                },
            )

        m.HkyGame.objects.filter(id=int(game_id)).update(stats_imported_at=now)

    if unmatched:
        messages.error(
            f"Imported stats for {imported} player(s). Unmatched: {', '.join([u for u in unmatched if u])}"
        )
    else:
        messages.success(request, f"Imported stats for {imported} player(s).")

    qs = urlencode(
        {"return_to": logic._safe_return_to_url(request.GET.get("return_to"), default="/schedule")}
    )
    return redirect(f"/hky/games/{int(game_id)}?{qs}")


def game_types(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        if name:
            try:
                from django.db import IntegrityError, transaction

                with transaction.atomic():
                    m.GameType.objects.create(name=name, is_default=False)
                messages.success(request, "Game type added")
            except IntegrityError:
                messages.error(request, "Failed to add game type (may already exist)")
        return redirect("/game_types")
    rows = list(m.GameType.objects.order_by("name").values("id", "name", "is_default"))
    return render(request, "game_types.html", {"game_types": rows})


def leagues_index(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    uid = _session_user_id(request)
    admin_ids = set(
        m.LeagueMember.objects.filter(user_id=uid, role__in=["admin", "owner"]).values_list(
            "league_id", flat=True
        )
    )
    leagues: list[dict[str, Any]] = []
    for row in (
        m.League.objects.filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
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
                "is_owner": is_owner,
                "is_admin": is_admin,
            }
        )
    selected_league_id = request.session.get("league_id")
    active_league: Optional[dict[str, Any]] = None
    try:
        selected_id_i = int(selected_league_id) if selected_league_id is not None else None
    except Exception:
        selected_id_i = None
    if selected_id_i is not None:
        for l0 in leagues:
            try:
                if int(l0.get("id") or 0) == int(selected_id_i):
                    active_league = l0
                    break
            except Exception:
                continue
    return render(request, "leagues.html", {"leagues": leagues, "active_league": active_league})


def leagues_new(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    name = str(request.POST.get("name") or "").strip()
    is_shared = 1 if str(request.POST.get("is_shared") or "") == "1" else 0
    is_public = 1 if str(request.POST.get("is_public") or "") == "1" else 0
    if not name:
        messages.error(request, "Name is required")
        return redirect("/leagues")
    _django_orm, m = _orm_modules()
    from django.db import IntegrityError, transaction

    uid = _session_user_id(request)
    now = dt.datetime.now()
    try:
        with transaction.atomic():
            league = m.League.objects.create(
                name=name,
                owner_user_id=uid,
                is_shared=bool(is_shared),
                is_public=bool(is_public),
                created_at=now,
                updated_at=None,
            )
            lid = int(league.id)
            m.LeagueMember.objects.get_or_create(
                league_id=lid,
                user_id=uid,
                defaults={"role": "admin", "created_at": now},
            )
    except IntegrityError:
        messages.error(request, "Failed to create league (name may already exist)")
        return redirect("/leagues")
    request.session["league_id"] = lid
    messages.success(request, "League created and selected")
    return redirect("/leagues")


def leagues_update(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")
    is_shared = 1 if str(request.POST.get("is_shared") or "") == "1" else 0
    is_public = 1 if str(request.POST.get("is_public") or "") == "1" else 0
    _django_orm, m = _orm_modules()
    m.League.objects.filter(id=int(league_id)).update(
        is_shared=bool(is_shared),
        is_public=bool(is_public),
        updated_at=dt.datetime.now(),
    )
    messages.success(request, "League settings updated")
    return redirect("/leagues")


def leagues_delete(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized to delete this league")
        return redirect("/leagues")

    def _chunks(ids: list[int], n: int = 500) -> list[list[int]]:
        return [ids[i : i + n] for i in range(0, len(ids), n)]

    _django_orm, m = _orm_modules()
    from django.db import transaction

    try:
        with transaction.atomic():
            game_ids = list(
                m.LeagueGame.objects.filter(league_id=int(league_id)).values_list(
                    "game_id", flat=True
                )
            )
            team_ids = list(
                m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                    "team_id", flat=True
                )
            )

            m.LeagueMember.objects.filter(league_id=int(league_id)).delete()
            m.LeaguePageView.objects.filter(league_id=int(league_id)).delete()
            m.LeagueGame.objects.filter(league_id=int(league_id)).delete()
            m.LeagueTeam.objects.filter(league_id=int(league_id)).delete()
            m.League.objects.filter(id=int(league_id)).delete()

            if game_ids:
                for chunk in _chunks(sorted({int(x) for x in game_ids}), n=500):
                    m.PlayerPeriodStat.objects.filter(game_id__in=[int(x) for x in chunk]).delete()
                    m.PlayerStat.objects.filter(game_id__in=[int(x) for x in chunk]).delete()
                    m.HkyGamePlayerStatsCsv.objects.filter(
                        game_id__in=[int(x) for x in chunk]
                    ).delete()
                    m.HkyGameEvent.objects.filter(game_id__in=[int(x) for x in chunk]).delete()
                    m.HkyGameStat.objects.filter(game_id__in=[int(x) for x in chunk]).delete()

                still_used = set(
                    m.LeagueGame.objects.filter(game_id__in=[int(x) for x in game_ids]).values_list(
                        "game_id", flat=True
                    )
                )
                safe_game_ids = [int(gid) for gid in game_ids if int(gid) not in still_used]
                for chunk in _chunks(sorted({int(x) for x in safe_game_ids}), n=500):
                    m.HkyGame.objects.filter(id__in=[int(x) for x in chunk]).delete()

            if team_ids:
                eligible_team_ids = list({int(x) for x in team_ids})
                still_used = set(
                    m.LeagueTeam.objects.filter(team_id__in=eligible_team_ids)
                    .exclude(league_id=int(league_id))
                    .values_list("team_id", flat=True)
                )
                still_used |= set(
                    m.HkyGame.objects.filter(team1_id__in=eligible_team_ids).values_list(
                        "team1_id", flat=True
                    )
                )
                still_used |= set(
                    m.HkyGame.objects.filter(team2_id__in=eligible_team_ids).values_list(
                        "team2_id", flat=True
                    )
                )
                safe_team_ids = [
                    int(tid) for tid in eligible_team_ids if int(tid) not in still_used
                ]
                for chunk in _chunks(sorted(safe_team_ids), n=500):
                    m.Team.objects.filter(id__in=[int(x) for x in chunk]).delete()
        if int(request.session.get("league_id") or 0) == int(league_id):
            request.session.pop("league_id", None)
        messages.success(request, "League and associated data deleted")
    except Exception:
        messages.error(request, "Failed to delete league")
    return redirect("/leagues")


def league_members(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")

    _django_orm, m = _orm_modules()

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        role = str(request.POST.get("role") or "viewer")
        if not email:
            messages.error(request, "Email required")
            return redirect(f"/leagues/{int(league_id)}/members")
        from django.db import transaction

        uid = m.User.objects.filter(email=email).values_list("id", flat=True).first()
        if uid is None:
            messages.error(request, "User not found. Ask them to register first.")
            return redirect(f"/leagues/{int(league_id)}/members")
        now = dt.datetime.now()
        with transaction.atomic():
            member, created = m.LeagueMember.objects.get_or_create(
                league_id=int(league_id),
                user_id=int(uid),
                defaults={"role": str(role or "viewer"), "created_at": now},
            )
            if not created and str(getattr(member, "role", "") or "") != str(role or "viewer"):
                m.LeagueMember.objects.filter(id=int(member.id)).update(role=str(role or "viewer"))
        messages.success(request, "Member added/updated")
        return redirect(f"/leagues/{int(league_id)}/members")

    rows = list(
        m.LeagueMember.objects.filter(league_id=int(league_id))
        .select_related("user")
        .order_by("user__email")
        .values("user_id", "user__email", "role")
    )
    members = [
        {"id": int(r0["user_id"]), "email": r0["user__email"], "role": (r0.get("role") or "admin")}
        for r0 in rows
    ]
    return render(request, "league_members.html", {"league_id": int(league_id), "members": members})


def league_members_remove(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")
    uid = int(request.POST.get("user_id") or 0)
    _django_orm, m = _orm_modules()
    m.LeagueMember.objects.filter(league_id=int(league_id), user_id=int(uid)).delete()
    messages.success(request, "Member removed")
    return redirect(f"/leagues/{int(league_id)}/members")


def leagues_recalc_div_ratings(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    if not league_id:
        messages.error(request, "Select an active league first.")
        return redirect("/leagues")
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")
    try:
        logic.recompute_league_mhr_ratings(None, int(league_id))
        messages.success(request, "Ratings recalculated.")
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"Failed to recalculate Ratings: {e}")
    return redirect("/leagues")


def public_leagues_index(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    _django_orm, m = _orm_modules()
    leagues = list(m.League.objects.filter(is_public=True).order_by("name").values("id", "name"))
    return render(request, "public_leagues.html", {"leagues": leagues})


def public_league_home(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    return redirect(f"/public/leagues/{int(league_id)}/teams")


def public_media_team_logo(
    request: HttpRequest, league_id: int, team_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    _django_orm, m = _orm_modules()
    row = (
        m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(team_id))
        .select_related("team")
        .values("team__logo_path")
        .first()
    )
    if not row or not row.get("team__logo_path"):
        raise Http404
    return _safe_file_response(Path(str(row["team__logo_path"])).resolve(), as_attachment=False)


def public_league_teams(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = (
        int(league.get("owner_user_id") or 0) if isinstance(league, dict) else None
    )

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
        entity_id=0,
        viewer_user_id=(viewer_user_id if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )
    is_league_owner = bool(
        viewer_user_id and league_owner_user_id and int(viewer_user_id) == int(league_owner_user_id)
    )

    _django_orm, m = _orm_modules()
    rows_raw = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id))
        .select_related("team")
        .values(
            "team_id",
            "team__user_id",
            "team__name",
            "team__logo_path",
            "team__is_external",
            "team__created_at",
            "team__updated_at",
            "division_name",
            "division_id",
            "conference_id",
            "mhr_rating",
            "mhr_agd",
            "mhr_sched",
            "mhr_games",
            "mhr_updated_at",
        )
    )
    rows: list[dict[str, Any]] = []
    for r0 in rows_raw:
        rows.append(
            {
                "id": int(r0["team_id"]),
                "user_id": int(r0["team__user_id"]),
                "name": r0.get("team__name"),
                "logo_path": r0.get("team__logo_path"),
                "is_external": bool(r0.get("team__is_external")),
                "created_at": r0.get("team__created_at"),
                "updated_at": r0.get("team__updated_at"),
                "division_name": r0.get("division_name"),
                "division_id": r0.get("division_id"),
                "conference_id": r0.get("conference_id"),
                "mhr_rating": r0.get("mhr_rating"),
                "mhr_agd": r0.get("mhr_agd"),
                "mhr_sched": r0.get("mhr_sched"),
                "mhr_games": r0.get("mhr_games"),
                "mhr_updated_at": r0.get("mhr_updated_at"),
            }
        )

    stats: dict[int, dict[str, Any]] = {}
    for t in rows:
        tid = int(t["id"])
        try:
            stats[tid] = logic.compute_team_stats_league(None, tid, int(league_id))
            stats[tid]["gp"] = (
                int(stats[tid].get("wins", 0) or 0)
                + int(stats[tid].get("losses", 0) or 0)
                + int(stats[tid].get("ties", 0) or 0)
            )
        except Exception:
            stats[tid] = {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for t in rows:
        dn = str(t.get("division_name") or "").strip() or "Unknown Division"
        grouped.setdefault(dn, []).append(t)
    divisions = []
    for dn in sorted(grouped.keys(), key=logic.division_sort_key):
        teams_sorted = sorted(
            grouped[dn],
            key=lambda tr: logic.sort_key_team_standings(tr, stats.get(int(tr["id"]), {})),
        )
        divisions.append({"name": dn, "teams": teams_sorted})

    league_page_views = None
    if is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
            "entity_id": 0,
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
            ),
        }
    return render(
        request,
        "teams.html",
        {
            "teams": rows,
            "divisions": divisions,
            "stats": stats,
            "include_external": True,
            "league_view": True,
            "public_league_id": int(league_id),
            "league_page_views": league_page_views,
        },
    )


def public_league_team_detail(
    request: HttpRequest, league_id: int, team_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = None
    try:
        league_owner_user_id = (
            int(league.get("owner_user_id")) if isinstance(league, dict) else None
        )
    except Exception:
        league_owner_user_id = None
    is_league_owner = bool(
        viewer_user_id
        and league_owner_user_id is not None
        and int(viewer_user_id) == int(league_owner_user_id)
    )

    recent_n_raw = request.GET.get("recent_n")
    try:
        recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
    except Exception:
        recent_n = 5
    recent_sort = str(request.GET.get("recent_sort") or "points").strip() or "points"
    recent_dir = str(request.GET.get("recent_dir") or "desc").strip().lower() or "desc"

    _django_orm, m = _orm_modules()
    team = (
        m.Team.objects.filter(id=int(team_id), league_teams__league_id=int(league_id))
        .values(
            "id",
            "user_id",
            "name",
            "logo_path",
            "is_external",
            "created_at",
            "updated_at",
        )
        .first()
    )
    if not team:
        raise Http404

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
        entity_id=int(team_id),
        viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )

    players = list(
        m.Player.objects.filter(team_id=int(team_id))
        .order_by("jersey_number", "name")
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    skaters, goalies, head_coaches, assistant_coaches = logic.split_roster(players or [])
    roster_players = list(skaters) + list(goalies)
    tstats = logic.compute_team_stats_league(None, int(team_id), int(league_id))

    from django.db.models import Q

    league_team_div_map = {
        int(tid): (str(dn).strip() if dn is not None else None)
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    schedule_rows_raw = list(
        m.LeagueGame.objects.filter(league_id=int(league_id))
        .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
        .select_related("game", "game__team1", "game__team2", "game__game_type")
        .values(
            "game_id",
            "division_name",
            "sort_order",
            "game__user_id",
            "game__team1_id",
            "game__team2_id",
            "game__game_type_id",
            "game__starts_at",
            "game__location",
            "game__notes",
            "game__team1_score",
            "game__team2_score",
            "game__is_final",
            "game__stats_imported_at",
            "game__created_at",
            "game__updated_at",
            "game__team1__name",
            "game__team2__name",
            "game__game_type__name",
        )
    )
    schedule_games: list[dict[str, Any]] = []
    for r0 in schedule_rows_raw:
        t1 = int(r0["game__team1_id"])
        t2 = int(r0["game__team2_id"])
        schedule_games.append(
            {
                "id": int(r0["game_id"]),
                "user_id": int(r0["game__user_id"]),
                "team1_id": t1,
                "team2_id": t2,
                "game_type_id": r0.get("game__game_type_id"),
                "starts_at": r0.get("game__starts_at"),
                "location": r0.get("game__location"),
                "notes": r0.get("game__notes"),
                "team1_score": r0.get("game__team1_score"),
                "team2_score": r0.get("game__team2_score"),
                "is_final": r0.get("game__is_final"),
                "stats_imported_at": r0.get("game__stats_imported_at"),
                "created_at": r0.get("game__created_at"),
                "updated_at": r0.get("game__updated_at"),
                "team1_name": r0.get("game__team1__name"),
                "team2_name": r0.get("game__team2__name"),
                "game_type_name": r0.get("game__game_type__name"),
                "division_name": r0.get("division_name"),
                "sort_order": r0.get("sort_order"),
                "team1_league_division_name": league_team_div_map.get(t1),
                "team2_league_division_name": league_team_div_map.get(t2),
            }
        )

    schedule_games = [
        g2
        for g2 in (schedule_games or [])
        if not logic._league_game_is_cross_division_non_external(g2)
    ]
    now_dt = dt.datetime.now()
    for g2 in schedule_games:
        sdt = g2.get("starts_at")
        started = False
        if sdt is not None:
            try:
                started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
            except Exception:
                started = False
        has_score = (
            (g2.get("team1_score") is not None)
            or (g2.get("team2_score") is not None)
            or bool(g2.get("is_final"))
        )
        g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None
    schedule_games = logic.sort_games_schedule_order(schedule_games or [])

    schedule_game_ids = [
        int(g2.get("id")) for g2 in (schedule_games or []) if g2.get("id") is not None
    ]
    ps_rows = list(
        m.PlayerStat.objects.filter(team_id=int(team_id), game_id__in=schedule_game_ids).values(
            "player_id", "game_id", *logic.PLAYER_STATS_SUM_KEYS
        )
    )

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"
    # Tournament-only players: show them on game pages, but not on team/division-level roster/stats.
    try:
        tournament_game_ids: set[int] = {
            int(g2.get("id"))
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "").strip().casefold() == "tournament"
            and g2.get("id") is not None
        }
        player_ids_with_any_stats: set[int] = set()
        player_ids_with_non_tournament_stats: set[int] = set()
        for r0 in ps_rows or []:
            try:
                pid_i = int(r0.get("player_id"))
                gid_i = int(r0.get("game_id"))
            except Exception:
                continue
            player_ids_with_any_stats.add(pid_i)
            if gid_i not in tournament_game_ids:
                player_ids_with_non_tournament_stats.add(pid_i)
        tournament_only_player_ids = (
            player_ids_with_any_stats - player_ids_with_non_tournament_stats
        )
    except Exception:
        tournament_only_player_ids = set()

    if tournament_only_player_ids:
        skaters = [p for p in skaters if int(p.get("id") or 0) not in tournament_only_player_ids]
        goalies = [p for p in goalies if int(p.get("id") or 0) not in tournament_only_player_ids]
        roster_players = list(skaters) + list(goalies)
        ps_rows = [
            r0 for r0 in ps_rows if int(r0.get("player_id") or 0) not in tournament_only_player_ids
        ]
    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [g2 for g2 in stats_schedule_games if logic._game_has_recorded_result(g2)]
    eligible_game_ids_in_order: list[int] = []
    for g2 in eligible_games:
        try:
            eligible_game_ids_in_order.append(int(g2.get("id")))
        except Exception:
            continue
    eligible_game_ids: set[int] = set(eligible_game_ids_in_order)
    ps_rows_filtered = []
    for r0 in ps_rows or []:
        try:
            if int(r0.get("game_id")) in eligible_game_ids:
                ps_rows_filtered.append(r0)
        except Exception:
            continue

    player_totals = logic._aggregate_player_totals_from_rows(
        player_stats_rows=ps_rows_filtered, allowed_game_ids=eligible_game_ids
    )
    player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(skaters, player_totals)
    )
    player_stats_columns = logic.filter_player_stats_display_columns_for_rows(
        logic.PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows
    )
    cov_counts, cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows_filtered, eligible_game_ids=eligible_game_ids_in_order
    )
    player_stats_columns = logic._player_stats_columns_with_coverage(
        columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
    )

    recent_scope_ids = (
        eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
    )
    recent_totals = logic.compute_recent_player_totals_from_rows(
        schedule_games=stats_schedule_games, player_stats_rows=ps_rows_filtered, n=recent_n
    )
    recent_player_stats_rows = logic.sort_player_stats_rows(
        logic.build_player_stats_table_rows(skaters, recent_totals),
        sort_key=recent_sort,
        sort_dir=recent_dir,
    )
    recent_player_stats_columns = logic.filter_player_stats_display_columns_for_rows(
        logic.PLAYER_STATS_DISPLAY_COLUMNS, recent_player_stats_rows
    )
    recent_cov_counts, recent_cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows_filtered, eligible_game_ids=recent_scope_ids
    )
    recent_player_stats_columns = logic._player_stats_columns_with_coverage(
        columns=recent_player_stats_columns,
        coverage_counts=recent_cov_counts,
        total_games=recent_cov_total,
    )

    player_stats_sources = logic._compute_team_player_stats_sources(
        None, eligible_game_ids=eligible_game_ids_in_order
    )
    selected_label = (
        "All"
        if selected_types is None
        else ", ".join(sorted(list(selected_types), key=lambda s: s.lower()))
    )
    game_type_filter_options = [
        {"label": gt, "checked": (selected_types is None) or (gt in selected_types)}
        for gt in game_type_options
    ]
    league_page_views = None
    if is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            "entity_id": int(team_id),
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM, entity_id=int(team_id)
            ),
        }
    return render(
        request,
        "team_detail.html",
        {
            "team": team,
            "roster_players": roster_players,
            "players": skaters,
            "head_coaches": head_coaches,
            "assistant_coaches": assistant_coaches,
            "player_stats_columns": player_stats_columns,
            "player_stats_rows": player_stats_rows,
            "recent_player_stats_columns": recent_player_stats_columns,
            "recent_player_stats_rows": recent_player_stats_rows,
            "recent_n": recent_n,
            "recent_sort": recent_sort,
            "recent_dir": recent_dir,
            "tstats": tstats,
            "schedule_games": schedule_games,
            "editable": False,
            "public_league_id": int(league_id),
            "player_stats_sources": player_stats_sources,
            "player_stats_coverage_total_games": cov_total,
            "player_stats_recent_coverage_total_games": recent_cov_total,
            "game_type_filter_options": game_type_filter_options,
            "game_type_filter_label": selected_label,
            "league_page_views": league_page_views,
        },
    )


def public_league_schedule(
    request: HttpRequest, league_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = None
    try:
        league_owner_user_id = (
            int(league.get("owner_user_id")) if isinstance(league, dict) else None
        )
    except Exception:
        league_owner_user_id = None
    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
        entity_id=0,
        viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )
    is_league_owner = bool(
        viewer_user_id
        and league_owner_user_id is not None
        and int(viewer_user_id) == int(league_owner_user_id)
    )

    selected_division = str(request.GET.get("division") or "").strip() or None
    selected_team_id = request.GET.get("team_id") or ""
    team_id_i: Optional[int] = None
    try:
        team_id_i = int(selected_team_id) if str(selected_team_id).strip() else None
    except Exception:
        team_id_i = None

    _django_orm, m = _orm_modules()
    divisions = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id))
        .exclude(division_name__isnull=True)
        .exclude(division_name="")
        .values_list("division_name", flat=True)
        .distinct()
    )
    divisions.sort(key=logic.division_sort_key)
    if selected_division:
        league_teams = list(
            m.Team.objects.filter(
                league_teams__league_id=int(league_id),
                league_teams__division_name=str(selected_division),
            )
            .distinct()
            .order_by("name")
            .values("id", "name")
        )
    else:
        league_teams = list(
            m.Team.objects.filter(league_teams__league_id=int(league_id))
            .distinct()
            .order_by("name")
            .values("id", "name")
        )
    if team_id_i is not None and not any(int(t["id"]) == int(team_id_i) for t in league_teams):
        team_id_i = None
        selected_team_id = ""

    from django.db.models import Q

    lg_qs = m.LeagueGame.objects.filter(league_id=int(league_id))
    if selected_division:
        lg_qs = lg_qs.filter(division_name=str(selected_division))
    if team_id_i is not None:
        lg_qs = lg_qs.filter(Q(game__team1_id=int(team_id_i)) | Q(game__team2_id=int(team_id_i)))

    league_team_div_map = {
        int(tid): (str(dn).strip() if dn is not None else None)
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    rows_raw = list(
        lg_qs.select_related("game", "game__team1", "game__team2", "game__game_type").values(
            "game_id",
            "division_name",
            "sort_order",
            "game__user_id",
            "game__team1_id",
            "game__team2_id",
            "game__game_type_id",
            "game__starts_at",
            "game__location",
            "game__notes",
            "game__team1_score",
            "game__team2_score",
            "game__is_final",
            "game__stats_imported_at",
            "game__created_at",
            "game__updated_at",
            "game__team1__name",
            "game__team2__name",
            "game__game_type__name",
        )
    )
    games: list[dict[str, Any]] = []
    for r0 in rows_raw:
        t1 = int(r0["game__team1_id"])
        t2 = int(r0["game__team2_id"])
        games.append(
            {
                "id": int(r0["game_id"]),
                "user_id": int(r0["game__user_id"]),
                "team1_id": t1,
                "team2_id": t2,
                "game_type_id": r0.get("game__game_type_id"),
                "starts_at": r0.get("game__starts_at"),
                "location": r0.get("game__location"),
                "notes": r0.get("game__notes"),
                "team1_score": r0.get("game__team1_score"),
                "team2_score": r0.get("game__team2_score"),
                "is_final": r0.get("game__is_final"),
                "stats_imported_at": r0.get("game__stats_imported_at"),
                "created_at": r0.get("game__created_at"),
                "updated_at": r0.get("game__updated_at"),
                "team1_name": r0.get("game__team1__name"),
                "team2_name": r0.get("game__team2__name"),
                "game_type_name": r0.get("game__game_type__name"),
                "division_name": r0.get("division_name"),
                "sort_order": r0.get("sort_order"),
                "team1_league_division_name": league_team_div_map.get(t1),
                "team2_league_division_name": league_team_div_map.get(t2),
            }
        )
    games = [
        g2 for g2 in (games or []) if not logic._league_game_is_cross_division_non_external(g2)
    ]
    now_dt = dt.datetime.now()
    for g2 in games or []:
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None
        sdt = g2.get("starts_at")
        started = False
        if sdt is not None:
            try:
                started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
            except Exception:
                started = False
        has_score = (
            (g2.get("team1_score") is not None)
            or (g2.get("team2_score") is not None)
            or bool(g2.get("is_final"))
        )
        g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
        g2["can_edit"] = False
    games = logic.sort_games_schedule_order(games or [])

    league_page_views = None
    if is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            "entity_id": 0,
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
            ),
        }

    return render(
        request,
        "schedule.html",
        {
            "games": games,
            "league_view": True,
            "divisions": divisions,
            "league_teams": league_teams,
            "selected_division": selected_division or "",
            "selected_team_id": str(team_id_i) if team_id_i is not None else "",
            "can_add_game": False,
            "public_league_id": int(league_id),
            "league_page_views": league_page_views,
        },
    )


def public_hky_game_detail(
    request: HttpRequest, league_id: int, game_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = None
    try:
        league_owner_user_id = (
            int(league.get("owner_user_id")) if isinstance(league, dict) else None
        )
    except Exception:
        league_owner_user_id = None
    is_league_owner = bool(
        viewer_user_id
        and league_owner_user_id is not None
        and int(viewer_user_id) == int(league_owner_user_id)
    )

    _django_orm, m = _orm_modules()
    row = (
        m.LeagueGame.objects.filter(league_id=int(league_id), game_id=int(game_id))
        .select_related("game", "game__team1", "game__team2")
        .values(
            "game_id",
            "division_name",
            "game__user_id",
            "game__team1_id",
            "game__team2_id",
            "game__game_type_id",
            "game__starts_at",
            "game__location",
            "game__notes",
            "game__team1_score",
            "game__team2_score",
            "game__is_final",
            "game__stats_imported_at",
            "game__created_at",
            "game__updated_at",
            "game__team1__name",
            "game__team2__name",
            "game__team1__is_external",
            "game__team2__is_external",
        )
        .first()
    )
    if not row:
        raise Http404

    team1_id = int(row["game__team1_id"])
    team2_id = int(row["game__team2_id"])
    div_map = {
        int(tid): (str(dn).strip() if dn is not None else None)
        for tid, dn in m.LeagueTeam.objects.filter(
            league_id=int(league_id),
            team_id__in=[team1_id, team2_id],
        ).values_list("team_id", "division_name")
    }
    game = {
        "id": int(row["game_id"]),
        "user_id": int(row["game__user_id"]),
        "team1_id": team1_id,
        "team2_id": team2_id,
        "game_type_id": row.get("game__game_type_id"),
        "starts_at": row.get("game__starts_at"),
        "location": row.get("game__location"),
        "notes": row.get("game__notes"),
        "team1_score": row.get("game__team1_score"),
        "team2_score": row.get("game__team2_score"),
        "is_final": row.get("game__is_final"),
        "stats_imported_at": row.get("game__stats_imported_at"),
        "created_at": row.get("game__created_at"),
        "updated_at": row.get("game__updated_at"),
        "team1_name": row.get("game__team1__name"),
        "team2_name": row.get("game__team2__name"),
        "team1_ext": row.get("game__team1__is_external"),
        "team2_ext": row.get("game__team2__is_external"),
        "division_name": row.get("division_name"),
        "team1_league_division_name": div_map.get(team1_id),
        "team2_league_division_name": div_map.get(team2_id),
    }
    try:
        game["game_video_url"] = logic._sanitize_http_url(
            logic._extract_game_video_url_from_notes(game.get("notes"))
        )
    except Exception:
        game["game_video_url"] = None
    if logic._league_game_is_cross_division_non_external(game):
        raise Http404

    now_dt = dt.datetime.now()
    sdt = game.get("starts_at")
    started = False
    if sdt is not None:
        try:
            started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
        except Exception:
            started = False
    has_score = (
        (game.get("team1_score") is not None)
        or (game.get("team2_score") is not None)
        or bool(game.get("is_final"))
    )
    can_view_summary = bool(has_score or (sdt is None) or started)
    if not can_view_summary:
        raise Http404

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME,
        entity_id=int(game_id),
        viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )

    team1_players = list(
        m.Player.objects.filter(team_id=int(game["team1_id"]))
        .order_by("jersey_number", "name")
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    team2_players = list(
        m.Player.objects.filter(team_id=int(game["team2_id"]))
        .order_by("jersey_number", "name")
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    stats_rows = list(m.PlayerStat.objects.filter(game_id=int(game_id)).values())
    game_stats_row = (
        m.HkyGameStat.objects.filter(game_id=int(game_id))
        .values("stats_json", "updated_at")
        .first()
    )
    team1_skaters, team1_goalies, team1_hc, team1_ac = logic.split_roster(team1_players)
    team2_skaters, team2_goalies, team2_hc, team2_ac = logic.split_roster(team2_players)
    team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
    team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
    stats_by_pid = {r0["player_id"]: r0 for r0 in stats_rows}

    game_stats = None
    game_stats_updated_at = None
    try:
        if game_stats_row and game_stats_row.get("stats_json"):
            game_stats = json.loads(game_stats_row["stats_json"])
            game_stats_updated_at = game_stats_row.get("updated_at")
    except Exception:
        game_stats = None
    game_stats = logic.filter_game_stats_for_display(game_stats)

    period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}
    tts_linked = logic._extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

    events_headers: list[str] = []
    events_rows: list[dict[str, str]] = []
    events_meta: Optional[dict[str, Any]] = None
    try:
        erow = (
            m.HkyGameEvent.objects.filter(game_id=int(game_id))
            .values("events_csv", "source_label", "updated_at")
            .first()
        )
        if erow and str(erow.get("events_csv") or "").strip():
            events_headers, events_rows = logic.parse_events_csv(str(erow.get("events_csv") or ""))
            events_headers, events_rows = logic.normalize_game_events_csv(
                events_headers, events_rows
            )
            events_rows = logic.filter_events_rows_prefer_timetoscore_for_goal_assist(
                events_rows, tts_linked=tts_linked
            )
            events_headers, events_rows = logic.normalize_events_video_time_for_display(
                events_headers, events_rows
            )
            events_headers, events_rows = logic.filter_events_headers_drop_empty_on_ice_split(
                events_headers, events_rows
            )
            events_rows = logic.sort_events_rows_default(events_rows)
            events_meta = {
                "source_label": erow.get("source_label"),
                "updated_at": erow.get("updated_at"),
                "count": len(events_rows),
                "sources": logic.summarize_event_sources(
                    events_rows, fallback_source_label=str(erow.get("source_label") or "")
                ),
            }
    except Exception:
        events_headers, events_rows, events_meta = [], [], None

    scoring_by_period_rows = logic.compute_team_scoring_by_period_from_events(
        events_rows, tts_linked=tts_linked
    )
    game_event_stats_rows = logic.compute_game_event_stats_by_side(events_rows)

    imported_player_stats_csv_text: Optional[str] = None
    player_stats_import_meta: Optional[dict[str, Any]] = None
    try:
        prow = (
            m.HkyGamePlayerStatsCsv.objects.filter(game_id=int(game_id))
            .values("player_stats_csv", "source_label", "updated_at")
            .first()
        )
        if prow and str(prow.get("player_stats_csv") or "").strip():
            imported_player_stats_csv_text = str(prow.get("player_stats_csv") or "")
            player_stats_import_meta = {
                "source_label": prow.get("source_label"),
                "updated_at": prow.get("updated_at"),
            }
    except Exception:
        imported_player_stats_csv_text, player_stats_import_meta = None, None

    (
        game_player_stats_columns,
        player_stats_cells_by_pid,
        player_stats_cell_conflicts_by_pid,
        player_stats_import_warning,
    ) = logic.build_game_player_stats_table(
        players=list(team1_skaters) + list(team2_skaters),
        stats_by_pid=stats_by_pid,
        imported_csv_text=imported_player_stats_csv_text,
        prefer_db_stats_for_keys={"goals", "assists"} if tts_linked else None,
    )
    team1_skaters_sorted = list(team1_skaters)
    team2_skaters_sorted = list(team2_skaters)
    try:

        def _sort_players_for_game(players_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows = []
            by_pid = {}
            for p in players_in:
                pid = int(p.get("id"))
                cells = (player_stats_cells_by_pid or {}).get(pid, {})
                g = logic._parse_int_from_cell_text(cells.get("goals", ""))
                a = logic._parse_int_from_cell_text(cells.get("assists", ""))
                rows.append(
                    {
                        "player_id": pid,
                        "name": str(p.get("name") or ""),
                        "goals": g,
                        "assists": a,
                        "points": g + a,
                    }
                )
                by_pid[pid] = p
            sorted_rows = logic.sort_players_table_default(rows)
            return [
                by_pid[int(r0["player_id"])] for r0 in sorted_rows if int(r0["player_id"]) in by_pid
            ]

        team1_skaters_sorted = _sort_players_for_game(team1_skaters_sorted)
        team2_skaters_sorted = _sort_players_for_game(team2_skaters_sorted)
    except Exception:
        team1_skaters_sorted = list(team1_skaters)
        team2_skaters_sorted = list(team2_skaters)

    default_back_url = f"/public/leagues/{int(league_id)}/schedule"
    return_to = logic._safe_return_to_url(request.GET.get("return_to"), default=default_back_url)
    public_is_logged_in = bool(viewer_user_id)

    league_page_views = None
    if is_league_owner:
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_GAME,
            "entity_id": int(game_id),
            "count": logic._get_league_page_view_count(
                None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME, entity_id=int(game_id)
            ),
        }

    return render(
        request,
        "hky_game_detail.html",
        {
            "game": game,
            "team1_roster": team1_roster,
            "team2_roster": team2_roster,
            "team1_players": team1_skaters_sorted,
            "team2_players": team2_skaters_sorted,
            "stats_by_pid": stats_by_pid,
            "period_stats_by_pid": period_stats_by_pid,
            "game_stats": game_stats,
            "game_stats_updated_at": game_stats_updated_at,
            "editable": False,
            "can_edit": False,
            "edit_mode": False,
            "public_league_id": int(league_id),
            "back_url": return_to,
            "return_to": return_to,
            "events_headers": events_headers,
            "events_rows": events_rows,
            "events_meta": events_meta,
            "scoring_by_period_rows": scoring_by_period_rows,
            "game_event_stats_rows": game_event_stats_rows,
            "user_video_clip_len_s": (
                logic.get_user_video_clip_len_s(None, int(viewer_user_id))
                if public_is_logged_in
                else None
            ),
            "user_is_logged_in": public_is_logged_in,
            "game_player_stats_columns": game_player_stats_columns,
            "player_stats_cells_by_pid": player_stats_cells_by_pid,
            "player_stats_cell_conflicts_by_pid": player_stats_cell_conflicts_by_pid,
            "player_stats_import_meta": player_stats_import_meta,
            "player_stats_import_warning": player_stats_import_warning,
            "league_page_views": league_page_views,
        },
    )


# ----------------------------
# Import/auth helpers (API)
# ----------------------------


def _get_import_token() -> Optional[str]:
    token = os.environ.get("HM_WEBAPP_IMPORT_TOKEN")
    if token:
        return str(token)
    try:
        cfg_path = os.environ.get("HM_DB_CONFIG", str(logic.CONFIG_PATH))
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        t = cfg.get("import_token")
        return str(t) if t else None
    except Exception:
        return None


def _require_import_auth(request: HttpRequest) -> Optional[JsonResponse]:
    required = _get_import_token()
    if required:
        supplied = None
        auth = str(request.META.get("HTTP_AUTHORIZATION") or "").strip()
        if auth.lower().startswith("bearer "):
            supplied = auth.split(" ", 1)[1].strip()
        if not supplied:
            supplied = str(
                request.META.get("HTTP_X_HM_IMPORT_TOKEN") or request.GET.get("token") or ""
            ).strip()
        required_s = str(required or "").strip()
        if not supplied or not secrets.compare_digest(str(supplied), required_s):
            return JsonResponse({"ok": False, "error": "unauthorized"}, status=401)
        return None

    if request.META.get("HTTP_X_FORWARDED_FOR"):
        return JsonResponse({"ok": False, "error": "import_token_required"}, status=403)
    if str(request.META.get("REMOTE_ADDR") or "") not in ("127.0.0.1", "::1"):
        return JsonResponse({"ok": False, "error": "import_token_required"}, status=403)
    return None


def _ensure_user_for_import(email: str, name: Optional[str] = None) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm:
        raise ValueError("owner_email is required")
    _django_orm, m = _orm_modules()
    existing = m.User.objects.filter(email=email_norm).values_list("id", flat=True).first()
    if existing is not None:
        return int(existing)
    pwd = generate_password_hash(secrets.token_hex(24))
    u = m.User.objects.create(
        email=email_norm,
        password_hash=pwd,
        name=(name or email_norm),
        created_at=dt.datetime.now(),
    )
    return int(u.id)


def _ensure_league_for_import(
    *,
    league_name: str,
    owner_user_id: int,
    is_shared: Optional[bool],
    source: Optional[str],
    external_key: Optional[str],
    commit: bool = True,
) -> int:
    del commit
    name = (league_name or "").strip()
    if not name:
        raise ValueError("league_name is required")
    _django_orm, m = _orm_modules()
    existing = (
        m.League.objects.filter(name=name)
        .values("id", "is_shared", "source", "external_key")
        .first()
    )
    now = dt.datetime.now()
    if existing:
        updates: dict[str, Any] = {}
        if is_shared is not None and bool(existing.get("is_shared")) != bool(is_shared):
            updates["is_shared"] = bool(is_shared)
        if source is not None and str(existing.get("source") or "") != str(source or ""):
            updates["source"] = source
        if external_key is not None and str(existing.get("external_key") or "") != str(
            external_key or ""
        ):
            updates["external_key"] = external_key
        if updates:
            updates["updated_at"] = now
            m.League.objects.filter(id=int(existing["id"])).update(**updates)
        return int(existing["id"])

    if is_shared is None:
        is_shared = True
    league = m.League.objects.create(
        name=name,
        owner_user_id=int(owner_user_id),
        is_shared=bool(is_shared),
        source=source,
        external_key=external_key,
        created_at=now,
        updated_at=None,
    )
    return int(league.id)


def _ensure_league_member_for_import(
    league_id: int, user_id: int, role: str, *, commit: bool = True
) -> None:
    del commit
    _django_orm, m = _orm_modules()
    m.LeagueMember.objects.get_or_create(
        league_id=int(league_id),
        user_id=int(user_id),
        defaults={"role": str(role or "viewer"), "created_at": dt.datetime.now()},
    )


def _normalize_import_game_type_name(raw: Any) -> Optional[str]:
    s = str(raw or "").strip()
    if not s:
        return None
    sl = s.casefold()
    if sl.startswith("regular"):
        return "Regular Season"
    if sl.startswith("preseason"):
        return "Preseason"
    if sl.startswith("exhibition"):
        return "Exhibition"
    if sl.startswith("tournament"):
        return "Tournament"
    return s


def _ensure_game_type_id_for_import(game_type_name: Any) -> Optional[int]:
    nm = _normalize_import_game_type_name(game_type_name)
    if not nm:
        return None
    _django_orm, m = _orm_modules()
    existing = m.GameType.objects.filter(name=str(nm)).values_list("id", flat=True).first()
    if existing is not None:
        return int(existing)
    gt = m.GameType.objects.create(name=str(nm), is_default=False)
    return int(gt.id)


def _ensure_external_team_for_import(owner_user_id: int, name: str, *, commit: bool = True) -> int:
    del commit

    def _norm_team_name(s: str) -> str:
        t = str(s or "").replace("\xa0", " ").strip()
        t = (
            t.replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2212", "-")
        )
        t = " ".join(t.split())
        t = re.sub(r"\s*\(\s*external\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
        t = t.casefold()
        t = re.sub(r"[^0-9a-z]+", " ", t)
        return " ".join(t.split())

    nm = _norm_team_name(name or "")
    if not nm:
        nm = "unknown"
    _django_orm, m = _orm_modules()
    raw_name = str(name or "").strip()
    exact = (
        m.Team.objects.filter(user_id=int(owner_user_id), name=raw_name)
        .values_list("id", flat=True)
        .first()
    )
    if exact is not None:
        return int(exact)
    for row in m.Team.objects.filter(user_id=int(owner_user_id)).values("id", "name"):
        if _norm_team_name(str(row.get("name") or "")) == nm:
            return int(row["id"])
    t = m.Team.objects.create(
        user_id=int(owner_user_id),
        name=raw_name or "UNKNOWN",
        is_external=True,
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(t.id)


def _ensure_player_for_import(
    owner_user_id: int,
    team_id: int,
    name: str,
    jersey_number: Optional[str],
    position: Optional[str],
    *,
    commit: bool = True,
) -> int:
    del commit
    nm = (name or "").strip()
    if not nm:
        raise ValueError("player name is required")
    _django_orm, m = _orm_modules()
    existing = (
        m.Player.objects.filter(user_id=int(owner_user_id), team_id=int(team_id), name=nm)
        .values_list("id", flat=True)
        .first()
    )
    if existing is not None:
        pid = int(existing)
        if jersey_number or position:
            updates: dict[str, Any] = {"updated_at": dt.datetime.now()}
            if jersey_number:
                updates["jersey_number"] = jersey_number
            if position:
                updates["position"] = position
            m.Player.objects.filter(id=pid).update(**updates)
        return pid

    p = m.Player.objects.create(
        user_id=int(owner_user_id),
        team_id=int(team_id),
        name=nm,
        jersey_number=jersey_number,
        position=position,
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(p.id)


def _merge_notes(existing: Optional[str], new_fields: dict[str, Any]) -> str:
    if not existing:
        return json.dumps(new_fields, sort_keys=True)
    try:
        cur = json.loads(existing)
        if isinstance(cur, dict):
            cur.update(new_fields)
            return json.dumps(cur, sort_keys=True)
    except Exception:
        pass
    return str(existing)


def _update_game_video_url_note(
    game_id: int, video_url: str, *, replace: bool, commit: bool = True
) -> None:
    del commit
    url = logic._sanitize_http_url(video_url)
    if not url:
        return
    _django_orm, m = _orm_modules()
    existing = str(
        m.HkyGame.objects.filter(id=int(game_id)).values_list("notes", flat=True).first() or ""
    ).strip()
    existing_url = logic._extract_game_video_url_from_notes(existing)
    if existing_url and not replace:
        return

    new_notes: str
    try:
        d = json.loads(existing) if existing else {}
        if isinstance(d, dict):
            d["game_video_url"] = url
            new_notes = json.dumps(d, sort_keys=True)
        else:
            raise ValueError("notes not dict")
    except Exception:
        suffix = f" game_video_url={url}"
        if existing and suffix.strip() in existing:
            new_notes = existing
        else:
            new_notes = (existing + "\n" + suffix.strip()).strip() if existing else suffix.strip()

    m.HkyGame.objects.filter(id=int(game_id)).update(notes=new_notes, updated_at=dt.datetime.now())


def _upsert_game_for_import(
    *,
    owner_user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[str],
    location: Optional[str],
    team1_score: Optional[int],
    team2_score: Optional[int],
    replace: bool,
    notes_json_fields: dict[str, Any],
    commit: bool = True,
) -> int:
    del commit
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    starts_dt = logic.to_dt(starts_at) if starts_at else None

    tts_int: Optional[int]
    try:
        tts_int = (
            int(notes_json_fields["timetoscore_game_id"])
            if notes_json_fields.get("timetoscore_game_id") is not None
            else None
        )
    except Exception:
        tts_int = None
    ext_key = str(notes_json_fields.get("external_game_key") or "").strip() or None

    existing_by_tts = None
    if tts_int is not None:
        existing_by_tts = (
            m.HkyGame.objects.filter(timetoscore_game_id=int(tts_int))
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        if existing_by_tts is None:
            token_json_nospace = f'"timetoscore_game_id":{int(tts_int)}'
            token_json_space = f'"timetoscore_game_id": {int(tts_int)}'
            token_plain = f"game_id={int(tts_int)}"
            existing_by_tts = (
                m.HkyGame.objects.filter(
                    Q(notes__contains=token_json_nospace)
                    | Q(notes__contains=token_json_space)
                    | Q(notes__contains=token_plain)
                )
                .values(
                    "id",
                    "notes",
                    "team1_score",
                    "team2_score",
                    "timetoscore_game_id",
                    "external_game_key",
                )
                .first()
            )

    existing_by_ext = None
    if ext_key:
        existing_by_ext = (
            m.HkyGame.objects.filter(user_id=int(owner_user_id), external_game_key=str(ext_key))
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        if existing_by_ext is None:
            try:
                ext_json = json.dumps(str(ext_key))
            except Exception:
                ext_json = f'"{str(ext_key)}"'
            token1 = f'"external_game_key":{ext_json}'
            token2 = f'"external_game_key": {ext_json}'
            existing_by_ext = (
                m.HkyGame.objects.filter(user_id=int(owner_user_id))
                .filter(Q(notes__contains=token1) | Q(notes__contains=token2))
                .values(
                    "id",
                    "notes",
                    "team1_score",
                    "team2_score",
                    "timetoscore_game_id",
                    "external_game_key",
                )
                .first()
            )

    if (
        existing_by_tts
        and existing_by_ext
        and int(existing_by_tts["id"]) != int(existing_by_ext["id"])
    ):
        _django_orm.merge_hky_games(
            keep_id=int(existing_by_tts["id"]), drop_id=int(existing_by_ext["id"])
        )
        existing_by_tts = (
            m.HkyGame.objects.filter(id=int(existing_by_tts["id"]))
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        existing_by_ext = None

    existing_by_time = None
    if starts_dt is not None:
        existing_by_time = (
            m.HkyGame.objects.filter(
                user_id=int(owner_user_id),
                team1_id=int(team1_id),
                team2_id=int(team2_id),
                starts_at=starts_dt,
            )
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )

    existing_row = existing_by_tts or existing_by_ext or existing_by_time

    now = dt.datetime.now()
    if existing_row is None:
        notes = json.dumps(notes_json_fields, sort_keys=True)
        g = m.HkyGame.objects.create(
            user_id=int(owner_user_id),
            team1_id=int(team1_id),
            team2_id=int(team2_id),
            game_type_id=int(game_type_id) if game_type_id is not None else None,
            starts_at=starts_dt,
            location=location,
            team1_score=team1_score,
            team2_score=team2_score,
            is_final=bool(team1_score is not None and team2_score is not None),
            notes=notes,
            stats_imported_at=now,
            timetoscore_game_id=tts_int,
            external_game_key=ext_key,
            created_at=now,
            updated_at=None,
        )
        return int(g.id)

    gid = int(existing_row["id"])
    merged_notes = _merge_notes(existing_row.get("notes"), notes_json_fields)

    existing_t1 = existing_row.get("team1_score")
    existing_t2 = existing_row.get("team2_score")

    updates: dict[str, Any] = {
        "notes": merged_notes,
        "stats_imported_at": now,
        "updated_at": now,
    }
    if tts_int is not None and existing_row.get("timetoscore_game_id") is None:
        updates["timetoscore_game_id"] = int(tts_int)
    if ext_key and not existing_row.get("external_game_key"):
        updates["external_game_key"] = str(ext_key)
    if game_type_id is not None:
        updates["game_type_id"] = int(game_type_id)
    if location is not None:
        updates["location"] = location

    if replace:
        updates["team1_score"] = team1_score
        updates["team2_score"] = team2_score
        if team1_score is not None and team2_score is not None:
            updates["is_final"] = True
    else:
        if existing_t1 is None and team1_score is not None:
            updates["team1_score"] = team1_score
        if existing_t2 is None and team2_score is not None:
            updates["team2_score"] = team2_score
        if (
            existing_t1 is None
            and existing_t2 is None
            and team1_score is not None
            and team2_score is not None
        ):
            updates["is_final"] = True

    m.HkyGame.objects.filter(id=gid).update(**updates)
    return gid


def _map_game_to_league_for_import(
    league_id: int,
    game_id: int,
    *,
    division_name: Optional[str] = None,
    division_id: Optional[int] = None,
    conference_id: Optional[int] = None,
    sort_order: Optional[int] = None,
    commit: bool = True,
) -> None:
    del commit
    dn = (division_name or "").strip() or None
    _django_orm, m = _orm_modules()
    obj, created = m.LeagueGame.objects.get_or_create(
        league_id=int(league_id),
        game_id=int(game_id),
        defaults={
            "division_name": dn,
            "division_id": division_id,
            "conference_id": conference_id,
            "sort_order": sort_order,
        },
    )
    if created:
        return

    updates: dict[str, Any] = {}
    allow_div_update = True
    if dn and logic.is_external_division_name(dn):
        existing_dn = str(getattr(obj, "division_name", "") or "").strip()
        if existing_dn and not logic.is_external_division_name(existing_dn):
            allow_div_update = False

    if dn and dn.strip() and dn.strip().lower() != "external" and allow_div_update:
        updates["division_name"] = dn
    if allow_div_update:
        if division_id is not None:
            updates["division_id"] = division_id
        if conference_id is not None:
            updates["conference_id"] = conference_id
    if sort_order is not None:
        updates["sort_order"] = sort_order
    if updates:
        m.LeagueGame.objects.filter(id=int(obj.id)).update(**updates)


def _map_team_to_league_for_import(
    league_id: int,
    team_id: int,
    *,
    division_name: Optional[str] = None,
    division_id: Optional[int] = None,
    conference_id: Optional[int] = None,
    commit: bool = True,
) -> None:
    del commit
    dn = (division_name or "").strip() or None
    _django_orm, m = _orm_modules()
    obj, created = m.LeagueTeam.objects.get_or_create(
        league_id=int(league_id),
        team_id=int(team_id),
        defaults={"division_name": dn, "division_id": division_id, "conference_id": conference_id},
    )
    if created:
        return
    updates: dict[str, Any] = {}
    allow_div_update = True
    if dn and logic.is_external_division_name(dn):
        existing_dn = str(getattr(obj, "division_name", "") or "").strip()
        if existing_dn and not logic.is_external_division_name(existing_dn):
            allow_div_update = False

    if dn and dn.strip() and dn.strip().lower() != "external" and allow_div_update:
        updates["division_name"] = dn
    if allow_div_update:
        if division_id is not None:
            updates["division_id"] = division_id
        if conference_id is not None:
            updates["conference_id"] = conference_id
    if updates:
        m.LeagueTeam.objects.filter(id=int(obj.id)).update(**updates)


def _ensure_team_logo_from_url_for_import(
    *, team_id: int, logo_url: Optional[str], replace: bool, commit: bool = True
) -> None:
    del commit
    url = str(logo_url or "").strip()
    if not url:
        return
    _django_orm, m = _orm_modules()
    existing = m.Team.objects.filter(id=int(team_id)).values_list("logo_path", flat=True).first()
    if existing and not replace:
        return
    try:
        import requests  # type: ignore
    except Exception:
        return

    headers: dict[str, str] = {"User-Agent": "Mozilla/5.0"}
    try:
        from urllib.parse import urlparse

        u = urlparse(url)
        if u.scheme and u.netloc:
            headers["Referer"] = f"{u.scheme}://{u.netloc}/"
    except Exception:
        pass

    try:
        resp = requests.get(url, timeout=(10, 30), headers=headers)  # type: ignore[attr-defined]
        resp.raise_for_status()
        data = resp.content
        ctype = str(resp.headers.get("Content-Type") or "")
    except Exception:
        return

    ext = None
    ctype_l = ctype.lower()
    if "png" in ctype_l:
        ext = ".png"
    elif "jpeg" in ctype_l or "jpg" in ctype_l:
        ext = ".jpg"
    elif "gif" in ctype_l:
        ext = ".gif"
    elif "webp" in ctype_l:
        ext = ".webp"
    elif "svg" in ctype_l:
        ext = ".svg"
    if ext is None:
        for cand in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
            if url.lower().split("?", 1)[0].endswith(cand):
                ext = cand
                break
    if ext is None:
        ext = ".png"

    try:
        logo_dir = Path(logic.INSTANCE_DIR) / "uploads" / "team_logos"
        logo_dir.mkdir(parents=True, exist_ok=True)
        dest = logo_dir / f"import_team{int(team_id)}{ext}"
        dest.write_bytes(data)
        try:
            os.chmod(dest, 0o644)
        except Exception:
            pass
        m.Team.objects.filter(id=int(team_id)).update(
            logo_path=str(dest), updated_at=dt.datetime.now()
        )
    except Exception:
        return


def _ensure_team_logo_for_import(
    *,
    team_id: int,
    logo_b64: Optional[str],
    logo_content_type: Optional[str],
    logo_url: Optional[str],
    replace: bool,
    commit: bool = True,
) -> None:
    del commit
    _django_orm, m = _orm_modules()
    existing = m.Team.objects.filter(id=int(team_id)).values_list("logo_path", flat=True).first()
    if existing and not replace:
        return

    b64_s = str(logo_b64 or "").strip()
    if b64_s:
        try:
            import base64

            data = base64.b64decode(b64_s.encode("ascii"), validate=False)
        except Exception:
            data = b""
        if not data or len(data) > 5 * 1024 * 1024:
            return

        ctype = str(logo_content_type or "").strip()
        ext = None
        ctype_l = ctype.lower()
        if "png" in ctype_l:
            ext = ".png"
        elif "jpeg" in ctype_l or "jpg" in ctype_l:
            ext = ".jpg"
        elif "gif" in ctype_l:
            ext = ".gif"
        elif "webp" in ctype_l:
            ext = ".webp"
        elif "svg" in ctype_l:
            ext = ".svg"
        if ext is None:
            url = str(logo_url or "").strip()
            for cand in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
                if url.lower().split("?", 1)[0].endswith(cand):
                    ext = cand
                    break
        if ext is None:
            ext = ".png"

        try:
            logo_dir = Path(logic.INSTANCE_DIR) / "uploads" / "team_logos"
            logo_dir.mkdir(parents=True, exist_ok=True)
            dest = logo_dir / f"import_team{int(team_id)}{ext}"
            dest.write_bytes(data)
            try:
                os.chmod(dest, 0o644)
            except Exception:
                pass
            m.Team.objects.filter(id=int(team_id)).update(
                logo_path=str(dest), updated_at=dt.datetime.now()
            )
            return
        except Exception:
            return

    _ensure_team_logo_from_url_for_import(team_id=int(team_id), logo_url=logo_url, replace=replace)


@csrf_exempt
def api_import_ensure_league(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    source = payload.get("source")
    external_key = payload.get("external_key")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(source) if source else None,
        external_key=str(external_key) if external_key else None,
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin")
    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id)}
    )


@csrf_exempt
def api_import_teams(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)

    teams = payload.get("teams") or []
    if not isinstance(teams, list) or not teams:
        return JsonResponse({"ok": False, "error": "teams must be a non-empty list"}, status=400)

    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(payload.get("source") or "timetoscore"),
        external_key=str(payload.get("external_key") or ""),
        commit=False,
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin", commit=False)

    results: list[dict[str, Any]] = []
    try:
        from django.db import transaction

        def _clean_division_name(dn: Any) -> Optional[str]:
            s = str(dn or "").strip()
            if not s:
                return None
            if s.lower() == "external":
                return None
            return s

        with transaction.atomic():
            for idx, team in enumerate(teams):
                if not isinstance(team, dict):
                    raise ValueError(f"teams[{idx}] must be an object")
                name = str(team.get("name") or "").strip()
                if not name:
                    continue

                team_replace = bool(team.get("replace", replace))

                division_name = _clean_division_name(team.get("division_name"))
                try:
                    division_id = (
                        int(team.get("division_id"))
                        if team.get("division_id") is not None
                        else None
                    )
                except Exception:
                    division_id = None
                try:
                    conference_id = (
                        int(team.get("conference_id"))
                        if team.get("conference_id") is not None
                        else None
                    )
                except Exception:
                    conference_id = None

                team_id = _ensure_external_team_for_import(owner_user_id, name, commit=False)
                _map_team_to_league_for_import(
                    league_id,
                    team_id,
                    division_name=division_name,
                    division_id=division_id,
                    conference_id=conference_id,
                    commit=False,
                )
                _ensure_team_logo_for_import(
                    team_id=int(team_id),
                    logo_b64=team.get("logo_b64") or team.get("team_logo_b64"),
                    logo_content_type=team.get("logo_content_type")
                    or team.get("team_logo_content_type"),
                    logo_url=team.get("logo_url") or team.get("team_logo_url"),
                    replace=team_replace,
                    commit=False,
                )
                results.append({"team_id": int(team_id), "name": name})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "owner_user_id": int(owner_user_id),
            "imported": int(len(results)),
            "results": results,
        }
    )


@csrf_exempt
def api_import_game(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(payload.get("source") or "timetoscore"),
        external_key=str(payload.get("external_key") or ""),
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin")

    game = payload.get("game") or {}
    home_name = str(game.get("home_name") or "").strip()
    away_name = str(game.get("away_name") or "").strip()
    if not home_name or not away_name:
        return JsonResponse(
            {"ok": False, "error": "home_name and away_name are required"}, status=400
        )

    division_name = str(game.get("division_name") or "").strip() or None
    home_division_name = str(game.get("home_division_name") or division_name or "").strip() or None
    away_division_name = str(game.get("away_division_name") or division_name or "").strip() or None
    try:
        division_id = int(game.get("division_id")) if game.get("division_id") is not None else None
    except Exception:
        division_id = None
    try:
        conference_id = (
            int(game.get("conference_id")) if game.get("conference_id") is not None else None
        )
    except Exception:
        conference_id = None
    try:
        home_division_id = (
            int(game.get("home_division_id"))
            if game.get("home_division_id") is not None
            else division_id
        )
    except Exception:
        home_division_id = division_id
    try:
        away_division_id = (
            int(game.get("away_division_id"))
            if game.get("away_division_id") is not None
            else division_id
        )
    except Exception:
        away_division_id = division_id
    try:
        home_conference_id = (
            int(game.get("home_conference_id"))
            if game.get("home_conference_id") is not None
            else conference_id
        )
    except Exception:
        home_conference_id = conference_id
    try:
        away_conference_id = (
            int(game.get("away_conference_id"))
            if game.get("away_conference_id") is not None
            else conference_id
        )
    except Exception:
        away_conference_id = conference_id

    team1_id = _ensure_external_team_for_import(owner_user_id, home_name)
    team2_id = _ensure_external_team_for_import(owner_user_id, away_name)
    _map_team_to_league_for_import(
        league_id,
        team1_id,
        division_name=home_division_name,
        division_id=home_division_id,
        conference_id=home_conference_id,
    )
    _map_team_to_league_for_import(
        league_id,
        team2_id,
        division_name=away_division_name,
        division_id=away_division_id,
        conference_id=away_conference_id,
    )
    _ensure_team_logo_for_import(
        team_id=int(team1_id),
        logo_b64=game.get("home_logo_b64") or game.get("team1_logo_b64"),
        logo_content_type=game.get("home_logo_content_type") or game.get("team1_logo_content_type"),
        logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
        replace=replace,
    )
    _ensure_team_logo_for_import(
        team_id=int(team2_id),
        logo_b64=game.get("away_logo_b64") or game.get("team2_logo_b64"),
        logo_content_type=game.get("away_logo_content_type") or game.get("team2_logo_content_type"),
        logo_url=game.get("away_logo_url") or game.get("team2_logo_url"),
        replace=replace,
    )

    starts_at = game.get("starts_at")
    starts_at_s = str(starts_at) if starts_at else None
    location = str(game.get("location")).strip() if game.get("location") else None
    team1_score = game.get("home_score")
    team2_score = game.get("away_score")
    tts_game_id = game.get("timetoscore_game_id")

    notes_fields: dict[str, Any] = {}
    if tts_game_id is not None:
        try:
            notes_fields["timetoscore_game_id"] = int(tts_game_id)
        except Exception:
            pass
    if game.get("season_id") is not None:
        try:
            notes_fields["timetoscore_season_id"] = int(game.get("season_id"))
        except Exception:
            pass
    if payload.get("source"):
        notes_fields["source"] = str(payload.get("source"))
    if game.get("timetoscore_type") is not None:
        notes_fields["timetoscore_type"] = str(game.get("timetoscore_type"))
    elif game.get("game_type_name") is not None:
        notes_fields["timetoscore_type"] = str(game.get("game_type_name"))
    elif game.get("type") is not None:
        notes_fields["timetoscore_type"] = str(game.get("type"))

    game_type_id = _ensure_game_type_id_for_import(
        game.get("game_type_name")
        or game.get("game_type")
        or game.get("timetoscore_type")
        or game.get("type")
    )

    try:
        t1s = int(team1_score) if team1_score is not None else None
    except Exception:
        t1s = None
    try:
        t2s = int(team2_score) if team2_score is not None else None
    except Exception:
        t2s = None

    gid = _upsert_game_for_import(
        owner_user_id=owner_user_id,
        team1_id=team1_id,
        team2_id=team2_id,
        game_type_id=game_type_id,
        starts_at=starts_at_s,
        location=location,
        team1_score=t1s,
        team2_score=t2s,
        replace=replace,
        notes_json_fields=notes_fields,
    )
    _map_game_to_league_for_import(
        league_id,
        gid,
        division_name=division_name or home_division_name or away_division_name,
        division_id=division_id or home_division_id or away_division_id,
        conference_id=conference_id or home_conference_id or away_conference_id,
    )

    roster_player_ids_by_team: dict[int, set[int]] = {int(team1_id): set(), int(team2_id): set()}
    for side_key, tid in (("home", team1_id), ("away", team2_id)):
        roster = game.get(f"{side_key}_roster") or []
        if isinstance(roster, list):
            for row in roster:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name") or "").strip()
                if not nm:
                    continue
                jersey = str(row.get("number") or "").strip() or None
                pos = str(row.get("position") or "").strip() or None
                pid = _ensure_player_for_import(owner_user_id, tid, nm, jersey, pos)
                try:
                    roster_player_ids_by_team[int(tid)].add(int(pid))
                except Exception:
                    pass

    def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
        _django_orm2, m2 = _orm_modules()
        pid = (
            m2.Player.objects.filter(
                user_id=int(owner_user_id), team_id=int(team_id), name=str(name)
            )
            .values_list("id", flat=True)
            .first()
        )
        return int(pid) if pid is not None else None

    stats_rows = game.get("player_stats") or []
    played = (
        bool(game.get("is_final"))
        or (t1s is not None and t2s is not None)
        or (isinstance(stats_rows, list) and bool(stats_rows))
    )

    _django_orm2, m2 = _orm_modules()
    from django.db import transaction

    with transaction.atomic():
        if played:
            to_create = []
            for tid, pids in roster_player_ids_by_team.items():
                for pid in sorted(pids):
                    to_create.append(
                        m2.PlayerStat(
                            user_id=int(owner_user_id),
                            team_id=int(tid),
                            game_id=int(gid),
                            player_id=int(pid),
                        )
                    )
            if to_create:
                m2.PlayerStat.objects.bulk_create(to_create, ignore_conflicts=True)

        if isinstance(stats_rows, list):
            for srow in stats_rows:
                if not isinstance(srow, dict):
                    continue
                pname = str(srow.get("name") or "").strip()
                if not pname:
                    continue
                goals = srow.get("goals")
                assists = srow.get("assists")
                try:
                    gval = int(goals) if goals is not None else 0
                except Exception:
                    gval = 0
                try:
                    aval = int(assists) if assists is not None else 0
                except Exception:
                    aval = 0

                team_ref = team1_id
                pid = _player_id_by_name(team1_id, pname)
                if pid is None:
                    pid = _player_id_by_name(team2_id, pname)
                    team_ref = team2_id if pid is not None else team1_id
                if pid is None:
                    pid = _ensure_player_for_import(owner_user_id, team_ref, pname, None, None)

                force_tts_scoring = bool(tts_game_id is not None)
                ps, _created = m2.PlayerStat.objects.get_or_create(
                    game_id=int(gid),
                    player_id=int(pid),
                    defaults={
                        "user_id": int(owner_user_id),
                        "team_id": int(team_ref),
                        "goals": gval,
                        "assists": aval,
                    },
                )
                if replace or force_tts_scoring:
                    m2.PlayerStat.objects.filter(id=ps.id).update(goals=gval, assists=aval)
                else:
                    m2.PlayerStat.objects.filter(id=ps.id, goals__isnull=True).update(goals=gval)
                    m2.PlayerStat.objects.filter(id=ps.id, assists__isnull=True).update(
                        assists=aval
                    )

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "owner_user_id": int(owner_user_id),
            "team1_id": int(team1_id),
            "team2_id": int(team2_id),
            "game_id": int(gid),
        }
    )


@csrf_exempt
def api_import_games_batch(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)

    games = payload.get("games") or []
    if not isinstance(games, list) or not games:
        return JsonResponse({"ok": False, "error": "games must be a non-empty list"}, status=400)

    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(payload.get("source") or "timetoscore"),
        external_key=str(payload.get("external_key") or ""),
        commit=False,
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin", commit=False)

    results: list[dict[str, Any]] = []
    try:
        _django_orm, m = _orm_modules()

        def _clean_division_name(dn: Any) -> Optional[str]:
            s = str(dn or "").strip()
            if not s:
                return None
            if s.lower() == "external":
                return None
            return s

        def _league_team_div_meta(
            lid: int, tid: int
        ) -> tuple[Optional[str], Optional[int], Optional[int]]:
            row = (
                m.LeagueTeam.objects.filter(league_id=int(lid), team_id=int(tid))
                .values("division_name", "division_id", "conference_id")
                .first()
            )
            if not row:
                return None, None, None
            try:
                did = int(row.get("division_id")) if row.get("division_id") is not None else None
            except Exception:
                did = None
            try:
                cid = (
                    int(row.get("conference_id")) if row.get("conference_id") is not None else None
                )
            except Exception:
                cid = None
            return _clean_division_name(row.get("division_name")), did, cid

        for idx, game in enumerate(games):
            if not isinstance(game, dict):
                raise ValueError(f"games[{idx}] must be an object")
            home_name = str(game.get("home_name") or "").strip()
            away_name = str(game.get("away_name") or "").strip()
            if not home_name or not away_name:
                raise ValueError(f"games[{idx}]: home_name and away_name are required")

            game_replace = bool(game.get("replace", replace))

            division_name = _clean_division_name(game.get("division_name"))
            home_division_name = _clean_division_name(
                game.get("home_division_name") or division_name
            )
            away_division_name = _clean_division_name(
                game.get("away_division_name") or division_name
            )
            try:
                division_id = (
                    int(game.get("division_id")) if game.get("division_id") is not None else None
                )
            except Exception:
                division_id = None
            try:
                conference_id = (
                    int(game.get("conference_id"))
                    if game.get("conference_id") is not None
                    else None
                )
            except Exception:
                conference_id = None
            try:
                home_division_id = (
                    int(game.get("home_division_id"))
                    if game.get("home_division_id") is not None
                    else division_id
                )
            except Exception:
                home_division_id = division_id
            try:
                away_division_id = (
                    int(game.get("away_division_id"))
                    if game.get("away_division_id") is not None
                    else division_id
                )
            except Exception:
                away_division_id = division_id
            try:
                home_conference_id = (
                    int(game.get("home_conference_id"))
                    if game.get("home_conference_id") is not None
                    else conference_id
                )
            except Exception:
                home_conference_id = conference_id
            try:
                away_conference_id = (
                    int(game.get("away_conference_id"))
                    if game.get("away_conference_id") is not None
                    else conference_id
                )
            except Exception:
                away_conference_id = conference_id

            team1_id = _ensure_external_team_for_import(owner_user_id, home_name, commit=False)
            team2_id = _ensure_external_team_for_import(owner_user_id, away_name, commit=False)
            _map_team_to_league_for_import(
                league_id,
                team1_id,
                division_name=home_division_name,
                division_id=home_division_id,
                conference_id=home_conference_id,
                commit=False,
            )
            _map_team_to_league_for_import(
                league_id,
                team2_id,
                division_name=away_division_name,
                division_id=away_division_id,
                conference_id=away_conference_id,
                commit=False,
            )
            _ensure_team_logo_for_import(
                team_id=int(team1_id),
                logo_b64=game.get("home_logo_b64") or game.get("team1_logo_b64"),
                logo_content_type=game.get("home_logo_content_type")
                or game.get("team1_logo_content_type"),
                logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
                replace=game_replace,
                commit=False,
            )
            _ensure_team_logo_for_import(
                team_id=int(team2_id),
                logo_b64=game.get("away_logo_b64") or game.get("team2_logo_b64"),
                logo_content_type=game.get("away_logo_content_type")
                or game.get("team2_logo_content_type"),
                logo_url=game.get("away_logo_url") or game.get("team2_logo_url"),
                replace=game_replace,
                commit=False,
            )

            starts_at = game.get("starts_at")
            starts_at_s = str(starts_at) if starts_at else None
            location = str(game.get("location")).strip() if game.get("location") else None
            team1_score = game.get("home_score")
            team2_score = game.get("away_score")
            tts_game_id = game.get("timetoscore_game_id")

            notes_fields: dict[str, Any] = {}
            if tts_game_id is not None:
                try:
                    notes_fields["timetoscore_game_id"] = int(tts_game_id)
                except Exception:
                    pass
            if game.get("season_id") is not None:
                try:
                    notes_fields["timetoscore_season_id"] = int(game.get("season_id"))
                except Exception:
                    pass
            if payload.get("source"):
                notes_fields["source"] = str(payload.get("source"))
            if game.get("timetoscore_type") is not None:
                notes_fields["timetoscore_type"] = str(game.get("timetoscore_type"))
            elif game.get("game_type_name") is not None:
                notes_fields["timetoscore_type"] = str(game.get("game_type_name"))
            elif game.get("type") is not None:
                notes_fields["timetoscore_type"] = str(game.get("type"))

            game_type_id = _ensure_game_type_id_for_import(
                game.get("game_type_name")
                or game.get("game_type")
                or game.get("timetoscore_type")
                or game.get("type")
            )

            try:
                t1s = int(team1_score) if team1_score is not None else None
            except Exception:
                t1s = None
            try:
                t2s = int(team2_score) if team2_score is not None else None
            except Exception:
                t2s = None

            gid = _upsert_game_for_import(
                owner_user_id=owner_user_id,
                team1_id=team1_id,
                team2_id=team2_id,
                game_type_id=game_type_id,
                starts_at=starts_at_s,
                location=location,
                team1_score=t1s,
                team2_score=t2s,
                replace=game_replace,
                notes_json_fields=notes_fields,
                commit=False,
            )

            effective_div_name = division_name or home_division_name or away_division_name
            effective_div_id = division_id or home_division_id or away_division_id
            effective_conf_id = conference_id or home_conference_id or away_conference_id
            if not effective_div_name:
                t1_dn, t1_did, t1_cid = _league_team_div_meta(int(league_id), int(team1_id))
                t2_dn, t2_did, t2_cid = _league_team_div_meta(int(league_id), int(team2_id))
                if t1_dn:
                    effective_div_name = t1_dn
                    effective_div_id = effective_div_id or t1_did
                    effective_conf_id = effective_conf_id or t1_cid
                elif t2_dn:
                    effective_div_name = t2_dn
                    effective_div_id = effective_div_id or t2_did
                    effective_conf_id = effective_conf_id or t2_cid
            _map_game_to_league_for_import(
                league_id,
                gid,
                division_name=effective_div_name,
                division_id=effective_div_id,
                conference_id=effective_conf_id,
                commit=False,
            )

            roster_player_ids_by_team: dict[int, set[int]] = {
                int(team1_id): set(),
                int(team2_id): set(),
            }
            for side_key, tid in (("home", team1_id), ("away", team2_id)):
                roster = game.get(f"{side_key}_roster") or []
                if isinstance(roster, list):
                    for row in roster:
                        if not isinstance(row, dict):
                            continue
                        nm = str(row.get("name") or "").strip()
                        if not nm:
                            continue
                        jersey = str(row.get("number") or "").strip() or None
                        pos = str(row.get("position") or "").strip() or None
                        pid = _ensure_player_for_import(
                            owner_user_id, tid, nm, jersey, pos, commit=False
                        )
                        try:
                            roster_player_ids_by_team[int(tid)].add(int(pid))
                        except Exception:
                            pass

            def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
                pid = (
                    m.Player.objects.filter(
                        user_id=int(owner_user_id), team_id=int(team_id), name=str(name)
                    )
                    .values_list("id", flat=True)
                    .first()
                )
                return int(pid) if pid is not None else None

            stats_rows = game.get("player_stats") or []
            events_csv = game.get("events_csv")
            game_stats_json = game.get("game_stats")
            played = (
                bool(game.get("is_final"))
                or (t1s is not None and t2s is not None)
                or (isinstance(stats_rows, list) and bool(stats_rows))
            )
            if played:
                to_create = []
                for tid, pids in roster_player_ids_by_team.items():
                    for pid in sorted(pids):
                        to_create.append(
                            m.PlayerStat(
                                user_id=int(owner_user_id),
                                team_id=int(tid),
                                game_id=int(gid),
                                player_id=int(pid),
                            )
                        )
                if to_create:
                    m.PlayerStat.objects.bulk_create(to_create, ignore_conflicts=True)

            if isinstance(stats_rows, list):
                for srow in stats_rows:
                    if not isinstance(srow, dict):
                        continue
                    pname = str(srow.get("name") or "").strip()
                    if not pname:
                        continue
                    goals = srow.get("goals")
                    assists = srow.get("assists")
                    pim = srow.get("pim")
                    try:
                        gval = int(goals) if goals is not None else 0
                    except Exception:
                        gval = 0
                    try:
                        aval = int(assists) if assists is not None else 0
                    except Exception:
                        aval = 0
                    try:
                        pim_val = int(pim) if pim is not None and str(pim).strip() != "" else None
                    except Exception:
                        pim_val = None

                    team_ref = team1_id
                    pid = _player_id_by_name(team1_id, pname)
                    if pid is None:
                        pid = _player_id_by_name(team2_id, pname)
                        team_ref = team2_id if pid is not None else team1_id
                    if pid is None:
                        pid = _ensure_player_for_import(
                            owner_user_id, team_ref, pname, None, None, commit=False
                        )

                    force_tts_scoring = bool(tts_game_id is not None)
                    ps, _created = m.PlayerStat.objects.get_or_create(
                        game_id=int(gid),
                        player_id=int(pid),
                        defaults={
                            "user_id": int(owner_user_id),
                            "team_id": int(team_ref),
                            "goals": gval,
                            "assists": aval,
                            "pim": pim_val,
                        },
                    )
                    if game_replace or force_tts_scoring:
                        updates = {"goals": gval, "assists": aval}
                        if pim_val is not None:
                            updates["pim"] = pim_val
                        m.PlayerStat.objects.filter(id=ps.id).update(**updates)
                    else:
                        m.PlayerStat.objects.filter(id=ps.id, goals__isnull=True).update(goals=gval)
                        m.PlayerStat.objects.filter(id=ps.id, assists__isnull=True).update(
                            assists=aval
                        )
                        if pim_val is not None:
                            m.PlayerStat.objects.filter(id=ps.id, pim__isnull=True).update(
                                pim=pim_val
                            )

            if isinstance(events_csv, str) and events_csv.strip():
                now2 = dt.datetime.now()
                if game_replace:
                    m.HkyGameEvent.objects.update_or_create(
                        game_id=int(gid),
                        defaults={
                            "events_csv": events_csv,
                            "source_label": "timetoscore",
                            "updated_at": now2,
                        },
                    )
                else:
                    existing_ev = (
                        m.HkyGameEvent.objects.filter(game_id=int(gid))
                        .values("events_csv", "source_label")
                        .first()
                    )
                    if not existing_ev:
                        m.HkyGameEvent.objects.create(
                            game_id=int(gid),
                            events_csv=events_csv,
                            source_label="timetoscore",
                            updated_at=now2,
                        )
                    else:
                        merged_csv, merged_source = logic.merge_events_csv_prefer_timetoscore(
                            existing_csv=str(existing_ev.get("events_csv") or ""),
                            existing_source_label=str(existing_ev.get("source_label") or ""),
                            incoming_csv=str(events_csv),
                            incoming_source_label="timetoscore",
                            protected_types={
                                "goal",
                                "assist",
                                "penalty",
                                "penalty expired",
                                "goaliechange",
                            },
                        )
                        m.HkyGameEvent.objects.update_or_create(
                            game_id=int(gid),
                            defaults={
                                "events_csv": merged_csv,
                                "source_label": merged_source or "timetoscore",
                                "updated_at": now2,
                            },
                        )

            if isinstance(game_stats_json, dict) and game_stats_json:
                try:
                    stats_json_text = json.dumps(game_stats_json)
                except Exception:
                    stats_json_text = None
                if stats_json_text:
                    now3 = dt.datetime.now()
                    if game_replace:
                        m.HkyGameStat.objects.update_or_create(
                            game_id=int(gid),
                            defaults={"stats_json": stats_json_text, "updated_at": now3},
                        )
                    else:
                        m.HkyGameStat.objects.get_or_create(
                            game_id=int(gid),
                            defaults={"stats_json": stats_json_text, "updated_at": now3},
                        )

            results.append({"game_id": gid, "team1_id": team1_id, "team2_id": team2_id})

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "owner_user_id": int(owner_user_id),
            "imported": int(len(results)),
            "results": results,
        }
    )


@csrf_exempt
def api_import_shift_package(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    replace = bool(payload.get("replace", False))

    game_id = payload.get("game_id")
    tts_game_id = payload.get("timetoscore_game_id")
    external_game_key = str(payload.get("external_game_key") or "").strip() or None
    team_side = str(payload.get("team_side") or "").strip().lower() or None
    if team_side not in {None, "", "home", "away"}:
        return JsonResponse(
            {"ok": False, "error": "team_side must be 'home' or 'away'"}, status=400
        )
    create_missing_players = bool(payload.get("create_missing_players", False))
    owner_email = str(payload.get("owner_email") or "").strip().lower() or None
    league_id_payload = payload.get("league_id")
    league_name = str(payload.get("league_name") or "").strip() or None
    division_name = str(payload.get("division_name") or "").strip() or None
    sort_order_payload = payload.get("sort_order")
    sort_order: Optional[int] = None
    try:
        sort_order = int(sort_order_payload) if sort_order_payload is not None else None
    except Exception:
        sort_order = None
    resolved_game_id: Optional[int] = None
    try:
        resolved_game_id = int(game_id) if game_id is not None else None
    except Exception:
        resolved_game_id = None

    _django_orm, m = _orm_modules()

    from django.db.models import Q

    tts_int: Optional[int]
    try:
        tts_int = int(tts_game_id) if tts_game_id is not None else None
    except Exception:
        tts_int = None

    if resolved_game_id is None and tts_int is not None:
        gid = (
            m.HkyGame.objects.filter(timetoscore_game_id=int(tts_int))
            .values_list("id", flat=True)
            .first()
        )
        if gid is None:
            token_json_nospace = f'"timetoscore_game_id":{int(tts_int)}'
            token_json_space = f'"timetoscore_game_id": {int(tts_int)}'
            token_plain = f"game_id={int(tts_int)}"
            gid = (
                m.HkyGame.objects.filter(
                    Q(notes__contains=token_json_nospace)
                    | Q(notes__contains=token_json_space)
                    | Q(notes__contains=token_plain)
                )
                .values_list("id", flat=True)
                .first()
            )
        if gid is not None:
            resolved_game_id = int(gid)

    if resolved_game_id is None and external_game_key and owner_email:
        owner_user_id_for_create = _ensure_user_for_import(owner_email)

        def _norm_team_name_for_match(s: str) -> str:
            t = str(s or "").replace("\xa0", " ").strip()
            t = (
                t.replace("\u2010", "-")
                .replace("\u2011", "-")
                .replace("\u2012", "-")
                .replace("\u2013", "-")
                .replace("\u2212", "-")
            )
            t = " ".join(t.split())
            t = re.sub(r"\s*\(\s*external\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
            t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
            t = t.casefold()
            t = re.sub(r"[^0-9a-z]+", " ", t)
            return " ".join(t.split())

        def _find_team_in_league_by_name(league_id_i: int, name: str) -> Optional[dict[str, Any]]:
            nm = _norm_team_name_for_match(name)
            if not nm:
                return None
            rows = list(
                m.LeagueTeam.objects.filter(league_id=int(league_id_i))
                .select_related("team")
                .values("team_id", "team__name", "division_name", "division_id", "conference_id")
            )
            matches = [
                {
                    "team_id": int(r["team_id"]),
                    "team_name": str(r.get("team__name") or ""),
                    "division_name": r.get("division_name"),
                    "division_id": r.get("division_id"),
                    "conference_id": r.get("conference_id"),
                }
                for r in rows
                if _norm_team_name_for_match(str(r.get("team__name") or "")) == nm
            ]
            if not matches:
                return None
            if len(matches) == 1:
                return matches[0]
            want_div = str(payload.get("division_name") or "").strip()
            if want_div and want_div.lower() != "external":
                by_div = [
                    cand
                    for cand in matches
                    if str(cand.get("division_name") or "").strip() == want_div
                ]
                if len(by_div) == 1:
                    return by_div[0]
            for cand in matches:
                dn = str(cand.get("division_name") or "").strip()
                if dn:
                    return cand
            return matches[0]

        gid = (
            m.HkyGame.objects.filter(
                user_id=int(owner_user_id_for_create), external_game_key=str(external_game_key)
            )
            .values_list("id", flat=True)
            .first()
        )
        if gid is None:
            try:
                ext_json = json.dumps(external_game_key)
            except Exception:
                ext_json = f'"{external_game_key}"'
            tokens = [f'"external_game_key":{ext_json}', f'"external_game_key": {ext_json}']
            gid = (
                m.HkyGame.objects.filter(user_id=int(owner_user_id_for_create))
                .filter(Q(notes__contains=tokens[0]) | Q(notes__contains=tokens[1]))
                .values_list("id", flat=True)
                .first()
            )
        if gid is not None:
            resolved_game_id = int(gid)
        else:
            home_team_name = str(payload.get("home_team_name") or "").strip()
            away_team_name = str(payload.get("away_team_name") or "").strip()
            if not home_team_name or not away_team_name:
                return JsonResponse(
                    {
                        "ok": False,
                        "error": "home_team_name and away_team_name are required to create an external game",
                    },
                    status=400,
                )

            league_id_i: Optional[int] = None
            try:
                league_id_i = int(league_id_payload) if league_id_payload is not None else None
            except Exception:
                league_id_i = None
            if league_id_i is None:
                if not league_name:
                    return JsonResponse(
                        {
                            "ok": False,
                            "error": "league_id or league_name is required to create an external game",
                        },
                        status=400,
                    )
                existing_lid = (
                    m.League.objects.filter(name=str(league_name))
                    .values_list("id", flat=True)
                    .first()
                )
                if existing_lid is not None:
                    league_id_i = int(existing_lid)
                else:
                    now = dt.datetime.now()
                    league = m.League.objects.create(
                        name=str(league_name),
                        owner_user_id=int(owner_user_id_for_create),
                        is_shared=False,
                        is_public=False,
                        source="shift_package",
                        external_key=None,
                        created_at=now,
                        updated_at=None,
                    )
                    league_id_i = int(league.id)

            match_home = (
                _find_team_in_league_by_name(int(league_id_i), home_team_name)
                if league_id_i
                else None
            )
            match_away = (
                _find_team_in_league_by_name(int(league_id_i), away_team_name)
                if league_id_i
                else None
            )

            team1_id = (
                int(match_home["team_id"])
                if match_home
                else _ensure_external_team_for_import(
                    owner_user_id_for_create, home_team_name, commit=False
                )
            )
            team2_id = (
                int(match_away["team_id"])
                if match_away
                else _ensure_external_team_for_import(
                    owner_user_id_for_create, away_team_name, commit=False
                )
            )

            game_division_name = str(division_name or "").strip() or "External"
            new_team_division_name = game_division_name

            _ensure_team_logo_for_import(
                team_id=team1_id,
                logo_b64=payload.get("home_logo_b64"),
                logo_content_type=payload.get("home_logo_content_type"),
                logo_url=payload.get("home_logo_url"),
                replace=replace,
                commit=False,
            )
            _ensure_team_logo_for_import(
                team_id=team2_id,
                logo_b64=payload.get("away_logo_b64"),
                logo_content_type=payload.get("away_logo_content_type"),
                logo_url=payload.get("away_logo_url"),
                replace=replace,
                commit=False,
            )

            team1_score = None
            team2_score = None
            try:
                parsed_gs = logic.parse_shift_stats_game_stats_csv(
                    str(payload.get("game_stats_csv") or "")
                )
                gf = parsed_gs.get("Goals For")
                ga = parsed_gs.get("Goals Against")
                gf_i = int(gf) if gf not in (None, "") else None
                ga_i = int(ga) if ga not in (None, "") else None
                if gf_i is not None and ga_i is not None and team_side in {"home", "away"}:
                    if team_side == "home":
                        team1_score, team2_score = gf_i, ga_i
                    else:
                        team1_score, team2_score = ga_i, gf_i
            except Exception:
                team1_score, team2_score = None, None

            starts_at = str(payload.get("starts_at") or "").strip() or None
            location = str(payload.get("location") or "").strip() or None
            notes_fields: dict[str, Any] = {"external_game_key": external_game_key}
            if tts_int is not None:
                notes_fields["timetoscore_game_id"] = int(tts_int)
            resolved_game_id = _upsert_game_for_import(
                owner_user_id=owner_user_id_for_create,
                team1_id=team1_id,
                team2_id=team2_id,
                game_type_id=None,
                starts_at=starts_at,
                location=location,
                team1_score=team1_score,
                team2_score=team2_score,
                replace=replace,
                notes_json_fields=notes_fields,
                commit=False,
            )
            if not match_home:
                _map_team_to_league_for_import(
                    int(league_id_i), team1_id, division_name=new_team_division_name, commit=False
                )
            if not match_away:
                _map_team_to_league_for_import(
                    int(league_id_i), team2_id, division_name=new_team_division_name, commit=False
                )
            _map_game_to_league_for_import(
                int(league_id_i),
                int(resolved_game_id),
                division_name=game_division_name,
                sort_order=sort_order,
                commit=False,
            )
            create_missing_players = True

    if resolved_game_id is None:
        return JsonResponse(
            {
                "ok": False,
                "error": "game_id, timetoscore_game_id, or external_game_key+owner_email+league_name+home_team_name+away_team_name is required",
            },
            status=400,
        )

    game_row = (
        m.HkyGame.objects.filter(id=int(resolved_game_id))
        .values(
            "id",
            "team1_id",
            "team2_id",
            "user_id",
            "notes",
            "timetoscore_game_id",
            "external_game_key",
        )
        .first()
    )
    if not game_row:
        return JsonResponse({"ok": False, "error": "game not found"}, status=404)

    key_fields: dict[str, Any] = {}
    if tts_int is not None:
        key_fields["timetoscore_game_id"] = int(tts_int)
    if external_game_key:
        key_fields["external_game_key"] = str(external_game_key)
    if key_fields:
        resolved_game_id = _upsert_game_for_import(
            owner_user_id=int(game_row.get("user_id") or 0),
            team1_id=int(game_row["team1_id"]),
            team2_id=int(game_row["team2_id"]),
            game_type_id=None,
            starts_at=None,
            location=None,
            team1_score=None,
            team2_score=None,
            replace=False,
            notes_json_fields=key_fields,
            commit=False,
        )
        game_row = (
            m.HkyGame.objects.filter(id=int(resolved_game_id))
            .values(
                "id",
                "team1_id",
                "team2_id",
                "user_id",
                "notes",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        if not game_row:
            return JsonResponse({"ok": False, "error": "game not found after merge"}, status=404)

    team1_id = int(game_row["team1_id"])
    team2_id = int(game_row["team2_id"])
    owner_user_id = int(game_row.get("user_id") or 0)
    tts_linked = bool(
        tts_int is not None
        or game_row.get("timetoscore_game_id") is not None
        or logic._extract_timetoscore_game_id_from_notes(game_row.get("notes")) is not None
    )

    player_stats_csv = payload.get("player_stats_csv")
    game_stats_csv = payload.get("game_stats_csv")
    events_csv = payload.get("events_csv")
    incoming_events_csv_raw: Optional[str] = None
    incoming_events_headers_raw: Optional[list[str]] = None
    incoming_events_rows_raw: Optional[list[dict[str, str]]] = None
    source_label = str(payload.get("source_label") or "").strip() or None
    game_video_url = (
        payload.get("game_video_url") or payload.get("game_video") or payload.get("video_url")
    )

    if isinstance(game_video_url, str) and game_video_url.strip():
        try:
            _update_game_video_url_note(
                int(resolved_game_id), str(game_video_url), replace=replace, commit=False
            )
        except Exception:
            pass

    if isinstance(events_csv, str) and events_csv.strip():
        incoming_events_csv_raw = str(events_csv)
        if tts_linked:
            try:
                incoming_events_headers_raw, incoming_events_rows_raw = logic.parse_events_csv(
                    incoming_events_csv_raw
                )
            except Exception:
                incoming_events_headers_raw, incoming_events_rows_raw = None, None
        drop_types = {"power play", "powerplay", "penalty kill", "penaltykill"}
        if tts_linked:
            drop_types |= {"goal", "assist"}
        try:
            events_csv = logic.filter_events_csv_drop_event_types(
                str(events_csv), drop_types=drop_types
            )
        except Exception:
            pass
        try:
            _h, _r = logic.parse_events_csv(str(events_csv))
            if not _r:
                events_csv = None
        except Exception:
            pass

    if league_id_payload is not None or league_name:
        league_id_i: Optional[int] = None
        try:
            league_id_i = int(league_id_payload) if league_id_payload is not None else None
        except Exception:
            league_id_i = None
        if league_id_i is None and league_name:
            existing_lid = (
                m.League.objects.filter(name=str(league_name)).values_list("id", flat=True).first()
            )
            if existing_lid is not None:
                league_id_i = int(existing_lid)
        if league_id_i is not None:
            _map_team_to_league_for_import(
                int(league_id_i), team1_id, division_name=division_name, commit=False
            )
            _map_team_to_league_for_import(
                int(league_id_i), team2_id, division_name=division_name, commit=False
            )
            _map_game_to_league_for_import(
                int(league_id_i),
                int(resolved_game_id),
                division_name=division_name,
                sort_order=sort_order,
                commit=False,
            )

    imported = 0
    unmatched: list[str] = []

    try:
        from django.db import transaction

        players = list(
            m.Player.objects.filter(team_id__in=[int(team1_id), int(team2_id)]).values(
                "id", "team_id", "name", "jersey_number"
            )
        )

        players_by_team: dict[int, list[dict[str, Any]]] = {}
        jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
        name_to_player_ids: dict[tuple[int, str], list[int]] = {}
        player_team_by_id: dict[int, int] = {}

        def _register_player(pid: int, tid: int, *, name: str, jersey_number: Any) -> None:
            player_team_by_id[int(pid)] = int(tid)
            p = {"id": int(pid), "team_id": int(tid), "name": name, "jersey_number": jersey_number}
            players_by_team.setdefault(int(tid), []).append(p)
            j = logic.normalize_jersey_number(jersey_number)
            if j:
                jersey_to_player_ids.setdefault((int(tid), j), []).append(int(pid))
            nm = logic.normalize_player_name(name or "")
            if nm:
                name_to_player_ids.setdefault((int(tid), nm), []).append(int(pid))

        for p in players:
            _register_player(
                int(p["id"]),
                int(p["team_id"]),
                name=str(p.get("name") or ""),
                jersey_number=p.get("jersey_number"),
            )

        def _resolve_player_id(jersey_norm: Optional[str], name_norm: str) -> Optional[int]:
            candidates: list[int] = []
            for tid in (team1_id, team2_id):
                if jersey_norm:
                    candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
            if len(set(candidates)) == 1:
                return int(list(set(candidates))[0])
            candidates = []
            for tid in (team1_id, team2_id):
                candidates.extend(name_to_player_ids.get((tid, name_norm), []))
            if len(set(candidates)) == 1:
                return int(list(set(candidates))[0])
            return None

        now = dt.datetime.now()
        with transaction.atomic():
            if isinstance(events_csv, str) and events_csv.strip():
                if replace:
                    m.HkyGameEvent.objects.update_or_create(
                        game_id=int(resolved_game_id),
                        defaults={
                            "events_csv": events_csv,
                            "source_label": source_label,
                            "updated_at": now,
                        },
                    )
                else:
                    existing = (
                        m.HkyGameEvent.objects.filter(game_id=int(resolved_game_id))
                        .values("events_csv", "source_label")
                        .first()
                    )
                    if not existing:
                        m.HkyGameEvent.objects.create(
                            game_id=int(resolved_game_id),
                            events_csv=events_csv,
                            source_label=source_label,
                            updated_at=now,
                        )
                    else:
                        try:
                            if isinstance(existing, dict):
                                existing_csv = str(existing.get("events_csv") or "")
                                existing_source = str(existing.get("source_label") or "")
                            else:
                                existing_csv = str(existing[0] or "")
                                existing_source = str(existing[1] or "")
                        except Exception:
                            existing_csv = ""
                            existing_source = ""

                        if (
                            tts_linked
                            and existing_csv.strip()
                            and existing_source.lower().startswith("timetoscore")
                        ):
                            try:
                                ex_headers, ex_rows = logic.parse_events_csv(existing_csv)
                                in_headers, in_rows = logic.parse_events_csv(events_csv)
                                if (
                                    incoming_events_headers_raw is not None
                                    and incoming_events_rows_raw is not None
                                ):
                                    ex_headers, ex_rows = (
                                        logic.enrich_timetoscore_goals_with_long_video_times(
                                            existing_headers=ex_headers,
                                            existing_rows=ex_rows,
                                            incoming_headers=incoming_events_headers_raw,
                                            incoming_rows=incoming_events_rows_raw,
                                        )
                                    )
                                    ex_headers, ex_rows = (
                                        logic.enrich_timetoscore_penalties_with_video_times(
                                            existing_headers=ex_headers,
                                            existing_rows=ex_rows,
                                            incoming_headers=incoming_events_headers_raw,
                                            incoming_rows=incoming_events_rows_raw,
                                        )
                                    )

                                def _norm_ev_type(v: Any) -> str:
                                    return str(v or "").strip().casefold()

                                protected_types = {
                                    "goal",
                                    "assist",
                                    "penalty",
                                    "penalty expired",
                                    "goaliechange",
                                }

                                def _key(r: dict[str, str]) -> tuple[str, str, str, str, str]:
                                    et = _norm_ev_type(r.get("Event Type") or r.get("Event") or "")
                                    per = str(r.get("Period") or "").strip()
                                    gs = str(
                                        r.get("Game Seconds") or r.get("GameSeconds") or ""
                                    ).strip()
                                    tr = (
                                        str(
                                            r.get("Team Side")
                                            or r.get("TeamSide")
                                            or r.get("Team Rel")
                                            or r.get("TeamRel")
                                            or r.get("Team")
                                            or ""
                                        )
                                        .strip()
                                        .casefold()
                                    )
                                    jerseys = str(r.get("Attributed Jerseys") or "").strip()
                                    return (et, per, gs, tr, jerseys)

                                seen = {_key(r) for r in (ex_rows or []) if isinstance(r, dict)}
                                merged_rows: list[dict[str, str]] = list(ex_rows or [])
                                for rr in in_rows or []:
                                    if not isinstance(rr, dict):
                                        continue
                                    et = _norm_ev_type(
                                        rr.get("Event Type") or rr.get("Event") or ""
                                    )
                                    if et in protected_types:
                                        continue
                                    k = _key(rr)
                                    if k in seen:
                                        continue
                                    seen.add(k)
                                    merged_rows.append(rr)

                                merged_headers = list(ex_headers or [])
                                for h in in_headers or []:
                                    if h not in merged_headers:
                                        merged_headers.append(h)
                                merged_csv = logic.to_csv_text(merged_headers, merged_rows)
                                m.HkyGameEvent.objects.filter(game_id=int(resolved_game_id)).update(
                                    events_csv=merged_csv, updated_at=now
                                )
                            except Exception:
                                pass

            if isinstance(player_stats_csv, str) and player_stats_csv.strip():
                try:
                    player_stats_csv = logic.sanitize_player_stats_csv_for_storage(player_stats_csv)
                except Exception:
                    pass
                if replace:
                    m.HkyGamePlayerStatsCsv.objects.update_or_create(
                        game_id=int(resolved_game_id),
                        defaults={
                            "player_stats_csv": player_stats_csv,
                            "source_label": source_label,
                            "updated_at": now,
                        },
                    )
                else:
                    if not m.HkyGamePlayerStatsCsv.objects.filter(
                        game_id=int(resolved_game_id)
                    ).exists():
                        m.HkyGamePlayerStatsCsv.objects.create(
                            game_id=int(resolved_game_id),
                            player_stats_csv=player_stats_csv,
                            source_label=source_label,
                            updated_at=now,
                        )

            if isinstance(game_stats_csv, str) and game_stats_csv.strip():
                try:
                    game_stats = logic.parse_shift_stats_game_stats_csv(game_stats_csv)
                except Exception:
                    game_stats = None
                if game_stats is not None:
                    game_stats = logic.filter_game_stats_for_display(game_stats)
                    m.HkyGameStat.objects.update_or_create(
                        game_id=int(resolved_game_id),
                        defaults={
                            "stats_json": json.dumps(game_stats, ensure_ascii=False),
                            "updated_at": now,
                        },
                    )

            if isinstance(player_stats_csv, str) and player_stats_csv.strip():
                parsed_rows = logic.parse_shift_stats_player_stats_csv(player_stats_csv)
                if replace:
                    m.PlayerStat.objects.filter(game_id=int(resolved_game_id)).update(
                        toi_seconds=None,
                        shifts=None,
                        video_toi_seconds=None,
                        sb_avg_shift_seconds=None,
                        sb_median_shift_seconds=None,
                        sb_longest_shift_seconds=None,
                        sb_shortest_shift_seconds=None,
                    )
                    m.PlayerPeriodStat.objects.filter(game_id=int(resolved_game_id)).delete()
                for row in parsed_rows:
                    jersey_norm = row.get("jersey_number")
                    name_norm = row.get("name_norm") or ""
                    pid = _resolve_player_id(jersey_norm, name_norm)
                    if pid is None:
                        if create_missing_players and team_side in {"home", "away"}:
                            target_team_id = team1_id if team_side == "home" else team2_id
                            disp = str(row.get("display_name") or "").strip()
                            if not disp:
                                disp = str(row.get("player_label") or "").strip()
                                match = re.match(r"^\s*\d+\s+(.*)$", disp)
                                if match:
                                    disp = str(match.group(1) or "").strip()
                            if disp:
                                try:
                                    pid = _ensure_player_for_import(
                                        owner_user_id,
                                        int(target_team_id),
                                        disp,
                                        str(jersey_norm or "").strip() or None,
                                        None,
                                        commit=False,
                                    )
                                    _register_player(
                                        int(pid),
                                        int(target_team_id),
                                        name=str(disp),
                                        jersey_number=jersey_norm,
                                    )
                                except Exception:
                                    pid = None
                        if pid is None:
                            unmatched.append(row.get("player_label") or "")
                            continue

                    team_id = player_team_by_id.get(int(pid))
                    if team_id is None:
                        if create_missing_players and team_side in {"home", "away"}:
                            team_id = team1_id if team_side == "home" else team2_id
                        else:
                            unmatched.append(row.get("player_label") or "")
                            continue

                    stats = row.get("stats") or {}
                    if tts_linked:
                        stats = dict(stats)
                        stats["goals"] = None
                        stats["assists"] = None
                    cols = [
                        "goals",
                        "assists",
                        "shots",
                        "pim",
                        "plus_minus",
                        "hits",
                        "blocks",
                        "faceoff_wins",
                        "faceoff_attempts",
                        "goalie_saves",
                        "goalie_ga",
                        "goalie_sa",
                        "sog",
                        "expected_goals",
                        "giveaways",
                        "turnovers_forced",
                        "created_turnovers",
                        "takeaways",
                        "controlled_entry_for",
                        "controlled_entry_against",
                        "controlled_exit_for",
                        "controlled_exit_against",
                        "gt_goals",
                        "gw_goals",
                        "ot_goals",
                        "ot_assists",
                        "gf_counted",
                        "ga_counted",
                    ]
                    defaults = {
                        "user_id": int(owner_user_id),
                        "team_id": int(team_id),
                        **{c: stats.get(c) for c in cols},
                    }
                    ps, created = m.PlayerStat.objects.get_or_create(
                        game_id=int(resolved_game_id),
                        player_id=int(pid),
                        defaults=defaults,
                    )
                    if not created:
                        updates = {c: stats.get(c) for c in cols if stats.get(c) is not None}
                        if updates:
                            m.PlayerStat.objects.filter(id=ps.id).update(**updates)
                    imported += 1

            if player_stats_csv or game_stats_csv or events_csv:
                m.HkyGame.objects.filter(id=int(resolved_game_id)).update(stats_imported_at=now)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return JsonResponse(
        {
            "ok": True,
            "game_id": int(resolved_game_id),
            "imported_players": int(imported),
            "unmatched": [u for u in unmatched if u],
        }
    )


@csrf_exempt
def api_internal_reset_league_data(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "").strip()
    owner_email = str(payload.get("owner_email") or "").strip().lower()
    if not league_name or not owner_email:
        return JsonResponse(
            {"ok": False, "error": "owner_email and league_name are required"}, status=400
        )

    _django_orm, m = _orm_modules()
    owner_user_id = m.User.objects.filter(email=owner_email).values_list("id", flat=True).first()
    if owner_user_id is None:
        return JsonResponse({"ok": False, "error": "owner_email_not_found"}, status=404)
    league_id = (
        m.League.objects.filter(name=league_name, owner_user_id=int(owner_user_id))
        .values_list("id", flat=True)
        .first()
    )
    if league_id is None:
        return JsonResponse({"ok": False, "error": "league_not_found_for_owner"}, status=404)

    try:
        stats = logic.reset_league_data(None, int(league_id), owner_user_id=int(owner_user_id))
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "league_id": int(league_id), "stats": stats})


@csrf_exempt
def api_internal_ensure_league_owner(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "").strip()
    owner_email = str(payload.get("owner_email") or "").strip().lower()
    owner_name = str(payload.get("owner_name") or owner_email).strip() or owner_email
    is_shared = bool(payload["shared"]) if "shared" in payload else None
    if not league_name or not owner_email:
        return JsonResponse(
            {"ok": False, "error": "owner_email and league_name are required"}, status=400
        )

    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
    _django_orm, m = _orm_modules()
    from django.db import transaction

    now = dt.datetime.now()
    with transaction.atomic():
        existing = m.League.objects.filter(name=league_name).values("id").first()
        if existing:
            league_id = int(existing["id"])
            updates: dict[str, Any] = {"owner_user_id": int(owner_user_id), "updated_at": now}
            if is_shared is not None:
                updates["is_shared"] = bool(is_shared)
            m.League.objects.filter(id=league_id).update(**updates)
        else:
            if is_shared is None:
                is_shared = True
            league = m.League.objects.create(
                name=league_name,
                owner_user_id=int(owner_user_id),
                is_shared=bool(is_shared),
                is_public=False,
                created_at=now,
                updated_at=None,
            )
            league_id = int(league.id)

        member, created = m.LeagueMember.objects.get_or_create(
            league_id=int(league_id),
            user_id=int(owner_user_id),
            defaults={"role": "owner", "created_at": now},
        )
        if not created and str(getattr(member, "role", "") or "") != "owner":
            m.LeagueMember.objects.filter(id=int(member.id)).update(role="owner")

    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id)}
    )


@csrf_exempt
def api_internal_ensure_user(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    email = str(payload.get("email") or payload.get("user_email") or "").strip().lower()
    name = str(payload.get("name") or payload.get("user_name") or email).strip() or email
    password = str(payload.get("password") or "password")
    if not email:
        return JsonResponse({"ok": False, "error": "email is required"}, status=400)
    _django_orm, m = _orm_modules()
    existing = m.User.objects.filter(email=email).values_list("id", flat=True).first()
    if existing is not None:
        return JsonResponse({"ok": True, "user_id": int(existing), "created": False})
    pwd_hash = generate_password_hash(password)
    now = dt.datetime.now()
    u = m.User.objects.create(
        email=email,
        password_hash=pwd_hash,
        name=name,
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    return JsonResponse({"ok": True, "user_id": int(u.id), "created": True})


@csrf_exempt
def api_internal_recalc_div_ratings(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_id_raw = payload.get("league_id") or payload.get("lid") or request.GET.get("league_id")
    league_name = str(
        payload.get("league_name") or payload.get("name") or request.GET.get("league_name") or ""
    ).strip()
    max_goal_diff_raw = payload.get("max_goal_diff") or payload.get("maxGoalDiff") or None
    min_games_raw = payload.get("min_games") or payload.get("minGames") or None

    max_goal_diff = int(max_goal_diff_raw) if max_goal_diff_raw is not None else 7
    min_games = int(min_games_raw) if min_games_raw is not None else 4

    _django_orm, m = _orm_modules()

    league_ids: list[int]
    if league_id_raw:
        try:
            league_ids = [int(league_id_raw)]
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid league_id"}, status=400)
    elif league_name:
        row = m.League.objects.filter(name=league_name).values("id").first()
        if not row:
            return JsonResponse({"ok": False, "error": "league_not_found"}, status=404)
        league_ids = [int(row["id"])]
    else:
        league_ids = list(m.League.objects.order_by("id").values_list("id", flat=True))

    ok_ids: list[int] = []
    failed: list[dict[str, Any]] = []
    for lid in league_ids:
        try:
            logic.recompute_league_mhr_ratings(
                None, int(lid), max_goal_diff=max_goal_diff, min_games=min_games
            )
            ok_ids.append(int(lid))
        except Exception as e:  # noqa: BLE001
            failed.append({"league_id": int(lid), "error": str(e)})

    if failed:
        return JsonResponse({"ok": False, "league_ids_ok": ok_ids, "failed": failed}, status=500)
    return JsonResponse({"ok": True, "league_ids": ok_ids})

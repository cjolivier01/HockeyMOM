from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.contrib import messages
from django.db.models import Q
from django.http import FileResponse, HttpRequest, HttpResponse, HttpResponseNotFound, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from werkzeug.security import check_password_hash, generate_password_hash

from .models import (
    Game,
    GameType,
    HkyGame,
    Job,
    League,
    LeagueGame,
    LeagueMember,
    LeagueTeam,
    Player,
    PlayerStat,
    Team,
    User,
)
from .utils import (
    aggregate_players_totals,
    compute_team_stats,
    parse_dt_or_none,
    read_dirwatch_state,
    send_email,
)


def _require_login(request: HttpRequest) -> Optional[HttpResponse]:
    if "user_id" not in request.session:
        return redirect("login")
    return None


def index(request: HttpRequest) -> HttpResponse:
    if request.session.get("user_id"):
        return redirect("games")
    return render(request, "index.html")


@require_http_methods(["GET", "POST"])
def register(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        name = request.POST.get("name", "").strip()
        if not email or not password:
            messages.error(request, "Email and password are required")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered")
        else:
            pw_hash = generate_password_hash(password)
            user = User.objects.create(
                email=email,
                password_hash=pw_hash,
                name=name or None,
                created_at=dt.datetime.now(),
            )
            request.session["user_id"] = user.id
            request.session["user_email"] = user.email
            request.session["user_name"] = user.name or user.email
            try:
                send_email(
                    to_addr=email,
                    subject="Welcome to HM WebApp",
                    body=(
                        f"Hello {name or email},\n\n"
                        "Your account has been created successfully.\n"
                        "You can now create a game, upload files, and run jobs.\n\n"
                        "Regards,\nHM"
                    ),
                )
            except Exception:
                pass
            return redirect("games")
    return render(request, "register.html")


@require_http_methods(["GET", "POST"])
def login_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            user = None
        if not user or not check_password_hash(user.password_hash, password):
            messages.error(request, "Invalid credentials")
        else:
            request.session["user_id"] = user.id
            request.session["user_email"] = user.email
            request.session["user_name"] = user.name or user.email
            return redirect("games")
    return render(request, "login.html")


@require_http_methods(["GET", "POST"])
def forgot_password(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        try:
            user = User.objects.filter(email=email).first()
            if user:
                import secrets

                token = secrets.token_urlsafe(32)
                expires = dt.datetime.now() + dt.timedelta(hours=1)
                Reset = getattr(__import__("hmwebapp.webapp.models", fromlist=["Reset"]), "Reset")
                Reset.objects.create(
                    user=user,
                    token=token,
                    expires_at=expires,
                    created_at=dt.datetime.now(),
                )
                base = request.build_absolute_uri("/").rstrip("/")
                link = f"{base}{reverse('reset_password', args=[token])}"
                send_email(
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
        messages.success(
            request,
            "If the account exists, a reset email has been sent.",
        )
        return redirect("login")
    return render(request, "forgot_password.html")


@require_http_methods(["GET", "POST"])
def reset_password(request: HttpRequest, token: str) -> HttpResponse:
    Reset = getattr(__import__("hmwebapp.webapp.models", fromlist=["Reset"]), "Reset")
    try:
        reset = Reset.objects.select_related("user").get(token=token)
    except Reset.DoesNotExist:
        messages.error(request, "Invalid or expired token")
        return redirect("login")
    now = dt.datetime.now()
    if reset.used_at or now > reset.expires_at:
        messages.error(request, "Invalid or expired token")
        return redirect("login")
    if request.method == "POST":
        pw1 = request.POST.get("password", "")
        pw2 = request.POST.get("password2", "")
        if not pw1 or pw1 != pw2:
            messages.error(request, "Passwords do not match")
            return render(request, "reset_password.html")
        reset.user.password_hash = generate_password_hash(pw1)
        reset.user.save(update_fields=["password_hash"])
        reset.used_at = dt.datetime.now()
        reset.save(update_fields=["used_at"])
        messages.success(request, "Password updated. Please log in.")
        return redirect("login")
    return render(request, "reset_password.html")


def logout_view(request: HttpRequest) -> HttpResponse:
    request.session.flush()
    return redirect("index")


@require_http_methods(["POST"])
def league_select(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    lid = request.POST.get("league_id")
    if lid and lid.isdigit():
        lid_i = int(lid)
        is_owner = League.objects.filter(id=lid_i, owner_user_id=uid).exists()
        is_member = LeagueMember.objects.filter(league_id=lid_i, user_id=uid).exists()
        if is_owner or is_member:
            request.session["league_id"] = lid_i
            User.objects.filter(pk=uid).update(default_league_id=lid_i)
    else:
        request.session.pop("league_id", None)
        User.objects.filter(pk=uid).update(default_league_id=None)
    return redirect(request.META.get("HTTP_REFERER") or reverse("index"))


def games(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    games_qs = Game.objects.filter(user_id=uid).order_by("-created_at")
    games_list: List[Game] = list(games_qs)
    state = read_dirwatch_state()
    for g in games_list:
        latest_status = state.get("processed", {}).get(g.dir_path, {}).get("status") or g.status
        setattr(g, "latest_status", latest_status)
    return render(request, "games.html", {"games": games_list})


@require_http_methods(["GET", "POST"])
def new_game(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        if not name:
            name = f"game-{dt.datetime.now():%Y%m%d-%H%M%S}"
        uid = int(request.session["user_id"])
        email = request.session.get("user_email") or ""
        ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        import secrets

        token = secrets.token_hex(4)
        d = Path(settings.WATCH_ROOT) / f"game_{uid}_{ts}_{token}"
        d.mkdir(parents=True, exist_ok=True)
        try:
            (d / ".dirwatch_meta.json").write_text(
                f'{{"user_email":"{email}","created":"{dt.datetime.now().isoformat()}"}}\n'
            )
        except Exception:
            pass
        game = Game.objects.create(
            user_id=uid,
            name=name,
            dir_path=str(d),
            created_at=dt.datetime.now(),
        )
        messages.success(request, "Game created")
        return redirect("game_detail", gid=game.id)
    return render(request, "new_game.html")


def game_detail(request: HttpRequest, gid: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    game = get_object_or_404(Game, id=gid, user_id=uid)
    files: List[str] = []
    try:
        if os.path.isdir(game.dir_path):
            all_files = os.listdir(game.dir_path)
        else:
            all_files = []

        def _is_user_file(fname: str) -> bool:
            if not fname:
                return False
            if fname.startswith("."):
                return False
            if fname.startswith("_"):
                return False
            if fname.startswith("slurm-"):
                return False
            return True

        files = [f for f in sorted(all_files) if _is_user_file(f)]
    except Exception:
        files = []

    latest_status: Optional[str] = None
    row = (
        Job.objects.filter(game_id=gid)
        .order_by("-id")
        .values_list("status", flat=True)
        .first()
    )
    if row:
        latest_status = str(row) if row is not None else None
    if not latest_status:
        state = read_dirwatch_state()
        latest_status = state.get("processed", {}).get(game.dir_path, {}).get("status") or game.status
    is_locked = False
    if row:
        is_locked = True
    final_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
    if latest_status and str(latest_status).upper() in final_states:
        is_locked = True
    return render(
        request,
        "game_detail.html",
        {"game": game, "files": files, "status": latest_status, "is_locked": is_locked},
    )


@require_http_methods(["GET", "POST"])
def delete_game(request: HttpRequest, gid: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    game = get_object_or_404(Game, id=gid, user_id=uid)
    latest = (
        Job.objects.filter(game_id=gid)
        .order_by("-id")
        .values("id", "slurm_job_id", "status")
        .first()
    )
    if request.method == "POST":
        token = (request.POST.get("confirm") or "").strip().upper()
        if token != "DELETE":
            messages.error(request, "Type DELETE to confirm permanent deletion.")
            return render(request, "confirm_delete.html", {"game": game})
        try:
            active_states = {"SUBMITTED", "RUNNING", "PENDING"}
            if latest and str(latest.get("status", "")).upper() in active_states:
                import subprocess
                import time

                dir_leaf = Path(game.dir_path).name
                job_name = f"dirwatch-{dir_leaf}"
                job_ids: List[str] = []
                jid = latest.get("slurm_job_id")
                if jid:
                    job_ids.append(str(jid))
                else:
                    try:
                        out = subprocess.check_output(["squeue", "-h", "-o", "%i %j"]).decode()
                        for line in out.splitlines():
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2 and parts[1] == job_name:
                                job_ids.append(parts[0])
                    except Exception:
                        pass
                for jid_ in job_ids:
                    subprocess.run(["scancel", str(jid_)], check=False)
                deadline = time.time() + 20.0
                while time.time() < deadline:
                    try:
                        out = subprocess.check_output(["squeue", "-h", "-o", "%i %j"]).decode()
                        still = False
                        for line in out.splitlines():
                            parts = line.strip().split(maxsplit=1)
                            if not parts:
                                continue
                            if parts[0] in job_ids or (len(parts) == 2 and parts[1] == job_name):
                                still = True
                                break
                        if not still:
                            break
                    except Exception:
                        break
                    time.sleep(1.0)
        except Exception:
            pass
        Job.objects.filter(game_id=gid).delete()
        game.delete()
        try:
            d = Path(game.dir_path).resolve()
            wr = Path(settings.WATCH_ROOT).resolve()
            if str(d).startswith(str(wr)):
                import shutil

                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        messages.success(request, "Game deleted.")
        return redirect("games")
    return render(request, "confirm_delete.html", {"game": game})


@require_http_methods(["POST"])
def upload_files(request: HttpRequest, gid: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    game = get_object_or_404(Game, id=gid, user_id=uid)
    latest = Job.objects.filter(game_id=gid).order_by("-id").first()
    if latest:
        messages.error(request, "Job already submitted; uploads disabled.")
        return redirect("game_detail", gid=gid)
    updir = Path(game.dir_path)
    updir.mkdir(parents=True, exist_ok=True)
    files = request.FILES.getlist("files")
    count = 0
    for f in files:
        if not f or not f.name:
            continue
        dest = updir / Path(f.name).name
        with dest.open("wb") as fh:
            for chunk in f.chunks():
                fh.write(chunk)
        count += 1
    messages.success(request, f"Uploaded {count} files")
    return redirect("game_detail", gid=gid)


@require_http_methods(["POST"])
def run_game(request: HttpRequest, gid: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    game = get_object_or_404(Game, id=gid, user_id=uid)
    existing = Job.objects.filter(game_id=gid).order_by("-id").first()
    if existing:
        messages.error(request, "Job already submitted.")
        return redirect("game_detail", gid=gid)
    dir_path = Path(game.dir_path)
    try:
        meta = dir_path / ".dirwatch_meta.json"
        meta.write_text(
            f'{{"user_email":"{request.session.get("user_email", "")}","game_id":{gid},"created":"{dt.datetime.now().isoformat()}"}}\n'
        )
    except Exception:
        pass
    (dir_path / "_READY").touch(exist_ok=True)
    game.status = "submitted"
    game.save(update_fields=["status"])
    Job.objects.create(
        user_id=uid,
        game=game,
        dir_path=str(dir_path),
        status="PENDING",
        created_at=dt.datetime.now(),
        user_email=request.session.get("user_email") or "",
    )
    messages.success(request, "Run requested. Job will start shortly.")
    return redirect("game_detail", gid=gid)


def serve_upload(request: HttpRequest, gid: int, name: str) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    game = get_object_or_404(Game, id=gid, user_id=uid)
    d = Path(game.dir_path).resolve()
    target = (d / name).resolve()
    if not str(target).startswith(str(d)) or not target.is_file():
        return HttpResponseNotFound("Not found")
    return FileResponse(target.open("rb"), as_attachment=True, filename=target.name)


def jobs(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    jobs_qs = Job.objects.filter(user_id=uid).order_by("-created_at")
    return render(request, "jobs.html", {"jobs": jobs_qs})


def healthcheck(request: HttpRequest) -> HttpResponse:
    from django.db import connections

    db_ok = True
    try:
        with connections["default"].cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception:
        db_ok = False
    status = "ok" if db_ok else "degraded"
    return JsonResponse({"status": status, "db": db_ok})


def media_team_logo(request: HttpRequest, team_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    team = get_object_or_404(Team, id=team_id, user_id=uid)
    if not team.logo_path:
        return HttpResponseNotFound("Not found")
    p = Path(team.logo_path).resolve()
    if not p.exists():
        return HttpResponseNotFound("Not found")
    return FileResponse(p.open("rb"), content_type="image/png")


def teams(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    include_external = request.GET.get("all", "0") == "1"
    league_id = request.session.get("league_id")
    if league_id:
        teams_qs = (
            Team.objects.filter(league_teams__league_id=league_id)
            .order_by("name")
            .distinct()
        )
    else:
        teams_qs = Team.objects.filter(user_id=uid)
        if not include_external:
            teams_qs = teams_qs.filter(is_external=False)
        teams_qs = teams_qs.order_by("name")
    teams_list: List[Team] = list(teams_qs)
    stats: Dict[int, Dict[str, int]] = {}
    if league_id:
        lg_game_ids = list(
            LeagueGame.objects.filter(league_id=league_id).values_list("game_id", flat=True)
        )
        rows = list(
            HkyGame.objects.filter(id__in=lg_game_ids)
            .values("team1_id", "team2_id", "team1_score", "team2_score", "is_final")
        )
        for t in teams_list:
            stats[t.id] = compute_team_stats(rows, t.id)
    else:
        all_rows = list(
            HkyGame.objects.filter(user_id=uid)
            .values("team1_id", "team2_id", "team1_score", "team2_score", "is_final")
        )
        for t in teams_list:
            stats[t.id] = compute_team_stats(all_rows, t.id)
    for t in teams_list:
        setattr(
            t,
            "stats",
            stats.get(
                t.id,
                {"wins": 0, "losses": 0, "ties": 0, "gf": 0, "ga": 0, "points": 0},
            ),
        )
    return render(
        request,
        "teams.html",
        {"teams": teams_list, "include_external": include_external},
    )


def leagues_index(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    leagues: List[Dict[str, Any]] = []
    qs = (
        League.objects.filter(Q(owner_user_id=uid) | Q(memberships__user_id=uid))
        .distinct()
        .order_by("name")
    )
    for l in qs:
        is_owner = int(l.owner_user_id) == uid
        is_admin = is_owner or LeagueMember.objects.filter(
            league=l, user_id=uid, role__in=["admin", "owner"]
        ).exists()
        leagues.append(
            {
                "id": l.id,
                "name": l.name,
                "is_shared": l.is_shared,
                "is_admin": is_admin,
            }
        )
    return render(request, "leagues.html", {"leagues": leagues})


@require_http_methods(["POST"])
def leagues_new(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    name = (request.POST.get("name") or "").strip()
    is_shared = request.POST.get("is_shared") == "1"
    if not name:
        messages.error(request, "Name is required")
        return redirect("leagues_index")
    try:
        league = League.objects.create(
            name=name,
            owner_user_id=uid,
            is_shared=is_shared,
            created_at=dt.datetime.now(),
        )
        LeagueMember.objects.create(
            league=league,
            user_id=uid,
            role="admin",
            created_at=dt.datetime.now(),
        )
    except Exception:
        messages.error(request, "Failed to create league (name may already exist)")
        return redirect("leagues_index")
    request.session["league_id"] = league.id
    messages.success(request, "League created and selected")
    return redirect("leagues_index")


@require_http_methods(["POST"])
def leagues_delete(request: HttpRequest, league_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    league = get_object_or_404(League, id=league_id)
    if int(league.owner_user_id) != uid:
        messages.error(request, "Not authorized to delete this league")
        return redirect("leagues_index")
    try:
        game_ids = list(
            LeagueGame.objects.filter(league_id=league_id).values_list("game_id", flat=True)
        )
        if game_ids:
            PlayerStat.objects.filter(game_id__in=game_ids).delete()
            LeagueGame.objects.filter(league_id=league_id).delete()
            other_refs = set(
                LeagueGame.objects.filter(game_id__in=game_ids)
                .exclude(league_id=league_id)
                .values_list("game_id", flat=True)
            )
            to_delete = [gid for gid in game_ids if gid not in other_refs]
            if to_delete:
                HkyGame.objects.filter(id__in=to_delete).delete()
        team_ids = list(
            LeagueTeam.objects.filter(league_id=league_id).values_list("team_id", flat=True)
        )
        if team_ids:
            LeagueTeam.objects.filter(league_id=league_id).delete()
            Player.objects.filter(team_id__in=team_ids).delete()
            still_mapped = set(
                LeagueTeam.objects.filter(team_id__in=team_ids).values_list("team_id", flat=True)
            )
            deleteable: List[int] = []
            for tid in team_ids:
                if tid in still_mapped:
                    continue
                cnt = HkyGame.objects.filter(Q(team1_id=tid) | Q(team2_id=tid)).count()
                if cnt == 0:
                    deleteable.append(tid)
            if deleteable:
                Team.objects.filter(id__in=deleteable).delete()
        LeagueMember.objects.filter(league_id=league_id).delete()
        league.delete()
        if request.session.get("league_id") == league_id:
            request.session.pop("league_id", None)
        messages.success(request, "League and associated data deleted")
    except Exception:
        messages.error(request, "Failed to delete league")
    return redirect("leagues_index")


def _is_league_admin(league_id: int, user_id: int) -> bool:
    league = League.objects.filter(id=league_id, owner_user_id=user_id).exists()
    if league:
        return True
    return LeagueMember.objects.filter(
        league_id=league_id, user_id=user_id, role__in=["admin", "owner"]
    ).exists()


def league_members(request: HttpRequest, league_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    if not _is_league_admin(league_id, uid):
        messages.error(request, "Not authorized")
        return redirect("leagues_index")
    members = list(
        LeagueMember.objects.filter(league_id=league_id)
        .select_related("user")
        .order_by("user__email")
    )
    rows = [
        {
            "id": m.user_id,
            "email": m.user.email,
            "role": m.role or "admin",
        }
        for m in members
    ]
    return render(
        request,
        "league_members.html",
        {"league_id": league_id, "members": rows},
    )


@require_http_methods(["POST"])
def league_members_add(request: HttpRequest, league_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    if not _is_league_admin(league_id, uid):
        messages.error(request, "Not authorized")
        return redirect("leagues_index")
    email = (request.POST.get("email") or "").strip().lower()
    role = request.POST.get("role", "viewer")
    if not email:
        messages.error(request, "Email required")
        return redirect("league_members", league_id=league_id)
    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        messages.error(request, "User not found. Ask them to register first.")
        return redirect("league_members", league_id=league_id)
    LeagueMember.objects.update_or_create(
        league_id=league_id,
        user=user,
        defaults={"role": role, "created_at": dt.datetime.now()},
    )
    messages.success(request, "Member added/updated")
    return redirect("league_members", league_id=league_id)


@require_http_methods(["POST"])
def league_members_remove(request: HttpRequest, league_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    if not _is_league_admin(league_id, uid):
        messages.error(request, "Not authorized")
        return redirect("leagues_index")
    user_id = int(request.POST.get("user_id") or 0)
    LeagueMember.objects.filter(league_id=league_id, user_id=user_id).delete()
    messages.success(request, "Member removed")
    return redirect("league_members", league_id=league_id)


@require_http_methods(["GET", "POST"])
def new_team(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        if not name:
            messages.error(request, "Team name is required")
            return render(request, "team_new.html")
        team = Team.objects.create(
            user_id=uid,
            name=name,
            is_external=False,
            created_at=dt.datetime.now(),
        )
        f = request.FILES.get("logo")
        if f and f.name:
            uploads = Path(Path(__file__).resolve().parents[2] / "instance" / "uploads" / "team_logos")
            uploads.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            dest = uploads / f"team{team.id}_{ts}_{Path(f.name).name}"
            with dest.open("wb") as fh:
                for chunk in f.chunks():
                    fh.write(chunk)
            team.logo_path = str(dest)
            team.save(update_fields=["logo_path"])
        messages.success(request, "Team created")
        return redirect("team_detail", team_id=team.id)
    return render(request, "team_new.html")


def team_detail(request: HttpRequest, team_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    team = get_object_or_404(Team, id=team_id, user_id=uid)
    players = list(
        Player.objects.filter(team_id=team_id, user_id=uid).order_by("jersey_number", "name")
    )
    rows = list(
        PlayerStat.objects.filter(team_id=team_id, user_id=uid)
        .values("player_id", "goals", "assists", "pim", "shots")
    )
    player_totals = aggregate_players_totals(rows)
    game_rows = list(
        HkyGame.objects.filter(user_id=uid).values(
            "team1_id", "team2_id", "team1_score", "team2_score", "is_final"
        )
    )
    tstats = compute_team_stats(game_rows, team_id)
    player_rows = []
    for p in players:
        agg = player_totals.get(
            p.id,
            {"goals": 0, "assists": 0, "points": 0, "shots": 0, "pim": 0},
        )
        player_rows.append({"player": p, "stats": agg})
    return render(
        request,
        "team_detail.html",
        {
            "team": team,
            "player_rows": player_rows,
            "tstats": tstats,
        },
    )


@require_http_methods(["GET", "POST"])
def team_edit(request: HttpRequest, team_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    team = get_object_or_404(Team, id=team_id, user_id=uid)
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        if name:
            team.name = name
            team.save(update_fields=["name"])
        f = request.FILES.get("logo")
        if f and f.name:
            uploads = Path(Path(__file__).resolve().parents[2] / "instance" / "uploads" / "team_logos")
            uploads.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            dest = uploads / f"team{team.id}_{ts}_{Path(f.name).name}"
            with dest.open("wb") as fh:
                for chunk in f.chunks():
                    fh.write(chunk)
            team.logo_path = str(dest)
            team.save(update_fields=["logo_path"])
        messages.success(request, "Team updated")
        return redirect("team_detail", team_id=team_id)
    return render(request, "team_edit.html", {"team": team})


@require_http_methods(["GET", "POST"])
def player_new(request: HttpRequest, team_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    team = get_object_or_404(Team, id=team_id, user_id=uid)
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        jersey = (request.POST.get("jersey_number") or "").strip()
        position = (request.POST.get("position") or "").strip()
        shoots = (request.POST.get("shoots") or "").strip()
        if not name:
            messages.error(request, "Player name is required")
            return render(request, "player_edit.html", {"team": team})
        Player.objects.create(
            user_id=uid,
            team=team,
            name=name,
            jersey_number=jersey or None,
            position=position or None,
            shoots=shoots or None,
            created_at=dt.datetime.now(),
        )
        messages.success(request, "Player added")
        return redirect("team_detail", team_id=team_id)
    return render(request, "player_edit.html", {"team": team})


@require_http_methods(["GET", "POST"])
def player_edit(request: HttpRequest, team_id: int, player_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    team = get_object_or_404(Team, id=team_id, user_id=uid)
    player = get_object_or_404(Player, id=player_id, team_id=team_id, user_id=uid)
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        jersey = (request.POST.get("jersey_number") or "").strip()
        position = (request.POST.get("position") or "").strip()
        shoots = (request.POST.get("shoots") or "").strip()
        player.name = name or player.name
        player.jersey_number = jersey or None
        player.position = position or None
        player.shoots = shoots or None
        player.save(
            update_fields=["name", "jersey_number", "position", "shoots"],
        )
        messages.success(request, "Player updated")
        return redirect("team_detail", team_id=team_id)
    return render(request, "player_edit.html", {"team": team, "player": player})


@require_http_methods(["POST"])
def player_delete(request: HttpRequest, team_id: int, player_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    Player.objects.filter(id=player_id, team_id=team_id, user_id=uid).delete()
    messages.success(request, "Player deleted")
    return redirect("team_detail", team_id=team_id)


def schedule(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    league_id = request.session.get("league_id")
    if league_id:
        games_qs = (
            LeagueGame.objects.filter(league_id=league_id)
            .select_related("game__team1", "game__team2", "game__game_type")
            .order_by("-game__starts_at", "-game__created_at")
        )
        games_list = [
            {
                "id": lg.game.id,
                "team1_name": lg.game.team1.name,
                "team2_name": lg.game.team2.name,
                "game_type_name": lg.game.game_type.name if lg.game.game_type else "",
                "starts_at": lg.game.starts_at or lg.game.created_at,
                "is_final": lg.game.is_final,
                "team1_score": lg.game.team1_score,
                "team2_score": lg.game.team2_score,
            }
            for lg in games_qs
        ]
    else:
        games_qs = (
            HkyGame.objects.filter(user_id=uid)
            .select_related("team1", "team2", "game_type")
            .order_by("-starts_at", "-created_at")
        )
        games_list = [
            {
                "id": g.id,
                "team1_name": g.team1.name,
                "team2_name": g.team2.name,
                "game_type_name": g.game_type.name if g.game_type else "",
                "starts_at": g.starts_at or g.created_at,
                "is_final": g.is_final,
                "team1_score": g.team1_score,
                "team2_score": g.team2_score,
            }
            for g in games_qs
        ]
    return render(request, "schedule.html", {"games": games_list})


@require_http_methods(["GET", "POST"])
def schedule_new(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    my_teams = list(
        Team.objects.filter(user_id=uid, is_external=False)
        .order_by("name")
        .values("id", "name")
    )
    game_types = list(GameType.objects.order_by("name").values("id", "name"))
    if request.method == "POST":
        team1_id = int(request.POST.get("team1_id") or 0)
        team2_id = int(request.POST.get("team2_id") or 0)
        opp_name = (request.POST.get("opponent_name") or "").strip()
        game_type_id = int(request.POST.get("game_type_id") or 0)
        starts_at = (request.POST.get("starts_at") or "").strip()
        location = (request.POST.get("location") or "").strip()
        if not team1_id and not team2_id:
            messages.error(request, "Select at least one of your teams")
            return render(
                request,
                "schedule_new.html",
                {"my_teams": my_teams, "game_types": game_types},
            )
        def _ensure_external_team(user_id: int, name: str) -> int:
            t = Team.objects.filter(user_id=user_id, name=name).first()
            if t:
                return t.id
            t = Team.objects.create(
                user_id=user_id,
                name=name,
                is_external=True,
                created_at=dt.datetime.now(),
            )
            return t.id

        if team1_id and not team2_id:
            team2_id = _ensure_external_team(uid, opp_name or "Opponent")
        elif team2_id and not team1_id:
            team1_id = _ensure_external_team(uid, opp_name or "Opponent")
        start_str = parse_dt_or_none(starts_at)
        starts_dt = dt.datetime.fromisoformat(start_str) if start_str else None
        game = HkyGame.objects.create(
            user_id=uid,
            team1_id=team1_id,
            team2_id=team2_id,
            game_type_id=game_type_id or None,
            starts_at=starts_dt,
            location=location or None,
            created_at=dt.datetime.now(),
        )
        league_id = request.session.get("league_id")
        if league_id:
            try:
                LeagueTeam.objects.get_or_create(league_id=league_id, team_id=team1_id)
                LeagueTeam.objects.get_or_create(league_id=league_id, team_id=team2_id)
                LeagueGame.objects.get_or_create(league_id=league_id, game=game)
            except Exception:
                pass
        messages.success(request, "Game created")
        return redirect("hky_game_detail", game_id=game.id)
    return render(
        request,
        "schedule_new.html",
        {"my_teams": my_teams, "game_types": game_types},
    )


@require_http_methods(["GET", "POST"])
def hky_game_detail(request: HttpRequest, game_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    uid = int(request.session["user_id"])
    league_id = request.session.get("league_id")
    game = (
        HkyGame.objects.select_related("team1", "team2")
        .filter(id=game_id, user_id=uid)
        .first()
    )
    if not game and league_id:
        game = (
            HkyGame.objects.select_related("team1", "team2")
            .filter(id=game_id, league_games__league_id=league_id)
            .first()
        )
    if not game:
        messages.error(request, "Not found")
        return redirect("schedule")
    team1_players = list(
        Player.objects.filter(team_id=game.team1_id, user_id=uid)
        .order_by("jersey_number", "name")
        .values()
    )
    team2_players = list(
        Player.objects.filter(team_id=game.team2_id, user_id=uid)
        .order_by("jersey_number", "name")
        .values()
    )
    stats_rows = list(PlayerStat.objects.filter(game_id=game_id).values())
    stats_by_pid = {int(r["player_id"]): r for r in stats_rows}
    editable = True
    if league_id and request.method == "POST":
        can_edit = LeagueMember.objects.filter(
            league_id=league_id, user_id=uid, role__in=["admin", "owner", "editor"]
        ).exists()
        if not can_edit:
            editable = False
            messages.error(
                request,
                "You do not have permission to edit this game in the selected league.",
            )
    if request.method == "POST" and editable:
        loc = (request.POST.get("location") or "").strip()
        starts_at = (request.POST.get("starts_at") or "").strip()
        t1_score = request.POST.get("team1_score")
        t2_score = request.POST.get("team2_score")
        is_final = bool(request.POST.get("is_final"))
        start_str = parse_dt_or_none(starts_at)
        starts_dt = dt.datetime.fromisoformat(start_str) if start_str else None
        game.location = loc or None
        game.starts_at = starts_dt
        game.team1_score = int(t1_score) if (t1_score or "").strip() else None
        game.team2_score = int(t2_score) if (t2_score or "").strip() else None
        game.is_final = bool(is_final)
        game.save(
            update_fields=[
                "location",
                "starts_at",
                "team1_score",
                "team2_score",
                "is_final",
            ]
        )

        def _collect(prefix: str, pid: int) -> Dict[str, Optional[int]]:
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
                "toi_seconds": _ival("toi"),
                "faceoff_wins": _ival("fow"),
                "faceoff_attempts": _ival("foa"),
                "goalie_saves": _ival("saves"),
                "goalie_ga": _ival("ga"),
                "goalie_sa": _ival("sa"),
            }

        for p in list(team1_players) + list(team2_players):
            pid = int(p["id"])
            vals = _collect("ps", pid)
            team_id = int(p["team_id"])
            ps, _created = PlayerStat.objects.get_or_create(
                game=game,
                player_id=pid,
                defaults={
                    "user_id": uid,
                    "team_id": team_id,
                },
            )
            for field, value in vals.items():
                setattr(ps, field, value)
            ps.save()
        messages.success(request, "Game updated")
        return redirect("hky_game_detail", game_id=game_id)
    team1_rows = []
    for p in team1_players:
        s = stats_by_pid.get(int(p["id"]), {})
        team1_rows.append({"player": p, "stats": s})
    team2_rows = []
    for p in team2_players:
        s = stats_by_pid.get(int(p["id"]), {})
        team2_rows.append({"player": p, "stats": s})
    return render(
        request,
        "hky_game_detail.html",
        {
            "game": game,
            "team1_rows": team1_rows,
            "team2_rows": team2_rows,
            "editable": editable,
        },
    )


@require_http_methods(["GET", "POST"])
def game_types(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        if name:
            try:
                GameType.objects.create(name=name, is_default=False)
                messages.success(request, "Game type added")
            except Exception:
                messages.error(request, "Failed to add game type (may already exist)")
        return redirect("game_types")
    rows = list(GameType.objects.order_by("name"))
    return render(request, "game_types.html", {"game_types": rows})

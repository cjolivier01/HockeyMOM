#!/usr/bin/env python3
import csv
import datetime as dt
from functools import lru_cache
import io
import json
import os
import re
import secrets
import sys
import traceback
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

from werkzeug.security import generate_password_hash

# Lazy import for pymysql to allow importing module without DB installed (e.g., tests)
try:
    import pymysql  # type: ignore
except Exception:  # pragma: no cover
    pymysql = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
CONFIG_PATH = BASE_DIR / "config.json"

WATCH_ROOT = os.environ.get("HM_WATCH_ROOT", "/data/incoming")

# Allow importing sibling modules (e.g., hockey_rankings.py) when app.py is loaded via file path in tests.
_base_dir_str = str(BASE_DIR)
if _base_dir_str not in sys.path:
    sys.path.insert(0, _base_dir_str)

from hockey_rankings import (  # noqa: E402
    GameScore,
    compute_mhr_like_ratings,
    parse_age_from_division_name,
    parse_level_from_division_name,
    filter_games_ignore_cross_age,
    scale_ratings_to_0_99_9_by_component,
)


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
        from django.db import IntegrityError, transaction
        from django.db.models import F

        now = dt.datetime.now()
        with transaction.atomic():
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


def to_dt(value: Any) -> Optional[dt.datetime]:
    """
    Parse a datetime from a DB value or string.
    Accepts:
      - datetime objects
      - 'YYYY-MM-DD HH:MM:SS'
      - 'YYYY-MM-DDTHH:MM[:SS]'
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return dt.datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _load_or_create_app_secret() -> str:
    env = os.environ.get("HM_WEBAPP_SECRET")
    if env:
        return str(env)

    # Best-effort: allow config.json to pin the secret for multi-worker deployments.
    try:
        cfg_path = os.environ.get("HM_DB_CONFIG", str(CONFIG_PATH))
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        for k in ("app_secret", "secret_key", "webapp_secret"):
            v = cfg.get(k)
            if v:
                return str(v)
    except Exception:
        pass

    # Fall back to a persistent secret under instance/ so all gunicorn workers share it.
    try:
        INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
        secret_path = INSTANCE_DIR / "app_secret.txt"
        if secret_path.exists():
            s = secret_path.read_text(encoding="utf-8").strip()
            if s:
                return s
        s = secrets.token_hex(32)
        secret_path.write_text(s + "\n", encoding="utf-8")
        try:
            os.chmod(secret_path, 0o600)
        except Exception:
            pass
        return s
    except Exception:
        # Last resort: non-persistent secret (may break sessions across workers).
        return secrets.token_hex(16)


APP_SECRET = _load_or_create_app_secret()


def create_app():
    # Lazily import Flask so this module remains importable in pure-Django deployments.
    from flask import (  # type: ignore
        Flask,
        flash,
        g,
        jsonify,
        redirect,
        render_template,
        request,
        send_from_directory,
        session,
        url_for,
    )
    from werkzeug.exceptions import HTTPException
    from werkzeug.security import check_password_hash

    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        instance_path=str(INSTANCE_DIR),
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    app.config.update(
        SECRET_KEY=APP_SECRET,
        MAX_CONTENT_LENGTH=1024 * 1024 * 500,  # 500MB
        UPLOAD_FOLDER=WATCH_ROOT,
    )

    @app.errorhandler(Exception)  # noqa: BLE001
    def _log_unhandled_exception(e: Exception):
        # Ensure unexpected errors always show up in gunicorn/systemd logs.
        if isinstance(e, HTTPException):
            return e
        try:
            print("[webapp] Unhandled exception:", file=sys.stderr)
            traceback.print_exc()
        except Exception:
            pass
        return ("Internal Server Error", 500)

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    @app.template_filter("fmt_toi")
    def _fmt_toi(seconds: Any) -> str:
        return format_seconds_to_mmss_or_hhmmss(seconds)

    def _to_dt(value: Any) -> Optional[dt.datetime]:
        return to_dt(value)

    @app.template_filter("fmt_date")
    def _fmt_date(value: Any) -> str:
        d = _to_dt(value)
        return d.strftime("%Y-%m-%d") if d else ""

    @app.template_filter("fmt_time")
    def _fmt_time(value: Any) -> str:
        d = _to_dt(value)
        return d.strftime("%H:%M") if d else ""

    @app.template_filter("fmt_stat")
    def _fmt_stat(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        try:
            f = float(value)
            if f.is_integer():
                return str(int(f))
            return f"{f:.2f}".rstrip("0").rstrip(".")
        except Exception:
            return str(value)

    @app.template_filter("youtube_best_quality_url")
    def _youtube_best_quality_url(url: Any) -> str:
        """
        Best-effort: append YouTube's `vq` hint (e.g. hd1080) to prefer higher playback quality.
        Note: YouTube ultimately chooses resolution based on bandwidth/device/player size.
        """
        s = str(url or "").strip()
        if not s:
            return ""
        try:
            from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

            u = urlparse(s)
            host = (u.hostname or "").lower()
            is_youtube = (
                ("youtube.com" in host) or ("youtu.be" in host) or ("youtube-nocookie.com" in host)
            )
            if not is_youtube:
                return s
            q = dict(parse_qsl(u.query or "", keep_blank_values=True))
            q.setdefault("vq", "hd1080")
            new_u = u._replace(query=urlencode(q, doseq=True))
            return urlunparse(new_u)
        except Exception:
            return s

    @app.before_request
    def open_db():
        _django_orm, m = _orm_modules()
        _django_orm.close_connections()
        # Legacy compatibility: older helper functions/routes pass `g.db` around; ORM ignores it.
        g.db = None

        # Ensure session league selection is valid or load user's default league
        try:
            if "user_id" not in session:
                return

            from django.db.models import Q

            uid = int(session["user_id"])  # type: ignore[arg-type]

            def _has_access(lid: int) -> bool:
                return (
                    m.League.objects.filter(id=int(lid))
                    .filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
                    .exists()
                )

            sid = session.get("league_id")
            if sid is not None:
                try:
                    lid = int(sid)  # type: ignore[arg-type]
                except Exception:
                    session.pop("league_id", None)
                    return
                if _has_access(lid):
                    return

                session.pop("league_id", None)
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
                session["league_id"] = pref_i
                return
            m.User.objects.filter(id=uid, default_league_id=pref_i).update(default_league=None)
        except Exception:
            # Non-fatal
            return

    @app.teardown_request
    def close_db(exc):  # noqa: ARG001
        _django_orm, _m = _orm_modules()
        _django_orm.close_connections()

    with app.app_context():
        if os.environ.get("HM_WEBAPP_SKIP_DB_INIT") != "1":
            init_db()

    @app.route("/")
    def index():
        if "user_id" in session:
            return redirect(url_for("games"))
        return render_template("index.html")

    @app.context_processor
    def inject_user_leagues():
        leagues = []
        selected = session.get("league_id")
        if "user_id" in session:
            try:
                _django_orm, m = _orm_modules()
                from django.db.models import Q

                uid = int(session["user_id"])  # type: ignore[arg-type]
                admin_ids = set(
                    m.LeagueMember.objects.filter(
                        user_id=uid, role__in=["admin", "owner"]
                    ).values_list("league_id", flat=True)
                )
                leagues = []
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

        def url_with_args(**kwargs: Any) -> str:
            # Merge current query params with overrides, preserving other args.
            params = request.args.to_dict(flat=True)
            for k, v in (kwargs or {}).items():
                if v is None or str(v).strip() == "":
                    params.pop(str(k), None)
                else:
                    params[str(k)] = str(v)
            qs = urlencode(params)
            return request.path + (f"?{qs}" if qs else "")

        return dict(user_leagues=leagues, selected_league_id=selected, url_with_args=url_with_args)

    @app.post("/league/select")
    def league_select():
        r = require_login()
        if r:
            return r
        lid = request.form.get("league_id")
        # Validate membership
        _django_orm, m = _orm_modules()
        from django.db.models import Q

        uid = int(session["user_id"])  # type: ignore[arg-type]
        if lid and lid.isdigit():
            lid_i = int(lid)
            ok = (
                m.League.objects.filter(id=lid_i)
                .filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
                .exists()
            )
            if ok:
                session["league_id"] = lid_i
                m.User.objects.filter(id=uid).update(default_league_id=lid_i)
        else:
            session.pop("league_id", None)
            m.User.objects.filter(id=uid).update(default_league=None)
        return redirect(request.headers.get("Referer") or url_for("index"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            name = request.form.get("name", "").strip()
            if not email or not password:
                flash("Email and password are required", "error")
            elif get_user_by_email(email):
                flash("Email already registered", "error")
            else:
                uid = create_user(email, password, name)
                session["user_id"] = uid
                session["user_email"] = email
                session["user_name"] = name
                # Send confirmation email (best-effort)
                try:
                    send_email(
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
                return redirect(url_for("games"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            u = get_user_by_email(email)
            if not u or not check_password_hash(u["password_hash"], password):
                flash("Invalid credentials", "error")
            else:
                session["user_id"] = u["id"]
                session["user_email"] = u["email"]
                session["user_name"] = u["name"] or u["email"]
                return redirect(url_for("games"))
        return render_template("login.html")

    @app.route("/forgot", methods=["GET", "POST"])
    def forgot():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            # Always say we sent an email to avoid user enumeration
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
                    # Compose link
                    base = request.url_root.rstrip("/")
                    link = f"{base}/reset/{token}"
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
            flash("If the account exists, a reset email has been sent.", "success")
            return redirect(url_for("login"))
        return render_template("forgot_password.html")

    @app.route("/reset/<token>", methods=["GET", "POST"])
    def reset(token: str):
        _django_orm, m = _orm_modules()
        row = (
            m.Reset.objects.select_related("user")
            .filter(token=str(token))
            .values("id", "user_id", "token", "expires_at", "used_at", "user__email")
            .first()
        )
        if not row:
            flash("Invalid or expired token", "error")
            return redirect(url_for("login"))
        # Check expiry and used
        now = dt.datetime.now()
        expires = (
            row["expires_at"]
            if isinstance(row["expires_at"], dt.datetime)
            else dt.datetime.fromisoformat(str(row["expires_at"]))
        )
        if row.get("used_at") or now > expires:
            flash("Invalid or expired token", "error")
            return redirect(url_for("login"))
        if request.method == "POST":
            pw1 = request.form.get("password", "")
            pw2 = request.form.get("password2", "")
            if not pw1 or pw1 != pw2:
                flash("Passwords do not match", "error")
                return render_template("reset_password.html")
            # Update password and mark token used
            newhash = generate_password_hash(pw1)
            from django.db import transaction

            now2 = dt.datetime.now()
            with transaction.atomic():
                m.User.objects.filter(id=int(row["user_id"])).update(password_hash=newhash)
                m.Reset.objects.filter(id=int(row["id"])).update(used_at=now2)
            flash("Password updated. Please log in.", "success")
            return redirect(url_for("login"))
        return render_template("reset_password.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    def require_login():
        if "user_id" not in session:
            return redirect(url_for("login"))
        return None

    @app.post("/api/user/video_clip_len")
    def api_user_video_clip_len():
        # Session-authenticated endpoint (public pages may call this if the user is logged in).
        if "user_id" not in session:
            return jsonify({"ok": False, "error": "login_required"}), 401
        payload = request.get_json(silent=True) or {}
        raw = payload.get("clip_len_s")
        try:
            v = int(raw)
        except Exception:
            return (
                jsonify(
                    {"ok": False, "error": "clip_len_s must be one of: 15, 20, 30, 45, 60, 90"}
                ),
                400,
            )
        if v not in {15, 20, 30, 45, 60, 90}:
            return (
                jsonify(
                    {"ok": False, "error": "clip_len_s must be one of: 15, 20, 30, 45, 60, 90"}
                ),
                400,
            )
        try:
            _django_orm, m = _orm_modules()
            m.User.objects.filter(id=int(session["user_id"])).update(video_clip_len_s=int(v))  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "clip_len_s": int(v)})

    @app.get("/api/leagues/<int:league_id>/page_views")
    def api_league_page_views(league_id: int):
        r = require_login()
        if r:
            return r
        user_id = int(session.get("user_id") or 0)
        if not user_id:
            return jsonify({"ok": False, "error": "login_required"}), 401
        _django_orm, m = _orm_modules()
        owner_id_i = (
            m.League.objects.filter(id=int(league_id))
            .values_list("owner_user_id", flat=True)
            .first()
        )
        owner_id_i = int(owner_id_i) if owner_id_i is not None else None
        if owner_id_i is None:
            return jsonify({"ok": False, "error": "not_found"}), 404
        if int(owner_id_i) != int(user_id):
            return jsonify({"ok": False, "error": "not_authorized"}), 403

        kind = str(request.args.get("kind") or "").strip()
        entity_id_raw = request.args.get("entity_id")
        try:
            entity_id = int(str(entity_id_raw or "0").strip() or "0")
        except Exception:
            entity_id = 0
        try:
            count = _get_league_page_view_count(
                None, int(league_id), kind=kind, entity_id=entity_id
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "league_id": int(league_id),
                "kind": kind,
                "entity_id": int(entity_id),
                "count": int(count),
            }
        )

    @app.route("/games")
    def games():
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        rows = list(m.Game.objects.filter(user_id=uid).order_by("-created_at").values())
        # Read dirwatcher state if present
        dw_state = read_dirwatch_state()
        return render_template("games.html", games=rows, state=dw_state)

    @app.route("/games/new", methods=["GET", "POST"])
    def new_game():
        r = require_login()
        if r:
            return r
        if request.method == "POST":
            name = request.form.get("name", "").strip() or f"game-{dt.datetime.now():%Y%m%d-%H%M%S}"
            gid, dir_path = create_game(session["user_id"], name, session.get("user_email") or "")
            flash("Game created", "success")
            return redirect(url_for("game_detail", gid=gid))
        return render_template("new_game.html")

    @app.route("/games/<int:gid>")
    def game_detail(gid: int):
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))
        files = []
        try:
            all_files = os.listdir(game["dir_path"]) if os.path.isdir(game["dir_path"]) else []

            def _is_user_file(fname: str) -> bool:
                # Hide control/meta files: dotfiles, sentinels, slurm outputs
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
        # Determine latest status from DB if present, else dirwatcher state, else game.status
        latest_status = None
        latest_job = m.Job.objects.filter(game_id=int(gid)).order_by("-id").values("status").first()
        if latest_job and latest_job.get("status") is not None:
            latest_status = str(latest_job["status"])
        if not latest_status:
            dw_state = read_dirwatch_state()
            latest_status = dw_state.get("processed", {}).get(game["dir_path"], {}).get(
                "status"
            ) or game.get("status")
        # Lock interactions once a job has been requested (any job row exists) or after completion
        is_locked = False
        if latest_job:
            is_locked = True
        final_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
        if latest_status and str(latest_status).upper() in final_states:
            is_locked = True
        return render_template(
            "game_detail.html", game=game, files=files, status=latest_status, is_locked=is_locked
        )

    @app.route("/games/<int:gid>/delete", methods=["GET", "POST"])
    def delete_game(gid: int):
        r = require_login()
        if r:
            return r
        # Load game
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))

        # Check latest job state for potential cancellation on delete
        latest = (
            m.Job.objects.filter(game_id=int(gid))
            .order_by("-id")
            .values("id", "slurm_job_id", "status")
            .first()
        )

        if request.method == "POST":
            token = request.form.get("confirm", "").strip().upper()
            if token != "DELETE":
                flash("Type DELETE to confirm permanent deletion.", "error")
                return render_template("confirm_delete.html", game=game)
            # If job is active, attempt scancel first (best-effort) and wait briefly
            try:
                active_states = ("SUBMITTED", "RUNNING", "PENDING")
                if latest and str(latest.get("status", "")).upper() in active_states:
                    import subprocess as _sp
                    import time as _time

                    dir_leaf = Path(game["dir_path"]).name
                    job_name = f"dirwatch-{dir_leaf}"
                    job_ids = []
                    jid = latest.get("slurm_job_id")
                    if jid:
                        job_ids.append(str(jid))
                    else:
                        # Fallback: discover job ids by job name prefix
                        try:
                            out = _sp.check_output(["squeue", "-h", "-o", "%i %j"]).decode()
                            for line in out.splitlines():
                                parts = line.strip().split(maxsplit=1)
                                if len(parts) == 2 and parts[1] == job_name:
                                    job_ids.append(parts[0])
                        except Exception:
                            pass
                    # Issue scancel and wait until jobs disappear from squeue or timeout
                    for jid_ in job_ids:
                        _sp.run(["scancel", str(jid_)], check=False)
                    deadline = _time.time() + 20.0
                    while _time.time() < deadline:
                        try:
                            out = _sp.check_output(["squeue", "-h", "-o", "%i %j"]).decode()
                            still = False
                            for line in out.splitlines():
                                parts = line.strip().split(maxsplit=1)
                                if not parts:
                                    continue
                                if parts[0] in job_ids or (
                                    len(parts) == 2 and parts[1] == job_name
                                ):
                                    still = True
                                    break
                            if not still:
                                break
                        except Exception:
                            break
                        _time.sleep(1.0)
            except Exception:
                pass

            # Delete from DB (jobs first), then remove directory
            from django.db import transaction

            with transaction.atomic():
                m.Job.objects.filter(game_id=int(gid)).delete()
                m.Game.objects.filter(id=int(gid), user_id=uid).delete()
            # Remove directory if under our watch root
            try:
                d = Path(game["dir_path"]).resolve()
                wr = Path(WATCH_ROOT).resolve()
                try:
                    # Python 3.9-compatible relative check
                    if str(d).startswith(str(wr)):
                        import shutil

                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
            except Exception:
                pass
            flash("Game deleted.", "success")
            return redirect(url_for("games"))

        return render_template("confirm_delete.html", game=game)

    @app.route("/games/<int:gid>/upload", methods=["POST"])
    def upload(gid: int):
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))
        # Block uploads if a job has been requested or finished
        if m.Job.objects.filter(game_id=int(gid)).exists():
            flash("Job already submitted; uploads disabled.", "error")
            return redirect(url_for("game_detail", gid=gid))
        # Save uploaded files
        updir = Path(game["dir_path"])
        updir.mkdir(parents=True, exist_ok=True)
        files = request.files.getlist("files")
        count = 0
        for f in files:
            if not f or not f.filename:
                continue
            dest = updir / Path(f.filename).name
            f.save(dest)
            count += 1
        flash(f"Uploaded {count} files", "success")
        return redirect(url_for("game_detail", gid=gid))

    @app.route("/games/<int:gid>/run", methods=["POST"])
    def run_game(gid: int):
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))
        # Prevent duplicate submissions
        if m.Job.objects.filter(game_id=int(gid)).exists():
            flash("Job already submitted.", "error")
            return redirect(url_for("game_detail", gid=gid))
        dir_path = Path(game["dir_path"])
        try:
            meta = dir_path / ".dirwatch_meta.json"
            meta.write_text(
                f'{{"user_email":"{session.get("user_email", "")}","game_id":{gid},"created":"{dt.datetime.now().isoformat()}"}}\n'
            )
        except Exception:
            pass
        # Create the READY file
        (dir_path / "_READY").touch(exist_ok=True)
        from django.db import transaction

        with transaction.atomic():
            m.Game.objects.filter(id=int(gid), user_id=uid).update(status="submitted")
            m.Job.objects.create(
                user_id=uid,
                game_id=int(gid),
                dir_path=str(dir_path),
                status="PENDING",
                created_at=dt.datetime.now(),
                user_email=str(session.get("user_email") or "") or None,
            )
        flash("Run requested. Job will start shortly.", "success")
        return redirect(url_for("game_detail", gid=gid))

    @app.route("/uploads/<int:gid>/<path:name>")
    def serve_upload(gid: int, name: str):
        # Optional: serve uploaded files back to user if needed
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        game = m.Game.objects.filter(id=int(gid), user_id=uid).values("dir_path").first()
        if not game:
            return ("Not found", 404)
        d = Path(game["dir_path"]).resolve()
        return send_from_directory(str(d), name, as_attachment=True)

    @app.route("/jobs")
    def jobs():
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
        jobs = list(m.Job.objects.filter(user_id=uid).order_by("-created_at").values())
        return render_template("jobs.html", jobs=jobs)

    # ---------------------------
    # League: Teams / Players / Games (Hockey)
    # ---------------------------

    @app.route("/media/team_logo/<int:team_id>")
    def media_team_logo(team_id: int):
        r = require_login()
        if r:
            return r
        league_id = session.get("league_id")
        _django_orm, m = _orm_modules()
        uid = int(session["user_id"])  # type: ignore[arg-type]
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
            return ("Not found", 404)
        p = Path(row["logo_path"]).resolve()
        if not p.exists():
            return ("Not found", 404)
        return send_from_directory(str(p.parent), p.name)

    @app.route("/teams")
    def teams():
        r = require_login()
        if r:
            return r
        include_external = request.args.get("all", "0") == "1"
        league_id = session.get("league_id")
        league_owner_user_id: Optional[int] = None
        is_league_owner = False
        if league_id:
            league_owner_user_id = _get_league_owner_user_id(None, int(league_id))
            is_league_owner = bool(
                league_owner_user_id is not None
                and int(league_owner_user_id) == int(session["user_id"])
            )
            _record_league_page_view(
                None,
                int(league_id),
                kind=LEAGUE_PAGE_VIEW_KIND_TEAMS,
                entity_id=0,
                viewer_user_id=int(session["user_id"]),
                league_owner_user_id=league_owner_user_id,
            )
        is_league_admin = False
        if league_id:
            try:
                is_league_admin = bool(_is_league_admin(int(league_id), int(session["user_id"])))
            except Exception:
                is_league_admin = False
        _django_orm, m = _orm_modules()

        uid = int(session["user_id"])  # type: ignore[arg-type]
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
            for r in rows_raw:
                rows.append(
                    {
                        "id": int(r["team_id"]),
                        "user_id": int(r["team__user_id"]),
                        "name": r.get("team__name"),
                        "logo_path": r.get("team__logo_path"),
                        "is_external": bool(r.get("team__is_external")),
                        "created_at": r.get("team__created_at"),
                        "updated_at": r.get("team__updated_at"),
                        "division_name": r.get("division_name"),
                        "division_id": r.get("division_id"),
                        "conference_id": r.get("conference_id"),
                        "mhr_rating": r.get("mhr_rating"),
                        "mhr_agd": r.get("mhr_agd"),
                        "mhr_sched": r.get("mhr_sched"),
                        "mhr_games": r.get("mhr_games"),
                        "mhr_updated_at": r.get("mhr_updated_at"),
                    }
                )
        else:
            qs = m.Team.objects.filter(user_id=uid)
            if not include_external:
                qs = qs.filter(is_external=False)
            rows = list(qs.order_by("name").values())
        # compute stats per team (wins/losses/ties/gf/ga/points)
        stats = {}
        for t in rows:
            if league_id:
                stats[t["id"]] = compute_team_stats_league(None, t["id"], int(league_id))
            else:
                stats[t["id"]] = compute_team_stats(None, t["id"], int(session["user_id"]))
        divisions = None
        if league_id:
            grouped: dict[str, list[dict]] = {}
            for t in rows:
                dn = str(t.get("division_name") or "").strip() or "Unknown Division"
                grouped.setdefault(dn, []).append(t)
            divisions = []
            for dn in sorted(grouped.keys(), key=division_sort_key):
                teams_sorted = sorted(
                    grouped[dn], key=lambda tr: sort_key_team_standings(tr, stats.get(tr["id"], {}))
                )
                divisions.append({"name": dn, "teams": teams_sorted})
        league_page_views = None
        if league_id and is_league_owner:
            league_page_views = {
                "league_id": int(league_id),
                "kind": LEAGUE_PAGE_VIEW_KIND_TEAMS,
                "entity_id": 0,
                "count": _get_league_page_view_count(
                    None, int(league_id), kind=LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
                ),
            }
        return render_template(
            "teams.html",
            teams=rows,
            divisions=divisions,
            stats=stats,
            include_external=include_external,
            league_view=bool(league_id),
            current_user_id=int(session["user_id"]),
            is_league_admin=is_league_admin,
            league_page_views=league_page_views,
        )

    @app.post("/leagues/recalc_div_ratings")
    def leagues_recalc_div_ratings():
        r = require_login()
        if r:
            return r
        league_id = session.get("league_id")
        if not league_id:
            flash("Select an active league first.", "error")
            return redirect(url_for("leagues_index"))
        if not _is_league_admin(int(league_id), int(session["user_id"])):
            flash("Not authorized", "error")
            return redirect(url_for("leagues_index"))
        try:
            recompute_league_mhr_ratings(None, int(league_id))
            flash("Ratings recalculated.", "success")
        except Exception as e:  # noqa: BLE001
            flash(f"Failed to recalculate Ratings: {e}", "error")
        return redirect(url_for("leagues_index"))

    @app.get("/leagues")
    def leagues_index():
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        from django.db.models import Q

        uid = int(session["user_id"])  # type: ignore[arg-type]
        admin_ids = set(
            m.LeagueMember.objects.filter(user_id=uid, role__in=["admin", "owner"]).values_list(
                "league_id", flat=True
            )
        )
        leagues: list[dict[str, Any]] = []
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
                    "is_owner": is_owner,
                    "is_admin": is_admin,
                }
            )
        return render_template("leagues.html", leagues=leagues)

    def _get_import_token() -> Optional[str]:
        token = os.environ.get("HM_WEBAPP_IMPORT_TOKEN")
        if token:
            return token
        try:
            cfg_path = os.environ.get("HM_DB_CONFIG", str(CONFIG_PATH))
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            t = cfg.get("import_token")
            return str(t) if t else None
        except Exception:
            return None

    def _require_import_auth():
        required = _get_import_token()
        if required:
            supplied = None
            auth = request.headers.get("Authorization", "").strip()
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
            if not supplied:
                supplied = request.headers.get("X-HM-Import-Token") or request.args.get("token")
            supplied = (supplied or "").strip()
            required = (required or "").strip()
            if not supplied or not secrets.compare_digest(supplied, required):
                return jsonify({"ok": False, "error": "unauthorized"}), 401
            return None

        if request.headers.get("X-Forwarded-For"):
            return jsonify({"ok": False, "error": "import_token_required"}), 403
        if request.remote_addr not in ("127.0.0.1", "::1"):
            return jsonify({"ok": False, "error": "import_token_required"}), 403
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

    def _ensure_external_team_for_import(
        owner_user_id: int, name: str, *, commit: bool = True
    ) -> int:
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
            # Case/punctuation-insensitive matching to avoid duplicate teams.
            t = t.casefold()
            t = re.sub(r"[^0-9a-z]+", " ", t)
            return " ".join(t.split())

        nm = _norm_team_name(name or "")
        if not nm:
            nm = "UNKNOWN"
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
        return existing

    def _extract_game_video_url_from_notes(notes: Optional[str]) -> Optional[str]:
        s = str(notes or "").strip()
        if not s:
            return None
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                for k in ("game_video_url", "game_video", "video_url"):
                    v = d.get(k)
                    if v is not None and str(v).strip():
                        return str(v).strip()
        except Exception:
            pass
        m = re.search(r"(?:^|[\\s|,;])game_video_url\\s*=\\s*([^\\s|,;]+)", s, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        m = re.search(r"(?:^|[\\s|,;])game_video\\s*=\\s*([^\\s|,;]+)", s, flags=re.IGNORECASE)
        if m:
            return str(m.group(1)).strip()
        return None

    def _extract_timetoscore_game_id_from_notes(notes: Optional[str]) -> Optional[int]:
        s = str(notes or "").strip()
        if not s:
            return None
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                v = d.get("timetoscore_game_id")
                if v is not None:
                    try:
                        return int(v)
                    except Exception:
                        return None
        except Exception:
            pass
        # Backward-compatible plain-text token used by older importers.
        m = re.search(r"(?:^|[\\s|,;])game_id\\s*=\\s*(\\d+)", s, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        m = re.search(r"\"timetoscore_game_id\"\\s*:\\s*(\\d+)", s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _update_game_video_url_note(
        game_id: int, video_url: str, *, replace: bool, commit: bool = True
    ) -> None:
        url = _sanitize_http_url(video_url)
        if not url:
            return
        _django_orm, m = _orm_modules()
        existing = str(
            m.HkyGame.objects.filter(id=int(game_id)).values_list("notes", flat=True).first() or ""
        ).strip()
        existing_url = _extract_game_video_url_from_notes(existing)
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
            # Preserve non-JSON notes (used by older importers for token matching).
            suffix = f" game_video_url={url}"
            if existing and suffix.strip() in existing:
                new_notes = existing
            else:
                new_notes = (
                    (existing + "\n" + suffix.strip()).strip() if existing else suffix.strip()
                )
        m.HkyGame.objects.filter(id=int(game_id)).update(
            notes=new_notes, updated_at=dt.datetime.now()
        )

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
        _django_orm, m = _orm_modules()
        from django.db.models import Q

        starts_dt = to_dt(starts_at) if starts_at else None

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

    def _map_team_to_league_for_import(
        league_id: int,
        team_id: int,
        *,
        division_name: Optional[str] = None,
        division_id: Optional[int] = None,
        conference_id: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        dn = (division_name or "").strip() or None
        _django_orm, m = _orm_modules()
        obj, created = m.LeagueTeam.objects.get_or_create(
            league_id=int(league_id),
            team_id=int(team_id),
            defaults={
                "division_name": dn,
                "division_id": division_id,
                "conference_id": conference_id,
            },
        )
        if created:
            return

        updates: dict[str, Any] = {}
        allow_div_update = True
        if dn and is_external_division_name(dn):
            existing_dn = str(getattr(obj, "division_name", "") or "").strip()
            if existing_dn and not is_external_division_name(existing_dn):
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
        if dn and is_external_division_name(dn):
            existing_dn = str(getattr(obj, "division_name", "") or "").strip()
            if existing_dn and not is_external_division_name(existing_dn):
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

    def _ensure_team_logo_from_url_for_import(
        *,
        team_id: int,
        logo_url: Optional[str],
        replace: bool,
        commit: bool = True,
    ) -> None:
        url = str(logo_url or "").strip()
        if not url:
            return
        _django_orm, m = _orm_modules()
        existing = (
            m.Team.objects.filter(id=int(team_id)).values_list("logo_path", flat=True).first()
        )
        if existing and not replace:
            return

        try:
            import requests
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
            resp = requests.get(url, timeout=(10, 30), headers=headers)
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
            logo_dir = INSTANCE_DIR / "uploads" / "team_logos"
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
        # Respect "replace": if a logo already exists, don't overwrite unless requested.
        _django_orm, m = _orm_modules()
        existing = (
            m.Team.objects.filter(id=int(team_id)).values_list("logo_path", flat=True).first()
        )
        if existing and not replace:
            return

        b64_s = str(logo_b64 or "").strip()
        if b64_s:
            try:
                import base64

                data = base64.b64decode(b64_s.encode("ascii"), validate=False)
            except Exception:
                data = b""

            # Basic size guardrail.
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
                # Fall back to guessing from URL, then default to png.
                url = str(logo_url or "").strip()
                for cand in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
                    if url.lower().split("?", 1)[0].endswith(cand):
                        ext = cand
                        break
            if ext is None:
                ext = ".png"

            try:
                logo_dir = INSTANCE_DIR / "uploads" / "team_logos"
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

        # Fallback: fetch from URL if the server has `requests` installed.
        _ensure_team_logo_from_url_for_import(
            team_id=int(team_id),
            logo_url=logo_url,
            replace=replace,
            commit=commit,
        )

    @app.post("/api/import/hockey/ensure_league")
    def api_import_ensure_league():
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
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
        return jsonify({"ok": True, "league_id": league_id, "owner_user_id": owner_user_id})

    @app.post("/api/import/hockey/teams")
    def api_import_teams():
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
        league_name = str(payload.get("league_name") or "CAHA")
        shared = bool(payload["shared"]) if "shared" in payload else None
        replace = bool(payload.get("replace", False))
        owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
        owner_name = str(payload.get("owner_name") or "CAHA Import")
        owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)

        teams = payload.get("teams") or []
        if not isinstance(teams, list) or not teams:
            return jsonify({"ok": False, "error": "teams must be a non-empty list"}), 400

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

            _django_orm, _m = _orm_modules()

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
            return jsonify({"ok": False, "error": str(e)}), 400

        return jsonify(
            {
                "ok": True,
                "league_id": league_id,
                "owner_user_id": owner_user_id,
                "imported": len(results),
                "results": results,
            }
        )

    @app.post("/api/import/hockey/game")
    def api_import_game():
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
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
            return jsonify({"ok": False, "error": "home_name and away_name are required"}), 400

        division_name = str(game.get("division_name") or "").strip() or None
        home_division_name = (
            str(game.get("home_division_name") or division_name or "").strip() or None
        )
        away_division_name = (
            str(game.get("away_division_name") or division_name or "").strip() or None
        )
        try:
            division_id = (
                int(game.get("division_id")) if game.get("division_id") is not None else None
            )
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
            logo_content_type=game.get("home_logo_content_type")
            or game.get("team1_logo_content_type"),
            logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
            replace=replace,
        )
        _ensure_team_logo_for_import(
            team_id=int(team2_id),
            logo_b64=game.get("away_logo_b64") or game.get("team2_logo_b64"),
            logo_content_type=game.get("away_logo_content_type")
            or game.get("team2_logo_content_type"),
            logo_url=game.get("away_logo_url") or game.get("team2_logo_url"),
            replace=replace,
        )

        starts_at = game.get("starts_at")
        starts_at_s = str(starts_at) if starts_at else None
        location = str(game.get("location")).strip() if game.get("location") else None
        team1_score = game.get("home_score")
        team2_score = game.get("away_score")
        tts_game_id = game.get("timetoscore_game_id")
        ext_game_key = str(game.get("external_game_key") or "").strip() or None

        notes_fields: dict[str, Any] = {}
        if tts_game_id is not None:
            try:
                notes_fields["timetoscore_game_id"] = int(tts_game_id)
            except Exception:
                pass
        if ext_game_key:
            notes_fields["external_game_key"] = str(ext_game_key)
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

        # Optional schedule metadata (used by CAHA schedule.pl imports).
        if game.get("caha_schedule_year") is not None:
            try:
                notes_fields["caha_schedule_year"] = int(game.get("caha_schedule_year"))
            except Exception:
                pass
        if game.get("caha_schedule_group") is not None:
            notes_fields["caha_schedule_group"] = str(game.get("caha_schedule_group"))
        if game.get("caha_schedule_game_number") is not None:
            try:
                notes_fields["caha_schedule_game_number"] = int(
                    game.get("caha_schedule_game_number")
                )
            except Exception:
                pass

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
                    pid = _ensure_player_for_import(owner_user_id, tid, nm, jersey, pos)
                    try:
                        roster_player_ids_by_team[int(tid)].add(int(pid))
                    except Exception:
                        pass

        def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
            _django_orm, m = _orm_modules()
            pid = (
                m.Player.objects.filter(
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
        _django_orm, m = _orm_modules()
        from django.db import transaction

        with transaction.atomic():
            if played:
                # Credit GP for roster players even when they have no scoring stats in TimeToScore.
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

                    # If this game is linked to TimeToScore, TimeToScore is the source of truth for
                    # goal/assist attribution. Always overwrite goals/assists from the TimeToScore import.
                    force_tts_scoring = bool(tts_game_id is not None)
                    ps, _created = m.PlayerStat.objects.get_or_create(
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
                        m.PlayerStat.objects.filter(id=ps.id).update(goals=gval, assists=aval)
                    else:
                        m.PlayerStat.objects.filter(id=ps.id, goals__isnull=True).update(goals=gval)
                        m.PlayerStat.objects.filter(id=ps.id, assists__isnull=True).update(
                            assists=aval
                        )

        return jsonify(
            {
                "ok": True,
                "league_id": league_id,
                "owner_user_id": owner_user_id,
                "team1_id": team1_id,
                "team2_id": team2_id,
                "game_id": gid,
            }
        )

    @app.post("/api/import/hockey/games_batch")
    def api_import_games_batch():
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
        league_name = str(payload.get("league_name") or "CAHA")
        shared = bool(payload["shared"]) if "shared" in payload else None
        replace = bool(payload.get("replace", False))
        owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
        owner_name = str(payload.get("owner_name") or "CAHA Import")
        owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)

        games = payload.get("games") or []
        if not isinstance(games, list) or not games:
            return jsonify({"ok": False, "error": "games must be a non-empty list"}), 400

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
                    did = (
                        int(row.get("division_id")) if row.get("division_id") is not None else None
                    )
                except Exception:
                    did = None
                try:
                    cid = (
                        int(row.get("conference_id"))
                        if row.get("conference_id") is not None
                        else None
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
                        int(game.get("division_id"))
                        if game.get("division_id") is not None
                        else None
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
                ext_game_key = str(game.get("external_game_key") or "").strip() or None

                notes_fields: dict[str, Any] = {}
                if tts_game_id is not None:
                    try:
                        notes_fields["timetoscore_game_id"] = int(tts_game_id)
                    except Exception:
                        pass
                if ext_game_key:
                    notes_fields["external_game_key"] = str(ext_game_key)
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

                # Optional schedule metadata (used by CAHA schedule.pl imports).
                if game.get("caha_schedule_year") is not None:
                    try:
                        notes_fields["caha_schedule_year"] = int(game.get("caha_schedule_year"))
                    except Exception:
                        pass
                if game.get("caha_schedule_group") is not None:
                    notes_fields["caha_schedule_group"] = str(game.get("caha_schedule_group"))
                if game.get("caha_schedule_game_number") is not None:
                    try:
                        notes_fields["caha_schedule_game_number"] = int(
                            game.get("caha_schedule_game_number")
                        )
                    except Exception:
                        pass

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
                    # If importer sends "External" (or no division), keep the game in the team's real division
                    # when the team already exists in the league.
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
                            pim_val = (
                                int(pim) if pim is not None and str(pim).strip() != "" else None
                            )
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

                        # If this game is linked to TimeToScore, TimeToScore is the source of truth for
                        # goal/assist attribution. Always overwrite goals/assists from the TimeToScore import.
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
                            m.PlayerStat.objects.filter(id=ps.id, goals__isnull=True).update(
                                goals=gval
                            )
                            m.PlayerStat.objects.filter(id=ps.id, assists__isnull=True).update(
                                assists=aval
                            )
                            if pim_val is not None:
                                m.PlayerStat.objects.filter(id=ps.id, pim__isnull=True).update(
                                    pim=pim_val
                                )

                if isinstance(events_csv, str) and events_csv.strip():
                    try:
                        from tools.webapp.django_app.views import (
                            _upsert_game_event_rows_from_events_csv as _upsert_event_rows,
                        )
                    except Exception:  # pragma: no cover
                        from django_app.views import (  # type: ignore
                            _upsert_game_event_rows_from_events_csv as _upsert_event_rows,
                        )

                    _upsert_event_rows(
                        game_id=int(gid),
                        events_csv=str(events_csv),
                        replace=bool(game_replace),
                        create_missing_players=False,
                        incoming_source_label="timetoscore",
                        prefer_incoming_for_event_types={
                            "goal",
                            "assist",
                            "penalty",
                            "penaltyexpired",
                            "goaliechange",
                        },
                    )

                results.append({"game_id": gid, "team1_id": team1_id, "team2_id": team2_id})

        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        return jsonify(
            {
                "ok": True,
                "league_id": league_id,
                "owner_user_id": owner_user_id,
                "imported": len(results),
                "results": results,
            }
        )

    @app.post("/api/import/hockey/shift_package")
    def api_import_shift_package():
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
        replace = bool(payload.get("replace", False))

        game_id = payload.get("game_id")
        tts_game_id = payload.get("timetoscore_game_id")
        external_game_key = str(payload.get("external_game_key") or "").strip() or None
        team_side = str(payload.get("team_side") or "").strip().lower() or None
        if team_side not in {None, "", "home", "away"}:
            return jsonify({"ok": False, "error": "team_side must be 'home' or 'away'"}), 400
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

        # External game flow: allow creating / matching games not in TimeToScore.
        if resolved_game_id is None and external_game_key and owner_email:
            owner_user_id_for_create = _ensure_user_for_import(owner_email)

            # Reuse existing league teams when names match (e.g., teams already imported from TimeToScore),
            # to avoid creating duplicates and to preserve their division mappings.
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
                # Many TimeToScore imports use a disambiguating "(Division)" suffix.
                # Strip a trailing parenthetical block for matching purposes so uploads that omit it still match.
                t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
                t = t.casefold()
                t = re.sub(r"[^0-9a-z]+", " ", t)
                return " ".join(t.split())

            def _find_team_in_league_by_name(
                league_id_i: int, name: str
            ) -> Optional[dict[str, Any]]:
                nm = _norm_team_name_for_match(name)
                if not nm:
                    return None
                rows = list(
                    m.LeagueTeam.objects.filter(league_id=int(league_id_i))
                    .select_related("team")
                    .values(
                        "team_id", "team__name", "division_name", "division_id", "conference_id"
                    )
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
                # Disambiguate by requested division if provided (and not External), otherwise by any existing match's division.
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
                tokens = [
                    f'"external_game_key":{ext_json}',
                    f'"external_game_key": {ext_json}',
                ]
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
                    return (
                        jsonify(
                            {
                                "ok": False,
                                "error": "home_team_name and away_team_name are required to create an external game",
                            }
                        ),
                        400,
                    )

                league_id_i: Optional[int] = None
                try:
                    league_id_i = int(league_id_payload) if league_id_payload is not None else None
                except Exception:
                    league_id_i = None
                if league_id_i is None:
                    if not league_name:
                        return (
                            jsonify(
                                {
                                    "ok": False,
                                    "error": "league_id or league_name is required to create an external game",
                                }
                            ),
                            400,
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

                # External games created from shift spreadsheets should be mapped to the provided division
                # (default: "External"), even when one side matches an existing league team.
                game_division_name = str(division_name or "").strip() or "External"
                new_team_division_name = game_division_name

                # Optional team icons for external games (do not overwrite unless replace).
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

                # Infer score from game_stats.csv if provided; Goals For/Against refer to the uploaded side.
                team1_score = None
                team2_score = None
                try:
                    parsed_gs = parse_shift_stats_game_stats_csv(
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
                        int(league_id_i),
                        team1_id,
                        division_name=new_team_division_name,
                        commit=False,
                    )
                if not match_away:
                    _map_team_to_league_for_import(
                        int(league_id_i),
                        team2_id,
                        division_name=new_team_division_name,
                        commit=False,
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
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "game_id, timetoscore_game_id, or external_game_key+owner_email+league_name+home_team_name+away_team_name is required",
                    }
                ),
                400,
            )

        # Load game + teams
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
            return jsonify({"ok": False, "error": "game not found"}), 404

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
                return jsonify({"ok": False, "error": "game not found after merge"}), 404

        team1_id = int(game_row["team1_id"])
        team2_id = int(game_row["team2_id"])
        owner_user_id = int(game_row.get("user_id") or 0)
        tts_linked = bool(
            tts_int is not None
            or game_row.get("timetoscore_game_id") is not None
            or _extract_timetoscore_game_id_from_notes(game_row.get("notes")) is not None
        )

        player_stats_csv = payload.get("player_stats_csv")
        game_stats_csv = payload.get("game_stats_csv")
        events_csv = payload.get("events_csv")
        game_video_url = (
            payload.get("game_video_url") or payload.get("game_video") or payload.get("video_url")
        )

        if isinstance(game_video_url, str) and game_video_url.strip():
            try:
                _update_game_video_url_note(
                    int(resolved_game_id), str(game_video_url), replace=replace, commit=False
                )
            except Exception:
                # Best-effort: failures updating the optional game video URL must not block the import.
                pass

        if isinstance(events_csv, str) and events_csv.strip():
            # Never store PP/PK span events; they are derived at render time from penalty windows.
            drop_types = {"power play", "powerplay", "penalty kill", "penaltykill"}
            try:
                events_csv = filter_events_csv_drop_event_types(
                    str(events_csv), drop_types=drop_types
                )
            except Exception:
                pass
            try:
                _h, _r = parse_events_csv(str(events_csv))
                if not _r:
                    events_csv = None
            except Exception:
                pass

        # Optional league mapping / ordering updates for existing games.
        if league_id_payload is not None or league_name:
            league_id_i: Optional[int] = None
            try:
                league_id_i = int(league_id_payload) if league_id_payload is not None else None
            except Exception:
                league_id_i = None
            if league_id_i is None and league_name:
                existing_lid = (
                    m.League.objects.filter(name=str(league_name))
                    .values_list("id", flat=True)
                    .first()
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
                p = {
                    "id": int(pid),
                    "team_id": int(tid),
                    "name": name,
                    "jersey_number": jersey_number,
                }
                players_by_team.setdefault(int(tid), []).append(p)
                j = normalize_jersey_number(jersey_number)
                if j:
                    jersey_to_player_ids.setdefault((int(tid), j), []).append(int(pid))
                nm = normalize_player_name(name or "")
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
                    has_existing = m.HkyGameEventRow.objects.filter(
                        game_id=int(resolved_game_id)
                    ).exists()
                    has_timetoscore = m.HkyGameEventRow.objects.filter(
                        game_id=int(resolved_game_id), source__icontains="timetoscore"
                    ).exists()

                    # Match legacy behavior: without replace, ignore follow-up shift-package uploads unless
                    # we have TimeToScore events to merge/augment.
                    should_upsert = True
                    replace_events = bool(replace)
                    if not replace_events and has_existing and not has_timetoscore:
                        should_upsert = False
                    # Avoid wiping authoritative TimeToScore events.
                    if replace_events and has_timetoscore:
                        replace_events = False

                    if should_upsert:
                        try:
                            from tools.webapp.django_app.views import (
                                _upsert_game_event_rows_from_events_csv as _upsert_event_rows,
                            )
                        except Exception:  # pragma: no cover
                            from django_app.views import (  # type: ignore
                                _upsert_game_event_rows_from_events_csv as _upsert_event_rows,
                            )

                        _upsert_event_rows(
                            game_id=int(resolved_game_id),
                            events_csv=str(events_csv),
                            replace=bool(replace_events),
                            create_missing_players=bool(create_missing_players),
                            incoming_source_label="shift_package",
                        )

                if isinstance(game_stats_csv, str) and game_stats_csv.strip():
                    try:
                        parsed_gs = parse_shift_stats_game_stats_csv(game_stats_csv)
                        gf = parsed_gs.get("Goals For")
                        ga = parsed_gs.get("Goals Against")
                        gf_i = int(gf) if gf not in (None, "") else None
                        ga_i = int(ga) if ga not in (None, "") else None
                        if (
                            gf_i is not None
                            and ga_i is not None
                            and team_side in {"home", "away"}
                            and int(resolved_game_id) > 0
                        ):
                            if team_side == "home":
                                t1_score_new, t2_score_new = gf_i, ga_i
                            else:
                                t1_score_new, t2_score_new = ga_i, gf_i
                            if replace:
                                m.HkyGame.objects.filter(id=int(resolved_game_id)).update(
                                    team1_score=int(t1_score_new),
                                    team2_score=int(t2_score_new),
                                    updated_at=now,
                                )
                            else:
                                m.HkyGame.objects.filter(
                                    id=int(resolved_game_id), team1_score__isnull=True
                                ).update(team1_score=int(t1_score_new), updated_at=now)
                                m.HkyGame.objects.filter(
                                    id=int(resolved_game_id), team2_score__isnull=True
                                ).update(team2_score=int(t2_score_new), updated_at=now)
                    except Exception:
                        pass

                if isinstance(player_stats_csv, str) and player_stats_csv.strip():
                    parsed_rows = parse_shift_stats_player_stats_csv(player_stats_csv)
                    if replace:
                        # Ensure shift/time-derived stats are not stored when replacing.
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
                        # Determine team_id for this player
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
                            "toi_seconds",
                            "shifts",
                            "video_toi_seconds",
                            "sb_avg_shift_seconds",
                            "sb_median_shift_seconds",
                            "sb_longest_shift_seconds",
                            "sb_shortest_shift_seconds",
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
            return jsonify({"ok": False, "error": str(e)}), 400

        return jsonify(
            {
                "ok": True,
                "game_id": int(resolved_game_id),
                "imported_players": int(imported),
                "unmatched": [u for u in unmatched if u],
            }
        )

    @app.post("/api/internal/reset_league_data")
    def api_internal_reset_league_data():
        """
        Hidden administrative endpoint used by tools/webapp/scripts/reset_league_data.py.
        Requires import auth (token or localhost rules). Not linked from the UI.
        """
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
        league_name = str(payload.get("league_name") or "").strip()
        owner_email = str(payload.get("owner_email") or "").strip().lower()
        if not league_name or not owner_email:
            return jsonify({"ok": False, "error": "owner_email and league_name are required"}), 400

        _django_orm, m = _orm_modules()
        owner_user_id = (
            m.User.objects.filter(email=owner_email).values_list("id", flat=True).first()
        )
        if owner_user_id is None:
            return jsonify({"ok": False, "error": "owner_email_not_found"}), 404
        league_id = (
            m.League.objects.filter(name=league_name, owner_user_id=int(owner_user_id))
            .values_list("id", flat=True)
            .first()
        )
        if league_id is None:
            return jsonify({"ok": False, "error": "league_not_found_for_owner"}), 404

        try:
            stats = reset_league_data(None, int(league_id), owner_user_id=int(owner_user_id))
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "league_id": int(league_id), "stats": stats})

    @app.post("/api/internal/ensure_league_owner")
    def api_internal_ensure_league_owner():
        """
        Hidden administrative endpoint to ensure a league exists and is owned by the specified user.
        Requires import auth (token or localhost rules). Not linked from the UI.
        """
        auth = _require_import_auth()
        if auth:
            return auth
        payload = request.get_json(silent=True) or {}
        league_name = str(payload.get("league_name") or "").strip()
        owner_email = str(payload.get("owner_email") or "").strip().lower()
        owner_name = str(payload.get("owner_name") or owner_email).strip() or owner_email
        is_shared = bool(payload["shared"]) if "shared" in payload else None
        if not league_name or not owner_email:
            return jsonify({"ok": False, "error": "owner_email and league_name are required"}), 400

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

        return jsonify(
            {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id)}
        )

    @app.post("/leagues/new")
    def leagues_new():
        r = require_login()
        if r:
            return r
        name = request.form.get("name", "").strip()
        is_shared = 1 if request.form.get("is_shared") == "1" else 0
        is_public = 1 if request.form.get("is_public") == "1" else 0
        if not name:
            flash("Name is required", "error")
            return redirect(url_for("leagues_index"))
        _django_orm, m = _orm_modules()
        from django.db import IntegrityError, transaction

        uid = int(session["user_id"])  # type: ignore[arg-type]
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
            flash("Failed to create league (name may already exist)", "error")
            return redirect(url_for("leagues_index"))
        session["league_id"] = lid
        flash("League created and selected", "success")
        return redirect(url_for("leagues_index"))

    @app.post("/leagues/<int:league_id>/update")
    def leagues_update(league_id: int):
        r = require_login()
        if r:
            return r
        if not _is_league_admin(league_id, session["user_id"]):
            flash("Not authorized", "error")
            return redirect(url_for("leagues_index"))
        is_shared = 1 if request.form.get("is_shared") == "1" else 0
        is_public = 1 if request.form.get("is_public") == "1" else 0
        _django_orm, m = _orm_modules()
        m.League.objects.filter(id=int(league_id)).update(
            is_shared=bool(is_shared),
            is_public=bool(is_public),
            updated_at=dt.datetime.now(),
        )
        flash("League settings updated", "success")
        return redirect(url_for("leagues_index"))

    @app.post("/leagues/<int:league_id>/delete")
    def leagues_delete(league_id: int):
        r = require_login()
        if r:
            return r
        if not _is_league_admin(league_id, session["user_id"]):
            flash("Not authorized to delete this league", "error")
            return redirect(url_for("leagues_index"))

        def _chunks(ids: list[int], n: int = 500) -> list[list[int]]:
            return [ids[i : i + n] for i in range(0, len(ids), n)]

        _django_orm, m = _orm_modules()
        from django.db import transaction
        from django.db.models import Q

        try:
            with transaction.atomic():
                league_row = (
                    m.League.objects.filter(id=int(league_id)).values("owner_user_id").first()
                )
                if not league_row:
                    flash("Not found", "error")
                    return redirect(url_for("leagues_index"))
                owner_user_id = int(league_row["owner_user_id"])

                # Only delete games/teams that are exclusively mapped to this league.
                mapped_game_ids = set(
                    m.LeagueGame.objects.filter(league_id=int(league_id)).values_list(
                        "game_id", flat=True
                    )
                )
                other_game_ids = set(
                    m.LeagueGame.objects.exclude(league_id=int(league_id))
                    .filter(game_id__in=mapped_game_ids)
                    .values_list("game_id", flat=True)
                )
                exclusive_game_ids = sorted(
                    {int(gid) for gid in (mapped_game_ids - other_game_ids)}
                )

                mapped_team_ids = set(
                    m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                        "team_id", flat=True
                    )
                )
                other_team_ids = set(
                    m.LeagueTeam.objects.exclude(league_id=int(league_id))
                    .filter(team_id__in=mapped_team_ids)
                    .values_list("team_id", flat=True)
                )
                exclusive_team_ids = sorted(
                    {int(tid) for tid in (mapped_team_ids - other_team_ids)}
                )

                # Remove mappings + league row first.
                m.LeagueGame.objects.filter(league_id=int(league_id)).delete()
                m.LeagueTeam.objects.filter(league_id=int(league_id)).delete()
                m.LeagueMember.objects.filter(league_id=int(league_id)).delete()
                m.League.objects.filter(id=int(league_id)).delete()

                # Delete exclusive games (cascades to player_stats etc).
                for chunk in _chunks(exclusive_game_ids, n=500):
                    m.HkyGame.objects.filter(id__in=[int(x) for x in chunk]).delete()

                # Delete eligible external teams owned by league owner that are now unused by any remaining games.
                eligible_team_ids = list(
                    m.Team.objects.filter(
                        id__in=[int(x) for x in exclusive_team_ids],
                        user_id=int(owner_user_id),
                        is_external=True,
                    ).values_list("id", flat=True)
                )
                safe_team_ids: list[int] = []
                if eligible_team_ids:
                    used_pairs = list(
                        m.HkyGame.objects.filter(
                            Q(team1_id__in=eligible_team_ids) | Q(team2_id__in=eligible_team_ids)
                        ).values_list("team1_id", "team2_id")
                    )
                    still_used: set[int] = set()
                    for a, b in used_pairs:
                        if a is not None:
                            still_used.add(int(a))
                        if b is not None:
                            still_used.add(int(b))
                    safe_team_ids = [
                        int(tid) for tid in eligible_team_ids if int(tid) not in still_used
                    ]
                for chunk in _chunks(sorted(safe_team_ids), n=500):
                    m.Team.objects.filter(id__in=[int(x) for x in chunk]).delete()
            if session.get("league_id") == league_id:
                session.pop("league_id", None)
            flash("League and associated data deleted", "success")
        except Exception:
            flash("Failed to delete league", "error")
        return redirect(url_for("leagues_index"))

    def _is_league_admin(league_id: int, user_id: int) -> bool:
        _django_orm, m = _orm_modules()
        if m.League.objects.filter(id=int(league_id), owner_user_id=int(user_id)).exists():
            return True
        return m.LeagueMember.objects.filter(
            league_id=int(league_id),
            user_id=int(user_id),
            role__in=["admin", "owner"],
        ).exists()

    def _is_public_league(league_id: int) -> Optional[dict]:
        _django_orm, m = _orm_modules()
        return (
            m.League.objects.filter(id=int(league_id), is_public=True)
            .values("id", "name", "owner_user_id")
            .first()
        )

    @app.get("/public/leagues")
    def public_leagues_index():
        _django_orm, m = _orm_modules()
        leagues = list(
            m.League.objects.filter(is_public=True).order_by("name").values("id", "name")
        )
        return render_template("public_leagues.html", leagues=leagues)

    @app.get("/public/leagues/<int:league_id>")
    def public_league_home(league_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        return redirect(url_for("public_league_teams", league_id=league_id))

    @app.get("/public/leagues/<int:league_id>/media/team_logo/<int:team_id>")
    def public_media_team_logo(league_id: int, team_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        _django_orm, m = _orm_modules()
        row = (
            m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(team_id))
            .select_related("team")
            .values("team__logo_path")
            .first()
        )
        if not row or not row.get("team__logo_path"):
            return ("Not found", 404)
        p = Path(row["team__logo_path"]).resolve()
        if not p.exists():
            return ("Not found", 404)
        return send_from_directory(str(p.parent), p.name)

    @app.get("/public/leagues/<int:league_id>/teams")
    def public_league_teams(league_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        viewer_user_id = int(session.get("user_id") or 0) if "user_id" in session else 0
        league_owner_user_id = None
        try:
            league_owner_user_id = (
                int(league.get("owner_user_id")) if isinstance(league, dict) else None
            )
        except Exception:
            league_owner_user_id = None
        _record_league_page_view(
            g.db,
            int(league_id),
            kind=LEAGUE_PAGE_VIEW_KIND_TEAMS,
            entity_id=0,
            viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
            league_owner_user_id=league_owner_user_id,
        )
        is_league_owner = bool(
            viewer_user_id
            and league_owner_user_id is not None
            and int(viewer_user_id) == int(league_owner_user_id)
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
        for r in rows_raw:
            rows.append(
                {
                    "id": int(r["team_id"]),
                    "user_id": int(r["team__user_id"]),
                    "name": r.get("team__name"),
                    "logo_path": r.get("team__logo_path"),
                    "is_external": bool(r.get("team__is_external")),
                    "created_at": r.get("team__created_at"),
                    "updated_at": r.get("team__updated_at"),
                    "division_name": r.get("division_name"),
                    "division_id": r.get("division_id"),
                    "conference_id": r.get("conference_id"),
                    "mhr_rating": r.get("mhr_rating"),
                    "mhr_agd": r.get("mhr_agd"),
                    "mhr_sched": r.get("mhr_sched"),
                    "mhr_games": r.get("mhr_games"),
                    "mhr_updated_at": r.get("mhr_updated_at"),
                }
            )
        stats = {t["id"]: compute_team_stats_league(None, t["id"], int(league_id)) for t in rows}
        grouped: dict[str, list[dict]] = {}
        for t in rows:
            dn = str(t.get("division_name") or "").strip() or "Unknown Division"
            grouped.setdefault(dn, []).append(t)
        divisions = []
        for dn in sorted(grouped.keys(), key=division_sort_key):
            teams_sorted = sorted(
                grouped[dn], key=lambda tr: sort_key_team_standings(tr, stats.get(tr["id"], {}))
            )
            divisions.append({"name": dn, "teams": teams_sorted})
        return render_template(
            "teams.html",
            teams=rows,
            divisions=divisions,
            stats=stats,
            include_external=True,
            league_view=True,
            current_user_id=-1,
            public_league_id=int(league_id),
            is_league_admin=False,
            league_page_views=(
                {
                    "league_id": int(league_id),
                    "kind": LEAGUE_PAGE_VIEW_KIND_TEAMS,
                    "entity_id": 0,
                    "count": _get_league_page_view_count(
                        g.db, int(league_id), kind=LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
                    ),
                }
                if is_league_owner
                else None
            ),
        )

    @app.get("/public/leagues/<int:league_id>/teams/<int:team_id>")
    def public_league_team_detail(league_id: int, team_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        viewer_user_id = int(session.get("user_id") or 0) if "user_id" in session else 0
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
        recent_n_raw = request.args.get("recent_n")
        try:
            recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
        except Exception:
            recent_n = 5
        recent_sort = str(request.args.get("recent_sort") or "points").strip() or "points"
        recent_dir = str(request.args.get("recent_dir") or "desc").strip().lower() or "desc"
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
            return ("Not found", 404)
        _record_league_page_view(
            g.db,
            int(league_id),
            kind=LEAGUE_PAGE_VIEW_KIND_TEAM,
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
        skaters, goalies, head_coaches, assistant_coaches = split_roster(players or [])
        roster_players = list(skaters) + list(goalies)
        tstats = compute_team_stats_league(g.db, team_id, int(league_id))
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
        for r in schedule_rows_raw:
            t1 = int(r["game__team1_id"])
            t2 = int(r["game__team2_id"])
            schedule_games.append(
                {
                    "id": int(r["game_id"]),
                    "user_id": int(r["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r.get("game__game_type_id"),
                    "starts_at": r.get("game__starts_at"),
                    "location": r.get("game__location"),
                    "notes": r.get("game__notes"),
                    "team1_score": r.get("game__team1_score"),
                    "team2_score": r.get("game__team2_score"),
                    "is_final": r.get("game__is_final"),
                    "stats_imported_at": r.get("game__stats_imported_at"),
                    "created_at": r.get("game__created_at"),
                    "updated_at": r.get("game__updated_at"),
                    "team1_name": r.get("game__team1__name"),
                    "team2_name": r.get("game__team2__name"),
                    "game_type_name": r.get("game__game_type__name"),
                    "division_name": r.get("division_name"),
                    "sort_order": r.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
        schedule_games = [
            g2
            for g2 in (schedule_games or [])
            if not _league_game_is_cross_division_non_external(g2)
        ]
        now_dt = dt.datetime.now()
        for g2 in schedule_games:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = _sanitize_http_url(
                    _extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
        schedule_games = sort_games_schedule_order(schedule_games or [])

        schedule_game_ids = [
            int(g2.get("id")) for g2 in (schedule_games or []) if g2.get("id") is not None
        ]
        ps_rows = list(
            m.PlayerStat.objects.filter(team_id=int(team_id), game_id__in=schedule_game_ids).values(
                "player_id", "game_id", *PLAYER_STATS_SUM_KEYS
            )
        )

        # Game type filter applies to player stats tables only.
        for g2 in schedule_games or []:
            try:
                g2["_game_type_label"] = _game_type_label_for_row(g2)
            except Exception:
                g2["_game_type_label"] = "Unknown"
        game_type_options = _dedupe_preserve_str(
            [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
        )
        selected_types = _parse_selected_game_type_labels(
            available=game_type_options, args=request.args
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
        eligible_games = [g2 for g2 in stats_schedule_games if _game_has_recorded_result(g2)]
        eligible_game_ids_in_order: list[int] = []
        for g2 in eligible_games:
            try:
                eligible_game_ids_in_order.append(int(g2.get("id")))
            except Exception:
                continue
        eligible_game_ids: set[int] = set(eligible_game_ids_in_order)
        ps_rows_filtered = []
        for r in ps_rows or []:
            try:
                if int(r.get("game_id")) in eligible_game_ids:
                    ps_rows_filtered.append(r)
            except Exception:
                continue

        player_totals = _aggregate_player_totals_from_rows(
            player_stats_rows=ps_rows_filtered, allowed_game_ids=eligible_game_ids
        )
        player_stats_rows = sort_players_table_default(
            build_player_stats_table_rows(skaters, player_totals)
        )
        player_stats_columns = filter_player_stats_display_columns_for_rows(
            PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows
        )
        cov_counts, cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=eligible_game_ids_in_order
        )
        player_stats_columns = _player_stats_columns_with_coverage(
            columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
        )

        recent_scope_ids = (
            eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
        )
        recent_totals = compute_recent_player_totals_from_rows(
            schedule_games=stats_schedule_games, player_stats_rows=ps_rows_filtered, n=recent_n
        )
        recent_player_stats_rows = sort_player_stats_rows(
            build_player_stats_table_rows(skaters, recent_totals),
            sort_key=recent_sort,
            sort_dir=recent_dir,
        )
        recent_player_stats_columns = filter_player_stats_display_columns_for_rows(
            PLAYER_STATS_DISPLAY_COLUMNS, recent_player_stats_rows
        )
        recent_cov_counts, recent_cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=recent_scope_ids
        )
        recent_player_stats_columns = _player_stats_columns_with_coverage(
            columns=recent_player_stats_columns,
            coverage_counts=recent_cov_counts,
            total_games=recent_cov_total,
        )

        player_stats_sources = _compute_team_player_stats_sources(
            g.db, eligible_game_ids=eligible_game_ids_in_order
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
        return render_template(
            "team_detail.html",
            team=team,
            roster_players=roster_players,
            players=skaters,
            head_coaches=head_coaches,
            assistant_coaches=assistant_coaches,
            player_stats_columns=player_stats_columns,
            player_stats_rows=player_stats_rows,
            recent_player_stats_columns=recent_player_stats_columns,
            recent_player_stats_rows=recent_player_stats_rows,
            recent_n=recent_n,
            recent_sort=recent_sort,
            recent_dir=recent_dir,
            tstats=tstats,
            schedule_games=schedule_games,
            editable=False,
            public_league_id=int(league_id),
            player_stats_sources=player_stats_sources,
            player_stats_coverage_total_games=cov_total,
            player_stats_recent_coverage_total_games=recent_cov_total,
            game_type_filter_options=game_type_filter_options,
            game_type_filter_label=selected_label,
            league_page_views=(
                {
                    "league_id": int(league_id),
                    "kind": LEAGUE_PAGE_VIEW_KIND_TEAM,
                    "entity_id": int(team_id),
                    "count": _get_league_page_view_count(
                        g.db,
                        int(league_id),
                        kind=LEAGUE_PAGE_VIEW_KIND_TEAM,
                        entity_id=int(team_id),
                    ),
                }
                if is_league_owner
                else None
            ),
        )

    @app.get("/public/leagues/<int:league_id>/schedule")
    def public_league_schedule(league_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        viewer_user_id = int(session.get("user_id") or 0) if "user_id" in session else 0
        league_owner_user_id = None
        try:
            league_owner_user_id = (
                int(league.get("owner_user_id")) if isinstance(league, dict) else None
            )
        except Exception:
            league_owner_user_id = None
        _record_league_page_view(
            g.db,
            int(league_id),
            kind=LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            entity_id=0,
            viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
            league_owner_user_id=league_owner_user_id,
        )
        is_league_owner = bool(
            viewer_user_id
            and league_owner_user_id is not None
            and int(viewer_user_id) == int(league_owner_user_id)
        )
        selected_division = (request.args.get("division") or "").strip() or None
        selected_team_id = request.args.get("team_id") or ""
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
            .order_by("division_name")
        )
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
        games: list[dict[str, Any]] = []
        for r in rows_raw:
            t1 = int(r["game__team1_id"])
            t2 = int(r["game__team2_id"])
            games.append(
                {
                    "id": int(r["game_id"]),
                    "user_id": int(r["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r.get("game__game_type_id"),
                    "starts_at": r.get("game__starts_at"),
                    "location": r.get("game__location"),
                    "notes": r.get("game__notes"),
                    "team1_score": r.get("game__team1_score"),
                    "team2_score": r.get("game__team2_score"),
                    "is_final": r.get("game__is_final"),
                    "stats_imported_at": r.get("game__stats_imported_at"),
                    "created_at": r.get("game__created_at"),
                    "updated_at": r.get("game__updated_at"),
                    "team1_name": r.get("game__team1__name"),
                    "team2_name": r.get("game__team2__name"),
                    "game_type_name": r.get("game__game_type__name"),
                    "division_name": r.get("division_name"),
                    "sort_order": r.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
        games = [g2 for g2 in (games or []) if not _league_game_is_cross_division_non_external(g2)]
        now_dt = dt.datetime.now()
        for g2 in games or []:
            try:
                g2["game_video_url"] = _sanitize_http_url(
                    _extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            # Hide game pages for future scheduled games that have not started and have no score yet.
            # If starts_at is missing (common for imported games), allow viewing.
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            g2["can_edit"] = False
        games = sort_games_schedule_order(games or [])
        return render_template(
            "schedule.html",
            games=games,
            league_view=True,
            divisions=divisions,
            league_teams=league_teams,
            selected_division=selected_division or "",
            selected_team_id=str(team_id_i) if team_id_i is not None else "",
            can_add_game=False,
            public_league_id=int(league_id),
            league_page_views=(
                {
                    "league_id": int(league_id),
                    "kind": LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
                    "entity_id": 0,
                    "count": _get_league_page_view_count(
                        g.db, int(league_id), kind=LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
                    ),
                }
                if is_league_owner
                else None
            ),
        )

    @app.get("/public/leagues/<int:league_id>/hky/games/<int:game_id>")
    def public_league_game_detail(league_id: int, game_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        viewer_user_id = int(session.get("user_id") or 0) if "user_id" in session else 0
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
            return ("Not found", 404)

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
            game["game_video_url"] = _sanitize_http_url(
                _extract_game_video_url_from_notes(game.get("notes"))
            )
        except Exception:
            game["game_video_url"] = None
        if _league_game_is_cross_division_non_external(game):
            return ("Not found", 404)
        now_dt = dt.datetime.now()
        sdt = game.get("starts_at")
        started = False
        if sdt is not None:
            try:
                started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
            except Exception:
                started = False
        has_score = (
            (game.get("team1_score") is not None)
            or (game.get("team2_score") is not None)
            or bool(game.get("is_final"))
        )
        can_view_summary = bool(has_score or (sdt is None) or started)
        if not can_view_summary:
            return ("Not found", 404)
        _record_league_page_view(
            g.db,
            int(league_id),
            kind=LEAGUE_PAGE_VIEW_KIND_GAME,
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
        team1_skaters, team1_goalies, team1_hc, team1_ac = split_roster(team1_players)
        team2_skaters, team2_goalies, team2_hc, team2_ac = split_roster(team2_players)
        team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
        team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
        stats_by_pid = {r["player_id"]: r for r in stats_rows}
        period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}
        tts_linked = _extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

        events_headers: list[str] = []
        events_rows: list[dict[str, str]] = []
        events_meta: Optional[dict[str, Any]] = None
        try:
            qs = (
                m.HkyGameEventRow.objects.filter(game_id=int(game_id))
                .select_related("event_type")
                .order_by("period", "game_seconds", "id")
            )
            qs = qs.exclude(
                import_key__in=m.HkyGameEventSuppression.objects.filter(
                    game_id=int(game_id)
                ).values_list("import_key", flat=True)
            )
            raw_rows = list(
                qs.values(
                    "event_type__name",
                    "event_id",
                    "source",
                    "team_raw",
                    "team_side",
                    "for_against",
                    "team_rel",
                    "period",
                    "game_time",
                    "video_time",
                    "game_seconds",
                    "game_seconds_end",
                    "video_seconds",
                    "details",
                    "attributed_players",
                    "attributed_jerseys",
                    "on_ice_players",
                    "on_ice_players_home",
                    "on_ice_players_away",
                    "created_at",
                    "updated_at",
                )
            )
            if raw_rows:
                events_headers = [
                    "Event Type",
                    "Event ID",
                    "Source",
                    "Team Raw",
                    "Team Side",
                    "For/Against",
                    "Team Rel",
                    "Period",
                    "Game Time",
                    "Video Time",
                    "Game Seconds",
                    "Game Seconds End",
                    "Video Seconds",
                    "Details",
                    "Attributed Players",
                    "Attributed Jerseys",
                    "On-Ice Players",
                    "On-Ice Players (Home)",
                    "On-Ice Players (Away)",
                ]
                max_ts: Optional[dt.datetime] = None
                for r in raw_rows:
                    ts = r.get("updated_at") or r.get("created_at")
                    if isinstance(ts, dt.datetime):
                        max_ts = ts if max_ts is None else max(max_ts, ts)

                    events_rows.append(
                        {
                            "Event Type": str(r.get("event_type__name") or "").strip(),
                            "Event ID": (
                                "" if r.get("event_id") is None else str(int(r["event_id"]))
                            ),
                            "Source": str(r.get("source") or "").strip(),
                            "Team Raw": str(r.get("team_raw") or "").strip(),
                            "Team Side": str(r.get("team_side") or "").strip(),
                            "For/Against": str(r.get("for_against") or "").strip(),
                            "Team Rel": str(r.get("team_rel") or "").strip(),
                            "Period": "" if r.get("period") is None else str(int(r["period"])),
                            "Game Time": str(r.get("game_time") or "").strip(),
                            "Video Time": str(r.get("video_time") or "").strip(),
                            "Game Seconds": (
                                "" if r.get("game_seconds") is None else str(int(r["game_seconds"]))
                            ),
                            "Game Seconds End": (
                                ""
                                if r.get("game_seconds_end") is None
                                else str(int(r["game_seconds_end"]))
                            ),
                            "Video Seconds": (
                                ""
                                if r.get("video_seconds") is None
                                else str(int(r["video_seconds"]))
                            ),
                            "Details": str(r.get("details") or "").strip(),
                            "Attributed Players": str(r.get("attributed_players") or "").strip(),
                            "Attributed Jerseys": str(r.get("attributed_jerseys") or "").strip(),
                            "On-Ice Players": str(r.get("on_ice_players") or "").strip(),
                            "On-Ice Players (Home)": str(
                                r.get("on_ice_players_home") or ""
                            ).strip(),
                            "On-Ice Players (Away)": str(
                                r.get("on_ice_players_away") or ""
                            ).strip(),
                        }
                    )

                events_headers, events_rows = normalize_game_events_csv(events_headers, events_rows)
                events_rows = filter_events_rows_prefer_timetoscore_for_goal_assist(
                    events_rows, tts_linked=tts_linked
                )
                events_headers, events_rows = normalize_events_video_time_for_display(
                    events_headers, events_rows
                )
                events_headers, events_rows = filter_events_headers_drop_empty_on_ice_split(
                    events_headers, events_rows
                )
                events_rows = sort_events_rows_default(events_rows)
                events_meta = {
                    "source_label": "db",
                    "updated_at": max_ts,
                    "count": len(events_rows),
                    "sources": summarize_event_sources(events_rows, fallback_source_label="db"),
                }
        except Exception:
            events_headers, events_rows, events_meta = [], [], None

        scoring_by_period_rows = compute_team_scoring_by_period_from_events(
            events_rows, tts_linked=tts_linked
        )
        game_event_stats_rows = compute_game_event_stats_by_side(events_rows)
        game_event_stats_rows = compute_game_event_stats_by_side(events_rows)

        imported_player_stats_csv_text: Optional[str] = None
        player_stats_import_meta: Optional[dict[str, Any]] = None

        (
            game_player_stats_columns,
            player_stats_cells_by_pid,
            player_stats_cell_conflicts_by_pid,
            player_stats_import_warning,
        ) = build_game_player_stats_table(
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
                    g = _parse_int_from_cell_text(cells.get("goals", ""))
                    a = _parse_int_from_cell_text(cells.get("assists", ""))
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
                sorted_rows = sort_players_table_default(rows)
                return [
                    by_pid[int(r["player_id"])]
                    for r in sorted_rows
                    if int(r["player_id"]) in by_pid
                ]

            team1_skaters_sorted = _sort_players_for_game(team1_skaters_sorted)
            team2_skaters_sorted = _sort_players_for_game(team2_skaters_sorted)
        except Exception:
            team1_skaters_sorted = list(team1_skaters)
            team2_skaters_sorted = list(team2_skaters)
        default_back_url = f"/public/leagues/{int(league_id)}/schedule"
        return_to = _safe_return_to_url(request.args.get("return_to"), default=default_back_url)
        public_user_id = int(session.get("user_id") or 0) if "user_id" in session else 0
        public_is_logged_in = bool(public_user_id)
        return render_template(
            "hky_game_detail.html",
            game=game,
            team1_roster=team1_roster,
            team2_roster=team2_roster,
            team1_players=team1_skaters_sorted,
            team2_players=team2_skaters_sorted,
            stats_by_pid=stats_by_pid,
            period_stats_by_pid=period_stats_by_pid,
            editable=False,
            can_edit=False,
            edit_mode=False,
            public_league_id=int(league_id),
            back_url=return_to,
            return_to=return_to,
            events_headers=events_headers,
            events_rows=events_rows,
            events_meta=events_meta,
            scoring_by_period_rows=scoring_by_period_rows,
            game_event_stats_rows=game_event_stats_rows,
            user_video_clip_len_s=(
                get_user_video_clip_len_s(g.db, public_user_id) if public_is_logged_in else None
            ),
            user_is_logged_in=public_is_logged_in,
            game_player_stats_columns=game_player_stats_columns,
            player_stats_cells_by_pid=player_stats_cells_by_pid,
            player_stats_cell_conflicts_by_pid=player_stats_cell_conflicts_by_pid,
            player_stats_import_meta=player_stats_import_meta,
            player_stats_import_warning=player_stats_import_warning,
            league_page_views=(
                {
                    "league_id": int(league_id),
                    "kind": LEAGUE_PAGE_VIEW_KIND_GAME,
                    "entity_id": int(game_id),
                    "count": _get_league_page_view_count(
                        g.db,
                        int(league_id),
                        kind=LEAGUE_PAGE_VIEW_KIND_GAME,
                        entity_id=int(game_id),
                    ),
                }
                if is_league_owner
                else None
            ),
        )

    @app.get("/leagues/<int:league_id>/members")
    def league_members(league_id: int):
        r = require_login()
        if r:
            return r
        if not _is_league_admin(league_id, session["user_id"]):
            flash("Not authorized", "error")
            return redirect(url_for("leagues_index"))
        _django_orm, m = _orm_modules()
        rows = list(
            m.LeagueMember.objects.filter(league_id=int(league_id))
            .select_related("user")
            .order_by("user__email")
            .values("user_id", "user__email", "role")
        )
        members = [
            {"id": int(r["user_id"]), "email": r["user__email"], "role": (r.get("role") or "admin")}
            for r in rows
        ]
        return render_template("league_members.html", league_id=league_id, members=members)

    @app.post("/leagues/<int:league_id>/members")
    def league_members_add(league_id: int):
        r = require_login()
        if r:
            return r
        if not _is_league_admin(league_id, session["user_id"]):
            flash("Not authorized", "error")
            return redirect(url_for("leagues_index"))
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "viewer")
        if not email:
            flash("Email required", "error")
            return redirect(url_for("league_members", league_id=league_id))
        _django_orm, m = _orm_modules()
        from django.db import transaction

        uid = m.User.objects.filter(email=email).values_list("id", flat=True).first()
        if uid is None:
            flash("User not found. Ask them to register first.", "error")
            return redirect(url_for("league_members", league_id=league_id))
        now = dt.datetime.now()
        with transaction.atomic():
            member, created = m.LeagueMember.objects.get_or_create(
                league_id=int(league_id),
                user_id=int(uid),
                defaults={"role": str(role or "viewer"), "created_at": now},
            )
            if not created and str(getattr(member, "role", "") or "") != str(role or "viewer"):
                m.LeagueMember.objects.filter(id=int(member.id)).update(role=str(role or "viewer"))
        flash("Member added/updated", "success")
        return redirect(url_for("league_members", league_id=league_id))

    @app.post("/leagues/<int:league_id>/members/remove")
    def league_members_remove(league_id: int):
        r = require_login()
        if r:
            return r
        if not _is_league_admin(league_id, session["user_id"]):
            flash("Not authorized", "error")
            return redirect(url_for("leagues_index"))
        uid = int(request.form.get("user_id") or 0)
        _django_orm, m = _orm_modules()
        m.LeagueMember.objects.filter(league_id=int(league_id), user_id=int(uid)).delete()
        flash("Member removed", "success")
        return redirect(url_for("league_members", league_id=league_id))

    @app.route("/teams/new", methods=["GET", "POST"])
    def new_team():
        r = require_login()
        if r:
            return r
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            if not name:
                flash("Team name is required", "error")
                return render_template("team_new.html")
            tid = create_team(session["user_id"], name, is_external=False)
            # handle logo upload
            f = request.files.get("logo")
            if f and f.filename:
                try:
                    p = save_team_logo(f, tid)
                    _django_orm, m = _orm_modules()
                    m.Team.objects.filter(id=int(tid), user_id=int(session["user_id"])).update(
                        logo_path=str(p)
                    )
                except Exception:
                    flash("Failed to save team logo", "error")
            flash("Team created", "success")
            return redirect(url_for("team_detail", team_id=tid))
        return render_template("team_new.html")

    @app.route("/teams/<int:team_id>")
    def team_detail(team_id: int):
        r = require_login()
        if r:
            return r
        recent_n_raw = request.args.get("recent_n")
        try:
            recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
        except Exception:
            recent_n = 5
        recent_sort = str(request.args.get("recent_sort") or "points").strip() or "points"
        recent_dir = str(request.args.get("recent_dir") or "desc").strip().lower() or "desc"
        league_id = session.get("league_id")
        league_owner_user_id: Optional[int] = None
        is_league_owner = False
        if league_id:
            league_owner_user_id = _get_league_owner_user_id(g.db, int(league_id))
            is_league_owner = bool(
                league_owner_user_id is not None
                and int(league_owner_user_id) == int(session["user_id"])
            )
        is_league_admin = False
        if league_id:
            try:
                is_league_admin = bool(_is_league_admin(int(league_id), int(session["user_id"])))
            except Exception:
                is_league_admin = False
        team = get_team(team_id, session["user_id"])
        editable = bool(team)
        if not team and league_id:
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
            flash("Not found", "error")
            return redirect(url_for("teams"))
        if league_id:
            _record_league_page_view(
                g.db,
                int(league_id),
                kind=LEAGUE_PAGE_VIEW_KIND_TEAM,
                entity_id=int(team_id),
                viewer_user_id=int(session["user_id"]),
                league_owner_user_id=league_owner_user_id,
            )
        team_owner_id = int(team["user_id"])
        _django_orm, m = _orm_modules()
        players_qs = m.Player.objects.filter(team_id=int(team_id))
        if editable:
            players_qs = players_qs.filter(user_id=int(session["user_id"]))  # type: ignore[arg-type]
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
        skaters, goalies, head_coaches, assistant_coaches = split_roster(players or [])
        roster_players = list(skaters) + list(goalies)
        if league_id:
            tstats = compute_team_stats_league(g.db, team_id, int(league_id))
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
            for r in schedule_rows_raw:
                t1 = int(r["game__team1_id"])
                t2 = int(r["game__team2_id"])
                schedule_games.append(
                    {
                        "id": int(r["game_id"]),
                        "user_id": int(r["game__user_id"]),
                        "team1_id": t1,
                        "team2_id": t2,
                        "game_type_id": r.get("game__game_type_id"),
                        "starts_at": r.get("game__starts_at"),
                        "location": r.get("game__location"),
                        "notes": r.get("game__notes"),
                        "team1_score": r.get("game__team1_score"),
                        "team2_score": r.get("game__team2_score"),
                        "is_final": r.get("game__is_final"),
                        "stats_imported_at": r.get("game__stats_imported_at"),
                        "created_at": r.get("game__created_at"),
                        "updated_at": r.get("game__updated_at"),
                        "team1_name": r.get("game__team1__name"),
                        "team2_name": r.get("game__team2__name"),
                        "game_type_name": r.get("game__game_type__name"),
                        "division_name": r.get("division_name"),
                        "sort_order": r.get("sort_order"),
                        "team1_league_division_name": league_team_div_map.get(t1),
                        "team2_league_division_name": league_team_div_map.get(t2),
                    }
                )

            schedule_games = [
                g2
                for g2 in (schedule_games or [])
                if not _league_game_is_cross_division_non_external(g2)
            ]
            now_dt = dt.datetime.now()
            for g2 in schedule_games:
                sdt = g2.get("starts_at")
                started = False
                if sdt is not None:
                    try:
                        started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                    except Exception:
                        started = False
                has_score = (
                    (g2.get("team1_score") is not None)
                    or (g2.get("team2_score") is not None)
                    or bool(g2.get("is_final"))
                )
                g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
                try:
                    g2["game_video_url"] = _sanitize_http_url(
                        _extract_game_video_url_from_notes(g2.get("notes"))
                    )
                except Exception:
                    g2["game_video_url"] = None
            schedule_games = sort_games_schedule_order(schedule_games or [])
            schedule_game_ids = [
                int(g2.get("id")) for g2 in (schedule_games or []) if g2.get("id") is not None
            ]
            ps_rows = list(
                m.PlayerStat.objects.filter(
                    team_id=int(team_id), game_id__in=schedule_game_ids
                ).values("player_id", "game_id", *PLAYER_STATS_SUM_KEYS)
            )
        else:
            tstats = compute_team_stats(g.db, team_id, team_owner_id)
            from django.db.models import Q

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
            for r in schedule_rows:
                schedule_games.append(
                    {
                        "id": int(r["id"]),
                        "user_id": int(r["user_id"]),
                        "team1_id": int(r["team1_id"]),
                        "team2_id": int(r["team2_id"]),
                        "game_type_id": r.get("game_type_id"),
                        "starts_at": r.get("starts_at"),
                        "location": r.get("location"),
                        "notes": r.get("notes"),
                        "team1_score": r.get("team1_score"),
                        "team2_score": r.get("team2_score"),
                        "is_final": r.get("is_final"),
                        "stats_imported_at": r.get("stats_imported_at"),
                        "created_at": r.get("created_at"),
                        "updated_at": r.get("updated_at"),
                        "team1_name": r.get("team1__name"),
                        "team2_name": r.get("team2__name"),
                        "game_type_name": r.get("game_type__name"),
                    }
                )
            now_dt = dt.datetime.now()
            for g2 in schedule_games:
                sdt = g2.get("starts_at")
                started = False
                if sdt is not None:
                    try:
                        started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                    except Exception:
                        started = False
                has_score = (
                    (g2.get("team1_score") is not None)
                    or (g2.get("team2_score") is not None)
                    or bool(g2.get("is_final"))
                )
                g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            schedule_game_ids = [
                int(g2.get("id")) for g2 in (schedule_games or []) if g2.get("id") is not None
            ]
            ps_rows = list(
                m.PlayerStat.objects.filter(
                    team_id=int(team_id), game_id__in=schedule_game_ids
                ).values("player_id", "game_id", *PLAYER_STATS_SUM_KEYS)
            )

        # Game type filter applies to player stats tables only.
        for g2 in schedule_games or []:
            try:
                g2["_game_type_label"] = _game_type_label_for_row(g2)
            except Exception:
                g2["_game_type_label"] = "Unknown"
        game_type_options = _dedupe_preserve_str(
            [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
        )
        selected_types = _parse_selected_game_type_labels(
            available=game_type_options, args=request.args
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
        eligible_games = [g2 for g2 in stats_schedule_games if _game_has_recorded_result(g2)]
        eligible_game_ids_in_order: list[int] = []
        for g2 in eligible_games:
            try:
                eligible_game_ids_in_order.append(int(g2.get("id")))
            except Exception:
                continue
        eligible_game_ids: set[int] = set(eligible_game_ids_in_order)
        ps_rows_filtered = []
        for r in ps_rows or []:
            try:
                if int(r.get("game_id")) in eligible_game_ids:
                    ps_rows_filtered.append(r)
            except Exception:
                continue

        player_totals = _aggregate_player_totals_from_rows(
            player_stats_rows=ps_rows_filtered, allowed_game_ids=eligible_game_ids
        )
        player_stats_rows = sort_players_table_default(
            build_player_stats_table_rows(skaters, player_totals)
        )
        player_stats_columns = filter_player_stats_display_columns_for_rows(
            PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows
        )
        cov_counts, cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=eligible_game_ids_in_order
        )
        player_stats_columns = _player_stats_columns_with_coverage(
            columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
        )
        recent_scope_ids = (
            eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
        )
        recent_totals = compute_recent_player_totals_from_rows(
            schedule_games=stats_schedule_games, player_stats_rows=ps_rows_filtered, n=recent_n
        )
        recent_player_stats_rows = sort_player_stats_rows(
            build_player_stats_table_rows(skaters, recent_totals),
            sort_key=recent_sort,
            sort_dir=recent_dir,
        )
        recent_player_stats_columns = filter_player_stats_display_columns_for_rows(
            PLAYER_STATS_DISPLAY_COLUMNS, recent_player_stats_rows
        )
        recent_cov_counts, recent_cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=recent_scope_ids
        )
        recent_player_stats_columns = _player_stats_columns_with_coverage(
            columns=recent_player_stats_columns,
            coverage_counts=recent_cov_counts,
            total_games=recent_cov_total,
        )

        player_stats_sources = _compute_team_player_stats_sources(
            g.db, eligible_game_ids=eligible_game_ids_in_order
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
                "kind": LEAGUE_PAGE_VIEW_KIND_TEAM,
                "entity_id": int(team_id),
                "count": _get_league_page_view_count(
                    g.db, int(league_id), kind=LEAGUE_PAGE_VIEW_KIND_TEAM, entity_id=int(team_id)
                ),
            }
        return render_template(
            "team_detail.html",
            team=team,
            roster_players=roster_players,
            players=skaters,
            head_coaches=head_coaches,
            assistant_coaches=assistant_coaches,
            player_stats_columns=player_stats_columns,
            player_stats_rows=player_stats_rows,
            recent_player_stats_columns=recent_player_stats_columns,
            recent_player_stats_rows=recent_player_stats_rows,
            recent_n=recent_n,
            recent_sort=recent_sort,
            recent_dir=recent_dir,
            tstats=tstats,
            schedule_games=schedule_games,
            editable=editable,
            is_league_admin=is_league_admin,
            player_stats_sources=player_stats_sources,
            player_stats_coverage_total_games=cov_total,
            player_stats_recent_coverage_total_games=recent_cov_total,
            game_type_filter_options=game_type_filter_options,
            game_type_filter_label=selected_label,
            league_page_views=league_page_views,
        )

    @app.route("/teams/<int:team_id>/edit", methods=["GET", "POST"])
    def team_edit(team_id: int):
        r = require_login()
        if r:
            return r
        team = get_team(team_id, session["user_id"])
        if not team:
            flash("Not found", "error")
            return redirect(url_for("teams"))
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            if name:
                _django_orm, m = _orm_modules()
                m.Team.objects.filter(id=int(team_id), user_id=int(session["user_id"])).update(
                    name=name
                )
            f = request.files.get("logo")
            if f and f.filename:
                p = save_team_logo(f, team_id)
                _django_orm, m = _orm_modules()
                m.Team.objects.filter(id=int(team_id), user_id=int(session["user_id"])).update(
                    logo_path=str(p)
                )
            flash("Team updated", "success")
            return redirect(url_for("team_detail", team_id=team_id))
        return render_template("team_edit.html", team=team)

    @app.route("/teams/<int:team_id>/players/new", methods=["GET", "POST"])
    def player_new(team_id: int):
        r = require_login()
        if r:
            return r
        team = get_team(team_id, session["user_id"])
        if not team:
            flash("Not found", "error")
            return redirect(url_for("teams"))
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            jersey = request.form.get("jersey_number", "").strip()
            position = request.form.get("position", "").strip()
            shoots = request.form.get("shoots", "").strip()
            if not name:
                flash("Player name is required", "error")
                return render_template("player_edit.html", team=team)
            _django_orm, m = _orm_modules()
            m.Player.objects.create(
                user_id=int(session["user_id"]),
                team_id=int(team_id),
                name=name,
                jersey_number=jersey or None,
                position=position or None,
                shoots=shoots or None,
                created_at=dt.datetime.now(),
                updated_at=None,
            )
            flash("Player added", "success")
            return redirect(url_for("team_detail", team_id=team_id))
        return render_template("player_edit.html", team=team)

    @app.route("/teams/<int:team_id>/players/<int:player_id>/edit", methods=["GET", "POST"])
    def player_edit(team_id: int, player_id: int):
        r = require_login()
        if r:
            return r
        team = get_team(team_id, session["user_id"])
        if not team:
            flash("Not found", "error")
            return redirect(url_for("teams"))
        _django_orm, m = _orm_modules()
        pl = (
            m.Player.objects.filter(
                id=int(player_id), team_id=int(team_id), user_id=int(session["user_id"])
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
            flash("Not found", "error")
            return redirect(url_for("team_detail", team_id=team_id))
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            jersey = request.form.get("jersey_number", "").strip()
            position = request.form.get("position", "").strip()
            shoots = request.form.get("shoots", "").strip()
            m.Player.objects.filter(
                id=int(player_id), team_id=int(team_id), user_id=int(session["user_id"])
            ).update(
                name=name or pl["name"],
                jersey_number=jersey or None,
                position=position or None,
                shoots=shoots or None,
                updated_at=dt.datetime.now(),
            )
            flash("Player updated", "success")
            return redirect(url_for("team_detail", team_id=team_id))
        return render_template("player_edit.html", team=team, player=pl)

    @app.route("/teams/<int:team_id>/players/<int:player_id>/delete", methods=["POST"])
    def player_delete(team_id: int, player_id: int):
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        m.Player.objects.filter(
            id=int(player_id), team_id=int(team_id), user_id=int(session["user_id"])
        ).delete()
        flash("Player deleted", "success")
        return redirect(url_for("team_detail", team_id=team_id))

    @app.route("/schedule")
    def schedule():
        r = require_login()
        if r:
            return r
        league_id = session.get("league_id")
        league_owner_user_id: Optional[int] = None
        is_league_owner = False
        if league_id:
            league_owner_user_id = _get_league_owner_user_id(g.db, int(league_id))
            is_league_owner = bool(
                league_owner_user_id is not None
                and int(league_owner_user_id) == int(session["user_id"])
            )
            _record_league_page_view(
                g.db,
                int(league_id),
                kind=LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
                entity_id=0,
                viewer_user_id=int(session["user_id"]),
                league_owner_user_id=league_owner_user_id,
            )
        selected_division = (request.args.get("division") or "").strip() or None
        selected_team_id = request.args.get("team_id") or ""
        team_id_i: Optional[int] = None
        try:
            team_id_i = int(selected_team_id) if str(selected_team_id).strip() else None
        except Exception:
            team_id_i = None
        divisions = []
        league_teams = []
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
                .order_by("division_name")
            )
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
            if team_id_i is not None and not any(
                int(t["id"]) == int(team_id_i) for t in league_teams
            ):
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
                lg_qs.select_related(
                    "game", "game__team1", "game__team2", "game__game_type"
                ).values(
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
            for r in rows_raw:
                t1 = int(r["game__team1_id"])
                t2 = int(r["game__team2_id"])
                games.append(
                    {
                        "id": int(r["game_id"]),
                        "user_id": int(r["game__user_id"]),
                        "team1_id": t1,
                        "team2_id": t2,
                        "game_type_id": r.get("game__game_type_id"),
                        "starts_at": r.get("game__starts_at"),
                        "location": r.get("game__location"),
                        "notes": r.get("game__notes"),
                        "team1_score": r.get("game__team1_score"),
                        "team2_score": r.get("game__team2_score"),
                        "is_final": r.get("game__is_final"),
                        "stats_imported_at": r.get("game__stats_imported_at"),
                        "created_at": r.get("game__created_at"),
                        "updated_at": r.get("game__updated_at"),
                        "team1_name": r.get("game__team1__name"),
                        "team2_name": r.get("game__team2__name"),
                        "game_type_name": r.get("game__game_type__name"),
                        "division_name": r.get("division_name"),
                        "sort_order": r.get("sort_order"),
                        "team1_league_division_name": league_team_div_map.get(t1),
                        "team2_league_division_name": league_team_div_map.get(t2),
                    }
                )
        else:
            rows = list(
                m.HkyGame.objects.filter(user_id=int(session["user_id"]))  # type: ignore[arg-type]
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
            for r in rows:
                games.append(
                    {
                        "id": int(r["id"]),
                        "user_id": int(r["user_id"]),
                        "team1_id": int(r["team1_id"]),
                        "team2_id": int(r["team2_id"]),
                        "game_type_id": r.get("game_type_id"),
                        "starts_at": r.get("starts_at"),
                        "location": r.get("location"),
                        "notes": r.get("notes"),
                        "team1_score": r.get("team1_score"),
                        "team2_score": r.get("team2_score"),
                        "is_final": r.get("is_final"),
                        "stats_imported_at": r.get("stats_imported_at"),
                        "created_at": r.get("created_at"),
                        "updated_at": r.get("updated_at"),
                        "team1_name": r.get("team1__name"),
                        "team2_name": r.get("team2__name"),
                        "game_type_name": r.get("game_type__name"),
                    }
                )
        if league_id:
            games = [
                g2 for g2 in (games or []) if not _league_game_is_cross_division_non_external(g2)
            ]
        now_dt = dt.datetime.now()
        is_league_admin = False
        if league_id:
            try:
                is_league_admin = bool(_is_league_admin(int(league_id), int(session["user_id"])))
            except Exception:
                is_league_admin = False
        for g2 in games or []:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            # Hide game pages for future scheduled games that have not started and have no score yet.
            # If starts_at is missing (common for imported games), allow viewing.
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = _sanitize_http_url(
                    _extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
            # Editing is gated to owners or league admins; UI still defaults to read-only unless Edit is clicked.
            try:
                g2["can_edit"] = bool(
                    int(g2.get("user_id") or 0) == int(session["user_id"]) or is_league_admin
                )
            except Exception:
                g2["can_edit"] = bool(is_league_admin)
        games = sort_games_schedule_order(games or [])
        league_page_views = None
        if league_id and is_league_owner:
            league_page_views = {
                "league_id": int(league_id),
                "kind": LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
                "entity_id": 0,
                "count": _get_league_page_view_count(
                    g.db, int(league_id), kind=LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
                ),
            }
        return render_template(
            "schedule.html",
            games=games,
            league_view=bool(league_id),
            divisions=divisions,
            league_teams=league_teams,
            selected_division=selected_division or "",
            selected_team_id=str(team_id_i) if team_id_i is not None else "",
            league_page_views=league_page_views,
        )

    @app.route("/schedule/new", methods=["GET", "POST"])
    def schedule_new():
        r = require_login()
        if r:
            return r
        # Load user's own teams (not external)
        _django_orm, m = _orm_modules()
        my_teams = list(
            m.Team.objects.filter(user_id=int(session["user_id"]), is_external=False)
            .order_by("name")
            .values("id", "name")
        )
        gt = list(m.GameType.objects.order_by("name").values("id", "name"))
        if request.method == "POST":
            team1_id = int(request.form.get("team1_id") or 0)
            team2_id = int(request.form.get("team2_id") or 0)
            opp_name = request.form.get("opponent_name", "").strip()
            game_type_id = int(request.form.get("game_type_id") or 0)
            starts_at = request.form.get("starts_at", "").strip()
            location = request.form.get("location", "").strip()
            # Validate at least one of the teams belongs to user
            if not team1_id and not team2_id:
                flash("Select at least one of your teams", "error")
                return render_template("schedule_new.html", my_teams=my_teams, game_types=gt)
            # If only one team is selected, create/find external opponent
            if team1_id and not team2_id:
                team2_id = ensure_external_team(session["user_id"], opp_name or "Opponent")
            elif team2_id and not team1_id:
                team1_id = ensure_external_team(session["user_id"], opp_name or "Opponent")
            # Create game
            gid = create_hky_game(
                user_id=session["user_id"],
                team1_id=team1_id,
                team2_id=team2_id,
                game_type_id=game_type_id or None,
                starts_at=to_dt(starts_at),
                location=location or None,
            )
            # If a league is selected, map teams and game into the league
            league_id = session.get("league_id")
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
                        m.LeagueGame.objects.get_or_create(
                            league_id=int(league_id), game_id=int(gid)
                        )
                except Exception:
                    pass
            flash("Game created", "success")
            return redirect(url_for("hky_game_detail", game_id=gid))
        return render_template("schedule_new.html", my_teams=my_teams, game_types=gt)

    @app.post("/hky/games/<int:game_id>/import_shift_stats")
    def hky_game_import_shift_stats(game_id: int):
        r = require_login()
        if r:
            return r
        league_id = session.get("league_id")

        # Load game (owner or via league mapping)
        _django_orm, m = _orm_modules()
        game = (
            m.HkyGame.objects.filter(id=int(game_id))
            .values("id", "user_id", "team1_id", "team2_id", "notes")
            .first()
        )
        if not game:
            flash("Not found", "error")
            return redirect(url_for("schedule"))

        tts_linked = _extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

        # Authorization: only allow edits if owner or league admin/owner.
        is_owner = int(game.get("user_id") or 0) == int(session["user_id"])
        if not is_owner:
            if (
                not league_id
                or not m.LeagueGame.objects.filter(
                    league_id=int(league_id), game_id=int(game_id)
                ).exists()
            ):
                flash("Not found", "error")
                return redirect(url_for("schedule"))
        can_edit = bool(is_owner)
        if league_id and not can_edit:
            can_edit = bool(_is_league_admin(int(league_id), int(session["user_id"])))
        if not can_edit:
            flash("You do not have permission to import stats for this game.", "error")
            return redirect(url_for("hky_game_detail", game_id=game_id))

        ps_file = request.files.get("player_stats_csv")
        if not ps_file or not ps_file.filename:
            flash("Select a player_stats.csv file to import.", "error")
            return redirect(url_for("hky_game_detail", game_id=game_id))

        try:
            ps_text = ps_file.stream.read().decode("utf-8", errors="replace")
        except Exception:
            flash("Failed to read uploaded player_stats.csv", "error")
            return redirect(url_for("hky_game_detail", game_id=game_id))

        try:
            parsed_rows = parse_shift_stats_player_stats_csv(ps_text)
        except Exception as e:
            flash(f"Failed to parse player_stats.csv: {e}", "error")
            return redirect(url_for("hky_game_detail", game_id=game_id))

        # Load players for both teams so we can map by jersey/name.
        owner_user_id = int(game.get("user_id") or 0)
        players = list(
            m.Player.objects.filter(
                user_id=int(owner_user_id),
                team_id__in=[int(game["team1_id"]), int(game["team2_id"])],
            ).values("id", "team_id", "name", "jersey_number")
        )

        players_by_team: dict[int, list[dict]] = {}
        jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
        name_to_player_ids: dict[tuple[int, str], list[int]] = {}
        player_team_by_id: dict[int, int] = {}

        for p in players:
            team_id = int(p["team_id"])
            player_team_by_id[int(p["id"])] = team_id
            players_by_team.setdefault(team_id, []).append(p)
            j = normalize_jersey_number(p.get("jersey_number"))
            if j:
                jersey_to_player_ids.setdefault((team_id, j), []).append(int(p["id"]))
            nm = normalize_player_name(p.get("name") or "")
            if nm:
                name_to_player_ids.setdefault((team_id, nm), []).append(int(p["id"]))

        def _resolve_player_id(jersey_norm: Optional[str], name_norm: str) -> Optional[int]:
            candidates: list[tuple[int, int]] = []  # (team_id, player_id)
            for team_id in (int(game["team1_id"]), int(game["team2_id"])):
                if jersey_norm:
                    for pid in jersey_to_player_ids.get((team_id, jersey_norm), []):
                        candidates.append((team_id, pid))
            if len(candidates) == 1:
                return candidates[0][1]

            # Fall back to name match within teams (exact normalized)
            candidates = []
            for team_id in (int(game["team1_id"]), int(game["team2_id"])):
                for pid in name_to_player_ids.get((team_id, name_norm), []):
                    candidates.append((team_id, pid))
            if len(candidates) == 1:
                return candidates[0][1]

            # Jersey match + fuzzy name tie-breaker
            if jersey_norm:
                fuzzy: list[int] = []
                for team_id in (int(game["team1_id"]), int(game["team2_id"])):
                    for pid in jersey_to_player_ids.get((team_id, jersey_norm), []):
                        pl = next(
                            (x for x in players_by_team.get(team_id, []) if int(x["id"]) == pid),
                            None,
                        )
                        if not pl:
                            continue
                        n2 = normalize_player_name(pl.get("name") or "")
                        if n2 and (n2 in name_norm or name_norm in n2):
                            fuzzy.append(pid)
                if len(fuzzy) == 1:
                    return fuzzy[0]

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
            "toi_seconds",
            "shifts",
            "video_toi_seconds",
            "sb_avg_shift_seconds",
            "sb_median_shift_seconds",
            "sb_longest_shift_seconds",
            "sb_shortest_shift_seconds",
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

            # Track import time
            m.HkyGame.objects.filter(id=int(game_id)).update(stats_imported_at=now)

        if unmatched:
            flash(
                f"Imported stats for {imported} player(s). Unmatched: {', '.join([u for u in unmatched if u])}",
                "error",
            )
        else:
            flash(f"Imported stats for {imported} player(s).", "success")
        return_to = _safe_return_to_url(request.args.get("return_to"), default="/schedule")
        return redirect(url_for("hky_game_detail", game_id=game_id, return_to=return_to))

    @app.route("/hky/games/<int:game_id>", methods=["GET", "POST"])
    def hky_game_detail(game_id: int):
        r = require_login()
        if r:
            return r
        league_id = session.get("league_id")
        league_owner_user_id: Optional[int] = None
        is_league_owner = False
        if league_id:
            league_owner_user_id = _get_league_owner_user_id(g.db, int(league_id))
            is_league_owner = bool(
                league_owner_user_id is not None
                and int(league_owner_user_id) == int(session["user_id"])
            )
        _django_orm, m = _orm_modules()
        session_uid = int(session["user_id"])  # type: ignore[arg-type]
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
            flash("Not found", "error")
            return redirect(url_for("schedule"))
        try:
            game["game_video_url"] = _sanitize_http_url(
                _extract_game_video_url_from_notes(game.get("notes"))
            )
        except Exception:
            game["game_video_url"] = None
        is_owner = int(game.get("user_id") or 0) == int(session["user_id"])
        if league_id and not is_owner and _league_game_is_cross_division_non_external(game):
            return ("Not found", 404)
        now_dt = dt.datetime.now()
        sdt = game.get("starts_at")
        started = False
        if sdt is not None:
            try:
                started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
            except Exception:
                started = False
        has_score = (
            (game.get("team1_score") is not None)
            or (game.get("team2_score") is not None)
            or bool(game.get("is_final"))
        )
        can_view_summary = bool(has_score or (sdt is None) or started)
        if not can_view_summary:
            return ("Not found", 404)
        if league_id:
            _record_league_page_view(
                g.db,
                int(league_id),
                kind=LEAGUE_PAGE_VIEW_KIND_GAME,
                entity_id=int(game_id),
                viewer_user_id=int(session["user_id"]),
                league_owner_user_id=league_owner_user_id,
            )
        # is_owner already computed above
        tts_linked = _extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

        return_to = _safe_return_to_url(request.args.get("return_to"), default="/schedule")

        # Authorization: editing requires ownership or league admin/owner.
        can_edit = bool(is_owner)
        if league_id and not can_edit:
            try:
                can_edit = bool(_is_league_admin(int(league_id), int(session["user_id"])))
            except Exception:
                can_edit = False
        edit_mode = bool(can_edit and (request.args.get("edit") == "1" or request.method == "POST"))

        # Load players from both teams (league view must not require ownership)
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
        team1_skaters, team1_goalies, team1_hc, team1_ac = split_roster(team1_players or [])
        team2_skaters, team2_goalies, team2_hc, team2_ac = split_roster(team2_players or [])
        team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
        team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
        stats_by_pid = {r["player_id"]: r for r in stats_rows}
        period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}

        events_headers: list[str] = []
        events_rows: list[dict[str, str]] = []
        events_meta: Optional[dict[str, Any]] = None
        try:
            qs = (
                m.HkyGameEventRow.objects.filter(game_id=int(game_id))
                .select_related("event_type")
                .order_by("period", "game_seconds", "id")
            )
            qs = qs.exclude(
                import_key__in=m.HkyGameEventSuppression.objects.filter(
                    game_id=int(game_id)
                ).values_list("import_key", flat=True)
            )
            raw_rows = list(
                qs.values(
                    "event_type__name",
                    "event_id",
                    "source",
                    "team_raw",
                    "team_side",
                    "for_against",
                    "team_rel",
                    "period",
                    "game_time",
                    "video_time",
                    "game_seconds",
                    "game_seconds_end",
                    "video_seconds",
                    "details",
                    "attributed_players",
                    "attributed_jerseys",
                    "on_ice_players",
                    "on_ice_players_home",
                    "on_ice_players_away",
                    "created_at",
                    "updated_at",
                )
            )
            if raw_rows:
                events_headers = [
                    "Event Type",
                    "Event ID",
                    "Source",
                    "Team Raw",
                    "Team Side",
                    "For/Against",
                    "Team Rel",
                    "Period",
                    "Game Time",
                    "Video Time",
                    "Game Seconds",
                    "Game Seconds End",
                    "Video Seconds",
                    "Details",
                    "Attributed Players",
                    "Attributed Jerseys",
                    "On-Ice Players",
                    "On-Ice Players (Home)",
                    "On-Ice Players (Away)",
                ]
                max_ts: Optional[dt.datetime] = None
                for r in raw_rows:
                    ts = r.get("updated_at") or r.get("created_at")
                    if isinstance(ts, dt.datetime):
                        max_ts = ts if max_ts is None else max(max_ts, ts)

                    events_rows.append(
                        {
                            "Event Type": str(r.get("event_type__name") or "").strip(),
                            "Event ID": (
                                "" if r.get("event_id") is None else str(int(r["event_id"]))
                            ),
                            "Source": str(r.get("source") or "").strip(),
                            "Team Raw": str(r.get("team_raw") or "").strip(),
                            "Team Side": str(r.get("team_side") or "").strip(),
                            "For/Against": str(r.get("for_against") or "").strip(),
                            "Team Rel": str(r.get("team_rel") or "").strip(),
                            "Period": "" if r.get("period") is None else str(int(r["period"])),
                            "Game Time": str(r.get("game_time") or "").strip(),
                            "Video Time": str(r.get("video_time") or "").strip(),
                            "Game Seconds": (
                                "" if r.get("game_seconds") is None else str(int(r["game_seconds"]))
                            ),
                            "Game Seconds End": (
                                ""
                                if r.get("game_seconds_end") is None
                                else str(int(r["game_seconds_end"]))
                            ),
                            "Video Seconds": (
                                ""
                                if r.get("video_seconds") is None
                                else str(int(r["video_seconds"]))
                            ),
                            "Details": str(r.get("details") or "").strip(),
                            "Attributed Players": str(r.get("attributed_players") or "").strip(),
                            "Attributed Jerseys": str(r.get("attributed_jerseys") or "").strip(),
                            "On-Ice Players": str(r.get("on_ice_players") or "").strip(),
                            "On-Ice Players (Home)": str(
                                r.get("on_ice_players_home") or ""
                            ).strip(),
                            "On-Ice Players (Away)": str(
                                r.get("on_ice_players_away") or ""
                            ).strip(),
                        }
                    )
                events_meta = {
                    "source_label": "db",
                    "updated_at": max_ts,
                    "count": len(events_rows),
                }
        except Exception:
            events_headers, events_rows, events_meta = [], [], None

        try:
            events_headers, events_rows = normalize_game_events_csv(events_headers, events_rows)
        except Exception:
            pass
        events_rows = filter_events_rows_prefer_timetoscore_for_goal_assist(
            events_rows, tts_linked=tts_linked
        )
        try:
            events_headers, events_rows = normalize_events_video_time_for_display(
                events_headers, events_rows
            )
            events_headers, events_rows = filter_events_headers_drop_empty_on_ice_split(
                events_headers, events_rows
            )
            events_rows = sort_events_rows_default(events_rows)
        except Exception:
            pass
        if events_meta is not None:
            try:
                events_meta["count"] = len(events_rows)
                events_meta["sources"] = summarize_event_sources(
                    events_rows, fallback_source_label=str(events_meta.get("source_label") or "")
                )
            except Exception:
                pass

        scoring_by_period_rows = compute_team_scoring_by_period_from_events(
            events_rows, tts_linked=tts_linked
        )
        try:
            game_event_stats_rows = compute_game_event_stats_by_side(events_rows)
        except Exception:
            game_event_stats_rows = []

        imported_player_stats_csv_text: Optional[str] = None
        player_stats_import_meta: Optional[dict[str, Any]] = None

        (
            game_player_stats_columns,
            player_stats_cells_by_pid,
            player_stats_cell_conflicts_by_pid,
            player_stats_import_warning,
        ) = build_game_player_stats_table(
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
                    g = _parse_int_from_cell_text(cells.get("goals", ""))
                    a = _parse_int_from_cell_text(cells.get("assists", ""))
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
                sorted_rows = sort_players_table_default(rows)
                return [
                    by_pid[int(r["player_id"])]
                    for r in sorted_rows
                    if int(r["player_id"]) in by_pid
                ]

            team1_skaters_sorted = _sort_players_for_game(team1_skaters_sorted)
            team2_skaters_sorted = _sort_players_for_game(team2_skaters_sorted)
        except Exception:
            team1_skaters_sorted = list(team1_skaters)
            team2_skaters_sorted = list(team2_skaters)

        if request.method == "POST" and not edit_mode:
            flash("You do not have permission to edit this game in the selected league.", "error")
            return redirect(url_for("hky_game_detail", game_id=game_id, return_to=return_to))

        if request.method == "POST" and edit_mode:
            # Update game meta and scores
            loc = request.form.get("location", "").strip()
            starts_at = request.form.get("starts_at", "").strip()
            t1_score = request.form.get("team1_score")
            t2_score = request.form.get("team2_score")
            is_final = bool(request.form.get("is_final"))
            from django.db import transaction

            starts_at_dt = to_dt(starts_at)
            updates = {
                "location": loc or None,
                "starts_at": starts_at_dt,
                "team1_score": int(t1_score) if (t1_score or "").strip() else None,
                "team2_score": int(t2_score) if (t2_score or "").strip() else None,
                "is_final": bool(is_final),
                "updated_at": dt.datetime.now(),
            }

            # Upsert player stats
            def _collect(prefix: str, pid: int) -> dict:
                def _ival(name: str) -> Optional[int]:
                    v = request.form.get(f"{prefix}_{name}_{pid}")
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
                        m.PlayerStat.objects.filter(id=ps.id).update(
                            **{c: vals.get(c) for c in cols}
                        )
            flash("Game updated", "success")
            return redirect(url_for("hky_game_detail", game_id=game_id, return_to=return_to))

        return render_template(
            "hky_game_detail.html",
            game=game,
            team1_roster=team1_roster,
            team2_roster=team2_roster,
            team1_players=team1_skaters_sorted,
            team2_players=team2_skaters_sorted,
            stats_by_pid=stats_by_pid,
            period_stats_by_pid=period_stats_by_pid,
            editable=bool(edit_mode),
            can_edit=bool(can_edit),
            edit_mode=bool(edit_mode),
            back_url=return_to,
            return_to=return_to,
            events_headers=events_headers,
            events_rows=events_rows,
            events_meta=events_meta,
            scoring_by_period_rows=scoring_by_period_rows,
            game_event_stats_rows=game_event_stats_rows,
            user_video_clip_len_s=get_user_video_clip_len_s(g.db, int(session.get("user_id") or 0)),
            user_is_logged_in=True,
            game_player_stats_columns=game_player_stats_columns,
            player_stats_cells_by_pid=player_stats_cells_by_pid,
            player_stats_cell_conflicts_by_pid=player_stats_cell_conflicts_by_pid,
            player_stats_import_meta=player_stats_import_meta,
            player_stats_import_warning=player_stats_import_warning,
            league_page_views=(
                {
                    "league_id": int(league_id),
                    "kind": LEAGUE_PAGE_VIEW_KIND_GAME,
                    "entity_id": int(game_id),
                    "count": _get_league_page_view_count(
                        g.db,
                        int(league_id),
                        kind=LEAGUE_PAGE_VIEW_KIND_GAME,
                        entity_id=int(game_id),
                    ),
                }
                if (league_id and is_league_owner)
                else None
            ),
        )

    @app.route("/game_types", methods=["GET", "POST"])
    def game_types():
        r = require_login()
        if r:
            return r
        _django_orm, m = _orm_modules()
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            if name:
                try:
                    from django.db import IntegrityError, transaction

                    with transaction.atomic():
                        m.GameType.objects.create(name=name, is_default=False)
                    flash("Game type added", "success")
                except IntegrityError:
                    flash("Failed to add game type (may already exist)", "error")
            return redirect(url_for("game_types"))
        rows = list(m.GameType.objects.order_by("name").values("id", "name", "is_default"))
        return render_template("game_types.html", game_types=rows)

    return app


def init_db():
    try:
        from tools.webapp import django_orm
    except Exception:  # pragma: no cover
        import django_orm  # type: ignore

    django_orm.ensure_schema()
    django_orm.ensure_bootstrap_data(default_admin_password_hash=generate_password_hash("admin"))


def get_user_by_email(email: str) -> Optional[dict]:
    _django_orm, m = _orm_modules()
    return m.User.objects.filter(email=str(email or "")).values().first()


def create_user(email: str, password: str, name: str) -> int:
    pw = generate_password_hash(password)
    _django_orm, m = _orm_modules()
    u = m.User.objects.create(
        email=str(email or ""),
        password_hash=pw,
        name=str(name or ""),
        created_at=dt.datetime.now(),
        default_league_id=None,
        video_clip_len_s=None,
    )
    return int(u.id)


def create_game(user_id: int, name: str, email: str):
    # Create dedicated dir: <watch_root>/<user_id>_<timestamp>_<rand>
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    token = secrets.token_hex(4)
    watch_root = os.environ.get("HM_WATCH_ROOT", WATCH_ROOT)
    d = Path(watch_root) / f"game_{user_id}_{ts}_{token}"
    d.mkdir(parents=True, exist_ok=True)
    # Create meta with user email
    try:
        (d / ".dirwatch_meta.json").write_text(
            f'{{"user_email":"{email}","created":"{dt.datetime.now().isoformat()}"}}\n'
        )
    except Exception:
        pass
    _django_orm, m = _orm_modules()
    g = m.Game.objects.create(
        user_id=int(user_id),
        name=str(name or ""),
        dir_path=str(d),
        status="new",
        created_at=dt.datetime.now(),
    )
    return int(g.id), str(d)


def read_dirwatch_state():
    state_path = Path("/var/lib/dirwatcher/state.json")
    try:
        import json

        return json.loads(state_path.read_text())
    except Exception:
        return {"processed": {}, "active": {}}


def send_email(to_addr: str, subject: str, body: str, from_addr: Optional[str] = None) -> None:
    # Use system sendmail preferred
    from_addr = from_addr or ("no-reply@" + os.uname().nodename)
    msg = (
        f"From: {from_addr}\nTo: {to_addr}\nSubject: {subject}\n"
        f"Content-Type: text/plain; charset=utf-8\n\n{body}\n"
    )
    import shutil as _sh
    import subprocess as _sp

    sendmail = _sh.which("sendmail")
    if sendmail:
        try:
            _sp.run([sendmail, "-t"], input=msg.encode("utf-8"), check=True)
            return
        except Exception:
            pass
    # no-op if email fails
    return


# ---------------------------
# Helpers for Teams/Players/Hockey Games
# ---------------------------


def create_team(user_id: int, name: str, is_external: bool = False) -> int:
    _django_orm, m = _orm_modules()
    t = m.Team.objects.create(
        user_id=int(user_id),
        name=str(name or ""),
        is_external=bool(is_external),
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(t.id)


def get_team(team_id: int, user_id: int) -> Optional[dict]:
    _django_orm, m = _orm_modules()
    return m.Team.objects.filter(id=int(team_id), user_id=int(user_id)).values().first()


def save_team_logo(file_storage, team_id: int) -> Path:
    # Save under instance/uploads/team_logos
    uploads = INSTANCE_DIR / "uploads" / "team_logos"
    uploads.mkdir(parents=True, exist_ok=True)
    # sanitize filename
    fname = Path(
        str(getattr(file_storage, "filename", None) or getattr(file_storage, "name", "") or "")
    ).name
    if not fname:
        fname = "logo"
    # prefix with team id and timestamp
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    dest = uploads / f"team{team_id}_{ts}_{fname}"
    if hasattr(file_storage, "save"):
        file_storage.save(dest)
    elif hasattr(file_storage, "chunks"):
        with dest.open("wb") as out:
            for chunk in file_storage.chunks():  # type: ignore[attr-defined]
                out.write(chunk)
    else:
        data = file_storage.read() if hasattr(file_storage, "read") else b""
        dest.write_bytes(data or b"")
    return dest


def ensure_external_team(user_id: int, name: str) -> int:
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
        return t

    name = _norm_team_name(name)
    _django_orm, m = _orm_modules()
    t, _created = m.Team.objects.get_or_create(
        user_id=int(user_id),
        name=str(name),
        defaults={
            "is_external": True,
            "logo_path": None,
            "created_at": dt.datetime.now(),
            "updated_at": None,
        },
    )
    return int(t.id)


def create_hky_game(
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[dt.datetime],
    location: Optional[str],
) -> int:
    _django_orm, m = _orm_modules()
    now = dt.datetime.now()
    g = m.HkyGame.objects.create(
        user_id=int(user_id),
        team1_id=int(team1_id),
        team2_id=int(team2_id),
        game_type_id=int(game_type_id) if game_type_id is not None else None,
        starts_at=starts_at,
        location=location,
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    return int(g.id)


def parse_dt_or_none(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    # Accept YYYY-MM-DD or YYYY-MM-DDTHH:MM
    try:
        if "T" in s:
            return dt.datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
        return dt.datetime.fromisoformat(s + "T00:00").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _safe_return_to_url(value: Optional[str], *, default: str) -> str:
    """
    Only allow returning to same-site relative URLs.
    Accepts paths like "/teams/123" or "/schedule?division=..." and rejects external URLs.
    """
    if not value:
        return str(default)
    s = str(value).strip()
    if not s:
        return str(default)
    if not s.startswith("/"):
        return str(default)
    if s.startswith("//"):
        return str(default)
    return s


def _sanitize_http_url(value: Optional[str]) -> Optional[str]:
    """
    Allow only http(s) URLs for external links (prevents javascript: etc).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    sl = s.lower()
    if sl.startswith("http://") or sl.startswith("https://"):
        return s
    return None


def _extract_game_video_url_from_notes(notes: Optional[str]) -> Optional[str]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            for k in ("game_video_url", "game_video", "video_url"):
                v = d.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass
    m = re.search(r"(?:^|[\\s|,;])game_video_url\\s*=\\s*([^\\s|,;]+)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip()
    m = re.search(r"(?:^|[\\s|,;])game_video\\s*=\\s*([^\\s|,;]+)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip()
    return None


def _extract_timetoscore_game_id_from_notes(notes: Optional[str]) -> Optional[int]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            v = d.get("timetoscore_game_id")
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    return None
    except Exception:
        pass
    m = re.search(r"(?:^|[\\s|,;])game_id\\s*=\\s*(\\d+)", s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\"timetoscore_game_id\"\\s*:\\s*(\\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def sort_games_schedule_order(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Stable schedule ordering:
      - games with known start datetime first
      - then by start date
      - then by start time
      - then league sort_order (if present)
      - then created_at
    """

    def _int_or_big(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 2147483647

    def _key(idx_game: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        idx, g = idx_game
        sdt = to_dt(g.get("starts_at"))
        has_dt = sdt is not None
        # Put unknown starts_at at the end but keep a deterministic order using sort_order/created_at.
        dt_key = sdt if sdt else dt.datetime.max
        so = _int_or_big(g.get("sort_order"))
        created = to_dt(g.get("created_at")) or dt.datetime.max
        return (0 if has_dt else 1, dt_key.date(), dt_key.time(), so, created, idx)

    return [g for _idx, g in sorted(list(enumerate(games or [])), key=_key)]


def is_external_division_name(name: Any) -> bool:
    return str(name or "").strip().casefold().startswith("external")


def division_sort_key(division_name: Any) -> tuple:
    """
    Sort key for league division names.

    Primary ordering:
      - age (10U/12AA/etc) ascending
      - level ordering: AAA, AA, A, BB, B, (everything else)
      - non-External before External (within the same age/level)
      - then lexicographic as a stable tie-breaker
    """
    raw = str(division_name or "").strip()
    if not raw:
        return (999, 99, 1, "", "")

    external = is_external_division_name(raw)
    base = raw
    if external:
        base = re.sub(r"(?i)^external\s*", "", base).strip()

    age = parse_age_from_division_name(base)
    age_key = int(age) if age is not None else 999

    m = re.search(
        r"(?i)(?:^|\b)\d{1,2}(?:u)?\s*(AAA|AA|BB|A|B)(?=\b|\s|$|[-–—])",
        base,
    )
    level_token = str(m.group(1)).upper() if m else ""
    level_rank = {"AAA": 0, "AA": 1, "A": 2, "BB": 3, "B": 4}.get(level_token, 99)

    return (age_key, int(level_rank), 1 if external else 0, base.casefold(), raw.casefold())


def _league_game_is_cross_division_non_external_row(
    game_division_name: Optional[str],
    team1_division_name: Optional[str],
    team2_division_name: Optional[str],
) -> bool:
    """
    True when both teams have known, non-External league divisions and they differ.
    """
    d1 = str(team1_division_name or "").strip()
    d2 = str(team2_division_name or "").strip()
    if not d1 or not d2:
        return False
    if is_external_division_name(d1) or is_external_division_name(d2):
        return False
    return d1 != d2


def recompute_league_mhr_ratings(
    db_conn, league_id: int, *, max_goal_diff: int = 7, min_games: int = 2
) -> dict[int, dict[str, Any]]:
    """
    Recompute and persist MyHockeyRankings-like ratings for teams in a league.
    Stores values on `league_teams` as:
      - mhr_rating (NULL if games < min_games)
      - mhr_agd, mhr_sched, mhr_games, mhr_updated_at
    """
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db import transaction

    league_team_rows = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id))
        .select_related("team")
        .values("team_id", "division_name", "team__name")
    )

    team_age: dict[int, Optional[int]] = {}
    team_level: dict[int, Optional[str]] = {}
    for r in league_team_rows:
        try:
            tid = int(r.get("team_id"))
        except Exception:
            continue
        dn = str(r.get("division_name") or "").strip()
        age = parse_age_from_division_name(dn)
        if age is None:
            age = parse_age_from_division_name(str(r.get("team__name") or "").strip())
        team_age[tid] = age
        lvl = parse_level_from_division_name(dn)
        if lvl is None:
            lvl = parse_level_from_division_name(str(r.get("team__name") or "").strip())
        team_level[tid] = lvl

    games: list[GameScore] = []
    for team1_id, team2_id, team1_score, team2_score in m.LeagueGame.objects.filter(
        league_id=int(league_id),
        game__team1_score__isnull=False,
        game__team2_score__isnull=False,
    ).values_list(
        "game__team1_id",
        "game__team2_id",
        "game__team1_score",
        "game__team2_score",
    ):
        try:
            games.append(
                GameScore(
                    team1_id=int(team1_id),
                    team2_id=int(team2_id),
                    team1_score=int(team1_score),
                    team2_score=int(team2_score),
                )
            )
        except Exception:
            continue

    # Eligibility rule: A team cannot have an MHR rating unless it has played at least one game
    # against another team in the same age group (age + level when known).
    age_group_games: dict[int, int] = {}
    for g in games:
        a = int(g.team1_id)
        b = int(g.team2_id)
        age_a = team_age.get(a)
        age_b = team_age.get(b)
        if age_a is None or age_b is None:
            continue
        if int(age_a) != int(age_b):
            continue
        lvl_a = str(team_level.get(a) or "").strip().upper() or None
        lvl_b = str(team_level.get(b) or "").strip().upper() or None
        if lvl_a is not None and lvl_b is not None and lvl_a != lvl_b:
            continue
        age_group_games[a] = int(age_group_games.get(a, 0)) + 1
        age_group_games[b] = int(age_group_games.get(b, 0)) + 1

    # Ignore cross-age games when computing ratings (no cross-age coupling).
    games_same_age = filter_games_ignore_cross_age(games, team_age=team_age)

    computed = compute_mhr_like_ratings(
        games=games_same_age,
        max_goal_diff=int(max_goal_diff),
        min_games_for_rating=int(min_games),
    )
    # Normalize per disconnected component: top team in each independent group becomes 99.9.
    computed_norm = scale_ratings_to_0_99_9_by_component(
        computed, games=games_same_age, key="rating"
    )

    now = dt.datetime.now()
    # Persist for all league teams (set NULL when unknown/insufficient).
    league_team_objs = list(m.LeagueTeam.objects.filter(league_id=int(league_id)))
    for lt in league_team_objs:
        tid = int(lt.team_id)
        row_norm = computed_norm.get(tid) or {}
        rating = row_norm.get("rating")
        # Use raw AGD/SCHED/GAMES from the base computation (before shifting).
        row_base = computed.get(tid) or {}
        if int(age_group_games.get(tid, 0)) <= 0:
            lt.mhr_rating = None
            lt.mhr_agd = None
            lt.mhr_sched = None
            lt.mhr_games = 0
        else:
            lt.mhr_rating = float(rating) if rating is not None else None
            lt.mhr_agd = float(row_base.get("agd")) if row_base.get("agd") is not None else None
            lt.mhr_sched = (
                float(row_base.get("sched")) if row_base.get("sched") is not None else None
            )
            lt.mhr_games = int(row_base.get("games")) if row_base.get("games") is not None else 0
        lt.mhr_updated_at = now
    if league_team_objs:
        with transaction.atomic():
            m.LeagueTeam.objects.bulk_update(
                league_team_objs,
                ["mhr_rating", "mhr_agd", "mhr_sched", "mhr_games", "mhr_updated_at"],
                batch_size=500,
            )
    return computed_norm


def parse_events_csv(events_csv: str) -> tuple[list[str], list[dict[str, str]]]:
    s = str(events_csv or "").strip()
    if not s:
        return [], []
    s = s.lstrip("\ufeff")
    f = io.StringIO(s)
    reader = csv.DictReader(f)
    headers = [str(h) for h in (reader.fieldnames or []) if h is not None]
    rows: list[dict[str, str]] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        rows.append({h: ("" if row.get(h) is None else str(row.get(h))) for h in headers})
    return headers, rows


def to_csv_text(headers: list[str], rows: list[dict[str, str]]) -> str:
    if not headers:
        return ""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows or []:
        w.writerow({h: ("" if (r.get(h) is None) else str(r.get(h))) for h in headers})
    return out.getvalue()


def sanitize_player_stats_csv_for_storage(player_stats_csv: str) -> str:
    """
    Sanitize game-level player_stats CSV before storage.

    We intentionally drop shift/ice-time and "per game"/"per shift" derived columns so they never
    appear in the web UI and don't accidentally get used for downstream calculations.
    """
    headers, rows = parse_events_csv(player_stats_csv)
    headers, rows = filter_single_game_player_stats_csv(headers, rows)
    return to_csv_text(headers, rows)


def filter_single_game_player_stats_csv(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Single-game player stats tables should not include any per-game normalized columns
    (e.g., 'Shots per Game', 'TOI per Game', 'PPG') because they are redundant for a
    one-game view and make the table unnecessarily wide.
    """

    def _drop_header(h: str) -> bool:
        key = str(h or "").strip().lower()
        if key in {"ppg", "gp"}:
            return True
        if "per game" in key:
            return True
        # Remove all shift/time related fields from the webapp UI.
        if "toi" in key or "ice time" in key:
            return True
        if "shift" in key or "per shift" in key:
            return True
        return False

    kept_headers = [h for h in headers if not _drop_header(h)]
    kept_rows = [{h: r.get(h, "") for h in kept_headers} for r in (rows or [])]
    return kept_headers, kept_rows


def normalize_game_events_csv(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Normalize the event table for display:
      - Ensure 'Event Type' is the leftmost column
      - Drop redundant 'Event Type Raw' (historical) if present
    """

    def _is_event_type_raw(h: str) -> bool:
        return str(h or "").strip().lower() in {"event type raw", "event_type_raw"}

    # Remove raw header if present.
    filtered_headers = [h for h in (headers or []) if not _is_event_type_raw(h)]
    filtered_rows = [{h: r.get(h, "") for h in filtered_headers} for r in (rows or [])]

    # Prefer explicit "Event Type"; fall back to "Event" (common legacy schema).
    event_header = None
    for h in filtered_headers:
        if str(h).strip().lower() == "event type":
            event_header = h
            break
    if event_header is None:
        for h in filtered_headers:
            if str(h).strip().lower() == "event":
                event_header = h
                break

    if event_header is None:
        return filtered_headers, filtered_rows

    # If the CSV uses "Event", rename it to "Event Type" for display.
    if str(event_header).strip().lower() == "event":
        renamed_headers: list[str] = []
        for h in filtered_headers:
            if h == event_header:
                renamed_headers.append("Event Type")
            else:
                renamed_headers.append(h)
        renamed_rows: list[dict[str, str]] = []
        for r in filtered_rows:
            out: dict[str, str] = {}
            for h in filtered_headers:
                if h == event_header:
                    out["Event Type"] = r.get(h, "")
                else:
                    out[h] = r.get(h, "")
            renamed_rows.append(out)
        filtered_headers, filtered_rows = renamed_headers, renamed_rows
        event_header = "Event Type"

    reordered_headers = [event_header] + [h for h in filtered_headers if h != event_header]
    reordered_rows = [{h: r.get(h, "") for h in reordered_headers} for r in filtered_rows]
    return reordered_headers, reordered_rows


def filter_events_headers_drop_empty_on_ice_split(
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    If Home/Away on-ice columns are present but completely empty, drop them from the display table.
    Keeps the raw CSV stored in DB unchanged; this only affects UI rendering.
    """
    if not headers or not rows:
        return headers, rows
    split_cols = ["On-Ice Players (Home)", "On-Ice Players (Away)"]
    present = [c for c in split_cols if c in headers]
    if not present:
        return headers, rows
    keep: set[str] = set(headers)
    for c in present:
        any_nonempty = False
        for r in rows:
            v = str((r or {}).get(c, "") or "").strip()
            if v:
                any_nonempty = True
                break
        if not any_nonempty:
            keep.discard(c)
    if keep == set(headers):
        return headers, rows
    new_headers = [h for h in headers if h in keep]
    new_rows = [
        {h: (r.get(h, "") if isinstance(r, dict) else "") for h in new_headers} for r in rows
    ]
    return new_headers, new_rows


def compute_team_scoring_by_period_from_events(
    events_rows: list[dict[str, str]],
    *,
    tts_linked: bool = False,
) -> list[dict[str, Any]]:
    """
    Compute per-period team GF/GA from stored goal events.

    Notes:
      - Uses event rows (e.g. from hky_game_events.events_csv).
      - Team mapping prefers absolute Home/Away via "Team Side" (or legacy "Team Rel" when it contains Home/Away).
      - "For/Against" is treated as a relative direction, not a synonym for Home/Away.
    Returns rows like:
      {'period': 1, 'team1_gf': 2, 'team1_ga': 1, 'team2_gf': 1, 'team2_ga': 2}
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _is_tts_row(row: dict[str, str]) -> bool:
        toks = _split_sources(row.get("Source") or "")
        return any(_norm(t) == "timetoscore" for t in toks)

    def _team_side_to_team_idx(row: dict[str, str]) -> Optional[int]:
        # Prefer explicit Team Side when available.
        side = _norm(row.get("Team Side") or row.get("TeamSide") or row.get("Side") or "")
        if side in {"home", "team1"}:
            return 1
        if side in {"away", "team2"}:
            return 2

        # Legacy: some tables stored Home/Away in Team Rel or Team Raw.
        tr = _norm(row.get("Team Rel") or row.get("TeamRel") or "")
        if tr in {"home", "team1"}:
            return 1
        if tr in {"away", "team2"}:
            return 2

        # Some simple/older tables use "Team" as Home/Away directly.
        team = _norm(row.get("Team") or "")
        if team in {"home", "team1"}:
            return 1
        if team in {"away", "team2"}:
            return 2

        tr2 = _norm(row.get("Team Raw") or row.get("TeamRaw") or "")
        if tr2 in {"home", "team1"}:
            return 1
        if tr2 in {"away", "team2"}:
            return 2
        return None

    by_period: dict[int, dict[str, int]] = {}
    # If the game is TimeToScore-linked, only count TimeToScore goal rows for scoring attribution.
    # Otherwise, prefer TimeToScore goal rows when they are present (but allow spreadsheet-only games).
    has_tts_goal_rows = False
    if tts_linked:
        has_tts_goal_rows = True
    else:
        for r0 in events_rows or []:
            if not isinstance(r0, dict):
                continue
            if _norm(r0.get("Event Type") or r0.get("Event") or "") != "goal":
                continue
            if _is_tts_row(r0):
                has_tts_goal_rows = True
                break

    for r in events_rows or []:
        if not isinstance(r, dict):
            continue
        et = _norm(r.get("Event Type") or r.get("Event") or "")
        if et != "goal":
            continue
        if has_tts_goal_rows and not _is_tts_row(r):
            continue
        per = _parse_int(r.get("Period"))
        if per is None:
            continue
        team_idx = _team_side_to_team_idx(r)
        if team_idx is None:
            continue
        rec = by_period.setdefault(
            per, {"team1_gf": 0, "team1_ga": 0, "team2_gf": 0, "team2_ga": 0}
        )
        if team_idx == 1:
            rec["team1_gf"] += 1
            rec["team2_ga"] += 1
        else:
            rec["team2_gf"] += 1
            rec["team1_ga"] += 1

    if not by_period:
        return []
    max_period = max(3, max(by_period.keys()))
    out: list[dict[str, Any]] = []
    for p in range(1, max_period + 1):
        rec = by_period.get(p) or {"team1_gf": 0, "team1_ga": 0, "team2_gf": 0, "team2_ga": 0}
        out.append({"period": p, **rec})
    return out


def filter_events_rows_prefer_timetoscore_for_goal_assist(
    events_rows: list[dict[str, str]],
    *,
    tts_linked: bool = False,
) -> list[dict[str, str]]:
    """
    If any TimeToScore Goal/Assist rows exist, drop non-TimeToScore Goal/Assist rows to
    avoid mixing attribution sources.
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _ev_type(r: dict[str, str]) -> str:
        return _norm(r.get("Event Type") or r.get("Event") or "")

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _is_tts_row(r: dict[str, str]) -> bool:
        toks = _split_sources(r.get("Source") or "")
        return any(_norm(t) == "timetoscore" for t in toks)

    if not events_rows:
        return []
    has_tts = any(
        _ev_type(r) in {"goal", "assist"} and _is_tts_row(r)
        for r in events_rows
        if isinstance(r, dict)
    )
    if not has_tts and not tts_linked:
        return list(events_rows)
    return [
        r
        for r in events_rows
        if isinstance(r, dict) and (_ev_type(r) not in {"goal", "assist"} or _is_tts_row(r))
    ]


def filter_events_csv_drop_event_types(csv_text: str, *, drop_types: set[str]) -> str:
    """
    Drop specific event types from an events CSV (case-insensitive match on Event Type/Event).
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    headers, rows = parse_events_csv(csv_text)
    if not headers:
        return csv_text
    kept_rows: list[dict[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        et = _norm(r.get("Event Type") or r.get("Event") or "")
        if et and et in drop_types:
            continue
        kept_rows.append(r)
    return to_csv_text(headers, kept_rows)


def summarize_event_sources(
    events_rows: list[dict[str, str]],
    *,
    fallback_source_label: Optional[str] = None,
) -> list[str]:
    """
    Return a de-duped, order-preserving list of event row sources for UI display.
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip()

    def _split_sources(raw: Any) -> list[str]:
        s = _norm(raw)
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _canon(s: str) -> str:
        sl = s.strip().casefold()
        if sl == "timetoscore":
            return "TimeToScore"
        if sl == "long":
            return "Long"
        if sl == "goals":
            return "Goals"
        if sl == "shift_package":
            return "Shift Package"
        if not s.strip():
            return ""
        return s.strip()

    out: list[str] = []
    seen: set[str] = set()
    for r in events_rows or []:
        if not isinstance(r, dict):
            continue
        src_raw = r.get("Source") or ""
        toks = _split_sources(src_raw)
        if not toks and fallback_source_label:
            toks = _split_sources(str(fallback_source_label))
        for t in toks:
            src = _canon(t)
            if not src:
                continue
            key = src.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(src)
    return out


def merge_events_csv_prefer_timetoscore(
    *,
    existing_csv: str,
    existing_source_label: str,
    incoming_csv: str,
    incoming_source_label: str,
    protected_types: set[str],
) -> tuple[str, str]:
    """
    Merge two events CSVs, preferring TimeToScore rows for protected event types (Goal/Assist/etc).
    Returns: (merged_csv, merged_source_label).
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _ev_type(r: dict[str, str]) -> str:
        return _norm(r.get("Event Type") or r.get("Event") or "")

    def _is_tts_row(r: dict[str, str], fallback_label: str) -> bool:
        src = _norm(r.get("Source") or "")
        if src:
            toks = [t for t in re.split(r"[,+;/\s]+", src) if t]
            return any(t == "timetoscore" for t in toks)
        return str(fallback_label or "").strip().lower().startswith("timetoscore")

    def _key(r: dict[str, str]) -> tuple[str, str, str, str, str]:
        et = _ev_type(r)
        per = str(r.get("Period") or "").strip()
        gs = str(r.get("Game Seconds") or r.get("GameSeconds") or "").strip()
        # Prefer absolute Home/Away when available.
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

    ex_headers, ex_rows = parse_events_csv(existing_csv)
    in_headers, in_rows = parse_events_csv(incoming_csv)

    def _iter_typed(rows: list[dict[str, str]], label: str) -> list[tuple[dict[str, str], str]]:
        out: list[tuple[dict[str, str], str]] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append((r, label))
        return out

    all_rows = _iter_typed(ex_rows, existing_source_label) + _iter_typed(
        in_rows, incoming_source_label
    )
    has_tts_protected = any(
        _ev_type(r) in protected_types and _is_tts_row(r, lbl) for r, lbl in all_rows
    )

    def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = str(row.get(k) or "").strip()
            if v:
                return v
        return ""

    def _overlay_missing_fields(dst: dict[str, str], src: dict[str, str]) -> None:
        for k, ks in (
            ("Event ID", ("Event ID", "EventID")),
            ("Video Time", ("Video Time", "VideoTime")),
            ("Video Seconds", ("Video Seconds", "VideoSeconds", "Video S", "VideoS")),
            ("On-Ice Players", ("On-Ice Players", "OnIce Players", "OnIcePlayers")),
            ("On-Ice Players (Home)", ("On-Ice Players (Home)", "OnIce Players (Home)")),
            ("On-Ice Players (Away)", ("On-Ice Players (Away)", "OnIce Players (Away)")),
        ):
            if str(dst.get(k) or "").strip():
                continue
            v = _first_non_empty(src, ks)
            if v:
                dst[k] = v

    fallback_rows_by_key: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}

    merged_rows: list[dict[str, str]] = []
    merged_by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for r, lbl in all_rows:
        et = _ev_type(r)
        if not et:
            continue
        k = _key(r)

        # If TimeToScore rows exist for protected types, keep them as authoritative rows, but
        # still remember non-TTS rows so we can copy missing clip/on-ice metadata onto the kept
        # rows (Video Time/Seconds, On-Ice Players, etc).
        if has_tts_protected and et in protected_types and not _is_tts_row(r, lbl):
            fallback_rows_by_key.setdefault(k, []).append(r)
            continue

        prev = merged_by_key.get(k)
        if prev is not None:
            _overlay_missing_fields(prev, r)
            continue

        rr = dict(r)
        merged_rows.append(rr)
        merged_by_key[k] = rr

    # Overlay missing fields for protected TimeToScore rows from any skipped non-TTS rows.
    for k, fb_rows in fallback_rows_by_key.items():
        dst = merged_by_key.get(k)
        if dst is None:
            continue
        for fb in fb_rows:
            _overlay_missing_fields(dst, fb)

    merged_headers = list(ex_headers or [])
    for h in in_headers or []:
        if h not in merged_headers:
            merged_headers.append(h)

    # Prefer a TimeToScore label if either input is TimeToScore.
    merged_source = (
        existing_source_label
        if str(existing_source_label or "").strip().lower().startswith("timetoscore")
        else (
            incoming_source_label
            if str(incoming_source_label or "").strip().lower().startswith("timetoscore")
            else (existing_source_label or incoming_source_label)
        )
    )

    return to_csv_text(merged_headers, merged_rows), str(merged_source or "")


def enrich_timetoscore_goals_with_long_video_times(
    *,
    existing_headers: list[str],
    existing_rows: list[dict[str, str]],
    incoming_headers: list[str],
    incoming_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    For TimeToScore-linked games, treat TimeToScore goal attribution as authoritative, but
    copy Video Time/Seconds and On-Ice player lists from matching spreadsheet-derived Goal events
    (same team + period + game time).

    - Only enriches existing rows whose Source contains "timetoscore"
    - Never adds new goal events; long-only goals are ignored
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _has_source(row: dict[str, str], token: str) -> bool:
        return any(_norm(t) == _norm(token) for t in _split_sources(row.get("Source") or ""))

    def _add_source(row: dict[str, str], token: str) -> None:
        toks = _split_sources(row.get("Source") or "")
        toks_cf = {_norm(t) for t in toks}
        if _norm(token) not in toks_cf:
            toks.append(str(token))
        row["Source"] = ",".join([t for t in toks if t])

    def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = str(row.get(k) or "").strip()
            if v:
                return v
        return ""

    def _set_if_blank(row: dict[str, str], key: str, value: str) -> bool:
        if str(row.get(key) or "").strip():
            return False
        if not str(value or "").strip():
            return False
        row[key] = str(value)
        return True

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _norm_side(row: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Side",
            "Team Rel",
            "TeamRel",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _ev_type(row: dict[str, str]) -> str:
        return _norm(row.get("Event Type") or row.get("Event") or "")

    def _period(row: dict[str, str]) -> Optional[int]:
        p = _parse_int(row.get("Period"))
        return p if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, str]) -> Optional[int]:
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is not None:
            return gs
        return parse_duration_seconds(
            row.get("Game Time") or row.get("GameTime") or row.get("Time")
        )

    def _video_seconds(row: dict[str, str]) -> Optional[int]:
        vs = _parse_int(row.get("Video Seconds") or row.get("VideoSeconds"))
        if vs is not None:
            return vs
        return parse_duration_seconds(row.get("Video Time") or row.get("VideoTime"))

    def _video_time(row: dict[str, str]) -> str:
        return str(row.get("Video Time") or row.get("VideoTime") or "").strip()

    # Lookups from incoming Goal rows.
    long_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
    on_ice_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
    on_ice_score_by_key: dict[tuple[int, str, int], int] = {}
    event_id_by_key: dict[tuple[int, str, int], tuple[int, str]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        if _ev_type(r) != "goal":
            continue
        per = _period(r)
        side = _norm_side(r)
        gs = _game_seconds(r)
        if per is None or side is None or gs is None:
            continue

        # Prefer a "goals" goal-row event id as the canonical id to copy onto TimeToScore goals
        # (long-sheet goal rows also have ids but are less stable).
        eid = _first_non_empty(r, ("Event ID", "EventID"))
        if eid:
            prio = 0 if _has_source(r, "goals") else (1 if _has_source(r, "long") else 2)
            prev = event_id_by_key.get((per, side, int(gs)))
            if prev is None or int(prio) < int(prev[0]):
                event_id_by_key[(per, side, int(gs))] = (int(prio), str(eid))

        # Video time enrichment prefers long-sheet rows.
        if _has_source(r, "long"):
            vs = _video_seconds(r)
            vt = _video_time(r)
            if vs is not None or vt:
                long_by_key[(per, side, int(gs))] = r

        # On-ice enrichment can come from any incoming Goal rows that carry those columns.
        home_on_ice = _first_non_empty(r, ("On-Ice Players (Home)", "OnIce Players (Home)"))
        away_on_ice = _first_non_empty(r, ("On-Ice Players (Away)", "OnIce Players (Away)"))
        legacy_on_ice = _first_non_empty(r, ("On-Ice Players", "OnIce Players"))
        score = (2 if home_on_ice else 0) + (2 if away_on_ice else 0) + (1 if legacy_on_ice else 0)
        if score > 0:
            k = (per, side, int(gs))
            if score > int(on_ice_score_by_key.get(k, 0)):
                on_ice_score_by_key[k] = int(score)
                on_ice_by_key[k] = r

    # Ensure destination headers include video fields.
    out_headers = list(existing_headers or [])
    for h in (
        "Event ID",
        "Video Time",
        "Video Seconds",
        "On-Ice Players",
        "On-Ice Players (Home)",
        "On-Ice Players (Away)",
    ):
        if h not in out_headers:
            out_headers.append(h)

    out_rows: list[dict[str, str]] = []
    for r in existing_rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        if _ev_type(rr) == "goal" and _has_source(rr, "timetoscore"):
            per = _period(rr)
            side = _norm_side(rr)
            gs = _game_seconds(rr)
            if per is not None and side is not None and gs is not None:
                k = (per, side, int(gs))
                match = long_by_key.get(k)
                if match is not None:
                    vs = _video_seconds(match)
                    vt = _video_time(match)
                    if vs is not None:
                        rr["Video Seconds"] = str(int(vs))
                    if vt:
                        rr["Video Time"] = vt
                    _add_source(rr, "long")

                if not str(rr.get("Event ID") or "").strip():
                    eid = event_id_by_key.get(k)
                    if eid is not None and str(eid[1]).strip():
                        rr["Event ID"] = str(eid[1]).strip()

                match_on_ice = on_ice_by_key.get(k)
                if match_on_ice is not None:
                    copied = False
                    copied |= _set_if_blank(
                        rr,
                        "On-Ice Players (Home)",
                        _first_non_empty(
                            match_on_ice, ("On-Ice Players (Home)", "OnIce Players (Home)")
                        ),
                    )
                    copied |= _set_if_blank(
                        rr,
                        "On-Ice Players (Away)",
                        _first_non_empty(
                            match_on_ice, ("On-Ice Players (Away)", "OnIce Players (Away)")
                        ),
                    )
                    copied |= _set_if_blank(
                        rr,
                        "On-Ice Players",
                        _first_non_empty(match_on_ice, ("On-Ice Players", "OnIce Players")),
                    )
                    if copied:
                        _add_source(rr, "shift_package")
        out_rows.append(rr)

    return out_headers, out_rows


def enrich_timetoscore_penalties_with_video_times(
    *,
    existing_headers: list[str],
    existing_rows: list[dict[str, str]],
    incoming_headers: list[str],
    incoming_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    For TimeToScore-linked games, keep TimeToScore penalty events as authoritative, but
    copy Video Time/Seconds from matching penalty rows found in the incoming spreadsheet
    events CSV (which can map scoreboard time -> video time using shift sync).

    This is what makes penalty icons in the game timeline clickable (the UI requires
    Video Time/Seconds in the row to open the video clip).
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _has_source(row: dict[str, str], token: str) -> bool:
        return any(_norm(t) == _norm(token) for t in _split_sources(row.get("Source") or ""))

    def _add_source(row: dict[str, str], token: str) -> None:
        toks = _split_sources(row.get("Source") or "")
        toks_cf = {_norm(t) for t in toks}
        if _norm(token) not in toks_cf:
            toks.append(str(token))
        row["Source"] = ",".join([t for t in toks if t])

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _norm_side(row: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Side",
            "Team Rel",
            "TeamRel",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _ev_type(row: dict[str, str]) -> str:
        return _norm(row.get("Event Type") or row.get("Event") or "")

    def _period(row: dict[str, str]) -> Optional[int]:
        p = _parse_int(row.get("Period"))
        return p if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, str]) -> Optional[int]:
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is not None:
            return gs
        return parse_duration_seconds(
            row.get("Game Time") or row.get("GameTime") or row.get("Time")
        )

    def _video_seconds(row: dict[str, str]) -> Optional[int]:
        vs = _parse_int(row.get("Video Seconds") or row.get("VideoSeconds"))
        if vs is not None:
            return vs
        return parse_duration_seconds(row.get("Video Time") or row.get("VideoTime"))

    def _video_time(row: dict[str, str]) -> str:
        return str(row.get("Video Time") or row.get("VideoTime") or "").strip()

    def _has_video(row: dict[str, str]) -> bool:
        return _video_seconds(row) is not None or bool(_video_time(row))

    # Build generic per-period mapping points from incoming rows with both game+video seconds.
    # These points already incorporate stoppages because they are derived from shift sync.
    mapping_by_period: dict[int, list[tuple[int, int, dict[str, str]]]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        per = _period(r)
        gs = _game_seconds(r)
        vs = _video_seconds(r)
        if per is None or gs is None or vs is None:
            continue
        mapping_by_period.setdefault(int(per), []).append((int(gs), int(vs), r))
    for per, pts in list(mapping_by_period.items()):
        # Deduplicate by game seconds, keeping the earliest (minimum) video timestamp for that instant.
        best_by_gs: dict[int, tuple[int, dict[str, str]]] = {}
        for gs, vs, rr in pts:
            prev = best_by_gs.get(int(gs))
            if prev is None or int(vs) < int(prev[0]):
                best_by_gs[int(gs)] = (int(vs), dict(rr))
        mapping_by_period[per] = sorted(
            [(gs, vs, rr) for gs, (vs, rr) in best_by_gs.items()], key=lambda x: x[0]
        )

    def _interp_video_seconds(
        period: int, game_s: int
    ) -> Optional[
        tuple[int, tuple[tuple[int, int, dict[str, str]], tuple[int, int, dict[str, str]]]]
    ]:
        pts = mapping_by_period.get(int(period)) or []
        if len(pts) < 2:
            return None
        for i in range(len(pts) - 1):
            g0, v0, r0 = pts[i]
            g1, v1, r1 = pts[i + 1]
            lo, hi = (g0, g1) if g0 <= g1 else (g1, g0)
            if not (lo <= int(game_s) <= hi):
                continue
            if g0 == g1:
                continue
            # Linear interpolation (works for both increasing and decreasing mappings).
            v = int(round(v0 + (int(game_s) - g0) * (v1 - v0) / (g1 - g0)))
            return v, ((int(g0), int(v0), dict(r0)), (int(g1), int(v1), dict(r1)))
        return None

    # Build lookup from incoming penalty rows with video times.
    incoming_by_key: dict[tuple[int, str, int, str], dict[str, str]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        if _ev_type(r) != "penalty":
            continue
        per = _period(r)
        side = _norm_side(r)
        gs = _game_seconds(r)
        if per is None or side is None or gs is None:
            continue
        if not _has_video(r):
            continue
        jerseys = str(r.get("Attributed Jerseys") or "").strip()
        incoming_by_key[(int(per), side, int(gs), jerseys)] = r

    if not incoming_by_key:
        return existing_headers, existing_rows

    # Ensure destination headers include video fields.
    out_headers = list(existing_headers or [])
    for h in ("Video Time", "Video Seconds"):
        if h not in out_headers:
            out_headers.append(h)

    out_rows: list[dict[str, str]] = []
    for r in existing_rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        if _ev_type(rr) == "penalty" and _has_source(rr, "timetoscore") and not _has_video(rr):
            per = _period(rr)
            side = _norm_side(rr)
            gs = _game_seconds(rr)
            jerseys = str(rr.get("Attributed Jerseys") or "").strip()
            if per is not None and side is not None and gs is not None:
                match = incoming_by_key.get((int(per), side, int(gs), jerseys))
                # Fallback: match ignoring jersey if the scraper / import disagrees about attribution.
                if match is None:
                    match = incoming_by_key.get((int(per), side, int(gs), ""))
                if match is not None:
                    vs = _video_seconds(match)
                    vt = _video_time(match)
                    if vs is not None:
                        rr["Video Seconds"] = str(int(vs))
                        if not vt:
                            rr["Video Time"] = format_seconds_to_mmss_or_hhmmss(int(vs))
                    if vt:
                        rr["Video Time"] = vt
                    _add_source(rr, "shift_spreadsheet")
                else:
                    # Fallback: interpolate from any available shift-synced mapping points.
                    interp = _interp_video_seconds(int(per), int(gs))
                    if interp is not None:
                        vs2, (a, b) = interp
                        rr["Video Seconds"] = str(int(vs2))
                        rr["Video Time"] = format_seconds_to_mmss_or_hhmmss(int(vs2))
                        _add_source(rr, "shift_spreadsheet")
        out_rows.append(rr)

    return out_headers, out_rows


def filter_game_stats_for_display(game_stats: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not game_stats:
        return game_stats

    def _drop_key(k: str) -> bool:
        kk = str(k or "").strip().lower()
        if not kk or kk == "_label":
            return False
        # No shift/ice-time related stats in the webapp.
        return ("toi" in kk) or ("ice time" in kk) or ("shift" in kk)

    out: dict[str, Any] = {}
    for k, v in (game_stats or {}).items():
        if _drop_key(str(k)):
            continue
        if k != "_label" and (v is None or str(v).strip() == ""):
            continue
        kk = str(k or "").strip().lower()
        if k != "_label" and ("ot" in kk):
            vv = str(v).strip()
            if vv in {"0", "0.0"}:
                continue
        out[k] = v
    return out


def normalize_event_type_key(raw: Any) -> str:
    """
    Stable key for event types across sources (e.g. "Expected Goal" vs "ExpectedGoal").
    """
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().casefold())


def compute_goalie_stats_for_game(
    event_rows: list[dict[str, Any]],
    *,
    home_goalies: Optional[list[dict[str, Any]]] = None,
    away_goalies: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Compute per-goalie stats for a single game from event rows.

    Uses:
      - goaliechange events (to determine which goalie is in net when)
      - goal events (goals against)
      - shot-on-goal events (shots against), from the implication chain:
          Goals ⊆ xG ⊆ SOG
        Concretely, we treat `goal`, `expectedgoal`, and `sog`/`shotongoal` rows as shot-on-goal
        evidence and de-duplicate per (period, game_seconds, shooter, side).
    """

    def _int_or_none(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    def _side_norm(row: dict[str, Any]) -> Optional[str]:
        for k in (
            "team_side_norm",
            "team_side",
            "Team Side",
            "TeamSide",
            "team_rel",
            "Team Rel",
            "TeamRel",
            "team_raw",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _period(row: dict[str, Any]) -> Optional[int]:
        p = _int_or_none(row.get("period") if "period" in row else row.get("Period"))
        return int(p) if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, Any]) -> Optional[int]:
        gs = _int_or_none(
            row.get("game_seconds")
            if "game_seconds" in row
            else row.get("Game Seconds") or row.get("GameSeconds")
        )
        if gs is not None:
            return int(gs)
        gt = row.get("game_time") if "game_time" in row else row.get("Game Time") or row.get("Time")
        return parse_duration_seconds(gt)

    def _event_type_key(row: dict[str, Any]) -> str:
        raw = (
            row.get("event_type_key")
            or row.get("event_type__key")
            or row.get("event_type")
            or row.get("Event Type")
            or row.get("Event")
        )
        return normalize_event_type_key(raw)

    def _attr_players(row: dict[str, Any]) -> str:
        return str(
            row.get("attributed_players")
            if "attributed_players" in row
            else row.get("Attributed Players") or row.get("AttributedPlayers") or ""
        ).strip()

    def _attr_jerseys(row: dict[str, Any]) -> str:
        return str(
            row.get("attributed_jerseys")
            if "attributed_jerseys" in row
            else row.get("Attributed Jerseys") or row.get("AttributedJerseys") or ""
        ).strip()

    def _details(row: dict[str, Any]) -> str:
        return str(row.get("details") if "details" in row else row.get("Details") or "").strip()

    def _player_id(row: dict[str, Any]) -> Optional[int]:
        return _int_or_none(row.get("player_id") or row.get("player") or row.get("Player ID"))

    def _event_id(row: dict[str, Any]) -> Optional[int]:
        return _int_or_none(row.get("event_id") or row.get("Event ID") or row.get("EventID"))

    def _goalie_maps(
        goalies: list[dict[str, Any]],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
        by_id: dict[int, dict[str, Any]] = {}
        name_to_ids: dict[str, set[int]] = {}
        for p in goalies or []:
            pid = _int_or_none(p.get("id") or p.get("player_id"))
            if pid is None:
                continue
            by_id[int(pid)] = p
            nm = normalize_player_name(str(p.get("name") or ""))
            if nm:
                name_to_ids.setdefault(nm, set()).add(int(pid))
        unique_by_name = {k: int(list(v)[0]) for k, v in name_to_ids.items() if len(v) == 1}
        return by_id, unique_by_name

    home_goalies = list(home_goalies or [])
    away_goalies = list(away_goalies or [])
    goalie_roster_by_side = {"home": home_goalies, "away": away_goalies}
    goalie_by_id: dict[str, dict[int, dict[str, Any]]] = {}
    goalie_id_by_name: dict[str, dict[str, int]] = {}
    goalie_side_by_pid: dict[int, str] = {}
    goalie_name_by_pid: dict[int, str] = {}
    for side, goalies in goalie_roster_by_side.items():
        by_id, by_name = _goalie_maps(goalies)
        goalie_by_id[side] = by_id
        goalie_id_by_name[side] = by_name
        for pid, rec in by_id.items():
            goalie_side_by_pid[int(pid)] = str(side)
            nm = str(rec.get("name") or "").strip()
            if nm and int(pid) not in goalie_name_by_pid:
                goalie_name_by_pid[int(pid)] = nm

    event_type_keys_present = {
        _event_type_key(r) for r in (event_rows or []) if isinstance(r, dict)
    }
    has_sog = bool(event_type_keys_present & {"goal", "expectedgoal", "sog", "shotongoal"})
    # Only consider xG stats available when ExpectedGoal rows exist (goals contribute to xG once xG
    # data exists, but goals alone should not force xG-based goalie columns to appear).
    has_xg = bool(event_type_keys_present & {"expectedgoal"})

    goalie_changes: dict[str, dict[int, list[tuple[int, Optional[int], str]]]] = {
        "home": {},
        "away": {},
    }
    for r in event_rows or []:
        if not isinstance(r, dict):
            continue
        if _event_type_key(r) != "goaliechange":
            continue
        side = _side_norm(r)
        per = _period(r)
        gs = _game_seconds(r)
        if side not in {"home", "away"} or per is None or gs is None:
            continue

        name = _attr_players(r)
        det = _details(r)
        empty_net = "emptynet" in normalize_player_name(
            name
        ) or "emptynet" in normalize_player_name(det)
        goalie_pid: Optional[int] = None if empty_net else _player_id(r)
        if goalie_pid is None and (not empty_net) and name:
            goalie_pid = goalie_id_by_name.get(str(side), {}).get(normalize_player_name(name))
        if goalie_pid is not None:
            goalie_side_by_pid[int(goalie_pid)] = str(side)

        goalie_name = name
        if goalie_pid is not None and not goalie_name:
            goalie_name = str(
                goalie_by_id.get(str(side), {}).get(int(goalie_pid), {}).get("name") or ""
            )
        if goalie_pid is not None and goalie_name and int(goalie_pid) not in goalie_name_by_pid:
            goalie_name_by_pid[int(goalie_pid)] = goalie_name

        goalie_changes.setdefault(str(side), {}).setdefault(int(per), []).append(
            (int(gs), goalie_pid, goalie_name)
        )

    def _infer_regulation_len_s() -> int:
        cand: list[int] = []
        for side in ("home", "away"):
            for gs, _pid, name in goalie_changes.get(side, {}).get(1, []):
                if "starting" in normalize_player_name(name):
                    cand.append(int(gs))
        if cand:
            return max(cand)

        cand = []
        for side in ("home", "away"):
            for gs, _pid, _name in goalie_changes.get(side, {}).get(1, []):
                cand.append(int(gs))
        if cand:
            return max(cand)

        cand = []
        for r in event_rows or []:
            if not isinstance(r, dict):
                continue
            per = _period(r)
            gs = _game_seconds(r)
            if per == 1 and gs is not None:
                cand.append(int(gs))
        if cand:
            return max(cand)
        return 15 * 60

    reg_len_s = int(_infer_regulation_len_s())

    def _period_len_s(period: int) -> int:
        if int(period) <= 3:
            return reg_len_s
        cand = []
        for r in event_rows or []:
            if not isinstance(r, dict):
                continue
            per = _period(r)
            gs = _game_seconds(r)
            if per == int(period) and gs is not None:
                cand.append(int(gs))
        return max(cand) if cand else reg_len_s

    periods = [1, 2, 3]
    extra_periods = sorted(
        {
            int(_period(r) or 0)
            for r in (event_rows or [])
            if isinstance(r, dict) and (_period(r) or 0) > 3
        }
    )
    periods.extend([p for p in extra_periods if p not in periods])

    start_goalie_by_side: dict[str, Optional[int]] = {"home": None, "away": None}
    for side in ("home", "away"):
        pts = goalie_changes.get(side, {}).get(1, [])
        if pts:
            pts_sorted = sorted(pts, key=lambda x: -int(x[0]))
            start_goalie_by_side[side] = pts_sorted[0][1]
        else:
            roster_goalies = goalie_roster_by_side.get(side) or []
            if len(roster_goalies) == 1:
                pid = _int_or_none(
                    roster_goalies[0].get("id") or roster_goalies[0].get("player_id")
                )
                start_goalie_by_side[side] = int(pid) if pid is not None else None
        if start_goalie_by_side.get(side) is not None:
            goalie_side_by_pid[int(start_goalie_by_side[side])] = str(side)

    points_by_side_period: dict[str, dict[int, list[tuple[int, Optional[int]]]]] = {
        "home": {},
        "away": {},
    }
    toi_by_goalie: dict[int, int] = {}

    for side in ("home", "away"):
        carry = start_goalie_by_side.get(side)
        for per in periods:
            per_len = int(_period_len_s(int(per)))
            changes = sorted(
                goalie_changes.get(side, {}).get(int(per), []), key=lambda x: -int(x[0])
            )
            for gs, pid, _name in changes:
                if int(gs) >= int(per_len) - 1:
                    carry = pid
                    break

            uniq: dict[int, Optional[int]] = {}
            uniq[int(per_len)] = carry
            for gs, pid, _name in changes:
                if int(gs) >= int(per_len) - 1:
                    continue
                uniq[int(gs)] = pid

            pts = sorted([(int(gs), pid) for gs, pid in uniq.items()], key=lambda x: -int(x[0]))
            points_by_side_period[side][int(per)] = pts

            for idx, (t0, goalie_pid) in enumerate(pts):
                t1 = pts[idx + 1][0] if idx + 1 < len(pts) else 0
                dur = int(t0) - int(t1)
                if dur > 0 and goalie_pid is not None:
                    toi_by_goalie[int(goalie_pid)] = toi_by_goalie.get(int(goalie_pid), 0) + int(
                        dur
                    )

            carry = pts[-1][1] if pts else carry

    def _active_goalie(side: str, *, period: int, game_s: int) -> Optional[int]:
        pts = points_by_side_period.get(str(side), {}).get(int(period), [])
        cur = None
        for t, pid in pts:
            if int(t) >= int(game_s):
                cur = pid
                continue
            break
        return cur

    goals_by_goalie: dict[int, set[tuple[int, int, str, str]]] = {}
    xg_shots_by_goalie: dict[int, set[tuple[int, int, str, str]]] = {}
    shots_by_goalie: dict[int, set[tuple[int, int, str, str]]] = {}

    for r in event_rows or []:
        if not isinstance(r, dict):
            continue
        et = _event_type_key(r)
        if et not in {"goal", "expectedgoal", "sog", "shotongoal"}:
            continue
        shooter_side = _side_norm(r)
        per = _period(r)
        gs = _game_seconds(r)
        if shooter_side not in {"home", "away"} or per is None or gs is None:
            continue
        defending_side = "away" if shooter_side == "home" else "home"
        goalie_pid = _active_goalie(defending_side, period=int(per), game_s=int(gs))
        if goalie_pid is None:
            continue

        shot_id = _player_id(r)
        if shot_id is None:
            shot_id = _int_or_none(normalize_jersey_number(_attr_jerseys(r)))
        if shot_id is None:
            shot_id = _event_id(r)
        shot_tag = (
            str(shot_id) if shot_id is not None else normalize_player_name(_attr_players(r)) or ""
        )
        shot_key = (int(per), int(gs), str(shooter_side), shot_tag)

        if et == "goal":
            goals_by_goalie.setdefault(int(goalie_pid), set()).add(shot_key)
        if has_xg and et in {"goal", "expectedgoal"}:
            xg_shots_by_goalie.setdefault(int(goalie_pid), set()).add(shot_key)
        if has_sog and et in {"goal", "expectedgoal", "sog", "shotongoal"}:
            shots_by_goalie.setdefault(int(goalie_pid), set()).add(shot_key)

    def _goalie_row(side: str, goalie_pid: int) -> dict[str, Any]:
        rec = goalie_by_id.get(str(side), {}).get(int(goalie_pid), {})
        name = str(rec.get("name") or goalie_name_by_pid.get(int(goalie_pid)) or "").strip()
        jersey = str(rec.get("jersey_number") or "").strip()
        toi = int(toi_by_goalie.get(int(goalie_pid), 0))
        ga = len(goals_by_goalie.get(int(goalie_pid), set()))
        xga = len(xg_shots_by_goalie.get(int(goalie_pid), set())) if has_xg else None
        xg_saves = (int(xga) - int(ga)) if (xga is not None) else None
        if xg_saves is not None and xg_saves < 0:
            xg_saves = 0
        xg_sv_pct = (
            (float(xg_saves) / float(xga))
            if (xga is not None and xga > 0 and xg_saves is not None)
            else None
        )
        sa = len(shots_by_goalie.get(int(goalie_pid), set())) if has_sog else None
        saves = (int(sa) - int(ga)) if (sa is not None) else None
        if saves is not None and saves < 0:
            saves = 0
        sv_pct = (
            (float(saves) / float(sa))
            if (sa is not None and sa > 0 and saves is not None)
            else None
        )
        gaa = (float(ga) * 60.0 / float(toi)) if toi > 0 else None
        return {
            "player_id": int(goalie_pid),
            "name": name,
            "jersey_number": jersey,
            "toi_seconds": toi,
            "ga": int(ga),
            "xga": int(xga) if xga is not None else None,
            "xg_saves": int(xg_saves) if xg_saves is not None else None,
            "xg_sv_pct": xg_sv_pct,
            "sa": int(sa) if sa is not None else None,
            "saves": int(saves) if saves is not None else None,
            "sv_pct": sv_pct,
            "gaa": gaa,
        }

    out: dict[str, Any] = {
        "meta": {"has_sog": bool(has_sog), "has_xg": bool(has_xg)},
        "home": [],
        "away": [],
    }
    for side in ("home", "away"):
        ids = set()
        ids.update(
            {
                int(_int_or_none(p.get("id") or p.get("player_id")) or 0)
                for p in goalie_roster_by_side.get(side, [])
            }
        )
        ids.update(
            {
                int(pid)
                for pid in toi_by_goalie.keys()
                if pid and goalie_side_by_pid.get(int(pid)) == str(side)
            }
        )
        ids.update(
            {
                int(pid)
                for pid in goals_by_goalie.keys()
                if pid and goalie_side_by_pid.get(int(pid)) == str(side)
            }
        )
        if has_xg:
            ids.update(
                {
                    int(pid)
                    for pid in xg_shots_by_goalie.keys()
                    if pid and goalie_side_by_pid.get(int(pid)) == str(side)
                }
            )
        if has_sog:
            ids.update(
                {
                    int(pid)
                    for pid in shots_by_goalie.keys()
                    if pid and goalie_side_by_pid.get(int(pid)) == str(side)
                }
            )
        ids = {i for i in ids if i > 0}
        rows = [_goalie_row(side, int(pid)) for pid in sorted(ids)]
        rows.sort(
            key=lambda r: (-int(r.get("toi_seconds") or 0), str(r.get("name") or "").casefold())
        )
        out[str(side)] = rows
    return out


def compute_goalie_stats_for_team_games(
    *,
    team_id: int,
    schedule_games: list[dict[str, Any]],
    event_rows_by_game_id: dict[int, list[dict[str, Any]]],
    goalies: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate goalie stats across a list of games for a single team.

    Shots against (SA) and derived stats are only accumulated for games that include SOG events.
    """

    def _int_or_none(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    game_by_id = {}
    for g in schedule_games or []:
        try:
            game_by_id[int(g.get("id"))] = g
        except Exception:
            continue

    totals: dict[int, dict[str, Any]] = {}
    has_any_sog = False
    has_any_xg = False
    for gid, rows in (event_rows_by_game_id or {}).items():
        g = game_by_id.get(int(gid))
        if not g:
            continue
        t1 = _int0(g.get("team1_id"))
        t2 = _int0(g.get("team2_id"))
        if int(team_id) == int(t1):
            our_side = "home"
            stats = compute_goalie_stats_for_game(rows, home_goalies=goalies, away_goalies=[])
        elif int(team_id) == int(t2):
            our_side = "away"
            stats = compute_goalie_stats_for_game(rows, home_goalies=[], away_goalies=goalies)
        else:
            continue

        meta = stats.get("meta") or {}
        if bool(meta.get("has_sog")):
            has_any_sog = True
        if bool(meta.get("has_xg")):
            has_any_xg = True

        for gr in stats.get(our_side, []) or []:
            pid = _int_or_none(gr.get("player_id"))
            if pid is None or pid <= 0:
                continue
            rec = totals.setdefault(
                int(pid),
                {
                    "player_id": int(pid),
                    "name": str(gr.get("name") or "").strip(),
                    "jersey_number": str(gr.get("jersey_number") or "").strip(),
                    "gp": 0,
                    "toi_seconds": 0,
                    "ga": 0,
                    "sa_sum": 0,
                    "saves_sum": 0,
                    "has_sog": False,
                    "xga_sum": 0,
                    "xg_saves_sum": 0,
                    "has_xg": False,
                },
            )
            if not rec.get("name") and gr.get("name"):
                rec["name"] = str(gr.get("name") or "").strip()
            if not rec.get("jersey_number") and gr.get("jersey_number"):
                rec["jersey_number"] = str(gr.get("jersey_number") or "").strip()

            toi = _int_or_none(gr.get("toi_seconds")) or 0
            ga = _int_or_none(gr.get("ga")) or 0
            rec["toi_seconds"] += int(toi)
            rec["ga"] += int(ga)
            if toi > 0:
                rec["gp"] += 1

            sa = gr.get("sa")
            if sa is not None:
                rec["has_sog"] = True
                rec["sa_sum"] += int(_int_or_none(sa) or 0)
                rec["saves_sum"] += int(_int_or_none(gr.get("saves")) or 0)

            xga = gr.get("xga")
            if xga is not None:
                rec["has_xg"] = True
                rec["xga_sum"] += int(_int_or_none(xga) or 0)
                rec["xg_saves_sum"] += int(_int_or_none(gr.get("xg_saves")) or 0)

    out_rows: list[dict[str, Any]] = []
    for pid, rec in totals.items():
        toi = int(rec.get("toi_seconds") or 0)
        ga = int(rec.get("ga") or 0)
        has_sog = bool(rec.get("has_sog"))
        sa = int(rec.get("sa_sum") or 0) if has_sog else None
        saves = int(rec.get("saves_sum") or 0) if has_sog else None
        sv_pct = (
            (float(saves) / float(sa))
            if (has_sog and sa and sa > 0 and saves is not None)
            else None
        )
        has_xg = bool(rec.get("has_xg"))
        xga = int(rec.get("xga_sum") or 0) if has_xg else None
        xg_saves = int(rec.get("xg_saves_sum") or 0) if has_xg else None
        xg_sv_pct = (
            (float(xg_saves) / float(xga))
            if (has_xg and xga and xga > 0 and xg_saves is not None)
            else None
        )
        gaa = (float(ga) * 60.0 / float(toi)) if toi > 0 else None
        out_rows.append(
            {
                "player_id": int(pid),
                "jersey_number": str(rec.get("jersey_number") or "").strip(),
                "name": str(rec.get("name") or "").strip(),
                "gp": int(rec.get("gp") or 0),
                "toi_seconds": toi,
                "ga": ga,
                "xga": xga,
                "xg_saves": xg_saves,
                "xg_sv_pct": xg_sv_pct,
                "sa": sa,
                "saves": saves,
                "sv_pct": sv_pct,
                "gaa": gaa,
            }
        )

    out_rows.sort(
        key=lambda r: (-int(r.get("toi_seconds") or 0), str(r.get("name") or "").casefold())
    )
    return {"rows": out_rows, "meta": {"has_sog": bool(has_any_sog), "has_xg": bool(has_any_xg)}}


def compute_game_event_stats_by_side(events_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Build a simple Home/Away event-count table from normalized game events rows.
    Returns rows: {"event_type": str, "home": int, "away": int}
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip()

    def _norm_cf(s: Any) -> str:
        return _norm(s).casefold()

    def _event_type(r: dict[str, str]) -> str:
        return _norm(r.get("Event Type") or r.get("Event") or r.get("Type") or "")

    def _side(r: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Team Rel",
            "TeamRel",
            "Side",
            "Team",
            "Team Raw",
            "TeamRaw",
        ):
            v = _norm_cf(r.get(k))
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
            if v in {"neutral"}:
                return None
        return None

    skip_types = {
        "assist",
        "penalty expired",
        "power play",
        "powerplay",
        "penalty kill",
        "penaltykill",
    }
    counts: dict[str, dict[str, int]] = {}
    for r in events_rows or []:
        if not isinstance(r, dict):
            continue
        et = _event_type(r)
        if not et:
            continue
        et_cf = et.casefold()
        if et_cf in skip_types:
            continue
        side = _side(r)
        if side not in {"home", "away"}:
            continue
        rec = counts.setdefault(et, {"home": 0, "away": 0})
        rec[side] += 1

    def _prio(et: str) -> int:
        key = et.casefold().replace(" ", "")
        order = [
            "goal",
            "penalty",
            "sog",
            "shot",
            "xg",
            "expectedgoal",
            "rush",
            "controlledentry",
            "controlledexit",
            "giveaway",
            "takeaway",
            "turnovers(forced)",
            "turnoverforced",
            "createdturnover",
            "goaliechange",
        ]
        try:
            return order.index(key)
        except Exception:
            return 10_000

    rows: list[dict[str, Any]] = []
    for et, rec in counts.items():
        if int(rec.get("home") or 0) == 0 and int(rec.get("away") or 0) == 0:
            continue
        rows.append(
            {"event_type": et, "home": int(rec.get("home") or 0), "away": int(rec.get("away") or 0)}
        )
    rows.sort(
        key=lambda r: (
            _prio(str(r.get("event_type") or "")),
            str(r.get("event_type") or "").casefold(),
        )
    )
    return rows


def normalize_events_video_time_for_display(
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Display-time normalization for events video fields:
      - Ensure a human-readable "Video Time" column exists when any video clip field exists.
      - Ensure a numeric "Video Seconds" column exists when any video clip field exists.
      - For each row, if one of (Video Time, Video Seconds) exists but the other is missing, derive it.

    This normalization does not affect the stored CSV/event rows in the database; it only impacts
    the returned headers/rows used for UI rendering and embedded JSON.
    """
    if not headers or not rows:
        return headers, rows

    def _hnorm(h: Any) -> str:
        return str(h or "").strip().lower()

    def _header_idx(headers_in: list[str], norms: set[str]) -> Optional[int]:
        for i, h in enumerate(headers_in or []):
            if _hnorm(h) in norms:
                return i
        return None

    def _has_any_video(rows_in: list[dict[str, str]]) -> bool:
        for r in rows_in or []:
            if not isinstance(r, dict):
                continue
            vt_raw = str(r.get("Video Time") or r.get("VideoTime") or "").strip()
            if vt_raw:
                return True
            vs = parse_duration_seconds(
                r.get("Video Seconds")
                or r.get("VideoSeconds")
                or r.get("Video S")
                or r.get("VideoS")
            )
            if vs is not None:
                return True
        return False

    # If the table doesn't contain any clip metadata, keep it unchanged.
    if not _has_any_video(rows):
        return headers, rows

    vt_norms = {"video time", "videotime"}
    vs_norms = {"video seconds", "videoseconds", "video s", "videos"}
    gt_norms = {"game time", "gametime", "time"}

    out_headers = list(headers)
    vt_idx = _header_idx(out_headers, vt_norms)
    vs_idx = _header_idx(out_headers, vs_norms)

    if vt_idx is None and vs_idx is not None:
        # Prefer to place it next to Video Seconds if present.
        out_headers.insert(int(vs_idx), "Video Time")
        vt_idx = int(vs_idx)
        vs_idx = _header_idx(out_headers, vs_norms)

    if vs_idx is None and vt_idx is not None:
        # Place seconds next to Video Time when Video Time exists.
        out_headers.insert(int(vt_idx) + 1, "Video Seconds")
        vs_idx = int(vt_idx) + 1

    if vt_idx is None and vs_idx is None:
        # Last resort: place both near Game Time, or append.
        try:
            gt_idx = next(i for i, h in enumerate(out_headers) if _hnorm(h) in gt_norms)
            out_headers.insert(int(gt_idx) + 1, "Video Time")
            out_headers.insert(int(gt_idx) + 2, "Video Seconds")
        except Exception:
            out_headers.append("Video Time")
            out_headers.append("Video Seconds")

    out_rows: list[dict[str, str]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)

        vt = str(rr.get("Video Time") or rr.get("VideoTime") or "").strip()
        vs = parse_duration_seconds(
            rr.get("Video Seconds")
            or rr.get("VideoSeconds")
            or rr.get("Video S")
            or rr.get("VideoS")
        )

        if vs is None and vt:
            vs = parse_duration_seconds(vt)
            if vs is not None:
                vs_s = str(int(vs))
                for k in ("Video Seconds", "VideoSeconds", "Video S", "VideoS"):
                    if not str(rr.get(k) or "").strip():
                        rr[k] = vs_s

        if not vt and vs is not None:
            vt2 = format_seconds_to_mmss_or_hhmmss(vs)
            if vt2:
                for k in ("Video Time", "VideoTime"):
                    if not str(rr.get(k) or "").strip():
                        rr[k] = vt2

        out_rows.append(rr)

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _period_and_game_seconds(row: dict[str, str]) -> Optional[tuple[int, int]]:
        p = _parse_int(row.get("Period"))
        if p is None or p <= 0:
            return None
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is None:
            gs = parse_duration_seconds(
                row.get("Game Time") or row.get("GameTime") or row.get("Time")
            )
        if gs is None:
            return None
        return int(p), int(gs)

    def _count_jerseys(raw: Any) -> int:
        s = str(raw or "").strip()
        if not s:
            return 0
        nums: set[int] = set()
        for m0 in re.findall(r"(\d+)", s):
            try:
                nums.add(int(m0))
            except Exception:
                continue
        return len(nums)

    def _normalize_on_ice_str(raw: Any, *, max_players: int = 6) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        # Most sources emit comma-separated "Name #Jersey" tokens; keep order but dedupe jerseys.
        parts = [p.strip() for p in re.split(r"[,;\n]+", s) if p and p.strip()]
        seen: set[tuple[str, Any]] = set()
        out: list[str] = []
        for p in parts:
            m = re.search(r"(\d+)", p)
            if m:
                try:
                    k = ("j", int(m.group(1)))
                except Exception:
                    k = ("s", p.casefold())
            else:
                k = ("s", p.casefold())
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        if len(out) > int(max_players):
            out = out[: int(max_players)]
            if out:
                out[-1] = out[-1] + " …"
        return ",".join(out)

    # Propagate missing clip/on-ice metadata across events at the same (period, game seconds).
    best_video_seconds: dict[tuple[int, int], dict[str, Any]] = {}
    best_on_ice_home: dict[tuple[int, int], str] = {}
    best_on_ice_away: dict[tuple[int, int], str] = {}

    for rr in out_rows:
        if not isinstance(rr, dict):
            continue
        k = _period_and_game_seconds(rr)
        if k is None:
            continue

        vt0, vs0 = normalize_video_time_and_seconds(
            rr.get("Video Time") or rr.get("VideoTime"),
            rr.get("Video Seconds")
            or rr.get("VideoSeconds")
            or rr.get("Video S")
            or rr.get("VideoS"),
        )
        if vs0 is not None:
            prev = best_video_seconds.get(k)
            if prev is None or int(vs0) < int(prev.get("video_seconds") or 0):
                best_video_seconds[k] = {
                    "video_seconds": int(vs0),
                    "video_time": str(vt0 or "").strip(),
                    "period": int(k[0]),
                    "game_seconds": int(k[1]),
                    "game_time": str(
                        rr.get("Game Time") or rr.get("GameTime") or rr.get("Time") or ""
                    ).strip(),
                }

        on_home = _normalize_on_ice_str(
            rr.get("On-Ice Players (Home)") or rr.get("OnIce Players (Home)") or ""
        )
        if on_home:
            prev = best_on_ice_home.get(k)
            if prev is None or _count_jerseys(on_home) > _count_jerseys(prev):
                best_on_ice_home[k] = on_home

        on_away = _normalize_on_ice_str(
            rr.get("On-Ice Players (Away)") or rr.get("OnIce Players (Away)") or ""
        )
        if on_away:
            prev = best_on_ice_away.get(k)
            if prev is None or _count_jerseys(on_away) > _count_jerseys(prev):
                best_on_ice_away[k] = on_away

    if best_video_seconds or best_on_ice_home or best_on_ice_away:
        for rr in out_rows:
            if not isinstance(rr, dict):
                continue
            k = _period_and_game_seconds(rr)
            if k is None:
                continue

            vt0, vs0 = normalize_video_time_and_seconds(
                rr.get("Video Time") or rr.get("VideoTime"),
                rr.get("Video Seconds")
                or rr.get("VideoSeconds")
                or rr.get("Video S")
                or rr.get("VideoS"),
            )
            if vs0 is None:
                best = best_video_seconds.get(k)
                if best is not None:
                    vs_best = best.get("video_seconds")
                    if vs_best is None:
                        continue
                    vs_s = str(int(vs_best))
                    rr.setdefault("Video Seconds", vs_s)
                    if not str(rr.get("Video Seconds") or "").strip():
                        rr["Video Seconds"] = vs_s
                    if not str(rr.get("VideoSeconds") or "").strip():
                        rr["VideoSeconds"] = vs_s
                    vt_best = format_seconds_to_mmss_or_hhmmss(vs_best)
                    if not str(rr.get("Video Time") or "").strip():
                        rr["Video Time"] = vt_best
                    if not str(rr.get("VideoTime") or "").strip():
                        rr["VideoTime"] = vt_best

            if not str(rr.get("On-Ice Players (Home)") or "").strip():
                home_best = best_on_ice_home.get(k)
                if home_best:
                    rr["On-Ice Players (Home)"] = home_best

            if not str(rr.get("On-Ice Players (Away)") or "").strip():
                away_best = best_on_ice_away.get(k)
                if away_best:
                    rr["On-Ice Players (Away)"] = away_best

    # Normalize on-ice strings for display (dedupe + clamp).
    for rr in out_rows:
        if not isinstance(rr, dict):
            continue
        home_norm = _normalize_on_ice_str(
            rr.get("On-Ice Players (Home)") or rr.get("OnIce Players (Home)") or ""
        )
        if home_norm:
            rr["On-Ice Players (Home)"] = home_norm
        away_norm = _normalize_on_ice_str(
            rr.get("On-Ice Players (Away)") or rr.get("OnIce Players (Away)") or ""
        )
        if away_norm:
            rr["On-Ice Players (Away)"] = away_norm
        legacy_norm = _normalize_on_ice_str(
            rr.get("On-Ice Players") or rr.get("OnIce Players") or rr.get("OnIcePlayers") or ""
        )
        if legacy_norm:
            rr["On-Ice Players"] = legacy_norm

    return out_headers, out_rows


def normalize_video_time_and_seconds(
    video_time: Any, video_seconds: Any
) -> tuple[str, Optional[int]]:
    """
    Best-effort bidirectional normalization for clip timestamps.

    Returns:
      - video_time: normalized string ('' when unknown)
      - video_seconds: int seconds (None when unknown/unparseable)
    """
    vt = str(video_time or "").strip()
    vs = parse_duration_seconds(video_seconds)
    if vs is None and vt:
        vs = parse_duration_seconds(vt)
    if not vt and vs is not None:
        vt = format_seconds_to_mmss_or_hhmmss(vs)
    return vt, vs


def _event_table_sort_key(r: dict[str, Any]) -> tuple[int, int, int]:
    """
    Stable default ordering for event tables:
      1) game datetime (descending; newest game first),
      2) period (ascending),
      3) game time within period (descending; clock counts down).
    """

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    gs = _parse_int(
        r.get("Game Seconds")
        or r.get("GameSeconds")
        or r.get("game_seconds")
        or r.get("gameSeconds")
    )
    if gs is None:
        gs = parse_duration_seconds(
            r.get("Game Time")
            or r.get("GameTime")
            or r.get("Time")
            or r.get("game_time")
            or r.get("gameTime")
        )
    game_seconds = int(gs) if gs is not None else -1

    p = _parse_int(r.get("Period") or r.get("period"))
    period = int(p) if p is not None and p > 0 else 999

    dt_raw = r.get("game_starts_at") or r.get("starts_at") or r.get("Date") or r.get("date")
    dt_obj = to_dt(dt_raw)
    if dt_obj is None:
        game_dt_key = 99999999999999
    else:
        game_dt_key = -int(dt_obj.strftime("%Y%m%d%H%M%S"))

    return (int(game_dt_key), int(period), -int(game_seconds))


def sort_events_rows_default(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Default ordering for the Game Events table: see `_event_table_sort_key`.
    """
    return sorted(
        [r for r in (rows or []) if isinstance(r, dict)],
        key=_event_table_sort_key,
    )


def sort_event_dicts_for_table_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sort event dicts for UI tables using `_event_table_sort_key`.
    """
    return sorted(
        [r for r in (rows or []) if isinstance(r, dict)],
        key=_event_table_sort_key,
    )


def get_user_video_clip_len_s(db_conn, user_id: Optional[int]) -> int:
    """
    Per-user clip length preference for timeline video clips.
    Defaults to 30 seconds when unset/unknown.
    """
    if not user_id:
        return 30
    try:
        del db_conn
        _django_orm, m = _orm_modules()
        v = (
            m.User.objects.filter(id=int(user_id))
            .values_list("video_clip_len_s", flat=True)
            .first()
        )
        try:
            iv = int(v) if v is not None else None
        except Exception:
            iv = None
        if iv in {15, 20, 30, 45, 60, 90}:
            return int(iv)
    except Exception:
        pass
    return 30


def reset_league_data(
    db_conn, league_id: int, *, owner_user_id: Optional[int] = None
) -> dict[str, int]:
    """
    Wipe imported hockey data for a league (games/teams/players/stats) while keeping:
      - users
      - league record and memberships

    This is used by `tools/webapp/scripts/reset_league_data.py` and the hidden REST endpoint.
    """
    stats: dict[str, int] = {
        "player_stats": 0,
        "league_games": 0,
        "hky_games": 0,
        "league_teams": 0,
        "players": 0,
        "teams": 0,
    }
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db import transaction
    from django.db.models import Q

    stats["league_games"] = m.LeagueGame.objects.filter(league_id=int(league_id)).count()
    stats["league_teams"] = m.LeagueTeam.objects.filter(league_id=int(league_id)).count()

    league_game_ids = list(
        m.LeagueGame.objects.filter(league_id=int(league_id)).values_list("game_id", flat=True)
    )
    other_game_ids = set()
    if league_game_ids:
        other_game_ids = set(
            m.LeagueGame.objects.exclude(league_id=int(league_id))
            .filter(game_id__in=league_game_ids)
            .values_list("game_id", flat=True)
        )
    exclusive_game_ids = sorted(
        {int(gid) for gid in league_game_ids if gid is not None and gid not in other_game_ids}
    )
    if exclusive_game_ids:
        stats["player_stats"] = m.PlayerStat.objects.filter(game_id__in=exclusive_game_ids).count()
        stats["hky_games"] = m.HkyGame.objects.filter(id__in=exclusive_game_ids).count()

    league_team_ids = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list("team_id", flat=True)
    )
    other_team_ids = set()
    if league_team_ids:
        other_team_ids = set(
            m.LeagueTeam.objects.exclude(league_id=int(league_id))
            .filter(team_id__in=league_team_ids)
            .values_list("team_id", flat=True)
        )
    exclusive_team_ids = sorted(
        {int(tid) for tid in league_team_ids if tid is not None and tid not in other_team_ids}
    )

    with transaction.atomic():
        # Remove league mappings (this is the "reset" behavior).
        m.LeagueGame.objects.filter(league_id=int(league_id)).delete()
        m.LeagueTeam.objects.filter(league_id=int(league_id)).delete()

        # Delete exclusive games (cascades to player_stats/hky_game_* tables).
        if exclusive_game_ids:
            m.HkyGame.objects.filter(id__in=exclusive_game_ids).delete()

        if exclusive_team_ids:
            eligible_qs = m.Team.objects.filter(id__in=exclusive_team_ids, is_external=True)
            if owner_user_id is not None:
                eligible_qs = eligible_qs.filter(user_id=int(owner_user_id))
            eligible_ids = list(eligible_qs.values_list("id", flat=True))

            if eligible_ids:
                eligible_set = {int(tid) for tid in eligible_ids}
                still_used: set[int] = set()
                for team1_id, team2_id in m.HkyGame.objects.filter(
                    Q(team1_id__in=eligible_ids) | Q(team2_id__in=eligible_ids)
                ).values_list("team1_id", "team2_id"):
                    if team1_id in eligible_set:
                        still_used.add(int(team1_id))
                    if team2_id in eligible_set:
                        still_used.add(int(team2_id))

                safe_team_ids = sorted(
                    [int(tid) for tid in eligible_set if int(tid) not in still_used]
                )
                if safe_team_ids:
                    stats["players"] = m.Player.objects.filter(team_id__in=safe_team_ids).count()
                    stats["teams"] = m.Team.objects.filter(id__in=safe_team_ids).count()
                    m.Team.objects.filter(id__in=safe_team_ids).delete()

    return stats


def compute_team_stats(db_conn, team_id: int, user_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    rows = list(
        m.HkyGame.objects.filter(
            user_id=int(user_id),
            team1_score__isnull=False,
            team2_score__isnull=False,
        )
        .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
        .values("team1_id", "team2_id", "team1_score", "team2_score")
    )
    wins = losses = ties = gf = ga = 0
    for r in rows:
        t1 = int(r["team1_id"]) == team_id
        my_score = (
            int(r["team1_score"])
            if t1
            else int(r["team2_score"]) if r["team2_score"] is not None else 0
        )
        op_score = (
            int(r["team2_score"])
            if t1
            else int(r["team1_score"]) if r["team1_score"] is not None else 0
        )
        gf += my_score
        ga += op_score
        if my_score > op_score:
            wins += 1
        elif my_score < op_score:
            losses += 1
        else:
            ties += 1
    points_total = wins * 2 + ties * 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "gf": gf,
        "ga": ga,
        "points": points_total,
        "points_total": points_total,
    }


def compute_team_stats_league(db_conn, team_id: int, league_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    league_team_div: dict[int, str] = {
        int(tid): str(dn or "").strip()
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    rows: list[dict[str, Any]] = []
    for lg in (
        m.LeagueGame.objects.filter(
            league_id=int(league_id),
            game__team1_score__isnull=False,
            game__team2_score__isnull=False,
        )
        .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
        .select_related("game", "game__game_type")
    ):
        g = lg.game
        t1_id = int(g.team1_id)
        t2_id = int(g.team2_id)
        rows.append(
            {
                "team1_id": t1_id,
                "team2_id": t2_id,
                "team1_score": g.team1_score,
                "team2_score": g.team2_score,
                "is_final": bool(g.is_final),
                "league_division_name": lg.division_name,
                "game_type_name": (g.game_type.name if g.game_type else None),
                "team1_league_division_name": league_team_div.get(t1_id),
                "team2_league_division_name": league_team_div.get(t2_id),
            }
        )
    wins = losses = ties = gf = ga = 0
    swins = slosses = sties = sgf = sga = 0

    def _is_cross_division_non_external_game(r: dict) -> bool:
        """
        Ignore TimeToScore cross-division games (e.g. 12AA vs 12A) when both teams have a known
        non-External league division. External games (or unknown opponent division) are kept.
        """
        d1 = str(r.get("team1_league_division_name") or "").strip()
        d2 = str(r.get("team2_league_division_name") or "").strip()
        if not d1 or not d2:
            return False
        if is_external_division_name(d1) or is_external_division_name(d2):
            return False
        ld = str(r.get("league_division_name") or "").strip()
        if is_external_division_name(ld):
            return False
        return d1 != d2

    def _is_regular_game(r: dict) -> bool:
        # Only regular-season games should contribute to standings points/rankings.
        gt = str(r.get("game_type_name") or "").strip()
        if not gt or not gt.lower().startswith("regular"):
            return False
        # Any game involving an External team, or mapped to the External division, does not count for standings.
        for key in (
            "league_division_name",
            "team1_league_division_name",
            "team2_league_division_name",
        ):
            dn = str(r.get(key) or "").strip()
            if is_external_division_name(dn):
                return False
        return True

    for r in rows:
        if _is_cross_division_non_external_game(r):
            continue
        t1 = int(r["team1_id"]) == team_id
        my_score = (
            int(r["team1_score"])
            if t1
            else int(r["team2_score"]) if r["team2_score"] is not None else 0
        )
        op_score = (
            int(r["team2_score"])
            if t1
            else int(r["team1_score"]) if r["team1_score"] is not None else 0
        )
        gf += my_score
        ga += op_score
        if my_score > op_score:
            wins += 1
        elif my_score < op_score:
            losses += 1
        else:
            ties += 1

        if _is_regular_game(r):
            sgf += my_score
            sga += op_score
            if my_score > op_score:
                swins += 1
            elif my_score < op_score:
                slosses += 1
            else:
                sties += 1

    points = swins * 2 + sties * 1
    points_total = wins * 2 + ties * 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "gf": gf,
        "ga": ga,
        "points": points,
        "points_total": points_total,
        # Used only for sorting/tiebreakers in standings; display fields above include all games.
        "standings_wins": swins,
        "standings_losses": slosses,
        "standings_ties": sties,
        "standings_gf": sgf,
        "standings_ga": sga,
    }


def sort_key_team_standings(team_row: dict, stats: dict) -> tuple:
    """Standard hockey standings sort (points, wins, goal diff, goals for, goals against, name)."""
    pts = int(stats.get("points", 0))
    wins = int(stats.get("standings_wins", stats.get("wins", 0)))
    gf = int(stats.get("standings_gf", stats.get("gf", 0)))
    ga = int(stats.get("standings_ga", stats.get("ga", 0)))
    gd = gf - ga
    name = str(team_row.get("name") or "")
    return (-pts, -wins, -gd, -gf, ga, name.lower())


PLAYER_STATS_SUM_KEYS: tuple[str, ...] = (
    "goals",
    "assists",
    "pim",
    "shots",
    "sog",
    "expected_goals",
    "plus_minus",
    "giveaways",
    "turnovers_forced",
    "created_turnovers",
    "takeaways",
    "controlled_entry_for",
    "controlled_entry_against",
    "controlled_exit_for",
    "controlled_exit_against",
    "gf_counted",
    "ga_counted",
    "gt_goals",
    "gw_goals",
    "ot_goals",
    "ot_assists",
    "hits",
    "blocks",
    "toi_seconds",
    "shifts",
    "faceoff_wins",
    "faceoff_attempts",
    "goalie_saves",
    "goalie_ga",
    "goalie_sa",
)

PLAYER_STATS_DISPLAY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("gp", "GP"),
    ("toi_seconds", "TOI"),
    ("toi_seconds_per_game", "TOI per Game"),
    ("shifts", "Shifts"),
    ("shifts_per_game", "Shifts per Game"),
    ("goals", "Goals"),
    ("assists", "Assists"),
    ("points", "Points"),
    ("ppg", "PPG"),
    ("plus_minus", "Goal +/-"),
    ("plus_minus_per_game", "Goal +/- per Game"),
    ("gf_counted", "GF Counted"),
    ("gf_per_game", "GF per Game"),
    ("ga_counted", "GA Counted"),
    ("ga_per_game", "GA per Game"),
    ("shots", "Shots"),
    ("shots_per_game", "Shots per Game"),
    ("sog", "SOG"),
    ("sog_per_game", "SOG per Game"),
    ("expected_goals", "xG"),
    ("expected_goals_per_game", "xG per Game"),
    ("expected_goals_per_sog", "xG per SOG"),
    ("turnovers_forced", "Turnovers (forced)"),
    ("turnovers_forced_per_game", "Turnovers (forced) per Game"),
    ("created_turnovers", "Created Turnovers"),
    ("created_turnovers_per_game", "Created Turnovers per Game"),
    ("giveaways", "Giveaways"),
    ("giveaways_per_game", "Giveaways per Game"),
    ("takeaways", "Takeaways"),
    ("takeaways_per_game", "Takeaways per Game"),
    ("controlled_entry_for", "Controlled Entry For (On-Ice)"),
    ("controlled_entry_for_per_game", "Controlled Entry For (On-Ice) per Game"),
    ("controlled_entry_against", "Controlled Entry Against (On-Ice)"),
    ("controlled_entry_against_per_game", "Controlled Entry Against (On-Ice) per Game"),
    ("controlled_exit_for", "Controlled Exit For (On-Ice)"),
    ("controlled_exit_for_per_game", "Controlled Exit For (On-Ice) per Game"),
    ("controlled_exit_against", "Controlled Exit Against (On-Ice)"),
    ("controlled_exit_against_per_game", "Controlled Exit Against (On-Ice) per Game"),
    ("gt_goals", "GT Goals"),
    ("gw_goals", "GW Goals"),
    ("ot_goals", "OT Goals"),
    ("ot_assists", "OT Assists"),
    ("pim", "PIM"),
    ("pim_per_game", "PIM per Game"),
)

OT_ONLY_PLAYER_STATS_KEYS: frozenset[str] = frozenset({"ot_goals", "ot_assists"})

GAME_PLAYER_STATS_COLUMNS: tuple[dict[str, Any], ...] = (
    {"id": "goals", "label": "G", "keys": ("goals",)},
    {"id": "assists", "label": "A", "keys": ("assists",)},
    {"id": "points", "label": "P", "keys": ("goals", "assists"), "op": "sum"},
    {"id": "toi_seconds", "label": "TOI", "keys": ("toi_seconds",)},
    {"id": "shifts", "label": "Shifts", "keys": ("shifts",)},
    {"id": "gt_goals", "label": "GT Goals", "keys": ("gt_goals",)},
    {"id": "gw_goals", "label": "GW Goals", "keys": ("gw_goals",)},
    {"id": "ot_goals", "label": "OT Goals", "keys": ("ot_goals",)},
    {"id": "ot_assists", "label": "OT Assists", "keys": ("ot_assists",)},
    {"id": "shots", "label": "S", "keys": ("shots",)},
    {"id": "pim", "label": "PIM", "keys": ("pim",)},
    {"id": "plus_minus", "label": "+/-", "keys": ("plus_minus",)},
    {"id": "sog", "label": "SOG", "keys": ("sog",)},
    {"id": "expected_goals", "label": "xG", "keys": ("expected_goals",)},
    {"id": "ce", "label": "CE (F/A)", "keys": ("controlled_entry_for", "controlled_entry_against")},
    {"id": "cx", "label": "CX (F/A)", "keys": ("controlled_exit_for", "controlled_exit_against")},
    {"id": "give_take", "label": "Give/Take", "keys": ("giveaways", "takeaways")},
    {"id": "gfga", "label": "GF/GA", "keys": ("gf_counted", "ga_counted")},
)

# Per-game player stats columns on the game page should use the same labels as the team page.
# Keep this set intentionally small; `filter_player_stats_display_columns_for_rows()` hides any
# column where all players have 0/blank values.
GAME_PLAYER_STATS_DISPLAY_KEYS: tuple[str, ...] = (
    "goals",
    "assists",
    "points",
    "toi_seconds",
    "shifts",
    "gt_goals",
    "gw_goals",
    "ot_goals",
    "ot_assists",
    "shots",
    "pim",
    "plus_minus",
    "sog",
    "expected_goals",
    "controlled_entry_for",
    "controlled_entry_against",
    "controlled_exit_for",
    "controlled_exit_against",
    "giveaways",
    "takeaways",
    "gf_counted",
    "ga_counted",
)


def build_game_player_stats_display_columns(
    *,
    rows: list[dict[str, Any]],
    base_keys: tuple[str, ...] = GAME_PLAYER_STATS_DISPLAY_KEYS,
) -> list[dict[str, Any]]:
    """
    Return per-game player stat columns for the game page, using team-page wording.
    """
    label_by_key: dict[str, str] = {str(k): str(label) for k, label in PLAYER_STATS_DISPLAY_COLUMNS}
    base_cols: list[tuple[str, str]] = []
    for k in base_keys:
        label = label_by_key.get(str(k))
        if not label:
            continue
        base_cols.append((str(k), label))
    filtered = filter_player_stats_display_columns_for_rows(tuple(base_cols), rows)
    return [{"key": str(k), "label": str(label), "show_count": False} for k, label in filtered]


_PLAYER_STATS_IDENTITY_HEADERS: frozenset[str] = frozenset(
    {
        "player",
        "name",
        "jersey #",
        "jersey",
        "jersey no",
        "jersey number",
        "pos",
        "position",
    }
)

_PLAYER_STATS_HEADER_TO_DB_KEY: dict[str, str] = {
    # Common short headers
    "g": "goals",
    "a": "assists",
    "goals": "goals",
    "assists": "assists",
    "shots": "shots",
    "pim": "pim",
    "hits": "hits",
    "blocks": "blocks",
    "faceoff wins": "faceoff_wins",
    "faceoffs won": "faceoff_wins",
    "faceoff attempts": "faceoff_attempts",
    "faceoffs": "faceoff_attempts",
    "saves": "goalie_saves",
    "goalie saves": "goalie_saves",
    "ga": "goalie_ga",
    "goalie ga": "goalie_ga",
    "sa": "goalie_sa",
    "goalie sa": "goalie_sa",
    "sog": "sog",
    "xg": "expected_goals",
    "giveaways": "giveaways",
    "turnovers (forced)": "turnovers_forced",
    "created turnovers": "created_turnovers",
    "takeaways": "takeaways",
    "controlled entry for (on-ice)": "controlled_entry_for",
    "controlled entry against (on-ice)": "controlled_entry_against",
    "controlled exit for (on-ice)": "controlled_exit_for",
    "controlled exit against (on-ice)": "controlled_exit_against",
    "gt goals": "gt_goals",
    "gw goals": "gw_goals",
    "ot goals": "ot_goals",
    "ot assists": "ot_assists",
    "plus minus": "plus_minus",
    "goal +/-": "plus_minus",
    "gf counted": "gf_counted",
    "ga counted": "ga_counted",
}


def _normalize_header_for_lookup(h: str) -> str:
    return str(h or "").strip().lower()


def _normalize_column_id(h: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(h or "").strip().lower()).strip("_") or "col"
    return base


def _parse_int_from_cell_text(s: Any) -> int:
    """
    Parse integers from a cell like "2", "1/2", "1 / 2", returning max part.
    """
    if s is None:
        return 0
    ss = str(s).strip()
    if not ss:
        return 0
    parts = [p.strip() for p in re.split(r"\s*/\s*", ss) if p.strip()]
    out = 0
    for p in parts:
        try:
            out = max(out, int(float(p)))
        except Exception:
            continue
    return out


def _build_game_player_stats_table_from_imported_csv(
    *,
    players: list[dict[str, Any]],
    stats_by_pid: dict[int, dict[str, Any]],
    imported_csv_text: str,
    prefer_db_stats_for_keys: Optional[set[str]] = None,
) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, str]], dict[int, dict[str, bool]], Optional[str]
]:
    """
    Display-first per-game table: preserve imported CSV columns (minus identity fields),
    with optional DB merge/conflict highlighting for known numeric stats.
    """
    try:
        headers, rows = parse_events_csv(imported_csv_text)
    except Exception as e:  # noqa: BLE001
        return (
            list(GAME_PLAYER_STATS_COLUMNS),
            {},
            {},
            f"Unable to parse imported player_stats_csv: {e}",
        )

    if not headers:
        return [], {}, {}, "Imported player_stats_csv has no headers"

    # Never show shift/TOI/per-game/per-shift columns in the web UI, even if older data is stored.
    headers, rows = filter_single_game_player_stats_csv(headers, rows)
    if not headers:
        return [], {}, {}, "Imported player_stats_csv has no displayable columns"

    team_ids = sorted(
        {int(p.get("team_id") or 0) for p in (players or []) if p.get("team_id") is not None}
    )
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    for p in players or []:
        try:
            pid = int(p.get("id"))
            tid = int(p.get("team_id") or 0)
        except Exception:
            continue
        jersey_norm = normalize_jersey_number(p.get("jersey_number"))
        if jersey_norm:
            jersey_to_player_ids.setdefault((tid, jersey_norm), []).append(pid)
        name_norm = normalize_player_name(str(p.get("name") or ""))
        if name_norm:
            name_to_player_ids.setdefault((tid, name_norm), []).append(pid)

    def _resolve_player_id(jersey_norm: Optional[str], name_norm: str) -> Optional[int]:
        candidates: list[int] = []
        for tid in team_ids:
            if jersey_norm:
                candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        candidates = []
        for tid in team_ids:
            candidates.extend(name_to_player_ids.get((tid, name_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        return None

    def _first_non_empty(d: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = str(d.get(k) or "").strip()
            if v:
                return v
        return ""

    imported_row_by_pid: dict[int, dict[str, str]] = {}
    for r in rows:
        jersey_raw = _first_non_empty(
            r,
            (
                "Jersey #",
                "Jersey",
                "Jersey No",
                "Jersey Number",
            ),
        )
        jersey_norm = normalize_jersey_number(jersey_raw) if jersey_raw else None
        player_name = _first_non_empty(r, ("Player", "Name"))
        name_part = player_name
        if jersey_norm is None:
            m = re.match(r"^\s*(\d+)\s+(.*)$", player_name)
            if m:
                jersey_norm = normalize_jersey_number(m.group(1))
                name_part = m.group(2).strip()
        name_norm = normalize_player_name(name_part)
        pid = _resolve_player_id(jersey_norm, name_norm)
        if pid is None:
            continue
        imported_row_by_pid[int(pid)] = dict(r)

    # Build columns in imported header order (minus identity headers).
    columns: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for h in headers:
        key = _normalize_header_for_lookup(h)
        if key in _PLAYER_STATS_IDENTITY_HEADERS:
            continue
        col_id = _PLAYER_STATS_HEADER_TO_DB_KEY.get(key) or _normalize_column_id(h)
        if col_id in used_ids:
            # De-dupe while preserving order.
            i = 2
            while f"{col_id}_{i}" in used_ids:
                i += 1
            col_id = f"{col_id}_{i}"
        used_ids.add(col_id)
        columns.append(
            {
                "id": col_id,
                "label": str(h),
                "header": str(h),
                "db_key": _PLAYER_STATS_HEADER_TO_DB_KEY.get(key),
            }
        )

    all_pids = [int(p.get("id")) for p in (players or []) if p.get("id") is not None]
    cell_text_by_pid: dict[int, dict[str, str]] = {pid: {} for pid in all_pids}
    cell_conf_by_pid: dict[int, dict[str, bool]] = {pid: {} for pid in all_pids}

    # Populate cells (imported first; merge with DB for known numeric keys).
    for pid in all_pids:
        db_row = stats_by_pid.get(pid) or {}
        imp_row = imported_row_by_pid.get(pid) or {}
        for col in columns:
            cid = str(col["id"])
            header = str(col.get("header") or "")
            db_key = col.get("db_key")
            raw_v = str(imp_row.get(header) or "").strip()
            if db_key:
                if prefer_db_stats_for_keys and str(db_key) in prefer_db_stats_for_keys:
                    v, s, is_conf = _merge_stat_values(db_row.get(str(db_key)), None)
                else:
                    v, s, is_conf = _merge_stat_values(db_row.get(str(db_key)), raw_v)
                cell_text_by_pid[pid][cid] = s
                cell_conf_by_pid[pid][cid] = bool(is_conf)
            else:
                cell_text_by_pid[pid][cid] = raw_v
                cell_conf_by_pid[pid][cid] = False

    # Hide columns that are entirely blank, and OT-only columns if all zero/blank.
    visible: list[dict[str, Any]] = []
    for col in columns:
        cid = str(col["id"])
        label_l = str(col.get("label") or "").strip().lower()
        vals = [cell_text_by_pid.get(pid, {}).get(cid, "") for pid in all_pids]
        if all(_is_blank_stat(v) or str(v).strip() == "" for v in vals):
            continue
        if label_l.startswith("ot ") or label_l in {"ot goals", "ot assists"}:
            if all(_is_zero_or_blank_stat(_parse_int_from_cell_text(v)) for v in vals):
                continue
        visible.append(col)

    # Add derived Points column (Goals + Assists) when those columns exist.
    goals_col_id = next(
        (str(c["id"]) for c in visible if str(c.get("db_key") or "") == "goals"), None
    )
    assists_col_id = next(
        (str(c["id"]) for c in visible if str(c.get("db_key") or "") == "assists"), None
    )
    if goals_col_id and assists_col_id:
        points_id = "points"
        if any(str(c.get("id")) == points_id for c in visible):
            points_id = "points_2"

        for pid in all_pids:
            g_txt = str(cell_text_by_pid.get(pid, {}).get(goals_col_id, "") or "")
            a_txt = str(cell_text_by_pid.get(pid, {}).get(assists_col_id, "") or "")
            any_part = bool(g_txt.strip() or a_txt.strip())
            if any_part:
                pts = _parse_int_from_cell_text(g_txt) + _parse_int_from_cell_text(a_txt)
                cell_text_by_pid[pid][points_id] = str(int(pts))
            else:
                cell_text_by_pid[pid][points_id] = ""
            cell_conf_by_pid[pid][points_id] = bool(
                cell_conf_by_pid.get(pid, {}).get(goals_col_id)
                or cell_conf_by_pid.get(pid, {}).get(assists_col_id)
            )

        pts_vals = [cell_text_by_pid.get(pid, {}).get(points_id, "") for pid in all_pids]
        if not all(_is_blank_stat(v) or str(v).strip() == "" for v in pts_vals):
            insert_at = next(
                (i + 1 for i, c in enumerate(visible) if str(c.get("id")) == assists_col_id), None
            )
            if insert_at is None:
                insert_at = 0
            visible.insert(
                int(insert_at), {"id": points_id, "label": "P", "header": "P", "db_key": None}
            )

    # Rebuild cell dicts to only include visible columns.
    vis_ids = {str(c["id"]) for c in visible}
    cell_text_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_text_by_pid.items()
    }
    cell_conf_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_conf_by_pid.items()
    }
    return visible, cell_text_by_pid, cell_conf_by_pid, None


def _int0(v: Any) -> int:
    try:
        if v is None:
            return 0
        return int(float(str(v)))
    except Exception:
        return 0


def _rate_or_none(numer: float, denom: float) -> Optional[float]:
    try:
        if denom <= 0:
            return None
        return float(numer) / float(denom)
    except Exception:
        return None


def compute_player_display_stats(sums: dict[str, Any]) -> dict[str, Any]:
    gp = _int0(sums.get("gp"))
    goals = _int0(sums.get("goals"))
    assists = _int0(sums.get("assists"))
    points = goals + assists

    toi_seconds = _int0(sums.get("toi_seconds"))
    shifts = _int0(sums.get("shifts"))

    shots = _int0(sums.get("shots"))
    sog = _int0(sums.get("sog"))
    xg = _int0(sums.get("expected_goals"))
    pim = _int0(sums.get("pim"))
    plus_minus = _int0(sums.get("plus_minus"))
    gf = _int0(sums.get("gf_counted"))
    ga = _int0(sums.get("ga_counted"))
    giveaways = _int0(sums.get("giveaways"))
    takeaways = _int0(sums.get("takeaways"))
    turnovers_forced = _int0(sums.get("turnovers_forced"))
    created_turnovers = _int0(sums.get("created_turnovers"))
    ce_for = _int0(sums.get("controlled_entry_for"))
    ce_against = _int0(sums.get("controlled_entry_against"))
    cx_for = _int0(sums.get("controlled_exit_for"))
    cx_against = _int0(sums.get("controlled_exit_against"))

    faceoff_wins = _int0(sums.get("faceoff_wins"))
    faceoff_attempts = _int0(sums.get("faceoff_attempts"))
    goalie_saves = _int0(sums.get("goalie_saves"))
    goalie_sa = _int0(sums.get("goalie_sa"))

    out: dict[str, Any] = dict(sums)
    # Stat implications for display/aggregation:
    #   Goals ⊆ xG ⊆ SOG ⊆ Shots
    xg = max(xg, goals)
    sog = max(sog, xg)
    shots = max(shots, sog)
    out["expected_goals"] = xg
    out["sog"] = sog
    out["shots"] = shots
    out["gp"] = gp
    out["points"] = points
    out["ppg"] = _rate_or_none(points, gp)

    # Per-game rates.
    out["toi_seconds_per_game"] = int(round(float(toi_seconds) / float(gp))) if gp > 0 else None
    out["shifts_per_game"] = _rate_or_none(shifts, gp)
    out["shots_per_game"] = _rate_or_none(shots, gp)
    out["sog_per_game"] = _rate_or_none(sog, gp)
    out["expected_goals_per_game"] = _rate_or_none(xg, gp)
    out["plus_minus_per_game"] = _rate_or_none(plus_minus, gp)
    out["gf_per_game"] = _rate_or_none(gf, gp)
    out["ga_per_game"] = _rate_or_none(ga, gp)
    out["giveaways_per_game"] = _rate_or_none(giveaways, gp)
    out["takeaways_per_game"] = _rate_or_none(takeaways, gp)
    out["turnovers_forced_per_game"] = _rate_or_none(turnovers_forced, gp)
    out["created_turnovers_per_game"] = _rate_or_none(created_turnovers, gp)
    out["controlled_entry_for_per_game"] = _rate_or_none(ce_for, gp)
    out["controlled_entry_against_per_game"] = _rate_or_none(ce_against, gp)
    out["controlled_exit_for_per_game"] = _rate_or_none(cx_for, gp)
    out["controlled_exit_against_per_game"] = _rate_or_none(cx_against, gp)
    out["pim_per_game"] = _rate_or_none(pim, gp)
    out["hits_per_game"] = _rate_or_none(_int0(sums.get("hits")), gp)
    out["blocks_per_game"] = _rate_or_none(_int0(sums.get("blocks")), gp)

    out["expected_goals_per_sog"] = _rate_or_none(xg, sog)
    out["faceoff_pct"] = _rate_or_none(faceoff_wins, faceoff_attempts)
    out["goalie_sv_pct"] = _rate_or_none(goalie_saves, goalie_sa)
    return out


def _classify_coach_position(pos: Any) -> Optional[str]:
    """
    Returns "HC" or "AC" if position indicates a coach; otherwise None.
    """
    p = str(pos or "").strip().upper()
    if p in {"HC", "HEAD COACH"}:
        return "HC"
    if p in {"AC", "ASSISTANT COACH"}:
        return "AC"
    return None


def _classify_roster_role(p: dict[str, Any]) -> Optional[str]:
    """
    Returns "HC", "AC", or "G" when the player dict is clearly a coach/goalie.
    Falls back to None for skaters/unknown.
    """
    jersey = str(p.get("jersey_number") or "").strip().upper()
    if jersey in {"HC", "HEAD COACH"}:
        return "HC"
    if jersey in {"AC", "ASSISTANT COACH"}:
        return "AC"
    if jersey in {"G", "GOALIE", "GOALTENDER"}:
        return "G"

    pos = p.get("position")
    role = _classify_coach_position(pos)
    if role:
        return role
    if _is_goalie_position(pos):
        return "G"

    name = str(p.get("name") or "").strip()
    if not name:
        return None
    name_up = name.upper()

    # Some imports encode coach role in the *name* field (position can be blank).
    if (
        re.match(r"^\s*HC\b", name_up)
        or re.search(r"\bHEAD\s+COACH\b", name_up)
        or re.search(r"\(HC\)", name_up)
    ):
        return "HC"
    if (
        re.match(r"^\s*AC\b", name_up)
        or re.search(r"\bASSISTANT\s+COACH\b", name_up)
        or re.search(r"\(AC\)", name_up)
    ):
        return "AC"

    # Conservative goalie hint when position is missing.
    if (
        re.search(r"\bGOALIE\b", name_up)
        or re.search(r"\bGOALTENDER\b", name_up)
        or re.search(r"\(G\)", name_up)
    ):
        return "G"

    return None


def _is_goalie_position(pos: Any) -> bool:
    p = str(pos or "").strip().upper()
    if not p:
        return False
    # Normalize common variants.
    p = re.sub(r"[()]", "", p).strip()
    if p in {"G", "GOALIE", "GOALTENDER"}:
        return True
    # Allow things like "G1", "G2", "G - Starter".
    if p.startswith("G"):
        return True
    return False


def split_players_and_coaches(
    players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Separate coaches (HC/AC) from players so they don't appear in player stats lists.
    Returns: (players_only, head_coaches, assistant_coaches)
    """
    players_only: list[dict[str, Any]] = []
    head_coaches: list[dict[str, Any]] = []
    assistant_coaches: list[dict[str, Any]] = []
    for p in players or []:
        role = _classify_roster_role(p)
        if role == "HC":
            head_coaches.append(p)
        elif role == "AC":
            assistant_coaches.append(p)
        else:
            players_only.append(p)
    return players_only, head_coaches, assistant_coaches


def split_roster(
    players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split a team roster into:
      - skaters (non-coach, non-goalie)
      - goalies
      - head coaches
      - assistant coaches

    This is used to keep coaches/goalies out of *stats* tables while still showing
    them in roster tables.
    """
    skaters: list[dict[str, Any]] = []
    goalies: list[dict[str, Any]] = []
    head_coaches: list[dict[str, Any]] = []
    assistant_coaches: list[dict[str, Any]] = []
    for p in players or []:
        role = _classify_roster_role(p)
        if role == "HC":
            head_coaches.append(p)
            continue
        if role == "AC":
            assistant_coaches.append(p)
            continue
        if role == "G":
            goalies.append(p)
            continue
        skaters.append(p)
    return skaters, goalies, head_coaches, assistant_coaches


def _is_blank_stat(v: Any) -> bool:
    return v is None or v == ""


def _is_zero_or_blank_stat(v: Any) -> bool:
    if _is_blank_stat(v):
        return True
    if isinstance(v, str):
        s = v.strip()
        if "/" in s:
            parts = s.split("/")
            if parts and all(_is_zero_or_blank_stat(part) for part in parts):
                return True
    try:
        return float(v) == 0.0  # type: ignore[arg-type]
    except Exception:
        return False


def filter_player_stats_display_columns_for_rows(
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """
    Hide:
      - Any column that is entirely blank (missing data)
      - Any column where all values are 0/blank
    """
    if not columns:
        return columns
    out: list[tuple[str, str]] = []
    gf_counted_all_zero: Optional[bool] = None
    ga_counted_all_zero: Optional[bool] = None
    for k, label in columns:
        vals = [r.get(k) for r in (rows or [])]
        all_zero = all(_is_zero_or_blank_stat(v) for v in vals)
        if k == "gf_counted":
            gf_counted_all_zero = all_zero
        elif k == "ga_counted":
            ga_counted_all_zero = all_zero
        if all_zero:
            continue
        out.append((k, label))
    if gf_counted_all_zero and ga_counted_all_zero:
        out = [
            (key, label)
            for key, label in out
            if key not in {"plus_minus", "plus_minus_per_game", "gf_counted", "ga_counted"}
        ]
    return tuple(out)


def _merge_stat_values(db_v: Any, imported_v: Any) -> tuple[Optional[int], str, bool]:
    """
    Returns: (merged_numeric_value_or_None, display_string, is_conflict)
    """

    def _to_int(v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except Exception:
            try:
                return int(float(str(v)))
            except Exception:
                return None

    a = _to_int(db_v)
    b = _to_int(imported_v)
    if a is None and b is None:
        return None, "", False
    if a is None:
        return b, str(b), False
    if b is None:
        return a, str(a), False
    if a == b:
        return a, str(a), False

    # Treat a single 0 vs non-zero as "missing" from one source (common in partial imports).
    if a == 0 and b != 0:
        return b, str(b), False
    if b == 0 and a != 0:
        return a, str(a), False

    return a, f"{a}/{b}", True


def _map_imported_shift_stats_to_player_ids(
    *,
    players: list[dict[str, Any]],
    imported_csv_text: Optional[str],
) -> tuple[dict[int, dict[str, Any]], Optional[str]]:
    """
    Returns (imported_stats_by_pid, parse_warning).
    """
    if not imported_csv_text or not str(imported_csv_text).strip():
        return {}, None
    try:
        parsed_rows = parse_shift_stats_player_stats_csv(str(imported_csv_text))
    except Exception as e:  # noqa: BLE001
        return {}, f"Unable to parse imported player_stats_csv: {e}"

    team_ids = sorted(
        {int(p.get("team_id") or 0) for p in (players or []) if p.get("team_id") is not None}
    )
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    for p in players or []:
        try:
            pid = int(p.get("id"))
            tid = int(p.get("team_id") or 0)
        except Exception:
            continue
        jersey_norm = normalize_jersey_number(p.get("jersey_number"))
        if jersey_norm:
            jersey_to_player_ids.setdefault((tid, jersey_norm), []).append(pid)
        name_norm = normalize_player_name(str(p.get("name") or ""))
        if name_norm:
            name_to_player_ids.setdefault((tid, name_norm), []).append(pid)

    def _resolve_player_id(jersey_norm: Optional[str], name_norm: str) -> Optional[int]:
        candidates: list[int] = []
        for tid in team_ids:
            if jersey_norm:
                candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        candidates = []
        for tid in team_ids:
            candidates.extend(name_to_player_ids.get((tid, name_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        return None

    imported_by_pid: dict[int, dict[str, Any]] = {}
    for row in parsed_rows:
        jersey_norm = row.get("jersey_number")
        name_norm = row.get("name_norm") or ""
        pid = _resolve_player_id(jersey_norm, name_norm)
        if pid is None:
            continue
        imported_by_pid[int(pid)] = dict(row.get("stats") or {})
    return imported_by_pid, None


def build_game_player_stats_table(
    *,
    players: list[dict[str, Any]],
    stats_by_pid: dict[int, dict[str, Any]],
    imported_csv_text: Optional[str],
    prefer_db_stats_for_keys: Optional[set[str]] = None,
) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, str]], dict[int, dict[str, bool]], Optional[str]
]:
    """
    Build a merged (DB + imported CSV) per-game player stats table.
    Returns: (visible_columns, cell_text_by_pid, cell_conflict_by_pid, imported_parse_warning)
    """
    if imported_csv_text and str(imported_csv_text).strip():
        return _build_game_player_stats_table_from_imported_csv(
            players=players,
            stats_by_pid=stats_by_pid,
            imported_csv_text=str(imported_csv_text),
            prefer_db_stats_for_keys=prefer_db_stats_for_keys,
        )

    imported_by_pid, imported_warning = _map_imported_shift_stats_to_player_ids(
        players=players, imported_csv_text=imported_csv_text
    )

    all_pids = [int(p.get("id")) for p in (players or []) if p.get("id") is not None]

    merged_vals: dict[int, dict[str, Optional[int]]] = {pid: {} for pid in all_pids}
    merged_disp: dict[int, dict[str, str]] = {pid: {} for pid in all_pids}
    merged_conf: dict[int, dict[str, bool]] = {pid: {} for pid in all_pids}

    all_keys: set[str] = set()
    for c in GAME_PLAYER_STATS_COLUMNS:
        for k in c.get("keys") or ():
            all_keys.add(str(k))

    for pid in all_pids:
        db_row = stats_by_pid.get(pid) or {}
        imp_row = imported_by_pid.get(pid) or {}
        for k in all_keys:
            v, s, is_conf = _merge_stat_values(db_row.get(k), imp_row.get(k))
            merged_vals[pid][k] = v
            merged_disp[pid][k] = s
            merged_conf[pid][k] = bool(is_conf)

    duration_keys: frozenset[str] = frozenset(
        {
            "toi_seconds",
            "video_toi_seconds",
            "sb_avg_shift_seconds",
            "sb_median_shift_seconds",
            "sb_longest_shift_seconds",
            "sb_shortest_shift_seconds",
        }
    )

    def _fmt_duration_display(raw: str) -> str:
        """
        Format merge display values for duration stats.
        Inputs are typically seconds like "754" or conflict strings like "754/760".
        """
        s = str(raw or "").strip()
        if not s:
            return ""
        if "/" in s:
            parts = [p.strip() for p in s.split("/", 1)]
            if len(parts) == 2:
                a = format_seconds_to_mmss_or_hhmmss(parts[0])
                b = format_seconds_to_mmss_or_hhmmss(parts[1])
                if a and b:
                    return f"{a}/{b}"
        out = format_seconds_to_mmss_or_hhmmss(s)
        return out or s

    for pid in all_pids:
        for k in duration_keys:
            disp = merged_disp[pid].get(k)
            if disp:
                merged_disp[pid][k] = _fmt_duration_display(disp)

    visible_columns: list[dict[str, Any]] = []
    for col in GAME_PLAYER_STATS_COLUMNS:
        keys = [str(k) for k in (col.get("keys") or ())]
        if keys and set(keys).issubset(OT_ONLY_PLAYER_STATS_KEYS):
            if all(
                all(_is_zero_or_blank_stat(merged_vals[pid].get(k)) for k in keys)
                for pid in all_pids
            ):
                continue
        if keys and all(all(merged_vals[pid].get(k) is None for k in keys) for pid in all_pids):
            continue
        visible_columns.append(dict(col))

    cell_text_by_pid: dict[int, dict[str, str]] = {}
    cell_conflict_by_pid: dict[int, dict[str, bool]] = {}
    for pid in all_pids:
        out_text: dict[str, str] = {}
        out_conf: dict[str, bool] = {}
        for col in visible_columns:
            col_id = str(col.get("id"))
            keys = [str(k) for k in (col.get("keys") or ())]
            op = str(col.get("op") or "").strip().lower()
            parts = [merged_disp[pid].get(k, "") for k in keys]
            any_part = any(str(p).strip() for p in parts)
            if len(keys) == 1:
                out_text[col_id] = parts[0] if parts else ""
                out_conf[col_id] = bool(keys and merged_conf[pid].get(keys[0]))
            else:
                if op == "sum":
                    if any_part:
                        out_text[col_id] = str(sum(_int0(merged_vals[pid].get(k)) for k in keys))
                    else:
                        out_text[col_id] = ""
                    out_conf[col_id] = any(bool(merged_conf[pid].get(k)) for k in keys)
                else:
                    if any_part:
                        filled = [p if str(p).strip() else "0" for p in parts]
                        out_text[col_id] = " / ".join(filled)
                    else:
                        out_text[col_id] = ""
                    out_conf[col_id] = any(bool(merged_conf[pid].get(k)) for k in keys)
        cell_text_by_pid[pid] = out_text
        cell_conflict_by_pid[pid] = out_conf

    # Hide any columns that are entirely blank.
    filtered_cols: list[dict[str, Any]] = []
    for col in visible_columns:
        cid = str(col.get("id"))
        vals = [str(cell_text_by_pid.get(pid, {}).get(cid, "") or "") for pid in all_pids]
        if all(_is_blank_stat(v) or v.strip() == "" for v in vals):
            continue
        filtered_cols.append(col)
    visible_columns = filtered_cols
    vis_ids = {str(c.get("id")) for c in visible_columns}
    cell_text_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_text_by_pid.items()
    }
    cell_conflict_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids}
        for pid, row in cell_conflict_by_pid.items()
    }

    return visible_columns, cell_text_by_pid, cell_conflict_by_pid, imported_warning


def _empty_player_display_stats(player_id: int) -> dict[str, Any]:
    base: dict[str, Any] = {"player_id": int(player_id), "gp": 0}
    for k in PLAYER_STATS_SUM_KEYS:
        base[k] = 0
    return compute_player_display_stats(base)


def build_player_stats_table_rows(
    players: list[dict[str, Any]],
    stats_by_player_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in players or []:
        pid = int(p.get("id"))
        s = stats_by_player_id.get(pid) or _empty_player_display_stats(pid)
        row = {
            "player_id": pid,
            "jersey_number": str(p.get("jersey_number") or ""),
            "name": str(p.get("name") or ""),
            "position": str(p.get("position") or ""),
        }
        for k, _label in PLAYER_STATS_DISPLAY_COLUMNS:
            row[k] = s.get(k)
        rows.append(row)
    return rows


def compute_recent_player_totals_from_rows(
    *,
    schedule_games: list[dict[str, Any]],
    player_stats_rows: list[dict[str, Any]],
    n: int,
) -> dict[int, dict[str, Any]]:
    """
    Compute per-player totals using each player's most recent N games (as defined by `schedule_games` order).
    """
    n_i = max(1, min(10, int(n)))
    order_idx: dict[int, int] = {}
    for idx, g in enumerate(schedule_games or []):
        try:
            order_idx[int(g.get("id"))] = int(idx)
        except Exception:
            continue

    rows_by_player: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for r in player_stats_rows or []:
        try:
            gid = int(r.get("game_id"))
            pid = int(r.get("player_id"))
        except Exception:
            continue
        idx = order_idx.get(gid)
        if idx is None:
            continue
        rows_by_player.setdefault(pid, []).append((idx, r))

    out: dict[int, dict[str, Any]] = {}
    for pid, items in rows_by_player.items():
        items.sort(key=lambda t: t[0], reverse=True)
        chosen = items[:n_i]
        sums: dict[str, Any] = {"player_id": int(pid), "gp": len(chosen)}
        for k in PLAYER_STATS_SUM_KEYS:
            sums[k] = sum(_int0(rr.get(k)) for _idx, rr in chosen)
        out[int(pid)] = compute_player_display_stats(sums)
    return out


def _dedupe_preserve_str(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items or []:
        s = str(it or "").strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _game_type_label_for_row(game_row: dict[str, Any]) -> str:
    gt = str(game_row.get("game_type_name") or "").strip()
    if not gt and is_external_division_name(game_row.get("division_name")):
        return "Tournament"
    return gt or "Unknown"


def _game_has_recorded_result(game_row: dict[str, Any]) -> bool:
    return (
        (game_row.get("team1_score") is not None)
        or (game_row.get("team2_score") is not None)
        or bool(game_row.get("is_final"))
    )


def _parse_selected_game_type_labels(
    *,
    available: list[str],
    args: Any,
) -> Optional[set[str]]:
    """
    Parse a game type filter from request args. Returns None to represent "no filtering" (all types).
    """
    avail = _dedupe_preserve_str(list(available or []))
    if not avail:
        return None
    raw: list[str] = []
    try:
        raw.extend(list(args.getlist("gt") or []))
    except Exception:
        pass
    try:
        v = args.get("gt")
        if v and isinstance(v, str) and "," in v:
            raw.extend([p.strip() for p in v.split(",") if p.strip()])
    except Exception:
        pass
    selected = _dedupe_preserve_str(raw)
    if not selected:
        return None
    avail_map = {a.casefold(): a for a in avail}
    chosen: set[str] = set()
    for s in selected:
        v = avail_map.get(s.casefold())
        if v:
            chosen.add(v)
    if not chosen or len(chosen) == len(avail):
        return None
    return chosen


def _aggregate_player_totals_from_rows(
    *,
    player_stats_rows: list[dict[str, Any]],
    allowed_game_ids: set[int],
) -> dict[int, dict[str, Any]]:
    sums_by_pid: dict[int, dict[str, Any]] = {}
    gp_by_pid: dict[int, int] = {}
    for r in player_stats_rows or []:
        if not isinstance(r, dict):
            continue
        try:
            gid = int(r.get("game_id"))
            pid = int(r.get("player_id"))
        except Exception:
            continue
        if gid not in allowed_game_ids:
            continue
        gp_by_pid[pid] = gp_by_pid.get(pid, 0) + 1
        acc = sums_by_pid.setdefault(pid, {"player_id": int(pid)})
        for k in PLAYER_STATS_SUM_KEYS:
            acc[k] = _int0(acc.get(k)) + _int0(r.get(k))
    out: dict[int, dict[str, Any]] = {}
    for pid, base in sums_by_pid.items():
        base["gp"] = int(gp_by_pid.get(pid, 0))
        out[int(pid)] = compute_player_display_stats(dict(base))
    return out


def _player_stats_required_sum_keys_for_display_key(col_key: str) -> tuple[str, ...]:
    k = str(col_key or "").strip()
    if not k:
        return tuple()
    if k in set(PLAYER_STATS_SUM_KEYS):
        return (k,)
    if k == "gp":
        return tuple()
    if k in {"points", "ppg"}:
        return ("goals", "assists")
    if k.endswith("_per_game"):
        base = k[: -len("_per_game")]
        if base in set(PLAYER_STATS_SUM_KEYS):
            return (base,)
    if k == "expected_goals_per_sog":
        return ("expected_goals", "sog")
    return tuple()


def _compute_team_player_stats_coverage(
    *,
    player_stats_rows: list[dict[str, Any]],
    eligible_game_ids: list[int],
) -> tuple[dict[str, int], int]:
    """
    Returns (coverage_counts_by_display_key, total_eligible_games).
    """
    eligible_set = {int(gid) for gid in (eligible_game_ids or [])}
    total = len(eligible_set)
    if total <= 0:
        return {}, 0

    has_any_ps: set[int] = set()
    has_key_by_game: dict[int, set[str]] = {}
    for r in player_stats_rows or []:
        if not isinstance(r, dict):
            continue
        try:
            gid = int(r.get("game_id"))
        except Exception:
            continue
        if gid not in eligible_set:
            continue
        has_any_ps.add(gid)
        keys = has_key_by_game.setdefault(gid, set())
        for sk in PLAYER_STATS_SUM_KEYS:
            if r.get(sk) is not None:
                keys.add(sk)

    counts: dict[str, int] = {"gp": len(has_any_ps)}
    for display_key, _label in PLAYER_STATS_DISPLAY_COLUMNS:
        dk = str(display_key)
        if dk in counts:
            continue
        req = _player_stats_required_sum_keys_for_display_key(dk)
        if not req:
            counts[dk] = len(has_any_ps)
            continue
        n = 0
        for gid in eligible_set:
            present = has_key_by_game.get(gid) or set()
            if all(rk in present for rk in req):
                n += 1
        counts[dk] = int(n)
    return counts, total


def _annotate_player_stats_column_labels(
    *,
    columns: list[tuple[str, str]],
    coverage_counts: dict[str, int],
    total_games: int,
) -> list[tuple[str, str]]:
    # Backwards-compatible wrapper: keep older call sites working.
    out: list[tuple[str, str]] = []
    for c in _player_stats_columns_with_coverage(
        columns=columns, coverage_counts=coverage_counts, total_games=total_games
    ):
        out.append((str(c["key"]), str(c["label"])))
    return out


def _player_stats_columns_with_coverage(
    *,
    columns: list[tuple[str, str]],
    coverage_counts: dict[str, int],
    total_games: int,
) -> list[dict[str, Any]]:
    """
    Return columns as dicts with optional coverage sublabel info for UI rendering.
    """
    out: list[dict[str, Any]] = []
    for k, label in columns or []:
        key = str(k)
        n = coverage_counts.get(key, total_games)
        show = bool(total_games > 0 and n != total_games)
        out.append(
            {
                "key": key,
                "label": str(label),
                "n_games": int(n) if n is not None else 0,
                "total_games": int(total_games) if total_games is not None else 0,
                "show_count": show,
            }
        )
    return out


def canon_event_source_key(raw: Any) -> str:
    """
    Return a canonical event source key used for ordering and display.

    Keys are intentionally coarse so we don't leak per-game/per-import labels into the UI.
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    sl = s.casefold()
    if sl in {"timetoscore", "t2s", "tts"}:
        return "timetoscore"
    if sl == "primary" or sl.startswith("parse_stats_inputs"):
        return "primary"
    if sl.startswith("parse_shift_spreadsheet"):
        return "primary"
    if sl == "shift_package":
        return "shift_package"
    if sl == "long":
        return "long"
    if sl == "goals":
        return "goals"
    return ""


def event_source_rank(raw: Any) -> int:
    """
    Rank event sources by preference for de-duping/selection.

    Lower is better.
    """
    k = canon_event_source_key(raw)
    if k == "timetoscore":
        return 0
    if k in {"primary", "shift_package"}:
        return 1
    if k == "long":
        return 2
    if k == "goals":
        return 3
    return 9


def _canon_source_label_for_ui(raw: Any) -> str:
    k = canon_event_source_key(raw)
    if k == "timetoscore":
        return "TimeToScore"
    if k == "long":
        return "Long"
    if k == "primary":
        return "Primary"
    if k == "shift_package":
        return "Shift Package"
    if k == "goals":
        return "Goals"
    return ""


def _compute_team_player_stats_sources(
    db_conn,
    *,
    eligible_game_ids: list[int],
) -> list[str]:
    gids = [int(g) for g in (eligible_game_ids or []) if int(g) > 0]
    if not gids:
        return []
    out: list[str] = []
    seen: set[str] = set()
    del db_conn
    try:
        _django_orm, m = _orm_modules()
    except Exception:
        return out

    def _add(src: Any) -> None:
        s = _canon_source_label_for_ui(src)
        if not s:
            return
        k = s.casefold()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    # Prefer scanning event row sources (multi-valued Source column semantics).
    try:
        for chunk in _django_orm.iter_chunks(gids, 200):
            sources = list(
                m.HkyGameEventRow.objects.filter(game_id__in=chunk).values_list("source", flat=True)
            )
            for src in sources:
                s = str(src or "").strip()
                if not s:
                    continue
                for tok in re.split(r"[,+;/\\s]+", s):
                    _add(tok)
    except Exception:
        pass
    return out


def sort_player_stats_rows(
    rows: list[dict[str, Any]],
    *,
    sort_key: str,
    sort_dir: str,
) -> list[dict[str, Any]]:
    key = str(sort_key or "").strip()
    direction = str(sort_dir or "").strip().lower()
    if direction not in {"asc", "desc"}:
        direction = "desc"

    def _val(r: dict[str, Any]) -> Any:
        if key in {"jersey", "jersey_number", "#"}:
            try:
                return int(str(r.get("jersey_number") or "0").strip() or "0")
            except Exception:
                return 0
        if key in {"name", "player"}:
            return str(r.get("name") or "").lower()
        if key == "position":
            return str(r.get("position") or "").lower()
        v = r.get(key)
        if v is None or v == "":
            return float("-inf") if direction == "desc" else float("inf")
        if isinstance(v, (int, float)):
            return v
        try:
            return float(str(v))
        except Exception:
            return str(v).lower()

    reverse = direction == "desc"

    # Stable tie-breakers (points desc, then name).
    def _tiebreak(r: dict[str, Any]) -> tuple:
        pts = r.get("points")
        try:
            pts_v = float(pts) if pts is not None else 0.0
        except Exception:
            pts_v = 0.0
        return (-pts_v, str(r.get("name") or "").lower())

    return sorted(list(rows or []), key=lambda r: (_val(r), _tiebreak(r)), reverse=reverse)


def sort_players_table_default(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Default stable sort order for the team-page Players table:
      - name ascending
      - assists descending
      - goals descending
      - points descending
    """
    out = list(rows or [])

    def _n(r: dict[str, Any], k: str) -> int:
        return _int0(r.get(k))

    # Stable sort: apply the least significant key first and the most significant key last.
    out.sort(key=lambda r: str(r.get("name") or "").lower())
    out.sort(key=lambda r: _n(r, "assists"), reverse=True)
    out.sort(key=lambda r: _n(r, "goals"), reverse=True)
    out.sort(key=lambda r: _n(r, "points"), reverse=True)
    return out


def aggregate_players_totals(db_conn, team_id: int, user_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Count, Sum
    from django.db.models.functions import Coalesce

    annotations: dict[str, Any] = {"gp": Count("id")}
    for k in PLAYER_STATS_SUM_KEYS:
        annotations[str(k)] = Coalesce(Sum(str(k)), 0)

    rows = (
        m.PlayerStat.objects.filter(team_id=int(team_id), user_id=int(user_id))
        .values("player_id")
        .annotate(**annotations)
    )
    out: dict[int, dict[str, Any]] = {}
    for r in rows or []:
        pid = int(r.get("player_id") if isinstance(r, dict) else r["player_id"])
        out[pid] = compute_player_display_stats(dict(r))
    return out


def aggregate_players_totals_league(db_conn, team_id: int, league_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Count, Q, Sum
    from django.db.models.functions import Coalesce

    league_team_div: dict[int, str] = {
        int(tid): str(dn or "").strip()
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    eligible_game_ids: list[int] = []
    for lg in (
        m.LeagueGame.objects.filter(league_id=int(league_id))
        .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
        .select_related("game")
    ):
        g = lg.game
        t1_id = int(g.team1_id)
        t2_id = int(g.team2_id)
        row = {
            "division_name": lg.division_name,
            "team1_league_division_name": league_team_div.get(t1_id),
            "team2_league_division_name": league_team_div.get(t2_id),
        }
        if _league_game_is_cross_division_non_external(row):
            continue
        eligible_game_ids.append(int(lg.game_id))

    if not eligible_game_ids:
        return {}

    annotations2: dict[str, Any] = {"gp": Count("id")}
    for k in PLAYER_STATS_SUM_KEYS:
        annotations2[str(k)] = Coalesce(Sum(str(k)), 0)

    rows = (
        m.PlayerStat.objects.filter(team_id=int(team_id), game_id__in=eligible_game_ids)
        .values("player_id")
        .annotate(**annotations2)
    )
    out: dict[int, dict[str, Any]] = {}
    for r in rows or []:
        pid = int(r.get("player_id") if isinstance(r, dict) else r["player_id"])
        out[pid] = compute_player_display_stats(dict(r))
    return out


def _league_game_is_cross_division_non_external(game_row: dict[str, Any]) -> bool:
    """
    Returns True if both teams have known, non-External league divisions and those divisions differ.
    """
    d1 = str(game_row.get("team1_league_division_name") or "").strip()
    d2 = str(game_row.get("team2_league_division_name") or "").strip()
    if not d1 or not d2:
        return False
    if is_external_division_name(d1) or is_external_division_name(d2):
        return False
    ld = str(game_row.get("division_name") or game_row.get("league_division_name") or "").strip()
    if is_external_division_name(ld):
        return False
    return d1 != d2


def normalize_jersey_number(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return str(int(m.group(1)))
    except Exception:
        return m.group(1)


def normalize_player_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().lower())


def parse_duration_seconds(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(float(sec))
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(float(sec))
        return int(float(s))
    except Exception:
        return None


def format_seconds_to_mmss_or_hhmmss(raw: Any) -> str:
    try:
        t = int(raw)  # type: ignore[arg-type]
    except Exception:
        return ""
    if t < 0:
        t = 0
    h = t // 3600
    r = t % 3600
    m = r // 60
    s = r % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_shift_stats_player_stats_csv(csv_text: str) -> list[dict[str, Any]]:
    """
    Parse stats/player_stats.csv written by scripts/parse_stats_inputs.py.

    Returns rows with:
      - player_label: original display label (e.g. "59 Ryan S Donahue")
      - jersey_number: normalized jersey number ("59") when present
      - name_norm: normalized player name for matching
      - stats: dict of DB column -> value
      - period_stats: dict[period:int] -> dict of period metrics
    """
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("missing CSV headers")

    out: list[dict[str, Any]] = []
    for raw_row in reader:
        row = {k.strip(): v for k, v in (raw_row or {}).items() if k}
        player_name = (row.get("Player") or "").strip()

        # Newer outputs write jersey and name as separate columns.
        jersey_raw = (
            row.get("Jersey #")
            or row.get("Jersey")
            or row.get("Jersey No")
            or row.get("Jersey Number")
            or ""
        )
        jersey_norm = normalize_jersey_number(jersey_raw) if str(jersey_raw).strip() else None

        # Back-compat: older outputs encoded jersey in the Player field (e.g. " 8 Adam Ro").
        name_part = player_name
        if jersey_norm is None:
            m = re.match(r"^\s*(\d+)\s+(.*)$", player_name)
            if m:
                jersey_norm = normalize_jersey_number(m.group(1))
                name_part = m.group(2).strip()

        player_label = f"{jersey_norm} {name_part}".strip() if jersey_norm else name_part
        name_norm = normalize_player_name(name_part)

        stats: dict[str, Any] = {
            "goals": _int_or_none(row.get("Goals")),
            "assists": _int_or_none(row.get("Assists")),
            "shots": _int_or_none(row.get("Shots")),
            "pim": _int_or_none(row.get("PIM")),
            "hits": _int_or_none(row.get("Hits")),
            "blocks": _int_or_none(row.get("Blocks")),
            "toi_seconds": parse_duration_seconds(row.get("TOI Total") or row.get("TOI")),
            "faceoff_wins": _int_or_none(row.get("Faceoff Wins") or row.get("Faceoffs Won")),
            "faceoff_attempts": _int_or_none(row.get("Faceoff Attempts") or row.get("Faceoffs")),
            "goalie_saves": _int_or_none(row.get("Saves") or row.get("Goalie Saves")),
            "goalie_ga": _int_or_none(row.get("GA") or row.get("Goalie GA")),
            "goalie_sa": _int_or_none(row.get("SA") or row.get("Goalie SA")),
            "sog": _int_or_none(row.get("SOG")),
            "expected_goals": _int_or_none(row.get("xG")),
            "giveaways": _int_or_none(row.get("Giveaways")),
            "turnovers_forced": _int_or_none(row.get("Turnovers (forced)")),
            "created_turnovers": _int_or_none(row.get("Created Turnovers")),
            "takeaways": _int_or_none(row.get("Takeaways")),
            "controlled_entry_for": _int_or_none(row.get("Controlled Entry For (On-Ice)")),
            "controlled_entry_against": _int_or_none(row.get("Controlled Entry Against (On-Ice)")),
            "controlled_exit_for": _int_or_none(row.get("Controlled Exit For (On-Ice)")),
            "controlled_exit_against": _int_or_none(row.get("Controlled Exit Against (On-Ice)")),
            "gt_goals": _int_or_none(row.get("GT Goals")),
            "gw_goals": _int_or_none(row.get("GW Goals")),
            "ot_goals": _int_or_none(row.get("OT Goals")),
            "ot_assists": _int_or_none(row.get("OT Assists")),
            "plus_minus": _int_or_none(row.get("Plus Minus") or row.get("Goal +/-")),
            "gf_counted": _int_or_none(row.get("GF Counted")),
            "ga_counted": _int_or_none(row.get("GA Counted")),
            "shifts": _int_or_none(row.get("Shifts")),
            "video_toi_seconds": parse_duration_seconds(
                row.get("TOI Total (Video)") or row.get("TOI (Video)")
            ),
            "sb_avg_shift_seconds": parse_duration_seconds(row.get("Average Shift")),
            "sb_median_shift_seconds": parse_duration_seconds(row.get("Median Shift")),
            "sb_longest_shift_seconds": parse_duration_seconds(row.get("Longest Shift")),
            "sb_shortest_shift_seconds": parse_duration_seconds(row.get("Shortest Shift")),
        }

        # Period stats: Period {n} GF/GA
        period_stats: dict[int, dict[str, Any]] = {}
        for k, v in row.items():
            if not k:
                continue
            m = re.match(r"^Period\s+(\d+)\s+GF$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["gf"] = _int_or_none(v)
                continue
            m = re.match(r"^Period\s+(\d+)\s+GA$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["ga"] = _int_or_none(v)
                continue
            m = re.match(r"^Period\s+(\d+)\s+TOI$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["toi_seconds"] = parse_duration_seconds(v)
                continue
            m = re.match(r"^Period\s+(\d+)\s+Shifts$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["shifts"] = _int_or_none(v)
                continue

        out.append(
            {
                "player_label": player_label,
                "jersey_number": jersey_norm,
                "name_norm": name_norm,
                "stats": stats,
                "period_stats": period_stats,
            }
        )
    return out


def parse_shift_stats_game_stats_csv(csv_text: str) -> dict[str, Any]:
    """
    Parse stats/game_stats.csv written by scripts/parse_stats_inputs.py.
    Format is a 2-column table: "Stat", "<game_label>".
    """
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    if not reader.fieldnames or "Stat" not in reader.fieldnames:
        raise ValueError("missing Stat column")
    value_col = next((c for c in reader.fieldnames if c != "Stat"), None)
    if not value_col:
        raise ValueError("missing value column")
    out: dict[str, Any] = {"_label": value_col}
    for row in reader:
        if not row:
            continue
        key = (row.get("Stat") or "").strip()
        if not key:
            continue
        out[key] = (row.get(value_col) or "").strip()
    return out


def _int_or_none(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


if __name__ == "__main__":  # pragma: no cover
    # Dev-only legacy entrypoint.
    create_app().run(host="127.0.0.1", port=8008, debug=True)

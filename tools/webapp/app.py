#!/usr/bin/env python3
import csv
import datetime as dt
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

from flask import (
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
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import HTTPException

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
    filter_games_ignore_cross_age,
    scale_ratings_to_0_99_9_by_component,
)


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


def create_app() -> Flask:
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
            is_youtube = ("youtube.com" in host) or ("youtu.be" in host) or ("youtube-nocookie.com" in host)
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
        g.db = get_db()
        # Ensure session league selection is valid or load user's default league
        try:
            if "user_id" in session:
                uid = int(session["user_id"])  # type: ignore[arg-type]

                def _has_access(lid: int) -> bool:
                    with g.db.cursor() as cur:
                        cur.execute(
                            """
                            SELECT 1 FROM leagues l
                            LEFT JOIN league_members m
                              ON m.league_id=l.id AND m.user_id=%s
                            WHERE l.id=%s AND (l.is_shared=1 OR l.owner_user_id=%s OR m.user_id=%s)
                            """,
                            (uid, lid, uid, uid),
                        )
                        return bool(cur.fetchone())

                # Validate existing session league
                sid = session.get("league_id")
                if sid is not None:
                    try:
                        lid = int(sid)  # type: ignore[arg-type]
                        if not _has_access(lid):
                            # Clear invalid selection and clear stored default if matches
                            session.pop("league_id", None)
                            with g.db.cursor() as cur:
                                cur.execute(
                                    "UPDATE users SET default_league_id=NULL WHERE id=%s AND default_league_id=%s",
                                    (uid, lid),
                                )
                            g.db.commit()
                    except Exception:
                        session.pop("league_id", None)
                else:
                    # Load user's default league if any
                    with g.db.cursor() as cur:
                        cur.execute("SELECT default_league_id FROM users WHERE id=%s", (uid,))
                        row = cur.fetchone()
                    if row and row[0] is not None:
                        try:
                            pref = int(row[0])
                            if _has_access(pref):
                                session["league_id"] = pref
                            else:
                                with g.db.cursor() as cur:
                                    cur.execute(
                                        "UPDATE users SET default_league_id=NULL WHERE id=%s AND default_league_id=%s",
                                        (uid, pref),
                                    )
                                g.db.commit()
                        except Exception:
                            pass
        except Exception:
            # Non-fatal
            pass

    @app.teardown_request
    def close_db(exc):  # noqa: ARG001
        db = g.pop("db", None)
        if db is not None:
            db.close()

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
                with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT l.id, l.name, l.is_shared, l.is_public, (l.owner_user_id=%s) AS is_owner,
                               CASE WHEN (l.owner_user_id=%s OR EXISTS (
                                   SELECT 1 FROM league_members m WHERE m.league_id=l.id AND m.user_id=%s AND m.role IN ('admin','owner')
                               )) THEN 1 ELSE 0 END AS is_admin
                        FROM leagues l
                        WHERE l.is_shared=1 OR l.owner_user_id=%s OR EXISTS (
                          SELECT 1 FROM league_members m WHERE m.league_id=l.id AND m.user_id=%s
                        )
                        ORDER BY l.name
                        """,
                        (
                            session["user_id"],
                            session["user_id"],
                            session["user_id"],
                            session["user_id"],
                            session["user_id"],
                        ),
                    )
                    leagues = cur.fetchall()
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
        if lid and lid.isdigit():
            lid_i = int(lid)
            with g.db.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM leagues l LEFT JOIN league_members m ON l.id=m.league_id AND m.user_id=%s WHERE l.id=%s AND (l.is_shared=1 OR l.owner_user_id=%s OR m.user_id=%s)",
                    (session["user_id"], lid_i, session["user_id"], session["user_id"]),
                )
                ok = cur.fetchone()
            if ok:
                session["league_id"] = lid_i
                # Persist preferred league in users table
                with g.db.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET default_league_id=%s WHERE id=%s",
                        (lid_i, session["user_id"]),
                    )
                g.db.commit()
        else:
            # Switch back to personal data; clear preferred league
            session.pop("league_id", None)
            with g.db.cursor() as cur:
                cur.execute(
                    "UPDATE users SET default_league_id=NULL WHERE id=%s", (session["user_id"],)
                )
            g.db.commit()
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
                u = get_user_by_email(email)
                if u:
                    token = secrets.token_urlsafe(32)
                    exp = (dt.datetime.now() + dt.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                    with g.db.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO resets(user_id, token, expires_at, created_at)
                            VALUES(%s,%s,%s,%s)
                            """,
                            (u["id"], token, exp, dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        )
                    g.db.commit()
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
        # Validate token
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT r.id, r.user_id, r.token, r.expires_at, r.used_at, u.email FROM resets r JOIN users u ON r.user_id=u.id WHERE r.token=%s",
                (token,),
            )
            row = cur.fetchone()
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
            with g.db.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash=%s WHERE id=%s", (newhash, row["user_id"])
                )
                cur.execute(
                    "UPDATE resets SET used_at=%s WHERE id=%s",
                    (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["id"]),
                )
            g.db.commit()
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
            return jsonify({"ok": False, "error": "clip_len_s must be 10, 20, or 30"}), 400
        if v not in {10, 20, 30}:
            return jsonify({"ok": False, "error": "clip_len_s must be 10, 20, or 30"}), 400
        try:
            with g.db.cursor() as cur:
                cur.execute("UPDATE users SET video_clip_len_s=%s WHERE id=%s", (int(v), int(session["user_id"])))
            g.db.commit()
        except Exception as e:  # noqa: BLE001
            try:
                g.db.rollback()
            except Exception:
                pass
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "clip_len_s": int(v)})

    @app.route("/games")
    def games():
        r = require_login()
        if r:
            return r
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT * FROM games WHERE user_id=%s ORDER BY created_at DESC",
                (session["user_id"],),
            )
            rows = cur.fetchall()
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM games WHERE id=%s AND user_id=%s", (gid, session["user_id"]))
            game = cur.fetchone()
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
        with g.db.cursor() as cur:
            cur.execute("SELECT status FROM jobs WHERE game_id=%s ORDER BY id DESC LIMIT 1", (gid,))
            row = cur.fetchone()
            if row:
                latest_status = str(row[0]) if row[0] is not None else None
        if not latest_status:
            dw_state = read_dirwatch_state()
            latest_status = dw_state.get("processed", {}).get(game["dir_path"], {}).get(
                "status"
            ) or game.get("status")
        # Lock interactions once a job has been requested (any job row exists) or after completion
        is_locked = False
        if row:
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM games WHERE id=%s AND user_id=%s", (gid, session["user_id"]))
            game = cur.fetchone()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))

        # Check latest job state for potential cancellation on delete
        latest = None
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, slurm_job_id, status FROM jobs WHERE game_id=%s ORDER BY id DESC LIMIT 1",
                (gid,),
            )
            latest = cur.fetchone()

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
            with g.db.cursor() as cur:
                cur.execute("DELETE FROM jobs WHERE game_id=%s", (gid,))
                cur.execute(
                    "DELETE FROM games WHERE id=%s AND user_id=%s", (gid, session["user_id"])
                )
            g.db.commit()
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM games WHERE id=%s AND user_id=%s", (gid, session["user_id"]))
            game = cur.fetchone()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))
        # Block uploads if a job has been requested or finished
        with g.db.cursor() as cur:
            cur.execute("SELECT status FROM jobs WHERE game_id=%s ORDER BY id DESC LIMIT 1", (gid,))
            row = cur.fetchone()
        if row:
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM games WHERE id=%s AND user_id=%s", (gid, session["user_id"]))
            game = cur.fetchone()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("games"))
        # Prevent duplicate submissions
        with g.db.cursor() as cur:
            cur.execute(
                "SELECT id,status FROM jobs WHERE game_id=%s ORDER BY id DESC LIMIT 1", (gid,)
            )
            row = cur.fetchone()
        if row:
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
        with g.db.cursor() as cur:
            cur.execute("UPDATE games SET status=%s WHERE id=%s", ("submitted", gid))
            # Insert job record (pending)
            cur.execute(
                """
                INSERT INTO jobs(user_id, game_id, dir_path, status, created_at)
                VALUES(%s,%s,%s,%s,%s)
                """,
                (session["user_id"], gid, str(dir_path), "PENDING", dt.datetime.now().isoformat()),
            )
        g.db.commit()
        flash("Run requested. Job will start shortly.", "success")
        return redirect(url_for("game_detail", gid=gid))

    @app.route("/uploads/<int:gid>/<path:name>")
    def serve_upload(gid: int, name: str):
        # Optional: serve uploaded files back to user if needed
        r = require_login()
        if r:
            return r
        game = g.db.execute(
            "SELECT * FROM games WHERE id=? AND user_id=?", (gid, session["user_id"])
        )
        game = game.fetchone()
        if not game:
            return ("Not found", 404)
        d = Path(game["dir_path"]).resolve()
        return send_from_directory(str(d), name, as_attachment=True)

    @app.route("/jobs")
    def jobs():
        r = require_login()
        if r:
            return r
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT * FROM jobs WHERE user_id=%s ORDER BY created_at DESC",
                (session["user_id"],),
            )
            jobs = cur.fetchall()
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, user_id, logo_path FROM teams WHERE id=%s AND user_id=%s",
                (team_id, session["user_id"]),
            )
            row = cur.fetchone()
            if not row and league_id:
                cur.execute(
                    """
                    SELECT t.id, t.user_id, t.logo_path
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s AND t.id=%s
                    """,
                    (league_id, team_id),
                )
                row = cur.fetchone()
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
        is_league_admin = False
        if league_id:
            try:
                is_league_admin = bool(_is_league_admin(int(league_id), int(session["user_id"])))
            except Exception:
                is_league_admin = False
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            if league_id:
                cur.execute(
                    """
                    SELECT t.*, lt.division_name, lt.division_id, lt.conference_id,
                           lt.mhr_rating, lt.mhr_agd, lt.mhr_sched, lt.mhr_games, lt.mhr_updated_at
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s
                    """,
                    (league_id,),
                )
            else:
                where = "user_id=%s" + ("" if include_external else " AND is_external=0")
                cur.execute(
                    f"SELECT * FROM teams WHERE {where} ORDER BY name ASC", (session["user_id"],)
                )
            rows = cur.fetchall()
        # compute stats per team (wins/losses/ties/gf/ga/points)
        stats = {}
        for t in rows:
            if league_id:
                stats[t["id"]] = compute_team_stats_league(g.db, t["id"], int(league_id))
            else:
                stats[t["id"]] = compute_team_stats(g.db, t["id"], session["user_id"])
        divisions = None
        if league_id:
            grouped: dict[str, list[dict]] = {}
            for t in rows:
                dn = str(t.get("division_name") or "").strip() or "Unknown Division"
                grouped.setdefault(dn, []).append(t)
            divisions = []
            for dn in sorted(grouped.keys(), key=lambda s: s.lower()):
                teams_sorted = sorted(grouped[dn], key=lambda tr: sort_key_team_standings(tr, stats.get(tr["id"], {})))
                divisions.append({"name": dn, "teams": teams_sorted})
        return render_template(
            "teams.html",
            teams=rows,
            divisions=divisions,
            stats=stats,
            include_external=include_external,
            league_view=bool(league_id),
            current_user_id=int(session["user_id"]),
            is_league_admin=is_league_admin,
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
            recompute_league_mhr_ratings(g.db, int(league_id))
            flash("Ratings recalculated.", "success")
        except Exception as e:  # noqa: BLE001
            try:
                g.db.rollback()
            except Exception as rollback_err:
                app.logger.exception("Database rollback failed after ratings recompute error: %s", rollback_err)
            flash(f"Failed to recalculate Ratings: {e}", "error")
        return redirect(url_for("leagues_index"))

    @app.get("/leagues")
    def leagues_index():
        r = require_login()
        if r:
            return r
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT l.id, l.name, l.is_shared, l.is_public, (l.owner_user_id=%s) AS is_owner,
                       CASE WHEN (l.owner_user_id=%s OR EXISTS (
                           SELECT 1 FROM league_members m WHERE m.league_id=l.id AND m.user_id=%s AND m.role IN ('admin','owner')
                       )) THEN 1 ELSE 0 END AS is_admin
                FROM leagues l
                WHERE l.is_shared=1 OR l.owner_user_id=%s OR EXISTS (
                  SELECT 1 FROM league_members m WHERE m.league_id=l.id AND m.user_id=%s
                )
                ORDER BY l.name
                """,
                (session["user_id"], session["user_id"], session["user_id"], session["user_id"], session["user_id"]),
            )
            leagues = cur.fetchall()
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
        with g.db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email_norm,))
            row = cur.fetchone()
            if row:
                return int(row[0])
            pwd = generate_password_hash(secrets.token_hex(24))
            cur.execute(
                "INSERT INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
                (email_norm, pwd, name or email_norm, dt.datetime.now().isoformat()),
            )
            g.db.commit()
            return int(cur.lastrowid)

    def _ensure_league_for_import(
        *,
        league_name: str,
        owner_user_id: int,
        is_shared: bool,
        source: Optional[str],
        external_key: Optional[str],
        commit: bool = True,
    ) -> int:
        name = (league_name or "").strip()
        if not name:
            raise ValueError("league_name is required")
        with g.db.cursor() as cur:
            cur.execute("SELECT id, is_shared FROM leagues WHERE name=%s", (name,))
            row = cur.fetchone()
            if row:
                lid = int(row[0])
                want_shared = 1 if is_shared else 0
                if int(row[1]) != want_shared:
                    cur.execute(
                        "UPDATE leagues SET is_shared=%s, updated_at=%s WHERE id=%s",
                        (want_shared, dt.datetime.now().isoformat(), lid),
                    )
                    if commit:
                        g.db.commit()
                return lid
            cur.execute(
                "INSERT INTO leagues(name, owner_user_id, is_shared, source, external_key, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
                (
                    name,
                    owner_user_id,
                    1 if is_shared else 0,
                    source,
                    external_key,
                    dt.datetime.now().isoformat(),
                ),
            )
            if commit:
                g.db.commit()
            return int(cur.lastrowid)

    def _ensure_league_member_for_import(league_id: int, user_id: int, role: str, *, commit: bool = True) -> None:
        with g.db.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO league_members(league_id, user_id, role, created_at) VALUES(%s,%s,%s,%s)",
                (league_id, user_id, role, dt.datetime.now().isoformat()),
            )
            if commit:
                g.db.commit()

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
        with g.db.cursor() as cur:
            cur.execute("SELECT id FROM game_types WHERE name=%s", (nm,))
            row = cur.fetchone()
            if row:
                return int(row[0])
            cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (nm, 0))
            g.db.commit()
            return int(cur.lastrowid)

    def _ensure_external_team_for_import(owner_user_id: int, name: str, *, commit: bool = True) -> int:
        def _norm_team_name(s: str) -> str:
            t = str(s or "").replace("\xa0", " ").strip()
            t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
            t = " ".join(t.split())
            t = re.sub(r"\s*\(\s*external\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
            # Case/punctuation-insensitive matching to avoid duplicate teams.
            t = t.casefold()
            t = re.sub(r"[^0-9a-z]+", " ", t)
            return " ".join(t.split())

        nm = _norm_team_name(name or "")
        if not nm:
            nm = "UNKNOWN"
        with g.db.cursor() as cur:
            # Keep the stored team name as-is; only use normalization for matching.
            raw_name = str(name or "").strip()
            cur.execute("SELECT id FROM teams WHERE user_id=%s AND name=%s", (owner_user_id, raw_name))
            row = cur.fetchone()
            if row:
                return int(row[0])
            # Robust match: normalize existing names to avoid duplicate external teams due to minor name variations.
            cur.execute("SELECT id, name FROM teams WHERE user_id=%s", (owner_user_id,))
            rows = cur.fetchall() or []
            for tid, tname in rows:
                if _norm_team_name(tname) == nm:
                    return int(tid)
            cur.execute(
                "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
                (owner_user_id, raw_name or "UNKNOWN", 1, dt.datetime.now().isoformat()),
            )
            if commit:
                g.db.commit()
            return int(cur.lastrowid)

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
        with g.db.cursor() as cur:
            cur.execute(
                "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                (owner_user_id, team_id, nm),
            )
            row = cur.fetchone()
            if row:
                pid = int(row[0])
                if jersey_number or position:
                    cur.execute(
                        "UPDATE players SET jersey_number=COALESCE(%s, jersey_number), position=COALESCE(%s, position), updated_at=%s WHERE id=%s",
                        (jersey_number, position, dt.datetime.now().isoformat(), pid),
                    )
                    if commit:
                        g.db.commit()
                return pid
            cur.execute(
                "INSERT INTO players(user_id, team_id, name, jersey_number, position, created_at) VALUES(%s,%s,%s,%s,%s,%s)",
                (owner_user_id, team_id, nm, jersey_number, position, dt.datetime.now().isoformat()),
            )
            if commit:
                g.db.commit()
            return int(cur.lastrowid)

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

    def _update_game_video_url_note(game_id: int, video_url: str, *, replace: bool, commit: bool = True) -> None:
        url = _sanitize_http_url(video_url)
        if not url:
            return
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT notes FROM hky_games WHERE id=%s", (int(game_id),))
            row = cur.fetchone()
        existing = str((row or {}).get("notes") or "").strip()
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
                new_notes = (existing + "\n" + suffix.strip()).strip() if existing else suffix.strip()
        with g.db.cursor() as cur:
            cur.execute("UPDATE hky_games SET notes=%s, updated_at=%s WHERE id=%s", (new_notes, dt.datetime.now().isoformat(), int(game_id)))
            if commit:
                g.db.commit()

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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            gid: Optional[int] = None
            if starts_at:
                cur.execute(
                    "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND team1_id=%s AND team2_id=%s AND starts_at=%s",
                    (owner_user_id, team1_id, team2_id, starts_at),
                )
                row = cur.fetchone()
                if row:
                    gid = int(row["id"])
            if gid is None and notes_json_fields.get("timetoscore_game_id") is not None:
                try:
                    tts_int = int(notes_json_fields["timetoscore_game_id"])
                except Exception:
                    tts_int = None
                if tts_int is not None:
                    row = None
                    for token in (
                        f"\"timetoscore_game_id\":{tts_int}",
                        f"\"timetoscore_game_id\": {tts_int}",
                    ):
                        cur.execute(
                            "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND notes LIKE %s",
                            (owner_user_id, f"%{token}%"),
                        )
                        row = cur.fetchone()
                        if row:
                            break
                    if row:
                        gid = int(row["id"])

            if gid is None and notes_json_fields.get("external_game_key"):
                ext = str(notes_json_fields.get("external_game_key") or "").strip()
                if ext:
                    try:
                        ext_json = json.dumps(ext)
                    except Exception:
                        ext_json = f"\"{ext}\""
                    row = None
                    for token in (
                        f"\"external_game_key\":{ext_json}",
                        f"\"external_game_key\": {ext_json}",
                    ):
                        cur.execute(
                            "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND notes LIKE %s",
                            (owner_user_id, f"%{token}%"),
                        )
                        row = cur.fetchone()
                        if row:
                            break
                    if row:
                        gid = int(row["id"])

            if gid is None:
                notes = json.dumps(notes_json_fields, sort_keys=True)
                cur.execute(
                    """
                    INSERT INTO hky_games(user_id, team1_id, team2_id, game_type_id, starts_at, location, team1_score, team2_score, is_final, notes, stats_imported_at, created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        owner_user_id,
                        team1_id,
                        team2_id,
                        game_type_id,
                        starts_at,
                        location,
                        team1_score,
                        team2_score,
                        1 if (team1_score is not None and team2_score is not None) else 0,
                        notes,
                        dt.datetime.now().isoformat(),
                        dt.datetime.now().isoformat(),
                    ),
                )
                if commit:
                    g.db.commit()
                return int(cur.lastrowid)

            cur.execute("SELECT notes, team1_score, team2_score FROM hky_games WHERE id=%s", (gid,))
            row2 = cur.fetchone()
            existing_notes = row2["notes"] if row2 else None
            merged_notes = _merge_notes(existing_notes, notes_json_fields)
            if replace:
                cur.execute(
                    """
                    UPDATE hky_games
                    SET game_type_id=COALESCE(%s, game_type_id),
                        location=COALESCE(%s, location),
                        team1_score=%s,
                        team2_score=%s,
                        is_final=CASE WHEN %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                        notes=%s,
                        stats_imported_at=%s,
                        updated_at=%s
                    WHERE id=%s
                    """,
                    (
                        game_type_id,
                        location,
                        team1_score,
                        team2_score,
                        team1_score,
                        team2_score,
                        merged_notes,
                        dt.datetime.now().isoformat(),
                        dt.datetime.now().isoformat(),
                        gid,
                    ),
                )
            else:
                cur.execute(
                    """
                    UPDATE hky_games
                    SET game_type_id=COALESCE(%s, game_type_id),
                        location=COALESCE(%s, location),
                        team1_score=COALESCE(team1_score, %s),
                        team2_score=COALESCE(team2_score, %s),
                        is_final=CASE WHEN team1_score IS NULL AND team2_score IS NULL AND %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                        notes=%s,
                        stats_imported_at=%s,
                        updated_at=%s
                    WHERE id=%s
                    """,
                    (
                        game_type_id,
                        location,
                        team1_score,
                        team2_score,
                        team1_score,
                        team2_score,
                        merged_notes,
                        dt.datetime.now().isoformat(),
                        dt.datetime.now().isoformat(),
                        gid,
                    ),
                )
            if commit:
                g.db.commit()
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
        with g.db.cursor() as cur:
            if dn is None and division_id is None and conference_id is None:
                cur.execute(
                    "INSERT IGNORE INTO league_teams(league_id, team_id) VALUES(%s,%s)",
                    (league_id, team_id),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO league_teams(league_id, team_id, division_name, division_id, conference_id)
                    VALUES(%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      division_name=CASE
                        WHEN VALUES(division_name) IS NULL OR VALUES(division_name)='' THEN division_name
                        WHEN division_name IS NULL OR division_name='' THEN VALUES(division_name)
                        WHEN VALUES(division_name)='External' THEN division_name
                        ELSE VALUES(division_name)
                      END,
                      division_id=COALESCE(VALUES(division_id), division_id),
                      conference_id=COALESCE(VALUES(conference_id), conference_id)
                    """,
                    (league_id, team_id, dn, division_id, conference_id),
                )
            if commit:
                g.db.commit()

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
        with g.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id, sort_order)
                VALUES(%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  division_name=CASE
                    WHEN VALUES(division_name) IS NULL OR VALUES(division_name)='' THEN division_name
                    WHEN division_name IS NULL OR division_name='' THEN VALUES(division_name)
                    WHEN VALUES(division_name)='External' THEN division_name
                    ELSE VALUES(division_name)
                  END,
                  division_id=COALESCE(VALUES(division_id), division_id),
                  conference_id=COALESCE(VALUES(conference_id), conference_id),
                  sort_order=COALESCE(VALUES(sort_order), sort_order)
                """,
                (league_id, game_id, dn, division_id, conference_id, sort_order),
            )
            if commit:
                g.db.commit()

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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT logo_path FROM teams WHERE id=%s", (int(team_id),))
            row = cur.fetchone()
            if row and row.get("logo_path") and not replace:
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
            with g.db.cursor() as cur:
                cur.execute(
                    "UPDATE teams SET logo_path=%s, updated_at=%s WHERE id=%s",
                    (str(dest), dt.datetime.now().isoformat(), int(team_id)),
                )
            if commit:
                g.db.commit()
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT logo_path FROM teams WHERE id=%s", (int(team_id),))
            row = cur.fetchone()
            if row and row.get("logo_path") and not replace:
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
                with g.db.cursor() as cur:
                    cur.execute(
                        "UPDATE teams SET logo_path=%s, updated_at=%s WHERE id=%s",
                        (str(dest), dt.datetime.now().isoformat(), int(team_id)),
                    )
                if commit:
                    g.db.commit()
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
        league_name = str(payload.get("league_name") or "Norcal")
        shared = bool(payload.get("shared", True))
        owner_email = str(payload.get("owner_email") or "norcal-import@hockeymom.local")
        owner_name = str(payload.get("owner_name") or "Norcal Import")
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
        league_name = str(payload.get("league_name") or "Norcal")
        shared = bool(payload.get("shared", True))
        replace = bool(payload.get("replace", False))
        owner_email = str(payload.get("owner_email") or "norcal-import@hockeymom.local")
        owner_name = str(payload.get("owner_name") or "Norcal Import")
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
            def _clean_division_name(dn: Any) -> Optional[str]:
                s = str(dn or "").strip()
                if not s:
                    return None
                if s.lower() == "external":
                    return None
                return s

            for idx, team in enumerate(teams):
                if not isinstance(team, dict):
                    raise ValueError(f"teams[{idx}] must be an object")
                name = str(team.get("name") or "").strip()
                if not name:
                    continue

                team_replace = bool(team.get("replace", replace))

                division_name = _clean_division_name(team.get("division_name"))
                try:
                    division_id = int(team.get("division_id")) if team.get("division_id") is not None else None
                except Exception:
                    division_id = None
                try:
                    conference_id = int(team.get("conference_id")) if team.get("conference_id") is not None else None
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
                    logo_content_type=team.get("logo_content_type") or team.get("team_logo_content_type"),
                    logo_url=team.get("logo_url") or team.get("team_logo_url"),
                    replace=team_replace,
                    commit=False,
                )
                results.append({"team_id": int(team_id), "name": name})

            g.db.commit()
        except Exception as e:
            try:
                g.db.rollback()
            except Exception:
                # Ignore rollback errors; the original exception `e` is already being handled.
                pass
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
        league_name = str(payload.get("league_name") or "Norcal")
        shared = bool(payload.get("shared", True))
        replace = bool(payload.get("replace", False))
        owner_email = str(payload.get("owner_email") or "norcal-import@hockeymom.local")
        owner_name = str(payload.get("owner_name") or "Norcal Import")
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
        home_division_name = str(game.get("home_division_name") or division_name or "").strip() or None
        away_division_name = str(game.get("away_division_name") or division_name or "").strip() or None
        try:
            division_id = int(game.get("division_id")) if game.get("division_id") is not None else None
        except Exception:
            division_id = None
        try:
            conference_id = int(game.get("conference_id")) if game.get("conference_id") is not None else None
        except Exception:
            conference_id = None
        try:
            home_division_id = int(game.get("home_division_id")) if game.get("home_division_id") is not None else division_id
        except Exception:
            home_division_id = division_id
        try:
            away_division_id = int(game.get("away_division_id")) if game.get("away_division_id") is not None else division_id
        except Exception:
            away_division_id = division_id
        try:
            home_conference_id = (
                int(game.get("home_conference_id")) if game.get("home_conference_id") is not None else conference_id
            )
        except Exception:
            home_conference_id = conference_id
        try:
            away_conference_id = (
                int(game.get("away_conference_id")) if game.get("away_conference_id") is not None else conference_id
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
            game.get("game_type_name") or game.get("game_type") or game.get("timetoscore_type") or game.get("type")
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
            with g.db.cursor() as cur:
                cur.execute(
                    "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                    (owner_user_id, team_id, name),
                )
                r = cur.fetchone()
                return int(r[0]) if r else None

        stats_rows = game.get("player_stats") or []
        played = bool(game.get("is_final")) or (t1s is not None and t2s is not None) or (isinstance(stats_rows, list) and bool(stats_rows))
        with g.db.cursor() as cur:
            if played:
                # Credit GP for roster players even when they have no scoring stats in TimeToScore.
                for tid, pids in roster_player_ids_by_team.items():
                    for pid in sorted(pids):
                        cur.execute(
                            """
                            INSERT INTO player_stats(user_id, team_id, game_id, player_id)
                            VALUES(%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE player_id=player_id
                            """,
                            (owner_user_id, int(tid), gid, int(pid)),
                        )
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
                    if replace or force_tts_scoring:
                        cur.execute(
                            """
                            INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                            VALUES(%s,%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE goals=VALUES(goals), assists=VALUES(assists)
                            """,
                            (owner_user_id, team_ref, gid, pid, gval, aval),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists)
                            VALUES(%s,%s,%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE goals=COALESCE(goals, VALUES(goals)), assists=COALESCE(assists, VALUES(assists))
                            """,
                            (owner_user_id, team_ref, gid, pid, gval, aval),
                        )
        g.db.commit()

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
        league_name = str(payload.get("league_name") or "Norcal")
        shared = bool(payload.get("shared", True))
        replace = bool(payload.get("replace", False))
        owner_email = str(payload.get("owner_email") or "norcal-import@hockeymom.local")
        owner_name = str(payload.get("owner_name") or "Norcal Import")
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
            def _clean_division_name(dn: Any) -> Optional[str]:
                s = str(dn or "").strip()
                if not s:
                    return None
                if s.lower() == "external":
                    return None
                return s

            def _league_team_div_meta(lid: int, tid: int) -> tuple[Optional[str], Optional[int], Optional[int]]:
                with g.db.cursor() as cur:
                    cur.execute(
                        "SELECT division_name, division_id, conference_id FROM league_teams WHERE league_id=%s AND team_id=%s",
                        (int(lid), int(tid)),
                    )
                    r = cur.fetchone()
                if not r:
                    return None, None, None
                dn = _clean_division_name(r[0])
                try:
                    did = int(r[1]) if r[1] is not None else None
                except Exception:
                    did = None
                try:
                    cid = int(r[2]) if r[2] is not None else None
                except Exception:
                    cid = None
                return dn, did, cid

            for idx, game in enumerate(games):
                if not isinstance(game, dict):
                    raise ValueError(f"games[{idx}] must be an object")
                home_name = str(game.get("home_name") or "").strip()
                away_name = str(game.get("away_name") or "").strip()
                if not home_name or not away_name:
                    raise ValueError(f"games[{idx}]: home_name and away_name are required")

                game_replace = bool(game.get("replace", replace))

                division_name = _clean_division_name(game.get("division_name"))
                home_division_name = _clean_division_name(game.get("home_division_name") or division_name)
                away_division_name = _clean_division_name(game.get("away_division_name") or division_name)
                try:
                    division_id = int(game.get("division_id")) if game.get("division_id") is not None else None
                except Exception:
                    division_id = None
                try:
                    conference_id = int(game.get("conference_id")) if game.get("conference_id") is not None else None
                except Exception:
                    conference_id = None
                try:
                    home_division_id = (
                        int(game.get("home_division_id")) if game.get("home_division_id") is not None else division_id
                    )
                except Exception:
                    home_division_id = division_id
                try:
                    away_division_id = (
                        int(game.get("away_division_id")) if game.get("away_division_id") is not None else division_id
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
                    logo_content_type=game.get("home_logo_content_type") or game.get("team1_logo_content_type"),
                    logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
                    replace=game_replace,
                    commit=False,
                )
                _ensure_team_logo_for_import(
                    team_id=int(team2_id),
                    logo_b64=game.get("away_logo_b64") or game.get("team2_logo_b64"),
                    logo_content_type=game.get("away_logo_content_type") or game.get("team2_logo_content_type"),
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
                    game.get("game_type_name") or game.get("game_type") or game.get("timetoscore_type") or game.get("type")
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
                            pid = _ensure_player_for_import(owner_user_id, tid, nm, jersey, pos, commit=False)
                            try:
                                roster_player_ids_by_team[int(tid)].add(int(pid))
                            except Exception:
                                pass

                def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
                    with g.db.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                            (owner_user_id, team_id, name),
                        )
                        r = cur.fetchone()
                        return int(r[0]) if r else None

                stats_rows = game.get("player_stats") or []
                events_csv = game.get("events_csv")
                game_stats_json = game.get("game_stats")
                played = bool(game.get("is_final")) or (t1s is not None and t2s is not None) or (
                    isinstance(stats_rows, list) and bool(stats_rows)
                )
                with g.db.cursor() as cur:
                    if played:
                        for tid, pids in roster_player_ids_by_team.items():
                            for pid in sorted(pids):
                                cur.execute(
                                    """
                                    INSERT INTO player_stats(user_id, team_id, game_id, player_id)
                                    VALUES(%s,%s,%s,%s)
                                    ON DUPLICATE KEY UPDATE player_id=player_id
                                    """,
                                    (owner_user_id, int(tid), gid, int(pid)),
                                )
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

                            # If this game is linked to TimeToScore, TimeToScore is the source of truth for
                            # goal/assist attribution. Always overwrite goals/assists from the TimeToScore import.
                            force_tts_scoring = bool(tts_game_id is not None)
                            if game_replace or force_tts_scoring:
                                cur.execute(
                                    """
                                    INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists, pim)
                                    VALUES(%s,%s,%s,%s,%s,%s,%s)
                                    ON DUPLICATE KEY UPDATE goals=VALUES(goals), assists=VALUES(assists), pim=COALESCE(VALUES(pim), pim)
                                    """,
                                    (owner_user_id, team_ref, gid, pid, gval, aval, pim_val),
                                )
                            else:
                                cur.execute(
                                    """
                                    INSERT INTO player_stats(user_id, team_id, game_id, player_id, goals, assists, pim)
                                    VALUES(%s,%s,%s,%s,%s,%s,%s)
                                    ON DUPLICATE KEY UPDATE goals=COALESCE(goals, VALUES(goals)), assists=COALESCE(assists, VALUES(assists)), pim=COALESCE(pim, VALUES(pim))
                                    """,
                                    (owner_user_id, team_ref, gid, pid, gval, aval, pim_val),
                                )

                    # TimeToScore imports may also provide a per-game events timeline and simple per-game stats.
                    if isinstance(events_csv, str) and events_csv.strip():
                        if game_replace:
                            cur.execute(
                                """
                                INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at)
                                VALUES(%s,%s,%s,%s)
                                ON DUPLICATE KEY UPDATE events_csv=VALUES(events_csv), source_label=VALUES(source_label), updated_at=VALUES(updated_at)
                                """,
                                (gid, events_csv, "timetoscore", dt.datetime.now().isoformat()),
                            )
                        else:
                            cur.execute("SELECT events_csv, source_label FROM hky_game_events WHERE game_id=%s", (gid,))
                            existing = cur.fetchone()
                            if not existing:
                                cur.execute(
                                    """
                                    INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at)
                                    VALUES(%s,%s,%s,%s)
                                    """,
                                    (gid, events_csv, "timetoscore", dt.datetime.now().isoformat()),
                                )
                            else:
                                try:
                                    existing_csv = str(existing[0] or "")
                                    existing_source = str(existing[1] or "")
                                except Exception:
                                    existing_csv = ""
                                    existing_source = ""
                                merged_csv, merged_source = merge_events_csv_prefer_timetoscore(
                                    existing_csv=existing_csv,
                                    existing_source_label=existing_source,
                                    incoming_csv=str(events_csv),
                                    incoming_source_label="timetoscore",
                                    protected_types={"goal", "assist", "penalty", "penalty expired", "goaliechange"},
                                )
                                cur.execute(
                                    """
                                    INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at)
                                    VALUES(%s,%s,%s,%s)
                                    ON DUPLICATE KEY UPDATE events_csv=VALUES(events_csv), source_label=VALUES(source_label), updated_at=VALUES(updated_at)
                                    """,
                                    (gid, merged_csv, merged_source or "timetoscore", dt.datetime.now().isoformat()),
                                )

                    if isinstance(game_stats_json, dict) and game_stats_json:
                        try:
                            stats_json_text = json.dumps(game_stats_json)
                        except Exception:
                            stats_json_text = None
                        if stats_json_text:
                            if game_replace:
                                cur.execute(
                                    """
                                    INSERT INTO hky_game_stats(game_id, stats_json, updated_at)
                                    VALUES(%s,%s,%s)
                                    ON DUPLICATE KEY UPDATE stats_json=VALUES(stats_json), updated_at=VALUES(updated_at)
                                    """,
                                    (gid, stats_json_text, dt.datetime.now().isoformat()),
                                )
                            else:
                                cur.execute("SELECT stats_json FROM hky_game_stats WHERE game_id=%s", (gid,))
                                if not cur.fetchone():
                                    cur.execute(
                                        """
                                        INSERT INTO hky_game_stats(game_id, stats_json, updated_at)
                                        VALUES(%s,%s,%s)
                                        """,
                                        (gid, stats_json_text, dt.datetime.now().isoformat()),
                                    )

                results.append({"game_id": gid, "team1_id": team1_id, "team2_id": team2_id})

            g.db.commit()
        except Exception as e:
            try:
                g.db.rollback()
            except Exception:
                pass
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

        if resolved_game_id is None and tts_game_id is not None:
            try:
                tts_int = int(tts_game_id)
            except Exception:
                tts_int = None
            if tts_int is not None:
                token_json_nospace = f"\"timetoscore_game_id\":{int(tts_int)}"
                token_json_space = f"\"timetoscore_game_id\": {int(tts_int)}"
                token_plain = f"game_id={int(tts_int)}"
                with g.db.cursor() as cur:
                    cur.execute("SELECT id FROM hky_games WHERE notes LIKE %s LIMIT 1", (f"%{token_json_nospace}%",))
                    r = cur.fetchone()
                    if not r:
                        cur.execute("SELECT id FROM hky_games WHERE notes LIKE %s LIMIT 1", (f"%{token_json_space}%",))
                        r = cur.fetchone()
                    if not r:
                        cur.execute("SELECT id FROM hky_games WHERE notes LIKE %s LIMIT 1", (f"%{token_plain}%",))
                        r = cur.fetchone()
                if r:
                    resolved_game_id = int(r[0])

        # External game flow: allow creating / matching games not in TimeToScore.
        if resolved_game_id is None and external_game_key and owner_email:
            owner_user_id_for_create = _ensure_user_for_import(owner_email)
            # Reuse existing league teams when names match (e.g., teams already imported from TimeToScore),
            # to avoid creating duplicates and to preserve their division mappings.
            def _norm_team_name_for_match(s: str) -> str:
                t = str(s or "").replace("\xa0", " ").strip()
                t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
                t = " ".join(t.split())
                t = re.sub(r"\s*\(\s*external\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
                # Many TimeToScore imports use a disambiguating "(Division)" suffix.
                # Strip a trailing parenthetical block for matching purposes so uploads that omit it still match.
                t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
                t = t.casefold()
                t = re.sub(r"[^0-9a-z]+", " ", t)
                return " ".join(t.split())

            def _find_team_in_league_by_name(league_id_i: int, name: str) -> Optional[dict[str, Any]]:
                nm = _norm_team_name_for_match(name)
                if not nm:
                    return None
                with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT t.id AS team_id, t.name AS team_name, lt.division_name, lt.division_id, lt.conference_id
                        FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                        WHERE lt.league_id=%s
                        """,
                        (int(league_id_i),),
                    )
                    rows = cur.fetchall() or []
                matches = [r for r in rows if _norm_team_name_for_match(str(r.get("team_name") or "")) == nm]
                if not matches:
                    return None
                if len(matches) == 1:
                    return matches[0]
                # Disambiguate by requested division if provided (and not External), otherwise by any existing match's division.
                want_div = str(payload.get("division_name") or "").strip()
                if want_div and want_div.lower() != "external":
                    by_div = [m for m in matches if str(m.get("division_name") or "").strip() == want_div]
                    if len(by_div) == 1:
                        return by_div[0]
                for m in matches:
                    dn = str(m.get("division_name") or "").strip()
                    if dn:
                        return m
                return matches[0]
                return None

            try:
                ext_json = json.dumps(external_game_key)
            except Exception:
                ext_json = f"\"{external_game_key}\""
            tokens = [
                f"\"external_game_key\":{ext_json}",
                f"\"external_game_key\": {ext_json}",
            ]
            with g.db.cursor() as cur:
                r = None
                for token in tokens:
                    cur.execute(
                        "SELECT id FROM hky_games WHERE user_id=%s AND notes LIKE %s LIMIT 1",
                        (owner_user_id_for_create, f"%{token}%"),
                    )
                    r = cur.fetchone()
                    if r:
                        break
            if r:
                resolved_game_id = int(r[0])
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
                    with g.db.cursor() as cur:
                        cur.execute("SELECT id FROM leagues WHERE name=%s", (league_name,))
                        lr = cur.fetchone()
                        if lr:
                            league_id_i = int(lr[0])
                        else:
                            cur.execute(
                                "INSERT INTO leagues(name, owner_user_id, is_shared, is_public, source, external_key, created_at) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                                (
                                    league_name,
                                    owner_user_id_for_create,
                                    0,
                                    0,
                                    "shift_package",
                                    None,
                                    dt.datetime.now().isoformat(),
                                ),
                            )
                            league_id_i = int(cur.lastrowid)

                match_home = _find_team_in_league_by_name(int(league_id_i), home_team_name) if league_id_i else None
                match_away = _find_team_in_league_by_name(int(league_id_i), away_team_name) if league_id_i else None

                team1_id = int(match_home["team_id"]) if match_home else _ensure_external_team_for_import(owner_user_id_for_create, home_team_name, commit=False)
                team2_id = int(match_away["team_id"]) if match_away else _ensure_external_team_for_import(owner_user_id_for_create, away_team_name, commit=False)

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
                    parsed_gs = parse_shift_stats_game_stats_csv(str(payload.get("game_stats_csv") or ""))
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
                    notes_json_fields={"external_game_key": external_game_key},
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
                g.db.commit()
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM hky_games WHERE id=%s", (resolved_game_id,))
            game_row = cur.fetchone()
        if not game_row:
            return jsonify({"ok": False, "error": "game not found"}), 404

        team1_id = int(game_row["team1_id"])
        team2_id = int(game_row["team2_id"])
        owner_user_id = int(game_row.get("user_id") or 0)
        tts_linked = bool(tts_game_id is not None or _extract_timetoscore_game_id_from_notes(game_row.get("notes")) is not None)

        player_stats_csv = payload.get("player_stats_csv")
        game_stats_csv = payload.get("game_stats_csv")
        events_csv = payload.get("events_csv")
        incoming_events_csv_raw: Optional[str] = None
        incoming_events_headers_raw: Optional[list[str]] = None
        incoming_events_rows_raw: Optional[list[dict[str, str]]] = None
        source_label = str(payload.get("source_label") or "").strip() or None
        game_video_url = (
            payload.get("game_video_url")
            or payload.get("game_video")
            or payload.get("video_url")
        )

        if isinstance(game_video_url, str) and game_video_url.strip():
            try:
                _update_game_video_url_note(int(resolved_game_id), str(game_video_url), replace=replace, commit=False)
            except Exception:
                # Best-effort: failures updating the optional game video URL must not block the import.
                pass

        if isinstance(events_csv, str) and events_csv.strip():
            incoming_events_csv_raw = str(events_csv)
            if tts_linked:
                try:
                    incoming_events_headers_raw, incoming_events_rows_raw = parse_events_csv(incoming_events_csv_raw)
                except Exception:
                    incoming_events_headers_raw, incoming_events_rows_raw = None, None
            # Never store PP/PK span events; they are derived at render time from penalty windows.
            # For TimeToScore-linked games, do not store spreadsheet Goal/Assist attribution.
            drop_types = {"power play", "powerplay", "penalty kill", "penaltykill"}
            if tts_linked:
                drop_types |= {"goal", "assist"}
            try:
                events_csv = filter_events_csv_drop_event_types(str(events_csv), drop_types=drop_types)
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
                with g.db.cursor() as cur:
                    cur.execute("SELECT id FROM leagues WHERE name=%s", (league_name,))
                    lr = cur.fetchone()
                    if lr:
                        league_id_i = int(lr[0])
            if league_id_i is not None:
                _map_team_to_league_for_import(int(league_id_i), team1_id, division_name=division_name, commit=False)
                _map_team_to_league_for_import(int(league_id_i), team2_id, division_name=division_name, commit=False)
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
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT id, team_id, name, jersey_number FROM players WHERE team_id IN (%s,%s)",
                    (team1_id, team2_id),
                )
                players = cur.fetchall() or []

            players_by_team: dict[int, list[dict[str, Any]]] = {}
            jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
            name_to_player_ids: dict[tuple[int, str], list[int]] = {}

            for p in players:
                tid = int(p["team_id"])
                players_by_team.setdefault(tid, []).append(p)
                j = normalize_jersey_number(p.get("jersey_number"))
                if j:
                    jersey_to_player_ids.setdefault((tid, j), []).append(int(p["id"]))
                nm = normalize_player_name(p.get("name") or "")
                if nm:
                    name_to_player_ids.setdefault((tid, nm), []).append(int(p["id"]))

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

            with g.db.cursor() as cur:
                if isinstance(events_csv, str) and events_csv.strip():
                    # Only overwrite existing events if replace is requested.
                    if replace:
                        cur.execute(
                            """
                            INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at)
                            VALUES(%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE events_csv=VALUES(events_csv), source_label=VALUES(source_label), updated_at=VALUES(updated_at)
                            """,
                                (resolved_game_id, events_csv, source_label, dt.datetime.now().isoformat()),
                            )
                    else:
                        cur.execute(
                            "SELECT events_csv, source_label, updated_at FROM hky_game_events WHERE game_id=%s",
                            (resolved_game_id,),
                        )
                        existing = cur.fetchone()
                        if not existing:
                            cur.execute(
                                """
                                INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at)
                                VALUES(%s,%s,%s,%s)
                                """,
                                (resolved_game_id, events_csv, source_label, dt.datetime.now().isoformat()),
                            )
                        else:
                            # For TimeToScore-linked games, treat TimeToScore as the source of truth for events,
                            # but still allow shift-spreadsheet uploads to *augment* the timeline with non-T2S events
                            # (e.g., SOG/xG/entries). We do not overwrite any existing rows.
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

                            if tts_linked and existing_csv.strip() and existing_source.lower().startswith("timetoscore"):
                                try:
                                    ex_headers, ex_rows = parse_events_csv(existing_csv)
                                    in_headers, in_rows = parse_events_csv(events_csv)
                                    if incoming_events_headers_raw is not None and incoming_events_rows_raw is not None:
                                        ex_headers, ex_rows = enrich_timetoscore_goals_with_long_video_times(
                                            existing_headers=ex_headers,
                                            existing_rows=ex_rows,
                                            incoming_headers=incoming_events_headers_raw,
                                            incoming_rows=incoming_events_rows_raw,
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
                                        gs = str(r.get("Game Seconds") or r.get("GameSeconds") or "").strip()
                                        tr = str(
                                            r.get("Team Side")
                                            or r.get("TeamSide")
                                            or r.get("Team Rel")
                                            or r.get("TeamRel")
                                            or r.get("Team")
                                            or ""
                                        ).strip().casefold()
                                        jerseys = str(r.get("Attributed Jerseys") or "").strip()
                                        return (et, per, gs, tr, jerseys)

                                    seen = {_key(r) for r in (ex_rows or []) if isinstance(r, dict)}
                                    merged_rows: list[dict[str, str]] = list(ex_rows or [])
                                    for rr in in_rows or []:
                                        if not isinstance(rr, dict):
                                            continue
                                        et = _norm_ev_type(rr.get("Event Type") or rr.get("Event") or "")
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
                                    merged_csv = to_csv_text(merged_headers, merged_rows)
                                    cur.execute(
                                        """
                                        INSERT INTO hky_game_events(game_id, events_csv, source_label, updated_at)
                                        VALUES(%s,%s,%s,%s)
                                        ON DUPLICATE KEY UPDATE events_csv=VALUES(events_csv), updated_at=VALUES(updated_at)
                                        """,
                                        (
                                            resolved_game_id,
                                            merged_csv,
                                            existing_source or source_label,
                                            dt.datetime.now().isoformat(),
                                        ),
                                    )
                                except Exception:
                                    # Best-effort merge; never fail the upload.
                                    pass

                if isinstance(player_stats_csv, str) and player_stats_csv.strip():
                    try:
                        player_stats_csv = sanitize_player_stats_csv_for_storage(player_stats_csv)
                    except Exception:
                        pass
                    # Persist the raw player_stats.csv for full-fidelity UI rendering.
                    if replace:
                        cur.execute(
                            """
                            INSERT INTO hky_game_player_stats_csv(game_id, player_stats_csv, source_label, updated_at)
                            VALUES(%s,%s,%s,%s)
                            ON DUPLICATE KEY UPDATE player_stats_csv=VALUES(player_stats_csv), source_label=VALUES(source_label), updated_at=VALUES(updated_at)
                            """,
                            (resolved_game_id, player_stats_csv, source_label, dt.datetime.now().isoformat()),
                        )
                    else:
                        cur.execute(
                            "SELECT player_stats_csv FROM hky_game_player_stats_csv WHERE game_id=%s",
                            (resolved_game_id,),
                        )
                        if not cur.fetchone():
                            cur.execute(
                                """
                                INSERT INTO hky_game_player_stats_csv(game_id, player_stats_csv, source_label, updated_at)
                                VALUES(%s,%s,%s,%s)
                                """,
                                (resolved_game_id, player_stats_csv, source_label, dt.datetime.now().isoformat()),
                            )

                if isinstance(game_stats_csv, str) and game_stats_csv.strip():
                    try:
                        game_stats = parse_shift_stats_game_stats_csv(game_stats_csv)
                    except Exception:
                        game_stats = None
                    if game_stats is not None:
                        game_stats = filter_game_stats_for_display(game_stats)
                        cur.execute(
                            """
                            INSERT INTO hky_game_stats(game_id, stats_json, updated_at)
                            VALUES(%s,%s,%s)
                            ON DUPLICATE KEY UPDATE stats_json=VALUES(stats_json), updated_at=VALUES(updated_at)
                            """,
                            (resolved_game_id, json.dumps(game_stats, ensure_ascii=False), dt.datetime.now().isoformat()),
                        )

                if isinstance(player_stats_csv, str) and player_stats_csv.strip():
                    parsed_rows = parse_shift_stats_player_stats_csv(player_stats_csv)
                    if replace:
                        # Ensure shift/time-derived stats are not stored when replacing.
                        cur.execute(
                            """
                            UPDATE player_stats
                            SET toi_seconds=NULL, shifts=NULL, video_toi_seconds=NULL,
                                sb_avg_shift_seconds=NULL, sb_median_shift_seconds=NULL,
                                sb_longest_shift_seconds=NULL, sb_shortest_shift_seconds=NULL
                            WHERE game_id=%s
                            """,
                            (resolved_game_id,),
                        )
                        cur.execute("DELETE FROM player_period_stats WHERE game_id=%s", (resolved_game_id,))
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
                                    m = re.match(r"^\s*\d+\s+(.*)$", disp)
                                    if m:
                                        disp = str(m.group(1) or "").strip()
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
                                    except Exception:
                                        pid = None
                            if pid is None:
                                unmatched.append(row.get("player_label") or "")
                                continue
                        # Determine team_id for this player
                        team_id = None
                        for t_players in players_by_team.values():
                            for pr in t_players:
                                if int(pr["id"]) == int(pid):
                                    team_id = int(pr["team_id"])
                                    break
                            if team_id is not None:
                                break
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
                        placeholders = ",".join(["%s"] * len(cols))
                        update_clause = ", ".join([f"{c}=COALESCE(VALUES({c}), {c})" for c in cols])
                        params = [owner_user_id, team_id, resolved_game_id, pid] + [stats.get(c) for c in cols]
                        cur.execute(
                            f"""
                            INSERT INTO player_stats(user_id, team_id, game_id, player_id, {', '.join(cols)})
                            VALUES(%s,%s,%s,%s,{placeholders})
                            ON DUPLICATE KEY UPDATE {update_clause}
                            """,
                            params,
                        )

                        imported += 1

                if player_stats_csv or game_stats_csv or events_csv:
                    cur.execute(
                        "UPDATE hky_games SET stats_imported_at=%s WHERE id=%s",
                        (dt.datetime.now().isoformat(), resolved_game_id),
                    )

            g.db.commit()
        except Exception as e:
            try:
                g.db.rollback()
            except Exception:
                pass
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
        Hidden administrative endpoint used by tools/webapp/reset_league_data.py.
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

        with g.db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (owner_email,))
            row = cur.fetchone()
            if not row:
                return jsonify({"ok": False, "error": "owner_email_not_found"}), 404
            owner_user_id = int(row[0])
            cur.execute("SELECT id FROM leagues WHERE name=%s AND owner_user_id=%s", (league_name, owner_user_id))
            lrow = cur.fetchone()
            if not lrow:
                return jsonify({"ok": False, "error": "league_not_found_for_owner"}), 404
            league_id = int(lrow[0])

        try:
            stats = reset_league_data(g.db, league_id, owner_user_id=owner_user_id)
        except Exception as e:  # noqa: BLE001
            try:
                g.db.rollback()
            except Exception:
                pass
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "league_id": league_id, "stats": stats})

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
        is_shared = bool(payload.get("shared", True))
        if not league_name or not owner_email:
            return jsonify({"ok": False, "error": "owner_email and league_name are required"}), 400

        owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
        with g.db.cursor() as cur:
            cur.execute("SELECT id FROM leagues WHERE name=%s", (league_name,))
            row = cur.fetchone()
            if row:
                league_id = int(row[0])
                cur.execute(
                    "UPDATE leagues SET owner_user_id=%s, is_shared=%s, updated_at=%s WHERE id=%s",
                    (owner_user_id, 1 if is_shared else 0, dt.datetime.now().isoformat(), league_id),
                )
            else:
                cur.execute(
                    "INSERT INTO leagues(name, owner_user_id, is_shared, is_public, created_at) VALUES(%s,%s,%s,%s,%s)",
                    (league_name, owner_user_id, 1 if is_shared else 0, 0, dt.datetime.now().isoformat()),
                )
                league_id = int(cur.lastrowid)
            cur.execute(
                """
                INSERT INTO league_members(league_id, user_id, role, created_at)
                VALUES(%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE role=%s
                """,
                (league_id, owner_user_id, "owner", dt.datetime.now().isoformat(), "owner"),
            )
        g.db.commit()
        return jsonify({"ok": True, "league_id": league_id, "owner_user_id": owner_user_id})

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
        with g.db.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO leagues(name, owner_user_id, is_shared, is_public, created_at) VALUES(%s,%s,%s,%s,%s)",
                    (name, session["user_id"], is_shared, is_public, dt.datetime.now().isoformat()),
                )
                lid = int(cur.lastrowid)
                # Add owner as admin member
                cur.execute(
                    "INSERT INTO league_members(league_id, user_id, role, created_at) VALUES(%s,%s,%s,%s)",
                    (lid, session["user_id"], "admin", dt.datetime.now().isoformat()),
                )
                g.db.commit()
            except Exception:
                g.db.rollback()
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
        with g.db.cursor() as cur:
            cur.execute(
                "UPDATE leagues SET is_shared=%s, is_public=%s, updated_at=%s WHERE id=%s",
                (is_shared, is_public, dt.datetime.now().isoformat(), league_id),
            )
        g.db.commit()
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

        try:
            with g.db.cursor() as cur:
                cur.execute("SELECT owner_user_id FROM leagues WHERE id=%s", (league_id,))
                row = cur.fetchone()
                if not row:
                    flash("Not found", "error")
                    return redirect(url_for("leagues_index"))
                owner_user_id = int(row[0])

                # Only delete games/teams that are exclusively mapped to this league.
                cur.execute(
                    """
                    SELECT game_id
                    FROM league_games
                    WHERE league_id=%s
                      AND game_id NOT IN (SELECT game_id FROM league_games WHERE league_id<>%s)
                    """,
                    (league_id, league_id),
                )
                exclusive_game_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})
                cur.execute(
                    """
                    SELECT team_id
                    FROM league_teams
                    WHERE league_id=%s
                      AND team_id NOT IN (SELECT team_id FROM league_teams WHERE league_id<>%s)
                    """,
                    (league_id, league_id),
                )
                exclusive_team_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})

                # Remove mappings + league row first.
                cur.execute("DELETE FROM league_games WHERE league_id=%s", (league_id,))
                cur.execute("DELETE FROM league_teams WHERE league_id=%s", (league_id,))
                cur.execute("DELETE FROM league_members WHERE league_id=%s", (league_id,))
                cur.execute("DELETE FROM leagues WHERE id=%s", (league_id,))

                # Delete exclusive games (cascades to player_stats etc).
                for chunk in _chunks(exclusive_game_ids, n=500):
                    q = ",".join(["%s"] * len(chunk))
                    cur.execute(f"DELETE FROM hky_games WHERE id IN ({q})", tuple(chunk))

                # Delete eligible external teams owned by league owner that are now unused by any remaining games.
                eligible_team_ids: list[int] = []
                if exclusive_team_ids:
                    q = ",".join(["%s"] * len(exclusive_team_ids))
                    cur.execute(f"SELECT id, user_id, is_external FROM teams WHERE id IN ({q})", tuple(exclusive_team_ids))
                    for tid, uid, is_ext in (cur.fetchall() or []):
                        if int(uid) == owner_user_id and int(is_ext) == 1:
                            eligible_team_ids.append(int(tid))
                safe_team_ids: list[int] = []
                if eligible_team_ids:
                    q2 = ",".join(["%s"] * len(eligible_team_ids))
                    cur.execute(
                        f"""
                        SELECT DISTINCT team1_id AS tid FROM hky_games WHERE team1_id IN ({q2})
                        UNION
                        SELECT DISTINCT team2_id AS tid FROM hky_games WHERE team2_id IN ({q2})
                        """,
                        tuple(eligible_team_ids) * 2,
                    )
                    still_used = {int(r[0]) for r in (cur.fetchall() or [])}
                    safe_team_ids = [tid for tid in eligible_team_ids if tid not in still_used]
                for chunk in _chunks(sorted(safe_team_ids), n=500):
                    q = ",".join(["%s"] * len(chunk))
                    cur.execute(f"DELETE FROM teams WHERE id IN ({q})", tuple(chunk))
            g.db.commit()
            if session.get("league_id") == league_id:
                session.pop("league_id", None)
            flash("League and associated data deleted", "success")
        except Exception:
            g.db.rollback()
            flash("Failed to delete league", "error")
        return redirect(url_for("leagues_index"))

    def _is_league_admin(league_id: int, user_id: int) -> bool:
        with g.db.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM leagues WHERE id=%s AND owner_user_id=%s",
                (league_id, user_id),
            )
            if cur.fetchone():
                return True
            cur.execute(
                "SELECT 1 FROM league_members WHERE league_id=%s AND user_id=%s AND role IN ('admin','owner')",
                (league_id, user_id),
            )
            return bool(cur.fetchone())

    def _is_public_league(league_id: int) -> Optional[dict]:
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT id, name FROM leagues WHERE id=%s AND is_public=1", (league_id,))
            return cur.fetchone()

    @app.get("/public/leagues")
    def public_leagues_index():
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT id, name FROM leagues WHERE is_public=1 ORDER BY name")
            leagues = cur.fetchall() or []
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT t.id, t.logo_path
                FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                WHERE lt.league_id=%s AND t.id=%s
                """,
                (league_id, team_id),
            )
            row = cur.fetchone()
        if not row or not row.get("logo_path"):
            return ("Not found", 404)
        p = Path(row["logo_path"]).resolve()
        if not p.exists():
            return ("Not found", 404)
        return send_from_directory(str(p.parent), p.name)

    @app.get("/public/leagues/<int:league_id>/teams")
    def public_league_teams(league_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT t.*, lt.division_name, lt.division_id, lt.conference_id,
                       lt.mhr_rating, lt.mhr_agd, lt.mhr_sched, lt.mhr_games, lt.mhr_updated_at
                FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                WHERE lt.league_id=%s
                """,
                (league_id,),
            )
            rows = cur.fetchall() or []
        stats = {t["id"]: compute_team_stats_league(g.db, t["id"], int(league_id)) for t in rows}
        grouped: dict[str, list[dict]] = {}
        for t in rows:
            dn = str(t.get("division_name") or "").strip() or "Unknown Division"
            grouped.setdefault(dn, []).append(t)
        divisions = []
        for dn in sorted(grouped.keys(), key=lambda s: s.lower()):
            teams_sorted = sorted(grouped[dn], key=lambda tr: sort_key_team_standings(tr, stats.get(tr["id"], {})))
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
        )

    @app.get("/public/leagues/<int:league_id>/teams/<int:team_id>")
    def public_league_team_detail(league_id: int, team_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        recent_n_raw = request.args.get("recent_n")
        try:
            recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
        except Exception:
            recent_n = 5
        recent_sort = str(request.args.get("recent_sort") or "points").strip() or "points"
        recent_dir = str(request.args.get("recent_dir") or "desc").strip().lower() or "desc"
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT t.*
                FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                WHERE lt.league_id=%s AND t.id=%s
                """,
                (league_id, team_id),
            )
            team = cur.fetchone()
        if not team:
            return ("Not found", 404)
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC", (team_id,))
            players = cur.fetchall() or []
        skaters, goalies, head_coaches, assistant_coaches = split_roster(players or [])
        roster_players = list(skaters) + list(goalies)
        tstats = compute_team_stats_league(g.db, team_id, int(league_id))
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name,
                       lg.division_name AS division_name, lg.sort_order AS sort_order,
                       lt1.division_name AS team1_league_division_name,
                       lt2.division_name AS team2_league_division_name
                FROM league_games lg
                  JOIN hky_games g ON lg.game_id=g.id
                  JOIN teams t1 ON g.team1_id=t1.id
                  JOIN teams t2 ON g.team2_id=t2.id
                  LEFT JOIN game_types gt ON g.game_type_id=gt.id
                  LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
                  LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
                WHERE lg.league_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
                ORDER BY (g.starts_at IS NULL) ASC, g.starts_at ASC, COALESCE(lg.sort_order, 2147483647) ASC, g.created_at ASC
                """,
                (int(league_id), team_id, team_id),
            )
            schedule_games = cur.fetchall() or []
            now_dt = dt.datetime.now()
            schedule_games = [g2 for g2 in (schedule_games or []) if not _league_game_is_cross_division_non_external(g2)]
            for g2 in schedule_games:
                sdt = g2.get("starts_at")
                started = False
                if sdt is not None:
                    try:
                        started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                    except Exception:
                        started = False
                has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
                g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
                try:
                    g2["game_video_url"] = _sanitize_http_url(_extract_game_video_url_from_notes(g2.get("notes")))
                except Exception:
                    g2["game_video_url"] = None
            schedule_games = sort_games_schedule_order(schedule_games or [])

        cols_sql = ", ".join([f"ps.{c}" for c in PLAYER_STATS_SUM_KEYS])
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                f"""
		                SELECT ps.player_id, ps.game_id, {cols_sql}
		                FROM league_games lg
		                  JOIN hky_games g ON lg.game_id=g.id
		                  JOIN player_stats ps ON lg.game_id=ps.game_id
		                  LEFT JOIN league_teams lt_self ON lt_self.league_id=lg.league_id AND lt_self.team_id=ps.team_id
		                  LEFT JOIN league_teams lt_opp ON lt_opp.league_id=lg.league_id AND lt_opp.team_id=(
		                       CASE WHEN g.team1_id=ps.team_id THEN g.team2_id ELSE g.team1_id END
		                  )
		                WHERE lg.league_id=%s AND ps.team_id=%s
		                  AND (
		                    LOWER(COALESCE(lg.division_name,''))='external'
		                    OR lt_opp.division_name IS NULL
		                    OR LOWER(COALESCE(lt_opp.division_name,''))='external'
		                    OR lt_self.division_name IS NULL
		                    OR lt_self.division_name=lt_opp.division_name
		                  )
		                """,
                (int(league_id), team_id),
            )
            ps_rows = cur.fetchall() or []

        # Game type filter applies to player stats tables only.
        for g2 in schedule_games or []:
            try:
                g2["_game_type_label"] = _game_type_label_for_row(g2)
            except Exception:
                g2["_game_type_label"] = "Unknown"
        game_type_options = _dedupe_preserve_str([str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])])
        selected_types = _parse_selected_game_type_labels(available=game_type_options, args=request.args)
        stats_schedule_games = (
            list(schedule_games or [])
            if selected_types is None
            else [g2 for g2 in (schedule_games or []) if str(g2.get("_game_type_label") or "") in selected_types]
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

        player_totals = _aggregate_player_totals_from_rows(player_stats_rows=ps_rows_filtered, allowed_game_ids=eligible_game_ids)
        player_stats_rows = sort_players_table_default(build_player_stats_table_rows(skaters, player_totals))
        player_stats_columns = filter_player_stats_display_columns_for_rows(PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows)
        cov_counts, cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=eligible_game_ids_in_order
        )
        player_stats_columns = _player_stats_columns_with_coverage(
            columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
        )

        recent_scope_ids = eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
        recent_totals = compute_recent_player_totals_from_rows(
            schedule_games=stats_schedule_games, player_stats_rows=ps_rows_filtered, n=recent_n
        )
        recent_player_stats_rows = sort_player_stats_rows(
            build_player_stats_table_rows(skaters, recent_totals), sort_key=recent_sort, sort_dir=recent_dir
        )
        recent_player_stats_columns = filter_player_stats_display_columns_for_rows(
            PLAYER_STATS_DISPLAY_COLUMNS, recent_player_stats_rows
        )
        recent_cov_counts, recent_cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=recent_scope_ids
        )
        recent_player_stats_columns = _player_stats_columns_with_coverage(
            columns=recent_player_stats_columns, coverage_counts=recent_cov_counts, total_games=recent_cov_total
        )

        player_stats_sources = _compute_team_player_stats_sources(g.db, eligible_game_ids=eligible_game_ids_in_order)
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
        )

    @app.get("/public/leagues/<int:league_id>/schedule")
    def public_league_schedule(league_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        selected_division = (request.args.get("division") or "").strip() or None
        selected_team_id = request.args.get("team_id") or ""
        team_id_i: Optional[int] = None
        try:
            team_id_i = int(selected_team_id) if str(selected_team_id).strip() else None
        except Exception:
            team_id_i = None
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT division_name
                FROM league_teams
                WHERE league_id=%s AND division_name IS NOT NULL AND division_name<>''
                ORDER BY division_name
                """,
                (league_id,),
            )
            divisions = [str(r["division_name"]) for r in (cur.fetchall() or []) if r.get("division_name")]
            if selected_division:
                cur.execute(
                    """
                    SELECT DISTINCT t.id, t.name
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s AND lt.division_name=%s
                    ORDER BY t.name
                    """,
                    (league_id, selected_division),
                )
            else:
                cur.execute(
                    """
                    SELECT DISTINCT t.id, t.name
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s
                    ORDER BY t.name
                    """,
                    (league_id,),
                )
            league_teams = cur.fetchall() or []
            if team_id_i is not None and not any(int(t["id"]) == int(team_id_i) for t in league_teams):
                team_id_i = None
                selected_team_id = ""
            where = ["lg.league_id=%s"]
            params: list[Any] = [league_id]
            if selected_division:
                where.append("lg.division_name=%s")
                params.append(selected_division)
            if team_id_i is not None:
                where.append("(g.team1_id=%s OR g.team2_id=%s)")
                params.extend([team_id_i, team_id_i])
            cur.execute(
                f"""
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name,
                       lg.division_name AS division_name,
                       lt1.division_name AS team1_league_division_name,
                       lt2.division_name AS team2_league_division_name
                FROM league_games lg
                  JOIN hky_games g ON lg.game_id=g.id
                  JOIN teams t1 ON g.team1_id=t1.id
                  JOIN teams t2 ON g.team2_id=t2.id
                  LEFT JOIN game_types gt ON g.game_type_id=gt.id
                  LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
                  LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
                WHERE {' AND '.join(where)}
                ORDER BY (g.starts_at IS NULL) ASC, g.starts_at ASC, COALESCE(lg.sort_order, 2147483647) ASC, g.created_at ASC
                """,
                tuple(params),
            )
            games = cur.fetchall() or []
        games = [g2 for g2 in (games or []) if not _league_game_is_cross_division_non_external(g2)]
        now_dt = dt.datetime.now()
        for g2 in games or []:
            try:
                g2["game_video_url"] = _sanitize_http_url(_extract_game_video_url_from_notes(g2.get("notes")))
            except Exception:
                g2["game_video_url"] = None
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
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
        )

    @app.get("/public/leagues/<int:league_id>/hky/games/<int:game_id>")
    def public_league_game_detail(league_id: int, game_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, t1.is_external AS team1_ext, t2.is_external AS team2_ext,
                       lg.division_name AS division_name,
                       lt1.division_name AS team1_league_division_name,
                       lt2.division_name AS team2_league_division_name
                FROM league_games lg JOIN hky_games g ON lg.game_id=g.id
                  JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id
                  LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
                  LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
                WHERE g.id=%s AND lg.league_id=%s
                """,
                (game_id, league_id),
            )
            game = cur.fetchone()
        if not game:
            return ("Not found", 404)
        try:
            game["game_video_url"] = _sanitize_http_url(_extract_game_video_url_from_notes(game.get("notes")))
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
        has_score = (game.get("team1_score") is not None) or (game.get("team2_score") is not None) or bool(game.get("is_final"))
        can_view_summary = bool(has_score or (sdt is None) or started)
        if not can_view_summary:
            return ("Not found", 404)
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC", (game["team1_id"],))
            team1_players = cur.fetchall() or []
            cur.execute("SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC", (game["team2_id"],))
            team2_players = cur.fetchall() or []
            cur.execute("SELECT * FROM player_stats WHERE game_id=%s", (game_id,))
            stats_rows = cur.fetchall() or []
            cur.execute("SELECT stats_json, updated_at FROM hky_game_stats WHERE game_id=%s", (game_id,))
            game_stats_row = cur.fetchone()
        team1_skaters, team1_goalies, team1_hc, team1_ac = split_roster(team1_players)
        team2_skaters, team2_goalies, team2_hc, team2_ac = split_roster(team2_players)
        team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
        team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
        stats_by_pid = {r["player_id"]: r for r in stats_rows}
        game_stats = None
        game_stats_updated_at = None
        try:
            if game_stats_row and game_stats_row.get("stats_json"):
                game_stats = json.loads(game_stats_row["stats_json"])
                game_stats_updated_at = game_stats_row.get("updated_at")
        except Exception:
            game_stats = None
        game_stats = filter_game_stats_for_display(game_stats)
        period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}
        tts_linked = _extract_timetoscore_game_id_from_notes(game.get("notes")) is not None
        tts_linked = _extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

        events_headers: list[str] = []
        events_rows: list[dict[str, str]] = []
        events_meta: Optional[dict[str, Any]] = None
        try:
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT events_csv, source_label, updated_at FROM hky_game_events WHERE game_id=%s",
                    (game_id,),
                )
                erow = cur.fetchone()
            if erow and str(erow.get("events_csv") or "").strip():
                events_headers, events_rows = parse_events_csv(str(erow.get("events_csv") or ""))
                events_headers, events_rows = normalize_game_events_csv(events_headers, events_rows)
                events_rows = filter_events_rows_prefer_timetoscore_for_goal_assist(events_rows, tts_linked=tts_linked)
                events_headers, events_rows = normalize_events_video_time_for_display(events_headers, events_rows)
                events_headers, events_rows = filter_events_headers_drop_empty_on_ice_split(events_headers, events_rows)
                events_rows = sort_events_rows_default(events_rows)
                events_meta = {
                    "source_label": erow.get("source_label"),
                    "updated_at": erow.get("updated_at"),
                    "count": len(events_rows),
                    "sources": summarize_event_sources(events_rows, fallback_source_label=str(erow.get("source_label") or "")),
                }
        except Exception:
            events_headers, events_rows, events_meta = [], [], None

        scoring_by_period_rows = compute_team_scoring_by_period_from_events(events_rows, tts_linked=tts_linked)
        game_event_stats_rows = compute_game_event_stats_by_side(events_rows)
        game_event_stats_rows = compute_game_event_stats_by_side(events_rows)

        imported_player_stats_csv_text: Optional[str] = None
        player_stats_import_meta: Optional[dict[str, Any]] = None
        try:
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT player_stats_csv, source_label, updated_at FROM hky_game_player_stats_csv WHERE game_id=%s",
                    (game_id,),
                )
                prow = cur.fetchone()
            if prow and str(prow.get("player_stats_csv") or "").strip():
                imported_player_stats_csv_text = str(prow.get("player_stats_csv") or "")
                player_stats_import_meta = {
                    "source_label": prow.get("source_label"),
                    "updated_at": prow.get("updated_at"),
                }
        except Exception:
            imported_player_stats_csv_text, player_stats_import_meta = None, None

        game_player_stats_columns, player_stats_cells_by_pid, player_stats_cell_conflicts_by_pid, player_stats_import_warning = (
            build_game_player_stats_table(
                players=list(team1_skaters) + list(team2_skaters),
                stats_by_pid=stats_by_pid,
                imported_csv_text=imported_player_stats_csv_text,
                prefer_db_stats_for_keys={"goals", "assists"} if tts_linked else None,
            )
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
                    rows.append({"player_id": pid, "name": str(p.get("name") or ""), "goals": g, "assists": a, "points": g + a})
                    by_pid[pid] = p
                sorted_rows = sort_players_table_default(rows)
                return [by_pid[int(r["player_id"])] for r in sorted_rows if int(r["player_id"]) in by_pid]

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
            game_stats=game_stats,
            game_stats_updated_at=game_stats_updated_at,
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
            user_video_clip_len_s=(get_user_video_clip_len_s(g.db, public_user_id) if public_is_logged_in else None),
            user_is_logged_in=public_is_logged_in,
            game_player_stats_columns=game_player_stats_columns,
            player_stats_cells_by_pid=player_stats_cells_by_pid,
            player_stats_cell_conflicts_by_pid=player_stats_cell_conflicts_by_pid,
            player_stats_import_meta=player_stats_import_meta,
            player_stats_import_warning=player_stats_import_warning,
        )

    @app.get("/leagues/<int:league_id>/members")
    def league_members(league_id: int):
        r = require_login()
        if r:
            return r
        if not _is_league_admin(league_id, session["user_id"]):
            flash("Not authorized", "error")
            return redirect(url_for("leagues_index"))
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT u.id, u.email, COALESCE(m.role,'admin') AS role FROM users u JOIN league_members m ON m.user_id=u.id WHERE m.league_id=%s ORDER BY u.email",
                (league_id,),
            )
            members = cur.fetchall()
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
        with g.db.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
            if not row:
                flash("User not found. Ask them to register first.", "error")
                return redirect(url_for("league_members", league_id=league_id))
            uid = int(row[0])
            cur.execute(
                "INSERT INTO league_members(league_id, user_id, role, created_at) VALUES(%s,%s,%s,%s) ON DUPLICATE KEY UPDATE role=VALUES(role)",
                (league_id, uid, role, dt.datetime.now().isoformat()),
            )
            g.db.commit()
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
        with g.db.cursor() as cur:
            cur.execute(
                "DELETE FROM league_members WHERE league_id=%s AND user_id=%s", (league_id, uid)
            )
            g.db.commit()
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
                    with g.db.cursor() as cur:
                        cur.execute(
                            "UPDATE teams SET logo_path=%s WHERE id=%s AND user_id=%s",
                            (str(p), tid, session["user_id"]),
                        )
                    g.db.commit()
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
        is_league_admin = False
        if league_id:
            try:
                is_league_admin = bool(_is_league_admin(int(league_id), int(session["user_id"])))
            except Exception:
                is_league_admin = False
        team = get_team(team_id, session["user_id"])
        editable = bool(team)
        if not team and league_id:
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT t.*
                    FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                    WHERE lt.league_id=%s AND t.id=%s
                    """,
                    (league_id, team_id),
                )
                team = cur.fetchone()
        if not team:
            flash("Not found", "error")
            return redirect(url_for("teams"))
        team_owner_id = int(team["user_id"])
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            if editable:
                cur.execute(
                    "SELECT * FROM players WHERE team_id=%s AND user_id=%s ORDER BY jersey_number ASC, name ASC",
                    (team_id, session["user_id"]),
                )
            else:
                cur.execute(
                    "SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC",
                    (team_id,),
                )
            players = cur.fetchall()
        skaters, goalies, head_coaches, assistant_coaches = split_roster(players or [])
        roster_players = list(skaters) + list(goalies)
        if league_id:
            tstats = compute_team_stats_league(g.db, team_id, int(league_id))
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name,
                           lg.division_name AS division_name, lg.sort_order AS sort_order,
                           lt1.division_name AS team1_league_division_name,
                           lt2.division_name AS team2_league_division_name
                    FROM league_games lg
                      JOIN hky_games g ON lg.game_id=g.id
                      JOIN teams t1 ON g.team1_id=t1.id
                      JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN game_types gt ON g.game_type_id=gt.id
                      LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
                      LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
                    WHERE lg.league_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
                    ORDER BY (g.starts_at IS NULL) ASC, g.starts_at ASC, COALESCE(lg.sort_order, 2147483647) ASC, g.created_at ASC
                    """,
                    (int(league_id), team_id, team_id),
                )
                schedule_games = cur.fetchall() or []

            schedule_games = [g2 for g2 in (schedule_games or []) if not _league_game_is_cross_division_non_external(g2)]
            now_dt = dt.datetime.now()
            for g2 in schedule_games:
                sdt = g2.get("starts_at")
                started = False
                if sdt is not None:
                    try:
                        started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                    except Exception:
                        started = False
                has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
                g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
                try:
                    g2["game_video_url"] = _sanitize_http_url(_extract_game_video_url_from_notes(g2.get("notes")))
                except Exception:
                    g2["game_video_url"] = None
            schedule_games = sort_games_schedule_order(schedule_games or [])
            cols_sql = ", ".join([f"ps.{c}" for c in PLAYER_STATS_SUM_KEYS])
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    f"""
	                    SELECT ps.player_id, ps.game_id, {cols_sql}
	                    FROM league_games lg
	                      JOIN hky_games g ON lg.game_id=g.id
	                      JOIN player_stats ps ON lg.game_id=ps.game_id
	                      LEFT JOIN league_teams lt_self ON lt_self.league_id=lg.league_id AND lt_self.team_id=ps.team_id
	                      LEFT JOIN league_teams lt_opp ON lt_opp.league_id=lg.league_id AND lt_opp.team_id=(
	                           CASE WHEN g.team1_id=ps.team_id THEN g.team2_id ELSE g.team1_id END
	                      )
	                    WHERE lg.league_id=%s AND ps.team_id=%s
	                      AND (
	                        LOWER(COALESCE(lg.division_name,''))='external'
	                        OR lt_opp.division_name IS NULL
	                        OR LOWER(COALESCE(lt_opp.division_name,''))='external'
	                        OR lt_self.division_name IS NULL
	                        OR lt_self.division_name=lt_opp.division_name
	                      )
	                    """,
                    (int(league_id), team_id),
                )
                ps_rows = cur.fetchall() or []
        else:
            tstats = compute_team_stats(g.db, team_id, team_owner_id)
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name
                    FROM hky_games g
                      JOIN teams t1 ON g.team1_id=t1.id
                      JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN game_types gt ON g.game_type_id=gt.id
                    WHERE g.user_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
                    ORDER BY (g.starts_at IS NULL) ASC, g.starts_at ASC, g.created_at ASC
                    """,
                    (team_owner_id, team_id, team_id),
                )
                schedule_games = cur.fetchall() or []
            now_dt = dt.datetime.now()
            for g2 in schedule_games:
                sdt = g2.get("starts_at")
                started = False
                if sdt is not None:
                    try:
                        started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                    except Exception:
                        started = False
                has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
                g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            cols_sql = ", ".join([str(c) for c in PLAYER_STATS_SUM_KEYS])
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    f"SELECT player_id, game_id, {cols_sql} FROM player_stats WHERE team_id=%s AND user_id=%s",
                    (team_id, team_owner_id),
                )
                ps_rows = cur.fetchall() or []

        # Game type filter applies to player stats tables only.
        for g2 in schedule_games or []:
            try:
                g2["_game_type_label"] = _game_type_label_for_row(g2)
            except Exception:
                g2["_game_type_label"] = "Unknown"
        game_type_options = _dedupe_preserve_str([str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])])
        selected_types = _parse_selected_game_type_labels(available=game_type_options, args=request.args)
        stats_schedule_games = (
            list(schedule_games or [])
            if selected_types is None
            else [g2 for g2 in (schedule_games or []) if str(g2.get("_game_type_label") or "") in selected_types]
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

        player_totals = _aggregate_player_totals_from_rows(player_stats_rows=ps_rows_filtered, allowed_game_ids=eligible_game_ids)
        player_stats_rows = sort_players_table_default(build_player_stats_table_rows(skaters, player_totals))
        player_stats_columns = filter_player_stats_display_columns_for_rows(PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows)
        cov_counts, cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=eligible_game_ids_in_order
        )
        player_stats_columns = _player_stats_columns_with_coverage(
            columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
        )
        recent_scope_ids = eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
        recent_totals = compute_recent_player_totals_from_rows(
            schedule_games=stats_schedule_games, player_stats_rows=ps_rows_filtered, n=recent_n
        )
        recent_player_stats_rows = sort_player_stats_rows(
            build_player_stats_table_rows(skaters, recent_totals), sort_key=recent_sort, sort_dir=recent_dir
        )
        recent_player_stats_columns = filter_player_stats_display_columns_for_rows(
            PLAYER_STATS_DISPLAY_COLUMNS, recent_player_stats_rows
        )
        recent_cov_counts, recent_cov_total = _compute_team_player_stats_coverage(
            player_stats_rows=ps_rows_filtered, eligible_game_ids=recent_scope_ids
        )
        recent_player_stats_columns = _player_stats_columns_with_coverage(
            columns=recent_player_stats_columns, coverage_counts=recent_cov_counts, total_games=recent_cov_total
        )

        player_stats_sources = _compute_team_player_stats_sources(g.db, eligible_game_ids=eligible_game_ids_in_order)
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
            editable=editable,
            is_league_admin=is_league_admin,
            player_stats_sources=player_stats_sources,
            player_stats_coverage_total_games=cov_total,
            player_stats_recent_coverage_total_games=recent_cov_total,
            game_type_filter_options=game_type_filter_options,
            game_type_filter_label=selected_label,
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
                with g.db.cursor() as cur:
                    cur.execute(
                        "UPDATE teams SET name=%s WHERE id=%s AND user_id=%s",
                        (name, team_id, session["user_id"]),
                    )
                g.db.commit()
            f = request.files.get("logo")
            if f and f.filename:
                p = save_team_logo(f, team_id)
                with g.db.cursor() as cur:
                    cur.execute(
                        "UPDATE teams SET logo_path=%s WHERE id=%s AND user_id=%s",
                        (str(p), team_id, session["user_id"]),
                    )
                g.db.commit()
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
            with g.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO players(user_id, team_id, name, jersey_number, position, shoots, created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        session["user_id"],
                        team_id,
                        name,
                        jersey or None,
                        position or None,
                        shoots or None,
                        dt.datetime.now().isoformat(),
                    ),
                )
            g.db.commit()
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT * FROM players WHERE id=%s AND team_id=%s AND user_id=%s",
                (player_id, team_id, session["user_id"]),
            )
            pl = cur.fetchone()
        if not pl:
            flash("Not found", "error")
            return redirect(url_for("team_detail", team_id=team_id))
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            jersey = request.form.get("jersey_number", "").strip()
            position = request.form.get("position", "").strip()
            shoots = request.form.get("shoots", "").strip()
            with g.db.cursor() as cur:
                cur.execute(
                    "UPDATE players SET name=%s, jersey_number=%s, position=%s, shoots=%s WHERE id=%s AND team_id=%s AND user_id=%s",
                    (
                        name or pl["name"],
                        jersey or None,
                        position or None,
                        shoots or None,
                        player_id,
                        team_id,
                        session["user_id"],
                    ),
                )
            g.db.commit()
            flash("Player updated", "success")
            return redirect(url_for("team_detail", team_id=team_id))
        return render_template("player_edit.html", team=team, player=pl)

    @app.route("/teams/<int:team_id>/players/<int:player_id>/delete", methods=["POST"])
    def player_delete(team_id: int, player_id: int):
        r = require_login()
        if r:
            return r
        with g.db.cursor() as cur:
            cur.execute(
                "DELETE FROM players WHERE id=%s AND team_id=%s AND user_id=%s",
                (player_id, team_id, session["user_id"]),
            )
        g.db.commit()
        flash("Player deleted", "success")
        return redirect(url_for("team_detail", team_id=team_id))

    @app.route("/schedule")
    def schedule():
        r = require_login()
        if r:
            return r
        league_id = session.get("league_id")
        selected_division = (request.args.get("division") or "").strip() or None
        selected_team_id = request.args.get("team_id") or ""
        team_id_i: Optional[int] = None
        try:
            team_id_i = int(selected_team_id) if str(selected_team_id).strip() else None
        except Exception:
            team_id_i = None
        divisions = []
        league_teams = []
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            if league_id:
                # Filter options for league schedule
                cur.execute(
                    """
                    SELECT DISTINCT division_name
                    FROM league_teams
                    WHERE league_id=%s AND division_name IS NOT NULL AND division_name<>''
                    ORDER BY division_name
                    """,
                    (league_id,),
                )
                divisions = [str(r["division_name"]) for r in (cur.fetchall() or []) if r.get("division_name")]
                if selected_division:
                    cur.execute(
                        """
                        SELECT DISTINCT t.id, t.name
                        FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                        WHERE lt.league_id=%s AND lt.division_name=%s
                        ORDER BY t.name
                        """,
                        (league_id, selected_division),
                    )
                else:
                    cur.execute(
                        """
                        SELECT DISTINCT t.id, t.name
                        FROM league_teams lt JOIN teams t ON lt.team_id=t.id
                        WHERE lt.league_id=%s
                        ORDER BY t.name
                        """,
                        (league_id,),
                    )
                league_teams = cur.fetchall() or []
                if team_id_i is not None and not any(int(t["id"]) == int(team_id_i) for t in league_teams):
                    team_id_i = None
                    selected_team_id = ""

                where = ["lg.league_id=%s"]
                params: list[Any] = [league_id]
                if selected_division:
                    where.append("lg.division_name=%s")
                    params.append(selected_division)
                if team_id_i is not None:
                    where.append("(g.team1_id=%s OR g.team2_id=%s)")
                    params.extend([team_id_i, team_id_i])
                cur.execute(
                    f"""
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name,
                           lg.division_name AS division_name,
                           lt1.division_name AS team1_league_division_name,
                           lt2.division_name AS team2_league_division_name
                    FROM league_games lg
                      JOIN hky_games g ON lg.game_id=g.id
                      JOIN teams t1 ON g.team1_id=t1.id
                      JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN game_types gt ON g.game_type_id=gt.id
                      LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
                      LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
                    WHERE {' AND '.join(where)}
                    ORDER BY (g.starts_at IS NULL) ASC, g.starts_at ASC, COALESCE(lg.sort_order, 2147483647) ASC, g.created_at ASC
                    """,
                    tuple(params),
                )
            else:
                cur.execute(
                    """
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name
                    FROM hky_games g
                      JOIN teams t1 ON g.team1_id=t1.id
                      JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN game_types gt ON g.game_type_id=gt.id
                    WHERE g.user_id=%s
                    ORDER BY (g.starts_at IS NULL) ASC, g.starts_at ASC, g.created_at ASC
                    """,
                    (session["user_id"],),
                )
            games = cur.fetchall()
        if league_id:
            games = [g2 for g2 in (games or []) if not _league_game_is_cross_division_non_external(g2)]
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
            has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
            # Hide game pages for future scheduled games that have not started and have no score yet.
            # If starts_at is missing (common for imported games), allow viewing.
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = _sanitize_http_url(_extract_game_video_url_from_notes(g2.get("notes")))
            except Exception:
                g2["game_video_url"] = None
            # Editing is gated to owners or league admins; UI still defaults to read-only unless Edit is clicked.
            try:
                g2["can_edit"] = bool(int(g2.get("user_id") or 0) == int(session["user_id"]) or is_league_admin)
            except Exception:
                g2["can_edit"] = bool(is_league_admin)
        games = sort_games_schedule_order(games or [])
        return render_template(
            "schedule.html",
            games=games,
            league_view=bool(league_id),
            divisions=divisions,
            league_teams=league_teams,
            selected_division=selected_division or "",
            selected_team_id=str(team_id_i) if team_id_i is not None else "",
        )

    @app.route("/schedule/new", methods=["GET", "POST"])
    def schedule_new():
        r = require_login()
        if r:
            return r
        # Load user's own teams (not external)
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, name FROM teams WHERE user_id=%s AND is_external=0 ORDER BY name",
                (session["user_id"],),
            )
            my_teams = cur.fetchall()
            cur.execute("SELECT id, name FROM game_types ORDER BY name")
            gt = cur.fetchall()
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
                starts_at=parse_dt_or_none(starts_at),
                location=location or None,
            )
            # If a league is selected, map teams and game into the league
            league_id = session.get("league_id")
            if league_id:
                try:
                    with g.db.cursor() as cur:
                        cur.execute(
                            "INSERT IGNORE INTO league_teams(league_id, team_id) VALUES(%s,%s)",
                            (league_id, team1_id),
                        )
                        cur.execute(
                            "INSERT IGNORE INTO league_teams(league_id, team_id) VALUES(%s,%s)",
                            (league_id, team2_id),
                        )
                        cur.execute(
                            "INSERT IGNORE INTO league_games(league_id, game_id) VALUES(%s,%s)",
                            (league_id, gid),
                        )
                    g.db.commit()
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
        game = None
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, t1.is_external AS team1_ext, t2.is_external AS team2_ext
                FROM hky_games g JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id
                WHERE g.id=%s AND g.user_id=%s
                """,
                (game_id, session["user_id"]),
            )
            game = cur.fetchone()
            if not game and league_id:
                cur.execute(
                    """
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, t1.is_external AS team1_ext, t2.is_external AS team2_ext
                    FROM league_games lg JOIN hky_games g ON lg.game_id=g.id
                      JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id
                    WHERE g.id=%s AND lg.league_id=%s
                    """,
                    (game_id, league_id),
                )
                game = cur.fetchone()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("schedule"))

        tts_linked = _extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

        # Authorization: only allow edits if owner or league admin/owner.
        editable = True
        if league_id:
            if not _is_league_admin(int(league_id), int(session["user_id"])):
                editable = False
        if not editable:
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

        # Optional game_stats.csv
        game_stats = None
        gs_file = request.files.get("game_stats_csv")
        if gs_file and gs_file.filename:
            try:
                gs_text = gs_file.stream.read().decode("utf-8", errors="replace")
                game_stats = parse_shift_stats_game_stats_csv(gs_text)
            except Exception:
                game_stats = None

        # Load players for both teams so we can map by jersey/name.
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, team_id, name, jersey_number FROM players WHERE user_id=%s AND team_id IN (%s,%s)",
                (session["user_id"], game["team1_id"], game["team2_id"]),
            )
            players = cur.fetchall()

        players_by_team: dict[int, list[dict]] = {}
        jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
        name_to_player_ids: dict[tuple[int, str], list[int]] = {}

        for p in players:
            team_id = int(p["team_id"])
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
                        pl = next((x for x in players_by_team.get(team_id, []) if int(x["id"]) == pid), None)
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

        with g.db.cursor() as cur:
            # Persist the raw player_stats.csv for full-fidelity UI rendering.
            try:
                ps_text_sanitized = sanitize_player_stats_csv_for_storage(ps_text)
                cur.execute(
                    """
                    INSERT INTO hky_game_player_stats_csv(game_id, player_stats_csv, source_label, updated_at)
                    VALUES(%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE player_stats_csv=VALUES(player_stats_csv), source_label=VALUES(source_label), updated_at=VALUES(updated_at)
                    """,
                    (game_id, ps_text_sanitized, "upload_form", dt.datetime.now().isoformat()),
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

                # Resolve team_id for this player_id (for FK consistency)
                team_id = None
                for team_players in players_by_team.values():
                    for p in team_players:
                        if int(p["id"]) == int(pid):
                            team_id = int(p["team_id"])
                            break
                    if team_id is not None:
                        break
                if team_id is None:
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
                placeholders = ",".join(["%s"] * len(cols))
                update_clause = ", ".join([f"{c}=COALESCE(VALUES({c}), {c})" for c in cols])
                params = [session["user_id"], team_id, game_id, pid] + [stats.get(c) for c in cols]
                cur.execute(
                    f"""
                    INSERT INTO player_stats(user_id, team_id, game_id, player_id, {', '.join(cols)})
                    VALUES(%s,%s,%s,%s,{placeholders})
                    ON DUPLICATE KEY UPDATE {update_clause}
                    """,
                    params,
                )

                imported += 1

            if game_stats is not None:
                game_stats = filter_game_stats_for_display(game_stats)
                cur.execute(
                    """
                    INSERT INTO hky_game_stats(game_id, stats_json, updated_at)
                    VALUES(%s,%s,%s)
                    ON DUPLICATE KEY UPDATE stats_json=VALUES(stats_json), updated_at=VALUES(updated_at)
                    """,
                    (game_id, json.dumps(game_stats, ensure_ascii=False), dt.datetime.now().isoformat()),
                )

            # Track import time
            cur.execute(
                "UPDATE hky_games SET stats_imported_at=%s WHERE id=%s",
                (dt.datetime.now().isoformat(), game_id),
            )
        g.db.commit()

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
        game = None
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, t1.is_external AS team1_ext, t2.is_external AS team2_ext
                FROM hky_games g JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id
                WHERE g.id=%s AND g.user_id=%s
                """,
                (game_id, session["user_id"]),
            )
            game = cur.fetchone()
            if not game and league_id:
                cur.execute(
                    """
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, t1.is_external AS team1_ext, t2.is_external AS team2_ext,
                           lg.division_name AS division_name,
                           lt1.division_name AS team1_league_division_name,
                           lt2.division_name AS team2_league_division_name
                    FROM league_games lg JOIN hky_games g ON lg.game_id=g.id
                      JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
                      LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
                    WHERE g.id=%s AND lg.league_id=%s
                    """,
                    (game_id, league_id),
                )
                game = cur.fetchone()
        if not game:
            flash("Not found", "error")
            return redirect(url_for("schedule"))
        try:
            game["game_video_url"] = _sanitize_http_url(_extract_game_video_url_from_notes(game.get("notes")))
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
        has_score = (game.get("team1_score") is not None) or (game.get("team2_score") is not None) or bool(game.get("is_final"))
        can_view_summary = bool(has_score or (sdt is None) or started)
        if not can_view_summary:
            return ("Not found", 404)
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            if is_owner:
                cur.execute(
                    "SELECT * FROM players WHERE team_id=%s AND user_id=%s ORDER BY jersey_number ASC, name ASC",
                    (game["team1_id"], session["user_id"]),
                )
            else:
                cur.execute(
                    "SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC",
                    (game["team1_id"],),
                )
            team1_players = cur.fetchall()
            if is_owner:
                cur.execute(
                    "SELECT * FROM players WHERE team_id=%s AND user_id=%s ORDER BY jersey_number ASC, name ASC",
                    (game["team2_id"], session["user_id"]),
                )
            else:
                cur.execute(
                    "SELECT * FROM players WHERE team_id=%s ORDER BY jersey_number ASC, name ASC",
                    (game["team2_id"],),
                )
            team2_players = cur.fetchall()
            # Load existing player stats rows for this game
            cur.execute("SELECT * FROM player_stats WHERE game_id=%s", (game_id,))
            stats_rows = cur.fetchall()
            cur.execute("SELECT stats_json, updated_at FROM hky_game_stats WHERE game_id=%s", (game_id,))
            game_stats_row = cur.fetchone()
        team1_skaters, team1_goalies, team1_hc, team1_ac = split_roster(team1_players or [])
        team2_skaters, team2_goalies, team2_hc, team2_ac = split_roster(team2_players or [])
        team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
        team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
        stats_by_pid = {r["player_id"]: r for r in stats_rows}

        game_stats = None
        game_stats_updated_at = None
        try:
            if game_stats_row and game_stats_row.get("stats_json"):
                game_stats = json.loads(game_stats_row["stats_json"])
                game_stats_updated_at = game_stats_row.get("updated_at")
        except Exception:
            game_stats = None

        game_stats = filter_game_stats_for_display(game_stats)
        period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}

        events_headers: list[str] = []
        events_rows: list[dict[str, str]] = []
        events_meta: Optional[dict[str, Any]] = None
        try:
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT events_csv, source_label, updated_at FROM hky_game_events WHERE game_id=%s",
                    (game_id,),
                )
                erow = cur.fetchone()
            if erow and str(erow.get("events_csv") or "").strip():
                events_headers, events_rows = parse_events_csv(str(erow.get("events_csv") or ""))
                events_meta = {
                    "source_label": erow.get("source_label"),
                    "updated_at": erow.get("updated_at"),
                    "count": len(events_rows),
                }
        except Exception:
            events_headers, events_rows, events_meta = [], [], None

        try:
            events_headers, events_rows = normalize_game_events_csv(events_headers, events_rows)
        except Exception:
            pass
        events_rows = filter_events_rows_prefer_timetoscore_for_goal_assist(events_rows, tts_linked=tts_linked)
        try:
            events_headers, events_rows = normalize_events_video_time_for_display(events_headers, events_rows)
            events_headers, events_rows = filter_events_headers_drop_empty_on_ice_split(events_headers, events_rows)
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

        scoring_by_period_rows = compute_team_scoring_by_period_from_events(events_rows, tts_linked=tts_linked)

        imported_player_stats_csv_text: Optional[str] = None
        player_stats_import_meta: Optional[dict[str, Any]] = None
        try:
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT player_stats_csv, source_label, updated_at FROM hky_game_player_stats_csv WHERE game_id=%s",
                    (game_id,),
                )
                prow = cur.fetchone()
            if prow and str(prow.get("player_stats_csv") or "").strip():
                imported_player_stats_csv_text = str(prow.get("player_stats_csv") or "")
                player_stats_import_meta = {
                    "source_label": prow.get("source_label"),
                    "updated_at": prow.get("updated_at"),
                }
        except Exception:
            imported_player_stats_csv_text, player_stats_import_meta = None, None

        game_player_stats_columns, player_stats_cells_by_pid, player_stats_cell_conflicts_by_pid, player_stats_import_warning = (
            build_game_player_stats_table(
                players=list(team1_skaters) + list(team2_skaters),
                stats_by_pid=stats_by_pid,
                imported_csv_text=imported_player_stats_csv_text,
                prefer_db_stats_for_keys={"goals", "assists"} if tts_linked else None,
            )
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
                    rows.append({"player_id": pid, "name": str(p.get("name") or ""), "goals": g, "assists": a, "points": g + a})
                    by_pid[pid] = p
                sorted_rows = sort_players_table_default(rows)
                return [by_pid[int(r["player_id"])] for r in sorted_rows if int(r["player_id"]) in by_pid]

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
            with g.db.cursor() as cur:
                if is_owner:
                    cur.execute(
                        "UPDATE hky_games SET location=%s, starts_at=%s, team1_score=%s, team2_score=%s, is_final=%s WHERE id=%s AND user_id=%s",
                        (
                            loc or None,
                            parse_dt_or_none(starts_at),
                            int(t1_score) if (t1_score or "").strip() else None,
                            int(t2_score) if (t2_score or "").strip() else None,
                            1 if is_final else 0,
                            game_id,
                            session["user_id"],
                        ),
                    )
                else:
                    cur.execute(
                        "UPDATE hky_games SET location=%s, starts_at=%s, team1_score=%s, team2_score=%s, is_final=%s WHERE id=%s",
                        (
                            loc or None,
                            parse_dt_or_none(starts_at),
                            int(t1_score) if (t1_score or "").strip() else None,
                            int(t2_score) if (t2_score or "").strip() else None,
                            1 if is_final else 0,
                            game_id,
                        ),
                    )

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

            with g.db.cursor() as cur:
                for p in list(team1_skaters) + list(team2_skaters):
                    pid = int(p["id"])
                    vals = _collect("ps", pid)
                    # Determine team_id for this player
                    team_id = int(p["team_id"])
                    # Determine if an entry exists
                    cur.execute(
                        "SELECT id FROM player_stats WHERE game_id=%s AND player_id=%s",
                        (game_id, pid),
                    )
                    row = cur.fetchone()
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
                    if row:
                        set_clause = ", ".join([f"{c}=%s" for c in cols])
                        params = [vals.get(c) for c in cols] + [game_id, pid]
                        cur.execute(
                            f"UPDATE player_stats SET {set_clause} WHERE game_id=%s AND player_id=%s",
                            params,
                        )
                    else:
                        placeholders = ",".join(["%s"] * len(cols))
                        params = [int(game.get("user_id") or session["user_id"]), team_id, game_id, pid] + [
                            vals.get(c) for c in cols
                        ]
                        cur.execute(
                            f"INSERT INTO player_stats(user_id, team_id, game_id, player_id, {', '.join(cols)}) VALUES(%s,%s,%s,%s,{placeholders})",
                            params,
                        )
            g.db.commit()
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
            game_stats=game_stats,
            game_stats_updated_at=game_stats_updated_at,
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
        )

    @app.route("/game_types", methods=["GET", "POST"])
    def game_types():
        r = require_login()
        if r:
            return r
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            if name:
                try:
                    with g.db.cursor() as cur:
                        cur.execute(
                            "INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 0)
                        )
                    g.db.commit()
                    flash("Game type added", "success")
                except Exception:
                    flash("Failed to add game type (may already exist)", "error")
            return redirect(url_for("game_types"))
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM game_types ORDER BY name")
            rows = cur.fetchall()
        return render_template("game_types.html", game_types=rows)

    return app


def init_db():
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INT AUTO_INCREMENT PRIMARY KEY,
              email VARCHAR(255) UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              name VARCHAR(255),
              created_at DATETIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Bootstrap a default admin login for fresh DBs.
        # Note: this is intentionally simple for quick setup; change/remove in real deployments.
        try:
            cur.execute(
                "INSERT IGNORE INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
                (
                    "admin",
                    generate_password_hash("admin"),
                    "admin",
                    dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        except Exception:
            pass
        # Leagues and sharing tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS leagues (
              id INT AUTO_INCREMENT PRIMARY KEY,
              name VARCHAR(255) UNIQUE NOT NULL,
              owner_user_id INT NOT NULL,
              is_shared TINYINT(1) NOT NULL DEFAULT 0,
              is_public TINYINT(1) NOT NULL DEFAULT 0,
              source VARCHAR(64) NULL,
              external_key VARCHAR(255) NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(owner_user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Add leagues.source and leagues.external_key if missing (older installs)
        try:
            cur.execute("SHOW COLUMNS FROM leagues LIKE 'source'")
            has_source = cur.fetchone()
            if not has_source:
                cur.execute("ALTER TABLE leagues ADD COLUMN source VARCHAR(64) NULL")
        except Exception:
            try:
                cur.execute("ALTER TABLE leagues ADD COLUMN source VARCHAR(64) NULL")
            except Exception:
                pass
        try:
            cur.execute("SHOW COLUMNS FROM leagues LIKE 'external_key'")
            has_ext = cur.fetchone()
            if not has_ext:
                cur.execute("ALTER TABLE leagues ADD COLUMN external_key VARCHAR(255) NULL")
        except Exception:
            try:
                cur.execute("ALTER TABLE leagues ADD COLUMN external_key VARCHAR(255) NULL")
            except Exception:
                pass
        # Add leagues.is_public if missing (older installs)
        try:
            cur.execute("SHOW COLUMNS FROM leagues LIKE 'is_public'")
            has_pub = cur.fetchone()
            if not has_pub:
                cur.execute("ALTER TABLE leagues ADD COLUMN is_public TINYINT(1) NOT NULL DEFAULT 0")
        except Exception:
            try:
                cur.execute("ALTER TABLE leagues ADD COLUMN is_public TINYINT(1) NOT NULL DEFAULT 0")
            except Exception:
                pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_members (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              user_id INT NOT NULL,
              role VARCHAR(32) NOT NULL DEFAULT 'viewer',
              created_at DATETIME NOT NULL,
              UNIQUE KEY uniq_member (league_id, user_id),
              INDEX(league_id), INDEX(user_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              name VARCHAR(255) NOT NULL,
              dir_path TEXT NOT NULL,
              status VARCHAR(32) NOT NULL DEFAULT 'new',
              created_at DATETIME NOT NULL,
              INDEX(user_id),
              FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              game_id INT,
              dir_path TEXT NOT NULL,
              slurm_job_id VARCHAR(64),
              status VARCHAR(32) NOT NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              finished_at DATETIME NULL,
              user_email VARCHAR(255) NULL,
              INDEX(user_id), INDEX(game_id), INDEX(slurm_job_id), INDEX(status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resets (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              token VARCHAR(128) UNIQUE NOT NULL,
              expires_at DATETIME NOT NULL,
              used_at DATETIME NULL,
              created_at DATETIME NOT NULL,
              INDEX(user_id), INDEX(token), INDEX(expires_at),
              FOREIGN KEY(user_id) REFERENCES users(id)
                ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Teams (owned by user)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS teams (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              name VARCHAR(255) NOT NULL,
              logo_path TEXT NULL,
              is_external TINYINT(1) NOT NULL DEFAULT 0,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(is_external),
              UNIQUE KEY uniq_team_user_name (user_id, name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Add users.default_league_id if missing
        try:
            cur.execute("SHOW COLUMNS FROM users LIKE 'default_league_id'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("ALTER TABLE users ADD COLUMN default_league_id INT NULL")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_users_default_league ON users(default_league_id)"
                )
        except Exception:
            # Fallback for MySQL variants without IF NOT EXISTS
            try:
                cur.execute("ALTER TABLE users ADD COLUMN default_league_id INT NULL")
            except Exception:
                pass
        # Add users.video_clip_len_s if missing (per-user UI preference)
        try:
            cur.execute("SHOW COLUMNS FROM users LIKE 'video_clip_len_s'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("ALTER TABLE users ADD COLUMN video_clip_len_s INT NULL")
        except Exception:
            try:
                cur.execute("ALTER TABLE users ADD COLUMN video_clip_len_s INT NULL")
            except Exception:
                pass
        # Players (belong to exactly one team)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS players (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              team_id INT NOT NULL,
              name VARCHAR(255) NOT NULL,
              jersey_number VARCHAR(16) NULL,
              position VARCHAR(32) NULL,
              shoots VARCHAR(8) NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(team_id), INDEX(name),
              FOREIGN KEY(team_id) REFERENCES teams(id)
                ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Hockey game types
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS game_types (
              id INT AUTO_INCREMENT PRIMARY KEY,
              name VARCHAR(64) UNIQUE NOT NULL,
              is_default TINYINT(1) NOT NULL DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Hockey games
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hky_games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              team1_id INT NOT NULL,
              team2_id INT NOT NULL,
              game_type_id INT NULL,
              starts_at DATETIME NULL,
              location VARCHAR(255) NULL,
              notes TEXT NULL,
              team1_score INT NULL,
              team2_score INT NULL,
              is_final TINYINT(1) NOT NULL DEFAULT 0,
              stats_imported_at DATETIME NULL,
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(team1_id), INDEX(team2_id), INDEX(game_type_id), INDEX(starts_at),
              FOREIGN KEY(team1_id) REFERENCES teams(id) ON DELETE RESTRICT ON UPDATE CASCADE,
              FOREIGN KEY(team2_id) REFERENCES teams(id) ON DELETE RESTRICT ON UPDATE CASCADE,
              FOREIGN KEY(game_type_id) REFERENCES game_types(id) ON DELETE SET NULL ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Add hky_games.stats_imported_at if missing (older installs)
        try:
            cur.execute("SHOW COLUMNS FROM hky_games LIKE 'stats_imported_at'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("ALTER TABLE hky_games ADD COLUMN stats_imported_at DATETIME NULL")
        except Exception:
            try:
                cur.execute("ALTER TABLE hky_games ADD COLUMN stats_imported_at DATETIME NULL")
            except Exception:
                pass
        # Player stats per game
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS player_stats (
              id INT AUTO_INCREMENT PRIMARY KEY,
              user_id INT NOT NULL,
              team_id INT NOT NULL,
              game_id INT NOT NULL,
              player_id INT NOT NULL,
              goals INT NULL,
              assists INT NULL,
              shots INT NULL,
              pim INT NULL,
              plus_minus INT NULL,
              hits INT NULL,
              blocks INT NULL,
              toi_seconds INT NULL,
              faceoff_wins INT NULL,
              faceoff_attempts INT NULL,
              goalie_saves INT NULL,
              goalie_ga INT NULL,
              goalie_sa INT NULL,
              sog INT NULL,
              expected_goals INT NULL,
              giveaways INT NULL,
              turnovers_forced INT NULL,
              created_turnovers INT NULL,
              takeaways INT NULL,
              controlled_entry_for INT NULL,
              controlled_entry_against INT NULL,
              controlled_exit_for INT NULL,
              controlled_exit_against INT NULL,
              gt_goals INT NULL,
              gw_goals INT NULL,
              ot_goals INT NULL,
              ot_assists INT NULL,
              shifts INT NULL,
              gf_counted INT NULL,
              ga_counted INT NULL,
              video_toi_seconds INT NULL,
              sb_avg_shift_seconds INT NULL,
              sb_median_shift_seconds INT NULL,
              sb_longest_shift_seconds INT NULL,
              sb_shortest_shift_seconds INT NULL,
              UNIQUE KEY uniq_game_player (game_id, player_id),
              INDEX(user_id), INDEX(team_id), INDEX(game_id), INDEX(player_id),
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Extend player_stats for shift spreadsheet stats (older installs)
        for col_ddl in [
            "sog INT NULL",
            "expected_goals INT NULL",
            "giveaways INT NULL",
            "turnovers_forced INT NULL",
            "created_turnovers INT NULL",
            "takeaways INT NULL",
            "controlled_entry_for INT NULL",
            "controlled_entry_against INT NULL",
            "controlled_exit_for INT NULL",
            "controlled_exit_against INT NULL",
            "gt_goals INT NULL",
            "gw_goals INT NULL",
            "ot_goals INT NULL",
            "ot_assists INT NULL",
            "shifts INT NULL",
            "gf_counted INT NULL",
            "ga_counted INT NULL",
            "video_toi_seconds INT NULL",
            "sb_avg_shift_seconds INT NULL",
            "sb_median_shift_seconds INT NULL",
            "sb_longest_shift_seconds INT NULL",
            "sb_shortest_shift_seconds INT NULL",
        ]:
            name = col_ddl.split(" ", 1)[0]
            try:
                cur.execute("SHOW COLUMNS FROM player_stats LIKE %s", (name,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(f"ALTER TABLE player_stats ADD COLUMN {col_ddl}")
            except Exception:
                # best-effort for mysql variants
                try:
                    cur.execute(f"ALTER TABLE player_stats ADD COLUMN {col_ddl}")
                except Exception:
                    pass

        # Per-player per-period stats (from shift spreadsheet outputs)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS player_period_stats (
              id INT AUTO_INCREMENT PRIMARY KEY,
              game_id INT NOT NULL,
              player_id INT NOT NULL,
              period INT NOT NULL,
              toi_seconds INT NULL,
              shifts INT NULL,
              gf INT NULL,
              ga INT NULL,
              UNIQUE KEY uniq_period (game_id, player_id, period),
              INDEX(game_id), INDEX(player_id), INDEX(period),
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        # Per-game stats key/value from shift spreadsheet outputs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hky_game_stats (
              game_id INT PRIMARY KEY,
              stats_json LONGTEXT NULL,
              updated_at DATETIME NULL,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        # Raw per-game events CSV (e.g., all_events_summary.csv from scripts/parse_stats_inputs.py)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hky_game_events (
              game_id INT PRIMARY KEY,
              events_csv MEDIUMTEXT NULL,
              source_label VARCHAR(255) NULL,
              updated_at DATETIME NULL,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )

        # Raw per-game player stats CSV (e.g., stats/player_stats.csv from scripts/parse_stats_inputs.py)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hky_game_player_stats_csv (
              game_id INT PRIMARY KEY,
              player_stats_csv MEDIUMTEXT NULL,
              source_label VARCHAR(255) NULL,
              updated_at DATETIME NULL,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # League mappings (created after dependent tables)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_teams (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              team_id INT NOT NULL,
              division_name VARCHAR(255) NULL,
              division_id INT NULL,
              conference_id INT NULL,
              mhr_div_rating DOUBLE NULL,
              mhr_rating DOUBLE NULL,
              mhr_agd DOUBLE NULL,
              mhr_sched DOUBLE NULL,
              mhr_games INT NULL,
              mhr_updated_at DATETIME NULL,
              UNIQUE KEY uniq_league_team (league_id, team_id),
              INDEX(league_id), INDEX(team_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS league_games (
              id INT AUTO_INCREMENT PRIMARY KEY,
              league_id INT NOT NULL,
              game_id INT NOT NULL,
              division_name VARCHAR(255) NULL,
              division_id INT NULL,
              conference_id INT NULL,
              sort_order INT NULL,
              UNIQUE KEY uniq_league_game (league_id, game_id),
              INDEX(league_id), INDEX(game_id),
              FOREIGN KEY(league_id) REFERENCES leagues(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        # Extend league_teams/league_games with division metadata (older installs)
        for table in ("league_teams", "league_games"):
            for col_ddl in [
                "division_name VARCHAR(255) NULL",
                "division_id INT NULL",
                "conference_id INT NULL",
            ]:
                col = col_ddl.split(" ", 1)[0]
                try:
                    cur.execute(f"SHOW COLUMNS FROM {table} LIKE %s", (col,))
                    exists = cur.fetchone()
                    if not exists:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_ddl}")
                except Exception:
                    try:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_ddl}")
                    except Exception:
                        pass
        # Extend league_teams with rating fields (older installs)
        for col_ddl in [
            "mhr_div_rating DOUBLE NULL",
            "mhr_rating DOUBLE NULL",
            "mhr_agd DOUBLE NULL",
            "mhr_sched DOUBLE NULL",
            "mhr_games INT NULL",
            "mhr_updated_at DATETIME NULL",
        ]:
            col = col_ddl.split(" ", 1)[0]
            try:
                cur.execute("SHOW COLUMNS FROM league_teams LIKE %s", (col,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(f"ALTER TABLE league_teams ADD COLUMN {col_ddl}")
            except Exception:
                try:
                    cur.execute(f"ALTER TABLE league_teams ADD COLUMN {col_ddl}")
                except Exception:
                    pass
        # Add league_games.sort_order if missing (older installs)
        try:
            cur.execute("SHOW COLUMNS FROM league_games LIKE %s", ("sort_order",))
            exists = cur.fetchone()
            if not exists:
                cur.execute("ALTER TABLE league_games ADD COLUMN sort_order INT NULL")
        except Exception:
            try:
                cur.execute("ALTER TABLE league_games ADD COLUMN sort_order INT NULL")
            except Exception:
                pass
    db.commit()
    # Seed default game types if empty
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM game_types")
        count = (cur.fetchone() or [0])[0]
        if int(count) == 0:
            for name in ("Preseason", "Regular Season", "Tournament", "Exhibition"):
                cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 1))
    db.commit()


def get_user_by_email(email: str) -> Optional[dict]:
    db = get_db()
    with db.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        return cur.fetchone()


def create_user(email: str, password: str, name: str) -> int:
    pw = generate_password_hash(password)
    now = dt.datetime.now().isoformat()
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO users(email, password_hash, name, created_at) VALUES(%s,%s,%s,%s)",
            (email, pw, name, now),
        )
        db.commit()
        return int(cur.lastrowid)


def create_game(user_id: int, name: str, email: str):
    # Create dedicated dir: <watch_root>/<user_id>_<timestamp>_<rand>
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    token = secrets.token_hex(4)
    d = Path(WATCH_ROOT) / f"game_{user_id}_{ts}_{token}"
    d.mkdir(parents=True, exist_ok=True)
    # Create meta with user email
    try:
        (d / ".dirwatch_meta.json").write_text(
            f'{{"user_email":"{email}","created":"{dt.datetime.now().isoformat()}"}}\n'
        )
    except Exception:
        pass
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO games(user_id, name, dir_path, created_at) VALUES(%s,%s,%s,%s)",
            (user_id, name, str(d), dt.datetime.now().isoformat()),
        )
        db.commit()
        return int(cur.lastrowid), str(d)


def read_dirwatch_state():
    state_path = Path("/var/lib/dirwatcher/state.json")
    try:
        import json

        return json.loads(state_path.read_text())
    except Exception:
        return {"processed": {}, "active": {}}


def get_db():
    # Load DB configuration
    cfg_path = os.environ.get("HM_DB_CONFIG", str(CONFIG_PATH))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    dbcfg = cfg.get("db", {})
    # Ensure pymysql is available at call time
    global pymysql  # type: ignore
    if pymysql is None:  # pragma: no cover
        import importlib

        pymysql = importlib.import_module("pymysql")  # type: ignore
    conn = pymysql.connect(
        host=dbcfg.get("host", "127.0.0.1"),
        port=int(dbcfg.get("port", 3306)),
        user=dbcfg.get("user", "hmapp"),
        password=dbcfg.get("pass", ""),
        database=dbcfg.get("name", "hm_app_db"),
        autocommit=False,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )
    return conn


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
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (user_id, name, 1 if is_external else 0, dt.datetime.now().isoformat()),
        )
        db.commit()
        return int(cur.lastrowid)


def get_team(team_id: int, user_id: int) -> Optional[dict]:
    # Prefer request-scoped connection if available
    db = getattr(g, "db", None) or get_db()
    with db.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT * FROM teams WHERE id=%s AND user_id=%s", (team_id, user_id))
        return cur.fetchone()


def save_team_logo(file_storage, team_id: int) -> Path:
    # Save under instance/uploads/team_logos
    uploads = INSTANCE_DIR / "uploads" / "team_logos"
    uploads.mkdir(parents=True, exist_ok=True)
    # sanitize filename
    fname = Path(file_storage.filename).name
    # prefix with team id and timestamp
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    dest = uploads / f"team{team_id}_{ts}_{fname}"
    file_storage.save(dest)
    return dest


def ensure_external_team(user_id: int, name: str) -> int:
    def _norm_team_name(s: str) -> str:
        t = str(s or "").replace("\xa0", " ").strip()
        t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
        t = " ".join(t.split())
        return t

    name = _norm_team_name(name)
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id FROM teams WHERE user_id=%s AND name=%s", (user_id, name))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur.execute(
            "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
            (user_id, name, 1, dt.datetime.now().isoformat()),
        )
        db.commit()
        return int(cur.lastrowid)


def create_hky_game(
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[dt.datetime],
    location: Optional[str],
) -> int:
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hky_games(user_id, team1_id, team2_id, game_type_id, starts_at, location, created_at)
            VALUES(%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                user_id,
                team1_id,
                team2_id,
                game_type_id,
                starts_at,
                location,
                dt.datetime.now().isoformat(),
            ),
        )
        db.commit()
        return int(cur.lastrowid)


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
    if d1.lower() == "external" or d2.lower() == "external":
        return False
    return d1 != d2


def recompute_league_mhr_ratings(db_conn, league_id: int, *, max_goal_diff: int = 7, min_games: int = 5) -> dict[int, dict[str, Any]]:
    """
    Recompute and persist MyHockeyRankings-like ratings for teams in a league.
    Stores values on `league_teams` as:
      - mhr_rating (NULL if games < min_games)
      - mhr_agd, mhr_sched, mhr_games, mhr_updated_at
    """
    with db_conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT team_id FROM league_teams WHERE league_id=%s", (int(league_id),))
        league_team_ids = sorted({int(r["team_id"]) for r in (cur.fetchall() or []) if r.get("team_id") is not None})

        cur.execute(
            """
            SELECT lt.team_id, lt.division_name, t.name AS team_name
            FROM league_teams lt JOIN teams t ON lt.team_id=t.id
            WHERE lt.league_id=%s
            """,
            (int(league_id),),
        )
        team_age: dict[int, Optional[int]] = {}
        for r in cur.fetchall() or []:
            try:
                tid = int(r.get("team_id"))
            except Exception:
                continue
            dn = str(r.get("division_name") or "").strip()
            age = parse_age_from_division_name(dn)
            if age is None:
                age = parse_age_from_division_name(str(r.get("team_name") or "").strip())
            team_age[tid] = age

        cur.execute(
            """
            SELECT g.team1_id, g.team2_id, g.team1_score, g.team2_score,
                   lg.division_name AS game_division_name,
                   lt1.division_name AS team1_league_division_name,
                   lt2.division_name AS team2_league_division_name
            FROM league_games lg
              JOIN hky_games g ON lg.game_id=g.id
              LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
              LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
            WHERE lg.league_id=%s
              AND g.team1_score IS NOT NULL
              AND g.team2_score IS NOT NULL
            """,
            (int(league_id),),
        )
        rows = cur.fetchall() or []

    games: list[GameScore] = []
    for r in rows:
        try:
            games.append(
                GameScore(
                    team1_id=int(r["team1_id"]),
                    team2_id=int(r["team2_id"]),
                    team1_score=int(r["team1_score"]),
                    team2_score=int(r["team2_score"]),
                )
            )
        except Exception:
            continue

    # Ignore cross-age games when computing ratings (no cross-age coupling).
    games_same_age = filter_games_ignore_cross_age(games, team_age=team_age)

    computed = compute_mhr_like_ratings(
        games=games_same_age,
        max_goal_diff=int(max_goal_diff),
        min_games_for_rating=int(min_games),
    )
    # Normalize per disconnected component: top team in each independent group becomes 99.9.
    computed_norm = scale_ratings_to_0_99_9_by_component(computed, games=games_same_age, key="rating")

    now_s = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Persist for all league teams (set NULL when unknown/insufficient).
    with db_conn.cursor() as cur:
        for tid in league_team_ids:
            row_norm = computed_norm.get(int(tid)) or {}
            rating = row_norm.get("rating")
            # Use raw AGD/SCHED/GAMES from the base computation (before shifting).
            row_base = computed.get(int(tid)) or {}
            cur.execute(
                """
                UPDATE league_teams
                SET mhr_rating=%s, mhr_agd=%s, mhr_sched=%s, mhr_games=%s, mhr_updated_at=%s
                WHERE league_id=%s AND team_id=%s
                """,
                (
                    float(rating) if rating is not None else None,
                    float(row_base.get("agd")) if row_base.get("agd") is not None else None,
                    float(row_base.get("sched")) if row_base.get("sched") is not None else None,
                    int(row_base.get("games")) if row_base.get("games") is not None else 0,
                    now_s,
                    int(league_id),
                    int(tid),
                ),
            )
    db_conn.commit()
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
    kept_set = set(kept_headers)
    kept_rows = [{h: r.get(h, "") for h in kept_headers} for r in (rows or [])]
    return kept_headers, kept_rows


def normalize_game_events_csv(headers: list[str], rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
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
    new_rows = [{h: (r.get(h, "") if isinstance(r, dict) else "") for h in new_headers} for r in rows]
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
        side = _norm(
            row.get("Team Side")
            or row.get("TeamSide")
            or row.get("Side")
            or ""
        )
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
        rec = by_period.setdefault(per, {"team1_gf": 0, "team1_ga": 0, "team2_gf": 0, "team2_ga": 0})
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
    has_tts = any(_ev_type(r) in {"goal", "assist"} and _is_tts_row(r) for r in events_rows if isinstance(r, dict))
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
        tr = str(
            r.get("Team Side")
            or r.get("TeamSide")
            or r.get("Team Rel")
            or r.get("TeamRel")
            or r.get("Team")
            or ""
        ).strip().casefold()
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

    all_rows = _iter_typed(ex_rows, existing_source_label) + _iter_typed(in_rows, incoming_source_label)
    has_tts_protected = any(_ev_type(r) in protected_types and _is_tts_row(r, lbl) for r, lbl in all_rows)

    merged_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for r, lbl in all_rows:
        et = _ev_type(r)
        if not et:
            continue
        if has_tts_protected and et in protected_types and not _is_tts_row(r, lbl):
            continue
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        merged_rows.append(r)

    merged_headers = list(ex_headers or [])
    for h in in_headers or []:
        if h not in merged_headers:
            merged_headers.append(h)

    # Prefer a TimeToScore label if either input is TimeToScore.
    merged_source = (
        existing_source_label
        if str(existing_source_label or "").strip().lower().startswith("timetoscore")
        else (incoming_source_label if str(incoming_source_label or "").strip().lower().startswith("timetoscore") else (existing_source_label or incoming_source_label))
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
    copy Video Time/Seconds from matching long-sheet Goal events (same team + period + game time).

    - Only uses incoming rows with Source containing "long"
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

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _norm_side(row: dict[str, str]) -> Optional[str]:
        for k in ("Team Side", "TeamSide", "Side", "Team Rel", "TeamRel", "Team Raw", "TeamRaw", "Team"):
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
        return parse_duration_seconds(row.get("Game Time") or row.get("GameTime") or row.get("Time"))

    def _video_seconds(row: dict[str, str]) -> Optional[int]:
        vs = _parse_int(row.get("Video Seconds") or row.get("VideoSeconds"))
        if vs is not None:
            return vs
        return parse_duration_seconds(row.get("Video Time") or row.get("VideoTime"))

    def _video_time(row: dict[str, str]) -> str:
        return str(row.get("Video Time") or row.get("VideoTime") or "").strip()

    # Build lookup from incoming long Goal rows.
    long_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        if _ev_type(r) != "goal":
            continue
        if not _has_source(r, "long"):
            continue
        per = _period(r)
        side = _norm_side(r)
        gs = _game_seconds(r)
        if per is None or side is None or gs is None:
            continue
        vs = _video_seconds(r)
        vt = _video_time(r)
        if vs is None and not vt:
            continue
        long_by_key[(per, side, int(gs))] = r

    if not long_by_key:
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
        if _ev_type(rr) == "goal" and _has_source(rr, "timetoscore"):
            per = _period(rr)
            side = _norm_side(rr)
            gs = _game_seconds(rr)
            if per is not None and side is not None and gs is not None:
                match = long_by_key.get((per, side, int(gs)))
                if match is not None:
                    vs = _video_seconds(match)
                    vt = _video_time(match)
                    if vs is not None:
                        rr["Video Seconds"] = str(int(vs))
                    if vt:
                        rr["Video Time"] = vt
                    _add_source(rr, "long")
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
        for k in ("Team Side", "TeamSide", "Team Rel", "TeamRel", "Side", "Team", "Team Raw", "TeamRaw"):
            v = _norm_cf(r.get(k))
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
            if v in {"neutral"}:
                return None
        return None

    skip_types = {"assist", "penalty expired", "power play", "powerplay", "penalty kill", "penaltykill"}
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
        rows.append({"event_type": et, "home": int(rec.get("home") or 0), "away": int(rec.get("away") or 0)})
    rows.sort(key=lambda r: (_prio(str(r.get("event_type") or "")), str(r.get("event_type") or "").casefold()))
    return rows


def normalize_events_video_time_for_display(
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Ensure the events table includes a human-readable Video Time column when Video Seconds exists.
    This is a display-time normalization only; it does not affect stored CSV.
    """
    if not headers or not rows:
        return headers, rows
    has_vs = any(str(h or "").strip().lower() in {"video seconds", "videoseconds"} for h in headers)
    if not has_vs:
        return headers, rows
    has_vt = any(str(h or "").strip().lower() in {"video time", "videotime"} for h in headers)

    out_headers = list(headers)
    if not has_vt:
        # Prefer to place it next to Video Seconds if present, otherwise near Game Time.
        try:
            vs_idx = next(i for i, h in enumerate(out_headers) if str(h or "").strip().lower() in {"video seconds", "videoseconds"})
            out_headers.insert(vs_idx, "Video Time")
        except Exception:
            try:
                gt_idx = next(i for i, h in enumerate(out_headers) if str(h or "").strip().lower() in {"game time", "gametime", "time"})
                out_headers.insert(gt_idx + 1, "Video Time")
            except Exception:
                out_headers.append("Video Time")

    out_rows: list[dict[str, str]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        vt = str(rr.get("Video Time") or rr.get("VideoTime") or "").strip()
        if not vt:
            vs = parse_duration_seconds(rr.get("Video Seconds") or rr.get("VideoSeconds"))
            if vs is not None:
                rr["Video Time"] = format_seconds_to_mmss_or_hhmmss(vs)
        out_rows.append(rr)
    return out_headers, out_rows


def sort_events_rows_default(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Default ordering for the Game Events table: period asc, then decreasing game time within period.
    """

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _period(r: dict[str, str]) -> int:
        p = _parse_int(r.get("Period"))
        return int(p) if p is not None and p > 0 else 999

    def _game_seconds(r: dict[str, str]) -> int:
        gs = _parse_int(r.get("Game Seconds") or r.get("GameSeconds"))
        if gs is not None:
            return int(gs)
        gt = parse_duration_seconds(r.get("Game Time") or r.get("GameTime") or r.get("Time"))
        return int(gt) if gt is not None else -1

    def _ev_type(r: dict[str, str]) -> str:
        return str(r.get("Event Type") or r.get("Event") or "").strip().casefold()

    return sorted(
        [r for r in (rows or []) if isinstance(r, dict)],
        key=lambda r: (_period(r), -_game_seconds(r), _ev_type(r)),
    )


def get_user_video_clip_len_s(db_conn, user_id: Optional[int]) -> int:
    """
    Per-user clip length preference for timeline video clips.
    Defaults to 10 seconds when unset/unknown.
    """
    if not user_id:
        return 10
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT video_clip_len_s FROM users WHERE id=%s", (int(user_id),))
            row = cur.fetchone()
        v = None
        if isinstance(row, dict):
            v = row.get("video_clip_len_s")
        elif isinstance(row, (list, tuple)) and row:
            v = row[0]
        try:
            iv = int(v) if v is not None else None
        except Exception:
            iv = None
        if iv in {10, 20, 30}:
            return int(iv)
    except Exception:
        pass
    return 10


def reset_league_data(db_conn, league_id: int, *, owner_user_id: Optional[int] = None) -> dict[str, int]:
    """
    Wipe imported hockey data for a league (games/teams/players/stats) while keeping:
      - users
      - league record and memberships

    This is used by `tools/webapp/reset_league_data.py` and the hidden REST endpoint.
    """
    stats: dict[str, int] = {
        "player_stats": 0,
        "league_games": 0,
        "hky_games": 0,
        "league_teams": 0,
        "players": 0,
        "teams": 0,
    }
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM league_games WHERE league_id=%s", (int(league_id),))
        stats["league_games"] = int((cur.fetchone() or [0])[0])
        cur.execute("SELECT COUNT(*) FROM league_teams WHERE league_id=%s", (int(league_id),))
        stats["league_teams"] = int((cur.fetchone() or [0])[0])

        # Exclusive games for this league (safe to delete without impacting other leagues).
        cur.execute(
            """
            SELECT game_id
            FROM league_games
            WHERE league_id=%s
              AND game_id NOT IN (SELECT game_id FROM league_games WHERE league_id<>%s)
            """,
            (int(league_id), int(league_id)),
        )
        exclusive_game_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})
        if exclusive_game_ids:
            ph = ",".join(["%s"] * len(exclusive_game_ids))
            cur.execute(f"SELECT COUNT(*) FROM player_stats WHERE game_id IN ({ph})", tuple(exclusive_game_ids))
            stats["player_stats"] = int((cur.fetchone() or [0])[0])
            cur.execute(f"SELECT COUNT(*) FROM hky_games WHERE id IN ({ph})", tuple(exclusive_game_ids))
            stats["hky_games"] = int((cur.fetchone() or [0])[0])

        # Exclusive teams for this league (safe candidates).
        cur.execute(
            """
            SELECT team_id
            FROM league_teams
            WHERE league_id=%s
              AND team_id NOT IN (SELECT team_id FROM league_teams WHERE league_id<>%s)
            """,
            (int(league_id), int(league_id)),
        )
        exclusive_team_ids = sorted({int(r[0]) for r in (cur.fetchall() or [])})

        # Remove league mappings (this is the "reset" behavior).
        cur.execute("DELETE FROM league_games WHERE league_id=%s", (int(league_id),))
        cur.execute("DELETE FROM league_teams WHERE league_id=%s", (int(league_id),))

        # Delete exclusive games (cascades to player_stats/hky_game_* tables).
        if exclusive_game_ids:
            ph = ",".join(["%s"] * len(exclusive_game_ids))
            cur.execute(f"DELETE FROM hky_games WHERE id IN ({ph})", tuple(exclusive_game_ids))
        if exclusive_team_ids:
            ph = ",".join(["%s"] * len(exclusive_team_ids))
            # Only delete external teams (and optionally only those owned by the league owner).
            cur.execute(
                f"SELECT id, user_id, is_external FROM teams WHERE id IN ({ph})",
                tuple(exclusive_team_ids),
            )
            team_rows = cur.fetchall() or []
            eligible = []
            for tid, uid, is_ext in team_rows:
                if int(is_ext or 0) != 1:
                    continue
                if owner_user_id is not None and int(uid) != int(owner_user_id):
                    continue
                eligible.append(int(tid))
            if eligible:
                ph2 = ",".join(["%s"] * len(eligible))
                cur.execute(
                    f"""
                    SELECT DISTINCT team1_id AS tid FROM hky_games WHERE team1_id IN ({ph2})
                    UNION
                    SELECT DISTINCT team2_id AS tid FROM hky_games WHERE team2_id IN ({ph2})
                    """,
                    tuple(eligible) * 2,
                )
                still_used = {int(r[0]) for r in (cur.fetchall() or [])}
                safe_team_ids = sorted([tid for tid in eligible if tid not in still_used])
                if safe_team_ids:
                    ph3 = ",".join(["%s"] * len(safe_team_ids))
                    cur.execute(f"SELECT COUNT(*) FROM players WHERE team_id IN ({ph3})", tuple(safe_team_ids))
                    stats["players"] = int((cur.fetchone() or [0])[0])
                    cur.execute(f"SELECT COUNT(*) FROM teams WHERE id IN ({ph3})", tuple(safe_team_ids))
                    stats["teams"] = int((cur.fetchone() or [0])[0])
                    cur.execute(f"DELETE FROM teams WHERE id IN ({ph3})", tuple(safe_team_ids))

    db_conn.commit()
    return stats


def compute_team_stats(db_conn, team_id: int, user_id: int) -> dict:
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT team1_id, team2_id, team1_score, team2_score, is_final
            FROM hky_games WHERE user_id=%s AND (team1_id=%s OR team2_id=%s) AND team1_score IS NOT NULL AND team2_score IS NOT NULL
            """,
            (user_id, team_id, team_id),
        )
        rows = cur.fetchall()
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
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT g.team1_id, g.team2_id, g.team1_score, g.team2_score, g.is_final,
                   lg.division_name AS league_division_name,
                   gt.name AS game_type_name,
                   lt1.division_name AS team1_league_division_name,
                   lt2.division_name AS team2_league_division_name
            FROM league_games lg
              JOIN hky_games g ON lg.game_id=g.id
              LEFT JOIN game_types gt ON g.game_type_id=gt.id
              LEFT JOIN league_teams lt1 ON lt1.league_id=lg.league_id AND lt1.team_id=g.team1_id
              LEFT JOIN league_teams lt2 ON lt2.league_id=lg.league_id AND lt2.team_id=g.team2_id
            WHERE lg.league_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
              AND g.team1_score IS NOT NULL AND g.team2_score IS NOT NULL
            """,
            (league_id, team_id, team_id),
        )
        rows = cur.fetchall()
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
        if d1.lower() == "external" or d2.lower() == "external":
            return False
        ld = str(r.get("league_division_name") or "").strip()
        if ld.lower() == "external":
            return False
        return d1 != d2

    def _is_regular_game(r: dict) -> bool:
        # Only regular-season games should contribute to standings points/rankings.
        gt = str(r.get("game_type_name") or "").strip()
        if not gt or not gt.lower().startswith("regular"):
            return False
        # Any game involving an External team, or mapped to the External division, does not count for standings.
        for key in ("league_division_name", "team1_league_division_name", "team2_league_division_name"):
            dn = str(r.get(key) or "").strip()
            if dn.lower() == "external":
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
    "faceoff_wins",
    "faceoff_attempts",
    "goalie_saves",
    "goalie_ga",
    "goalie_sa",
)

PLAYER_STATS_DISPLAY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("gp", "GP"),
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
) -> tuple[list[dict[str, Any]], dict[int, dict[str, str]], dict[int, dict[str, bool]], Optional[str]]:
    """
    Display-first per-game table: preserve imported CSV columns (minus identity fields),
    with optional DB merge/conflict highlighting for known numeric stats.
    """
    try:
        headers, rows = parse_events_csv(imported_csv_text)
    except Exception as e:  # noqa: BLE001
        return list(GAME_PLAYER_STATS_COLUMNS), {}, {}, f"Unable to parse imported player_stats_csv: {e}"

    if not headers:
        return [], {}, {}, "Imported player_stats_csv has no headers"

    # Never show shift/TOI/per-game/per-shift columns in the web UI, even if older data is stored.
    headers, rows = filter_single_game_player_stats_csv(headers, rows)
    if not headers:
        return [], {}, {}, "Imported player_stats_csv has no displayable columns"

    team_ids = sorted({int(p.get("team_id") or 0) for p in (players or []) if p.get("team_id") is not None})
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    for p in (players or []):
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
        columns.append({"id": col_id, "label": str(h), "header": str(h), "db_key": _PLAYER_STATS_HEADER_TO_DB_KEY.get(key)})

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
    goals_col_id = next((str(c["id"]) for c in visible if str(c.get("db_key") or "") == "goals"), None)
    assists_col_id = next((str(c["id"]) for c in visible if str(c.get("db_key") or "") == "assists"), None)
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
                cell_conf_by_pid.get(pid, {}).get(goals_col_id) or cell_conf_by_pid.get(pid, {}).get(assists_col_id)
            )

        pts_vals = [cell_text_by_pid.get(pid, {}).get(points_id, "") for pid in all_pids]
        if not all(_is_blank_stat(v) or str(v).strip() == "" for v in pts_vals):
            insert_at = next((i + 1 for i, c in enumerate(visible) if str(c.get("id")) == assists_col_id), None)
            if insert_at is None:
                insert_at = 0
            visible.insert(int(insert_at), {"id": points_id, "label": "P", "header": "P", "db_key": None})

    # Rebuild cell dicts to only include visible columns.
    vis_ids = {str(c["id"]) for c in visible}
    cell_text_by_pid = {pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_text_by_pid.items()}
    cell_conf_by_pid = {pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_conf_by_pid.items()}
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
    goalie_ga = _int0(sums.get("goalie_ga"))

    out: dict[str, Any] = dict(sums)
    out["gp"] = gp
    out["points"] = points
    out["ppg"] = _rate_or_none(points, gp)

    # Per-game rates.
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
    if re.match(r"^\s*HC\b", name_up) or re.search(r"\bHEAD\s+COACH\b", name_up) or re.search(r"\(HC\)", name_up):
        return "HC"
    if re.match(r"^\s*AC\b", name_up) or re.search(r"\bASSISTANT\s+COACH\b", name_up) or re.search(r"\(AC\)", name_up):
        return "AC"

    # Conservative goalie hint when position is missing.
    if re.search(r"\bGOALIE\b", name_up) or re.search(r"\bGOALTENDER\b", name_up) or re.search(r"\(G\)", name_up):
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
    for p in (players or []):
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
    for p in (players or []):
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
    for k, label in columns:
        vals = [r.get(k) for r in (rows or [])]
        if all(_is_zero_or_blank_stat(v) for v in vals):
            continue
        out.append((k, label))
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

    team_ids = sorted({int(p.get("team_id") or 0) for p in (players or []) if p.get("team_id") is not None})
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    for p in (players or []):
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
) -> tuple[list[dict[str, Any]], dict[int, dict[str, str]], dict[int, dict[str, bool]], Optional[str]]:
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

    visible_columns: list[dict[str, Any]] = []
    for col in GAME_PLAYER_STATS_COLUMNS:
        keys = [str(k) for k in (col.get("keys") or ())]
        if keys and set(keys).issubset(OT_ONLY_PLAYER_STATS_KEYS):
            if all(all(_is_zero_or_blank_stat(merged_vals[pid].get(k)) for k in keys) for pid in all_pids):
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
    cell_text_by_pid = {pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_text_by_pid.items()}
    cell_conflict_by_pid = {pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_conflict_by_pid.items()}

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
    for p in (players or []):
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
    for r in (player_stats_rows or []):
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
    if not gt and str(game_row.get("division_name") or "").strip().lower() == "external":
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
    for r in (player_stats_rows or []):
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
    for r in (player_stats_rows or []):
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
    for c in _player_stats_columns_with_coverage(columns=columns, coverage_counts=coverage_counts, total_games=total_games):
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
    for k, label in (columns or []):
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


def _canon_source_label_for_ui(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    sl = s.casefold()
    if sl in {"t2s", "tts"}:
        return "TimeToScore"
    if sl == "timetoscore":
        return "TimeToScore"
    if sl == "long":
        return "Long"
    if sl == "primary":
        return "Primary"
    if sl.startswith("parse_stats_inputs:") or sl == "parse_stats_inputs":
        return "Primary"
    if sl.startswith("parse_shift_spreadsheet:") or sl == "parse_shift_spreadsheet":
        return "Primary"
    if sl == "shift_package":
        return "Shift Package"
    if sl == "goals":
        return "Goals"
    # Only show recognized, high-level source types (avoid per-game labels like "sharks-12-2-r1").
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

    def _add(src: Any) -> None:
        s = _canon_source_label_for_ui(src)
        if not s:
            return
        k = s.casefold()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    # Prefer scanning events CSV sources (multi-valued Source column), then include player_stats CSV labels.
    try:
        with db_conn.cursor(pymysql.cursors.DictCursor) as cur:
            for i in range(0, len(gids), 200):
                chunk = gids[i : i + 200]
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"SELECT game_id, events_csv, source_label FROM hky_game_events WHERE game_id IN ({ph})", chunk)
                rows = cur.fetchall() or []
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    csv_text = str(r.get("events_csv") or "").strip()
                    if csv_text:
                        try:
                            _h, ev_rows = parse_events_csv(csv_text)
                            for s in summarize_event_sources(ev_rows, fallback_source_label=str(r.get("source_label") or "")):
                                _add(s)
                            continue
                        except Exception:
                            pass
                    _add(r.get("source_label"))
    except Exception:
        pass
    try:
        with db_conn.cursor(pymysql.cursors.DictCursor) as cur:
            for i in range(0, len(gids), 200):
                chunk = gids[i : i + 200]
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(
                    f"SELECT game_id, source_label FROM hky_game_player_stats_csv WHERE game_id IN ({ph})", chunk
                )
                rows = cur.fetchall() or []
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    _add(r.get("source_label"))
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
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
	            SELECT player_id,
	                   COUNT(*) AS gp,
	                   COALESCE(SUM(goals),0) AS goals,
	                   COALESCE(SUM(assists),0) AS assists,
	                   COALESCE(SUM(pim),0) AS pim,
	                   COALESCE(SUM(shots),0) AS shots,
	                   COALESCE(SUM(sog),0) AS sog,
	                   COALESCE(SUM(expected_goals),0) AS expected_goals,
	                   COALESCE(SUM(giveaways),0) AS giveaways,
	                   COALESCE(SUM(turnovers_forced),0) AS turnovers_forced,
	                   COALESCE(SUM(created_turnovers),0) AS created_turnovers,
	                   COALESCE(SUM(takeaways),0) AS takeaways,
	                   COALESCE(SUM(controlled_entry_for),0) AS controlled_entry_for,
	                   COALESCE(SUM(controlled_entry_against),0) AS controlled_entry_against,
	                   COALESCE(SUM(controlled_exit_for),0) AS controlled_exit_for,
	                   COALESCE(SUM(controlled_exit_against),0) AS controlled_exit_against,
	                   COALESCE(SUM(plus_minus),0) AS plus_minus,
	                   COALESCE(SUM(gf_counted),0) AS gf_counted,
	                   COALESCE(SUM(ga_counted),0) AS ga_counted,
	                   COALESCE(SUM(gt_goals),0) AS gt_goals,
	                   COALESCE(SUM(gw_goals),0) AS gw_goals,
	                   COALESCE(SUM(ot_goals),0) AS ot_goals,
	                   COALESCE(SUM(ot_assists),0) AS ot_assists,
	                   COALESCE(SUM(hits),0) AS hits,
	                   COALESCE(SUM(blocks),0) AS blocks,
	                   COALESCE(SUM(faceoff_wins),0) AS faceoff_wins,
	                   COALESCE(SUM(faceoff_attempts),0) AS faceoff_attempts,
	                   COALESCE(SUM(goalie_saves),0) AS goalie_saves,
	                   COALESCE(SUM(goalie_ga),0) AS goalie_ga,
	                   COALESCE(SUM(goalie_sa),0) AS goalie_sa
	            FROM player_stats WHERE team_id=%s AND user_id=%s
	            GROUP BY player_id
            """,
            (team_id, user_id),
        )
        rows = cur.fetchall()
    out: dict[int, dict[str, Any]] = {}
    for r in (rows or []):
        pid = int(r.get("player_id") if isinstance(r, dict) else r["player_id"])
        out[pid] = compute_player_display_stats(dict(r))
    return out


def aggregate_players_totals_league(db_conn, team_id: int, league_id: int) -> dict:
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
	            SELECT ps.player_id,
	                   COUNT(*) AS gp,
	                   COALESCE(SUM(ps.goals),0) AS goals,
	                   COALESCE(SUM(ps.assists),0) AS assists,
	                   COALESCE(SUM(ps.pim),0) AS pim,
	                   COALESCE(SUM(ps.shots),0) AS shots,
	                   COALESCE(SUM(ps.sog),0) AS sog,
	                   COALESCE(SUM(ps.expected_goals),0) AS expected_goals,
	                   COALESCE(SUM(ps.giveaways),0) AS giveaways,
	                   COALESCE(SUM(ps.turnovers_forced),0) AS turnovers_forced,
	                   COALESCE(SUM(ps.created_turnovers),0) AS created_turnovers,
	                   COALESCE(SUM(ps.takeaways),0) AS takeaways,
	                   COALESCE(SUM(ps.controlled_entry_for),0) AS controlled_entry_for,
	                   COALESCE(SUM(ps.controlled_entry_against),0) AS controlled_entry_against,
	                   COALESCE(SUM(ps.controlled_exit_for),0) AS controlled_exit_for,
	                   COALESCE(SUM(ps.controlled_exit_against),0) AS controlled_exit_against,
	                   COALESCE(SUM(ps.plus_minus),0) AS plus_minus,
	                   COALESCE(SUM(ps.gf_counted),0) AS gf_counted,
	                   COALESCE(SUM(ps.ga_counted),0) AS ga_counted,
	                   COALESCE(SUM(ps.gt_goals),0) AS gt_goals,
	                   COALESCE(SUM(ps.gw_goals),0) AS gw_goals,
	                   COALESCE(SUM(ps.ot_goals),0) AS ot_goals,
	                   COALESCE(SUM(ps.ot_assists),0) AS ot_assists,
	                   COALESCE(SUM(ps.hits),0) AS hits,
	                   COALESCE(SUM(ps.blocks),0) AS blocks,
	                   COALESCE(SUM(ps.faceoff_wins),0) AS faceoff_wins,
	                   COALESCE(SUM(ps.faceoff_attempts),0) AS faceoff_attempts,
	                   COALESCE(SUM(ps.goalie_saves),0) AS goalie_saves,
	                   COALESCE(SUM(ps.goalie_ga),0) AS goalie_ga,
	                   COALESCE(SUM(ps.goalie_sa),0) AS goalie_sa
	            FROM league_games lg
	              JOIN hky_games g ON lg.game_id=g.id
	              JOIN player_stats ps ON lg.game_id=ps.game_id
	              LEFT JOIN league_teams lt_self ON lt_self.league_id=lg.league_id AND lt_self.team_id=ps.team_id
	              LEFT JOIN league_teams lt_opp ON lt_opp.league_id=lg.league_id AND lt_opp.team_id=(
	                   CASE WHEN g.team1_id=ps.team_id THEN g.team2_id ELSE g.team1_id END
	              )
	            WHERE lg.league_id=%s AND ps.team_id=%s
	              AND (
	                LOWER(COALESCE(lg.division_name,''))='external'
	                OR lt_opp.division_name IS NULL
	                OR LOWER(COALESCE(lt_opp.division_name,''))='external'
	                OR lt_self.division_name IS NULL
	                OR lt_self.division_name=lt_opp.division_name
	              )
	            GROUP BY ps.player_id
            """,
            (league_id, team_id),
        )
        rows = cur.fetchall()
    out: dict[int, dict[str, Any]] = {}
    for r in (rows or []):
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
    if d1.lower() == "external" or d2.lower() == "external":
        return False
    ld = str(game_row.get("division_name") or game_row.get("league_division_name") or "").strip()
    if ld.lower() == "external":
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


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8008, debug=True)

#!/usr/bin/env python3
import csv
import datetime as dt
import io
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any, Optional

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

# Lazy import for pymysql to allow importing module without DB installed (e.g., tests)
try:
    import pymysql  # type: ignore
except Exception:  # pragma: no cover
    pymysql = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
CONFIG_PATH = BASE_DIR / "config.json"

WATCH_ROOT = os.environ.get("HM_WATCH_ROOT", "/data/incoming")


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

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    @app.template_filter("fmt_toi")
    def _fmt_toi(seconds: Any) -> str:
        return format_seconds_to_mmss_or_hhmmss(seconds)

    def _to_dt(value: Any) -> Optional[dt.datetime]:
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

    @app.template_filter("fmt_date")
    def _fmt_date(value: Any) -> str:
        d = _to_dt(value)
        return d.strftime("%Y-%m-%d") if d else ""

    @app.template_filter("fmt_time")
    def _fmt_time(value: Any) -> str:
        d = _to_dt(value)
        return d.strftime("%H:%M") if d else ""

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
        return dict(user_leagues=leagues, selected_league_id=selected)

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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            if league_id:
                cur.execute(
                    """
                    SELECT t.*, lt.division_name, lt.division_id, lt.conference_id
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
        )

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
            if supplied != required:
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

    def _ensure_external_team_for_import(owner_user_id: int, name: str, *, commit: bool = True) -> int:
        nm = (name or "").strip()
        if not nm:
            nm = "UNKNOWN"
        with g.db.cursor() as cur:
            cur.execute("SELECT id FROM teams WHERE user_id=%s AND name=%s", (owner_user_id, nm))
            row = cur.fetchone()
            if row:
                return int(row[0])
            cur.execute(
                "INSERT INTO teams(user_id, name, is_external, created_at) VALUES(%s,%s,%s,%s)",
                (owner_user_id, nm, 1, dt.datetime.now().isoformat()),
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

    def _upsert_game_for_import(
        *,
        owner_user_id: int,
        team1_id: int,
        team2_id: int,
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
                token = f'\"timetoscore_game_id\":{int(notes_json_fields["timetoscore_game_id"])}'
                cur.execute(
                    "SELECT id, notes, team1_score, team2_score FROM hky_games WHERE user_id=%s AND notes LIKE %s",
                    (owner_user_id, f"%{token}%"),
                )
                row = cur.fetchone()
                if row:
                    gid = int(row["id"])

            if gid is None:
                notes = json.dumps(notes_json_fields, sort_keys=True)
                cur.execute(
                    """
                    INSERT INTO hky_games(user_id, team1_id, team2_id, starts_at, location, team1_score, team2_score, is_final, notes, stats_imported_at, created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        owner_user_id,
                        team1_id,
                        team2_id,
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
                    SET location=COALESCE(%s, location),
                        team1_score=%s,
                        team2_score=%s,
                        is_final=CASE WHEN %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                        notes=%s,
                        stats_imported_at=%s,
                        updated_at=%s
                    WHERE id=%s
                    """,
                    (
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
                    SET location=COALESCE(%s, location),
                        team1_score=COALESCE(team1_score, %s),
                        team2_score=COALESCE(team2_score, %s),
                        is_final=CASE WHEN team1_score IS NULL AND team2_score IS NULL AND %s IS NOT NULL AND %s IS NOT NULL THEN 1 ELSE is_final END,
                        notes=%s,
                        stats_imported_at=%s,
                        updated_at=%s
                    WHERE id=%s
                    """,
                    (
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
                      division_name=COALESCE(VALUES(division_name), division_name),
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
        commit: bool = True,
    ) -> None:
        dn = (division_name or "").strip() or None
        with g.db.cursor() as cur:
            if dn is None and division_id is None and conference_id is None:
                cur.execute(
                    "INSERT IGNORE INTO league_games(league_id, game_id) VALUES(%s,%s)",
                    (league_id, game_id),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO league_games(league_id, game_id, division_name, division_id, conference_id)
                    VALUES(%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      division_name=COALESCE(VALUES(division_name), division_name),
                      division_id=COALESCE(VALUES(division_id), division_id),
                      conference_id=COALESCE(VALUES(conference_id), conference_id)
                    """,
                    (league_id, game_id, dn, division_id, conference_id),
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

        try:
            resp = requests.get(url, timeout=(10, 30))
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
        _ensure_team_logo_from_url_for_import(
            team_id=int(team1_id),
            logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
            replace=replace,
        )
        _ensure_team_logo_from_url_for_import(
            team_id=int(team2_id),
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
                    _ensure_player_for_import(owner_user_id, tid, nm, jersey, pos)

        def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
            with g.db.cursor() as cur:
                cur.execute(
                    "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                    (owner_user_id, team_id, name),
                )
                r = cur.fetchone()
                return int(r[0]) if r else None

        stats_rows = game.get("player_stats") or []
        if isinstance(stats_rows, list):
            with g.db.cursor() as cur:
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

                    if replace:
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
            for idx, game in enumerate(games):
                if not isinstance(game, dict):
                    raise ValueError(f"games[{idx}] must be an object")
                home_name = str(game.get("home_name") or "").strip()
                away_name = str(game.get("away_name") or "").strip()
                if not home_name or not away_name:
                    raise ValueError(f"games[{idx}]: home_name and away_name are required")

                game_replace = bool(game.get("replace", replace))

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
                _ensure_team_logo_from_url_for_import(
                    team_id=int(team1_id),
                    logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
                    replace=game_replace,
                    commit=False,
                )
                _ensure_team_logo_from_url_for_import(
                    team_id=int(team2_id),
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
                    starts_at=starts_at_s,
                    location=location,
                    team1_score=t1s,
                    team2_score=t2s,
                    replace=game_replace,
                    notes_json_fields=notes_fields,
                    commit=False,
                )
                _map_game_to_league_for_import(
                    league_id,
                    gid,
                    division_name=division_name or home_division_name or away_division_name,
                    division_id=division_id or home_division_id or away_division_id,
                    conference_id=conference_id or home_conference_id or away_conference_id,
                    commit=False,
                )

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
                            _ensure_player_for_import(
                                owner_user_id, tid, nm, jersey, pos, commit=False
                            )

                def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
                    with g.db.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM players WHERE user_id=%s AND team_id=%s AND name=%s",
                            (owner_user_id, team_id, name),
                        )
                        r = cur.fetchone()
                        return int(r[0]) if r else None

                stats_rows = game.get("player_stats") or []
                if isinstance(stats_rows, list):
                    with g.db.cursor() as cur:
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
                                pid = _ensure_player_for_import(
                                    owner_user_id, team_ref, pname, None, None, commit=False
                                )

                            if game_replace:
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
                SELECT t.*, lt.division_name, lt.division_id, lt.conference_id
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
        )

    @app.get("/public/leagues/<int:league_id>/teams/<int:team_id>")
    def public_league_team_detail(league_id: int, team_id: int):
        league = _is_public_league(league_id)
        if not league:
            return ("Not found", 404)
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
        player_totals = aggregate_players_totals_league(g.db, team_id, int(league_id))
        tstats = compute_team_stats_league(g.db, team_id, int(league_id))
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name,
                       lg.division_name AS division_name
                FROM league_games lg
                  JOIN hky_games g ON lg.game_id=g.id
                  JOIN teams t1 ON g.team1_id=t1.id
                  JOIN teams t2 ON g.team2_id=t2.id
                  LEFT JOIN game_types gt ON g.game_type_id=gt.id
                WHERE lg.league_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
                ORDER BY COALESCE(g.starts_at, g.created_at) DESC
                """,
                (int(league_id), team_id, team_id),
            )
            schedule_games = cur.fetchall() or []
        return render_template(
            "team_detail.html",
            team=team,
            players=players,
            player_totals=player_totals,
            tstats=tstats,
            schedule_games=schedule_games,
            editable=False,
            public_league_id=int(league_id),
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
                       lg.division_name AS division_name
                FROM league_games lg
                  JOIN hky_games g ON lg.game_id=g.id
                  JOIN teams t1 ON g.team1_id=t1.id
                  JOIN teams t2 ON g.team2_id=t2.id
                  LEFT JOIN game_types gt ON g.game_type_id=gt.id
                WHERE {' AND '.join(where)}
                ORDER BY COALESCE(g.starts_at, g.created_at) DESC
                """,
                tuple(params),
            )
            games = cur.fetchall() or []
        now_dt = dt.datetime.now()
        for g2 in games or []:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
            g2["can_view_summary"] = bool(started or has_score)
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
                SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, t1.is_external AS team1_ext, t2.is_external AS team2_ext
                FROM league_games lg JOIN hky_games g ON lg.game_id=g.id
                  JOIN teams t1 ON g.team1_id=t1.id JOIN teams t2 ON g.team2_id=t2.id
                WHERE g.id=%s AND lg.league_id=%s
                """,
                (game_id, league_id),
            )
            game = cur.fetchone()
        if not game:
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
            cur.execute(
                "SELECT player_id, period, toi_seconds, shifts, gf, ga FROM player_period_stats WHERE game_id=%s",
                (game_id,),
            )
            period_rows = cur.fetchall() or []
        stats_by_pid = {r["player_id"]: r for r in stats_rows}
        game_stats = None
        game_stats_updated_at = None
        try:
            if game_stats_row and game_stats_row.get("stats_json"):
                game_stats = json.loads(game_stats_row["stats_json"])
                game_stats_updated_at = game_stats_row.get("updated_at")
        except Exception:
            game_stats = None
        period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}
        for r in period_rows:
            pid = int(r["player_id"])
            period = int(r["period"])
            period_stats_by_pid.setdefault(pid, {})[period] = {
                "toi_seconds": r.get("toi_seconds"),
                "shifts": r.get("shifts"),
                "gf": r.get("gf"),
                "ga": r.get("ga"),
            }
        return render_template(
            "hky_game_detail.html",
            game=game,
            team1_players=team1_players,
            team2_players=team2_players,
            stats_by_pid=stats_by_pid,
            period_stats_by_pid=period_stats_by_pid,
            game_stats=game_stats,
            game_stats_updated_at=game_stats_updated_at,
            editable=False,
            public_league_id=int(league_id),
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
        league_id = session.get("league_id")
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
        if league_id:
            player_totals = aggregate_players_totals_league(g.db, team_id, int(league_id))
            tstats = compute_team_stats_league(g.db, team_id, int(league_id))
            with g.db.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    """
                    SELECT g.*, t1.name AS team1_name, t2.name AS team2_name, gt.name AS game_type_name,
                           lg.division_name AS division_name
                    FROM league_games lg
                      JOIN hky_games g ON lg.game_id=g.id
                      JOIN teams t1 ON g.team1_id=t1.id
                      JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN game_types gt ON g.game_type_id=gt.id
                    WHERE lg.league_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
                    ORDER BY COALESCE(g.starts_at, g.created_at) DESC
                    """,
                    (int(league_id), team_id, team_id),
                )
                schedule_games = cur.fetchall() or []
        else:
            player_totals = aggregate_players_totals(g.db, team_id, team_owner_id)
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
                    ORDER BY COALESCE(g.starts_at, g.created_at) DESC
                    """,
                    (team_owner_id, team_id, team_id),
                )
                schedule_games = cur.fetchall() or []
        return render_template(
            "team_detail.html",
            team=team,
            players=players,
            player_totals=player_totals,
            tstats=tstats,
            schedule_games=schedule_games,
            editable=editable,
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
                           lg.division_name AS division_name
                    FROM league_games lg
                      JOIN hky_games g ON lg.game_id=g.id
                      JOIN teams t1 ON g.team1_id=t1.id
                      JOIN teams t2 ON g.team2_id=t2.id
                      LEFT JOIN game_types gt ON g.game_type_id=gt.id
                    WHERE {' AND '.join(where)}
                    ORDER BY COALESCE(g.starts_at, g.created_at) DESC
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
                    ORDER BY COALESCE(g.starts_at, g.created_at) DESC
                    """,
                    (session["user_id"],),
                )
            games = cur.fetchall()
        now_dt = dt.datetime.now()
        for g2 in games or []:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = _to_dt(sdt) is not None and _to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (g2.get("team1_score") is not None) or (g2.get("team2_score") is not None) or bool(g2.get("is_final"))
            g2["can_view_summary"] = bool(started or has_score)
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

        # Authorization: only allow edits if owner or league editor/admin
        editable = True
        if league_id:
            with g.db.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(CASE WHEN role IN ('admin','owner','editor') THEN 1 ELSE 0 END),0) FROM league_members WHERE league_id=%s AND user_id=%s",
                    (league_id, session["user_id"]),
                )
                can_edit = int((cur.fetchone() or [0])[0]) == 1
            if not can_edit:
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
                cols = [
                    "goals",
                    "assists",
                    "shots",
                    "plus_minus",
                    "toi_seconds",
                    "sog",
                    "expected_goals",
                    "giveaways",
                    "takeaways",
                    "controlled_entry_for",
                    "controlled_entry_against",
                    "controlled_exit_for",
                    "controlled_exit_against",
                    "gt_goals",
                    "gw_goals",
                    "ot_goals",
                    "ot_assists",
                    "shifts",
                    "gf_counted",
                    "ga_counted",
                    "video_toi_seconds",
                    "sb_avg_shift_seconds",
                    "sb_median_shift_seconds",
                    "sb_longest_shift_seconds",
                    "sb_shortest_shift_seconds",
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

                # Upsert per-period stats (optional)
                for per, per_stats in (row.get("period_stats") or {}).items():
                    cur.execute(
                        """
                        INSERT INTO player_period_stats(game_id, player_id, period, toi_seconds, shifts, gf, ga)
                        VALUES(%s,%s,%s,%s,%s,%s,%s)
                        ON DUPLICATE KEY UPDATE
                          toi_seconds=COALESCE(VALUES(toi_seconds), toi_seconds),
                          shifts=COALESCE(VALUES(shifts), shifts),
                          gf=COALESCE(VALUES(gf), gf),
                          ga=COALESCE(VALUES(ga), ga)
                        """,
                        (
                            game_id,
                            pid,
                            int(per),
                            per_stats.get("toi_seconds"),
                            per_stats.get("shifts"),
                            per_stats.get("gf"),
                            per_stats.get("ga"),
                        ),
                    )

                imported += 1

            if game_stats is not None:
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
        return redirect(url_for("hky_game_detail", game_id=game_id))

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
        is_owner = int(game.get("user_id") or 0) == int(session["user_id"])

        # Authorization: allow edits if owner or league editor/admin.
        editable = is_owner
        if not editable and league_id:
            with g.db.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(CASE WHEN role IN ('admin','owner','editor') THEN 1 ELSE 0 END),0) FROM league_members WHERE league_id=%s AND user_id=%s",
                    (league_id, session["user_id"]),
                )
                editable = int((cur.fetchone() or [0])[0]) == 1

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
        stats_by_pid = {r["player_id"]: r for r in stats_rows}

        game_stats = None
        game_stats_updated_at = None
        try:
            if game_stats_row and game_stats_row.get("stats_json"):
                game_stats = json.loads(game_stats_row["stats_json"])
                game_stats_updated_at = game_stats_row.get("updated_at")
        except Exception:
            game_stats = None

        period_stats_by_pid: dict[int, dict[int, dict[str, Any]]] = {}
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT player_id, period, toi_seconds, shifts, gf, ga FROM player_period_stats WHERE game_id=%s",
                (game_id,),
            )
            for r in cur.fetchall():
                pid = int(r["player_id"])
                period = int(r["period"])
                period_stats_by_pid.setdefault(pid, {})[period] = {
                    "toi_seconds": r.get("toi_seconds"),
                    "shifts": r.get("shifts"),
                    "gf": r.get("gf"),
                    "ga": r.get("ga"),
                }

        if request.method == "POST" and not editable:
            flash("You do not have permission to edit this game in the selected league.", "error")

        if request.method == "POST" and editable:
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
                    "toi_seconds": _ival("toi"),
                    "faceoff_wins": _ival("fow"),
                    "faceoff_attempts": _ival("foa"),
                    "goalie_saves": _ival("saves"),
                    "goalie_ga": _ival("ga"),
                    "goalie_sa": _ival("sa"),
                }

            with g.db.cursor() as cur:
                for p in list(team1_players) + list(team2_players):
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
                        "toi_seconds",
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
            return redirect(url_for("hky_game_detail", game_id=game_id))

        return render_template(
            "hky_game_detail.html",
            game=game,
            team1_players=team1_players,
            team2_players=team2_players,
            stats_by_pid=stats_by_pid,
            period_stats_by_pid=period_stats_by_pid,
            game_stats=game_stats,
            game_stats_updated_at=game_stats_updated_at,
            editable=editable,
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
    points = wins * 2 + ties * 1
    return {"wins": wins, "losses": losses, "ties": ties, "gf": gf, "ga": ga, "points": points}


def compute_team_stats_league(db_conn, team_id: int, league_id: int) -> dict:
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT g.team1_id, g.team2_id, g.team1_score, g.team2_score, g.is_final
            FROM league_games lg JOIN hky_games g ON lg.game_id=g.id
            WHERE lg.league_id=%s AND (g.team1_id=%s OR g.team2_id=%s)
              AND g.team1_score IS NOT NULL AND g.team2_score IS NOT NULL
            """,
            (league_id, team_id, team_id),
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
    points = wins * 2 + ties * 1
    return {"wins": wins, "losses": losses, "ties": ties, "gf": gf, "ga": ga, "points": points}


def sort_key_team_standings(team_row: dict, stats: dict) -> tuple:
    """Standard hockey standings sort (points, wins, goal diff, goals for, goals against, name)."""
    pts = int(stats.get("points", 0))
    wins = int(stats.get("wins", 0))
    gf = int(stats.get("gf", 0))
    ga = int(stats.get("ga", 0))
    gd = gf - ga
    name = str(team_row.get("name") or "")
    return (-pts, -wins, -gd, -gf, ga, name.lower())


def aggregate_players_totals(db_conn, team_id: int, user_id: int) -> dict:
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT player_id,
                   COALESCE(SUM(goals),0) AS goals,
                   COALESCE(SUM(assists),0) AS assists,
                   COALESCE(SUM(pim),0) AS pim,
                   COALESCE(SUM(shots),0) AS shots,
                   COALESCE(SUM(sog),0) AS sog,
                   COALESCE(SUM(expected_goals),0) AS expected_goals,
                   COALESCE(SUM(giveaways),0) AS giveaways,
                   COALESCE(SUM(takeaways),0) AS takeaways,
                   COALESCE(SUM(controlled_entry_for),0) AS controlled_entry_for,
                   COALESCE(SUM(controlled_entry_against),0) AS controlled_entry_against,
                   COALESCE(SUM(controlled_exit_for),0) AS controlled_exit_for,
                   COALESCE(SUM(controlled_exit_against),0) AS controlled_exit_against,
                   COALESCE(SUM(plus_minus),0) AS plus_minus,
                   COALESCE(SUM(toi_seconds),0) AS toi_seconds,
                   COALESCE(SUM(video_toi_seconds),0) AS video_toi_seconds,
                   COALESCE(SUM(shifts),0) AS shifts,
                   COALESCE(SUM(gf_counted),0) AS gf_counted,
                   COALESCE(SUM(ga_counted),0) AS ga_counted
            FROM player_stats WHERE team_id=%s AND user_id=%s
            GROUP BY player_id
            """,
            (team_id, user_id),
        )
        rows = cur.fetchall()
    out = {}
    for r in rows:
        out[int(r["player_id"])] = {
            "goals": int(r["goals"] or 0),
            "assists": int(r["assists"] or 0),
            "points": int(r["goals"] or 0) + int(r["assists"] or 0),
            "shots": int(r["shots"] or 0),
            "sog": int(r.get("sog") or 0),
            "expected_goals": int(r.get("expected_goals") or 0),
            "giveaways": int(r.get("giveaways") or 0),
            "takeaways": int(r.get("takeaways") or 0),
            "controlled_entry_for": int(r.get("controlled_entry_for") or 0),
            "controlled_entry_against": int(r.get("controlled_entry_against") or 0),
            "controlled_exit_for": int(r.get("controlled_exit_for") or 0),
            "controlled_exit_against": int(r.get("controlled_exit_against") or 0),
            "plus_minus": int(r.get("plus_minus") or 0),
            "toi_seconds": int(r.get("toi_seconds") or 0),
            "video_toi_seconds": int(r.get("video_toi_seconds") or 0),
            "shifts": int(r.get("shifts") or 0),
            "gf_counted": int(r.get("gf_counted") or 0),
            "ga_counted": int(r.get("ga_counted") or 0),
            "pim": int(r["pim"] or 0),
        }
    return out


def aggregate_players_totals_league(db_conn, team_id: int, league_id: int) -> dict:
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, "cursors", None)) else None  # type: ignore
    with db_conn.cursor(curtype) if curtype else db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT ps.player_id,
                   COALESCE(SUM(ps.goals),0) AS goals,
                   COALESCE(SUM(ps.assists),0) AS assists,
                   COALESCE(SUM(ps.pim),0) AS pim,
                   COALESCE(SUM(ps.shots),0) AS shots,
                   COALESCE(SUM(ps.sog),0) AS sog,
                   COALESCE(SUM(ps.expected_goals),0) AS expected_goals,
                   COALESCE(SUM(ps.giveaways),0) AS giveaways,
                   COALESCE(SUM(ps.takeaways),0) AS takeaways,
                   COALESCE(SUM(ps.controlled_entry_for),0) AS controlled_entry_for,
                   COALESCE(SUM(ps.controlled_entry_against),0) AS controlled_entry_against,
                   COALESCE(SUM(ps.controlled_exit_for),0) AS controlled_exit_for,
                   COALESCE(SUM(ps.controlled_exit_against),0) AS controlled_exit_against,
                   COALESCE(SUM(ps.plus_minus),0) AS plus_minus,
                   COALESCE(SUM(ps.toi_seconds),0) AS toi_seconds,
                   COALESCE(SUM(ps.video_toi_seconds),0) AS video_toi_seconds,
                   COALESCE(SUM(ps.shifts),0) AS shifts,
                   COALESCE(SUM(ps.gf_counted),0) AS gf_counted,
                   COALESCE(SUM(ps.ga_counted),0) AS ga_counted
            FROM league_games lg JOIN player_stats ps ON lg.game_id=ps.game_id
            WHERE lg.league_id=%s AND ps.team_id=%s
            GROUP BY ps.player_id
            """,
            (league_id, team_id),
        )
        rows = cur.fetchall()
    out = {}
    for r in rows:
        out[int(r["player_id"])] = {
            "goals": int(r["goals"] or 0),
            "assists": int(r["assists"] or 0),
            "points": int(r["goals"] or 0) + int(r["assists"] or 0),
            "shots": int(r["shots"] or 0),
            "sog": int(r.get("sog") or 0),
            "expected_goals": int(r.get("expected_goals") or 0),
            "giveaways": int(r.get("giveaways") or 0),
            "takeaways": int(r.get("takeaways") or 0),
            "controlled_entry_for": int(r.get("controlled_entry_for") or 0),
            "controlled_entry_against": int(r.get("controlled_entry_against") or 0),
            "controlled_exit_for": int(r.get("controlled_exit_for") or 0),
            "controlled_exit_against": int(r.get("controlled_exit_against") or 0),
            "plus_minus": int(r.get("plus_minus") or 0),
            "toi_seconds": int(r.get("toi_seconds") or 0),
            "video_toi_seconds": int(r.get("video_toi_seconds") or 0),
            "shifts": int(r.get("shifts") or 0),
            "gf_counted": int(r.get("gf_counted") or 0),
            "ga_counted": int(r.get("ga_counted") or 0),
            "pim": int(r["pim"] or 0),
        }
    return out


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
    Parse stats/player_stats.csv written by scripts/parse_shift_spreadsheet.py.

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
        player_label = (row.get("Player") or "").strip()
        jersey_norm = None
        name_part = player_label
        m = re.match(r"^\s*(\d+)\s+(.*)$", player_label)
        if m:
            jersey_norm = normalize_jersey_number(m.group(1))
            name_part = m.group(2).strip()
        name_norm = normalize_player_name(name_part)

        giveaways_unforced = _int_or_none(row.get("Giveaways"))
        turnovers_forced = _int_or_none(row.get("Turnovers (forced)"))
        giveaways_total = None
        if giveaways_unforced is not None or turnovers_forced is not None:
            giveaways_total = int(giveaways_unforced or 0) + int(turnovers_forced or 0)

        stats: dict[str, Any] = {
            "goals": _int_or_none(row.get("Goals")),
            "assists": _int_or_none(row.get("Assists")),
            "shots": _int_or_none(row.get("Shots")),
            "sog": _int_or_none(row.get("SOG")),
            "expected_goals": _int_or_none(row.get("xG")),
            # DB only stores one "giveaways" integer; treat it as total turnovers
            # (forced turnovers + unforced giveaways) from the spreadsheet outputs.
            "giveaways": giveaways_total,
            "takeaways": _int_or_none(row.get("Takeaways")),
            "controlled_entry_for": _int_or_none(row.get("Controlled Entry For (On-Ice)")),
            "controlled_entry_against": _int_or_none(row.get("Controlled Entry Against (On-Ice)")),
            "controlled_exit_for": _int_or_none(row.get("Controlled Exit For (On-Ice)")),
            "controlled_exit_against": _int_or_none(row.get("Controlled Exit Against (On-Ice)")),
            "gt_goals": _int_or_none(row.get("GT Goals")),
            "gw_goals": _int_or_none(row.get("GW Goals")),
            "ot_goals": _int_or_none(row.get("OT Goals")),
            "ot_assists": _int_or_none(row.get("OT Assists")),
            "plus_minus": _int_or_none(row.get("Plus Minus")),
            "gf_counted": _int_or_none(row.get("GF Counted")),
            "ga_counted": _int_or_none(row.get("GA Counted")),
            "shifts": _int_or_none(row.get("Shifts")),
            "toi_seconds": parse_duration_seconds(row.get("TOI Total")),
            "video_toi_seconds": parse_duration_seconds(row.get("TOI Total (Video)")),
            "sb_avg_shift_seconds": parse_duration_seconds(row.get("Average Shift")),
            "sb_median_shift_seconds": parse_duration_seconds(row.get("Median Shift")),
            "sb_longest_shift_seconds": parse_duration_seconds(row.get("Longest Shift")),
            "sb_shortest_shift_seconds": parse_duration_seconds(row.get("Shortest Shift")),
        }

        # Period stats: Period {n} TOI/Shifts/GF/GA
        period_stats: dict[int, dict[str, Any]] = {}
        for k, v in row.items():
            if not k:
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
    Parse stats/game_stats.csv written by scripts/parse_shift_spreadsheet.py.
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

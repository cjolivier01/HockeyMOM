#!/usr/bin/env python3
import os
import json
import secrets
# Lazy import for pymysql to allow importing module without DB installed (e.g., tests)
try:
    import pymysql  # type: ignore
except Exception:  # pragma: no cover
    pymysql = None  # type: ignore
import secrets
import datetime as dt
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.security import generate_password_hash, check_password_hash


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
CONFIG_PATH = BASE_DIR / "config.json"

WATCH_ROOT = os.environ.get("HM_WATCH_ROOT", "/data/incoming")
APP_SECRET = os.environ.get("HM_WEBAPP_SECRET") or secrets.token_hex(16)


def create_app() -> Flask:
    app = Flask(__name__, instance_path=str(INSTANCE_DIR))
    app.config.update(
        SECRET_KEY=APP_SECRET,
        MAX_CONTENT_LENGTH=1024 * 1024 * 500,  # 500MB
        UPLOAD_FOLDER=WATCH_ROOT,
    )

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    @app.before_request
    def open_db():
        g.db = get_db()

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
        expires = row["expires_at"] if isinstance(row["expires_at"], dt.datetime) else dt.datetime.fromisoformat(str(row["expires_at"]))
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
                cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (newhash, row["user_id"]))
                cur.execute("UPDATE resets SET used_at=%s WHERE id=%s", (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row["id"]))
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
            cur.execute("SELECT * FROM games WHERE user_id=%s ORDER BY created_at DESC", (session["user_id"],))
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
                if fname.startswith('.'):
                    return False
                if fname.startswith('_'):
                    return False
                if fname.startswith('slurm-'):
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
            latest_status = dw_state.get("processed", {}).get(game["dir_path"], {}).get("status") or game.get("status")
        # Lock interactions once a job has been requested (any job row exists) or after completion
        is_locked = False
        if row:
            is_locked = True
        final_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
        if latest_status and str(latest_status).upper() in final_states:
            is_locked = True
        return render_template("game_detail.html", game=game, files=files, status=latest_status, is_locked=is_locked)

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
            cur.execute("SELECT id, slurm_job_id, status FROM jobs WHERE game_id=%s ORDER BY id DESC LIMIT 1", (gid,))
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
                    import subprocess as _sp, time as _time
                    dir_leaf = Path(game["dir_path"]).name
                    job_name = f"dirwatch-{dir_leaf}"
                    job_ids = []
                    jid = latest.get("slurm_job_id")
                    if jid:
                        job_ids.append(str(jid))
                    else:
                        # Fallback: discover job ids by job name prefix
                        try:
                            out = _sp.check_output(["squeue", "-h", "-o", "%i %j"]) .decode()
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
                                if parts[0] in job_ids or (len(parts) == 2 and parts[1] == job_name):
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
                cur.execute("DELETE FROM games WHERE id=%s AND user_id=%s", (gid, session["user_id"]))
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
            cur.execute("SELECT id,status FROM jobs WHERE game_id=%s ORDER BY id DESC LIMIT 1", (gid,))
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
        game = g.db.execute("SELECT * FROM games WHERE id=? AND user_id=?", (gid, session["user_id"]))
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                "SELECT id, user_id, logo_path FROM teams WHERE id=%s AND user_id=%s",
                (team_id, session["user_id"]),
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
        where = "user_id=%s" + ("" if include_external else " AND is_external=0")
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(f"SELECT * FROM teams WHERE {where} ORDER BY name ASC", (session["user_id"],))
            rows = cur.fetchall()
        # compute stats per team (wins/losses/ties/gf/ga/points)
        stats = {}
        for t in rows:
            stats[t["id"]] = compute_team_stats(g.db, t["id"], session["user_id"]) 
        return render_template("teams.html", teams=rows, stats=stats, include_external=include_external)

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
                        cur.execute("UPDATE teams SET logo_path=%s WHERE id=%s AND user_id=%s", (str(p), tid, session["user_id"]))
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
        team = get_team(team_id, session["user_id"])
        if not team:
            flash("Not found", "error")
            return redirect(url_for("teams"))
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM players WHERE team_id=%s AND user_id=%s ORDER BY jersey_number ASC, name ASC", (team_id, session["user_id"]))
            players = cur.fetchall()
        # Aggregate player career stats
        player_totals = aggregate_players_totals(g.db, team_id, session["user_id"])  # pid -> dict
        tstats = compute_team_stats(g.db, team_id, session["user_id"])  # team totals from games
        return render_template("team_detail.html", team=team, players=players, player_totals=player_totals, tstats=tstats)

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
                    cur.execute("UPDATE teams SET name=%s WHERE id=%s AND user_id=%s", (name, team_id, session["user_id"]))
                g.db.commit()
            f = request.files.get("logo")
            if f and f.filename:
                p = save_team_logo(f, team_id)
                with g.db.cursor() as cur:
                    cur.execute("UPDATE teams SET logo_path=%s WHERE id=%s AND user_id=%s", (str(p), team_id, session["user_id"]))
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
                    (session["user_id"], team_id, name, jersey or None, position or None, shoots or None, dt.datetime.now().isoformat()),
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
            cur.execute("SELECT * FROM players WHERE id=%s AND team_id=%s AND user_id=%s", (player_id, team_id, session["user_id"]))
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
                    (name or pl["name"], jersey or None, position or None, shoots or None, player_id, team_id, session["user_id"]),
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
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
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
        return render_template("schedule.html", games=games)

    @app.route("/schedule/new", methods=["GET", "POST"])
    def schedule_new():
        r = require_login()
        if r:
            return r
        # Load user's own teams (not external)
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT id, name FROM teams WHERE user_id=%s AND is_external=0 ORDER BY name", (session["user_id"],))
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
            flash("Game created", "success")
            return redirect(url_for("hky_game_detail", game_id=gid))
        return render_template("schedule_new.html", my_teams=my_teams, game_types=gt)

    @app.route("/hky/games/<int:game_id>", methods=["GET", "POST"]) 
    def hky_game_detail(game_id: int):
        r = require_login()
        if r:
            return r
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
        if not game:
            flash("Not found", "error")
            return redirect(url_for("schedule"))
        # Load players from both teams
        with g.db.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SELECT * FROM players WHERE team_id=%s AND user_id=%s ORDER BY jersey_number ASC, name ASC", (game["team1_id"], session["user_id"]))
            team1_players = cur.fetchall()
            cur.execute("SELECT * FROM players WHERE team_id=%s AND user_id=%s ORDER BY jersey_number ASC, name ASC", (game["team2_id"], session["user_id"]))
            team2_players = cur.fetchall()
            # Load existing player stats rows for this game
            cur.execute("SELECT * FROM player_stats WHERE game_id=%s", (game_id,))
            stats_rows = cur.fetchall()
        stats_by_pid = {r["player_id"]: r for r in stats_rows}

        if request.method == "POST":
            # Update game meta and scores
            loc = request.form.get("location", "").strip()
            starts_at = request.form.get("starts_at", "").strip()
            t1_score = request.form.get("team1_score")
            t2_score = request.form.get("team2_score")
            is_final = bool(request.form.get("is_final"))
            with g.db.cursor() as cur:
                cur.execute(
                    "UPDATE hky_games SET location=%s, starts_at=%s, team1_score=%s, team2_score=%s, is_final=%s WHERE id=%s AND user_id=%s",
                    (
                        loc or None,
                        parse_dt_or_none(starts_at),
                        int(t1_score) if (t1_score or '').strip() else None,
                        int(t2_score) if (t2_score or '').strip() else None,
                        1 if is_final else 0,
                        game_id,
                        session["user_id"],
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
                    cur.execute("SELECT id FROM player_stats WHERE game_id=%s AND player_id=%s", (game_id, pid))
                    row = cur.fetchone()
                    cols = ["goals","assists","shots","pim","plus_minus","hits","blocks","toi_seconds","faceoff_wins","faceoff_attempts","goalie_saves","goalie_ga","goalie_sa"]
                    if row:
                        set_clause = ", ".join([f"{c}=%s" for c in cols])
                        params = [vals.get(c) for c in cols] + [game_id, pid]
                        cur.execute(f"UPDATE player_stats SET {set_clause} WHERE game_id=%s AND player_id=%s", params)
                    else:
                        placeholders = ",".join(["%s"] * len(cols))
                        params = [session["user_id"], team_id, game_id, pid] + [vals.get(c) for c in cols]
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
                        cur.execute("INSERT INTO game_types(name, is_default) VALUES(%s,%s)", (name, 0))
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
              created_at DATETIME NOT NULL,
              updated_at DATETIME NULL,
              INDEX(user_id), INDEX(team1_id), INDEX(team2_id), INDEX(game_type_id), INDEX(starts_at),
              FOREIGN KEY(team1_id) REFERENCES teams(id) ON DELETE RESTRICT ON UPDATE CASCADE,
              FOREIGN KEY(team2_id) REFERENCES teams(id) ON DELETE RESTRICT ON UPDATE CASCADE,
              FOREIGN KEY(game_type_id) REFERENCES game_types(id) ON DELETE SET NULL ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
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
              UNIQUE KEY uniq_game_player (game_id, player_id),
              INDEX(user_id), INDEX(team_id), INDEX(game_id), INDEX(player_id),
              FOREIGN KEY(game_id) REFERENCES hky_games(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(player_id) REFERENCES players(id) ON DELETE CASCADE ON UPDATE CASCADE,
              FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
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
        (d / ".dirwatch_meta.json").write_text(f'{{"user_email":"{email}","created":"{dt.datetime.now().isoformat()}"}}\n')
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
    import shutil as _sh, subprocess as _sp
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
    db = getattr(g, 'db', None) or get_db()
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


def create_hky_game(user_id: int, team1_id: int, team2_id: int, game_type_id: Optional[int], starts_at: Optional[dt.datetime], location: Optional[str]) -> int:
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hky_games(user_id, team1_id, team2_id, game_type_id, starts_at, location, created_at)
            VALUES(%s,%s,%s,%s,%s,%s,%s)
            """,
            (user_id, team1_id, team2_id, game_type_id, starts_at, location, dt.datetime.now().isoformat()),
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
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, 'cursors', None)) else None  # type: ignore
    with (db_conn.cursor(curtype) if curtype else db_conn.cursor()) as cur:
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
        my_score = int(r["team1_score"]) if t1 else int(r["team2_score"]) if r["team2_score"] is not None else 0
        op_score = int(r["team2_score"]) if t1 else int(r["team1_score"]) if r["team1_score"] is not None else 0
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


def aggregate_players_totals(db_conn, team_id: int, user_id: int) -> dict:
    curtype = pymysql.cursors.DictCursor if (pymysql and getattr(pymysql, 'cursors', None)) else None  # type: ignore
    with (db_conn.cursor(curtype) if curtype else db_conn.cursor()) as cur:
        cur.execute(
            """
            SELECT player_id,
                   COALESCE(SUM(goals),0) AS goals,
                   COALESCE(SUM(assists),0) AS assists,
                   COALESCE(SUM(pim),0) AS pim,
                   COALESCE(SUM(shots),0) AS shots
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
            "pim": int(r["pim"] or 0),
        }
    return out


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8008, debug=True)

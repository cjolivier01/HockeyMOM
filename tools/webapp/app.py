#!/usr/bin/env python3
import os
import json
import secrets
import pymysql
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
            files = os.listdir(game["dir_path"]) if os.path.isdir(game["dir_path"]) else []
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


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8008, debug=True)

#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

SERVICE_NAME = "hm-webapp.service"
NGINX_SITE = "/etc/nginx/sites-available/hm-webapp"


def sudo_write_text(path: str | Path, content: str):
    """Write text to a root-protected file via sudo tee."""
    cmd = ["sudo", "tee", str(path)]
    subprocess.run(cmd, input=content.encode(), check=True)


def _sudo_capture_text(cmd: list[str]) -> str:
    proc = subprocess.run(
        ["sudo", *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout


def _ensure_port_available_for_nginx(listen_port: int, *, disable_apache2: bool) -> None:
    listeners = _sudo_capture_text(["ss", "-ltnHp", f"sport = :{listen_port}"]).strip()
    if not listeners:
        return

    if '("nginx",' in listeners:
        return

    if '("apache2",' in listeners and disable_apache2:
        print(
            f"apache2 is listening on port {listen_port}; stopping/disabling apache2 so nginx can bind..."
        )
        subprocess.run(["sudo", "systemctl", "disable", "--now", "apache2"], check=False)
        listeners = _sudo_capture_text(["ss", "-ltnHp", f"sport = :{listen_port}"]).strip()
        if not listeners:
            return
        if '("nginx",' in listeners:
            return

    msg = [
        f"ERROR: Port {listen_port} is already in use, so nginx cannot bind to it.",
        "",
        "Current listeners:",
        listeners,
        "",
        "Fix options:",
        f"- Stop/disable the service using port {listen_port} (common culprit: apache2).",
        "- Or re-run this installer with `--nginx-port <other-port>`.",
        "",
    ]
    if '("apache2",' in listeners and not disable_apache2:
        msg.append("Tip: re-run with `--disable-apache2` to stop+disable apache2 automatically.")
        msg.append("")
    raise SystemExit("\n".join(msg))


def main():
    ap = argparse.ArgumentParser(description="Install HockeyMOM WebApp (Django + Nginx)")
    ap.add_argument("--install-root", default="/opt/hm-webapp")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER") or os.environ.get("USER"))
    ap.add_argument("--watch-root", default="/data/incoming")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument(
        "--nginx-port",
        type=int,
        default=80,
        help="Port nginx listens on (default: 80).",
    )
    ap.add_argument(
        "--bind-address",
        default="127.0.0.1",
        help="Bind address for gunicorn (default: 127.0.0.1). Use 0.0.0.0 to expose the app port.",
    )
    ap.add_argument(
        "--disable-apache2",
        action="store_true",
        help="If apache2 is running and using the nginx listen port, stop+disable apache2 so nginx can start.",
    )
    ap.add_argument("--server-name", default="_")
    ap.add_argument("--client-max-body-size", default="500M")
    ap.add_argument("--db-name", default="hm_app_db")
    ap.add_argument("--db-user", default="hmapp")
    ap.add_argument("--db-pass", default="hmapp_pass")
    ap.add_argument("--db-host", default="127.0.0.1")
    ap.add_argument("--db-port", type=int, default=3306)
    ap.add_argument("--python-bin", default="", help="Python interpreter to run the app")
    ap.add_argument(
        "--import-token",
        default="",
        help="If set, require this bearer token for /api/import/* endpoints (send via Authorization: Bearer ...).",
    )
    ap.add_argument(
        "--no-default-user",
        action="store_true",
        help="Skip creating a default webapp user from git config (user.email/user.name).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    install_root = Path(args.install_root)
    app_dir = install_root / "app"
    templates_dir = app_dir / "templates"
    static_dir = app_dir / "static"

    print("Installing OS packages (nginx + mariadb + python venv tools)...")
    subprocess.check_call(["sudo", "apt-get", "update", "-y"])
    subprocess.check_call(
        [
            "sudo",
            "apt-get",
            "install",
            "-y",
            "nginx",
            "python3-venv",
            "mariadb-server",
            "mariadb-client",
        ]
    )
    # Ensure DB service is running (service name varies by distro).
    subprocess.run(["sudo", "systemctl", "enable", "--now", "mariadb"], check=False)
    subprocess.run(["sudo", "systemctl", "enable", "--now", "mysql"], check=False)
    subprocess.check_call(["sudo", "mkdir", "-p", args.watch_root])
    subprocess.check_call(["sudo", "chown", f"{args.user}:{args.user}", args.watch_root])

    def _do_copy(src, dst):
        subprocess.check_call(["sudo", "cp", "-r", str(src), str(dst)])

    print("Copying webapp code...")
    subprocess.check_call(["sudo", "mkdir", "-p", str(app_dir)])
    _do_copy(repo_root / "tools/webapp/app.py", app_dir / "app.py")
    _do_copy(repo_root / "tools/webapp/manage.py", app_dir / "manage.py")
    _do_copy(repo_root / "tools/webapp/django_orm.py", app_dir / "django_orm.py")
    _do_copy(repo_root / "tools/webapp/django_settings.py", app_dir / "django_settings.py")
    _do_copy(repo_root / "tools/webapp/urls.py", app_dir / "urls.py")
    _do_copy(repo_root / "tools/webapp/wsgi.py", app_dir / "wsgi.py")
    _do_copy(repo_root / "tools/webapp/django_app", app_dir)
    _do_copy(repo_root / "tools/webapp/hm_webapp", app_dir)
    _do_copy(repo_root / "tools/webapp/core", app_dir)
    _do_copy(repo_root / "tools/webapp/hockey_rankings.py", app_dir / "hockey_rankings.py")
    _do_copy(
        repo_root / "tools/webapp/scripts/recalc_div_ratings.py", app_dir / "recalc_div_ratings.py"
    )
    subprocess.check_call(["sudo", "mkdir", "-p", str(templates_dir)])
    for t in (repo_root / "tools/webapp/templates").glob("*.html"):
        _do_copy(t, templates_dir / t.name)
    subprocess.check_call(["sudo", "mkdir", "-p", str(static_dir)])
    for s in (repo_root / "tools/webapp/static").glob("*"):
        if s.is_file():
            _do_copy(s, static_dir / s.name)
    subprocess.check_call(["sudo", "mkdir", "-p", str(app_dir / "instance")])

    # Resolve python binary
    if not args.python_bin:
        try:
            base_python = subprocess.check_output(
                [
                    "sudo",
                    "-H",
                    "-u",
                    args.user,
                    "bash",
                    "-lc",
                    "command -v python || command -v python3",
                ],
                text=True,
            ).strip()
            base_python = base_python or "/usr/bin/python3"
        except Exception:
            base_python = "/usr/bin/python3"
    else:
        base_python = args.python_bin

    # Ensure the install root is owned by the app user before creating the venv/pip installing
    subprocess.check_call(["sudo", "chown", "-R", f"{args.user}:{args.user}", str(install_root)])

    venv_dir = install_root / "venv"
    print(f"Using python: {base_python}")
    subprocess.check_call(
        [
            "sudo",
            "-H",
            "-u",
            args.user,
            "bash",
            "-lc",
            f"{base_python} -m venv {venv_dir}",
        ]
    )
    python_bin = venv_dir / "bin/python"
    print(f"Using virtualenv: {python_bin}")
    subprocess.check_call(
        [
            "sudo",
            "-H",
            "-u",
            args.user,
            "bash",
            "-lc",
            f"{python_bin} -m pip install --upgrade pip wheel gunicorn werkzeug pymysql django openpyxl",
        ]
    )

    print("Configuring MariaDB schema and user...")
    sql = f"""
CREATE DATABASE IF NOT EXISTS `{args.db_name}` CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS '{args.db_user}'@'localhost' IDENTIFIED BY '{args.db_pass}';
CREATE USER IF NOT EXISTS '{args.db_user}'@'127.0.0.1' IDENTIFIED BY '{args.db_pass}';
GRANT ALL PRIVILEGES ON `{args.db_name}`.* TO '{args.db_user}'@'localhost';
GRANT ALL PRIVILEGES ON `{args.db_name}`.* TO '{args.db_user}'@'127.0.0.1';
CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'admin';
CREATE USER IF NOT EXISTS 'admin'@'127.0.0.1' IDENTIFIED BY 'admin';
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'127.0.0.1' WITH GRANT OPTION;
FLUSH PRIVILEGES;
"""
    subprocess.check_call(["sudo", "bash", "-lc", f"cat <<'SQL' | mysql -u root\n{sql}\nSQL\n"])

    cfg = {
        "watch_root": args.watch_root,
        "db": {
            "host": args.db_host,
            "port": args.db_port,
            "name": args.db_name,
            "user": args.db_user,
            "pass": args.db_pass,
        },
        "email": {"from": os.environ.get("HM_FROM_EMAIL", "")},
    }
    config_json = app_dir / "config.json"
    if args.import_token:
        cfg["import_token"] = args.import_token
    else:
        # Preserve existing token on redeploy unless explicitly overridden.
        try:
            if config_json.exists():
                prev = json.loads(config_json.read_text(encoding="utf-8"))
                if prev.get("import_token"):
                    cfg["import_token"] = prev["import_token"]
        except Exception:
            pass
    config_json.write_text(json.dumps(cfg, indent=2))
    subprocess.check_call(["sudo", "chown", f"{args.user}:{args.user}", str(config_json)])
    subprocess.check_call(["sudo", "chmod", "600", str(config_json)])

    print("Writing systemd service...")
    unit = f"""
[Unit]
Description=HockeyMOM WebApp (Django via gunicorn)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={args.user}
Group={args.user}
Environment=PYTHONUNBUFFERED=1
Environment=HM_WATCH_ROOT={args.watch_root}
Environment=MSMTP_CONFIG=/etc/msmtprc
Environment=MSMTPRC=/etc/msmtprc
Environment=HM_DB_CONFIG={app_dir}/config.json
Environment=DJANGO_SETTINGS_MODULE=hm_webapp.settings
WorkingDirectory={app_dir}
ExecStart={python_bin} -m gunicorn -b {args.bind_address}:{args.port} --access-logfile - --error-logfile - --capture-output --log-level info wsgi:application
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
""".lstrip()

    sudo_write_text(f"/etc/systemd/system/{SERVICE_NAME}", unit)

    print("Writing weekly Ratings systemd timer...")
    ratings_service = f"""
[Unit]
Description=HockeyMOM: Recompute Ratings (weekly)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User={args.user}
Group={args.user}
Environment=PYTHONUNBUFFERED=1
Environment=HM_DB_CONFIG={app_dir}/config.json
Environment=HM_WATCH_ROOT={args.watch_root}
WorkingDirectory={app_dir}
ExecStart={python_bin} {app_dir}/recalc_div_ratings.py --config {app_dir}/config.json
""".lstrip()
    sudo_write_text("/etc/systemd/system/hm-webapp-div-ratings.service", ratings_service)
    ratings_timer = """
[Unit]
Description=HockeyMOM: Weekly Ratings recalculation

[Timer]
OnCalendar=Wed *-*-* 00:00:00
Persistent=true

[Install]
WantedBy=timers.target
""".lstrip()
    sudo_write_text("/etc/systemd/system/hm-webapp-div-ratings.timer", ratings_timer)

    print("Writing nginx site...")
    nginx_conf = f"""
server {{
    listen {args.nginx_port} default_server;
    server_name {args.server_name};
    client_max_body_size {args.client_max_body_size};

    location /static/ {{
        alias {static_dir}/;
    }}

    location / {{
        proxy_pass http://127.0.0.1:{args.port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Webapp import endpoints can run for minutes; keep nginx timeouts >= gunicorn's --timeout.
        proxy_connect_timeout 60s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }}
}}
""".lstrip()
    sudo_write_text(NGINX_SITE, nginx_conf)

    sites_enabled = Path("/etc/nginx/sites-enabled/hm-webapp")
    if not sites_enabled.exists():
        subprocess.run(["sudo", "ln", "-s", NGINX_SITE, str(sites_enabled)], check=False)
    default_site = Path("/etc/nginx/sites-enabled/default")
    if default_site.exists():
        subprocess.run(["sudo", "rm", "-f", str(default_site)])

    print("Setting ownership and enabling services...")
    subprocess.check_call(["sudo", "chown", "-R", f"{args.user}:{args.user}", str(install_root)])
    subprocess.check_call(["sudo", "systemctl", "daemon-reload"])
    subprocess.check_call(["sudo", "systemctl", "enable", "--now", SERVICE_NAME])
    subprocess.check_call(["sudo", "systemctl", "enable", "--now", "hm-webapp-div-ratings.timer"])

    _ensure_port_available_for_nginx(args.nginx_port, disable_apache2=args.disable_apache2)
    subprocess.check_call(["sudo", "systemctl", "enable", "--now", "nginx"])
    subprocess.check_call(["sudo", "nginx", "-t"])
    subprocess.check_call(["sudo", "systemctl", "restart", "nginx"])

    def _git_cfg(key: str) -> str:
        try:
            return subprocess.check_output(
                ["sudo", "-H", "-u", args.user, "bash", "-lc", f"git config --get {key}"],
                text=True,
            ).strip()
        except Exception:
            return ""

    def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
        import socket

        start = time.time()
        while time.time() - start < timeout_s:
            try:
                s = socket.socket()
                s.settimeout(0.5)
                s.connect((host, int(port)))
                s.close()
                return True
            except Exception:
                try:
                    s.close()
                except Exception:
                    pass
                time.sleep(0.25)
        return False

    if not args.no_default_user:
        email = _git_cfg("user.email")
        name = _git_cfg("user.name")
        if email and name:
            host = "127.0.0.1"
            if not _wait_for_port(host, int(args.port), timeout_s=20.0):
                print("[!] Skipping default user creation: webapp port not reachable yet.")
            else:
                token = str(cfg.get("import_token") or "").strip()
                headers = {"Content-Type": "application/json"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    headers["X-HM-Import-Token"] = token
                body = json.dumps({"email": email, "name": name, "password": "password"}).encode(
                    "utf-8"
                )
                url = f"http://{host}:{int(args.port)}/api/internal/ensure_user"
                try:
                    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        out = json.loads(resp.read().decode("utf-8"))
                    if out.get("ok"):
                        created = bool(out.get("created"))
                        print(
                            f"[ok] Default webapp user ensured: {email} (password='password'){'' if created else ' (already existed)'}"
                        )
                    else:
                        print(f"[!] Default user creation failed: {out}")
                except Exception as e:
                    print(f"[!] Default user creation failed: {e}")

    print("Installed webapp:")
    if args.nginx_port == 80:
        print("  http://localhost/")
    else:
        print(f"  http://localhost:{args.nginx_port}/")
    print(f"  install_root: {install_root}")
    print(f"  uploads root (watch): {args.watch_root}")


if __name__ == "__main__":
    main()

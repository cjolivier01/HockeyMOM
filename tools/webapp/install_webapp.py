#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

SERVICE_NAME = "hm-webapp.service"
NGINX_SITE = "/etc/nginx/sites-available/hm-webapp"


def sudo_write_text(path: str | Path, content: str):
    """Write text to a root-protected file via sudo tee."""
    cmd = ["sudo", "tee", str(path)]
    subprocess.run(cmd, input=content.encode(), check=True)


def main():
    ap = argparse.ArgumentParser(description="Install HM WebApp (Flask + Nginx)")
    ap.add_argument("--install-root", default="/opt/hm-webapp")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER") or os.environ.get("USER"))
    ap.add_argument("--watch-root", default="/data/incoming")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--server-name", default="_")
    ap.add_argument("--client-max-body-size", default="500M")
    ap.add_argument("--db-name", default="hm_app_db")
    ap.add_argument("--db-user", default="hmapp")
    ap.add_argument("--db-pass", default="hmapp_pass")
    ap.add_argument("--db-host", default="127.0.0.1")
    ap.add_argument("--db-port", type=int, default=3306)
    ap.add_argument("--python-bin", default="", help="Python interpreter to run the app")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    install_root = Path(args.install_root)
    app_dir = install_root / "app"
    templates_dir = app_dir / "templates"
    static_dir = app_dir / "static"

    print("Installing OS packages (nginx + python venv tools)...")
    subprocess.check_call(["sudo", "apt-get", "update", "-y"])
    subprocess.check_call(["sudo", "apt-get", "install", "-y", "nginx", "python3-venv"])
    subprocess.check_call(["sudo", "mkdir", "-p", args.watch_root])
    subprocess.check_call(["sudo", "chown", f"{args.user}:{args.user}", args.watch_root])

    def _do_copy(src, dst):
        subprocess.check_call(["sudo", "cp", "-r", str(src), str(dst)])

    print("Copying webapp code...")
    subprocess.check_call(["sudo", "mkdir", "-p", str(app_dir)])
    _do_copy(repo_root / "tools/webapp/app.py", app_dir / "app.py")
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
            f"{python_bin} -m pip install --upgrade pip wheel flask gunicorn werkzeug pymysql",
        ]
    )

    print("Configuring MariaDB schema and user...")
    sql = f"""
CREATE DATABASE IF NOT EXISTS `{args.db_name}` CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS '{args.db_user}'@'localhost' IDENTIFIED BY '{args.db_pass}';
GRANT ALL PRIVILEGES ON `{args.db_name}`.* TO '{args.db_user}'@'localhost';
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
    config_json.write_text(json.dumps(cfg, indent=2))
    subprocess.check_call(["sudo", "chown", f"{args.user}:{args.user}", str(config_json)])
    subprocess.check_call(["sudo", "chmod", "600", str(config_json)])

    print("Writing systemd service...")
    unit = f"""
[Unit]
Description=HM WebApp (Flask via gunicorn)
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
WorkingDirectory={app_dir}
ExecStart={python_bin} -m gunicorn -b 127.0.0.1:{args.port} app:app
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
""".lstrip()

    sudo_write_text(f"/etc/systemd/system/{SERVICE_NAME}", unit)

    print("Writing nginx site...")
    nginx_conf = f"""
server {{
    listen 80 default_server;
    server_name {args.server_name};
    client_max_body_size {args.client_max_body_size};

    location / {{
        proxy_pass http://127.0.0.1:{args.port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""
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
    subprocess.check_call(["sudo", "systemctl", "restart", "nginx"])

    print("Installed webapp:")
    print("  http://localhost/")
    print(f"  install_root: {install_root}")
    print(f"  uploads root (watch): {args.watch_root}")


if __name__ == "__main__":
    main()

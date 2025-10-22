#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

SERVICE_NAME = "hm-webapp.service"
NGINX_SITE = "/etc/nginx/sites-available/hm-webapp"


def main():
    ap = argparse.ArgumentParser(description="Install HM WebApp (Flask + Nginx)")
    ap.add_argument("--install-root", default="/opt/hm-webapp")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER") or os.environ.get("USER"))
    ap.add_argument("--watch-root", default="/data/incoming")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--server-name", default="_")  # nginx server_name
    ap.add_argument("--client-max-body-size", default="500M")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    install_root = Path(args.install_root)
    venv = install_root / "venv"
    app_dir = install_root / "app"
    templates_dir = app_dir / "templates"

    print("Installing OS packages (nginx, python3-venv)...")
    subprocess.check_call(["sudo", "apt-get", "update", "-y"]) 
    subprocess.check_call(["sudo", "apt-get", "install", "-y", "nginx", "python3-venv"])
    subprocess.check_call(["sudo", "mkdir", "-p", args.watch_root])

    print("Copying webapp code...")
    subprocess.check_call(["sudo", "mkdir", "-p", str(app_dir)])
    shutil.copy2(repo_root / "tools/webapp/app.py", app_dir / "app.py")
    subprocess.check_call(["sudo", "mkdir", "-p", str(templates_dir)])
    for t in (repo_root / "tools/webapp/templates").glob("*.html"):
        shutil.copy2(t, templates_dir / t.name)
    # Ensure instance dir is present and owned by service user
    subprocess.check_call(["sudo", "mkdir", "-p", str(app_dir / "instance")])

    print("Creating virtualenv and installing deps...")
    subprocess.check_call(["sudo", "python3", "-m", "venv", str(venv)])
    pip = str(venv / "bin/pip")
    subprocess.check_call(["sudo", pip, "install", "--upgrade", "pip", "wheel"])
    subprocess.check_call(["sudo", pip, "install", "flask", "gunicorn", "werkzeug"])

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
WorkingDirectory={app_dir}
ExecStart={venv}/bin/gunicorn -b 127.0.0.1:{args.port} app:app
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
""".lstrip()
    unit_path = Path("/etc/systemd/system") / SERVICE_NAME
    Path(unit_path).write_text(unit)

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
    Path(NGINX_SITE).write_text(nginx_conf)
    sites_enabled = Path("/etc/nginx/sites-enabled/hm-webapp")
    if not sites_enabled.exists():
        subprocess.run(["sudo", "ln", "-s", NGINX_SITE, str(sites_enabled)], check=False)
    # remove default site if present (optional)
    default_site = Path("/etc/nginx/sites-enabled/default")
    if default_site.exists():
        subprocess.run(["sudo", "rm", "-f", str(default_site)])

    print("Setting ownership and enabling services...")
    subprocess.check_call(["sudo", "chown", "-R", f"{args.user}:{args.user}", str(install_root)])
    subprocess.check_call(["sudo", "systemctl", "daemon-reload"]) 
    subprocess.check_call(["sudo", "systemctl", "enable", "--now", SERVICE_NAME])
    subprocess.check_call(["sudo", "systemctl", "restart", "nginx"]) 

    print("Installed webapp:")
    print(f"  http://localhost/")
    print(f"  install_root: {install_root}")
    print(f"  uploads root (watch): {args.watch_root}")


if __name__ == "__main__":
    main()

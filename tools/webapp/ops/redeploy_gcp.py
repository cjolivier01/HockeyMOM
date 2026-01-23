#!/usr/bin/env python3
"""
Fast redeploy helper for the GCE-based webapp deployment.

This does NOT recreate the VM or rerun the full installer. It:
  - copies updated `tools/webapp/app.py`, ORM files, `templates/`, and `static/` to the VM's `/opt/hm-webapp/app/`
  - ensures Django is installed in `/opt/hm-webapp/venv`
  - ensures the MariaDB login user `admin/admin` exists (idempotent)
  - restarts `hm-webapp.service`

If you changed Python dependencies or system packages, rerun `tools/webapp/ops/deploy_gcp.py` instead.
"""

from __future__ import annotations

import argparse
import shutil
import textwrap
import shlex
import subprocess
import tempfile
from pathlib import Path


WEBAPP_UPLOAD_IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".DS_Store",
    "instance",
    "*.sqlite3",
    "*.db",
    "*.log",
)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, check=check)


def main() -> int:
    ap = argparse.ArgumentParser(description="Redeploy tools/webapp code to an existing GCE VM.")
    ap.add_argument("--project", required=True, help="GCP project id.")
    ap.add_argument("--zone", default="us-central1-a", help="GCE zone.")
    ap.add_argument("--instance", default="hm-webapp", help="GCE instance name.")
    args = ap.parse_args()

    # Stage current repo's tools/webapp on the VM, then copy into /opt/hm-webapp/app.
    _run(
        [
            "gcloud",
            "--quiet",
            "--project",
            args.project,
            "compute",
            "ssh",
            args.instance,
            "--zone",
            args.zone,
            "--command",
            "rm -rf /tmp/hm/tools/webapp && mkdir -p /tmp/hm/tools",
        ]
    )
    repo_root = Path(__file__).resolve().parents[3]
    src_webapp_dir = repo_root / "tools" / "webapp"
    if not src_webapp_dir.exists():
        raise SystemExit(f"Missing webapp source dir: {src_webapp_dir}")
    with tempfile.TemporaryDirectory(prefix="hm_webapp_gcp_stage_") as td:
        stage_dir = Path(td) / "webapp"
        shutil.copytree(
            src_webapp_dir,
            stage_dir,
            ignore=shutil.ignore_patterns(*WEBAPP_UPLOAD_IGNORE_PATTERNS),
        )
        _run(
            [
                "gcloud",
                "--quiet",
                "--project",
                args.project,
                "compute",
                "scp",
                "--recurse",
                str(stage_dir),
                f"{args.instance}:/tmp/hm/tools",
                "--zone",
                args.zone,
            ]
        )
    _run(
        [
            "gcloud",
            "--quiet",
            "--project",
            args.project,
            "compute",
            "ssh",
            args.instance,
            "--zone",
            args.zone,
            "--command",
            textwrap.dedent(
                """
                set -euo pipefail

                if ! /opt/hm-webapp/venv/bin/python -c 'import django' >/dev/null 2>&1; then
                  sudo /opt/hm-webapp/venv/bin/python -m pip install django
                fi
                if ! /opt/hm-webapp/venv/bin/python -c 'import openpyxl' >/dev/null 2>&1; then
                  sudo /opt/hm-webapp/venv/bin/python -m pip install openpyxl
                fi

                sudo mkdir -p /opt/hm-webapp/app/templates /opt/hm-webapp/app/static
                sudo cp /tmp/hm/tools/webapp/app.py /opt/hm-webapp/app/app.py
                sudo cp /tmp/hm/tools/webapp/manage.py /opt/hm-webapp/app/manage.py
                sudo cp /tmp/hm/tools/webapp/django_orm.py /opt/hm-webapp/app/django_orm.py
                sudo cp /tmp/hm/tools/webapp/django_settings.py /opt/hm-webapp/app/django_settings.py
                sudo cp /tmp/hm/tools/webapp/urls.py /opt/hm-webapp/app/urls.py
                sudo cp /tmp/hm/tools/webapp/wsgi.py /opt/hm-webapp/app/wsgi.py
                sudo cp -r /tmp/hm/tools/webapp/django_app /opt/hm-webapp/app/
                sudo cp -r /tmp/hm/tools/webapp/hm_webapp /opt/hm-webapp/app/
                sudo cp -r /tmp/hm/tools/webapp/core /opt/hm-webapp/app/
                sudo cp /tmp/hm/tools/webapp/hockey_rankings.py /opt/hm-webapp/app/hockey_rankings.py
                sudo cp /tmp/hm/tools/webapp/scripts/recalc_div_ratings.py /opt/hm-webapp/app/recalc_div_ratings.py
                sudo cp -r /tmp/hm/tools/webapp/templates/. /opt/hm-webapp/app/templates/
                sudo cp -r /tmp/hm/tools/webapp/static/. /opt/hm-webapp/app/static/
                sudo chown -R colivier:colivier /opt/hm-webapp/app

                # Certbot may generate a TLS server block that omits /static; patch it back in.
                if sudo test -f /etc/nginx/sites-available/hm-webapp; then
                  sudo python3 - <<'PY'
                from __future__ import annotations

                import re
                from pathlib import Path

                path = Path("/etc/nginx/sites-available/hm-webapp")
                text = path.read_text(encoding="utf-8")
                changed = False

                if "location /static/" not in text:
                    m = re.search(r"(?m)^(?P<indent>[ \\t]*)location\\s+/\\s*\\{", text)
                    if m:
                        indent = m.group("indent")
                        block = (
                            f"{indent}location /static/ {{\\n"
                            f"{indent}    alias /opt/hm-webapp/app/static/;\\n"
                            f"{indent}}}\\n\\n"
                        )
                        text = text[: m.start()] + block + text[m.start() :]
                        changed = True

                directives = [
                    ("proxy_connect_timeout", "60s"),
                    ("proxy_send_timeout", "600s"),
                    ("proxy_read_timeout", "600s"),
                ]
                missing = [
                    (name, value)
                    for name, value in directives
                    if re.search(rf"(?m)^\\s*{re.escape(name)}\\b", text) is None
                ]
                if missing:
                    m = re.search(
                        r"(?m)^(?P<indent>[ \\t]*)proxy_pass\\s+http://127\\.0\\.0\\.1:\\d+;\\s*$",
                        text,
                    )
                    if m:
                        indent = m.group("indent")
                        block = "\\n".join(f"{indent}{name} {value};" for name, value in missing)
                        text = text[: m.end()] + "\\n" + block + text[m.end() :]
                        changed = True

                if changed:
                    path.write_text(text, encoding="utf-8")
                PY
                  sudo nginx -t
                  sudo systemctl reload nginx || sudo systemctl restart nginx
                fi

                # Upgrade old systemd service entrypoint to Django (idempotent).
                if sudo test -f /etc/systemd/system/hm-webapp.service && sudo grep -Fq "app:app" /etc/systemd/system/hm-webapp.service; then
                  sudo sed -i 's|Description=HockeyMOM WebApp (Flask via gunicorn)|Description=HockeyMOM WebApp (Django via gunicorn)|g' /etc/systemd/system/hm-webapp.service || true
                  sudo sed -i 's|app:app|wsgi:application|g' /etc/systemd/system/hm-webapp.service
                  if ! sudo grep -q '^Environment=DJANGO_SETTINGS_MODULE=' /etc/systemd/system/hm-webapp.service; then
                    sudo sed -i '/^Environment=HM_DB_CONFIG=/a Environment=DJANGO_SETTINGS_MODULE=hm_webapp.settings' /etc/systemd/system/hm-webapp.service || true
                  fi
                  sudo systemctl daemon-reload
                fi

                sudo systemctl start mariadb >/dev/null 2>&1 || sudo systemctl start mysql >/dev/null 2>&1 || true
                if sudo mysql --connect-timeout=5 -u root -e 'SELECT 1;' >/dev/null 2>&1; then
                  sudo mysql --connect-timeout=5 -u root <<'SQL' || true
                CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'admin';
                CREATE USER IF NOT EXISTS 'admin'@'127.0.0.1' IDENTIFIED BY 'admin';
                GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;
                GRANT ALL PRIVILEGES ON *.* TO 'admin'@'127.0.0.1' WITH GRANT OPTION;
                FLUSH PRIVILEGES;
                SQL
                else
                  echo '[!] Warning: cannot connect to MariaDB as root; skipping DB admin user provisioning' >&2
                fi

                sudo systemctl restart hm-webapp.service
                sudo systemctl is-active hm-webapp.service
                """
            ).strip(),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

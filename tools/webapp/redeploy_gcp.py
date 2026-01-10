#!/usr/bin/env python3
"""
Fast redeploy helper for the GCE-based webapp deployment.

This does NOT recreate the VM or rerun the full installer. It:
  - copies updated `tools/webapp/app.py`, ORM files, `templates/`, and `static/` to the VM's `/opt/hm-webapp/app/`
  - ensures Django is installed in `/opt/hm-webapp/venv`
  - ensures the MariaDB login user `admin/admin` exists (idempotent)
  - restarts `hm-webapp.service`

If you changed Python dependencies or system packages, rerun `tools/webapp/deploy_gcp.py` instead.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


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
            "mkdir -p /tmp/hm/tools",
        ]
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
            "tools/webapp",
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
            "set -euo pipefail; "
            "if ! /opt/hm-webapp/venv/bin/python -c 'import django' >/dev/null 2>&1; then "
            "  sudo /opt/hm-webapp/venv/bin/python -m pip install django; "
            "fi; "
            "sudo mkdir -p /opt/hm-webapp/app/templates /opt/hm-webapp/app/static; "
            "sudo cp /tmp/hm/tools/webapp/app.py /opt/hm-webapp/app/app.py; "
            "sudo cp /tmp/hm/tools/webapp/django_orm.py /opt/hm-webapp/app/django_orm.py; "
            "sudo cp /tmp/hm/tools/webapp/django_settings.py /opt/hm-webapp/app/django_settings.py; "
            "sudo cp /tmp/hm/tools/webapp/urls.py /opt/hm-webapp/app/urls.py; "
            "sudo cp /tmp/hm/tools/webapp/wsgi.py /opt/hm-webapp/app/wsgi.py; "
            "sudo cp -r /tmp/hm/tools/webapp/django_app /opt/hm-webapp/app/; "
            "sudo cp /tmp/hm/tools/webapp/hockey_rankings.py /opt/hm-webapp/app/hockey_rankings.py; "
            "sudo cp /tmp/hm/tools/webapp/recalc_div_ratings.py /opt/hm-webapp/app/recalc_div_ratings.py; "
            "sudo cp -r /tmp/hm/tools/webapp/templates/. /opt/hm-webapp/app/templates/; "
            "sudo cp -r /tmp/hm/tools/webapp/static/. /opt/hm-webapp/app/static/; "
            "sudo chown -R colivier:colivier /opt/hm-webapp/app; "
            "sudo systemctl start mariadb >/dev/null 2>&1 || sudo systemctl start mysql >/dev/null 2>&1 || true; "
            "if sudo mysql --connect-timeout=5 -u root -e 'SELECT 1;' >/dev/null 2>&1; then "
            "  sudo mysql --connect-timeout=5 -u root <<'SQL' || true\n"
            "CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'admin';\n"
            "CREATE USER IF NOT EXISTS 'admin'@'127.0.0.1' IDENTIFIED BY 'admin';\n"
            "GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;\n"
            "GRANT ALL PRIVILEGES ON *.* TO 'admin'@'127.0.0.1' WITH GRANT OPTION;\n"
            "FLUSH PRIVILEGES;\n"
            "SQL\n"
            "else "
            "  echo '[!] Warning: cannot connect to MariaDB as root; skipping DB admin user provisioning' >&2; "
            "fi; "
            "sudo systemctl restart hm-webapp.service; "
            "sudo systemctl is-active hm-webapp.service",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

# Re-deploy updated HM WebApp code (Django) to system install and restart services.
# - Copies app.py, manage.py, hmwebapp/, templates, static to /opt/hm-webapp/app
# - Restarts hm-webapp and nginx
# - Optionally runs the smoke UI script when RUN_SMOKE=1
#
# Usage:
#   tools/webapp/redeploy_local.sh
#   RUN_SMOKE=1 tools/webapp/redeploy_local.sh

APP_DIR="/opt/hm-webapp/app"
SMOKE="${RUN_SMOKE:-0}"

if [ ! -d "$APP_DIR" ]; then
  echo "[!] $APP_DIR not found. Ensure the webapp is installed first." >&2
  exit 1
fi

echo "[i] Copying Django app code, templates, and static to $APP_DIR (sudo required)"
sudo install -m 0644 -D tools/webapp/app.py "$APP_DIR/app.py"
sudo install -m 0755 -D tools/webapp/manage.py "$APP_DIR/manage.py"
sudo mkdir -p "$APP_DIR/hmwebapp"
sudo rsync -a tools/webapp/hmwebapp/ "$APP_DIR/hmwebapp/"
sudo mkdir -p "$APP_DIR/templates" "$APP_DIR/static"
sudo rsync -a tools/webapp/templates/ "$APP_DIR/templates/"
sudo rsync -a tools/webapp/static/ "$APP_DIR/static/"

echo "[i] Restarting services"
sudo systemctl daemon-reload || true
sudo systemctl restart hm-webapp nginx

echo "[i] Verifying endpoints"
sleep 0.5
if curl -sSfI http://127.0.0.1/teams >/dev/null; then
  echo "[ok] /teams reachable (likely 302 to /login if not logged in)"
else
  echo "[!] /teams not reachable. Check service logs: sudo journalctl -u hm-webapp -n 100 --no-pager" >&2
  exit 1
fi

if [ "$SMOKE" = "1" ]; then
  echo "[i] Running smoke UI script"
  bash tools/webapp/smoke_ui.sh
fi

echo "[ok] Redeploy complete"

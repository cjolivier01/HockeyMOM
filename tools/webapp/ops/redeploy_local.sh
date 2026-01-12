#!/usr/bin/env bash
set -euo pipefail

# Re-deploy updated HockeyMOM WebApp code to system install and restart services.
# - Copies app.py, templates, static to /opt/hm-webapp/app
# - Ensures local DB is reachable (when configured)
# - Restarts hm-webapp and nginx, then verifies both :8008 and nginx proxy
# - Optionally runs the smoke UI script when RUN_SMOKE=1
#
# Usage:
#   tools/webapp/ops/redeploy_local.sh
#   RUN_SMOKE=1 tools/webapp/ops/redeploy_local.sh

APP_DIR="/opt/hm-webapp/app"
SMOKE="${RUN_SMOKE:-0}"
PORT="${HM_WEBAPP_PORT:-8008}"
SERVICE_FILE="/etc/systemd/system/hm-webapp.service"

die() {
  echo "[!]" "$@" >&2
  exit 1
}

json_get() {
  local path="$1"
  local dotted="$2"
  python3 - "$path" "$dotted" <<'PY'
import json
import sys

path = sys.argv[1]
dotted = sys.argv[2]
cur = json.load(open(path))
for key in dotted.split("."):
    cur = cur[key]
print(cur)
PY
}

tcp_check() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect((host, port))
except Exception:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    try:
        s.close()
    except Exception:
        pass
PY
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-30}"
  local start
  start="$(date +%s)"
  while true; do
    if tcp_check "$host" "$port"; then
      return 0
    fi
    if [ "$(( $(date +%s) - start ))" -ge "$timeout_s" ]; then
      return 1
    fi
    sleep 0.25
  done
}

ensure_db_admin_user() {
  local db_host="$1"
  local db_port="$2"
  if [[ "$db_host" != "127.0.0.1" && "$db_host" != "localhost" ]]; then
    echo "[i] Skipping DB admin user provisioning (remote DB host=${db_host}:${db_port})"
    return 0
  fi
  echo "[i] Ensuring MariaDB login user admin/admin exists (requires sudo)"
  if ! sudo mysql --connect-timeout=5 -u root -e "SELECT 1;" >/dev/null 2>&1; then
    echo "[!] Cannot connect to MariaDB as root via sudo; skipping DB admin user provisioning" >&2
    return 0
  fi
  if ! sudo mysql --connect-timeout=5 -u root >/dev/null <<'SQL'
CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'admin';
CREATE USER IF NOT EXISTS 'admin'@'127.0.0.1' IDENTIFIED BY 'admin';
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'admin'@'127.0.0.1' WITH GRANT OPTION;
FLUSH PRIVILEGES;
SQL
  then
    echo "[!] Failed to provision DB admin user (admin/admin). Try: sudo mysql -u root" >&2
  fi
}

curl_ok() {
  local url="$1"
  local code
  code="$(curl -sS -o /dev/null -m 8 -w '%{http_code}' "$url" || true)"
  [[ "$code" == "200" || "$code" == "302" ]]
}

print_debug() {
  echo "[i] Port listeners:"
  (ss -ltn 2>/dev/null || true) | grep -E ":(80|${PORT}|3306)\\b" || true

  echo "[i] hm-webapp status:"
  sudo systemctl status hm-webapp -n 120 --no-pager || true

  echo "[i] nginx status:"
  sudo systemctl status nginx -n 60 --no-pager || true

  echo "[i] mariadb status:"
  sudo systemctl status mariadb -n 60 --no-pager || true

  echo "[i] hm-webapp logs:"
  sudo journalctl -u hm-webapp -n 200 --no-pager || true
}

on_err() {
  local code="$?"
  echo "[!] Redeploy failed (exit=$code). Debug info follows." >&2
  print_debug
  exit "$code"
}
trap on_err ERR

if [ ! -d "$APP_DIR" ]; then
  die "$APP_DIR not found. Ensure the webapp is installed first."
fi

echo "[i] Validating sudo access"
if ! sudo -n true 2>/dev/null; then
  die "sudo requires a password/tty. Run: sudo -v  (in a terminal), then re-run: tools/webapp/ops/redeploy_local.sh"
fi

echo "[i] Copying app.py, templates, and static to $APP_DIR (sudo required)"
sudo install -m 0644 -D tools/webapp/app.py "$APP_DIR/app.py"
sudo install -m 0755 -D tools/webapp/manage.py "$APP_DIR/manage.py"
sudo install -m 0644 -D tools/webapp/django_orm.py "$APP_DIR/django_orm.py"
sudo install -m 0644 -D tools/webapp/django_settings.py "$APP_DIR/django_settings.py"
sudo install -m 0644 -D tools/webapp/urls.py "$APP_DIR/urls.py"
sudo install -m 0644 -D tools/webapp/wsgi.py "$APP_DIR/wsgi.py"
sudo install -m 0644 -D tools/webapp/hockey_rankings.py "$APP_DIR/hockey_rankings.py"
sudo install -m 0755 -D tools/webapp/scripts/recalc_div_ratings.py "$APP_DIR/recalc_div_ratings.py"
sudo mkdir -p "$APP_DIR/templates" "$APP_DIR/static"
sudo rsync -a tools/webapp/templates/ "$APP_DIR/templates/"
sudo rsync -a tools/webapp/static/ "$APP_DIR/static/"
sudo mkdir -p "$APP_DIR/django_app"
sudo rsync -a tools/webapp/django_app/ "$APP_DIR/django_app/"
sudo mkdir -p "$APP_DIR/hm_webapp"
sudo rsync -a tools/webapp/hm_webapp/ "$APP_DIR/hm_webapp/"

CONFIG_JSON="$APP_DIR/config.json"
if [ ! -f "$CONFIG_JSON" ]; then
  die "Missing $CONFIG_JSON; re-run installer (tools/webapp/ops/install_webapp.py)."
fi

echo "[i] Ensuring Django is installed in the webapp venv"
if ! /opt/hm-webapp/venv/bin/python -c "import django" >/dev/null 2>&1; then
  echo "[i] Installing Django into /opt/hm-webapp/venv"
  /opt/hm-webapp/venv/bin/python -m pip install -q django 2>/dev/null \
    || sudo /opt/hm-webapp/venv/bin/python -m pip install django
fi

# Ensure gunicorn logs to journald (so Internal Server Error always has a traceback in `journalctl`).
if sudo test -f "$SERVICE_FILE"; then
  if sudo grep -q -- "app:app" "$SERVICE_FILE"; then
    echo "[i] Updating $SERVICE_FILE gunicorn entrypoint to Django WSGI (wsgi:application)"
    sudo perl -0777 -i -pe 's/^(ExecStart=.*?gunicorn\\b[^\\n]*?)\\s+app:app\\s*$/\\1 wsgi:application/m' "$SERVICE_FILE"
  fi
  if ! sudo grep -q -- "--error-logfile" "$SERVICE_FILE"; then
    echo "[i] Updating $SERVICE_FILE to capture gunicorn output"
    sudo perl -0777 -i -pe 's/^(ExecStart=.*?gunicorn\\b[^\\n]*?)\\s+wsgi:application\\s*$/\\1 --access-logfile - --error-logfile - --capture-output --log-level info wsgi:application/m' "$SERVICE_FILE"
  fi
fi

# Ensure nginx serves /static/ directly for the Django UI.
NGINX_SITE="/etc/nginx/sites-available/hm-webapp"
if sudo test -f "$NGINX_SITE"; then
  if ! sudo grep -q "location /static/" "$NGINX_SITE"; then
    echo "[i] Updating $NGINX_SITE to serve /static/ from $APP_DIR/static"
    sudo python3 - "$NGINX_SITE" "$APP_DIR" <<'PY'
import os
import re
import sys
import tempfile

path = sys.argv[1]
app_dir = sys.argv[2].rstrip("/")

with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

if "location /static/" in text:
    raise SystemExit(0)

insert = f\"\"\"\n    location /static/ {{\n        alias {app_dir}/static/;\n    }}\n\"\"\".rstrip()

m = re.search(r\"\\n\\}\\s*$\", text, flags=re.S)
if not m:
    raise SystemExit(0)

text2 = text[: m.start()] + \"\\n\\n\" + insert + \"\\n\" + text[m.start() :]

dir_name = os.path.dirname(path) or \".\"
fd, tmp = tempfile.mkstemp(prefix=\"hm-webapp-nginx-\", dir=dir_name)
try:
    with os.fdopen(fd, \"w\", encoding=\"utf-8\") as fh:
        fh.write(text2)
    os.replace(tmp, path)
finally:
    try:
        if os.path.exists(tmp):
            os.unlink(tmp)
    except Exception:
        pass
PY
  fi
fi

DB_HOST="$(json_get "$CONFIG_JSON" "db.host")"
DB_PORT="$(json_get "$CONFIG_JSON" "db.port")"

echo "[i] Restarting services (hm-webapp port=$PORT)"
sudo systemctl daemon-reload || true

echo "[i] Stopping hm-webapp (to avoid port conflicts)"
sudo systemctl stop hm-webapp || true

echo "[i] Ensuring DB reachable (${DB_HOST}:${DB_PORT})"
if [[ "$DB_HOST" == "127.0.0.1" || "$DB_HOST" == "localhost" ]]; then
  if ! wait_for_tcp "$DB_HOST" "$DB_PORT" 2; then
    echo "[i] Local MySQL/MariaDB not reachable; starting DB service"
    sudo systemctl start mariadb || true
    sudo systemctl start mysql || true
  fi
fi
if ! wait_for_tcp "$DB_HOST" "$DB_PORT" 20; then
  die "DB still not reachable on ${DB_HOST}:${DB_PORT}. Check: sudo journalctl -u mariadb -n 200 --no-pager"
fi

ensure_db_admin_user "$DB_HOST" "$DB_PORT"

echo "[i] Starting hm-webapp and nginx"
sudo systemctl restart hm-webapp
sudo systemctl restart nginx

echo "[i] Ensuring weekly Ratings timer is installed (Wed 00:00)"
APP_USER="$(stat -c '%U' "$APP_DIR" 2>/dev/null || echo "${SUDO_USER:-$USER}")"
UNIT_SERVICE="/etc/systemd/system/hm-webapp-div-ratings.service"
UNIT_TIMER="/etc/systemd/system/hm-webapp-div-ratings.timer"
sudo tee "$UNIT_SERVICE" >/dev/null <<EOF
[Unit]
Description=HockeyMOM: Recompute Ratings (weekly)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${APP_USER}
Group=${APP_USER}
Environment=PYTHONUNBUFFERED=1
Environment=HM_DB_CONFIG=${APP_DIR}/config.json
Environment=HM_WATCH_ROOT=/data/incoming
WorkingDirectory=${APP_DIR}
ExecStart=/opt/hm-webapp/venv/bin/python ${APP_DIR}/recalc_div_ratings.py --config ${APP_DIR}/config.json
EOF

sudo tee "$UNIT_TIMER" >/dev/null <<EOF
[Unit]
Description=HockeyMOM: Weekly Ratings recalculation

[Timer]
OnCalendar=Wed *-*-* 00:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload || true
sudo systemctl enable --now hm-webapp-div-ratings.timer || true

echo "[i] Waiting for hm-webapp to listen on :$PORT"
if ! wait_for_tcp "127.0.0.1" "$PORT" 20; then
  die "hm-webapp not listening on :$PORT"
fi

echo "[i] Verifying endpoints"
sleep 0.5
if curl_ok "http://127.0.0.1:${PORT}/teams"; then
  echo "[ok] http://127.0.0.1:${PORT}/teams reachable"
else
  die "hm-webapp not reachable on :$PORT (try: sudo journalctl -u hm-webapp -n 200 --no-pager)"
fi

if curl_ok "http://127.0.0.1/teams"; then
  echo "[ok] http://127.0.0.1/teams reachable (nginx proxy)"
else
  die "nginx proxy not reachable on port 80 (/teams); check: sudo systemctl status nginx hm-webapp"
fi

if [ "$SMOKE" = "1" ]; then
  echo "[i] Running smoke UI script"
  bash tools/webapp/ops/smoke_ui.sh
fi

echo "[ok] Redeploy complete"

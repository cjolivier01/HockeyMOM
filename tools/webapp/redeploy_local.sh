#!/usr/bin/env bash
set -euo pipefail

# Re-deploy updated HockeyMOM WebApp code to system install and restart services.
# - Copies app.py, templates, static to /opt/hm-webapp/app
# - Ensures local DB is reachable (when configured)
# - Restarts hm-webapp and nginx, then verifies both :8008 and nginx proxy
# - Optionally runs the smoke UI script when RUN_SMOKE=1
#
# Usage:
#   tools/webapp/redeploy_local.sh
#   RUN_SMOKE=1 tools/webapp/redeploy_local.sh

APP_DIR="/opt/hm-webapp/app"
SMOKE="${RUN_SMOKE:-0}"
PORT="${HM_WEBAPP_PORT:-8008}"

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

echo "[i] Validating sudo access (may prompt)"
sudo -v

echo "[i] Copying app.py, templates, and static to $APP_DIR (sudo required)"
sudo install -m 0644 -D tools/webapp/app.py "$APP_DIR/app.py"
sudo install -m 0644 -D tools/webapp/hockey_rankings.py "$APP_DIR/hockey_rankings.py"
sudo mkdir -p "$APP_DIR/templates" "$APP_DIR/static"
sudo rsync -a tools/webapp/templates/ "$APP_DIR/templates/"
sudo rsync -a tools/webapp/static/ "$APP_DIR/static/"

CONFIG_JSON="$APP_DIR/config.json"
if [ ! -f "$CONFIG_JSON" ]; then
  die "Missing $CONFIG_JSON; re-run installer (tools/webapp/install_webapp.py)."
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

echo "[i] Starting hm-webapp and nginx"
sudo systemctl restart hm-webapp
sudo systemctl restart nginx

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
  bash tools/webapp/smoke_ui.sh
fi

echo "[ok] Redeploy complete"

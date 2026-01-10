#!/usr/bin/env bash
set -euo pipefail

WEBAPP_URL="${WEBAPP_URL:-http://127.0.0.1:8008}"
WEB_ACCESS_KEY="${WEB_ACCESS_KEY:-}"
SHIFT_FILE_LIST="${SHIFT_FILE_LIST:-}"
DEPLOY_ONLY=0
DROP_DB=0
DROP_DB_ONLY=0
SPREADSHEETS_ONLY=0
PARSE_ONLY=0
T2S_SCRAPE=0
NO_DEFAULT_USER=0

GIT_USER_EMAIL="$(git config --get user.email 2>/dev/null || true)"
GIT_USER_NAME="$(git config --get user.name 2>/dev/null || true)"
OWNER_EMAIL="${OWNER_EMAIL:-${GIT_USER_EMAIL:-cjolivier01@gmail.com}}"
OWNER_NAME="${OWNER_NAME:-${GIT_USER_NAME:-$OWNER_EMAIL}}"

usage() {
  cat <<'EOF'
Usage: ./import_webapp.sh [--deploy-only] [--drop-db | --drop-db-only] [--spreadsheets-only] [--parse-only]

Environment:
  WEBAPP_URL        Webapp base URL (default: http://127.0.0.1:8008)
  WEB_ACCESS_KEY    Optional token args passed through to import scripts
  SHIFT_FILE_LIST   Shift spreadsheet file list (default: ~/RVideos/game_list_long.txt if present, else ~/Videos/game_list_long.txt)

Options:
  --deploy-only     Only redeploy/restart the webapp and exit (no reset/import/upload)
  --drop-db         Drop (and recreate) the local webapp MariaDB database, then continue
  --drop-db-only    Drop (and recreate) the database, then exit (implies --drop-db)
  --no-default-user Skip auto-creating a default webapp user from git config (user.email/user.name)
  --spreadsheets-only  Seed only from shift spreadsheets (skip TimeToScore import; avoids T2S usage in parse_stats_inputs)
  --parse-only      Only run scripts/parse_stats_inputs.py upload (skip reset + TimeToScore import); forces --webapp-replace
  --scrape          Force re-scraping TimeToScore game pages (overrides local cache) when running the T2S import step
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-only) DEPLOY_ONLY=1; shift ;;
    --drop-db) DROP_DB=1; shift ;;
    --drop-db-only) DROP_DB=1; DROP_DB_ONLY=1; shift ;;
    --no-default-user) NO_DEFAULT_USER=1; shift ;;
    --spreadsheets-only|--no-time2score|--no-t2s) SPREADSHEETS_ONLY=1; shift ;;
    --parse-only|--parse-stats-only|--spreadsheets-upload-only) PARSE_ONLY=1; shift ;;
    --scrape|--t2s-scrape) T2S_SCRAPE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[!] Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "./p" ]]; then
  echo "[!] Missing executable: ./p" >&2
  echo "    Run from repo root, or use ./run.sh / bazel run helpers." >&2
  exit 2
fi

if [[ "${DROP_DB_ONLY}" == "1" && "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[!] --drop-db-only cannot be combined with --deploy-only" >&2
  exit 2
fi
if [[ "${PARSE_ONLY}" == "1" && "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[!] --parse-only cannot be combined with --deploy-only" >&2
  exit 2
fi
if [[ "${PARSE_ONLY}" == "1" && "${DROP_DB_ONLY}" == "1" ]]; then
  echo "[!] --parse-only cannot be combined with --drop-db-only" >&2
  exit 2
fi

drop_db_local() {
  local cfg="/opt/hm-webapp/app/config.json"
  echo "[i] Dropping/recreating local DB (requires sudo)"
  if ! sudo -n true 2>/dev/null; then
    echo "[!] sudo requires a password/tty. Run: sudo -v  (in a terminal), then re-run this script." >&2
    exit 2
  fi

  sudo systemctl stop hm-webapp.service >/dev/null 2>&1 || sudo systemctl stop hm-webapp >/dev/null 2>&1 || true
  sudo systemctl start mariadb >/dev/null 2>&1 || sudo systemctl start mysql >/dev/null 2>&1 || true

  python3 - <<'PY' | sudo mysql -u root
import json
from pathlib import Path

cfg_path = Path("/opt/hm-webapp/app/config.json")
cfg = {}
try:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception:
    cfg = {}

db = cfg.get("db") or {}
db_name = str(db.get("name") or "hm_app_db")
db_user = str(db.get("user") or "hmapp")
db_pass = str(db.get("pass") or "hmapp_pass")

def esc_ident(s: str) -> str:
    return s.replace("`", "``")

def esc_str(s: str) -> str:
    return s.replace("\\\\", "\\\\\\\\").replace("'", "''")

name_i = esc_ident(db_name)
user_s = esc_str(db_user)
pass_s = esc_str(db_pass)

print(f"DROP DATABASE IF EXISTS `{name_i}`;")
print(f"CREATE DATABASE IF NOT EXISTS `{name_i}` CHARACTER SET utf8mb4;")
print(f"CREATE USER IF NOT EXISTS '{user_s}'@'localhost' IDENTIFIED BY '{pass_s}';")
print(f"CREATE USER IF NOT EXISTS '{user_s}'@'127.0.0.1' IDENTIFIED BY '{pass_s}';")
print(f"GRANT ALL PRIVILEGES ON `{name_i}`.* TO '{user_s}'@'localhost';")
print(f"GRANT ALL PRIVILEGES ON `{name_i}`.* TO '{user_s}'@'127.0.0.1';")
print("CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'admin';")
print("CREATE USER IF NOT EXISTS 'admin'@'127.0.0.1' IDENTIFIED BY 'admin';")
print("GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;")
print("GRANT ALL PRIVILEGES ON *.* TO 'admin'@'127.0.0.1' WITH GRANT OPTION;")
print("FLUSH PRIVILEGES;")
PY

  sudo systemctl restart hm-webapp.service >/dev/null 2>&1 || sudo systemctl restart hm-webapp >/dev/null 2>&1 || true
}

# If your server requires an import token, set one of:
#   export HM_WEBAPP_IMPORT_TOKEN="...secret..."
#   export WEB_ACCESS_KEY="--import-token=...secret..."
# (Local dev defaults typically do not require a token.)

# ./p tools/webapp/scripts/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com
# ./p tools/webapp/scripts/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com --division 6:0

# PROJECT_ID="sage-courier-241217"
# REDEPLOY_WEB="python3 tools/webapp/ops/redeploy_gcp.py --project ${PROJECT_ID} --zone us-central1-a --instance hm-webapp"
# $REDEPLOY_WEB

if [[ "${DROP_DB}" == "1" ]]; then
  drop_db_local
  if [[ "${DROP_DB_ONLY}" == "1" ]]; then
    echo "[i] --drop-db-only: done"
    exit 0
  fi
fi

echo "[i] Redeploying local webapp"
./tools/webapp/ops/redeploy_local.sh

echo "[i] Verifying webapp endpoint"
curl -sS -o /dev/null -m 10 -f -I "${WEBAPP_URL}/teams" >/dev/null

if [[ "${NO_DEFAULT_USER}" == "0" && -n "${GIT_USER_EMAIL}" && -n "${GIT_USER_NAME}" ]]; then
  echo "[i] Ensuring default webapp user from git config: ${GIT_USER_EMAIL} (password='password')"
  IMPORT_TOKEN="${HM_WEBAPP_IMPORT_TOKEN:-}"
  if [[ -z "${IMPORT_TOKEN}" ]]; then
    IMPORT_TOKEN="$(python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("/opt/hm-webapp/app/config.json")
try:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception:
    cfg = {}
print(str(cfg.get("import_token") or "").strip())
PY
)"
  fi
  HDRS=( -H "Content-Type: application/json" )
  if [[ -n "${IMPORT_TOKEN}" ]]; then
    HDRS+=( -H "Authorization: Bearer ${IMPORT_TOKEN}" -H "X-HM-Import-Token: ${IMPORT_TOKEN}" )
  fi
  PAYLOAD="$(GIT_USER_EMAIL="${GIT_USER_EMAIL}" GIT_USER_NAME="${GIT_USER_NAME}" python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "email": os.environ.get("GIT_USER_EMAIL") or "",
            "name": os.environ.get("GIT_USER_NAME") or "",
            "password": "password",
        }
    )
)
PY
)"
  if ! curl -sS -f -X POST "${HDRS[@]}" --data "${PAYLOAD}" "${WEBAPP_URL}/api/internal/ensure_user" >/dev/null; then
    echo "[!] Failed to ensure default webapp user via ${WEBAPP_URL}/api/internal/ensure_user" >&2
    echo "    Hint: if WEBAPP_URL points at nginx (port 80), you may need an import token; try WEBAPP_URL=http://127.0.0.1:8008" >&2
  fi
fi

if [[ "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[i] --deploy-only: skipping reset/import/upload"
  exit 0
fi

if [[ "${PARSE_ONLY}" == "1" ]]; then
  echo "[i] --parse-only: skipping league reset and TimeToScore import"
else
  echo "[i] Resetting league data"
  ./p tools/webapp/scripts/reset_league_data.py --force

  if [[ "${SPREADSHEETS_ONLY}" == "1" ]]; then
    echo "[i] --spreadsheets-only: skipping TimeToScore import"
  else
    echo "[i] Importing TimeToScore (caha -> Norcal)"
    # python3 tools/webapp/scripts/import_time2score.py --source=caha --league-name=Norcal --season 0 --api-url "${WEBAPP_URL}" ${WEB_ACCESS_KEY} --user-email cjolivier01@gmail.com --division 6:0
    T2S_ARGS=()
    if [[ "${T2S_SCRAPE}" == "1" ]]; then
      T2S_ARGS+=( "--scrape" )
    fi
    ./p tools/webapp/scripts/import_time2score.py --source=caha --league-name=Norcal --season 0 --api-url "${WEBAPP_URL}" "${T2S_ARGS[@]}" ${WEB_ACCESS_KEY} --user-email "${OWNER_EMAIL}"
  fi
fi

#echo "[i] Uploading shift spreadsheets"
if [[ -z "${SHIFT_FILE_LIST}" ]]; then
  if [[ -f "$HOME/RVideos/game_list_long.txt" ]]; then
    SHIFT_FILE_LIST="$HOME/RVideos/game_list_long.txt"
  else
    SHIFT_FILE_LIST="$HOME/Videos/game_list_long.txt"
  fi
fi
if [[ ! -f "${SHIFT_FILE_LIST}" ]]; then
  echo "[!] SHIFT_FILE_LIST not found: ${SHIFT_FILE_LIST}" >&2
  echo "    Set it explicitly, e.g.: export SHIFT_FILE_LIST=~/RVideos/game_list_long.txt" >&2
  exit 2
fi
SPREADSHEET_ARGS=()
if [[ "${SPREADSHEETS_ONLY}" == "1" ]]; then
  SPREADSHEET_ARGS+=( "--no-time2score" )
fi
if [[ "${PARSE_ONLY}" == "1" ]]; then
  SPREADSHEET_ARGS+=( "--webapp-replace" )
fi
./p scripts/parse_stats_inputs.py \
  --file-list "${SHIFT_FILE_LIST}" \
  --shifts \
  --no-scripts \
  --upload-webapp \
  --webapp-url="${WEBAPP_URL}" \
  "${SPREADSHEET_ARGS[@]}" \
  ${WEB_ACCESS_KEY} \
  --webapp-owner-email "${OWNER_EMAIL}" \
  --webapp-league-name=Norcal

echo "[i] Recalculating Ratings (REST)"
IMPORT_TOKEN="${HM_WEBAPP_IMPORT_TOKEN:-}"
if [[ -z "${IMPORT_TOKEN}" ]]; then
  IMPORT_TOKEN="$(python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("/opt/hm-webapp/app/config.json")
try:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
except Exception:
    cfg = {}
print(str(cfg.get("import_token") or "").strip())
PY
)"
fi
HDRS=( -H "Content-Type: application/json" )
if [[ -n "${IMPORT_TOKEN}" ]]; then
  HDRS+=( -H "Authorization: Bearer ${IMPORT_TOKEN}" -H "X-HM-Import-Token: ${IMPORT_TOKEN}" )
fi
RATINGS_PAYLOAD="$(python3 - <<'PY'
import json

print(json.dumps({"league_name": "Norcal"}))
PY
)"
curl -sS -m 300 -f -X POST "${HDRS[@]}" --data "${RATINGS_PAYLOAD}" "${WEBAPP_URL}/api/internal/recalc_div_ratings" >/dev/null

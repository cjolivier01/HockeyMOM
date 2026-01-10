#!/usr/bin/env bash
set -euo pipefail

# GCP-deployed webapp URL (default points at the production instance we set up).
WEBAPP_URL="${WEBAPP_URL:-https://www.jrsharks2013.org}"

# Import token (required for production; local dev typically does not require it).
# Do NOT hardcode this in the script; pass it via env.
HM_WEBAPP_IMPORT_TOKEN="${HM_WEBAPP_IMPORT_TOKEN:-}"

# League/user defaults.
LEAGUE_NAME="${LEAGUE_NAME:-Norcal}"
NO_DEFAULT_USER=0

GIT_USER_EMAIL="$(git config --get user.email 2>/dev/null || true)"
GIT_USER_NAME="$(git config --get user.name 2>/dev/null || true)"
OWNER_EMAIL="${OWNER_EMAIL:-${GIT_USER_EMAIL:-cjolivier01@gmail.com}}"
OWNER_NAME="${OWNER_NAME:-${GIT_USER_NAME:-$OWNER_EMAIL}}"

# Game list for shift spreadsheet uploader.
if [[ -z "${SHIFT_FILE_LIST:-}" ]]; then
  if [[ -f "$HOME/RVideos/game_list_long.txt" ]]; then
    SHIFT_FILE_LIST="$HOME/RVideos/game_list_long.txt"
  else
    SHIFT_FILE_LIST="$HOME/Videos/game_list_long.txt"
  fi
fi

# GCP instance info for code redeploy.
PROJECT_ID="${PROJECT_ID:-sage-courier-241217}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-hm-webapp}"
DEPLOY_ONLY=0
DROP_DB=0
DROP_DB_ONLY=0
SPREADSHEETS_ONLY=0
T2S_SCRAPE=0

require_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "[!] Missing required command: ${name}" >&2
    return 1
  fi
}

usage() {
  cat <<'EOF'
Usage: ./gcp_import_webapp.sh [--deploy-only] [--drop-db | --drop-db-only] [--spreadsheets-only]

Environment:
  WEBAPP_URL              Webapp base URL (default: https://www.jrsharks2013.org)
  HM_WEBAPP_IMPORT_TOKEN  Import token (required for reset/import/upload; not needed for --deploy-only)
  LEAGUE_NAME             League name (default: Norcal)
  OWNER_EMAIL             League owner email (default: git config user.email, else cjolivier01@gmail.com)
  OWNER_NAME              League owner display name (default: git config user.name, else OWNER_EMAIL)
  SHIFT_FILE_LIST         Shift spreadsheet file list (default: ~/RVideos/game_list_long.txt if present, else ~/Videos/game_list_long.txt)
  PROJECT_ID              GCP project id (default: sage-courier-241217)
  ZONE                    GCE zone (default: us-central1-a)
  INSTANCE                GCE instance name (default: hm-webapp)

Options:
  --deploy-only           Only redeploy/restart the webapp and exit (no reset/import/upload)
  --drop-db               Drop (and recreate) the webapp MariaDB database on the GCE instance, then continue
  --drop-db-only          Drop (and recreate) the database, then exit (implies --drop-db)
  --no-default-user       Skip auto-creating a default webapp user from git config (user.email/user.name)
  --spreadsheets-only     Seed only from shift spreadsheets (skip TimeToScore import; avoids T2S usage in parse_stats_inputs)
  --scrape                Force re-scraping TimeToScore game pages (overrides local cache) when running the T2S import step
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-only) DEPLOY_ONLY=1; shift ;;
    --drop-db) DROP_DB=1; shift ;;
    --drop-db-only) DROP_DB=1; DROP_DB_ONLY=1; shift ;;
    --no-default-user) NO_DEFAULT_USER=1; shift ;;
    --spreadsheets-only|--no-time2score|--no-t2s) SPREADSHEETS_ONLY=1; shift ;;
    --scrape|--t2s-scrape) T2S_SCRAPE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[!] Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${DROP_DB_ONLY}" == "1" && "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[!] --drop-db-only cannot be combined with --deploy-only" >&2
    exit 2
fi

if [[ ! -x "./p" ]]; then
  echo "[!] Missing executable: ./p" >&2
  echo "    Run from repo root, or use ./run.sh / bazel run helpers." >&2
  exit 2
fi

require_cmd curl || exit 2
require_cmd python3 || exit 2
require_cmd gcloud || {
  echo "    Install the Google Cloud CLI and run: gcloud auth login" >&2
  echo "    See: https://cloud.google.com/sdk/docs/install" >&2
  exit 2
}

is_local_url() {
  case "${WEBAPP_URL}" in
    http://127.0.0.1:*|http://localhost:*|https://127.0.0.1:*|https://localhost:* ) return 0 ;;
    *) return 1 ;;
  esac
}

read_default_token() {
  local token_file="$HOME/.ssh/hm-webapp-token.txt"
  if [[ -n "${HM_WEBAPP_IMPORT_TOKEN:-}" ]]; then
    return 0
  fi
  if [[ -f "${token_file}" ]]; then
    HM_WEBAPP_IMPORT_TOKEN="$(cat "${token_file}")"
  fi
}

gcp_ssh() {
  local cmd="$1"
  gcloud --quiet --project "${PROJECT_ID}" compute ssh "${INSTANCE}" --zone "${ZONE}" --command "${cmd}"
}

drop_db_gcp() {
  local remote_cmd
  remote_cmd="$(cat <<'EOF'
set -euo pipefail

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
EOF
)"
  echo "[i] Dropping/recreating DB on ${INSTANCE} via gcloud ssh"
  gcp_ssh "${remote_cmd}"
}

TOKEN_ARGS=()
RESET_TOKEN_ARGS=()
if [[ "${DROP_DB}" == "1" ]]; then
  drop_db_gcp
  if [[ "${DROP_DB_ONLY}" == "1" ]]; then
    echo "[i] --drop-db-only: done"
    exit 0
  fi
fi

echo "[i] Redeploying GCP webapp (code-only)"
python3 tools/webapp/redeploy_gcp.py --project "${PROJECT_ID}" --zone "${ZONE}" --instance "${INSTANCE}"

echo "[i] Verifying webapp endpoint"
curl -sS -o /dev/null -m 15 -f -I "${WEBAPP_URL}/" >/dev/null

read_default_token
if [[ "${NO_DEFAULT_USER}" == "0" && -n "${GIT_USER_EMAIL}" && -n "${GIT_USER_NAME}" ]]; then
  echo "[i] Ensuring default webapp user from git config: ${GIT_USER_EMAIL} (password='password')"
  HDRS=( -H "Content-Type: application/json" )
  if [[ -n "${HM_WEBAPP_IMPORT_TOKEN:-}" ]]; then
    HDRS+=( -H "Authorization: Bearer ${HM_WEBAPP_IMPORT_TOKEN}" -H "X-HM-Import-Token: ${HM_WEBAPP_IMPORT_TOKEN}" )
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
  if ! curl -sS -m 30 -f -X POST "${HDRS[@]}" --data "${PAYLOAD}" "${WEBAPP_URL}/api/internal/ensure_user" >/dev/null; then
    echo "[!] Failed to ensure default webapp user via ${WEBAPP_URL}/api/internal/ensure_user" >&2
    echo "    Hint: if your server requires an import token, set: export HM_WEBAPP_IMPORT_TOKEN='...'" >&2
  fi
fi

if [[ "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[i] --deploy-only: skipping reset/import/upload"
  exit 0
fi

if [[ -n "${HM_WEBAPP_IMPORT_TOKEN:-}" ]]; then
  TOKEN_ARGS+=( "--import-token=${HM_WEBAPP_IMPORT_TOKEN}" )
  RESET_TOKEN_ARGS+=( "--webapp-token=${HM_WEBAPP_IMPORT_TOKEN}" )
else
  if ! is_local_url; then
    echo "[!] HM_WEBAPP_IMPORT_TOKEN is not set. Production imports will be rejected." >&2
    echo "    Set it like: export HM_WEBAPP_IMPORT_TOKEN='...'" >&2
    exit 2
  fi
fi

echo "[i] Ensuring league '${LEAGUE_NAME}' is owned by ${OWNER_EMAIL}"
LEAGUE_OWNER_PAYLOAD="$(
  LEAGUE_NAME="${LEAGUE_NAME}" OWNER_EMAIL="${OWNER_EMAIL}" python3 - <<'PY'
import json, os
print(json.dumps({"league_name": os.environ["LEAGUE_NAME"], "owner_email": os.environ["OWNER_EMAIL"], "shared": True}))
PY
)"
AUTH_HEADER=()
if [[ -n "${HM_WEBAPP_IMPORT_TOKEN:-}" ]]; then
  AUTH_HEADER+=( -H "Authorization: Bearer ${HM_WEBAPP_IMPORT_TOKEN}" )
fi
curl -sS -m 30 -f \
  -X POST "${WEBAPP_URL}/api/internal/ensure_league_owner" \
  "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d "${LEAGUE_OWNER_PAYLOAD}" >/dev/null

echo "[i] Resetting league data (REST)"
./p tools/webapp/reset_league_data.py \
  --force \
  --webapp-url "${WEBAPP_URL}" \
  "${RESET_TOKEN_ARGS[@]}" \
  --webapp-owner-email "${OWNER_EMAIL}" \
  --league-name "${LEAGUE_NAME}"

if [[ "${SPREADSHEETS_ONLY}" == "1" ]]; then
  echo "[i] --spreadsheets-only: skipping TimeToScore import"
else
  echo "[i] Importing TimeToScore (caha -> ${LEAGUE_NAME}) via REST"
  T2S_ARGS=()
  if [[ "${T2S_SCRAPE}" == "1" ]]; then
    T2S_ARGS+=( "--scrape" )
  fi
  ./p tools/webapp/import_time2score.py \
    --source=caha \
    --league-name="${LEAGUE_NAME}" \
    --season 0 \
    --api-url "${WEBAPP_URL}" \
    "${T2S_ARGS[@]}" \
    "${TOKEN_ARGS[@]}" \
    --user-email "${OWNER_EMAIL}"
fi

echo "[i] Uploading shift spreadsheets via REST"
if [[ ! -f "${SHIFT_FILE_LIST}" ]]; then
  echo "[!] SHIFT_FILE_LIST not found: ${SHIFT_FILE_LIST}" >&2
  echo "    Set it explicitly, e.g.: export SHIFT_FILE_LIST=~/RVideos/game_list_long.txt" >&2
  exit 2
fi
SPREADSHEET_ARGS=()
if [[ "${SPREADSHEETS_ONLY}" == "1" ]]; then
  SPREADSHEET_ARGS+=( "--no-time2score" )
fi
./p scripts/parse_stats_inputs.py \
  --file-list "${SHIFT_FILE_LIST}" \
  --shifts \
  --upload-webapp \
  --webapp-url="${WEBAPP_URL}" \
  "${SPREADSHEET_ARGS[@]}" \
  "${TOKEN_ARGS[@]}" \
  --webapp-owner-email "${OWNER_EMAIL}" \
  --webapp-league-name "${LEAGUE_NAME}"

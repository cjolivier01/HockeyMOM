#!/usr/bin/env bash
set -euo pipefail

# GCP-deployed webapp URL (default points at the production instance we set up).
WEBAPP_URL="${WEBAPP_URL:-https://www.jrsharks2013.org}"

# Import token (required for production; local dev typically does not require it).
# Do NOT hardcode this in the script; pass it via env.
HM_WEBAPP_IMPORT_TOKEN="${HM_WEBAPP_IMPORT_TOKEN:-}"

# League/user defaults.
LEAGUE_NAME="${LEAGUE_NAME:-Norcal}"
OWNER_EMAIL="${OWNER_EMAIL:-cjolivier01@gmail.com}"

# Game list for shift spreadsheet uploader.
SHIFT_FILE_LIST="${SHIFT_FILE_LIST:-$HOME/Videos/game_list_long.txt}"

# GCP instance info for code redeploy.
PROJECT_ID="${PROJECT_ID:-sage-courier-241217}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-hm-webapp}"
DEPLOY_ONLY=0
DROP_DB=0
DROP_DB_ONLY=0
SPREADSHEETS_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./gcp_import_webapp.sh [--deploy-only] [--drop-db | --drop-db-only] [--spreadsheets-only]

Environment:
  WEBAPP_URL              Webapp base URL (default: https://www.jrsharks2013.org)
  HM_WEBAPP_IMPORT_TOKEN  Import token (required for reset/import/upload; not needed for --deploy-only)
  LEAGUE_NAME             League name (default: Norcal)
  OWNER_EMAIL             League owner email (default: cjolivier01@gmail.com)
  SHIFT_FILE_LIST         Shift spreadsheet file list (default: ~/Videos/game_list_long.txt)
  PROJECT_ID              GCP project id (default: sage-courier-241217)
  ZONE                    GCE zone (default: us-central1-a)
  INSTANCE                GCE instance name (default: hm-webapp)

Options:
  --deploy-only           Only redeploy/restart the webapp and exit (no reset/import/upload)
  --drop-db               Drop (and recreate) the webapp MariaDB database on the GCE instance, then continue
  --drop-db-only          Drop (and recreate) the database, then exit (implies --drop-db)
  --spreadsheets-only     Seed only from shift spreadsheets (skip TimeToScore import; avoids T2S usage in parse_stats_inputs)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-only) DEPLOY_ONLY=1; shift ;;
    --drop-db) DROP_DB=1; shift ;;
    --drop-db-only) DROP_DB=1; DROP_DB_ONLY=1; shift ;;
    --spreadsheets-only|--no-time2score|--no-t2s) SPREADSHEETS_ONLY=1; shift ;;
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

if [[ "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[i] --deploy-only: skipping reset/import/upload"
  exit 0
fi

read_default_token
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
curl -sS -m 30 -f \
  -X POST "${WEBAPP_URL}/api/internal/ensure_league_owner" \
  -H "Authorization: Bearer ${HM_WEBAPP_IMPORT_TOKEN}" \
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
  ./p tools/webapp/import_time2score.py \
    --source=caha \
    --league-name="${LEAGUE_NAME}" \
    --season 0 \
    --api-url "${WEBAPP_URL}" \
    "${TOKEN_ARGS[@]}" \
    --user-email "${OWNER_EMAIL}"
fi

echo "[i] Uploading shift spreadsheets via REST"
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

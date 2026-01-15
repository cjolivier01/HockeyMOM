#!/usr/bin/env bash
set -euo pipefail

WEBAPP_URL="${WEBAPP_URL:-http://127.0.0.1:8008}"
WEB_ACCESS_KEY="${WEB_ACCESS_KEY:-}"
SHIFT_FILE_LIST="${SHIFT_FILE_LIST:-}"
LEAGUE_NAME="${LEAGUE_NAME:-CAHA}"
DEPLOY_ONLY=0
DROP_DB=0
DROP_DB_ONLY=0
SPREADSHEETS_ONLY=0
PARSE_ONLY=0
T2S_SCRAPE=0
NO_DEFAULT_USER=0
REBUILD=0
INCLUDE_SHIFTS=0
T2S_LEAGUES=(3 5 18)
T2S_LEAGUES_SET=0

GIT_USER_EMAIL="$(git config --get user.email 2>/dev/null || true)"
GIT_USER_NAME="$(git config --get user.name 2>/dev/null || true)"
OWNER_EMAIL="${OWNER_EMAIL:-${GIT_USER_EMAIL:-cjolivier01@gmail.com}}"
OWNER_NAME="${OWNER_NAME:-${GIT_USER_NAME:-$OWNER_EMAIL}}"

usage() {
  cat <<'EOF'
Usage: ./import_webapp.sh [--deploy-only] [--drop-db | --drop-db-only] [--spreadsheets-only] [--parse-only] [--shifts]

Environment:
  WEBAPP_URL        Webapp base URL (default: http://127.0.0.1:8008)
  WEB_ACCESS_KEY    Optional token args passed through to import scripts
  SHIFT_FILE_LIST   Shift spreadsheet file list (default: ~/RVideos/game_list_long.yaml if present, else ~/Videos/game_list_long.yaml, else .txt)
  LEAGUE_NAME       League name to import into (default: CAHA)

Options:
  --deploy-only     Only redeploy/restart the webapp and exit (no reset/import/upload)
  --drop-db         Drop (and recreate) the local webapp MariaDB database, then continue
  --drop-db-only    Drop (and recreate) the database, then exit (implies --drop-db)
  --no-default-user Skip auto-creating a default webapp user from git config (user.email/user.name)
  --t2s-league ID   Only import these TimeToScore league ids (repeatable or comma-separated; default: 3,5,18)
  --rebuild         Reset (delete) existing league hockey data before importing (destructive)
  --spreadsheets-only  Seed only from shift spreadsheets (skip TimeToScore import; scrape only per-game T2S lookups)
  --parse-only      Only run scripts/parse_stats_inputs.py upload (skip reset + TimeToScore import); forces --webapp-replace
  --scrape          Force re-scraping TimeToScore game pages (overrides local cache) when running the T2S import step
  --shifts          Include TOI/Shifts stats from shift spreadsheets (adds TOI/Shifts columns in webapp tables)
EOF
}

add_t2s_leagues() {
  local raw="$1"
  if [[ "${T2S_LEAGUES_SET}" == "0" ]]; then
    T2S_LEAGUES=()
    T2S_LEAGUES_SET=1
  fi

  local -a parts=()
  IFS=',' read -ra parts <<<"${raw}"
  local p
  for p in "${parts[@]}"; do
    p="${p//[[:space:]]/}"
    if [[ -z "${p}" || ! "${p}" =~ ^[0-9]+$ ]]; then
      echo "[!] Invalid --t2s-league value: ${raw}" >&2
      exit 2
    fi
    T2S_LEAGUES+=( "${p}" )
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-only) DEPLOY_ONLY=1; shift ;;
    --drop-db) DROP_DB=1; shift ;;
    --drop-db-only) DROP_DB=1; DROP_DB_ONLY=1; shift ;;
    --no-default-user) NO_DEFAULT_USER=1; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --shifts) INCLUDE_SHIFTS=1; shift ;;
    --t2s-league=*)
      T2S_LEAGUE_RAW="${1#*=}"
      shift
      add_t2s_leagues "${T2S_LEAGUE_RAW}"
      ;;
    --t2s-league)
      shift
      if [[ $# -lt 1 ]]; then
        echo "[!] --t2s-league requires an integer argument" >&2
        exit 2
      fi
      T2S_LEAGUE_RAW="${1}"
      shift
      add_t2s_leagues "${T2S_LEAGUE_RAW}"
      ;;
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

# ./p tools/webapp/scripts/import_time2score.py --source=caha --league-name=CAHA --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com
# ./p tools/webapp/scripts/import_time2score.py --source=caha --league-name=CAHA --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com --division 6:0

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

echo "[i] Ensuring league '${LEAGUE_NAME}' is shared and owned by ${OWNER_EMAIL}"
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
LEAGUE_OWNER_PAYLOAD="$(
  LEAGUE_NAME="${LEAGUE_NAME}" OWNER_EMAIL="${OWNER_EMAIL}" OWNER_NAME="${OWNER_NAME}" python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "league_name": os.environ.get("LEAGUE_NAME") or "",
            "owner_email": os.environ.get("OWNER_EMAIL") or "",
            "owner_name": os.environ.get("OWNER_NAME") or "",
            "shared": True,
        }
    )
)
PY
)"
if ! curl -sS -m 30 -f -X POST "${HDRS[@]}" --data "${LEAGUE_OWNER_PAYLOAD}" "${WEBAPP_URL}/api/internal/ensure_league_owner" >/dev/null; then
  echo "[!] Failed to ensure league ownership via ${WEBAPP_URL}/api/internal/ensure_league_owner" >&2
  echo "    Hint: if WEBAPP_URL points at nginx (port 80), you may need an import token; try WEBAPP_URL=http://127.0.0.1:8008" >&2
fi

if [[ "${PARSE_ONLY}" == "1" ]]; then
  if [[ "${REBUILD}" == "1" ]]; then
    echo "[!] --parse-only ignores --rebuild (reset requires re-importing TimeToScore games first)" >&2
  fi
  echo "[i] --parse-only: skipping league reset and TimeToScore import"
else
  if [[ "${REBUILD}" == "1" ]]; then
    echo "[i] Resetting league data (REST)"
    RESET_PAYLOAD="$(LEAGUE_NAME="${LEAGUE_NAME}" OWNER_EMAIL="${OWNER_EMAIL}" python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "league_name": os.environ.get("LEAGUE_NAME") or "",
            "owner_email": os.environ.get("OWNER_EMAIL") or "",
        }
    )
)
PY
)"
    curl -sS -m 300 -f -X POST "${HDRS[@]}" --data "${RESET_PAYLOAD}" "${WEBAPP_URL}/api/internal/reset_league_data" >/dev/null
  else
    echo "[i] Skipping league reset (incremental import). Use --rebuild to wipe existing league data."
  fi

  if [[ "${SPREADSHEETS_ONLY}" == "1" ]]; then
    echo "[i] --spreadsheets-only: skipping TimeToScore import"
  else
    echo "[i] Importing TimeToScore (caha league=${T2S_LEAGUES[*]} -> ${LEAGUE_NAME})"
    # python3 tools/webapp/scripts/import_time2score.py --source=caha --league-name=CAHA --season 0 --api-url "${WEBAPP_URL}" ${WEB_ACCESS_KEY} --user-email cjolivier01@gmail.com --division 6:0
    T2S_ARGS=()
    if [[ "${T2S_SCRAPE}" == "1" ]]; then
      T2S_ARGS+=( "--scrape" )
    fi
    for T2S_LEAGUE_ID in "${T2S_LEAGUES[@]}"; do
      echo "[i]   - CAHA TimeToScore league=${T2S_LEAGUE_ID}"
      ./p tools/webapp/scripts/import_time2score.py \
        --source=caha \
        --t2s-league-id "${T2S_LEAGUE_ID}" \
        --league-name="${LEAGUE_NAME}" \
        --season 0 \
        --api-url "${WEBAPP_URL}" \
        "${T2S_ARGS[@]}" \
        ${WEB_ACCESS_KEY} \
        --user-email "${OWNER_EMAIL}"
    done

    echo "[i] Importing CAHA tier schedule.pl (AA/AAA Major/Minor/15O AAA) -> ${LEAGUE_NAME}"
    ./p tools/webapp/scripts/import_caha_schedule_pl.py \
      --league-name="${LEAGUE_NAME}" \
      --api-url "${WEBAPP_URL}" \
      ${WEB_ACCESS_KEY} \
      --owner-email "${OWNER_EMAIL}"
  fi
fi

#echo "[i] Uploading shift spreadsheets"
if [[ -z "${SHIFT_FILE_LIST}" ]]; then
  if [[ -f "$HOME/RVideos/game_list_long.yaml" ]]; then
    SHIFT_FILE_LIST="$HOME/RVideos/game_list_long.yaml"
  elif [[ -f "$HOME/RVideos/game_list_long.yml" ]]; then
    SHIFT_FILE_LIST="$HOME/RVideos/game_list_long.yml"
  elif [[ -f "$HOME/RVideos/game_list_long.txt" ]]; then
    SHIFT_FILE_LIST="$HOME/RVideos/game_list_long.txt"
  elif [[ -f "$HOME/Videos/game_list_long.yaml" ]]; then
    SHIFT_FILE_LIST="$HOME/Videos/game_list_long.yaml"
  elif [[ -f "$HOME/Videos/game_list_long.yml" ]]; then
    SHIFT_FILE_LIST="$HOME/Videos/game_list_long.yml"
  else
    SHIFT_FILE_LIST="$HOME/Videos/game_list_long.txt"
  fi
fi
if [[ ! -f "${SHIFT_FILE_LIST}" ]]; then
  echo "[!] SHIFT_FILE_LIST not found: ${SHIFT_FILE_LIST}" >&2
  echo "    Set it explicitly, e.g.: export SHIFT_FILE_LIST=~/RVideos/game_list_long.yaml" >&2
  exit 2
fi
SPREADSHEET_ARGS=()
if [[ "${SPREADSHEETS_ONLY}" == "1" ]]; then
  SPREADSHEET_ARGS+=( "--t2s-scrape-only" )
fi
if [[ "${PARSE_ONLY}" == "1" ]]; then
  SPREADSHEET_ARGS+=( "--webapp-replace" )
fi
if [[ "${INCLUDE_SHIFTS}" == "1" ]]; then
  SPREADSHEET_ARGS+=( "--shifts" )
fi
./p scripts/parse_stats_inputs.py \
  --file-list "${SHIFT_FILE_LIST}" \
  --no-scripts \
  --upload-webapp \
  --webapp-url="${WEBAPP_URL}" \
  "${SPREADSHEET_ARGS[@]}" \
  ${WEB_ACCESS_KEY} \
  --webapp-owner-email "${OWNER_EMAIL}" \
  --webapp-league-name="${LEAGUE_NAME}"

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
RATINGS_PAYLOAD="$(LEAGUE_NAME="${LEAGUE_NAME}" python3 - <<'PY'
import json

import os

print(json.dumps({"league_name": os.environ.get("LEAGUE_NAME") or "CAHA"}))
PY
)"
curl -sS -m 300 -f -X POST "${HDRS[@]}" --data "${RATINGS_PAYLOAD}" "${WEBAPP_URL}/api/internal/recalc_div_ratings" >/dev/null

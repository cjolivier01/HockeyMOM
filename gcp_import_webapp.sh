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

is_local_url() {
  case "${WEBAPP_URL}" in
    http://127.0.0.1:*|http://localhost:*|https://127.0.0.1:*|https://localhost:* ) return 0 ;;
    *) return 1 ;;
  esac
}

TOKEN_ARGS=()
RESET_TOKEN_ARGS=()
if [[ -n "${HM_WEBAPP_IMPORT_TOKEN}" ]]; then
  TOKEN_ARGS+=( "--import-token=${HM_WEBAPP_IMPORT_TOKEN}" )
  RESET_TOKEN_ARGS+=( "--webapp-token=${HM_WEBAPP_IMPORT_TOKEN}" )
else
  if ! is_local_url; then
    echo "[!] HM_WEBAPP_IMPORT_TOKEN is not set. Production imports will be rejected." >&2
    echo "    Set it like: export HM_WEBAPP_IMPORT_TOKEN='...'" >&2
    exit 2
  fi
fi

echo "[i] Redeploying GCP webapp (code-only)"
python3 tools/webapp/redeploy_gcp.py --project "${PROJECT_ID}" --zone "${ZONE}" --instance "${INSTANCE}"

echo "[i] Verifying webapp endpoint"
curl -sS -o /dev/null -m 15 -f -I "${WEBAPP_URL}/" >/dev/null

echo "[i] Resetting league data (REST)"
./p tools/webapp/reset_league_data.py \
  --force \
  --webapp-url "${WEBAPP_URL}" \
  "${RESET_TOKEN_ARGS[@]}" \
  --webapp-owner-email "${OWNER_EMAIL}" \
  --league-name "${LEAGUE_NAME}"

echo "[i] Importing TimeToScore (caha -> ${LEAGUE_NAME}) via REST"
./p tools/webapp/import_time2score.py \
  --source=caha \
  --league-name="${LEAGUE_NAME}" \
  --season 0 \
  --api-url "${WEBAPP_URL}" \
  "${TOKEN_ARGS[@]}" \
  --user-email "${OWNER_EMAIL}"

echo "[i] Uploading shift spreadsheets via REST"
./p scripts/parse_shift_spreadsheet.py \
  --file-list "${SHIFT_FILE_LIST}" \
  --shifts \
  --upload-webapp \
  --webapp-url="${WEBAPP_URL}" \
  "${TOKEN_ARGS[@]}" \
  --webapp-owner-email "${OWNER_EMAIL}" \
  --webapp-league-name "${LEAGUE_NAME}"


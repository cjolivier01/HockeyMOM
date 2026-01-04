#!/usr/bin/env bash
set -euo pipefail

WEBAPP_URL="${WEBAPP_URL:-http://127.0.0.1:8008}"
WEB_ACCESS_KEY="${WEB_ACCESS_KEY:-}"
DEPLOY_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./import_webapp.sh [--deploy-only]

Environment:
  WEBAPP_URL        Webapp base URL (default: http://127.0.0.1:8008)
  WEB_ACCESS_KEY    Optional token args passed through to import scripts

Options:
  --deploy-only     Only redeploy/restart the webapp and exit (no reset/import/upload)
EOF
}

for arg in "$@"; do
  case "$arg" in
    --deploy-only) DEPLOY_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[!] Unknown arg: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# If your server requires an import token, set one of:
#   export HM_WEBAPP_IMPORT_TOKEN="...secret..."
#   export WEB_ACCESS_KEY="--import-token=...secret..."
# (Local dev defaults typically do not require a token.)

# ./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com
# ./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --config /opt/hm-webapp/app/config.json --user-email cjolivier01@gmail.com --division 6:0

# PROJECT_ID="sage-courier-241217"
# REDEPLOY_WEB="python3 tools/webapp/redeploy_gcp.py --project ${PROJECT_ID} --zone us-central1-a --instance hm-webapp"
# $REDEPLOY_WEB

echo "[i] Redeploying local webapp"
./tools/webapp/redeploy_local.sh

echo "[i] Verifying webapp endpoint"
curl -sS -o /dev/null -m 10 -f -I "${WEBAPP_URL}/teams" >/dev/null

if [[ "${DEPLOY_ONLY}" == "1" ]]; then
  echo "[i] --deploy-only: skipping reset/import/upload"
  exit 0
fi

echo "[i] Resetting league data"
./p tools/webapp/reset_league_data.py --force

echo "[i] Importing TimeToScore (caha -> Norcal)"
# ./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --api-url "${WEBAPP_URL}" ${WEB_ACCESS_KEY} --user-email cjolivier01@gmail.com --division 6:0
./p tools/webapp/import_time2score.py --source=caha --league-name=Norcal --season 0 --api-url "${WEBAPP_URL}" ${WEB_ACCESS_KEY} --user-email cjolivier01@gmail.com

echo "[i] Uploading shift spreadsheets"
./p scripts/parse_shift_spreadsheet.py --file-list ~/Videos/game_list_long.txt --shifts --upload-webapp --webapp-url="${WEBAPP_URL}" ${WEB_ACCESS_KEY} --webapp-owner-email cjolivier01@gmail.com --webapp-league-name=Norcal

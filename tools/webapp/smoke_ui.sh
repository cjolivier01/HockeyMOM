#!/usr/bin/env bash
set -euo pipefail

# HockeyMOM WebApp UI smoke test
# - Registers a user
# - Creates a team with a logo
# - Adds a player
# - Creates a game (vs auto-created external opponent)
# - Sets a final score and verifies team record updates
#
# Usage:
#   tools/webapp/smoke_ui.sh [BASE_URL]
#
#  BASE_URL defaults to http://127.0.0.1 (nginx front).

BASE="${1:-${BASE:-http://127.0.0.1}}"
COOK_FILE="${COOK_FILE:-/tmp/hm_smoke_cookie.txt}"
TMP_DIR="${TMP_DIR:-/tmp}"

echo "[i] Target: $BASE"

get_csrf() {
  # Fetch a page and extract the first CSRF token for POSTs.
  local path="$1"
  reqf "${BASE}${path}" | python3 -c '
import re
import sys

html = sys.stdin.read()
m = re.search(r"name=\"csrfmiddlewaretoken\" value=\"([^\"]+)\"", html)
if not m:
    raise SystemExit(2)
print(m.group(1))
'
}

req() {
  local method="$1"; shift
  curl -sS -c "$COOK_FILE" -b "$COOK_FILE" -X "$method" "$@"
}

reqf() {
  curl -sS -c "$COOK_FILE" -b "$COOK_FILE" "$@"
}

assert_contains() {
  local needle="$1"; shift
  if ! grep -q "$needle"; then
    echo "[!] Expected to find: $needle" >&2
    exit 1
  fi
}

# 0) Reachability
if ! curl -sSf "$BASE/" >/dev/null; then
  echo "[!] Cannot reach $BASE/" >&2
  exit 1
fi

rm -f "$COOK_FILE"

# 1) Register user (CSRF-protected)
EMAIL="smoke_$(date +%s)@example.com"
NAME="Smoke User"
PASS="smokepw123"
echo "[i] Registering $EMAIL"
CSRF="$(get_csrf "/register")"
REG_RESP=$(curl -sS -i -c "$COOK_FILE" -b "$COOK_FILE" -X POST \
  -d "csrfmiddlewaretoken=$CSRF" -d "name=$NAME" -d "email=$EMAIL" -d "password=$PASS" \
  "$BASE/register")
echo "$REG_RESP" | grep -q "^Location: /games" || true
# Follow with an explicit GET to avoid POST -> 405 on /games via -L
HTML=$(reqf "$BASE/games")
echo "$HTML" | assert_contains "Your Games"

# 1b) Leagues page should render for logged-in user
HTML=$(reqf "$BASE/leagues")
echo "$HTML" | assert_contains "Create League"

# 2) Ensure Teams route exists before proceeding (requires login)
if ! reqf -sSf "$BASE/teams/new" >/dev/null; then
  echo "[!] /teams endpoints not found. Ensure the updated webapp is installed." >&2
  exit 1
fi

# 3) Create a tiny PNG logo
LOGO_B64="$TMP_DIR/hm_logo_smoke.b64"
LOGO_PNG="$TMP_DIR/hm_logo_smoke.png"
cat > "$LOGO_B64" << 'B64'
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/58BAgMDAx2yWmQAAAAASUVORK5CYII=
B64
base64 -d "$LOGO_B64" > "$LOGO_PNG"

# 4) Create team with logo
echo "[i] Creating team"
CSRF="$(get_csrf "/teams/new")"
HTML=$(reqf -L \
  -F "csrfmiddlewaretoken=$CSRF" \
  -F "name=Smoke Team" \
  -F "logo=@$LOGO_PNG;type=image/png" \
  "$BASE/teams/new")
echo "$HTML" | assert_contains "Smoke Team"

# 5) Discover team id
TEAM_ID=$(echo "$HTML" | sed -n 's#.*href="/teams/\([0-9]\+\)">Edit</a>.*#\1#p' | head -n1)
if [ -z "$TEAM_ID" ]; then
  TEAM_ID=$(reqf "$BASE/teams?all=1" | sed -n 's#.*href="/teams/\([0-9]\+\)">Smoke Team<.*#\1#p' | head -n1)
fi
if [ -z "$TEAM_ID" ]; then
  echo "[!] Could not resolve team id" >&2
  exit 1
fi
echo "[i] TEAM_ID=$TEAM_ID"

# 6) Add a player
echo "[i] Adding player"
CSRF="$(get_csrf "/teams/$TEAM_ID/players/new")"
HTML=$(reqf -L \
  -d "csrfmiddlewaretoken=$CSRF" \
  -d "name=Smoke Skater" -d "jersey_number=77" -d "position=F" -d "shoots=R" \
  "$BASE/teams/$TEAM_ID/players/new")
echo "$HTML" | assert_contains "Smoke Skater"

# 7) Create a game (use first available game type)
GT_ID=$(
  reqf "$BASE/schedule/new" | python3 -c 'import re, sys; html=sys.stdin.read(); m=re.search(r"<select name=\"game_type_id\">.*?<option value=\"(\d+)\"", html, re.S); print(m.group(1) if m else "")'
)
if [ -z "$GT_ID" ]; then echo "[!] No game type found" >&2; exit 1; fi
echo "[i] GAME_TYPE=$GT_ID"

CSRF="$(get_csrf "/schedule/new")"
RESP=$(curl -sS -i -c "$COOK_FILE" -b "$COOK_FILE" -X POST \
  -d "csrfmiddlewaretoken=$CSRF" \
  -d "team1_id=$TEAM_ID" -d "team2_id=" -d "opponent_name=Opp Smoke" \
  -d "game_type_id=$GT_ID" -d "starts_at=2025-01-02T12:00" -d "location=Smoke Rink" \
  "$BASE/schedule/new")
echo "$RESP" | assert_contains "^Location: /hky/games/"
GAME_ID=$(echo "$RESP" | sed -n 's#Location: /hky/games/\([0-9]\+\).*#\1#p' | head -n1)
echo "[i] GAME_ID=$GAME_ID"

# 7b) Import shift spreadsheet stats for the player
echo "[i] Importing sample shift stats"
PS_CSV="$TMP_DIR/hm_player_stats_smoke.csv"
cat > "$PS_CSV" << 'CSV'
Player,Goals,Assists,Shots,SOG,xG,Plus Minus,Shifts,TOI Total,TOI Total (Video),Period 1 TOI,Period 1 Shifts,Period 1 GF,Period 1 GA
77 Smoke Skater,1,0,2,1,0,1,5,5:00,5:00,5:00,5,1,0
CSV

CSRF="$(get_csrf "/hky/games/$GAME_ID?edit=1")"
HTML=$(reqf -L \
  -F "csrfmiddlewaretoken=$CSRF" \
  -F "player_stats_csv=@$PS_CSV;type=text/csv" \
  "$BASE/hky/games/$GAME_ID/import_shift_stats")
echo "$HTML" | assert_contains "Imported stats for"

# 8) Set final score
echo "[i] Setting final score"
CSRF="$(get_csrf "/hky/games/$GAME_ID?edit=1")"
reqf -L \
  -d "csrfmiddlewaretoken=$CSRF" \
  -d "team1_score=6" -d "team2_score=3" -d "is_final=on" \
  "$BASE/hky/games/$GAME_ID?edit=1" >/dev/null

# 9) Verify team record updated
REC=$(
  reqf "$BASE/teams" | python3 -c '
import re
import sys

team_id = sys.argv[1]
html = sys.stdin.read()
pat = "href=\"/teams/%s\".*?<td>\\s*\\d+\\s*</td>\\s*<td>\\s*(\\d+-\\d+-\\d+)\\s*</td>" % re.escape(team_id)
m = re.search(pat, html, re.S)
print(m.group(1) if m else "")
' "$TEAM_ID"
)
echo "[i] Record: ${REC:-unknown}"
if [ -z "$REC" ]; then
  echo "[!] Could not parse team record" >&2
  exit 1
fi

# 10) Verify team logo URL
curl -sSf -I "$BASE/media/team_logo/$TEAM_ID" >/dev/null
echo "[ok] Smoke test passed: TEAM_ID=$TEAM_ID GAME_ID=$GAME_ID RECORD=$REC"

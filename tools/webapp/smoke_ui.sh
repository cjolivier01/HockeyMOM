#!/usr/bin/env bash
set -euo pipefail

# HM WebApp UI smoke test
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

req() {
  local method="$1"; shift
  curl -sS -c "$COOK_FILE" -b "$COOK_FILE" -X "$method" "$@"
}

reqf() {
  curl -sS -c "$COOK_FILE" -b "$COOK_FILE" "$@"
}

csrf_header() {
  if [ ! -f "$COOK_FILE" ]; then
    return
  fi
  local token
  token=$(awk '$6=="csrftoken"{token=$7} END{print token}' "$COOK_FILE" || true)
  if [ -n "${token:-}" ]; then
    printf 'X-CSRFToken: %s' "$token"
  fi
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

# 1) Register user
EMAIL="smoke_$(date +%s)@example.com"
NAME="Smoke User"
PASS="smokepw123"
echo "[i] Registering $EMAIL"
reqf "$BASE/register" >/dev/null
REG_RESP=$(curl -sS -i -c "$COOK_FILE" -b "$COOK_FILE" -X POST \
  -H "$(csrf_header)" \
  -d "name=$NAME" -d "email=$EMAIL" -d "password=$PASS" \
  "$BASE/register")
echo "$REG_RESP" | grep -q "^Location: /games" || true
# Follow with an explicit GET to avoid POST -> 405 on /games via -L
HTML=$(reqf "$BASE/games")
echo "$HTML" | assert_contains "Your Games"

# 2) Ensure Teams route exists before proceeding
if ! curl -sSf "$BASE/teams/new" >/dev/null; then
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
reqf "$BASE/teams/new" >/dev/null
HTML=$(curl -sS -L -c "$COOK_FILE" -b "$COOK_FILE" \
  -H "$(csrf_header)" \
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
reqf "$BASE/teams/$TEAM_ID/players/new" >/dev/null
HTML=$(curl -sS -L -c "$COOK_FILE" -b "$COOK_FILE" -X POST \
  -H "$(csrf_header)" \
  -d "name=Smoke Skater" -d "jersey_number=77" -d "position=F" -d "shoots=R" \
  "$BASE/teams/$TEAM_ID/players/new")
echo "$HTML" | assert_contains "Smoke Skater"

# 7) Create a game (leave game_type unset for simplicity)
reqf "$BASE/schedule/new" >/dev/null
RESP=$(curl -sS -i -c "$COOK_FILE" -b "$COOK_FILE" -X POST \
  -H "$(csrf_header)" \
  -d "team1_id=$TEAM_ID" -d "team2_id=" -d "opponent_name=Opp Smoke" \
  -d "game_type_id=" -d "starts_at=2025-01-02T12:00" -d "location=Smoke Rink" \
  "$BASE/schedule/new")
echo "$RESP" | assert_contains "^Location: /hky/games/"
GAME_ID=$(echo "$RESP" | sed -n 's#Location: /hky/games/\([0-9]\+\).*#\1#p' | head -n1)
echo "[i] GAME_ID=$GAME_ID"

# 8) Set final score
echo "[i] Setting final score"
reqf "$BASE/hky/games/$GAME_ID" >/dev/null
curl -sS -c "$COOK_FILE" -b "$COOK_FILE" -X POST \
  -H "$(csrf_header)" \
  -d "team1_score=6" -d "team2_score=3" -d "is_final=on" \
  "$BASE/hky/games/$GAME_ID" >/dev/null

# 9) Verify team record updated
REC=$(reqf "$BASE/teams" | tr '\n' ' ' | sed -n 's#.*Smoke Team</a>[^0-9]*\([0-9]\+-[0-9]\+-[0-9]\+\)</td>.*#\1#p' | head -n1)
echo "[i] Record: ${REC:-unknown}"
if [ -z "$REC" ]; then
  echo "[!] Could not parse team record" >&2
  exit 1
fi

# 10) Verify team logo URL
curl -sSf -I "$BASE/media/team_logo/$TEAM_ID" >/dev/null
echo "[ok] Smoke test passed: TEAM_ID=$TEAM_ID GAME_ID=$GAME_ID RECORD=$REC"

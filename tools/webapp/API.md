# HockeyMOM Webapp REST API

The webapp serves both:
- a browser UI (Django views + templates), and
- JSON REST endpoints used by import/reset tooling.

This document focuses on the JSON endpoints under `/api/...`.

## Authentication model

### UI/session endpoints
- Browser endpoints use Django sessions (cookie-based) after `/login` or `/register`.
- UI form POSTs use CSRF (standard Django behavior).

### Import/internal endpoints (`/api/import/*`, `/api/internal/*`)
These endpoints are intended for automation (import/reset/deploy helpers).

They use **import-token auth** when configured:
- Configure a token via either:
  - environment: `HM_WEBAPP_IMPORT_TOKEN`, or
  - `config.json`: `"import_token": "..."` (deployed installs).
- Send the token via either:
  - `Authorization: Bearer <token>` (preferred), or
  - `X-HM-Import-Token: <token>`.

If **no token is configured**, these endpoints only allow **local direct** requests:
- Requests with `X-Forwarded-For` are rejected (`403 import_token_required`).
- Requests not from `127.0.0.1`/`::1` are rejected (`403 import_token_required`).

Implication: for local dev without a token, call import/internal endpoints via gunicorn (example: `http://127.0.0.1:8008`),
not through nginx on port 80.

## Conventions

- Requests: JSON body, `Content-Type: application/json`
- Responses: JSON with `{"ok": true, ...}` or `{"ok": false, "error": "..."}`.

## Import API (`/api/import/hockey/*`)

All endpoints in this section require the import-token rules above.

### `POST /api/import/hockey/ensure_league`
Create or update a league and ensure an owner user exists.

Body:
- `league_name` (string, default `"CAHA"`)
- `owner_email` (string, default `"caha-import@hockeymom.local"`)
- `owner_name` (string, default `"CAHA Import"`)
- `shared` (bool, optional)
- `source` (string, optional)
- `external_key` (string, optional)

Response:
- `league_id` (int)
- `owner_user_id` (int)

### `POST /api/import/hockey/teams`
Ensure a league exists, then import/update teams and their league mappings.

Body:
- `league_name`, `owner_email`, `owner_name`, `shared`, `source`, `external_key` (same semantics as `ensure_league`)
- `replace` (bool, optional): default behavior for team assets (logos)
- `teams` (list, required): each entry is an object containing:
  - `name` (string, required)
  - `division_name` (string, optional)
  - `division_id` (int, optional)
  - `conference_id` (int, optional)
  - team logo fields (optional; any of these work):
    - `logo_b64` / `team_logo_b64`
    - `logo_content_type` / `team_logo_content_type`
    - `logo_url` / `team_logo_url`
  - `replace` (bool, optional): per-team override

Response:
- `league_id`, `owner_user_id`
- `imported` (int)
- `results` (list of `{team_id, name}`)

### `POST /api/import/hockey/game`
Upsert a single game (and its roster/stats) into a league.

Body:
- `league_name`, `owner_email`, `owner_name`, `shared`, `source`, `external_key`
- `replace` (bool, optional): whether to overwrite existing values in some cases
- `game` (object, required):
  - teams: `home_name`, `away_name` (required)
  - optional metadata: `starts_at`, `location`
  - optional scores: `home_score`, `away_score`
  - optional division metadata: `division_name`, `division_id`, `conference_id`, plus `home_*`/`away_*` overrides
  - optional TimeToScore metadata: `timetoscore_game_id`, `season_id`, `timetoscore_type`/`game_type_name`
  - optional rosters: `home_roster` / `away_roster` (list of `{name, number, position}`)
  - optional player stats: `player_stats` (list; schema depends on importer)

Response includes (at least):
- `game_id`, `league_id`, `owner_user_id`, `team1_id`, `team2_id`

### `POST /api/import/hockey/games_batch`
Batch variant of `game` import.

Body:
- same top-level fields as `game`
- `games` (list, required): each entry has the same shape as the `game` object above.

Response:
- `imported` (int) and a `results` list with per-game identifiers.

### `POST /api/import/hockey/shift_package`
Upload a “shift package” for a game (shift stats CSVs and related metadata). This endpoint is typically called by
`scripts/parse_stats_inputs.py --upload-webapp` rather than by hand.

Body highlights (many optional fields exist):
- `game_id` (int, optional) or `timetoscore_game_id` (int, optional) or `external_game_key` (string, optional)
- `league_id` (int, optional) or `league_name` (string, optional) (used when creating external games)
- `owner_email` (string, optional; used when creating external games)
- `team_side` (`"home"` or `"away"`, optional; used to interpret Goals For/Against)
- `replace` (bool, optional)
- `create_missing_players` (bool, optional)
- CSV payloads (strings): `events_csv`, `player_stats_csv`, plus shift-related CSVs:
  - `shift_rows_csv` (optional): per-player shift intervals (used to derive TOI/Shifts at runtime and optionally render on/off-ice markers)
  - `game_stats_csv` (optional): only used to fill missing final scores
- Shift row behavior:
  - `shift_rows_csv` requires `team_side` and will be stored in `hky_game_shift_rows` (separate from `hky_game_event_rows`).
  - When `shift_rows_csv` is present, stored TOI/Shifts fields in `PlayerStat` are cleared and are expected to be derived at runtime.
  - `replace_shift_rows` (bool, optional): if true, existing shift rows for the game+team_side are deleted before importing new ones.

Response:
- `game_id`, plus import counts and any “unmatched” player names when relevant.

## Internal API (`/api/internal/*`)

All endpoints in this section require the import-token rules above.

### `POST /api/internal/reset_league_data`
Reset (delete) league data for a specific owner+league.

Body:
- `league_name` (string, required)
- `owner_email` (string, required)

Response:
- `league_id`
- `stats` (object of counts)

### `POST /api/internal/ensure_league_owner`
Ensure an owner user exists and is the owner of the league (also ensures owner membership).

Body:
- `league_name` (string, required)
- `owner_email` (string, required)
- `owner_name` (string, optional)
- `shared` (bool, optional)

Response:
- `league_id`
- `owner_user_id`

### `POST /api/internal/ensure_user`
Idempotently create a user if missing (used by install/import scripts).

Body:
- `email` (string, required)
- `name` (string, optional; default: `email`)
- `password` (string, optional; default: `"password"`)

Response:
- `user_id`
- `created` (bool)

### `POST /api/internal/recalc_div_ratings`
Recompute and persist MyHockeyRankings-like Ratings for teams in a league (stored on `league_teams.*mhr_*`).

Body (choose one):
- `league_id` (int), or
- `league_name` (string), or
- omit both to recompute for **all** leagues.

Optional tuning:
- `max_goal_diff` (int, default `7`)
- `min_games` (int, default `2`)

Response:
- `league_ids` (list of ints) on success

This is invoked automatically by `./import_webapp.sh` and `./gcp_import_webapp.sh` after shift spreadsheet upload.

### `POST /api/internal/apply_event_corrections`
Apply idempotent event corrections (suppression + upsert) for one or more games.

Body:
- `corrections` (list, required): each entry identifies a game and includes optional operations:
  - game identifier (choose one):
    - `game_id` (int), or
    - `timetoscore_game_id` (int)
  - `suppress` (list, optional): list of event specs to suppress (prevent re-import from re-adding)
  - `upsert` (list, optional): list of event specs to insert/update
- `create_missing_players` (bool, optional): allow creating missing players when upserting

Event spec fields (subset; best-effort):
- `event_type` (string, required)
- `period` (int, optional)
- `game_time` (string, optional) or `game_seconds` (int, optional)
- `team_side` (`Home`/`Away`, optional)
- `jersey` / `player` / `attributed_jerseys` (optional; used for event identity + player mapping)
- `details` (optional; used for penalty disambiguation)

Response:
- `stats`: counts of suppressed/upserted actions

## User/analytics API

### `POST /api/user/video_clip_len`
Set the logged-in user’s preferred video clip length.

Auth: requires a logged-in session cookie.

Body:
- `clip_len_s` (int): one of `15, 20, 30, 45, 60, 90`

### `GET /api/leagues/<league_id>/page_views`
Get page-view counts for a league (owner-only).

Query params:
- `kind` (string)
- `entity_id` (int, optional)

## Events API

### `GET /api/hky/games/<game_id>/events`
Query normalized per-row game events (backed by `hky_event_types` + `hky_game_event_rows`).

Auth: requires a logged-in session and access to the game (owner, or viewed through the currently selected league).

Query params (all optional):
- `player_id` (int): filter to events attributed to a specific player
- `event_type` (string): filter by event type (matches normalized key; e.g. `Goal`, `ExpectedGoal`, `Penalty Expired`)
- `period` (int)
- `limit` (int, default `2000`, max `5000`)

Response:
- `events` (list): each event includes `event_type`, `period`, `game_time`, `video_time`, `details`, `source`, plus ids and raw columns

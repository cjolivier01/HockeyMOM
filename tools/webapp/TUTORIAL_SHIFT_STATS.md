# HockeyMOM WebApp: Import and View Shift Spreadsheet Stats (Event-Driven)

This tutorial shows how to take the `stats/` outputs produced by `scripts/parse_stats_inputs.py` and import them into the HockeyMOM webapp so you can view per-game events and aggregated player/team stats.

The webapp no longer imports `player_stats.csv` / `game_stats.csv` into DB tables; all scoring and aggregates are computed at runtime from imported event/shift rows.

## 1) Generate `stats/all_events_summary.csv` and `stats/shift_rows.csv`

Run `scripts/parse_stats_inputs.py` on your game sheet (or game directory) and confirm that the output directory contains:

- `stats/all_events_summary.csv` (the event log uploaded to the webapp as `events_csv`)
- `stats/shift_rows.csv` (per-player shift intervals uploaded as `shift_rows_csv`)

Notes:
- These files are used for webapp ingestion; `player_stats.csv` may still be written for convenience, but it is not imported.

## 2) Create teams and players in the webapp (UI)

1. Open the webapp and register/login.
2. Go to `Teams` → `New team` and create your team.
3. Click into your team → `Add player` and add each player.

Important:
- Set each player’s `Jersey #` to match the jersey numbers used in your shift spreadsheets/events.
- Matching is done by jersey number first; name matching is a fallback.

## 3) Upload to the webapp

Recommended options:
- End-to-end local import: `./import_webapp.sh --spreadsheets-only` (uploads `events_csv` + `shift_rows_csv`).
- Direct upload from the parser: run `scripts/parse_stats_inputs.py` with `--upload-webapp ...` for your file-list YAML or game directory.

What happens:
- The webapp upserts `hky_game_event_rows` from `events_csv` and `hky_game_shift_rows` from `shift_rows_csv`.
- Player/game/team stats tables are computed from those event/shift tables at runtime (no `player_stats` table).

## 4) Verify the stats

- Game page:
  - **Game Events** shows the imported event rows.
  - Player stats tables are derived from events (and shift rows when enabled for the league).
- Team page:
  - The Players table aggregates across eligible games and is derived from events/shift rows.

## Troubleshooting

- “Unmatched players”:
  - Ensure each player exists on the team in the webapp and has the correct jersey number.
  - Re-run the import after fixing jersey numbers.

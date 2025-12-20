# HockeyMOM WebApp: Import and View Shift Spreadsheet Stats

This tutorial shows how to take the `stats/` outputs produced by `scripts/parse_shift_spreadsheet.py` (e.g. from `~/Videos/stockton-r3/stat` and `~/Videos/sharks-12-1-r2/stats`) and import them into the HockeyMOM WebApp so you can view them per game, per player, and aggregated on the team page.

## 1) Generate `stats/player_stats.csv` + `stats/game_stats.csv`

Run `scripts/parse_shift_spreadsheet.py` on your game sheet (or game directory) and confirm that the output directory contains:

- `stats/player_stats.csv`
- `stats/game_stats.csv` (optional but recommended)

Notes:
- If you want TOI/shifts columns in the outputs, run with `--shifts` (the script only writes TOI/shift columns when enabled).
- The webapp importer expects the CSV headers produced by this script (e.g. `Player`, `Goals`, `Assists`, `TOI Total`, `SOG`, `xG`, `Controlled Entry For (On-Ice)`, etc.).

## 2) Create teams and players in the WebApp (UI)

1. Open the webapp and register/login.
2. Go to `Teams` → `New team` and create your team (e.g. `Sharks`).
3. Click into your team → `Add player` and add each player.

Important:
- Set each player’s `Jersey #` to match the jersey numbers shown in `player_stats.csv` (`Player` column starts with the jersey number).
- Import matching is done by jersey number first; name matching is only a fallback.

## 3) Create the game (UI)

1. Go to `Schedule` → `Add game`.
2. Select your team as `Team 1` (or `Team 2`).
3. If you don’t have the opponent team created, leave the other team blank and enter `Opponent name` to auto-create an external opponent.
4. Save the game; you’ll be redirected to the game page.

## 4) Import the stats (UI)

1. Open the game page (`Schedule` → click the game).
2. In **Import Shift Spreadsheet Stats**:
   - Upload `stats/player_stats.csv`
   - Optionally upload `stats/game_stats.csv`
3. Click `Import stats`.

What happens:
- The app upserts rows into `player_stats` for the matching players for that game.
- Per-period rows (e.g. `Period 1 TOI`, `Period 1 Shifts`, `Period 1 GF/GA`) are stored in `player_period_stats` and shown on the game page.
- Game-level key/value stats from `game_stats.csv` are stored in `hky_game_stats` and shown on the game page.

## 5) Verify the stats

- Game page:
  - **Imported Game Stats** shows the key/value table from `game_stats.csv`.
  - **Imported Shift/Video Stats (Read-only)** shows TOI, video TOI, shifts, SOG, xG, entries/exits, giveaways/takeaways, per-period splits, etc.
- Team page:
  - The Players table now includes aggregate totals for SOG, xG, TOI, shifts, and plus/minus (in addition to goals/assists/shots).

## Troubleshooting

- “Unmatched players” during import:
  - Ensure each player exists on the team in the webapp and has the correct jersey number.
  - Re-import after fixing jersey numbers.
- Missing TOI/shifts columns in the imported table:
  - Re-run `scripts/parse_shift_spreadsheet.py` with `--shifts` so it writes TOI/shift columns to `player_stats.csv`.

# scripts/AGENTS.md

This file applies to the `scripts/` subtree.

## `scripts/parse_shift_spreadsheet.py`

Parses a per-game “shift spreadsheet” and produces:
- Parent-facing per-player stats (CSV/XLSX/text), optionally including TOI/shift metrics.
- Per-game (team) stats (CSV/XLSX).
- Timestamp files + helper `clip_*.sh` scripts for generating per-player and per-event highlight clips.

It supports two spreadsheet “families”:
1) **Primary shift sheet** (the “normal” spreadsheet): shift start/end times per player, plus an optional event-log section.
2) **Companion `*-long*` sheet**: parses the *leftmost* per-period event table (shots/SOG/goals/xG, controlled entries/exits, rush, turnovers).

### Quick usage

Single game (pass the `stats/` dir; it will auto-discover files):
```bash
python scripts/parse_shift_spreadsheet.py \
  --input /home/colivier/RVideos/sharks-12-1-r2/stats \
  --outdir player_shifts
```

Season run (file list can contain files *or directories*; optional `:HOME`/`:AWAY` per line):
```bash
python scripts/parse_shift_spreadsheet.py \
  --file-list /mnt/ripper-data/Videos/game_list.txt \
  --outdir season_stats
```

Include TOI/shift counts in parent-facing outputs (off by default):
```bash
python scripts/parse_shift_spreadsheet.py --input <game stats dir> --outdir <out> --shifts
```

If `*-long*` team-color inference fails, force your team:
```bash
python scripts/parse_shift_spreadsheet.py --input <game stats dir> --outdir <out> --dark   # Blue is “us”
python scripts/parse_shift_spreadsheet.py --input <game stats dir> --outdir <out> --light  # White is “us”
```

### Inputs

#### `--input`
- Accepts a single `.xls/.xlsx` file **or** a directory.
- If a directory is provided, it expands to:
  - exactly one “primary” (non-`*-long*`) shift sheet, and
  - zero or more `*-long*` sheets in the same dir.
- You can also pass a game directory (it will try `<dir>/stats/`).
- Directory inputs must contain exactly one “game label” (derived from filename); otherwise it errors and you should use `--file-list`.

#### `--file-list`
- One file/dir per line; `#` comments and blank lines are ignored.
- Each line may end with `:HOME` or `:AWAY` (used for TimeToScore GF/GA mapping).
- Relative paths are resolved relative to the file-list’s directory.

#### Goals (GF/GA) sources and priority
The script tries, in order:
1) `--goal` / `--goals-file`
2) TimeToScore via `--t2s` (or inferred from filenames like `game-54111.xlsx` where the trailing number is `>= 10000`), using `--home/--away` or `:HOME/:AWAY` if needed.
3) `goals.xlsx` next to the primary sheet (fallback when no TimeToScore id is in use).

#### `*-long*` companion sheets
Only the leftmost per-period event table is used. It is identified by period header rows in column A
(e.g. `1st Period`, `2nd Period`, …) and per-row `Team` values (`Blue`/`White`).

### Outputs

For a single game, writes under `<outdir>/per_player/` (and possibly `<outdir>/event_log/` depending on sheet layout).
For multi-game runs, writes under `<outdir>/<game label>/per_player/` plus consolidated season outputs at:
- `<outdir>/player_stats_consolidated.xlsx` (multi-sheet workbook with `Cumulative` + per-game sheets)
- `<outdir>/player_stats_consolidated.csv` (the `Cumulative` sheet)
- `<outdir>/game_stats_consolidated.xlsx` / `<outdir>/game_stats_consolidated.csv` (stats as rows, games as columns)
  - Note: per-game averages for long-sheet-derived stats only count games where that stat exists; the `... per Game` columns in `Cumulative` include `(N)` to show the denominator.

Key outputs (per game):
- `per_player/stats/player_stats.txt`, `per_player/stats/player_stats.csv`, `per_player/stats/player_stats.xlsx`
  - By default, **no** shift counts or TOI columns are included (privacy). Use `--shifts` to include them.
  - Includes `SOG` plus `xG` columns and per-game ratios (e.g., `xG per SOG`).
- `per_player/stats/game_stats.csv`, `per_player/stats/game_stats.xlsx`
  - Team/game-level stats only (no TOI).
  - Layout is transposed: **stats as rows**, and the game name as the value column header.
- Timestamp files for video clipping:
  - Per-player shifts: `*_video_times.txt`, `*_scoreboard_times.txt` (skipped with `--no-scripts`; used only for clip generation).
  - Goals: `goals_for.txt`, `goals_against.txt` (video windows).
  - Long-sheet events: `events_<Event>_<Team>_video_times.txt` and `events_<Event>_<Team>_scoreboard_times.txt` (skipped with `--no-scripts`).
  - Per-player highlights (when long sheet present): `events_Goal_<player>_video_times.txt`, `events_SOG_<player>_video_times.txt`, etc. (skipped with `--no-scripts`).
- Clip helper scripts (unless `--no-scripts`):
  - `clip_<player>.sh` (per-player shifts)
  - `clip_events_<Event>_<Team>.sh` (per event type)
  - `clip_goal_<player>.sh`, `clip_sog_<player>.sh` (per-player highlight reels)
  - `clip_all.sh` / `clip_events_all.sh` (batch runners)

### Event model and stat semantics

Internal event type names:
- `Shot`, `SOG`, `Goal`, `Assist`
- `ExpectedGoal` (displayed as **`xG`** everywhere user-facing)
- `ControlledEntry`, `ControlledExit`, `Rush`
- `Giveaway`, `Takeaway` (from `Turnover` rows in `*-long*` sheets)

`xG` rule:
- Any explicit “Expected Goal” row counts as `xG`.
- **Any goal also counts as `xG`** (so goal rows increment both `Goal` and `ExpectedGoal`).

Turnovers (`*-long*`):
- A row labeled `Turnover` produces:
  - `Giveaway` credited to the row’s `Team` from the “Shots” column jersey.
  - `Takeaway` credited to the **other** team from the “Shots on Goal” column (often formatted like `Caused by #91`).

### Clip window durations

Clip windows are centered on each event time and then merged when close together:
- Non-goal events: **10s before, 5s after**.
- Goals: **20s before, 10s after**.
- See constants near the top of the file: `EVENT_CLIP_PRE_S`, `EVENT_CLIP_POST_S`, `GOAL_CLIP_PRE_S`, `GOAL_CLIP_POST_S`.

### Architecture map (where to look in code)

High-level flow (CLI):
- Argument parsing is at the bottom of `parse_shift_spreadsheet.py`.
- Inputs are expanded with `_parse_input_token()` and `_expand_dir_input_to_game_sheets()`.
- Each primary sheet is processed via `process_sheet()` which:
  - parses shifts and/or event logs,
  - parses any discovered `*-long*` sheets and merges their events,
  - resolves goals (manual / T2S / goals.xlsx),
  - writes per-player timestamps + scripts,
  - writes stats files (`player_stats.*`, per-player `*_stats.txt`, and `game_stats.*`).
- Multi-game runs aggregate per-game rows into `player_stats_consolidated.xlsx`.
- Multi-game runs also join per-game `game_stats.csv` into `game_stats_consolidated.xlsx` / `game_stats_consolidated.csv`.

Primary sheet parsing:
- `_parse_per_player_layout()` handles the standard “Period N / Jersey No” shift layout.
- `_parse_event_log_layout()` handles spreadsheets that contain an event-log table.

`*-long*` parsing:
- `_parse_long_left_event_table()` extracts per-row events from the leftmost per-period table.
- `_infer_focus_team_from_long_sheet()` decides whether “us” is Blue or White by roster overlap (or CLI override `--dark/--light`).
- `_event_log_context_from_long_events()` converts long events into the shared `EventLogContext` used by the rest of the pipeline.

Stats outputs:
- `_write_player_stats_text_and_csv()` writes player stats CSV/XLSX/text; TOI/shift columns are gated by `include_shifts_in_stats`.
- `_write_game_stats_files()` writes `game_stats.csv/xlsx` (transposed layout).
- `_display_col_name()` contains user-facing column labels (notably `expected_goals` → `xG`).
  - Per-player `stats/*_stats.txt` “Event Counts” will include an `xG: 0` line when the player has at least one `SOG`.

Excel styling:
- `_apply_excel_table_style()` applies teal title/header, gray banding, white grid borders.
- `_wrap_header_after_words()` wraps headers (currently after 2 words for generated XLSX files).

### Known “gotchas” / future work

- Long-sheet time parsing is intentionally heuristic (`_parse_long_mmss_time_to_seconds()`); if you encounter a new time encoding variant, adjust it carefully.
- Internal event names still use `ExpectedGoal`; display uses `xG` (via `_display_col_name()` / `_display_event_type()`).
- Per-player highlight scripts are currently generated for `Goal` and `SOG` only; extend `_write_player_event_highlights(..., highlight_types=...)` if needed.

HockeyMOM WebApp (Uploads + Run)
=========================

Overview
--------
Django-based web app (DTL + Django ORM) that lets users:
- Create an account and log in
- Create a new "game" (project) directory for uploads and trigger Slurm jobs via DirWatcher
- NEW: Manage hockey teams and players, create games, and track stats

It integrates with the DirWatcher service which submits Slurm jobs and tracks completion. DirWatcher optionally emails the user on job completion and logs failures.

Install
-------

Prereqs:
- Slurm and DirWatcher installed and configured
- Nginx and Python 3 available (script installs Nginx + venv)

Install and start:

```
sudo python3 tools/webapp/ops/install_webapp.py \
  --watch-root /data/incoming \
  --server-name _ \
  --port 8008
```

This will:
- Copy the app to `/opt/hm-webapp/app`
- Create a venv + install Django/Gunicorn (and PyMySQL)
- Install a systemd unit `hm-webapp.service`
- Install an Nginx site proxying `http://127.0.0.1:8008` and restart Nginx

If nginx fails to start with `Address already in use`, something else is already bound to the nginx listen port (default: 80; common culprit: `apache2`).
Either stop/disable the conflicting service, or re-run the installer with `--nginx-port <other-port>` (and optionally `--disable-apache2`).

Deploy to Google Cloud (smallest VM)
------------------------------------
This repo includes a helper that deploys the webapp + local MariaDB to a tiny Compute Engine VM (`e2-micro`) using the `gcloud` CLI.

1) Install/auth `gcloud` (if needed):
- https://cloud.google.com/sdk/docs/install
- `gcloud init`
- `gcloud auth login`

2) Deploy:

```
python3 tools/webapp/ops/deploy_gcp.py --project <PROJECT_ID> --zone us-central1-a
```

3) Delete everything created by the script:

```
python3 tools/webapp/ops/deploy_gcp.py --project <PROJECT_ID> --zone us-central1-a --delete
```

Redeploy (code-only)
--------------------
If you only changed `tools/webapp/app.py` / templates / static assets, you can do a fast redeploy that just copies files to the VM and restarts the service:

```
python3 tools/webapp/ops/redeploy_gcp.py --project <PROJECT_ID> --zone us-central1-a --instance hm-webapp
```

Uninstall
---------

```
sudo python3 tools/webapp/ops/uninstall_webapp.py
```

Usage
-----
- Open `http://<server>/`
- Upload/Run jobs: use the Jobs section as before
- Teams/Players: create teams (with logo), add/edit/delete players, view player and team stats
- Schedule: create games between one or two of your teams
  - If you select only one of your teams, enter an opponent name to auto-create an external team that is hidden from your team list by default
  - Edit a game to set scores and enter per-player stats (goals, assists, shots, PIM, +/-). Team standings are computed automatically from game results.

Dev (repo)
----------
- Run a dev server: `cd tools/webapp && python3 manage.py runserver`
- Project settings live in `tools/webapp/hm_webapp/settings.py` (`tools/webapp/django_settings.py` is a back-compat shim for installs/tests).

Import Shift Spreadsheet Stats
------------------------------
To import shift-spreadsheet stats from `scripts/parse_stats_inputs.py` into the webapp:

- Run `scripts/parse_stats_inputs.py` with `--upload-webapp ...` (or use `./import_webapp.sh`).
- The script uploads `stats/all_events_summary.csv` as `events_csv` and `stats/shift_rows.csv` as `shift_rows_csv` to
  `/api/import/hockey/shift_package`.
- The webapp computes player/game/team stats at runtime from `hky_game_event_rows` and `hky_game_shift_rows`
  (no `player_stats.csv` / `game_stats.csv` imports).

See `tools/webapp/TUTORIAL_SHIFT_STATS.md`.

### Shift spreadsheet file-list YAML (`game_list_long.yaml`)

`./import_webapp.sh` and `./gcp_import_webapp.sh` pass `SHIFT_FILE_LIST` to `scripts/parse_stats_inputs.py --file-list`.
Prefer a readable YAML mapping format (avoid legacy one-line `|key=value` strings).

Recommended structure:

```yaml
teams: # optional; per-team defaults (avoids repeating logos for every game)
  - name: "San Jose Jr Sharks 12AA-2"
    icon: /home/colivier/RVideos/logos/jrsharks.png
    # replace_logo defaults to true (overrides existing logos like TimeToScore)
    # replace_logo: false
  - name: "Texas Warriors 12AA"
    icon: /home/colivier/RVideos/logos/texas.png

games:
  # Spreadsheet-backed game (directory or .xlsx path)
  - path: /home/colivier/RVideos/stockton-r2/stats
    side: HOME # optional (HOME/AWAY)
    label: stockton-r2 # optional
    metadata: # alias: meta
      home_team: "San Jose Jr Sharks 12AA-2"
      away_team: "Stockton Colts 12AA"
      date: "2025-12-07"
      time: "16:15" # quote "HH:MM" (YAML may parse unquoted times as base-60)
      game_video: /home/colivier/RVideos/stockton-r2/game.mp4
      # Optional per-game logo overrides (these take precedence over `teams:` above):
      # home_logo: /home/colivier/RVideos/stockton-r2/home_logo.png
      # away_logo: /home/colivier/RVideos/stockton-r2/away_logo.png
      # Alias keys (same behavior as home_logo/away_logo):
      # home_team_icon: /home/colivier/RVideos/stockton-r2/home_logo.png
      # away_team_icon: /home/colivier/RVideos/stockton-r2/away_logo.png

  # TimeToScore-only game (no spreadsheets)
  - t2s: 51602
    side: HOME # optional
    label: stockton-r2 # optional
    metadata:
      game_video: "https://youtu.be/…"
```

Notes:
- Relative paths are resolved relative to the YAML file’s directory.
- `metadata:` is optional; you can also put metadata keys directly under the game mapping (e.g., `home_team: ...`).

### Event Corrections

The webapp supports idempotent event corrections (suppress bad imported events + upsert corrected ones). Corrected events render in **red** in the Player Events popup, and hover shows what changed (plus any note).

#### Inline in `--file-list` YAML (recommended)

Add `event_corrections:` under a game entry in your file-list YAML. When you run `scripts/parse_stats_inputs.py --file-list <file>.yaml --upload-webapp ...`, corrections are applied automatically after that game uploads.

```yaml
games:
  - label: utah-1
    path: utah-1/stats
    side: AWAY
    metadata:
      home_team: Texas Warriors 12AA
      away_team: San Jose Jr Sharks 12AA-2
      date: "2026-01-16"
      game_video: "https://youtu.be/..."
    event_corrections:
      reason: "Swap scorer/assist for goal at P2 03:14"
      patch:
        - match:
            event_type: Goal
            period: 2
            game_time: "03:14"
            team_side: Away
            jersey: "3"
          set:
            jersey: "13"
            video_time: "10:20"
          note: "see video"
```

#### Separate corrections file (`--corrections-yaml`)

You can also apply corrections from a separate YAML file via `scripts/parse_stats_inputs.py --corrections-yaml <file> --upload-webapp` (calls `POST /api/internal/apply_event_corrections`).

Recommended structure (patch format):

```yaml
create_missing_players: true # optional
corrections:
  - external_game_key: utah-1 # or: timetoscore_game_id: 51602, or: game_id: 123
    owner_email: you@example.com # recommended for external games
    reason: "Fix scorer/assists for goal at P2 03:14"
    patch:
      - match:
          event_type: Goal
          period: 2
          game_time: "03:14"
          team_side: Away
          jersey: "3"
        set:
          jersey: "13"
          video_time: "10:20"
        note: "see video"
```

Notes:
- `match` identifies the existing event; `set` applies overrides. Use `details` and/or `event_id` to disambiguate if multiple events can exist at the same timestamp.
- If `video_time` is present and the game has `metadata.game_video`, the Player Events popup can open the associated clip.

Demo Data
---------
To quickly demo the Teams/Schedule features, seed sample data into your configured DB:

```
python3 tools/webapp/scripts/seed_demo.py --config /opt/hm-webapp/app/config.json \
  --email demo@example.com --name "Demo User"
```

This creates a demo user, two teams, 10 players per team, default game types, two games (one completed with random per-player stats), and an external opponent team.

Import From time2score
----------------------
Use the import script to populate teams and games from the CAHA TimeToScore site into the webapp database. It will:
- Discover games by division schedules or explicit game ids (avoids Division Player Stats)
- Scrape per-game box scores to derive teams, rosters and scores
- Upsert teams as external teams for a selected user
- Upsert games with start time, location, scores, and optional league grouping

Example usage:

```
python3 tools/webapp/scripts/import_time2score.py \
  --config /opt/hm-webapp/app/config.json \
  --source caha \
  --season 0 \
  --user-email demo@example.com
```

Flags:
- `--source`: `caha` (CAHA/TimeToScore) or `sharksice` (SharksIce adult, league=1)
- `--t2s-league-id`: CAHA TimeToScore `league=` id (default: 3). Common values:
  - `3`: CAHA regular season
  - `5`: CAHA tier teams
  - `18`: CAHA tournaments (imported as game type "Tournament")
- `--season`: season id (0 = current/latest)
- `--user-email`: user that will own imported teams/games (teams marked is_external=1)
- `--division`: filter to only divisions you want (repeatable). Accepts:
  - substring name match (e.g., `--division 12AA`) — case-insensitive
  - level only (e.g., `--division 12`) to include all conferences at that level
  - exact level:conference (e.g., `--division 12:1`)
- `--list-seasons`: print the known season ids and exit
- `--list-divisions`: print divisions for the selected season and exit
- `--game-id` / `--games-file`: import from explicit game ids (one per line)
- `--limit`: cap number of games (useful for testing)
- `--team`: filter to games that involve a team name containing this substring (repeatable)
- League grouping and sharing:
  - `--league-name` (default: `CAHA-<season>`), `--shared`, `--share-with <email>`
  - Adds records to `leagues`, `league_members`, `league_teams`, `league_games`
- Non-destructive by default:
  - `--replace` overwrites existing game scores and `player_stats`; otherwise importer only fills missing values.

Import "CAHA"
-------------
To import CAHA/TimeToScore into a league named `CAHA`, use `import_time2score.py` with league flags:

```
python3 tools/webapp/scripts/import_time2score.py \
  --config /opt/hm-webapp/app/config.json \
  --source caha \
  --season 0 \
  --user-email demo@example.com \
  --league-name CAHA \
  --shared
```

Then in the web UI (Leagues page), mark the league as Public if you want it viewable without login.

Reset Hockey Data
-----------------
Wipe teams, players, hockey games, and player_stats while keeping users and league permissions intact. Useful before a fresh re-import.

Examples:

```
# Wipe everything (prompts for confirmation)
python3 tools/webapp/scripts/reset_league_data.py --config /opt/hm-webapp/app/config.json

# Wipe only a specific league by name
python3 tools/webapp/scripts/reset_league_data.py --config /opt/hm-webapp/app/config.json --league-name CAHA-Current-12U

# Non-interactive
python3 tools/webapp/scripts/reset_league_data.py --config /opt/hm-webapp/app/config.json --force
```

What it does:
- Global reset: deletes from `player_stats`, `league_games`, `hky_games`, `league_teams`, `players`, `teams` in FK-safe order.
- League-scoped reset: clears mappings for that league, deletes related `player_stats`; then removes `hky_games`/`teams` only if not referenced by other leagues or remaining games.
- Preserves: `users`, `leagues`, and `league_members`.

Notes:
- The hockey DB schema does not include a season column; the script stores season metadata in the `hky_games.notes` JSON field for reference.
- If the user does not exist, pass `--create-user --password-hash <pbkdf2_hash>` to create it.

Testing
-------
- Route/logic smoke tests for the new hockey features are in `tests/test_webapp_hockey.py`.
- They do not require a database. Tests import the app with `HM_WEBAPP_SKIP_DB_INIT=1` to avoid DB initialization and use fake cursors to validate stat computations.
- Register a new account and log in
- Create a game
- Upload files
- Click Run to create `_READY` in the game directory. DirWatcher sees it and submits the job.

Email Notifications
-------------------
DirWatcher tries to email the user on job completion (success or failure):
- Preferred: `/usr/sbin/sendmail` if available
- Fallback: SMTP if configured in `/etc/dirwatcher/config.yaml` under `behavior`:

```
behavior:
  from_email: no-reply@example.com
  smtp_host: smtp.example.com
  smtp_port: 587
  smtp_user: username
  smtp_pass: password
  smtp_use_tls: true
```

Failures
--------
- Slurm job errors are appended to `/var/log/dirwatcher/failed_jobs.log` by DirWatcher.
- Web app logs go to `journalctl -u hm-webapp` and Nginx error logs.

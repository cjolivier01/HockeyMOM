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

Import Shift Spreadsheet Stats
------------------------------
To import the `stats/player_stats.csv` + `stats/game_stats.csv` outputs written by `scripts/parse_stats_inputs.py` and view them per game/player/team:

- See `tools/webapp/TUTORIAL_SHIFT_STATS.md`.

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
- `--source`: `caha` (league=3) or `sharksice` (league=1)
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

Import "Norcal"
---------------
To import CAHA/TimeToScore into a league named `Norcal`, use `import_time2score.py` with league flags:

```
python3 tools/webapp/scripts/import_time2score.py \
  --config /opt/hm-webapp/app/config.json \
  --source caha \
  --season 0 \
  --user-email demo@example.com \
  --league-name Norcal \
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

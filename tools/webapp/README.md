HM WebApp (Uploads + Run)
=========================

Overview
--------
Django-based web app that lets users:
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
sudo python3 tools/webapp/install_webapp.py \
  --watch-root /data/incoming \
  --server-name _ \
  --port 8008
```

This will:
- Copy the app to `/opt/hm-webapp/app`
- Create a venv + install Django/Gunicorn
- Install a systemd unit `hm-webapp.service`
- Install an Nginx site proxying `http://127.0.0.1:8008` and restart Nginx

Uninstall
---------

```
sudo python3 tools/webapp/uninstall_webapp.py
```

Usage
-----
- Open `http://<server>/`
- Upload/Run jobs: use the Jobs section as before
- Teams/Players: create teams (with logo), add/edit/delete players, view player and team stats
- Schedule: create games between one or two of your teams
  - If you select only one of your teams, enter an opponent name to auto-create an external team that is hidden from your team list by default
  - Edit a game to set scores and enter per-player stats (goals, assists, shots, PIM, +/-). Team standings are computed automatically from game results.

Local Development (dev server)
------------------------------

From the repo checkout you can run the Django dev server directly:

```bash
cd tools/webapp
# Optional: point at the system DB/config instead of the default SQLite
export HM_DB_CONFIG=/opt/hm-webapp/app/config.json
./run_dev.sh          # runs manage.py migrate (best-effort) then runserver 127.0.0.1:8008
```

Then visit `http://127.0.0.1:8008/` in a browser. For most development it is fine to use the default SQLite DB; pointing at `/opt/hm-webapp/app/config.json` lets you test against the production-like MySQL schema.

Demo Data
---------
To quickly demo the Teams/Schedule features, seed sample data into your configured DB:

```
python3 tools/webapp/seed_demo.py --config /opt/hm-webapp/app/config.json \
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
python3 tools/webapp/import_time2score.py \
  --config /opt/hm-webapp/app/config.json \
  --season 2024 \
  --user-email demo@example.com \
  --sync --stats
```

Flags:
- `--season`: season id (0 = current)
- `--user-email`: user that will own imported teams/games (teams marked is_external=1)
- `--sync`: scrape and cache seasons/divisions/teams/games before import
- `--stats`: fetch box scores to fill missing scores if available
- `--tts-db-dir`: directory to place the temporary sqlite file used by the scraper (default: `tools/webapp/instance/time2score_db`)
- `--division`: filter to only divisions you want (repeatable). Accepts:
  - substring name match (e.g., `--division 12AA`) — case-insensitive
  - level only (e.g., `--division 12`) to include all conferences at that level
  - exact level:conference (e.g., `--division 12:1`)
- `--list-seasons`: print the known season ids and exit
- `--list-divisions`: print divisions for the selected season and exit
- `--game-id` / `--games-file`: import from explicit game ids (one per line)
- `--limit`: cap number of games (useful for testing)
- `--team`: filter to games that involve a team name containing this substring (repeatable)
- `--logo-dir`: directory to save team logos downloaded from the team page; importer updates `teams.logo_path`
- League grouping and sharing:
  - `--league-name` (default: `CAHA-<season>`), `--shared`, `--share-with <email>`
  - Adds records to `leagues`, `league_members`, `league_teams`, `league_games`

Reset Hockey Data
-----------------
Wipe teams, players, hockey games, and player_stats while keeping users and league permissions intact. Useful before a fresh re-import.

Examples:

```
# Wipe everything (prompts for confirmation)
python3 tools/webapp/reset_league_data.py --config /opt/hm-webapp/app/config.json

# Wipe only a specific league by name
python3 tools/webapp/reset_league_data.py --config /opt/hm-webapp/app/config.json --league-name CAHA-Current-12U

# Non-interactive
python3 tools/webapp/reset_league_data.py --config /opt/hm-webapp/app/config.json --force
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
- Unit-style tests for the hockey helpers live in `tests/test_webapp_hockey.py`:

  ```bash
  PYTHONPATH=tools/webapp python -m pytest tests/test_webapp_hockey.py
  ```

  These tests import the Django URL config and utility functions with `HM_WEBAPP_SKIP_DB_INIT=1` to avoid DB initialization and only exercise date/aggregate helpers and route wiring.

- End-to-end UI smoke test (against a running instance, e.g. via nginx on localhost):

  ```bash
  # Assumes hm-webapp.service and nginx are running
  tools/webapp/smoke_ui.sh http://127.0.0.1
  ```

  The smoke script:
  - Registers a throwaway user
  - Creates a team with a logo
  - Adds a player
  - Creates a scheduled game and marks it final
  - Verifies the team record and logo endpoint

Management commands and health checks
-------------------------------------

From the installed app directory (`/opt/hm-webapp/app`) using the system venv:

```bash
cd /opt/hm-webapp/app
source ../venv/bin/activate

# Initialize/upgrade the webapp schema (users/games/teams/etc.)
HM_DB_CONFIG=/opt/hm-webapp/app/config.json python manage.py init_hm_db

# Check DB connectivity (exit code 0 on success)
HM_DB_CONFIG=/opt/hm-webapp/app/config.json python manage.py hm_healthcheck
```

For HTTP health checks, hit the built-in endpoint:

```bash
curl http://127.0.0.1/healthz
```

which returns JSON including the DB status (e.g. `{"status": "ok", "db": true}`).

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

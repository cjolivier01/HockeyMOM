HM WebApp (Uploads + Run)
=========================

Overview
--------
Flask-based web app that lets users:
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
- Create a venv + install Flask/Gunicorn
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

Demo Data
---------
To quickly demo the Teams/Schedule features, seed sample data into your configured DB:

```
python3 tools/webapp/seed_demo.py --config /opt/hm-webapp/app/config.json \
  --email demo@example.com --name "Demo User"
```

This creates a demo user, two teams, 10 players per team, default game types, two games (one completed with random per-player stats), and an external opponent team.

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

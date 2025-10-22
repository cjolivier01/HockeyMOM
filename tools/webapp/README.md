HM WebApp (Uploads + Run)
=========================

Overview
--------
Flask-based web app that lets users:
- Create an account and log in
- Create a new "game" (project)
- Upload files to a dedicated subdirectory under the dirwatcher root
- Click "Run" to create the ready-file, triggering the Slurm job

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


DirWatcher Slurm Trigger
========================

Overview
--------
DirWatcher is a small Python service that monitors a configured root directory for new subdirectories. When a subdirectory contains a specific ready-file (e.g., `_READY`) that is stable (not changing and not locked), DirWatcher submits a Slurm job that processes that directory. After successful job completion, the subdirectory can optionally be deleted.

Key features:
- Poll-based watcher; no extra dependencies.
- Ready-file stability checks (size/mtime + non-blocking flock).
- Submits `sbatch` with the subdirectory path appended to your command/script.
- Tracks job completion using `sacct` (requires Slurm accounting).
- Optional cleanup on success; logs failed jobs to a file.

Components
----------
- `dirwatcher_service.py` — the watcher daemon.
- `process_dir.py` — a sample Slurm job script (Python) used by default.
- `install_service.py` — installs the service, config, and systemd unit.

Installation
------------
Prerequisites:
- A working Slurm setup on this node (slurmctld/slurmd `active`).
- Slurm accounting enabled (`slurmdbd` running), because DirWatcher uses `sacct` to detect final job state.

Install the service (example):

```
sudo python3 tools/dirwatch/install_service.py \
  --watch-root /data/incoming \
  --signal-file _READY \
  --partition main \
  --account dev \
  --gres gpu:1 \
  --time 00:10:00
```

Flags of note:
- `--delete-on-success`: delete the subdirectory after a COMPLETED job.
- `--wrap-cmd`: the command run by `sbatch --wrap` (default uses the sample job: `python3 /opt/dirwatcher/process_dir.py`).
- `--install-root`: where the scripts are installed (default `/opt/dirwatcher`).
- `--config-path`: service config file (default `/etc/dirwatcher/config.yaml`).
- `--failure-log`: path where failed job records are appended (default `/var/log/dirwatcher/failed_jobs.log`).

Service control:
- `sudo systemctl status dirwatcher`
- `sudo systemctl restart dirwatcher`
- `sudo systemctl stop dirwatcher`

Configuration
-------------
The installer writes a YAML config (default: `/etc/dirwatcher/config.yaml`). Example:

```
watch:
  root: /data/incoming
  signal_filename: _READY
  poll_interval_sec: 5
  stability_checks: 2
  stability_interval_sec: 1

job:
  wrap_cmd: python3 /opt/dirwatcher/process_dir.py
  partition: main
  account: dev
  gres: gpu:1
  time_limit: 00:10:00
  job_name_prefix: dirwatch
  env: {}
  extra_args: []
  chdir_to_subdir: true

behavior:
  delete_on_success: false
  state_file: /var/lib/dirwatcher/state.json
  failure_log: /var/log/dirwatcher/failed_jobs.log
```

How it Works
------------
1. DirWatcher polls `watch.root` for subdirectories containing `signal_filename`.
2. It checks the ready-file is stable and not locked.
3. It submits an sbatch job. The directory path is appended to your `wrap_cmd` or passed as an arg to `job_script`.
4. It polls `sacct` for job completion. On `COMPLETED`, it optionally deletes the subdirectory.
5. On failures (FAILED, CANCELLED, TIMEOUT), it appends a record to `behavior.failure_log`.

Usage
-----
- Prepare data: `mkdir -p /data/incoming/case1; cp ... /data/incoming/case1/`
- Signal ready: `touch /data/incoming/case1/_READY`
- The service submits a job similar to:
  `sbatch --partition=main --account=dev --gres=gpu:1 --chdir /data/incoming/case1 --wrap 'python3 /opt/dirwatcher/process_dir.py /data/incoming/case1'`
- Check accounting: `sacct -j <jobid> --format=JobID,State,ReqTRES%60,AllocTRES%60`
- Inspect failed jobs log: `/var/log/dirwatcher/failed_jobs.log`

Troubleshooting
---------------
- Service logs: `journalctl -u dirwatcher -f`
- Ensure `slurmdbd` is running if `sacct` returns no data.
- Verify the partition/account/gres values match your Slurm configuration.


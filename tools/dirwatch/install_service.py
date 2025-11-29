#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

SERVICE_NAME = "dirwatcher.service"


def write_config(
    cfg_path: Path,
    watch_root: str,
    signal_file: str,
    delete_on_success: bool,
    wrap_cmd: str,
    partition: str,
    account: str,
    gres: str,
    time_limit: str,
    state_file: str,
    failure_log: str,
    from_email: str = "",
    smtp_host: str = "",
    smtp_port: int = 0,
    smtp_user: str = "",
    smtp_pass: str = "",
    smtp_use_tls: bool = True,
    db_host: str = "127.0.0.1",
    db_port: int = 3306,
    db_name: str = "hm_app_db",
    db_user: str = "hmapp",
    db_pass: str = "",
) -> None:
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = f"""
watch:
  root: {watch_root}
  signal_filename: {signal_file}
  poll_interval_sec: 5
  stability_checks: 2
  stability_interval_sec: 1

job:
  wrap_cmd: {wrap_cmd}
  partition: {partition}
  account: {account}
  gres: {gres}
  time_limit: {time_limit}
  job_name_prefix: dirwatch
  env: {{}}
  extra_args: []
  chdir_to_subdir: true

behavior:
  delete_on_success: {str(delete_on_success).lower()}
  state_file: {state_file}
  failure_log: {failure_log}
  from_email: {from_email}
  smtp_host: {smtp_host}
  smtp_port: {smtp_port}
  smtp_user: {smtp_user}
  smtp_pass: {smtp_pass}
  smtp_use_tls: {str(smtp_use_tls).lower()}
db:
  host: {db_host}
  port: {db_port}
  name: {db_name}
  user: {db_user}
  pass: {db_pass}
""".strip()
    cfg_path.write_text(cfg + "\n")


def install_files(repo_root: Path, install_root: Path) -> None:
    install_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "tools/dirwatch/dirwatcher_service.py", install_root / "dirwatcher_service.py")
    shutil.copy2(repo_root / "tools/dirwatch/process_dir.py", install_root / "process_dir.py")
    (install_root / "process_dir.py").chmod(0o755)


def write_service(unit_path: Path, user: str, config_path: Path, python_bin: str, install_root: Path) -> None:
    unit = f"""
[Unit]
Description=DirWatcher Service (Slurm trigger)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={user}
Environment=PYTHONUNBUFFERED=1
Environment=MSMTP_CONFIG=/etc/msmtprc
Environment=MSMTPRC=/etc/msmtprc
ExecStart={python_bin} {install_root}/dirwatcher_service.py --config {config_path}
Restart=on-failure
RestartSec=3
Nice=10

[Install]
WantedBy=multi-user.target
""".lstrip()
    unit_path.write_text(unit)


def systemctl(*args: str) -> None:
    subprocess.run(["sudo", "-n", "true"], check=False)
    subprocess.check_call(["sudo", "systemctl", *args])


def configure_msmtp(
    user: str, from_email: str, smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str, use_tls: bool
) -> None:
    # Install msmtp and provide a sendmail alternative
    subprocess.check_call(["sudo", "apt-get", "update", "-y"])
    subprocess.check_call(["sudo", "apt-get", "install", "-y", "msmtp", "msmtp-mta", "ca-certificates"])
    msmtprc = Path("/etc/msmtprc")
    tls_line = "tls on" if use_tls else "tls off"
    # For port 587 (STARTTLS), turn starttls on; for 465 (SMTPS), turn it off
    tls_starttls = (
        "tls_starttls on"
        if (use_tls and smtp_port == 587)
        else ("tls_starttls off" if (use_tls and smtp_port == 465) else "")
    )
    auth_lines = []
    if smtp_user:
        # Gmail app passwords are shown with spaces; strip them
        clean_pass = smtp_pass.replace(" ", "")
        auth_lines = [
            f"user {smtp_user}",
            f"password {clean_pass}",
            "auth on",
        ]
    else:
        auth_lines = ["auth off"]
    lines = [
        "defaults",
        "  logfile /var/log/dirwatcher/msmtp.log",
        "account default",
        f"  host {smtp_host or 'localhost'}",
        f"  port {smtp_port or 25}",
        f"  from {from_email or user+'@'+os.uname().nodename}",
        f"  {tls_line}",
    ]
    if tls_starttls:
        lines.append(f"  {tls_starttls}")
    lines.append("  tls_trust_file /etc/ssl/certs/ca-certificates.crt")
    lines.extend(["  " + ln for ln in auth_lines])
    conf = "\n".join(lines) + "\n"
    msmtprc.write_text(conf)
    # Ensure log directory exists and is writeable by the service user
    subprocess.run(["sudo", "mkdir", "-p", "/var/log/dirwatcher"], check=False)
    subprocess.run(["sudo", "chown", f"{user}:{user}", "/var/log/dirwatcher"], check=False)
    # Restrict to root owner and service user group for read access (contains credentials)
    try:
        subprocess.check_call(["sudo", "chown", f"root:{user}", str(msmtprc)])
    except Exception:
        subprocess.check_call(["sudo", "chown", "root:root", str(msmtprc)])
    subprocess.check_call(["sudo", "chmod", "640", str(msmtprc)])
    # Mark stamp so we can uninstall later
    stamp = Path("/var/lib/dirwatcher/.installed_msmtp")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("1\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Install the dirwatcher systemd service")
    ap.add_argument("--watch-root", required=True, help="Directory to watch for subdirectories")
    ap.add_argument("--signal-file", default="_READY", help="Ready signal filename (default: _READY)")
    ap.add_argument(
        "--delete-on-success", action="store_true", help="Delete subdirectory after successful job completion"
    )
    ap.add_argument(
        "--user",
        default=os.environ.get("SUDO_USER") or os.environ.get("USER"),
        help="Service user (defaults to current user)",
    )
    ap.add_argument("--python-bin", default="", help="Python interpreter (prefer conda env of service user)")

    # Slurm job configuration
    ap.add_argument("--partition", default="main")
    ap.add_argument("--account", default="dev")
    ap.add_argument("--gres", default="gpu:1")
    ap.add_argument("--time", dest="time_limit", default="00:10:00")

    # Command to run within sbatch; default to sample script
    ap.add_argument(
        "--wrap-cmd",
        default="python3 /opt/dirwatcher/process_dir.py",
        help="Command to wrap with sbatch (directory path is appended)",
    )

    ap.add_argument(
        "--install-root",
        default="/opt/dirwatcher",
        help="Directory to install service scripts",
    )
    ap.add_argument(
        "--config-path",
        default="/etc/dirwatcher/config.yaml",
        help="Path to write service config (YAML)",
    )
    ap.add_argument(
        "--state-file",
        default="/var/lib/dirwatcher/state.json",
        help="Path to write state file",
    )
    ap.add_argument(
        "--failure-log",
        default="/var/log/dirwatcher/failed_jobs.log",
        help="Path to append failed job records",
    )
    # DB options (shared with webapp)
    ap.add_argument("--db-name", default="hm_app_db")
    ap.add_argument("--db-user", default="hmapp")
    ap.add_argument("--db-pass", default="hmapp_pass")
    ap.add_argument("--db-host", default="127.0.0.1")
    ap.add_argument("--db-port", type=int, default=3306)
    # SMTP options
    ap.add_argument("--smtp-setup", action="store_true", help="Install and configure msmtp for email sending")
    ap.add_argument("--from-email", default="", help="From email address for notifications")
    ap.add_argument("--smtp-host", default="", help="SMTP host (e.g., smtp.example.com)")
    ap.add_argument("--smtp-port", type=int, default=587, help="SMTP port (default 587)")
    ap.add_argument("--smtp-user", default="", help="SMTP username (for authenticated SMTP)")
    ap.add_argument("--smtp-pass", default="", help="SMTP password")
    ap.add_argument("--smtp-use-tls", action="store_true", default=True, help="Use STARTTLS (default on)")
    ap.add_argument("--no-smtp-use-tls", action="store_false", dest="smtp_use_tls", help="Disable STARTTLS")

    # Uninstall mode
    ap.add_argument("--uninstall", action="store_true", help="Uninstall the DirWatcher service and optional SMTP")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    install_root = Path(args.install_root)
    config_path = Path(args.config_path)
    state_file = Path(args.state_file)
    failure_log = Path(args.failure_log)
    unit_path = Path("/etc/systemd/system") / SERVICE_NAME

    # Uninstall mode
    if args.uninstall:
        try:
            systemctl("disable", "--now", "dirwatcher")
        except Exception:
            pass
        try:
            subprocess.run(["sudo", "rm", "-f", "/etc/systemd/system/dirwatcher.service"], check=False)
            subprocess.run(["sudo", "systemctl", "daemon-reload"], check=False)
        except Exception:
            pass
        # Remove installed files
        subprocess.run(["sudo", "rm", "-rf", "/opt/dirwatcher"], check=False)
        # Config/state/logs
        subprocess.run(["sudo", "rm", "-rf", "/etc/dirwatcher"], check=False)
        subprocess.run(["sudo", "rm", "-rf", "/var/lib/dirwatcher"], check=False)
        # Remove msmtp if we installed it
        stamp = Path("/var/lib/dirwatcher/.installed_msmtp")
        if stamp.exists():
            subprocess.run(["sudo", "apt-get", "remove", "-y", "msmtp", "msmtp-mta"], check=False)
            subprocess.run(["sudo", "rm", "-f", "/etc/msmtprc"], check=False)
        print("Uninstall complete.")
        return 0

    # Prepare filesystem
    install_files(repo_root, install_root)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    (config_path.parent).mkdir(parents=True, exist_ok=True)

    # Determine python for the service user (prefer their conda env)
    if not args.python_bin:
        try:
            user_py = subprocess.check_output(
                ["sudo", "-H", "-u", args.user, "bash", "-lc", "command -v python || command -v python3"],
                text=True,
            ).strip()
            python_bin = user_py or "/usr/bin/python3"
        except Exception:
            python_bin = "/usr/bin/python3"
    else:
        python_bin = args.python_bin
    # Ensure pymysql is available in that environment
    try:
        subprocess.check_call(
            ["sudo", "-H", "-u", args.user, "bash", "-lc", f"{python_bin} -m pip install --upgrade pip wheel pymysql"]
        )
    except Exception:
        pass

    write_config(
        config_path,
        watch_root=args.watch_root,
        signal_file=args.signal_file,
        delete_on_success=args.delete_on_success,
        wrap_cmd=args.wrap_cmd,
        partition=args.partition,
        account=args.account,
        gres=args.gres,
        time_limit=args.time_limit,
        state_file=str(state_file),
        failure_log=str(failure_log),
        from_email=args.from_email,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_pass=args.smtp_pass,
        smtp_use_tls=bool(args.smtp_use_tls),
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_pass=args.db_pass,
    )

    write_service(unit_path, args.user, config_path, python_bin, install_root)

    # Ensure state and log paths are writeable by the service user
    try:
        if not state_file.exists():
            state_file.write_text("{" "'" '\n  "active": {},\n  "processed": {}\n\n' "'" "}")
        # Ensure both the state file and its directory are owned by the service user
        subprocess.check_call(["sudo", "chown", "-R", f"{args.user}:{args.user}", str(state_file.parent)])
        subprocess.check_call(["sudo", "chown", "-R", f"{args.user}:{args.user}", str(failure_log.parent)])
        # Protect config file since it may contain SMTP credentials
        subprocess.check_call(["sudo", "chown", f"{args.user}:{args.user}", str(config_path)])
        subprocess.check_call(["sudo", "chmod", "600", str(config_path)])
    except Exception:
        pass

    # Optionally install SMTP (msmtp) to provide sendmail and SMTP relay
    if args.smtp_setup:
        configure_msmtp(
            args.user,
            args.from_email,
            args.smtp_host,
            args.smtp_port,
            args.smtp_user,
            args.smtp_pass,
            args.smtp_use_tls,
        )

    # Reload and start service
    systemctl("daemon-reload")
    systemctl("enable", "--now", SERVICE_NAME)

    print(f"Installed and started {SERVICE_NAME}")
    print(f"  Config:  {config_path}")
    print(f"  Scripts: {install_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import dataclasses
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-Linux fallback
    fcntl = None  # type: ignore


@dataclass
class JobConfig:
    wrap_cmd: Optional[str] = None
    job_script: Optional[str] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    gres: Optional[str] = None
    time_limit: Optional[str] = None  # e.g., 00:10:00
    job_name_prefix: str = "dirwatch"
    env: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    chdir_to_subdir: bool = True


@dataclass
class WatchConfig:
    root: str
    signal_filename: str = "_READY"
    poll_interval_sec: float = 5.0
    stability_checks: int = 2
    stability_interval_sec: float = 1.0


@dataclass
class BehaviorConfig:
    delete_on_success: bool = False
    state_file: Optional[str] = None
    failure_log: Optional[str] = None
    from_email: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    smtp_use_tls: bool = True


@dataclass
class Config:
    watch: WatchConfig
    job: JobConfig
    behavior: BehaviorConfig


@dataclass
class State:
    processed: Dict[str, Dict[str, str]] = field(default_factory=dict)  # subdir -> info
    active: Dict[str, str] = field(default_factory=dict)  # jobid -> subdir


def load_yaml(path: Path) -> Dict:
    # Minimal YAML loader to avoid PyYAML dependency (accepts JSON superset)
    import re

    text = path.read_text()
    # If it looks like JSON, parse directly
    txt = text.strip()
    if txt.startswith("{") or txt.startswith("["):
        return json.loads(text)

    # Tiny YAML-to-JSON-ish conversion for a simple dict+lists
    # This is intentionally minimal; recommend providing JSON or simple YAML.
    # For robustness, allow key: value lines and list items with '-'.
    result: Dict = {}
    stack: List[Tuple[int, Dict]] = [(0, result)]
    current = result
    indent_re = re.compile(r"^(?P<indent>\s*)(?P<body>.*)$")

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = indent_re.match(line)
        assert m
        indent = len(m.group("indent"))
        body = m.group("body")

        # adjust current by indent
        while stack and indent < stack[-1][0]:
            stack.pop()
        if stack:
            current = stack[-1][1]

        if body.startswith("-"):
            # list item; find last key which should be a list
            key = next(reversed(current))
            if not isinstance(current[key], list):
                current[key] = [current[key]]
            val = body[1:].strip()
            current[key].append(_parse_scalar(val))
        elif ":" in body:
            key, val = body.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val == "":
                # nested dict
                d: Dict = {}
                current[key] = d
                stack.append((indent + 2, d))
                current = d
            else:
                current[key] = _parse_scalar(val)
        else:
            raise ValueError(f"Unsupported config line: {raw_line}")
    return result


def _parse_scalar(val: str):
    # Try JSON for inline dict/list
    if val.startswith("{") or val.startswith("["):
        try:
            return json.loads(val)
        except Exception:
            pass
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if val.isdigit():
        return int(val)
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def read_config(path: Path) -> Config:
    raw = load_yaml(path)

    watch = raw.get("watch", {})
    job = raw.get("job", {})
    behavior = raw.get("behavior", {})

    cfg = Config(
        watch=WatchConfig(
            root=str(watch["root"]),
            signal_filename=str(watch.get("signal_filename", "_READY")),
            poll_interval_sec=float(watch.get("poll_interval_sec", 5.0)),
            stability_checks=int(watch.get("stability_checks", 2)),
            stability_interval_sec=float(watch.get("stability_interval_sec", 1.0)),
        ),
        job=JobConfig(
            wrap_cmd=job.get("wrap_cmd"),
            job_script=job.get("job_script"),
            partition=job.get("partition"),
            account=job.get("account"),
            gres=job.get("gres"),
            time_limit=job.get("time_limit"),
            job_name_prefix=str(job.get("job_name_prefix", "dirwatch")),
            env=dict(job.get("env", {})),
            extra_args=list(job.get("extra_args", [])),
            chdir_to_subdir=bool(job.get("chdir_to_subdir", True)),
        ),
        behavior=BehaviorConfig(
            delete_on_success=bool(behavior.get("delete_on_success", False)),
            state_file=behavior.get("state_file"),
            failure_log=behavior.get("failure_log"),
            from_email=behavior.get("from_email"),
            smtp_host=behavior.get("smtp_host"),
            smtp_port=(int(behavior.get("smtp_port")) if behavior.get("smtp_port") else None),
            smtp_user=behavior.get("smtp_user"),
            smtp_pass=behavior.get("smtp_pass"),
            smtp_use_tls=bool(behavior.get("smtp_use_tls", True)),
        ),
    )
    return cfg


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    try:
        data = json.loads(path.read_text())
        return State(processed=data.get("processed", {}), active=data.get("active", {}))
    except Exception:
        return State()


def save_state(path: Path, state: State) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True))
    tmp.replace(path)


def is_file_stable(p: Path, stability_checks: int, interval: float) -> bool:
    try:
        last_stat = None
        for _ in range(stability_checks):
            st = p.stat()
            cur = (st.st_size, st.st_mtime_ns)
            if last_stat is None:
                last_stat = cur
            elif cur != last_stat:
                return False
            # also try a non-blocking exclusive lock if available
            if fcntl is not None:
                try:
                    with p.open("rb") as fh:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    return False
            time.sleep(interval)
        return True
    except FileNotFoundError:
        return False


def submit_sbatch(subdir: Path, job_cfg: JobConfig) -> Optional[str]:
    cmd: List[str] = ["sbatch", "--parsable"]

    job_name = f"{job_cfg.job_name_prefix}-{subdir.name}"
    cmd += ["--job-name", job_name]

    if job_cfg.partition:
        cmd += ["--partition", job_cfg.partition]
    if job_cfg.account:
        cmd += ["--account", job_cfg.account]
    if job_cfg.gres:
        cmd += ["--gres", job_cfg.gres]
    if job_cfg.time_limit:
        cmd += ["--time", job_cfg.time_limit]
    if job_cfg.chdir_to_subdir:
        cmd += ["--chdir", str(subdir)]

    # Environment export
    env_parts = ["ALL", f"WATCH_DIR={subdir}"]
    for k, v in job_cfg.env.items():
        env_parts.append(f"{k}={v}")
    cmd += ["--export", ",".join(env_parts)]

    # Extra args (advanced users)
    cmd += job_cfg.extra_args

    # Payload
    if job_cfg.wrap_cmd:
        # Build a wrapped command passing the directory
        tokens = shlex.split(job_cfg.wrap_cmd)
        tokens.append(str(subdir))
        wrap_str = " ".join(shlex.quote(t) for t in tokens)
        cmd += ["--wrap", wrap_str]
    elif job_cfg.job_script:
        cmd += [job_cfg.job_script, str(subdir)]
    else:
        raise ValueError("Either job.wrap_cmd or job.job_script must be provided")

    logging.info("Submitting: %s", shlex.join(cmd))
    try:
        out = subprocess.check_output(cmd, text=True)
        job_id = out.strip().split(";", 1)[0]
        logging.info("Submitted job %s for %s", job_id, subdir)
        return job_id
    except subprocess.CalledProcessError as e:
        logging.exception("sbatch submission failed: %s", e)
        return None


def get_job_state(job_id: str) -> Optional[str]:
    # Use sacct to query final state; returns None if not yet available
    try:
        out = subprocess.check_output(
            [
                "sacct",
                "-j",
                job_id,
                "-n",
                "-P",
                "--format=JobID,State",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return None

    # Parse primary job line (not .batch or .extern)
    state_map: Dict[str, str] = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            state_map[parts[0].strip()] = parts[1].strip()

    # Prefer exact job_id; otherwise, first root-like entry
    if job_id in state_map:
        return state_map[job_id]
    # Fallback: if nothing, None
    return None


def discover_ready_subdirs(watch_root: Path, signal_name: str, stability_checks: int, stability_interval: float) -> List[Path]:
    ready: List[Path] = []
    if not watch_root.exists():
        return ready
    for entry in watch_root.iterdir():
        if not entry.is_dir():
            continue
        sig = entry / signal_name
        if sig.exists() and is_file_stable(sig, stability_checks, stability_interval):
            ready.append(entry)
    return ready


def run_service(cfg: Config) -> None:
    watch_root = Path(cfg.watch.root)
    watch_root.mkdir(parents=True, exist_ok=True)

    # State location
    if cfg.behavior.state_file:
        state_path = Path(cfg.behavior.state_file)
    else:
        state_path = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "dirwatcher/state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Prepare failure log dir if configured
    if cfg.behavior.failure_log:
        try:
            Path(cfg.behavior.failure_log).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            logging.exception("Failed to create parent directory for failure_log")
    state = load_state(state_path)

    stop = threading.Event()

    def _handle_sig(signum, frame):  # noqa: ARG001
        logging.info("Signal %s received; shutting down.", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    logging.info("Starting dirwatcher: root=%s signal=%s", watch_root, cfg.watch.signal_filename)

    try:
        while not stop.is_set():
            # 1) Submit jobs for newly ready subdirs
            ready = discover_ready_subdirs(
                watch_root,
                cfg.watch.signal_filename,
                cfg.watch.stability_checks,
                cfg.watch.stability_interval_sec,
            )
            for sub in ready:
                sub_key = str(sub.resolve())
                if sub_key in state.processed:
                    continue
                job_id = submit_sbatch(sub, cfg.job)
                if job_id:
                    state.active[job_id] = sub_key
                    state.processed[sub_key] = {"status": "SUBMITTED", "job_id": job_id}
                    save_state(state_path, state)
                else:
                    # sbatch submission failed; record into failure log
                    if cfg.behavior.failure_log:
                        try:
                            with open(cfg.behavior.failure_log, "a", encoding="utf-8") as fh:
                                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} SUBMIT_FAIL dir={sub_key}\n")
                        except Exception:
                            logging.exception("Failed to write failure_log")

            # 2) Check active jobs
            if state.active:
                finished: List[str] = []
                for job_id, sub_key in list(state.active.items()):
                    st = get_job_state(job_id)
                    if not st:
                        continue
                    logging.info("Job %s state: %s", job_id, st)
                    # Normalize e.g., COMPLETED, FAILED, CANCELLED, TIMEOUT
                    final = any(st.startswith(x) for x in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"))
                    if not final:
                        continue
                    # Update status
                    info = state.processed.get(sub_key, {})
                    info["status"] = st
                    state.processed[sub_key] = info
                    finished.append(job_id)

                    # Notify via email if possible
                    try:
                        to_addr = _read_user_email_from_meta(Path(sub_key))
                        if to_addr:
                            subj = f"DirWatcher job {st} - {Path(sub_key).name}"
                            body = (
                                f"Directory: {sub_key}\nJob ID: {job_id}\nState: {st}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )
                            send_email(to_addr, subj, body, cfg.behavior)
                    except Exception:
                        logging.exception("Email notification failed")

                    # On success, optionally delete subdir
                    if st.startswith("COMPLETED") and cfg.behavior.delete_on_success:
                        try:
                            from shutil import rmtree

                            logging.info("Deleting processed directory: %s", sub_key)
                            rmtree(sub_key)
                        except Exception:
                            logging.exception("Failed to delete %s", sub_key)
                    # On failure-like states, append to failure log with details
                    if (not st.startswith("COMPLETED")) and cfg.behavior.failure_log:
                        try:
                            # Try to get exit code and reason from sacct for .batch step
                            try:
                                q = subprocess.check_output(
                                    ["sacct", "-j", job_id, "-n", "-P", "--format=JobID,ExitCode,Reason"],
                                    text=True,
                                ).strip()
                            except Exception:
                                q = ""
                            with open(cfg.behavior.failure_log, "a", encoding="utf-8") as fh:
                                fh.write(
                                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')} JOB_FAIL job={job_id} state={st} dir={sub_key} details={q}\n"
                                )
                        except Exception:
                            logging.exception("Failed to write failure_log")

                for job_id in finished:
                    state.active.pop(job_id, None)
                if finished:
                    save_state(state_path, state)

            stop.wait(timeout=cfg.watch.poll_interval_sec)
    finally:
        save_state(state_path, state)
        logging.info("Dirwatcher stopped.")


def _read_user_email_from_meta(d: Path) -> Optional[str]:
    meta = d / ".dirwatch_meta.json"
    try:
        import json as _json

        data = _json.loads(meta.read_text())
        em = data.get("user_email")
        if isinstance(em, str) and "@" in em:
            return em
    except Exception:
        return None
    return None


def send_email(to_addr: str, subject: str, body: str, bcfg: BehaviorConfig) -> None:
    # Only send if email is configured. If neither SMTP nor from_email is set,
    # treat email as disabled and return silently.
    if not (bcfg.smtp_host or bcfg.from_email):
        logging.info("Email disabled (no smtp_host/from_email configured) for %s", to_addr)
        return

    from_addr = bcfg.from_email or ("no-reply@" + os.uname().nodename)
    msg = (
        f"From: {from_addr}\n"
        f"To: {to_addr}\n"
        f"Subject: {subject}\n"
        f"Content-Type: text/plain; charset=utf-8\n\n"
        f"{body}\n"
    )
    # Prefer system sendmail (msmtp-mta) with config path pinned via environment
    import shutil as _sh
    sendmail = _sh.which("sendmail")
    if sendmail:
        try:
            subprocess.run([sendmail, "-t"], input=msg.encode("utf-8"), check=True)
            logging.info("Email sent to %s via sendmail", to_addr)
            return
        except Exception:
            logging.exception("sendmail failed; falling back to SMTP if configured")
    # SMTP fallback
    if bcfg.smtp_host:
        import smtplib

        try:
            server = smtplib.SMTP(bcfg.smtp_host, bcfg.smtp_port or 587, timeout=10)
            if bcfg.smtp_use_tls:
                server.starttls()
            if bcfg.smtp_user and bcfg.smtp_pass:
                server.login(bcfg.smtp_user, bcfg.smtp_pass)
            server.sendmail(from_addr, [to_addr], msg)
            server.quit()
            logging.info("Email sent to %s via SMTP", to_addr)
            return
        except Exception:
            logging.exception("SMTP send failed")
    logging.warning("No email method available to notify %s", to_addr)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Directory watcher that triggers Slurm jobs")
    p.add_argument("--config", default="/etc/dirwatcher/config.yaml", help="Path to config (YAML or JSON)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = read_config(Path(args.config))
    run_service(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

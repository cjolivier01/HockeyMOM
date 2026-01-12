#!/usr/bin/env python3
"""
Deploy the HockeyMOM webapp (tools/webapp) to Google Cloud using the smallest GCE instance size.

This script uses the `gcloud` CLI (recommended by Google) rather than direct GCP APIs.
It creates:
  - A Compute Engine VM (default: e2-micro)
  - A firewall rule allowing inbound HTTP (tcp:80) to the VM via a network tag

Then it copies `tools/webapp` to the VM and runs `tools/webapp/ops/install_webapp.py` there, which installs:
  - nginx + gunicorn + Django app
  - MariaDB (local) + schema/user

Use --delete to remove the VM and firewall rule created by this script.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_ACCOUNT_EMAIL = "cjolivier01@gmail.com"
GCLOUD_FALLBACKS = (
    "/snap/google-cloud-sdk/current/bin/gcloud",
    "/snap/google-cloud-sdk/632/bin/gcloud",
)
GCLOUD_BIN = "gcloud"


@dataclass(frozen=True)
class GcpNames:
    instance: str
    firewall_rule: str
    network_tag: str


def _resolve_gcloud_bin() -> str:
    def _works(path: str) -> bool:
        try:
            cp = subprocess.run([path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            return False
        if cp.returncode != 0:
            return False
        return "Google Cloud SDK" in (cp.stdout or "") or "Google Cloud SDK" in (cp.stderr or "")

    gcloud = shutil.which("gcloud")
    if gcloud and _works(gcloud):
        return gcloud
    for p in GCLOUD_FALLBACKS:
        if p and shutil.which(p) is not None and _works(p):
            return p
        if p and shutil.os.path.exists(p) and _works(p):
            return p
    raise RuntimeError(
        "Missing a working `gcloud` CLI. Install it first:\n"
        "  https://cloud.google.com/sdk/docs/install\n"
        "Then run:\n"
        "  gcloud init\n"
        "  gcloud auth login\n"
    )


def _run(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
    )


def _gcloud(cmd: list[str], *, project: str | None, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    base = [GCLOUD_BIN, "--quiet"]
    if project:
        base += ["--project", project]
    return _run(base + cmd, capture=capture, check=check)


def _require_gcloud() -> None:
    # Resolved during module init (main).
    return


def _default_project() -> str | None:
    # `gcloud config get-value project` prints "(unset)" if unset.
    try:
        p = _run([GCLOUD_BIN, "config", "get-value", "project"], capture=True, check=True).stdout.strip()
    except Exception:
        return None
    if not p or p == "(unset)":
        return None
    return p


def _active_account() -> str | None:
    try:
        out = _run(
            [GCLOUD_BIN, "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
            capture=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None
    return out or None


def _best_effort_set_account(project: str | None, account_email: str | None) -> None:
    if not account_email:
        return
    # This does not log you in; it only selects an account if already authenticated.
    _gcloud(["config", "set", "account", account_email], project=project, check=False)


def _resource_exists(project: str, gcloud_args: list[str]) -> bool:
    cp = _gcloud(gcloud_args, project=project, capture=True, check=False)
    return cp.returncode == 0


def _detect_instance_zone(project: str, instance: str) -> str | None:
    cp = _gcloud(
        ["compute", "instances", "list", "--filter", f"name=({instance})", "--format=value(zone)"],
        project=project,
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        return None
    zone = (cp.stdout or "").strip()
    if not zone:
        return None
    return zone.split("/")[-1]


def _wait_for_ssh(project: str, instance: str, zone: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while True:
        cp = _gcloud(
            ["compute", "ssh", instance, "--zone", zone, "--command", "true"],
            project=project,
            capture=True,
            check=False,
        )
        if cp.returncode == 0:
            return
        if time.time() > deadline:
            stderr = (cp.stderr or "").strip()
            raise RuntimeError(f"Timed out waiting for SSH to {instance} in {zone}. Last error:\n{stderr}")
        time.sleep(5)


def _instance_external_ip(project: str, instance: str, zone: str) -> str | None:
    cp = _gcloud(
        [
            "compute",
            "instances",
            "describe",
            instance,
            "--zone",
            zone,
            "--format=value(networkInterfaces[0].accessConfigs[0].natIP)",
        ],
        project=project,
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        return None
    ip = (cp.stdout or "").strip()
    return ip or None


def _deploy(args: argparse.Namespace, names: GcpNames) -> None:
    project = args.project
    zone = args.zone

    _best_effort_set_account(project, args.account_email)
    active = _active_account()
    if not active:
        raise RuntimeError("No active gcloud account. Run `gcloud auth login` (and `gcloud init`).")
    if args.account_email and active != args.account_email:
        print(f"Note: active account is {active!r} (requested {args.account_email!r}).", file=sys.stderr)

    print("Enabling Compute Engine API (best-effort)...")
    _gcloud(["services", "enable", "compute.googleapis.com"], project=project, check=False)

    if not _resource_exists(project, ["compute", "firewall-rules", "describe", names.firewall_rule]):
        allowed = f"tcp:80,tcp:{args.app_port}"
        if args.enable_https:
            allowed = f"{allowed},tcp:443"
        firewall_created = True
        print(f"Creating firewall rule {names.firewall_rule!r} (tcp:80)...")
        _gcloud(
            [
                "compute",
                "firewall-rules",
                "create",
                names.firewall_rule,
                "--network",
                args.network,
                "--allow",
                allowed,
                "--direction",
                "INGRESS",
                "--source-ranges",
                "0.0.0.0/0",
                "--target-tags",
                names.network_tag,
                "--description",
                f"Allow inbound hm-webapp ports ({allowed})",
            ],
            project=project,
            check=True,
        )
    else:
        firewall_created = False
        print(f"Firewall rule {names.firewall_rule!r} already exists; skipping.")

    if args.enable_https and not firewall_created:
        https_rule = f"{args.name}-allow-https"
        if not _resource_exists(project, ["compute", "firewall-rules", "describe", https_rule]):
            # Separate rule so existing deployments can enable HTTPS later without changing the main rule.
            _gcloud(
                [
                    "compute",
                    "firewall-rules",
                    "create",
                    https_rule,
                    "--network",
                    args.network,
                    "--allow",
                    "tcp:443",
                    "--direction",
                    "INGRESS",
                    "--source-ranges",
                    "0.0.0.0/0",
                    "--target-tags",
                    names.network_tag,
                    "--description",
                    "Allow inbound HTTPS to hm webapp VM",
                ],
                project=project,
                check=True,
            )

    if not _resource_exists(project, ["compute", "instances", "describe", names.instance, "--zone", zone]):
        print(f"Creating instance {names.instance!r} in {zone!r} ({args.machine_type})...")
        _gcloud(
            [
                "compute",
                "instances",
                "create",
                names.instance,
                "--zone",
                zone,
                "--machine-type",
                args.machine_type,
                "--network",
                args.network,
                "--image-family",
                args.image_family,
                "--image-project",
                args.image_project,
                "--boot-disk-size",
                args.boot_disk_size,
                "--tags",
                names.network_tag,
            ],
            project=project,
            check=True,
        )
    else:
        print(f"Instance {names.instance!r} already exists; skipping create.")

    print("Waiting for SSH connectivity...")
    _wait_for_ssh(project, names.instance, zone, timeout_s=args.ssh_timeout_s)

    print("Preparing remote staging directory...")
    _gcloud(
        [
            "compute",
            "ssh",
            names.instance,
            "--zone",
            zone,
            "--command",
            "mkdir -p /tmp/hm/tools",
        ],
        project=project,
        check=True,
    )

    print("Copying `tools/webapp` to the VM...")
    _gcloud(
        [
            "compute",
            "scp",
            "--recurse",
            "tools/webapp",
            f"{names.instance}:/tmp/hm/tools",
            "--zone",
            zone,
        ],
        project=project,
        check=True,
    )

    server_name = args.server_name
    if server_name == "AUTO":
        server_name = _instance_external_ip(project, names.instance, zone) or "_"

    print("Installing webapp on the VM (nginx + mariadb + gunicorn)...")
    install_cmd = [
        "sudo",
        "python3",
        "/tmp/hm/tools/webapp/ops/install_webapp.py",
        "--watch-root",
        args.watch_root,
        "--server-name",
        server_name,
        "--port",
        str(args.app_port),
        "--bind-address",
        args.bind_address,
        "--db-name",
        args.db_name,
        "--db-user",
        args.db_user,
        "--db-pass",
        args.db_pass,
    ]
    if args.import_token:
        install_cmd += ["--import-token", args.import_token]
    _gcloud(
        [
            "compute",
            "ssh",
            names.instance,
            "--zone",
            zone,
            "--command",
            shlex.join(install_cmd),
        ],
        project=project,
        check=True,
    )

    if args.enable_https:
        if not args.domains:
            raise RuntimeError("--enable-https requires --domains (e.g. --domains jrsharks2013.org,www.jrsharks2013.org)")
        if not args.https_email:
            raise RuntimeError("--enable-https requires --https-email")
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if not domains:
            raise RuntimeError("--domains must not be empty")
        server_names = " ".join(domains)
        certbot_cmd = (
            "set -euo pipefail; "
            f"sudo sed -i 's/^\\s*server_name .*;/    server_name {server_names};/' /etc/nginx/sites-available/hm-webapp; "
            "sudo nginx -t; sudo systemctl reload nginx; "
            "sudo apt-get update -y; "
            "sudo apt-get install -y certbot python3-certbot-nginx; "
            f"sudo certbot --nginx -n --agree-tos --email {shlex.quote(args.https_email)} "
            + " ".join([f"-d {shlex.quote(d)}" for d in domains])
            + " --redirect; "
            "sudo nginx -t; sudo systemctl reload nginx; "
            "sudo systemctl enable --now certbot.timer || true"
        )
        print(f"Enabling HTTPS for: {', '.join(domains)}")
        _gcloud(
            [
                "compute",
                "ssh",
                names.instance,
                "--zone",
                zone,
                "--command",
                certbot_cmd,
            ],
            project=project,
            check=True,
        )

    ip = _instance_external_ip(project, names.instance, zone)
    if ip:
        print("")
        print("Deployed.")
        print(f"URL: http://{ip}/")
        print(f"SSH: gcloud compute ssh {names.instance} --zone {zone} --project {project}")
    else:
        print("Deployed, but could not determine external IP. Use:")
        print(f"  gcloud compute instances describe {names.instance} --zone {zone} --project {project}")


def _delete(args: argparse.Namespace, names: GcpNames) -> None:
    project = args.project
    zone = args.zone or _detect_instance_zone(project, names.instance)

    _best_effort_set_account(project, args.account_email)

    if zone and _resource_exists(project, ["compute", "instances", "describe", names.instance, "--zone", zone]):
        print(f"Deleting instance {names.instance!r} in {zone!r}...")
        _gcloud(
            ["compute", "instances", "delete", names.instance, "--zone", zone],
            project=project,
            check=True,
        )
    else:
        print(f"Instance {names.instance!r} not found; skipping.")

    if _resource_exists(project, ["compute", "firewall-rules", "describe", names.firewall_rule]):
        print(f"Deleting firewall rule {names.firewall_rule!r}...")
        _gcloud(["compute", "firewall-rules", "delete", names.firewall_rule], project=project, check=True)
    else:
        print(f"Firewall rule {names.firewall_rule!r} not found; skipping.")

    https_rule = f"{args.name}-allow-https"
    if _resource_exists(project, ["compute", "firewall-rules", "describe", https_rule]):
        print(f"Deleting firewall rule {https_rule!r}...")
        _gcloud(["compute", "firewall-rules", "delete", https_rule], project=project, check=True)

    print("Delete complete.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy tools/webapp to Google Cloud (smallest GCE instance).")
    ap.add_argument(
        "--gcloud",
        default="",
        help="Path to gcloud binary (optional). If unset, auto-detects and may fall back to snap locations.",
    )
    ap.add_argument("--project", default=None, help="GCP project id (defaults to `gcloud config get-value project`).")
    ap.add_argument("--account-email", default=DEFAULT_ACCOUNT_EMAIL, help="GCP account email (best-effort select).")
    ap.add_argument("--name", default="hm-webapp", help="Resource name prefix (instance/firewall).")
    ap.add_argument("--zone", default="us-central1-a", help="Compute Engine zone.")
    ap.add_argument("--network", default="default", help="VPC network name (default: 'default').")
    ap.add_argument("--machine-type", default="e2-micro", help="Smallest/recommended: e2-micro.")
    ap.add_argument("--image-family", default="debian-12", help="VM image family.")
    ap.add_argument("--image-project", default="debian-cloud", help="VM image project.")
    ap.add_argument("--boot-disk-size", default="10GB", help="Boot disk size, e.g. 10GB.")

    ap.add_argument("--watch-root", default="/data/incoming", help="Upload/watch directory.")
    ap.add_argument("--app-port", type=int, default=8008, help="Local gunicorn port (nginx listens on 80).")
    ap.add_argument(
        "--bind-address",
        default="0.0.0.0",
        help="Address to bind the app port on the VM (default: 0.0.0.0 for public access).",
    )
    ap.add_argument(
        "--server-name",
        default="AUTO",
        help="nginx server_name; use AUTO to set to the instance external IP, or '_' for default_server.",
    )

    ap.add_argument("--db-name", default="hm_app_db")
    ap.add_argument("--db-user", default="hmapp")
    ap.add_argument("--db-pass", default="hmapp_pass")
    ap.add_argument(
        "--import-token",
        default="",
        help="If set, require this bearer token for /api/import/* endpoints (send via Authorization: Bearer ...).",
    )

    ap.add_argument("--enable-https", action="store_true", help="Enable HTTPS via certbot (requires DNS pointing at VM).")
    ap.add_argument("--domains", default="", help="Comma-separated domains for the cert, e.g. example.com,www.example.com")
    ap.add_argument("--https-email", default="", help="Email address for Let's Encrypt registration.")

    ap.add_argument("--ssh-timeout-s", type=int, default=300, help="Seconds to wait for SSH readiness.")
    ap.add_argument("--delete", action="store_true", help="Delete the created resources instead of deploying.")
    args = ap.parse_args()

    global GCLOUD_BIN
    if args.gcloud:
        GCLOUD_BIN = args.gcloud
    else:
        GCLOUD_BIN = _resolve_gcloud_bin()
    _require_gcloud()

    project = args.project or _default_project()
    if not project:
        print("Error: missing --project and no default gcloud project is set.", file=sys.stderr)
        print("Fix by running:", file=sys.stderr)
        print("  gcloud config set project <PROJECT_ID>", file=sys.stderr)
        return 2
    args.project = project

    names = GcpNames(
        instance=args.name,
        firewall_rule=f"{args.name}-allow-http",
        network_tag=args.name,
    )

    try:
        if args.delete:
            _delete(args, names)
        else:
            _deploy(args, names)
    except subprocess.CalledProcessError as e:
        stderr = getattr(e, "stderr", None)
        if stderr:
            print(stderr, file=sys.stderr)
        return e.returncode or 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

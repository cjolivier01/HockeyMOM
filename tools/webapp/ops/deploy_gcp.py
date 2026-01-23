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
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACCOUNT_EMAIL = "cjolivier01@gmail.com"
DEFAULT_DOMAIN_CANDIDATES = ("californiahockey.org", "californiayouthhockey.org")
GCLOUD_FALLBACKS = (
    "/snap/google-cloud-sdk/current/bin/gcloud",
    "/snap/google-cloud-sdk/632/bin/gcloud",
)
GCLOUD_BIN = "gcloud"

# Do not upload local build artifacts / caches / secrets to the VM.
WEBAPP_UPLOAD_IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    ".DS_Store",
    # Local secret/config state.
    "instance",
    # Common local data artifacts.
    "*.sqlite3",
    "*.db",
    "*.log",
)


@dataclass(frozen=True)
class GcpNames:
    instance: str
    firewall_rule: str
    network_tag: str


def _resolve_gcloud_bin() -> str:
    def _works(path: str) -> bool:
        try:
            cp = subprocess.run(
                [path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
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


def _run(
    cmd: list[str], *, capture: bool = False, check: bool = True, cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        text=True,
        cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
    )


def _gcloud(
    cmd: list[str],
    *,
    project: str | None,
    capture: bool = False,
    check: bool = True,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    base = [GCLOUD_BIN, "--quiet"]
    if project:
        base += ["--project", project]
    return _run(base + cmd, capture=capture, check=check, cwd=cwd)


def _require_gcloud() -> None:
    # Resolved during module init (main).
    return


def _default_project() -> str | None:
    # `gcloud config get-value project` prints "(unset)" if unset.
    try:
        p = _run(
            [GCLOUD_BIN, "config", "get-value", "project"], capture=True, check=True
        ).stdout.strip()
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


def _domain_register_params(project: str, domain: str) -> dict:
    cp = _gcloud(
        ["domains", "registrations", "get-register-parameters", domain, "--format=json"],
        project=project,
        capture=True,
        check=True,
    )
    try:
        return json.loads(cp.stdout or "{}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse gcloud output for domain {domain!r}: {e}") from e


def _domain_yearly_price(params: dict) -> str:
    price = params.get("yearlyPrice") or {}
    currency = price.get("currencyCode") or "USD"
    units = int(price.get("units") or 0)
    nanos = int(price.get("nanos") or 0)
    value = units + nanos / 1_000_000_000
    return f"{value:.2f} {currency}"


def _select_domain(project: str, preferred_domains: list[str]) -> tuple[str, dict]:
    errors: list[str] = []
    for domain in preferred_domains:
        domain = domain.strip()
        if not domain:
            continue
        try:
            params = _domain_register_params(project, domain)
        except Exception as e:
            errors.append(f"{domain}: {e}")
            continue
        if (params.get("availability") or "").upper() == "AVAILABLE":
            return domain, params
        errors.append(f"{domain}: availability={params.get('availability')!r}")
    msg = "No preferred domains appear to be available."
    if errors:
        msg += "\n" + "\n".join(errors)
    raise RuntimeError(msg)


def _default_cloud_dns_zone_name(domain: str) -> str:
    # Cloud DNS managed-zone name: lower-case letters, digits, hyphen. Keep it simple.
    return domain.lower().replace(".", "-")


def _ensure_cloud_dns_zone(project: str, zone_name: str, domain: str) -> None:
    if _resource_exists(project, ["dns", "managed-zones", "describe", zone_name]):
        return
    dns_name = domain if domain.endswith(".") else f"{domain}."
    _gcloud(
        [
            "dns",
            "managed-zones",
            "create",
            zone_name,
            "--dns-name",
            dns_name,
            "--description",
            f"Public DNS zone for {domain}",
        ],
        project=project,
        check=True,
    )


def _upsert_a_record(project: str, zone_name: str, fqdn: str, ipv4: str, ttl_s: int = 300) -> None:
    fqdn = fqdn if fqdn.endswith(".") else f"{fqdn}."
    existing_cp = _gcloud(
        [
            "dns",
            "record-sets",
            "list",
            "--zone",
            zone_name,
            "--name",
            fqdn,
            "--type",
            "A",
            "--format=json",
        ],
        project=project,
        capture=True,
        check=True,
    )
    try:
        records = json.loads(existing_cp.stdout or "[]")
    except json.JSONDecodeError:
        records = []
    existing_rrdatas: list[str] = []
    existing_ttl: int | None = None
    if records:
        r0 = records[0] or {}
        existing_rrdatas = list(r0.get("rrdatas") or [])
        existing_ttl = r0.get("ttl")

    if existing_rrdatas == [ipv4] and (existing_ttl is None or int(existing_ttl) == ttl_s):
        return

    with tempfile.TemporaryDirectory(prefix="hm-gcp-dns-") as td:
        _gcloud(
            ["dns", "record-sets", "transaction", "start", "--zone", zone_name],
            project=project,
            check=True,
            cwd=td,
        )
        if existing_rrdatas:
            _gcloud(
                [
                    "dns",
                    "record-sets",
                    "transaction",
                    "remove",
                    "--zone",
                    zone_name,
                    "--name",
                    fqdn,
                    "--type",
                    "A",
                    "--ttl",
                    str(existing_ttl or ttl_s),
                    *existing_rrdatas,
                ],
                project=project,
                check=True,
                cwd=td,
            )
        _gcloud(
            [
                "dns",
                "record-sets",
                "transaction",
                "add",
                "--zone",
                zone_name,
                "--name",
                fqdn,
                "--type",
                "A",
                "--ttl",
                str(ttl_s),
                ipv4,
            ],
            project=project,
            check=True,
            cwd=td,
        )
        _gcloud(
            ["dns", "record-sets", "transaction", "execute", "--zone", zone_name],
            project=project,
            check=True,
            cwd=td,
        )


def _register_domain(
    project: str,
    domain: str,
    *,
    contact_yaml: str,
    contact_privacy: str,
    cloud_dns_zone: str,
    validate_only: bool,
) -> None:
    params = _domain_register_params(project, domain)
    if (params.get("availability") or "").upper() != "AVAILABLE":
        raise RuntimeError(
            f"Domain {domain!r} is not available (availability={params.get('availability')!r})."
        )
    yearly_price = _domain_yearly_price(params)
    cmd = [
        "domains",
        "registrations",
        "register",
        domain,
        "--contact-data-from-file",
        contact_yaml,
        "--contact-privacy",
        contact_privacy,
        "--yearly-price",
        yearly_price,
        "--cloud-dns-zone",
        cloud_dns_zone,
    ]
    if validate_only:
        cmd += ["--validate-only"]
    _gcloud(cmd, project=project, check=True)


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
            raise RuntimeError(
                f"Timed out waiting for SSH to {instance} in {zone}. Last error:\n{stderr}"
            )
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
        print(
            f"Note: active account is {active!r} (requested {args.account_email!r}).",
            file=sys.stderr,
        )

    print("Enabling Compute Engine API (best-effort)...")
    _gcloud(["services", "enable", "compute.googleapis.com"], project=project, check=False)
    wants_domains = (
        args.domain_register
        or args.domain_configure_dns
        or (args.enable_https and args.domains.strip().upper() == "AUTO")
    )
    if wants_domains:
        print("Enabling Cloud Domains + Cloud DNS APIs (best-effort)...")
        _gcloud(["services", "enable", "domains.googleapis.com"], project=project, check=False)
        _gcloud(["services", "enable", "dns.googleapis.com"], project=project, check=False)

    if not _resource_exists(
        project, ["compute", "firewall-rules", "describe", names.firewall_rule]
    ):
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

    if not _resource_exists(
        project, ["compute", "instances", "describe", names.instance, "--zone", zone]
    ):
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
            "rm -rf /tmp/hm/tools/webapp && mkdir -p /tmp/hm/tools",
        ],
        project=project,
        check=True,
    )

    print("Copying `tools/webapp` to the VM (excluding caches/bytecode)...")
    repo_root = Path(__file__).resolve().parents[3]
    src_webapp_dir = repo_root / "tools" / "webapp"
    if not src_webapp_dir.exists():
        raise RuntimeError(f"Missing webapp source dir: {src_webapp_dir}")
    with tempfile.TemporaryDirectory(prefix="hm_webapp_gcp_stage_") as td:
        stage_dir = Path(td) / "webapp"
        shutil.copytree(
            src_webapp_dir,
            stage_dir,
            ignore=shutil.ignore_patterns(*WEBAPP_UPLOAD_IGNORE_PATTERNS),
        )
        _gcloud(
            [
                "compute",
                "scp",
                "--recurse",
                str(stage_dir),
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
            raise RuntimeError(
                "--enable-https requires --domains "
                "(e.g. --domains jrsharks2013.org,www.jrsharks2013.org or --domains AUTO)"
            )
        if not args.https_email:
            raise RuntimeError("--enable-https requires --https-email")
        preferred = [d.strip() for d in args.domains_auto_candidates.split(",") if d.strip()]
        if not preferred:
            preferred = list(DEFAULT_DOMAIN_CANDIDATES)

        if args.domains.strip().upper() == "AUTO":
            apex, params = _select_domain(project, preferred)
            domains = [apex, f"www.{apex}"]
            print(f"Selected domain: {apex} (yearly price: {_domain_yearly_price(params)})")
        else:
            domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if not domains:
            raise RuntimeError("--domains must not be empty")

        apex_domain = next((d for d in domains if not d.lower().startswith("www.")), domains[0])
        cloud_dns_zone = args.domain_cloud_dns_zone or _default_cloud_dns_zone_name(apex_domain)

        if args.domain_register:
            if not args.domain_contact_yaml:
                raise RuntimeError("--domain-register requires --domain-contact-yaml")
            print(f"Registering domain (Cloud Domains): {apex_domain}")
            _ensure_cloud_dns_zone(project, cloud_dns_zone, apex_domain)
            _register_domain(
                project,
                apex_domain,
                contact_yaml=args.domain_contact_yaml,
                contact_privacy=args.domain_contact_privacy,
                cloud_dns_zone=cloud_dns_zone,
                validate_only=args.domain_register_validate_only,
            )

        if args.domain_configure_dns:
            ip_for_dns = _instance_external_ip(project, names.instance, zone)
            if not ip_for_dns:
                raise RuntimeError(
                    "Could not determine instance external IP for DNS A record setup."
                )
            print(f"Configuring Cloud DNS A records in zone {cloud_dns_zone!r} -> {ip_for_dns} ...")
            _ensure_cloud_dns_zone(project, cloud_dns_zone, apex_domain)
            _upsert_a_record(project, cloud_dns_zone, apex_domain, ip_for_dns)
            _upsert_a_record(project, cloud_dns_zone, f"www.{apex_domain}", ip_for_dns)

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
        ensure_nginx_cmd = textwrap.dedent(
            """
            set -euo pipefail

            sudo python3 - <<'PY'
            from __future__ import annotations

            import re
            from pathlib import Path

            path = Path("/etc/nginx/sites-available/hm-webapp")
            text = path.read_text(encoding="utf-8")
            changed = False

            # Certbot may generate a TLS server block that omits /static; patch it back in.
            if "location /static/" not in text:
                m = re.search(r"(?m)^(?P<indent>[ \\t]*)location\\s+/\\s*\\{", text)
                if m:
                    indent = m.group("indent")
                    block = (
                        f"{indent}location /static/ {{\\n"
                        f"{indent}    alias /opt/hm-webapp/app/static/;\\n"
                        f"{indent}}}\\n\\n"
                    )
                    text = text[: m.start()] + block + text[m.start() :]
                    changed = True

            directives = [
                ("proxy_connect_timeout", "60s"),
                ("proxy_send_timeout", "600s"),
                ("proxy_read_timeout", "600s"),
            ]
            missing = [
                (name, value)
                for name, value in directives
                if re.search(rf"(?m)^\\s*{re.escape(name)}\\b", text) is None
            ]
            if missing:
                m = re.search(
                    r"(?m)^(?P<indent>[ \\t]*)proxy_pass\\s+http://127\\.0\\.0\\.1:\\d+;\\s*$",
                    text,
                )
                if m:
                    indent = m.group("indent")
                    block = "\\n".join(f"{indent}{name} {value};" for name, value in missing)
                    text = text[: m.end()] + "\\n" + block + text[m.end() :]
                    changed = True

            if changed:
                path.write_text(text, encoding="utf-8")
            PY

            sudo nginx -t
            sudo systemctl reload nginx
            """
        ).strip()
        _gcloud(
            [
                "compute",
                "ssh",
                names.instance,
                "--zone",
                zone,
                "--command",
                ensure_nginx_cmd,
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
        print(
            f"  gcloud compute instances describe {names.instance} --zone {zone} --project {project}"
        )


def _delete(args: argparse.Namespace, names: GcpNames) -> None:
    project = args.project
    zone = args.zone or _detect_instance_zone(project, names.instance)

    _best_effort_set_account(project, args.account_email)

    if zone and _resource_exists(
        project, ["compute", "instances", "describe", names.instance, "--zone", zone]
    ):
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
        _gcloud(
            ["compute", "firewall-rules", "delete", names.firewall_rule],
            project=project,
            check=True,
        )
    else:
        print(f"Firewall rule {names.firewall_rule!r} not found; skipping.")

    https_rule = f"{args.name}-allow-https"
    if _resource_exists(project, ["compute", "firewall-rules", "describe", https_rule]):
        print(f"Deleting firewall rule {https_rule!r}...")
        _gcloud(["compute", "firewall-rules", "delete", https_rule], project=project, check=True)

    print("Delete complete.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Deploy tools/webapp to Google Cloud (smallest GCE instance)."
    )
    ap.add_argument(
        "--gcloud",
        default="",
        help="Path to gcloud binary (optional). If unset, auto-detects and may fall back to snap locations.",
    )
    ap.add_argument(
        "--project",
        default=None,
        help="GCP project id (defaults to `gcloud config get-value project`).",
    )
    ap.add_argument(
        "--account-email",
        default=DEFAULT_ACCOUNT_EMAIL,
        help="GCP account email (best-effort select).",
    )
    ap.add_argument("--name", default="hm-webapp", help="Resource name prefix (instance/firewall).")
    ap.add_argument("--zone", default="us-central1-a", help="Compute Engine zone.")
    ap.add_argument("--network", default="default", help="VPC network name (default: 'default').")
    ap.add_argument("--machine-type", default="e2-micro", help="Smallest/recommended: e2-micro.")
    ap.add_argument("--image-family", default="debian-12", help="VM image family.")
    ap.add_argument("--image-project", default="debian-cloud", help="VM image project.")
    ap.add_argument("--boot-disk-size", default="10GB", help="Boot disk size, e.g. 10GB.")

    ap.add_argument("--watch-root", default="/data/incoming", help="Upload/watch directory.")
    ap.add_argument(
        "--app-port", type=int, default=8008, help="Local gunicorn port (nginx listens on 80)."
    )
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

    ap.add_argument(
        "--enable-https",
        action="store_true",
        help="Enable HTTPS via certbot (requires DNS pointing at VM).",
    )
    ap.add_argument(
        "--domains",
        default="",
        help="Comma-separated domains for the cert, e.g. jrsharks2013.org,www.jrsharks2013.org (or AUTO).",
    )
    ap.add_argument(
        "--domains-auto-candidates",
        default=",".join(DEFAULT_DOMAIN_CANDIDATES),
        help="Comma-separated domain candidates (used when --domains=AUTO).",
    )
    ap.add_argument(
        "--https-email", default="", help="Email address for Let's Encrypt registration."
    )

    ap.add_argument(
        "--domain-register",
        action="store_true",
        help="If set, register the apex domain via Cloud Domains before enabling HTTPS (requires --domain-contact-yaml).",
    )
    ap.add_argument(
        "--domain-register-validate-only",
        action="store_true",
        help="Validate domain registration args without actually purchasing/registering the domain.",
    )
    ap.add_argument(
        "--domain-contact-yaml",
        default="",
        help="Contact YAML for gcloud domains registrations register (template: tools/webapp/ops/domain_contacts_template.yaml).",
    )
    ap.add_argument(
        "--domain-contact-privacy",
        default="redacted-contact-data",
        help="Contact privacy for the domain registration (e.g. redacted-contact-data, public-contact-data).",
    )
    ap.add_argument(
        "--domain-cloud-dns-zone",
        default="",
        help="Cloud DNS managed-zone name to use (defaults to apex domain with dots replaced).",
    )
    ap.add_argument(
        "--domain-configure-dns",
        dest="domain_configure_dns",
        action="store_true",
        help="If set, create/update Cloud DNS A records for apex + www to point at the VM external IP.",
    )

    ap.add_argument(
        "--ssh-timeout-s", type=int, default=300, help="Seconds to wait for SSH readiness."
    )
    ap.add_argument(
        "--delete", action="store_true", help="Delete the created resources instead of deploying."
    )
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

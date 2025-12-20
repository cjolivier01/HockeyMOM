#!/usr/bin/env python3
import argparse
import subprocess

SERVICE_NAME = "hm-webapp.service"
NGINX_SITE = "/etc/nginx/sites-available/hm-webapp"


def main():
    ap = argparse.ArgumentParser(description="Uninstall HockeyMOM WebApp")
    ap.add_argument("--install-root", default="/opt/hm-webapp")
    args = ap.parse_args()

    print("Stopping services...")
    subprocess.run(["sudo", "systemctl", "disable", "--now", SERVICE_NAME], check=False)

    print("Removing nginx site...")
    subprocess.run(["sudo", "rm", "-f", "/etc/nginx/sites-enabled/hm-webapp"], check=False)
    subprocess.run(["sudo", "rm", "-f", NGINX_SITE], check=False)
    subprocess.run(["sudo", "systemctl", "restart", "nginx"], check=False)

    print("Removing systemd unit...")
    subprocess.run(["sudo", "rm", "-f", f"/etc/systemd/system/{SERVICE_NAME}"], check=False)
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=False)

    print("(Optional) Remove install root with sudo rm -rf", args.install_root)
    print("Uninstall complete.")


if __name__ == "__main__":
    main()

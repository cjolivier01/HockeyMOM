#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: str, check: bool = True, capture: bool = False, env=None):
    print(f"+ {cmd}")
    result = subprocess.run(
        cmd, shell=True, check=check, text=True, capture_output=capture, env=env
    )
    if capture:
        return result.stdout.strip()
    return ""


def require_ubuntu():
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        raise SystemExit("/etc/os-release not found; only Ubuntu is supported.")
    data = os_release.read_text()
    if "ID=ubuntu" not in data:
        raise SystemExit("This installer currently supports Ubuntu only.")


def pkg_install():
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    run("sudo apt-get update -y", env=env)
    # Base Slurm + Munge and commonly needed plugins
    pkgs = [
        "munge",
        "libmunge2",
        "libmunge-dev",
        "slurm-wlm",
        "slurmctld",
        "slurmd",
        "slurm-client",
        # Optional plugins: harmless if unused
        "slurm-wlm-basic-plugins",
    ]
    run("sudo apt-get install -y " + " ".join(map(shlex.quote, pkgs)), env=env)


def ensure_munge():
    # Create munge key if absent
    if not Path("/etc/munge/munge.key").exists():
        # Prefer create-munge-key if available; otherwise fallback to dd
        if shutil.which("create-munge-key"):
            run("sudo create-munge-key")
        elif Path("/usr/sbin/create-munge-key").exists():
            run("sudo /usr/sbin/create-munge-key")
        else:
            run("sudo dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024 status=none")
    run("sudo chown -R munge:munge /etc/munge")
    run("sudo chmod 0700 /etc/munge")
    run("sudo chmod 0400 /etc/munge/munge.key")


def detect_topology():
    # Use slurmd -C to get Sockets/Cores/Threads/RealMemory
    line = run("slurmd -C | head -n1", capture=True)
    def field(name):
        for tok in line.split():
            if tok.startswith(name + "="):
                return tok.split("=", 1)[1]
        return None

    topo = {
        "CPUs": field("CPUs"),
        "Sockets": field("SocketsPerBoard") or field("Sockets"),
        "CoresPerSocket": field("CoresPerSocket"),
        "ThreadsPerCore": field("ThreadsPerCore"),
        "RealMemory": field("RealMemory"),
    }
    # Basic fallback if slurmd -C parsing fails
    for k, v in topo.items():
        if not v:
            raise SystemExit(f"Failed to detect topology field: {k}: from '{line}'")
    return topo


def detect_gpus():
    # Prefer nvidia-smi; otherwise count /dev/nvidia[0-9]*
    count = 0
    if shutil.which("nvidia-smi"):
        try:
            out = run("nvidia-smi --query-gpu=name --format=csv,noheader", capture=True)
            count = len([l for l in out.splitlines() if l.strip()])
        except subprocess.CalledProcessError:
            count = 0
    if count == 0:
        devs = list(Path("/dev").glob("nvidia[0-9]*"))
        count = len(devs)
    return count


def write_configs():
    host = run("hostname -s", capture=True)
    topo = detect_topology()
    gpu_count = detect_gpus()

    # Ensure directories
    run("sudo mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd")
    run("sudo chown -R slurm:slurm /var/spool/slurmctld /var/spool/slurmd")

    slurm_conf = f"""
ClusterName=local
SlurmctldHost={host}

SlurmUser=slurm
SlurmdUser=root
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SwitchType=switch/none
MpiDefault=none
SlurmctldPort=6817
SlurmdPort=6818
ProctrackType=proctrack/cgroup
TaskPlugin=task/affinity,task/cgroup
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/linux

ReturnToService=2
SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
KillWait=30
MinJobAge=300
Waittime=0

GresTypes=gpu

NodeName={host} Sockets={topo['Sockets']} CoresPerSocket={topo['CoresPerSocket']} ThreadsPerCore={topo['ThreadsPerCore']} RealMemory={topo['RealMemory']} Gres=gpu:{gpu_count} State=UNKNOWN
PartitionName=debug Nodes={host} Default=YES MaxTime=INFINITE State=UP
""".strip()
    Path("/tmp/slurm.conf").write_text(slurm_conf + "\n")
    run("sudo install -m 0644 /tmp/slurm.conf /etc/slurm/slurm.conf")

    # Explicit GRES config lines for each /dev/nvidiaX so it works even without NVML
    gres_lines = ["# Explicit GPU device mapping"]
    gpu_indexes = []
    for dev in sorted(Path("/dev").glob("nvidia[0-9]*")):
        name = dev.name
        if name.startswith("nvidia") and name[6:].isdigit():
            gpu_indexes.append(int(name[6:]))
    if gpu_indexes:
        for i in sorted(gpu_indexes):
            gres_lines.append(f"Name=gpu File=/dev/nvidia{i}")
    else:
        # Fallback: leave empty but keep file present
        gres_lines.append("# No /dev/nvidia* devices detected")
    Path("/tmp/gres.conf").write_text("\n".join(gres_lines) + "\n")
    run("sudo install -m 0644 /tmp/gres.conf /etc/slurm/gres.conf")

    cgroup_conf = """
CgroupPlugin=cgroup/v2
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
ConstrainSwapSpace=no
""".strip()
    Path("/tmp/cgroup.conf").write_text(cgroup_conf + "\n")
    run("sudo install -m 0644 /tmp/cgroup.conf /etc/slurm/cgroup.conf")

    run("sudo chown -R root:root /etc/slurm")
    run("sudo chmod 0644 /etc/slurm/*.conf")


def start_services():
    run("sudo systemctl daemon-reload || true")
    run("sudo systemctl enable --now munge")
    run("sleep 1")
    run("sudo systemctl enable --now slurmctld")
    run("sleep 1")
    run("sudo systemctl enable --now slurmd")


def resume_node():
    host = run("hostname -s", capture=True)
    # Clear previous drain reasons and resume
    run(f"sudo scontrol update nodename={host} state=resume reason=reset", check=False)


def verify():
    host = run("hostname -s", capture=True)
    out = run("sinfo -Nel", capture=True)
    print(out)
    # Check that node is listed and not down/drained
    if host not in out:
        raise SystemExit("Node not visible in sinfo output.")
    if any(s in out for s in ["down", "drain", "drained", "invalid", "inval"]):
        print(out)
        raise SystemExit("Node not ready; check configuration and logs.")
    # Check GRES assignment presence
    sctl = run(f"scontrol show node {host}", capture=True)
    if "Gres=gpu:" not in sctl:
        print(sctl)
        raise SystemExit("GPU GRES not visible on node.")
    # Try a quick allocation if GPU present
    if "Gres=gpu:" in sctl:
        try:
            print("Attempting srun nvidia-smi -L ...")
            test = run("srun --gres=gpu:1 --ntasks=1 --nodes=1 nvidia-smi -L", capture=True)
            print(test)
        except subprocess.CalledProcessError:
            print("srun test failed; GPU might be busy or a prior job is running.")


def install():
    require_ubuntu()
    pkg_install()
    ensure_munge()
    write_configs()
    start_services()
    resume_node()
    verify()


def uninstall():
    # Stop services first
    run("sudo systemctl disable --now slurmd || true", check=False)
    run("sudo systemctl disable --now slurmctld || true", check=False)
    run("sudo systemctl disable --now munge || true", check=False)
    # Purge packages
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    run(
        "sudo apt-get purge -y 'slurm-*' 'munge*' 'libmunge*' || true",
        check=False,
        env=env,
    )
    run("sudo apt-get autoremove -y || true", check=False, env=env)
    # Remove config and state
    for p in [
        "/etc/slurm",
        "/var/spool/slurmd",
        "/var/spool/slurmctld",
        "/var/log/slurm",
        "/etc/munge",
        "/var/lib/munge",
    ]:
        run(f"sudo rm -rf {shlex.quote(p)} || true", check=False)
    print("Uninstall complete.")


def main():
    ap = argparse.ArgumentParser(description="Install/Uninstall single-node Slurm with GPU GRES (Ubuntu)")
    ap.add_argument("action", choices=["install", "uninstall", "verify"], help="Action to perform")
    args = ap.parse_args()
    if args.action == "install":
        install()
    elif args.action == "verify":
        verify()
    else:
        uninstall()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_default_tag(repo_root: Path) -> str:
    tag_path = repo_root / "env" / "tag"
    if tag_path.is_file():
        tag = tag_path.read_text(encoding="utf-8").strip()
        if tag:
            return tag
    return "hm-cuda"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, env=env, check=True)


def _docker_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("DOCKER_BUILDKIT", "1")
    return env


def _docker_bridge_interface_exists() -> bool:
    # Docker's default bridge network typically uses an interface named "docker0".
    # If that interface is missing, `docker run`/`docker build` with bridge networking
    # can fail with "adding interface veth... to bridge docker0 failed: Device does not exist".
    return Path("/sys/class/net/docker0").exists()


def _default_network() -> str:
    return "bridge" if _docker_bridge_interface_exists() else "host"


def cmd_build(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    tag = args.tag or _read_default_tag(repo_root)

    build_cmd = [
        "docker",
        "build",
        "--network",
        args.network,
        "-f",
        str(repo_root / "env" / "Dockerfile"),
        "-t",
        tag,
        "--build-arg",
        f"USERNAME={args.username}",
        "--build-arg",
        f"UID={args.uid}",
        "--build-arg",
        f"GID={args.gid}",
        "--build-arg",
        f"CUDA_BASE={args.cuda_base}",
        "--build-arg",
        f"TORCH_INDEX_URL={args.torch_index_url}",
        "--build-arg",
        f"TORCH_VERSION={args.torch_version}",
        "--build-arg",
        f"TORCHVISION_VERSION={args.torchvision_version}",
        "--build-arg",
        f"TORCHAUDIO_VERSION={args.torchaudio_version}",
        str(repo_root),
    ]
    _run(build_cmd, env=_docker_env())


def cmd_run(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    tag = args.tag or _read_default_tag(repo_root)

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--gpus",
        "all",
        "--network",
        args.network,
        "--shm-size",
        args.shm_size,
    ]

    if args.name:
        docker_cmd += ["--name", args.name]

    if args.videos_mount:
        host_videos = Path(args.videos_mount).expanduser().resolve()
        container_videos = f"/home/{args.username}/Videos"
        docker_cmd += ["-v", f"{host_videos}:{container_videos}:rw"]

    if args.dev_mount:
        docker_cmd += ["-v", f"{repo_root}:/workspace/hm:rw", "-w", "/workspace/hm"]
        docker_cmd += ["-e", "PYTHONPATH=/workspace/hm:/workspace/hm/src"]

    cmd = args.command if args.command else ["bash"]
    docker_cmd.append(tag)
    docker_cmd += cmd

    _run(docker_cmd)


def main(argv: list[str]) -> int:
    repo_root = _repo_root()
    default_tag = _read_default_tag(repo_root)
    default_network = _default_network()

    parser = argparse.ArgumentParser(
        prog="hm_cuda_container.py",
        description=(
            "Build and run a CUDA Docker image with HockeyMOM fully installed (Bazel-built wheels). "
            "By default, only $HOME/Videos is mounted into the container."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tag", default=default_tag, help=f"Docker tag (default: {default_tag})")
    common.add_argument("--username", default=os.environ.get("USER", "hm"), help="Username in image")
    common.add_argument("--uid", type=int, default=os.getuid(), help="UID for user in image")
    common.add_argument("--gid", type=int, default=os.getgid(), help="GID for user in image")

    build = subparsers.add_parser("build", parents=[common], help="Build the CUDA image")
    build.add_argument(
        "--network",
        default=default_network,
        help=f"Docker networking mode for build containers (default: {default_network})",
    )
    build.add_argument(
        "--cuda-base",
        default="nvidia/cuda:12.4.1-devel-ubuntu22.04",
        help="Base CUDA image (must include nvcc for CUDA builds)",
    )
    build.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu124")
    build.add_argument("--torch-version", default="2.4.1")
    build.add_argument("--torchvision-version", default="0.19.1")
    build.add_argument("--torchaudio-version", default="2.4.1")
    build.set_defaults(func=cmd_build)

    run = subparsers.add_parser("run", parents=[common], help="Run the CUDA container with GPU access")
    run.add_argument(
        "--network",
        default=default_network,
        help=f"Docker networking mode for the container (default: {default_network})",
    )
    run.add_argument("--name", default=None, help="Optional container name")
    run.add_argument(
        "--videos-mount",
        default=str(Path.home() / "Videos"),
        help="Host videos directory to mount into /home/<user>/Videos (default: ~/Videos)",
    )
    run.add_argument(
        "--no-videos-mount",
        dest="videos_mount",
        action="store_const",
        const=None,
        help="Do not mount any host videos directory",
    )
    run.add_argument(
        "--dev-mount",
        action="store_true",
        help="Bind-mount this repo into /workspace/hm for development (overrides PYTHONPATH)",
    )
    run.add_argument("--shm-size", default="2g", help="Shared memory size (default: 2g)")
    run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        default=["bash"],
        help="Command to run in the container (default: bash)",
    )
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

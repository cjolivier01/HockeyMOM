#!/usr/bin/env python3
import argparse
import glob
import os
import shutil

from hmlib.log import get_logger


logger = get_logger(__name__)


def find_wheel(name: str):
    # Look for wheel in various possible locations
    patterns = [
        f"{name}-*.whl",
        f"*/{name}-*.whl",
        f"*/*/{name}-*.whl",
        f"bazel-out/*/bin/{name}/{name}-*.whl",
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]

    # Check runfiles
    runfiles_dir = os.environ.get("RUNFILES_DIR", "")
    if runfiles_dir:
        for pattern in patterns:
            files = glob.glob(os.path.join(runfiles_dir, pattern))
            if files:
                return files[0]

    return None


def main(name: str):
    wheel_path = find_wheel(name=name)
    if not wheel_path:
        logger.error("Error: Could not find wheel file")
        return 1

    # Get workspace root
    workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    dist_dir = os.path.join(workspace_root, "dist")

    # Create dist directory
    os.makedirs(dist_dir, exist_ok=True)

    # Copy wheel
    wheel_name = os.path.basename(wheel_path)
    dest_path = os.path.join(dist_dir, wheel_name)
    shutil.copy2(wheel_path, dest_path)
    os.chmod(dest_path, 0o644)
    # Remove execute flag
    logger.info("Installed %s to %s/", wheel_name, dist_dir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", type=str, default="hmlib", help="Package file name of the wheel file")
    args = parser.parse_args()
    exit(main(args.name))

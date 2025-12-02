#!/usr/bin/env python3
import os
import sys
import sysconfig


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())

    # Prefer SOABI passed from Bazel (matches the built extension),
    # fall back to the runtime Python's SOABI if not provided.
    soabi = sys.argv[1] if len(sys.argv) > 1 else sysconfig.get_config_var("SOABI")
    if not soabi:
        print("Error: SOABI is not defined for this Python", file=sys.stderr)
        return 1

    so_name = f"_hockeymom.{soabi}.so"

    # Location of the built extension as seen from the workspace.
    src = os.path.join(workspace, "bazel-bin", "hockeymom", so_name)
    if not os.path.exists(src):
        print(f"Error: built extension not found at {src}", file=sys.stderr)
        return 1

    pkg_dir = os.path.join(workspace, "hockeymom")
    os.makedirs(pkg_dir, exist_ok=True)
    dst = os.path.join(pkg_dir, so_name)

    # Remove any existing file so we can replace it with a symlink.
    try:
        if os.path.lexists(dst):
            os.remove(dst)
    except OSError as exc:  # pragma: no cover - best-effort cleanup
        print(f"Warning: could not remove existing {dst}: {exc}", file=sys.stderr)

    # Create a relative symlink so it remains valid if the workspace is moved.
    rel_src = os.path.relpath(src, pkg_dir)
    os.symlink(rel_src, dst)
    print(f"Linked {dst} -> {rel_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

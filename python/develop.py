#!/usr/bin/env python3
"""
Install jetson_utils in editable/develop mode (similar to ``python setup.py develop``).

Adds a .pth pointing to the source tree and the built extension directory so imports
resolve without copying files.
"""

import argparse
import os
import sys
import sysconfig
from pathlib import Path
from textwrap import dedent


def _find_workspace_root() -> Path:
    env_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "WORKSPACE").exists() or (parent / "WORKSPACE.bazel").exists():
            return parent
    return Path.cwd().resolve()


def _find_extension_dir(workspace: Path) -> Path | None:
    candidates = []
    candidates.extend(workspace.glob("bazel-bin/python/bindings/jetson_utils_python.*.so"))
    candidates.extend(workspace.glob("bazel-out/**/python/bindings/jetson_utils_python.*.so"))
    for match in candidates:
        if match.is_file():
            return match.parent.resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Install jetson_utils in develop/editable mode.")
    parser.add_argument("--workspace", type=Path, default=None, help="Override the workspace root.")
    args = parser.parse_args()

    workspace = (args.workspace or _find_workspace_root()).resolve()
    pkg_root = workspace / "python" / "python"
    ext_dir = _find_extension_dir(workspace)

    site_packages = Path(sysconfig.get_paths()["purelib"])
    site_packages.mkdir(parents=True, exist_ok=True)
    pth_path = site_packages / "jetson_utils-development.pth"

    lines = [str(pkg_root)]
    if ext_dir:
        lines.append(str(ext_dir))
    pth_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    scripts_dir = Path(sysconfig.get_paths()["scripts"])
    scripts_dir.mkdir(parents=True, exist_ok=True)
    launcher = scripts_dir / "jetson_utils_develop_info"
    launcher.write_text(
        dedent(
            f"""\
            #!{sys.executable}
            import sys
            print("jetson_utils develop paths:")
            print("  package root:", r"{pkg_root}")
            print("  extension dir:", r"{ext_dir if ext_dir else 'not found'}")
            """
        ),
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    print(f"Added {pth_path}")
    if not ext_dir:
        print("Warning: built extension not found; run `bazel build //python/bindings:jetson_utils_python_ext`")
    else:
        print(f"Extension directory added: {ext_dir}")
    print(f"Helper script installed: {launcher}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

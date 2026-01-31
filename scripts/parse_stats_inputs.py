#!/usr/bin/env python3
"""
Back-compat shim.

The shift spreadsheet parser lives in the HockeyMOMWeb repo now:
  ../HockeyMOMWeb/scripts/parse_stats_inputs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    hm_root = Path(__file__).resolve().parents[1]
    new_repo_root = hm_root.parent / "HockeyMOMWeb"
    new_script = new_repo_root / "scripts" / "parse_stats_inputs.py"
    if new_script.is_file():
        os.execv(sys.executable, [sys.executable, str(new_script), *sys.argv[1:]])

    raise SystemExit(
        "ERROR: HockeyMOMWeb not found.\n\n"
        "This script moved out of the `hm` repo. Clone HockeyMOMWeb next to it:\n"
        f"  {new_repo_root}\n\n"
        "Then run:\n"
        "  ../HockeyMOMWeb/scripts/parse_stats_inputs.py --help\n"
    )


if __name__ == "__main__":
    main()

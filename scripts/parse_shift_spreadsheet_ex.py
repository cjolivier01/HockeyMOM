#!/usr/bin/env python3
"""
Shim runner for the extended shift spreadsheet parser (event-log + per-player).

Executes the logic in "scripts/parse_shift_spreadsheet ex.py" but provides a
space-free filename for convenience.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:  # pragma: no cover - thin wrapper
    here = Path(__file__).resolve()
    target = here.with_name("parse_shift_spreadsheet ex.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()


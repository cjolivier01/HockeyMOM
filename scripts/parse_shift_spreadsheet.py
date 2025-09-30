#!/usr/bin/env python3
"""
This tool has moved. Please use the packaged CLI instead:

  hmanalyze_shifts --input <xls/xlsx> [--t2s <game-id>] [...]

Backwards-compatible wrapper to invoke hmlib.cli.analyze_shifts.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from hmlib.cli.analyze_shifts import main as _main
    except Exception as e:  # noqa: BLE001
        print(
            "Failed to import hmlib.cli.analyze_shifts. Ensure hmlib is on PYTHONPATH.",
            file=sys.stderr,
        )
        raise
    _main()


if __name__ == "__main__":
    main()


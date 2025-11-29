#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sample processing job for a directory")
    ap.add_argument("directory", help="Directory to process")
    ap.add_argument("--sleep", type=float, default=1.0, help="Simulated processing time (seconds)")
    args = ap.parse_args(argv)

    d = Path(args.directory)
    if not d.exists() or not d.is_dir():
        print(f"Directory does not exist: {d}", file=sys.stderr)
        return 2

    print(f"[process_dir] Starting on {d}")
    print(f"[process_dir] WATCH_DIR env: {os.environ.get('WATCH_DIR')}")
    # If GPU is present, optionally show it (no failure if missing)
    try:
        import subprocess

        subprocess.run(["nvidia-smi", "-L"], check=False)
    except Exception:
        pass

    # Simulate work
    time.sleep(args.sleep)

    # Write a result file in the directory
    (d / "_PROCESSED").write_text("ok\n")
    print("[process_dir] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

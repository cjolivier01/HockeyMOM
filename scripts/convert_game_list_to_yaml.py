#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _read_tokens(txt_path: Path) -> list[str]:
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip().lstrip("\ufeff")
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Convert parse_stats_inputs --file-list .txt to .yaml")
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Input .txt file list path")
    p.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=None,
        help="Output .yaml path (default: <in>.yaml)",
    )
    args = p.parse_args()

    inp: Path = args.inp.expanduser()
    out: Path = args.out.expanduser() if args.out else inp.with_suffix(".yaml")

    tokens = _read_tokens(inp)
    payload = {"games": tokens}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

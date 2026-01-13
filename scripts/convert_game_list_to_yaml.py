#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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


def _parse_t2s_spec(raw: str) -> tuple[int, str | None, str | None] | None:
    """
    Parse a TimeToScore-only token like:
      - t2s=51602
      - t2s=51602:HOME
      - t2s=51602:HOME:stockton-r2
    """
    m = re.match(r"(?i)^\\s*t2s\\s*=\\s*(\\d+)(.*)$", str(raw or ""))
    if not m:
        return None
    t2s_id = int(m.group(1))
    rest = (m.group(2) or "").strip()
    if rest.startswith(":"):
        rest = rest[1:]
    parts = [p.strip() for p in rest.split(":")] if rest else []
    side: str | None = None
    label: str | None = None
    if parts:
        first = parts[0].upper()
        if first in {"HOME", "AWAY"}:
            side = first
            label = ":".join(parts[1:]).strip() if len(parts) > 1 else None
        else:
            label = ":".join(parts).strip()
    if label == "":
        label = None
    return t2s_id, side, label


def _parse_inline_meta_from_colons(token: str) -> tuple[str, str | None, dict[str, str]]:
    """
    Parse a legacy non-t2s token with optional ':HOME' / ':AWAY' suffix and optional
    inline ':key=value' segments (rare; back-compat).
    """
    raw = str(token or "").strip()
    side: str | None = None
    meta: dict[str, str] = {}
    if ":" in raw:
        parts = raw.split(":")
        while len(parts) > 1 and "=" in parts[-1]:
            k, v = parts[-1].split("=", 1)
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv and kk not in meta:
                meta[kk] = vv
            parts.pop()
        if len(parts) > 1 and parts[-1].upper() in {"HOME", "AWAY"}:
            side = parts[-1].upper()
            parts.pop()
        raw = ":".join(parts)
    return raw, side, meta


def _parse_legacy_file_list_line(line: str) -> dict:
    """
    Convert one legacy `--file-list` line (text file) into a structured YAML mapping entry.

    Supported legacy syntax:
    - /path/to/stats[:HOME|:AWAY][|key=value|key=value...]
    - t2s=<id>[:HOME|:AWAY][:label][|key=value|...]
    """
    parts = [p.strip() for p in str(line or "").split("|") if p.strip()]
    token = parts[0] if parts else ""
    meta: dict[str, str] = {}
    for seg in parts[1:]:
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        kk = str(k or "").strip()
        vv = str(v or "").strip()
        if kk and vv:
            meta[kk] = vv

    t2s_parsed = _parse_t2s_spec(token)
    if t2s_parsed is not None:
        t2s_id, side, label = t2s_parsed
        out: dict[str, object] = {"t2s": int(t2s_id)}
        if side:
            out["side"] = side
        if label:
            out["label"] = label
        if meta:
            out["metadata"] = meta
        return out

    path, side, inline_meta = _parse_inline_meta_from_colons(token)
    for k, v in inline_meta.items():
        meta.setdefault(k, v)
    out = {"path": path}
    if side:
        out["side"] = side
    if meta:
        out["metadata"] = meta
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
    payload = {"games": [_parse_legacy_file_list_line(t) for t in tokens]}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

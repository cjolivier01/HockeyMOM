#!/usr/bin/env python3
"""
Post-process shift analysis outputs for Home/Away runs.

Inputs: one or more output directories produced by analyze_shifts.py
        (e.g., ..._home and ..._away)

Outputs (written alongside the first directory):
  - combined_player_stats.tsv: concatenation of per-player stats with a 'side' column
  - player_comparison.tsv: left-join by player of Home vs Away key metrics
  - report.txt: quick summary of GF/GA counts from goals_for/against files per side
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def _read_tsv(p: Path) -> Tuple[List[str], List[List[str]]]:
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [], []
    header = lines[0].split("\t")
    rows: List[List[str]] = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    return header, rows


def _rows_to_map(header: List[str], rows: List[List[str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        m: Dict[str, str] = {}
        for i, col in enumerate(header):
            m[col] = r[i] if i < len(r) else ""
        out.append(m)
    return out


def _count_goals(dirp: Path) -> Tuple[int, int]:
    gf = (
        (dirp / "goals_for.txt").read_text(encoding="utf-8").splitlines()
        if (dirp / "goals_for.txt").exists()
        else []
    )
    ga = (
        (dirp / "goals_against.txt").read_text(encoding="utf-8").splitlines()
        if (dirp / "goals_against.txt").exists()
        else []
    )
    gf = [ln for ln in gf if ln.strip() and not ln.strip().startswith("#")]
    ga = [ln for ln in ga if ln.strip() and not ln.strip().startswith("#")]
    return len(gf), len(ga)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-process Home/Away analyze_shifts outputs.")
    p.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="One or more output directories (e.g., *_home and/or *_away)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    dirs: List[Path] = [d for d in args.dirs if d.is_dir()]
    if not dirs:
        raise SystemExit("No valid directories provided")

    # Use the first directory as output location
    out_base = dirs[0]

    combined_rows: List[Dict[str, str]] = []
    combined_header: List[str] | None = None

    # Load each dir's player_stats.txt
    for d in dirs:
        side = (
            "home"
            if d.name.endswith("_home")
            else ("away" if d.name.endswith("_away") else "unknown")
        )
        stats_path = d / "player_stats.txt"
        if not stats_path.exists():
            continue
        header, rows = _read_tsv(stats_path)
        maps = _rows_to_map(header, rows)
        for m in maps:
            m = dict(m)  # copy
            m["side"] = side
            combined_rows.append(m)
        if combined_header is None:
            combined_header = list(header) + ["side"]

    if not combined_rows or combined_header is None:
        raise SystemExit("No player_stats.txt files found in provided directories")

    # Write combined_player_stats.tsv
    comb_path = out_base / "combined_player_stats.tsv"
    lines = ["\t".join(combined_header)]
    for m in sorted(combined_rows, key=lambda r: (r.get("player", ""), r.get("side", ""))):
        lines.append("\t".join(m.get(c, "") for c in combined_header))
    comb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Build player comparison (Home vs Away) for key metrics
    # Index rows by (player, side)
    idx: Dict[Tuple[str, str], Dict[str, str]] = {}
    for m in combined_rows:
        idx[(m.get("player", ""), m.get("side", ""))] = m

    players = sorted({m.get("player", "") for m in combined_rows if m.get("player")})
    cmp_header = [
        "player",
        "plus_minus_home",
        "plus_minus_away",
        "plus_minus_delta",
        "toi_total_home",
        "toi_total_away",
    ]
    cmp_lines = ["\t".join(cmp_header)]
    for pl in players:
        h = idx.get((pl, "home"), {})
        a = idx.get((pl, "away"), {})
        pm_h = h.get("plus_minus", "")
        pm_a = a.get("plus_minus", "")
        try:
            delta = str(int(pm_h) - int(pm_a)) if pm_h and pm_a else ""
        except Exception:
            delta = ""
        row = [
            pl,
            pm_h,
            pm_a,
            delta,
            h.get("toi_total", ""),
            a.get("toi_total", ""),
        ]
        cmp_lines.append("\t".join(row))
    (out_base / "player_comparison.tsv").write_text("\n".join(cmp_lines) + "\n", encoding="utf-8")

    # Quick report of GF/GA from goals files per side
    report_lines = ["Post-process summary:\n"]
    for d in dirs:
        side = (
            "home"
            if d.name.endswith("_home")
            else ("away" if d.name.endswith("_away") else "unknown")
        )
        gf, ga = _count_goals(d)
        report_lines.append(f"- {d.name} | side={side} | GF={gf} GA={ga}")
    (out_base / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"✅ Wrote: {comb_path}")
    print(f"✅ Wrote: {out_base / 'player_comparison.tsv'}")
    print(f"✅ Wrote: {out_base / 'report.txt'}")


if __name__ == "__main__":
    main()

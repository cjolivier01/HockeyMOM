#!/usr/bin/env python3
"""
Extract shifts and stats from the 'dh-tv-12-1.xls' style sheet.

Outputs per-player:
  - {Jersey}_{Name}_video_times.txt          -> "videoStart videoEnd"
  - {Jersey}_{Name}_scoreboard_times.txt     -> "period scoreboardStart scoreboardEnd"
  - {Jersey}_{Name}_stats.txt                -> scoreboard-time-based stats (TOI, #shifts, avg, etc., + plus/minus)

Goals can be specified via:
  --goal GF:2/13:45 --goal GA:1/05:12 ...
or as a file with lines like:
  GF:1/13:47
  GA:2/09:15
  # comments and blank lines allowed

Install deps (for .xls):
  pip install pandas xlrd

Example:
  python extract_shift_times.py \
      --input dh-tv-12-1.xls \
      --outdir player_shifts \
      --goal GF:1/12:20 --goal GA:2/10:05 \
      --goals-file goals.txt \
      --keep-goalies
"""

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Header labels as they appear in the sheet
LABEL_START_SB = "Shift Start (Scoreboard Time)"
LABEL_END_SB   = "Shift End (Scoreboard Time)"
LABEL_START_V  = "Shift Start (Video Time)"
LABEL_END_V    = "Shift End (Video Time)"


# ----------------------------- utilities -----------------------------

def sanitize_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)

def is_period_label(x: object) -> bool:
    s = str(x).strip()
    return bool(re.fullmatch(r"Period\s+[1-9]\d*", s))


def parse_flex_time_to_seconds(s: str) -> int:
    """
    Accepts H:MM:SS or M:SS or MM:SS.
    Returns total seconds (int).
    """
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + int(sec)
    elif len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + int(sec)
    else:
        raise ValueError(f"Invalid time format '{s}'. Expected M:SS or H:MM:SS.")


def seconds_to_mmss_or_hhmmss(t: int) -> str:
    """Pretty printer for seconds: HH:MM:SS if >= 3600 else M:SS with minutes not zero-padded."""
    if t < 0:
        t = 0
    h = t // 3600
    r = t % 3600
    m = r // 60
    s = r % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"

def seconds_to_hhmmss(t: int) -> str:
    """Always format seconds as HH:MM:SS with zero-padded hours."""
    if t < 0:
        t = 0
    h = t // 3600
    r = t % 3600
    m = r // 60
    s = r % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def forward_fill_header_labels(header_row: pd.Series) -> Dict[str, List[int]]:
    """
    Given a header row with merged-like cells (label then NaNs across its span),
    forward-fill labels across columns to group column indices by label.
    """
    labels_by_col: List[Optional[str]] = []
    current = None
    for c in range(len(header_row)):
        val = header_row.iloc[c]
        if pd.notna(val) and str(val).strip():
            current = str(val).strip()
        labels_by_col.append(current)

    groups: Dict[str, List[int]] = {}
    for idx, lab in enumerate(labels_by_col):
        if not lab:
            continue
        groups.setdefault(lab, []).append(idx)
    return groups


def extract_pairs_from_row(row: pd.Series, start_cols: List[int], end_cols: List[int]) -> List[Tuple[str, str]]:
    """
    From start/end column groups, collect non-empty strings and pair positionally.
    Start/End order in the sheet can be higher->lower or lower->higher; pairing is positional only.
    """
    starts = [str(row[c]).strip() for c in start_cols if pd.notna(row[c]) and str(row[c]).strip()]
    ends = [str(row[c]).strip() for c in end_cols if pd.notna(row[c]) and str(row[c]).strip()]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]


# ----------------------------- parsing sheet -----------------------------

def find_period_blocks(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Returns a list of (period_number, start_row_idx, end_row_idx_exclusive) for each 'Period X' block.
    """
    # Use position-based indexing for robustness: some Excel files may not
    # create a column label "0" even with header=None, so df[0] can KeyError.
    col0 = df.iloc[:, 0]
    idxs = [i for i, v in col0.items() if is_period_label(v)]
    idxs.append(len(df))
    blocks = []
    for i in range(len(idxs) - 1):
        period_num = int(str(df.iloc[idxs[i], 0]).strip().split()[-1])
        blocks.append((period_num, idxs[i], idxs[i + 1]))
    return blocks

def find_header_row(df: pd.DataFrame, start: int, end: int) -> Optional[int]:
    """
    Within [start, end), locate the header row (typically the row with 'Jersey No' in col 0).
    Often pattern is: 'Period', blank, header.
    """
    for r in range(start, min(end, start + 12)):
        if str(df.iloc[r, 0]).strip().lower() == "jersey no":
            return r
    # Fallback (Period + blank + header)
    return start + 2 if start + 2 < end else None


# ----------------------------- goals parsing -----------------------------


class GoalEvent:
    __slots__ = ("kind", "period", "t_str", "t_sec")

    def __init__(self, kind: str, period: int, t_str: str):
        self.kind = kind  # "GF" or "GA"
        self.period = period
        self.t_str = t_str.strip()
        self.t_sec = parse_flex_time_to_seconds(self.t_str)

    def __repr__(self) -> str:
        return f"{self.kind}:{self.period}/{self.t_str}"


def parse_goal_token(token: str) -> GoalEvent:
    """
    Token: GF:2/13:45 or GA:1/05:12 (case-insensitive on GF/GA).
    """
    token = token.strip()
    m = re.fullmatch(r"(?i)(GF|GA)\s*:\s*([1-9]\d*)\s*/\s*([0-9:]+)", token)
    if not m:
        raise ValueError(f"Bad goal token '{token}'. Expected GF:period/time or GA:period/time")
    kind = m.group(1).upper()
    period = int(m.group(2))
    t_str = m.group(3)
    return GoalEvent(kind, period, t_str)


def load_goals(goals_inline: Iterable[str], goals_file: Optional[Path]) -> List[GoalEvent]:
    events: List[GoalEvent] = []
    # Inline
    for tok in goals_inline or []:
        if not tok:
            continue
        events.append(parse_goal_token(tok))
    # File
    if goals_file:
        with goals_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                events.append(parse_goal_token(line))
    return events


# ----------------------------- core processing -----------------------------


def compute_interval_seconds(a: str, b: str) -> Tuple[int, int]:
    """
    Return (lo, hi) in seconds for a scoreboard interval defined by two times (order-agnostic).
    Works even if the scoreboard counts down (e.g., 15:00 -> 09:21) or up (e.g., 11:27 -> 29:54).
    """
    sa, sb = parse_flex_time_to_seconds(a), parse_flex_time_to_seconds(b)
    return (sa, sb) if sa <= sb else (sb, sa)


def interval_contains(t: int, lo: int, hi: int) -> bool:
    return lo <= t <= hi


def fmt_pairs_for_file(pairs: List[Tuple[str, str]]) -> str:
    return "\n".join(f"{a} {b}" for a, b in pairs)


def summarize_shift_lengths_sec(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Given scoreboard time string pairs, compute durations in seconds and summarize.
    """
    lengths = []
    for a, b in pairs:
        lo, hi = compute_interval_seconds(a, b)
        lengths.append(hi - lo)
    if not lengths:
        return {
            "num_shifts": "0",
            "toi_total": "0:00",
            "toi_avg": "0:00",
            "toi_median": "0:00",
            "toi_longest": "0:00",
            "toi_shortest": "0:00",
        }
    return {
        "num_shifts": str(len(lengths)),
        "toi_total": seconds_to_mmss_or_hhmmss(sum(lengths)),
        "toi_avg": seconds_to_mmss_or_hhmmss(int(sum(lengths) / len(lengths))),
        "toi_median": seconds_to_mmss_or_hhmmss(int(statistics.median(lengths))),
        "toi_longest": seconds_to_mmss_or_hhmmss(max(lengths)),
        "toi_shortest": seconds_to_mmss_or_hhmmss(min(lengths)),
    }


def per_period_toi(pairs_by_period: Dict[int, List[Tuple[str, str]]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for period, pairs in pairs_by_period.items():
        total = 0
        for a, b in pairs:
            lo, hi = compute_interval_seconds(a, b)
            total += hi - lo
        out[period] = seconds_to_mmss_or_hhmmss(total)
    return out


def process_sheet(
    xls_path: Path,
    sheet_name: Optional[str],
    outdir: Path,
    keep_goalies: bool,
    goals: List[GoalEvent],
) -> None:
    # If sheet_name is None, pandas returns a dict of DataFrames.
    # We want the first sheet by default, so coerce None -> 0.
    target_sheet = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(xls_path, sheet_name=target_sheet, header=None)
    blocks = find_period_blocks(df)
    if not blocks:
        raise ValueError("No 'Period N' sections found in column A.")

    outdir.mkdir(parents=True, exist_ok=True)

    # Accumulators
    video_pairs_by_player: Dict[str, List[Tuple[str, str]]] = {}
    sb_pairs_by_player: Dict[str, List[Tuple[int, str, str]]] = {}
    # For scoreboard->video conversion: collect mapping segments per period
    # Each segment: (sb_start_sec, sb_end_sec, v_start_sec, v_end_sec)
    conv_segments_by_period: Dict[int, List[Tuple[int, int, int, int]]] = {}

    # Parse each period block
    for period_num, blk_start, blk_end in blocks:
        header_row_idx = find_header_row(df, blk_start, blk_end)
        if header_row_idx is None:
            raise ValueError(f"Could not locate header row for Period {period_num}")
        data_start = header_row_idx + 1

        header = df.iloc[header_row_idx]
        groups = forward_fill_header_labels(header)

        start_sb_cols = groups.get(LABEL_START_SB, [])
        end_sb_cols   = groups.get(LABEL_END_SB, [])
        start_v_cols  = groups.get(LABEL_START_V, [])
        end_v_cols    = groups.get(LABEL_END_V, [])
        for lab, cols in [
            (LABEL_START_SB, start_sb_cols),
            (LABEL_END_SB, end_sb_cols),
            (LABEL_START_V, start_v_cols),
            (LABEL_END_V, end_v_cols),
        ]:
            if not cols:
                raise ValueError(f"Missing header columns for '{lab}' in Period {period_num}")

        # Iterate rows (players) in the block
        for r in range(data_start, blk_end):
            jersey = str(df.iloc[r, 0]).strip()
            name   = str(df.iloc[r, 1]).strip()

            if is_period_label(df.iloc[r, 0]) or (not jersey and not name):
                break
            if not jersey or jersey.lower() == "nan":
                continue
            # Skip goalies like "(G) 37"
            if not keep_goalies and "(" in jersey and ")" in jersey:
                continue

            player_key = f"{sanitize_name(jersey)}_{sanitize_name(name)}"

            video_pairs = extract_pairs_from_row(df.iloc[r], start_v_cols, end_v_cols)
            sb_pairs    = extract_pairs_from_row(df.iloc[r], start_sb_cols, end_sb_cols)

            if video_pairs:
                video_pairs_by_player.setdefault(player_key, []).extend(video_pairs)
            if sb_pairs:
                sb_pairs_by_player.setdefault(player_key, []).extend((period_num, a, b) for a, b in sb_pairs)

            # Build conversion segments for this row where both SB and Video pairs exist positionally
            nseg = min(len(video_pairs), len(sb_pairs))
            for idx in range(nseg):
                sva, svb = video_pairs[idx]
                sba, sbb = sb_pairs[idx]
                try:
                    v1 = parse_flex_time_to_seconds(sva)
                    v2 = parse_flex_time_to_seconds(svb)
                    s1 = parse_flex_time_to_seconds(sba)
                    s2 = parse_flex_time_to_seconds(sbb)
                except Exception:
                    continue
                conv_segments_by_period.setdefault(period_num, []).append((s1, s2, v1, v2))

    # Write per-player times files
    for player_key, v_pairs in video_pairs_by_player.items():
        # Normalize video times to HH:MM:SS
        norm_pairs = []
        for a, b in v_pairs:
            try:
                sa = parse_flex_time_to_seconds(a)
                sb = parse_flex_time_to_seconds(b)
            except Exception:
                # Skip unparseable pairs
                continue
            norm_pairs.append((seconds_to_hhmmss(sa), seconds_to_hhmmss(sb)))
        p = outdir / f"{player_key}_video_times.txt"
        p.write_text("\n".join(f"{a} {b}" for a, b in norm_pairs) + ("\n" if norm_pairs else ""), encoding="utf-8")

        # Create a convenience bash script to run the video clipper for this player
        script_path = outdir / f"clip_{player_key}.sh"
        player_label = player_key.replace("_", " ")
        script_body = """#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT="$1"
OPP="$2"
THIS_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TS_FILE="$THIS_DIR/{player_key}_video_times.txt"
# Parse optional flags
QUICK=0
HQ=0
shift 2 || true
for ARG in "$@"; do
  if [ "$ARG" = "--quick" ] || [ "$ARG" = "-q" ]; then
    QUICK=1
  elif [ "$ARG" = "--hq" ]; then
    HQ=1
  fi
done

EXTRA_FLAGS=()
if [ "$QUICK" -gt 0 ]; then
  EXTRA_FLAGS+=("--quick" "1")
fi
if [ "$HQ" -gt 0 ]; then
  export VIDEO_CLIPPER_HQ=1
fi

python -m hmlib.cli.video_clipper -j {nr_jobs} --input "$INPUT" --timestamps "$TS_FILE" --temp-dir "$THIS_DIR/temp_clips/{player_key}" "{player_label} vs $OPP" "${{EXTRA_FLAGS[@]}}"
""".format(
            nr_jobs=5, player_key=player_key, player_label=player_label
        )
        script_path.write_text(script_body, encoding="utf-8")
        try:
            import os
            os.chmod(script_path, 0o755)
        except Exception:
            pass

    for player_key, sb_list in sb_pairs_by_player.items():
        p = outdir / f"{player_key}_scoreboard_times.txt"
        lines = [f"{period} {a} {b}" for (period, a, b) in sb_list]
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ---------- Stats & Plus/Minus ----------
    # Index goal events by period for quick lookup
    goals_by_period: Dict[int, List[GoalEvent]] = {}
    for ev in goals:
        goals_by_period.setdefault(ev.period, []).append(ev)

    # For each player, compute:
    #  - TOI (total and per period) based on scoreboard times
    #  - #shifts, avg/median/longest/shortest shift
    #  - Plus/Minus (GF = +1, GA = -1) if goal time ∈ any shift interval of that period
    # Collect rows for a consolidated player_stats table
    stats_table_rows: List[Dict[str, str]] = []
    all_periods_seen: set[int] = set()

    for player_key, sb_list in sb_pairs_by_player.items():
        # By period grouping (for per-period TOI & goal checks)
        sb_by_period: Dict[int, List[Tuple[str, str]]] = {}
        for period, a, b in sb_list:
            sb_by_period.setdefault(period, []).append((a, b))

        # Shift stats (overall, scoreboard-based)
        all_pairs = [(a, b) for (_, a, b) in sb_list]
        shift_summary = summarize_shift_lengths_sec(all_pairs)
        per_period_toi_map = per_period_toi(sb_by_period)

        # Plus/Minus
        plus_minus = 0
        counted_gf: List[str] = []
        counted_ga: List[str] = []
        counted_gf_by_period: Dict[int, int] = {}
        counted_ga_by_period: Dict[int, int] = {}
        for period, pairs in sb_by_period.items():
            if period not in goals_by_period:
                continue
            # For each goal in this period, check against each shift pair using the
            # labeled start/end times (not the normalized interval) to handle edge rules.
            for ev in goals_by_period[period]:
                matched = False
                for (a, b) in pairs:
                    a_sec = parse_flex_time_to_seconds(a)
                    b_sec = parse_flex_time_to_seconds(b)
                    lo, hi = (a_sec, b_sec) if a_sec <= b_sec else (b_sec, a_sec)
                    # Inside inclusive range first
                    if not (lo <= ev.t_sec <= hi):
                        continue
                    # Apply edge rules using labeled endpoints:
                    # - GA at exact shift START (time == a) does NOT count.
                    # - GF at exact shift END (time == b) DOES count (already covered by inclusive range).
                    if ev.kind == "GA" and ev.t_sec == a_sec:
                        continue
                    elif ev.kind == "GF" and ev.t_sec == a_sec:
                        continue
                    matched = True
                    break
                if matched:
                    if ev.kind == "GF":
                        plus_minus += 1
                        counted_gf.append(f"P{period}:{ev.t_str}")
                        counted_gf_by_period[period] = counted_gf_by_period.get(period, 0) + 1
                    else:
                        plus_minus -= 1
                        counted_ga.append(f"P{period}:{ev.t_str}")
                        counted_ga_by_period[period] = counted_ga_by_period.get(period, 0) + 1

        # Stats file
        stats_lines = []
        stats_lines.append(f"Player: {player_key}")
        stats_lines.append(f"Shifts (scoreboard): {shift_summary['num_shifts']}")
        stats_lines.append(f"TOI total (scoreboard): {shift_summary['toi_total']}")
        stats_lines.append(f"Avg shift: {shift_summary['toi_avg']}")
        stats_lines.append(f"Median shift: {shift_summary['toi_median']}")
        stats_lines.append(f"Longest shift: {shift_summary['toi_longest']}")
        stats_lines.append(f"Shortest shift: {shift_summary['toi_shortest']}")
        # Per-period TOI
        if per_period_toi_map:
            stats_lines.append("Per-period TOI (scoreboard):")
            for period in sorted(per_period_toi_map.keys()):
                stats_lines.append(f"  Period {period}: {per_period_toi_map[period]}")
        # Plus/Minus
        stats_lines.append(f"Plus/Minus: {plus_minus}")
        if counted_gf:
            stats_lines.append("  GF counted at: " + ", ".join(sorted(counted_gf)))
        if counted_ga:
            stats_lines.append("  GA counted at: " + ", ".join(sorted(counted_ga)))

        # (Optional) interesting extras you can glean:
        # - Shifts per period
        for period, pairs in sorted(sb_by_period.items()):
            stats_lines.append(f"Shifts in Period {period}: {len(pairs)}")

        # Write file
        (outdir / f"{player_key}_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

        # Accumulate consolidated table row (omit explicit goal listings)
        row_map: Dict[str, str] = {
            "player": player_key,
            # Summary
            "shifts": shift_summary["num_shifts"],
            "plus_minus": str(plus_minus),
            # Scoreboard-based shift stats
            "sb_toi_total": shift_summary["toi_total"],
            "sb_avg": shift_summary["toi_avg"],
            "sb_median": shift_summary["toi_median"],
            "sb_longest": shift_summary["toi_longest"],
            "sb_shortest": shift_summary["toi_shortest"],
        }
        # Counted goals (totals only)
        row_map["gf_counted"] = str(len(counted_gf))
        row_map["ga_counted"] = str(len(counted_ga))

        # Video-based total TOI if available
        v_pairs = video_pairs_by_player.get(player_key, [])
        if v_pairs:
            v_sum = 0
            for a, b in v_pairs:
                lo, hi = compute_interval_seconds(a, b)
                v_sum += hi - lo
            row_map["video_toi_total"] = seconds_to_mmss_or_hhmmss(v_sum)
        else:
            row_map["video_toi_total"] = ""

        # Per-period aggregates (scoreboard)
        for period, toi in per_period_toi_map.items():
            row_map[f"P{period}_toi"] = toi
            all_periods_seen.add(period)
        for period, pairs in sb_by_period.items():
            row_map[f"P{period}_shifts"] = str(len(pairs))
        for period, cnt in counted_gf_by_period.items():
            row_map[f"P{period}_GF"] = str(cnt)
            all_periods_seen.add(period)
        for period, cnt in counted_ga_by_period.items():
            row_map[f"P{period}_GA"] = str(cnt)
            all_periods_seen.add(period)

        stats_table_rows.append(row_map)

    # Global summary CSV (optional quick view)
    summary_rows = []
    for player_key, sb_list in sb_pairs_by_player.items():
        all_pairs = [(a, b) for (_, a, b) in sb_list]
        shift_summary = summarize_shift_lengths_sec(all_pairs)
        row = {
            "player": player_key,
            "num_shifts": int(shift_summary["num_shifts"]),
            "toi_total_sec": (
                parse_flex_time_to_seconds(shift_summary["toi_total"]) if ":" in shift_summary["toi_total"] else 0
            ),
            "toi_avg_sec": (
                parse_flex_time_to_seconds(shift_summary["toi_avg"]) if ":" in shift_summary["toi_avg"] else 0
            ),
            "toi_median_sec": (
                parse_flex_time_to_seconds(shift_summary["toi_median"]) if ":" in shift_summary["toi_median"] else 0
            ),
            "toi_longest_sec": (
                parse_flex_time_to_seconds(shift_summary["toi_longest"]) if ":" in shift_summary["toi_longest"] else 0
            ),
            "toi_shortest_sec": (
                parse_flex_time_to_seconds(shift_summary["toi_shortest"]) if ":" in shift_summary["toi_shortest"] else 0
            ),
        }
        summary_rows.append(row)
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(by="player").to_csv(outdir / "summary_stats.csv", index=False)

    # Consolidated player stats text table
    if stats_table_rows:
        # Column groups (logical ordering)
        periods = sorted(all_periods_seen)
        summary_cols = ["player", "shifts", "plus_minus", "gf_counted", "ga_counted"]
        sb_cols = ["sb_toi_total", "sb_avg", "sb_median", "sb_longest", "sb_shortest"]
        video_cols = ["video_toi_total"]
        period_toi_cols = [f"P{p}_toi" for p in periods]
        period_shift_cols = [f"P{p}_shifts" for p in periods]
        period_gf_cols = [f"P{p}_GF" for p in periods]
        period_ga_cols = [f"P{p}_GA" for p in periods]
        cols = summary_cols + sb_cols + video_cols + period_toi_cols + period_shift_cols + period_gf_cols + period_ga_cols

        # Build rows with missing period cols filled as empty
        rows_for_print: List[List[str]] = []
        for r in sorted(stats_table_rows, key=lambda x: x.get("player", "")):
            rows_for_print.append([r.get(c, "") for c in cols])

        # Compute column widths
        widths = [len(c) for c in cols]
        for row in rows_for_print:
            for i, cell in enumerate(row):
                if len(str(cell)) > widths[i]:
                    widths[i] = len(str(cell))

        # Render table
        def fmt_row(values: List[str]) -> str:
            return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

        lines = [fmt_row(cols)]
        lines.append(fmt_row(["-" * w for w in widths]))
        for row in rows_for_print:
            lines.append(fmt_row(row))
        (outdir / "player_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # CSV export with identical columns and sorting
        import csv  # local import to avoid global
        csv_rows = [dict(zip(cols, row)) for row in rows_for_print]
        try:
            pd.DataFrame(csv_rows).to_csv(outdir / "player_stats.csv", index=False, columns=cols)
        except Exception:
            with (outdir / "player_stats.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in csv_rows:
                    w.writerow(r)

    # ---------- Goals window files (optional) ----------
    # If any goals were provided, write goals_for.txt and goals_against.txt
    if goals:
        # Helper: map a scoreboard second to video seconds using row-derived segments
        def map_sb_to_video(period: int, t_sb: int) -> Optional[int]:
            if period not in conv_segments_by_period:
                return None
            segs = conv_segments_by_period[period]
            for (s1, s2, v1, v2) in segs:
                lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
                if lo <= t_sb <= hi and s1 != s2:
                    # Linear interpolation using labeled endpoints (preserve direction)
                    return int(round(v1 + (t_sb - s1) * (v2 - v1) / (s2 - s1)))
            return None

        # Determine observed scoreboard max per period for capping
        max_sb_by_period: Dict[int, int] = {}
        for period, segs in conv_segments_by_period.items():
            mx = 0
            for (s1, s2, _, _) in segs:
                mx = max(mx, s1, s2)
            max_sb_by_period[period] = mx

        gf_lines: List[str] = []
        ga_lines: List[str] = []
        for ev in goals:
            # Cap scoreboard window to [0, max_seen_period]
            sb_max = max_sb_by_period.get(ev.period, None)
            start_sb = ev.t_sec - 30
            end_sb = ev.t_sec + 10
            if sb_max is not None:
                start_sb = max(0, start_sb)
                end_sb = min(sb_max, end_sb)
            else:
                start_sb = max(0, start_sb)

            # Prefer mapping around the goal center time for robust window sizes in video space
            v_center = map_sb_to_video(ev.period, ev.t_sec)
            if v_center is not None:
                v_start = max(0, v_center - 30)
                v_end = v_center + 10
                start_str = seconds_to_hhmmss(v_start)
                end_str = seconds_to_hhmmss(v_end)
            else:
                # Fallback: attempt endpoint-wise mapping; if still not possible, leave as-is
                v_start = map_sb_to_video(ev.period, start_sb)
                v_end = map_sb_to_video(ev.period, end_sb)
                if v_start is not None and v_end is not None:
                    start_str = seconds_to_hhmmss(max(0, v_start))
                    end_str = seconds_to_hhmmss(max(0, v_end))
                else:
                    # Last resort: keep scoreboard times; note these are not converted
                    start_str = seconds_to_hhmmss(max(0, start_sb))
                    end_str = seconds_to_hhmmss(max(0, end_sb))

            # Goals files use video time format (no period prefix), matching *_video_times.txt
            line = f"{start_str} {end_str}"
            if ev.kind == "GF":
                gf_lines.append(line)
            else:
                ga_lines.append(line)

        (outdir / "goals_for.txt").write_text("\n".join(gf_lines) + ("\n" if gf_lines else ""), encoding="utf-8")
        (outdir / "goals_against.txt").write_text("\n".join(ga_lines) + ("\n" if ga_lines else ""), encoding="utf-8")

    # ---------- Aggregate clip launcher ----------
    # Write a convenience script to run all per-player clip scripts.
    # Usage: ./clip_all.sh <input_video> <opposing_team> [--quick|-q] [--hq]
    clip_all_path = outdir / "clip_all.sh"
    clip_all_body = """#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT="$1"
OPP="$2"
shift 2 || true
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$THIS_DIR"/clip_*.sh; do
  [ -x "$s" ] || continue
  echo "Running $s..."
  "$s" "$INPUT" "$OPP" "$@"
done
"""
    clip_all_path.write_text(clip_all_body, encoding="utf-8")
    try:
        import os as _os
        _os.chmod(clip_all_path, 0o755)
    except Exception:
        pass


# ----------------------------- CLI -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-player shifts & stats from an Excel sheet like 'dh-tv-12-1.xls'."
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to input .xls/.xlsx file.")
    p.add_argument("--sheet", "-s", type=str, default=None, help="Worksheet name (default: first sheet).")
    p.add_argument("--outdir", "-o", type=Path, default=Path("player_shifts"), help="Output directory.")
    p.add_argument(
        "--keep-goalies",
        action="store_true",
        help="By default, rows like '(G) 37' are skipped. Use this flag to include them.",
    )
    # Goals
    p.add_argument(
        "--goal",
        "-g",
        action="append",
        default=[],
        help="Goal event token. Example: 'GF:2/13:45' or 'GA:1/05:12'. Can repeat.",
    )
    p.add_argument(
        "--goals-file",
        type=Path,
        default=None,
        help="Path to a text file with one goal per line (GF:period/time or GA:period/time). '#' lines ignored.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    goals = load_goals(args.goal, args.goals_file)
    process_sheet(
        xls_path=args.input,
        sheet_name=args.sheet,
        outdir=args.outdir,
        keep_goalies=args.keep_goalies,
        goals=goals,
    )
    print(f"✅ Done. Wrote per-player files to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

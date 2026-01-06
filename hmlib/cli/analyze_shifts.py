#!/usr/bin/env python3
"""
Analyze shifts and stats from an Excel sheet and optionally enrich with
TimeToScore game data to auto-fill Goals For/Against.

This is a migrated and extended version of scripts/parse_stats_inputs.py
with an additional --t2s option to fetch goals from CAHA TimeToScore.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Optional import of TimeToScore API (available in this repo)
try:  # pragma: no cover - optional at runtime
    from hmlib.time2score import api as t2s_api
except Exception:  # noqa: BLE001
    t2s_api = None  # type: ignore[assignment]


# Header labels as they appear in the sheet
LABEL_START_SB = "Shift Start (Scoreboard Time)"
LABEL_END_SB = "Shift End (Scoreboard Time)"
LABEL_START_V = "Shift Start (Video Time)"
LABEL_END_V = "Shift End (Video Time)"


# ----------------------------- utilities -----------------------------


def sanitize_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)


def is_period_label(x: object) -> bool:
    s = str(x).strip()
    return bool(re.fullmatch(r"Period\s+[1-9]\d*", s))


def _int_floor_seconds_component(sec_str: str) -> int:
    """Return integer seconds, flooring if fractional (e.g., '12.7' -> 12)."""
    try:
        return int(float(sec_str))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid seconds component '{sec_str}': {e}")


def parse_flex_time_to_seconds(s: str) -> int:
    """
    Accepts H:MM:SS(.fff) or M:SS(.fff) or MM:SS(.fff).
    Returns total seconds (int), rounding fractional seconds down.
    """
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + _int_floor_seconds_component(sec)
    elif len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + _int_floor_seconds_component(sec)
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


def extract_pairs_from_row(
    row: pd.Series, start_cols: List[int], end_cols: List[int]
) -> List[Tuple[str, str]]:
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

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}:{self.period}/{self.t_str}"


def parse_goal_token(token: str) -> GoalEvent:
    """
    Token: GF:2/13:45 or GA:1/05:12 (case-insensitive on GF/GA).
    """
    token = token.strip()
    m = re.fullmatch(r"(?i)(GF|GA)\s*:\s*([1-9]\d*)\s*/\s*([0-9:]+(?:\.[0-9]+)?)", token)
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


def _prompt_select_team(game_info: dict) -> str:
    """Prompt user to select 'home' or 'away'. Returns 'home' or 'away'."""
    home_name = (game_info.get("home") or {}).get("team") or {}
    away_name = (game_info.get("away") or {}).get("team") or {}
    home_label = home_name.get("name") if isinstance(home_name, dict) else None
    away_label = away_name.get("name") if isinstance(away_name, dict) else None
    options = [
        f"1) Home: {home_label or 'Unknown'}",
        f"2) Away: {away_label or 'Unknown'}",
    ]
    print("Select your team (for GF/GA classification):")
    for line in options:
        print("  ", line)
    while True:
        choice = input("Enter 1 for Home or 2 for Away: ").strip()
        if choice == "1":
            return "home"
        if choice == "2":
            return "away"
        print("Invalid choice. Please enter 1 or 2.")


def goals_from_t2s(
    game_id: int, *, team_side: Optional[str] = None
) -> Tuple[List[GoalEvent], Optional[str]]:
    """
    Retrieve goals from TimeToScore for a game id and map them to GF/GA based on
    the selected side (home/away).
    """
    if t2s_api is None:
        raise RuntimeError("TimeToScore API not available in this environment")

    info = t2s_api.get_game_details(game_id)
    stats = info.get("stats") or {}
    if not stats:
        print(
            f"Warning: no stats available for game {game_id}; proceeding without T2S goals.",
            file=sys.stderr,
        )
        return [], team_side

    if team_side not in ("home", "away"):
        team_side = _prompt_select_team(info)

    # Collect scoring rows
    home_sc = stats.get("homeScoring") or []
    away_sc = stats.get("awayScoring") or []

    events: List[GoalEvent] = []

    # Helper to ensure period is int and time is mm:ss string (floor fractional seconds)
    def _mk_event(kind: str, period_val, time_val) -> Optional[GoalEvent]:
        if period_val is None or time_val is None:
            return None
        try:
            per = int(str(period_val).strip())
        except Exception:
            return None
        t_str = str(time_val).strip()
        # Normalize fractional seconds by flooring
        try:
            sec = parse_flex_time_to_seconds(t_str)
        except Exception:
            return None
        mm = sec // 60
        ss = sec % 60
        return GoalEvent(kind, per, f"{mm}:{ss:02d}")

    # Home goals are GF if team_side == home else GA
    for row in home_sc:
        ev = _mk_event("GF" if team_side == "home" else "GA", row.get("period"), row.get("time"))
        if ev:
            events.append(ev)
    # Away goals are GF if team_side == away else GA
    for row in away_sc:
        ev = _mk_event("GF" if team_side == "away" else "GA", row.get("period"), row.get("time"))
        if ev:
            events.append(ev)

    # Sort by period then time for determinism
    events.sort(key=lambda e: (e.period, e.t_sec))
    return events, team_side


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
    skip_validation: bool = False,
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

    # Validation settings
    MAX_SHIFT_SECONDS = 30 * 60  # 30 minutes

    # Simple helper to report validation issues
    def _report_validation(
        kind: str, period: int, player_key: str, a: str, b: str, reason: str
    ) -> None:
        print(
            f"[validation] {kind} | Player={player_key} | Period={period} | start='{a}' end='{b}' -> {reason}",
            file=sys.stderr,
        )

    validation_errors = 0

    # Parse each period block
    for period_num, blk_start, blk_end in blocks:
        header_row_idx = find_header_row(df, blk_start, blk_end)
        if header_row_idx is None:
            raise ValueError(f"Could not locate header row for Period {period_num}")
        data_start = header_row_idx + 1

        header = df.iloc[header_row_idx]
        groups = forward_fill_header_labels(header)

        start_sb_cols = groups.get(LABEL_START_SB, [])
        end_sb_cols = groups.get(LABEL_END_SB, [])
        start_v_cols = groups.get(LABEL_START_V, [])
        end_v_cols = groups.get(LABEL_END_V, [])
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
            name = str(df.iloc[r, 1]).strip()

            if is_period_label(df.iloc[r, 0]) or (not jersey and not name):
                break
            if not jersey or jersey.lower() == "nan":
                continue
            # Skip goalies like "(G) 37"
            if not keep_goalies and "(" in jersey and ")" in jersey:
                continue

            player_key = f"{sanitize_name(jersey)}_{sanitize_name(name)}"

            video_pairs = extract_pairs_from_row(df.iloc[r], start_v_cols, end_v_cols)
            sb_pairs = extract_pairs_from_row(df.iloc[r], start_sb_cols, end_sb_cols)

            # ---------------- Validation (per-row) ----------------
            if not skip_validation:
                # Video times: must be strictly increasing and not excessively long
                for va, vb in video_pairs:
                    try:
                        vsa = parse_flex_time_to_seconds(va)
                        vsb = parse_flex_time_to_seconds(vb)
                    except Exception as e:  # noqa: BLE001
                        _report_validation(
                            "VIDEO",
                            period_num,
                            player_key,
                            va,
                            vb,
                            f"unparseable time: {e}",
                        )
                        validation_errors += 1
                        continue
                    if vsa >= vsb:
                        _report_validation(
                            "VIDEO",
                            period_num,
                            player_key,
                            va,
                            vb,
                            "start must be before end (strictly increasing)",
                        )
                        validation_errors += 1
                    dur = vsb - vsa if vsb >= vsa else 0
                    if dur > MAX_SHIFT_SECONDS:
                        _report_validation(
                            "VIDEO",
                            period_num,
                            player_key,
                            va,
                            vb,
                            f"duration {seconds_to_mmss_or_hhmmss(dur)} exceeds limit 30:00",
                        )
                        validation_errors += 1

                # Scoreboard times: allow either count-up or count-down, but must not be equal and not excessively long
                for sa, sb in sb_pairs:
                    try:
                        ssa = parse_flex_time_to_seconds(sa)
                        ssb = parse_flex_time_to_seconds(sb)
                    except Exception as e:  # noqa: BLE001
                        _report_validation(
                            "SCOREBOARD",
                            period_num,
                            player_key,
                            sa,
                            sb,
                            f"unparseable time: {e}",
                        )
                        validation_errors += 1
                        continue
                    if ssa == ssb:
                        _report_validation(
                            "SCOREBOARD",
                            period_num,
                            player_key,
                            sa,
                            sb,
                            "start equals end (zero-length shift)",
                        )
                        validation_errors += 1
                    dur = abs(ssb - ssa)
                    if dur > MAX_SHIFT_SECONDS:
                        _report_validation(
                            "SCOREBOARD",
                            period_num,
                            player_key,
                            sa,
                            sb,
                            f"duration {seconds_to_mmss_or_hhmmss(dur)} exceeds limit 30:00",
                        )
                        validation_errors += 1

            if video_pairs:
                video_pairs_by_player.setdefault(player_key, []).extend(video_pairs)
            if sb_pairs:
                sb_pairs_by_player.setdefault(player_key, []).extend(
                    (period_num, a, b) for a, b in sb_pairs
                )

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
        p.write_text(
            "\n".join(f"{a} {b}" for a, b in norm_pairs) + ("\n" if norm_pairs else ""),
            encoding="utf-8",
        )

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
            nr_jobs=4, player_key=player_key, player_label=player_label
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
                for a, b in pairs:
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

        all_periods_seen.update(sb_by_period.keys())

        row = {
            "player": player_key,
            "num_shifts": shift_summary["num_shifts"],
            "toi_total": shift_summary["toi_total"],
            "toi_avg": shift_summary["toi_avg"],
            "toi_median": shift_summary["toi_median"],
            "toi_longest": shift_summary["toi_longest"],
            "toi_shortest": shift_summary["toi_shortest"],
            "plus_minus": str(plus_minus),
        }
        # Include per-period TOI columns
        for period in sorted(sb_by_period.keys()):
            row[f"toi_p{period}"] = per_period_toi_map.get(period, "0:00")
            row[f"gf_p{period}"] = str(counted_gf_by_period.get(period, 0))
            row[f"ga_p{period}"] = str(counted_ga_by_period.get(period, 0))

        stats_table_rows.append(row)

    # Write consolidated stats file
    # Use a stable column order
    base_cols = [
        "player",
        "num_shifts",
        "toi_total",
        "toi_avg",
        "toi_median",
        "toi_longest",
        "toi_shortest",
        "plus_minus",
    ]
    # Extend with per-period columns that were seen
    periods_sorted = sorted(
        {
            int(k.split("_p")[-1])
            for r in stats_table_rows
            for k in r
            if re.search(r"^[gftoia_]*p\d+$", k)
        }
    )
    cols = list(base_cols)
    for pidx in periods_sorted:
        cols.extend([f"toi_p{pidx}", f"gf_p{pidx}", f"ga_p{pidx}"])
    # Create CSV-like text (tab-separated for readability)
    lines = []
    header_cols = cols
    lines.append("\t".join(header_cols))
    for r in stats_table_rows:
        lines.append("\t".join(r.get(c, "") for c in header_cols))
    (outdir / "player_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- Goals window extraction (video times) ----------
    # Convert scoreboard goal timestamps to approximate video windows using the conv segments
    def map_sb_to_video(period: int, t_sb: int) -> Optional[int]:
        if period not in conv_segments_by_period:
            return None
        segs = conv_segments_by_period[period]
        for s1, s2, v1, v2 in segs:
            lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
            if lo <= t_sb <= hi and s1 != s2:
                # Linear interpolation using labeled endpoints (preserve direction)
                return int(round(v1 + (t_sb - s1) * (v2 - v1) / (s2 - s1)))
        return None

    # Determine observed scoreboard max per period for capping
    max_sb_by_period: Dict[int, int] = {}
    for period, segs in conv_segments_by_period.items():
        mx = 0
        for s1, s2, _, _ in segs:
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

    (outdir / "goals_for.txt").write_text(
        "\n".join(gf_lines) + ("\n" if gf_lines else ""), encoding="utf-8"
    )
    (outdir / "goals_against.txt").write_text(
        "\n".join(ga_lines) + ("\n" if ga_lines else ""), encoding="utf-8"
    )

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
  if [ "$s" = "$THIS_DIR/clip_all.sh" ]; then
    continue
  fi
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

    # Final validation summary (if any)
    if not skip_validation and validation_errors > 0:
        print(
            f"[validation] Completed with {validation_errors} issue(s). See messages above.",
            file=sys.stderr,
        )


# ----------------------------- CLI -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract per-player shifts & stats from an Excel sheet like 'dh-tv-12-1.xls'. "
            "Optionally fetch goals via --t2s to auto-fill GF/GA."
        )
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to input .xls/.xlsx file.")
    p.add_argument(
        "--sheet", "-s", type=str, default=None, help="Worksheet name (default: first sheet)."
    )
    p.add_argument(
        "--outdir", "-o", type=Path, default=Path("player_shifts"), help="Output directory."
    )
    p.add_argument(
        "--keep-goalies",
        action="store_true",
        help="By default, rows like '(G) 37' are skipped. Use this flag to include them.",
    )
    # Goals (manual)
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
        help=(
            "Path to a text file with one goal per line (GF:period/time or GA:period/time). '#' lines ignored."
        ),
    )
    p.add_argument(
        "--t2s",
        type=int,
        default=None,
        help=(
            "TimeToScore game id. If set, fetches goals and prompts you to select home/away to map GF/GA."
        ),
    )
    p.add_argument(
        "--side",
        type=str,
        choices=["home", "away"],
        default=None,
        help=(
            "Team side to use with --t2s (home/away). Skips prompt and also appends the side to the output directory name."
        ),
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks on start/end ordering and excessive durations.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    goals: List[GoalEvent] = []
    selected_side: Optional[str] = None
    # Prefer T2S if provided; fall back to manual goals
    if args.t2s is not None:
        try:
            goals, selected_side = goals_from_t2s(int(args.t2s), team_side=args.side)
        except Exception as e:  # noqa: BLE001
            print(f"[t2s] Failed to fetch goals for game {args.t2s}: {e}", file=sys.stderr)
            goals = []
            selected_side = None

    # If still empty, use manually supplied
    if not goals:
        goals = load_goals(args.goal, args.goals_file)

    # If we have a selected side, keep Home/Away outputs separated by suffixing the leaf directory
    outdir = args.outdir
    if selected_side in ("home", "away"):
        name = outdir.name
        # Only append if not already suffixed
        if not (name.endswith("_home") or name.endswith("_away")):
            outdir = outdir.with_name(f"{name}_{selected_side}")

    process_sheet(
        xls_path=args.input,
        sheet_name=args.sheet,
        outdir=outdir,
        keep_goalies=args.keep_goalies,
        goals=goals,
        skip_validation=args.skip_validation,
    )
    print(f"✅ Done. Wrote per-player files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

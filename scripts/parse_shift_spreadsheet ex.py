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

Alternatively, with a TimeToScore game id:
  - Two-team (event-log) sheets: goals are derived from the sheet; use
    --home=Blue or --home=White (or --away=Blue/White) to force which color
    maps to the Home/Away roster for roster naming/validation.
  - Per-player sheets: use --your-side=home or --your-side=away to map GF/GA
    when fetching goals from TimeToScore.

Install deps (for .xls):
  pip install pandas xlrd

Example:
  python scripts/parse_shift_spreadsheet.py \
      --input dh-tv-12-1.xls \
      --outdir player_shifts \
      --t2s 51602 --away \
      --keep-goalies
"""

import argparse
import datetime
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def clean_team_display_name(s: Optional[str]) -> str:
    if not s:
        return ""
    t = str(s)
    # Remove explicit '(Shifts)' marker, tolerate spaces and case
    t = re.sub(r"\(\s*Shifts\s*\)", "", t, flags=re.IGNORECASE)
    # Collapse multiple spaces
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


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
    Returns total seconds (int), flooring fractional seconds.
    """
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + _int_floor_seconds_component(sec)
    elif len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + _int_floor_seconds_component(sec)
    elif len(parts) == 1:
        # Support 'SS(.fff)' format (e.g., '58.2' -> 0:58)
        return _int_floor_seconds_component(parts[0])
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
    If a value is a time (datetime.time or Timestamp with time), keep only hour:minute.
    """

    def format_cell(val) -> str:
        if pd.isna(val):
            return ""
        # Already string → return trimmed
        if isinstance(val, str):
            return val.strip()
        # datetime.time (Excel time)
        if isinstance(val, datetime.time):
            return val.strftime("%H:%M")
        # pandas Timestamp (could include date/time)
        if isinstance(val, pd.Timestamp):
            if val.time() != datetime.time(0, 0):  # has time portion
                return val.strftime("%H:%M")
            else:
                return val.strftime("%Y-%m-%d")  # just a date
        # Fallback
        return str(val).strip()

    starts = [format_cell(row[c]) for c in start_cols if format_cell(row[c])]
    ends = [format_cell(row[c]) for c in end_cols if format_cell(row[c])]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]


# Treat period end as 0:00 on the scoreboard when sheets encode it as the period start time.
_PERIOD_START_TIMES = {"12:00", "15:00", "20:00"}

def _normalize_sb_end_time(t: str) -> str:
    try:
        ts = str(t).strip()
    except Exception:
        return t
    if ts in _PERIOD_START_TIMES:
        return "0:00"
    return t


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


@dataclass
class GoalEvent:
    kind: str  # "GF" or "GA"
    period: int
    t_str: str
    t_sec: int = field(init=False)

    def __post_init__(self) -> None:
        self.t_str = self.t_str.strip()
        self.t_sec = parse_flex_time_to_seconds(self.t_str)

    def __str__(self) -> str:  # preserve prior textual representation
        return f"{self.kind}:{self.period}/{self.t_str}"

    __repr__ = __str__


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


def goals_from_t2s(game_id: int, *, side: str) -> List[GoalEvent]:
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
        return []

    home_sc = stats.get("homeScoring") or []
    away_sc = stats.get("awayScoring") or []

    events: List[GoalEvent] = []

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

    # Home goals are GF if side == home else GA
    for row in home_sc:
        ev = _mk_event("GF" if side == "home" else "GA", row.get("period"), row.get("time"))
        if ev:
            events.append(ev)
    # Away goals are GF if side == away else GA
    for row in away_sc:
        ev = _mk_event("GF" if side == "away" else "GA", row.get("period"), row.get("time"))
        if ev:
            events.append(ev)

    # Sort by period then time for determinism
    events.sort(key=lambda e: (e.period, e.t_sec))
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


@dataclass
class EventLogContext:
    event_counts_by_player: Dict[str, Dict[str, int]]
    event_counts_by_type_team: Dict[Tuple[str, str], int]
    event_instances: Dict[Tuple[str, str], List[Dict[str, Any]]]
    event_player_rows: List[Dict[str, Any]]
    team_roster: Dict[str, List[int]]
    team_excluded: Dict[str, List[int]]
    # Derived from on-ice inference at event time (by scoreboard time)
    on_ice_counts_by_player: Dict[str, Dict[str, int]]
    # Map logical sides ("Blue"/"White") -> display team names from sheet
    team_display: Dict[str, str]
    # Roster numbers observed strictly from the shift tables (excludes left event table)
    team_shift_roster: Dict[str, List[int]]
    # Per-player PP/SH shift counts (keys like 'Blue_12')
    pp_shifts_by_player: Dict[str, int]
    sh_shifts_by_player: Dict[str, int]


def _detect_event_log_headers(df: pd.DataFrame) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Backward-compatible detector for legacy headers. Kept for completeness, but
    newer sheets might not contain these exact labels.
    """
    def _find_cell(value: str) -> Optional[Tuple[int, int]]:
        needle = value.strip().lower()
        for rr in range(min(8, df.shape[0])):
            for cc in range(df.shape[1]):
                try:
                    v = df.iat[rr, cc]
                except Exception:
                    continue
                if pd.isna(v):
                    continue
                if str(v).strip().lower() == needle:
                    return (rr, cc)
        return None

    blue_hdr = _find_cell("Shifts (Blue)") or _find_cell("Shifts (blue)")
    white_hdr = _find_cell("Shifts (white)")
    return blue_hdr, white_hdr


def _detect_generic_shift_tables(df: pd.DataFrame) -> List[Tuple[Tuple[int, int], str, Optional[str]]]:
    """
    Detect shift tables by locating repeated 'Video time' / 'Game time' headers in the
    top rows, ignoring the leftmost event summary table. Returns a list of
    ((header_row, approx_team_col), inferred_team_prefix) for up to two teams.
    Team prefix is inferred as 'Blue'/'White' when possible; otherwise defaults
    to 'Blue' for the rightmost table and 'White' for the other.
    """
    matches: List[Tuple[int, int]] = []
    for rr in range(min(6, df.shape[0])):
        for cc in range(df.shape[1]):
            v = df.iat[rr, cc]
            if isinstance(v, str) and v.strip().lower() == "video time":
                # Must have a corresponding 'game time' nearby on same row
                ok = False
                for dc in (1, 2, -1):
                    cc2 = cc + dc
                    if 0 <= cc2 < df.shape[1]:
                        gv = df.iat[rr, cc2]
                        if isinstance(gv, str) and gv.strip().lower() in {"game time", "scoreboard"}:
                            ok = True
                            break
                if ok:
                    matches.append((rr, cc))

    # Filter out very-left occurrences (likely left event table), and de-dup by column proximity
    matches = [(r, c) for (r, c) in matches if c >= 8]
    matches.sort(key=lambda x: x[1])
    dedup: List[Tuple[int, int]] = []
    for rc in matches:
        if not dedup or abs(rc[1] - dedup[-1][1]) > 3:
            dedup.append(rc)

    # Keep up to the two rightmost
    cand = dedup[-2:] if len(dedup) >= 2 else dedup
    if not cand:
        return []

    # For each candidate, try to find a team-identifying label in row 0 nearby
    out: List[Tuple[Tuple[int, int], str, Optional[str]]] = []
    for (rr, cc) in cand:
        team_text = None
        # search row 0 around cc
        row0 = 0
        for dc in range(-2, 6):
            c2 = cc + dc
            if 0 <= c2 < df.shape[1]:
                v = df.iat[row0, c2]
                if isinstance(v, str) and v.strip():
                    team_text = v.strip()
                    break
        prefix = ""
        if team_text:
            low = team_text.lower()
            if "(blue)" in low or low.endswith(" blue"):
                prefix = "Blue"
            elif "(white)" in low or low.endswith(" white"):
                prefix = "White"
        out.append(((row0, cc), prefix, team_text))

    # Ensure distinct prefixes; fill blanks deterministically
    rightsorted = sorted(out, key=lambda x: x[0][1])
    # Default rightmost as Blue, the other as White when unknown
    if len(rightsorted) == 2:
        if not rightsorted[1][1]:
            rightsorted[1] = (rightsorted[1][0], "Blue", rightsorted[1][2])
        if not rightsorted[0][1]:
            rightsorted[0] = (rightsorted[0][0], "White", rightsorted[0][2])
    elif len(rightsorted) == 1:
        if not rightsorted[0][1]:
            rightsorted[0] = (rightsorted[0][0], "Blue", rightsorted[0][2])
    return rightsorted


def _parse_event_log_layout(df: pd.DataFrame) -> Tuple[
    bool,
    Dict[str, List[Tuple[str, str]]],
    Dict[str, List[Tuple[int, str, str]]],
    Dict[int, List[Tuple[int, int, int, int]]],
    Optional[EventLogContext],
]:
    blue_hdr, white_hdr = _detect_event_log_headers(df)
    headers_g: List[Tuple[Tuple[int, int], str, Optional[str]]] = []
    def _find_row0_label(col: int) -> Optional[str]:
        # Search first row for a likely team label near column
        row0 = 0
        for dc in range(-4, 8):
            c2 = col + dc
            if 0 <= c2 < df.shape[1]:
                v = df.iat[row0, c2]
                if isinstance(v, str) and v.strip() and ("video time" not in v.lower()) and ("game time" not in v.lower()):
                    return v.strip()
        return None
    if blue_hdr:
        headers_g.append((blue_hdr, "Blue", _find_row0_label(blue_hdr[1])))
    if white_hdr:
        headers_g.append((white_hdr, "White", _find_row0_label(white_hdr[1])))
    if not headers_g:
        # Fallback to generic detector
        headers_g = _detect_generic_shift_tables(df)
    if not headers_g:
        return False, {}, {}, {}, None

    # Accumulators
    video_pairs_by_player: Dict[str, List[Tuple[str, str]]] = {}
    sb_pairs_by_player: Dict[str, List[Tuple[int, str, str]]] = {}
    conv_segments_by_period: Dict[int, List[Tuple[int, int, int, int]]] = {}

    # Roster cap handling
    MAX_TEAM_PLAYERS = 20
    team_roster: Dict[str, List[int]] = {}
    team_excluded: Dict[str, List[int]] = {}
    team_shift_roster: Dict[str, List[int]] = {}
    # Per-player PP/SH counts keyed as 'Blue_12' or 'White_34'
    pp_shifts_by_player: Dict[str, int] = {}
    sh_shifts_by_player: Dict[str, int] = {}

    def _register_and_flag(team: str, jerseys: List[int]) -> List[int]:
        if not team:
            return []
        seen_local = set()
        ordered: List[int] = []
        for j in jerseys:
            if j in seen_local:
                continue
            seen_local.add(j)
            ordered.append(j)
        roster = team_roster.setdefault(team, [])
        excluded = team_excluded.setdefault(team, [])
        for j in ordered:
            if j in roster:
                continue
            if len(roster) < MAX_TEAM_PLAYERS:
                roster.append(j)
            else:
                if j not in excluded:
                    excluded.append(j)
        return ordered

    def _parse_event_time(cell: Any) -> Optional[int]:
        if cell is None or (isinstance(cell, str) and not cell.strip()):
            return None
        s = str(cell).strip()
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, _s = parts
                return int(h) * 60 + int(m)
            if len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + int(sec)
            if len(parts) == 1:
                return int(float(parts[0]))
        except Exception:
            return None
        return None

    def _period_num_from_label(lbl: Optional[str]) -> Optional[int]:
        if not lbl:
            return None
        s = str(lbl).strip().lower()
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    def _parse_event_block(header_rc: Tuple[int, int], team_prefix: str) -> None:
        base_r, base_c = header_rc
        # Locate columns for video/game time near header
        video_col = None
        game_col = None
        period_col = None
        for r in range(base_r + 1, min(df.shape[0], base_r + 4)):
            for c in range(max(0, base_c - 4), min(df.shape[1], base_c + 6)):
                val = df.iat[r, c]
                if pd.notna(val) and isinstance(val, str):
                    s = val.strip().lower()
                    if s == "video time":
                        video_col = c
                    elif s == "game time":
                        game_col = c
                    elif "period" in s:
                        period_col = c
            if video_col is not None and game_col is not None:
                break
        if video_col is None:
            video_col = max(0, base_c - 2)
        if game_col is None:
            game_col = max(0, base_c - 1)
        if period_col is None:
            period_col = base_c + 2

        players_start = base_c
        # Determine players_width by scanning to the right until a fully blank column separates tables
        def _col_is_blank(col: int) -> bool:
            if col >= df.shape[1]:
                return True
            for rr in range(base_r + 1, min(df.shape[0], base_r + 120)):
                vv = df.iat[rr, col]
                if pd.isna(vv):
                    continue
                if str(vv).strip():
                    return False
            return True

        players_width = 0
        for c2 in range(players_start, min(df.shape[1], players_start + 40)):
            if _col_is_blank(c2):
                break
            players_width += 1
        if players_width <= 0:
            players_width = 12

        # Build events
        current_period_label: Optional[str] = None
        current_strength: Optional[str] = None  # 'PP' or 'SH'
        events: List[Dict[str, Any]] = []
        for r in range(base_r + 1, df.shape[0]):
            vcell = df.iat[r, video_col] if video_col < df.shape[1] else None
            if isinstance(vcell, str) and vcell.strip().lower() == "video time":
                plbl = df.iat[r, period_col] if period_col < df.shape[1] else None
                current_period_label = str(plbl).strip() if pd.notna(plbl) else current_period_label
                continue
            gcell = df.iat[r, game_col] if game_col < df.shape[1] else None
            vsec = _parse_event_time(vcell)
            gsec = _parse_event_time(gcell)
            players: List[int] = []
            saw_pp = False
            saw_sh = False
            for k in range(players_width):
                c = players_start + k
                if c >= df.shape[1]:
                    break
                val = df.iat[r, c]
                if pd.isna(val):
                    continue
                if isinstance(val, (int, float)):
                    n = int(val)
                    if 1 <= n <= 98:
                        players.append(n)
                    continue
                if hasattr(val, "hour") and hasattr(val, "minute"):
                    continue
                s = str(val).strip()
                if not s:
                    continue
                if s.upper() in {"PP", "SH"}:
                    if s.upper() == "PP":
                        saw_pp = True
                    if s.upper() == "SH":
                        saw_sh = True
                    continue
                if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", s):
                    continue
                for m in re.finditer(r"#?(\d{1,2})(?!\d)", s):
                    try:
                        n = int(m.group(1))
                    except Exception:
                        continue
                    if 1 <= n <= 98:
                        players.append(n)
            if players:
                players = sorted(set(players))
            # Determine strength marker for this row
            row_strength: Optional[str] = None
            if saw_pp and not saw_sh:
                row_strength = "PP"
            elif saw_sh and not saw_pp:
                row_strength = "SH"
            # If only PP/SH marker present, update current strength and skip
            if (not players) and (vsec is None and gsec is None) and row_strength is not None:
                current_strength = row_strength
                continue
            if vsec is None and gsec is None and not players:
                continue
            players = _register_and_flag(team_prefix, players)
            # Track shift-table-only roster per team for robust T2S mapping/validation
            if players:
                sr = team_shift_roster.setdefault(team_prefix, [])
                for j in players:
                    if j not in sr:
                        sr.append(j)
            events.append({
                "period": _period_num_from_label(current_period_label),
                "v": vsec,
                "g": gsec,
                "players": players,
                "strength": (row_strength or current_strength),
            })

        # Walk events to generate per-player shifts
        open_shift: Dict[int, Dict[str, Any]] = {}
        last_period: Optional[int] = None
        for ev in events:
            cur_p = ev.get("period")
            # Close open shifts at period change (scoreboard end -> 0:00)
            if last_period is not None and cur_p is not None and cur_p != last_period:
                for pid, sh in list(open_shift.items()):
                    sv, sg = sh.get("sv"), sh.get("sg")
                    evv, egg = ev.get("v"), 0
                    key = f"{team_prefix}_{pid}"
                    if sv is not None and evv is not None:
                        video_pairs_by_player.setdefault(key, []).append(
                            (seconds_to_hhmmss(int(sv)), seconds_to_hhmmss(int(evv)))
                        )
                    if sg is not None:
                        sb_pairs_by_player.setdefault(key, []).append(
                            (last_period, seconds_to_mmss_or_hhmmss(int(sg)), seconds_to_mmss_or_hhmmss(int(egg)))
                        )
                        if sv is not None and evv is not None:
                            conv_segments_by_period.setdefault(last_period, []).append((int(sg), int(egg), int(sv), int(evv)))
                    st = sh.get("strength")
                    if st == "PP":
                        pp_shifts_by_player[key] = pp_shifts_by_player.get(key, 0) + 1
                    elif st == "SH":
                        sh_shifts_by_player[key] = sh_shifts_by_player.get(key, 0) + 1
                    del open_shift[pid]
            if cur_p is not None:
                last_period = cur_p

            on_ice = set(ev.get("players") or [])
            # Close shifts for players no longer on ice
            for pid, sh in list(open_shift.items()):
                if pid not in on_ice:
                    sv, sg = sh.get("sv"), sh.get("sg")
                    evv, egg = ev.get("v"), ev.get("g")
                    key = f"{team_prefix}_{pid}"
                    if sv is not None and evv is not None:
                        video_pairs_by_player.setdefault(key, []).append(
                            (seconds_to_hhmmss(int(sv)), seconds_to_hhmmss(int(evv)))
                        )
                    if sg is not None and cur_p is not None:
                        end_g = egg if egg is not None else 0
                        sb_pairs_by_player.setdefault(key, []).append(
                            (cur_p, seconds_to_mmss_or_hhmmss(int(sg)), seconds_to_mmss_or_hhmmss(int(end_g)))
                        )
                        if sv is not None and evv is not None:
                            conv_segments_by_period.setdefault(cur_p, []).append((int(sg), int(end_g), int(sv), int(evv)))
                    st = sh.get("strength")
                    if st == "PP":
                        pp_shifts_by_player[key] = pp_shifts_by_player.get(key, 0) + 1
                    elif st == "SH":
                        sh_shifts_by_player[key] = sh_shifts_by_player.get(key, 0) + 1
                    del open_shift[pid]

            # Open shifts for players now on
            for pid in on_ice:
                if pid not in open_shift:
                    open_shift[pid] = {"sv": ev.get("v"), "sg": ev.get("g"), "period": ev.get("period"), "strength": ev.get("strength")}

        # Close any remaining open shifts at last event, scoreboard -> 0:00
        if events and open_shift:
            last_ev = events[-1]
            for pid, sh in list(open_shift.items()):
                key = f"{team_prefix}_{pid}"
                sv, sg = sh.get("sv"), sh.get("sg")
                evv, egg = last_ev.get("v"), 0
                per = sh.get("period") or last_ev.get("period")
                if sv is not None and evv is not None:
                    video_pairs_by_player.setdefault(key, []).append(
                        (seconds_to_hhmmss(int(sv)), seconds_to_hhmmss(int(evv)))
                    )
                if sg is not None and per is not None:
                    sb_pairs_by_player.setdefault(key, []).append(
                        (per, seconds_to_mmss_or_hhmmss(int(sg)), seconds_to_mmss_or_hhmmss(int(egg)))
                    )
                    if sv is not None and evv is not None:
                        conv_segments_by_period.setdefault(int(per), []).append((int(sg), int(egg), int(sv), int(evv)))
                st = sh.get("strength")
                if st == "PP":
                    pp_shifts_by_player[key] = pp_shifts_by_player.get(key, 0) + 1
                elif st == "SH":
                    sh_shifts_by_player[key] = sh_shifts_by_player.get(key, 0) + 1

    # Prepare team display mapping (Blue/White -> actual names if present)
    team_display: Dict[str, str] = {}
    for (_rc, pref, disp) in headers_g:
        if pref:
            if disp and isinstance(disp, str):
                team_display[pref] = clean_team_display_name(disp)
            else:
                team_display[pref] = pref

    for (rc, team_prefix, _disp) in headers_g:
        _parse_event_block(rc, team_prefix or "Blue")

    # ---- Parse left-side event columns ----
    event_counts_by_player: Dict[str, Dict[str, int]] = {}
    event_counts_by_type_team: Dict[Tuple[str, str], int] = {}
    event_instances: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    event_player_rows: List[Dict[str, Any]] = []

    # Find left header row and relevant columns up to the Blue/White shifts header column
    limit_col = (blue_hdr[1] if blue_hdr else df.shape[1])
    left_header_row: Optional[int] = None
    for r in range(min(8, df.shape[0])):
        for c in range(min(limit_col, df.shape[1])):
            v = df.iat[r, c]
            if pd.notna(v) and isinstance(v, str) and v.strip().lower() == "video time":
                left_header_row = r
                break
        if left_header_row is not None:
            break
    if left_header_row is not None:
        # Map labels -> columns from the header row
        label_to_col: Dict[str, int] = {}
        for c in range(min(limit_col, df.shape[1])):
            v = df.iat[left_header_row, c]
            if pd.notna(v) and isinstance(v, str) and v.strip():
                label_to_col[v.strip().lower()] = c

        vt_col = label_to_col.get("video time")
        gt_col = label_to_col.get("scoreboard")
        shots_col = label_to_col.get("shots")
        goals_col = label_to_col.get("goals")
        assists_col = label_to_col.get("assist")
        sog_col = None
        # handle alternate header names like 'Shots on Goal' / 'SOG'
        for k, c in label_to_col.items():
            kl = k.lower()
            if kl == "shots on goal" or kl == "sog":
                sog_col = c
        # tolerate wording variations and capitalization
        entries_col = None
        exits_col = None
        for k, c in label_to_col.items():
            kl = k.lower()
            if "controlled" in kl and "blue" in kl and "entr" in kl:
                entries_col = c
            if "controlled" in kl and "exit" in kl:
                exits_col = c

        # Guess team column
        team_col: Optional[int] = None
        best_count = 0
        for c in range(min(limit_col, df.shape[1])):
            cnt = 0
            for r in range(left_header_row + 1, min(df.shape[0], left_header_row + 80)):
                v = df.iat[r, c]
                if isinstance(v, str) and v.strip():
                    vs = v.strip().lower()
                    if ("blu" in vs) or ("whi" in vs):
                        cnt += 1
            if cnt > best_count:
                best_count = cnt
                team_col = c

        def _parse_team_from_text(s: Optional[str]) -> Optional[str]:
            if not s:
                return None
            t = s.lower()
            if "blue" in t:
                return "Blue"
            if "white" in t:
                return "White"
            return None

        def _extract_nums(s: Optional[str]) -> List[int]:
            if not s:
                return []
            s = s.strip()
            if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", s):
                return []
            nums: List[int] = []
            for m in re.finditer(r"#?(\d{1,2})(?!\d)", s):
                try:
                    n = int(m.group(1))
                except Exception:
                    continue
                if 1 <= n <= 98:
                    nums.append(n)
            seen = set()
            out: List[int] = []
            for n in nums:
                if n in seen:
                    continue
                seen.add(n)
                out.append(n)
            return out

        def _record_event(
            kind: str,
            team: Optional[str],
            jersey_list: List[int],
            period_label: Optional[str],
            vsec: Optional[int],
            gsec: Optional[int],
        ) -> None:
            if not team:
                return
            event_counts_by_type_team[(kind, team)] = event_counts_by_type_team.get((kind, team), 0) + 1
            filtered = _register_and_flag(team, jersey_list)
            period_num = None
            if period_label is not None:
                m = re.search(r"(\d+)", str(period_label))
                if m:
                    period_num = int(m.group(1))
            for j in filtered:
                pk = f"{team}_{int(j)}"
                d = event_counts_by_player.setdefault(pk, {})
                d[kind] = d.get(kind, 0) + 1
                event_player_rows.append(
                    {
                        "event_type": kind,
                        "team": team,
                        "player": pk,
                        "jersey": int(j),
                        "period": period_num,
                        "video_s": vsec,
                        "game_s": gsec,
                    }
                )
            event_instances.setdefault((kind, team), []).append({"period": period_num, "video_s": vsec, "game_s": gsec})

        def _validate_left_roster(team: Optional[str], jersey_list: List[int]) -> None:
            """Validate that jersey numbers listed in the left event table appear in the
            shift-table-derived roster for the specified team color (Blue/White).
            """
            if not team or not jersey_list:
                return
            try:
                roster_nums = set(team_shift_roster.get(team, []) or [])
                missing = [n for n in jersey_list if n not in roster_nums]
                if missing:
                    import sys as _sys
                    miss_s = ", ".join(str(x) for x in sorted(set(missing))[:20])
                    print(
                        f"[validation] LEFT_TABLE_ROSTER | Team={team} | jersey(s) not on sheet roster: {miss_s}",
                        file=_sys.stderr,
                    )
            except Exception:
                pass

        # Walk data rows to collect events
        current_period: Optional[str] = None
        for r in range(left_header_row + 1, df.shape[0]):
            row_vals = [df.iat[r, c] for c in range(min(limit_col, df.shape[1]))]
            for v in row_vals:
                if isinstance(v, str) and "period" in v.lower():
                    current_period = v.strip()
                    break

            team_val = None
            if team_col is not None:
                tv = df.iat[r, team_col]
                if isinstance(tv, str) and tv.strip() in ("Blue", "White"):
                    team_val = tv.strip()
            row_vsec = _parse_event_time(df.iat[r, vt_col]) if vt_col is not None else None
            row_gsec = _parse_event_time(df.iat[r, gt_col]) if gt_col is not None else None

            if shots_col is not None:
                sv = df.iat[r, shots_col]
                if isinstance(sv, str) and sv.strip():
                    t = team_val or _parse_team_from_text(sv)
                    jerseys = _extract_nums(sv)
                    _validate_left_roster(t, jerseys)
                    _record_event("Shot", t, jerseys, current_period, row_vsec, row_gsec)
            if goals_col is not None:
                gv = df.iat[r, goals_col]
                if isinstance(gv, str) and gv.strip():
                    t = team_val or _parse_team_from_text(gv)
                    jerseys = _extract_nums(gv)
                    _validate_left_roster(t, jerseys)
                    _record_event("Goal", t, jerseys, current_period, row_vsec, row_gsec)
            # If there's a 'Shots on Goal' column, parse SOG/GOAL markers
            if sog_col is not None:
                sv2 = df.iat[r, sog_col]
                if isinstance(sv2, str) and sv2.strip():
                    s2 = sv2.strip().lower()
                    t = team_val or _parse_team_from_text(sv2)
                    # SOG marker
                    if "sog" in s2:
                        _record_event("SOG", t, [], current_period, row_vsec, row_gsec)
                    # Goal marker (also counts as SOG)
                    if "goal" in s2:
                        jerseys: List[int] = []
                        if shots_col is not None:
                            sv_sh = df.iat[r, shots_col]
                            if isinstance(sv_sh, str) and sv_sh.strip():
                                jerseys = _extract_nums(sv_sh)
                        if not jerseys and assists_col is not None:
                            av_sh = df.iat[r, assists_col]
                            if isinstance(av_sh, str) and av_sh.strip():
                                jerseys = _extract_nums(av_sh)
                        _record_event("SOG", t, [], current_period, row_vsec, row_gsec)
                        _validate_left_roster(t, jerseys)
                        _record_event("Goal", t, jerseys, current_period, row_vsec, row_gsec)
            if assists_col is not None:
                av = df.iat[r, assists_col]
                if isinstance(av, str) and av.strip():
                    t = team_val or _parse_team_from_text(av)
                    jerseys = _extract_nums(av)
                    _validate_left_roster(t, jerseys)
                    _record_event("Assist", t, jerseys, current_period, row_vsec, row_gsec)
            if entries_col is not None:
                ev = df.iat[r, entries_col]
                if isinstance(ev, str) and ev.strip():
                    t = team_val or _parse_team_from_text(ev)
                    jerseys = _extract_nums(ev)
                    _validate_left_roster(t, jerseys)
                    _record_event("ControlledEntry", t, jerseys, current_period, row_vsec, row_gsec)
            if exits_col is not None:
                xv = df.iat[r, exits_col]
                if isinstance(xv, str) and xv.strip():
                    t = team_val or _parse_team_from_text(xv)
                    jerseys = _extract_nums(xv)
                    _validate_left_roster(t, jerseys)
                    _record_event("ControlledExit", t, jerseys, current_period, row_vsec, row_gsec)
            label = df.iat[r, 0]
            if isinstance(label, str) and label.strip().lower() == "expected goal":
                text_cell = None
                if shots_col is not None:
                    text_cell = df.iat[r, shots_col]
                if not (isinstance(text_cell, str) and text_cell.strip()) and goals_col is not None:
                    text_cell = df.iat[r, goals_col]
                t = None
                jerseys: List[int] = []
                if isinstance(text_cell, str) and text_cell.strip():
                    t = team_val or _parse_team_from_text(text_cell)
                    jerseys = _extract_nums(text_cell)
                _record_event("ExpectedGoal", t, jerseys, current_period, row_vsec, row_gsec)
            elif isinstance(label, str):
                lab = label.strip().lower()
                if ("controlled" in lab) and ("blue" in lab) and ("entr" in lab):
                    # Controlled blue-line entries row
                    _record_event("ControlledEntry", team_val, [], current_period, row_vsec, row_gsec)
                elif ("controlled" in lab) and ("exit" in lab):
                    # Controlled exits row
                    _record_event("ControlledExit", team_val, [], current_period, row_vsec, row_gsec)
                elif "rush" in lab:
                    # Handle '2/3 Man Rushes'
                    _record_event("Rush", team_val, [], current_period, row_vsec, row_gsec)
            elif isinstance(label, str) and "rush" in label.strip().lower():
                # Handle '2/3 Man Rushes' style rows; team is from the Team column
                t = team_val
                _record_event("Rush", t, [], current_period, row_vsec, row_gsec)

    # ---- Build on-ice event counts (by scoreboard time) ----
    # Build per-player scoreboard intervals for quick membership checks
    intervals_by_player: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
    for pk, lst in sb_pairs_by_player.items():
        for period, a, b in lst:
            try:
                lo, hi = compute_interval_seconds(a, b)
            except Exception:
                continue
            intervals_by_player.setdefault(pk, {}).setdefault(period, []).append((lo, hi))

    def on_ice_sets(period: int, gsec: int) -> Tuple[set[int], set[int]]:
        blue: set[int] = set()
        white: set[int] = set()
        for pk, byp in intervals_by_player.items():
            if period not in byp:
                continue
            for lo, hi in byp[period]:
                if interval_contains(gsec, lo, hi):
                    # pk format: "Team_#"
                    if pk.startswith("Blue_"):
                        try:
                            white_or_blue_num = int(pk.split("_", 1)[1])
                        except Exception:
                            continue
                        blue.add(white_or_blue_num)
                    elif pk.startswith("White_"):
                        try:
                            num = int(pk.split("_", 1)[1])
                        except Exception:
                            continue
                        white.add(num)
                    break
        return blue, white

    def _inc(d: Dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    on_ice_counts_by_player: Dict[str, Dict[str, int]] = {}
    # Iterate over all event instances
    for (etype, team), lst in event_instances.items():
        if team not in ("Blue", "White"):
            continue
        for it in lst:
            p = it.get("period")
            gs = it.get("game_s")
            if not (isinstance(p, int) and isinstance(gs, (int, float))):
                continue
            period = int(p)
            gsec = int(gs)
            blue_set, white_set = on_ice_sets(period, gsec)
            for_or_against_key_for = f"{etype}_For"
            for_or_against_key_against = f"{etype}_Against"
            # Increment for players on the executing team
            if team == "Blue":
                for j in sorted(blue_set):
                    pk = f"Blue_{j}"
                    _inc(on_ice_counts_by_player.setdefault(pk, {}), for_or_against_key_for)
                for j in sorted(white_set):
                    pk = f"White_{j}"
                    _inc(on_ice_counts_by_player.setdefault(pk, {}), for_or_against_key_against)
            else:
                for j in sorted(white_set):
                    pk = f"White_{j}"
                    _inc(on_ice_counts_by_player.setdefault(pk, {}), for_or_against_key_for)
                for j in sorted(blue_set):
                    pk = f"Blue_{j}"
                    _inc(on_ice_counts_by_player.setdefault(pk, {}), for_or_against_key_against)

    event_log_context = EventLogContext(
        event_counts_by_player=event_counts_by_player,
        event_counts_by_type_team=event_counts_by_type_team,
        event_instances=event_instances,
        event_player_rows=event_player_rows,
        team_roster=team_roster,
        team_excluded=team_excluded,
        on_ice_counts_by_player=on_ice_counts_by_player,
        team_display=team_display,
        team_shift_roster=team_shift_roster,
        pp_shifts_by_player=pp_shifts_by_player,
        sh_shifts_by_player=sh_shifts_by_player,
    )

    return True, video_pairs_by_player, sb_pairs_by_player, conv_segments_by_period, event_log_context


def _parse_per_player_layout(
    df: pd.DataFrame, keep_goalies: bool, skip_validation: bool
) -> Tuple[
    Dict[str, List[Tuple[str, str]]],
    Dict[str, List[Tuple[int, str, str]]],
    Dict[int, List[Tuple[int, int, int, int]]],
    int,
]:
    blocks = find_period_blocks(df)
    if not blocks:
        raise ValueError("No 'Period N' sections found in column A.")

    video_pairs_by_player: Dict[str, List[Tuple[str, str]]] = {}
    sb_pairs_by_player: Dict[str, List[Tuple[int, str, str]]] = {}
    conv_segments_by_period: Dict[int, List[Tuple[int, int, int, int]]] = {}

    MAX_SHIFT_SECONDS = 30 * 60  # 30 minutes

    def _report_validation(kind: str, period: int, player_key: str, a: str, b: str, reason: str) -> None:
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
            if sb_pairs:
                sb_pairs = [(a, _normalize_sb_end_time(b)) for a, b in sb_pairs]

            if not skip_validation:
                for va, vb in video_pairs:
                    try:
                        vsa = parse_flex_time_to_seconds(va)
                        vsb = parse_flex_time_to_seconds(vb)
                    except Exception as e:
                        _report_validation("VIDEO", period_num, player_key, va, vb, f"unparseable time: {e}")
                        validation_errors += 1
                        continue
                    if vsa >= vsb:
                        _report_validation(
                            "VIDEO", period_num, player_key, va, vb, "start must be before end (strictly increasing)"
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

                for sa, sb in sb_pairs:
                    try:
                        ssa = parse_flex_time_to_seconds(sa)
                        ssb = parse_flex_time_to_seconds(sb)
                    except Exception as e:
                        _report_validation("SCOREBOARD", period_num, player_key, sa, sb, f"unparseable time: {e}")
                        validation_errors += 1
                        continue
                    if ssa == ssb:
                        _report_validation(
                            "SCOREBOARD", period_num, player_key, sa, sb, "start equals end (zero-length shift)"
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
                sb_pairs_by_player.setdefault(player_key, []).extend((period_num, a, b) for a, b in sb_pairs)

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

    return video_pairs_by_player, sb_pairs_by_player, conv_segments_by_period, validation_errors


def _split_team_and_key(player_key: str) -> Tuple[Optional[str], str]:
    """If key is like 'Blue_12_Name', return ('Blue', '12_Name'). Else (None, key)."""
    if "_" in player_key:
        team, rest = player_key.split("_", 1)
        if team in {"Blue", "White"}:
            return team, rest
    return None, player_key


def _write_video_times_and_scripts(
    outdir: Path,
    video_pairs_by_player: Dict[str, List[Tuple[str, str]]],
    nr_jobs: int,
    *,
    split_by_team: bool = False,
) -> None:
    for player_key, v_pairs in video_pairs_by_player.items():
        norm_pairs = []
        for a, b in v_pairs:
            try:
                sa = parse_flex_time_to_seconds(a)
                sb = parse_flex_time_to_seconds(b)
            except Exception:
                continue
            norm_pairs.append((seconds_to_hhmmss(sa), seconds_to_hhmmss(sb)))
        p = outdir / f"{player_key}_video_times.txt"
        p.write_text("\n".join(f"{a} {b}" for a, b in norm_pairs) + ("\n" if norm_pairs else ""), encoding="utf-8")

        script_path = outdir / f"clip_{player_key}.sh"
        player_label = player_key.replace("_", " ")
        script_body = """#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT=\"$1\"
OPP=\"$2\"
THIS_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
TS_FILE=\"$THIS_DIR/{player_key}_video_times.txt\"
# Parse optional flags
QUICK=0
HQ=0
shift 2 || true
for ARG in \"$@\"; do
  if [ \"$ARG\" = \"--quick\" ] || [ \"$ARG\" = \"-q\" ]; then
    QUICK=1
  elif [ \"$ARG\" = \"--hq\" ]; then
    HQ=1
  fi
done

EXTRA_FLAGS=()
if [ \"$QUICK\" -gt 0 ]; then
  EXTRA_FLAGS+=(\"--quick\" \"1\")
fi
if [ \"$HQ\" -gt 0 ]; then
  export VIDEO_CLIPPER_HQ=1
fi

python -m hmlib.cli.video_clipper -j {nr_jobs} --input \"$INPUT\" --timestamps \"$TS_FILE\" --temp-dir \"$THIS_DIR/temp_clips/{player_key}\" \"{player_label} vs $OPP\" \"${{EXTRA_FLAGS[@]}}\"
""".format(
            nr_jobs=nr_jobs, player_key=player_key, player_label=player_label
        )
        script_path.write_text(script_body, encoding="utf-8")
        try:
            import os

            os.chmod(script_path, 0o755)
        except Exception:
            pass


def _write_scoreboard_times(outdir: Path, sb_pairs_by_player: Dict[str, List[Tuple[int, str, str]]]) -> None:
    for player_key, sb_list in sb_pairs_by_player.items():
        p = outdir / f"{player_key}_scoreboard_times.txt"
        lines = [f"{period} {a} {b}" for (period, a, b) in sb_list]
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_global_summary_csv(
    outdir: Path,
    sb_pairs_by_player: Dict[str, List[Tuple[int, str, str]]],
    pp_map: Optional[Dict[str, int]] = None,
    sh_map: Optional[Dict[str, int]] = None,
    team_color: Optional[str] = None,
) -> None:
    summary_rows = []
    for player_key, sb_list in sb_pairs_by_player.items():
        all_pairs = [(a, b) for (_, a, b) in sb_list]
        shift_summary = summarize_shift_lengths_sec(all_pairs)
        row: Dict[str, Any] = {
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
        # Optionally include PP/SH shifts if provided
        if pp_map is not None and sh_map is not None and team_color in ("Blue", "White"):
            try:
                base = str(player_key).split("_", 1)[0]
                num = int(base)
                k = f"{team_color}_{num}"
                row["pp_shifts"] = int(pp_map.get(k, 0))
                row["sh_shifts"] = int(sh_map.get(k, 0))
            except Exception:
                pass
        summary_rows.append(row)
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(by="player").to_csv(outdir / "summary_stats.csv", index=False)


def _compute_player_stats(
    player_key: str,
    sb_list: List[Tuple[int, str, str]],
    video_pairs_by_player: Dict[str, List[Tuple[str, str]]],
    goals_by_period: Dict[int, List[GoalEvent]],
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int], Dict[int, str]]:
    sb_by_period: Dict[int, List[Tuple[str, str]]] = {}
    for period, a, b in sb_list:
        sb_by_period.setdefault(period, []).append((a, b))

    all_pairs = [(a, b) for (_, a, b) in sb_list]
    shift_summary = summarize_shift_lengths_sec(all_pairs)
    per_period_toi_map = per_period_toi(sb_by_period)

    plus_minus = 0
    counted_gf: List[str] = []
    counted_ga: List[str] = []
    counted_gf_by_period: Dict[int, int] = {}
    counted_ga_by_period: Dict[int, int] = {}
    for period, pairs in sb_by_period.items():
        if period not in goals_by_period:
            continue
        for ev in goals_by_period[period]:
            matched = False
            for a, b in pairs:
                a_sec = parse_flex_time_to_seconds(a)
                b_sec = parse_flex_time_to_seconds(b)
                lo, hi = (a_sec, b_sec) if a_sec <= b_sec else (b_sec, a_sec)
                if not (lo <= ev.t_sec <= hi):
                    continue
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

    row_map: Dict[str, str] = {
        "player": player_key,
        "shifts": shift_summary["num_shifts"],
        "plus_minus": str(plus_minus),
        "sb_toi_total": shift_summary["toi_total"],
        "sb_avg": shift_summary["toi_avg"],
        "sb_median": shift_summary["toi_median"],
        "sb_longest": shift_summary["toi_longest"],
        "sb_shortest": shift_summary["toi_shortest"],
    }
    row_map["gf_counted"] = str(len(counted_gf))
    row_map["ga_counted"] = str(len(counted_ga))

    v_pairs = video_pairs_by_player.get(player_key, [])
    if v_pairs:
        v_sum = 0
        for a, b in v_pairs:
            lo, hi = compute_interval_seconds(a, b)
            v_sum += hi - lo
        row_map["video_toi_total"] = seconds_to_mmss_or_hhmmss(v_sum)
    else:
        row_map["video_toi_total"] = ""

    # per-period values
    per_counts: Dict[str, int] = {}
    per_counts_gf: Dict[str, int] = {}
    for period, toi in per_period_toi_map.items():
        row_map[f"P{period}_toi"] = toi
    for period, pairs in sb_by_period.items():
        per_counts[f"P{period}_shifts"] = len(pairs)
    for period, cnt in counted_gf_by_period.items():
        per_counts_gf[f"P{period}_GF"] = cnt
    for period, cnt in counted_ga_by_period.items():
        per_counts_gf[f"P{period}_GA"] = per_counts_gf.get(f"P{period}_GA", 0)  # placeholder to ensure keys
    # Return row_map and per-period counts; plus per_period_toi_map for columns
    return row_map, per_counts, {**{k: 0 for k in []}}, per_period_toi_map


def _write_player_stats_text_and_csv(
    outdir: Path,
    stats_table_rows: List[Dict[str, str]],
    all_periods_seen: List[int],
) -> None:
    periods = sorted(all_periods_seen)
    summary_cols = ["player", "shifts", "plus_minus", "gf_counted", "ga_counted"]
    sb_cols = ["sb_toi_total", "sb_avg", "sb_median", "sb_longest", "sb_shortest"]
    video_cols = ["video_toi_total"]
    period_toi_cols = [f"P{p}_toi" for p in periods]
    period_shift_cols = [f"P{p}_shifts" for p in periods]
    period_gf_cols = [f"P{p}_GF" for p in periods]
    period_ga_cols = [f"P{p}_GA" for p in periods]
    # Detect extra event count columns present in rows and include them deterministically
    event_order_types = [
        "Shot",
        "SOG",
        "Goal",
        "Assist",
        "ControlledEntry",
        "ControlledExit",
        "ExpectedGoal",
        "Rush",
    ]
    detected_event_cols = []
    for t in event_order_types:
        for suf in ("For", "Against"):
            k = f"{t}_{suf}"
            if any(k in r for r in stats_table_rows):
                detected_event_cols.append(k)
    # Include PP/SH columns if present in any row
    include_ppsh = any(("pp_shifts" in r or "sh_shifts" in r) for r in stats_table_rows)
    ppsh_cols = ["pp_shifts", "sh_shifts"] if include_ppsh else []
    cols = (
        summary_cols
        + sb_cols
        + video_cols
        + ppsh_cols
        + detected_event_cols
        + period_toi_cols
        + period_shift_cols
        + period_gf_cols
        + period_ga_cols
    )

    rows_for_print: List[List[str]] = []
    for r in sorted(stats_table_rows, key=lambda x: x.get("player", "")):
        rows_for_print.append([r.get(c, "") for c in cols])

    widths = [len(c) for c in cols]
    for row in rows_for_print:
        for i, cell in enumerate(row):
            if len(str(cell)) > widths[i]:
                widths[i] = len(str(cell))

    def fmt_row(values: List[str]) -> str:
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

    lines = [fmt_row(cols)]
    lines.append(fmt_row(["-" * w for w in widths]))
    for row in rows_for_print:
        lines.append(fmt_row(row))
    (outdir / "player_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import csv  # local import

    csv_rows = [dict(zip(cols, row)) for row in rows_for_print]
    try:
        pd.DataFrame(csv_rows).to_csv(outdir / "player_stats.csv", index=False, columns=cols)
    except Exception:
        with (outdir / "player_stats.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)


def _write_event_summaries_and_clips(
    outdir: Path,
    event_log_context: EventLogContext,
    conv_segments_by_period: Dict[int, List[Tuple[int, int, int, int]]],
    nr_jobs: int,
) -> None:
    evt_by_team = event_log_context.event_counts_by_type_team
    rows_evt = []
    for (et, tm), cnt in sorted(evt_by_team.items()):
        tm_disp = event_log_context.team_display.get(tm, tm)
        rows_evt.append({"event_type": et, "team": tm_disp, "count": cnt})
    if rows_evt:
        pd.DataFrame(rows_evt).to_csv(outdir / "event_summary.csv", index=False)

    player_event_rows = event_log_context.event_player_rows or []
    if player_event_rows:
        def _fmt_v(x):
            return seconds_to_hhmmss(int(x)) if isinstance(x, int) else (seconds_to_hhmmss(int(x)) if isinstance(x, float) else "")
        def _fmt_g(x):
            return seconds_to_mmss_or_hhmmss(int(x)) if isinstance(x, int) else (seconds_to_mmss_or_hhmmss(int(x)) if isinstance(x, float) else "")
        rows = []
        for r in player_event_rows:
            tm = r.get('team')
            tm_disp = event_log_context.team_display.get(tm, tm)
            rows.append({
                'event_type': r.get('event_type'),
                'team': tm_disp,
                'player': r.get('player'),
                'jersey': r.get('jersey'),
                'period': r.get('period'),
                'video_time': _fmt_v(r.get('video_s')),
                'game_time': _fmt_g(r.get('game_s')),
            })
        pd.DataFrame(rows).to_csv(outdir / "event_players.csv", index=False)

    instances = event_log_context.event_instances or {}

    def map_sb_to_video(period: int, t_sb: int) -> Optional[int]:
        segs = conv_segments_by_period.get(period)
        if not segs:
            return None
        for s1, s2, v1, v2 in segs:
            lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
            if lo <= t_sb <= hi and s1 != s2:
                return int(round(v1 + (t_sb - s1) * (v2 - v1) / (s2 - s1)))
        return None

    max_sb_by_period: Dict[int, int] = {}
    for period, segs in conv_segments_by_period.items():
        mx = 0
        for s1, s2, _, _ in segs:
            mx = max(mx, s1, s2)
        if mx > 0:
            max_sb_by_period[period] = mx
    for _, lst in instances.items():
        for it in lst:
            p = it.get('period')
            gs = it.get('game_s')
            if isinstance(p, int) and isinstance(gs, (int, float)):
                v = int(gs)
                max_sb_by_period[p] = max(max_sb_by_period.get(p, 0), v)

    def merge_windows(win: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not win:
            return []
        win = sorted(win)
        out = [list(win[0])]
        for a, b in win[1:]:
            la, lb = out[-1]
            if a <= lb + 10:
                out[-1][1] = max(lb, b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    clip_scripts = []
    for (etype, team), lst in sorted(instances.items()):
        v_windows: List[Tuple[int, int]] = []
        sb_windows_by_period: Dict[int, List[Tuple[int, int]]] = {}
        for it in lst:
            p = it.get('period')
            v = it.get('video_s')
            g = it.get('game_s')
            vsec = None
            if isinstance(v, (int, float)):
                vsec = int(v)
            elif isinstance(g, (int, float)) and isinstance(p, int):
                vsec = map_sb_to_video(int(p), int(g))
            if vsec is not None:
                # Use shorter clips for CE/CBE/Rush, otherwise default to ±15s
                if etype in ("ControlledEntry", "ControlledExit", "Rush"):
                    pre, post = 10, 10
                else:
                    pre, post = 15, 15
                start = max(0, vsec - pre)
                end = vsec + post
                v_windows.append((start, end))
            if isinstance(g, (int, float)) and isinstance(p, int):
                gsec = int(g)
                sb_max = max_sb_by_period.get(int(p), None)
                # For CE/CBE/Rush scoreboard windows, also use ±10s
                if etype in ("ControlledEntry", "ControlledExit", "Rush"):
                    pre_sb, post_sb = 10, 10
                else:
                    pre_sb, post_sb = 15, 15
                sb_start = gsec + post_sb
                if sb_max is not None:
                    sb_start = min(sb_max, sb_start)
                sb_end = max(0, gsec - pre_sb)
                lo, hi = (sb_end, sb_start) if sb_end <= sb_start else (sb_start, sb_end)
                sb_windows_by_period.setdefault(int(p), []).append((lo, hi))

        v_windows = merge_windows(v_windows)
        if v_windows:
            team_disp = event_log_context.team_display.get(team, team)
            team_tag = sanitize_name(team_disp)
            vfile = outdir / f"events_{etype}_{team_tag}_video_times.txt"
            v_lines = [f"{seconds_to_hhmmss(a)} {seconds_to_hhmmss(b)}" for a, b in v_windows]
            vfile.write_text("\n".join(v_lines) + "\n", encoding="utf-8")
            script = outdir / f"clip_events_{etype}_{team_tag}.sh"
            label = f"{etype} ({team_disp})"
            body = f"""#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT=\"$1\"
OPP=\"$2\"
THIS_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
TS_FILE=\"$THIS_DIR/{vfile.name}\"
shift 2 || true
python -m hmlib.cli.video_clipper -j {nr_jobs} --input \"$INPUT\" --timestamps \"$TS_FILE\" --temp-dir \"$THIS_DIR/temp_clips/{etype}_{team_tag}\" \"{label} vs $OPP\" \"$@\"
"""
            script.write_text(body, encoding="utf-8")
            try:
                import os as _os
                _os.chmod(script, 0o755)
            except Exception:
                pass
            clip_scripts.append(script.name)

        if sb_windows_by_period:
            sfile = outdir / f"events_{etype}_{team_tag}_scoreboard_times.txt"
            s_lines = []
            for p, wins in sorted(sb_windows_by_period.items()):
                wins = merge_windows(wins)
                for lo, hi in wins:
                    s_lines.append(f"{p} {seconds_to_mmss_or_hhmmss(hi)} {seconds_to_mmss_or_hhmmss(lo)}")
            if s_lines:
                sfile.write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    # Build combined ControlledBoth (entries + exits) per team with 10s windows
    teams_seen = sorted({team for (_, team) in instances.keys()})
    for team in teams_seen:
        lst_combined = []
        for et in ("ControlledEntry", "ControlledExit"):
            lst_combined.extend(instances.get((et, team), []) or [])
        if not lst_combined:
            continue
        v_windows: List[Tuple[int, int]] = []
        sb_windows_by_period: Dict[int, List[Tuple[int, int]]] = {}
        for it in lst_combined:
            p = it.get('period')
            v = it.get('video_s')
            g = it.get('game_s')
            vsec = None
            if isinstance(v, (int, float)):
                vsec = int(v)
            elif isinstance(g, (int, float)) and isinstance(p, int):
                vsec = map_sb_to_video(int(p), int(g))
            if vsec is not None:
                start = max(0, vsec - 10)
                end = vsec + 10
                v_windows.append((start, end))
            if isinstance(g, (int, float)) and isinstance(p, int):
                gsec = int(g)
                sb_max = max_sb_by_period.get(int(p), None)
                sb_start = gsec + 10
                if sb_max is not None:
                    sb_start = min(sb_max, sb_start)
                sb_end = max(0, gsec - 10)
                lo, hi = (sb_end, sb_start) if sb_end <= sb_start else (sb_start, sb_end)
                sb_windows_by_period.setdefault(int(p), []).append((lo, hi))

        v_windows = merge_windows(v_windows)
        if v_windows:
            team_disp = event_log_context.team_display.get(team, team)
            team_tag = sanitize_name(team_disp)
            vfile = outdir / f"events_ControlledBoth_{team_tag}_video_times.txt"
            v_lines = [f"{seconds_to_hhmmss(a)} {seconds_to_hhmmss(b)}" for a, b in v_windows]
            vfile.write_text("\n".join(v_lines) + "\n", encoding="utf-8")
            script = outdir / f"clip_events_ControlledBoth_{team_tag}.sh"
            label = f"Controlled Entry/Exit ({team_disp})"
            body = f"""#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT=\"$1\"
OPP=\"$2\"
THIS_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
TS_FILE=\"$THIS_DIR/{vfile.name}\"
shift 2 || true
python -m hmlib.cli.video_clipper -j {nr_jobs} --input \"$INPUT\" --timestamps \"$TS_FILE\" --temp-dir \"$THIS_DIR/temp_clips/ControlledBoth_{team_tag}\" \"{label} vs $OPP\" \"$@\"
"""
            script.write_text(body, encoding="utf-8")
            try:
                import os as _os
                _os.chmod(script, 0o755)
            except Exception:
                pass
            clip_scripts.append(script.name)
        if sb_windows_by_period:
            sfile = outdir / f"events_ControlledBoth_{team_tag}_scoreboard_times.txt"
            s_lines = []
            for p, wins in sorted(sb_windows_by_period.items()):
                wins = merge_windows(wins)
                for lo, hi in wins:
                    s_lines.append(f"{p} {seconds_to_mmss_or_hhmmss(hi)} {seconds_to_mmss_or_hhmmss(lo)}")
            if s_lines:
                sfile.write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    if clip_scripts:
        all_script = outdir / "clip_events_all.sh"
        all_body = """#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT=\"$1\"
OPP=\"$2\"
shift 2 || true
THIS_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
for s in {scripts}; do
  echo \"Running $s...\"
  \"$THIS_DIR/$s\" \"$INPUT\" \"$OPP\" \"$@\"
done
""".replace("{scripts}", " ".join(sorted(clip_scripts)))
        all_script.write_text(all_body, encoding="utf-8")
        try:
            import os as _os
            _os.chmod(all_script, 0o755)
        except Exception:
            pass

    team_excluded = event_log_context.team_excluded or {}
    if any(v for v in team_excluded.values()):
        import sys as _sys
        for team, excl in team_excluded.items():
            if not excl:
                continue
            excl_str = ", ".join(str(x) for x in excl[:20])
            print(
                f"[warning] Team {team} exceeded 20 unique jerseys; additional jerseys seen: {excl_str} (data kept)",
                file=_sys.stderr,
            )


def _write_goal_window_files(
    outdir: Path,
    goals: List[GoalEvent],
    conv_segments_by_period: Dict[int, List[Tuple[int, int, int, int]]],
) -> None:
    if not goals:
        return

    def map_sb_to_video(period: int, t_sb: int) -> Optional[int]:
        segs = conv_segments_by_period.get(period)
        if not segs:
            return None
        for s1, s2, v1, v2 in segs:
            lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
            if lo <= t_sb <= hi and s1 != s2:
                return int(round(v1 + (t_sb - s1) * (v2 - v1) / (s2 - s1)))
        return None

    max_sb_by_period: Dict[int, int] = {}
    for period, segs in conv_segments_by_period.items():
        mx = 0
        for s1, s2, _, _ in segs:
            mx = max(mx, s1, s2)
        max_sb_by_period[period] = mx

    gf_lines: List[str] = []
    ga_lines: List[str] = []
    for ev in goals:
        sb_max = max_sb_by_period.get(ev.period, None)
        start_sb = ev.t_sec - 30
        end_sb = ev.t_sec + 10
        if sb_max is not None:
            start_sb = max(0, start_sb)
            end_sb = min(sb_max, end_sb)
        else:
            start_sb = max(0, start_sb)

        v_center = map_sb_to_video(ev.period, ev.t_sec)
        if v_center is not None:
            v_start = max(0, v_center - 30)
            v_end = v_center + 10
            start_str = seconds_to_hhmmss(v_start)
            end_str = seconds_to_hhmmss(v_end)
        else:
            v_start = map_sb_to_video(ev.period, start_sb)
            v_end = map_sb_to_video(ev.period, end_sb)
            if v_start is not None and v_end is not None:
                start_str = seconds_to_hhmmss(max(0, v_start))
                end_str = seconds_to_hhmmss(max(0, v_end))
            else:
                start_str = seconds_to_hhmmss(max(0, start_sb))
                end_str = seconds_to_hhmmss(max(0, end_sb))

        line = f"{start_str} {end_str}"
        if ev.kind == "GF":
            gf_lines.append(line)
        else:
            ga_lines.append(line)

    (outdir / "goals_for.txt").write_text("\n".join(gf_lines) + ("\n" if gf_lines else ""), encoding="utf-8")
    (outdir / "goals_against.txt").write_text("\n".join(ga_lines) + ("\n" if ga_lines else ""), encoding="utf-8")


def _write_clip_all_runner(outdir: Path) -> None:
    clip_all_path = outdir / "clip_all.sh"
    clip_all_body = """#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT=\"$1\"
OPP=\"$2\"
shift 2 || true
THIS_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
for s in \"$THIS_DIR\"/clip_*.sh; do
  [ -x \"$s\" ] || continue
  if [ \"$s\" = \"$THIS_DIR/clip_all.sh\" ]; then
    continue
  fi
  echo \"Running $s...\"
  \"$s\" \"$INPUT\" \"$OPP\" \"$@\"
done
"""
    clip_all_path.write_text(clip_all_body, encoding="utf-8")
    try:
        import os as _os
        _os.chmod(clip_all_path, 0o755)
    except Exception:
        pass


def process_sheet(
    xls_path: Path,
    sheet_name: Optional[str],
    outdir: Path,
    keep_goalies: bool,
    goals: List[GoalEvent],
    skip_validation: bool = False,
    nr_jobs: int = 4,
    t2s_game_id: Optional[int] = None,
    home: Optional[str] = None,
    away: Optional[str] = None,
) -> Path:
    target_sheet = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(xls_path, sheet_name=target_sheet, header=None)
    outdir.mkdir(parents=True, exist_ok=True)

    # Try event-log layout first
    (
        used_event_log,
        video_pairs_by_player,
        sb_pairs_by_player,
        conv_segments_by_period,
        event_log_context,
    ) = _parse_event_log_layout(df)

    validation_errors = 0
    if not used_event_log:
        (
            video_pairs_by_player,
            sb_pairs_by_player,
            conv_segments_by_period,
            validation_errors,
        ) = _parse_per_player_layout(df, keep_goalies=keep_goalies, skip_validation=skip_validation)

    # If this is the legacy per-player layout, keep the existing single-directory behavior.
    if not used_event_log:
        # Output subdir
        legacy_dir = outdir / "per_player"
        legacy_dir.mkdir(parents=True, exist_ok=True)

        # Per-player time files and clip scripts
        _write_video_times_and_scripts(legacy_dir, video_pairs_by_player, nr_jobs)
        _write_scoreboard_times(legacy_dir, sb_pairs_by_player)

        # Stats and plus/minus
        goals_by_period: Dict[int, List[GoalEvent]] = {}
        for ev in goals:
            goals_by_period.setdefault(ev.period, []).append(ev)

        stats_table_rows: List[Dict[str, str]] = []
        all_periods_seen: set[int] = set()
        for player_key, sb_list in sb_pairs_by_player.items():
            sb_by_period: Dict[int, List[Tuple[str, str]]] = {}
            for period, a, b in sb_list:
                sb_by_period.setdefault(period, []).append((a, b))
            all_pairs = [(a, b) for (_, a, b) in sb_list]
            shift_summary = summarize_shift_lengths_sec(all_pairs)
            per_period_toi_map = per_period_toi(sb_by_period)

            plus_minus = 0
            counted_gf: List[str] = []
            counted_ga: List[str] = []
            counted_gf_by_period: Dict[int, int] = {}
            counted_ga_by_period: Dict[int, int] = {}
            for period, pairs in sb_by_period.items():
                if period not in goals_by_period:
                    continue
                for ev in goals_by_period[period]:
                    matched = False
                    for a, b in pairs:
                        a_sec = parse_flex_time_to_seconds(a)
                        b_sec = parse_flex_time_to_seconds(b)
                        lo, hi = (a_sec, b_sec) if a_sec <= b_sec else (b_sec, a_sec)
                        if not (lo <= ev.t_sec <= hi):
                            continue
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

            stats_lines = []
            stats_lines.append(f"Player: {player_key}")
            stats_lines.append(f"Shifts (scoreboard): {shift_summary['num_shifts']}")
            stats_lines.append(f"TOI total (scoreboard): {shift_summary['toi_total']}")
            stats_lines.append(f"Avg shift: {shift_summary['toi_avg']}")
            stats_lines.append(f"Median shift: {shift_summary['toi_median']}")
            stats_lines.append(f"Longest shift: {shift_summary['toi_longest']}")
            stats_lines.append(f"Shortest shift: {shift_summary['toi_shortest']}")
            if per_period_toi_map:
                stats_lines.append("Per-period TOI (scoreboard):")
                for period in sorted(per_period_toi_map.keys()):
                    stats_lines.append(f"  Period {period}: {per_period_toi_map[period]}")
            stats_lines.append(f"Plus/Minus: {plus_minus}")
            # PP/SH shift counts (from shift tables)
            pp_ct = 0
            sh_ct = 0
            if event_log_context is not None:
                base_key_num = str(player_key).split("_", 1)[0]
                try:
                    int(base_key_num)
                    pk_pref = f"{team}_{base_key_num}"
                    pp_ct = (event_log_context.pp_shifts_by_player or {}).get(pk_pref, 0)
                    sh_ct = (event_log_context.sh_shifts_by_player or {}).get(pk_pref, 0)
                except Exception:
                    pass
            stats_lines.append(f"PP shifts: {pp_ct}")
            stats_lines.append(f"SH shifts: {sh_ct}")
            if counted_gf:
                stats_lines.append("  GF counted at: " + ", ".join(sorted(counted_gf)))
            if counted_ga:
                stats_lines.append("  GA counted at: " + ", ".join(sorted(counted_ga)))

            for period, pairs in sorted(sb_by_period.items()):
                stats_lines.append(f"Shifts in Period {period}: {len(pairs)}")

            (legacy_dir / f"{player_key}_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

            row_map: Dict[str, str] = {
                "player": player_key,
                "shifts": shift_summary["num_shifts"],
                "plus_minus": str(plus_minus),
                "sb_toi_total": shift_summary["toi_total"],
                "sb_avg": shift_summary["toi_avg"],
                "sb_median": shift_summary["toi_median"],
                "sb_longest": shift_summary["toi_longest"],
                "sb_shortest": shift_summary["toi_shortest"],
            }
            row_map["gf_counted"] = str(len(counted_gf))
            row_map["ga_counted"] = str(len(counted_ga))
            v_pairs = video_pairs_by_player.get(player_key, [])
            if v_pairs:
                v_sum = 0
                for a, b in v_pairs:
                    lo, hi = compute_interval_seconds(a, b)
                    v_sum += hi - lo
                row_map["video_toi_total"] = seconds_to_mmss_or_hhmmss(v_sum)
            else:
                row_map["video_toi_total"] = ""
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

        # Global CSV, consolidated stats, and goal windows
        _write_global_summary_csv(legacy_dir, sb_pairs_by_player)
        if stats_table_rows:
            _write_player_stats_text_and_csv(legacy_dir, stats_table_rows, sorted(all_periods_seen))
        _write_goal_window_files(legacy_dir, goals, conv_segments_by_period)
        _write_clip_all_runner(legacy_dir)

        if (not skip_validation) and validation_errors > 0:
            print(
                f"[validation] Completed with {validation_errors} issue(s). See messages above.",
                file=sys.stderr,
            )
        return legacy_dir

    # Event-log layout: split into team subdirectories and derive goals from left table
    # Build team-specific goal events from parsed instances
    goals_blue: List[GoalEvent] = []
    goals_white: List[GoalEvent] = []
    if event_log_context is not None:
        instances = event_log_context.event_instances or {}

        def _mk_goals_for(team_name: str) -> List[GoalEvent]:
            lst: List[GoalEvent] = []
            for it in instances.get(("Goal", team_name), []) or []:
                p = it.get("period")
                gs = it.get("game_s")
                if isinstance(p, int) and isinstance(gs, (int, float)):
                    sec = int(gs)
                    mm, ss = divmod(sec, 60)
                    lst.append(GoalEvent("GF", int(p), f"{mm}:{ss:02d}"))
            lst.sort(key=lambda e: (e.period, e.t_sec))
            return lst

        blue_gf = _mk_goals_for("Blue")
        white_gf = _mk_goals_for("White")
        goals_blue = blue_gf + [GoalEvent("GA", g.period, g.t_str) for g in white_gf]
        goals_white = white_gf + [GoalEvent("GA", g.period, g.t_str) for g in blue_gf]

    # If a TimeToScore game id was provided, fetch rosters and try to map Blue/White
    # to home/away using jersey overlap (and team name similarity), then prepare
    # per-color roster maps for name enrichment and jersey validation.
    roster_by_color: Dict[str, Dict[int, str]] = {}
    if used_event_log and t2s_game_id is not None and t2s_api is not None and event_log_context is not None:
        try:
            info = t2s_api.get_game_details(int(t2s_game_id))
            stats = info.get("stats") or {}

            def _build_roster(rows: Any) -> Dict[int, str]:
                out: Dict[int, str] = {}
                for r in rows or []:
                    try:
                        num = int(str((r or {}).get("number")).strip())
                    except Exception:
                        continue
                    name = str((r or {}).get("name") or "").strip()
                    if not (1 <= num <= 99):
                        continue
                    if not name:
                        continue
                    out[num] = name
                return out

            home_roster = _build_roster((stats or {}).get("homePlayers"))
            away_roster = _build_roster((stats or {}).get("awayPlayers"))

            # Observed sheet jersey sets — prefer shift tables (less ambiguous) over left events
            if event_log_context.team_shift_roster:
                blue_set = set(event_log_context.team_shift_roster.get("Blue", []) or [])
                white_set = set(event_log_context.team_shift_roster.get("White", []) or [])
            else:
                blue_set = set(event_log_context.team_roster.get("Blue", []) or [])
                white_set = set(event_log_context.team_roster.get("White", []) or [])

            # Simple side inference by overlap + name similarity
            def _norm(s: str) -> str:
                import re as _re
                return _re.sub(r"[^a-z0-9]", "", (s or "").lower())

            td_blue = event_log_context.team_display.get("Blue", "Blue")
            td_white = event_log_context.team_display.get("White", "White")
            home_name = (((info.get("home") or {}).get("team") or {}) or {}).get("name") or "Home"
            away_name = (((info.get("away") or {}).get("team") or {}) or {}).get("name") or "Away"
            n_td_blue, n_td_white = _norm(td_blue), _norm(td_white)
            n_home, n_away = _norm(home_name), _norm(away_name)

            def _name_score(a: str, b: str) -> int:
                if not a or not b:
                    return 0
                if a == b:
                    return 3
                if a in b or b in a:
                    return 2
                return 0

            blue_home_ov = len(blue_set & set(home_roster.keys()))
            blue_away_ov = len(blue_set & set(away_roster.keys()))
            white_home_ov = len(white_set & set(home_roster.keys()))
            white_away_ov = len(white_set & set(away_roster.keys()))

            # Optional explicit override from CLI: --home=Blue/White or --away=Blue/White
            def _norm_color(s: Optional[str]) -> Optional[str]:
                if not s:
                    return None
                sl = s.strip().lower()
                if sl in ("blue", "white"):
                    return sl.capitalize()
                return None

            home_color = _norm_color(home)
            away_color = _norm_color(away)
            if home_color and away_color and home_color == away_color:
                # invalid, ignore away_color
                away_color = None
            if home_color:
                color_to_side = {home_color: "home", ("White" if home_color == "Blue" else "Blue"): "away"}
            elif away_color:
                color_to_side = {away_color: "away", ("White" if away_color == "Blue" else "Blue"): "home"}
            else:
                # Primary criterion: choose assignment that minimizes mismatches
                mismatch_home = len([n for n in blue_set if n not in home_roster]) + len(
                    [n for n in white_set if n not in away_roster]
                )
                mismatch_away = len([n for n in blue_set if n not in away_roster]) + len(
                    [n for n in white_set if n not in home_roster]
                )

                # Secondary tiebreaker: overlap counts + name similarity
                blue_home_score = blue_home_ov + _name_score(n_td_blue, n_home)
                blue_away_score = blue_away_ov + _name_score(n_td_blue, n_away)
                white_home_score = white_home_ov + _name_score(n_td_white, n_home)
                white_away_score = white_away_ov + _name_score(n_td_white, n_away)

                if mismatch_home < mismatch_away:
                    color_to_side = {"Blue": "home", "White": "away"}
                elif mismatch_away < mismatch_home:
                    color_to_side = {"Blue": "away", "White": "home"}
                else:
                    # Tie: prefer higher combined overlap+name score
                    score_home = blue_home_score + white_away_score
                    score_away = blue_away_score + white_home_score
                    if score_home >= score_away:
                        color_to_side = {"Blue": "home", "White": "away"}
                    else:
                        color_to_side = {"Blue": "away", "White": "home"}

            roster_by_color = {
                "Blue": home_roster if color_to_side.get("Blue") == "home" else away_roster,
                "White": away_roster if color_to_side.get("Blue") == "home" else home_roster,
            }

            # Log the decision for visibility
            try:
                import sys as _sys

                # Make mismatch counts only if we computed them
                if 'mismatch_home' in locals() and 'mismatch_away' in locals():
                    mm_info = f"; mismatch if H/A: {mismatch_home}/{mismatch_away}"
                else:
                    mm_info = ""
                print((
                    f"[t2s] Blue-> {color_to_side['Blue'].upper()} (ovl H={blue_home_ov}, A={blue_away_ov}{mm_info}). "
                    f"Home='{home_name}', Away='{away_name}'."
                ), file=_sys.stderr)
            except Exception:
                pass

            # Validation: any jersey numbers observed but not on roster
            import sys as _sys
            for color, seen in ("Blue", blue_set), ("White", white_set):
                roster_nums = set((roster_by_color.get(color) or {}).keys())
                missing = sorted([n for n in seen if n not in roster_nums])
                if missing:
                    team_disp = event_log_context.team_display.get(color, color)
                    print(
                        f"[validation] ROSTER | Team={color} -> {team_disp} | jersey(s) not on T2S roster: "
                        + ", ".join(str(x) for x in missing[:20]),
                        file=_sys.stderr,
                    )
        except Exception:
            roster_by_color = {}

    # Helper to split dicts like {"Blue_11": [...], "White_9": [...]} into per-team without prefix
    def _split_prefix(prefix: str, m: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        pref = prefix + "_"
        for k, v in m.items():
            if k.startswith(pref):
                out[k[len(pref) :]] = v
        return out

    teams = []
    if any(k.startswith("Blue_") for k in video_pairs_by_player) or any(
        k.startswith("Blue_") for k in sb_pairs_by_player
    ):
        teams.append("Blue")
    if any(k.startswith("White_") for k in video_pairs_by_player) or any(
        k.startswith("White_") for k in sb_pairs_by_player
    ):
        teams.append("White")

    root_out = outdir  # keep root as the return value
    for team in teams:
        team_disp = event_log_context.team_display.get(team, team) if event_log_context is not None else team
        team_dirname = sanitize_name(team_disp)
        team_dir = root_out / team_dirname
        team_dir.mkdir(parents=True, exist_ok=True)

        v_pairs_team = _split_prefix(team, video_pairs_by_player)
        sb_pairs_team = _split_prefix(team, sb_pairs_by_player)

        # If a roster is available for this color, rename keys to include player names
        roster_map = roster_by_color.get(team) if roster_by_color else None
        if roster_map:
            def _remap_keys(m: Dict[str, Any]) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for k, v in m.items():
                    base = str(k).strip()
                    try:
                        num = int(base.split("_", 1)[0])
                    except Exception:
                        out[k] = v
                        continue
                    nm = roster_map.get(num)
                    if nm:
                        nk = f"{num}_{sanitize_name(nm)}"
                    else:
                        nk = base
                    out[nk] = v
                return out
            v_pairs_team = _remap_keys(v_pairs_team)
            sb_pairs_team = _remap_keys(sb_pairs_team)

        _write_video_times_and_scripts(team_dir, v_pairs_team, nr_jobs)
        _write_scoreboard_times(team_dir, sb_pairs_team)

        # Team-specific goals_by_period
        team_goal_events = goals_blue if team == "Blue" else goals_white
        goals_by_period: Dict[int, List[GoalEvent]] = {}
        for ev in team_goal_events:
            goals_by_period.setdefault(ev.period, []).append(ev)

        stats_table_rows: List[Dict[str, str]] = []
        all_periods_seen: set[int] = set()
        for player_key, sb_list in sb_pairs_team.items():
            sb_by_period: Dict[int, List[Tuple[str, str]]] = {}
            for period, a, b in sb_list:
                sb_by_period.setdefault(period, []).append((a, b))
            all_pairs = [(a, b) for (_, a, b) in sb_list]
            shift_summary = summarize_shift_lengths_sec(all_pairs)
            per_period_toi_map = per_period_toi(sb_by_period)

            plus_minus = 0
            counted_gf: List[str] = []
            counted_ga: List[str] = []
            counted_gf_by_period: Dict[int, int] = {}
            counted_ga_by_period: Dict[int, int] = {}
            for period, pairs in sb_by_period.items():
                if period not in goals_by_period:
                    continue
                for ev in goals_by_period[period]:
                    matched = False
                    for a, b in pairs:
                        a_sec = parse_flex_time_to_seconds(a)
                        b_sec = parse_flex_time_to_seconds(b)
                        lo, hi = (a_sec, b_sec) if a_sec <= b_sec else (b_sec, a_sec)
                        if not (lo <= ev.t_sec <= hi):
                            continue
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

            stats_lines = []
            stats_lines.append(f"Player: {player_key}")
            stats_lines.append(f"Shifts (scoreboard): {shift_summary['num_shifts']}")
            stats_lines.append(f"TOI total (scoreboard): {shift_summary['toi_total']}")
            stats_lines.append(f"Avg shift: {shift_summary['toi_avg']}")
            stats_lines.append(f"Median shift: {shift_summary['toi_median']}")
            stats_lines.append(f"Longest shift: {shift_summary['toi_longest']}")
            stats_lines.append(f"Shortest shift: {shift_summary['toi_shortest']}")
            if per_period_toi_map:
                stats_lines.append("Per-period TOI (scoreboard):")
                for p in sorted(per_period_toi_map.keys()):
                    stats_lines.append(f"  Period {p}: {per_period_toi_map[p]}")
            stats_lines.append(f"Plus/Minus: {plus_minus}")
            if counted_gf:
                stats_lines.append("  GF counted at: " + ", ".join(sorted(counted_gf)))
            if counted_ga:
                stats_lines.append("  GA counted at: " + ", ".join(sorted(counted_ga)))

            for p, pairs in sorted(sb_by_period.items()):
                stats_lines.append("")
                stats_lines.append(f"Period {p} shifts:")
                for a, b in pairs:
                    stats_lines.append(f"  {a} -> {b}")
            # On-ice event counts (For/Against) from left table
            if event_log_context is not None:
                oc_all = event_log_context.on_ice_counts_by_player or {}
                # on-ice counts keyed by team_jersey; extract jersey component
                base_key = str(player_key).split("_", 1)[0]
                pref_key = f"{team}_{base_key}"
                oc = oc_all.get(pref_key, {})
                if oc:
                    stats_lines.append("")
                    stats_lines.append("Event Counts (On-Ice):")
                    ev_order = [
                        "Shot",
                        "SOG",
                        "Goal",
                        "Assist",
                        "ControlledEntry",
                        "ControlledExit",
                        "ExpectedGoal",
                        "Rush",
                    ]
                    for tname in ev_order:
                        fkey = f"{tname}_For"
                        akey = f"{tname}_Against"
                        fv = oc.get(fkey, 0)
                        av = oc.get(akey, 0)
                        if fv or av:
                            stats_lines.append(f"  {tname}: For {fv}, Against {av}")
            (team_dir / f"{player_key}_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

            # Row for team summary CSV/text
            row_map: Dict[str, str] = {
                "player": player_key,
                "shifts": shift_summary["num_shifts"],
                "plus_minus": str(plus_minus),
                "sb_toi_total": shift_summary["toi_total"],
                "sb_avg": shift_summary["toi_avg"],
                "sb_median": shift_summary["toi_median"],
                "sb_longest": shift_summary["toi_longest"],
                "sb_shortest": shift_summary["toi_shortest"],
            }
            row_map["gf_counted"] = str(len(counted_gf))
            row_map["ga_counted"] = str(len(counted_ga))
            v_pairs = v_pairs_team.get(player_key, [])
            if v_pairs:
                v_sum = 0
                for a, b in v_pairs:
                    lo, hi = compute_interval_seconds(a, b)
                    v_sum += hi - lo
                row_map["video_toi_total"] = seconds_to_mmss_or_hhmmss(v_sum)
            else:
                row_map["video_toi_total"] = ""
            for p, toi in per_period_toi_map.items():
                row_map[f"P{p}_toi"] = toi
                all_periods_seen.add(p)
            for p, pairs in sb_by_period.items():
                row_map[f"P{p}_shifts"] = str(len(pairs))
            for p, cnt in counted_gf_by_period.items():
                row_map[f"P{p}_GF"] = str(cnt)
                all_periods_seen.add(p)
            for p, cnt in counted_ga_by_period.items():
                row_map[f"P{p}_GA"] = str(cnt)
                all_periods_seen.add(p)

            # Include PP/SH shift counts in CSV row
            pp_ct = 0
            sh_ct = 0
            if event_log_context is not None:
                base_key_num = str(player_key).split("_", 1)[0]
                try:
                    int(base_key_num)
                    pk_pref = f"{team}_{base_key_num}"
                    pp_ct = (event_log_context.pp_shifts_by_player or {}).get(pk_pref, 0)
                    sh_ct = (event_log_context.sh_shifts_by_player or {}).get(pk_pref, 0)
                except Exception:
                    pass
            row_map["pp_shifts"] = str(pp_ct)
            row_map["sh_shifts"] = str(sh_ct)

            # Add on-ice event counts into row_map for CSV
            if event_log_context is not None:
                oc_all = event_log_context.on_ice_counts_by_player or {}
                base_key = str(player_key).split("_", 1)[0]
                pref_key = f"{team}_{base_key}"
                oc = oc_all.get(pref_key, {})
                for k, v in oc.items():
                    # k like 'Shot_For'
                    row_map[k] = str(v)

            stats_table_rows.append(row_map)

        # Team-level summaries
        _write_global_summary_csv(
            team_dir,
            sb_pairs_team,
            pp_map=(event_log_context.pp_shifts_by_player if event_log_context is not None else None),
            sh_map=(event_log_context.sh_shifts_by_player if event_log_context is not None else None),
            team_color=team,
        )
        if stats_table_rows:
            _write_player_stats_text_and_csv(team_dir, stats_table_rows, sorted(all_periods_seen))
        if team_goal_events:
            _write_goal_window_files(team_dir, team_goal_events, conv_segments_by_period)

        _write_clip_all_runner(team_dir)

    # For event-log layout, also write a one-time aggregate event summary at the root
    if event_log_context is not None:
        _write_event_summaries_and_clips(root_out, event_log_context, conv_segments_by_period, nr_jobs)

    return root_out


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
    # TimeToScore integration
    p.add_argument(
        "--t2s",
        type=int,
        default=None,
        help=(
            "TimeToScore game id. If set and no --goal/--goals-file provided, fetch goals from T2S (auto-detects side when possible). "
            "For two-team sheets, rosters are used to name players and side flags are not required."
        ),
    )
    # Force Blue/White mapping to Home/Away (two-team event-log sheets)
    p.add_argument(
        "--home",
        type=str,
        choices=["Blue", "White", "blue", "white"],
        default=None,
        help="Force which color (Blue/White) maps to the Home roster (two-team event-log).",
    )
    p.add_argument(
        "--away",
        type=str,
        choices=["Blue", "White", "blue", "white"],
        default=None,
        help="Force which color (Blue/White) maps to the Away roster (two-team event-log).",
    )
    # Legacy per-player sheets GF/GA perspective when fetching T2S goals
    p.add_argument(
        "--your-side",
        type=str,
        choices=["home", "away"],
        default=None,
        help="For per-player sheets with --t2s and no manual goals: specify if your team is 'home' or 'away'.",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks on start/end ordering and excessive durations.",
    )
    p.add_argument(
        "--nr-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for video clipping scripts (default: 4)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    goals = load_goals(args.goal, args.goals_file)

    # If no manual goals provided, and a T2S game id is given, optionally fetch goals
    # from T2S. In two-team event-log layout, goals are derived from the sheet's
    # left event table, so --home/--away is not required. For legacy per-player sheets,
    # infer side automatically by jersey overlap when possible; otherwise fall back to flags.
    if not goals and args.t2s is not None:
        try:
            target_sheet = 0 if args.sheet is None else args.sheet
            df_peek = pd.read_excel(args.input, sheet_name=target_sheet, header=None)
            used_ev, _vp, _sp, _conv, ctx = _parse_event_log_layout(df_peek)
        except Exception:
            used_ev, ctx = False, None

        # If color mapping flags are present, assume two-team event-log mode even if
        # header detection is inconclusive. In event-log mode, goals come from the
        # left event table; do not fetch T2S goals.
        def _norm_color(s: Optional[str]) -> Optional[str]:
            if not s:
                return None
            sl = s.strip().lower()
            if sl in ("blue", "white"):
                return sl.capitalize()
            return None

        forced_home = _norm_color(args.home)
        forced_away = _norm_color(args.away)
        assume_event_log = used_ev or bool(forced_home or forced_away)

        # Two-team event log -> validate color flags if provided and skip T2S goals
        if assume_event_log:
            # Validate color mapping flags if provided
            hcol = forced_home
            acol = forced_away
            if hcol and acol and hcol == acol:
                print("Error: --home and --away cannot be the same color.", file=sys.stderr)
                sys.exit(2)
            if (args.home or args.away) and not (hcol or acol):
                print("Error: --home/--away must be 'Blue' or 'White'.", file=sys.stderr)
                sys.exit(2)
        else:
            # Legacy per-player sheet: fetch from T2S. Try auto-detect if possible.
            side: Optional[str] = None
            if t2s_api is not None:
                try:
                    # Try overlap against any jersey numbers present in the sheet first column
                    # (jersey) for a simple heuristic.
                    df0 = df_peek if 'df_peek' in locals() else pd.read_excel(args.input, sheet_name=target_sheet, header=None)
                    sheet_nums: set[int] = set()
                    try:
                        for i in range(min(200, df0.shape[0])):
                            v = df0.iat[i, 0]
                            if isinstance(v, (int, float)):
                                n = int(v)
                                if 1 <= n <= 98:
                                    sheet_nums.add(n)
                            elif isinstance(v, str):
                                for m in re.finditer(r"#?(\d{1,2})(?!\d)", v):
                                    n = int(m.group(1))
                                    if 1 <= n <= 98:
                                        sheet_nums.add(n)
                    except Exception:
                        sheet_nums = set()

                    info = t2s_api.get_game_details(int(args.t2s))
                    stats = info.get("stats") or {}
                    def _nums(rows: Any) -> set[int]:
                        out: set[int] = set()
                        for r in rows or []:
                            try:
                                out.add(int(str(r.get("number")).strip()))
                            except Exception:
                                pass
                        return out
                    home_set = _nums(stats.get("homePlayers"))
                    away_set = _nums(stats.get("awayPlayers"))
                    # Decide side by which T2S roster overlaps more with jersey numbers in sheet
                    home_overlap = len(sheet_nums & home_set)
                    away_overlap = len(sheet_nums & away_set)
                    if home_overlap > 0 or away_overlap > 0:
                        side = "home" if home_overlap >= away_overlap else "away"
                        print(
                            f"[t2s] Auto-detected your side as {side.upper()} (overlap home={home_overlap}, away={away_overlap}).",
                            file=sys.stderr,
                        )
                except Exception:
                    side = None

            # Fallback to explicit flag if auto-detect was not possible
            if side is None:
                if args.your_side in ("home", "away"):
                    side = args.your_side
                else:
                    print(
                        "Error: --t2s was provided but --your-side was not specified, and auto-detection failed.",
                        file=sys.stderr,
                    )
                    sys.exit(2)

            try:
                goals = goals_from_t2s(int(args.t2s), side=side)
                if not goals:
                    print(
                        f"[t2s] No goals found for game {args.t2s}; continuing without GF/GA.",
                        file=sys.stderr,
                    )
                else:
                    for g in reversed(sorted([str(g) for g in goals])):
                        print(g)
            except Exception as e:  # noqa: BLE001
                print(f"[t2s] Failed to fetch goals for game {args.t2s}: {e}", file=sys.stderr)
                # proceed with empty goals
    final_outdir = process_sheet(
        xls_path=args.input,
        sheet_name=args.sheet,
        outdir=args.outdir,
        keep_goalies=args.keep_goalies,
        goals=goals,
        skip_validation=args.skip_validation,
        nr_jobs=int(args.nr_jobs),
        t2s_game_id=args.t2s,
        home=args.home,
        away=args.away,
    )
    try:
        print(f"✅ Done. Wrote per-player files to: {final_outdir.resolve()}")
    except Exception:
        print("✅ Done.")


if __name__ == "__main__":
    main()

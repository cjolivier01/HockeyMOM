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

Alternatively, provide a TimeToScore game id to auto-fill goals:
  --t2s 51602 --home   # Your team is home (home scoring = GF)
  --t2s 51602 --away   # Your team is away (away scoring = GF)

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
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

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
    """
    starts = [str(row[c]).strip() for c in start_cols if pd.notna(row[c]) and str(row[c]).strip()]
    ends = [str(row[c]).strip() for c in end_cols if pd.notna(row[c]) and str(row[c]).strip()]
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


def process_sheet(
    xls_path: Path,
    sheet_name: Optional[str],
    outdir: Path,
    keep_goalies: bool,
    goals: List[GoalEvent],
    skip_validation: bool = False,
) -> Path:
    # If sheet_name is None, pandas returns a dict of DataFrames.
    # We want the first sheet by default, so coerce None -> 0.
    target_sheet = 0 if sheet_name is None else sheet_name
    df = pd.read_excel(xls_path, sheet_name=target_sheet, header=None)
    outdir.mkdir(parents=True, exist_ok=True)

    # Accumulators (used by both formats)
    video_pairs_by_player: Dict[str, List[Tuple[str, str]]] = {}
    sb_pairs_by_player: Dict[str, List[Tuple[int, str, str]]] = {}
    # For scoreboard->video conversion: collect mapping segments per period
    # Each segment: (sb_start_sec, sb_end_sec, v_start_sec, v_end_sec)
    conv_segments_by_period: Dict[int, List[Tuple[int, int, int, int]]] = {}

    # -------- Attempt to parse "Shifts (Blue)/(white)" event-log layout --------
    used_event_log = False

    def _find_cell(value: str) -> Optional[Tuple[int, int]]:
        needle = value.strip().lower()
        for rr in range(df.shape[0]):
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

    if blue_hdr or white_hdr:
        used_event_log = True
        # Roster cap: at most 20 unique jerseys per team
        MAX_TEAM_PLAYERS = 20
        team_roster: Dict[str, List[int]] = {}
        team_excluded: Dict[str, List[int]] = {}

        def _register_and_flag(team: str, jerseys: List[int]) -> List[int]:
            if not team:
                return []
            # de-dup preserve order
            seen_local = set()
            ordered = []
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
            # Return all jerseys (data kept), only flagged if beyond 20
            return ordered

        def _parse_event_time(cell: Any) -> Optional[int]:
            if cell is None or (isinstance(cell, str) and not cell.strip()):
                return None
            # Accept values like 01:43:00 (H:MM:SS) or 24:45 (M:SS) or 1:02
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
            players_width = 12

            # Build events
            current_period_label: Optional[str] = None
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
                for k in range(players_width):
                    c = players_start + k
                    if c >= df.shape[1]:
                        break
                    val = df.iat[r, c]
                    if pd.isna(val):
                        continue
                    # Accept only jersey-like values 1..98
                    if isinstance(val, (int, float)):
                        n = int(val)
                        if 1 <= n <= 98:
                            players.append(n)
                        continue
                    if hasattr(val, "hour") and hasattr(val, "minute"):
                        # datetime.time -> ignore
                        continue
                    s = str(val).strip()
                    if not s:
                        continue
                    if s.upper() in {"PP", "SH"}:
                        continue
                    # Ignore time-like strings
                    if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", s):
                        continue
                    for m in re.finditer(r"#?(\d{1,2})(?!\d)", s):
                        try:
                            n = int(m.group(1))
                        except Exception:
                            continue
                        if 1 <= n <= 98:
                            players.append(n)
                # Deduplicate
                if players:
                    players = sorted(set(players))
                if vsec is None and gsec is None and not players:
                    continue
                # Register and flag if team exceeds 20, but keep all players
                players = _register_and_flag(team_prefix, players)
                events.append({
                    "period": _period_num_from_label(current_period_label),
                    "v": vsec,
                    "g": gsec,
                    "players": players,
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
                                conv_segments_by_period.setdefault(last_period, []).append(
                                    (int(sg), int(egg), int(sv), int(evv))
                                )
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
                                conv_segments_by_period.setdefault(cur_p, []).append(
                                    (int(sg), int(end_g), int(sv), int(evv))
                                )
                        del open_shift[pid]

                # Open shifts for players now on
                for pid in on_ice:
                    if pid not in open_shift:
                        open_shift[pid] = {"sv": ev.get("v"), "sg": ev.get("g"), "period": ev.get("period")}

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
                            conv_segments_by_period.setdefault(int(per), []).append(
                                (int(sg), int(egg), int(sv), int(evv))
                            )

        if blue_hdr:
            _parse_event_block(blue_hdr, "Blue")
        if white_hdr:
            _parse_event_block(white_hdr, "White")

        # ---- Parse left-side event columns (shots, goals, assists, entries, exits, expected goal) ----
        event_counts_by_player: Dict[str, Dict[str, int]] = {}
        event_counts_by_type_team: Dict[Tuple[str, str], int] = {}
        event_instances: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        event_player_rows: List[Dict[str, Any]] = []

        def _record_event(kind: str, team: Optional[str], jersey_list: List[int], period_label: Optional[str], vsec: Optional[int], gsec: Optional[int]) -> None:
            if not team:
                return
            # team/type count
            event_counts_by_type_team[(kind, team)] = event_counts_by_type_team.get((kind, team), 0) + 1
            # cap jerseys per team and update per-player stats
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
                event_player_rows.append({
                    'event_type': kind,
                    'team': team,
                    'player': pk,
                    'jersey': int(j),
                    'period': period_num,
                    'video_s': vsec,
                    'game_s': gsec,
                })
            # instance for windowing (one per row event)
            event_instances.setdefault((kind, team), []).append({
                'period': period_num,
                'video_s': vsec,
                'game_s': gsec,
            })

        # Find left header row and relevant columns up to the Blue/White shifts header column
        limit_col = (blue_hdr[1] if blue_hdr else df.shape[1])
        left_header_row = None
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
            # tolerate wording variations and capitalization
            entries_col = None
            exits_col = None
            for k, c in label_to_col.items():
                kl = k.lower()
                if "controlled" in kl and "blue" in kl and "entr" in kl:
                    entries_col = c
                if "controlled" in kl and "exit" in kl:
                    exits_col = c

            # Guess team column: choose the col in [0, limit_col) with most exact 'Blue'/'White'
            team_col = None
            best_count = 0
            for c in range(min(limit_col, df.shape[1])):
                cnt = 0
                for r in range(left_header_row + 1, min(df.shape[0], left_header_row + 80)):
                    v = df.iat[r, c]
                    if isinstance(v, str) and v.strip() in ("Blue", "White"):
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
                # Ignore time-like strings wholesale
                if re.match(r"^\d{1,2}:\d{2}(?::\d{2})?$", s):
                    return []
                nums: List[int] = []
                # Prefer patterns like '#59' or '59' with 1-2 digits
                for m in re.finditer(r"#?(\d{1,2})(?!\d)", s):
                    try:
                        n = int(m.group(1))
                    except Exception:
                        continue
                    if 1 <= n <= 98:
                        nums.append(n)
                # Dedup while preserving order
                seen = set()
                out: List[int] = []
                for n in nums:
                    if n in seen:
                        continue
                    seen.add(n)
                    out.append(n)
                return out

            # Walk data rows to collect events
            current_period = None
            for r in range(left_header_row + 1, df.shape[0]):
                # Update period if any cell contains 'Period'
                row_vals = [df.iat[r, c] for c in range(min(limit_col, df.shape[1]))]
                for v in row_vals:
                    if isinstance(v, str) and "period" in v.lower():
                        current_period = v.strip()
                        break

                # Team value for this row, if present
                team_val = None
                if team_col is not None:
                    tv = df.iat[r, team_col]
                    if isinstance(tv, str) and tv.strip() in ("Blue", "White"):
                        team_val = tv.strip()
                # Row times (center)
                row_vsec = _parse_event_time(df.iat[r, vt_col]) if vt_col is not None else None
                row_gsec = _parse_event_time(df.iat[r, gt_col]) if gt_col is not None else None

                # Shots
                if shots_col is not None:
                    sv = df.iat[r, shots_col]
                    if isinstance(sv, str) and sv.strip():
                        t = team_val or _parse_team_from_text(sv)
                        jerseys = _extract_nums(sv)
                        _record_event("Shot", t, jerseys, current_period, row_vsec, row_gsec)

                # Goals
                if goals_col is not None:
                    gv = df.iat[r, goals_col]
                    if isinstance(gv, str) and gv.strip():
                        t = team_val or _parse_team_from_text(gv)
                        jerseys = _extract_nums(gv)
                        _record_event("Goal", t, jerseys, current_period, row_vsec, row_gsec)

                # Assists (may list multiple jerseys)
                if assists_col is not None:
                    av = df.iat[r, assists_col]
                    if isinstance(av, str) and av.strip():
                        t = team_val or _parse_team_from_text(av)
                        jerseys = _extract_nums(av)
                        _record_event("Assist", t, jerseys, current_period, row_vsec, row_gsec)

                # Controlled entries
                if entries_col is not None:
                    ev = df.iat[r, entries_col]
                    if isinstance(ev, str) and ev.strip():
                        t = team_val or _parse_team_from_text(ev)
                        jerseys = _extract_nums(ev)
                        _record_event("ControlledEntry", t, jerseys, current_period, row_vsec, row_gsec)

                # Controlled exits
                if exits_col is not None:
                    xv = df.iat[r, exits_col]
                    if isinstance(xv, str) and xv.strip():
                        t = team_val or _parse_team_from_text(xv)
                        jerseys = _extract_nums(xv)
                        _record_event("ControlledExit", t, jerseys, current_period, row_vsec, row_gsec)

                # Expected Goal from event label (fallback), often lists a player in the 'Shots' column
                label = df.iat[r, 0]
                if isinstance(label, str) and label.strip().lower() == "expected goal":
                    # Try to parse from shots/col4 or goals col if present
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

        # Save event summaries under the output directory (selected later)
        event_log_context = {
            "event_counts_by_player": event_counts_by_player,
            "event_counts_by_type_team": event_counts_by_type_team,
            "event_instances": event_instances,
            "event_player_rows": event_player_rows,
            "team_roster": team_roster,
            "team_excluded": team_excluded,
        }
    else:
        event_log_context = None

    # If not event-log format, fall back to per-player Start/End columns
    if not used_event_log:
        blocks = find_period_blocks(df)
        if not blocks:
            raise ValueError("No 'Period N' sections found in column A.")

        # Validation settings
        MAX_SHIFT_SECONDS = 30 * 60  # 30 minutes

        # Simple helper to report validation issues
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
                # Normalize scoreboard end times that incorrectly use the period start
                # time (e.g., 12:00/15:00/20:00) to represent end-of-period. Treat as 0:00.
                if sb_pairs:
                    sb_pairs = [(a, _normalize_sb_end_time(b)) for a, b in sb_pairs]

                # ---------------- Validation (per-row) ----------------
                if not skip_validation:
                    # Video times: must be strictly increasing and not excessively long
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

                    # Scoreboard times: allow either count-up or count-down, but must not be equal and not excessively long
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


    # Select subdirectory by detected format
    format_dir = "event_log" if ('used_event_log' in locals() and used_event_log) else "per_player"
    outdir = outdir / format_dir
    outdir.mkdir(parents=True, exist_ok=True)

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

        # Add event counts (event-log format only)
        if event_log_context is not None:
            per_player_events = event_log_context.get("event_counts_by_player", {})
            ev_counts = per_player_events.get(player_key, {})
            if ev_counts:
                stats_lines.append("")
                stats_lines.append("Event Counts:")
                # Order common events first
                order = [
                    "Shot",
                    "Goal",
                    "Assist",
                    "ControlledEntry",
                    "ControlledExit",
                    "ExpectedGoal",
                ]
                for kind in order:
                    if kind in ev_counts and ev_counts[kind] > 0:
                        stats_lines.append(f"  {kind}: {ev_counts[kind]}")
                # Any other events
                for kind, cnt in sorted(ev_counts.items()):
                    if kind in order:
                        continue
                    stats_lines.append(f"  {kind}: {cnt}")

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

    # Event summaries for event-log sheets
    if event_log_context is not None:
        evt_by_team = event_log_context.get("event_counts_by_type_team", {})
        rows_evt = [
            {"event_type": et, "team": tm, "count": cnt}
            for (et, tm), cnt in sorted(evt_by_team.items())
        ]
        if rows_evt:
            pd.DataFrame(rows_evt).to_csv(outdir / "event_summary.csv", index=False)

        # Per-instance event rows with timestamps
        player_event_rows = event_log_context.get("event_player_rows", []) or []
        if player_event_rows:
            # Add formatted time strings
            def _fmt_v(x):
                return seconds_to_hhmmss(int(x)) if isinstance(x, int) else (seconds_to_hhmmss(int(x)) if isinstance(x, float) else "")
            def _fmt_g(x):
                return seconds_to_mmss_or_hhmmss(int(x)) if isinstance(x, int) else (seconds_to_mmss_or_hhmmss(int(x)) if isinstance(x, float) else "")
            rows = []
            for r in player_event_rows:
                rows.append({
                    'event_type': r.get('event_type'),
                    'team': r.get('team'),
                    'player': r.get('player'),
                    'jersey': r.get('jersey'),
                    'period': r.get('period'),
                    'video_time': _fmt_v(r.get('video_s')),
                    'game_time': _fmt_g(r.get('game_s')),
                })
            pd.DataFrame(rows).to_csv(outdir / "event_players.csv", index=False)

        # Build clip windows per (event_type, team)
        instances = event_log_context.get("event_instances", {}) or {}
        # Helper: map scoreboard sec -> video sec using conv segments
        def map_sb_to_video(period: int, t_sb: int) -> Optional[int]:
            segs = conv_segments_by_period.get(period)
            if not segs:
                return None
            for s1, s2, v1, v2 in segs:
                lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
                if lo <= t_sb <= hi and s1 != s2:
                    return int(round(v1 + (t_sb - s1) * (v2 - v1) / (s2 - s1)))
            return None

        # Determine observed scoreboard max per period for capping
        max_sb_by_period: Dict[int, int] = {}
        for period, segs in conv_segments_by_period.items():
            mx = 0
            for s1, s2, _, _ in segs:
                mx = max(mx, s1, s2)
            if mx > 0:
                max_sb_by_period[period] = mx
        # Also incorporate event-side scoreboard times
        for key, lst in instances.items():
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
                if a <= lb + 10:  # merge if overlap or <=10s gap
                    out[-1][1] = max(lb, b)
                else:
                    out.append([a, b])
            return [(a, b) for a, b in out]

        # Generate files and scripts
        clip_scripts = []
        for (etype, team), lst in sorted(instances.items()):
            # Build video windows centered at event time
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
                    start = max(0, vsec - 15)
                    end = vsec + 15
                    v_windows.append((start, end))
                # Scoreboard window
                if isinstance(g, (int, float)) and isinstance(p, int):
                    gsec = int(g)
                    sb_max = max_sb_by_period.get(int(p), None)
                    sb_start = gsec + 15
                    if sb_max is not None:
                        sb_start = min(sb_max, sb_start)
                    sb_end = max(0, gsec - 15)
                    lo, hi = (sb_end, sb_start) if sb_end <= sb_start else (sb_start, sb_end)
                    sb_windows_by_period.setdefault(int(p), []).append((lo, hi))

            v_windows = merge_windows(v_windows)
            # Write video times file
            if v_windows:
                vfile = outdir / f"events_{etype}_{team}_video_times.txt"
                v_lines = [f"{seconds_to_hhmmss(a)} {seconds_to_hhmmss(b)}" for a, b in v_windows]
                vfile.write_text("\n".join(v_lines) + "\n", encoding="utf-8")
                # Create clip script
                script = outdir / f"clip_events_{etype}_{team}.sh"
                label = f"{etype} ({team})"
                body = f"""#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT="$1"
OPP="$2"
THIS_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TS_FILE="$THIS_DIR/{vfile.name}"
shift 2 || true
python -m hmlib.cli.video_clipper -j 4 --input "$INPUT" --timestamps "$TS_FILE" --temp-dir "$THIS_DIR/temp_clips/{etype}_{team}" "{label} vs $OPP" "$@"
"""
                script.write_text(body, encoding="utf-8")
                try:
                    import os as _os
                    _os.chmod(script, 0o755)
                except Exception:
                    pass
                clip_scripts.append(script.name)

            # Write scoreboard windows per period
            if sb_windows_by_period:
                sfile = outdir / f"events_{etype}_{team}_scoreboard_times.txt"
                s_lines = []
                for p, wins in sorted(sb_windows_by_period.items()):
                    wins = merge_windows(wins)
                    for lo, hi in wins:
                        # For display, use hi (larger) as start then lo as end
                        s_lines.append(f"{p} {seconds_to_mmss_or_hhmmss(hi)} {seconds_to_mmss_or_hhmmss(lo)}")
                if s_lines:
                    sfile.write_text("\n".join(s_lines) + "\n", encoding="utf-8")

        # Aggregate runner for all event clips
        if clip_scripts:
            all_script = outdir / "clip_events_all.sh"
            all_body = """#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video> <opposing_team> [--quick|-q] [--hq]"
  exit 1
fi
INPUT="$1"
OPP="$2"
shift 2 || true
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in {scripts}; do
  echo "Running $s..."
  "$THIS_DIR/$s" "$INPUT" "$OPP" "$@"
done
""".replace("{scripts}", " ".join(sorted(clip_scripts)))
            all_script.write_text(all_body, encoding="utf-8")
            try:
                import os as _os
                _os.chmod(all_script, 0o755)
            except Exception:
                pass

        # Roster cap warnings
        team_excluded = event_log_context.get("team_excluded", {}) or {}
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
        cols = (
            summary_cols + sb_cols + video_cols + period_toi_cols + period_shift_cols + period_gf_cols + period_ga_cols
        )

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

    # Final validation summary (if any) — only relevant for per-player format
    if ('used_event_log' in locals()) and (not used_event_log) and (not skip_validation) and ('validation_errors' in locals()) and validation_errors > 0:
        print(f"[validation] Completed with {validation_errors} issue(s). See messages above.", file=sys.stderr)

    return outdir


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
        help=("TimeToScore game id. If set and no --goal/--goals-file provided, fetch goals from T2S."),
    )
    side_group = p.add_mutually_exclusive_group()
    side_group.add_argument(
        "--home",
        action="store_true",
        help="Your team is the home team (with --t2s).",
    )
    side_group.add_argument(
        "--away",
        action="store_true",
        help="Your team is the away team (with --t2s).",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks on start/end ordering and excessive durations.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    goals = load_goals(args.goal, args.goals_file)

    # If no manual goals provided, but a T2S game id is given, use T2S.
    if not goals and args.t2s is not None:
        side: Optional[str]
        if args.home:
            side = "home"
        elif args.away:
            side = "away"
        else:
            print(
                "Error: --t2s was provided but neither --home nor --away was specified.",
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
    )
    try:
        print(f"✅ Done. Wrote per-player files to: {final_outdir.resolve()}")
    except Exception:
        print("✅ Done.")


if __name__ == "__main__":
    main()

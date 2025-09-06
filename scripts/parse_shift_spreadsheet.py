#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

LABEL_START_SB = "Shift Start (Scoreboard Time)"
LABEL_END_SB   = "Shift End (Scoreboard Time)"
LABEL_START_V  = "Shift Start (Video Time)"
LABEL_END_V    = "Shift End (Video Time)"


def sanitize_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    # keep alnum, underscore, dash; drop the rest
    return re.sub(r"[^A-Za-z0-9_\-]", "", s)


def is_period_label(x: object) -> bool:
    s = str(x).strip()
    return bool(re.fullmatch(r"Period\s+[1-3]", s))


def find_period_blocks(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Returns a list of (period_number, start_row_idx, end_row_idx_exclusive)
    spanning each 'Period X' section.
    """
    idxs = [i for i, v in df[0].items() if is_period_label(v)]
    idxs.append(len(df))
    blocks = []
    for i in range(len(idxs) - 1):
        period_num = int(str(df.iloc[idxs[i], 0]).strip().split()[-1])
        blocks.append((period_num, idxs[i], idxs[i + 1]))
    return blocks


def find_header_row(df: pd.DataFrame, start: int, end: int) -> Optional[int]:
    """
    Within [start, end), find the row that begins the header (contains 'Jersey No' in col 0).
    Your file has: 'Period X' row, then a blank row, then this header row.
    """
    for r in range(start, min(end, start + 8)):  # small window is enough
        if str(df.iloc[r, 0]).strip().lower() == "jersey no":
            return r
    return None


def forward_fill_header_labels(header_row: pd.Series) -> Dict[str, List[int]]:
    """
    The header row has merged-like labels: the key word appears once, then NaNs
    across the span. We forward-fill across columns to learn which columns belong
    to which label.
    Returns a mapping: label -> list of column indices.
    """
    labels_by_col: List[str] = []
    current_label = None
    for c in range(len(header_row)):
        val = header_row.iloc[c]
        if pd.notna(val) and str(val).strip():
            current_label = str(val).strip()
        labels_by_col.append(current_label)

    groups: Dict[str, List[int]] = {}
    for col_idx, lab in enumerate(labels_by_col):
        if not lab:
            continue
        groups.setdefault(lab, []).append(col_idx)
    return groups


def extract_pairs_from_row(row: pd.Series, start_cols: List[int], end_cols: List[int]) -> List[Tuple[str, str]]:
    """
    Given a row and the columns for starts and ends (already separated),
    pair them positionally. Ignore blanks/NaNs. Truncate to the min length.
    """
    starts = [str(row[c]).strip() for c in start_cols if pd.notna(row[c]) and str(row[c]).strip()]
    ends   = [str(row[c]).strip() for c in end_cols   if pd.notna(row[c]) and str(row[c]).strip()]
    n = min(len(starts), len(ends))
    return [(starts[i], ends[i]) for i in range(n)]


def parse_file(
    xls_path: Path,
    output_dir: Path,
    skip_goalies: bool = True,
) -> None:
    df = pd.read_excel(xls_path, header=None)

    # Identify period blocks
    blocks = find_period_blocks(df)
    if not blocks:
        raise ValueError("No 'Period N' sections found in column A of the sheet.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate per-player across all periods
    per_player_video: Dict[str, List[Tuple[str, str]]] = {}
    per_player_score: Dict[str, List[Tuple[int, str, str]]] = {}

    for period_num, blk_start, blk_end in blocks:
        header_row_idx = find_header_row(df, blk_start, blk_end)
        if header_row_idx is None:
            # Fallback to the common ‘Period; blank; header’ pattern
            header_row_idx = blk_start + 2
        data_start = header_row_idx + 1

        header = df.iloc[header_row_idx]
        groups = forward_fill_header_labels(header)

        # Which columns belong to which label?
        start_sb_cols = groups.get(LABEL_START_SB, [])
        end_sb_cols   = groups.get(LABEL_END_SB, [])
        start_v_cols  = groups.get(LABEL_START_V, [])
        end_v_cols    = groups.get(LABEL_END_V, [])

        # Required labels must exist
        for lab, cols in [
            (LABEL_START_SB, start_sb_cols),
            (LABEL_END_SB, end_sb_cols),
            (LABEL_START_V, start_v_cols),
            (LABEL_END_V, end_v_cols),
        ]:
            if not cols:
                raise ValueError(f"Could not locate header columns for: '{lab}' in Period {period_num}")

        # Iterate each player row
        for r in range(data_start, blk_end):
            jersey = str(df.iloc[r, 0]).strip()
            name   = str(df.iloc[r, 1]).strip()

            # Stop if we hit the next period or empty tail
            if is_period_label(df.iloc[r, 0]) or (not jersey and not name):
                break

            # Skip blank lines
            if not jersey or jersey.lower() == "nan":
                continue

            # Skip goalies `(G) 37` etc.
            if skip_goalies and "(" in jersey and ")" in jersey:
                continue

            player_key = f"{sanitize_name(jersey)}_{sanitize_name(name)}"

            # Extract and pair times
            video_pairs = extract_pairs_from_row(df.iloc[r], start_v_cols, end_v_cols)
            sb_pairs    = extract_pairs_from_row(df.iloc[r], start_sb_cols, end_sb_cols)

            if video_pairs:
                per_player_video.setdefault(player_key, []).extend(video_pairs)
            if sb_pairs:
                per_player_score.setdefault(player_key, []).extend((period_num, s, e) for s, e in sb_pairs)

    # Write files
    for player_key, v_pairs in per_player_video.items():
        with (output_dir / f"{player_key}_video_times.txt").open("w", encoding="utf-8") as f:
            for s, e in v_pairs:
                f.write(f"{s} {e}\n")

    for player_key, sb_list in per_player_score.items():
        with (output_dir / f"{player_key}_scoreboard_times.txt").open("w", encoding="utf-8") as f:
            for period_num, s, e in sb_list:
                f.write(f"{period_num} {s} {e}\n")


if __name__ == "__main__":
    # --- Configure here or pass in via flags if you want to CLI-ify ---
    # xls_path = Path("dh-tv-12-1.xls")   # your .xls file
    xls_path = Path("/mnt/ripper-data/Videos/dh-tv-12-1/stats/dh-tv-12-1.xls")  # Replace with your .xls filename
    out_dir  = Path("player_shifts")    # output folder
    parse_file(xls_path, out_dir, skip_goalies=True)
    print(f"✅ Done. Wrote per-player files in: {out_dir.resolve()}")

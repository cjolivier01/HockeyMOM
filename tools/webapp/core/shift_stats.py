import csv
import io
import re
from typing import Any, Optional


def normalize_jersey_number(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    try:
        return str(int(m.group(1)))
    except Exception:
        return m.group(1)


def normalize_player_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().lower())


def strip_jersey_from_player_name(raw: str, jersey_number: Optional[str]) -> str:
    """
    Strip embedded jersey numbers from a player name when the jersey matches.
    """
    name = str(raw or "").strip()
    if not name:
        return ""
    jersey_norm = normalize_jersey_number(jersey_number) if jersey_number else None
    if jersey_norm:
        tail = re.sub(rf"\s*[\(#]?\s*{re.escape(jersey_norm)}\s*\)?\s*$", "", name).strip()
        if tail:
            name = tail
        head = re.sub(rf"^#?\s*{re.escape(jersey_norm)}\s+", "", name).strip()
        if head:
            name = head
    return name


def normalize_player_name_no_middle(raw: str) -> str:
    """
    Normalize a name while dropping single-letter middle initials (e.g. "Brock T Birkey").
    """
    parts = [p for p in str(raw or "").strip().split() if p]
    if len(parts) >= 3:
        kept: list[str] = []
        for idx, token in enumerate(parts):
            t = token.strip(".")
            if 0 < idx < (len(parts) - 1) and len(t) == 1:
                continue
            kept.append(token)
        parts = kept or parts
    return normalize_player_name(" ".join(parts))


def parse_duration_seconds(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(float(sec))
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(float(sec))
        return int(float(s))
    except Exception:
        return None


def format_seconds_to_mmss_or_hhmmss(raw: Any) -> str:
    try:
        t = int(raw)  # type: ignore[arg-type]
    except Exception:
        return ""
    if t < 0:
        t = 0
    h = t // 3600
    r = t % 3600
    m = r // 60
    s = r % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_shift_stats_player_stats_csv(csv_text: str) -> list[dict[str, Any]]:
    """
    Parse stats/player_stats.csv written by scripts/parse_stats_inputs.py.

    Returns rows with:
      - player_label: original display label (e.g. "59 Ryan S Donahue")
      - jersey_number: normalized jersey number ("59") when present
      - name_norm: normalized player name for matching
      - stats: dict of DB column -> value
      - period_stats: dict[period:int] -> dict of period metrics
    """
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("missing CSV headers")

    out: list[dict[str, Any]] = []
    for raw_row in reader:
        row = {k.strip(): v for k, v in (raw_row or {}).items() if k}
        player_name = (row.get("Player") or "").strip()

        # Newer outputs write jersey and name as separate columns.
        jersey_raw = (
            row.get("Jersey #")
            or row.get("Jersey")
            or row.get("Jersey No")
            or row.get("Jersey Number")
            or ""
        )
        jersey_norm = normalize_jersey_number(jersey_raw) if str(jersey_raw).strip() else None

        # Back-compat: older outputs encoded jersey in the Player field (e.g. " 8 Adam Ro").
        name_part = player_name
        m = re.match(r"^\s*(\d+)\s+(.*)$", player_name)
        if m:
            jersey_from_name = normalize_jersey_number(m.group(1))
            if jersey_norm is None:
                jersey_norm = jersey_from_name
            if jersey_from_name and jersey_norm and jersey_from_name == jersey_norm:
                name_part = m.group(2).strip()

        player_label = f"{jersey_norm} {name_part}".strip() if jersey_norm else name_part
        name_norm = normalize_player_name(name_part)
        name_norm_no_middle = normalize_player_name_no_middle(name_part)

        stats: dict[str, Any] = {
            "goals": _int_or_none(row.get("Goals")),
            "assists": _int_or_none(row.get("Assists")),
            "shots": _int_or_none(row.get("Shots")),
            "pim": _int_or_none(row.get("PIM")),
            "hits": _int_or_none(row.get("Hits")),
            "blocks": _int_or_none(row.get("Blocks")),
            "toi_seconds": parse_duration_seconds(row.get("TOI Total") or row.get("TOI")),
            "faceoff_wins": _int_or_none(row.get("Faceoff Wins") or row.get("Faceoffs Won")),
            "faceoff_attempts": _int_or_none(row.get("Faceoff Attempts") or row.get("Faceoffs")),
            "goalie_saves": _int_or_none(row.get("Saves") or row.get("Goalie Saves")),
            "goalie_ga": _int_or_none(row.get("GA") or row.get("Goalie GA")),
            "goalie_sa": _int_or_none(row.get("SA") or row.get("Goalie SA")),
            "sog": _int_or_none(row.get("SOG")),
            "expected_goals": _int_or_none(row.get("xG")),
            "completed_passes": _int_or_none(
                row.get("Completed Passes") or row.get("Completed Pass")
            ),
            "giveaways": _int_or_none(row.get("Giveaways")),
            "turnovers_forced": _int_or_none(row.get("Turnovers (forced)")),
            "created_turnovers": _int_or_none(row.get("Created Turnovers")),
            "takeaways": _int_or_none(row.get("Takeaways")),
            "controlled_entry_for": _int_or_none(row.get("Controlled Entry For (On-Ice)")),
            "controlled_entry_against": _int_or_none(row.get("Controlled Entry Against (On-Ice)")),
            "controlled_exit_for": _int_or_none(row.get("Controlled Exit For (On-Ice)")),
            "controlled_exit_against": _int_or_none(row.get("Controlled Exit Against (On-Ice)")),
            "gt_goals": _int_or_none(row.get("GT Goals")),
            "gw_goals": _int_or_none(row.get("GW Goals")),
            "ot_goals": _int_or_none(row.get("OT Goals")),
            "ot_assists": _int_or_none(row.get("OT Assists")),
            "plus_minus": _int_or_none(row.get("Plus Minus") or row.get("Goal +/-")),
            "gf_counted": _int_or_none(row.get("GF Counted")),
            "ga_counted": _int_or_none(row.get("GA Counted")),
            "shifts": _int_or_none(row.get("Shifts")),
            "video_toi_seconds": parse_duration_seconds(
                row.get("TOI Total (Video)") or row.get("TOI (Video)")
            ),
            "sb_avg_shift_seconds": parse_duration_seconds(row.get("Average Shift")),
            "sb_median_shift_seconds": parse_duration_seconds(row.get("Median Shift")),
            "sb_longest_shift_seconds": parse_duration_seconds(row.get("Longest Shift")),
            "sb_shortest_shift_seconds": parse_duration_seconds(row.get("Shortest Shift")),
        }

        # Period stats: Period {n} GF/GA
        period_stats: dict[int, dict[str, Any]] = {}
        for k, v in row.items():
            if not k:
                continue
            m = re.match(r"^Period\s+(\d+)\s+GF$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["gf"] = _int_or_none(v)
                continue
            m = re.match(r"^Period\s+(\d+)\s+GA$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["ga"] = _int_or_none(v)
                continue
            m = re.match(r"^Period\s+(\d+)\s+TOI$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["toi_seconds"] = parse_duration_seconds(v)
                continue
            m = re.match(r"^Period\s+(\d+)\s+Shifts$", k)
            if m:
                per = int(m.group(1))
                period_stats.setdefault(per, {})["shifts"] = _int_or_none(v)
                continue

        out.append(
            {
                "player_label": player_label,
                "jersey_number": jersey_norm,
                "name_norm": name_norm,
                "name_norm_no_middle": name_norm_no_middle,
                "stats": stats,
                "period_stats": period_stats,
            }
        )
    return out


def parse_shift_stats_game_stats_csv(csv_text: str) -> dict[str, Any]:
    """
    Parse stats/game_stats.csv written by scripts/parse_stats_inputs.py.
    Format is a 2-column table: "Stat", "<game_label>".
    """
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    if not reader.fieldnames or "Stat" not in reader.fieldnames:
        raise ValueError("missing Stat column")
    value_col = next((c for c in reader.fieldnames if c != "Stat"), None)
    if not value_col:
        raise ValueError("missing value column")
    out: dict[str, Any] = {"_label": value_col}
    for row in reader:
        if not row:
            continue
        key = (row.get("Stat") or "").strip()
        if not key:
            continue
        out[key] = (row.get(value_col) or "").strip()
    return out


def parse_shift_rows_csv(csv_text: str) -> list[dict[str, Any]]:
    """
    Parse stats/shift_rows.csv written by scripts/parse_stats_inputs.py.

    Returns rows with:
      - player_label: display label (e.g. "59 Ryan S Donahue")
      - jersey_number: normalized jersey number ("59") when present
      - name_norm/name_norm_no_middle: normalized names for matching
      - period: int
      - game_seconds/game_seconds_end: within-period seconds (typically "remaining" clock)
      - video_seconds/video_seconds_end: optional video seconds
      - import_key: optional idempotency key (caller may derive if missing)
    """
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("missing CSV headers")

    def _get(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for k in keys:
            if k in row:
                return row.get(k)
        return None

    def _int_or_none_flex(v: Any) -> Optional[int]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return None
        return parse_duration_seconds(s)

    out: list[dict[str, Any]] = []
    for raw_row in reader:
        row = {k.strip(): v for k, v in (raw_row or {}).items() if k}

        player_name = str(_get(row, ("Player", "Name")) or "").strip()
        jersey_raw = _get(
            row,
            ("Jersey #", "Jersey", "Jersey No", "JerseyNo", "Jersey Number", "JerseyNumber"),
        )
        jersey_norm = normalize_jersey_number(jersey_raw) if str(jersey_raw or "").strip() else None

        # Back-compat: allow jersey in the Player cell (e.g. " 8 Adam Ro").
        name_part = player_name
        m = re.match(r"^\s*(\d+)\s+(.*)$", player_name)
        if m:
            jersey_from_name = normalize_jersey_number(m.group(1))
            if jersey_norm is None:
                jersey_norm = jersey_from_name
            if jersey_from_name and jersey_norm and jersey_from_name == jersey_norm:
                name_part = m.group(2).strip()

        player_label = f"{jersey_norm} {name_part}".strip() if jersey_norm else name_part
        name_norm = normalize_player_name(name_part)
        name_norm_no_middle = normalize_player_name_no_middle(name_part)

        period = _int_or_none(_get(row, ("Period",)))
        game_seconds = _int_or_none_flex(
            _get(row, ("Game Seconds", "GameSeconds", "Start Game Seconds", "StartGameSeconds"))
        )
        game_seconds_end = _int_or_none_flex(
            _get(
                row,
                (
                    "Game Seconds End",
                    "GameSecondsEnd",
                    "End Game Seconds",
                    "EndGameSeconds",
                    "Game Seconds (End)",
                ),
            )
        )
        video_seconds = _int_or_none_flex(
            _get(row, ("Video Seconds", "VideoSeconds", "Start Video Seconds", "StartVideoSeconds"))
        )
        video_seconds_end = _int_or_none_flex(
            _get(
                row,
                (
                    "Video Seconds End",
                    "VideoSecondsEnd",
                    "End Video Seconds",
                    "EndVideoSeconds",
                    "Video Seconds (End)",
                ),
            )
        )
        import_key = str(_get(row, ("Import Key", "ImportKey")) or "").strip() or None
        source = str(_get(row, ("Source",)) or "").strip() or None

        out.append(
            {
                "player_label": player_label,
                "jersey_number": jersey_norm,
                "name_norm": name_norm,
                "name_norm_no_middle": name_norm_no_middle,
                "period": period,
                "game_seconds": game_seconds,
                "game_seconds_end": game_seconds_end,
                "video_seconds": video_seconds,
                "video_seconds_end": video_seconds_end,
                "import_key": import_key,
                "source": source,
            }
        )
    return out


def _int_or_none(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

import csv
import io
import re
from typing import Any, Optional

from .shift_stats import (
    format_seconds_to_mmss_or_hhmmss,
    normalize_jersey_number,
    normalize_player_name,
    parse_duration_seconds,
)
from .utils import to_dt


def _int0(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def parse_events_csv(events_csv: str) -> tuple[list[str], list[dict[str, str]]]:
    s = str(events_csv or "").strip()
    if not s:
        return [], []
    s = s.lstrip("\ufeff")
    f = io.StringIO(s)
    reader = csv.DictReader(f)
    headers = [str(h) for h in (reader.fieldnames or []) if h is not None]
    rows: list[dict[str, str]] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        rows.append({h: ("" if row.get(h) is None else str(row.get(h))) for h in headers})
    return headers, rows


def to_csv_text(headers: list[str], rows: list[dict[str, str]]) -> str:
    if not headers:
        return ""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows or []:
        w.writerow({h: ("" if (r.get(h) is None) else str(r.get(h))) for h in headers})
    return out.getvalue()


def sanitize_player_stats_csv_for_storage(player_stats_csv: str) -> str:
    """
    Sanitize game-level player_stats CSV before storage.

    We intentionally drop shift/ice-time and "per game"/"per shift" derived columns so they never
    appear in the web UI and don't accidentally get used for downstream calculations.
    """
    headers, rows = parse_events_csv(player_stats_csv)
    headers, rows = filter_single_game_player_stats_csv(headers, rows)
    return to_csv_text(headers, rows)


def filter_single_game_player_stats_csv(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Single-game player stats tables should not include any per-game normalized columns
    (e.g., 'Shots per Game', 'TOI per Game', 'PPG') because they are redundant for a
    one-game view and make the table unnecessarily wide.
    """

    def _drop_header(h: str) -> bool:
        key = str(h or "").strip().lower()
        if key in {"ppg", "gp"}:
            return True
        if "per game" in key:
            return True
        # Remove all shift/time related fields from the webapp UI.
        if "toi" in key or "ice time" in key:
            return True
        if "shift" in key or "per shift" in key:
            return True
        return False

    def _meta_items(r: dict[str, str]) -> dict[str, str]:
        # Preserve internal row metadata (used by some views for per-row actions).
        return {k: v for k, v in (r or {}).items() if str(k).startswith("__hm_")}

    kept_headers = [h for h in headers if not _drop_header(h)]
    kept_rows: list[dict[str, str]] = []
    for r in rows or []:
        out = {h: r.get(h, "") for h in kept_headers}
        out.update(_meta_items(r))
        kept_rows.append(out)
    return kept_headers, kept_rows


def normalize_game_events_csv(
    headers: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Normalize the event table for display:
      - Ensure 'Event Type' is the leftmost column
      - Drop redundant 'Event Type Raw' (historical) if present
    """

    def _is_event_type_raw(h: str) -> bool:
        return str(h or "").strip().lower() in {"event type raw", "event_type_raw"}

    def _meta_items(r: dict[str, str]) -> dict[str, str]:
        # Preserve internal row metadata (used by some views for per-row actions).
        return {k: v for k, v in (r or {}).items() if str(k).startswith("__hm_")}

    # Remove raw header if present.
    filtered_headers = [h for h in (headers or []) if not _is_event_type_raw(h)]
    filtered_rows: list[dict[str, str]] = []
    for r in rows or []:
        out = {h: r.get(h, "") for h in filtered_headers}
        out.update(_meta_items(r))
        filtered_rows.append(out)

    # Prefer explicit "Event Type"; fall back to "Event" (common legacy schema).
    event_header = None
    for h in filtered_headers:
        if str(h).strip().lower() == "event type":
            event_header = h
            break
    if event_header is None:
        for h in filtered_headers:
            if str(h).strip().lower() == "event":
                event_header = h
                break

    if event_header is None:
        return filtered_headers, filtered_rows

    # If the CSV uses "Event", rename it to "Event Type" for display.
    if str(event_header).strip().lower() == "event":
        renamed_headers: list[str] = []
        for h in filtered_headers:
            if h == event_header:
                renamed_headers.append("Event Type")
            else:
                renamed_headers.append(h)
        renamed_rows: list[dict[str, str]] = []
        for r in filtered_rows:
            out: dict[str, str] = {}
            for h in filtered_headers:
                if h == event_header:
                    out["Event Type"] = r.get(h, "")
                else:
                    out[h] = r.get(h, "")
            out.update(_meta_items(r))
            renamed_rows.append(out)
        filtered_headers, filtered_rows = renamed_headers, renamed_rows
        event_header = "Event Type"

    reordered_headers = [event_header] + [h for h in filtered_headers if h != event_header]
    reordered_rows: list[dict[str, str]] = []
    for r in filtered_rows:
        out = {h: r.get(h, "") for h in reordered_headers}
        out.update(_meta_items(r))
        reordered_rows.append(out)
    return reordered_headers, reordered_rows


def filter_events_headers_drop_empty_on_ice_split(
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    If Home/Away on-ice columns are present but completely empty, drop them from the display table.
    Keeps the raw CSV stored in DB unchanged; this only affects UI rendering.
    """
    if not headers or not rows:
        return headers, rows
    split_cols = ["On-Ice Players (Home)", "On-Ice Players (Away)"]
    present = [c for c in split_cols if c in headers]
    if not present:
        return headers, rows
    keep: set[str] = set(headers)
    for c in present:
        any_nonempty = False
        for r in rows:
            v = str((r or {}).get(c, "") or "").strip()
            if v:
                any_nonempty = True
                break
        if not any_nonempty:
            keep.discard(c)
    if keep == set(headers):
        return headers, rows
    new_headers = [h for h in headers if h in keep]
    new_rows: list[dict[str, str]] = []
    for r in rows:
        base = {h: (r.get(h, "") if isinstance(r, dict) else "") for h in new_headers}
        if isinstance(r, dict):
            base.update({k: v for k, v in r.items() if str(k).startswith("__hm_")})
        new_rows.append(base)
    return new_headers, new_rows


def compute_team_scoring_by_period_from_events(
    events_rows: list[dict[str, str]],
    *,
    tts_linked: bool = False,
) -> list[dict[str, Any]]:
    """
    Compute per-period team GF/GA from stored goal events.

    Notes:
      - Uses event rows (e.g. from hky_game_events.events_csv).
      - Team mapping prefers absolute Home/Away via "Team Side" (or legacy "Team Rel" when it contains Home/Away).
      - "For/Against" is treated as a relative direction, not a synonym for Home/Away.
    Returns rows like:
      {'period': 1, 'team1_gf': 2, 'team1_ga': 1, 'team2_gf': 1, 'team2_ga': 2}
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _is_tts_row(row: dict[str, str]) -> bool:
        toks = _split_sources(row.get("Source") or "")
        return any(_norm(t) == "timetoscore" for t in toks)

    def _team_side_to_team_idx(row: dict[str, str]) -> Optional[int]:
        # Prefer explicit Team Side when available.
        side = _norm(row.get("Team Side") or row.get("TeamSide") or row.get("Side") or "")
        if side in {"home", "team1"}:
            return 1
        if side in {"away", "team2"}:
            return 2

        # Legacy: some tables stored Home/Away in Team Rel or Team Raw.
        tr = _norm(row.get("Team Rel") or row.get("TeamRel") or "")
        if tr in {"home", "team1"}:
            return 1
        if tr in {"away", "team2"}:
            return 2

        # Some simple/older tables use "Team" as Home/Away directly.
        team = _norm(row.get("Team") or "")
        if team in {"home", "team1"}:
            return 1
        if team in {"away", "team2"}:
            return 2

        tr2 = _norm(row.get("Team Raw") or row.get("TeamRaw") or "")
        if tr2 in {"home", "team1"}:
            return 1
        if tr2 in {"away", "team2"}:
            return 2
        return None

    by_period: dict[int, dict[str, int]] = {}
    # If the game is TimeToScore-linked, only count TimeToScore goal rows for scoring attribution.
    # Otherwise, prefer TimeToScore goal rows when they are present (but allow spreadsheet-only games).
    has_tts_goal_rows = False
    if tts_linked:
        has_tts_goal_rows = True
    else:
        for r0 in events_rows or []:
            if not isinstance(r0, dict):
                continue
            if _norm(r0.get("Event Type") or r0.get("Event") or "") != "goal":
                continue
            if _is_tts_row(r0):
                has_tts_goal_rows = True
                break

    for r in events_rows or []:
        if not isinstance(r, dict):
            continue
        et = _norm(r.get("Event Type") or r.get("Event") or "")
        if et != "goal":
            continue
        if has_tts_goal_rows and not _is_tts_row(r):
            continue
        per = _parse_int(r.get("Period"))
        if per is None:
            continue
        team_idx = _team_side_to_team_idx(r)
        if team_idx is None:
            continue
        rec = by_period.setdefault(
            per, {"team1_gf": 0, "team1_ga": 0, "team2_gf": 0, "team2_ga": 0}
        )
        if team_idx == 1:
            rec["team1_gf"] += 1
            rec["team2_ga"] += 1
        else:
            rec["team2_gf"] += 1
            rec["team1_ga"] += 1

    if not by_period:
        return []
    max_period = max(3, max(by_period.keys()))
    out: list[dict[str, Any]] = []
    for p in range(1, max_period + 1):
        rec = by_period.get(p) or {"team1_gf": 0, "team1_ga": 0, "team2_gf": 0, "team2_ga": 0}
        out.append({"period": p, **rec})
    return out


def filter_events_rows_prefer_timetoscore_for_goal_assist(
    events_rows: list[dict[str, str]],
    *,
    tts_linked: bool = False,
) -> list[dict[str, str]]:
    """
    If any TimeToScore Goal/Assist rows exist, drop non-TimeToScore Goal/Assist rows to
    avoid mixing attribution sources.
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _ev_type(r: dict[str, str]) -> str:
        return _norm(r.get("Event Type") or r.get("Event") or "")

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _is_tts_row(r: dict[str, str]) -> bool:
        def _is_t2s_token(tok: str) -> bool:
            t = _norm(tok)
            if not t:
                return False
            # Common values seen in DB and CSV sources.
            if t in {"timetoscore", "t2s", "tts"}:
                return True
            # Some sources are stamped like "t2s:54183" or "timetoscore:54183".
            return t.startswith("t2s") or t.startswith("timetoscore")

        toks = _split_sources(r.get("Source") or "")
        return any(_is_t2s_token(t) for t in toks)

    if not events_rows:
        return []
    has_tts = any(
        _ev_type(r) in {"goal", "assist"} and _is_tts_row(r)
        for r in events_rows
        if isinstance(r, dict)
    )
    # If we don't actually have TimeToScore Goal/Assist rows, don't drop anything: keep whatever we have.
    if not has_tts:
        return list(events_rows)
    return [
        r
        for r in events_rows
        if isinstance(r, dict) and (_ev_type(r) not in {"goal", "assist"} or _is_tts_row(r))
    ]


def filter_events_csv_drop_event_types(csv_text: str, *, drop_types: set[str]) -> str:
    """
    Drop specific event types from an events CSV (case-insensitive match on Event Type/Event).
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    headers, rows = parse_events_csv(csv_text)
    if not headers:
        return csv_text
    kept_rows: list[dict[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        et = _norm(r.get("Event Type") or r.get("Event") or "")
        if et and et in drop_types:
            continue
        kept_rows.append(r)
    return to_csv_text(headers, kept_rows)


def summarize_event_sources(
    events_rows: list[dict[str, str]],
    *,
    fallback_source_label: Optional[str] = None,
) -> list[str]:
    """
    Return a de-duped, order-preserving list of event row sources for UI display.
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip()

    def _split_sources(raw: Any) -> list[str]:
        s = _norm(raw)
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _canon(s: str) -> str:
        sl = s.strip().casefold()
        if (
            sl in {"timetoscore", "t2s", "tts"}
            or sl.startswith("t2s")
            or sl.startswith("timetoscore")
        ):
            return "TimeToScore"
        if sl == "long":
            return "Long"
        if sl == "goals":
            return "Goals"
        if sl == "shift_package":
            return "Shift Package"
        if not s.strip():
            return ""
        return s.strip()

    out: list[str] = []
    seen: set[str] = set()
    for r in events_rows or []:
        if not isinstance(r, dict):
            continue
        src_raw = r.get("Source") or ""
        toks = _split_sources(src_raw)
        if not toks and fallback_source_label:
            toks = _split_sources(str(fallback_source_label))
        for t in toks:
            src = _canon(t)
            if not src:
                continue
            key = src.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(src)
    return out


def merge_events_csv_prefer_timetoscore(
    *,
    existing_csv: str,
    existing_source_label: str,
    incoming_csv: str,
    incoming_source_label: str,
    protected_types: set[str],
) -> tuple[str, str]:
    """
    Merge two events CSVs, preferring TimeToScore rows for protected event types (Goal/Assist/etc).
    Returns: (merged_csv, merged_source_label).
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _ev_type(r: dict[str, str]) -> str:
        return _norm(r.get("Event Type") or r.get("Event") or "")

    def _is_tts_row(r: dict[str, str], fallback_label: str) -> bool:
        src = _norm(r.get("Source") or "")
        if src:
            toks = [t for t in re.split(r"[,+;/\s]+", src) if t]
            return any(t == "timetoscore" for t in toks)
        return str(fallback_label or "").strip().lower().startswith("timetoscore")

    def _key(r: dict[str, str]) -> tuple[str, str, str, str, str]:
        et = _ev_type(r)
        per = str(r.get("Period") or "").strip()
        gs = str(r.get("Game Seconds") or r.get("GameSeconds") or "").strip()
        # Prefer absolute Home/Away when available.
        tr = (
            str(
                r.get("Team Side")
                or r.get("TeamSide")
                or r.get("Team Rel")
                or r.get("TeamRel")
                or r.get("Team")
                or ""
            )
            .strip()
            .casefold()
        )
        jerseys = str(r.get("Attributed Jerseys") or "").strip()
        return (et, per, gs, tr, jerseys)

    ex_headers, ex_rows = parse_events_csv(existing_csv)
    in_headers, in_rows = parse_events_csv(incoming_csv)

    def _iter_typed(rows: list[dict[str, str]], label: str) -> list[tuple[dict[str, str], str]]:
        out: list[tuple[dict[str, str], str]] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append((r, label))
        return out

    all_rows = _iter_typed(ex_rows, existing_source_label) + _iter_typed(
        in_rows, incoming_source_label
    )
    has_tts_protected = any(
        _ev_type(r) in protected_types and _is_tts_row(r, lbl) for r, lbl in all_rows
    )

    def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = str(row.get(k) or "").strip()
            if v:
                return v
        return ""

    def _overlay_missing_fields(dst: dict[str, str], src: dict[str, str]) -> None:
        for k, ks in (
            ("Event ID", ("Event ID", "EventID")),
            ("Video Time", ("Video Time", "VideoTime")),
            ("Video Seconds", ("Video Seconds", "VideoSeconds", "Video S", "VideoS")),
            ("On-Ice Players", ("On-Ice Players", "OnIce Players", "OnIcePlayers")),
            ("On-Ice Players (Home)", ("On-Ice Players (Home)", "OnIce Players (Home)")),
            ("On-Ice Players (Away)", ("On-Ice Players (Away)", "OnIce Players (Away)")),
        ):
            if str(dst.get(k) or "").strip():
                continue
            v = _first_non_empty(src, ks)
            if v:
                dst[k] = v

    fallback_rows_by_key: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}

    merged_rows: list[dict[str, str]] = []
    merged_by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for r, lbl in all_rows:
        et = _ev_type(r)
        if not et:
            continue
        k = _key(r)

        # If TimeToScore rows exist for protected types, keep them as authoritative rows, but
        # still remember non-TTS rows so we can copy missing clip/on-ice metadata onto the kept
        # rows (Video Time/Seconds, On-Ice Players, etc).
        if has_tts_protected and et in protected_types and not _is_tts_row(r, lbl):
            fallback_rows_by_key.setdefault(k, []).append(r)
            continue

        prev = merged_by_key.get(k)
        if prev is not None:
            _overlay_missing_fields(prev, r)
            continue

        rr = dict(r)
        merged_rows.append(rr)
        merged_by_key[k] = rr

    # Overlay missing fields for protected TimeToScore rows from any skipped non-TTS rows.
    for k, fb_rows in fallback_rows_by_key.items():
        dst = merged_by_key.get(k)
        if dst is None:
            continue
        for fb in fb_rows:
            _overlay_missing_fields(dst, fb)

    merged_headers = list(ex_headers or [])
    for h in in_headers or []:
        if h not in merged_headers:
            merged_headers.append(h)

    # Prefer a TimeToScore label if either input is TimeToScore.
    merged_source = (
        existing_source_label
        if str(existing_source_label or "").strip().lower().startswith("timetoscore")
        else (
            incoming_source_label
            if str(incoming_source_label or "").strip().lower().startswith("timetoscore")
            else (existing_source_label or incoming_source_label)
        )
    )

    return to_csv_text(merged_headers, merged_rows), str(merged_source or "")


def enrich_timetoscore_goals_with_long_video_times(
    *,
    existing_headers: list[str],
    existing_rows: list[dict[str, str]],
    incoming_headers: list[str],
    incoming_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    For TimeToScore-linked games, treat TimeToScore goal attribution as authoritative, but
    copy Video Time/Seconds and On-Ice player lists from matching spreadsheet-derived Goal events
    (same team + period + game time).

    - Only enriches existing rows whose Source contains "timetoscore"
    - Never adds new goal events; long-only goals are ignored
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _has_source(row: dict[str, str], token: str) -> bool:
        return any(_norm(t) == _norm(token) for t in _split_sources(row.get("Source") or ""))

    def _add_source(row: dict[str, str], token: str) -> None:
        toks = _split_sources(row.get("Source") or "")
        toks_cf = {_norm(t) for t in toks}
        if _norm(token) not in toks_cf:
            toks.append(str(token))
        row["Source"] = ",".join([t for t in toks if t])

    def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = str(row.get(k) or "").strip()
            if v:
                return v
        return ""

    def _set_if_blank(row: dict[str, str], key: str, value: str) -> bool:
        if str(row.get(key) or "").strip():
            return False
        if not str(value or "").strip():
            return False
        row[key] = str(value)
        return True

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _norm_side(row: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Side",
            "Team Rel",
            "TeamRel",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _ev_type(row: dict[str, str]) -> str:
        return _norm(row.get("Event Type") or row.get("Event") or "")

    def _period(row: dict[str, str]) -> Optional[int]:
        p = _parse_int(row.get("Period"))
        return p if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, str]) -> Optional[int]:
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is not None:
            return gs
        return parse_duration_seconds(
            row.get("Game Time") or row.get("GameTime") or row.get("Time")
        )

    def _video_seconds(row: dict[str, str]) -> Optional[int]:
        vs = _parse_int(row.get("Video Seconds") or row.get("VideoSeconds"))
        if vs is not None:
            return vs
        return parse_duration_seconds(row.get("Video Time") or row.get("VideoTime"))

    def _video_time(row: dict[str, str]) -> str:
        return str(row.get("Video Time") or row.get("VideoTime") or "").strip()

    # Lookups from incoming Goal rows.
    long_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
    on_ice_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
    on_ice_score_by_key: dict[tuple[int, str, int], int] = {}
    event_id_by_key: dict[tuple[int, str, int], tuple[int, str]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        if _ev_type(r) != "goal":
            continue
        per = _period(r)
        side = _norm_side(r)
        gs = _game_seconds(r)
        if per is None or side is None or gs is None:
            continue

        # Prefer a "goals" goal-row event id as the canonical id to copy onto TimeToScore goals
        # (long-sheet goal rows also have ids but are less stable).
        eid = _first_non_empty(r, ("Event ID", "EventID"))
        if eid:
            prio = 0 if _has_source(r, "goals") else (1 if _has_source(r, "long") else 2)
            prev = event_id_by_key.get((per, side, int(gs)))
            if prev is None or int(prio) < int(prev[0]):
                event_id_by_key[(per, side, int(gs))] = (int(prio), str(eid))

        # Video time enrichment prefers long-sheet rows.
        if _has_source(r, "long"):
            vs = _video_seconds(r)
            vt = _video_time(r)
            if vs is not None or vt:
                long_by_key[(per, side, int(gs))] = r

        # On-ice enrichment can come from any incoming Goal rows that carry those columns.
        home_on_ice = _first_non_empty(r, ("On-Ice Players (Home)", "OnIce Players (Home)"))
        away_on_ice = _first_non_empty(r, ("On-Ice Players (Away)", "OnIce Players (Away)"))
        legacy_on_ice = _first_non_empty(r, ("On-Ice Players", "OnIce Players"))
        score = (2 if home_on_ice else 0) + (2 if away_on_ice else 0) + (1 if legacy_on_ice else 0)
        if score > 0:
            k = (per, side, int(gs))
            if score > int(on_ice_score_by_key.get(k, 0)):
                on_ice_score_by_key[k] = int(score)
                on_ice_by_key[k] = r

    # Ensure destination headers include video fields.
    out_headers = list(existing_headers or [])
    for h in (
        "Event ID",
        "Video Time",
        "Video Seconds",
        "On-Ice Players",
        "On-Ice Players (Home)",
        "On-Ice Players (Away)",
    ):
        if h not in out_headers:
            out_headers.append(h)

    out_rows: list[dict[str, str]] = []
    for r in existing_rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        if _ev_type(rr) == "goal" and _has_source(rr, "timetoscore"):
            per = _period(rr)
            side = _norm_side(rr)
            gs = _game_seconds(rr)
            if per is not None and side is not None and gs is not None:
                k = (per, side, int(gs))
                match = long_by_key.get(k)
                if match is not None:
                    vs = _video_seconds(match)
                    vt = _video_time(match)
                    if vs is not None:
                        rr["Video Seconds"] = str(int(vs))
                    if vt:
                        rr["Video Time"] = vt
                    _add_source(rr, "long")

                if not str(rr.get("Event ID") or "").strip():
                    eid = event_id_by_key.get(k)
                    if eid is not None and str(eid[1]).strip():
                        rr["Event ID"] = str(eid[1]).strip()

                match_on_ice = on_ice_by_key.get(k)
                if match_on_ice is not None:
                    copied = False
                    copied |= _set_if_blank(
                        rr,
                        "On-Ice Players (Home)",
                        _first_non_empty(
                            match_on_ice, ("On-Ice Players (Home)", "OnIce Players (Home)")
                        ),
                    )
                    copied |= _set_if_blank(
                        rr,
                        "On-Ice Players (Away)",
                        _first_non_empty(
                            match_on_ice, ("On-Ice Players (Away)", "OnIce Players (Away)")
                        ),
                    )
                    copied |= _set_if_blank(
                        rr,
                        "On-Ice Players",
                        _first_non_empty(match_on_ice, ("On-Ice Players", "OnIce Players")),
                    )
                    if copied:
                        _add_source(rr, "shift_package")
        out_rows.append(rr)

    return out_headers, out_rows


def enrich_goal_video_times_from_long_events(
    *,
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    If a game has long-sheet events, prefer their goal clip timing for goal events in the unified
    game events stream.

    Motivation: some imports provide Goal rows from TimeToScore/goals.xlsx, while the long
    spreadsheet has better video_time/video_seconds alignment for the corresponding goal moment.

    Policy:
      - Never modifies rows that were explicitly corrected (e.g. via embedded YAML event_corrections).
      - Overwrites the goal row's Video Time/Seconds when a matching long-sheet event exists.
      - Adds "long" to the Source field when enrichment is applied.
    """
    if not headers or not rows:
        return headers, rows

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _has_source(row: dict[str, str], token: str) -> bool:
        return any(_norm(t) == _norm(token) for t in _split_sources(row.get("Source") or ""))

    def _add_source(row: dict[str, str], token: str) -> None:
        toks = _split_sources(row.get("Source") or "")
        toks_cf = {_norm(t) for t in toks}
        if _norm(token) not in toks_cf:
            toks.append(str(token))
        row["Source"] = ",".join([t for t in toks if t])

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _norm_side(row: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Side",
            "Team Rel",
            "TeamRel",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _period(row: dict[str, str]) -> Optional[int]:
        p = _parse_int(row.get("Period"))
        return p if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, str]) -> Optional[int]:
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is not None:
            return gs
        return parse_duration_seconds(
            row.get("Game Time") or row.get("GameTime") or row.get("Time")
        )

    def _video_seconds(row: dict[str, str]) -> Optional[int]:
        vs = _parse_int(row.get("Video Seconds") or row.get("VideoSeconds"))
        if vs is not None:
            return vs
        return parse_duration_seconds(row.get("Video Time") or row.get("VideoTime"))

    def _video_time(row: dict[str, str]) -> str:
        return str(row.get("Video Time") or row.get("VideoTime") or "").strip()

    def _player_id(row: dict[str, str]) -> Optional[int]:
        pid = _parse_int(row.get("__hm_player_id"))
        return pid if pid is not None and pid > 0 else None

    def _is_corrected(row: dict[str, str]) -> bool:
        # `_load_game_events_for_display()` injects `__hm_has_correction` for corrected rows.
        if str(row.get("__hm_has_correction") or "").strip():
            return True
        # Best-effort fallback for older rows/tests.
        return _has_source(row, "correction")

    # Build a lookup of long-sheet events that can anchor a goal clip.
    # Prefer Goal > xG > SOG > Shot at the same instant (and player when available).
    def _etype_prio(row: dict[str, str]) -> int:
        k = normalize_event_type_key(row.get("Event Type") or row.get("Event") or "")
        if k == "goal":
            return 0
        if k in {"xg", "expectedgoal", "expectedgoals"}:
            return 1
        if k in {"sog", "shotongoal", "shotsongoal"}:
            return 2
        if k in {"shot", "shots"}:
            return 3
        return 9

    def _row_prio(row: dict[str, str]) -> tuple[int, int, int]:
        vs = _video_seconds(row)
        vt = _video_time(row)
        return (_etype_prio(row), 0 if vs is not None else 1, 0 if vt else 1)

    long_by_pid: dict[tuple[int, str, int, int], dict[str, str]] = {}
    long_by_time: dict[tuple[int, str, int], dict[str, str]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not _has_source(r, "long"):
            continue
        per = _period(r)
        side = _norm_side(r)
        gs = _game_seconds(r)
        if per is None or side is None or gs is None:
            continue
        vs = _video_seconds(r)
        vt = _video_time(r)
        if vs is None and not vt:
            continue
        pid = _player_id(r)
        if pid is not None:
            k1 = (int(per), str(side), int(gs), int(pid))
            prev = long_by_pid.get(k1)
            if prev is None or _row_prio(r) < _row_prio(prev):
                long_by_pid[k1] = r
        k2 = (int(per), str(side), int(gs))
        prev2 = long_by_time.get(k2)
        if prev2 is None or _row_prio(r) < _row_prio(prev2):
            long_by_time[k2] = r

    if not long_by_pid and not long_by_time:
        return headers, rows

    out_rows: list[dict[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        if normalize_event_type_key(rr.get("Event Type") or rr.get("Event") or "") == "goal":
            if not _is_corrected(rr):
                per = _period(rr)
                side = _norm_side(rr)
                gs = _game_seconds(rr)
                if per is not None and side is not None and gs is not None:
                    match: Optional[dict[str, str]] = None
                    pid = _player_id(rr)
                    if pid is not None:
                        match = long_by_pid.get((int(per), str(side), int(gs), int(pid)))
                    if match is None:
                        match = long_by_time.get((int(per), str(side), int(gs)))
                    if match is not None:
                        vs = _video_seconds(match)
                        vt = _video_time(match)
                        if vs is not None:
                            rr["Video Seconds"] = str(int(vs))
                        if vt:
                            rr["Video Time"] = str(vt)
                        _add_source(rr, "long")
        out_rows.append(rr)

    return headers, out_rows


def enrich_timetoscore_penalties_with_video_times(
    *,
    existing_headers: list[str],
    existing_rows: list[dict[str, str]],
    incoming_headers: list[str],
    incoming_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    For TimeToScore-linked games, keep TimeToScore penalty events as authoritative, but
    copy Video Time/Seconds from matching penalty rows found in the incoming spreadsheet
    events CSV (which can map scoreboard time -> video time using shift sync).

    This is what makes penalty icons in the game timeline clickable (the UI requires
    Video Time/Seconds in the row to open the video clip).
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip().casefold()

    def _split_sources(raw: Any) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        parts = re.split(r"[,+;/\s]+", s)
        return [p for p in (pp.strip() for pp in parts) if p]

    def _has_source(row: dict[str, str], token: str) -> bool:
        return any(_norm(t) == _norm(token) for t in _split_sources(row.get("Source") or ""))

    def _add_source(row: dict[str, str], token: str) -> None:
        toks = _split_sources(row.get("Source") or "")
        toks_cf = {_norm(t) for t in toks}
        if _norm(token) not in toks_cf:
            toks.append(str(token))
        row["Source"] = ",".join([t for t in toks if t])

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _norm_side(row: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Side",
            "Team Rel",
            "TeamRel",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _ev_type(row: dict[str, str]) -> str:
        return _norm(row.get("Event Type") or row.get("Event") or "")

    def _period(row: dict[str, str]) -> Optional[int]:
        p = _parse_int(row.get("Period"))
        return p if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, str]) -> Optional[int]:
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is not None:
            return gs
        return parse_duration_seconds(
            row.get("Game Time") or row.get("GameTime") or row.get("Time")
        )

    def _video_seconds(row: dict[str, str]) -> Optional[int]:
        vs = _parse_int(row.get("Video Seconds") or row.get("VideoSeconds"))
        if vs is not None:
            return vs
        return parse_duration_seconds(row.get("Video Time") or row.get("VideoTime"))

    def _video_time(row: dict[str, str]) -> str:
        return str(row.get("Video Time") or row.get("VideoTime") or "").strip()

    def _has_video(row: dict[str, str]) -> bool:
        return _video_seconds(row) is not None or bool(_video_time(row))

    # Build generic per-period mapping points from incoming rows with both game+video seconds.
    # These points already incorporate stoppages because they are derived from shift sync.
    mapping_by_period: dict[int, list[tuple[int, int, dict[str, str]]]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        per = _period(r)
        gs = _game_seconds(r)
        vs = _video_seconds(r)
        if per is None or gs is None or vs is None:
            continue
        mapping_by_period.setdefault(int(per), []).append((int(gs), int(vs), r))
    for per, pts in list(mapping_by_period.items()):
        # Deduplicate by game seconds, keeping the earliest (minimum) video timestamp for that instant.
        best_by_gs: dict[int, tuple[int, dict[str, str]]] = {}
        for gs, vs, rr in pts:
            prev = best_by_gs.get(int(gs))
            if prev is None or int(vs) < int(prev[0]):
                best_by_gs[int(gs)] = (int(vs), dict(rr))
        mapping_by_period[per] = sorted(
            [(gs, vs, rr) for gs, (vs, rr) in best_by_gs.items()], key=lambda x: x[0]
        )

    def _interp_video_seconds(
        period: int, game_s: int
    ) -> Optional[
        tuple[int, tuple[tuple[int, int, dict[str, str]], tuple[int, int, dict[str, str]]]]
    ]:
        pts = mapping_by_period.get(int(period)) or []
        if len(pts) < 2:
            return None
        for i in range(len(pts) - 1):
            g0, v0, r0 = pts[i]
            g1, v1, r1 = pts[i + 1]
            lo, hi = (g0, g1) if g0 <= g1 else (g1, g0)
            if not (lo <= int(game_s) <= hi):
                continue
            if g0 == g1:
                continue
            # Linear interpolation (works for both increasing and decreasing mappings).
            v = int(round(v0 + (int(game_s) - g0) * (v1 - v0) / (g1 - g0)))
            return v, ((int(g0), int(v0), dict(r0)), (int(g1), int(v1), dict(r1)))
        return None

    # Build lookup from incoming penalty rows with video times.
    incoming_by_key: dict[tuple[int, str, int, str], dict[str, str]] = {}
    for r in incoming_rows or []:
        if not isinstance(r, dict):
            continue
        if _ev_type(r) != "penalty":
            continue
        per = _period(r)
        side = _norm_side(r)
        gs = _game_seconds(r)
        if per is None or side is None or gs is None:
            continue
        if not _has_video(r):
            continue
        jerseys = str(r.get("Attributed Jerseys") or "").strip()
        incoming_by_key[(int(per), side, int(gs), jerseys)] = r

    if not incoming_by_key:
        return existing_headers, existing_rows

    # Ensure destination headers include video fields.
    out_headers = list(existing_headers or [])
    for h in ("Video Time", "Video Seconds"):
        if h not in out_headers:
            out_headers.append(h)

    out_rows: list[dict[str, str]] = []
    for r in existing_rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        if _ev_type(rr) == "penalty" and _has_source(rr, "timetoscore") and not _has_video(rr):
            per = _period(rr)
            side = _norm_side(rr)
            gs = _game_seconds(rr)
            jerseys = str(rr.get("Attributed Jerseys") or "").strip()
            if per is not None and side is not None and gs is not None:
                match = incoming_by_key.get((int(per), side, int(gs), jerseys))
                # Fallback: match ignoring jersey if the scraper / import disagrees about attribution.
                if match is None:
                    match = incoming_by_key.get((int(per), side, int(gs), ""))
                if match is not None:
                    vs = _video_seconds(match)
                    vt = _video_time(match)
                    if vs is not None:
                        rr["Video Seconds"] = str(int(vs))
                        if not vt:
                            rr["Video Time"] = format_seconds_to_mmss_or_hhmmss(int(vs))
                    if vt:
                        rr["Video Time"] = vt
                    _add_source(rr, "shift_spreadsheet")
                else:
                    # Fallback: interpolate from any available shift-synced mapping points.
                    interp = _interp_video_seconds(int(per), int(gs))
                    if interp is not None:
                        vs2, _ = interp
                        rr["Video Seconds"] = str(int(vs2))
                        rr["Video Time"] = format_seconds_to_mmss_or_hhmmss(int(vs2))
                        _add_source(rr, "shift_spreadsheet")
        out_rows.append(rr)

    return out_headers, out_rows


def filter_game_stats_for_display(game_stats: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not game_stats:
        return game_stats

    def _drop_key(k: str) -> bool:
        kk = str(k or "").strip().lower()
        if not kk or kk == "_label":
            return False
        # No shift/ice-time related stats in the webapp.
        return ("toi" in kk) or ("ice time" in kk) or ("shift" in kk)

    out: dict[str, Any] = {}
    for k, v in (game_stats or {}).items():
        if _drop_key(str(k)):
            continue
        if k != "_label" and (v is None or str(v).strip() == ""):
            continue
        kk = str(k or "").strip().lower()
        if k != "_label" and ("ot" in kk):
            vv = str(v).strip()
            if vv in {"0", "0.0"}:
                continue
        out[k] = v
    return out


def normalize_event_type_key(raw: Any) -> str:
    """
    Stable key for event types across sources (e.g. "Expected Goal" vs "ExpectedGoal").
    """
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().casefold())


def compute_goalie_stats_for_game(
    event_rows: list[dict[str, Any]],
    *,
    home_goalies: Optional[list[dict[str, Any]]] = None,
    away_goalies: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Compute per-goalie stats for a single game from event rows.

    Uses:
      - goaliechange events (to determine which goalie is in net when)
      - goal events (goals against)
      - shot-on-goal events (shots against), from the implication chain:
          Goals ⊆ xG ⊆ SOG
        Concretely, we treat `goal`, `expectedgoal`, and `sog`/`shotongoal` rows as shot-on-goal
        evidence and de-duplicate per (period, game_seconds, shooter, side).
    """

    def _int_or_none(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    def _side_norm(row: dict[str, Any]) -> Optional[str]:
        for k in (
            "team_side_norm",
            "team_side",
            "Team Side",
            "TeamSide",
            "team_rel",
            "Team Rel",
            "TeamRel",
            "team_raw",
            "Team Raw",
            "TeamRaw",
            "Team",
        ):
            v = str(row.get(k) or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
        return None

    def _period(row: dict[str, Any]) -> Optional[int]:
        p = _int_or_none(row.get("period") if "period" in row else row.get("Period"))
        return int(p) if p is not None and p > 0 else None

    def _game_seconds(row: dict[str, Any]) -> Optional[int]:
        gs = _int_or_none(
            row.get("game_seconds")
            if "game_seconds" in row
            else row.get("Game Seconds") or row.get("GameSeconds")
        )
        if gs is not None:
            return int(gs)
        gt = row.get("game_time") if "game_time" in row else row.get("Game Time") or row.get("Time")
        return parse_duration_seconds(gt)

    def _event_type_key(row: dict[str, Any]) -> str:
        raw = (
            row.get("event_type_key")
            or row.get("event_type__key")
            or row.get("event_type")
            or row.get("Event Type")
            or row.get("Event")
        )
        return normalize_event_type_key(raw)

    def _attr_players(row: dict[str, Any]) -> str:
        return str(
            row.get("attributed_players")
            if "attributed_players" in row
            else row.get("Attributed Players") or row.get("AttributedPlayers") or ""
        ).strip()

    def _attr_jerseys(row: dict[str, Any]) -> str:
        return str(
            row.get("attributed_jerseys")
            if "attributed_jerseys" in row
            else row.get("Attributed Jerseys") or row.get("AttributedJerseys") or ""
        ).strip()

    def _details(row: dict[str, Any]) -> str:
        return str(row.get("details") if "details" in row else row.get("Details") or "").strip()

    def _player_id(row: dict[str, Any]) -> Optional[int]:
        return _int_or_none(row.get("player_id") or row.get("player") or row.get("Player ID"))

    def _event_id(row: dict[str, Any]) -> Optional[int]:
        return _int_or_none(row.get("event_id") or row.get("Event ID") or row.get("EventID"))

    def _goalie_maps(
        goalies: list[dict[str, Any]],
    ) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
        by_id: dict[int, dict[str, Any]] = {}
        name_to_ids: dict[str, set[int]] = {}
        for p in goalies or []:
            pid = _int_or_none(p.get("id") or p.get("player_id"))
            if pid is None:
                continue
            by_id[int(pid)] = p
            nm = normalize_player_name(str(p.get("name") or ""))
            if nm:
                name_to_ids.setdefault(nm, set()).add(int(pid))
        unique_by_name = {k: int(list(v)[0]) for k, v in name_to_ids.items() if len(v) == 1}
        return by_id, unique_by_name

    home_goalies = list(home_goalies or [])
    away_goalies = list(away_goalies or [])
    goalie_roster_by_side = {"home": home_goalies, "away": away_goalies}
    goalie_by_id: dict[str, dict[int, dict[str, Any]]] = {}
    goalie_id_by_name: dict[str, dict[str, int]] = {}
    goalie_side_by_pid: dict[int, str] = {}
    goalie_name_by_pid: dict[int, str] = {}
    for side, goalies in goalie_roster_by_side.items():
        by_id, by_name = _goalie_maps(goalies)
        goalie_by_id[side] = by_id
        goalie_id_by_name[side] = by_name
        for pid, rec in by_id.items():
            goalie_side_by_pid[int(pid)] = str(side)
            nm = str(rec.get("name") or "").strip()
            if nm and int(pid) not in goalie_name_by_pid:
                goalie_name_by_pid[int(pid)] = nm

    event_type_keys_present = {
        _event_type_key(r) for r in (event_rows or []) if isinstance(r, dict)
    }
    has_sog = bool(event_type_keys_present & {"goal", "expectedgoal", "sog", "shotongoal"})
    # Only consider xG stats available when ExpectedGoal rows exist (goals contribute to xG once xG
    # data exists, but goals alone should not force xG-based goalie columns to appear).
    has_xg = bool(event_type_keys_present & {"expectedgoal"})

    goalie_changes: dict[str, dict[int, list[tuple[int, Optional[int], str]]]] = {
        "home": {},
        "away": {},
    }
    for r in event_rows or []:
        if not isinstance(r, dict):
            continue
        if _event_type_key(r) != "goaliechange":
            continue
        side = _side_norm(r)
        per = _period(r)
        gs = _game_seconds(r)
        if side not in {"home", "away"} or per is None or gs is None:
            continue

        name = _attr_players(r)
        det = _details(r)
        empty_net = "emptynet" in normalize_player_name(
            name
        ) or "emptynet" in normalize_player_name(det)
        goalie_pid: Optional[int] = None if empty_net else _player_id(r)
        if goalie_pid is None and (not empty_net) and name:
            goalie_pid = goalie_id_by_name.get(str(side), {}).get(normalize_player_name(name))
        if goalie_pid is not None:
            goalie_side_by_pid[int(goalie_pid)] = str(side)

        goalie_name = name
        if goalie_pid is not None and not goalie_name:
            goalie_name = str(
                goalie_by_id.get(str(side), {}).get(int(goalie_pid), {}).get("name") or ""
            )
        if goalie_pid is not None and goalie_name and int(goalie_pid) not in goalie_name_by_pid:
            goalie_name_by_pid[int(goalie_pid)] = goalie_name

        goalie_changes.setdefault(str(side), {}).setdefault(int(per), []).append(
            (int(gs), goalie_pid, goalie_name)
        )

    def _infer_regulation_len_s() -> int:
        cand: list[int] = []
        for side in ("home", "away"):
            for gs, _pid, name in goalie_changes.get(side, {}).get(1, []):
                if "starting" in normalize_player_name(name):
                    cand.append(int(gs))
        if cand:
            return max(cand)

        cand = []
        for side in ("home", "away"):
            for gs, _pid, _name in goalie_changes.get(side, {}).get(1, []):
                cand.append(int(gs))
        if cand:
            return max(cand)

        cand = []
        for r in event_rows or []:
            if not isinstance(r, dict):
                continue
            per = _period(r)
            gs = _game_seconds(r)
            if per == 1 and gs is not None:
                cand.append(int(gs))
        if cand:
            return max(cand)
        return 15 * 60

    reg_len_s = int(_infer_regulation_len_s())

    def _period_len_s(period: int) -> int:
        if int(period) <= 3:
            return reg_len_s
        cand = []
        for r in event_rows or []:
            if not isinstance(r, dict):
                continue
            per = _period(r)
            gs = _game_seconds(r)
            if per == int(period) and gs is not None:
                cand.append(int(gs))
        return max(cand) if cand else reg_len_s

    periods = [1, 2, 3]
    extra_periods = sorted(
        {
            int(_period(r) or 0)
            for r in (event_rows or [])
            if isinstance(r, dict) and (_period(r) or 0) > 3
        }
    )
    periods.extend([p for p in extra_periods if p not in periods])

    start_goalie_by_side: dict[str, Optional[int]] = {"home": None, "away": None}
    for side in ("home", "away"):
        pts = goalie_changes.get(side, {}).get(1, [])
        if pts:
            pts_sorted = sorted(pts, key=lambda x: -int(x[0]))
            start_goalie_by_side[side] = pts_sorted[0][1]
        else:
            roster_goalies = goalie_roster_by_side.get(side) or []
            if len(roster_goalies) == 1:
                pid = _int_or_none(
                    roster_goalies[0].get("id") or roster_goalies[0].get("player_id")
                )
                start_goalie_by_side[side] = int(pid) if pid is not None else None
        if start_goalie_by_side.get(side) is not None:
            goalie_side_by_pid[int(start_goalie_by_side[side])] = str(side)

    points_by_side_period: dict[str, dict[int, list[tuple[int, Optional[int]]]]] = {
        "home": {},
        "away": {},
    }
    toi_by_goalie: dict[int, int] = {}

    for side in ("home", "away"):
        carry = start_goalie_by_side.get(side)
        for per in periods:
            per_len = int(_period_len_s(int(per)))
            changes = sorted(
                goalie_changes.get(side, {}).get(int(per), []), key=lambda x: -int(x[0])
            )
            for gs, pid, _name in changes:
                if int(gs) >= int(per_len) - 1:
                    carry = pid
                    break

            uniq: dict[int, Optional[int]] = {}
            uniq[int(per_len)] = carry
            for gs, pid, _name in changes:
                if int(gs) >= int(per_len) - 1:
                    continue
                uniq[int(gs)] = pid

            pts = sorted([(int(gs), pid) for gs, pid in uniq.items()], key=lambda x: -int(x[0]))
            points_by_side_period[side][int(per)] = pts

            for idx, (t0, goalie_pid) in enumerate(pts):
                t1 = pts[idx + 1][0] if idx + 1 < len(pts) else 0
                dur = int(t0) - int(t1)
                if dur > 0 and goalie_pid is not None:
                    toi_by_goalie[int(goalie_pid)] = toi_by_goalie.get(int(goalie_pid), 0) + int(
                        dur
                    )

            carry = pts[-1][1] if pts else carry

    def _active_goalie(side: str, *, period: int, game_s: int) -> Optional[int]:
        pts = points_by_side_period.get(str(side), {}).get(int(period), [])
        cur = None
        for t, pid in pts:
            if int(t) >= int(game_s):
                cur = pid
                continue
            break
        return cur

    goals_by_goalie: dict[int, set[tuple[int, int, str, str]]] = {}
    xg_shots_by_goalie: dict[int, set[tuple[int, int, str, str]]] = {}
    shots_by_goalie: dict[int, set[tuple[int, int, str, str]]] = {}

    for r in event_rows or []:
        if not isinstance(r, dict):
            continue
        et = _event_type_key(r)
        if et not in {"goal", "expectedgoal", "sog", "shotongoal"}:
            continue
        shooter_side = _side_norm(r)
        per = _period(r)
        gs = _game_seconds(r)
        if shooter_side not in {"home", "away"} or per is None or gs is None:
            continue
        defending_side = "away" if shooter_side == "home" else "home"
        goalie_pid = _active_goalie(defending_side, period=int(per), game_s=int(gs))
        if goalie_pid is None:
            continue

        shot_id = _player_id(r)
        if shot_id is None:
            shot_id = _int_or_none(normalize_jersey_number(_attr_jerseys(r)))
        if shot_id is None:
            shot_id = _event_id(r)
        shot_tag = (
            str(shot_id) if shot_id is not None else normalize_player_name(_attr_players(r)) or ""
        )
        shot_key = (int(per), int(gs), str(shooter_side), shot_tag)

        if et == "goal":
            goals_by_goalie.setdefault(int(goalie_pid), set()).add(shot_key)
        if has_xg and et in {"goal", "expectedgoal"}:
            xg_shots_by_goalie.setdefault(int(goalie_pid), set()).add(shot_key)
        if has_sog and et in {"goal", "expectedgoal", "sog", "shotongoal"}:
            shots_by_goalie.setdefault(int(goalie_pid), set()).add(shot_key)

    def _goalie_row(side: str, goalie_pid: int) -> dict[str, Any]:
        rec = goalie_by_id.get(str(side), {}).get(int(goalie_pid), {})
        name = str(rec.get("name") or goalie_name_by_pid.get(int(goalie_pid)) or "").strip()
        jersey = str(rec.get("jersey_number") or "").strip()
        toi = int(toi_by_goalie.get(int(goalie_pid), 0))
        ga = len(goals_by_goalie.get(int(goalie_pid), set()))
        xga = len(xg_shots_by_goalie.get(int(goalie_pid), set())) if has_xg else None
        xg_saves = (int(xga) - int(ga)) if (xga is not None) else None
        if xg_saves is not None and xg_saves < 0:
            xg_saves = 0
        xg_sv_pct = (
            (float(xg_saves) / float(xga))
            if (xga is not None and xga > 0 and xg_saves is not None)
            else None
        )
        sa = len(shots_by_goalie.get(int(goalie_pid), set())) if has_sog else None
        saves = (int(sa) - int(ga)) if (sa is not None) else None
        if saves is not None and saves < 0:
            saves = 0
        sv_pct = (
            (float(saves) / float(sa))
            if (sa is not None and sa > 0 and saves is not None)
            else None
        )
        gaa = (float(ga) * 60.0 / float(toi)) if toi > 0 else None
        return {
            "player_id": int(goalie_pid),
            "name": name,
            "jersey_number": jersey,
            "toi_seconds": toi,
            "ga": int(ga),
            "xga": int(xga) if xga is not None else None,
            "xg_saves": int(xg_saves) if xg_saves is not None else None,
            "xg_sv_pct": xg_sv_pct,
            "sa": int(sa) if sa is not None else None,
            "saves": int(saves) if saves is not None else None,
            "sv_pct": sv_pct,
            "gaa": gaa,
        }

    out: dict[str, Any] = {
        "meta": {"has_sog": bool(has_sog), "has_xg": bool(has_xg)},
        "home": [],
        "away": [],
    }
    for side in ("home", "away"):
        ids = set()
        ids.update(
            {
                int(_int_or_none(p.get("id") or p.get("player_id")) or 0)
                for p in goalie_roster_by_side.get(side, [])
            }
        )
        ids.update(
            {
                int(pid)
                for pid in toi_by_goalie.keys()
                if pid and goalie_side_by_pid.get(int(pid)) == str(side)
            }
        )
        ids.update(
            {
                int(pid)
                for pid in goals_by_goalie.keys()
                if pid and goalie_side_by_pid.get(int(pid)) == str(side)
            }
        )
        if has_xg:
            ids.update(
                {
                    int(pid)
                    for pid in xg_shots_by_goalie.keys()
                    if pid and goalie_side_by_pid.get(int(pid)) == str(side)
                }
            )
        if has_sog:
            ids.update(
                {
                    int(pid)
                    for pid in shots_by_goalie.keys()
                    if pid and goalie_side_by_pid.get(int(pid)) == str(side)
                }
            )
        ids = {i for i in ids if i > 0}
        rows = [_goalie_row(side, int(pid)) for pid in sorted(ids)]
        rows.sort(
            key=lambda r: (-int(r.get("toi_seconds") or 0), str(r.get("name") or "").casefold())
        )
        out[str(side)] = rows
    return out


def compute_goalie_stats_for_team_games(
    *,
    team_id: int,
    schedule_games: list[dict[str, Any]],
    event_rows_by_game_id: dict[int, list[dict[str, Any]]],
    goalies: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aggregate goalie stats across a list of games for a single team.

    Shots against (SA) and derived stats are only accumulated for games that include SOG events.
    """

    def _int_or_none(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    game_by_id = {}
    for g in schedule_games or []:
        try:
            game_by_id[int(g.get("id"))] = g
        except Exception:
            continue

    totals: dict[int, dict[str, Any]] = {}
    has_any_sog = False
    has_any_xg = False
    for gid, rows in (event_rows_by_game_id or {}).items():
        g = game_by_id.get(int(gid))
        if not g:
            continue
        t1 = _int0(g.get("team1_id"))
        t2 = _int0(g.get("team2_id"))
        if int(team_id) == int(t1):
            our_side = "home"
            stats = compute_goalie_stats_for_game(rows, home_goalies=goalies, away_goalies=[])
        elif int(team_id) == int(t2):
            our_side = "away"
            stats = compute_goalie_stats_for_game(rows, home_goalies=[], away_goalies=goalies)
        else:
            continue

        meta = stats.get("meta") or {}
        if bool(meta.get("has_sog")):
            has_any_sog = True
        if bool(meta.get("has_xg")):
            has_any_xg = True

        for gr in stats.get(our_side, []) or []:
            pid = _int_or_none(gr.get("player_id"))
            if pid is None or pid <= 0:
                continue
            rec = totals.setdefault(
                int(pid),
                {
                    "player_id": int(pid),
                    "name": str(gr.get("name") or "").strip(),
                    "jersey_number": str(gr.get("jersey_number") or "").strip(),
                    "gp": 0,
                    "toi_seconds": 0,
                    "ga": 0,
                    "sa_sum": 0,
                    "saves_sum": 0,
                    "has_sog": False,
                    "xga_sum": 0,
                    "xg_saves_sum": 0,
                    "has_xg": False,
                },
            )
            if not rec.get("name") and gr.get("name"):
                rec["name"] = str(gr.get("name") or "").strip()
            if not rec.get("jersey_number") and gr.get("jersey_number"):
                rec["jersey_number"] = str(gr.get("jersey_number") or "").strip()

            toi = _int_or_none(gr.get("toi_seconds")) or 0
            ga = _int_or_none(gr.get("ga")) or 0
            rec["toi_seconds"] += int(toi)
            rec["ga"] += int(ga)
            if toi > 0:
                rec["gp"] += 1

            sa = gr.get("sa")
            if sa is not None:
                rec["has_sog"] = True
                rec["sa_sum"] += int(_int_or_none(sa) or 0)
                rec["saves_sum"] += int(_int_or_none(gr.get("saves")) or 0)

            xga = gr.get("xga")
            if xga is not None:
                rec["has_xg"] = True
                rec["xga_sum"] += int(_int_or_none(xga) or 0)
                rec["xg_saves_sum"] += int(_int_or_none(gr.get("xg_saves")) or 0)

    out_rows: list[dict[str, Any]] = []
    for pid, rec in totals.items():
        toi = int(rec.get("toi_seconds") or 0)
        ga = int(rec.get("ga") or 0)
        has_sog = bool(rec.get("has_sog"))
        sa = int(rec.get("sa_sum") or 0) if has_sog else None
        saves = int(rec.get("saves_sum") or 0) if has_sog else None
        sv_pct = (
            (float(saves) / float(sa))
            if (has_sog and sa and sa > 0 and saves is not None)
            else None
        )
        has_xg = bool(rec.get("has_xg"))
        xga = int(rec.get("xga_sum") or 0) if has_xg else None
        xg_saves = int(rec.get("xg_saves_sum") or 0) if has_xg else None
        xg_sv_pct = (
            (float(xg_saves) / float(xga))
            if (has_xg and xga and xga > 0 and xg_saves is not None)
            else None
        )
        gaa = (float(ga) * 60.0 / float(toi)) if toi > 0 else None
        out_rows.append(
            {
                "player_id": int(pid),
                "jersey_number": str(rec.get("jersey_number") or "").strip(),
                "name": str(rec.get("name") or "").strip(),
                "gp": int(rec.get("gp") or 0),
                "toi_seconds": toi,
                "ga": ga,
                "xga": xga,
                "xg_saves": xg_saves,
                "xg_sv_pct": xg_sv_pct,
                "sa": sa,
                "saves": saves,
                "sv_pct": sv_pct,
                "gaa": gaa,
            }
        )

    out_rows.sort(
        key=lambda r: (-int(r.get("toi_seconds") or 0), str(r.get("name") or "").casefold())
    )
    return {"rows": out_rows, "meta": {"has_sog": bool(has_any_sog), "has_xg": bool(has_any_xg)}}


def compute_game_event_stats_by_side(events_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Build a simple Home/Away event-count table from normalized game events rows.
    Returns rows: {"event_type": str, "home": int, "away": int}
    """

    def _norm(s: Any) -> str:
        return str(s or "").strip()

    def _norm_cf(s: Any) -> str:
        return _norm(s).casefold()

    def _event_type(r: dict[str, str]) -> str:
        return _norm(r.get("Event Type") or r.get("Event") or r.get("Type") or "")

    def _side(r: dict[str, str]) -> Optional[str]:
        for k in (
            "Team Side",
            "TeamSide",
            "Team Rel",
            "TeamRel",
            "Side",
            "Team",
            "Team Raw",
            "TeamRaw",
        ):
            v = _norm_cf(r.get(k))
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
            if v in {"neutral"}:
                return None
        return None

    skip_types = {
        "assist",
        "penalty expired",
        "power play",
        "powerplay",
        "penalty kill",
        "penaltykill",
    }
    counts: dict[str, dict[str, int]] = {}
    for r in events_rows or []:
        if not isinstance(r, dict):
            continue
        et = _event_type(r)
        if not et:
            continue
        et_cf = et.casefold()
        if et_cf in skip_types:
            continue
        side = _side(r)
        if side not in {"home", "away"}:
            continue
        rec = counts.setdefault(et, {"home": 0, "away": 0})
        rec[side] += 1

    # Shot-tracking display semantics:
    #   Goals ⊆ xG ⊆ SOG ⊆ Shots
    #
    # Some event sources log shot attempts as mutually-exclusive categories:
    #   - "Goal" rows (goals)
    #   - "xG"/"Expected Goal" rows (non-goal xG)
    #   - "SOG"/"Shot on Goal" rows (non-xG shots on goal)
    #   - "Shot" rows (non-SOG shots)
    #
    # Player stat rows use these to build cumulative totals (shots_total = goals + xg + sog + shot),
    # so apply the same implication to the game-level event-count table for consistency.
    def _norm_et_key(et: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(et or "").casefold())

    goal_keys = {"goal"}
    xg_keys = {"xg", "expectedgoal", "expectedgoals"}
    sog_keys = {"sog", "shotongoal", "shotsongoal"}
    shot_keys = {"shot", "shots"}

    # Compute cumulative totals once, then override per-row counts for the corresponding types.
    derived: dict[str, dict[str, int]] = {
        "xg": {"home": 0, "away": 0},
        "sog": {"home": 0, "away": 0},
        "shot": {"home": 0, "away": 0},
    }
    if counts:
        for side in ("home", "away"):
            goals_n = sum(
                int(rec.get(side) or 0)
                for et, rec in counts.items()
                if _norm_et_key(et) in goal_keys
            )
            xg_n = sum(
                int(rec.get(side) or 0) for et, rec in counts.items() if _norm_et_key(et) in xg_keys
            )
            sog_n = sum(
                int(rec.get(side) or 0)
                for et, rec in counts.items()
                if _norm_et_key(et) in sog_keys
            )
            shot_n = sum(
                int(rec.get(side) or 0)
                for et, rec in counts.items()
                if _norm_et_key(et) in shot_keys
            )

            expected_goals_total = int(goals_n) + int(xg_n)
            sog_total = int(expected_goals_total) + int(sog_n)
            shots_total = int(sog_total) + int(shot_n)

            derived["xg"][side] = int(expected_goals_total)
            derived["sog"][side] = int(sog_total)
            derived["shot"][side] = int(shots_total)

        for et, rec in counts.items():
            k = _norm_et_key(et)
            if k in xg_keys:
                rec["home"] = int(derived["xg"]["home"])
                rec["away"] = int(derived["xg"]["away"])
            elif k in sog_keys:
                rec["home"] = int(derived["sog"]["home"])
                rec["away"] = int(derived["sog"]["away"])
            elif k in shot_keys:
                rec["home"] = int(derived["shot"]["home"])
                rec["away"] = int(derived["shot"]["away"])

    def _prio(et: str) -> int:
        key = et.casefold().replace(" ", "")
        order = [
            "goal",
            "penalty",
            "sog",
            "shot",
            "xg",
            "expectedgoal",
            "rush",
            "controlledentry",
            "controlledexit",
            "giveaway",
            "takeaway",
            "turnovers(forced)",
            "turnoverforced",
            "createdturnover",
            "goaliechange",
        ]
        try:
            return order.index(key)
        except Exception:
            return 10_000

    rows: list[dict[str, Any]] = []
    for et, rec in counts.items():
        if int(rec.get("home") or 0) == 0 and int(rec.get("away") or 0) == 0:
            continue
        rows.append(
            {"event_type": et, "home": int(rec.get("home") or 0), "away": int(rec.get("away") or 0)}
        )
    rows.sort(
        key=lambda r: (
            _prio(str(r.get("event_type") or "")),
            str(r.get("event_type") or "").casefold(),
        )
    )
    return rows


def normalize_events_video_time_for_display(
    headers: list[str],
    rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Display-time normalization for events video fields:
      - Ensure a human-readable "Video Time" column exists when any video clip field exists.
      - Ensure a numeric "Video Seconds" column exists when any video clip field exists.
      - For each row, if one of (Video Time, Video Seconds) exists but the other is missing, derive it.

    This normalization does not affect the stored CSV/event rows in the database; it only impacts
    the returned headers/rows used for UI rendering and embedded JSON.
    """
    if not headers or not rows:
        return headers, rows

    def _hnorm(h: Any) -> str:
        return str(h or "").strip().lower()

    def _header_idx(headers_in: list[str], norms: set[str]) -> Optional[int]:
        for i, h in enumerate(headers_in or []):
            if _hnorm(h) in norms:
                return i
        return None

    def _has_any_video(rows_in: list[dict[str, str]]) -> bool:
        for r in rows_in or []:
            if not isinstance(r, dict):
                continue
            vt_raw = str(r.get("Video Time") or r.get("VideoTime") or "").strip()
            if vt_raw:
                return True
            vs = parse_duration_seconds(
                r.get("Video Seconds")
                or r.get("VideoSeconds")
                or r.get("Video S")
                or r.get("VideoS")
            )
            if vs is not None:
                return True
        return False

    # If the table doesn't contain any clip metadata, keep it unchanged.
    if not _has_any_video(rows):
        return headers, rows

    vt_norms = {"video time", "videotime"}
    vs_norms = {"video seconds", "videoseconds", "video s", "videos"}
    gt_norms = {"game time", "gametime", "time"}

    out_headers = list(headers)
    vt_idx = _header_idx(out_headers, vt_norms)
    vs_idx = _header_idx(out_headers, vs_norms)

    if vt_idx is None and vs_idx is not None:
        # Prefer to place it next to Video Seconds if present.
        out_headers.insert(int(vs_idx), "Video Time")
        vt_idx = int(vs_idx)
        vs_idx = _header_idx(out_headers, vs_norms)

    if vs_idx is None and vt_idx is not None:
        # Place seconds next to Video Time when Video Time exists.
        out_headers.insert(int(vt_idx) + 1, "Video Seconds")
        vs_idx = int(vt_idx) + 1

    if vt_idx is None and vs_idx is None:
        # Last resort: place both near Game Time, or append.
        try:
            gt_idx = next(i for i, h in enumerate(out_headers) if _hnorm(h) in gt_norms)
            out_headers.insert(int(gt_idx) + 1, "Video Time")
            out_headers.insert(int(gt_idx) + 2, "Video Seconds")
        except Exception:
            out_headers.append("Video Time")
            out_headers.append("Video Seconds")

    out_rows: list[dict[str, str]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)

        vt = str(rr.get("Video Time") or rr.get("VideoTime") or "").strip()
        vs = parse_duration_seconds(
            rr.get("Video Seconds")
            or rr.get("VideoSeconds")
            or rr.get("Video S")
            or rr.get("VideoS")
        )

        if vs is None and vt:
            vs = parse_duration_seconds(vt)
            if vs is not None:
                vs_s = str(int(vs))
                for k in ("Video Seconds", "VideoSeconds", "Video S", "VideoS"):
                    if not str(rr.get(k) or "").strip():
                        rr[k] = vs_s

        if not vt and vs is not None:
            vt2 = format_seconds_to_mmss_or_hhmmss(vs)
            if vt2:
                for k in ("Video Time", "VideoTime"):
                    if not str(rr.get(k) or "").strip():
                        rr[k] = vt2

        out_rows.append(rr)

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    def _period_and_game_seconds(row: dict[str, str]) -> Optional[tuple[int, int]]:
        p = _parse_int(row.get("Period"))
        if p is None or p <= 0:
            return None
        gs = _parse_int(row.get("Game Seconds") or row.get("GameSeconds"))
        if gs is None:
            gs = parse_duration_seconds(
                row.get("Game Time") or row.get("GameTime") or row.get("Time")
            )
        if gs is None:
            return None
        return int(p), int(gs)

    def _count_jerseys(raw: Any) -> int:
        s = str(raw or "").strip()
        if not s:
            return 0
        nums: set[int] = set()
        for m0 in re.findall(r"(\d+)", s):
            try:
                nums.add(int(m0))
            except Exception:
                continue
        return len(nums)

    def _normalize_on_ice_str(raw: Any, *, max_players: int = 6) -> str:
        s = str(raw or "").strip()
        if not s:
            return ""
        # Most sources emit comma-separated "Name #Jersey" tokens; keep order but dedupe jerseys.
        parts = [p.strip() for p in re.split(r"[,;\n]+", s) if p and p.strip()]
        seen: set[tuple[str, Any]] = set()
        out: list[str] = []
        for p in parts:
            m = re.search(r"(\d+)", p)
            if m:
                try:
                    k = ("j", int(m.group(1)))
                except Exception:
                    k = ("s", p.casefold())
            else:
                k = ("s", p.casefold())
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        if len(out) > int(max_players):
            out = out[: int(max_players)]
            if out:
                out[-1] = out[-1] + " …"
        return ",".join(out)

    # Propagate missing clip/on-ice metadata across events at the same (period, game seconds).
    best_video_seconds: dict[tuple[int, int], dict[str, Any]] = {}
    best_on_ice_home: dict[tuple[int, int], str] = {}
    best_on_ice_away: dict[tuple[int, int], str] = {}

    for rr in out_rows:
        if not isinstance(rr, dict):
            continue
        k = _period_and_game_seconds(rr)
        if k is None:
            continue

        vt0, vs0 = normalize_video_time_and_seconds(
            rr.get("Video Time") or rr.get("VideoTime"),
            rr.get("Video Seconds")
            or rr.get("VideoSeconds")
            or rr.get("Video S")
            or rr.get("VideoS"),
        )
        if vs0 is not None:
            prev = best_video_seconds.get(k)
            if prev is None or int(vs0) < int(prev.get("video_seconds") or 0):
                best_video_seconds[k] = {
                    "video_seconds": int(vs0),
                    "video_time": str(vt0 or "").strip(),
                    "period": int(k[0]),
                    "game_seconds": int(k[1]),
                    "game_time": str(
                        rr.get("Game Time") or rr.get("GameTime") or rr.get("Time") or ""
                    ).strip(),
                }

        on_home = _normalize_on_ice_str(
            rr.get("On-Ice Players (Home)") or rr.get("OnIce Players (Home)") or ""
        )
        if on_home:
            prev = best_on_ice_home.get(k)
            if prev is None or _count_jerseys(on_home) > _count_jerseys(prev):
                best_on_ice_home[k] = on_home

        on_away = _normalize_on_ice_str(
            rr.get("On-Ice Players (Away)") or rr.get("OnIce Players (Away)") or ""
        )
        if on_away:
            prev = best_on_ice_away.get(k)
            if prev is None or _count_jerseys(on_away) > _count_jerseys(prev):
                best_on_ice_away[k] = on_away

    if best_video_seconds or best_on_ice_home or best_on_ice_away:
        for rr in out_rows:
            if not isinstance(rr, dict):
                continue
            k = _period_and_game_seconds(rr)
            if k is None:
                continue

            vt0, vs0 = normalize_video_time_and_seconds(
                rr.get("Video Time") or rr.get("VideoTime"),
                rr.get("Video Seconds")
                or rr.get("VideoSeconds")
                or rr.get("Video S")
                or rr.get("VideoS"),
            )
            if vs0 is None:
                best = best_video_seconds.get(k)
                if best is not None:
                    vs_best = best.get("video_seconds")
                    if vs_best is None:
                        continue
                    vs_s = str(int(vs_best))
                    rr.setdefault("Video Seconds", vs_s)
                    if not str(rr.get("Video Seconds") or "").strip():
                        rr["Video Seconds"] = vs_s
                    if not str(rr.get("VideoSeconds") or "").strip():
                        rr["VideoSeconds"] = vs_s
                    vt_best = format_seconds_to_mmss_or_hhmmss(vs_best)
                    if not str(rr.get("Video Time") or "").strip():
                        rr["Video Time"] = vt_best
                    if not str(rr.get("VideoTime") or "").strip():
                        rr["VideoTime"] = vt_best

            if not str(rr.get("On-Ice Players (Home)") or "").strip():
                home_best = best_on_ice_home.get(k)
                if home_best:
                    rr["On-Ice Players (Home)"] = home_best

            if not str(rr.get("On-Ice Players (Away)") or "").strip():
                away_best = best_on_ice_away.get(k)
                if away_best:
                    rr["On-Ice Players (Away)"] = away_best

    # Normalize on-ice strings for display (dedupe + clamp).
    for rr in out_rows:
        if not isinstance(rr, dict):
            continue
        home_norm = _normalize_on_ice_str(
            rr.get("On-Ice Players (Home)") or rr.get("OnIce Players (Home)") or ""
        )
        if home_norm:
            rr["On-Ice Players (Home)"] = home_norm
        away_norm = _normalize_on_ice_str(
            rr.get("On-Ice Players (Away)") or rr.get("OnIce Players (Away)") or ""
        )
        if away_norm:
            rr["On-Ice Players (Away)"] = away_norm
        legacy_norm = _normalize_on_ice_str(
            rr.get("On-Ice Players") or rr.get("OnIce Players") or rr.get("OnIcePlayers") or ""
        )
        if legacy_norm:
            rr["On-Ice Players"] = legacy_norm

    return out_headers, out_rows


def normalize_video_time_and_seconds(
    video_time: Any, video_seconds: Any
) -> tuple[str, Optional[int]]:
    """
    Best-effort bidirectional normalization for clip timestamps.

    Returns:
      - video_time: normalized string ('' when unknown)
      - video_seconds: int seconds (None when unknown/unparseable)
    """
    vt = str(video_time or "").strip()
    vs = parse_duration_seconds(video_seconds)
    if vs is None and vt:
        vs = parse_duration_seconds(vt)
    if not vt and vs is not None:
        vt = format_seconds_to_mmss_or_hhmmss(vs)
    return vt, vs


def _event_table_sort_key(r: dict[str, Any]) -> tuple[int, int, int]:
    """
    Stable default ordering for event tables:
      1) game datetime (descending; newest game first),
      2) period (ascending),
      3) game time within period (descending; clock counts down).
    """

    def _parse_int(v: Any) -> Optional[int]:
        try:
            return int(str(v or "").strip())
        except Exception:
            return None

    gs = _parse_int(
        r.get("Game Seconds")
        or r.get("GameSeconds")
        or r.get("game_seconds")
        or r.get("gameSeconds")
    )
    if gs is None:
        gs = parse_duration_seconds(
            r.get("Game Time")
            or r.get("GameTime")
            or r.get("Time")
            or r.get("game_time")
            or r.get("gameTime")
        )
    game_seconds = int(gs) if gs is not None else -1

    p = _parse_int(r.get("Period") or r.get("period"))
    period = int(p) if p is not None and p > 0 else 999

    dt_raw = r.get("game_starts_at") or r.get("starts_at") or r.get("Date") or r.get("date")
    dt_obj = to_dt(dt_raw)
    if dt_obj is None:
        game_dt_key = 99999999999999
    else:
        game_dt_key = -int(dt_obj.strftime("%Y%m%d%H%M%S"))

    return (int(game_dt_key), int(period), -int(game_seconds))


def sort_events_rows_default(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Default ordering for the Game Events table: see `_event_table_sort_key`.
    """
    return sorted(
        [r for r in (rows or []) if isinstance(r, dict)],
        key=_event_table_sort_key,
    )


def sort_event_dicts_for_table_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sort event dicts for UI tables using `_event_table_sort_key`.
    """
    return sorted(
        [r for r in (rows or []) if isinstance(r, dict)],
        key=_event_table_sort_key,
    )

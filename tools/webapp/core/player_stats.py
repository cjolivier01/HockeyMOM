import re
from typing import Any, Optional

from .events import filter_single_game_player_stats_csv, parse_events_csv
from .hockey import is_external_division_name
from .orm import _get_league_name, _orm_modules
from .shift_stats import (
    format_seconds_to_mmss_or_hhmmss,
    normalize_jersey_number,
    normalize_player_name,
    normalize_player_name_no_middle,
    parse_shift_stats_player_stats_csv,
)


# Stats persisted in `player_stats` DB rows. Derived shift/time stats (TOI/shifts) are computed
# from `hky_game_shift_rows` at runtime and are intentionally excluded here.
PLAYER_STATS_DB_KEYS: tuple[str, ...] = (
    "goals",
    "assists",
    "pim",
    "shots",
    "sog",
    "expected_goals",
    "plus_minus",
    "completed_passes",
    "giveaways",
    "turnovers_forced",
    "created_turnovers",
    "takeaways",
    # Shift-derived on-ice shot metrics (SOG for/against) are computed at runtime from
    # `hky_game_event_rows` + `hky_game_shift_rows` when enabled.
    "sog_for_on_ice",
    "sog_against_on_ice",
    # Shift-derived on-ice shot attempt metrics (for Pseudo-CF%) computed at runtime from
    # `hky_game_event_rows` + `hky_game_shift_rows` when enabled.
    "shots_for_on_ice",
    "shots_against_on_ice",
    "controlled_entry_for",
    "controlled_entry_against",
    "controlled_exit_for",
    "controlled_exit_against",
    "gf_counted",
    "ga_counted",
    "gt_goals",
    "gw_goals",
    "ot_goals",
    "ot_assists",
)

# Keys used for aggregation/display. Some of these are derived (e.g. TOI/shifts) and are not
# stored in the `player_stats` table.
PLAYER_STATS_SUM_KEYS: tuple[str, ...] = PLAYER_STATS_DB_KEYS + (
    "toi_seconds",
    "shifts",
)

PLAYER_STATS_DISPLAY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("gp", "GP"),
    ("toi_seconds", "TOI"),
    ("toi_seconds_per_game", "TOI per Game"),
    ("shifts", "Shifts"),
    ("shifts_per_game", "Shifts per Game"),
    ("goals", "Goals"),
    ("assists", "Assists"),
    ("points", "Points"),
    ("ppg", "PPG"),
    ("plus_minus", "Goal +/-"),
    ("plus_minus_per_game", "Goal +/- per Game"),
    ("gf_counted", "GF Counted"),
    ("gf_per_game", "GF per Game"),
    ("ga_counted", "GA Counted"),
    ("ga_per_game", "GA per Game"),
    ("shots", "Shots"),
    ("shots_per_game", "Shots per Game"),
    ("pseudo_cf_pct", "Pseudo-CF%"),
    ("sog", "SOG"),
    ("sog_per_game", "SOG per Game"),
    ("expected_goals", "xG"),
    ("expected_goals_per_game", "xG per Game"),
    ("expected_goals_per_sog", "xG per SOG"),
    ("completed_passes", "Completed Passes"),
    ("completed_passes_per_game", "Completed Passes per Game"),
    ("turnovers_forced", "Turnovers (forced)"),
    ("turnovers_forced_per_game", "Turnovers (forced) per Game"),
    ("created_turnovers", "Created Turnovers"),
    ("created_turnovers_per_game", "Created Turnovers per Game"),
    ("giveaways", "Giveaways"),
    ("giveaways_per_game", "Giveaways per Game"),
    ("takeaways", "Takeaways"),
    ("takeaways_per_game", "Takeaways per Game"),
    ("sog_for_on_ice", "SOG For (On-Ice)"),
    ("sog_against_on_ice", "SOG Against (On-Ice)"),
    ("controlled_entry_for", "Controlled Entry For (On-Ice)"),
    ("controlled_entry_for_per_game", "Controlled Entry For (On-Ice) per Game"),
    ("controlled_entry_against", "Controlled Entry Against (On-Ice)"),
    ("controlled_entry_against_per_game", "Controlled Entry Against (On-Ice) per Game"),
    ("controlled_exit_for", "Controlled Exit For (On-Ice)"),
    ("controlled_exit_for_per_game", "Controlled Exit For (On-Ice) per Game"),
    ("controlled_exit_against", "Controlled Exit Against (On-Ice)"),
    ("controlled_exit_against_per_game", "Controlled Exit Against (On-Ice) per Game"),
    ("gt_goals", "GT Goals"),
    ("gw_goals", "GW Goals"),
    ("ot_goals", "OT Goals"),
    ("ot_assists", "OT Assists"),
    ("pim", "PIM"),
    ("pim_per_game", "PIM per Game"),
)

OT_ONLY_PLAYER_STATS_KEYS: frozenset[str] = frozenset({"ot_goals", "ot_assists"})

GAME_PLAYER_STATS_COLUMNS: tuple[dict[str, Any], ...] = (
    {"id": "goals", "label": "G", "keys": ("goals",)},
    {"id": "assists", "label": "A", "keys": ("assists",)},
    {"id": "points", "label": "P", "keys": ("goals", "assists"), "op": "sum"},
    {"id": "toi_seconds", "label": "TOI", "keys": ("toi_seconds",)},
    {"id": "shifts", "label": "Shifts", "keys": ("shifts",)},
    {"id": "gt_goals", "label": "GT Goals", "keys": ("gt_goals",)},
    {"id": "gw_goals", "label": "GW Goals", "keys": ("gw_goals",)},
    {"id": "ot_goals", "label": "OT Goals", "keys": ("ot_goals",)},
    {"id": "ot_assists", "label": "OT Assists", "keys": ("ot_assists",)},
    {"id": "shots", "label": "S", "keys": ("shots",)},
    {"id": "pim", "label": "PIM", "keys": ("pim",)},
    {"id": "plus_minus", "label": "+/-", "keys": ("plus_minus",)},
    {"id": "sog", "label": "SOG", "keys": ("sog",)},
    {"id": "expected_goals", "label": "xG", "keys": ("expected_goals",)},
    {"id": "ce", "label": "CE (F/A)", "keys": ("controlled_entry_for", "controlled_entry_against")},
    {"id": "cx", "label": "CX (F/A)", "keys": ("controlled_exit_for", "controlled_exit_against")},
    {"id": "give_take", "label": "Give/Take", "keys": ("giveaways", "takeaways")},
    {"id": "gfga", "label": "GF/GA", "keys": ("gf_counted", "ga_counted")},
)

# Per-game player stats columns on the game page should use the same labels as the team page.
# Keep this set intentionally small; `filter_player_stats_display_columns_for_rows()` hides any
# column where all players have 0/blank values.
GAME_PLAYER_STATS_DISPLAY_KEYS: tuple[str, ...] = (
    "goals",
    "assists",
    "points",
    "toi_seconds",
    "shifts",
    "gt_goals",
    "gw_goals",
    "ot_goals",
    "ot_assists",
    "shots",
    "pseudo_cf_pct",
    "pim",
    "plus_minus",
    "sog",
    "expected_goals",
    "completed_passes",
    "controlled_entry_for",
    "controlled_entry_against",
    "controlled_exit_for",
    "controlled_exit_against",
    "giveaways",
    "takeaways",
    "gf_counted",
    "ga_counted",
)


def build_game_player_stats_display_columns(
    *,
    rows: list[dict[str, Any]],
    base_keys: tuple[str, ...] = GAME_PLAYER_STATS_DISPLAY_KEYS,
) -> list[dict[str, Any]]:
    """
    Return per-game player stat columns for the game page, using team-page wording.
    """
    label_by_key: dict[str, str] = {str(k): str(label) for k, label in PLAYER_STATS_DISPLAY_COLUMNS}
    base_cols: list[tuple[str, str]] = []
    for k in base_keys:
        label = label_by_key.get(str(k))
        if not label:
            continue
        base_cols.append((str(k), label))
    filtered = filter_player_stats_display_columns_for_rows(tuple(base_cols), rows)
    return [{"key": str(k), "label": str(label), "show_count": False} for k, label in filtered]


_PLAYER_STATS_IDENTITY_HEADERS: frozenset[str] = frozenset(
    {
        "player",
        "name",
        "jersey #",
        "jersey",
        "jersey no",
        "jersey number",
        "pos",
        "position",
    }
)

_PLAYER_STATS_HEADER_TO_DB_KEY: dict[str, str] = {
    # Common short headers
    "g": "goals",
    "a": "assists",
    "goals": "goals",
    "assists": "assists",
    "shots": "shots",
    "pim": "pim",
    "hits": "hits",
    "blocks": "blocks",
    "faceoff wins": "faceoff_wins",
    "faceoffs won": "faceoff_wins",
    "faceoff attempts": "faceoff_attempts",
    "faceoffs": "faceoff_attempts",
    "saves": "goalie_saves",
    "goalie saves": "goalie_saves",
    "ga": "goalie_ga",
    "goalie ga": "goalie_ga",
    "sa": "goalie_sa",
    "goalie sa": "goalie_sa",
    "sog": "sog",
    "xg": "expected_goals",
    "completed passes": "completed_passes",
    "completed pass": "completed_passes",
    "giveaways": "giveaways",
    "turnovers (forced)": "turnovers_forced",
    "created turnovers": "created_turnovers",
    "takeaways": "takeaways",
    "controlled entry for (on-ice)": "controlled_entry_for",
    "controlled entry against (on-ice)": "controlled_entry_against",
    "controlled exit for (on-ice)": "controlled_exit_for",
    "controlled exit against (on-ice)": "controlled_exit_against",
    "gt goals": "gt_goals",
    "gw goals": "gw_goals",
    "ot goals": "ot_goals",
    "ot assists": "ot_assists",
    "plus minus": "plus_minus",
    "goal +/-": "plus_minus",
    "gf counted": "gf_counted",
    "ga counted": "ga_counted",
}


def _normalize_header_for_lookup(h: str) -> str:
    return str(h or "").strip().lower()


def _normalize_column_id(h: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(h or "").strip().lower()).strip("_") or "col"
    return base


def _parse_int_from_cell_text(s: Any) -> int:
    """
    Parse integers from a cell like "2", "1/2", "1 / 2", returning max part.
    """
    if s is None:
        return 0
    ss = str(s).strip()
    if not ss:
        return 0
    parts = [p.strip() for p in re.split(r"\s*/\s*", ss) if p.strip()]
    out = 0
    for p in parts:
        try:
            out = max(out, int(float(p)))
        except Exception:
            continue
    return out


def _build_game_player_stats_table_from_imported_csv(
    *,
    players: list[dict[str, Any]],
    stats_by_pid: dict[int, dict[str, Any]],
    imported_csv_text: str,
    prefer_db_stats_for_keys: Optional[set[str]] = None,
) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, str]], dict[int, dict[str, bool]], Optional[str]
]:
    """
    Display-first per-game table: preserve imported CSV columns (minus identity fields),
    with optional DB merge/conflict highlighting for known numeric stats.
    """
    try:
        headers, rows = parse_events_csv(imported_csv_text)
    except Exception as e:  # noqa: BLE001
        return (
            list(GAME_PLAYER_STATS_COLUMNS),
            {},
            {},
            f"Unable to parse imported player_stats_csv: {e}",
        )

    if not headers:
        return [], {}, {}, "Imported player_stats_csv has no headers"

    # Never show shift/TOI/per-game/per-shift columns in the web UI, even if older data is stored.
    headers, rows = filter_single_game_player_stats_csv(headers, rows)
    if not headers:
        return [], {}, {}, "Imported player_stats_csv has no displayable columns"

    team_ids = sorted(
        {int(p.get("team_id") or 0) for p in (players or []) if p.get("team_id") is not None}
    )
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids_no_middle: dict[tuple[int, str], list[int]] = {}
    for p in players or []:
        try:
            pid = int(p.get("id"))
            tid = int(p.get("team_id") or 0)
        except Exception:
            continue
        jersey_norm = normalize_jersey_number(p.get("jersey_number"))
        if jersey_norm:
            jersey_to_player_ids.setdefault((tid, jersey_norm), []).append(pid)
        name_norm = normalize_player_name(str(p.get("name") or ""))
        if name_norm:
            name_to_player_ids.setdefault((tid, name_norm), []).append(pid)
        name_norm_no_middle = normalize_player_name_no_middle(str(p.get("name") or ""))
        if name_norm_no_middle:
            name_to_player_ids_no_middle.setdefault((tid, name_norm_no_middle), []).append(pid)

    def _resolve_player_id(
        jersey_norm: Optional[str], name_norm: str, name_norm_no_middle: str
    ) -> Optional[int]:
        candidates: list[int] = []
        for tid in team_ids:
            if jersey_norm:
                candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        candidates = []
        for tid in team_ids:
            candidates.extend(name_to_player_ids.get((tid, name_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        if name_norm_no_middle and name_norm_no_middle != name_norm:
            candidates = []
            for tid in team_ids:
                candidates.extend(name_to_player_ids_no_middle.get((tid, name_norm_no_middle), []))
            if len(set(candidates)) == 1:
                return int(list(set(candidates))[0])
        return None

    def _first_non_empty(d: dict[str, str], keys: tuple[str, ...]) -> str:
        for k in keys:
            v = str(d.get(k) or "").strip()
            if v:
                return v
        return ""

    imported_row_by_pid: dict[int, dict[str, str]] = {}
    for r in rows:
        jersey_raw = _first_non_empty(
            r,
            (
                "Jersey #",
                "Jersey",
                "Jersey No",
                "Jersey Number",
            ),
        )
        jersey_norm = normalize_jersey_number(jersey_raw) if jersey_raw else None
        player_name = _first_non_empty(r, ("Player", "Name"))
        name_part = player_name
        if jersey_norm is None:
            m = re.match(r"^\s*(\d+)\s+(.*)$", player_name)
            if m:
                jersey_norm = normalize_jersey_number(m.group(1))
                name_part = m.group(2).strip()
        name_norm = normalize_player_name(name_part)
        name_norm_no_middle = normalize_player_name_no_middle(name_part)
        pid = _resolve_player_id(jersey_norm, name_norm, name_norm_no_middle)
        if pid is None:
            continue
        imported_row_by_pid[int(pid)] = dict(r)

    # Build columns in imported header order (minus identity headers).
    columns: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for h in headers:
        key = _normalize_header_for_lookup(h)
        if key in _PLAYER_STATS_IDENTITY_HEADERS:
            continue
        col_id = _PLAYER_STATS_HEADER_TO_DB_KEY.get(key) or _normalize_column_id(h)
        if col_id in used_ids:
            # De-dupe while preserving order.
            i = 2
            while f"{col_id}_{i}" in used_ids:
                i += 1
            col_id = f"{col_id}_{i}"
        used_ids.add(col_id)
        columns.append(
            {
                "id": col_id,
                "label": str(h),
                "header": str(h),
                "db_key": _PLAYER_STATS_HEADER_TO_DB_KEY.get(key),
            }
        )

    all_pids = [int(p.get("id")) for p in (players or []) if p.get("id") is not None]
    cell_text_by_pid: dict[int, dict[str, str]] = {pid: {} for pid in all_pids}
    cell_conf_by_pid: dict[int, dict[str, bool]] = {pid: {} for pid in all_pids}

    # Populate cells (imported first; merge with DB for known numeric keys).
    for pid in all_pids:
        db_row = stats_by_pid.get(pid) or {}
        imp_row = imported_row_by_pid.get(pid) or {}
        for col in columns:
            cid = str(col["id"])
            header = str(col.get("header") or "")
            db_key = col.get("db_key")
            raw_v = str(imp_row.get(header) or "").strip()
            if db_key:
                if prefer_db_stats_for_keys and str(db_key) in prefer_db_stats_for_keys:
                    v, s, is_conf = _merge_stat_values(db_row.get(str(db_key)), None)
                else:
                    v, s, is_conf = _merge_stat_values(db_row.get(str(db_key)), raw_v)
                cell_text_by_pid[pid][cid] = s
                cell_conf_by_pid[pid][cid] = bool(is_conf)
            else:
                cell_text_by_pid[pid][cid] = raw_v
                cell_conf_by_pid[pid][cid] = False

    # Hide columns that are entirely blank, and OT-only columns if all zero/blank.
    visible: list[dict[str, Any]] = []
    for col in columns:
        cid = str(col["id"])
        label_l = str(col.get("label") or "").strip().lower()
        vals = [cell_text_by_pid.get(pid, {}).get(cid, "") for pid in all_pids]
        if all(_is_blank_stat(v) or str(v).strip() == "" for v in vals):
            continue
        if label_l.startswith("ot ") or label_l in {"ot goals", "ot assists"}:
            if all(_is_zero_or_blank_stat(_parse_int_from_cell_text(v)) for v in vals):
                continue
        visible.append(col)

    # Add derived Points column (Goals + Assists) when those columns exist.
    goals_col_id = next(
        (str(c["id"]) for c in visible if str(c.get("db_key") or "") == "goals"), None
    )
    assists_col_id = next(
        (str(c["id"]) for c in visible if str(c.get("db_key") or "") == "assists"), None
    )
    if goals_col_id and assists_col_id:
        points_id = "points"
        if any(str(c.get("id")) == points_id for c in visible):
            points_id = "points_2"

        for pid in all_pids:
            g_txt = str(cell_text_by_pid.get(pid, {}).get(goals_col_id, "") or "")
            a_txt = str(cell_text_by_pid.get(pid, {}).get(assists_col_id, "") or "")
            any_part = bool(g_txt.strip() or a_txt.strip())
            if any_part:
                pts = _parse_int_from_cell_text(g_txt) + _parse_int_from_cell_text(a_txt)
                cell_text_by_pid[pid][points_id] = str(int(pts))
            else:
                cell_text_by_pid[pid][points_id] = ""
            cell_conf_by_pid[pid][points_id] = bool(
                cell_conf_by_pid.get(pid, {}).get(goals_col_id)
                or cell_conf_by_pid.get(pid, {}).get(assists_col_id)
            )

        pts_vals = [cell_text_by_pid.get(pid, {}).get(points_id, "") for pid in all_pids]
        if not all(_is_blank_stat(v) or str(v).strip() == "" for v in pts_vals):
            insert_at = next(
                (i + 1 for i, c in enumerate(visible) if str(c.get("id")) == assists_col_id), None
            )
            if insert_at is None:
                insert_at = 0
            visible.insert(
                int(insert_at), {"id": points_id, "label": "P", "header": "P", "db_key": None}
            )

    # Rebuild cell dicts to only include visible columns.
    vis_ids = {str(c["id"]) for c in visible}
    cell_text_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_text_by_pid.items()
    }
    cell_conf_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_conf_by_pid.items()
    }
    return visible, cell_text_by_pid, cell_conf_by_pid, None


def _int0(v: Any) -> int:
    try:
        if v is None:
            return 0
        return int(float(str(v)))
    except Exception:
        return 0


def _rate_or_none(numer: float, denom: float) -> Optional[float]:
    try:
        if denom <= 0:
            return None
        return float(numer) / float(denom)
    except Exception:
        return None


def compute_player_display_stats(
    sums: dict[str, Any], *, per_game_denoms: Optional[dict[str, int]] = None
) -> dict[str, Any]:
    gp = _int0(sums.get("gp"))
    goals = _int0(sums.get("goals"))
    assists = _int0(sums.get("assists"))
    points = goals + assists

    toi_seconds = _int0(sums.get("toi_seconds"))
    shifts = _int0(sums.get("shifts"))

    shots = _int0(sums.get("shots"))
    sog = _int0(sums.get("sog"))
    xg = _int0(sums.get("expected_goals"))
    pim = _int0(sums.get("pim"))
    plus_minus = _int0(sums.get("plus_minus"))
    gf = _int0(sums.get("gf_counted"))
    ga = _int0(sums.get("ga_counted"))
    giveaways = _int0(sums.get("giveaways"))
    takeaways = _int0(sums.get("takeaways"))
    completed_passes = _int0(sums.get("completed_passes"))
    turnovers_forced = _int0(sums.get("turnovers_forced"))
    created_turnovers = _int0(sums.get("created_turnovers"))
    ce_for = _int0(sums.get("controlled_entry_for"))
    ce_against = _int0(sums.get("controlled_entry_against"))
    cx_for = _int0(sums.get("controlled_exit_for"))
    cx_against = _int0(sums.get("controlled_exit_against"))

    faceoff_wins = _int0(sums.get("faceoff_wins"))
    faceoff_attempts = _int0(sums.get("faceoff_attempts"))
    goalie_saves = _int0(sums.get("goalie_saves"))
    goalie_sa = _int0(sums.get("goalie_sa"))

    shots_for_on_ice = _int0(sums.get("shots_for_on_ice"))
    shots_against_on_ice = _int0(sums.get("shots_against_on_ice"))

    def _denom_for(key: str) -> int:
        if per_game_denoms is not None and str(key) in per_game_denoms:
            try:
                v = int(per_game_denoms.get(str(key)) or 0)
            except Exception:
                v = 0
            return max(0, int(v))
        return int(gp)

    out: dict[str, Any] = dict(sums)
    # Stat implications for display/aggregation:
    #   Goals ⊆ xG ⊆ SOG ⊆ Shots
    #
    # Important: for team-page aggregates we may be mixing games that have full shot-tracking
    # (long spreadsheet) with games that only have goals. When `per_game_denoms` is provided,
    # treat shots/SOG/xG as "measured" stats and do not infer them from goals, otherwise per-game
    # rates become misleading.
    if per_game_denoms is None:
        xg = max(xg, goals)
        sog = max(sog, xg)
        shots = max(shots, sog)
    out["expected_goals"] = xg
    out["sog"] = sog
    out["shots"] = shots
    out["gp"] = gp
    out["points"] = points
    out["ppg"] = _rate_or_none(points, _denom_for("ppg"))

    # Per-game rates.
    toi_gp = _denom_for("toi_seconds_per_game")
    out["toi_seconds_per_game"] = (
        int(round(float(toi_seconds) / float(toi_gp))) if toi_gp > 0 else None
    )
    out["shifts_per_game"] = _rate_or_none(shifts, _denom_for("shifts_per_game"))
    out["shots_per_game"] = _rate_or_none(shots, _denom_for("shots_per_game"))
    out["sog_per_game"] = _rate_or_none(sog, _denom_for("sog_per_game"))
    out["expected_goals_per_game"] = _rate_or_none(xg, _denom_for("expected_goals_per_game"))
    out["plus_minus_per_game"] = _rate_or_none(plus_minus, _denom_for("plus_minus_per_game"))
    out["gf_per_game"] = _rate_or_none(gf, _denom_for("gf_per_game"))
    out["ga_per_game"] = _rate_or_none(ga, _denom_for("ga_per_game"))
    out["giveaways_per_game"] = _rate_or_none(giveaways, _denom_for("giveaways_per_game"))
    out["takeaways_per_game"] = _rate_or_none(takeaways, _denom_for("takeaways_per_game"))
    out["completed_passes_per_game"] = _rate_or_none(
        completed_passes, _denom_for("completed_passes_per_game")
    )
    out["turnovers_forced_per_game"] = _rate_or_none(
        turnovers_forced, _denom_for("turnovers_forced_per_game")
    )
    out["created_turnovers_per_game"] = _rate_or_none(
        created_turnovers, _denom_for("created_turnovers_per_game")
    )
    out["controlled_entry_for_per_game"] = _rate_or_none(
        ce_for, _denom_for("controlled_entry_for_per_game")
    )
    out["controlled_entry_against_per_game"] = _rate_or_none(
        ce_against, _denom_for("controlled_entry_against_per_game")
    )
    out["controlled_exit_for_per_game"] = _rate_or_none(
        cx_for, _denom_for("controlled_exit_for_per_game")
    )
    out["controlled_exit_against_per_game"] = _rate_or_none(
        cx_against, _denom_for("controlled_exit_against_per_game")
    )
    out["pim_per_game"] = _rate_or_none(pim, _denom_for("pim_per_game"))
    out["hits_per_game"] = _rate_or_none(_int0(sums.get("hits")), _denom_for("hits_per_game"))
    out["blocks_per_game"] = _rate_or_none(_int0(sums.get("blocks")), _denom_for("blocks_per_game"))

    out["expected_goals_per_sog"] = _rate_or_none(xg, sog)
    out["faceoff_pct"] = _rate_or_none(faceoff_wins, faceoff_attempts)
    out["goalie_sv_pct"] = _rate_or_none(goalie_saves, goalie_sa)

    pseudo_cf = _rate_or_none(
        float(shots_for_on_ice), float(shots_for_on_ice + shots_against_on_ice)
    )
    out["pseudo_cf_pct"] = (float(pseudo_cf) * 100.0) if pseudo_cf is not None else None
    if per_game_denoms is not None:
        out["_per_game_denoms"] = {str(k): int(v) for k, v in dict(per_game_denoms).items()}
    return out


def _classify_coach_position(pos: Any) -> Optional[str]:
    """
    Returns "HC" or "AC" if position indicates a coach; otherwise None.
    """
    p = str(pos or "").strip().upper()
    if p in {"HC", "HEAD COACH"}:
        return "HC"
    if p in {"AC", "ASSISTANT COACH"}:
        return "AC"
    return None


def _classify_roster_role(p: dict[str, Any]) -> Optional[str]:
    """
    Returns "HC", "AC", or "G" when the player dict is clearly a coach/goalie.
    Falls back to None for skaters/unknown.
    """
    jersey = str(p.get("jersey_number") or "").strip().upper()
    if jersey in {"HC", "HEAD COACH"}:
        return "HC"
    if jersey in {"AC", "ASSISTANT COACH"}:
        return "AC"
    if jersey in {"G", "GOALIE", "GOALTENDER"}:
        return "G"

    pos = p.get("position")
    role = _classify_coach_position(pos)
    if role:
        return role
    if _is_goalie_position(pos):
        return "G"

    name = str(p.get("name") or "").strip()
    if not name:
        return None
    name_up = name.upper()

    # Some imports encode coach role in the *name* field (position can be blank).
    if (
        re.match(r"^\s*HC\b", name_up)
        or re.search(r"\bHEAD\s+COACH\b", name_up)
        or re.search(r"\(HC\)", name_up)
    ):
        return "HC"
    if (
        re.match(r"^\s*AC\b", name_up)
        or re.search(r"\bASSISTANT\s+COACH\b", name_up)
        or re.search(r"\(AC\)", name_up)
    ):
        return "AC"

    # Conservative goalie hint when position is missing.
    if (
        re.search(r"\bGOALIE\b", name_up)
        or re.search(r"\bGOALTENDER\b", name_up)
        or re.search(r"\(G\)", name_up)
    ):
        return "G"

    return None


def _is_goalie_position(pos: Any) -> bool:
    p = str(pos or "").strip().upper()
    if not p:
        return False
    # Normalize common variants.
    p = re.sub(r"[()]", "", p).strip()
    if p in {"G", "GOALIE", "GOALTENDER"}:
        return True
    # Allow things like "G1", "G2", "G - Starter".
    if p.startswith("G"):
        return True
    return False


def split_players_and_coaches(
    players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Separate coaches (HC/AC) from players so they don't appear in player stats lists.
    Returns: (players_only, head_coaches, assistant_coaches)
    """
    players_only: list[dict[str, Any]] = []
    head_coaches: list[dict[str, Any]] = []
    assistant_coaches: list[dict[str, Any]] = []
    for p in players or []:
        role = _classify_roster_role(p)
        if role == "HC":
            head_coaches.append(p)
        elif role == "AC":
            assistant_coaches.append(p)
        else:
            players_only.append(p)
    return players_only, head_coaches, assistant_coaches


def split_roster(
    players: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split a team roster into:
      - skaters (non-coach, non-goalie)
      - goalies
      - head coaches
      - assistant coaches

    This is used to keep coaches/goalies out of *stats* tables while still showing
    them in roster tables.
    """
    skaters: list[dict[str, Any]] = []
    goalies: list[dict[str, Any]] = []
    head_coaches: list[dict[str, Any]] = []
    assistant_coaches: list[dict[str, Any]] = []
    for p in players or []:
        role = _classify_roster_role(p)
        if role == "HC":
            head_coaches.append(p)
            continue
        if role == "AC":
            assistant_coaches.append(p)
            continue
        if role == "G":
            goalies.append(p)
            continue
        skaters.append(p)
    return skaters, goalies, head_coaches, assistant_coaches


def _is_blank_stat(v: Any) -> bool:
    return v is None or v == ""


def _is_zero_or_blank_stat(v: Any) -> bool:
    if _is_blank_stat(v):
        return True
    if isinstance(v, str):
        s = v.strip()
        if "/" in s:
            parts = s.split("/")
            if parts and all(_is_zero_or_blank_stat(part) for part in parts):
                return True
    try:
        return float(v) == 0.0  # type: ignore[arg-type]
    except Exception:
        return False


def filter_player_stats_display_columns_for_rows(
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """
    Hide:
      - Any column that is entirely blank (missing data)
      - Any column where all values are 0/blank
    """
    if not columns:
        return columns
    out: list[tuple[str, str]] = []
    gf_counted_all_zero: Optional[bool] = None
    ga_counted_all_zero: Optional[bool] = None
    for k, label in columns:
        vals = [r.get(k) for r in (rows or [])]
        all_zero = all(_is_zero_or_blank_stat(v) for v in vals)
        if k == "gf_counted":
            gf_counted_all_zero = all_zero
        elif k == "ga_counted":
            ga_counted_all_zero = all_zero
        if all_zero:
            continue
        out.append((k, label))
    if gf_counted_all_zero and ga_counted_all_zero:
        out = [
            (key, label)
            for key, label in out
            if key not in {"plus_minus", "plus_minus_per_game", "gf_counted", "ga_counted"}
        ]
    return tuple(out)


def _merge_stat_values(db_v: Any, imported_v: Any) -> tuple[Optional[int], str, bool]:
    """
    Returns: (merged_numeric_value_or_None, display_string, is_conflict)
    """

    def _to_int(v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except Exception:
            try:
                return int(float(str(v)))
            except Exception:
                return None

    a = _to_int(db_v)
    b = _to_int(imported_v)
    if a is None and b is None:
        return None, "", False
    if a is None:
        return b, str(b), False
    if b is None:
        return a, str(a), False
    if a == b:
        return a, str(a), False

    # Treat a single 0 vs non-zero as "missing" from one source (common in partial imports).
    if a == 0 and b != 0:
        return b, str(b), False
    if b == 0 and a != 0:
        return a, str(a), False

    return a, f"{a}/{b}", True


def _map_imported_shift_stats_to_player_ids(
    *,
    players: list[dict[str, Any]],
    imported_csv_text: Optional[str],
) -> tuple[dict[int, dict[str, Any]], Optional[str]]:
    """
    Returns (imported_stats_by_pid, parse_warning).
    """
    if not imported_csv_text or not str(imported_csv_text).strip():
        return {}, None
    try:
        parsed_rows = parse_shift_stats_player_stats_csv(str(imported_csv_text))
    except Exception as e:  # noqa: BLE001
        return {}, f"Unable to parse imported player_stats_csv: {e}"

    team_ids = sorted(
        {int(p.get("team_id") or 0) for p in (players or []) if p.get("team_id") is not None}
    )
    jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
    name_to_player_ids: dict[tuple[int, str], list[int]] = {}
    for p in players or []:
        try:
            pid = int(p.get("id"))
            tid = int(p.get("team_id") or 0)
        except Exception:
            continue
        jersey_norm = normalize_jersey_number(p.get("jersey_number"))
        if jersey_norm:
            jersey_to_player_ids.setdefault((tid, jersey_norm), []).append(pid)
        name_norm = normalize_player_name(str(p.get("name") or ""))
        if name_norm:
            name_to_player_ids.setdefault((tid, name_norm), []).append(pid)

    def _resolve_player_id(jersey_norm: Optional[str], name_norm: str) -> Optional[int]:
        candidates: list[int] = []
        for tid in team_ids:
            if jersey_norm:
                candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        candidates = []
        for tid in team_ids:
            candidates.extend(name_to_player_ids.get((tid, name_norm), []))
        if len(set(candidates)) == 1:
            return int(list(set(candidates))[0])
        return None

    imported_by_pid: dict[int, dict[str, Any]] = {}
    for row in parsed_rows:
        jersey_norm = row.get("jersey_number")
        name_norm = row.get("name_norm") or ""
        pid = _resolve_player_id(jersey_norm, name_norm)
        if pid is None:
            continue
        imported_by_pid[int(pid)] = dict(row.get("stats") or {})
    return imported_by_pid, None


def build_game_player_stats_table(
    *,
    players: list[dict[str, Any]],
    stats_by_pid: dict[int, dict[str, Any]],
    imported_csv_text: Optional[str],
    prefer_db_stats_for_keys: Optional[set[str]] = None,
) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, str]], dict[int, dict[str, bool]], Optional[str]
]:
    """
    Build a merged (DB + imported CSV) per-game player stats table.
    Returns: (visible_columns, cell_text_by_pid, cell_conflict_by_pid, imported_parse_warning)
    """
    if imported_csv_text and str(imported_csv_text).strip():
        return _build_game_player_stats_table_from_imported_csv(
            players=players,
            stats_by_pid=stats_by_pid,
            imported_csv_text=str(imported_csv_text),
            prefer_db_stats_for_keys=prefer_db_stats_for_keys,
        )

    imported_by_pid, imported_warning = _map_imported_shift_stats_to_player_ids(
        players=players, imported_csv_text=imported_csv_text
    )

    all_pids = [int(p.get("id")) for p in (players or []) if p.get("id") is not None]

    merged_vals: dict[int, dict[str, Optional[int]]] = {pid: {} for pid in all_pids}
    merged_disp: dict[int, dict[str, str]] = {pid: {} for pid in all_pids}
    merged_conf: dict[int, dict[str, bool]] = {pid: {} for pid in all_pids}

    all_keys: set[str] = set()
    for c in GAME_PLAYER_STATS_COLUMNS:
        for k in c.get("keys") or ():
            all_keys.add(str(k))

    for pid in all_pids:
        db_row = stats_by_pid.get(pid) or {}
        imp_row = imported_by_pid.get(pid) or {}
        for k in all_keys:
            v, s, is_conf = _merge_stat_values(db_row.get(k), imp_row.get(k))
            merged_vals[pid][k] = v
            merged_disp[pid][k] = s
            merged_conf[pid][k] = bool(is_conf)

    duration_keys: frozenset[str] = frozenset(
        {
            "toi_seconds",
            "video_toi_seconds",
            "sb_avg_shift_seconds",
            "sb_median_shift_seconds",
            "sb_longest_shift_seconds",
            "sb_shortest_shift_seconds",
        }
    )

    def _fmt_duration_display(raw: str) -> str:
        """
        Format merge display values for duration stats.
        Inputs are typically seconds like "754" or conflict strings like "754/760".
        """
        s = str(raw or "").strip()
        if not s:
            return ""
        if "/" in s:
            parts = [p.strip() for p in s.split("/", 1)]
            if len(parts) == 2:
                a = format_seconds_to_mmss_or_hhmmss(parts[0])
                b = format_seconds_to_mmss_or_hhmmss(parts[1])
                if a and b:
                    return f"{a}/{b}"
        out = format_seconds_to_mmss_or_hhmmss(s)
        return out or s

    for pid in all_pids:
        for k in duration_keys:
            disp = merged_disp[pid].get(k)
            if disp:
                merged_disp[pid][k] = _fmt_duration_display(disp)

    visible_columns: list[dict[str, Any]] = []
    for col in GAME_PLAYER_STATS_COLUMNS:
        keys = [str(k) for k in (col.get("keys") or ())]
        if keys and set(keys).issubset(OT_ONLY_PLAYER_STATS_KEYS):
            if all(
                all(_is_zero_or_blank_stat(merged_vals[pid].get(k)) for k in keys)
                for pid in all_pids
            ):
                continue
        if keys and all(all(merged_vals[pid].get(k) is None for k in keys) for pid in all_pids):
            continue
        visible_columns.append(dict(col))

    cell_text_by_pid: dict[int, dict[str, str]] = {}
    cell_conflict_by_pid: dict[int, dict[str, bool]] = {}
    for pid in all_pids:
        out_text: dict[str, str] = {}
        out_conf: dict[str, bool] = {}
        for col in visible_columns:
            col_id = str(col.get("id"))
            keys = [str(k) for k in (col.get("keys") or ())]
            op = str(col.get("op") or "").strip().lower()
            parts = [merged_disp[pid].get(k, "") for k in keys]
            any_part = any(str(p).strip() for p in parts)
            if len(keys) == 1:
                out_text[col_id] = parts[0] if parts else ""
                out_conf[col_id] = bool(keys and merged_conf[pid].get(keys[0]))
            else:
                if op == "sum":
                    if any_part:
                        out_text[col_id] = str(sum(_int0(merged_vals[pid].get(k)) for k in keys))
                    else:
                        out_text[col_id] = ""
                    out_conf[col_id] = any(bool(merged_conf[pid].get(k)) for k in keys)
                else:
                    if any_part:
                        filled = [p if str(p).strip() else "0" for p in parts]
                        out_text[col_id] = " / ".join(filled)
                    else:
                        out_text[col_id] = ""
                    out_conf[col_id] = any(bool(merged_conf[pid].get(k)) for k in keys)
        cell_text_by_pid[pid] = out_text
        cell_conflict_by_pid[pid] = out_conf

    # Hide any columns that are entirely blank.
    filtered_cols: list[dict[str, Any]] = []
    for col in visible_columns:
        cid = str(col.get("id"))
        vals = [str(cell_text_by_pid.get(pid, {}).get(cid, "") or "") for pid in all_pids]
        if all(_is_blank_stat(v) or v.strip() == "" for v in vals):
            continue
        filtered_cols.append(col)
    visible_columns = filtered_cols
    vis_ids = {str(c.get("id")) for c in visible_columns}
    cell_text_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids} for pid, row in cell_text_by_pid.items()
    }
    cell_conflict_by_pid = {
        pid: {k: v for k, v in row.items() if k in vis_ids}
        for pid, row in cell_conflict_by_pid.items()
    }

    return visible_columns, cell_text_by_pid, cell_conflict_by_pid, imported_warning


def _empty_player_display_stats(player_id: int) -> dict[str, Any]:
    base: dict[str, Any] = {"player_id": int(player_id), "gp": 0}
    for k in PLAYER_STATS_SUM_KEYS:
        base[k] = 0
    return compute_player_display_stats(base)


def build_player_stats_table_rows(
    players: list[dict[str, Any]],
    stats_by_player_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in players or []:
        pid = int(p.get("id"))
        s = stats_by_player_id.get(pid) or _empty_player_display_stats(pid)
        row = {
            "player_id": pid,
            "jersey_number": str(p.get("jersey_number") or ""),
            "name": str(p.get("name") or ""),
            "position": str(p.get("position") or ""),
        }
        for k, _label in PLAYER_STATS_DISPLAY_COLUMNS:
            row[k] = s.get(k)
        denoms = s.get("_per_game_denoms")
        if isinstance(denoms, dict):
            row["_per_game_denoms"] = denoms
        rows.append(row)
    return rows


def compute_recent_player_totals_from_rows(
    *,
    schedule_games: list[dict[str, Any]],
    player_stats_rows: list[dict[str, Any]],
    n: int,
) -> dict[int, dict[str, Any]]:
    """
    Compute per-player totals using each player's most recent N games (as defined by `schedule_games` order).
    """
    n_i = max(1, min(10, int(n)))
    order_idx: dict[int, int] = {}
    for idx, g in enumerate(schedule_games or []):
        try:
            order_idx[int(g.get("id"))] = int(idx)
        except Exception:
            continue

    rows_by_player: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for r in player_stats_rows or []:
        try:
            gid = int(r.get("game_id"))
            pid = int(r.get("player_id"))
        except Exception:
            continue
        idx = order_idx.get(gid)
        if idx is None:
            continue
        rows_by_player.setdefault(pid, []).append((idx, r))

    out: dict[int, dict[str, Any]] = {}
    for pid, items in rows_by_player.items():
        items.sort(key=lambda t: t[0], reverse=True)
        chosen = items[:n_i]
        sums: dict[str, Any] = {"player_id": int(pid), "gp": len(chosen)}
        present: dict[str, int] = {}
        ppg_games = 0
        for k in PLAYER_STATS_SUM_KEYS:
            total = 0
            n_present = 0
            for _idx, rr in chosen:
                if rr.get(k) is not None:
                    n_present += 1
                total += _int0(rr.get(k))
            sums[k] = total
            present[k] = int(n_present)
        for _idx, rr in chosen:
            if rr.get("goals") is not None and rr.get("assists") is not None:
                ppg_games += 1
        denoms = {
            "ppg": int(ppg_games),
            "toi_seconds_per_game": int(present.get("toi_seconds", 0) or 0),
            "shifts_per_game": int(present.get("shifts", 0) or 0),
            "shots_per_game": int(present.get("shots", 0) or 0),
            "sog_per_game": int(present.get("sog", 0) or 0),
            "expected_goals_per_game": int(present.get("expected_goals", 0) or 0),
            "plus_minus_per_game": int(present.get("plus_minus", 0) or 0),
            "gf_per_game": int(present.get("gf_counted", 0) or 0),
            "ga_per_game": int(present.get("ga_counted", 0) or 0),
            "giveaways_per_game": int(present.get("giveaways", 0) or 0),
            "takeaways_per_game": int(present.get("takeaways", 0) or 0),
            "completed_passes_per_game": int(present.get("completed_passes", 0) or 0),
            "turnovers_forced_per_game": int(present.get("turnovers_forced", 0) or 0),
            "created_turnovers_per_game": int(present.get("created_turnovers", 0) or 0),
            "controlled_entry_for_per_game": int(present.get("controlled_entry_for", 0) or 0),
            "controlled_entry_against_per_game": int(
                present.get("controlled_entry_against", 0) or 0
            ),
            "controlled_exit_for_per_game": int(present.get("controlled_exit_for", 0) or 0),
            "controlled_exit_against_per_game": int(present.get("controlled_exit_against", 0) or 0),
            "pim_per_game": int(present.get("pim", 0) or 0),
            "hits_per_game": int(present.get("hits", 0) or 0),
            "blocks_per_game": int(present.get("blocks", 0) or 0),
        }
        out[int(pid)] = compute_player_display_stats(sums, per_game_denoms=denoms)
    return out


def _dedupe_preserve_str(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for it in items or []:
        s = str(it or "").strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _game_type_label_for_row(game_row: dict[str, Any]) -> str:
    gt = str(game_row.get("game_type_name") or "").strip()
    div_name = (
        game_row.get("division_name")
        if game_row.get("division_name") is not None
        else game_row.get("league_division_name")
    )
    if not gt and is_external_division_name(div_name):
        return "Tournament"
    return gt or "Unknown"


def _game_has_recorded_result(game_row: dict[str, Any]) -> bool:
    return (
        (game_row.get("team1_score") is not None)
        or (game_row.get("team2_score") is not None)
        or bool(game_row.get("is_final"))
    )


def _norm_division_name_for_compare(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().casefold()


def _game_goal_diff(game_row: dict[str, Any]) -> Optional[int]:
    try:
        s1 = game_row.get("team1_score")
        s2 = game_row.get("team2_score")
        if s1 is None or s2 is None:
            return None
        return abs(int(s1) - int(s2))
    except Exception:
        return None


def game_exclusion_reason_for_stats(
    game_row: dict[str, Any],
    *,
    team_id: Optional[int] = None,
    league_name: Optional[str] = None,
) -> Optional[str]:
    """
    Return a short reason when a game should be excluded from team/player statistics.

    Rules:
      - Tournament games with goal differential >= 7 are excluded (blowouts can heavily skew rates/averages).
      - CAHA preseason games are excluded for a given team if the game was played in a different league
        division than the team's current CAHA league division (relegation/evaluation games are not representative).
    """
    gt_label = str(_game_type_label_for_row(game_row) or "").strip()
    gt_cf = gt_label.casefold()

    if gt_cf.startswith("tournament"):
        gd = _game_goal_diff(game_row)
        if gd is not None and int(gd) >= 7:
            return "Tournament blowout (goal differential ≥ 7)"

    league_cf = str(league_name or "").strip().casefold()
    if league_cf == "caha" and "preseason" in gt_cf:
        if team_id is None:
            return None
        try:
            tid = int(team_id)
        except Exception:
            return None

        game_div = (
            game_row.get("division_name")
            if game_row.get("division_name") is not None
            else game_row.get("league_division_name")
        )
        game_div_cf = _norm_division_name_for_compare(game_div)

        if not game_div_cf or is_external_division_name(game_div):
            return None

        t1 = game_row.get("team1_id")
        t2 = game_row.get("team2_id")
        team_div = None
        if t1 is not None and int(t1) == tid:
            team_div = game_row.get("team1_league_division_name")
        elif t2 is not None and int(t2) == tid:
            team_div = game_row.get("team2_league_division_name")
        team_div_cf = _norm_division_name_for_compare(team_div)

        if team_div_cf and not is_external_division_name(team_div) and team_div_cf != game_div_cf:
            return "CAHA preseason (played in a different division than the team's final placement)"

    return None


def game_is_eligible_for_stats(
    game_row: dict[str, Any],
    *,
    team_id: Optional[int] = None,
    league_name: Optional[str] = None,
) -> bool:
    return (
        game_exclusion_reason_for_stats(game_row, team_id=team_id, league_name=league_name) is None
    )


def _parse_selected_game_type_labels(
    *,
    available: list[str],
    args: Any,
) -> Optional[set[str]]:
    """
    Parse a game type filter from request args. Returns None to represent "no filtering" (all types).
    """
    avail = _dedupe_preserve_str(list(available or []))
    if not avail:
        return None
    raw: list[str] = []
    try:
        raw.extend(list(args.getlist("gt") or []))
    except Exception:
        pass
    try:
        v = args.get("gt")
        if v and isinstance(v, str) and "," in v:
            raw.extend([p.strip() for p in v.split(",") if p.strip()])
    except Exception:
        pass
    selected = _dedupe_preserve_str(raw)
    if not selected:
        return None
    avail_map = {a.casefold(): a for a in avail}
    chosen: set[str] = set()
    for s in selected:
        v = avail_map.get(s.casefold())
        if v:
            chosen.add(v)
    if not chosen or len(chosen) == len(avail):
        return None
    return chosen


def _aggregate_player_totals_from_rows(
    *,
    player_stats_rows: list[dict[str, Any]],
    allowed_game_ids: set[int],
) -> dict[int, dict[str, Any]]:
    sums_by_pid: dict[int, dict[str, Any]] = {}
    gp_by_pid: dict[int, int] = {}
    present_counts_by_pid: dict[int, dict[str, int]] = {}
    ppg_games_by_pid: dict[int, int] = {}
    for r in player_stats_rows or []:
        if not isinstance(r, dict):
            continue
        try:
            gid = int(r.get("game_id"))
            pid = int(r.get("player_id"))
        except Exception:
            continue
        if gid not in allowed_game_ids:
            continue
        gp_by_pid[pid] = gp_by_pid.get(pid, 0) + 1
        if r.get("goals") is not None and r.get("assists") is not None:
            ppg_games_by_pid[pid] = ppg_games_by_pid.get(pid, 0) + 1
        acc = sums_by_pid.setdefault(pid, {"player_id": int(pid)})
        for k in PLAYER_STATS_SUM_KEYS:
            if r.get(k) is not None:
                per = present_counts_by_pid.setdefault(pid, {})
                per[k] = per.get(k, 0) + 1
            acc[k] = _int0(acc.get(k)) + _int0(r.get(k))
    out: dict[int, dict[str, Any]] = {}
    for pid, base in sums_by_pid.items():
        base["gp"] = int(gp_by_pid.get(pid, 0))
        present = present_counts_by_pid.get(int(pid)) or {}
        denoms = {
            "ppg": int(ppg_games_by_pid.get(int(pid), 0) or 0),
            "toi_seconds_per_game": int(present.get("toi_seconds", 0) or 0),
            "shifts_per_game": int(present.get("shifts", 0) or 0),
            "shots_per_game": int(present.get("shots", 0) or 0),
            "sog_per_game": int(present.get("sog", 0) or 0),
            "expected_goals_per_game": int(present.get("expected_goals", 0) or 0),
            "plus_minus_per_game": int(present.get("plus_minus", 0) or 0),
            "gf_per_game": int(present.get("gf_counted", 0) or 0),
            "ga_per_game": int(present.get("ga_counted", 0) or 0),
            "giveaways_per_game": int(present.get("giveaways", 0) or 0),
            "takeaways_per_game": int(present.get("takeaways", 0) or 0),
            "completed_passes_per_game": int(present.get("completed_passes", 0) or 0),
            "turnovers_forced_per_game": int(present.get("turnovers_forced", 0) or 0),
            "created_turnovers_per_game": int(present.get("created_turnovers", 0) or 0),
            "controlled_entry_for_per_game": int(present.get("controlled_entry_for", 0) or 0),
            "controlled_entry_against_per_game": int(
                present.get("controlled_entry_against", 0) or 0
            ),
            "controlled_exit_for_per_game": int(present.get("controlled_exit_for", 0) or 0),
            "controlled_exit_against_per_game": int(present.get("controlled_exit_against", 0) or 0),
            "pim_per_game": int(present.get("pim", 0) or 0),
            "hits_per_game": int(present.get("hits", 0) or 0),
            "blocks_per_game": int(present.get("blocks", 0) or 0),
        }
        out[int(pid)] = compute_player_display_stats(dict(base), per_game_denoms=denoms)
    return out


def _player_stats_required_sum_keys_for_display_key(col_key: str) -> tuple[str, ...]:
    k = str(col_key or "").strip()
    if not k:
        return tuple()
    if k == "gf_per_game":
        return ("gf_counted",)
    if k == "ga_per_game":
        return ("ga_counted",)
    if k == "pseudo_cf_pct":
        return ("shots_for_on_ice", "shots_against_on_ice")
    if k in set(PLAYER_STATS_SUM_KEYS):
        return (k,)
    if k == "gp":
        return tuple()
    if k in {"points", "ppg"}:
        return ("goals", "assists")
    if k.endswith("_per_game"):
        base = k[: -len("_per_game")]
        if base in set(PLAYER_STATS_SUM_KEYS):
            return (base,)
    if k == "expected_goals_per_sog":
        return ("expected_goals", "sog")
    return tuple()


def _compute_team_player_stats_coverage(
    *,
    player_stats_rows: list[dict[str, Any]],
    eligible_game_ids: list[int],
) -> tuple[dict[str, int], int]:
    """
    Returns (coverage_counts_by_display_key, total_eligible_games).
    """
    eligible_set = {int(gid) for gid in (eligible_game_ids or [])}
    total = len(eligible_set)
    if total <= 0:
        return {}, 0

    has_any_ps: set[int] = set()
    has_key_by_game: dict[int, set[str]] = {}
    for r in player_stats_rows or []:
        if not isinstance(r, dict):
            continue
        try:
            gid = int(r.get("game_id"))
        except Exception:
            continue
        if gid not in eligible_set:
            continue
        has_any_ps.add(gid)
        keys = has_key_by_game.setdefault(gid, set())
        for sk in PLAYER_STATS_SUM_KEYS:
            if r.get(sk) is not None:
                keys.add(sk)

    counts: dict[str, int] = {"gp": len(has_any_ps)}
    for display_key, _label in PLAYER_STATS_DISPLAY_COLUMNS:
        dk = str(display_key)
        if dk in counts:
            continue
        req = _player_stats_required_sum_keys_for_display_key(dk)
        if not req:
            counts[dk] = len(has_any_ps)
            continue
        n = 0
        for gid in eligible_set:
            present = has_key_by_game.get(gid) or set()
            if all(rk in present for rk in req):
                n += 1
        counts[dk] = int(n)
    return counts, total


def _annotate_player_stats_column_labels(
    *,
    columns: list[tuple[str, str]],
    coverage_counts: dict[str, int],
    total_games: int,
) -> list[tuple[str, str]]:
    # Backwards-compatible wrapper: keep older call sites working.
    out: list[tuple[str, str]] = []
    for c in _player_stats_columns_with_coverage(
        columns=columns, coverage_counts=coverage_counts, total_games=total_games
    ):
        out.append((str(c["key"]), str(c["label"])))
    return out


def _player_stats_columns_with_coverage(
    *,
    columns: list[tuple[str, str]],
    coverage_counts: dict[str, int],
    total_games: int,
) -> list[dict[str, Any]]:
    """
    Return columns as dicts with optional coverage sublabel info for UI rendering.
    """
    out: list[dict[str, Any]] = []
    for k, label in columns or []:
        key = str(k)
        n = coverage_counts.get(key, total_games)
        show = bool(total_games > 0 and n != total_games)
        out.append(
            {
                "key": key,
                "label": str(label),
                "n_games": int(n) if n is not None else 0,
                "total_games": int(total_games) if total_games is not None else 0,
                "show_count": show,
            }
        )
    return out


def canon_event_source_key(raw: Any) -> str:
    """
    Return a canonical event source key used for ordering and display.

    Keys are intentionally coarse so we don't leak per-game/per-import labels into the UI.
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    sl = s.casefold()
    if sl in {"timetoscore", "t2s", "tts"}:
        return "timetoscore"
    if sl == "primary" or sl.startswith("parse_stats_inputs"):
        return "primary"
    if sl.startswith("parse_shift_spreadsheet"):
        return "primary"
    if sl == "shift_package":
        return "shift_package"
    if sl == "long":
        return "long"
    if sl == "goals":
        return "goals"
    return ""


def event_source_rank(raw: Any) -> int:
    """
    Rank event sources by preference for de-duping/selection.

    Lower is better.
    """
    k = canon_event_source_key(raw)
    if k == "timetoscore":
        return 0
    if k in {"primary", "shift_package"}:
        return 1
    if k == "long":
        return 2
    if k == "goals":
        return 3
    return 9


def _canon_source_label_for_ui(raw: Any) -> str:
    k = canon_event_source_key(raw)
    if k == "timetoscore":
        return "TimeToScore"
    if k == "long":
        return "Long"
    if k == "primary":
        return "Primary"
    if k == "shift_package":
        return "Shift Package"
    if k == "goals":
        return "Goals"
    return ""


def _compute_team_player_stats_sources(
    db_conn,
    *,
    eligible_game_ids: list[int],
) -> list[str]:
    gids = [int(g) for g in (eligible_game_ids or []) if int(g) > 0]
    if not gids:
        return []
    out: list[str] = []
    seen: set[str] = set()
    del db_conn
    try:
        _django_orm, m = _orm_modules()
    except Exception:
        return out

    def _add(src: Any) -> None:
        s = _canon_source_label_for_ui(src)
        if not s:
            return
        k = s.casefold()
        if k in seen:
            return
        seen.add(k)
        out.append(s)

    # Prefer scanning event row sources (multi-valued Source column semantics).
    try:
        for chunk in _django_orm.iter_chunks(gids, 200):
            sources = list(
                m.HkyGameEventRow.objects.filter(game_id__in=chunk).values_list("source", flat=True)
            )
            for src in sources:
                s = str(src or "").strip()
                if not s:
                    continue
                for tok in re.split(r"[,+;/\s]+", s):
                    _add(tok)
    except Exception:
        pass
    return out


def sort_player_stats_rows(
    rows: list[dict[str, Any]],
    *,
    sort_key: str,
    sort_dir: str,
) -> list[dict[str, Any]]:
    key = str(sort_key or "").strip()
    direction = str(sort_dir or "").strip().lower()
    if direction not in {"asc", "desc"}:
        direction = "desc"

    def _val(r: dict[str, Any]) -> Any:
        if key in {"jersey", "jersey_number", "#"}:
            try:
                return int(str(r.get("jersey_number") or "0").strip() or "0")
            except Exception:
                return 0
        if key in {"name", "player"}:
            return str(r.get("name") or "").lower()
        if key == "position":
            return str(r.get("position") or "").lower()
        v = r.get(key)
        if v is None or v == "":
            return float("-inf") if direction == "desc" else float("inf")
        if isinstance(v, (int, float)):
            return v
        try:
            return float(str(v))
        except Exception:
            return str(v).lower()

    reverse = direction == "desc"

    # Stable tie-breakers (points desc, then name).
    def _tiebreak(r: dict[str, Any]) -> tuple:
        pts = r.get("points")
        try:
            pts_v = float(pts) if pts is not None else 0.0
        except Exception:
            pts_v = 0.0
        return (-pts_v, str(r.get("name") or "").lower())

    return sorted(list(rows or []), key=lambda r: (_val(r), _tiebreak(r)), reverse=reverse)


def sort_players_table_default(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Default stable sort order for the team-page Players table:
      - name ascending
      - assists descending
      - goals descending
      - points descending
    """
    out = list(rows or [])

    def _n(r: dict[str, Any], k: str) -> int:
        return _int0(r.get(k))

    # Stable sort: apply the least significant key first and the most significant key last.
    out.sort(key=lambda r: str(r.get("name") or "").lower())
    out.sort(key=lambda r: _n(r, "assists"), reverse=True)
    out.sort(key=lambda r: _n(r, "goals"), reverse=True)
    out.sort(key=lambda r: _n(r, "points"), reverse=True)
    return out


def aggregate_players_totals(db_conn, team_id: int, user_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Count, F, IntegerField, Sum
    from django.db.models.functions import Coalesce
    from django.db.models.functions import Abs

    # Exclude outlier games from aggregated totals (outliers are allowed for MHR-like ratings only).
    from django.db.models import Q

    eligible_game_ids: list[int] = []
    for r in (
        m.HkyGame.objects.filter(user_id=int(user_id))
        .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
        .select_related("game_type")
        .values("id", "team1_score", "team2_score", "game_type__name")
    ):
        try:
            gid = int(r.get("id") or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        row = dict(r)
        row["game_type_name"] = row.get("game_type__name")
        if not game_is_eligible_for_stats(row, team_id=int(team_id), league_name=None):
            continue
        eligible_game_ids.append(int(gid))

    if not eligible_game_ids:
        return {}

    game_ids = [int(x) for x in eligible_game_ids]

    def _parse_pim_minutes(details: Any) -> int:
        s = str(details or "").strip()
        if not s:
            return 0
        m0 = re.search(r"(\\d+)\\s*(?:m|min)\\b", s, flags=re.IGNORECASE)
        if not m0:
            return 0
        try:
            return max(0, int(m0.group(1)))
        except Exception:
            return 0

    # Track which games count toward a player's GP.
    games_by_pid: dict[int, set[int]] = {}
    for pid, gid in m.HkyGamePlayer.objects.filter(
        game_id__in=game_ids, team_id=int(team_id)
    ).values_list("player_id", "game_id"):
        try:
            games_by_pid.setdefault(int(pid), set()).add(int(gid))
        except Exception:
            continue

    # Accumulate event-derived totals.
    sums_by_pid: dict[int, dict[str, Any]] = {}
    xg_by_pid: dict[int, int] = {}
    sog_by_pid: dict[int, int] = {}
    shot_by_pid: dict[int, int] = {}

    for r in (
        m.HkyGameEventRow.objects.filter(game_id__in=game_ids, player__team_id=int(team_id))
        .select_related("event_type")
        .values("game_id", "player_id", "event_type__key", "period", "details")
    ):
        pid_raw = r.get("player_id")
        gid_raw = r.get("game_id")
        if pid_raw is None or gid_raw is None:
            continue
        try:
            pid = int(pid_raw)
            gid = int(gid_raw)
        except Exception:
            continue
        if pid <= 0 or gid <= 0:
            continue
        games_by_pid.setdefault(int(pid), set()).add(int(gid))

        sums = sums_by_pid.get(int(pid))
        if sums is None:
            sums = {"player_id": int(pid)}
            for k in PLAYER_STATS_DB_KEYS:
                sums[str(k)] = 0
            sums["toi_seconds"] = 0
            sums["shifts"] = 0
            sums_by_pid[int(pid)] = sums

        et_key = str(r.get("event_type__key") or "").strip().casefold()
        per_raw = r.get("period")
        try:
            per = int(per_raw) if per_raw is not None else None
        except Exception:
            per = None
        is_ot = bool(per is not None and int(per) >= 4)

        if et_key == "goal":
            sums["goals"] = int(sums.get("goals") or 0) + 1
            if is_ot:
                sums["ot_goals"] = int(sums.get("ot_goals") or 0) + 1
        elif et_key == "assist":
            sums["assists"] = int(sums.get("assists") or 0) + 1
            if is_ot:
                sums["ot_assists"] = int(sums.get("ot_assists") or 0) + 1
        elif et_key in {"xg", "expectedgoal"}:
            xg_by_pid[int(pid)] = int(xg_by_pid.get(int(pid), 0) or 0) + 1
        elif et_key in {"sog", "shotongoal"}:
            sog_by_pid[int(pid)] = int(sog_by_pid.get(int(pid), 0) or 0) + 1
        elif et_key == "shot":
            shot_by_pid[int(pid)] = int(shot_by_pid.get(int(pid), 0) or 0) + 1
        elif et_key == "penalty":
            sums["pim"] = int(sums.get("pim") or 0) + _parse_pim_minutes(r.get("details"))
        elif et_key == "completedpass":
            sums["completed_passes"] = int(sums.get("completed_passes") or 0) + 1
        elif et_key == "giveaway":
            sums["giveaways"] = int(sums.get("giveaways") or 0) + 1
        elif et_key == "turnoversforced":
            sums["turnovers_forced"] = int(sums.get("turnovers_forced") or 0) + 1
        elif et_key == "createdturnovers":
            sums["created_turnovers"] = int(sums.get("created_turnovers") or 0) + 1
        elif et_key == "takeaway":
            sums["takeaways"] = int(sums.get("takeaways") or 0) + 1

    # Compute xG/SOG/Shots with the implication:
    #   Goals ⊆ xG ⊆ SOG ⊆ Shots
    for pid, sums in sums_by_pid.items():
        goals = _int0(sums.get("goals"))
        xg = _int0(xg_by_pid.get(int(pid)))
        sog_extra = _int0(sog_by_pid.get(int(pid)))
        shot_extra = _int0(shot_by_pid.get(int(pid)))
        expected_goals = int(goals) + int(xg)
        sog_total = int(expected_goals) + int(sog_extra)
        shots_total = int(sog_total) + int(shot_extra)
        sums["expected_goals"] = int(expected_goals)
        sums["sog"] = int(sog_total)
        sums["shots"] = int(shots_total)

    base_by_pid: dict[int, dict[str, Any]] = {}
    for pid, gids in games_by_pid.items():
        sums = dict(sums_by_pid.get(int(pid)) or {"player_id": int(pid)})
        for k in PLAYER_STATS_DB_KEYS:
            sums.setdefault(str(k), 0)
        sums.setdefault("toi_seconds", 0)
        sums.setdefault("shifts", 0)
        sums["gp"] = int(len(gids))
        base_by_pid[int(pid)] = sums

    if base_by_pid:
        dur_expr = Abs(F("game_seconds_end") - F("game_seconds"))
        for srow in (
            m.HkyGameShiftRow.objects.filter(game_id__in=game_ids)
            .filter(player__team_id=int(team_id))
            .exclude(game_seconds__isnull=True)
            .exclude(game_seconds_end__isnull=True)
            .values("player_id")
            .annotate(toi=Coalesce(Sum(dur_expr, output_field=IntegerField()), 0), n=Count("id"))
        ):
            try:
                pid = int(srow.get("player_id") or 0)
            except Exception:
                continue
            if pid <= 0 or pid not in base_by_pid:
                continue
            base_by_pid[pid]["toi_seconds"] = int(srow.get("toi") or 0)
            base_by_pid[pid]["shifts"] = int(srow.get("n") or 0)

    out: dict[int, dict[str, Any]] = {}
    for pid, sums in base_by_pid.items():
        out[int(pid)] = compute_player_display_stats(dict(sums))
    return out


def aggregate_players_totals_league(db_conn, team_id: int, league_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Count, F, IntegerField, Q, Sum
    from django.db.models.functions import Coalesce
    from django.db.models.functions import Abs

    league_name = _get_league_name(None, int(league_id))
    league_team_div: dict[int, str] = {
        int(tid): str(dn or "").strip()
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    eligible_game_ids: list[int] = []
    for lg in (
        m.LeagueGame.objects.filter(league_id=int(league_id))
        .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
        .select_related("game", "game__game_type")
    ):
        g = lg.game
        t1_id = int(g.team1_id)
        t2_id = int(g.team2_id)
        row: dict[str, Any] = {
            "team1_id": t1_id,
            "team2_id": t2_id,
            "team1_score": g.team1_score,
            "team2_score": g.team2_score,
            "is_final": bool(getattr(g, "is_final", False)),
            "game_type_name": (g.game_type.name if g.game_type else None),
            "division_name": lg.division_name,
            "team1_league_division_name": league_team_div.get(t1_id),
            "team2_league_division_name": league_team_div.get(t2_id),
        }
        if not game_is_eligible_for_stats(row, team_id=int(team_id), league_name=league_name):
            continue
        eligible_game_ids.append(int(lg.game_id))

    if not eligible_game_ids:
        return {}

    game_ids = [int(x) for x in eligible_game_ids]

    def _parse_pim_minutes(details: Any) -> int:
        s = str(details or "").strip()
        if not s:
            return 0
        m0 = re.search(r"(\\d+)\\s*(?:m|min)\\b", s, flags=re.IGNORECASE)
        if not m0:
            return 0
        try:
            return max(0, int(m0.group(1)))
        except Exception:
            return 0

    games_by_pid: dict[int, set[int]] = {}
    for pid, gid in m.HkyGamePlayer.objects.filter(
        game_id__in=game_ids, team_id=int(team_id)
    ).values_list("player_id", "game_id"):
        try:
            games_by_pid.setdefault(int(pid), set()).add(int(gid))
        except Exception:
            continue

    sums_by_pid: dict[int, dict[str, Any]] = {}
    xg_by_pid: dict[int, int] = {}
    sog_by_pid: dict[int, int] = {}
    shot_by_pid: dict[int, int] = {}

    for r in (
        m.HkyGameEventRow.objects.filter(game_id__in=game_ids, player__team_id=int(team_id))
        .select_related("event_type")
        .values("game_id", "player_id", "event_type__key", "period", "details")
    ):
        pid_raw = r.get("player_id")
        gid_raw = r.get("game_id")
        if pid_raw is None or gid_raw is None:
            continue
        try:
            pid = int(pid_raw)
            gid = int(gid_raw)
        except Exception:
            continue
        if pid <= 0 or gid <= 0:
            continue
        games_by_pid.setdefault(int(pid), set()).add(int(gid))

        sums = sums_by_pid.get(int(pid))
        if sums is None:
            sums = {"player_id": int(pid)}
            for k in PLAYER_STATS_DB_KEYS:
                sums[str(k)] = 0
            sums["toi_seconds"] = 0
            sums["shifts"] = 0
            sums_by_pid[int(pid)] = sums

        et_key = str(r.get("event_type__key") or "").strip().casefold()
        per_raw = r.get("period")
        try:
            per = int(per_raw) if per_raw is not None else None
        except Exception:
            per = None
        is_ot = bool(per is not None and int(per) >= 4)

        if et_key == "goal":
            sums["goals"] = int(sums.get("goals") or 0) + 1
            if is_ot:
                sums["ot_goals"] = int(sums.get("ot_goals") or 0) + 1
        elif et_key == "assist":
            sums["assists"] = int(sums.get("assists") or 0) + 1
            if is_ot:
                sums["ot_assists"] = int(sums.get("ot_assists") or 0) + 1
        elif et_key in {"xg", "expectedgoal"}:
            xg_by_pid[int(pid)] = int(xg_by_pid.get(int(pid), 0) or 0) + 1
        elif et_key in {"sog", "shotongoal"}:
            sog_by_pid[int(pid)] = int(sog_by_pid.get(int(pid), 0) or 0) + 1
        elif et_key == "shot":
            shot_by_pid[int(pid)] = int(shot_by_pid.get(int(pid), 0) or 0) + 1
        elif et_key == "penalty":
            sums["pim"] = int(sums.get("pim") or 0) + _parse_pim_minutes(r.get("details"))
        elif et_key == "completedpass":
            sums["completed_passes"] = int(sums.get("completed_passes") or 0) + 1
        elif et_key == "giveaway":
            sums["giveaways"] = int(sums.get("giveaways") or 0) + 1
        elif et_key == "turnoversforced":
            sums["turnovers_forced"] = int(sums.get("turnovers_forced") or 0) + 1
        elif et_key == "createdturnovers":
            sums["created_turnovers"] = int(sums.get("created_turnovers") or 0) + 1
        elif et_key == "takeaway":
            sums["takeaways"] = int(sums.get("takeaways") or 0) + 1

    for pid, sums in sums_by_pid.items():
        goals = _int0(sums.get("goals"))
        xg = _int0(xg_by_pid.get(int(pid)))
        sog_extra = _int0(sog_by_pid.get(int(pid)))
        shot_extra = _int0(shot_by_pid.get(int(pid)))
        expected_goals = int(goals) + int(xg)
        sog_total = int(expected_goals) + int(sog_extra)
        shots_total = int(sog_total) + int(shot_extra)
        sums["expected_goals"] = int(expected_goals)
        sums["sog"] = int(sog_total)
        sums["shots"] = int(shots_total)

    base_by_pid: dict[int, dict[str, Any]] = {}
    for pid, gids in games_by_pid.items():
        sums = dict(sums_by_pid.get(int(pid)) or {"player_id": int(pid)})
        for k in PLAYER_STATS_DB_KEYS:
            sums.setdefault(str(k), 0)
        sums.setdefault("toi_seconds", 0)
        sums.setdefault("shifts", 0)
        sums["gp"] = int(len(gids))
        base_by_pid[int(pid)] = sums

    if base_by_pid:
        if eligible_game_ids:
            dur_expr = Abs(F("game_seconds_end") - F("game_seconds"))
            for srow in (
                m.HkyGameShiftRow.objects.filter(game_id__in=[int(x) for x in eligible_game_ids])
                .filter(player__team_id=int(team_id))
                .exclude(game_seconds__isnull=True)
                .exclude(game_seconds_end__isnull=True)
                .values("player_id")
                .annotate(
                    toi=Coalesce(Sum(dur_expr, output_field=IntegerField()), 0), n=Count("id")
                )
            ):
                try:
                    pid = int(srow.get("player_id") or 0)
                except Exception:
                    continue
                if pid <= 0 or pid not in base_by_pid:
                    continue
                base_by_pid[pid]["toi_seconds"] = int(srow.get("toi") or 0)
                base_by_pid[pid]["shifts"] = int(srow.get("n") or 0)

    out: dict[int, dict[str, Any]] = {}
    for pid, sums in base_by_pid.items():
        out[int(pid)] = compute_player_display_stats(dict(sums))
    return out

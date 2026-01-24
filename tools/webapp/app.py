#!/usr/bin/env python3
"""
Public facade for the HockeyMOM webapp.

This module intentionally stays small; the implementation lives in `tools/webapp/core/`.
"""

# ruff: noqa: F401

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
CONFIG_PATH = BASE_DIR / "config.json"

WATCH_ROOT = os.environ.get("HM_WATCH_ROOT", "/data/incoming")

# Allow importing the webapp as either:
#   - `import tools.webapp.app` (namespace package)
#   - importlib file-loader tests (`spec_from_file_location(..., tools/webapp/app.py)`)
_repo_root = str(BASE_DIR.parent.parent)
_base_dir_str = str(BASE_DIR)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _base_dir_str not in sys.path:
    sys.path.insert(0, _base_dir_str)

# Lazy import for pymysql to allow importing module without DB installed (e.g., tests)
try:
    import pymysql  # type: ignore
except Exception:  # pragma: no cover
    pymysql = None  # type: ignore


from core.orm import (  # noqa: E402
    LEAGUE_PAGE_VIEW_KIND_EVENT_CLIP,
    LEAGUE_PAGE_VIEW_KIND_GAME,
    LEAGUE_PAGE_VIEW_KIND_PLAYER_EVENTS,
    LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
    LEAGUE_PAGE_VIEW_KIND_TEAM,
    LEAGUE_PAGE_VIEW_KIND_TEAMS,
    LEAGUE_PAGE_VIEW_KINDS,
    _canon_league_page_view_kind_entity,
    _get_league_name,
    _get_league_owner_user_id,
    _get_league_page_view_baseline_count,
    _get_league_page_view_count,
    _orm_modules,
    _record_league_page_view,
    _set_league_page_view_baseline_count,
    init_db,
)
from core.secret import load_or_create_app_secret  # noqa: E402
from core.system import (  # noqa: E402
    create_game,
    create_user,
    get_user_by_email,
    get_user_video_clip_len_s,
    read_dirwatch_state,
    send_email,
)
from core.utils import parse_dt_or_none, to_dt  # noqa: E402

from core.hockey import (  # noqa: E402
    _extract_game_stats_note_from_notes,
    _extract_game_video_url_from_notes,
    _extract_timetoscore_game_id_from_notes,
    _league_game_is_cross_division_non_external,
    _league_game_is_cross_division_non_external_row,
    _safe_return_to_url,
    _sanitize_http_url,
    create_hky_game,
    create_team,
    division_sort_key,
    ensure_external_team,
    get_team,
    is_external_division_name,
    parse_age_from_division_name,
    parse_level_from_division_name,
    recompute_league_mhr_ratings,
    sort_games_schedule_order,
)
from core.events import (  # noqa: E402
    _event_table_sort_key,
    compute_game_event_stats_by_side,
    compute_goalie_stats_for_game,
    compute_goalie_stats_for_team_games,
    compute_team_scoring_by_period_from_events,
    enrich_timetoscore_goals_with_long_video_times,
    enrich_timetoscore_penalties_with_video_times,
    filter_events_csv_drop_event_types,
    filter_events_headers_drop_empty_on_ice_split,
    filter_events_rows_prefer_timetoscore_for_goal_assist,
    filter_game_stats_for_display,
    filter_single_game_player_stats_csv,
    merge_events_csv_prefer_timetoscore,
    normalize_event_type_key,
    normalize_events_video_time_for_display,
    normalize_game_events_csv,
    normalize_video_time_and_seconds,
    parse_events_csv,
    sanitize_player_stats_csv_for_storage,
    sort_event_dicts_for_table_display,
    sort_events_rows_default,
    summarize_event_sources,
    to_csv_text,
)
from core.league_admin import reset_league_data  # noqa: E402
from core.team_stats import (  # noqa: E402
    compute_team_stats,
    compute_team_stats_league,
    division_seed_team_id,
    division_standings_team_ids,
    sort_key_team_standings,
)
from core.seed_placeholders import (  # noqa: E402
    SEED_PLACEHOLDER_TEAM_NAME,
    ensure_seed_placeholder_team_for_import,
    is_seed_placeholder_name,
    parse_seed_placeholder_name,
)
from core.player_stats import (  # noqa: E402
    GAME_PLAYER_STATS_COLUMNS,
    GAME_PLAYER_STATS_DISPLAY_KEYS,
    OT_ONLY_PLAYER_STATS_KEYS,
    PLAYER_STATS_DB_KEYS,
    PLAYER_STATS_DISPLAY_COLUMNS,
    PLAYER_STATS_SUM_KEYS,
    _aggregate_player_totals_from_rows,
    _annotate_player_stats_column_labels,
    _build_game_player_stats_table_from_imported_csv,
    _canon_source_label_for_ui,
    _compute_team_player_stats_coverage,
    _compute_team_player_stats_sources,
    _int0,
    _is_blank_stat,
    _is_goalie_position,
    _is_zero_or_blank_stat,
    _map_imported_shift_stats_to_player_ids,
    _merge_stat_values,
    _normalize_column_id,
    _normalize_header_for_lookup,
    _norm_division_name_for_compare,
    _parse_int_from_cell_text,
    _parse_selected_game_type_labels,
    _player_stats_columns_with_coverage,
    _player_stats_required_sum_keys_for_display_key,
    _rate_or_none,
    _classify_coach_position,
    _classify_roster_role,
    _dedupe_preserve_str,
    _empty_player_display_stats,
    _game_goal_diff,
    _game_has_recorded_result,
    _game_type_label_for_row,
    aggregate_players_totals,
    aggregate_players_totals_league,
    build_game_player_stats_display_columns,
    build_game_player_stats_table,
    build_player_stats_table_rows,
    canon_event_source_key,
    compute_player_display_stats,
    compute_recent_player_totals_from_rows,
    event_source_rank,
    filter_player_stats_display_columns_for_rows,
    game_exclusion_reason_for_stats,
    game_is_eligible_for_stats,
    sort_player_stats_rows,
    sort_players_table_default,
    split_players_and_coaches,
    split_roster,
)
from core.shift_stats import (  # noqa: E402
    _int_or_none,
    format_seconds_to_mmss_or_hhmmss,
    normalize_jersey_number,
    normalize_player_name,
    normalize_player_name_no_middle,
    parse_duration_seconds,
    parse_shift_rows_csv,
    parse_shift_stats_game_stats_csv,
    parse_shift_stats_player_stats_csv,
    strip_jersey_from_player_name,
)


def _load_or_create_app_secret() -> str:
    return load_or_create_app_secret(instance_dir=INSTANCE_DIR, config_path=CONFIG_PATH)


APP_SECRET = _load_or_create_app_secret()


def save_team_logo(file_storage, team_id: int) -> Path:
    from core.hockey import save_team_logo as _save_team_logo  # noqa: PLC0415

    return _save_team_logo(file_storage, int(team_id), instance_dir=INSTANCE_DIR)

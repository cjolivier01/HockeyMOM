from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any, Optional

from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from werkzeug.security import check_password_hash, generate_password_hash


def _import_logic():
    try:
        from tools.webapp import app as logic  # type: ignore

        return logic
    except Exception:  # pragma: no cover
        import app as logic  # type: ignore

        return logic


logic = _import_logic()

logger = logging.getLogger(__name__)


def _orm_modules():
    return logic._orm_modules()


def _session_user_id(request: HttpRequest) -> int:
    try:
        return int(request.session.get("user_id") or 0)
    except Exception:
        return 0


def _require_login(request: HttpRequest) -> Optional[HttpResponse]:
    if not _session_user_id(request):
        return redirect("/login")
    return None


def _is_league_admin(league_id: int, user_id: int) -> bool:
    _django_orm, m = _orm_modules()
    del _django_orm
    if m.League.objects.filter(id=int(league_id), owner_user_id=int(user_id)).exists():
        return True
    return m.LeagueMember.objects.filter(
        league_id=int(league_id),
        user_id=int(user_id),
        role__in=["admin", "owner"],
    ).exists()


def _is_public_league(league_id: int) -> Optional[dict[str, Any]]:
    _django_orm, m = _orm_modules()
    return (
        m.League.objects.filter(id=int(league_id), is_public=True)
        .values("id", "name", "owner_user_id")
        .first()
    )


def _league_show_goalie_stats(league_id: int) -> bool:
    _django_orm, m = _orm_modules()
    try:
        v = (
            m.League.objects.filter(id=int(league_id))
            .values_list("show_goalie_stats", flat=True)
            .first()
        )
        return bool(v)
    except Exception:
        return False


def _league_show_shift_data(league_id: int) -> bool:
    _django_orm, m = _orm_modules()
    try:
        v = (
            m.League.objects.filter(id=int(league_id))
            .values_list("show_shift_data", flat=True)
            .first()
        )
        return bool(v)
    except Exception:
        return False


def _safe_file_response(path: Path, *, as_attachment: bool = False) -> FileResponse:
    if not path.exists() or not path.is_file():
        raise Http404
    resp = FileResponse(path.open("rb"))
    if as_attachment:
        resp["Content-Disposition"] = f'attachment; filename="{path.name}"'
    return resp


def _json_body(request: HttpRequest) -> dict[str, Any]:
    try:
        raw = request.body or b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def _event_type_key(raw: Any) -> str:
    """
    Stable key for event types across sources (e.g. "Expected Goal" vs "ExpectedGoal").
    """
    s = str(raw or "").strip().casefold()
    return re.sub(r"[^a-z0-9]+", "", s)


def _normalize_player_name_no_middle(raw: Any) -> str:
    parts = [p for p in str(raw or "").strip().split() if p]
    if len(parts) >= 3:
        kept: list[str] = []
        for idx, token in enumerate(parts):
            t = token.strip(".")
            if 0 < idx < (len(parts) - 1) and len(t) == 1:
                continue
            kept.append(token)
        parts = kept or parts
    return logic.normalize_player_name(" ".join(parts))


def _strip_jersey_from_player_name(raw: Any, jersey_number: Optional[str]) -> str:
    name = str(raw or "").strip()
    if not name:
        return ""
    jersey_norm = logic.normalize_jersey_number(jersey_number) if jersey_number else None
    if jersey_norm:
        tail = re.sub(rf"\s*[\(#]?\s*{re.escape(jersey_norm)}\s*\)?\s*$", "", name).strip()
        if tail:
            name = tail
        head = re.sub(rf"^#?\s*{re.escape(jersey_norm)}\s+", "", name).strip()
        if head:
            name = head
    return name


def _norm_ws(raw: Any) -> str:
    return " ".join(str(raw or "").strip().split())


_LEVEL_ORDER = {"AAA": 0, "AA": 1, "A": 2, "BB": 3, "B": 4}
_AGE_LABELS = {8: "8U (Mite)"}


def _age_label(age: int) -> str:
    if age in _AGE_LABELS:
        return _AGE_LABELS[int(age)]
    return f"{int(age)}U"


def _level_sort_key(level: str) -> tuple[int, str]:
    token = str(level or "").strip().upper()
    return (_LEVEL_ORDER.get(token, 99), token)


def _norm_division_name(raw: Any) -> str:
    return str(raw or "").strip() or "Unknown Division"


def _ival(raw: Any) -> Optional[int]:
    try:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _normalize_team_side(raw: Any) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (side_norm, side_label) where:
      - side_norm is "home" / "away" / None
      - side_label is "Home" / "Away" / None (for storage/display)
    """
    s = str(raw or "").strip()
    if not s:
        return None, None
    cf = s.casefold()
    if cf in {"home", "team1"}:
        return "home", "Home"
    if cf in {"away", "team2"}:
        return "away", "Away"
    return None, s


def _best_effort_side_from_row(row: dict[str, str]) -> tuple[Optional[str], Optional[str]]:
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
        if k in row and str(row.get(k) or "").strip():
            return _normalize_team_side(row.get(k))
    return None, None


def _events_group_rank(event_type_key: str) -> Optional[int]:
    """
    Rank within the shot implication chain used for de-duplication:
      Goal > ExpectedGoal > SOG > Shot
    """
    k = str(event_type_key or "").casefold()
    if k == "goal":
        return 3
    if k == "expectedgoal":
        return 2
    if k == "sog":
        return 1
    if k == "shot":
        return 0
    return None


def _compute_event_import_key(
    *,
    event_type_key: str,
    period: Optional[int],
    game_seconds: Optional[int],
    team_side_norm: Optional[str],
    jersey_norm: Optional[str],
    event_id: Optional[int],
    details: Optional[str],
    game_seconds_end: Optional[int],
) -> str:
    """
    Compute a stable import key for idempotent upserts.
    """
    et_key = str(event_type_key or "").casefold()

    # Some event types should be unique by team+time (not by player):
    #   - goal: prevent accidental duplicates (same team cannot score twice at the same game time)
    #   - goaliechange: one change per team per time (starting goalie uses period start time)
    jersey_part = str(jersey_norm or "")
    if et_key in {"goal", "goaliechange"}:
        jersey_part = ""

    parts = [
        str(event_type_key or ""),
        str(int(period)) if period is not None else "",
        str(int(game_seconds)) if game_seconds is not None else "",
        str(team_side_norm or ""),
        jersey_part,
    ]

    # Penalties may have multiple simultaneous infractions for the same player.
    if et_key in {"penalty", "penaltyexpired"}:
        parts.append(_norm_ws(details))
        parts.append(str(int(game_seconds_end)) if game_seconds_end is not None else "")
    # Goals/assists may be enriched later with `event_id` (e.g., from shift/xG data); keep the
    # import key stable so we update the existing row instead of creating a duplicate.
    elif et_key in {"goal", "assist"}:
        pass
    elif event_id is not None:
        parts.append(str(int(event_id)))

    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _synthesize_event_id(import_key: str) -> int:
    """
    Deterministic integer `event_id` for sources that don't provide one (notably penalties).

    Uses the already-stable `import_key` (sha1 hex) to derive a positive signed-32-bit integer
    so it can be stored in MySQL/Django `IntegerField`.
    """
    key = str(import_key or "").strip()
    try:
        n = int(key[:8], 16) & 0x7FFFFFFF
    except Exception:
        n = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
    return 1 if n == 0 else int(n)


def _upsert_game_event_rows_from_events_csv(
    *,
    game_id: int,
    events_csv: str,
    replace: bool,
    create_missing_players: bool = False,
    incoming_source_label: Optional[str] = None,
    prefer_incoming_for_event_types: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Parse a merged events CSV and upsert normalized per-row events into DB tables.
    """
    _django_orm, m = _orm_modules()
    from django.db import transaction

    now = dt.datetime.now()
    headers, rows = logic.parse_events_csv(str(events_csv or ""))
    if not headers or not rows:
        return {"ok": True, "created": 0, "updated": 0, "linked_players": 0}

    game = (
        m.HkyGame.objects.filter(id=int(game_id))
        .values("id", "team1_id", "team2_id", "user_id")
        .first()
    )
    if not game:
        return {"ok": False, "error": "game_not_found"}
    team1_id = int(game["team1_id"])
    team2_id = int(game["team2_id"])
    owner_user_id = int(game.get("user_id") or 0)

    # Preload players for jersey/name resolution.
    players = list(
        m.Player.objects.filter(team_id__in=[team1_id, team2_id]).values(
            "id", "team_id", "name", "jersey_number", "position"
        )
    )
    pid_to_team_id: dict[int, int] = {}
    jersey_to_pids: dict[tuple[int, str], list[int]] = {}
    name_to_pids: dict[tuple[int, str], list[int]] = {}
    for p in players:
        tid = int(p.get("team_id") or 0)
        pid = int(p.get("id") or 0)
        if not tid or not pid:
            continue
        pid_to_team_id[int(pid)] = int(tid)
        jn = logic.normalize_jersey_number(p.get("jersey_number"))
        if jn:
            jersey_to_pids.setdefault((tid, jn), []).append(pid)
        nn = logic.normalize_player_name(str(p.get("name") or ""))
        if nn:
            name_to_pids.setdefault((tid, nn), []).append(pid)

    # For some imports (e.g. TimeToScore) roster rows may be present even when jersey numbers
    # are missing. As a best-effort fallback, if a game+team has a single roster player
    # (`hky_game_players`) and an event row includes a jersey/name but can't be resolved to a
    # specific player, attribute it to that unique roster player.
    roster_pids_by_team: dict[int, list[int]] = {}
    try:
        for tid0, pid0 in m.HkyGamePlayer.objects.filter(
            game_id=int(game_id), team_id__in=[int(team1_id), int(team2_id)]
        ).values_list("team_id", "player_id"):
            try:
                tid_i = int(tid0 or 0)
                pid_i = int(pid0 or 0)
            except Exception:
                continue
            if tid_i > 0 and pid_i > 0:
                roster_pids_by_team.setdefault(int(tid_i), []).append(int(pid_i))
    except Exception:
        roster_pids_by_team = {}

    def _unique_game_roster_pid(tid: int) -> Optional[int]:
        uniq = sorted(set(int(x) for x in (roster_pids_by_team.get(int(tid)) or []) if int(x) > 0))
        if len(uniq) == 1:
            return int(uniq[0])
        return None

    # Ensure event types exist.
    needed_types: dict[str, str] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        raw_type = str(r.get("Event Type") or r.get("Event") or "").strip()
        if not raw_type:
            continue
        k = _event_type_key(raw_type)
        if k:
            needed_types.setdefault(k, raw_type)
    if needed_types:
        existing_keys = set(
            m.HkyEventType.objects.filter(key__in=list(needed_types.keys())).values_list(
                "key", flat=True
            )
        )
        to_create = [
            m.HkyEventType(key=k, name=needed_types[k], created_at=now)
            for k in needed_types
            if k not in existing_keys
        ]
        if to_create:
            m.HkyEventType.objects.bulk_create(to_create, ignore_conflicts=True)
    type_by_key: dict[str, int] = {
        str(k): int(i)
        for k, i in m.HkyEventType.objects.filter(key__in=list(needed_types.keys())).values_list(
            "key", "id"
        )
    }

    suppressed_keys: set[str] = set(
        str(k)
        for k in m.HkyGameEventSuppression.objects.filter(game_id=int(game_id)).values_list(
            "import_key", flat=True
        )
    )

    default_source = _norm_ws(incoming_source_label) if incoming_source_label else ""
    prefer_incoming_types = {
        str(k or "").casefold() for k in (prefer_incoming_for_event_types or set())
    }
    incoming_label_cf = str(incoming_source_label or "").strip().casefold()
    incoming_is_timetoscore = incoming_label_cf.startswith("timetoscore") or incoming_label_cf in {
        "timetoscore",
        "t2s",
        "tts",
    }

    parsed: list[dict[str, Any]] = []

    def _unique_pid(candidates: list[int]) -> Optional[int]:
        uniq = sorted(set(int(x) for x in (candidates or []) if int(x) > 0))
        if len(uniq) == 1:
            return int(uniq[0])
        return None

    def _resolve_unique_pid_any_team(
        *, jersey_norm: Optional[str], attributed_players: str
    ) -> Optional[int]:
        if jersey_norm:
            cands: list[int] = []
            for tid0 in (team1_id, team2_id):
                cands.extend(jersey_to_pids.get((int(tid0), str(jersey_norm)), []))
            pid = _unique_pid(cands)
            if pid is not None:
                return int(pid)
        if attributed_players:
            name_norm = logic.normalize_player_name(attributed_players)
            cands = []
            for tid0 in (team1_id, team2_id):
                cands.extend(name_to_pids.get((int(tid0), str(name_norm)), []))
            pid = _unique_pid(cands)
            if pid is not None:
                return int(pid)
        return None

    for r in rows:
        if not isinstance(r, dict):
            continue
        raw_type = str(r.get("Event Type") or r.get("Event") or "").strip()
        et_key = _event_type_key(raw_type)
        et_id = type_by_key.get(et_key)
        if not raw_type or not et_key or et_id is None:
            continue

        event_id = _ival(r.get("Event ID") or r.get("EventID"))
        period = _ival(r.get("Period"))
        game_seconds = _ival(r.get("Game Seconds") or r.get("GameSeconds"))
        if game_seconds is None:
            game_seconds = logic.parse_duration_seconds(
                r.get("Game Time") or r.get("GameTime") or r.get("Time")
            )
        game_seconds_end = _ival(r.get("Game Seconds End") or r.get("GameSecondsEnd"))
        video_time, video_seconds = logic.normalize_video_time_and_seconds(
            _norm_ws(r.get("Video Time") or r.get("VideoTime")),
            r.get("Video Seconds") or r.get("VideoSeconds") or r.get("Video S") or r.get("VideoS"),
        )

        side_norm, side_label = _best_effort_side_from_row(r)

        player_cell = _norm_ws(r.get("Player"))
        attributed_jerseys = _norm_ws(r.get("Attributed Jerseys") or r.get("AttributedJerseys"))
        if not attributed_jerseys and player_cell:
            attributed_jerseys = player_cell
        jersey_norm = logic.normalize_jersey_number(attributed_jerseys)
        attributed_players = _norm_ws(r.get("Attributed Players") or r.get("AttributedPlayers"))
        if not attributed_players and player_cell:
            # Common legacy encoding: "#9 Alice" or "9 Alice"
            attributed_players = re.sub(r"^#?\s*\d+\s*", "", str(player_cell)).strip()

        player_id_any: Optional[int] = None
        if side_norm == "home":
            team_id = team1_id
        elif side_norm == "away":
            team_id = team2_id
        else:
            team_id = None
            # For Goal events, we require a resolvable scoring team (Home/Away).
            # If Team Side is missing, attempt to infer it from the scorer jersey/name when possible.
            if et_key == "goal":
                player_id_any = _resolve_unique_pid_any_team(
                    jersey_norm=jersey_norm, attributed_players=attributed_players
                )
                inferred_team_id = (
                    int(pid_to_team_id.get(int(player_id_any)))
                    if player_id_any is not None and int(player_id_any) in pid_to_team_id
                    else None
                )
                if inferred_team_id == int(team1_id):
                    team_id = int(team1_id)
                    side_norm, side_label = "home", "Home"
                elif inferred_team_id == int(team2_id):
                    team_id = int(team2_id)
                    side_norm, side_label = "away", "Away"

        player_id: Optional[int] = None
        if player_id_any is not None and team_id is not None:
            player_id = int(player_id_any)
        if player_id is None and team_id is not None and jersey_norm:
            candidates = jersey_to_pids.get((int(team_id), str(jersey_norm)), [])
            if len(set(candidates)) == 1:
                player_id = int(list(set(candidates))[0])
        if player_id is None and team_id is not None and attributed_players:
            name_norm = logic.normalize_player_name(attributed_players)
            candidates = name_to_pids.get((int(team_id), name_norm), [])
            if len(set(candidates)) == 1:
                player_id = int(list(set(candidates))[0])

        if player_id is None and team_id is not None and (jersey_norm or attributed_players):
            roster_pid = _unique_game_roster_pid(int(team_id))
            if roster_pid is not None:
                player_id = int(roster_pid)

        if (
            player_id is None
            and create_missing_players
            and team_id is not None
            and attributed_players
        ):
            try:
                pid = _ensure_player_for_import(
                    int(owner_user_id),
                    int(team_id),
                    str(attributed_players),
                    jersey_norm,
                    None,
                    commit=False,
                )
                player_id = int(pid)
            except Exception:
                player_id = None

        details = _norm_ws(r.get("Details"))
        import_key = _compute_event_import_key(
            event_type_key=et_key,
            period=period,
            game_seconds=game_seconds,
            team_side_norm=side_norm,
            jersey_norm=jersey_norm,
            event_id=event_id,
            details=details,
            game_seconds_end=game_seconds_end,
        )
        if event_id is None and et_key in {"penalty", "penaltyexpired"}:
            event_id = _synthesize_event_id(import_key)
        if suppressed_keys and import_key in suppressed_keys:
            continue

        source = _norm_ws(r.get("Source"))
        if not source and default_source:
            source = default_source

        game_time_txt = _norm_ws(r.get("Game Time") or r.get("GameTime") or r.get("Time"))
        if not game_time_txt and game_seconds is not None:
            game_time_txt = logic.format_seconds_to_mmss_or_hhmmss(int(game_seconds))

        # Tight validation for Goal events: they must be timestamped and tied to a scoring team.
        # This is required for correctness of goal-based derived stats (plus/minus, pair overlap GF/GA, etc.).
        if et_key == "goal":
            if period is None or int(period) <= 0:
                raise ValueError(
                    "invalid_goal_event_row: missing Period "
                    f"(game_id={int(game_id)}, game_seconds={game_seconds}, team_side={side_label!r}, "
                    f"attributed_jerseys={attributed_jerseys!r})"
                )
            if game_seconds is None:
                raise ValueError(
                    "invalid_goal_event_row: missing Game Seconds/Game Time "
                    f"(game_id={int(game_id)}, period={int(period)}, team_side={side_label!r}, "
                    f"attributed_jerseys={attributed_jerseys!r})"
                )
            if team_id is None or side_norm not in {"home", "away"}:
                raise ValueError(
                    "invalid_goal_event_row: missing/unknown Team Side (must be Home/Away or inferable from jersey) "
                    f"(game_id={int(game_id)}, period={int(period)}, game_seconds={int(game_seconds)}, "
                    f"team_side={side_label!r}, team_raw={_norm_ws(r.get('Team') or r.get('Team Raw') or '')!r}, "
                    f"attributed_jerseys={attributed_jerseys!r}, attributed_players={attributed_players!r})"
                )

        parsed.append(
            {
                "import_key": import_key,
                "event_type_key": et_key,
                "event_type_id": int(et_id),
                "team_side_norm": side_norm,
                "team_id": int(team_id) if team_id is not None else None,
                "player_id": int(player_id) if player_id is not None else None,
                "source": source or None,
                "event_id": event_id,
                "team_raw": _norm_ws(r.get("Team Raw") or r.get("TeamRaw") or r.get("Team")),
                "team_side": side_label or _norm_ws(r.get("Team Side") or r.get("TeamSide")),
                "for_against": _norm_ws(r.get("For/Against") or r.get("ForAgainst")),
                "team_rel": _norm_ws(r.get("Team Rel") or r.get("TeamRel")),
                "period": int(period) if period is not None else None,
                "game_time": game_time_txt,
                "video_time": video_time,
                "game_seconds": int(game_seconds) if game_seconds is not None else None,
                "game_seconds_end": (
                    int(game_seconds_end) if game_seconds_end is not None else None
                ),
                "video_seconds": int(video_seconds) if video_seconds is not None else None,
                "details": details or None,
                "attributed_players": attributed_players or None,
                "attributed_jerseys": attributed_jerseys or None,
                "jersey_norm": jersey_norm,
                "on_ice_players": _norm_ws(r.get("On-Ice Players") or r.get("OnIce Players"))
                or None,
                "on_ice_players_home": _norm_ws(
                    r.get("On-Ice Players (Home)") or r.get("OnIce Players (Home)")
                )
                or None,
                "on_ice_players_away": _norm_ws(
                    r.get("On-Ice Players (Away)") or r.get("OnIce Players (Away)")
                )
                or None,
            }
        )

    # Propagate clip metadata across all events at the same instant so any event row can be used
    # as a video anchor (e.g. assists share the same clip time as the goal / xG / SOG at that time).
    best_video: dict[tuple[int, int], dict[str, Any]] = {}
    for rec in parsed:
        per = rec.get("period")
        gs = rec.get("game_seconds")
        vs = rec.get("video_seconds")
        if not isinstance(per, int) or not isinstance(gs, int) or not isinstance(vs, int):
            continue
        prev = best_video.get((int(per), int(gs)))
        if prev is None or int(vs) < int(prev.get("video_seconds") or 0):
            best_video[(int(per), int(gs))] = {
                "video_seconds": int(vs),
                "video_time": str(rec.get("video_time") or "").strip(),
            }

    if best_video:
        for rec in parsed:
            per = rec.get("period")
            gs = rec.get("game_seconds")
            if not isinstance(per, int) or not isinstance(gs, int):
                continue
            if rec.get("video_seconds") is not None:
                continue
            best = best_video.get((int(per), int(gs)))
            if not best:
                continue
            best_vs = best.get("video_seconds")
            if not isinstance(best_vs, int):
                continue
            rec["video_seconds"] = int(best_vs)
            rec["video_time"] = logic.format_seconds_to_mmss_or_hhmmss(int(best_vs))

    # De-duplicate redundant shot implication rows (Goal > ExpectedGoal > SOG > Shot) at the same instant/player/side.
    chain_keys = {"goal", "expectedgoal", "sog", "shot"}
    groups: dict[
        tuple[Optional[int], Optional[int], Optional[str], Optional[str]],
        list[dict[str, Any]],
    ] = {}
    for rec in parsed:
        if str(rec.get("event_type_key") or "") not in chain_keys:
            continue
        groups.setdefault(
            (
                rec.get("period"),
                rec.get("game_seconds"),
                str(rec.get("team_side_norm") or ""),
                str(rec.get("jersey_norm") or ""),
            ),
            [],
        ).append(rec)

    to_drop: set[str] = set()
    for inst_list in groups.values():
        if len(inst_list) <= 1:
            continue
        ranks = [_events_group_rank(str(r.get("event_type_key") or "")) for r in inst_list]
        best = max([r for r in ranks if r is not None], default=None)
        if best is None:
            continue
        for rec in inst_list:
            rnk = _events_group_rank(str(rec.get("event_type_key") or ""))
            if rnk is None or rnk < best:
                to_drop.add(str(rec.get("import_key") or ""))

    if to_drop:
        parsed = [rec for rec in parsed if str(rec.get("import_key") or "") not in to_drop]

    # Ensure we never attempt to bulk_create/bulk_update multiple rows for the same import_key
    # within a single import (can happen when some event types use team+time keys).
    def _is_blank(v: object) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and not v.strip():
            return True
        return False

    collisions = 0
    deduped: dict[str, dict[str, Any]] = {}
    for rec in parsed:
        import_key = str(rec.get("import_key") or "")
        prev = deduped.get(import_key)
        if prev is None:
            deduped[import_key] = rec
            continue

        collisions += 1
        for k, v in rec.items():
            if k not in prev:
                prev[k] = v
                continue
            if _is_blank(prev[k]) and not _is_blank(v):
                prev[k] = v

    if collisions:
        logger.warning(
            "Deduped %s event row collisions for game_id=%s",
            collisions,
            int(game_id),
        )
        parsed = list(deduped.values())

    with transaction.atomic():
        if replace:
            m.HkyGameEventRow.objects.filter(game_id=int(game_id)).delete()

        def _is_blank(v: object) -> bool:
            if v is None:
                return True
            if isinstance(v, str) and not v.strip():
                return True
            return False

        import_keys = [str(rec.get("import_key") or "") for rec in parsed]
        existing_rows = list(
            m.HkyGameEventRow.objects.filter(
                game_id=int(game_id), import_key__in=import_keys
            ).values(
                "id",
                "import_key",
                "event_type_id",
                "team_id",
                "player_id",
                "source",
                "event_id",
                "team_raw",
                "team_side",
                "for_against",
                "team_rel",
                "period",
                "game_time",
                "video_time",
                "game_seconds",
                "game_seconds_end",
                "video_seconds",
                "details",
                "correction",
                "attributed_players",
                "attributed_jerseys",
                "on_ice_players",
                "on_ice_players_home",
                "on_ice_players_away",
            )
        )
        existing_by_key: dict[str, dict[str, Any]] = {
            str(r["import_key"]): dict(r) for r in (existing_rows or []) if isinstance(r, dict)
        }

        to_create: list[Any] = []
        to_update: list[Any] = []
        linked: set[tuple[int, int]] = set()

        for rec in parsed:
            import_key = str(rec["import_key"])
            existing_row = existing_by_key.get(import_key)
            if existing_row is None:
                to_create.append(
                    m.HkyGameEventRow(
                        game_id=int(game_id),
                        event_type_id=int(rec["event_type_id"]),
                        import_key=import_key,
                        team_id=rec.get("team_id"),
                        player_id=rec.get("player_id"),
                        source=rec.get("source") or None,
                        event_id=rec.get("event_id"),
                        team_raw=rec.get("team_raw") or None,
                        team_side=rec.get("team_side") or None,
                        for_against=rec.get("for_against") or None,
                        team_rel=rec.get("team_rel") or None,
                        period=rec.get("period"),
                        game_time=rec.get("game_time") or None,
                        video_time=rec.get("video_time") or None,
                        game_seconds=rec.get("game_seconds"),
                        game_seconds_end=rec.get("game_seconds_end"),
                        video_seconds=rec.get("video_seconds"),
                        details=rec.get("details") or None,
                        attributed_players=rec.get("attributed_players") or None,
                        attributed_jerseys=rec.get("attributed_jerseys") or None,
                        on_ice_players=rec.get("on_ice_players") or None,
                        on_ice_players_home=rec.get("on_ice_players_home") or None,
                        on_ice_players_away=rec.get("on_ice_players_away") or None,
                        created_at=now,
                        updated_at=None,
                    )
                )
            else:
                event_type_key = str(rec.get("event_type_key") or "").casefold()
                incoming_record_source = str(rec.get("source") or "").casefold()
                existing_record_source = str(existing_row.get("source") or "").casefold()
                incoming_record_is_timetoscore = bool(
                    incoming_is_timetoscore or ("timetoscore" in incoming_record_source)
                )
                existing_record_is_timetoscore = bool("timetoscore" in existing_record_source)
                prefer_incoming = bool(
                    incoming_record_is_timetoscore
                    and event_type_key
                    and event_type_key in prefer_incoming_types
                )
                existing_is_corrected = bool(
                    not _is_blank(existing_row.get("correction"))
                    or "correction" in str(existing_row.get("source") or "").casefold()
                )
                incoming_is_correction = bool("correction" in incoming_record_source)

                def _merge(field: str) -> Any:
                    existing_val = existing_row.get(field)
                    incoming_val = rec.get(field)
                    if (
                        existing_is_corrected
                        and not incoming_is_correction
                        and field
                        in {
                            "player_id",
                            "attributed_players",
                            "attributed_jerseys",
                            "details",
                            "correction",
                        }
                        and not _is_blank(existing_val)
                    ):
                        return existing_val
                    if (
                        event_type_key in {"goal", "goaliechange"}
                        and field in {"player_id", "attributed_players", "attributed_jerseys"}
                        and not _is_blank(incoming_val)
                        and (incoming_record_is_timetoscore or not existing_record_is_timetoscore)
                    ):
                        return incoming_val
                    if prefer_incoming:
                        return incoming_val if not _is_blank(incoming_val) else existing_val
                    return existing_val if not _is_blank(existing_val) else incoming_val

                merged_team_id = _merge("team_id")
                merged_player_id = _merge("player_id")
                to_update.append(
                    m.HkyGameEventRow(
                        id=int(existing_row["id"]),
                        game_id=int(game_id),
                        event_type_id=_merge("event_type_id"),
                        import_key=import_key,
                        team_id=merged_team_id,
                        player_id=merged_player_id,
                        source=_merge("source") or None,
                        event_id=_merge("event_id"),
                        team_raw=_merge("team_raw") or None,
                        team_side=_merge("team_side") or None,
                        for_against=_merge("for_against") or None,
                        team_rel=_merge("team_rel") or None,
                        period=_merge("period"),
                        game_time=_merge("game_time") or None,
                        video_time=_merge("video_time") or None,
                        game_seconds=_merge("game_seconds"),
                        game_seconds_end=_merge("game_seconds_end"),
                        video_seconds=_merge("video_seconds"),
                        details=_merge("details") or None,
                        correction=_merge("correction") or None,
                        attributed_players=_merge("attributed_players") or None,
                        attributed_jerseys=_merge("attributed_jerseys") or None,
                        on_ice_players=_merge("on_ice_players") or None,
                        on_ice_players_home=_merge("on_ice_players_home") or None,
                        on_ice_players_away=_merge("on_ice_players_away") or None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                if merged_player_id is not None and merged_team_id is not None:
                    linked.add((int(merged_player_id), int(merged_team_id)))
            if (
                existing_row is None
                and rec.get("player_id") is not None
                and rec.get("team_id") is not None
            ):
                linked.add((int(rec["player_id"]), int(rec["team_id"])))

        created = 0
        updated = 0
        if to_create:
            m.HkyGameEventRow.objects.bulk_create(to_create, ignore_conflicts=True, batch_size=500)
            created = len(to_create)
        if to_update:
            m.HkyGameEventRow.objects.bulk_update(
                to_update,
                [
                    "event_type",
                    "import_key",
                    "team",
                    "player",
                    "source",
                    "event_id",
                    "team_raw",
                    "team_side",
                    "for_against",
                    "team_rel",
                    "period",
                    "game_time",
                    "video_time",
                    "game_seconds",
                    "game_seconds_end",
                    "video_seconds",
                    "details",
                    "correction",
                    "attributed_players",
                    "attributed_jerseys",
                    "on_ice_players",
                    "on_ice_players_home",
                    "on_ice_players_away",
                    "updated_at",
                ],
                batch_size=500,
            )
            updated = len(to_update)

        # After merging multiple sources, propagate clip metadata across all events at the same
        # instant so any row (including goals/assists/penalties) can act as a video anchor.
        try:
            all_rows = list(
                m.HkyGameEventRow.objects.filter(
                    game_id=int(game_id),
                    period__isnull=False,
                    game_seconds__isnull=False,
                ).values(
                    "id",
                    "period",
                    "game_seconds",
                    "video_time",
                    "video_seconds",
                )
            )
            best_by_time: dict[tuple[int, int], dict[str, Any]] = {}
            for r0 in all_rows:
                per = r0.get("period")
                gs = r0.get("game_seconds")
                if per is None or gs is None:
                    continue
                vt, vs = logic.normalize_video_time_and_seconds(
                    r0.get("video_time"), r0.get("video_seconds")
                )
                if vs is None:
                    continue
                key = (int(per), int(gs))
                prev = best_by_time.get(key)
                if prev is None or int(vs) < int(prev.get("video_seconds") or 0):
                    best_by_time[key] = {
                        "video_seconds": int(vs),
                        "video_time": str(vt or "").strip(),
                    }

            if best_by_time:
                to_fill: list[Any] = []
                for r0 in all_rows:
                    per = r0.get("period")
                    gs = r0.get("game_seconds")
                    if per is None or gs is None:
                        continue
                    best = best_by_time.get((int(per), int(gs)))
                    if not best:
                        continue
                    best_vs = best.get("video_seconds")
                    if not isinstance(best_vs, int):
                        continue

                    has_video_time = bool(str(r0.get("video_time") or "").strip())
                    has_video_seconds = r0.get("video_seconds") is not None
                    if has_video_time and has_video_seconds:
                        continue

                    vt1, vs1 = logic.normalize_video_time_and_seconds(
                        r0.get("video_time"), r0.get("video_seconds")
                    )
                    if vs1 is None:
                        vt1, vs1 = logic.normalize_video_time_and_seconds("", int(best_vs))
                    if vs1 is None:
                        continue

                    to_fill.append(
                        m.HkyGameEventRow(
                            id=int(r0["id"]),
                            video_time=str(vt1 or "").strip() or None,
                            video_seconds=int(vs1),
                            updated_at=now,
                        )
                    )
                if to_fill:
                    m.HkyGameEventRow.objects.bulk_update(
                        to_fill,
                        ["video_time", "video_seconds", "updated_at"],
                        batch_size=500,
                    )
        except Exception:
            logger.warning(
                "Failed to propagate video fields across events for game_id=%s",
                int(game_id),
                exc_info=True,
            )

        linked_players = 0
        if linked:
            to_link = [
                m.HkyGamePlayer(
                    game_id=int(game_id),
                    player_id=int(pid),
                    team_id=int(tid),
                    created_at=now,
                    updated_at=None,
                )
                for pid, tid in sorted(linked)
            ]
            if to_link:
                m.HkyGamePlayer.objects.bulk_create(to_link, ignore_conflicts=True, batch_size=500)
                linked_players = len(to_link)

    return {"ok": True, "created": created, "updated": updated, "linked_players": linked_players}


def _load_game_events_for_display(
    *,
    game_id: int,
) -> tuple[list[str], list[dict[str, str]], Optional[dict[str, Any]]]:
    """
    Load normalized DB event rows for the game.
    Returns (headers, rows, meta) suitable for the existing templates/JS.
    """
    _django_orm, m = _orm_modules()

    qs = (
        m.HkyGameEventRow.objects.filter(game_id=int(game_id))
        .select_related("event_type")
        .order_by("period", "game_seconds", "id")
    )
    qs = qs.exclude(
        import_key__in=m.HkyGameEventSuppression.objects.filter(game_id=int(game_id)).values_list(
            "import_key", flat=True
        )
    )
    if not qs.exists():
        return [], [], None

    raw_rows = list(
        qs.values(
            "id",
            "event_type__name",
            "event_id",
            "source",
            "team_raw",
            "team_side",
            "for_against",
            "team_rel",
            "period",
            "game_time",
            "video_time",
            "game_seconds",
            "game_seconds_end",
            "video_seconds",
            "details",
            "attributed_players",
            "attributed_jerseys",
            "on_ice_players",
            "on_ice_players_home",
            "on_ice_players_away",
            "created_at",
            "updated_at",
        )
    )

    headers = [
        "Event Type",
        "Event ID",
        "Source",
        "Team Raw",
        "Team Side",
        "For/Against",
        "Team Rel",
        "Period",
        "Game Time",
        "Video Time",
        "Game Seconds",
        "Game Seconds End",
        "Video Seconds",
        "Details",
        "Attributed Players",
        "Attributed Jerseys",
        "On-Ice Players",
        "On-Ice Players (Home)",
        "On-Ice Players (Away)",
    ]

    out_rows: list[dict[str, str]] = []
    max_ts: Optional[dt.datetime] = None
    for r in raw_rows:
        ts = r.get("updated_at") or r.get("created_at")
        if isinstance(ts, dt.datetime):
            max_ts = ts if max_ts is None else max(max_ts, ts)

        out_rows.append(
            {
                "__hm_event_row_id": str(int(r.get("id") or 0)),
                "Event Type": str(r.get("event_type__name") or "").strip(),
                "Event ID": "" if r.get("event_id") is None else str(int(r["event_id"])),
                "Source": str(r.get("source") or "").strip(),
                "Team Raw": str(r.get("team_raw") or "").strip(),
                "Team Side": str(r.get("team_side") or "").strip(),
                "For/Against": str(r.get("for_against") or "").strip(),
                "Team Rel": str(r.get("team_rel") or "").strip(),
                "Period": "" if r.get("period") is None else str(int(r["period"])),
                "Game Time": str(r.get("game_time") or "").strip(),
                "Video Time": str(r.get("video_time") or "").strip(),
                "Game Seconds": (
                    "" if r.get("game_seconds") is None else str(int(r["game_seconds"]))
                ),
                "Game Seconds End": (
                    "" if r.get("game_seconds_end") is None else str(int(r["game_seconds_end"]))
                ),
                "Video Seconds": (
                    "" if r.get("video_seconds") is None else str(int(r["video_seconds"]))
                ),
                "Details": str(r.get("details") or "").strip(),
                "Attributed Players": str(r.get("attributed_players") or "").strip(),
                "Attributed Jerseys": str(r.get("attributed_jerseys") or "").strip(),
                "On-Ice Players": str(r.get("on_ice_players") or "").strip(),
                "On-Ice Players (Home)": str(r.get("on_ice_players_home") or "").strip(),
                "On-Ice Players (Away)": str(r.get("on_ice_players_away") or "").strip(),
            }
        )

    meta = {"source_label": "db", "updated_at": max_ts, "count": len(out_rows)}
    return headers, out_rows, meta


def _load_game_shift_rows_for_timeline(*, game_id: int) -> list[dict[str, str]]:
    """
    Load shift intervals from the DB and convert to lightweight "event rows" for timeline rendering.

    These are intentionally NOT included in the Game Events table; they are a separate data stream
    that can be toggled on/off at the league level.
    """
    _django_orm, m = _orm_modules()

    rows = list(
        m.HkyGameShiftRow.objects.filter(game_id=int(game_id))
        .select_related("player")
        .values(
            "team_side",
            "period",
            "game_seconds",
            "game_seconds_end",
            "video_seconds",
            "video_seconds_end",
            "player__name",
            "player__jersey_number",
        )
    )
    if not rows:
        return []

    out: list[dict[str, str]] = []
    for r in rows:
        side = str(r.get("team_side") or "").strip()
        if side.casefold() not in {"home", "away"}:
            continue
        try:
            period = int(r.get("period") or 0)
        except Exception:
            continue
        if period <= 0:
            continue
        gs0 = r.get("game_seconds")
        gs1 = r.get("game_seconds_end")
        if gs0 is None or gs1 is None:
            continue
        try:
            start_gs = int(gs0)
            end_gs = int(gs1)
        except Exception:
            continue
        vs0 = r.get("video_seconds")
        vs1 = r.get("video_seconds_end")
        try:
            start_vs = int(vs0) if vs0 is not None else None
        except Exception:
            start_vs = None
        try:
            end_vs = int(vs1) if vs1 is not None else None
        except Exception:
            end_vs = None

        jersey = logic.normalize_jersey_number(r.get("player__jersey_number"))
        name = str(r.get("player__name") or "").strip()
        label = name
        if jersey and name:
            label = f"{name} (#{jersey})"

        # Treat stored game_seconds/game_seconds_end as shift start/end in "within-period" seconds.
        out.append(
            {
                "Event Type": "On-Ice",
                "Event ID": "",
                "Source": "shifts",
                "Team Raw": "",
                "Team Side": side,
                "For/Against": "",
                "Team Rel": side,
                "Period": str(int(period)),
                "Game Time": "",
                "Video Time": "",
                "Game Seconds": str(int(start_gs)),
                "Game Seconds End": "",
                "Video Seconds": str(int(start_vs)) if start_vs is not None else "",
                "Details": label,
                "Attributed Players": label,
                "Attributed Jerseys": str(jersey or ""),
                "On-Ice Players": "",
                "On-Ice Players (Home)": "",
                "On-Ice Players (Away)": "",
            }
        )
        out.append(
            {
                "Event Type": "Off-Ice",
                "Event ID": "",
                "Source": "shifts",
                "Team Raw": "",
                "Team Side": side,
                "For/Against": "",
                "Team Rel": side,
                "Period": str(int(period)),
                "Game Time": "",
                "Video Time": "",
                "Game Seconds": str(int(end_gs)),
                "Game Seconds End": "",
                "Video Seconds": str(int(end_vs)) if end_vs is not None else "",
                "Details": label,
                "Attributed Players": label,
                "Attributed Jerseys": str(jersey or ""),
                "On-Ice Players": "",
                "On-Ice Players (Home)": "",
                "On-Ice Players (Away)": "",
            }
        )

    return out


def _overlay_game_player_stats_from_event_rows(
    *, game_id: int, stats_by_pid: dict[int, dict[str, Any]]
) -> None:
    """
    Best-effort: compute goal-derived per-game stats from normalized event rows and overlay onto the
    per-game `stats_by_pid` dict used by the game detail view:

      - game-winning goals (`gw_goals`)
      - game-tying goals (`gt_goals`)
      - on-ice goal GF/GA and +/- from goal event on-ice lists (`gf_counted`/`ga_counted`/`plus_minus`)

    This intentionally avoids stored aggregates and does not require shift rows.
    """
    _django_orm, m = _orm_modules()

    # Game-winning / game-tying goals + on-ice goal +/- are computed from goal event rows.
    try:
        game = (
            m.HkyGame.objects.filter(id=int(game_id))
            .values("team1_id", "team2_id", "team1_score", "team2_score", "is_final")
            .first()
        )
        if not game:
            return
        team1_id = int(game.get("team1_id") or 0)
        team2_id = int(game.get("team2_id") or 0)
        if team1_id <= 0 or team2_id <= 0:
            return

        players = list(
            m.Player.objects.filter(team_id__in=[team1_id, team2_id]).values(
                "id", "team_id", "jersey_number", "name"
            )
        )

        jersey_to_pids: dict[tuple[int, str], list[int]] = {}
        team_id_by_player_id: dict[int, int] = {}
        all_player_ids: set[int] = set()

        def _player_jersey_norm(p: dict[str, Any]) -> Optional[str]:
            j0 = logic.normalize_jersey_number(p.get("jersey_number"))
            if j0:
                return j0
            nm = str(p.get("name") or "").strip()
            m0 = re.match(r"^\s*#?\s*(\d+)\b", nm)
            if m0:
                return logic.normalize_jersey_number(m0.group(1))
            m1 = re.search(r"\(\s*#\s*(\d+)\s*\)\s*$", nm)
            if m1:
                return logic.normalize_jersey_number(m1.group(1))
            return None

        for p in players:
            try:
                pid = int(p.get("id") or 0)
                tid = int(p.get("team_id") or 0)
            except Exception:
                continue
            if pid <= 0 or tid <= 0:
                continue
            all_player_ids.add(pid)
            team_id_by_player_id[pid] = tid
            jn = _player_jersey_norm(p)
            if not jn:
                continue
            jersey_to_pids.setdefault((tid, str(jn)), []).append(pid)

        def _extract_numbers(raw: Any) -> set[str]:
            s = str(raw or "")
            out: set[str] = set()
            for m0 in re.findall(r"(\d+)", s):
                try:
                    out.add(str(int(m0)))
                except Exception:
                    continue
            return out

        def _norm_side(raw: Any) -> Optional[str]:
            v = str(raw or "").strip().casefold()
            if v in {"home", "team1"}:
                return "home"
            if v in {"away", "team2"}:
                return "away"
            return None

        goal_qs = m.HkyGameEventRow.objects.filter(game_id=int(game_id), event_type__key="goal")
        goal_qs = goal_qs.exclude(
            import_key__in=m.HkyGameEventSuppression.objects.filter(
                game_id=int(game_id)
            ).values_list("import_key", flat=True)
        )
        goal_rows = list(
            goal_qs.values(
                "id",
                "event_id",
                "period",
                "game_seconds",
                "game_time",
                "team_id",
                "team_side",
                "team_rel",
                "team_raw",
                "player_id",
                "attributed_jerseys",
                "on_ice_players_home",
                "on_ice_players_away",
            )
        )

        def _int_or_none(raw: Any) -> Optional[int]:
            try:
                if raw is None:
                    return None
                s = str(raw).strip()
                if not s:
                    return None
                return int(float(s))
            except Exception:
                return None

        def _scoring_team_id(row: dict[str, Any]) -> Optional[int]:
            tid = _int_or_none(row.get("team_id"))
            if tid in {team1_id, team2_id}:
                return int(tid)
            pid = _int_or_none(row.get("player_id"))
            if pid is not None:
                ptid = team_id_by_player_id.get(int(pid))
                if ptid in {team1_id, team2_id}:
                    return int(ptid)
            side = (
                _norm_side(row.get("team_side"))
                or _norm_side(row.get("team_rel"))
                or _norm_side(row.get("team_raw"))
            )
            if side == "home":
                return int(team1_id)
            if side == "away":
                return int(team2_id)
            return None

        def _goal_sort_key(row: dict[str, Any]) -> tuple:
            per = _int_or_none(row.get("period"))
            period_key = int(per) if per is not None and per > 0 else 999
            gs = _int_or_none(row.get("game_seconds"))
            if gs is None:
                gs = logic.parse_duration_seconds(row.get("game_time"))
            gs_missing = 1 if gs is None else 0
            gs_key = -int(gs) if gs is not None else 0  # time remaining: higher => earlier
            eid = _int_or_none(row.get("event_id"))
            eid_key = int(eid) if eid is not None else 1_000_000_000
            row_id = _int_or_none(row.get("id")) or 0
            return (period_key, gs_missing, gs_key, eid_key, row_id)

        def _scorer_pid(row: dict[str, Any], scoring_tid: Optional[int]) -> Optional[int]:
            pid = _int_or_none(row.get("player_id"))
            if pid is not None and pid > 0:
                return int(pid)
            if scoring_tid not in {team1_id, team2_id}:
                return None
            jerseys_raw = str(row.get("attributed_jerseys") or "").strip()
            if not jerseys_raw:
                return None
            nums = sorted(
                _extract_numbers(jerseys_raw),
                key=lambda x: int(x) if str(x).isdigit() else 9999,
            )
            if not nums:
                return None
            scorer_num = nums[0]
            candidates = jersey_to_pids.get((int(scoring_tid), str(scorer_num)), [])
            if len(set(candidates)) == 1:
                return int(list(set(candidates))[0])
            return None

        # Derive GT/GW goals from goal ordering and final score when possible.
        if goal_rows and bool(game.get("is_final")):
            t1_score = _int_or_none(game.get("team1_score"))
            t2_score = _int_or_none(game.get("team2_score"))

            if t1_score is None or t2_score is None:
                # Fall back to event-derived totals if the game record doesn't have a score.
                t1_score = 0
                t2_score = 0
                for r0 in goal_rows:
                    tid = _scoring_team_id(r0)
                    if tid == team1_id:
                        t1_score += 1
                    elif tid == team2_id:
                        t2_score += 1
                    else:
                        t1_score = None
                        t2_score = None
                        break

            if (
                t1_score is not None
                and t2_score is not None
                and int(t1_score) != int(t2_score)
                and int(t1_score) >= 0
                and int(t2_score) >= 0
            ):
                winner_team_id = team1_id if int(t1_score) > int(t2_score) else team2_id
                loser_score = min(int(t1_score), int(t2_score))
                target_winner_goal_num = int(loser_score) + 1

                gt_goals_by_pid: dict[int, int] = {}
                gw_pid: Optional[int] = None
                score_team1 = 0
                score_team2 = 0
                winner_goal_num = 0
                for r0 in sorted(goal_rows, key=_goal_sort_key):
                    scoring_tid = _scoring_team_id(r0)
                    if scoring_tid is None:
                        gt_goals_by_pid = {}
                        gw_pid = None
                        break

                    if scoring_tid == team1_id:
                        score_team1 += 1
                    elif scoring_tid == team2_id:
                        score_team2 += 1
                    else:
                        continue

                    pid = _scorer_pid(r0, scoring_tid)
                    if pid is not None and score_team1 == score_team2:
                        gt_goals_by_pid[int(pid)] = gt_goals_by_pid.get(int(pid), 0) + 1

                    if scoring_tid == int(winner_team_id):
                        winner_goal_num += 1
                        if winner_goal_num == int(target_winner_goal_num):
                            gw_pid = int(pid) if pid is not None else None
                            break

                if gw_pid is not None:
                    stats_by_pid.setdefault(
                        gw_pid, {"player_id": int(gw_pid), "game_id": int(game_id)}
                    )
                    stats_by_pid[gw_pid]["gw_goals"] = 1

                for pid, n in gt_goals_by_pid.items():
                    if int(n) <= 0:
                        continue
                    stats_by_pid.setdefault(pid, {"player_id": int(pid), "game_id": int(game_id)})
                    stats_by_pid[pid]["gt_goals"] = int(n)

        total_goals = 0
        complete_goals = 0
        gf_by_pid: dict[int, int] = {}
        ga_by_pid: dict[int, int] = {}

        for r in goal_rows:
            scoring_tid = _scoring_team_id(r)
            if scoring_tid == team1_id:
                side = "home"
            elif scoring_tid == team2_id:
                side = "away"
            else:
                side = (
                    _norm_side(r.get("team_side"))
                    or _norm_side(r.get("team_rel"))
                    or _norm_side(r.get("team_raw"))
                )
            if side not in {"home", "away"}:
                continue
            total_goals += 1

            home_raw = str(r.get("on_ice_players_home") or "").strip()
            away_raw = str(r.get("on_ice_players_away") or "").strip()
            if not home_raw or not away_raw:
                continue

            complete_goals += 1
            home_nums = _extract_numbers(home_raw)
            away_nums = _extract_numbers(away_raw)

            for j in home_nums:
                candidates = jersey_to_pids.get((team1_id, j), [])
                if len(set(candidates)) != 1:
                    continue
                pid = int(list(set(candidates))[0])
                if side == "home":
                    gf_by_pid[pid] = gf_by_pid.get(pid, 0) + 1
                else:
                    ga_by_pid[pid] = ga_by_pid.get(pid, 0) + 1

            for j in away_nums:
                candidates = jersey_to_pids.get((team2_id, j), [])
                if len(set(candidates)) != 1:
                    continue
                pid = int(list(set(candidates))[0])
                if side == "away":
                    gf_by_pid[pid] = gf_by_pid.get(pid, 0) + 1
                else:
                    ga_by_pid[pid] = ga_by_pid.get(pid, 0) + 1

        # Only compute on-ice GF/GA when at least one goal has complete on-ice lists.
        # Note: GF/GA are computed only from goals that have complete on-ice lists ("counted" goals).
        # Plus/Minus is only meaningful when *all* goals have complete on-ice lists.
        if total_goals <= 0 or complete_goals <= 0:
            return

        for pid in all_player_ids:
            gf = int(gf_by_pid.get(pid, 0) or 0)
            ga = int(ga_by_pid.get(pid, 0) or 0)
            stats_by_pid.setdefault(pid, {"player_id": int(pid), "game_id": int(game_id)})
            stats_by_pid[pid]["gf_counted"] = int(gf)
            stats_by_pid[pid]["ga_counted"] = int(ga)
            if complete_goals == total_goals:
                stats_by_pid[pid]["plus_minus"] = int(gf) - int(ga)
    except Exception:
        # Best-effort: this overlay is optional and should not break page rendering.
        logger.exception(
            "Failed to overlay on-ice GF/GA into player stat rows (game_id=%s)", game_id
        )


def _overlay_on_ice_goal_stats_from_shift_rows(
    *,
    game_id: int,
    stats_by_pid: dict[int, dict[str, Any]],
) -> None:
    """
    Compute on-ice GF/GA/+/- from goal events + shift intervals (hky_game_shift_rows) and overlay
    into `stats_by_pid`.

    This intentionally avoids relying on per-goal on-ice lists (which may be missing) and instead
    uses the canonical shift intervals imported from spreadsheets.
    """
    _django_orm, m = _orm_modules()

    game = m.HkyGame.objects.filter(id=int(game_id)).values("team1_id", "team2_id").first()
    if not game:
        return
    team1_id = int(game.get("team1_id") or 0)
    team2_id = int(game.get("team2_id") or 0)
    if team1_id <= 0 or team2_id <= 0:
        return

    suppressed = m.HkyGameEventSuppression.objects.filter(game_id=int(game_id)).values_list(
        "import_key", flat=True
    )

    goal_rows = list(
        m.HkyGameEventRow.objects.filter(game_id=int(game_id), event_type__key="goal")
        .exclude(import_key__in=suppressed)
        .values(
            "source",
            "period",
            "game_seconds",
            "game_time",
            "team_id",
            "team_side",
            "team_rel",
            "team_raw",
            "player_id",
        )
    )
    if not goal_rows:
        return

    # Prefer TimeToScore goals when present to avoid duplicate goal streams.
    if any("timetoscore" in str(r.get("source") or "").casefold() for r in goal_rows):
        goal_rows = [r for r in goal_rows if "timetoscore" in str(r.get("source") or "").casefold()]
        if not goal_rows:
            return

    team_id_by_player_id: dict[int, int] = {}
    for p in m.Player.objects.filter(team_id__in=[team1_id, team2_id]).values("id", "team_id"):
        try:
            pid = int(p.get("id") or 0)
            tid = int(p.get("team_id") or 0)
        except Exception:
            continue
        if pid > 0 and tid > 0:
            team_id_by_player_id[int(pid)] = int(tid)

    def _int_or_none(raw: Any) -> Optional[int]:
        try:
            if raw is None:
                return None
            s = str(raw).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    def _norm_side(raw: Any) -> Optional[str]:
        v = str(raw or "").strip().casefold()
        if v in {"home", "team1"}:
            return "home"
        if v in {"away", "team2"}:
            return "away"
        return None

    def _scoring_side(row: dict[str, Any]) -> Optional[str]:
        tid = _int_or_none(row.get("team_id"))
        if tid == team1_id:
            return "home"
        if tid == team2_id:
            return "away"
        pid = _int_or_none(row.get("player_id"))
        if pid is not None:
            ptid = team_id_by_player_id.get(int(pid))
            if ptid == team1_id:
                return "home"
            if ptid == team2_id:
                return "away"
        side = (
            _norm_side(row.get("team_side"))
            or _norm_side(row.get("team_rel"))
            or _norm_side(row.get("team_raw"))
        )
        if side in {"home", "away"}:
            return side
        return None

    def _goal_seconds(row: dict[str, Any]) -> Optional[int]:
        gs = _int_or_none(row.get("game_seconds"))
        if gs is not None:
            return int(gs)
        return logic.parse_duration_seconds(row.get("game_time"))

    shift_rows = list(
        m.HkyGameShiftRow.objects.filter(game_id=int(game_id), player_id__isnull=False)
        .exclude(period__isnull=True)
        .exclude(game_seconds__isnull=True)
        .exclude(game_seconds_end__isnull=True)
        .values("player_id", "team_side", "period", "game_seconds", "game_seconds_end")
    )
    if not shift_rows:
        return

    shifts_by_period: dict[int, dict[str, list[tuple[int, int, int]]]] = {}
    all_pids: set[int] = set()
    for r in shift_rows:
        try:
            pid = int(r.get("player_id") or 0)
            per = int(r.get("period") or 0)
            gs0 = int(r.get("game_seconds") or 0)
            gs1 = int(r.get("game_seconds_end") or 0)
        except Exception:
            continue
        if pid <= 0 or per <= 0:
            continue
        side = _norm_side(r.get("team_side"))
        if side not in {"home", "away"}:
            continue
        all_pids.add(int(pid))
        shifts_by_period.setdefault(int(per), {}).setdefault(str(side), []).append((pid, gs0, gs1))

    if not shifts_by_period:
        return

    def _shift_contains(*, t: int, start: int, end: int) -> bool:
        lo = start if start <= end else end
        hi = end if start <= end else start
        if t < lo or t > hi:
            return False
        # Match scripts/parse_stats_inputs.py: don't count goals exactly at shift start.
        return int(t) != int(start)

    gf_by_pid: dict[int, int] = {}
    ga_by_pid: dict[int, int] = {}

    for gr in goal_rows:
        try:
            per = int(gr.get("period") or 0)
        except Exception:
            continue
        if per <= 0:
            continue
        t = _goal_seconds(gr)
        if t is None:
            continue
        side = _scoring_side(gr)
        if side not in {"home", "away"}:
            continue

        period_shifts = shifts_by_period.get(int(per), {})
        home_shifts = period_shifts.get("home", [])
        away_shifts = period_shifts.get("away", [])

        home_on: set[int] = set()
        for pid, start, end in home_shifts:
            if _shift_contains(t=int(t), start=int(start), end=int(end)):
                home_on.add(int(pid))
        away_on: set[int] = set()
        for pid, start, end in away_shifts:
            if _shift_contains(t=int(t), start=int(start), end=int(end)):
                away_on.add(int(pid))

        if side == "home":
            for pid in home_on:
                gf_by_pid[int(pid)] = gf_by_pid.get(int(pid), 0) + 1
            for pid in away_on:
                ga_by_pid[int(pid)] = ga_by_pid.get(int(pid), 0) + 1
        else:
            for pid in away_on:
                gf_by_pid[int(pid)] = gf_by_pid.get(int(pid), 0) + 1
            for pid in home_on:
                ga_by_pid[int(pid)] = ga_by_pid.get(int(pid), 0) + 1

    for pid in all_pids:
        gf = int(gf_by_pid.get(int(pid), 0) or 0)
        ga = int(ga_by_pid.get(int(pid), 0) or 0)
        stats_by_pid.setdefault(pid, {"player_id": int(pid), "game_id": int(game_id)})
        stats_by_pid[pid]["gf_counted"] = int(gf)
        stats_by_pid[pid]["ga_counted"] = int(ga)
        stats_by_pid[pid]["plus_minus"] = int(gf) - int(ga)


def _overlay_completed_passes_into_player_stat_rows(
    *,
    game_ids: list[int],
    player_stats_rows: list[dict[str, Any]],
) -> None:
    """
    Best-effort: if `PlayerStat.completed_passes` is missing/0, derive it from normalized event rows.
    This makes Completed Passes show up in team-level aggregated player stats even when only events
    were imported.
    """
    if not game_ids or not player_stats_rows:
        return
    _django_orm, m = _orm_modules()
    from django.db.models import Count

    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for r in player_stats_rows:
        try:
            pid = int(r.get("player_id") or 0)
            gid = int(r.get("game_id") or 0)
        except Exception:
            continue
        if pid > 0 and gid > 0:
            by_key[(pid, gid)] = r
    if not by_key:
        return

    gids = [int(x) for x in game_ids]
    pids = sorted({int(pid) for (pid, _gid) in by_key.keys()})
    if not pids:
        return

    suppressed = m.HkyGameEventSuppression.objects.filter(game_id__in=gids).values_list(
        "import_key", flat=True
    )
    qs = (
        m.HkyGameEventRow.objects.filter(
            game_id__in=gids,
            player_id__in=pids,
            event_type__key="completedpass",
        )
        .exclude(import_key__in=suppressed)
        .values("player_id", "game_id")
        .annotate(n=Count("id"))
    )
    for r in qs:
        try:
            pid = int(r.get("player_id") or 0)
            gid = int(r.get("game_id") or 0)
            n = int(r.get("n") or 0)
        except Exception:
            continue
        if pid <= 0 or gid <= 0 or n <= 0:
            continue
        row = by_key.get((pid, gid))
        if not row:
            continue
        try:
            existing = int(row.get("completed_passes") or 0)
        except Exception:
            existing = 0
        if existing <= 0:
            row["completed_passes"] = int(n)

    return


def _overlay_scoring_into_player_stat_rows_from_event_rows(
    *,
    game_ids: list[int],
    player_stats_rows: list[dict[str, Any]],
) -> None:
    """
    Best-effort: if scoring-derived `PlayerStat` fields (Goals/Assists/Shots/SOG/xG) are missing,
    derive them from normalized event rows and overlay into the in-memory player stat rows used for
    team-level aggregation/rendering.

    This is important for "events-only" uploads (e.g. external games / spreadsheets-only flows)
    where game pages can compute scoring from `hky_game_event_rows`, but `player_stats` rows may not
    have explicit scoring columns populated.
    """
    if not game_ids or not player_stats_rows:
        return
    _django_orm, m = _orm_modules()
    from django.db.models import Count

    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for r in player_stats_rows:
        try:
            pid = int(r.get("player_id") or 0)
            gid = int(r.get("game_id") or 0)
        except Exception:
            continue
        if pid > 0 and gid > 0:
            by_key[(pid, gid)] = r
    if not by_key:
        return

    gids = [int(x) for x in game_ids]
    pids = sorted({int(pid) for (pid, _gid) in by_key.keys()})
    if not pids:
        return

    suppressed = m.HkyGameEventSuppression.objects.filter(game_id__in=gids).values_list(
        "import_key", flat=True
    )
    qs = (
        m.HkyGameEventRow.objects.filter(
            game_id__in=gids,
            player_id__in=pids,
            event_type__key__in=[
                "goal",
                "assist",
                "expectedgoal",
                "sog",
                "shotongoal",
                "shot",
            ],
        )
        .exclude(import_key__in=suppressed)
        .values("player_id", "game_id", "event_type__key")
        .annotate(n=Count("id"))
    )

    counts_by_key: dict[tuple[int, int], dict[str, int]] = {}
    present_by_game_id: dict[int, set[str]] = {}
    for r in qs:
        try:
            pid = int(r.get("player_id") or 0)
            gid = int(r.get("game_id") or 0)
            n = int(r.get("n") or 0)
        except Exception:
            continue
        if pid <= 0 or gid <= 0 or n <= 0:
            continue
        k = str(r.get("event_type__key") or "").strip().casefold()
        if not k:
            continue
        present_by_game_id.setdefault(gid, set()).add(k)
        counts_by_key.setdefault((pid, gid), {})
        counts_by_key[(pid, gid)][k] = counts_by_key[(pid, gid)].get(k, 0) + int(n)

    scoring_game_ids = {
        int(gid)
        for gid, keys in (present_by_game_id or {}).items()
        if ("goal" in keys) or ("assist" in keys)
    }
    shot_game_ids = {
        int(gid)
        for gid, keys in (present_by_game_id or {}).items()
        if any(k in keys for k in ("expectedgoal", "sog", "shotongoal", "shot"))
    }

    for (pid, gid), row in by_key.items():
        if not row:
            continue

        c = counts_by_key.get((pid, gid), {})
        goals = int(c.get("goal") or 0)
        assists = int(c.get("assist") or 0)
        xg_direct = int(c.get("expectedgoal") or 0)
        sog_direct = int(c.get("sog") or 0) + int(c.get("shotongoal") or 0)
        shots_direct = int(c.get("shot") or 0)

        xg = goals + xg_direct
        sog = xg + sog_direct
        shots = sog + shots_direct

        # Only fill missing (NULL) values; do not override explicit DB values.
        # When we have event data for a game, treat missing stats as 0 so per-game rates (PPG, shots/game)
        # don't use a denominator of 1 just because the player had 0 in that stat.
        if gid in scoring_game_ids:
            if row.get("goals") is None:
                row["goals"] = int(goals)
            if row.get("assists") is None:
                row["assists"] = int(assists)
        if gid in shot_game_ids:
            if row.get("expected_goals") is None:
                row["expected_goals"] = int(xg)
            if row.get("sog") is None:
                row["sog"] = int(sog)
            if row.get("shots") is None:
                row["shots"] = int(shots)


def _overlay_shift_stats_into_player_stat_rows(
    *,
    game_ids: list[int],
    player_stats_rows: list[dict[str, Any]],
) -> None:
    """
    Best-effort: derive TOI and shift counts from imported shift rows (hky_game_shift_rows)
    and overlay into the in-memory player stat rows used for aggregation/rendering.

    This keeps shift stats opt-in at the league level and avoids storing derived aggregates.
    """
    if not game_ids or not player_stats_rows:
        return
    _django_orm, m = _orm_modules()
    from django.db.models import Count, F, IntegerField, Sum
    from django.db.models.functions import Abs

    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for r in player_stats_rows:
        try:
            pid = int(r.get("player_id") or 0)
            gid = int(r.get("game_id") or 0)
        except Exception:
            continue
        if pid > 0 and gid > 0:
            by_key[(pid, gid)] = r
    if not by_key:
        return

    gids = [int(x) for x in game_ids]
    pids = sorted({int(pid) for (pid, _gid) in by_key.keys()})
    if not pids:
        return

    dur_expr = Abs(F("game_seconds_end") - F("game_seconds"))
    qs = (
        m.HkyGameShiftRow.objects.filter(game_id__in=gids, player_id__in=pids)
        .exclude(game_seconds__isnull=True)
        .exclude(game_seconds_end__isnull=True)
        .values("player_id", "game_id")
        .annotate(toi=Sum(dur_expr, output_field=IntegerField()), shifts=Count("id"))
    )
    for r in qs:
        try:
            pid = int(r.get("player_id") or 0)
            gid = int(r.get("game_id") or 0)
        except Exception:
            continue
        row = by_key.get((pid, gid))
        if not row:
            continue
        try:
            toi = int(r.get("toi") or 0)
        except Exception:
            toi = 0
        try:
            shifts = int(r.get("shifts") or 0)
        except Exception:
            shifts = 0
        row["toi_seconds"] = int(toi)
        row["shifts"] = int(shifts)

    return


def _mask_shift_stats_in_player_stat_rows(player_stats_rows: list[dict[str, Any]]) -> None:
    """
    When a league has shift data hidden, ensure shift-derived stats are not shown even if they
    exist in PlayerStat from older imports.
    """
    if not player_stats_rows:
        return
    shift_keys = (
        "toi_seconds",
        "shifts",
        "video_toi_seconds",
        "sb_avg_shift_seconds",
        "sb_median_shift_seconds",
        "sb_longest_shift_seconds",
        "sb_shortest_shift_seconds",
    )
    for r in player_stats_rows:
        if not isinstance(r, dict):
            continue
        for k in shift_keys:
            if k in r:
                r[k] = None


def _overlay_timetoscore_zero_stats_into_player_stat_rows(
    *,
    game_ids: list[int],
    player_stats_rows: list[dict[str, Any]],
) -> None:
    """
    TimeToScore-linked games have explicit scoring/penalty records. For those games, missing values
    should be treated as 0 (not "unknown") so per-game denominators are stable for players who had
    zero goals/assists/PIM.
    """
    if not game_ids or not player_stats_rows:
        return
    _django_orm, m = _orm_modules()
    t2s_game_ids = set(
        m.HkyGame.objects.filter(
            id__in=[int(x) for x in game_ids], timetoscore_game_id__isnull=False
        ).values_list("id", flat=True)
    )
    if not t2s_game_ids:
        return
    keys = ("goals", "assists", "pim")
    for r in player_stats_rows:
        if not isinstance(r, dict):
            continue
        try:
            gid = int(r.get("game_id") or 0)
        except Exception:
            continue
        if gid <= 0 or gid not in t2s_game_ids:
            continue
        for k in keys:
            if r.get(k) is None:
                r[k] = 0


def _player_stat_rows_from_event_tables_for_team_games(
    *,
    team_id: int,
    schedule_games: list[dict[str, Any]],
    roster_players: list[dict[str, Any]],
    show_shift_data: bool,
) -> list[dict[str, Any]]:
    """
    Build per-player/per-game stats rows for team-page aggregation from event/shift tables.

    This intentionally avoids using `player_stats` DB rows for any stat values.
    """
    if not schedule_games or not roster_players:
        return []

    _django_orm, m = _orm_modules()

    games_by_id: dict[int, dict[str, Any]] = {}
    game_ids: list[int] = []
    side_by_game_id: dict[int, str] = {}
    for g2 in schedule_games or []:
        try:
            gid = int(g2.get("id") or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        games_by_id[int(gid)] = dict(g2)
        game_ids.append(int(gid))
        side: Optional[str] = None
        try:
            if int(g2.get("team1_id") or 0) == int(team_id):
                side = "home"
            elif int(g2.get("team2_id") or 0) == int(team_id):
                side = "away"
        except Exception:
            side = None
        if side:
            side_by_game_id[int(gid)] = str(side)

    if not game_ids:
        return []

    def _norm_name(raw: Any) -> str:
        return re.sub(r"\s+", " ", str(raw or "").strip()).casefold()

    # Roster lookup helpers for jersey/name matching (used for events where player_id is NULL).
    roster_by_pid: dict[int, dict[str, Any]] = {}
    jersey_to_players: dict[str, list[dict[str, Any]]] = {}
    name_jersey_to_pid: dict[tuple[str, str], int] = {}
    for p in roster_players or []:
        try:
            pid = int(p.get("id") or 0)
        except Exception:
            continue
        if pid <= 0:
            continue
        name = str(p.get("name") or "").strip()
        pos = str(p.get("position") or "").strip().upper()
        jn = logic.normalize_jersey_number(p.get("jersey_number"))
        roster_by_pid[int(pid)] = {"id": int(pid), "name": name, "position": pos, "jersey": jn}
        if jn:
            jersey_to_players.setdefault(str(jn), []).append(roster_by_pid[int(pid)])
            name_jersey_to_pid[(str(_norm_name(name)), str(jn))] = int(pid)

    def _pick_pid_for_jersey(jersey: str, *, prefer_goalie: Optional[bool] = None) -> Optional[int]:
        candidates = list(jersey_to_players.get(str(jersey), []))
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0]["id"])
        if prefer_goalie is not None:
            if prefer_goalie:
                for c in candidates:
                    if str(c.get("position") or "").upper() == "G":
                        return int(c["id"])
            else:
                for c in candidates:
                    if str(c.get("position") or "").upper() != "G":
                        return int(c["id"])
        # Default: prefer skaters for skater-centric events (goals/assists/penalties/shots).
        for c in candidates:
            if str(c.get("position") or "").upper() != "G":
                return int(c["id"])
        return int(candidates[0]["id"])

    def _pid_for_name_jersey(name: Optional[str], jersey: Optional[str]) -> Optional[int]:
        if name and jersey:
            pid = name_jersey_to_pid.get((_norm_name(name), str(jersey)))
            if pid is not None:
                return int(pid)
        if jersey:
            return _pick_pid_for_jersey(str(jersey), prefer_goalie=False)
        return None

    def _extract_numbers(raw: Any) -> list[str]:
        out: list[str] = []
        for m0 in re.findall(r"(\d+)", str(raw or "")):
            jn = logic.normalize_jersey_number(m0)
            if jn:
                out.append(str(jn))
        return out

    def _parse_on_ice_list(raw: Any) -> list[tuple[Optional[str], Optional[str]]]:
        s = str(raw or "").strip()
        if not s:
            return []
        out: list[tuple[Optional[str], Optional[str]]] = []
        for tok in [t.strip() for t in s.split(",") if t.strip()]:
            m0 = re.match(r"^(.*?)\(\s*#?\s*(\d+)\s*\)\s*$", tok)
            if m0:
                out.append((m0.group(1).strip(), logic.normalize_jersey_number(m0.group(2))))
                continue
            nums = _extract_numbers(tok)
            if nums:
                out.append((tok, nums[0]))
        return out

    def _int_or_none(raw: Any) -> Optional[int]:
        try:
            if raw is None:
                return None
            s = str(raw).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    def _event_team_id(row: dict[str, Any]) -> Optional[int]:
        tid = _int_or_none(row.get("team_id"))
        if tid is not None and tid > 0:
            return int(tid)
        gid = _int_or_none(row.get("game_id"))
        if gid is None:
            return None
        g2 = games_by_id.get(int(gid)) or {}
        side = str(row.get("team_side") or "").strip().casefold()
        if side in {"home", "team1"}:
            return _int_or_none(g2.get("team1_id"))
        if side in {"away", "team2"}:
            return _int_or_none(g2.get("team2_id"))
        return None

    def _goal_sort_key(row: dict[str, Any]) -> tuple:
        per = _int_or_none(row.get("period"))
        period_key = int(per) if per is not None and per > 0 else 999
        gs = _int_or_none(row.get("game_seconds"))
        if gs is None:
            gs = logic.parse_duration_seconds(row.get("game_time"))
        gs_missing = 1 if gs is None else 0
        gs_key = -int(gs) if gs is not None else 0
        row_id = _int_or_none(row.get("id")) or 0
        return (period_key, gs_missing, gs_key, row_id)

    def _parse_pim_minutes(details: Any) -> int:
        s = str(details or "").strip()
        if not s:
            return 0
        m0 = re.search(r"(\d+)\s*(?:m|min)\b", s, flags=re.IGNORECASE)
        if not m0:
            return 0
        try:
            return max(0, int(m0.group(1)))
        except Exception:
            return 0

    # ----------------------------
    # Fetch relevant event rows once.
    # ----------------------------
    suppressed = m.HkyGameEventSuppression.objects.filter(game_id__in=game_ids).values_list(
        "import_key", flat=True
    )
    event_rows = list(
        m.HkyGameEventRow.objects.filter(
            game_id__in=game_ids,
            event_type__key__in=[
                "goal",
                "assist",
                "xg",
                "expectedgoal",
                "sog",
                "shotongoal",
                "shot",
                "penalty",
                "completedpass",
                "giveaway",
                "turnoversforced",
                "createdturnovers",
                "takeaway",
                "controlledentry",
                "controlledexit",
            ],
        )
        .exclude(import_key__in=suppressed)
        .select_related("event_type")
        .values(
            "id",
            "game_id",
            "event_type__key",
            "source",
            "team_id",
            "team_side",
            "for_against",
            "period",
            "game_seconds",
            "game_time",
            "player_id",
            "attributed_jerseys",
            "details",
            "on_ice_players_home",
            "on_ice_players_away",
        )
    )

    # Per-game coverage signals.
    goal_rows_by_game: dict[int, list[dict[str, Any]]] = {}
    has_any_goal_row: set[int] = set()
    scoring_game_ids: set[int] = set()
    shot_game_ids: set[int] = set()
    pim_game_ids: set[int] = set()
    completed_pass_game_ids: set[int] = set()
    giveaway_game_ids: set[int] = set()
    turnovers_forced_game_ids: set[int] = set()
    created_turnovers_game_ids: set[int] = set()
    takeaway_game_ids: set[int] = set()
    on_ice_goal_game_ids: set[int] = set()
    on_ice_goal_complete_game_ids: set[int] = set()
    ce_game_ids: set[int] = set()
    cx_game_ids: set[int] = set()

    # Direct per-player counts (by roster pid).
    direct_counts: dict[tuple[int, int], dict[str, int]] = {}
    ot_goals_by_pid_gid: dict[tuple[int, int], int] = {}
    ot_assists_by_pid_gid: dict[tuple[int, int], int] = {}

    for r0 in event_rows or []:
        try:
            gid = int(r0.get("game_id") or 0)
        except Exception:
            continue
        if gid <= 0 or gid not in games_by_id:
            continue
        et = str(r0.get("event_type__key") or "").strip().casefold()
        if not et:
            continue

        # Goal rows are used for scoring coverage + GT/GW derivation + on-ice goal stats.
        if et == "goal":
            has_any_goal_row.add(int(gid))
            goal_rows_by_game.setdefault(int(gid), []).append(r0)

        # Track per-game availability for "measured" stats.
        if et in {"goal", "assist"}:
            scoring_game_ids.add(int(gid))
        if et in {"xg", "expectedgoal", "sog", "shotongoal", "shot"}:
            shot_game_ids.add(int(gid))
        if et == "penalty":
            pim_game_ids.add(int(gid))
        if et == "completedpass":
            completed_pass_game_ids.add(int(gid))
        if et == "giveaway":
            giveaway_game_ids.add(int(gid))
        if et == "turnoversforced":
            turnovers_forced_game_ids.add(int(gid))
        if et == "createdturnovers":
            created_turnovers_game_ids.add(int(gid))
        if et == "takeaway":
            takeaway_game_ids.add(int(gid))

        # On-ice goal coverage: any goal row that has both home/away on-ice lists.
        if et == "goal":
            home_raw = str(r0.get("on_ice_players_home") or "").strip()
            away_raw = str(r0.get("on_ice_players_away") or "").strip()
            if home_raw and away_raw:
                on_ice_goal_game_ids.add(int(gid))

        # Controlled entry/exit on-ice coverage: any row with on-ice list for our side.
        if et in {"controlledentry", "controlledexit"}:
            side = side_by_game_id.get(int(gid))
            if side == "home":
                raw = str(r0.get("on_ice_players_home") or "").strip()
            elif side == "away":
                raw = str(r0.get("on_ice_players_away") or "").strip()
            else:
                raw = ""
            if raw:
                if et == "controlledentry":
                    ce_game_ids.add(int(gid))
                else:
                    cx_game_ids.add(int(gid))

        # Direct per-player counting for our roster.
        pid_raw = _int_or_none(r0.get("player_id"))
        pid: Optional[int] = None
        if pid_raw is not None and pid_raw > 0 and pid_raw in roster_by_pid:
            pid = int(pid_raw)
        elif et in {
            "goal",
            "assist",
            "xg",
            "expectedgoal",
            "sog",
            "shotongoal",
            "shot",
            "penalty",
            "completedpass",
            "giveaway",
            "turnoversforced",
            "createdturnovers",
            "takeaway",
        }:
            # Jersey-based fallback for rows with unresolved player_id.
            ev_tid = _event_team_id(r0)
            if ev_tid is not None and int(ev_tid) == int(team_id):
                jerseys = _extract_numbers(r0.get("attributed_jerseys"))
                if len(jerseys) == 1:
                    pid = _pick_pid_for_jersey(str(jerseys[0]), prefer_goalie=False)

        if pid is None:
            continue

        key = (int(pid), int(gid))
        acc = direct_counts.setdefault(key, {})
        if et == "goal":
            acc["goals"] = acc.get("goals", 0) + 1
            per = _int_or_none(r0.get("period")) or 0
            if int(per) >= 4:
                ot_goals_by_pid_gid[key] = ot_goals_by_pid_gid.get(key, 0) + 1
        elif et == "assist":
            acc["assists"] = acc.get("assists", 0) + 1
            per = _int_or_none(r0.get("period")) or 0
            if int(per) >= 4:
                ot_assists_by_pid_gid[key] = ot_assists_by_pid_gid.get(key, 0) + 1
        elif et in {"xg", "expectedgoal"}:
            acc["xg"] = acc.get("xg", 0) + 1
        elif et in {"sog", "shotongoal"}:
            acc["sog"] = acc.get("sog", 0) + 1
        elif et == "shot":
            acc["shot"] = acc.get("shot", 0) + 1
        elif et == "completedpass":
            acc["completed_passes"] = acc.get("completed_passes", 0) + 1
        elif et == "giveaway":
            acc["giveaways"] = acc.get("giveaways", 0) + 1
        elif et == "turnoversforced":
            acc["turnovers_forced"] = acc.get("turnovers_forced", 0) + 1
        elif et == "createdturnovers":
            acc["created_turnovers"] = acc.get("created_turnovers", 0) + 1
        elif et == "takeaway":
            acc["takeaways"] = acc.get("takeaways", 0) + 1
        elif et == "penalty":
            mins = _parse_pim_minutes(r0.get("details"))
            if mins > 0:
                acc["pim"] = acc.get("pim", 0) + int(mins)

    # TimeToScore-linked games: treat PIM as "covered" even if 0 penalties were recorded.
    try:
        t2s_game_ids = set(
            m.HkyGame.objects.filter(
                id__in=[int(x) for x in game_ids], timetoscore_game_id__isnull=False
            ).values_list("id", flat=True)
        )
    except Exception:
        t2s_game_ids = set()
    pim_game_ids |= {int(gid) for gid in t2s_game_ids or []}

    # ----------------------------
    # Determine per-game participation ("played") for GP/denominators.
    # ----------------------------
    roster_link_pairs: set[tuple[int, int]] = set()
    shift_pairs: set[tuple[int, int]] = set()
    shift_game_ids: set[int] = set()
    shifts_by_gid_period: dict[int, dict[int, list[tuple[int, int, int]]]] = {}

    # Game roster (TimeToScore rosters + goals.xlsx rosters uploaded via shift_package).
    try:
        roster_link_pairs = {
            (int(pid), int(gid))
            for pid, gid in m.HkyGamePlayer.objects.filter(
                game_id__in=game_ids,
                team_id=int(team_id),
                player_id__isnull=False,
            )
            .values_list("player_id", "game_id")
            .distinct()
        }
    except Exception:
        roster_link_pairs = set()

    # Shift presence + shift intervals (for skater GP and on-ice GF/GA/+/-).
    try:
        for r0 in (
            m.HkyGameShiftRow.objects.filter(
                game_id__in=game_ids,
                team_id=int(team_id),
                player_id__isnull=False,
            )
            .exclude(period__isnull=True)
            .exclude(game_seconds__isnull=True)
            .exclude(game_seconds_end__isnull=True)
            .values("game_id", "player_id", "period", "game_seconds", "game_seconds_end")
        ):
            try:
                gid = int(r0.get("game_id") or 0)
                pid = int(r0.get("player_id") or 0)
                per = int(r0.get("period") or 0)
                gs0 = int(r0.get("game_seconds") or 0)
                gs1 = int(r0.get("game_seconds_end") or 0)
            except Exception:
                continue
            if gid <= 0 or pid <= 0 or per <= 0:
                continue
            shift_game_ids.add(int(gid))
            if pid not in roster_by_pid:
                continue
            shift_pairs.add((int(pid), int(gid)))
            shifts_by_gid_period.setdefault(int(gid), {}).setdefault(int(per), []).append(
                (int(pid), int(gs0), int(gs1))
            )
    except Exception:
        shift_pairs = set()
        shift_game_ids = set()
        shifts_by_gid_period = {}

    direct_event_pairs = {(int(pid), int(gid)) for (pid, gid) in (direct_counts or {}).keys()}
    on_ice_pairs: set[tuple[int, int]] = set()
    for r0 in event_rows or []:
        gid = _int_or_none(r0.get("game_id")) or 0
        if gid <= 0 or gid not in games_by_id:
            continue
        my_side = side_by_game_id.get(int(gid))
        if my_side not in {"home", "away"}:
            continue
        raw = (
            str(r0.get("on_ice_players_home") or "").strip()
            if my_side == "home"
            else str(r0.get("on_ice_players_away") or "").strip()
        )
        if not raw:
            continue
        for nm, jn in _parse_on_ice_list(raw):
            pid = _pid_for_name_jersey(nm, jn)
            if pid is None:
                continue
            on_ice_pairs.add((int(pid), int(gid)))

    played_pairs: set[tuple[int, int]] = set()
    for gid in game_ids:
        has_shift_data = int(gid) in shift_game_ids
        for pid, p0 in roster_by_pid.items():
            pos = str(p0.get("position") or "").strip().upper()
            is_goalie = pos == "G"
            key = (int(pid), int(gid))
            has_shift = key in shift_pairs
            in_roster = key in roster_link_pairs
            has_event = key in direct_event_pairs
            has_on_ice = key in on_ice_pairs

            if not is_goalie and has_shift_data:
                # Primary/long spreadsheet semantics: skaters "played" when they have at least one shift.
                # If shift rows are incomplete, allow an attributed event row to imply participation.
                if has_shift or has_event or has_on_ice:
                    played_pairs.add(key)
            else:
                # No shift data for this team/game: fall back to the per-game roster (T2S/goals.xlsx).
                if in_roster or has_shift or has_event or has_on_ice:
                    played_pairs.add(key)

    # ----------------------------
    # Create per-player/per-game rows (one row per played game).
    # ----------------------------
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for gid in game_ids:
        if gid not in games_by_id:
            continue
        for p in roster_players or []:
            try:
                pid = int(p.get("id") or 0)
            except Exception:
                continue
            if pid <= 0:
                continue
            if (int(pid), int(gid)) not in played_pairs:
                continue
            r: dict[str, Any] = {"player_id": int(pid), "game_id": int(gid)}
            for k in logic.PLAYER_STATS_SUM_KEYS:
                r[str(k)] = None
            rows.append(r)
            by_key[(int(pid), int(gid))] = r

    # Fill direct counts for players with events.
    for (pid, gid), c in (direct_counts or {}).items():
        row = by_key.get((int(pid), int(gid)))
        if not row:
            continue
        if "goals" in c:
            row["goals"] = int(c.get("goals") or 0)
        if "assists" in c:
            row["assists"] = int(c.get("assists") or 0)
        if "completed_passes" in c:
            row["completed_passes"] = int(c.get("completed_passes") or 0)
        if "giveaways" in c:
            row["giveaways"] = int(c.get("giveaways") or 0)
        if "turnovers_forced" in c:
            row["turnovers_forced"] = int(c.get("turnovers_forced") or 0)
        if "created_turnovers" in c:
            row["created_turnovers"] = int(c.get("created_turnovers") or 0)
        if "takeaways" in c:
            row["takeaways"] = int(c.get("takeaways") or 0)
        if "pim" in c:
            row["pim"] = int(c.get("pim") or 0)

        # Shot-tracking is only "covered" in some games; compute derived counts when covered.
        if int(gid) in shot_game_ids:
            goals = int(c.get("goals") or 0)
            xg = int(c.get("xg") or 0)
            sog = int(c.get("sog") or 0)
            shots = int(c.get("shot") or 0)
            expected_goals = int(goals) + int(xg)
            sog_total = int(expected_goals) + int(sog)
            shots_total = int(sog_total) + int(shots)
            row["expected_goals"] = int(expected_goals)
            row["sog"] = int(sog_total)
            row["shots"] = int(shots_total)

        # OT goals/assists.
        if int(gid) in scoring_game_ids:
            og = int(ot_goals_by_pid_gid.get((int(pid), int(gid)), 0) or 0)
            oa = int(ot_assists_by_pid_gid.get((int(pid), int(gid)), 0) or 0)
            if og:
                row["ot_goals"] = int(og)
            if oa:
                row["ot_assists"] = int(oa)

    # Backfill zeros for covered stat groups so denominators are stable.
    for (pid, gid), row in by_key.items():
        if gid in scoring_game_ids:
            if row.get("goals") is None:
                row["goals"] = 0
            if row.get("assists") is None:
                row["assists"] = 0
            if row.get("ot_goals") is None:
                row["ot_goals"] = 0
            if row.get("ot_assists") is None:
                row["ot_assists"] = 0
        if gid in pim_game_ids and row.get("pim") is None:
            row["pim"] = 0
        if gid in completed_pass_game_ids and row.get("completed_passes") is None:
            row["completed_passes"] = 0
        if gid in giveaway_game_ids and row.get("giveaways") is None:
            row["giveaways"] = 0
        if gid in turnovers_forced_game_ids and row.get("turnovers_forced") is None:
            row["turnovers_forced"] = 0
        if gid in created_turnovers_game_ids and row.get("created_turnovers") is None:
            row["created_turnovers"] = 0
        if gid in takeaway_game_ids and row.get("takeaways") is None:
            row["takeaways"] = 0
        if gid in shot_game_ids:
            if row.get("expected_goals") is None:
                row["expected_goals"] = 0
            if row.get("sog") is None:
                row["sog"] = 0
            if row.get("shots") is None:
                row["shots"] = 0

    # ----------------------------
    # On-ice goal stats (GF/GA and +/-).
    #
    # Prefer shift-derived on-ice attribution when shift rows exist for this team/game, even when
    # the goal event rows lack explicit on-ice lists (e.g. TimeToScore-only goal streams).
    # ----------------------------
    def _goal_seconds(row: dict[str, Any]) -> Optional[int]:
        gs = _int_or_none(row.get("game_seconds"))
        if gs is not None:
            return int(gs)
        return logic.parse_duration_seconds(row.get("game_time"))

    def _shift_contains(*, t: int, start: int, end: int) -> bool:
        lo = start if start <= end else end
        hi = end if start <= end else start
        if t < lo or t > hi:
            return False
        # Match scripts/parse_stats_inputs.py: don't count goals exactly at shift start.
        return int(t) != int(start)

    for gid in sorted(list(shift_game_ids)):
        g2 = games_by_id.get(int(gid)) or {}
        team1_id = _int_or_none(g2.get("team1_id")) or 0
        team2_id = _int_or_none(g2.get("team2_id")) or 0
        if team1_id <= 0 or team2_id <= 0:
            continue

        goals_all = list(goal_rows_by_game.get(int(gid), []) or [])
        # Prefer TimeToScore goals when present to avoid duplicate goal streams.
        if any("timetoscore" in str(r.get("source") or "").casefold() for r in goals_all):
            goals_all = [
                r for r in goals_all if "timetoscore" in str(r.get("source") or "").casefold()
            ]

        gf_by_pid: dict[int, int] = {}
        ga_by_pid: dict[int, int] = {}

        goals_seen = 0
        goals_counted = 0
        for gr in goals_all:
            scoring_tid = _event_team_id(gr)
            if scoring_tid not in {team1_id, team2_id}:
                continue
            per = _int_or_none(gr.get("period"))
            t = _goal_seconds(gr)
            if per is None or int(per) <= 0 or t is None:
                continue
            goals_seen += 1

            period_shifts = shifts_by_gid_period.get(int(gid), {}).get(int(per), []) or []
            on_ice: set[int] = set()
            for pid, start, end in period_shifts:
                if _shift_contains(t=int(t), start=int(start), end=int(end)):
                    on_ice.add(int(pid))

            if not on_ice:
                continue
            goals_counted += 1
            is_for_us = bool(int(scoring_tid) == int(team_id))
            if is_for_us:
                for pid in on_ice:
                    gf_by_pid[int(pid)] = gf_by_pid.get(int(pid), 0) + 1
            else:
                for pid in on_ice:
                    ga_by_pid[int(pid)] = ga_by_pid.get(int(pid), 0) + 1

        # If we have goal events but couldn't map any of them to on-ice shifts, treat the
        # on-ice goal stats as unknown for this game.
        if goals_seen > 0 and goals_counted <= 0:
            continue

        # Only apply on-ice stats to players that have shift rows in this game.
        for (pid, g0), row in list(by_key.items()):
            if int(g0) != int(gid):
                continue
            if (int(pid), int(gid)) not in shift_pairs:
                continue
            gf = int(gf_by_pid.get(int(pid), 0) or 0)
            ga = int(ga_by_pid.get(int(pid), 0) or 0)
            row["gf_counted"] = int(gf)
            row["ga_counted"] = int(ga)
            row["plus_minus"] = int(gf) - int(ga)

    # Fallback for games without shift rows: use goal event on-ice lists when present.
    for gid, goals in (goal_rows_by_game or {}).items():
        if int(gid) in shift_game_ids:
            continue
        g2 = games_by_id.get(int(gid)) or {}
        my_side = side_by_game_id.get(int(gid))
        if my_side not in {"home", "away"}:
            continue
        team1_id = _int_or_none(g2.get("team1_id")) or 0
        team2_id = _int_or_none(g2.get("team2_id")) or 0
        if team1_id <= 0 or team2_id <= 0:
            continue

        total_goals = 0
        complete_goals = 0
        gf_by_pid: dict[int, int] = {}
        ga_by_pid: dict[int, int] = {}

        for r0 in goals or []:
            scoring_tid = _event_team_id(r0)
            if scoring_tid not in {team1_id, team2_id}:
                continue
            total_goals += 1

            home_raw = str(r0.get("on_ice_players_home") or "").strip()
            away_raw = str(r0.get("on_ice_players_away") or "").strip()
            if not home_raw or not away_raw:
                continue
            complete_goals += 1

            my_raw = home_raw if my_side == "home" else away_raw
            for nm, jn in _parse_on_ice_list(my_raw):
                pid = _pid_for_name_jersey(nm, jn)
                if pid is None:
                    continue
                if scoring_tid == int(team_id):
                    gf_by_pid[int(pid)] = gf_by_pid.get(int(pid), 0) + 1
                else:
                    ga_by_pid[int(pid)] = ga_by_pid.get(int(pid), 0) + 1

        if total_goals <= 0 or complete_goals <= 0:
            continue

        on_ice_goal_game_ids.add(int(gid))
        if complete_goals == total_goals:
            on_ice_goal_complete_game_ids.add(int(gid))

        for p in roster_players or []:
            try:
                pid = int(p.get("id") or 0)
            except Exception:
                continue
            if pid <= 0:
                continue
            row = by_key.get((int(pid), int(gid)))
            if not row:
                continue
            gf = int(gf_by_pid.get(int(pid), 0) or 0)
            ga = int(ga_by_pid.get(int(pid), 0) or 0)
            row["gf_counted"] = int(gf)
            row["ga_counted"] = int(ga)
            if int(gid) in on_ice_goal_complete_game_ids:
                row["plus_minus"] = int(gf) - int(ga)

    # ----------------------------
    # On-ice controlled entry/exit stats.
    # ----------------------------
    for r0 in event_rows or []:
        et = str(r0.get("event_type__key") or "").strip().casefold()
        if et not in {"controlledentry", "controlledexit"}:
            continue
        gid = _int_or_none(r0.get("game_id")) or 0
        if gid <= 0 or gid not in games_by_id:
            continue
        my_side = side_by_game_id.get(int(gid))
        if my_side not in {"home", "away"}:
            continue
        raw = (
            str(r0.get("on_ice_players_home") or "").strip()
            if my_side == "home"
            else str(r0.get("on_ice_players_away") or "").strip()
        )
        if not raw:
            continue
        ev_tid = _event_team_id(r0)
        is_for_us = bool(ev_tid is not None and int(ev_tid) == int(team_id))
        for nm, jn in _parse_on_ice_list(raw):
            pid = _pid_for_name_jersey(nm, jn)
            if pid is None:
                continue
            row = by_key.get((int(pid), int(gid)))
            if not row:
                continue
            if et == "controlledentry":
                key = "controlled_entry_for" if is_for_us else "controlled_entry_against"
            else:
                key = "controlled_exit_for" if is_for_us else "controlled_exit_against"
            row[key] = logic._int0(row.get(key)) + 1

    # Default 0 for on-ice CE/CX in covered games.
    for (pid, gid), row in by_key.items():
        if gid in ce_game_ids:
            if row.get("controlled_entry_for") is None:
                row["controlled_entry_for"] = 0
            if row.get("controlled_entry_against") is None:
                row["controlled_entry_against"] = 0
        if gid in cx_game_ids:
            if row.get("controlled_exit_for") is None:
                row["controlled_exit_for"] = 0
            if row.get("controlled_exit_against") is None:
                row["controlled_exit_against"] = 0

    # Shift-derived TOI/shifts.
    if show_shift_data:
        try:
            _overlay_shift_stats_into_player_stat_rows(
                game_ids=list(game_ids), player_stats_rows=rows
            )
        except Exception:
            logger.exception(
                "Failed to overlay shift stats into event-derived player stat rows for game_ids=%s",
                game_ids,
            )

    return rows


def _compute_player_has_events_by_pid_for_game(
    *,
    events_rows: list[dict[str, Any]],
    team1_players: list[dict[str, Any]],
    team2_players: list[dict[str, Any]],
) -> dict[int, bool]:
    """
    Best-effort: mark which players have any underlying events for the Player Events popup.
    """
    player_has_events_by_pid: dict[int, bool] = {}

    home_by_jersey: dict[str, list[int]] = {}
    for p in team1_players or []:
        try:
            pid_i = int(p.get("id") or 0)
        except Exception:
            continue
        if pid_i <= 0:
            continue
        player_has_events_by_pid[pid_i] = False
        j = logic.normalize_jersey_number(p.get("jersey_number"))
        if j:
            home_by_jersey.setdefault(j, []).append(pid_i)

    away_by_jersey: dict[str, list[int]] = {}
    for p in team2_players or []:
        try:
            pid_i = int(p.get("id") or 0)
        except Exception:
            continue
        if pid_i <= 0:
            continue
        player_has_events_by_pid[pid_i] = False
        j = logic.normalize_jersey_number(p.get("jersey_number"))
        if j:
            away_by_jersey.setdefault(j, []).append(pid_i)

    def _num_set(raw: Any) -> set[str]:
        if not raw:
            return set()
        return {str(int(x)) for x in re.findall(r"([0-9]+)", str(raw))}

    for row in events_rows or []:
        side_raw = (
            str(
                row.get("Team Side")
                or row.get("TeamSide")
                or row.get("Team Rel")
                or row.get("TeamRel")
                or row.get("Team")
                or ""
            )
            .strip()
            .lower()
        )
        side = ""
        if side_raw in {"home", "team1"}:
            side = "home"
        elif side_raw in {"away", "team2"}:
            side = "away"

        # Attributed events (team-side specific).
        attr = _num_set(row.get("Attributed Jerseys") or row.get("AttributedJerseys"))
        if attr and side:
            jmap = home_by_jersey if side == "home" else away_by_jersey
            for j in attr:
                for pid_i in jmap.get(j, []):
                    player_has_events_by_pid[int(pid_i)] = True

        # On-ice columns can be split Home/Away or a single legacy column.
        on_home = row.get("On-Ice Players (Home)") or row.get("On-Ice Players Home")
        on_away = row.get("On-Ice Players (Away)") or row.get("On-Ice Players Away")
        if on_home:
            for j in _num_set(on_home):
                for pid_i in home_by_jersey.get(j, []):
                    player_has_events_by_pid[int(pid_i)] = True
        if on_away:
            for j in _num_set(on_away):
                for pid_i in away_by_jersey.get(j, []):
                    player_has_events_by_pid[int(pid_i)] = True

        if not on_home and not on_away and side:
            on_generic = row.get("On-Ice Players") or row.get("On-Ice Players (PM)")
            if on_generic:
                jmap = home_by_jersey if side == "home" else away_by_jersey
                for j in _num_set(on_generic):
                    for pid_i in jmap.get(j, []):
                        player_has_events_by_pid[int(pid_i)] = True

    return player_has_events_by_pid


# ----------------------------
# Landing/auth
# ----------------------------


def index(request: HttpRequest) -> HttpResponse:
    if _session_user_id(request):
        return redirect("/games")
    return render(request, "index.html")


def register(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        password = str(request.POST.get("password") or "")
        name = str(request.POST.get("name") or "").strip()
        if not email or not password:
            messages.error(request, "Email and password are required")
        elif logic.get_user_by_email(email):
            messages.error(request, "Email already registered")
        else:
            uid = logic.create_user(email, password, name)
            request.session["user_id"] = int(uid)
            request.session["user_email"] = email
            request.session["user_name"] = name
            try:
                logic.send_email(
                    to_addr=email,
                    subject="Welcome to HockeyMOM",
                    body=(
                        f"Hello {name or email},\n\n"
                        "Your account has been created successfully.\n"
                        "You can now create a game, upload files, and run jobs.\n\n"
                        "Regards,\nHockeyMOM"
                    ),
                )
            except Exception:
                pass
            return redirect("/games")
    return render(request, "register.html")


def login_view(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        password = str(request.POST.get("password") or "")
        u = logic.get_user_by_email(email)
        if not u or not check_password_hash(str(u.get("password_hash") or ""), password):
            messages.error(request, "Invalid credentials")
        else:
            request.session["user_id"] = int(u["id"])
            request.session["user_email"] = str(u["email"])
            request.session["user_name"] = str(u.get("name") or u["email"])
            return redirect("/games")
    return render(request, "login.html")


def forgot_password(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        try:
            _django_orm, m = _orm_modules()
            u = m.User.objects.filter(email=email).values("id").first()
            if u and u.get("id"):
                token = secrets.token_urlsafe(32)
                now = dt.datetime.now()
                m.Reset.objects.create(
                    user_id=int(u["id"]),
                    token=token,
                    expires_at=(now + dt.timedelta(hours=1)),
                    created_at=now,
                )
                link = request.build_absolute_uri(f"/reset/{token}")
                logic.send_email(
                    to_addr=email,
                    subject="Password reset",
                    body=(
                        "We received a request to reset your password.\n\n"
                        f"Use this link within 1 hour: {link}\n\n"
                        "If you did not request this, you can ignore this message."
                    ),
                )
        except Exception:
            pass
        messages.success(request, "If the account exists, a reset email has been sent.")
        return redirect("/login")
    return render(request, "forgot_password.html")


def reset_password(request: HttpRequest, token: str) -> HttpResponse:
    from django.contrib import messages

    _django_orm, m = _orm_modules()
    row = (
        m.Reset.objects.select_related("user")
        .filter(token=str(token))
        .values("id", "user_id", "token", "expires_at", "used_at", "user__email")
        .first()
    )
    if not row:
        messages.error(request, "Invalid or expired token")
        return redirect("/login")
    now = dt.datetime.now()
    expires_raw = row.get("expires_at")
    expires = (
        expires_raw
        if isinstance(expires_raw, dt.datetime)
        else dt.datetime.fromisoformat(str(expires_raw))
    )
    if row.get("used_at") or now > expires:
        messages.error(request, "Invalid or expired token")
        return redirect("/login")
    if request.method == "POST":
        pw1 = str(request.POST.get("password") or "")
        pw2 = str(request.POST.get("password2") or "")
        if not pw1 or pw1 != pw2:
            messages.error(request, "Passwords do not match")
            return render(request, "reset_password.html")
        newhash = generate_password_hash(pw1)
        from django.db import transaction

        now2 = dt.datetime.now()
        with transaction.atomic():
            m.User.objects.filter(id=int(row["user_id"])).update(password_hash=newhash)
            m.Reset.objects.filter(id=int(row["id"])).update(used_at=now2)
        messages.success(request, "Password updated. Please log in.")
        return redirect("/login")
    return render(request, "reset_password.html")


def logout_view(request: HttpRequest) -> HttpResponse:
    try:
        request.session.flush()
    except Exception:
        request.session.clear()
    return redirect("/")


def league_select(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    lid = str(request.POST.get("league_id") or "").strip()
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    uid = _session_user_id(request)
    if lid and lid.isdigit():
        lid_i = int(lid)
        ok = (
            m.League.objects.filter(id=lid_i)
            .filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
            .exists()
        )
        if ok:
            request.session["league_id"] = lid_i
            m.User.objects.filter(id=uid).update(default_league_id=lid_i)
    else:
        request.session.pop("league_id", None)
        m.User.objects.filter(id=uid).update(default_league=None)
    referer = str(request.META.get("HTTP_REFERER") or "").strip()
    return redirect(referer or "/")


# ----------------------------
# API
# ----------------------------


@csrf_exempt
def api_user_video_clip_len(request: HttpRequest) -> JsonResponse:
    if not _session_user_id(request):
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    payload = _json_body(request)
    raw = payload.get("clip_len_s")
    try:
        v = int(raw)
    except Exception:
        return JsonResponse(
            {"ok": False, "error": "clip_len_s must be one of: 15, 20, 30, 45, 60, 90"}, status=400
        )
    if v not in {15, 20, 30, 45, 60, 90}:
        return JsonResponse(
            {"ok": False, "error": "clip_len_s must be one of: 15, 20, 30, 45, 60, 90"}, status=400
        )
    try:
        _django_orm, m = _orm_modules()
        m.User.objects.filter(id=_session_user_id(request)).update(video_clip_len_s=int(v))
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "clip_len_s": int(v)})


@csrf_exempt
def api_export_table_xlsx(request: HttpRequest) -> HttpResponse:
    """
    Export a rendered HTML table as an XLSX spreadsheet.

    This is a stateless endpoint: callers POST the table headers/rows and receive a styled XLSX.
    Shift/TOI columns are always removed from exports (privacy/sensitivity).
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)
    try:
        payload = json.loads(request.body or b"{}")
    except Exception:
        return JsonResponse({"ok": False, "error": "invalid_json"}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "invalid_payload"}, status=400)

    title = str(payload.get("title") or payload.get("sheet_name") or "Table").strip() or "Table"
    sheet_name = str(payload.get("sheet_name") or title).strip() or title
    filename = str(payload.get("filename") or title).strip() or title

    headers_raw = payload.get("headers") or []
    if not isinstance(headers_raw, list):
        return JsonResponse({"ok": False, "error": "invalid_headers"}, status=400)
    headers = [str(h or "") for h in headers_raw]

    rows_raw = payload.get("rows") or []
    if not isinstance(rows_raw, list):
        return JsonResponse({"ok": False, "error": "invalid_rows"}, status=400)
    rows: list[list[str]] = []
    for r in rows_raw:
        if not isinstance(r, list):
            continue
        rows.append([str(v or "") for v in r])

    col_keys_raw = payload.get("col_keys")
    col_keys = None
    if isinstance(col_keys_raw, list):
        col_keys = [str(k).strip() if k is not None else None for k in col_keys_raw]

    try:
        freeze_cols = int(payload.get("freeze_cols") or 0)
    except Exception:
        freeze_cols = 0
    if freeze_cols < 0:
        freeze_cols = 0

    # Safety guardrails (avoid pathological payload sizes).
    if len(headers) > 250:
        return JsonResponse({"ok": False, "error": "too_many_columns"}, status=400)
    if len(rows) > 50_000:
        return JsonResponse({"ok": False, "error": "too_many_rows"}, status=400)

    try:
        from .xlsx_export import export_table_xlsx

        out_name, out_bytes = export_table_xlsx(
            title=title,
            headers=headers,
            rows=rows,
            col_keys=col_keys,
            freeze_cols=int(freeze_cols),
            sheet_name=sheet_name,
            filename=filename,
        )
    except Exception:
        logger.exception("api_export_table_xlsx failed")
        return JsonResponse({"ok": False, "error": "xlsx_export_failed"}, status=500)

    resp = HttpResponse(
        out_bytes,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = f'attachment; filename="{out_name}"'
    return resp


def api_league_page_views(request: HttpRequest, league_id: int) -> JsonResponse:
    r = _require_login(request)
    if r:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    user_id = _session_user_id(request)
    _django_orm, m = _orm_modules()
    owner_id = (
        m.League.objects.filter(id=int(league_id)).values_list("owner_user_id", flat=True).first()
    )
    owner_id_i = int(owner_id) if owner_id is not None else None
    if owner_id_i is None:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    is_admin = False
    try:
        is_admin = bool(_is_league_admin(int(league_id), int(user_id)))
    except Exception:
        is_admin = False
    if int(owner_id_i) != int(user_id) and not is_admin:
        return JsonResponse({"ok": False, "error": "not_authorized"}, status=403)

    kind = str(request.GET.get("kind") or "").strip()
    entity_id_raw = request.GET.get("entity_id")
    try:
        entity_id = int(str(entity_id_raw or "0").strip() or "0")
    except Exception:
        entity_id = 0
    try:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=kind, entity_id=entity_id
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    baseline_count = logic._get_league_page_view_baseline_count(
        None, int(league_id), kind=kind, entity_id=entity_id
    )
    delta_count = None
    if baseline_count is not None:
        try:
            delta_count = max(0, int(count) - int(baseline_count))
        except Exception:
            delta_count = 0
    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "kind": kind,
            "entity_id": int(entity_id),
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }
    )


def api_league_page_views_mark(request: HttpRequest, league_id: int) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    r = _require_login(request)
    if r:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    user_id = _session_user_id(request)
    _django_orm, m = _orm_modules()
    owner_id = (
        m.League.objects.filter(id=int(league_id)).values_list("owner_user_id", flat=True).first()
    )
    owner_id_i = int(owner_id) if owner_id is not None else None
    if owner_id_i is None:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    is_admin = False
    try:
        is_admin = bool(_is_league_admin(int(league_id), int(user_id)))
    except Exception:
        is_admin = False
    if int(owner_id_i) != int(user_id) and not is_admin:
        return JsonResponse({"ok": False, "error": "not_authorized"}, status=403)

    kind = str(request.POST.get("kind") or "").strip()
    entity_id_raw = request.POST.get("entity_id")
    try:
        entity_id = int(str(entity_id_raw or "0").strip() or "0")
    except Exception:
        entity_id = 0

    try:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=kind, entity_id=entity_id
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    try:
        ok = logic._set_league_page_view_baseline_count(
            None, int(league_id), kind=kind, entity_id=entity_id, baseline_count=int(count)
        )
    except Exception:
        ok = False
    if not ok:
        return JsonResponse({"ok": False, "error": "failed_to_mark"}, status=500)

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "kind": kind,
            "entity_id": int(entity_id),
            "count": int(count),
            "baseline_count": int(count),
            "delta_count": 0,
        }
    )


def _parse_entity_ids_csv(raw: Any, *, max_ids: int = 500) -> list[int]:
    if raw is None:
        return []
    s = str(raw or "").strip()
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        t = str(tok or "").strip()
        if not t:
            continue
        try:
            v = int(t)
        except Exception:
            continue
        if v <= 0:
            continue
        out.append(int(v))
        if len(out) >= int(max_ids):
            break
    # Dedupe preserving order.
    seen: set[int] = set()
    uniq: list[int] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def _validate_league_page_view_entity_ids(
    *,
    m,
    league_id: int,
    kind: str,
    entity_ids: list[int],
) -> bool:
    """
    Ensure all entity_ids belong to the league for the given kind.

    This prevents querying/marking view counts for cross-league entities via batch APIs.
    """
    kind_s = str(kind or "").strip()
    ids = [int(x) for x in (entity_ids or []) if int(x) > 0]
    if not ids:
        return True
    if kind_s in {logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE}:
        return True
    if kind_s == logic.LEAGUE_PAGE_VIEW_KIND_TEAM:
        ok_ids = set(
            m.LeagueTeam.objects.filter(
                league_id=int(league_id), team_id__in=list(ids)
            ).values_list("team_id", flat=True)
        )
        return len(ok_ids) == len(set(ids))
    if kind_s == logic.LEAGUE_PAGE_VIEW_KIND_GAME:
        ok_ids = set(
            m.LeagueGame.objects.filter(
                league_id=int(league_id), game_id__in=list(ids)
            ).values_list("game_id", flat=True)
        )
        return len(ok_ids) == len(set(ids))
    if kind_s == logic.LEAGUE_PAGE_VIEW_KIND_PLAYER_EVENTS:
        rows = list(m.Player.objects.filter(id__in=list(ids)).values_list("id", "team_id"))
        team_ids = {int(tid) for _pid, tid in rows if tid is not None}
        ok_team_ids = set(
            m.LeagueTeam.objects.filter(
                league_id=int(league_id), team_id__in=list(team_ids)
            ).values_list("team_id", flat=True)
        )
        for _pid, tid in rows:
            if int(tid) not in ok_team_ids:
                return False
        return True
    if kind_s == logic.LEAGUE_PAGE_VIEW_KIND_EVENT_CLIP:
        rows = list(m.HkyGameEventRow.objects.filter(id__in=list(ids)).values_list("id", "game_id"))
        game_ids = {int(gid) for _eid, gid in rows if gid is not None}
        ok_game_ids = set(
            m.LeagueGame.objects.filter(
                league_id=int(league_id), game_id__in=list(game_ids)
            ).values_list("game_id", flat=True)
        )
        for _eid, gid in rows:
            if int(gid) not in ok_game_ids:
                return False
        return True
    return False


def api_league_page_views_batch(request: HttpRequest, league_id: int) -> JsonResponse:
    r = _require_login(request)
    if r:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    user_id = _session_user_id(request)
    _django_orm, m = _orm_modules()
    owner_id = (
        m.League.objects.filter(id=int(league_id)).values_list("owner_user_id", flat=True).first()
    )
    owner_id_i = int(owner_id) if owner_id is not None else None
    if owner_id_i is None:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    is_admin = False
    try:
        is_admin = bool(_is_league_admin(int(league_id), int(user_id)))
    except Exception:
        is_admin = False
    if int(owner_id_i) != int(user_id) and not is_admin:
        return JsonResponse({"ok": False, "error": "not_authorized"}, status=403)

    kind = str(request.GET.get("kind") or "").strip()
    entity_ids = _parse_entity_ids_csv(request.GET.get("entity_ids") or request.GET.get("ids"))
    if not entity_ids:
        entity_ids = _parse_entity_ids_csv(request.GET.get("entity_id"))
    if not entity_ids:
        return JsonResponse({"ok": False, "error": "entity_ids_required"}, status=400)

    try:
        kind_canon, _ = logic._canon_league_page_view_kind_entity(
            kind=kind, entity_id=int(entity_ids[0])
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    if not _validate_league_page_view_entity_ids(
        m=m, league_id=int(league_id), kind=str(kind_canon), entity_ids=list(entity_ids or [])
    ):
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    view_counts: dict[int, int] = {}
    for eid, vc in m.LeaguePageView.objects.filter(
        league_id=int(league_id), page_kind=str(kind_canon), entity_id__in=list(entity_ids)
    ).values_list("entity_id", "view_count"):
        try:
            view_counts[int(eid)] = int(vc or 0)
        except Exception:
            view_counts[int(eid)] = 0

    baseline_counts: dict[int, int] = {}
    for eid, bc in m.LeaguePageViewBaseline.objects.filter(
        league_id=int(league_id), page_kind=str(kind_canon), entity_id__in=list(entity_ids)
    ).values_list("entity_id", "baseline_count"):
        try:
            baseline_counts[int(eid)] = int(bc or 0)
        except Exception:
            baseline_counts[int(eid)] = 0

    results: dict[str, Any] = {}
    for eid in entity_ids:
        count = int(view_counts.get(int(eid), 0))
        baseline = baseline_counts.get(int(eid))
        delta = None
        if baseline is not None:
            try:
                delta = max(0, int(count) - int(baseline))
            except Exception:
                delta = 0
        results[str(int(eid))] = {
            "entity_id": int(eid),
            "count": int(count),
            "baseline_count": int(baseline) if baseline is not None else None,
            "delta_count": int(delta) if delta is not None else None,
        }
    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "kind": kind_canon, "results": results}
    )


def api_league_page_views_mark_batch(request: HttpRequest, league_id: int) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    r = _require_login(request)
    if r:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    user_id = _session_user_id(request)
    _django_orm, m = _orm_modules()
    owner_id = (
        m.League.objects.filter(id=int(league_id)).values_list("owner_user_id", flat=True).first()
    )
    owner_id_i = int(owner_id) if owner_id is not None else None
    if owner_id_i is None:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    is_admin = False
    try:
        is_admin = bool(_is_league_admin(int(league_id), int(user_id)))
    except Exception:
        is_admin = False
    if int(owner_id_i) != int(user_id) and not is_admin:
        return JsonResponse({"ok": False, "error": "not_authorized"}, status=403)

    kind = str(request.POST.get("kind") or "").strip()
    entity_ids = _parse_entity_ids_csv(request.POST.get("entity_ids") or request.POST.get("ids"))
    if not entity_ids:
        entity_ids = _parse_entity_ids_csv(request.POST.get("entity_id"))
    if not entity_ids:
        return JsonResponse({"ok": False, "error": "entity_ids_required"}, status=400)

    try:
        kind_canon, _ = logic._canon_league_page_view_kind_entity(
            kind=kind, entity_id=int(entity_ids[0])
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    if not _validate_league_page_view_entity_ids(
        m=m, league_id=int(league_id), kind=str(kind_canon), entity_ids=list(entity_ids or [])
    ):
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    view_counts: dict[int, int] = {}
    for eid, vc in m.LeaguePageView.objects.filter(
        league_id=int(league_id), page_kind=str(kind_canon), entity_id__in=list(entity_ids)
    ).values_list("entity_id", "view_count"):
        try:
            view_counts[int(eid)] = int(vc or 0)
        except Exception:
            view_counts[int(eid)] = 0

    from django.db import transaction

    now = dt.datetime.now()
    try:
        with transaction.atomic():
            existing = list(
                m.LeaguePageViewBaseline.objects.filter(
                    league_id=int(league_id),
                    page_kind=str(kind_canon),
                    entity_id__in=list(entity_ids),
                )
            )
            existing_ids = {int(b.entity_id) for b in existing}
            for b in existing:
                eid = int(b.entity_id)
                b.baseline_count = int(view_counts.get(eid, 0))
                b.updated_at = now
            if existing:
                m.LeaguePageViewBaseline.objects.bulk_update(
                    existing, ["baseline_count", "updated_at"]
                )

            to_create = []
            for eid in entity_ids:
                if int(eid) in existing_ids:
                    continue
                to_create.append(
                    m.LeaguePageViewBaseline(
                        league_id=int(league_id),
                        page_kind=str(kind_canon),
                        entity_id=int(eid),
                        baseline_count=int(view_counts.get(int(eid), 0)),
                        created_at=now,
                        updated_at=now,
                    )
                )
            if to_create:
                m.LeaguePageViewBaseline.objects.bulk_create(to_create, ignore_conflicts=True)
    except Exception:
        return JsonResponse({"ok": False, "error": "failed_to_mark"}, status=500)

    results: dict[str, Any] = {}
    for eid in entity_ids:
        count = int(view_counts.get(int(eid), 0))
        results[str(int(eid))] = {
            "entity_id": int(eid),
            "count": int(count),
            "baseline_count": int(count),
            "delta_count": 0,
        }
    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "kind": kind_canon, "results": results}
    )


@csrf_exempt
def api_league_page_views_record(request: HttpRequest, league_id: int) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    payload = _json_body(request)
    kind = str(payload.get("kind") or request.POST.get("kind") or "").strip()
    entity_id_raw = payload.get("entity_id")
    if entity_id_raw is None:
        entity_id_raw = request.POST.get("entity_id")
    try:
        entity_id = int(str(entity_id_raw or "0").strip() or "0")
    except Exception:
        entity_id = 0

    _django_orm, m = _orm_modules()
    viewer_user_id = _session_user_id(request)
    league_row = (
        m.League.objects.filter(id=int(league_id))
        .values("id", "owner_user_id", "is_shared", "is_public")
        .first()
    )
    if not league_row:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    owner_user_id = int(league_row["owner_user_id"])
    is_public = bool(league_row.get("is_public"))

    if not viewer_user_id and not is_public:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    if viewer_user_id and not is_public:
        from django.db.models import Q

        ok = (
            m.League.objects.filter(id=int(league_id))
            .filter(
                Q(is_shared=True)
                | Q(owner_user_id=int(viewer_user_id))
                | Q(members__user_id=int(viewer_user_id))
            )
            .exists()
        )
        if not ok:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    # Validate entity belongs to the league, to avoid cross-league injection of view counts.
    kind_s = str(kind or "").strip()
    if kind_s == logic.LEAGUE_PAGE_VIEW_KIND_TEAMS:
        entity_id = 0
    elif kind_s == logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE:
        entity_id = 0
    elif kind_s == logic.LEAGUE_PAGE_VIEW_KIND_TEAM:
        if (
            entity_id <= 0
            or not m.LeagueTeam.objects.filter(
                league_id=int(league_id), team_id=int(entity_id)
            ).exists()
        ):
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    elif kind_s == logic.LEAGUE_PAGE_VIEW_KIND_GAME:
        if (
            entity_id <= 0
            or not m.LeagueGame.objects.filter(
                league_id=int(league_id), game_id=int(entity_id)
            ).exists()
        ):
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    elif kind_s == logic.LEAGUE_PAGE_VIEW_KIND_PLAYER_EVENTS:
        if entity_id <= 0:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        p = m.Player.objects.filter(id=int(entity_id)).values_list("team_id", flat=True).first()
        if p is None:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        if not m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(p)).exists():
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    elif kind_s == logic.LEAGUE_PAGE_VIEW_KIND_EVENT_CLIP:
        if entity_id <= 0:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        gid = (
            m.HkyGameEventRow.objects.filter(id=int(entity_id))
            .values_list("game_id", flat=True)
            .first()
        )
        if gid is None:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        if not m.LeagueGame.objects.filter(league_id=int(league_id), game_id=int(gid)).exists():
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
    else:
        return JsonResponse({"ok": False, "error": "unsupported_kind"}, status=400)

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=str(kind_s),
        entity_id=int(entity_id),
        viewer_user_id=int(viewer_user_id) if viewer_user_id else None,
        league_owner_user_id=int(owner_user_id) if owner_user_id else None,
    )
    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "kind": kind_s, "entity_id": int(entity_id)}
    )


def api_hky_game_events(request: HttpRequest, game_id: int) -> JsonResponse:
    """
    Query normalized per-row game events.

    Auth: requires a logged-in session and access to the game (owner or selected-league view).
    """
    if request.method != "GET":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    session_uid = _session_user_id(request)
    if not session_uid:
        return JsonResponse({"ok": False, "error": "login_required"}, status=401)

    _django_orm, m = _orm_modules()

    owned = (
        m.HkyGame.objects.filter(id=int(game_id), user_id=int(session_uid))
        .values_list("id", flat=True)
        .first()
    )
    if owned is None:
        league_id = request.session.get("league_id")
        try:
            league_id_i = int(league_id) if league_id is not None else None
        except Exception:
            league_id_i = None
        if (
            league_id_i is None
            or not m.LeagueGame.objects.filter(
                league_id=int(league_id_i), game_id=int(game_id)
            ).exists()
        ):
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    qs = (
        m.HkyGameEventRow.objects.filter(game_id=int(game_id))
        .select_related("event_type")
        .order_by("period", "game_seconds", "id")
    )
    qs = qs.exclude(
        import_key__in=m.HkyGameEventSuppression.objects.filter(game_id=int(game_id)).values_list(
            "import_key", flat=True
        )
    )

    # Filters
    player_id = request.GET.get("player_id")
    if player_id is not None and str(player_id).strip():
        try:
            qs = qs.filter(player_id=int(player_id))
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid player_id"}, status=400)

    period = request.GET.get("period")
    if period is not None and str(period).strip():
        try:
            qs = qs.filter(period=int(period))
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid period"}, status=400)

    event_type = request.GET.get("event_type")
    if event_type is not None and str(event_type).strip():
        et_key = _event_type_key(event_type)
        if et_key:
            qs = qs.filter(event_type__key=str(et_key))
        else:
            return JsonResponse({"ok": False, "error": "invalid event_type"}, status=400)

    limit_raw = request.GET.get("limit")
    limit = 2000
    if limit_raw is not None and str(limit_raw).strip():
        try:
            limit = int(limit_raw)
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid limit"}, status=400)
    limit = max(1, min(int(limit), 5000))

    rows = list(
        qs[:limit].values(
            "id",
            "import_key",
            "event_type__name",
            "source",
            "event_id",
            "team_id",
            "player_id",
            "team_raw",
            "team_side",
            "for_against",
            "team_rel",
            "period",
            "game_time",
            "video_time",
            "game_seconds",
            "game_seconds_end",
            "video_seconds",
            "details",
            "attributed_players",
            "attributed_jerseys",
            "on_ice_players",
            "on_ice_players_home",
            "on_ice_players_away",
            "created_at",
            "updated_at",
        )
    )
    out: list[dict[str, Any]] = []
    for r0 in rows:
        video_time, video_seconds = logic.normalize_video_time_and_seconds(
            r0.get("video_time"), r0.get("video_seconds")
        )
        out.append(
            {
                "id": int(r0["id"]),
                "import_key": str(r0.get("import_key") or ""),
                "event_type": str(r0.get("event_type__name") or ""),
                "source": str(r0.get("source") or ""),
                "event_id": r0.get("event_id"),
                "team_id": r0.get("team_id"),
                "player_id": r0.get("player_id"),
                "team_raw": str(r0.get("team_raw") or ""),
                "team_side": str(r0.get("team_side") or ""),
                "for_against": str(r0.get("for_against") or ""),
                "team_rel": str(r0.get("team_rel") or ""),
                "period": r0.get("period"),
                "game_time": str(r0.get("game_time") or ""),
                "video_time": video_time,
                "game_seconds": r0.get("game_seconds"),
                "game_seconds_end": r0.get("game_seconds_end"),
                "video_seconds": video_seconds,
                "details": str(r0.get("details") or ""),
                "attributed_players": str(r0.get("attributed_players") or ""),
                "attributed_jerseys": str(r0.get("attributed_jerseys") or ""),
                "on_ice_players": str(r0.get("on_ice_players") or ""),
                "on_ice_players_home": str(r0.get("on_ice_players_home") or ""),
                "on_ice_players_away": str(r0.get("on_ice_players_away") or ""),
                "created_at": r0.get("created_at"),
                "updated_at": r0.get("updated_at"),
            }
        )

    out = logic.sort_event_dicts_for_table_display(out)

    return JsonResponse(
        {
            "ok": True,
            "game_id": int(game_id),
            "count": int(len(out)),
            "events": out,
        }
    )


def api_hky_team_player_events(request: HttpRequest, team_id: int, player_id: int) -> JsonResponse:
    """
    Return underlying event rows contributing to a player's stats on a team's page.

    Supports:
      - attributed events (player_id match), and
      - on-ice goal events (Goal rows where the player's jersey is listed as on-ice).

    Auth:
      - Logged-in session is required for private leagues/teams.
      - For public leagues, `league_id` can be passed as a query param and will be honored without login.
    """
    if request.method != "GET":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    _django_orm, m = _orm_modules()
    session_uid = _session_user_id(request)

    league_id_param = request.GET.get("league_id") or request.GET.get("lid")
    league_id_session = request.session.get("league_id")
    league_id_raw = league_id_param if league_id_param is not None else league_id_session
    try:
        league_id_i = int(league_id_raw) if league_id_raw is not None else None
    except Exception:
        league_id_i = None

    public_mode = False
    if not session_uid:
        if league_id_i is None:
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        if not _is_public_league(int(league_id_i)):
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        public_mode = True
    else:
        # If a league id is selected, ensure membership to prevent leaking league data.
        if league_id_i is not None:
            from django.db.models import Q

            ok = (
                m.League.objects.filter(id=int(league_id_i))
                .filter(
                    Q(is_shared=True)
                    | Q(owner_user_id=int(session_uid))
                    | Q(members__user_id=int(session_uid))
                )
                .exists()
            )
            if not ok:
                return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    if (
        league_id_i is not None
        and not m.LeagueTeam.objects.filter(
            league_id=int(league_id_i), team_id=int(team_id)
        ).exists()
    ):
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    # Validate player/team relationship.
    prow = (
        m.Player.objects.filter(id=int(player_id), team_id=int(team_id))
        .values("id", "team_id", "name", "jersey_number", "position")
        .first()
    )
    if not prow:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    if league_id_i is not None:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id_i))
        logic._record_league_page_view(
            None,
            int(league_id_i),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_PLAYER_EVENTS,
            entity_id=int(player_id),
            viewer_user_id=int(session_uid) if session_uid else None,
            league_owner_user_id=league_owner_user_id,
        )

    player_name = str(prow.get("name") or "").strip()
    player_pos = str(prow.get("position") or "").strip()
    jersey_norm = logic.normalize_jersey_number(prow.get("jersey_number"))

    def _on_ice_numbers(raw: Optional[str]) -> set[str]:
        if not raw:
            return set()
        return {str(int(x)) for x in re.findall(r"([0-9]+)", str(raw))}

    # Determine schedule games and apply the same game-type and "eligible game" filtering as the team page.
    schedule_games: list[dict[str, Any]] = []
    if league_id_i is not None:
        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id_i)).values_list(
                "team_id", "division_name"
            )
        }
        from django.db.models import Q

        schedule_rows_raw = list(
            m.LeagueGame.objects.filter(league_id=int(league_id_i))
            .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
            .select_related("game", "game__team1", "game__team2", "game__game_type")
            .values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        for r0 in schedule_rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            schedule_games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])
    else:
        if not session_uid:
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        team_owned = (
            m.Team.objects.filter(id=int(team_id), user_id=int(session_uid))
            .values_list("id", flat=True)
            .first()
        )
        if team_owned is None:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        from django.db.models import Q

        schedule_rows = list(
            m.HkyGame.objects.filter(user_id=int(session_uid))
            .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        for r0 in schedule_rows:
            schedule_games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])

    league_name: Optional[str] = None
    if league_id_i is not None:
        try:
            league_name = logic._get_league_name(None, int(league_id_i))
        except Exception:
            league_name = None

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None

    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [
        g2
        for g2 in stats_schedule_games
        if logic._game_has_recorded_result(g2)
        and logic.game_is_eligible_for_stats(
            g2,
            team_id=int(team_id),
            league_name=league_name,
        )
    ]
    eligible_game_ids = [int(g2["id"]) for g2 in eligible_games if g2.get("id") is not None]
    eligible_game_ids_set = set(eligible_game_ids)

    recent_n_raw = request.GET.get("recent_n")
    recent_n: Optional[int] = None
    if recent_n_raw is not None and str(recent_n_raw).strip():
        try:
            recent_n = int(str(recent_n_raw).strip())
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid recent_n"}, status=400)
        recent_n = max(1, min(10, recent_n))

    # Optional scoping: restrict to the player's most recent N eligible games.
    if recent_n is not None and eligible_game_ids:
        order_idx: dict[int, int] = {}
        for idx, gid in enumerate(eligible_game_ids):
            order_idx[int(gid)] = int(idx)

        pid_game_ids_raw = list(
            m.HkyGamePlayer.objects.filter(
                player_id=int(player_id),
                game_id__in=eligible_game_ids,
            )
            .values_list("game_id", flat=True)
            .distinct()
        )
        pid_games_with_order: list[tuple[int, int]] = []
        for gid in pid_game_ids_raw:
            try:
                gid_i = int(gid)
            except Exception:
                continue
            idx = order_idx.get(gid_i)
            if idx is None:
                continue
            pid_games_with_order.append((int(idx), int(gid_i)))

        pid_games_with_order.sort(key=lambda t: t[0], reverse=True)
        chosen_ids = [gid for _idx, gid in pid_games_with_order[: int(recent_n)]]
        eligible_game_ids = [int(gid) for gid in chosen_ids]
        eligible_game_ids_set = set(eligible_game_ids)

    # Optional scoping: restrict to an explicit game selection (team page "Select Games" filter).
    selected_game_ids_raw = list(request.GET.getlist("gid") or [])
    selected_game_ids: set[int] = set()
    for raw in selected_game_ids_raw:
        try:
            gid = int(str(raw).strip())
        except Exception:
            continue
        if gid > 0:
            selected_game_ids.add(int(gid))
    if selected_game_ids and eligible_game_ids:
        chosen = [int(gid) for gid in eligible_game_ids if int(gid) in selected_game_ids]
        if chosen and set(chosen) != set(eligible_game_ids):
            eligible_game_ids = chosen
            eligible_game_ids_set = set(eligible_game_ids)
        eligible_games = [
            g2 for g2 in eligible_games if int(g2.get("id") or 0) in eligible_game_ids_set
        ]

    if not eligible_game_ids:
        return JsonResponse(
            {
                "ok": True,
                "team_id": int(team_id),
                "player_id": int(player_id),
                "player_name": player_name,
                "jersey_number": jersey_norm or "",
                "position": player_pos,
                "eligible_games": 0,
                "events": [],
                "on_ice_goals": [],
            }
        )

    games_by_id: dict[int, dict[str, Any]] = {
        int(g2["id"]): dict(g2) for g2 in eligible_games if g2.get("id") is not None
    }

    def _game_paths(gid: int) -> tuple[Optional[str], Optional[str]]:
        game_url = None
        video_url = None
        g2 = games_by_id.get(int(gid)) or {}
        video_url = str(g2.get("game_video_url") or "").strip() or None
        try:
            if public_mode and league_id_i is not None:
                game_url = reverse(
                    "public_hky_game_detail",
                    kwargs={"league_id": int(league_id_i), "game_id": int(gid)},
                )
            else:
                game_url = reverse("hky_game_detail", kwargs={"game_id": int(gid)})
        except Exception:
            game_url = None
        return game_url, video_url

    qs = (
        m.HkyGameEventRow.objects.filter(game_id__in=eligible_game_ids)
        .select_related("event_type")
        .order_by("game_id", "period", "game_seconds", "id")
    )
    qs = qs.exclude(
        import_key__in=m.HkyGameEventSuppression.objects.filter(
            game_id__in=eligible_game_ids
        ).values_list("import_key", flat=True)
    )

    limit_raw = request.GET.get("limit")
    limit = 4000
    if limit_raw is not None and str(limit_raw).strip():
        try:
            limit = int(limit_raw)
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid limit"}, status=400)
    limit = max(1, min(int(limit), 10000))

    attributed_values = (
        "id",
        "import_key",
        "game_id",
        "event_type__name",
        "event_type__key",
        "source",
        "event_id",
        "team_id",
        "team_raw",
        "team_side",
        "for_against",
        "team_rel",
        "period",
        "game_time",
        "video_time",
        "game_seconds",
        "video_seconds",
        "details",
        "correction",
        "attributed_players",
        "attributed_jerseys",
        "player_id",
    )

    direct_rows = list(qs.filter(player_id=int(player_id))[:limit].values(*attributed_values))
    fallback_rows: list[dict[str, Any]] = []
    if jersey_norm:
        jersey_re = rf"(^|[^0-9]){re.escape(jersey_norm)}([^0-9]|$)"
        # Only fill gaps for core counting stats; other event types should be attributed via player_id.
        fallback_rows = list(
            qs.filter(
                player_id__isnull=True,
                team_id=int(team_id),
                event_type__key__in=["goal", "assist"],
                attributed_jerseys__regex=str(jersey_re),
            )[:limit].values(*attributed_values)
        )

    # De-dupe by row id (defensive).
    attributed_rows: list[dict[str, Any]] = []
    seen_row_ids: set[int] = set()
    for r0 in list(direct_rows) + list(fallback_rows):
        try:
            rid = int(r0.get("id") or 0)
        except Exception:
            rid = 0
        if rid and rid in seen_row_ids:
            continue
        if rid:
            seen_row_ids.add(rid)
        attributed_rows.append(r0)

    events_out: list[dict[str, Any]] = []

    for r0 in attributed_rows:
        gid = int(r0.get("game_id") or 0)
        if gid not in eligible_game_ids_set:
            continue
        g2 = games_by_id.get(int(gid)) or {}
        game_url, video_url = _game_paths(int(gid))
        video_time, video_seconds = logic.normalize_video_time_and_seconds(
            r0.get("video_time"), r0.get("video_seconds")
        )

        correction: Any = None
        corr_raw = r0.get("correction")
        if corr_raw:
            try:
                correction = json.loads(str(corr_raw))
            except Exception:
                correction = {"raw": str(corr_raw)}

        details_txt = str(r0.get("details") or "")
        et_key = str(r0.get("event_type__key") or "").strip().lower()
        if et_key == "goal":
            # For goal rows, show only assists (filled below) to avoid redundant scorer/assist text.
            details_txt = ""

        events_out.append(
            {
                "kind": "attributed",
                "game_id": int(gid),
                "game_starts_at": g2.get("starts_at"),
                "game_type": str(g2.get("_game_type_label") or ""),
                "opponent": (
                    str(g2.get("team2_name") or "")
                    if int(g2.get("team1_id") or 0) == int(team_id)
                    else str(g2.get("team1_name") or "")
                ),
                "game_url": game_url,
                "video_url": video_url,
                "event_id": r0.get("event_id"),
                "event_type": str(r0.get("event_type__name") or ""),
                "event_type_key": str(r0.get("event_type__key") or ""),
                "team_side": str(r0.get("team_side") or ""),
                "period": r0.get("period"),
                "game_time": str(r0.get("game_time") or ""),
                "video_time": video_time,
                "game_seconds": r0.get("game_seconds"),
                "video_seconds": video_seconds,
                "details": details_txt,
                "correction": correction,
                "for_against": str(r0.get("for_against") or ""),
                "source": str(r0.get("source") or ""),
            }
        )

    # For goal events, show assists in the Details column (with jersey + name).
    team_players = list(
        m.Player.objects.filter(team_id=int(team_id)).values("name", "jersey_number", "position")
    )
    jersey_to_name: dict[str, str] = {}
    jersey_to_is_goalie: dict[str, bool] = {}
    for p in team_players:
        j = logic.normalize_jersey_number(p.get("jersey_number"))
        if not j:
            continue
        nm = str(p.get("name") or "").strip()
        is_goalie = str(p.get("position") or "").strip().upper() == "G"
        if j not in jersey_to_name:
            jersey_to_name[str(j)] = nm
            jersey_to_is_goalie[str(j)] = bool(is_goalie)
        else:
            # If jerseys collide, prefer non-goalies for assist display.
            if jersey_to_is_goalie.get(str(j)) and not is_goalie:
                jersey_to_name[str(j)] = nm
                jersey_to_is_goalie[str(j)] = False

    def _jersey_name(j: str) -> str:
        nm = str(jersey_to_name.get(str(j)) or "").strip()
        if nm:
            return f"#{j} {nm}"
        return f"#{j}"

    assist_rows = list(
        qs.filter(event_type__key="assist").values(
            "game_id",
            "period",
            "team_side",
            "game_seconds",
            "game_time",
            "attributed_jerseys",
            "id",
        )
    )
    assists_by_time: dict[tuple[int, int, int], list[str]] = {}
    assists_by_side_time: dict[tuple[int, int, int, str], list[str]] = {}
    for r0 in assist_rows:
        try:
            gid = int(r0.get("game_id") or 0)
            period = int(r0.get("period") or 0)
        except Exception:
            continue
        if gid <= 0 or period <= 0:
            continue
        gs = r0.get("game_seconds")
        try:
            gs_i = int(gs) if gs is not None else None
        except Exception:
            gs_i = None
        if gs_i is None:
            gs_i = logic.parse_duration_seconds(str(r0.get("game_time") or ""))  # type: ignore[arg-type]
        if gs_i is None:
            continue
        side = str(r0.get("team_side") or "").strip()
        jerseys_raw = str(r0.get("attributed_jerseys") or "")
        jerseys = [logic.normalize_jersey_number(x) for x in re.findall(r"([0-9]+)", jerseys_raw)]
        jerseys_norm = [j for j in jerseys if j]
        if not jerseys_norm:
            continue
        for j in jerseys_norm:
            k0 = (gid, period, int(gs_i))
            lst0 = assists_by_time.setdefault(k0, [])
            if j not in lst0:
                lst0.append(j)
            if side:
                k1 = (gid, period, int(gs_i), side)
                lst1 = assists_by_side_time.setdefault(k1, [])
                if j not in lst1:
                    lst1.append(j)

    for r in events_out:
        if str(r.get("event_type_key") or "").strip().lower() != "goal":
            continue
        try:
            gid = int(r.get("game_id") or 0)
            period = int(r.get("period") or 0)
        except Exception:
            continue
        if gid <= 0 or period <= 0:
            continue
        gs_val = r.get("game_seconds")
        try:
            gs_i = int(gs_val) if gs_val is not None else None
        except Exception:
            gs_i = None
        if gs_i is None:
            gs_i = logic.parse_duration_seconds(str(r.get("game_time") or ""))  # type: ignore[arg-type]
        if gs_i is None:
            continue
        side = str(r.get("team_side") or "").strip()
        assists = assists_by_side_time.get((gid, period, int(gs_i), side)) or assists_by_time.get(
            (gid, period, int(gs_i))
        )
        if not assists:
            continue
        lines = []
        if len(assists) == 1:
            lines.append(f"A: {_jersey_name(str(assists[0]))}")
        else:
            lines.append(f"A1: {_jersey_name(str(assists[0]))}")
            lines.append(f"A2: {_jersey_name(str(assists[1]))}")
        r["details"] = "\n".join(lines)

    # On-ice goal events (for GF/GA and plus/minus drilldown).
    goal_rows = list(
        qs.filter(event_type__key="goal")[:limit].values(
            "id",
            "import_key",
            "game_id",
            "event_type__name",
            "event_type__key",
            "source",
            "event_id",
            "team_id",
            "team_side",
            "period",
            "game_time",
            "video_time",
            "game_seconds",
            "video_seconds",
            "details",
            "correction",
            "attributed_players",
            "attributed_jerseys",
            "on_ice_players",
            "on_ice_players_home",
            "on_ice_players_away",
        )
    )

    on_ice_goals_out: list[dict[str, Any]] = []
    if jersey_norm:
        for r0 in goal_rows:
            gid = int(r0.get("game_id") or 0)
            if gid not in eligible_game_ids_set:
                continue
            g2 = games_by_id.get(int(gid)) or {}
            if int(g2.get("team1_id") or 0) == int(team_id):
                on_ice_raw = str(r0.get("on_ice_players_home") or "").strip() or str(
                    r0.get("on_ice_players") or ""
                )
            elif int(g2.get("team2_id") or 0) == int(team_id):
                on_ice_raw = str(r0.get("on_ice_players_away") or "").strip() or str(
                    r0.get("on_ice_players") or ""
                )
            else:
                continue
            on_ice = _on_ice_numbers(on_ice_raw)
            if jersey_norm not in on_ice:
                continue
            scoring_team_id = r0.get("team_id")
            try:
                scoring_team_id_i = int(scoring_team_id) if scoring_team_id is not None else None
            except Exception:
                scoring_team_id_i = None
            rel = None
            if scoring_team_id_i is not None:
                rel = "For" if int(scoring_team_id_i) == int(team_id) else "Against"
            game_url, video_url = _game_paths(int(gid))
            video_time, video_seconds = logic.normalize_video_time_and_seconds(
                r0.get("video_time"), r0.get("video_seconds")
            )
            correction: Any = None
            corr_raw = r0.get("correction")
            if corr_raw:
                try:
                    correction = json.loads(str(corr_raw))
                except Exception:
                    correction = {"raw": str(corr_raw)}
            on_ice_goals_out.append(
                {
                    "id": int(r0.get("id") or 0),
                    "kind": "on_ice_goal",
                    "for_against": rel or "",
                    "game_id": int(gid),
                    "game_starts_at": g2.get("starts_at"),
                    "game_type": str(g2.get("_game_type_label") or ""),
                    "opponent": (
                        str(g2.get("team2_name") or "")
                        if int(g2.get("team1_id") or 0) == int(team_id)
                        else str(g2.get("team1_name") or "")
                    ),
                    "game_url": game_url,
                    "video_url": video_url,
                    "period": r0.get("period"),
                    "game_time": str(r0.get("game_time") or ""),
                    "video_time": video_time,
                    "game_seconds": r0.get("game_seconds"),
                    "video_seconds": video_seconds,
                    "details": str(r0.get("details") or ""),
                    "correction": correction,
                    "source": str(r0.get("source") or ""),
                }
            )

    # Stable ordering for event tables.
    events_out = logic.sort_event_dicts_for_table_display(events_out)
    on_ice_goals_out = logic.sort_event_dicts_for_table_display(on_ice_goals_out)

    return JsonResponse(
        {
            "ok": True,
            "team_id": int(team_id),
            "player_id": int(player_id),
            "player_name": player_name,
            "jersey_number": jersey_norm or "",
            "position": player_pos,
            "eligible_games": int(len(eligible_game_ids)),
            "events": events_out,
            "on_ice_goals": on_ice_goals_out,
        }
    )


def api_hky_team_pair_on_ice(request: HttpRequest, team_id: int) -> JsonResponse:
    """
    Compute a consolidated "pair on-ice" style table for a team from:
      - hky_game_shift_rows (interval overlap), and
      - hky_game_event_rows (goal timestamps + scoring team, plus Goal/Assist attribution when available).

    This is intended to match the structure of scripts/parse_stats_inputs.py pair_on_ice outputs, but computed
    at runtime from DB tables (no stored CSV dependence).

    Auth:
      - Logged-in session is required for private leagues/teams.
      - For public leagues, `league_id` can be passed as a query param and will be honored without login.
    """
    if request.method != "GET":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    _django_orm, m = _orm_modules()
    session_uid = _session_user_id(request)

    league_id_param = request.GET.get("league_id") or request.GET.get("lid")
    league_id_session = request.session.get("league_id")
    league_id_raw = league_id_param if league_id_param is not None else league_id_session
    try:
        league_id_i = int(league_id_raw) if league_id_raw is not None else None
    except Exception:
        league_id_i = None

    if not session_uid:
        if league_id_i is None:
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        if not _is_public_league(int(league_id_i)):
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
    else:
        if league_id_i is not None:
            from django.db.models import Q

            ok = (
                m.League.objects.filter(id=int(league_id_i))
                .filter(
                    Q(is_shared=True)
                    | Q(owner_user_id=int(session_uid))
                    | Q(members__user_id=int(session_uid))
                )
                .exists()
            )
            if not ok:
                return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    if (
        league_id_i is not None
        and not m.LeagueTeam.objects.filter(
            league_id=int(league_id_i), team_id=int(team_id)
        ).exists()
    ):
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    # ----------------------------
    # Determine eligible games (same filtering semantics as the team stats page).
    # ----------------------------
    from django.db.models import Q

    schedule_games: list[dict[str, Any]] = []
    if league_id_i is not None:
        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id_i)).values_list(
                "team_id", "division_name"
            )
        }
        schedule_rows_raw = list(
            m.LeagueGame.objects.filter(league_id=int(league_id_i))
            .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
            .select_related("game", "game__team1", "game__team2", "game__game_type")
            .values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        for r0 in schedule_rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            schedule_games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])
    else:
        if not session_uid:
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        team_owned = (
            m.Team.objects.filter(id=int(team_id), user_id=int(session_uid))
            .values_list("id", flat=True)
            .first()
        )
        if team_owned is None:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        schedule_rows = list(
            m.HkyGame.objects.filter(user_id=int(session_uid))
            .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        for r0 in schedule_rows:
            schedule_games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])

    league_name: Optional[str] = None
    if league_id_i is not None:
        try:
            league_name = logic._get_league_name(None, int(league_id_i))
        except Exception:
            league_name = None

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"

    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [
        g2
        for g2 in stats_schedule_games
        if logic._game_has_recorded_result(g2)
        and logic.game_is_eligible_for_stats(
            g2,
            team_id=int(team_id),
            league_name=league_name,
        )
    ]
    eligible_game_ids_in_order: list[int] = []
    for g2 in eligible_games:
        try:
            eligible_game_ids_in_order.append(int(g2.get("id")))
        except Exception:
            continue

    recent_n_raw = request.GET.get("recent_n")
    recent_n: Optional[int] = None
    if recent_n_raw is not None and str(recent_n_raw).strip():
        try:
            recent_n = int(str(recent_n_raw).strip())
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid recent_n"}, status=400)
        recent_n = max(1, min(10, int(recent_n)))

    if recent_n is not None and eligible_game_ids_in_order:
        eligible_game_ids_in_order = list(eligible_game_ids_in_order)[-int(recent_n) :]

    # Optional scoping: restrict to an explicit game selection (team page "Select Games" filter).
    selected_game_ids_raw = list(request.GET.getlist("gid") or [])
    selected_game_ids: set[int] = set()
    for raw in selected_game_ids_raw:
        try:
            gid = int(str(raw).strip())
        except Exception:
            continue
        if gid > 0:
            selected_game_ids.add(int(gid))
    if selected_game_ids and eligible_game_ids_in_order:
        chosen = [int(gid) for gid in eligible_game_ids_in_order if int(gid) in selected_game_ids]
        if chosen and set(chosen) != set(eligible_game_ids_in_order):
            eligible_game_ids_in_order = chosen

    if not eligible_game_ids_in_order:
        return JsonResponse(
            {
                "ok": True,
                "team_id": int(team_id),
                "eligible_games": 0,
                "shift_games": 0,
                "rows": [],
            }
        )

    eligible_game_ids_set = {int(x) for x in eligible_game_ids_in_order}

    # ----------------------------
    # Compute pair-on-ice stats from shift+goal tables.
    # ----------------------------
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        merged: list[tuple[int, int]] = []
        for lo, hi in sorted(intervals, key=lambda x: (x[0], x[1])):
            if not merged:
                merged.append((lo, hi))
                continue
            prev_lo, prev_hi = merged[-1]
            if lo <= prev_hi:
                merged[-1] = (prev_lo, max(prev_hi, hi))
            else:
                merged.append((lo, hi))
        return merged

    def _intersection_seconds(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
        i = 0
        j = 0
        total = 0
        while i < len(a) and j < len(b):
            alo, ahi = a[i]
            blo, bhi = b[j]
            lo = max(alo, blo)
            hi = min(ahi, bhi)
            if hi > lo:
                total += hi - lo
            if ahi <= bhi:
                i += 1
            else:
                j += 1
        return total

    def _interval_contains(t: int, lo: int, hi: int) -> bool:
        return lo <= t <= hi

    def _first_jersey(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        m0 = re.search(r"([0-9]+)", s)
        if not m0:
            return None
        try:
            return str(int(m0.group(1)))
        except Exception:
            return m0.group(1)

    def _all_jerseys(raw: Any) -> list[str]:
        if raw is None:
            return []
        vals = []
        for tok in re.findall(r"([0-9]+)", str(raw)):
            try:
                vals.append(str(int(tok)))
            except Exception:
                vals.append(tok)
        out: list[str] = []
        seen: set[str] = set()
        for j in vals:
            if not j or j in seen:
                continue
            seen.add(j)
            out.append(j)
        return out

    shift_rows = list(
        m.HkyGameShiftRow.objects.filter(
            game_id__in=list(eligible_game_ids_set),
            team_id=int(team_id),
        )
        .exclude(period__isnull=True)
        .exclude(game_seconds__isnull=True)
        .exclude(game_seconds_end__isnull=True)
        .select_related("player")
        .values(
            "game_id",
            "player_id",
            "player__name",
            "player__jersey_number",
            "period",
            "game_seconds",
            "game_seconds_end",
        )
    )
    if not shift_rows:
        return JsonResponse(
            {
                "ok": True,
                "team_id": int(team_id),
                "eligible_games": int(len(eligible_game_ids_in_order)),
                "shift_games": 0,
                "rows": [],
            }
        )

    shift_game_ids = sorted({int(r.get("game_id") or 0) for r in shift_rows if r.get("game_id")})
    shift_game_ids_set = set(shift_game_ids)
    shift_game_ids_in_order = [
        int(gid) for gid in eligible_game_ids_in_order if int(gid) in shift_game_ids_set
    ]

    goal_rows = list(
        m.HkyGameEventRow.objects.filter(
            game_id__in=list(eligible_game_ids_set), event_type__key="goal"
        )
        .exclude(period__isnull=True)
        .exclude(game_seconds__isnull=True)
        .exclude(team_id__isnull=True)
        .values(
            "game_id",
            "team_id",
            "team_side",
            "period",
            "game_seconds",
            "attributed_jerseys",
        )
    )
    assist_rows = list(
        m.HkyGameEventRow.objects.filter(
            game_id__in=list(eligible_game_ids_set), event_type__key="assist"
        )
        .exclude(period__isnull=True)
        .exclude(game_seconds__isnull=True)
        .values(
            "game_id",
            "team_id",
            "team_side",
            "period",
            "game_seconds",
            "attributed_jerseys",
        )
    )

    shifts_by_game: dict[int, list[dict[str, Any]]] = {}
    for r0 in shift_rows:
        gid = logic._int0(r0.get("game_id"))
        if gid and gid in eligible_game_ids_set:
            shifts_by_game.setdefault(int(gid), []).append(r0)

    goals_by_game: dict[int, list[dict[str, Any]]] = {}
    for r0 in goal_rows:
        gid = logic._int0(r0.get("game_id"))
        if gid and gid in eligible_game_ids_set:
            goals_by_game.setdefault(int(gid), []).append(r0)

    assists_by_game: dict[int, list[dict[str, Any]]] = {}
    for r0 in assist_rows:
        gid = logic._int0(r0.get("game_id"))
        if gid and gid in eligible_game_ids_set:
            assists_by_game.setdefault(int(gid), []).append(r0)

    # Aggregation across all games with shift rows.
    agg: dict[tuple[int, int], dict[str, Any]] = {}
    total_pm_by_player: dict[int, int] = {}
    shift_games_by_player: dict[int, int] = {}
    player_info: dict[int, dict[str, str]] = {}

    for gid in shift_game_ids_in_order:
        rows = shifts_by_game.get(int(gid)) or []
        if not rows:
            continue

        intervals_by_pid_period: dict[int, dict[int, list[tuple[int, int]]]] = {}
        raw_intervals_by_pid_period: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        toi_by_pid: dict[int, int] = {}

        for sr in rows:
            pid = logic._int0(sr.get("player_id"))
            if pid <= 0:
                continue
            per = logic._int0(sr.get("period"))
            if per <= 0:
                continue
            try:
                gs = int(sr.get("game_seconds"))  # start (as recorded)
                ge = int(sr.get("game_seconds_end"))
            except Exception:
                continue
            lo = min(int(gs), int(ge))
            hi = max(int(gs), int(ge))
            intervals_by_pid_period.setdefault(int(pid), {}).setdefault(int(per), []).append(
                (int(lo), int(hi))
            )
            raw_intervals_by_pid_period.setdefault(int(pid), {}).setdefault(int(per), []).append(
                (int(lo), int(hi), int(gs))
            )
            player_info.setdefault(
                int(pid),
                {
                    "name": str(sr.get("player__name") or "").strip(),
                    "jersey_number": str(sr.get("player__jersey_number") or "").strip(),
                },
            )

        merged_by_pid_period: dict[int, dict[int, list[tuple[int, int]]]] = {}
        for pid, by_period in (intervals_by_pid_period or {}).items():
            toi = 0
            merged_by_pid_period[int(pid)] = {}
            for per, ints in (by_period or {}).items():
                merged = _merge_intervals(list(ints or []))
                merged_by_pid_period[int(pid)][int(per)] = merged
                toi += sum(int(hi) - int(lo) for lo, hi in merged)
            toi_by_pid[int(pid)] = int(toi)

        players = [pid for pid in sorted(toi_by_pid.keys()) if int(toi_by_pid.get(pid, 0) or 0) > 0]
        if not players:
            continue

        for pid in players:
            shift_games_by_player[int(pid)] = int(shift_games_by_player.get(int(pid), 0) or 0) + 1

        # Track player totals even when there are not enough players for any pair rows.
        pm_game: dict[int, int] = {int(pid): 0 for pid in players}

        # Jersey -> player mapping (only when unique across shift-tracked players).
        jersey_to_pid: dict[str, Optional[int]] = {}
        for pid in players:
            j = logic.normalize_jersey_number(player_info.get(int(pid), {}).get("jersey_number"))
            if not j:
                continue
            if str(j) not in jersey_to_pid:
                jersey_to_pid[str(j)] = int(pid)
            elif jersey_to_pid.get(str(j)) is not None:
                jersey_to_pid[str(j)] = None

        overlap_by_pair: dict[tuple[int, int], int] = {}
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a = int(players[i])
                b = int(players[j])
                overlap = 0
                a_periods = merged_by_pid_period.get(a, {})
                b_periods = merged_by_pid_period.get(b, {})
                for per in set(a_periods.keys()) & set(b_periods.keys()):
                    overlap += _intersection_seconds(a_periods.get(per, []), b_periods.get(per, []))
                overlap_by_pair[(min(a, b), max(a, b))] = int(overlap)

        def _on_ice_for_goal(pid: int, per: int, t_sec: int) -> bool:
            for lo, hi, start_sec in raw_intervals_by_pid_period.get(int(pid), {}).get(
                int(per), []
            ):
                if not _interval_contains(int(t_sec), int(lo), int(hi)):
                    continue
                if int(t_sec) == int(start_sec):
                    continue
                return True
            return False

        def _on_ice_any(pid: int, per: int, t_sec: int) -> bool:
            for lo, hi in merged_by_pid_period.get(int(pid), {}).get(int(per), []):
                if _interval_contains(int(t_sec), int(lo), int(hi)):
                    return True
            return False

        goals_by_pair: dict[tuple[int, int], tuple[int, int]] = {}
        player_goals_on_ice_by_pair: dict[tuple[int, int], int] = {}
        player_assists_on_ice_by_pair: dict[tuple[int, int], int] = {}
        collab_goals_by_pair: dict[tuple[int, int], int] = {}
        collab_assists_by_pair: dict[tuple[int, int], int] = {}

        # Group assists by (period, game_seconds, team_id/team_side) for associating with goals.
        assists_by_team_key: dict[tuple[int, int, Optional[int]], list[str]] = {}
        assists_by_side_key: dict[tuple[int, int, str], list[str]] = {}
        for ar in assists_by_game.get(int(gid), []) or []:
            per = logic._int0(ar.get("period"))
            gs_raw = ar.get("game_seconds")
            try:
                gs = int(gs_raw) if gs_raw is not None else None
            except Exception:
                gs = None
            if per <= 0 or gs is None:
                continue
            tid = ar.get("team_id")
            try:
                tid_i = int(tid) if tid is not None else None
            except Exception:
                tid_i = None
            for j in _all_jerseys(ar.get("attributed_jerseys")):
                assists_by_team_key.setdefault((int(per), int(gs), tid_i), []).append(j)
                side = str(ar.get("team_side") or "").strip()
                if side:
                    assists_by_side_key.setdefault((int(per), int(gs), side), []).append(j)

        for gr in sorted(
            goals_by_game.get(int(gid), []) or [],
            key=lambda r: (logic._int0(r.get("period")), logic._int0(r.get("game_seconds"))),
        ):
            per = logic._int0(gr.get("period"))
            gs_raw = gr.get("game_seconds")
            try:
                gs = int(gs_raw) if gs_raw is not None else None
            except Exception:
                gs = None
            if per <= 0 or gs is None:
                continue
            scoring_team_id_raw = gr.get("team_id")
            try:
                scoring_team_id = (
                    int(scoring_team_id_raw) if scoring_team_id_raw is not None else None
                )
            except Exception:
                scoring_team_id = None
            if scoring_team_id is None:
                continue
            kind = "GF" if int(scoring_team_id) == int(team_id) else "GA"

            on_ice = [pid for pid in players if _on_ice_for_goal(int(pid), int(per), int(gs))]
            on_ice_any = [pid for pid in players if _on_ice_any(int(pid), int(per), int(gs))]

            for pid in on_ice:
                pm_game[int(pid)] = int(pm_game.get(int(pid), 0) or 0) + (1 if kind == "GF" else -1)

            if kind == "GF":
                scorer_num = _first_jersey(gr.get("attributed_jerseys"))
                scorer_pid = (
                    int(jersey_to_pid.get(str(scorer_num)) or 0)
                    if scorer_num and jersey_to_pid.get(str(scorer_num)) is not None
                    else None
                )
                assist_nums = (
                    assists_by_team_key.get((int(per), int(gs), int(scoring_team_id)), []) or []
                )
                if not assist_nums:
                    side = str(gr.get("team_side") or "").strip()
                    if side:
                        assist_nums = assists_by_side_key.get((int(per), int(gs), side), []) or []
                assist_pids: list[int] = []
                for j in assist_nums:
                    pid0 = jersey_to_pid.get(str(j))
                    if pid0 is not None and pid0:
                        assist_pids.append(int(pid0))
                # De-dup assists.
                assist_pids = list(dict.fromkeys(assist_pids))

                if scorer_pid is not None and int(scorer_pid) in on_ice_any:
                    for teammate in on_ice_any:
                        if int(teammate) == int(scorer_pid):
                            continue
                        k2 = (int(scorer_pid), int(teammate))
                        player_goals_on_ice_by_pair[k2] = (
                            int(player_goals_on_ice_by_pair.get(k2, 0) or 0) + 1
                        )

                for ap in assist_pids:
                    if int(ap) not in on_ice_any:
                        continue
                    for teammate in on_ice_any:
                        if int(teammate) == int(ap):
                            continue
                        k2 = (int(ap), int(teammate))
                        player_assists_on_ice_by_pair[k2] = (
                            int(player_assists_on_ice_by_pair.get(k2, 0) or 0) + 1
                        )

                if scorer_pid is not None:
                    for ap in assist_pids:
                        if int(ap) == int(scorer_pid):
                            continue
                        k_goal = (int(scorer_pid), int(ap))
                        collab_goals_by_pair[k_goal] = (
                            int(collab_goals_by_pair.get(k_goal, 0) or 0) + 1
                        )
                        k_ast = (int(ap), int(scorer_pid))
                        collab_assists_by_pair[k_ast] = (
                            int(collab_assists_by_pair.get(k_ast, 0) or 0) + 1
                        )

            if len(on_ice) >= 2:
                for i in range(len(on_ice)):
                    for j in range(i + 1, len(on_ice)):
                        a = int(on_ice[i])
                        b = int(on_ice[j])
                        key = (min(a, b), max(a, b))
                        gf, ga = goals_by_pair.get(key, (0, 0))
                        if kind == "GF":
                            gf += 1
                        else:
                            ga += 1
                        goals_by_pair[key] = (int(gf), int(ga))

        for pid, pm in pm_game.items():
            total_pm_by_player[int(pid)] = int(total_pm_by_player.get(int(pid), 0) or 0) + int(
                pm or 0
            )

        # Pair rows (directed) only when at least two players have TOI in this game.
        if len(players) < 2:
            continue

        for pid in players:
            ptoi = int(toi_by_pid.get(int(pid), 0) or 0)
            for teammate in players:
                if int(teammate) == int(pid):
                    continue
                ukey = (min(int(pid), int(teammate)), max(int(pid), int(teammate)))
                overlap = int(overlap_by_pair.get(ukey, 0) or 0)
                gf, ga = goals_by_pair.get(ukey, (0, 0))
                dkey = (int(pid), int(teammate))
                dest = agg.setdefault(
                    dkey,
                    {
                        "shift_games": 0,
                        "player_toi_seconds": 0,
                        "overlap_seconds": 0,
                        "gf_together": 0,
                        "ga_together": 0,
                        "player_goals_on_ice_together": 0,
                        "player_assists_on_ice_together": 0,
                        "goals_collab_with_teammate": 0,
                        "assists_collab_with_teammate": 0,
                    },
                )
                dest["shift_games"] = int(dest.get("shift_games", 0) or 0) + 1
                dest["player_toi_seconds"] = int(dest.get("player_toi_seconds", 0) or 0) + int(ptoi)
                dest["overlap_seconds"] = int(dest.get("overlap_seconds", 0) or 0) + int(overlap)
                dest["gf_together"] = int(dest.get("gf_together", 0) or 0) + int(gf)
                dest["ga_together"] = int(dest.get("ga_together", 0) or 0) + int(ga)
                dest["player_goals_on_ice_together"] = int(
                    dest.get("player_goals_on_ice_together", 0) or 0
                ) + int(player_goals_on_ice_by_pair.get(dkey, 0) or 0)
                dest["player_assists_on_ice_together"] = int(
                    dest.get("player_assists_on_ice_together", 0) or 0
                ) + int(player_assists_on_ice_by_pair.get(dkey, 0) or 0)
                dest["goals_collab_with_teammate"] = int(
                    dest.get("goals_collab_with_teammate", 0) or 0
                ) + int(collab_goals_by_pair.get(dkey, 0) or 0)
                dest["assists_collab_with_teammate"] = int(
                    dest.get("assists_collab_with_teammate", 0) or 0
                ) + int(collab_assists_by_pair.get(dkey, 0) or 0)

    out_rows: list[dict[str, Any]] = []
    for (pid, teammate), d in agg.items():
        p = player_info.get(int(pid), {})
        t = player_info.get(int(teammate), {})
        toi = int(d.get("player_toi_seconds", 0) or 0)
        overlap = int(d.get("overlap_seconds", 0) or 0)
        pct = 100.0 * overlap / toi if toi > 0 else 0.0
        gf = int(d.get("gf_together", 0) or 0)
        ga = int(d.get("ga_together", 0) or 0)
        out_rows.append(
            {
                "player_id": int(pid),
                "player_jersey": str(p.get("jersey_number") or ""),
                "player_name": str(p.get("name") or ""),
                "teammate_id": int(teammate),
                "teammate_jersey": str(t.get("jersey_number") or ""),
                "teammate_name": str(t.get("name") or ""),
                "shift_games": int(d.get("shift_games", 0) or 0),
                "overlap_pct": float(pct),
                "gf_together": int(gf),
                "ga_together": int(ga),
                "player_goals_on_ice_together": int(d.get("player_goals_on_ice_together", 0) or 0),
                "player_assists_on_ice_together": int(
                    d.get("player_assists_on_ice_together", 0) or 0
                ),
                "goals_collab_with_teammate": int(d.get("goals_collab_with_teammate", 0) or 0),
                "assists_collab_with_teammate": int(d.get("assists_collab_with_teammate", 0) or 0),
                "plus_minus_together": int(gf) - int(ga),
                "player_total_plus_minus": int(total_pm_by_player.get(int(pid), 0) or 0),
                "teammate_total_plus_minus": int(total_pm_by_player.get(int(teammate), 0) or 0),
                "player_shift_games": int(shift_games_by_player.get(int(pid), 0) or 0),
                "teammate_shift_games": int(shift_games_by_player.get(int(teammate), 0) or 0),
            }
        )

    out_rows.sort(
        key=lambda r: (
            str(r.get("player_name") or "").casefold(),
            -float(r.get("overlap_pct") or 0.0),
            str(r.get("teammate_name") or "").casefold(),
        )
    )

    return JsonResponse(
        {
            "ok": True,
            "team_id": int(team_id),
            "eligible_games": int(len(eligible_game_ids_in_order)),
            "shift_games": int(len(shift_game_ids_in_order)),
            "rows": out_rows,
        }
    )


def api_hky_team_goalie_stats(request: HttpRequest, team_id: int, player_id: int) -> JsonResponse:
    """
    Per-goalie breakdown for the goalie stats tables on the team page.

    Auth:
      - Logged-in session is required for private teams/leagues.
      - For public leagues, `league_id` can be passed as a query param and will be honored without login.
    """
    if request.method != "GET":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    _django_orm, m = _orm_modules()
    session_uid = _session_user_id(request)

    league_id_param = request.GET.get("league_id") or request.GET.get("lid")
    league_id_session = request.session.get("league_id")
    league_id_raw = league_id_param if league_id_param is not None else league_id_session
    try:
        league_id_i = int(league_id_raw) if league_id_raw is not None else None
    except Exception:
        league_id_i = None

    public_mode = False
    if not session_uid:
        if league_id_i is None:
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        if not _is_public_league(int(league_id_i)):
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        public_mode = True
    else:
        # If a league id is selected, ensure membership to prevent leaking league data.
        if league_id_i is not None:
            from django.db.models import Q

            ok = (
                m.League.objects.filter(id=int(league_id_i))
                .filter(
                    Q(is_shared=True)
                    | Q(owner_user_id=int(session_uid))
                    | Q(members__user_id=int(session_uid))
                )
                .exists()
            )
            if not ok:
                return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    if (
        league_id_i is not None
        and not m.LeagueTeam.objects.filter(
            league_id=int(league_id_i), team_id=int(team_id)
        ).exists()
    ):
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    prow = (
        m.Player.objects.filter(id=int(player_id), team_id=int(team_id))
        .values("id", "team_id", "name", "jersey_number", "position")
        .first()
    )
    if not prow:
        return JsonResponse({"ok": False, "error": "not_found"}, status=404)

    player_name = str(prow.get("name") or "").strip()
    jersey_norm = logic.normalize_jersey_number(prow.get("jersey_number")) or ""

    # Determine schedule games and apply the same game-type and "eligible game" filtering as the team page.
    schedule_games: list[dict[str, Any]] = []
    if league_id_i is not None:
        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id_i)).values_list(
                "team_id", "division_name"
            )
        }
        from django.db.models import Q

        schedule_rows_raw = list(
            m.LeagueGame.objects.filter(league_id=int(league_id_i))
            .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
            .select_related("game", "game__team1", "game__team2", "game__game_type")
            .values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        for r0 in schedule_rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            schedule_games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])
    else:
        if not session_uid:
            return JsonResponse({"ok": False, "error": "login_required"}, status=401)
        team_owned = (
            m.Team.objects.filter(id=int(team_id), user_id=int(session_uid))
            .values_list("id", flat=True)
            .first()
        )
        if team_owned is None:
            return JsonResponse({"ok": False, "error": "not_found"}, status=404)
        from django.db.models import Q

        schedule_rows = list(
            m.HkyGame.objects.filter(user_id=int(session_uid))
            .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        for r0 in schedule_rows:
            schedule_games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])

    league_name: Optional[str] = None
    if league_id_i is not None:
        try:
            league_name = logic._get_league_name(None, int(league_id_i))
        except Exception:
            league_name = None

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None

    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [
        g2
        for g2 in stats_schedule_games
        if logic._game_has_recorded_result(g2)
        and logic.game_is_eligible_for_stats(
            g2,
            team_id=int(team_id),
            league_name=league_name,
        )
    ]
    eligible_game_ids_in_order: list[int] = []
    for g2 in eligible_games:
        try:
            eligible_game_ids_in_order.append(int(g2.get("id")))
        except Exception:
            continue

    recent_n_raw = request.GET.get("recent_n")
    recent_n: Optional[int] = None
    if recent_n_raw is not None and str(recent_n_raw).strip():
        try:
            recent_n = int(str(recent_n_raw).strip())
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid recent_n"}, status=400)
        recent_n = max(1, min(10, int(recent_n)))

    if recent_n is not None and eligible_game_ids_in_order:
        eligible_game_ids_in_order = eligible_game_ids_in_order[-int(recent_n) :]
        elig_set = {int(gid) for gid in eligible_game_ids_in_order}
        eligible_games = [g2 for g2 in eligible_games if int(g2.get("id") or 0) in elig_set]

    # Optional scoping: restrict to an explicit game selection (team page "Select Games" filter).
    selected_game_ids_raw = list(request.GET.getlist("gid") or [])
    selected_game_ids: set[int] = set()
    for raw in selected_game_ids_raw:
        try:
            gid = int(str(raw).strip())
        except Exception:
            continue
        if gid > 0:
            selected_game_ids.add(int(gid))
    if selected_game_ids and eligible_game_ids_in_order:
        chosen = [int(gid) for gid in eligible_game_ids_in_order if int(gid) in selected_game_ids]
        if chosen and set(chosen) != set(eligible_game_ids_in_order):
            eligible_game_ids_in_order = chosen
            elig_set = {int(gid) for gid in eligible_game_ids_in_order}
            eligible_games = [g2 for g2 in eligible_games if int(g2.get("id") or 0) in elig_set]

    if not eligible_game_ids_in_order:
        return JsonResponse(
            {
                "ok": True,
                "team_id": int(team_id),
                "player_id": int(player_id),
                "player_name": player_name,
                "jersey_number": jersey_norm,
                "eligible_games": 0,
                "games": [],
                "totals": {"gp": 0, "toi_seconds": 0, "ga": 0},
                "meta": {"has_sog": False, "has_xg": False},
            }
        )

    # Team roster goalies (for names/jerseys and starting-goalie inference).
    team_players = list(
        m.Player.objects.filter(team_id=int(team_id)).values(
            "id", "name", "jersey_number", "position"
        )
    )
    _sk, goalies, _hc, _ac = logic.split_roster(team_players)
    goalies = list(goalies or [])
    goalie_ids = {int(g.get("id") or 0) for g in goalies}
    if int(player_id) not in goalie_ids:
        goalies.append(
            {"id": int(player_id), "name": player_name, "jersey_number": prow.get("jersey_number")}
        )

    goalie_event_rows = list(
        m.HkyGameEventRow.objects.filter(
            game_id__in=list(eligible_game_ids_in_order),
            event_type__key__in=["goal", "expectedgoal", "sog", "shotongoal", "goaliechange"],
        )
        .select_related("event_type")
        .values(
            "game_id",
            "event_type__key",
            "event_id",
            "team_side",
            "period",
            "game_seconds",
            "player_id",
            "attributed_players",
            "attributed_jerseys",
            "details",
        )
    )
    rows_by_gid: dict[int, list[dict[str, Any]]] = {}
    for r0 in goalie_event_rows:
        try:
            gid_i = int(r0.get("game_id") or 0)
        except Exception:
            continue
        if gid_i <= 0:
            continue
        rows_by_gid.setdefault(int(gid_i), []).append(r0)

    games_by_id: dict[int, dict[str, Any]] = {
        int(g2["id"]): dict(g2) for g2 in eligible_games if g2.get("id") is not None
    }

    def _game_url(gid: int) -> Optional[str]:
        try:
            if public_mode and league_id_i is not None:
                return reverse(
                    "public_hky_game_detail",
                    kwargs={"league_id": int(league_id_i), "game_id": int(gid)},
                )
            return reverse("hky_game_detail", kwargs={"game_id": int(gid)})
        except Exception:
            return None

    out_games: list[dict[str, Any]] = []
    toi_sum = 0
    ga_sum = 0
    sa_sum = 0
    saves_sum = 0
    xga_sum = 0
    xg_saves_sum = 0
    has_any_sog = False
    has_any_xg = False

    for gid in list(eligible_game_ids_in_order)[::-1]:
        g2 = games_by_id.get(int(gid)) or {}
        t1 = int(g2.get("team1_id") or 0)
        t2 = int(g2.get("team2_id") or 0)
        if int(team_id) == int(t1):
            our_side = "home"
            opp = str(g2.get("team2_name") or "").strip()
            stats = logic.compute_goalie_stats_for_game(
                rows_by_gid.get(int(gid), []), home_goalies=goalies, away_goalies=[]
            )
        elif int(team_id) == int(t2):
            our_side = "away"
            opp = str(g2.get("team1_name") or "").strip()
            stats = logic.compute_goalie_stats_for_game(
                rows_by_gid.get(int(gid), []), home_goalies=[], away_goalies=goalies
            )
        else:
            continue

        our_rows = list(stats.get(str(our_side)) or [])
        row = next(
            (r for r in our_rows if int(r.get("player_id") or 0) == int(player_id)),
            None,
        )
        if not row:
            continue
        toi = int(row.get("toi_seconds") or 0)
        if toi <= 0:
            continue

        ga = int(row.get("ga") or 0)
        sa = row.get("sa")
        saves = row.get("saves")
        xga = row.get("xga")
        xg_saves = row.get("xg_saves")

        toi_sum += toi
        ga_sum += ga

        if sa is not None:
            has_any_sog = True
            try:
                sa_sum += int(sa)
            except Exception:
                pass
            try:
                saves_sum += int(saves or 0)
            except Exception:
                pass
        if xga is not None:
            has_any_xg = True
            try:
                xga_sum += int(xga)
            except Exception:
                pass
            try:
                xg_saves_sum += int(xg_saves or 0)
            except Exception:
                pass

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

        team1_score = _int_or_none(g2.get("team1_score"))
        team2_score = _int_or_none(g2.get("team2_score"))
        if our_side == "home":
            score_for = team1_score
            score_against = team2_score
        else:
            score_for = team2_score
            score_against = team1_score

        out_games.append(
            {
                "game_id": int(gid),
                "game_url": _game_url(int(gid)),
                "game_starts_at": g2.get("starts_at"),
                "game_type": str(g2.get("_game_type_label") or ""),
                "opponent": opp,
                "our_side": our_side,
                "score_for": score_for,
                "score_against": score_against,
                "toi_seconds": toi,
                "ga": ga,
                "xga": xga,
                "xg_saves": xg_saves,
                "xg_sv_pct": row.get("xg_sv_pct"),
                "sa": sa,
                "saves": saves,
                "sv_pct": row.get("sv_pct"),
                "gaa": row.get("gaa"),
            }
        )

    totals: dict[str, Any] = {
        "gp": int(len(out_games)),
        "toi_seconds": int(toi_sum),
        "ga": int(ga_sum),
    }
    if has_any_sog:
        totals["sa"] = int(sa_sum)
        totals["saves"] = int(saves_sum)
        totals["sv_pct"] = (float(saves_sum) / float(sa_sum)) if sa_sum > 0 else None
    if has_any_xg:
        totals["xga"] = int(xga_sum)
        totals["xg_saves"] = int(xg_saves_sum)
        totals["xg_sv_pct"] = (float(xg_saves_sum) / float(xga_sum)) if xga_sum > 0 else None
    totals["gaa"] = (float(ga_sum) * 60.0 / float(toi_sum)) if toi_sum > 0 else None

    return JsonResponse(
        {
            "ok": True,
            "team_id": int(team_id),
            "player_id": int(player_id),
            "player_name": player_name,
            "jersey_number": jersey_norm,
            "eligible_games": int(len(eligible_game_ids_in_order)),
            "games": out_games,
            "totals": totals,
            "meta": {"has_sog": bool(has_any_sog), "has_xg": bool(has_any_xg)},
        }
    )


# ----------------------------
# Legacy uploads/jobs UI
# ----------------------------


def games(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    rows = list(m.Game.objects.filter(user_id=uid).order_by("-created_at").values())
    dw_state = logic.read_dirwatch_state()
    for g in rows:
        try:
            st = (
                (dw_state.get("processed", {}) or {}).get(str(g.get("dir_path") or ""), {}) or {}
            ).get("status") or g.get("status")
            g["display_status"] = st
        except Exception:
            g["display_status"] = g.get("status")
    return render(request, "games.html", {"games": rows, "state": dw_state})


def new_game(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if request.method == "POST":
        name = (
            str(request.POST.get("name") or "").strip() or f"game-{dt.datetime.now():%Y%m%d-%H%M%S}"
        )
        gid, _dir_path = logic.create_game(
            _session_user_id(request), name, str(request.session.get("user_email") or "")
        )
        messages.success(request, "Game created")
        return redirect(f"/games/{gid}")
    return render(request, "new_game.html")


def game_detail(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        messages.error(request, "Not found")
        return redirect("/games")
    files: list[str] = []
    try:
        all_files = (
            os.listdir(str(game.get("dir_path") or ""))
            if os.path.isdir(str(game.get("dir_path") or ""))
            else []
        )

        def _is_user_file(fname: str) -> bool:
            if not fname:
                return False
            if fname.startswith(".") or fname.startswith("_") or fname.startswith("slurm-"):
                return False
            return True

        files = [f for f in sorted(all_files) if _is_user_file(f)]
    except Exception:
        files = []

    latest_status: Optional[str] = None
    latest_job = m.Job.objects.filter(game_id=int(gid)).order_by("-id").values("status").first()
    if latest_job and latest_job.get("status") is not None:
        latest_status = str(latest_job["status"])
    if not latest_status:
        dw_state = logic.read_dirwatch_state()
        latest_status = (dw_state.get("processed", {}) or {}).get(
            str(game.get("dir_path") or ""), {}
        ).get("status") or str(game.get("status") or "")

    is_locked = bool(latest_job)
    final_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
    if latest_status and str(latest_status).upper() in final_states:
        is_locked = True
    return render(
        request,
        "game_detail.html",
        {"game": game, "files": files, "status": latest_status, "is_locked": is_locked},
    )


def delete_game(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        messages.error(request, "Not found")
        return redirect("/games")

    latest = (
        m.Job.objects.filter(game_id=int(gid))
        .order_by("-id")
        .values("id", "slurm_job_id", "status")
        .first()
    )
    if request.method == "POST":
        token = str(request.POST.get("confirm") or "").strip().upper()
        if token != "DELETE":
            messages.error(request, "Type DELETE to confirm permanent deletion.")
            return render(request, "confirm_delete.html", {"game": game})

        try:
            active_states = ("SUBMITTED", "RUNNING", "PENDING")
            if latest and str(latest.get("status", "")).upper() in active_states:
                import subprocess as _sp
                import time as _time

                dir_leaf = Path(str(game["dir_path"])).name
                job_name = f"dirwatch-{dir_leaf}"
                job_ids: list[str] = []
                jid = latest.get("slurm_job_id")
                if jid:
                    job_ids.append(str(jid))
                else:
                    try:
                        out = _sp.check_output(["squeue", "-h", "-o", "%i %j"]).decode()
                        for line in out.splitlines():
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2 and parts[1] == job_name:
                                job_ids.append(parts[0])
                    except Exception:
                        pass
                for jid2 in job_ids:
                    try:
                        _sp.run(["scancel", str(jid2)], check=False)
                    except Exception:
                        pass
                _time.sleep(0.5)
        except Exception:
            pass

        from django.db import transaction

        with transaction.atomic():
            m.Job.objects.filter(game_id=int(gid)).delete()
            m.Game.objects.filter(id=int(gid)).delete()
        try:
            import shutil

            shutil.rmtree(str(game.get("dir_path") or ""), ignore_errors=True)
        except Exception:
            pass
        messages.success(request, "Game deleted")
        return redirect("/games")
    return render(request, "confirm_delete.html", {"game": game})


def upload_to_game(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        raise Http404
    files = request.FILES.getlist("files")
    if not files:
        messages.error(request, "No files selected")
        return redirect(f"/games/{gid}")

    try:
        watch_root = Path(os.environ.get("HM_WATCH_ROOT", logic.WATCH_ROOT)).resolve()
        gp = Path(str(game.get("dir_path") or "")).resolve()
        if watch_root not in gp.parents and gp != watch_root:
            messages.error(request, "Invalid upload path")
            return redirect(f"/games/{gid}")
    except Exception:
        pass

    base_dir = Path(str(game.get("dir_path") or ""))
    for f in files:
        if not f or not getattr(f, "name", ""):
            continue
        fname = Path(str(f.name)).name
        dest = base_dir / fname
        with dest.open("wb") as out:
            for chunk in f.chunks():
                out.write(chunk)
    messages.success(request, "Uploaded")
    return redirect(f"/games/{gid}")


def run_game(request: HttpRequest, gid: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values().first()
    if not game:
        raise Http404
    ready = Path(str(game.get("dir_path") or "")) / "_READY"
    ready.write_text("ready\n", encoding="utf-8")
    now = dt.datetime.now()
    m.Job.objects.create(
        user_id=uid,
        game_id=int(gid),
        dir_path=str(game.get("dir_path") or ""),
        slurm_job_id=None,
        status="SUBMITTED",
        created_at=now,
        updated_at=now,
        finished_at=None,
        user_email=str(request.session.get("user_email") or ""),
    )
    messages.success(request, "Job submitted")
    return redirect(f"/games/{gid}")


def serve_upload(request: HttpRequest, gid: int, name: str) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    game = m.Game.objects.filter(id=int(gid), user_id=uid).values("dir_path").first()
    if not game:
        raise Http404
    base = Path(str(game.get("dir_path") or "")).resolve()
    target = (base / str(name)).resolve()
    if base not in target.parents and target != base:
        raise Http404
    return _safe_file_response(target, as_attachment=True)


def jobs(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    rows = list(m.Job.objects.filter(user_id=uid).order_by("-created_at").values())
    return render(request, "jobs.html", {"jobs": rows})


# ----------------------------
# Hockey UI (partial; ported in follow-up patch)
# ----------------------------


def media_team_logo(request: HttpRequest, team_id: int) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    row = (
        m.Team.objects.filter(id=int(team_id), user_id=uid)
        .values("id", "user_id", "logo_path")
        .first()
    )
    if not row and league_id:
        row = (
            m.Team.objects.filter(id=int(team_id), league_teams__league_id=int(league_id))
            .values("id", "user_id", "logo_path")
            .first()
        )
    if not row or not row.get("logo_path"):
        raise Http404
    return _safe_file_response(Path(str(row["logo_path"])).resolve(), as_attachment=False)


def teams(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    include_external = str(request.GET.get("all") or "0") == "1"
    league_id = request.session.get("league_id")
    selected_age_raw = str(request.GET.get("age") or "").strip()
    selected_level_raw = str(request.GET.get("level") or "").strip()
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
            entity_id=0,
            viewer_user_id=_session_user_id(request),
            league_owner_user_id=league_owner_user_id,
        )
    is_league_admin = bool(
        league_id and _is_league_admin(int(league_id), _session_user_id(request))
    )

    _django_orm, m = _orm_modules()
    uid = _session_user_id(request)
    if league_id:
        rows_raw = list(
            m.LeagueTeam.objects.filter(league_id=int(league_id))
            .select_related("team")
            .values(
                "team_id",
                "team__user_id",
                "team__name",
                "team__logo_path",
                "team__is_external",
                "team__created_at",
                "team__updated_at",
                "division_name",
                "division_id",
                "conference_id",
                "mhr_rating",
                "mhr_agd",
                "mhr_sched",
                "mhr_games",
                "mhr_updated_at",
            )
        )
        schedule_team_ids: Optional[set[int]] = None
        if any("girls" in str(r0.get("team__name") or "").lower() for r0 in rows_raw):
            schedule_team_ids = set()
            for t1_id, t2_id in m.LeagueGame.objects.filter(league_id=int(league_id)).values_list(
                "game__team1_id", "game__team2_id"
            ):
                if t1_id is not None:
                    schedule_team_ids.add(int(t1_id))
                if t2_id is not None:
                    schedule_team_ids.add(int(t2_id))
        rows: list[dict[str, Any]] = []
        for r0 in rows_raw:
            tid = int(r0["team_id"])
            name = str(r0.get("team__name") or "")
            if name.strip() == logic.SEED_PLACEHOLDER_TEAM_NAME or logic.is_seed_placeholder_name(
                name
            ):
                continue
            if (
                schedule_team_ids is not None
                and "girls" in name.lower()
                and tid not in schedule_team_ids
            ):
                continue
            rows.append(
                {
                    "id": tid,
                    "user_id": int(r0["team__user_id"]),
                    "name": name,
                    "logo_path": r0.get("team__logo_path"),
                    "is_external": bool(r0.get("team__is_external")),
                    "created_at": r0.get("team__created_at"),
                    "updated_at": r0.get("team__updated_at"),
                    "division_name": r0.get("division_name"),
                    "division_id": r0.get("division_id"),
                    "conference_id": r0.get("conference_id"),
                    "mhr_rating": r0.get("mhr_rating"),
                    "mhr_agd": r0.get("mhr_agd"),
                    "mhr_sched": r0.get("mhr_sched"),
                    "mhr_games": r0.get("mhr_games"),
                    "mhr_updated_at": r0.get("mhr_updated_at"),
                }
            )
    else:
        qs = m.Team.objects.filter(user_id=uid)
        if not include_external:
            qs = qs.filter(is_external=False)
        rows = list(qs.order_by("name").values())

    age_options: list[dict[str, str]] = []
    level_options: list[str] = []
    selected_age = ""
    selected_level = ""
    if league_id:
        meta_by_div: dict[str, dict[str, Any]] = {}
        for t in rows:
            dn = _norm_division_name(t.get("division_name"))
            if dn not in meta_by_div:
                meta_by_div[dn] = {
                    "age": logic.parse_age_from_division_name(dn),
                    "level": logic.parse_level_from_division_name(dn),
                }

        ages = sorted({int(v["age"]) for v in meta_by_div.values() if v.get("age") is not None})
        age_options = [{"value": str(a), "label": _age_label(a)} for a in ages]

        selected_age_u: Optional[int] = None
        if selected_age_raw:
            m_age = re.search(r"(\d{1,2})", selected_age_raw)
            if m_age:
                try:
                    candidate = int(m_age.group(1))
                except Exception:
                    candidate = None
                if candidate is not None and candidate in set(ages):
                    selected_age_u = candidate

        candidates = list(meta_by_div.values())
        if selected_age_u is not None:
            candidates = [
                v
                for v in candidates
                if v.get("age") is not None and int(v["age"]) == selected_age_u
            ]

        levels = {str(v["level"]).strip().upper() for v in candidates if v.get("level")}
        level_options = sorted(levels, key=_level_sort_key)
        if any(v.get("level") is None for v in candidates):
            level_options.append("Other")

        selected_level_norm = selected_level_raw.strip().upper()
        if selected_level_norm == "OTHER":
            selected_level_norm = "Other"
        if selected_level_norm in set(level_options):
            selected_level = selected_level_norm

        if selected_age_u is not None:
            selected_age = str(selected_age_u)
            rows = [
                t
                for t in rows
                if meta_by_div.get(_norm_division_name(t.get("division_name")), {}).get("age")
                == selected_age_u
            ]
        if selected_level:
            if selected_level == "Other":
                rows = [
                    t
                    for t in rows
                    if meta_by_div.get(_norm_division_name(t.get("division_name")), {}).get("level")
                    is None
                ]
            else:
                rows = [
                    t
                    for t in rows
                    if str(
                        meta_by_div.get(_norm_division_name(t.get("division_name")), {}).get(
                            "level"
                        )
                        or ""
                    )
                    .strip()
                    .upper()
                    == selected_level
                ]

    stats: dict[int, dict[str, Any]] = {}
    for t in rows:
        tid = int(t["id"])
        if league_id:
            stats[tid] = logic.compute_team_stats_league(None, tid, int(league_id))
        else:
            stats[tid] = logic.compute_team_stats(None, tid, uid)
        try:
            s = stats[tid]
            s["gp"] = (
                int(s.get("wins", 0) or 0)
                + int(s.get("losses", 0) or 0)
                + int(s.get("ties", 0) or 0)
            )
        except Exception:
            pass

    divisions = None
    if league_id:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for t in rows:
            dn = _norm_division_name(t.get("division_name"))
            grouped.setdefault(dn, []).append(t)
        divisions = []
        for dn in sorted(grouped.keys(), key=logic.division_sort_key):
            ranked_ids = logic.division_standings_team_ids(int(league_id), str(dn))
            order = {int(tid): i for i, tid in enumerate(ranked_ids)}
            teams_sorted = sorted(
                grouped[dn],
                key=lambda tr: (
                    order.get(int(tr["id"]), 10**9),
                    str(tr.get("name") or "").casefold(),
                ),
            )
            divisions.append({"name": dn, "teams": teams_sorted})

    league_page_views = None
    if league_id and is_league_owner:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
            "entity_id": 0,
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }
    return render(
        request,
        "teams.html",
        {
            "teams": rows,
            "divisions": divisions,
            "age_options": age_options,
            "level_options": level_options,
            "selected_age": selected_age,
            "selected_level": selected_level,
            "stats": stats,
            "include_external": include_external,
            "league_view": bool(league_id),
            "current_user_id": uid,
            "is_league_admin": is_league_admin,
            "league_page_views": league_page_views,
        },
    )


# ----------------------------
# Team/player management
# ----------------------------


def new_team(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        if not name:
            messages.error(request, "Team name is required")
            return render(request, "team_new.html")
        tid = logic.create_team(_session_user_id(request), name, is_external=False)
        f = request.FILES.get("logo")
        if f and getattr(f, "name", ""):
            try:
                p = logic.save_team_logo(f, tid)
                _django_orm, m = _orm_modules()
                m.Team.objects.filter(id=int(tid), user_id=_session_user_id(request)).update(
                    logo_path=str(p)
                )
            except Exception:
                messages.error(request, "Failed to save team logo")
        messages.success(request, "Team created")
        return redirect(f"/teams/{tid}")
    return render(request, "team_new.html")


def _player_totals_has_any_events(stats: dict[str, Any]) -> bool:
    for k in logic.PLAYER_STATS_SUM_KEYS:
        try:
            if int(stats.get(k) or 0) != 0:
                return True
        except Exception:
            continue
    return False


def team_detail(request: HttpRequest, team_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r

    session_uid = _session_user_id(request)
    recent_n_raw = request.GET.get("recent_n")
    try:
        recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
    except Exception:
        recent_n = 5

    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None and int(league_owner_user_id) == session_uid
        )

    show_shift_data = True
    if league_id:
        try:
            show_shift_data = _league_show_shift_data(int(league_id))
        except Exception:
            show_shift_data = False

    is_league_admin = False
    if league_id:
        try:
            is_league_admin = bool(_is_league_admin(int(league_id), session_uid))
        except Exception:
            is_league_admin = False

    team = logic.get_team(int(team_id), session_uid)
    editable = bool(team)
    _django_orm, m = _orm_modules()
    if not team and league_id:
        team = (
            m.Team.objects.filter(id=int(team_id), league_teams__league_id=int(league_id))
            .values(
                "id",
                "user_id",
                "name",
                "logo_path",
                "is_external",
                "created_at",
                "updated_at",
            )
            .first()
        )
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")

    # Page view counts (and per-event clip counts) are league-scoped. If the user has not selected a
    # league in-session, but this team belongs to exactly one league where the user is an owner/admin,
    # infer it so the team page Player Events modal can still show consistent view counts.
    page_views_league_id: Optional[int] = None
    if league_id:
        page_views_league_id = int(league_id)
    else:
        try:
            candidate_ids = list(
                m.LeagueTeam.objects.filter(team_id=int(team_id))
                .values_list("league_id", flat=True)
                .distinct()
            )
            candidate_ids_i = [int(x) for x in candidate_ids if x is not None]
            if candidate_ids_i:
                owner_ids = set(
                    m.League.objects.filter(
                        id__in=candidate_ids_i, owner_user_id=int(session_uid)
                    ).values_list("id", flat=True)
                )
                admin_ids = set(
                    m.LeagueMember.objects.filter(
                        league_id__in=candidate_ids_i,
                        user_id=int(session_uid),
                        role__in=["admin", "owner"],
                    ).values_list("league_id", flat=True)
                )
                eligible_ids = sorted({int(x) for x in (owner_ids | admin_ids)})
                if len(eligible_ids) == 1:
                    page_views_league_id = int(eligible_ids[0])
        except Exception:
            page_views_league_id = None

    # Recompute league-owner/admin flags against the effective league id used for view counts.
    if page_views_league_id:
        try:
            league_owner_user_id = logic._get_league_owner_user_id(None, int(page_views_league_id))
        except Exception:
            league_owner_user_id = None
        is_league_owner = bool(
            league_owner_user_id is not None and int(league_owner_user_id) == int(session_uid)
        )
        try:
            is_league_admin = bool(_is_league_admin(int(page_views_league_id), int(session_uid)))
        except Exception:
            is_league_admin = False

    if page_views_league_id:
        logic._record_league_page_view(
            None,
            int(page_views_league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            entity_id=int(team_id),
            viewer_user_id=session_uid,
            league_owner_user_id=league_owner_user_id,
        )

    team_owner_id = int(team["user_id"])
    players_qs = m.Player.objects.filter(team_id=int(team_id))
    if editable:
        players_qs = players_qs.filter(user_id=session_uid)
    players = list(
        players_qs.order_by("jersey_number", "name").values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    skaters, goalies, head_coaches, assistant_coaches = logic.split_roster(players or [])
    roster_players = list(skaters) + list(goalies)

    from django.db.models import Q

    if league_id:
        tstats = logic.compute_team_stats_league(None, int(team_id), int(league_id))
        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                "team_id", "division_name"
            )
        }
        schedule_rows_raw = list(
            m.LeagueGame.objects.filter(league_id=int(league_id))
            .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
            .select_related("game", "game__team1", "game__team2", "game__game_type")
            .values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__stats_imported_at",
                "game__created_at",
                "game__updated_at",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        schedule_games: list[dict[str, Any]] = []
        for r0 in schedule_rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            schedule_games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "stats_imported_at": r0.get("game__stats_imported_at"),
                    "created_at": r0.get("game__created_at"),
                    "updated_at": r0.get("game__updated_at"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )

        now_dt = dt.datetime.now()
        for g2 in schedule_games:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = logic._sanitize_http_url(
                    logic._extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
        schedule_games = logic.sort_games_schedule_order(schedule_games or [])
        ps_rows = _player_stat_rows_from_event_tables_for_team_games(
            team_id=int(team_id),
            schedule_games=list(schedule_games or []),
            roster_players=list(roster_players or []),
            show_shift_data=bool(show_shift_data),
        )
    else:
        tstats = logic.compute_team_stats(None, int(team_id), team_owner_id)
        schedule_rows = list(
            m.HkyGame.objects.filter(user_id=int(team_owner_id))
            .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "stats_imported_at",
                "created_at",
                "updated_at",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        schedule_games = []
        for r0 in schedule_rows:
            schedule_games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "stats_imported_at": r0.get("stats_imported_at"),
                    "created_at": r0.get("created_at"),
                    "updated_at": r0.get("updated_at"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )
        now_dt = dt.datetime.now()
        for g2 in schedule_games:
            sdt = g2.get("starts_at")
            started = False
            if sdt is not None:
                try:
                    started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
                except Exception:
                    started = False
            has_score = (
                (g2.get("team1_score") is not None)
                or (g2.get("team2_score") is not None)
                or bool(g2.get("is_final"))
            )
            g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
            try:
                g2["game_video_url"] = logic._sanitize_http_url(
                    logic._extract_game_video_url_from_notes(g2.get("notes"))
                )
            except Exception:
                g2["game_video_url"] = None
        ps_rows = _player_stat_rows_from_event_tables_for_team_games(
            team_id=int(team_id),
            schedule_games=list(schedule_games or []),
            roster_players=list(roster_players or []),
            show_shift_data=bool(show_shift_data),
        )

    _attach_schedule_stats_icons(schedule_games)

    league_name: Optional[str] = None
    if league_id:
        try:
            league_name = logic._get_league_name(None, int(league_id))
        except Exception:
            league_name = None

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"
        try:
            reason = logic.game_exclusion_reason_for_stats(
                g2, team_id=int(team_id), league_name=league_name
            )
        except Exception:
            reason = None
        g2["excluded_from_stats_reason"] = reason
        g2["excluded_from_stats"] = bool(reason)
    # Tournament-only players: show them on game pages, but not on team/division-level roster/stats.
    try:
        tournament_game_ids: set[int] = {
            int(g2.get("id"))
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "").strip().casefold().startswith("tournament")
            and g2.get("id") is not None
        }
        player_ids_with_any_stats: set[int] = set()
        player_ids_with_non_tournament_stats: set[int] = set()
        for r0 in ps_rows or []:
            try:
                pid_i = int(r0.get("player_id"))
                gid_i = int(r0.get("game_id"))
            except Exception:
                continue
            player_ids_with_any_stats.add(pid_i)
            if gid_i not in tournament_game_ids:
                player_ids_with_non_tournament_stats.add(pid_i)
        tournament_only_player_ids = (
            player_ids_with_any_stats - player_ids_with_non_tournament_stats
        )
    except Exception:
        tournament_only_player_ids = set()

    team_is_external = bool(team.get("is_external"))
    if tournament_only_player_ids and not team_is_external:
        skaters = [p for p in skaters if int(p.get("id") or 0) not in tournament_only_player_ids]
        goalies = [p for p in goalies if int(p.get("id") or 0) not in tournament_only_player_ids]
        roster_players = list(skaters) + list(goalies)
        ps_rows = [
            r0 for r0 in ps_rows if int(r0.get("player_id") or 0) not in tournament_only_player_ids
        ]
    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [
        g2
        for g2 in stats_schedule_games
        if logic._game_has_recorded_result(g2)
        and logic.game_is_eligible_for_stats(
            g2,
            team_id=int(team_id),
            league_name=league_name,
        )
    ]
    eligible_game_ids_in_order: list[int] = []
    for g2 in eligible_games:
        try:
            eligible_game_ids_in_order.append(int(g2.get("id")))
        except Exception:
            continue
    eligible_game_ids_set: set[int] = set(eligible_game_ids_in_order)

    selected_game_ids_raw = list(request.GET.getlist("gid") or [])
    selected_game_ids_req: list[int] = []
    seen_gid: set[int] = set()
    for raw in selected_game_ids_raw:
        try:
            gid = int(str(raw).strip())
        except Exception:
            continue
        if gid <= 0 or gid in seen_gid:
            continue
        seen_gid.add(gid)
        selected_game_ids_req.append(int(gid))
    selected_game_ids_req_set = set(selected_game_ids_req)
    selected_game_ids_in_order = [
        int(gid) for gid in eligible_game_ids_in_order if int(gid) in selected_game_ids_req_set
    ]
    selected_game_ids_set = set(selected_game_ids_in_order)

    use_selected_games = (
        bool(selected_game_ids_set) and selected_game_ids_set != eligible_game_ids_set
    )
    stats_game_ids_in_order = (
        list(selected_game_ids_in_order) if use_selected_games else list(eligible_game_ids_in_order)
    )
    stats_game_ids_set = set(stats_game_ids_in_order)

    ps_rows_filtered: list[dict[str, Any]] = []
    for r0 in ps_rows or []:
        try:
            if int(r0.get("game_id")) in stats_game_ids_set:
                ps_rows_filtered.append(r0)
        except Exception:
            continue

    player_totals = logic._aggregate_player_totals_from_rows(
        player_stats_rows=ps_rows_filtered, allowed_game_ids=stats_game_ids_set
    )
    player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(skaters, player_totals)
    )
    for r0 in player_stats_rows:
        try:
            pid_i = int(r0.get("player_id") or 0)
        except Exception:
            continue
        r0["has_events"] = _player_totals_has_any_events(player_totals.get(pid_i) or {})
    player_stats_columns = logic.filter_player_stats_display_columns_for_rows(
        logic.PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows
    )
    cov_counts, cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows_filtered, eligible_game_ids=stats_game_ids_in_order
    )
    player_stats_columns = logic._player_stats_columns_with_coverage(
        columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
    )

    recent_scope_ids = (
        eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
    )
    has_pair_on_ice_all = False
    try:
        if stats_game_ids_in_order:
            has_pair_on_ice_all = bool(
                m.HkyGameShiftRow.objects.filter(
                    game_id__in=list(stats_game_ids_in_order), team_id=int(team_id)
                ).exists()
            )
    except Exception:
        has_pair_on_ice_all = False

    player_stats_sources = logic._compute_team_player_stats_sources(
        None, eligible_game_ids=stats_game_ids_in_order
    )
    selected_label = (
        "All"
        if selected_types is None
        else ", ".join(sorted(list(selected_types), key=lambda s: s.lower()))
    )
    game_type_filter_options = [
        {"label": gt, "checked": (selected_types is None) or (gt in selected_types)}
        for gt in game_type_options
    ]

    # "Select Games" UI: show eligible games (already filtered by Game Types) and highlight partial selection.
    select_games_options: list[dict[str, Any]] = []

    def _select_game_label(g2: dict[str, Any]) -> str:
        gid = g2.get("id")
        dt0 = logic.to_dt(g2.get("starts_at"))
        date_s = dt0.strftime("%Y-%m-%d") if dt0 else ""
        gt = str(g2.get("_game_type_label") or "").strip()
        opp = ""
        try:
            if int(g2.get("team1_id") or 0) == int(team_id):
                opp = str(g2.get("team2_name") or "").strip()
            else:
                opp = str(g2.get("team1_name") or "").strip()
        except Exception:
            opp = ""
        bits = []
        if date_s:
            bits.append(date_s)
        if opp:
            bits.append(f"vs {opp}")
        if gt:
            bits.append(f"({gt})")
        if not bits:
            return f"Game {gid}" if gid is not None else "Game"
        return " ".join(bits)

    for g2 in eligible_games or []:
        try:
            gid_i = int(g2.get("id") or 0)
        except Exception:
            continue
        if gid_i <= 0:
            continue
        select_games_options.append(
            {
                "id": int(gid_i),
                "label": _select_game_label(g2),
                "checked": (not use_selected_games) or (int(gid_i) in stats_game_ids_set),
            }
        )

    select_games_label = "All"
    if use_selected_games:
        select_games_label = f"{len(stats_game_ids_in_order)} of {len(eligible_game_ids_in_order)}"

    goalie_stats_rows: list[dict[str, Any]] = []
    goalie_stats_has_sog = False
    goalie_stats_has_xg = False
    recent_goalie_stats_rows: list[dict[str, Any]] = []
    recent_goalie_stats_has_sog = False
    recent_goalie_stats_has_xg = False
    show_goalie_stats = True
    if league_id:
        try:
            show_goalie_stats = _league_show_goalie_stats(int(league_id))
        except Exception:
            show_goalie_stats = False
    try:
        if show_goalie_stats and eligible_game_ids_in_order:
            goalie_event_rows = list(
                m.HkyGameEventRow.objects.filter(
                    game_id__in=list(eligible_game_ids_in_order),
                    event_type__key__in=[
                        "goal",
                        "expectedgoal",
                        "sog",
                        "shotongoal",
                        "goaliechange",
                    ],
                )
                .select_related("event_type")
                .values(
                    "game_id",
                    "event_type__key",
                    "event_id",
                    "team_side",
                    "period",
                    "game_seconds",
                    "player_id",
                    "attributed_players",
                    "attributed_jerseys",
                    "details",
                )
            )
            by_gid: dict[int, list[dict[str, Any]]] = {}
            for r0 in goalie_event_rows:
                try:
                    gid = int(r0.get("game_id") or 0)
                except Exception:
                    continue
                if gid <= 0:
                    continue
                by_gid.setdefault(gid, []).append(r0)
            out = logic.compute_goalie_stats_for_team_games(
                team_id=int(team_id),
                schedule_games=eligible_games,
                event_rows_by_game_id=by_gid,
                goalies=goalies,
            )
            goalie_stats_rows = list(out.get("rows") or [])
            goalie_stats_has_sog = bool((out.get("meta") or {}).get("has_sog"))
            goalie_stats_has_xg = bool((out.get("meta") or {}).get("has_xg"))

            recent_ids = {int(gid) for gid in recent_scope_ids}
            if recent_ids:
                recent_games = [g2 for g2 in eligible_games if int(g2.get("id") or 0) in recent_ids]
                out_recent = logic.compute_goalie_stats_for_team_games(
                    team_id=int(team_id),
                    schedule_games=recent_games,
                    event_rows_by_game_id=by_gid,
                    goalies=goalies,
                )
                recent_goalie_stats_rows = list(out_recent.get("rows") or [])
                recent_goalie_stats_has_sog = bool((out_recent.get("meta") or {}).get("has_sog"))
                recent_goalie_stats_has_xg = bool((out_recent.get("meta") or {}).get("has_xg"))
    except Exception:
        goalie_stats_rows = []
        goalie_stats_has_sog = False
        goalie_stats_has_xg = False
        recent_goalie_stats_rows = []
        recent_goalie_stats_has_sog = False
        recent_goalie_stats_has_xg = False

    can_view_league_page_views = bool(page_views_league_id and (is_league_owner or is_league_admin))
    league_page_views = None
    if page_views_league_id and can_view_league_page_views:
        count = logic._get_league_page_view_count(
            None,
            int(page_views_league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            entity_id=int(team_id),
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None,
            int(page_views_league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            entity_id=int(team_id),
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(page_views_league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            "entity_id": int(team_id),
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }
    excluded_games = [g2 for g2 in (schedule_games or []) if bool(g2.get("excluded_from_stats"))]
    return render(
        request,
        "team_detail.html",
        {
            "team": team,
            "roster_players": roster_players,
            "players": skaters,
            "head_coaches": head_coaches,
            "assistant_coaches": assistant_coaches,
            "player_stats_columns": player_stats_columns,
            "player_stats_rows": player_stats_rows,
            "select_games_options": select_games_options,
            "select_games_label": select_games_label,
            "select_games_partial": bool(use_selected_games),
            "recent_n": recent_n,
            "tstats": tstats,
            "schedule_games": schedule_games,
            "excluded_schedule_games_count": len(excluded_games),
            "show_caha_preseason_exclusion_note": str(league_name or "").strip().casefold()
            == "caha",
            "editable": editable,
            "is_league_admin": is_league_admin,
            "can_view_league_page_views": can_view_league_page_views,
            "page_views_league_id": page_views_league_id,
            "player_stats_sources": player_stats_sources,
            "player_stats_coverage_total_games": cov_total,
            "game_type_filter_options": game_type_filter_options,
            "game_type_filter_label": selected_label,
            "goalie_stats_rows": goalie_stats_rows,
            "goalie_stats_has_sog": goalie_stats_has_sog,
            "goalie_stats_has_xg": goalie_stats_has_xg,
            "recent_goalie_stats_rows": recent_goalie_stats_rows,
            "recent_goalie_stats_has_sog": recent_goalie_stats_has_sog,
            "recent_goalie_stats_has_xg": recent_goalie_stats_has_xg,
            "has_pair_on_ice_all": has_pair_on_ice_all,
            "league_page_views": league_page_views,
        },
    )


def team_edit(request: HttpRequest, team_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    team = logic.get_team(int(team_id), _session_user_id(request))
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        if name:
            _django_orm, m = _orm_modules()
            m.Team.objects.filter(id=int(team_id), user_id=_session_user_id(request)).update(
                name=name
            )
        f = request.FILES.get("logo")
        if f and getattr(f, "name", ""):
            p = logic.save_team_logo(f, int(team_id))
            _django_orm, m = _orm_modules()
            m.Team.objects.filter(id=int(team_id), user_id=_session_user_id(request)).update(
                logo_path=str(p)
            )
        messages.success(request, "Team updated")
        return redirect(f"/teams/{team_id}")
    return render(request, "team_edit.html", {"team": team})


def player_new(request: HttpRequest, team_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    team = logic.get_team(int(team_id), _session_user_id(request))
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        jersey = str(request.POST.get("jersey_number") or "").strip()
        position = str(request.POST.get("position") or "").strip()
        shoots = str(request.POST.get("shoots") or "").strip()
        if not name:
            messages.error(request, "Player name is required")
            return render(request, "player_edit.html", {"team": team})
        _django_orm, m = _orm_modules()
        m.Player.objects.create(
            user_id=_session_user_id(request),
            team_id=int(team_id),
            name=name,
            jersey_number=jersey or None,
            position=position or None,
            shoots=shoots or None,
            created_at=dt.datetime.now(),
            updated_at=None,
        )
        messages.success(request, "Player added")
        return redirect(f"/teams/{team_id}")
    return render(request, "player_edit.html", {"team": team})


def player_edit(request: HttpRequest, team_id: int, player_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    team = logic.get_team(int(team_id), _session_user_id(request))
    if not team:
        messages.error(request, "Not found")
        return redirect("/teams")
    _django_orm, m = _orm_modules()
    pl = (
        m.Player.objects.filter(
            id=int(player_id), team_id=int(team_id), user_id=_session_user_id(request)
        )
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
        .first()
    )
    if not pl:
        messages.error(request, "Not found")
        return redirect(f"/teams/{team_id}")
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        jersey = str(request.POST.get("jersey_number") or "").strip()
        position = str(request.POST.get("position") or "").strip()
        shoots = str(request.POST.get("shoots") or "").strip()
        m.Player.objects.filter(
            id=int(player_id), team_id=int(team_id), user_id=_session_user_id(request)
        ).update(
            name=name or pl["name"],
            jersey_number=jersey or None,
            position=position or None,
            shoots=shoots or None,
            updated_at=dt.datetime.now(),
        )
        messages.success(request, "Player updated")
        return redirect(f"/teams/{team_id}")
    return render(request, "player_edit.html", {"team": team, "player": pl})


def player_delete(request: HttpRequest, team_id: int, player_id: int) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    m.Player.objects.filter(
        id=int(player_id), team_id=int(team_id), user_id=_session_user_id(request)
    ).delete()
    messages.success(request, "Player deleted")
    return redirect(f"/teams/{team_id}")


def _attach_schedule_stats_icons(games: list[dict[str, Any]]) -> None:
    """
    Add `stats_icons` (+ optional `stats_note`) to schedule game dicts for UI badges.
    """

    def _compact_ws(v: Any) -> Any:
        if v is None:
            return None
        s = str(v).replace("\xa0", " ").strip()
        return " ".join(s.split())

    for g in games or []:
        for k in ("division_name", "team1_name", "team2_name", "game_type_name", "location"):
            if k not in g:
                continue
            try:
                g[k] = _compact_ws(g.get(k))
            except Exception:
                continue

    gids: list[int] = []
    for g in games or []:
        try:
            gid = int(g.get("id") or 0)
        except Exception:
            continue
        if gid > 0:
            gids.append(gid)
    if not gids:
        return
    gids = sorted(set(gids))

    _django_orm, m = _orm_modules()

    types_by_gid: dict[int, set[str]] = {int(gid): set() for gid in gids}

    try:
        for gid, tts_id in m.HkyGame.objects.filter(id__in=gids).values_list(
            "id", "timetoscore_game_id"
        ):
            try:
                gid_i = int(gid or 0)
            except Exception:
                continue
            if gid_i <= 0:
                continue
            if tts_id is not None:
                types_by_gid.setdefault(int(gid_i), set()).add("timetoscore")
    except Exception:
        pass

    # Shift rows imply at least "primary" spreadsheet stats are present.
    try:
        for gid in m.HkyGameShiftRow.objects.filter(game_id__in=gids).values_list(
            "game_id", flat=True
        ):
            try:
                gid_i = int(gid or 0)
            except Exception:
                continue
            if gid_i <= 0:
                continue
            types_by_gid.setdefault(int(gid_i), set()).add("primary")
    except Exception:
        pass

    # Event rows can come from multiple sources (CSV `Source` column is multi-valued).
    try:
        for gid0, src0 in (
            m.HkyGameEventRow.objects.filter(game_id__in=gids)
            .values_list("game_id", "source")
            .distinct()
        ):
            try:
                gid_i = int(gid0 or 0)
            except Exception:
                continue
            if gid_i <= 0:
                continue
            src = str(src0 or "").strip()
            if not src:
                continue
            recognized = False
            for tok in re.split(r"[,+;/\s]+", src):
                k = logic.canon_event_source_key(tok)
                if k == "shift_package":
                    k = "primary"
                if not k:
                    continue
                types_by_gid.setdefault(int(gid_i), set()).add(str(k))
                recognized = True
            if not recognized:
                types_by_gid.setdefault(int(gid_i), set()).add("primary")
    except Exception:
        pass

    icon_spec: list[tuple[str, str, str]] = [
        ("timetoscore", "TimeToScore", "T"),
        ("primary", "Primary", "P"),
        ("long", "Long", "L"),
        ("goals", "Goals", "G"),
        ("yaml-only", "YAML-only", "Y"),
    ]

    for g in games or []:
        try:
            gid = int(g.get("id") or 0)
        except Exception:
            continue
        if gid <= 0:
            continue
        types = set(types_by_gid.get(int(gid)) or set())
        try:
            if logic._extract_timetoscore_game_id_from_notes(g.get("notes")) is not None:
                types.add("timetoscore")
        except Exception:
            pass
        yaml_only = not types
        if yaml_only:
            types = {"yaml-only"}
        stats_note: Optional[str] = None
        try:
            stats_note = logic._extract_game_stats_note_from_notes(g.get("notes"))
        except Exception:
            stats_note = None
        g["stats_note"] = stats_note

        icons: list[dict[str, Any]] = []
        for key, label, text in icon_spec:
            if key == "yaml-only":
                if not yaml_only:
                    continue
            else:
                if key not in types:
                    continue
            title = label
            if stats_note:
                title = f"{label}: {stats_note}"
            icons.append(
                {
                    "key": key,
                    "title": title,
                    "text": text,
                }
            )
        g["stats_icons"] = icons


def _extract_seed_placeholder_spec(
    *, notes: Optional[str], side: str, team_name: Optional[str]
) -> Optional[dict[str, Any]]:
    """
    Return a seed placeholder spec for "team1"/"team2" if present, otherwise None.

    Supports:
      - New imports: JSON notes field "seed_placeholders".
      - Legacy data: team name matches "<division> Seed <n>".
    """
    side_key = str(side or "").strip()
    if side_key not in {"team1", "team2"}:
        return None

    if notes:
        try:
            d = json.loads(notes)
        except Exception:
            d = None
        if isinstance(d, dict):
            sp = d.get("seed_placeholders")
            if isinstance(sp, dict):
                raw = sp.get(side_key)
                if isinstance(raw, dict) and raw.get("seed") is not None:
                    return dict(raw)

    ph = logic.parse_seed_placeholder_name(str(team_name or ""))
    if ph is None:
        return None
    return {"seed": int(ph.seed), "division_token": str(ph.division_token), "raw": str(ph.raw)}


def _resolve_seed_placeholders_in_schedule_games(
    *, league_id: int, games: list[dict[str, Any]]
) -> None:
    """
    Mutate `games` in-place: replace "<division> Seed <n>" placeholders with the current
    team occupying that seed in the division standings.
    """
    if not league_id or not games:
        return

    # Collect placeholder requests grouped by division.
    needed: dict[str, set[int]] = {}
    spec_by_game_side: dict[tuple[int, str], dict[str, Any]] = {}
    for g in games:
        gid = int(g.get("id") or 0)
        if gid <= 0:
            continue
        for side in ("team1", "team2"):
            spec = _extract_seed_placeholder_spec(
                notes=g.get("notes"),
                side=side,
                team_name=(g.get(f"{side}_name") if side in {"team1", "team2"} else None),
            )
            if spec is None:
                continue
            try:
                seed_i = int(spec.get("seed"))
            except Exception:
                continue
            if seed_i <= 0:
                continue
            div_hint = str(spec.get("division_name_hint") or "").strip()
            if not div_hint:
                div_hint = str(g.get("division_name") or "").strip()
            if not div_hint:
                continue
            needed.setdefault(div_hint, set()).add(seed_i)
            spec_by_game_side[(gid, side)] = spec

    if not needed:
        return

    _django_orm, m = _orm_modules()

    ranked_by_div: dict[str, list[int]] = {}
    resolved_ids: set[int] = set()
    for div_name, seeds in needed.items():
        ranked = logic.division_standings_team_ids(int(league_id), str(div_name))
        ranked_by_div[str(div_name)] = list(ranked)
        for seed_i in seeds:
            if seed_i <= 0 or seed_i > len(ranked):
                continue
            tid = int(ranked[seed_i - 1])
            resolved_ids.add(tid)

    name_by_id = {
        int(r["id"]): str(r.get("name") or "")
        for r in m.Team.objects.filter(id__in=sorted(resolved_ids)).values("id", "name")
    }

    for g in games:
        gid = int(g.get("id") or 0)
        if gid <= 0:
            continue
        for side in ("team1", "team2"):
            spec = spec_by_game_side.get((gid, side))
            if spec is None:
                continue
            try:
                seed_i = int(spec.get("seed"))
            except Exception:
                continue
            div_hint = (
                str(spec.get("division_name_hint") or "").strip()
                or str(g.get("division_name") or "").strip()
            )
            ranked = ranked_by_div.get(div_hint) or []
            if 0 < seed_i <= len(ranked):
                tid = int(ranked[seed_i - 1])
                g[f"{side}_id"] = tid
                g[f"{side}_name"] = name_by_id.get(tid) or g.get(f"{side}_name")
            else:
                # Fall back to a readable placeholder string instead of the generic placeholder team.
                if spec.get("raw"):
                    g[f"{side}_name"] = str(spec.get("raw"))


def schedule(request: HttpRequest) -> HttpResponse:
    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            entity_id=0,
            viewer_user_id=_session_user_id(request),
            league_owner_user_id=league_owner_user_id,
        )

    selected_division = str(request.GET.get("division") or "").strip() or None
    selected_team_id = str(request.GET.get("team_id") or "")
    team_id_i: Optional[int] = None
    try:
        team_id_i = int(selected_team_id) if selected_team_id.strip() else None
    except Exception:
        team_id_i = None

    divisions: list[Any] = []
    league_teams: list[dict[str, Any]] = []
    _django_orm, m = _orm_modules()

    games: list[dict[str, Any]] = []
    if league_id:
        divisions = list(
            m.LeagueTeam.objects.filter(league_id=int(league_id))
            .exclude(division_name__isnull=True)
            .exclude(division_name="")
            .values_list("division_name", flat=True)
            .distinct()
        )
        divisions.sort(key=logic.division_sort_key)
        if selected_division:
            league_teams = list(
                m.Team.objects.filter(
                    league_teams__league_id=int(league_id),
                    league_teams__division_name=str(selected_division),
                )
                .distinct()
                .order_by("name")
                .values("id", "name")
            )
        else:
            league_teams = list(
                m.Team.objects.filter(league_teams__league_id=int(league_id))
                .distinct()
                .order_by("name")
                .values("id", "name")
            )
        league_teams = [
            t
            for t in league_teams
            if str(t.get("name") or "").strip() != logic.SEED_PLACEHOLDER_TEAM_NAME
            and not logic.is_seed_placeholder_name(str(t.get("name") or ""))
        ]
        if team_id_i is not None and not any(int(t["id"]) == int(team_id_i) for t in league_teams):
            team_id_i = None
            selected_team_id = ""

        lg_qs = m.LeagueGame.objects.filter(league_id=int(league_id))
        if selected_division:
            lg_qs = lg_qs.filter(division_name=str(selected_division))

        league_team_div_map = {
            int(tid): (str(dn).strip() if dn is not None else None)
            for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                "team_id", "division_name"
            )
        }
        rows_raw = list(
            lg_qs.select_related("game", "game__team1", "game__team2", "game__game_type").values(
                "game_id",
                "division_name",
                "sort_order",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__stats_imported_at",
                "game__created_at",
                "game__updated_at",
                "game__team1__name",
                "game__team2__name",
                "game__game_type__name",
            )
        )
        for r0 in rows_raw:
            t1 = int(r0["game__team1_id"])
            t2 = int(r0["game__team2_id"])
            games.append(
                {
                    "id": int(r0["game_id"]),
                    "user_id": int(r0["game__user_id"]),
                    "team1_id": t1,
                    "team2_id": t2,
                    "game_type_id": r0.get("game__game_type_id"),
                    "starts_at": r0.get("game__starts_at"),
                    "location": r0.get("game__location"),
                    "notes": r0.get("game__notes"),
                    "team1_score": r0.get("game__team1_score"),
                    "team2_score": r0.get("game__team2_score"),
                    "is_final": r0.get("game__is_final"),
                    "stats_imported_at": r0.get("game__stats_imported_at"),
                    "created_at": r0.get("game__created_at"),
                    "updated_at": r0.get("game__updated_at"),
                    "team1_name": r0.get("game__team1__name"),
                    "team2_name": r0.get("game__team2__name"),
                    "game_type_name": r0.get("game__game_type__name"),
                    "division_name": r0.get("division_name"),
                    "sort_order": r0.get("sort_order"),
                    "team1_league_division_name": league_team_div_map.get(t1),
                    "team2_league_division_name": league_team_div_map.get(t2),
                }
            )
        _resolve_seed_placeholders_in_schedule_games(league_id=int(league_id), games=games)
        if team_id_i is not None:
            games = [
                g
                for g in games
                if int(g.get("team1_id") or 0) == int(team_id_i)
                or int(g.get("team2_id") or 0) == int(team_id_i)
            ]
    else:
        rows = list(
            m.HkyGame.objects.filter(user_id=_session_user_id(request))
            .select_related("team1", "team2", "game_type")
            .values(
                "id",
                "user_id",
                "team1_id",
                "team2_id",
                "game_type_id",
                "starts_at",
                "location",
                "notes",
                "team1_score",
                "team2_score",
                "is_final",
                "stats_imported_at",
                "created_at",
                "updated_at",
                "team1__name",
                "team2__name",
                "game_type__name",
            )
        )
        for r0 in rows:
            games.append(
                {
                    "id": int(r0["id"]),
                    "user_id": int(r0["user_id"]),
                    "team1_id": int(r0["team1_id"]),
                    "team2_id": int(r0["team2_id"]),
                    "game_type_id": r0.get("game_type_id"),
                    "starts_at": r0.get("starts_at"),
                    "location": r0.get("location"),
                    "notes": r0.get("notes"),
                    "team1_score": r0.get("team1_score"),
                    "team2_score": r0.get("team2_score"),
                    "is_final": r0.get("is_final"),
                    "stats_imported_at": r0.get("stats_imported_at"),
                    "created_at": r0.get("created_at"),
                    "updated_at": r0.get("updated_at"),
                    "team1_name": r0.get("team1__name"),
                    "team2_name": r0.get("team2__name"),
                    "game_type_name": r0.get("game_type__name"),
                }
            )

    now_dt = dt.datetime.now()
    is_league_admin = bool(
        league_id and _is_league_admin(int(league_id), _session_user_id(request))
    )
    for g2 in games or []:
        sdt = g2.get("starts_at")
        started = False
        if sdt is not None:
            try:
                started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
            except Exception:
                started = False
        has_score = (
            (g2.get("team1_score") is not None)
            or (g2.get("team2_score") is not None)
            or bool(g2.get("is_final"))
        )
        g2["can_view_summary"] = bool(has_score or (sdt is None) or started)
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None
        try:
            g2["can_edit"] = bool(
                int(g2.get("user_id") or 0) == _session_user_id(request) or is_league_admin
            )
        except Exception:
            g2["can_edit"] = bool(is_league_admin)

    _attach_schedule_stats_icons(games)
    games = logic.sort_games_schedule_order(games or [])
    league_page_views = None
    if league_id and is_league_owner:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            "entity_id": 0,
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }
    return render(
        request,
        "schedule.html",
        {
            "games": games,
            "league_view": bool(league_id),
            "divisions": divisions,
            "league_teams": league_teams,
            "selected_division": selected_division or "",
            "selected_team_id": str(team_id_i) if team_id_i is not None else "",
            "league_page_views": league_page_views,
        },
    )


def schedule_new(request: HttpRequest) -> HttpResponse:
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    my_teams = list(
        m.Team.objects.filter(user_id=_session_user_id(request), is_external=False)
        .order_by("name")
        .values("id", "name")
    )
    gt = list(m.GameType.objects.order_by("name").values("id", "name"))
    if request.method == "POST":
        team1_id = int(request.POST.get("team1_id") or 0)
        team2_id = int(request.POST.get("team2_id") or 0)
        opp_name = str(request.POST.get("opponent_name") or "").strip()
        game_type_id = int(request.POST.get("game_type_id") or 0)
        starts_at = str(request.POST.get("starts_at") or "").strip()
        location = str(request.POST.get("location") or "").strip()

        if not team1_id and not team2_id:
            messages.error(request, "Select at least one of your teams")
            return render(request, "schedule_new.html", {"my_teams": my_teams, "game_types": gt})
        if team1_id and not team2_id:
            team2_id = logic.ensure_external_team(_session_user_id(request), opp_name or "Opponent")
        elif team2_id and not team1_id:
            team1_id = logic.ensure_external_team(_session_user_id(request), opp_name or "Opponent")

        gid = logic.create_hky_game(
            user_id=_session_user_id(request),
            team1_id=int(team1_id),
            team2_id=int(team2_id),
            game_type_id=int(game_type_id) if game_type_id else None,
            starts_at=logic.to_dt(starts_at),
            location=location or None,
        )

        league_id = request.session.get("league_id")
        if league_id:
            from django.db import transaction

            try:
                with transaction.atomic():
                    m.LeagueTeam.objects.get_or_create(
                        league_id=int(league_id), team_id=int(team1_id)
                    )
                    m.LeagueTeam.objects.get_or_create(
                        league_id=int(league_id), team_id=int(team2_id)
                    )
                    m.LeagueGame.objects.get_or_create(league_id=int(league_id), game_id=int(gid))
            except Exception:
                pass
        messages.success(request, "Game created")
        return redirect(f"/hky/games/{gid}")
    return render(request, "schedule_new.html", {"my_teams": my_teams, "game_types": gt})


def hky_game_detail(request: HttpRequest, game_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    league_owner_user_id: Optional[int] = None
    is_league_owner = False
    if league_id:
        league_owner_user_id = logic._get_league_owner_user_id(None, int(league_id))
        is_league_owner = bool(
            league_owner_user_id is not None
            and int(league_owner_user_id) == _session_user_id(request)
        )

    show_shift_data = True
    if league_id:
        try:
            show_shift_data = _league_show_shift_data(int(league_id))
        except Exception:
            show_shift_data = False

    _django_orm, m = _orm_modules()
    session_uid = _session_user_id(request)
    is_league_admin = False
    if league_id:
        try:
            is_league_admin = bool(_is_league_admin(int(league_id), int(session_uid)))
        except Exception:
            is_league_admin = False
    owned_row = (
        m.HkyGame.objects.filter(id=int(game_id), user_id=session_uid)
        .select_related("team1", "team2")
        .values(
            "id",
            "user_id",
            "team1_id",
            "team2_id",
            "game_type_id",
            "starts_at",
            "location",
            "notes",
            "team1_score",
            "team2_score",
            "is_final",
            "stats_imported_at",
            "created_at",
            "updated_at",
            "team1__name",
            "team2__name",
            "team1__is_external",
            "team2__is_external",
        )
        .first()
    )
    game: Optional[dict[str, Any]] = None
    if owned_row:
        game = {
            "id": int(owned_row["id"]),
            "user_id": int(owned_row["user_id"]),
            "team1_id": int(owned_row["team1_id"]),
            "team2_id": int(owned_row["team2_id"]),
            "game_type_id": owned_row.get("game_type_id"),
            "starts_at": owned_row.get("starts_at"),
            "location": owned_row.get("location"),
            "notes": owned_row.get("notes"),
            "team1_score": owned_row.get("team1_score"),
            "team2_score": owned_row.get("team2_score"),
            "is_final": owned_row.get("is_final"),
            "stats_imported_at": owned_row.get("stats_imported_at"),
            "created_at": owned_row.get("created_at"),
            "updated_at": owned_row.get("updated_at"),
            "team1_name": owned_row.get("team1__name"),
            "team2_name": owned_row.get("team2__name"),
            "team1_ext": owned_row.get("team1__is_external"),
            "team2_ext": owned_row.get("team2__is_external"),
        }
    elif league_id:
        league_row = (
            m.LeagueGame.objects.filter(league_id=int(league_id), game_id=int(game_id))
            .select_related("game", "game__team1", "game__team2")
            .values(
                "game_id",
                "division_name",
                "game__user_id",
                "game__team1_id",
                "game__team2_id",
                "game__game_type_id",
                "game__starts_at",
                "game__location",
                "game__notes",
                "game__team1_score",
                "game__team2_score",
                "game__is_final",
                "game__stats_imported_at",
                "game__created_at",
                "game__updated_at",
                "game__team1__name",
                "game__team2__name",
                "game__team1__is_external",
                "game__team2__is_external",
            )
            .first()
        )
        if league_row:
            t1 = int(league_row["game__team1_id"])
            t2 = int(league_row["game__team2_id"])
            div_map = {
                int(tid): (str(dn).strip() if dn is not None else None)
                for tid, dn in m.LeagueTeam.objects.filter(
                    league_id=int(league_id),
                    team_id__in=[t1, t2],
                ).values_list("team_id", "division_name")
            }
            game = {
                "id": int(league_row["game_id"]),
                "user_id": int(league_row["game__user_id"]),
                "team1_id": t1,
                "team2_id": t2,
                "game_type_id": league_row.get("game__game_type_id"),
                "starts_at": league_row.get("game__starts_at"),
                "location": league_row.get("game__location"),
                "notes": league_row.get("game__notes"),
                "team1_score": league_row.get("game__team1_score"),
                "team2_score": league_row.get("game__team2_score"),
                "is_final": league_row.get("game__is_final"),
                "stats_imported_at": league_row.get("game__stats_imported_at"),
                "created_at": league_row.get("game__created_at"),
                "updated_at": league_row.get("game__updated_at"),
                "team1_name": league_row.get("game__team1__name"),
                "team2_name": league_row.get("game__team2__name"),
                "team1_ext": league_row.get("game__team1__is_external"),
                "team2_ext": league_row.get("game__team2__is_external"),
                "division_name": league_row.get("division_name"),
                "team1_league_division_name": div_map.get(t1),
                "team2_league_division_name": div_map.get(t2),
            }

    if not game:
        messages.error(request, "Not found")
        return redirect("/schedule")

    try:
        game["game_video_url"] = logic._sanitize_http_url(
            logic._extract_game_video_url_from_notes(game.get("notes"))
        )
    except Exception:
        game["game_video_url"] = None

    is_owner = int(game.get("user_id") or 0) == int(session_uid)

    now_dt = dt.datetime.now()
    sdt = game.get("starts_at")
    started = False
    if sdt is not None:
        try:
            started = logic.to_dt(sdt) is not None and logic.to_dt(sdt) <= now_dt
        except Exception:
            started = False
    has_score = (
        (game.get("team1_score") is not None)
        or (game.get("team2_score") is not None)
        or bool(game.get("is_final"))
    )
    can_view_summary = bool(has_score or (sdt is None) or started)
    if not can_view_summary:
        raise Http404

    if league_id:
        logic._record_league_page_view(
            None,
            int(league_id),
            kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME,
            entity_id=int(game_id),
            viewer_user_id=session_uid,
            league_owner_user_id=league_owner_user_id,
        )

    tts_linked = logic._extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

    return_to = logic._safe_return_to_url(request.GET.get("return_to"), default="/schedule")

    can_edit = bool(is_owner)
    if league_id and not can_edit:
        try:
            can_edit = bool(_is_league_admin(int(league_id), session_uid))
        except Exception:
            can_edit = False
    edit_mode = bool(can_edit and (request.GET.get("edit") == "1" or request.method == "POST"))

    team1_players_qs = m.Player.objects.filter(team_id=int(game["team1_id"]))
    team2_players_qs = m.Player.objects.filter(team_id=int(game["team2_id"]))
    if is_owner:
        team1_players_qs = team1_players_qs.filter(user_id=session_uid)
        team2_players_qs = team2_players_qs.filter(user_id=session_uid)

    team1_players = list(
        team1_players_qs.order_by("jersey_number", "name").values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    team2_players = list(
        team2_players_qs.order_by("jersey_number", "name").values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    player_stats_imported_by_pid: dict[int, dict[str, Optional[int]]] = {}
    stats_rows: list[dict[str, Any]] = []
    try:
        stats_rows.extend(
            _player_stat_rows_from_event_tables_for_team_games(
                team_id=int(game["team1_id"]),
                schedule_games=[dict(game)],
                roster_players=list(team1_players or []),
                show_shift_data=bool(show_shift_data),
            )
        )
        stats_rows.extend(
            _player_stat_rows_from_event_tables_for_team_games(
                team_id=int(game["team2_id"]),
                schedule_games=[dict(game)],
                roster_players=list(team2_players or []),
                show_shift_data=bool(show_shift_data),
            )
        )
    except Exception:
        logger.exception("Failed to compute event-derived player stat rows for game_id=%s", game_id)
        stats_rows = []
    if not show_shift_data:
        _mask_shift_stats_in_player_stat_rows(stats_rows)
    team1_skaters, team1_goalies, team1_hc, team1_ac = logic.split_roster(team1_players or [])
    team2_skaters, team2_goalies, team2_hc, team2_ac = logic.split_roster(team2_players or [])
    team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
    team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
    stats_by_pid = {r0["player_id"]: r0 for r0 in stats_rows}
    try:
        _overlay_game_player_stats_from_event_rows(game_id=int(game_id), stats_by_pid=stats_by_pid)
    except Exception:
        logger.exception("Failed to overlay goal-derived stats for game_id=%s", game_id)
    if show_shift_data:
        try:
            _overlay_on_ice_goal_stats_from_shift_rows(
                game_id=int(game_id), stats_by_pid=stats_by_pid
            )
        except Exception:
            # Best-effort: shift-derived overlays are optional and should not break page rendering.
            pass

    events_headers, events_rows, events_meta = _load_game_events_for_display(game_id=int(game_id))

    try:
        events_headers, events_rows = logic.normalize_game_events_csv(events_headers, events_rows)
    except Exception:
        pass
    if tts_linked:
        try:
            events_headers, events_rows = logic.enrich_timetoscore_goals_with_long_video_times(
                existing_headers=events_headers,
                existing_rows=events_rows,
                incoming_headers=events_headers,
                incoming_rows=events_rows,
            )
            events_headers, events_rows = logic.enrich_timetoscore_penalties_with_video_times(
                existing_headers=events_headers,
                existing_rows=events_rows,
                incoming_headers=events_headers,
                incoming_rows=events_rows,
            )
        except Exception:
            # Best-effort: keep UI working even if enrichment fails.
            pass
    events_rows = logic.filter_events_rows_prefer_timetoscore_for_goal_assist(
        events_rows, tts_linked=tts_linked
    )
    try:
        events_headers, events_rows = logic.normalize_events_video_time_for_display(
            events_headers, events_rows
        )
        events_headers, events_rows = logic.filter_events_headers_drop_empty_on_ice_split(
            events_headers, events_rows
        )
        events_rows = logic.sort_events_rows_default(events_rows)
    except Exception:
        pass
    if events_meta is not None:
        try:
            events_meta["count"] = len(events_rows)
            events_meta["sources"] = logic.summarize_event_sources(
                events_rows, fallback_source_label=str(events_meta.get("source_label") or "")
            )
        except Exception:
            pass

    scoring_by_period_rows = logic.compute_team_scoring_by_period_from_events(
        events_rows, tts_linked=tts_linked
    )
    try:
        game_event_stats_rows = logic.compute_game_event_stats_by_side(events_rows)
    except Exception:
        game_event_stats_rows = []

    goalie_stats = {"home": [], "away": [], "meta": {"has_sog": False}}
    show_goalie_stats = True
    if league_id:
        try:
            show_goalie_stats = _league_show_goalie_stats(int(league_id))
        except Exception:
            show_goalie_stats = False
    if show_goalie_stats:
        try:
            goalie_event_rows = list(
                m.HkyGameEventRow.objects.filter(
                    game_id=int(game_id),
                    event_type__key__in=[
                        "goal",
                        "expectedgoal",
                        "sog",
                        "shotongoal",
                        "goaliechange",
                    ],
                )
                .select_related("event_type")
                .values(
                    "event_type__key",
                    "event_id",
                    "team_side",
                    "period",
                    "game_seconds",
                    "player_id",
                    "attributed_players",
                    "attributed_jerseys",
                    "details",
                )
            )
            goalie_stats = logic.compute_goalie_stats_for_game(
                goalie_event_rows,
                home_goalies=team1_goalies,
                away_goalies=team2_goalies,
            )
        except Exception:
            goalie_stats = {"home": [], "away": [], "meta": {"has_sog": False}}

    player_stats_import_meta: Optional[dict[str, Any]] = None

    try:
        player_has_events_by_pid = _compute_player_has_events_by_pid_for_game(
            events_rows=events_rows,
            team1_players=list(team1_skaters),
            team2_players=list(team2_skaters),
        )
    except Exception:
        player_has_events_by_pid = {}

    display_by_pid: dict[int, dict[str, Any]] = {}
    for p in list(team1_skaters) + list(team2_skaters):
        try:
            pid = int(p.get("id") or 0)
        except Exception:
            continue
        if pid <= 0:
            continue
        db_row = stats_by_pid.get(pid) or {}
        base: dict[str, Any] = {"player_id": int(pid), "gp": 1}
        for k in logic.PLAYER_STATS_SUM_KEYS:
            base[str(k)] = logic._int0(db_row.get(str(k)))
        display_by_pid[int(pid)] = logic.compute_player_display_stats(base)

    team1_player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(list(team1_skaters), display_by_pid)
    )
    team2_player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(list(team2_skaters), display_by_pid)
    )
    for r0 in list(team1_player_stats_rows) + list(team2_player_stats_rows):
        try:
            pid_i = int(r0.get("player_id") or 0)
        except Exception:
            continue
        r0["has_events"] = bool(player_has_events_by_pid.get(pid_i))

    game_player_stats_columns = logic.build_game_player_stats_display_columns(
        rows=list(team1_player_stats_rows) + list(team2_player_stats_rows)
    )
    player_stats_cell_conflicts_by_pid: dict[int, dict[str, str]] = {}
    if show_shift_data and player_stats_imported_by_pid:
        for pid, imported in (player_stats_imported_by_pid or {}).items():
            try:
                pid_i = int(pid)
            except Exception:
                continue
            cur = stats_by_pid.get(int(pid_i)) or {}

            def _cmp_int(key: str) -> tuple[Optional[int], Optional[int]]:
                try:
                    a0 = imported.get(key)
                    a = int(a0) if a0 is not None else None
                except Exception:
                    a = None
                try:
                    b0 = cur.get(key)
                    b = int(b0) if b0 is not None else None
                except Exception:
                    b = None
                return a, b

            for k in ("toi_seconds", "shifts", "gf_counted", "ga_counted", "plus_minus"):
                a, b = _cmp_int(k)
                if a is None or b is None or int(a) == int(b):
                    continue

                msg = f"Imported {k}={a}, computed from shifts={b}"
                if k == "plus_minus":
                    gf = logic._int0(cur.get("gf_counted"))
                    ga = logic._int0(cur.get("ga_counted"))
                    msg = (
                        f"Imported +/-={a:+d}, computed from shifts={b:+d} "
                        f"(GF={int(gf)}, GA={int(ga)}; goals at shift start excluded)"
                    )
                elif k == "gf_counted":
                    msg = f"Imported GF counted={a}, computed from shifts={b}"
                elif k == "ga_counted":
                    msg = f"Imported GA counted={a}, computed from shifts={b}"
                elif k == "toi_seconds":
                    msg = f"Imported TOI={a}s, computed from shifts={b}s"
                elif k == "shifts":
                    msg = f"Imported shifts={a}, computed from shift rows={b}"

                player_stats_cell_conflicts_by_pid.setdefault(int(pid_i), {})[str(k)] = str(msg)
    player_stats_import_warning: Optional[str] = None

    if request.method == "POST" and not edit_mode:
        messages.error(
            request, "You do not have permission to edit this game in the selected league."
        )
        return redirect(f"/hky/games/{int(game_id)}?return_to={return_to}")

    if request.method == "POST" and edit_mode:
        loc = str(request.POST.get("location") or "").strip()
        starts_at = str(request.POST.get("starts_at") or "").strip()
        t1_score = request.POST.get("team1_score")
        t2_score = request.POST.get("team2_score")
        is_final = bool(request.POST.get("is_final"))
        from django.db import transaction

        starts_at_dt = logic.to_dt(starts_at)
        updates = {
            "location": loc or None,
            "starts_at": starts_at_dt,
            "team1_score": int(t1_score) if (t1_score or "").strip() else None,
            "team2_score": int(t2_score) if (t2_score or "").strip() else None,
            "is_final": bool(is_final),
            "updated_at": dt.datetime.now(),
        }
        with transaction.atomic():
            if is_owner:
                m.HkyGame.objects.filter(id=int(game_id), user_id=session_uid).update(**updates)
            else:
                m.HkyGame.objects.filter(id=int(game_id)).update(**updates)

        messages.success(request, "Game updated")
        return redirect(f"/hky/games/{int(game_id)}?return_to={return_to}")

    can_view_league_page_views = bool(league_id and (is_league_owner or is_league_admin))
    league_page_views = None
    if league_id and can_view_league_page_views:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME, entity_id=int(game_id)
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME, entity_id=int(game_id)
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_GAME,
            "entity_id": int(game_id),
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }

    shift_timeline_rows: list[dict[str, str]] = []
    if show_shift_data:
        try:
            shift_timeline_rows = _load_game_shift_rows_for_timeline(game_id=int(game_id))
        except Exception:
            shift_timeline_rows = []

    return render(
        request,
        "hky_game_detail.html",
        {
            "game": game,
            "team1_roster": team1_roster,
            "team2_roster": team2_roster,
            "team1_player_stats_rows": team1_player_stats_rows,
            "team2_player_stats_rows": team2_player_stats_rows,
            "editable": bool(edit_mode),
            "can_edit": bool(can_edit),
            "edit_mode": bool(edit_mode),
            "back_url": return_to,
            "return_to": return_to,
            "events_headers": events_headers,
            "events_rows": events_rows,
            "events_meta": events_meta,
            "shift_timeline_rows": shift_timeline_rows,
            "show_shift_data": bool(show_shift_data),
            "scoring_by_period_rows": scoring_by_period_rows,
            "game_event_stats_rows": game_event_stats_rows,
            "user_video_clip_len_s": logic.get_user_video_clip_len_s(
                None, int(request.session.get("user_id") or 0)
            ),
            "user_is_logged_in": True,
            "game_player_stats_columns": game_player_stats_columns,
            "player_stats_cell_conflicts_by_pid": player_stats_cell_conflicts_by_pid,
            "player_stats_import_meta": player_stats_import_meta,
            "player_stats_import_warning": player_stats_import_warning,
            "goalie_stats_home_rows": goalie_stats.get("home") or [],
            "goalie_stats_away_rows": goalie_stats.get("away") or [],
            "goalie_stats_has_sog": bool((goalie_stats.get("meta") or {}).get("has_sog")),
            "goalie_stats_has_xg": bool((goalie_stats.get("meta") or {}).get("has_xg")),
            "can_view_league_page_views": can_view_league_page_views,
            "league_page_views": league_page_views,
        },
    )


def game_types(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    if request.method == "POST":
        name = str(request.POST.get("name") or "").strip()
        if name:
            try:
                from django.db import IntegrityError, transaction

                with transaction.atomic():
                    m.GameType.objects.create(name=name, is_default=False)
                messages.success(request, "Game type added")
            except IntegrityError:
                messages.error(request, "Failed to add game type (may already exist)")
        return redirect("/game_types")
    rows = list(m.GameType.objects.order_by("name").values("id", "name", "is_default"))
    return render(request, "game_types.html", {"game_types": rows})


def leagues_index(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    r = _require_login(request)
    if r:
        return r
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    uid = _session_user_id(request)
    admin_ids = set(
        m.LeagueMember.objects.filter(user_id=uid, role__in=["admin", "owner"]).values_list(
            "league_id", flat=True
        )
    )
    leagues: list[dict[str, Any]] = []
    for row in (
        m.League.objects.filter(Q(is_shared=True) | Q(owner_user_id=uid) | Q(members__user_id=uid))
        .distinct()
        .order_by("name")
        .values(
            "id",
            "name",
            "is_shared",
            "is_public",
            "show_goalie_stats",
            "show_shift_data",
            "owner_user_id",
        )
    ):
        lid = int(row["id"])
        is_owner = int(int(row["owner_user_id"]) == uid)
        is_admin = 1 if is_owner or lid in admin_ids else 0
        leagues.append(
            {
                "id": lid,
                "name": row["name"],
                "is_shared": bool(row["is_shared"]),
                "is_public": bool(row.get("is_public")),
                "show_goalie_stats": bool(row.get("show_goalie_stats")),
                "show_shift_data": bool(row.get("show_shift_data")),
                "is_owner": is_owner,
                "is_admin": is_admin,
            }
        )
    selected_league_id = request.session.get("league_id")
    active_league: Optional[dict[str, Any]] = None
    try:
        selected_id_i = int(selected_league_id) if selected_league_id is not None else None
    except Exception:
        selected_id_i = None
    if selected_id_i is not None:
        for l0 in leagues:
            try:
                if int(l0.get("id") or 0) == int(selected_id_i):
                    active_league = l0
                    break
            except Exception:
                continue
    return render(request, "leagues.html", {"leagues": leagues, "active_league": active_league})


def leagues_new(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    name = str(request.POST.get("name") or "").strip()
    is_shared = 1 if str(request.POST.get("is_shared") or "") == "1" else 0
    is_public = 1 if str(request.POST.get("is_public") or "") == "1" else 0
    show_goalie_stats = 1 if str(request.POST.get("show_goalie_stats") or "") == "1" else 0
    show_shift_data = 1 if str(request.POST.get("show_shift_data") or "") == "1" else 0
    if not name:
        messages.error(request, "Name is required")
        return redirect("/leagues")
    _django_orm, m = _orm_modules()
    from django.db import IntegrityError, transaction

    uid = _session_user_id(request)
    now = dt.datetime.now()
    try:
        with transaction.atomic():
            league = m.League.objects.create(
                name=name,
                owner_user_id=uid,
                is_shared=bool(is_shared),
                is_public=bool(is_public),
                show_goalie_stats=bool(show_goalie_stats),
                show_shift_data=bool(show_shift_data),
                created_at=now,
                updated_at=None,
            )
            lid = int(league.id)
            m.LeagueMember.objects.get_or_create(
                league_id=lid,
                user_id=uid,
                defaults={"role": "admin", "created_at": now},
            )
    except IntegrityError:
        messages.error(request, "Failed to create league (name may already exist)")
        return redirect("/leagues")
    request.session["league_id"] = lid
    messages.success(request, "League created and selected")
    return redirect("/leagues")


def leagues_update(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")
    is_shared = 1 if str(request.POST.get("is_shared") or "") == "1" else 0
    is_public = 1 if str(request.POST.get("is_public") or "") == "1" else 0
    show_goalie_stats = 1 if str(request.POST.get("show_goalie_stats") or "") == "1" else 0
    show_shift_data = 1 if str(request.POST.get("show_shift_data") or "") == "1" else 0
    _django_orm, m = _orm_modules()
    m.League.objects.filter(id=int(league_id)).update(
        is_shared=bool(is_shared),
        is_public=bool(is_public),
        show_goalie_stats=bool(show_goalie_stats),
        show_shift_data=bool(show_shift_data),
        updated_at=dt.datetime.now(),
    )
    messages.success(request, "League settings updated")
    return redirect("/leagues")


def leagues_delete(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized to delete this league")
        return redirect("/leagues")

    def _chunks(ids: list[int], n: int = 500) -> list[list[int]]:
        return [ids[i : i + n] for i in range(0, len(ids), n)]

    _django_orm, m = _orm_modules()
    from django.db import transaction

    try:
        with transaction.atomic():
            game_ids = list(
                m.LeagueGame.objects.filter(league_id=int(league_id)).values_list(
                    "game_id", flat=True
                )
            )
            team_ids = list(
                m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                    "team_id", flat=True
                )
            )

            m.LeagueMember.objects.filter(league_id=int(league_id)).delete()
            m.LeaguePageView.objects.filter(league_id=int(league_id)).delete()
            m.LeagueGame.objects.filter(league_id=int(league_id)).delete()
            m.LeagueTeam.objects.filter(league_id=int(league_id)).delete()
            m.League.objects.filter(id=int(league_id)).delete()

            if game_ids:
                for chunk in _chunks(sorted({int(x) for x in game_ids}), n=500):
                    m.PlayerPeriodStat.objects.filter(game_id__in=[int(x) for x in chunk]).delete()
                    m.HkyGameEventSuppression.objects.filter(
                        game_id__in=[int(x) for x in chunk]
                    ).delete()
                    m.HkyGamePlayer.objects.filter(game_id__in=[int(x) for x in chunk]).delete()
                    m.HkyGameEventRow.objects.filter(game_id__in=[int(x) for x in chunk]).delete()

                still_used = set(
                    m.LeagueGame.objects.filter(game_id__in=[int(x) for x in game_ids]).values_list(
                        "game_id", flat=True
                    )
                )
                safe_game_ids = [int(gid) for gid in game_ids if int(gid) not in still_used]
                for chunk in _chunks(sorted({int(x) for x in safe_game_ids}), n=500):
                    m.HkyGame.objects.filter(id__in=[int(x) for x in chunk]).delete()

            if team_ids:
                eligible_team_ids = list({int(x) for x in team_ids})
                still_used = set(
                    m.LeagueTeam.objects.filter(team_id__in=eligible_team_ids)
                    .exclude(league_id=int(league_id))
                    .values_list("team_id", flat=True)
                )
                still_used |= set(
                    m.HkyGame.objects.filter(team1_id__in=eligible_team_ids).values_list(
                        "team1_id", flat=True
                    )
                )
                still_used |= set(
                    m.HkyGame.objects.filter(team2_id__in=eligible_team_ids).values_list(
                        "team2_id", flat=True
                    )
                )
                safe_team_ids = [
                    int(tid) for tid in eligible_team_ids if int(tid) not in still_used
                ]
                for chunk in _chunks(sorted(safe_team_ids), n=500):
                    m.Team.objects.filter(id__in=[int(x) for x in chunk]).delete()
        if int(request.session.get("league_id") or 0) == int(league_id):
            request.session.pop("league_id", None)
        messages.success(request, "League and associated data deleted")
    except Exception:
        messages.error(request, "Failed to delete league")
    return redirect("/leagues")


def league_members(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")

    _django_orm, m = _orm_modules()

    if request.method == "POST":
        email = str(request.POST.get("email") or "").strip().lower()
        role = str(request.POST.get("role") or "viewer")
        if not email:
            messages.error(request, "Email required")
            return redirect(f"/leagues/{int(league_id)}/members")
        from django.db import transaction

        uid = m.User.objects.filter(email=email).values_list("id", flat=True).first()
        if uid is None:
            messages.error(request, "User not found. Ask them to register first.")
            return redirect(f"/leagues/{int(league_id)}/members")
        now = dt.datetime.now()
        with transaction.atomic():
            member, created = m.LeagueMember.objects.get_or_create(
                league_id=int(league_id),
                user_id=int(uid),
                defaults={"role": str(role or "viewer"), "created_at": now},
            )
            if not created and str(getattr(member, "role", "") or "") != str(role or "viewer"):
                m.LeagueMember.objects.filter(id=int(member.id)).update(role=str(role or "viewer"))
        messages.success(request, "Member added/updated")
        return redirect(f"/leagues/{int(league_id)}/members")

    rows = list(
        m.LeagueMember.objects.filter(league_id=int(league_id))
        .select_related("user")
        .order_by("user__email")
        .values("user_id", "user__email", "role")
    )
    members = [
        {"id": int(r0["user_id"]), "email": r0["user__email"], "role": (r0.get("role") or "admin")}
        for r0 in rows
    ]
    return render(request, "league_members.html", {"league_id": int(league_id), "members": members})


def league_members_remove(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")
    uid = int(request.POST.get("user_id") or 0)
    _django_orm, m = _orm_modules()
    m.LeagueMember.objects.filter(league_id=int(league_id), user_id=int(uid)).delete()
    messages.success(request, "Member removed")
    return redirect(f"/leagues/{int(league_id)}/members")


def leagues_recalc_div_ratings(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    from django.contrib import messages

    r = _require_login(request)
    if r:
        return r
    league_id = request.session.get("league_id")
    if not league_id:
        messages.error(request, "Select an active league first.")
        return redirect("/leagues")
    if not _is_league_admin(int(league_id), _session_user_id(request)):
        messages.error(request, "Not authorized")
        return redirect("/leagues")
    try:
        logic.recompute_league_mhr_ratings(None, int(league_id))
        messages.success(request, "Ratings recalculated.")
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"Failed to recalculate Ratings: {e}")
    return redirect("/leagues")


def public_leagues_index(request: HttpRequest) -> HttpResponse:  # pragma: no cover
    _django_orm, m = _orm_modules()
    leagues = list(m.League.objects.filter(is_public=True).order_by("name").values("id", "name"))
    return render(request, "public_leagues.html", {"leagues": leagues})


def public_league_home(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    return redirect(f"/public/leagues/{int(league_id)}/teams")


def public_media_team_logo(
    request: HttpRequest, league_id: int, team_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    _django_orm, m = _orm_modules()
    row = (
        m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(team_id))
        .select_related("team")
        .values("team__logo_path")
        .first()
    )
    if not row or not row.get("team__logo_path"):
        raise Http404
    return _safe_file_response(Path(str(row["team__logo_path"])).resolve(), as_attachment=False)


def public_league_teams(request: HttpRequest, league_id: int) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    selected_age_raw = str(request.GET.get("age") or "").strip()
    selected_level_raw = str(request.GET.get("level") or "").strip()
    league_owner_user_id = (
        int(league.get("owner_user_id") or 0) if isinstance(league, dict) else None
    )

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
        entity_id=0,
        viewer_user_id=(viewer_user_id if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )
    is_league_owner = bool(
        viewer_user_id and league_owner_user_id and int(viewer_user_id) == int(league_owner_user_id)
    )

    _django_orm, m = _orm_modules()
    rows_raw = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id))
        .select_related("team")
        .values(
            "team_id",
            "team__user_id",
            "team__name",
            "team__logo_path",
            "team__is_external",
            "team__created_at",
            "team__updated_at",
            "division_name",
            "division_id",
            "conference_id",
            "mhr_rating",
            "mhr_agd",
            "mhr_sched",
            "mhr_games",
            "mhr_updated_at",
        )
    )
    rows: list[dict[str, Any]] = []
    for r0 in rows_raw:
        nm = str(r0.get("team__name") or "")
        if nm.strip() == logic.SEED_PLACEHOLDER_TEAM_NAME or logic.is_seed_placeholder_name(nm):
            continue
        rows.append(
            {
                "id": int(r0["team_id"]),
                "user_id": int(r0["team__user_id"]),
                "name": nm,
                "logo_path": r0.get("team__logo_path"),
                "is_external": bool(r0.get("team__is_external")),
                "created_at": r0.get("team__created_at"),
                "updated_at": r0.get("team__updated_at"),
                "division_name": r0.get("division_name"),
                "division_id": r0.get("division_id"),
                "conference_id": r0.get("conference_id"),
                "mhr_rating": r0.get("mhr_rating"),
                "mhr_agd": r0.get("mhr_agd"),
                "mhr_sched": r0.get("mhr_sched"),
                "mhr_games": r0.get("mhr_games"),
                "mhr_updated_at": r0.get("mhr_updated_at"),
            }
        )

    meta_by_div: dict[str, dict[str, Any]] = {}
    for t in rows:
        dn = _norm_division_name(t.get("division_name"))
        if dn not in meta_by_div:
            meta_by_div[dn] = {
                "age": logic.parse_age_from_division_name(dn),
                "level": logic.parse_level_from_division_name(dn),
            }

    ages = sorted({int(v["age"]) for v in meta_by_div.values() if v.get("age") is not None})
    age_options = [{"value": str(a), "label": _age_label(a)} for a in ages]

    selected_age_u: Optional[int] = None
    if selected_age_raw:
        m_age = re.search(r"(\d{1,2})", selected_age_raw)
        if m_age:
            try:
                candidate = int(m_age.group(1))
            except Exception:
                candidate = None
            if candidate is not None and candidate in set(ages):
                selected_age_u = candidate

    candidates = list(meta_by_div.values())
    if selected_age_u is not None:
        candidates = [
            v for v in candidates if v.get("age") is not None and int(v["age"]) == selected_age_u
        ]

    levels = {str(v["level"]).strip().upper() for v in candidates if v.get("level")}
    level_options = sorted(levels, key=_level_sort_key)
    if any(v.get("level") is None for v in candidates):
        level_options.append("Other")

    selected_age = str(selected_age_u) if selected_age_u is not None else ""
    selected_level = ""
    selected_level_norm = selected_level_raw.strip().upper()
    if selected_level_norm == "OTHER":
        selected_level_norm = "Other"
    if selected_level_norm in set(level_options):
        selected_level = selected_level_norm

    if selected_age_u is not None:
        rows = [
            t
            for t in rows
            if meta_by_div.get(_norm_division_name(t.get("division_name")), {}).get("age")
            == selected_age_u
        ]
    if selected_level:
        if selected_level == "Other":
            rows = [
                t
                for t in rows
                if meta_by_div.get(_norm_division_name(t.get("division_name")), {}).get("level")
                is None
            ]
        else:
            rows = [
                t
                for t in rows
                if str(
                    meta_by_div.get(_norm_division_name(t.get("division_name")), {}).get("level")
                    or ""
                )
                .strip()
                .upper()
                == selected_level
            ]

    stats: dict[int, dict[str, Any]] = {}
    for t in rows:
        tid = int(t["id"])
        try:
            stats[tid] = logic.compute_team_stats_league(None, tid, int(league_id))
            stats[tid]["gp"] = (
                int(stats[tid].get("wins", 0) or 0)
                + int(stats[tid].get("losses", 0) or 0)
                + int(stats[tid].get("ties", 0) or 0)
            )
        except Exception:
            stats[tid] = {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for t in rows:
        dn = _norm_division_name(t.get("division_name"))
        grouped.setdefault(dn, []).append(t)
    divisions = []
    for dn in sorted(grouped.keys(), key=logic.division_sort_key):
        ranked_ids = logic.division_standings_team_ids(int(league_id), str(dn))
        order = {int(tid): i for i, tid in enumerate(ranked_ids)}
        teams_sorted = sorted(
            grouped[dn],
            key=lambda tr: (
                order.get(int(tr["id"]), 10**9),
                str(tr.get("name") or "").casefold(),
            ),
        )
        divisions.append({"name": dn, "teams": teams_sorted})

    league_page_views = None
    if is_league_owner:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAMS, entity_id=0
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAMS,
            "entity_id": 0,
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }
    return render(
        request,
        "teams.html",
        {
            "teams": rows,
            "divisions": divisions,
            "age_options": age_options,
            "level_options": level_options,
            "selected_age": selected_age,
            "selected_level": selected_level,
            "stats": stats,
            "include_external": True,
            "league_view": True,
            "public_league_id": int(league_id),
            "league_page_views": league_page_views,
        },
    )


def public_league_team_detail(
    request: HttpRequest, league_id: int, team_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = None
    try:
        league_owner_user_id = (
            int(league.get("owner_user_id")) if isinstance(league, dict) else None
        )
    except Exception:
        league_owner_user_id = None
    is_league_owner = bool(
        viewer_user_id
        and league_owner_user_id is not None
        and int(viewer_user_id) == int(league_owner_user_id)
    )
    is_league_admin = False
    if viewer_user_id:
        try:
            is_league_admin = bool(_is_league_admin(int(league_id), int(viewer_user_id)))
        except Exception:
            is_league_admin = False
    can_view_league_page_views = bool(is_league_owner or is_league_admin)

    show_shift_data = False
    try:
        show_shift_data = _league_show_shift_data(int(league_id))
    except Exception:
        show_shift_data = False
    try:
        league_name = (
            str(league.get("name") or "").strip() if isinstance(league, dict) else None
        ) or logic._get_league_name(None, int(league_id))
    except Exception:
        league_name = None

    recent_n_raw = request.GET.get("recent_n")
    try:
        recent_n = max(1, min(10, int(str(recent_n_raw or "5"))))
    except Exception:
        recent_n = 5

    _django_orm, m = _orm_modules()
    team = (
        m.Team.objects.filter(id=int(team_id), league_teams__league_id=int(league_id))
        .values(
            "id",
            "user_id",
            "name",
            "logo_path",
            "is_external",
            "created_at",
            "updated_at",
        )
        .first()
    )
    if not team:
        raise Http404

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
        entity_id=int(team_id),
        viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )

    players = list(
        m.Player.objects.filter(team_id=int(team_id))
        .order_by("jersey_number", "name")
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    skaters, goalies, head_coaches, assistant_coaches = logic.split_roster(players or [])
    roster_players = list(skaters) + list(goalies)
    tstats = logic.compute_team_stats_league(None, int(team_id), int(league_id))

    from django.db.models import Q

    league_team_div_map = {
        int(tid): (str(dn).strip() if dn is not None else None)
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    schedule_rows_raw = list(
        m.LeagueGame.objects.filter(league_id=int(league_id))
        .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
        .select_related("game", "game__team1", "game__team2", "game__game_type")
        .values(
            "game_id",
            "division_name",
            "sort_order",
            "game__user_id",
            "game__team1_id",
            "game__team2_id",
            "game__game_type_id",
            "game__starts_at",
            "game__location",
            "game__notes",
            "game__team1_score",
            "game__team2_score",
            "game__is_final",
            "game__stats_imported_at",
            "game__created_at",
            "game__updated_at",
            "game__team1__name",
            "game__team2__name",
            "game__game_type__name",
        )
    )
    schedule_games: list[dict[str, Any]] = []
    for r0 in schedule_rows_raw:
        t1 = int(r0["game__team1_id"])
        t2 = int(r0["game__team2_id"])
        schedule_games.append(
            {
                "id": int(r0["game_id"]),
                "user_id": int(r0["game__user_id"]),
                "team1_id": t1,
                "team2_id": t2,
                "game_type_id": r0.get("game__game_type_id"),
                "starts_at": r0.get("game__starts_at"),
                "location": r0.get("game__location"),
                "notes": r0.get("game__notes"),
                "team1_score": r0.get("game__team1_score"),
                "team2_score": r0.get("game__team2_score"),
                "is_final": r0.get("game__is_final"),
                "stats_imported_at": r0.get("game__stats_imported_at"),
                "created_at": r0.get("game__created_at"),
                "updated_at": r0.get("game__updated_at"),
                "team1_name": r0.get("game__team1__name"),
                "team2_name": r0.get("game__team2__name"),
                "game_type_name": r0.get("game__game_type__name"),
                "division_name": r0.get("division_name"),
                "sort_order": r0.get("sort_order"),
                "team1_league_division_name": league_team_div_map.get(t1),
                "team2_league_division_name": league_team_div_map.get(t2),
            }
        )

    for g2 in schedule_games:
        has_score = (
            (g2.get("team1_score") is not None)
            or (g2.get("team2_score") is not None)
            or bool(g2.get("is_final"))
        )
        # Public pages should be stable and conservative: only show game detail when a score exists.
        g2["can_view_summary"] = bool(has_score)
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None
    schedule_games = logic.sort_games_schedule_order(schedule_games or [])
    _attach_schedule_stats_icons(schedule_games)

    ps_rows = _player_stat_rows_from_event_tables_for_team_games(
        team_id=int(team_id),
        schedule_games=list(schedule_games or []),
        roster_players=list(roster_players or []),
        show_shift_data=bool(show_shift_data),
    )

    for g2 in schedule_games or []:
        try:
            g2["_game_type_label"] = logic._game_type_label_for_row(g2)
        except Exception:
            g2["_game_type_label"] = "Unknown"
        try:
            reason = logic.game_exclusion_reason_for_stats(
                g2, team_id=int(team_id), league_name=league_name
            )
        except Exception:
            reason = None
        g2["excluded_from_stats_reason"] = reason
        g2["excluded_from_stats"] = bool(reason)
    # Tournament-only players: show them on game pages, but not on team/division-level roster/stats.
    try:
        tournament_game_ids: set[int] = {
            int(g2.get("id"))
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "").strip().casefold().startswith("tournament")
            and g2.get("id") is not None
        }
        player_ids_with_any_stats: set[int] = set()
        player_ids_with_non_tournament_stats: set[int] = set()
        for r0 in ps_rows or []:
            try:
                pid_i = int(r0.get("player_id"))
                gid_i = int(r0.get("game_id"))
            except Exception:
                continue
            player_ids_with_any_stats.add(pid_i)
            if gid_i not in tournament_game_ids:
                player_ids_with_non_tournament_stats.add(pid_i)
        tournament_only_player_ids = (
            player_ids_with_any_stats - player_ids_with_non_tournament_stats
        )
    except Exception:
        tournament_only_player_ids = set()

    team_is_external = bool(team.get("is_external"))
    if tournament_only_player_ids and not team_is_external:
        skaters = [p for p in skaters if int(p.get("id") or 0) not in tournament_only_player_ids]
        goalies = [p for p in goalies if int(p.get("id") or 0) not in tournament_only_player_ids]
        roster_players = list(skaters) + list(goalies)
        ps_rows = [
            r0 for r0 in ps_rows if int(r0.get("player_id") or 0) not in tournament_only_player_ids
        ]
    game_type_options = logic._dedupe_preserve_str(
        [str(g2.get("_game_type_label") or "") for g2 in (schedule_games or [])]
    )
    selected_types = logic._parse_selected_game_type_labels(
        available=game_type_options, args=request.GET
    )
    stats_schedule_games = (
        list(schedule_games or [])
        if selected_types is None
        else [
            g2
            for g2 in (schedule_games or [])
            if str(g2.get("_game_type_label") or "") in selected_types
        ]
    )
    eligible_games = [
        g2
        for g2 in stats_schedule_games
        if logic._game_has_recorded_result(g2)
        and logic.game_is_eligible_for_stats(
            g2,
            team_id=int(team_id),
            league_name=league_name,
        )
    ]
    eligible_game_ids_in_order: list[int] = []
    for g2 in eligible_games:
        try:
            eligible_game_ids_in_order.append(int(g2.get("id")))
        except Exception:
            continue
    eligible_game_ids_set: set[int] = set(eligible_game_ids_in_order)

    selected_game_ids_raw = list(request.GET.getlist("gid") or [])
    selected_game_ids_req: list[int] = []
    seen_gid: set[int] = set()
    for raw in selected_game_ids_raw:
        try:
            gid = int(str(raw).strip())
        except Exception:
            continue
        if gid <= 0 or gid in seen_gid:
            continue
        seen_gid.add(gid)
        selected_game_ids_req.append(int(gid))
    selected_game_ids_req_set = set(selected_game_ids_req)
    selected_game_ids_in_order = [
        int(gid) for gid in eligible_game_ids_in_order if int(gid) in selected_game_ids_req_set
    ]
    selected_game_ids_set = set(selected_game_ids_in_order)

    use_selected_games = (
        bool(selected_game_ids_set) and selected_game_ids_set != eligible_game_ids_set
    )
    stats_game_ids_in_order = (
        list(selected_game_ids_in_order) if use_selected_games else list(eligible_game_ids_in_order)
    )
    stats_game_ids_set = set(stats_game_ids_in_order)

    ps_rows_filtered: list[dict[str, Any]] = []
    for r0 in ps_rows or []:
        try:
            if int(r0.get("game_id")) in stats_game_ids_set:
                ps_rows_filtered.append(r0)
        except Exception:
            continue

    player_totals = logic._aggregate_player_totals_from_rows(
        player_stats_rows=ps_rows_filtered, allowed_game_ids=stats_game_ids_set
    )
    player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(skaters, player_totals)
    )
    for r0 in player_stats_rows:
        try:
            pid_i = int(r0.get("player_id") or 0)
        except Exception:
            continue
        r0["has_events"] = _player_totals_has_any_events(player_totals.get(pid_i) or {})
    player_stats_columns = logic.filter_player_stats_display_columns_for_rows(
        logic.PLAYER_STATS_DISPLAY_COLUMNS, player_stats_rows
    )
    cov_counts, cov_total = logic._compute_team_player_stats_coverage(
        player_stats_rows=ps_rows_filtered, eligible_game_ids=stats_game_ids_in_order
    )
    player_stats_columns = logic._player_stats_columns_with_coverage(
        columns=player_stats_columns, coverage_counts=cov_counts, total_games=cov_total
    )

    recent_scope_ids = (
        eligible_game_ids_in_order[-int(recent_n) :] if eligible_game_ids_in_order else []
    )
    has_pair_on_ice_all = False
    try:
        if stats_game_ids_in_order:
            has_pair_on_ice_all = bool(
                m.HkyGameShiftRow.objects.filter(
                    game_id__in=list(stats_game_ids_in_order), team_id=int(team_id)
                ).exists()
            )
    except Exception:
        has_pair_on_ice_all = False

    player_stats_sources = logic._compute_team_player_stats_sources(
        None, eligible_game_ids=stats_game_ids_in_order
    )
    selected_label = (
        "All"
        if selected_types is None
        else ", ".join(sorted(list(selected_types), key=lambda s: s.lower()))
    )
    game_type_filter_options = [
        {"label": gt, "checked": (selected_types is None) or (gt in selected_types)}
        for gt in game_type_options
    ]

    # "Select Games" UI: show eligible games (already filtered by Game Types) and highlight partial selection.
    select_games_options: list[dict[str, Any]] = []

    def _select_game_label(g2: dict[str, Any]) -> str:
        gid = g2.get("id")
        dt0 = logic.to_dt(g2.get("starts_at"))
        date_s = dt0.strftime("%Y-%m-%d") if dt0 else ""
        gt = str(g2.get("_game_type_label") or "").strip()
        opp = ""
        try:
            if int(g2.get("team1_id") or 0) == int(team_id):
                opp = str(g2.get("team2_name") or "").strip()
            else:
                opp = str(g2.get("team1_name") or "").strip()
        except Exception:
            opp = ""
        bits = []
        if date_s:
            bits.append(date_s)
        if opp:
            bits.append(f"vs {opp}")
        if gt:
            bits.append(f"({gt})")
        if not bits:
            return f"Game {gid}" if gid is not None else "Game"
        return " ".join(bits)

    for g2 in eligible_games or []:
        try:
            gid_i = int(g2.get("id") or 0)
        except Exception:
            continue
        if gid_i <= 0:
            continue
        select_games_options.append(
            {
                "id": int(gid_i),
                "label": _select_game_label(g2),
                "checked": (not use_selected_games) or (int(gid_i) in stats_game_ids_set),
            }
        )

    select_games_label = "All"
    if use_selected_games:
        select_games_label = f"{len(stats_game_ids_in_order)} of {len(eligible_game_ids_in_order)}"

    goalie_stats_rows: list[dict[str, Any]] = []
    goalie_stats_has_sog = False
    goalie_stats_has_xg = False
    recent_goalie_stats_rows: list[dict[str, Any]] = []
    recent_goalie_stats_has_sog = False
    recent_goalie_stats_has_xg = False
    show_goalie_stats = True
    if league_id:
        try:
            show_goalie_stats = _league_show_goalie_stats(int(league_id))
        except Exception:
            show_goalie_stats = False
    try:
        if show_goalie_stats and eligible_game_ids_in_order:
            goalie_event_rows = list(
                m.HkyGameEventRow.objects.filter(
                    game_id__in=list(eligible_game_ids_in_order),
                    event_type__key__in=[
                        "goal",
                        "expectedgoal",
                        "sog",
                        "shotongoal",
                        "goaliechange",
                    ],
                )
                .select_related("event_type")
                .values(
                    "game_id",
                    "event_type__key",
                    "event_id",
                    "team_side",
                    "period",
                    "game_seconds",
                    "player_id",
                    "attributed_players",
                    "attributed_jerseys",
                    "details",
                )
            )
            by_gid: dict[int, list[dict[str, Any]]] = {}
            for r0 in goalie_event_rows:
                try:
                    gid = int(r0.get("game_id") or 0)
                except Exception:
                    continue
                if gid <= 0:
                    continue
                by_gid.setdefault(gid, []).append(r0)
            out = logic.compute_goalie_stats_for_team_games(
                team_id=int(team_id),
                schedule_games=eligible_games,
                event_rows_by_game_id=by_gid,
                goalies=goalies,
            )
            goalie_stats_rows = list(out.get("rows") or [])
            goalie_stats_has_sog = bool((out.get("meta") or {}).get("has_sog"))
            goalie_stats_has_xg = bool((out.get("meta") or {}).get("has_xg"))

            recent_ids = {int(gid) for gid in recent_scope_ids}
            if recent_ids:
                recent_games = [g2 for g2 in eligible_games if int(g2.get("id") or 0) in recent_ids]
                out_recent = logic.compute_goalie_stats_for_team_games(
                    team_id=int(team_id),
                    schedule_games=recent_games,
                    event_rows_by_game_id=by_gid,
                    goalies=goalies,
                )
                recent_goalie_stats_rows = list(out_recent.get("rows") or [])
                recent_goalie_stats_has_sog = bool((out_recent.get("meta") or {}).get("has_sog"))
                recent_goalie_stats_has_xg = bool((out_recent.get("meta") or {}).get("has_xg"))
    except Exception:
        goalie_stats_rows = []
        goalie_stats_has_sog = False
        goalie_stats_has_xg = False
        recent_goalie_stats_rows = []
        recent_goalie_stats_has_sog = False
        recent_goalie_stats_has_xg = False

    league_page_views = None
    if can_view_league_page_views:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM, entity_id=int(team_id)
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_TEAM, entity_id=int(team_id)
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_TEAM,
            "entity_id": int(team_id),
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }
    excluded_games = [g2 for g2 in (schedule_games or []) if bool(g2.get("excluded_from_stats"))]
    return render(
        request,
        "team_detail.html",
        {
            "team": team,
            "roster_players": roster_players,
            "players": skaters,
            "head_coaches": head_coaches,
            "assistant_coaches": assistant_coaches,
            "player_stats_columns": player_stats_columns,
            "player_stats_rows": player_stats_rows,
            "select_games_options": select_games_options,
            "select_games_label": select_games_label,
            "select_games_partial": bool(use_selected_games),
            "recent_n": recent_n,
            "tstats": tstats,
            "schedule_games": schedule_games,
            "excluded_schedule_games_count": len(excluded_games),
            "show_caha_preseason_exclusion_note": str(league_name or "").strip().casefold()
            == "caha",
            "editable": False,
            "public_league_id": int(league_id),
            "is_league_admin": is_league_admin,
            "can_view_league_page_views": can_view_league_page_views,
            "player_stats_sources": player_stats_sources,
            "player_stats_coverage_total_games": cov_total,
            "game_type_filter_options": game_type_filter_options,
            "game_type_filter_label": selected_label,
            "goalie_stats_rows": goalie_stats_rows,
            "goalie_stats_has_sog": goalie_stats_has_sog,
            "goalie_stats_has_xg": goalie_stats_has_xg,
            "recent_goalie_stats_rows": recent_goalie_stats_rows,
            "recent_goalie_stats_has_sog": recent_goalie_stats_has_sog,
            "recent_goalie_stats_has_xg": recent_goalie_stats_has_xg,
            "has_pair_on_ice_all": has_pair_on_ice_all,
            "league_page_views": league_page_views,
        },
    )


def public_league_schedule(
    request: HttpRequest, league_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = None
    try:
        league_owner_user_id = (
            int(league.get("owner_user_id")) if isinstance(league, dict) else None
        )
    except Exception:
        league_owner_user_id = None
    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
        entity_id=0,
        viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )
    is_league_owner = bool(
        viewer_user_id
        and league_owner_user_id is not None
        and int(viewer_user_id) == int(league_owner_user_id)
    )

    selected_division = str(request.GET.get("division") or "").strip() or None
    selected_team_id = request.GET.get("team_id") or ""
    team_id_i: Optional[int] = None
    try:
        team_id_i = int(selected_team_id) if str(selected_team_id).strip() else None
    except Exception:
        team_id_i = None

    _django_orm, m = _orm_modules()
    divisions = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id))
        .exclude(division_name__isnull=True)
        .exclude(division_name="")
        .values_list("division_name", flat=True)
        .distinct()
    )
    divisions.sort(key=logic.division_sort_key)
    if selected_division:
        league_teams = list(
            m.Team.objects.filter(
                league_teams__league_id=int(league_id),
                league_teams__division_name=str(selected_division),
            )
            .distinct()
            .order_by("name")
            .values("id", "name")
        )
    else:
        league_teams = list(
            m.Team.objects.filter(league_teams__league_id=int(league_id))
            .distinct()
            .order_by("name")
            .values("id", "name")
        )
    league_teams = [
        t
        for t in league_teams
        if str(t.get("name") or "").strip() != logic.SEED_PLACEHOLDER_TEAM_NAME
        and not logic.is_seed_placeholder_name(str(t.get("name") or ""))
    ]
    if team_id_i is not None and not any(int(t["id"]) == int(team_id_i) for t in league_teams):
        team_id_i = None
        selected_team_id = ""

    lg_qs = m.LeagueGame.objects.filter(league_id=int(league_id))
    if selected_division:
        lg_qs = lg_qs.filter(division_name=str(selected_division))

    league_team_div_map = {
        int(tid): (str(dn).strip() if dn is not None else None)
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    rows_raw = list(
        lg_qs.select_related("game", "game__team1", "game__team2", "game__game_type").values(
            "game_id",
            "division_name",
            "sort_order",
            "game__user_id",
            "game__team1_id",
            "game__team2_id",
            "game__game_type_id",
            "game__starts_at",
            "game__location",
            "game__notes",
            "game__team1_score",
            "game__team2_score",
            "game__is_final",
            "game__stats_imported_at",
            "game__created_at",
            "game__updated_at",
            "game__team1__name",
            "game__team2__name",
            "game__game_type__name",
        )
    )
    games: list[dict[str, Any]] = []
    for r0 in rows_raw:
        t1 = int(r0["game__team1_id"])
        t2 = int(r0["game__team2_id"])
        games.append(
            {
                "id": int(r0["game_id"]),
                "user_id": int(r0["game__user_id"]),
                "team1_id": t1,
                "team2_id": t2,
                "game_type_id": r0.get("game__game_type_id"),
                "starts_at": r0.get("game__starts_at"),
                "location": r0.get("game__location"),
                "notes": r0.get("game__notes"),
                "team1_score": r0.get("game__team1_score"),
                "team2_score": r0.get("game__team2_score"),
                "is_final": r0.get("game__is_final"),
                "stats_imported_at": r0.get("game__stats_imported_at"),
                "created_at": r0.get("game__created_at"),
                "updated_at": r0.get("game__updated_at"),
                "team1_name": r0.get("game__team1__name"),
                "team2_name": r0.get("game__team2__name"),
                "game_type_name": r0.get("game__game_type__name"),
                "division_name": r0.get("division_name"),
                "sort_order": r0.get("sort_order"),
                "team1_league_division_name": league_team_div_map.get(t1),
                "team2_league_division_name": league_team_div_map.get(t2),
            }
        )
    _resolve_seed_placeholders_in_schedule_games(league_id=int(league_id), games=games)
    if team_id_i is not None:
        games = [
            g
            for g in games
            if int(g.get("team1_id") or 0) == int(team_id_i)
            or int(g.get("team2_id") or 0) == int(team_id_i)
        ]
    games = list(games or [])
    for g2 in games or []:
        try:
            g2["game_video_url"] = logic._sanitize_http_url(
                logic._extract_game_video_url_from_notes(g2.get("notes"))
            )
        except Exception:
            g2["game_video_url"] = None
        has_score = (
            (g2.get("team1_score") is not None)
            or (g2.get("team2_score") is not None)
            or bool(g2.get("is_final"))
        )
        # Public pages should be stable and conservative: only show game detail when a score exists.
        g2["can_view_summary"] = bool(has_score)
        g2["can_edit"] = False
    _attach_schedule_stats_icons(games)
    games = logic.sort_games_schedule_order(games or [])

    league_page_views = None
    if is_league_owner:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE, entity_id=0
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_SCHEDULE,
            "entity_id": 0,
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }

    return render(
        request,
        "schedule.html",
        {
            "games": games,
            "league_view": True,
            "divisions": divisions,
            "league_teams": league_teams,
            "selected_division": selected_division or "",
            "selected_team_id": str(team_id_i) if team_id_i is not None else "",
            "can_add_game": False,
            "public_league_id": int(league_id),
            "league_page_views": league_page_views,
        },
    )


def public_hky_game_detail(
    request: HttpRequest, league_id: int, game_id: int
) -> HttpResponse:  # pragma: no cover
    league = _is_public_league(int(league_id))
    if not league:
        raise Http404
    viewer_user_id = _session_user_id(request)
    league_owner_user_id = None
    try:
        league_owner_user_id = (
            int(league.get("owner_user_id")) if isinstance(league, dict) else None
        )
    except Exception:
        league_owner_user_id = None
    is_league_owner = bool(
        viewer_user_id
        and league_owner_user_id is not None
        and int(viewer_user_id) == int(league_owner_user_id)
    )
    is_league_admin = False
    if viewer_user_id:
        try:
            is_league_admin = bool(_is_league_admin(int(league_id), int(viewer_user_id)))
        except Exception:
            is_league_admin = False
    can_view_league_page_views = bool(is_league_owner or is_league_admin)

    show_shift_data = False
    try:
        show_shift_data = _league_show_shift_data(int(league_id))
    except Exception:
        show_shift_data = False

    _django_orm, m = _orm_modules()
    row = (
        m.LeagueGame.objects.filter(league_id=int(league_id), game_id=int(game_id))
        .select_related("game", "game__team1", "game__team2")
        .values(
            "game_id",
            "division_name",
            "game__user_id",
            "game__team1_id",
            "game__team2_id",
            "game__game_type_id",
            "game__starts_at",
            "game__location",
            "game__notes",
            "game__team1_score",
            "game__team2_score",
            "game__is_final",
            "game__stats_imported_at",
            "game__created_at",
            "game__updated_at",
            "game__team1__name",
            "game__team2__name",
            "game__team1__is_external",
            "game__team2__is_external",
        )
        .first()
    )
    if not row:
        raise Http404

    team1_id = int(row["game__team1_id"])
    team2_id = int(row["game__team2_id"])
    div_map = {
        int(tid): (str(dn).strip() if dn is not None else None)
        for tid, dn in m.LeagueTeam.objects.filter(
            league_id=int(league_id),
            team_id__in=[team1_id, team2_id],
        ).values_list("team_id", "division_name")
    }
    game = {
        "id": int(row["game_id"]),
        "user_id": int(row["game__user_id"]),
        "team1_id": team1_id,
        "team2_id": team2_id,
        "game_type_id": row.get("game__game_type_id"),
        "starts_at": row.get("game__starts_at"),
        "location": row.get("game__location"),
        "notes": row.get("game__notes"),
        "team1_score": row.get("game__team1_score"),
        "team2_score": row.get("game__team2_score"),
        "is_final": row.get("game__is_final"),
        "stats_imported_at": row.get("game__stats_imported_at"),
        "created_at": row.get("game__created_at"),
        "updated_at": row.get("game__updated_at"),
        "team1_name": row.get("game__team1__name"),
        "team2_name": row.get("game__team2__name"),
        "team1_ext": row.get("game__team1__is_external"),
        "team2_ext": row.get("game__team2__is_external"),
        "division_name": row.get("division_name"),
        "team1_league_division_name": div_map.get(team1_id),
        "team2_league_division_name": div_map.get(team2_id),
    }
    try:
        game["game_video_url"] = logic._sanitize_http_url(
            logic._extract_game_video_url_from_notes(game.get("notes"))
        )
    except Exception:
        game["game_video_url"] = None
    has_score = (
        (game.get("team1_score") is not None)
        or (game.get("team2_score") is not None)
        or bool(game.get("is_final"))
    )
    # Public pages should be stable and conservative: only show game detail when a score exists.
    can_view_summary = bool(has_score)
    if not can_view_summary:
        raise Http404

    logic._record_league_page_view(
        None,
        int(league_id),
        kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME,
        entity_id=int(game_id),
        viewer_user_id=(int(viewer_user_id) if viewer_user_id else None),
        league_owner_user_id=league_owner_user_id,
    )

    team1_players = list(
        m.Player.objects.filter(team_id=int(game["team1_id"]))
        .order_by("jersey_number", "name")
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    team2_players = list(
        m.Player.objects.filter(team_id=int(game["team2_id"]))
        .order_by("jersey_number", "name")
        .values(
            "id",
            "user_id",
            "team_id",
            "name",
            "jersey_number",
            "position",
            "shoots",
            "created_at",
            "updated_at",
        )
    )
    stats_rows: list[dict[str, Any]] = []
    player_stats_imported_by_pid: dict[int, dict[str, Optional[int]]] = {}
    try:
        stats_rows.extend(
            _player_stat_rows_from_event_tables_for_team_games(
                team_id=int(game["team1_id"]),
                schedule_games=[dict(game)],
                roster_players=list(team1_players or []),
                show_shift_data=bool(show_shift_data),
            )
        )
        stats_rows.extend(
            _player_stat_rows_from_event_tables_for_team_games(
                team_id=int(game["team2_id"]),
                schedule_games=[dict(game)],
                roster_players=list(team2_players or []),
                show_shift_data=bool(show_shift_data),
            )
        )
    except Exception:
        logger.exception("Failed to compute event-derived player stat rows for game_id=%s", game_id)
        stats_rows = []
    if not show_shift_data:
        _mask_shift_stats_in_player_stat_rows(stats_rows)
    team1_skaters, team1_goalies, team1_hc, team1_ac = logic.split_roster(team1_players)
    team2_skaters, team2_goalies, team2_hc, team2_ac = logic.split_roster(team2_players)
    team1_roster = list(team1_skaters) + list(team1_goalies) + list(team1_hc) + list(team1_ac)
    team2_roster = list(team2_skaters) + list(team2_goalies) + list(team2_hc) + list(team2_ac)
    stats_by_pid = {r0["player_id"]: r0 for r0 in stats_rows}
    if show_shift_data:
        try:
            _overlay_on_ice_goal_stats_from_shift_rows(
                game_id=int(game_id), stats_by_pid=stats_by_pid
            )
        except Exception:
            # Best-effort: shift-derived overlays are optional and should not break page rendering.
            pass
    tts_linked = logic._extract_timetoscore_game_id_from_notes(game.get("notes")) is not None

    events_headers, events_rows, events_meta = _load_game_events_for_display(game_id=int(game_id))
    try:
        events_headers, events_rows = logic.normalize_game_events_csv(events_headers, events_rows)
    except Exception:
        # Best-effort: keep UI working even if legacy CSV normalization fails.
        pass
    if tts_linked:
        try:
            events_headers, events_rows = logic.enrich_timetoscore_goals_with_long_video_times(
                existing_headers=events_headers,
                existing_rows=events_rows,
                incoming_headers=events_headers,
                incoming_rows=events_rows,
            )
            events_headers, events_rows = logic.enrich_timetoscore_penalties_with_video_times(
                existing_headers=events_headers,
                existing_rows=events_rows,
                incoming_headers=events_headers,
                incoming_rows=events_rows,
            )
        except Exception:
            # Best-effort: keep UI working even if enrichment fails.
            pass
    events_rows = logic.filter_events_rows_prefer_timetoscore_for_goal_assist(
        events_rows, tts_linked=tts_linked
    )
    try:
        events_headers, events_rows = logic.normalize_events_video_time_for_display(
            events_headers, events_rows
        )
        events_headers, events_rows = logic.filter_events_headers_drop_empty_on_ice_split(
            events_headers, events_rows
        )
        events_rows = logic.sort_events_rows_default(events_rows)
    except Exception:
        # Best-effort: keep UI working even if video-time normalization fails.
        pass
    if events_meta is not None:
        try:
            events_meta["count"] = len(events_rows)
            events_meta["sources"] = logic.summarize_event_sources(
                events_rows, fallback_source_label=str(events_meta.get("source_label") or "")
            )
        except Exception:
            # Best-effort: event source summary is optional and should not break page rendering.
            pass

    scoring_by_period_rows = logic.compute_team_scoring_by_period_from_events(
        events_rows, tts_linked=tts_linked
    )
    game_event_stats_rows = logic.compute_game_event_stats_by_side(events_rows)

    goalie_stats = {"home": [], "away": [], "meta": {"has_sog": False}}
    show_goalie_stats = True
    if league_id:
        try:
            show_goalie_stats = _league_show_goalie_stats(int(league_id))
        except Exception:
            show_goalie_stats = False
    if show_goalie_stats:
        try:
            goalie_event_rows = list(
                m.HkyGameEventRow.objects.filter(
                    game_id=int(game_id),
                    event_type__key__in=[
                        "goal",
                        "expectedgoal",
                        "sog",
                        "shotongoal",
                        "goaliechange",
                    ],
                )
                .select_related("event_type")
                .values(
                    "event_type__key",
                    "event_id",
                    "team_side",
                    "period",
                    "game_seconds",
                    "player_id",
                    "attributed_players",
                    "attributed_jerseys",
                    "details",
                )
            )
            goalie_stats = logic.compute_goalie_stats_for_game(
                goalie_event_rows,
                home_goalies=team1_goalies,
                away_goalies=team2_goalies,
            )
        except Exception:
            goalie_stats = {"home": [], "away": [], "meta": {"has_sog": False}}

    player_stats_import_meta: Optional[dict[str, Any]] = None

    try:
        player_has_events_by_pid = _compute_player_has_events_by_pid_for_game(
            events_rows=events_rows,
            team1_players=list(team1_skaters),
            team2_players=list(team2_skaters),
        )
    except Exception:
        player_has_events_by_pid = {}

    display_by_pid: dict[int, dict[str, Any]] = {}
    for p in list(team1_skaters) + list(team2_skaters):
        try:
            pid = int(p.get("id") or 0)
        except Exception:
            continue
        if pid <= 0:
            continue
        db_row = stats_by_pid.get(pid) or {}
        base: dict[str, Any] = {"player_id": int(pid), "gp": 1}
        for k in logic.PLAYER_STATS_SUM_KEYS:
            base[str(k)] = logic._int0(db_row.get(str(k)))
        display_by_pid[int(pid)] = logic.compute_player_display_stats(base)

    team1_player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(list(team1_skaters), display_by_pid)
    )
    team2_player_stats_rows = logic.sort_players_table_default(
        logic.build_player_stats_table_rows(list(team2_skaters), display_by_pid)
    )
    for r0 in list(team1_player_stats_rows) + list(team2_player_stats_rows):
        try:
            pid_i = int(r0.get("player_id") or 0)
        except Exception:
            continue
        r0["has_events"] = bool(player_has_events_by_pid.get(pid_i))

    game_player_stats_columns = logic.build_game_player_stats_display_columns(
        rows=list(team1_player_stats_rows) + list(team2_player_stats_rows)
    )
    player_stats_cell_conflicts_by_pid: dict[int, dict[str, str]] = {}
    if show_shift_data and player_stats_imported_by_pid:
        for pid, imported in (player_stats_imported_by_pid or {}).items():
            try:
                pid_i = int(pid)
            except Exception:
                continue
            cur = stats_by_pid.get(int(pid_i)) or {}

            def _cmp_int(key: str) -> tuple[Optional[int], Optional[int]]:
                try:
                    a0 = imported.get(key)
                    a = int(a0) if a0 is not None else None
                except Exception:
                    a = None
                try:
                    b0 = cur.get(key)
                    b = int(b0) if b0 is not None else None
                except Exception:
                    b = None
                return a, b

            for k in ("toi_seconds", "shifts", "gf_counted", "ga_counted", "plus_minus"):
                a, b = _cmp_int(k)
                if a is None or b is None or int(a) == int(b):
                    continue

                msg = f"Imported {k}={a}, computed from shifts={b}"
                if k == "plus_minus":
                    gf = logic._int0(cur.get("gf_counted"))
                    ga = logic._int0(cur.get("ga_counted"))
                    msg = (
                        f"Imported +/-={a:+d}, computed from shifts={b:+d} "
                        f"(GF={int(gf)}, GA={int(ga)}; goals at shift start excluded)"
                    )
                elif k == "gf_counted":
                    msg = f"Imported GF counted={a}, computed from shifts={b}"
                elif k == "ga_counted":
                    msg = f"Imported GA counted={a}, computed from shifts={b}"
                elif k == "toi_seconds":
                    msg = f"Imported TOI={a}s, computed from shifts={b}s"
                elif k == "shifts":
                    msg = f"Imported shifts={a}, computed from shift rows={b}"

                player_stats_cell_conflicts_by_pid.setdefault(int(pid_i), {})[str(k)] = str(msg)
    player_stats_import_warning: Optional[str] = None

    default_back_url = f"/public/leagues/{int(league_id)}/schedule"
    return_to = logic._safe_return_to_url(request.GET.get("return_to"), default=default_back_url)
    public_is_logged_in = bool(viewer_user_id)

    league_page_views = None
    if can_view_league_page_views:
        count = logic._get_league_page_view_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME, entity_id=int(game_id)
        )
        baseline_count = logic._get_league_page_view_baseline_count(
            None, int(league_id), kind=logic.LEAGUE_PAGE_VIEW_KIND_GAME, entity_id=int(game_id)
        )
        delta_count = (
            max(0, int(count) - int(baseline_count)) if baseline_count is not None else None
        )
        league_page_views = {
            "league_id": int(league_id),
            "kind": logic.LEAGUE_PAGE_VIEW_KIND_GAME,
            "entity_id": int(game_id),
            "count": int(count),
            "baseline_count": int(baseline_count) if baseline_count is not None else None,
            "delta_count": int(delta_count) if delta_count is not None else None,
        }

    shift_timeline_rows: list[dict[str, str]] = []
    if show_shift_data:
        try:
            shift_timeline_rows = _load_game_shift_rows_for_timeline(game_id=int(game_id))
        except Exception:
            shift_timeline_rows = []

    return render(
        request,
        "hky_game_detail.html",
        {
            "game": game,
            "team1_roster": team1_roster,
            "team2_roster": team2_roster,
            "team1_player_stats_rows": team1_player_stats_rows,
            "team2_player_stats_rows": team2_player_stats_rows,
            "editable": False,
            "can_edit": False,
            "edit_mode": False,
            "public_league_id": int(league_id),
            "back_url": return_to,
            "return_to": return_to,
            "events_headers": events_headers,
            "events_rows": events_rows,
            "events_meta": events_meta,
            "shift_timeline_rows": shift_timeline_rows,
            "show_shift_data": bool(show_shift_data),
            "scoring_by_period_rows": scoring_by_period_rows,
            "game_event_stats_rows": game_event_stats_rows,
            "user_video_clip_len_s": (
                logic.get_user_video_clip_len_s(None, int(viewer_user_id))
                if public_is_logged_in
                else None
            ),
            "user_is_logged_in": public_is_logged_in,
            "game_player_stats_columns": game_player_stats_columns,
            "player_stats_cell_conflicts_by_pid": player_stats_cell_conflicts_by_pid,
            "player_stats_import_meta": player_stats_import_meta,
            "player_stats_import_warning": player_stats_import_warning,
            "goalie_stats_home_rows": goalie_stats.get("home") or [],
            "goalie_stats_away_rows": goalie_stats.get("away") or [],
            "goalie_stats_has_sog": bool((goalie_stats.get("meta") or {}).get("has_sog")),
            "goalie_stats_has_xg": bool((goalie_stats.get("meta") or {}).get("has_xg")),
            "is_league_admin": is_league_admin,
            "can_view_league_page_views": can_view_league_page_views,
            "league_page_views": league_page_views,
        },
    )


# ----------------------------
# Import/auth helpers (API)
# ----------------------------


def _get_import_token() -> Optional[str]:
    token = os.environ.get("HM_WEBAPP_IMPORT_TOKEN")
    if token:
        return str(token)
    try:
        cfg_path = os.environ.get("HM_DB_CONFIG", str(logic.CONFIG_PATH))
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        t = cfg.get("import_token")
        return str(t) if t else None
    except Exception:
        return None


def _require_import_auth(request: HttpRequest) -> Optional[JsonResponse]:
    required = _get_import_token()
    if required:
        supplied = None
        auth = str(request.META.get("HTTP_AUTHORIZATION") or "").strip()
        if auth.lower().startswith("bearer "):
            supplied = auth.split(" ", 1)[1].strip()
        if not supplied:
            supplied = str(
                request.META.get("HTTP_X_HM_IMPORT_TOKEN") or request.GET.get("token") or ""
            ).strip()
        required_s = str(required or "").strip()
        if not supplied or not secrets.compare_digest(str(supplied), required_s):
            return JsonResponse({"ok": False, "error": "unauthorized"}, status=401)
        return None

    if request.META.get("HTTP_X_FORWARDED_FOR"):
        return JsonResponse({"ok": False, "error": "import_token_required"}, status=403)
    if str(request.META.get("REMOTE_ADDR") or "") not in ("127.0.0.1", "::1"):
        return JsonResponse({"ok": False, "error": "import_token_required"}, status=403)
    return None


def _ensure_user_for_import(email: str, name: Optional[str] = None) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm:
        raise ValueError("owner_email is required")
    _django_orm, m = _orm_modules()
    existing = m.User.objects.filter(email=email_norm).values_list("id", flat=True).first()
    if existing is not None:
        return int(existing)
    pwd = generate_password_hash(secrets.token_hex(24))
    u = m.User.objects.create(
        email=email_norm,
        password_hash=pwd,
        name=(name or email_norm),
        created_at=dt.datetime.now(),
    )
    return int(u.id)


def _ensure_league_for_import(
    *,
    league_name: str,
    owner_user_id: int,
    is_shared: Optional[bool],
    source: Optional[str],
    external_key: Optional[str],
    commit: bool = True,
) -> int:
    del commit
    name = (league_name or "").strip()
    if not name:
        raise ValueError("league_name is required")
    _django_orm, m = _orm_modules()
    existing = (
        m.League.objects.filter(name=name)
        .values("id", "is_shared", "source", "external_key")
        .first()
    )
    now = dt.datetime.now()
    if existing:
        updates: dict[str, Any] = {}
        if is_shared is not None and bool(existing.get("is_shared")) != bool(is_shared):
            updates["is_shared"] = bool(is_shared)
        if source is not None and str(existing.get("source") or "") != str(source or ""):
            updates["source"] = source
        if external_key is not None and str(existing.get("external_key") or "") != str(
            external_key or ""
        ):
            updates["external_key"] = external_key
        if updates:
            updates["updated_at"] = now
            m.League.objects.filter(id=int(existing["id"])).update(**updates)
        return int(existing["id"])

    if is_shared is None:
        is_shared = True
    league = m.League.objects.create(
        name=name,
        owner_user_id=int(owner_user_id),
        is_shared=bool(is_shared),
        source=source,
        external_key=external_key,
        created_at=now,
        updated_at=None,
    )
    return int(league.id)


def _ensure_league_member_for_import(
    league_id: int, user_id: int, role: str, *, commit: bool = True
) -> None:
    del commit
    _django_orm, m = _orm_modules()
    m.LeagueMember.objects.get_or_create(
        league_id=int(league_id),
        user_id=int(user_id),
        defaults={"role": str(role or "viewer"), "created_at": dt.datetime.now()},
    )


def _normalize_import_game_type_name(raw: Any) -> Optional[str]:
    s = str(raw or "").strip()
    if not s:
        return None
    sl = s.casefold()
    if sl.startswith("regular"):
        return "Regular Season"
    if sl.startswith("preseason"):
        return "Preseason"
    if sl.startswith("exhibition"):
        return "Exhibition"
    if sl.startswith("tournament"):
        return "Tournament"
    return s


def _ensure_game_type_id_for_import(game_type_name: Any) -> Optional[int]:
    nm = _normalize_import_game_type_name(game_type_name)
    if not nm:
        return None
    _django_orm, m = _orm_modules()
    existing = m.GameType.objects.filter(name=str(nm)).values_list("id", flat=True).first()
    if existing is not None:
        return int(existing)
    gt = m.GameType.objects.create(name=str(nm), is_default=False)
    return int(gt.id)


def _ensure_external_team_for_import(owner_user_id: int, name: str, *, commit: bool = True) -> int:
    del commit

    ph = logic.parse_seed_placeholder_name(str(name or ""))
    if ph is not None:
        return logic.ensure_seed_placeholder_team_for_import(int(owner_user_id), commit=False)

    def _norm_team_name(s: str) -> str:
        t = str(s or "").replace("\xa0", " ").strip()
        t = (
            t.replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2212", "-")
        )
        t = " ".join(t.split())
        t = re.sub(r"\s*\(\s*external\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
        t = t.casefold()
        t = re.sub(r"[^0-9a-z]+", " ", t)
        return " ".join(t.split())

    nm = _norm_team_name(name or "")
    if not nm:
        nm = "unknown"
    _django_orm, m = _orm_modules()
    raw_name = str(name or "").strip()
    exact = (
        m.Team.objects.filter(user_id=int(owner_user_id), name=raw_name)
        .values_list("id", flat=True)
        .first()
    )
    if exact is not None:
        return int(exact)
    for row in m.Team.objects.filter(user_id=int(owner_user_id)).values("id", "name"):
        if _norm_team_name(str(row.get("name") or "")) == nm:
            return int(row["id"])
    t = m.Team.objects.create(
        user_id=int(owner_user_id),
        name=raw_name or "UNKNOWN",
        is_external=True,
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(t.id)


def _ensure_player_for_import(
    owner_user_id: int,
    team_id: int,
    name: str,
    jersey_number: Optional[str],
    position: Optional[str],
    *,
    commit: bool = True,
) -> int:
    del commit
    raw_name = str(name or "").strip()
    nm = _strip_jersey_from_player_name(raw_name, jersey_number)
    if not nm:
        raise ValueError("player name is required")
    _django_orm, m = _orm_modules()
    existing = (
        m.Player.objects.filter(user_id=int(owner_user_id), team_id=int(team_id), name=nm)
        .values_list("id", flat=True)
        .first()
    )
    if existing is None and raw_name and raw_name != nm:
        existing_raw = (
            m.Player.objects.filter(user_id=int(owner_user_id), team_id=int(team_id), name=raw_name)
            .values_list("id", flat=True)
            .first()
        )
        if existing_raw is not None:
            m.Player.objects.filter(id=int(existing_raw)).update(
                name=nm, updated_at=dt.datetime.now()
            )
            existing = int(existing_raw)
    if existing is not None:
        pid = int(existing)
        if jersey_number or position:
            updates: dict[str, Any] = {"updated_at": dt.datetime.now()}
            if jersey_number:
                updates["jersey_number"] = jersey_number
            if position:
                updates["position"] = position
            m.Player.objects.filter(id=pid).update(**updates)
        return pid

    p = m.Player.objects.create(
        user_id=int(owner_user_id),
        team_id=int(team_id),
        name=nm,
        jersey_number=jersey_number,
        position=position,
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(p.id)


def _merge_notes(existing: Optional[str], new_fields: dict[str, Any]) -> str:
    if not existing:
        return json.dumps(new_fields, sort_keys=True)
    try:
        cur = json.loads(existing)
        if isinstance(cur, dict):
            cur.update(new_fields)
            return json.dumps(cur, sort_keys=True)
    except Exception:
        pass
    return str(existing)


def _update_game_video_url_note(
    game_id: int, video_url: str, *, replace: bool, commit: bool = True
) -> None:
    del commit
    url = logic._sanitize_http_url(video_url)
    if not url:
        return
    _django_orm, m = _orm_modules()
    existing = str(
        m.HkyGame.objects.filter(id=int(game_id)).values_list("notes", flat=True).first() or ""
    ).strip()
    existing_url = logic._extract_game_video_url_from_notes(existing)
    if existing_url and not replace:
        return

    new_notes: str
    try:
        d = json.loads(existing) if existing else {}
        if isinstance(d, dict):
            d["game_video_url"] = url
            new_notes = json.dumps(d, sort_keys=True)
        else:
            raise ValueError("notes not dict")
    except Exception:
        suffix = f" game_video_url={url}"
        if existing and suffix.strip() in existing:
            new_notes = existing
        else:
            new_notes = (existing + "\n" + suffix.strip()).strip() if existing else suffix.strip()

    m.HkyGame.objects.filter(id=int(game_id)).update(notes=new_notes, updated_at=dt.datetime.now())


def _update_game_stats_note(
    game_id: int, stats_note: str, *, replace: bool, commit: bool = True
) -> None:
    del commit
    del replace
    note = str(stats_note or "").strip()
    if not note:
        return
    note = " ".join(note.split())
    if not note:
        return
    _django_orm, m = _orm_modules()
    existing = str(
        m.HkyGame.objects.filter(id=int(game_id)).values_list("notes", flat=True).first() or ""
    ).strip()
    existing_note = None
    try:
        existing_note = logic._extract_game_stats_note_from_notes(existing)
    except Exception:
        existing_note = None
    if existing_note is not None and str(existing_note).strip() == note:
        return

    new_notes: str
    try:
        d = json.loads(existing) if existing else {}
        if isinstance(d, dict):
            d["stats_note"] = note
            new_notes = json.dumps(d, sort_keys=True)
        else:
            raise ValueError("notes not dict")
    except Exception:
        lines = [ln for ln in str(existing or "").splitlines() if ln is not None]
        replaced = False
        for idx, ln in enumerate(lines):
            if re.match(r"^\\s*(?:stats_note|schedule_note)\\s*[:=]", str(ln), flags=re.IGNORECASE):
                lines[idx] = f"stats_note: {note}"
                replaced = True
                break
        if not replaced:
            lines.append(f"stats_note: {note}")
        new_notes = "\n".join([ln for ln in lines if str(ln).strip()]).strip()

    m.HkyGame.objects.filter(id=int(game_id)).update(notes=new_notes, updated_at=dt.datetime.now())


def _upsert_game_for_import(
    *,
    owner_user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[str],
    location: Optional[str],
    team1_score: Optional[int],
    team2_score: Optional[int],
    replace: bool,
    notes_json_fields: dict[str, Any],
    commit: bool = True,
) -> int:
    del commit
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    starts_dt = logic.to_dt(starts_at) if starts_at else None

    tts_int: Optional[int]
    try:
        tts_int = (
            int(notes_json_fields["timetoscore_game_id"])
            if notes_json_fields.get("timetoscore_game_id") is not None
            else None
        )
    except Exception:
        tts_int = None
    ext_key = str(notes_json_fields.get("external_game_key") or "").strip() or None

    existing_by_tts = None
    if tts_int is not None:
        existing_by_tts = (
            m.HkyGame.objects.filter(timetoscore_game_id=int(tts_int))
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        if existing_by_tts is None:
            token_json_nospace = f'"timetoscore_game_id":{int(tts_int)}'
            token_json_space = f'"timetoscore_game_id": {int(tts_int)}'
            token_plain = f"game_id={int(tts_int)}"
            existing_by_tts = (
                m.HkyGame.objects.filter(
                    Q(notes__contains=token_json_nospace)
                    | Q(notes__contains=token_json_space)
                    | Q(notes__contains=token_plain)
                )
                .values(
                    "id",
                    "notes",
                    "team1_score",
                    "team2_score",
                    "timetoscore_game_id",
                    "external_game_key",
                )
                .first()
            )

    existing_by_ext = None
    if ext_key:
        existing_by_ext = (
            m.HkyGame.objects.filter(user_id=int(owner_user_id), external_game_key=str(ext_key))
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        if existing_by_ext is None:
            try:
                ext_json = json.dumps(str(ext_key))
            except Exception:
                ext_json = f'"{str(ext_key)}"'
            token1 = f'"external_game_key":{ext_json}'
            token2 = f'"external_game_key": {ext_json}'
            existing_by_ext = (
                m.HkyGame.objects.filter(user_id=int(owner_user_id))
                .filter(Q(notes__contains=token1) | Q(notes__contains=token2))
                .values(
                    "id",
                    "notes",
                    "team1_score",
                    "team2_score",
                    "timetoscore_game_id",
                    "external_game_key",
                )
                .first()
            )

    if (
        existing_by_tts
        and existing_by_ext
        and int(existing_by_tts["id"]) != int(existing_by_ext["id"])
    ):
        _django_orm.merge_hky_games(
            keep_id=int(existing_by_tts["id"]), drop_id=int(existing_by_ext["id"])
        )
        existing_by_tts = (
            m.HkyGame.objects.filter(id=int(existing_by_tts["id"]))
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        existing_by_ext = None

    existing_by_time = None
    if starts_dt is not None:
        existing_by_time = (
            m.HkyGame.objects.filter(
                user_id=int(owner_user_id),
                team1_id=int(team1_id),
                team2_id=int(team2_id),
                starts_at=starts_dt,
            )
            .values(
                "id",
                "notes",
                "team1_score",
                "team2_score",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )

    existing_row = existing_by_tts or existing_by_ext or existing_by_time

    now = dt.datetime.now()
    if existing_row is None:
        notes = json.dumps(notes_json_fields, sort_keys=True)
        g = m.HkyGame.objects.create(
            user_id=int(owner_user_id),
            team1_id=int(team1_id),
            team2_id=int(team2_id),
            game_type_id=int(game_type_id) if game_type_id is not None else None,
            starts_at=starts_dt,
            location=location,
            team1_score=team1_score,
            team2_score=team2_score,
            is_final=bool(team1_score is not None and team2_score is not None),
            notes=notes,
            stats_imported_at=now,
            timetoscore_game_id=tts_int,
            external_game_key=ext_key,
            created_at=now,
            updated_at=None,
        )
        return int(g.id)

    gid = int(existing_row["id"])
    merged_notes = _merge_notes(existing_row.get("notes"), notes_json_fields)

    existing_t1 = existing_row.get("team1_score")
    existing_t2 = existing_row.get("team2_score")

    updates: dict[str, Any] = {
        "notes": merged_notes,
        "stats_imported_at": now,
        "updated_at": now,
    }
    if tts_int is not None and existing_row.get("timetoscore_game_id") is None:
        updates["timetoscore_game_id"] = int(tts_int)
    if ext_key and not existing_row.get("external_game_key"):
        updates["external_game_key"] = str(ext_key)
    if game_type_id is not None:
        updates["game_type_id"] = int(game_type_id)
    if location is not None:
        updates["location"] = location

    if replace:
        updates["team1_score"] = team1_score
        updates["team2_score"] = team2_score
        if team1_score is not None and team2_score is not None:
            updates["is_final"] = True
    else:
        if existing_t1 is None and team1_score is not None:
            updates["team1_score"] = team1_score
        if existing_t2 is None and team2_score is not None:
            updates["team2_score"] = team2_score
        if (
            existing_t1 is None
            and existing_t2 is None
            and team1_score is not None
            and team2_score is not None
        ):
            updates["is_final"] = True

    m.HkyGame.objects.filter(id=gid).update(**updates)
    return gid


def _map_game_to_league_for_import(
    league_id: int,
    game_id: int,
    *,
    division_name: Optional[str] = None,
    division_id: Optional[int] = None,
    conference_id: Optional[int] = None,
    sort_order: Optional[int] = None,
    commit: bool = True,
) -> None:
    del commit
    dn = (division_name or "").strip() or None
    _django_orm, m = _orm_modules()
    obj, created = m.LeagueGame.objects.get_or_create(
        league_id=int(league_id),
        game_id=int(game_id),
        defaults={
            "division_name": dn,
            "division_id": division_id,
            "conference_id": conference_id,
            "sort_order": sort_order,
        },
    )
    if created:
        return

    updates: dict[str, Any] = {}
    allow_div_update = True
    if dn and logic.is_external_division_name(dn):
        existing_dn = str(getattr(obj, "division_name", "") or "").strip()
        if existing_dn and not logic.is_external_division_name(existing_dn):
            allow_div_update = False

    if dn and dn.strip() and dn.strip().lower() != "external" and allow_div_update:
        updates["division_name"] = dn
    if allow_div_update:
        if division_id is not None:
            updates["division_id"] = division_id
        if conference_id is not None:
            updates["conference_id"] = conference_id
    if sort_order is not None:
        updates["sort_order"] = sort_order
    if updates:
        m.LeagueGame.objects.filter(id=int(obj.id)).update(**updates)


def _map_team_to_league_for_import(
    league_id: int,
    team_id: int,
    *,
    division_name: Optional[str] = None,
    division_id: Optional[int] = None,
    conference_id: Optional[int] = None,
    commit: bool = True,
) -> None:
    del commit
    dn = (division_name or "").strip() or None
    _django_orm, m = _orm_modules()
    obj, created = m.LeagueTeam.objects.get_or_create(
        league_id=int(league_id),
        team_id=int(team_id),
        defaults={"division_name": dn, "division_id": division_id, "conference_id": conference_id},
    )
    if created:
        return
    updates: dict[str, Any] = {}
    allow_div_update = True
    if dn and logic.is_external_division_name(dn):
        existing_dn = str(getattr(obj, "division_name", "") or "").strip()
        if existing_dn and not logic.is_external_division_name(existing_dn):
            allow_div_update = False

    if dn and dn.strip() and dn.strip().lower() != "external" and allow_div_update:
        updates["division_name"] = dn
    if allow_div_update:
        if division_id is not None:
            updates["division_id"] = division_id
        if conference_id is not None:
            updates["conference_id"] = conference_id
    if updates:
        m.LeagueTeam.objects.filter(id=int(obj.id)).update(**updates)


def _ensure_team_logo_from_url_for_import(
    *, team_id: int, logo_url: Optional[str], replace: bool, commit: bool = True
) -> None:
    del commit
    url = str(logo_url or "").strip()
    if not url:
        return
    _django_orm, m = _orm_modules()
    existing = m.Team.objects.filter(id=int(team_id)).values_list("logo_path", flat=True).first()
    if existing and not replace:
        return
    try:
        import requests  # type: ignore
    except Exception:
        return

    headers: dict[str, str] = {"User-Agent": "Mozilla/5.0"}
    try:
        from urllib.parse import urlparse

        u = urlparse(url)
        if u.scheme and u.netloc:
            headers["Referer"] = f"{u.scheme}://{u.netloc}/"
    except Exception:
        pass

    try:
        resp = requests.get(url, timeout=(10, 30), headers=headers)  # type: ignore[attr-defined]
        resp.raise_for_status()
        data = resp.content
        ctype = str(resp.headers.get("Content-Type") or "")
    except Exception:
        return

    ext = None
    ctype_l = ctype.lower()
    if "png" in ctype_l:
        ext = ".png"
    elif "jpeg" in ctype_l or "jpg" in ctype_l:
        ext = ".jpg"
    elif "gif" in ctype_l:
        ext = ".gif"
    elif "webp" in ctype_l:
        ext = ".webp"
    elif "svg" in ctype_l:
        ext = ".svg"
    if ext is None:
        for cand in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
            if url.lower().split("?", 1)[0].endswith(cand):
                ext = cand
                break
    if ext is None:
        ext = ".png"

    try:
        logo_dir = Path(logic.INSTANCE_DIR) / "uploads" / "team_logos"
        logo_dir.mkdir(parents=True, exist_ok=True)
        dest = logo_dir / f"import_team{int(team_id)}{ext}"
        dest.write_bytes(data)
        try:
            os.chmod(dest, 0o644)
        except Exception:
            pass
        m.Team.objects.filter(id=int(team_id)).update(
            logo_path=str(dest), updated_at=dt.datetime.now()
        )
    except Exception:
        return


def _ensure_team_logo_for_import(
    *,
    team_id: int,
    logo_b64: Optional[str],
    logo_content_type: Optional[str],
    logo_url: Optional[str],
    replace: bool,
    commit: bool = True,
) -> None:
    del commit
    _django_orm, m = _orm_modules()
    existing = m.Team.objects.filter(id=int(team_id)).values_list("logo_path", flat=True).first()
    if existing and not replace:
        return

    b64_s = str(logo_b64 or "").strip()
    if b64_s:
        try:
            import base64

            data = base64.b64decode(b64_s.encode("ascii"), validate=False)
        except Exception:
            data = b""
        if not data or len(data) > 5 * 1024 * 1024:
            return

        ctype = str(logo_content_type or "").strip()
        ext = None
        ctype_l = ctype.lower()
        if "png" in ctype_l:
            ext = ".png"
        elif "jpeg" in ctype_l or "jpg" in ctype_l:
            ext = ".jpg"
        elif "gif" in ctype_l:
            ext = ".gif"
        elif "webp" in ctype_l:
            ext = ".webp"
        elif "svg" in ctype_l:
            ext = ".svg"
        if ext is None:
            url = str(logo_url or "").strip()
            for cand in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
                if url.lower().split("?", 1)[0].endswith(cand):
                    ext = cand
                    break
        if ext is None:
            ext = ".png"

        try:
            logo_dir = Path(logic.INSTANCE_DIR) / "uploads" / "team_logos"
            logo_dir.mkdir(parents=True, exist_ok=True)
            dest = logo_dir / f"import_team{int(team_id)}{ext}"
            dest.write_bytes(data)
            try:
                os.chmod(dest, 0o644)
            except Exception:
                pass
            m.Team.objects.filter(id=int(team_id)).update(
                logo_path=str(dest), updated_at=dt.datetime.now()
            )
            return
        except Exception:
            return

    _ensure_team_logo_from_url_for_import(team_id=int(team_id), logo_url=logo_url, replace=replace)


@csrf_exempt
def api_import_ensure_league(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    source = payload.get("source")
    external_key = payload.get("external_key")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(source) if source else None,
        external_key=str(external_key) if external_key else None,
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin")
    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id)}
    )


@csrf_exempt
def api_import_teams(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)

    teams = payload.get("teams") or []
    if not isinstance(teams, list) or not teams:
        return JsonResponse({"ok": False, "error": "teams must be a non-empty list"}, status=400)

    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(payload.get("source") or "timetoscore"),
        external_key=str(payload.get("external_key") or ""),
        commit=False,
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin", commit=False)

    results: list[dict[str, Any]] = []
    try:
        from django.db import transaction

        def _clean_division_name(dn: Any) -> Optional[str]:
            s = str(dn or "").strip()
            if not s:
                return None
            if s.lower() == "external":
                return None
            return s

        with transaction.atomic():
            for idx, team in enumerate(teams):
                if not isinstance(team, dict):
                    raise ValueError(f"teams[{idx}] must be an object")
                name = str(team.get("name") or "").strip()
                if not name:
                    continue

                team_replace = bool(team.get("replace", replace))

                division_name = _clean_division_name(team.get("division_name"))
                try:
                    division_id = (
                        int(team.get("division_id"))
                        if team.get("division_id") is not None
                        else None
                    )
                except Exception:
                    division_id = None
                try:
                    conference_id = (
                        int(team.get("conference_id"))
                        if team.get("conference_id") is not None
                        else None
                    )
                except Exception:
                    conference_id = None

                team_id = _ensure_external_team_for_import(owner_user_id, name, commit=False)
                _map_team_to_league_for_import(
                    league_id,
                    team_id,
                    division_name=division_name,
                    division_id=division_id,
                    conference_id=conference_id,
                    commit=False,
                )
                _ensure_team_logo_for_import(
                    team_id=int(team_id),
                    logo_b64=team.get("logo_b64") or team.get("team_logo_b64"),
                    logo_content_type=team.get("logo_content_type")
                    or team.get("team_logo_content_type"),
                    logo_url=team.get("logo_url") or team.get("team_logo_url"),
                    replace=team_replace,
                    commit=False,
                )
                results.append({"team_id": int(team_id), "name": name})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "owner_user_id": int(owner_user_id),
            "imported": int(len(results)),
            "results": results,
        }
    )


@csrf_exempt
def api_import_game(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(payload.get("source") or "timetoscore"),
        external_key=str(payload.get("external_key") or ""),
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin")

    game = payload.get("game") or {}
    home_name = str(game.get("home_name") or "").strip()
    away_name = str(game.get("away_name") or "").strip()
    if not home_name or not away_name:
        return JsonResponse(
            {"ok": False, "error": "home_name and away_name are required"}, status=400
        )

    home_ph = logic.parse_seed_placeholder_name(home_name)
    away_ph = logic.parse_seed_placeholder_name(away_name)
    is_home_placeholder = bool(home_ph is not None)
    is_away_placeholder = bool(away_ph is not None)

    division_name = str(game.get("division_name") or "").strip() or None
    home_division_name = str(game.get("home_division_name") or division_name or "").strip() or None
    away_division_name = str(game.get("away_division_name") or division_name or "").strip() or None
    try:
        division_id = int(game.get("division_id")) if game.get("division_id") is not None else None
    except Exception:
        division_id = None
    try:
        conference_id = (
            int(game.get("conference_id")) if game.get("conference_id") is not None else None
        )
    except Exception:
        conference_id = None
    try:
        home_division_id = (
            int(game.get("home_division_id"))
            if game.get("home_division_id") is not None
            else division_id
        )
    except Exception:
        home_division_id = division_id
    try:
        away_division_id = (
            int(game.get("away_division_id"))
            if game.get("away_division_id") is not None
            else division_id
        )
    except Exception:
        away_division_id = division_id
    try:
        home_conference_id = (
            int(game.get("home_conference_id"))
            if game.get("home_conference_id") is not None
            else conference_id
        )
    except Exception:
        home_conference_id = conference_id
    try:
        away_conference_id = (
            int(game.get("away_conference_id"))
            if game.get("away_conference_id") is not None
            else conference_id
        )
    except Exception:
        away_conference_id = conference_id

    if is_home_placeholder:
        team1_id = logic.ensure_seed_placeholder_team_for_import(int(owner_user_id))
    else:
        team1_id = _ensure_external_team_for_import(owner_user_id, home_name)
    if is_away_placeholder:
        team2_id = logic.ensure_seed_placeholder_team_for_import(int(owner_user_id))
    else:
        team2_id = _ensure_external_team_for_import(owner_user_id, away_name)

    if not is_home_placeholder:
        _map_team_to_league_for_import(
            league_id,
            team1_id,
            division_name=home_division_name,
            division_id=home_division_id,
            conference_id=home_conference_id,
        )
    if not is_away_placeholder:
        _map_team_to_league_for_import(
            league_id,
            team2_id,
            division_name=away_division_name,
            division_id=away_division_id,
            conference_id=away_conference_id,
        )
    if not is_home_placeholder:
        _ensure_team_logo_for_import(
            team_id=int(team1_id),
            logo_b64=game.get("home_logo_b64") or game.get("team1_logo_b64"),
            logo_content_type=game.get("home_logo_content_type")
            or game.get("team1_logo_content_type"),
            logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
            replace=replace,
        )
    if not is_away_placeholder:
        _ensure_team_logo_for_import(
            team_id=int(team2_id),
            logo_b64=game.get("away_logo_b64") or game.get("team2_logo_b64"),
            logo_content_type=game.get("away_logo_content_type")
            or game.get("team2_logo_content_type"),
            logo_url=game.get("away_logo_url") or game.get("team2_logo_url"),
            replace=replace,
        )

    starts_at = game.get("starts_at")
    starts_at_s = str(starts_at) if starts_at else None
    location = str(game.get("location")).strip() if game.get("location") else None
    team1_score = game.get("home_score")
    team2_score = game.get("away_score")
    tts_game_id = game.get("timetoscore_game_id")

    notes_fields: dict[str, Any] = {}
    if tts_game_id is not None:
        try:
            notes_fields["timetoscore_game_id"] = int(tts_game_id)
        except Exception:
            pass
    if game.get("season_id") is not None:
        try:
            notes_fields["timetoscore_season_id"] = int(game.get("season_id"))
        except Exception:
            pass
    if payload.get("source"):
        notes_fields["source"] = str(payload.get("source"))
    if game.get("timetoscore_type") is not None:
        notes_fields["timetoscore_type"] = str(game.get("timetoscore_type"))
    elif game.get("game_type_name") is not None:
        notes_fields["timetoscore_type"] = str(game.get("game_type_name"))
    elif game.get("type") is not None:
        notes_fields["timetoscore_type"] = str(game.get("type"))
    if is_home_placeholder or is_away_placeholder:
        seed_placeholders: dict[str, Any] = {}
        if home_ph is not None:
            seed_placeholders["team1"] = {
                "seed": int(home_ph.seed),
                "division_token": str(home_ph.division_token),
                "raw": str(home_ph.raw),
                "division_name_hint": str(home_division_name or division_name or ""),
            }
        if away_ph is not None:
            seed_placeholders["team2"] = {
                "seed": int(away_ph.seed),
                "division_token": str(away_ph.division_token),
                "raw": str(away_ph.raw),
                "division_name_hint": str(away_division_name or division_name or ""),
            }
        notes_fields["seed_placeholders"] = seed_placeholders

    game_type_id = _ensure_game_type_id_for_import(
        game.get("game_type_name")
        or game.get("game_type")
        or game.get("timetoscore_type")
        or game.get("type")
    )

    try:
        t1s = int(team1_score) if team1_score is not None else None
    except Exception:
        t1s = None
    try:
        t2s = int(team2_score) if team2_score is not None else None
    except Exception:
        t2s = None

    gid = _upsert_game_for_import(
        owner_user_id=owner_user_id,
        team1_id=team1_id,
        team2_id=team2_id,
        game_type_id=game_type_id,
        starts_at=starts_at_s,
        location=location,
        team1_score=t1s,
        team2_score=t2s,
        replace=replace,
        notes_json_fields=notes_fields,
    )
    _map_game_to_league_for_import(
        league_id,
        gid,
        division_name=division_name or home_division_name or away_division_name,
        division_id=division_id or home_division_id or away_division_id,
        conference_id=conference_id or home_conference_id or away_conference_id,
    )

    roster_player_ids_by_team: dict[int, set[int]] = {int(team1_id): set(), int(team2_id): set()}
    for side_key, tid, is_placeholder in (
        ("home", team1_id, is_home_placeholder),
        ("away", team2_id, is_away_placeholder),
    ):
        if is_placeholder:
            continue
        roster = game.get(f"{side_key}_roster") or []
        if isinstance(roster, list):
            for row in roster:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name") or "").strip()
                if not nm:
                    continue
                jersey = str(row.get("number") or "").strip() or None
                pos = str(row.get("position") or "").strip() or None
                pid = _ensure_player_for_import(owner_user_id, tid, nm, jersey, pos)
                try:
                    roster_player_ids_by_team[int(tid)].add(int(pid))
                except Exception:
                    pass

    def _player_id_by_name(team_id: int, name: str) -> Optional[int]:
        _django_orm2, m2 = _orm_modules()
        pid = (
            m2.Player.objects.filter(
                user_id=int(owner_user_id), team_id=int(team_id), name=str(name)
            )
            .values_list("id", flat=True)
            .first()
        )
        return int(pid) if pid is not None else None

    _django_orm2, m2 = _orm_modules()
    from django.db import transaction

    with transaction.atomic():
        events_csv = game.get("events_csv")
        played = (
            bool(game.get("is_final"))
            or (t1s is not None and t2s is not None)
            or (isinstance(events_csv, str) and events_csv.strip())
        )
        now = dt.datetime.now()
        to_create_links = []
        if played and roster_player_ids_by_team:
            for tid, pids in roster_player_ids_by_team.items():
                for pid in sorted(pids):
                    to_create_links.append(
                        m2.HkyGamePlayer(
                            game_id=int(gid),
                            player_id=int(pid),
                            team_id=int(tid),
                            created_at=now,
                            updated_at=None,
                        )
                    )
            if to_create_links:
                m2.HkyGamePlayer.objects.bulk_create(to_create_links, ignore_conflicts=True)

        if isinstance(events_csv, str) and events_csv.strip():
            try:
                _upsert_game_event_rows_from_events_csv(
                    game_id=int(gid),
                    events_csv=str(events_csv),
                    replace=bool(replace),
                    create_missing_players=False,
                    incoming_source_label="timetoscore",
                    prefer_incoming_for_event_types={
                        "goal",
                        "assist",
                        "penalty",
                        "penaltyexpired",
                        "goaliechange",
                    },
                )
            except Exception:
                logger.warning(
                    "api_import_game: failed to upsert event rows (game_id=%s, replace=%s)",
                    int(gid),
                    bool(replace),
                    exc_info=True,
                )
                raise

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "owner_user_id": int(owner_user_id),
            "team1_id": int(team1_id),
            "team2_id": int(team2_id),
            "game_id": int(gid),
        }
    )


@csrf_exempt
def api_import_games_batch(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "CAHA")
    shared = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "caha-import@hockeymom.local")
    owner_name = str(payload.get("owner_name") or "CAHA Import")
    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)

    games = payload.get("games") or []
    if not isinstance(games, list) or not games:
        return JsonResponse({"ok": False, "error": "games must be a non-empty list"}, status=400)

    league_id = _ensure_league_for_import(
        league_name=league_name,
        owner_user_id=owner_user_id,
        is_shared=shared,
        source=str(payload.get("source") or "timetoscore"),
        external_key=str(payload.get("external_key") or ""),
        commit=False,
    )
    _ensure_league_member_for_import(league_id, owner_user_id, role="admin", commit=False)

    results: list[dict[str, Any]] = []
    try:
        _django_orm, m = _orm_modules()

        def _clean_division_name(dn: Any) -> Optional[str]:
            s = str(dn or "").strip()
            if not s:
                return None
            if s.lower() == "external":
                return None
            return s

        def _league_team_div_meta(
            lid: int, tid: int
        ) -> tuple[Optional[str], Optional[int], Optional[int]]:
            row = (
                m.LeagueTeam.objects.filter(league_id=int(lid), team_id=int(tid))
                .values("division_name", "division_id", "conference_id")
                .first()
            )
            if not row:
                return None, None, None
            try:
                did = int(row.get("division_id")) if row.get("division_id") is not None else None
            except Exception:
                did = None
            try:
                cid = (
                    int(row.get("conference_id")) if row.get("conference_id") is not None else None
                )
            except Exception:
                cid = None
            return _clean_division_name(row.get("division_name")), did, cid

        for idx, game in enumerate(games):
            if not isinstance(game, dict):
                raise ValueError(f"games[{idx}] must be an object")
            home_name = str(game.get("home_name") or "").strip()
            away_name = str(game.get("away_name") or "").strip()
            if not home_name or not away_name:
                raise ValueError(f"games[{idx}]: home_name and away_name are required")

            game_replace = bool(game.get("replace", replace))

            division_name = _clean_division_name(game.get("division_name"))
            home_division_name = _clean_division_name(
                game.get("home_division_name") or division_name
            )
            away_division_name = _clean_division_name(
                game.get("away_division_name") or division_name
            )
            try:
                division_id = (
                    int(game.get("division_id")) if game.get("division_id") is not None else None
                )
            except Exception:
                division_id = None
            try:
                conference_id = (
                    int(game.get("conference_id"))
                    if game.get("conference_id") is not None
                    else None
                )
            except Exception:
                conference_id = None
            try:
                home_division_id = (
                    int(game.get("home_division_id"))
                    if game.get("home_division_id") is not None
                    else division_id
                )
            except Exception:
                home_division_id = division_id
            try:
                away_division_id = (
                    int(game.get("away_division_id"))
                    if game.get("away_division_id") is not None
                    else division_id
                )
            except Exception:
                away_division_id = division_id
            try:
                home_conference_id = (
                    int(game.get("home_conference_id"))
                    if game.get("home_conference_id") is not None
                    else conference_id
                )
            except Exception:
                home_conference_id = conference_id
            try:
                away_conference_id = (
                    int(game.get("away_conference_id"))
                    if game.get("away_conference_id") is not None
                    else conference_id
                )
            except Exception:
                away_conference_id = conference_id

            home_ph = logic.parse_seed_placeholder_name(home_name)
            away_ph = logic.parse_seed_placeholder_name(away_name)
            is_home_placeholder = bool(home_ph is not None)
            is_away_placeholder = bool(away_ph is not None)

            if is_home_placeholder:
                team1_id = logic.ensure_seed_placeholder_team_for_import(
                    int(owner_user_id), commit=False
                )
            else:
                team1_id = _ensure_external_team_for_import(owner_user_id, home_name, commit=False)
            if is_away_placeholder:
                team2_id = logic.ensure_seed_placeholder_team_for_import(
                    int(owner_user_id), commit=False
                )
            else:
                team2_id = _ensure_external_team_for_import(owner_user_id, away_name, commit=False)

            if not is_home_placeholder:
                _map_team_to_league_for_import(
                    league_id,
                    team1_id,
                    division_name=home_division_name,
                    division_id=home_division_id,
                    conference_id=home_conference_id,
                    commit=False,
                )
            if not is_away_placeholder:
                _map_team_to_league_for_import(
                    league_id,
                    team2_id,
                    division_name=away_division_name,
                    division_id=away_division_id,
                    conference_id=away_conference_id,
                    commit=False,
                )
            if not is_home_placeholder:
                _ensure_team_logo_for_import(
                    team_id=int(team1_id),
                    logo_b64=game.get("home_logo_b64") or game.get("team1_logo_b64"),
                    logo_content_type=game.get("home_logo_content_type")
                    or game.get("team1_logo_content_type"),
                    logo_url=game.get("home_logo_url") or game.get("team1_logo_url"),
                    replace=game_replace,
                    commit=False,
                )
            if not is_away_placeholder:
                _ensure_team_logo_for_import(
                    team_id=int(team2_id),
                    logo_b64=game.get("away_logo_b64") or game.get("team2_logo_b64"),
                    logo_content_type=game.get("away_logo_content_type")
                    or game.get("team2_logo_content_type"),
                    logo_url=game.get("away_logo_url") or game.get("team2_logo_url"),
                    replace=game_replace,
                    commit=False,
                )

            starts_at = game.get("starts_at")
            starts_at_s = str(starts_at) if starts_at else None
            location = str(game.get("location")).strip() if game.get("location") else None
            team1_score = game.get("home_score")
            team2_score = game.get("away_score")
            tts_game_id = game.get("timetoscore_game_id")
            ext_game_key = str(game.get("external_game_key") or "").strip() or None

            notes_fields: dict[str, Any] = {}
            if tts_game_id is not None:
                try:
                    notes_fields["timetoscore_game_id"] = int(tts_game_id)
                except Exception:
                    pass
            if ext_game_key:
                notes_fields["external_game_key"] = str(ext_game_key)
            if game.get("season_id") is not None:
                try:
                    notes_fields["timetoscore_season_id"] = int(game.get("season_id"))
                except Exception:
                    pass
            if payload.get("source"):
                notes_fields["source"] = str(payload.get("source"))
            if game.get("timetoscore_type") is not None:
                notes_fields["timetoscore_type"] = str(game.get("timetoscore_type"))
            elif game.get("game_type_name") is not None:
                notes_fields["timetoscore_type"] = str(game.get("game_type_name"))
            elif game.get("type") is not None:
                notes_fields["timetoscore_type"] = str(game.get("type"))

            # Optional schedule metadata (used by CAHA schedule.pl imports).
            if game.get("caha_schedule_year") is not None:
                try:
                    notes_fields["caha_schedule_year"] = int(game.get("caha_schedule_year"))
                except Exception:
                    pass
            if game.get("caha_schedule_group") is not None:
                notes_fields["caha_schedule_group"] = str(game.get("caha_schedule_group"))
            if game.get("caha_schedule_game_number") is not None:
                try:
                    notes_fields["caha_schedule_game_number"] = int(
                        game.get("caha_schedule_game_number")
                    )
                except Exception:
                    pass
            if is_home_placeholder or is_away_placeholder:
                seed_placeholders: dict[str, Any] = {}
                if home_ph is not None:
                    seed_placeholders["team1"] = {
                        "seed": int(home_ph.seed),
                        "division_token": str(home_ph.division_token),
                        "raw": str(home_ph.raw),
                        "division_name_hint": str(home_division_name or division_name or ""),
                    }
                if away_ph is not None:
                    seed_placeholders["team2"] = {
                        "seed": int(away_ph.seed),
                        "division_token": str(away_ph.division_token),
                        "raw": str(away_ph.raw),
                        "division_name_hint": str(away_division_name or division_name or ""),
                    }
                notes_fields["seed_placeholders"] = seed_placeholders

            game_type_id = _ensure_game_type_id_for_import(
                game.get("game_type_name")
                or game.get("game_type")
                or game.get("timetoscore_type")
                or game.get("type")
            )

            try:
                t1s = int(team1_score) if team1_score is not None else None
            except Exception:
                t1s = None
            try:
                t2s = int(team2_score) if team2_score is not None else None
            except Exception:
                t2s = None

            gid = _upsert_game_for_import(
                owner_user_id=owner_user_id,
                team1_id=team1_id,
                team2_id=team2_id,
                game_type_id=game_type_id,
                starts_at=starts_at_s,
                location=location,
                team1_score=t1s,
                team2_score=t2s,
                replace=game_replace,
                notes_json_fields=notes_fields,
                commit=False,
            )

            effective_div_name = division_name or home_division_name or away_division_name
            effective_div_id = division_id or home_division_id or away_division_id
            effective_conf_id = conference_id or home_conference_id or away_conference_id
            if not effective_div_name:
                t1_dn, t1_did, t1_cid = _league_team_div_meta(int(league_id), int(team1_id))
                t2_dn, t2_did, t2_cid = _league_team_div_meta(int(league_id), int(team2_id))
                if t1_dn:
                    effective_div_name = t1_dn
                    effective_div_id = effective_div_id or t1_did
                    effective_conf_id = effective_conf_id or t1_cid
                elif t2_dn:
                    effective_div_name = t2_dn
                    effective_div_id = effective_div_id or t2_did
                    effective_conf_id = effective_conf_id or t2_cid
            _map_game_to_league_for_import(
                league_id,
                gid,
                division_name=effective_div_name,
                division_id=effective_div_id,
                conference_id=effective_conf_id,
                commit=False,
            )

            roster_player_ids_by_team: dict[int, set[int]] = {
                int(team1_id): set(),
                int(team2_id): set(),
            }
            for side_key, tid in (("home", team1_id), ("away", team2_id)):
                roster = game.get(f"{side_key}_roster") or []
                if isinstance(roster, list):
                    for row in roster:
                        if not isinstance(row, dict):
                            continue
                        nm = str(row.get("name") or "").strip()
                        if not nm:
                            continue
                        jersey = str(row.get("number") or "").strip() or None
                        pos = str(row.get("position") or "").strip() or None
                        pid = _ensure_player_for_import(
                            owner_user_id, tid, nm, jersey, pos, commit=False
                        )
                        try:
                            roster_player_ids_by_team[int(tid)].add(int(pid))
                        except Exception:
                            pass

            events_csv = game.get("events_csv")
            played = (
                bool(game.get("is_final"))
                or (t1s is not None and t2s is not None)
                or (isinstance(events_csv, str) and events_csv.strip())
            )
            if played and roster_player_ids_by_team:
                now = dt.datetime.now()
                to_create_links = []
                for tid, pids in roster_player_ids_by_team.items():
                    for pid in sorted(pids):
                        to_create_links.append(
                            m.HkyGamePlayer(
                                game_id=int(gid),
                                player_id=int(pid),
                                team_id=int(tid),
                                created_at=now,
                                updated_at=None,
                            )
                        )
                if to_create_links:
                    m.HkyGamePlayer.objects.bulk_create(to_create_links, ignore_conflicts=True)

            if isinstance(events_csv, str) and events_csv.strip():
                try:
                    _upsert_game_event_rows_from_events_csv(
                        game_id=int(gid),
                        events_csv=str(events_csv),
                        replace=bool(game_replace),
                        create_missing_players=False,
                        incoming_source_label="timetoscore",
                        prefer_incoming_for_event_types={
                            "goal",
                            "assist",
                            "penalty",
                            "penaltyexpired",
                            "goaliechange",
                        },
                    )
                except Exception:
                    logger.warning(
                        "api_import_games_batch: failed to upsert event rows (game_id=%s, replace=%s)",
                        int(gid),
                        bool(game_replace),
                        exc_info=True,
                    )
                    raise

            results.append({"game_id": gid, "team1_id": team1_id, "team2_id": team2_id})

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return JsonResponse(
        {
            "ok": True,
            "league_id": int(league_id),
            "owner_user_id": int(owner_user_id),
            "imported": int(len(results)),
            "results": results,
        }
    )


@csrf_exempt
def api_import_shift_package(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    replace = bool(payload.get("replace", False))

    game_id = payload.get("game_id")
    tts_game_id = payload.get("timetoscore_game_id")
    external_game_key = str(payload.get("external_game_key") or "").strip() or None
    team_side = str(payload.get("team_side") or "").strip().lower() or None
    if team_side not in {None, "", "home", "away"}:
        return JsonResponse(
            {"ok": False, "error": "team_side must be 'home' or 'away'"}, status=400
        )
    stats_note: Optional[str] = None
    raw_stats_note = payload.get("stats_note") or payload.get("schedule_note")
    if isinstance(raw_stats_note, str) and raw_stats_note.strip():
        stats_note = " ".join(str(raw_stats_note).strip().split()) or None
    if "create_missing_players" in payload:
        create_missing_players = bool(payload.get("create_missing_players"))
    else:
        create_missing_players = True
    owner_email = str(payload.get("owner_email") or "").strip().lower() or None
    league_id_payload = payload.get("league_id")
    league_name = str(payload.get("league_name") or "").strip() or None
    division_name = str(payload.get("division_name") or "").strip() or None
    sort_order_payload = payload.get("sort_order")
    sort_order: Optional[int] = None
    try:
        sort_order = int(sort_order_payload) if sort_order_payload is not None else None
    except Exception:
        sort_order = None
    resolved_game_id: Optional[int] = None
    try:
        resolved_game_id = int(game_id) if game_id is not None else None
    except Exception:
        resolved_game_id = None

    _django_orm, m = _orm_modules()

    from django.db.models import Q

    tts_int: Optional[int]
    try:
        tts_int = int(tts_game_id) if tts_game_id is not None else None
    except Exception:
        tts_int = None

    if resolved_game_id is None and tts_int is not None:
        gid = (
            m.HkyGame.objects.filter(timetoscore_game_id=int(tts_int))
            .values_list("id", flat=True)
            .first()
        )
        if gid is None:
            token_json_nospace = f'"timetoscore_game_id":{int(tts_int)}'
            token_json_space = f'"timetoscore_game_id": {int(tts_int)}'
            token_plain = f"game_id={int(tts_int)}"
            gid = (
                m.HkyGame.objects.filter(
                    Q(notes__contains=token_json_nospace)
                    | Q(notes__contains=token_json_space)
                    | Q(notes__contains=token_plain)
                )
                .values_list("id", flat=True)
                .first()
            )
        if gid is not None:
            resolved_game_id = int(gid)

    if resolved_game_id is None and external_game_key and owner_email:
        owner_user_id_for_create = _ensure_user_for_import(owner_email)

        def _norm_team_name_for_match(s: str) -> str:
            t = str(s or "").replace("\xa0", " ").strip()
            t = (
                t.replace("\u2010", "-")
                .replace("\u2011", "-")
                .replace("\u2012", "-")
                .replace("\u2013", "-")
                .replace("\u2212", "-")
            )
            t = " ".join(t.split())
            t = re.sub(r"\s*\(\s*external\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
            t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
            t = t.casefold()
            t = re.sub(r"[^0-9a-z]+", " ", t)
            return " ".join(t.split())

        def _find_team_in_league_by_name(league_id_i: int, name: str) -> Optional[dict[str, Any]]:
            nm = _norm_team_name_for_match(name)
            if not nm:
                return None
            rows = list(
                m.LeagueTeam.objects.filter(league_id=int(league_id_i))
                .select_related("team")
                .values("team_id", "team__name", "division_name", "division_id", "conference_id")
            )
            matches = [
                {
                    "team_id": int(r["team_id"]),
                    "team_name": str(r.get("team__name") or ""),
                    "division_name": r.get("division_name"),
                    "division_id": r.get("division_id"),
                    "conference_id": r.get("conference_id"),
                }
                for r in rows
                if _norm_team_name_for_match(str(r.get("team__name") or "")) == nm
            ]
            if not matches:
                return None
            if len(matches) == 1:
                return matches[0]
            want_div = str(payload.get("division_name") or "").strip()
            if want_div and want_div.lower() != "external":
                by_div = [
                    cand
                    for cand in matches
                    if str(cand.get("division_name") or "").strip() == want_div
                ]
                if len(by_div) == 1:
                    return by_div[0]
            for cand in matches:
                dn = str(cand.get("division_name") or "").strip()
                if dn:
                    return cand
            return matches[0]

        gid = (
            m.HkyGame.objects.filter(
                user_id=int(owner_user_id_for_create), external_game_key=str(external_game_key)
            )
            .values_list("id", flat=True)
            .first()
        )
        if gid is None:
            try:
                ext_json = json.dumps(external_game_key)
            except Exception:
                ext_json = f'"{external_game_key}"'
            tokens = [f'"external_game_key":{ext_json}', f'"external_game_key": {ext_json}']
            gid = (
                m.HkyGame.objects.filter(user_id=int(owner_user_id_for_create))
                .filter(Q(notes__contains=tokens[0]) | Q(notes__contains=tokens[1]))
                .values_list("id", flat=True)
                .first()
            )
        if gid is not None:
            resolved_game_id = int(gid)
        else:
            home_team_name = str(payload.get("home_team_name") or "").strip()
            away_team_name = str(payload.get("away_team_name") or "").strip()
            if not home_team_name or not away_team_name:
                return JsonResponse(
                    {
                        "ok": False,
                        "error": "home_team_name and away_team_name are required to create an external game",
                    },
                    status=400,
                )

            league_id_i: Optional[int] = None
            try:
                league_id_i = int(league_id_payload) if league_id_payload is not None else None
            except Exception:
                league_id_i = None
            if league_id_i is None:
                if not league_name:
                    return JsonResponse(
                        {
                            "ok": False,
                            "error": "league_id or league_name is required to create an external game",
                        },
                        status=400,
                    )
                existing_lid = (
                    m.League.objects.filter(name=str(league_name))
                    .values_list("id", flat=True)
                    .first()
                )
                if existing_lid is not None:
                    league_id_i = int(existing_lid)
                else:
                    now = dt.datetime.now()
                    league = m.League.objects.create(
                        name=str(league_name),
                        owner_user_id=int(owner_user_id_for_create),
                        is_shared=False,
                        is_public=False,
                        source="shift_package",
                        external_key=None,
                        created_at=now,
                        updated_at=None,
                    )
                    league_id_i = int(league.id)

            match_home = (
                _find_team_in_league_by_name(int(league_id_i), home_team_name)
                if league_id_i
                else None
            )
            match_away = (
                _find_team_in_league_by_name(int(league_id_i), away_team_name)
                if league_id_i
                else None
            )

            team1_id = (
                int(match_home["team_id"])
                if match_home
                else _ensure_external_team_for_import(
                    owner_user_id_for_create, home_team_name, commit=False
                )
            )
            team2_id = (
                int(match_away["team_id"])
                if match_away
                else _ensure_external_team_for_import(
                    owner_user_id_for_create, away_team_name, commit=False
                )
            )

            def _infer_base_division_name(s: Optional[str]) -> Optional[str]:
                raw = str(s or "").strip()
                if not raw:
                    return None
                if logic.is_external_division_name(raw):
                    raw = re.sub(r"(?i)^external\s*", "", raw).strip()
                m_age_level = re.search(
                    r"(?i)(?:^|\b)(\d{1,2})(?:u)?\s*(AAA|AA|BB|A|B)(?=\b|\s|$|[-–—])",
                    raw,
                )
                if m_age_level:
                    try:
                        age = int(m_age_level.group(1))
                    except Exception:
                        age = None
                    level = str(m_age_level.group(2) or "").strip().upper() or None
                    if age is not None and level:
                        return f"{age} {level}"
                raw_cf = raw.casefold()
                if "mite" in raw_cf:
                    return "8U"
                if "squirt" in raw_cf:
                    return "10U"
                if "peewee" in raw_cf or "pee wee" in raw_cf:
                    return "12U"
                if "bantam" in raw_cf:
                    return "14U"
                if "midget" in raw_cf:
                    return "16U"
                if "junior" in raw_cf:
                    return "18U"
                m_age_u = re.search(r"(?i)(?:^|\b)(\d{1,2})u(?=\b|\s|$|[-–—])", raw)
                if m_age_u:
                    try:
                        age = int(m_age_u.group(1))
                    except Exception:
                        age = None
                    if age is not None:
                        return f"{age}U"
                return None

            base_division_name = (
                _infer_base_division_name(division_name)
                or _infer_base_division_name(home_team_name)
                or _infer_base_division_name(away_team_name)
            )
            external_division_name = (
                f"External {base_division_name}".strip() if base_division_name else "External"
            )
            game_division_name = external_division_name
            new_team_division_name = external_division_name
            division_name = external_division_name

            _ensure_team_logo_for_import(
                team_id=team1_id,
                logo_b64=payload.get("home_logo_b64"),
                logo_content_type=payload.get("home_logo_content_type"),
                logo_url=payload.get("home_logo_url"),
                replace=replace,
                commit=False,
            )
            _ensure_team_logo_for_import(
                team_id=team2_id,
                logo_b64=payload.get("away_logo_b64"),
                logo_content_type=payload.get("away_logo_content_type"),
                logo_url=payload.get("away_logo_url"),
                replace=replace,
                commit=False,
            )

            team1_score = None
            team2_score = None
            try:
                raw_events_csv = payload.get("events_csv")
                if isinstance(raw_events_csv, str) and raw_events_csv.strip():
                    _h, _rows = logic.parse_events_csv(str(raw_events_csv))
                    if _rows:
                        home_goals = 0
                        away_goals = 0
                        for r0 in _rows:
                            if not isinstance(r0, dict):
                                continue
                            et = str(
                                r0.get("Event Type") or r0.get("Event") or r0.get("Type") or ""
                            ).strip()
                            if not et or et.casefold() != "goal":
                                continue
                            side = str(r0.get("Team Side") or r0.get("TeamSide") or "").strip()
                            if side.casefold() == "home":
                                home_goals += 1
                            elif side.casefold() == "away":
                                away_goals += 1
                        team1_score = int(home_goals)
                        team2_score = int(away_goals)
            except Exception:
                team1_score, team2_score = None, None

            starts_at = str(payload.get("starts_at") or "").strip() or None
            location = str(payload.get("location") or "").strip() or None
            notes_fields: dict[str, Any] = {"external_game_key": external_game_key}
            if tts_int is not None:
                notes_fields["timetoscore_game_id"] = int(tts_int)
            if stats_note:
                notes_fields["stats_note"] = str(stats_note)
            resolved_game_id = _upsert_game_for_import(
                owner_user_id=owner_user_id_for_create,
                team1_id=team1_id,
                team2_id=team2_id,
                game_type_id=None,
                starts_at=starts_at,
                location=location,
                team1_score=team1_score,
                team2_score=team2_score,
                replace=replace,
                notes_json_fields=notes_fields,
                commit=False,
            )
            if not match_home:
                _map_team_to_league_for_import(
                    int(league_id_i), team1_id, division_name=new_team_division_name, commit=False
                )
            if not match_away:
                _map_team_to_league_for_import(
                    int(league_id_i), team2_id, division_name=new_team_division_name, commit=False
                )
            _map_game_to_league_for_import(
                int(league_id_i),
                int(resolved_game_id),
                division_name=game_division_name,
                sort_order=sort_order,
                commit=False,
            )
            # Do not implicitly create players for external games; respect the payload flag.

    if resolved_game_id is None:
        return JsonResponse(
            {
                "ok": False,
                "error": "game_id, timetoscore_game_id, or external_game_key+owner_email+league_name+home_team_name+away_team_name is required",
            },
            status=400,
        )

    game_row = (
        m.HkyGame.objects.filter(id=int(resolved_game_id))
        .values(
            "id",
            "team1_id",
            "team2_id",
            "user_id",
            "notes",
            "timetoscore_game_id",
            "external_game_key",
        )
        .first()
    )
    if not game_row:
        return JsonResponse({"ok": False, "error": "game not found"}, status=404)

    key_fields: dict[str, Any] = {}
    if tts_int is not None:
        key_fields["timetoscore_game_id"] = int(tts_int)
    if external_game_key:
        key_fields["external_game_key"] = str(external_game_key)
    if key_fields:
        resolved_game_id = _upsert_game_for_import(
            owner_user_id=int(game_row.get("user_id") or 0),
            team1_id=int(game_row["team1_id"]),
            team2_id=int(game_row["team2_id"]),
            game_type_id=None,
            starts_at=None,
            location=None,
            team1_score=None,
            team2_score=None,
            replace=False,
            notes_json_fields=key_fields,
            commit=False,
        )
        game_row = (
            m.HkyGame.objects.filter(id=int(resolved_game_id))
            .values(
                "id",
                "team1_id",
                "team2_id",
                "user_id",
                "notes",
                "timetoscore_game_id",
                "external_game_key",
            )
            .first()
        )
        if not game_row:
            return JsonResponse({"ok": False, "error": "game not found after merge"}, status=404)

    team1_id = int(game_row["team1_id"])
    team2_id = int(game_row["team2_id"])
    owner_user_id = int(game_row.get("user_id") or 0)

    # Team logos can be provided in shift_package payloads (e.g., from file-list YAML).
    # Apply them even when the game already exists so rerunning imports can backfill missing icons.
    try:
        _ensure_team_logo_for_import(
            team_id=int(team1_id),
            logo_b64=payload.get("home_logo_b64"),
            logo_content_type=payload.get("home_logo_content_type"),
            logo_url=payload.get("home_logo_url"),
            replace=replace,
            commit=False,
        )
        _ensure_team_logo_for_import(
            team_id=int(team2_id),
            logo_b64=payload.get("away_logo_b64"),
            logo_content_type=payload.get("away_logo_content_type"),
            logo_url=payload.get("away_logo_url"),
            replace=replace,
            commit=False,
        )
    except Exception:
        logger.warning(
            "api_import_shift_package: failed to apply team logos (game_id=%s)",
            int(resolved_game_id),
            exc_info=True,
        )

    events_csv = payload.get("events_csv")
    shift_rows_csv = payload.get("shift_rows_csv")
    replace_shift_rows_payload = payload.get("replace_shift_rows")
    game_video_url = (
        payload.get("game_video_url") or payload.get("game_video") or payload.get("video_url")
    )

    if isinstance(game_video_url, str) and game_video_url.strip():
        try:
            _update_game_video_url_note(
                int(resolved_game_id), str(game_video_url), replace=replace, commit=False
            )
        except Exception:
            pass
    if stats_note:
        try:
            _update_game_stats_note(
                int(resolved_game_id), str(stats_note), replace=replace, commit=False
            )
        except Exception:
            pass

    if isinstance(events_csv, str) and events_csv.strip():
        drop_types = {"power play", "powerplay", "penalty kill", "penaltykill"}
        try:
            events_csv = logic.filter_events_csv_drop_event_types(
                str(events_csv), drop_types=drop_types
            )
        except Exception:
            pass
        try:
            _h, _r = logic.parse_events_csv(str(events_csv))
            if not _r:
                events_csv = None
        except Exception:
            pass

    if league_id_payload is not None or league_name:
        league_id_i: Optional[int] = None
        try:
            league_id_i = int(league_id_payload) if league_id_payload is not None else None
        except Exception:
            league_id_i = None
        if league_id_i is None and league_name:
            existing_lid = (
                m.League.objects.filter(name=str(league_name)).values_list("id", flat=True).first()
            )
            if existing_lid is not None:
                league_id_i = int(existing_lid)
        if league_id_i is not None:
            _map_team_to_league_for_import(
                int(league_id_i), team1_id, division_name=division_name, commit=False
            )
            _map_team_to_league_for_import(
                int(league_id_i), team2_id, division_name=division_name, commit=False
            )
            _map_game_to_league_for_import(
                int(league_id_i),
                int(resolved_game_id),
                division_name=division_name,
                sort_order=sort_order,
                commit=False,
            )

    imported = 0
    imported_shifts = 0
    unmatched: list[str] = []

    if isinstance(shift_rows_csv, str) and shift_rows_csv.strip():
        if team_side not in {"home", "away"}:
            return JsonResponse(
                {
                    "ok": False,
                    "error": "shift_rows_csv requires team_side='home' or 'away'",
                },
                status=400,
            )

    # Optional roster seed (provided by client, e.g., parse_stats_inputs scraping TimeToScore).
    # This lets the webapp create missing roster players (including goalies) without contacting T2S.
    roster_player_ids_by_team: dict[int, set[int]] = {
        int(team1_id): set(),
        int(team2_id): set(),
    }
    roster_home = payload.get("roster_home") or []
    roster_away = payload.get("roster_away") or []
    if isinstance(roster_home, list) or isinstance(roster_away, list):
        try:
            for side, roster in (("home", roster_home), ("away", roster_away)):
                if not isinstance(roster, list):
                    continue
                tid = team1_id if side == "home" else team2_id
                for rec in roster:
                    if not isinstance(rec, dict):
                        continue
                    nm = str(rec.get("name") or "").strip()
                    if not nm:
                        continue
                    jersey_norm = logic.normalize_jersey_number(rec.get("jersey_number"))
                    pos = str(rec.get("position") or "").strip() or None
                    pid = _ensure_player_for_import(
                        int(owner_user_id),
                        int(tid),
                        nm,
                        str(jersey_norm) if jersey_norm else None,
                        pos,
                        commit=False,
                    )
                    try:
                        roster_player_ids_by_team[int(tid)].add(int(pid))
                    except Exception:
                        pass
        except Exception:
            logger.exception(
                "Error while creating/importing roster players (game_id=%s)", resolved_game_id
            )

    try:
        from django.db import transaction

        players = list(
            m.Player.objects.filter(team_id__in=[int(team1_id), int(team2_id)]).values(
                "id", "team_id", "name", "jersey_number"
            )
        )

        players_by_team: dict[int, list[dict[str, Any]]] = {}
        jersey_to_player_ids: dict[tuple[int, str], list[int]] = {}
        name_to_player_ids: dict[tuple[int, str], list[int]] = {}
        player_team_by_id: dict[int, int] = {}

        def _register_player(pid: int, tid: int, *, name: str, jersey_number: Any) -> None:
            player_team_by_id[int(pid)] = int(tid)
            p = {"id": int(pid), "team_id": int(tid), "name": name, "jersey_number": jersey_number}
            players_by_team.setdefault(int(tid), []).append(p)
            j = logic.normalize_jersey_number(jersey_number)
            if j:
                jersey_to_player_ids.setdefault((int(tid), j), []).append(int(pid))
            nm = logic.normalize_player_name(name or "")
            if nm:
                name_to_player_ids.setdefault((int(tid), nm), []).append(int(pid))
            nm_no_mid = _normalize_player_name_no_middle(name or "")
            if nm_no_mid:
                name_to_player_ids.setdefault((int(tid), nm_no_mid), []).append(int(pid))

        for p in players:
            _register_player(
                int(p["id"]),
                int(p["team_id"]),
                name=str(p.get("name") or ""),
                jersey_number=p.get("jersey_number"),
            )

        def _resolve_player_id(
            jersey_norm: Optional[str],
            name_norm: str,
            name_norm_no_middle: str,
        ) -> Optional[int]:
            candidates: list[int] = []
            for tid in (team1_id, team2_id):
                if jersey_norm:
                    candidates.extend(jersey_to_player_ids.get((tid, jersey_norm), []))
            if len(set(candidates)) == 1:
                return int(list(set(candidates))[0])
            candidates = []
            for tid in (team1_id, team2_id):
                candidates.extend(name_to_player_ids.get((tid, name_norm), []))
            if len(set(candidates)) == 1:
                return int(list(set(candidates))[0])
            if name_norm_no_middle and name_norm_no_middle != name_norm:
                candidates = []
                for tid in (team1_id, team2_id):
                    candidates.extend(name_to_player_ids.get((tid, name_norm_no_middle), []))
                if len(set(candidates)) == 1:
                    return int(list(set(candidates))[0])
            return None

        def _boolish(raw: Any) -> bool:
            if isinstance(raw, bool):
                return bool(raw)
            s = str(raw or "").strip().casefold()
            if not s:
                return False
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
            return bool(raw)

        now = dt.datetime.now()
        with transaction.atomic():
            has_stats_payload = bool(
                (isinstance(events_csv, str) and events_csv.strip())
                or (isinstance(shift_rows_csv, str) and shift_rows_csv.strip())
            )
            if has_stats_payload and roster_player_ids_by_team:
                # Credit GP for roster players even when they have no shift-package stats/events.
                to_create_links = []
                for tid, pids in roster_player_ids_by_team.items():
                    for pid in sorted(pids):
                        to_create_links.append(
                            m.HkyGamePlayer(
                                game_id=int(resolved_game_id),
                                player_id=int(pid),
                                team_id=int(tid),
                                created_at=now,
                                updated_at=None,
                            )
                        )
                if to_create_links:
                    m.HkyGamePlayer.objects.bulk_create(to_create_links, ignore_conflicts=True)

            if isinstance(events_csv, str) and events_csv.strip():
                try:
                    has_existing = m.HkyGameEventRow.objects.filter(
                        game_id=int(resolved_game_id)
                    ).exists()
                    has_timetoscore = m.HkyGameEventRow.objects.filter(
                        game_id=int(resolved_game_id), source__icontains="timetoscore"
                    ).exists()

                    # Match legacy behavior: without replace, ignore follow-up shift-package uploads unless
                    # we have TimeToScore events to merge/augment.
                    should_upsert = True
                    replace_events = bool(replace)
                    if not replace_events and has_existing and not has_timetoscore:
                        should_upsert = False
                    # Avoid wiping authoritative TimeToScore events.
                    if replace_events and has_timetoscore:
                        replace_events = False

                    if should_upsert:
                        _upsert_game_event_rows_from_events_csv(
                            game_id=int(resolved_game_id),
                            events_csv=str(events_csv),
                            replace=bool(replace_events),
                            create_missing_players=bool(create_missing_players),
                            incoming_source_label="shift_package",
                        )
                except Exception:
                    logger.warning(
                        "api_import_shift_package: failed to upsert event rows (game_id=%s)",
                        int(resolved_game_id),
                        exc_info=True,
                    )
                    raise

            if (
                isinstance(shift_rows_csv, str)
                and shift_rows_csv.strip()
                and team_side in {"home", "away"}
            ):
                side_norm, side_label = _normalize_team_side(team_side)
                team_id_for_rows = team1_id if side_norm == "home" else team2_id
                source_label = (
                    str(payload.get("source_label") or "shift_package").strip() or "shift_package"
                )

                replace_shift_rows = (
                    bool(replace)
                    if replace_shift_rows_payload is None
                    else bool(_boolish(replace_shift_rows_payload))
                )

                parsed_shift_rows = logic.parse_shift_rows_csv(str(shift_rows_csv))

                if replace_shift_rows and side_label:
                    m.HkyGameShiftRow.objects.filter(
                        game_id=int(resolved_game_id),
                        team_side=str(side_label),
                    ).delete()

                to_create: list[Any] = []
                seen_keys: set[str] = set()
                for r0 in parsed_shift_rows:
                    period = r0.get("period")
                    game_s = r0.get("game_seconds")
                    game_s_end = r0.get("game_seconds_end")
                    if period is None or game_s is None or game_s_end is None:
                        continue
                    try:
                        per_i = int(period)
                        gs_i = int(game_s)
                        ge_i = int(game_s_end)
                    except Exception:
                        continue
                    if per_i <= 0:
                        continue

                    jersey_norm = r0.get("jersey_number")
                    name_norm = str(r0.get("name_norm") or "")
                    name_norm_no_middle = str(r0.get("name_norm_no_middle") or "")
                    pid = _resolve_player_id(jersey_norm, name_norm, name_norm_no_middle)
                    if pid is None:
                        if create_missing_players and team_side in {"home", "away"}:
                            disp = str(r0.get("player_label") or "").strip()
                            match = re.match(r"^\s*\d+\s+(.*)$", disp)
                            if match:
                                disp = str(match.group(1) or "").strip()
                            if disp:
                                try:
                                    pid = _ensure_player_for_import(
                                        owner_user_id,
                                        int(team_id_for_rows),
                                        disp,
                                        str(jersey_norm or "").strip() or None,
                                        None,
                                        commit=False,
                                    )
                                    _register_player(
                                        int(pid),
                                        int(team_id_for_rows),
                                        name=str(disp),
                                        jersey_number=jersey_norm,
                                    )
                                except Exception:
                                    pid = None
                        if pid is None:
                            unmatched.append(r0.get("player_label") or "")
                            continue

                    import_key = str(r0.get("import_key") or "").strip()
                    if not import_key:
                        base = "|".join(
                            [
                                str(side_label or ""),
                                str(jersey_norm or ""),
                                str(name_norm or ""),
                                str(per_i),
                                str(gs_i),
                                str(ge_i),
                                str(r0.get("video_seconds") or ""),
                                str(r0.get("video_seconds_end") or ""),
                            ]
                        )
                        import_key = hashlib.sha1(base.encode("utf-8")).hexdigest()[:40]
                    if len(import_key) > 64:
                        import_key = import_key[:64]
                    if import_key in seen_keys:
                        continue
                    seen_keys.add(import_key)

                    vs0 = r0.get("video_seconds")
                    vs1 = r0.get("video_seconds_end")
                    try:
                        vs_i = int(vs0) if vs0 is not None else None
                    except Exception:
                        vs_i = None
                    try:
                        ve_i = int(vs1) if vs1 is not None else None
                    except Exception:
                        ve_i = None

                    to_create.append(
                        m.HkyGameShiftRow(
                            game_id=int(resolved_game_id),
                            import_key=str(import_key),
                            source=str(r0.get("source") or source_label),
                            team_id=int(team_id_for_rows),
                            player_id=int(pid),
                            team_side=str(side_label or ""),
                            period=int(per_i),
                            game_seconds=int(gs_i),
                            game_seconds_end=int(ge_i),
                            video_seconds=vs_i,
                            video_seconds_end=ve_i,
                            created_at=now,
                            updated_at=None,
                        )
                    )

                if to_create:
                    m.HkyGameShiftRow.objects.bulk_create(to_create, ignore_conflicts=True)
                    imported_shifts += len(to_create)

            if events_csv or shift_rows_csv:
                m.HkyGame.objects.filter(id=int(resolved_game_id)).update(stats_imported_at=now)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return JsonResponse(
        {
            "ok": True,
            "game_id": int(resolved_game_id),
            "imported_players": int(imported),
            "imported_shifts": int(imported_shifts),
            "unmatched": [u for u in unmatched if u],
        }
    )


@csrf_exempt
def api_internal_reset_league_data(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "").strip()
    owner_email = str(payload.get("owner_email") or "").strip().lower()
    if not league_name or not owner_email:
        return JsonResponse(
            {"ok": False, "error": "owner_email and league_name are required"}, status=400
        )

    _django_orm, m = _orm_modules()
    owner_user_id = m.User.objects.filter(email=owner_email).values_list("id", flat=True).first()
    if owner_user_id is None:
        return JsonResponse({"ok": False, "error": "owner_email_not_found"}, status=404)
    league_id = (
        m.League.objects.filter(name=league_name, owner_user_id=int(owner_user_id))
        .values_list("id", flat=True)
        .first()
    )
    if league_id is None:
        return JsonResponse({"ok": False, "error": "league_not_found_for_owner"}, status=404)

    try:
        stats = logic.reset_league_data(None, int(league_id), owner_user_id=int(owner_user_id))
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"ok": False, "error": str(e)}, status=500)
    return JsonResponse({"ok": True, "league_id": int(league_id), "stats": stats})


@csrf_exempt
def api_internal_ensure_league_owner(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_name = str(payload.get("league_name") or "").strip()
    owner_email = str(payload.get("owner_email") or "").strip().lower()
    owner_name = str(payload.get("owner_name") or owner_email).strip() or owner_email
    is_shared = bool(payload["shared"]) if "shared" in payload else None
    is_public = None
    if "is_public" in payload:
        is_public = bool(payload["is_public"])
    elif "public" in payload:
        is_public = bool(payload["public"])
    if not league_name or not owner_email:
        return JsonResponse(
            {"ok": False, "error": "owner_email and league_name are required"}, status=400
        )

    owner_user_id = _ensure_user_for_import(owner_email, name=owner_name)
    _django_orm, m = _orm_modules()
    from django.db import transaction

    now = dt.datetime.now()
    with transaction.atomic():
        existing = m.League.objects.filter(name=league_name).values("id").first()
        if existing:
            league_id = int(existing["id"])
            updates: dict[str, Any] = {"owner_user_id": int(owner_user_id), "updated_at": now}
            if is_shared is not None:
                updates["is_shared"] = bool(is_shared)
            if is_public is not None:
                updates["is_public"] = bool(is_public)
            m.League.objects.filter(id=league_id).update(**updates)
        else:
            if is_shared is None:
                is_shared = True
            if is_public is None:
                is_public = False
            league = m.League.objects.create(
                name=league_name,
                owner_user_id=int(owner_user_id),
                is_shared=bool(is_shared),
                is_public=bool(is_public),
                created_at=now,
                updated_at=None,
            )
            league_id = int(league.id)

        member, created = m.LeagueMember.objects.get_or_create(
            league_id=int(league_id),
            user_id=int(owner_user_id),
            defaults={"role": "owner", "created_at": now},
        )
        if not created and str(getattr(member, "role", "") or "") != "owner":
            m.LeagueMember.objects.filter(id=int(member.id)).update(role="owner")

    return JsonResponse(
        {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id)}
    )


@csrf_exempt
def api_internal_ensure_user(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    email = str(payload.get("email") or payload.get("user_email") or "").strip().lower()
    name = str(payload.get("name") or payload.get("user_name") or email).strip() or email
    password = str(payload.get("password") or "password")
    if not email:
        return JsonResponse({"ok": False, "error": "email is required"}, status=400)
    _django_orm, m = _orm_modules()
    existing = m.User.objects.filter(email=email).values_list("id", flat=True).first()
    if existing is not None:
        return JsonResponse({"ok": True, "user_id": int(existing), "created": False})
    pwd_hash = generate_password_hash(password)
    now = dt.datetime.now()
    u = m.User.objects.create(
        email=email,
        password_hash=pwd_hash,
        name=name,
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    return JsonResponse({"ok": True, "user_id": int(u.id), "created": True})


@csrf_exempt
def api_internal_recalc_div_ratings(request: HttpRequest) -> JsonResponse:
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    league_id_raw = payload.get("league_id") or payload.get("lid") or request.GET.get("league_id")
    league_name = str(
        payload.get("league_name") or payload.get("name") or request.GET.get("league_name") or ""
    ).strip()
    max_goal_diff_raw = payload.get("max_goal_diff") or payload.get("maxGoalDiff") or None
    min_games_raw = payload.get("min_games") or payload.get("minGames") or None

    max_goal_diff = int(max_goal_diff_raw) if max_goal_diff_raw is not None else 7
    min_games = int(min_games_raw) if min_games_raw is not None else 2

    _django_orm, m = _orm_modules()

    league_ids: list[int]
    if league_id_raw:
        try:
            league_ids = [int(league_id_raw)]
        except Exception:
            return JsonResponse({"ok": False, "error": "invalid league_id"}, status=400)
    elif league_name:
        row = m.League.objects.filter(name=league_name).values("id").first()
        if not row:
            return JsonResponse({"ok": False, "error": "league_not_found"}, status=404)
        league_ids = [int(row["id"])]
    else:
        league_ids = list(m.League.objects.order_by("id").values_list("id", flat=True))

    ok_ids: list[int] = []
    failed: list[dict[str, Any]] = []
    for lid in league_ids:
        try:
            logic.recompute_league_mhr_ratings(
                None, int(lid), max_goal_diff=max_goal_diff, min_games=min_games
            )
            ok_ids.append(int(lid))
        except Exception as e:  # noqa: BLE001
            failed.append({"league_id": int(lid), "error": str(e)})

    if failed:
        return JsonResponse({"ok": False, "league_ids_ok": ok_ids, "failed": failed}, status=500)
    return JsonResponse({"ok": True, "league_ids": ok_ids})


@csrf_exempt
def api_internal_apply_event_corrections(request: HttpRequest) -> JsonResponse:
    """
    Persist idempotent event corrections:
      - `suppress`: hide an imported event key for a game (prevents re-import from re-adding it)
      - `upsert`: insert/update an event row (typically paired with suppress of the original)
      - `patch`: suppress + upsert in one step, with stored diff metadata for UI

    Payload:
      {
        "corrections": [
          {
            "game_id": 1001 | null,
            "timetoscore_game_id": 123 | null,
            "external_game_key": "utah-1" | null,
            "suppress": [ {event spec...}, ... ],
            "upsert": [ {event spec...}, ... ],
            "patch": [
              {
                "match": {event spec...},
                "set": {event spec overrides...},
                "note": "see video"
              }
            ]
          }
        ]
      }
    """
    auth = _require_import_auth(request)
    if auth:
        return auth
    payload = _json_body(request)
    corrections = payload.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        return JsonResponse(
            {"ok": False, "error": "corrections must be a non-empty list"}, status=400
        )

    create_missing_players = bool(payload.get("create_missing_players", False))

    _django_orm, m = _orm_modules()
    from django.db import transaction

    def _resolve_game_id(c: dict[str, Any]) -> Optional[int]:
        gid_raw = c.get("game_id")
        if gid_raw is not None and str(gid_raw).strip():
            try:
                return int(gid_raw)
            except Exception:
                return None
        tts_raw = c.get("timetoscore_game_id") or c.get("tts_game_id")
        if tts_raw is not None and str(tts_raw).strip():
            try:
                tts_i = int(tts_raw)
            except Exception:
                tts_i = None
            if tts_i is not None:
                gid = (
                    m.HkyGame.objects.filter(timetoscore_game_id=int(tts_i))
                    .values_list("id", flat=True)
                    .first()
                )
                if gid is not None:
                    return int(gid)
                token = f'"timetoscore_game_id":{int(tts_i)}'
                gid = (
                    m.HkyGame.objects.filter(notes__contains=token)
                    .values_list("id", flat=True)
                    .first()
                )
                if gid is not None:
                    return int(gid)

        ext_key_raw = (
            c.get("external_game_key")
            or c.get("external_key")
            or c.get("external_game_id")
            or c.get("label")
        )
        ext_key = str(ext_key_raw or "").strip() or None
        if not ext_key:
            return None

        owner_user_id: Optional[int] = None
        owner_user_id_raw = c.get("owner_user_id") or payload.get("owner_user_id")
        if owner_user_id_raw is not None and str(owner_user_id_raw).strip():
            try:
                owner_user_id = int(owner_user_id_raw)
            except Exception:
                owner_user_id = None
        owner_email = (
            str(c.get("owner_email") or payload.get("owner_email") or "").strip().lower() or None
        )
        if owner_user_id is None and owner_email:
            owner_user_id = (
                m.User.objects.filter(email=str(owner_email)).values_list("id", flat=True).first()
            )
            owner_user_id = int(owner_user_id) if owner_user_id is not None else None

        qs = m.HkyGame.objects.filter(external_game_key=str(ext_key))
        if owner_user_id is not None:
            qs = qs.filter(user_id=int(owner_user_id))

        ids = list(qs.values_list("id", flat=True)[:2])
        if len(ids) == 1 and ids[0] is not None:
            return int(ids[0])
        return None

    def _norm_side(raw: Any) -> tuple[Optional[str], Optional[str]]:
        side_norm, side_label = _normalize_team_side(raw)
        if side_norm in {"home", "away"}:
            return side_norm, side_label
        return None, side_label

    def _event_spec_import_key(ev: dict[str, Any]) -> tuple[str, Optional[str], Optional[str]]:
        et_raw = str(ev.get("event_type") or ev.get("event") or "").strip()
        et_key = _event_type_key(et_raw)
        if not et_key:
            raise ValueError("event_type is required")
        period = _ival(ev.get("period"))
        game_seconds = _ival(ev.get("game_seconds"))
        if game_seconds is None:
            game_seconds = logic.parse_duration_seconds(ev.get("game_time") or ev.get("time"))
        game_seconds_end = _ival(ev.get("game_seconds_end"))
        side_norm, side_label = _norm_side(ev.get("team_side") or ev.get("side") or ev.get("team"))
        jersey_norm = logic.normalize_jersey_number(
            ev.get("jersey")
            or ev.get("attributed_jerseys")
            or ev.get("player")
            or ev.get("attributed_players")
        )
        details = _norm_ws(ev.get("details"))
        event_id = _ival(ev.get("event_id"))
        return (
            _compute_event_import_key(
                event_type_key=et_key,
                period=period,
                game_seconds=game_seconds,
                team_side_norm=side_norm,
                jersey_norm=jersey_norm,
                event_id=event_id,
                details=details,
                game_seconds_end=game_seconds_end,
            ),
            side_norm,
            side_label,
        )

    stats = {"suppressed": 0, "unsuppressed": 0, "upserted": 0, "deleted_existing": 0}
    now = dt.datetime.now()

    def _event_spec_norm_for_diff(ev: dict[str, Any]) -> dict[str, Any]:
        et_raw = str(ev.get("event_type") or ev.get("event") or "").strip()
        et_key = _event_type_key(et_raw) or ""
        period = _ival(ev.get("period"))
        game_time = _norm_ws(ev.get("game_time") or ev.get("time")) or ""
        game_seconds = _ival(ev.get("game_seconds"))
        if game_seconds is None:
            game_seconds = logic.parse_duration_seconds(game_time)
        side_norm, side_label = _norm_side(ev.get("team_side") or ev.get("side") or ev.get("team"))
        jersey_norm = logic.normalize_jersey_number(
            ev.get("jersey")
            or ev.get("attributed_jerseys")
            or ev.get("player")
            or ev.get("attributed_players")
        )
        video_time, video_seconds = logic.normalize_video_time_and_seconds(
            _norm_ws(ev.get("video_time")), ev.get("video_seconds")
        )
        return {
            "event_type": et_raw,
            "event_type_key": et_key,
            "period": int(period) if period is not None else None,
            "team_side": side_label,
            "team_side_norm": side_norm,
            "game_time": game_time,
            "game_seconds": int(game_seconds) if game_seconds is not None else None,
            "jersey": jersey_norm,
            "player_name": str(
                ev.get("attributed_players") or ev.get("player_name") or ev.get("player") or ""
            ).strip()
            or None,
            "details": _norm_ws(ev.get("details")) or None,
            "event_id": _ival(ev.get("event_id")),
            "video_time": video_time,
            "video_seconds": int(video_seconds) if video_seconds is not None else None,
        }

    def _correction_json(
        *,
        match_ev: dict[str, Any],
        upsert_ev: dict[str, Any],
        note: Optional[str],
        reason: Optional[str],
    ) -> str:
        before = _event_spec_norm_for_diff(match_ev)
        after = _event_spec_norm_for_diff(upsert_ev)
        fields = [
            "event_type",
            "team_side",
            "period",
            "game_time",
            "jersey",
            "player_name",
            "details",
            "video_time",
        ]
        changes: list[dict[str, Any]] = []
        for f in fields:
            if before.get(f) != after.get(f):
                changes.append({"field": f, "from": before.get(f), "to": after.get(f)})
        try:
            import_key_before, _sn1, _sl1 = _event_spec_import_key(match_ev)
        except Exception:
            import_key_before = ""
        try:
            import_key_after, _sn2, _sl2 = _event_spec_import_key(upsert_ev)
        except Exception:
            import_key_after = ""
        return json.dumps(
            {
                "version": 1,
                "note": str(note).strip() if note and str(note).strip() else None,
                "reason": str(reason).strip() if reason and str(reason).strip() else None,
                "changes": changes,
                "match": before,
                "set": after,
                "match_import_key": import_key_before or None,
                "upsert_import_key": import_key_after or None,
            },
            sort_keys=True,
        )

    with transaction.atomic():
        for idx, c in enumerate(corrections):
            if not isinstance(c, dict):
                return JsonResponse(
                    {"ok": False, "error": f"corrections[{idx}] must be an object"}, status=400
                )
            gid = _resolve_game_id(c)
            if gid is None:
                return JsonResponse(
                    {
                        "ok": False,
                        "error": f"corrections[{idx}] missing/invalid game_id or timetoscore_game_id",
                    },
                    status=400,
                )

            game_row = (
                m.HkyGame.objects.filter(id=int(gid))
                .values("id", "team1_id", "team2_id", "user_id")
                .first()
            )
            if not game_row:
                return JsonResponse({"ok": False, "error": f"game_not_found: {gid}"}, status=404)

            team1_id = int(game_row["team1_id"])
            team2_id = int(game_row["team2_id"])
            owner_user_id = int(game_row.get("user_id") or 0)

            patch_list = c.get("patch") or c.get("patches") or []
            if patch_list:
                if not isinstance(patch_list, list):
                    return JsonResponse(
                        {"ok": False, "error": f"corrections[{idx}].patch must be a list"},
                        status=400,
                    )
                for pidx, patch in enumerate(patch_list):
                    if not isinstance(patch, dict):
                        continue
                    match_ev = patch.get("match") or patch.get("from") or patch.get("old")
                    set_ev = patch.get("set") or patch.get("to") or patch.get("new") or {}
                    if not isinstance(match_ev, dict) or not isinstance(set_ev, dict):
                        return JsonResponse(
                            {
                                "ok": False,
                                "error": f"corrections[{idx}].patch[{pidx}] must contain match/set objects",
                            },
                            status=400,
                        )

                    # Build upsert event by applying overrides to the match spec.
                    upsert_ev: dict[str, Any] = dict(match_ev)
                    upsert_ev.update(dict(set_ev))

                    reason = str(patch.get("reason") or c.get("reason") or "").strip() or None
                    note = str(patch.get("note") or c.get("note") or "").strip() or None

                    try:
                        import_key_match, _sn, _sl = _event_spec_import_key(match_ev)
                    except Exception as e:
                        return JsonResponse(
                            {
                                "ok": False,
                                "error": f"corrections[{idx}].patch[{pidx}] invalid match spec: {e}",
                            },
                            status=400,
                        )
                    try:
                        import_key_upsert, side_norm, side_label = _event_spec_import_key(upsert_ev)
                    except Exception as e:
                        return JsonResponse(
                            {
                                "ok": False,
                                "error": f"corrections[{idx}].patch[{pidx}] invalid set spec: {e}",
                            },
                            status=400,
                        )

                    match_exists = m.HkyGameEventRow.objects.filter(
                        game_id=int(gid), import_key=str(import_key_match)
                    ).exists()
                    if not match_exists:
                        # Idempotency: once applied, the matched row may be deleted while the import_key remains
                        # suppressed and the corrected upsert row exists.
                        already_suppressed = m.HkyGameEventSuppression.objects.filter(
                            game_id=int(gid), import_key=str(import_key_match)
                        ).exists()
                        already_upserted = m.HkyGameEventRow.objects.filter(
                            game_id=int(gid), import_key=str(import_key_upsert)
                        ).exists()
                        if not (already_suppressed and already_upserted):
                            return JsonResponse(
                                {
                                    "ok": False,
                                    "error": (
                                        f"corrections[{idx}].patch[{pidx}] match did not match any event "
                                        f"(import_key={import_key_match})"
                                    ),
                                },
                                status=400,
                            )

                    # If the patch doesn't change the event import_key (notably: goal scorer corrections),
                    # update the existing row in-place (do not suppress; suppressed keys are excluded from views).
                    if str(import_key_match) != str(import_key_upsert):
                        obj, created = m.HkyGameEventSuppression.objects.get_or_create(
                            game_id=int(gid),
                            import_key=str(import_key_match),
                            defaults={"reason": reason, "created_at": now, "updated_at": None},
                        )
                        if (
                            not created
                            and reason
                            and str(getattr(obj, "reason", "") or "") != reason
                        ):
                            m.HkyGameEventSuppression.objects.filter(id=int(obj.id)).update(
                                reason=reason, updated_at=now
                            )
                        stats["suppressed"] += 1
                        stats["deleted_existing"] += m.HkyGameEventRow.objects.filter(
                            game_id=int(gid), import_key=str(import_key_match)
                        ).delete()[0]
                    else:
                        m.HkyGameEventSuppression.objects.filter(
                            game_id=int(gid), import_key=str(import_key_match)
                        ).delete()

                    # 2) Upsert the corrected event.
                    et_raw = str(
                        upsert_ev.get("event_type") or upsert_ev.get("event") or ""
                    ).strip()
                    et_key = _event_type_key(et_raw)
                    if not et_key:
                        continue
                    et_obj, _ = m.HkyEventType.objects.get_or_create(
                        key=str(et_key),
                        defaults={"name": et_raw or et_key, "created_at": now},
                    )

                    team_id = upsert_ev.get("team_id")
                    if team_id is not None and str(team_id).strip():
                        try:
                            team_id = int(team_id)
                        except Exception:
                            team_id = None
                    elif side_norm == "home":
                        team_id = team1_id
                    elif side_norm == "away":
                        team_id = team2_id
                    else:
                        team_id = None

                    player_id = upsert_ev.get("player_id")
                    if player_id is not None and str(player_id).strip():
                        try:
                            player_id = int(player_id)
                        except Exception:
                            player_id = None
                    else:
                        jersey_norm = logic.normalize_jersey_number(
                            upsert_ev.get("jersey")
                            or upsert_ev.get("attributed_jerseys")
                            or upsert_ev.get("player")
                        )
                        player_name = str(
                            upsert_ev.get("attributed_players")
                            or upsert_ev.get("player_name")
                            or ""
                        ).strip()
                        if team_id is not None and jersey_norm:
                            player_id = (
                                m.Player.objects.filter(
                                    team_id=int(team_id), jersey_number=str(jersey_norm)
                                )
                                .values_list("id", flat=True)
                                .first()
                            )
                            player_id = int(player_id) if player_id is not None else None
                        if player_id is None and team_id is not None and player_name:
                            player_id = (
                                m.Player.objects.filter(team_id=int(team_id), name=str(player_name))
                                .values_list("id", flat=True)
                                .first()
                            )
                            player_id = int(player_id) if player_id is not None else None
                        if (
                            player_id is None
                            and create_missing_players
                            and team_id is not None
                            and player_name
                        ):
                            try:
                                player_id = _ensure_player_for_import(
                                    int(owner_user_id),
                                    int(team_id),
                                    str(player_name),
                                    jersey_norm,
                                    None,
                                    commit=False,
                                )
                            except Exception:
                                player_id = None

                    period = _ival(upsert_ev.get("period"))
                    game_time = _norm_ws(upsert_ev.get("game_time") or upsert_ev.get("time"))
                    game_seconds = _ival(upsert_ev.get("game_seconds"))
                    if game_seconds is None:
                        game_seconds = logic.parse_duration_seconds(game_time)
                    game_seconds_end = _ival(upsert_ev.get("game_seconds_end"))
                    video_time, video_seconds = logic.normalize_video_time_and_seconds(
                        _norm_ws(upsert_ev.get("video_time")), upsert_ev.get("video_seconds")
                    )

                    details = _norm_ws(upsert_ev.get("details")) or None
                    attributed_players = _norm_ws(upsert_ev.get("attributed_players")) or None
                    attributed_jerseys = (
                        _norm_ws(
                            upsert_ev.get("attributed_jerseys")
                            or upsert_ev.get("player")
                            or upsert_ev.get("jersey")
                        )
                        or None
                    )
                    source = (
                        str(upsert_ev.get("source") or patch.get("source") or "correction").strip()
                        or "correction"
                    )

                    corr_json = _correction_json(
                        match_ev=match_ev, upsert_ev=upsert_ev, note=note, reason=reason
                    )

                    m.HkyGameEventRow.objects.update_or_create(
                        game_id=int(gid),
                        import_key=str(import_key_upsert),
                        defaults={
                            "event_type_id": int(et_obj.id),
                            "team_id": int(team_id) if team_id is not None else None,
                            "player_id": int(player_id) if player_id is not None else None,
                            "source": source,
                            "event_id": _ival(upsert_ev.get("event_id")),
                            "team_raw": _norm_ws(upsert_ev.get("team_raw") or upsert_ev.get("team"))
                            or None,
                            "team_side": side_label
                            or _norm_ws(upsert_ev.get("team_side") or upsert_ev.get("side"))
                            or None,
                            "for_against": _norm_ws(upsert_ev.get("for_against")) or None,
                            "team_rel": _norm_ws(upsert_ev.get("team_rel")) or None,
                            "period": int(period) if period is not None else None,
                            "game_time": game_time or None,
                            "video_time": video_time or None,
                            "game_seconds": int(game_seconds) if game_seconds is not None else None,
                            "game_seconds_end": (
                                int(game_seconds_end) if game_seconds_end is not None else None
                            ),
                            "video_seconds": (
                                int(video_seconds) if video_seconds is not None else None
                            ),
                            "details": details,
                            "correction": corr_json,
                            "attributed_players": attributed_players,
                            "attributed_jerseys": attributed_jerseys,
                            "on_ice_players": _norm_ws(upsert_ev.get("on_ice_players")) or None,
                            "on_ice_players_home": _norm_ws(upsert_ev.get("on_ice_players_home"))
                            or None,
                            "on_ice_players_away": _norm_ws(upsert_ev.get("on_ice_players_away"))
                            or None,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    stats["upserted"] += 1

                    m.HkyGameEventSuppression.objects.filter(
                        game_id=int(gid), import_key=str(import_key_upsert)
                    ).delete()

                    if player_id is not None and team_id is not None:
                        m.HkyGamePlayer.objects.get_or_create(
                            game_id=int(gid),
                            player_id=int(player_id),
                            defaults={
                                "team_id": int(team_id),
                                "created_at": now,
                                "updated_at": None,
                            },
                        )

            suppress_list = c.get("suppress") or []
            if suppress_list:
                if not isinstance(suppress_list, list):
                    return JsonResponse(
                        {"ok": False, "error": f"corrections[{idx}].suppress must be a list"},
                        status=400,
                    )
                for ev in suppress_list:
                    if not isinstance(ev, dict):
                        continue
                    import_key, _side_norm, _side_label = _event_spec_import_key(ev)
                    reason = str(ev.get("reason") or c.get("reason") or "").strip() or None
                    obj, created = m.HkyGameEventSuppression.objects.get_or_create(
                        game_id=int(gid),
                        import_key=str(import_key),
                        defaults={"reason": reason, "created_at": now, "updated_at": None},
                    )
                    if not created and reason and str(getattr(obj, "reason", "") or "") != reason:
                        m.HkyGameEventSuppression.objects.filter(id=int(obj.id)).update(
                            reason=reason, updated_at=now
                        )
                    stats["suppressed"] += 1
                    stats["deleted_existing"] += m.HkyGameEventRow.objects.filter(
                        game_id=int(gid), import_key=str(import_key)
                    ).delete()[0]

            upsert_list = c.get("upsert") or []
            if upsert_list:
                if not isinstance(upsert_list, list):
                    return JsonResponse(
                        {"ok": False, "error": f"corrections[{idx}].upsert must be a list"},
                        status=400,
                    )
                for ev in upsert_list:
                    if not isinstance(ev, dict):
                        continue
                    et_raw = str(ev.get("event_type") or ev.get("event") or "").strip()
                    et_key = _event_type_key(et_raw)
                    if not et_key:
                        continue
                    et_obj, _ = m.HkyEventType.objects.get_or_create(
                        key=str(et_key),
                        defaults={"name": et_raw or et_key, "created_at": now},
                    )
                    import_key, side_norm, side_label = _event_spec_import_key(ev)

                    team_id = ev.get("team_id")
                    if team_id is not None and str(team_id).strip():
                        try:
                            team_id = int(team_id)
                        except Exception:
                            team_id = None
                    elif side_norm == "home":
                        team_id = team1_id
                    elif side_norm == "away":
                        team_id = team2_id
                    else:
                        team_id = None

                    player_id = ev.get("player_id")
                    if player_id is not None and str(player_id).strip():
                        try:
                            player_id = int(player_id)
                        except Exception:
                            player_id = None
                    else:
                        jersey_norm = logic.normalize_jersey_number(
                            ev.get("jersey") or ev.get("attributed_jerseys") or ev.get("player")
                        )
                        player_name = str(
                            ev.get("attributed_players") or ev.get("player_name") or ""
                        ).strip()
                        if team_id is not None and jersey_norm:
                            player_id = (
                                m.Player.objects.filter(
                                    team_id=int(team_id), jersey_number=str(jersey_norm)
                                )
                                .values_list("id", flat=True)
                                .first()
                            )
                            player_id = int(player_id) if player_id is not None else None
                        if player_id is None and team_id is not None and player_name:
                            player_id = (
                                m.Player.objects.filter(team_id=int(team_id), name=str(player_name))
                                .values_list("id", flat=True)
                                .first()
                            )
                            player_id = int(player_id) if player_id is not None else None
                        if (
                            player_id is None
                            and create_missing_players
                            and team_id is not None
                            and player_name
                        ):
                            try:
                                player_id = _ensure_player_for_import(
                                    int(owner_user_id),
                                    int(team_id),
                                    str(player_name),
                                    jersey_norm,
                                    None,
                                    commit=False,
                                )
                            except Exception:
                                player_id = None

                    period = _ival(ev.get("period"))
                    game_time = _norm_ws(ev.get("game_time") or ev.get("time"))
                    game_seconds = _ival(ev.get("game_seconds"))
                    if game_seconds is None:
                        game_seconds = logic.parse_duration_seconds(game_time)
                    game_seconds_end = _ival(ev.get("game_seconds_end"))
                    video_time, video_seconds = logic.normalize_video_time_and_seconds(
                        _norm_ws(ev.get("video_time")), ev.get("video_seconds")
                    )

                    details = _norm_ws(ev.get("details")) or None
                    attributed_players = _norm_ws(ev.get("attributed_players")) or None
                    attributed_jerseys = (
                        _norm_ws(
                            ev.get("attributed_jerseys") or ev.get("player") or ev.get("jersey")
                        )
                        or None
                    )
                    source = str(ev.get("source") or "correction").strip() or "correction"

                    m.HkyGameEventRow.objects.update_or_create(
                        game_id=int(gid),
                        import_key=str(import_key),
                        defaults={
                            "event_type_id": int(et_obj.id),
                            "team_id": int(team_id) if team_id is not None else None,
                            "player_id": int(player_id) if player_id is not None else None,
                            "source": source,
                            "event_id": _ival(ev.get("event_id")),
                            "team_raw": _norm_ws(ev.get("team_raw") or ev.get("team")) or None,
                            "team_side": side_label
                            or _norm_ws(ev.get("team_side") or ev.get("side"))
                            or None,
                            "for_against": _norm_ws(ev.get("for_against")) or None,
                            "team_rel": _norm_ws(ev.get("team_rel")) or None,
                            "period": int(period) if period is not None else None,
                            "game_time": game_time or None,
                            "video_time": video_time or None,
                            "game_seconds": int(game_seconds) if game_seconds is not None else None,
                            "game_seconds_end": (
                                int(game_seconds_end) if game_seconds_end is not None else None
                            ),
                            "video_seconds": (
                                int(video_seconds) if video_seconds is not None else None
                            ),
                            "details": details,
                            "correction": None,
                            "attributed_players": attributed_players,
                            "attributed_jerseys": attributed_jerseys,
                            "on_ice_players": _norm_ws(ev.get("on_ice_players")) or None,
                            "on_ice_players_home": _norm_ws(ev.get("on_ice_players_home")) or None,
                            "on_ice_players_away": _norm_ws(ev.get("on_ice_players_away")) or None,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    stats["upserted"] += 1

                    m.HkyGameEventSuppression.objects.filter(
                        game_id=int(gid), import_key=str(import_key)
                    ).delete()

                    if player_id is not None and team_id is not None:
                        m.HkyGamePlayer.objects.get_or_create(
                            game_id=int(gid),
                            player_id=int(player_id),
                            defaults={
                                "team_id": int(team_id),
                                "created_at": now,
                                "updated_at": None,
                            },
                        )

    return JsonResponse({"ok": True, "stats": stats})

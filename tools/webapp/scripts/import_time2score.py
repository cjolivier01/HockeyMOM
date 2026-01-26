#!/usr/bin/env python3
"""Import TimeToScore data into the HockeyMOM webapp DB (optionally using a local sqlite cache).

This script scrapes TimeToScore directly via `hmlib.time2score` and upserts:
- teams (as external teams owned by the specified user)
- games (hky_games)
- players + per-row game events (hky_game_event_rows)

By default it targets CAHA youth (TimeToScore `league=3`). SharksIce (adult, league=1) is also supported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import base64
import csv
from functools import lru_cache
import io
import json
import os
import random
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from contextlib import contextmanager


@lru_cache(maxsize=1)
def _orm_modules():
    try:
        from tools.webapp import django_orm  # type: ignore
    except Exception:  # pragma: no cover
        import django_orm  # type: ignore

    django_orm.setup_django()
    django_orm.ensure_schema()
    django_orm.ensure_bootstrap_data()

    try:
        from tools.webapp.django_app import models as m  # type: ignore
    except Exception:  # pragma: no cover
        from django_app import models as m  # type: ignore

    return django_orm, m


def _parse_period_token(val: Any) -> Optional[int]:
    s = str(val or "").strip()
    if not s:
        return None
    sl = s.casefold()
    if sl in {"ot", "overtime"}:
        return 4
    m = re.search(r"(\d+)", sl)
    if not m:
        return None
    try:
        n = int(m.group(1))
        return n if n > 0 else None
    except Exception:
        return None


def _parse_mmss_to_seconds(val: Any, *, period_len_s: Optional[int] = None) -> Optional[int]:
    s = str(val or "").strip()
    if not s:
        return None
    m = re.match(r"^\s*(\d+):(\d{2})\s*$", s)
    if not m:
        # Some TimeToScore pages use '.' instead of ':' (e.g. "10.0" meaning "10:00").
        m = re.match(r"^\s*(\d+)[.](\d{1,2})\s*$", s)
        if not m:
            return None
        try:
            a = int(m.group(1))
            b = int(m.group(2))
        except Exception:
            return None

        # Disambiguate:
        #   - "10.0" => 10:00 (mm:ss)
        #   - "54.8" (in a 15:00 period) => 54.8 seconds (ss.d)
        # Heuristic: if the "minutes" part exceeds the period length in minutes, treat it as seconds.
        try:
            if period_len_s is not None and a > max(1, int(period_len_s) // 60):
                return int(float(s))
        except Exception:
            pass
        return a * 60 + b

    try:
        return int(m.group(1)) * 60 + int(m.group(2))
    except Exception:
        return None


def build_timetoscore_goal_and_assist_events(
    *,
    stats: dict[str, Any],
    period_len_s: int,
    num_to_name_home: dict[str, str],
    num_to_name_away: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build Goal + Assist event rows from TimeToScore scoring tables.

    Assists share the goal's (period, game time/seconds) so video mappings can be propagated
    across events at the same instant.
    """

    def _side_label(side: str) -> str:
        return "Home" if str(side).strip().lower() == "home" else "Away"

    def _norm_jersey(val: Any) -> Optional[str]:
        s = str(val or "").strip()
        if not s:
            return None
        m = re.search(r"(\d+)", s)
        return m.group(1) if m else None

    def _cell_text(raw: Any) -> str:
        if isinstance(raw, dict):
            txt = str(raw.get("text") or "").strip()
            if txt:
                return txt
        return str(raw or "").strip()

    def _resolve_player_name(
        raw: Any, jersey_num: Optional[str], roster_map: dict[str, str]
    ) -> str:
        if jersey_num and jersey_num in roster_map:
            return str(roster_map.get(jersey_num) or "").strip()
        return _cell_text(raw)

    goal_events: list[dict[str, Any]] = []
    assist_events: list[dict[str, Any]] = []

    for side_key, roster_map in (("home", num_to_name_home), ("away", num_to_name_away)):
        scoring = stats.get(f"{side_key}Scoring") or []
        if not isinstance(scoring, list):
            continue
        for srow in scoring:
            if not isinstance(srow, dict):
                continue
            per = _parse_period_token(srow.get("period"))
            if per is None:
                continue
            time_txt = str(srow.get("time") or "").strip()
            time_s = _parse_mmss_to_seconds(time_txt, period_len_s=period_len_s)
            scorer_raw = srow.get("goal")
            a1_raw = srow.get("assist1")
            a2_raw = srow.get("assist2")

            scorer_num = _norm_jersey(scorer_raw)
            scorer_name = _resolve_player_name(scorer_raw, scorer_num, roster_map)

            a1_num = _norm_jersey(a1_raw)
            a2_num = _norm_jersey(a2_raw)
            a1_name = _resolve_player_name(a1_raw, a1_num, roster_map)
            a2_name = _resolve_player_name(a2_raw, a2_num, roster_map)

            assists_txt = ", ".join([x for x in [a1_name, a2_name] if x])
            details = f"{scorer_name}" + (f" (A: {assists_txt})" if assists_txt else "")

            goal_events.append(
                {
                    "Event Type": "Goal",
                    "Source": "timetoscore",
                    "Team Side": _side_label(side_key),
                    "For/Against": "For",
                    "Team Rel": _side_label(side_key),
                    "Team Raw": _side_label(side_key),
                    "Period": int(per),
                    "Game Time": time_txt,
                    "Game Seconds": time_s if time_s is not None else "",
                    "Game Seconds End": "",
                    "Details": details,
                    "Attributed Players": scorer_name if scorer_name else "",
                    "Attributed Jerseys": scorer_num or "",
                }
            )

            for a_num, a_name in ((a1_num, a1_name), (a2_num, a2_name)):
                if not a_name:
                    continue
                a_details = f"Assist: {a_name}"
                if scorer_name:
                    a_details += f" (Goal: {scorer_name})"
                assist_events.append(
                    {
                        "Event Type": "Assist",
                        "Source": "timetoscore",
                        "Team Side": _side_label(side_key),
                        "For/Against": "For",
                        "Team Rel": _side_label(side_key),
                        "Team Raw": _side_label(side_key),
                        "Period": int(per),
                        "Game Time": time_txt,
                        "Game Seconds": time_s if time_s is not None else "",
                        "Game Seconds End": "",
                        "Details": a_details,
                        "Attributed Players": a_name,
                        "Attributed Jerseys": a_num or "",
                    }
                )

    return goal_events, assist_events


def _format_mmss(seconds: int) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _within_to_elapsed(within_s: int, *, mode: str, period_len_s: int) -> int:
    m = str(mode or "").strip().lower()
    if m == "remaining":
        return max(0, int(period_len_s) - int(within_s))
    return max(0, int(within_s))


def _infer_end_period_for_within_times(
    *,
    start_period: int,
    start_within_s: Optional[int],
    end_within_s: Optional[int],
    mode: str,
    period_len_s: int,
    max_advance_periods: int = 3,
) -> int:
    """
    TimeToScore penalty rows sometimes include an end time that is in the *next* period without
    bumping the period number (e.g. P1 1:00 -> end 14:00 meaning P2 14:00).

    Given a start period/time and end time (both in the same within-period time mode), infer the
    smallest end period >= start_period such that end occurs after start in absolute time.
    """
    per0 = int(start_period)
    if start_within_s is None or end_within_s is None:
        return per0

    start_elapsed = _within_to_elapsed(int(start_within_s), mode=mode, period_len_s=period_len_s)
    end_elapsed = _within_to_elapsed(int(end_within_s), mode=mode, period_len_s=period_len_s)
    start_abs = (per0 - 1) * int(period_len_s) + int(start_elapsed)

    per_end = per0
    end_abs = (per_end - 1) * int(period_len_s) + int(end_elapsed)
    while end_abs + 0.5 < start_abs and per_end < per0 + max_advance_periods:
        per_end += 1
        end_abs = (per_end - 1) * int(period_len_s) + int(end_elapsed)
    return int(per_end)


def _compute_end_from_start_and_delta(
    *,
    start_period: int,
    start_within_s: int,
    delta_s: int,
    mode: str,
    period_len_s: int,
    max_advance_periods: int = 3,
) -> tuple[int, int]:
    """
    Compute a penalty end time from start within-period time + duration, allowing it to wrap to
    subsequent periods (rather than clamping at 0 or period_len_s).
    """
    per0 = int(start_period)
    start_w = max(0, int(start_within_s))
    delta = max(0, int(delta_s))
    m = str(mode or "").strip().lower()

    if m == "remaining":
        remaining = int(start_w) - int(delta)
        per_end = per0
        while remaining < 0 and per_end < per0 + max_advance_periods:
            per_end += 1
            remaining = int(period_len_s) + int(remaining)
        remaining = max(0, min(int(period_len_s), int(remaining)))
        return int(per_end), int(remaining)

    elapsed = int(start_w) + int(delta)
    per_end = per0
    while elapsed > int(period_len_s) and per_end < per0 + max_advance_periods:
        elapsed -= int(period_len_s)
        per_end += 1
    elapsed = max(0, min(int(period_len_s), int(elapsed)))
    return int(per_end), int(elapsed)


def _to_csv_text(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not headers:
        return ""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows or []:
        w.writerow({h: ("" if r.get(h) is None else str(r.get(h))) for h in headers})
    return out.getvalue()


@contextmanager
def _working_directory(path: Path):
    prev = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def ensure_defaults(conn) -> None:
    del conn
    try:
        django_orm, _m = _orm_modules()
        django_orm.ensure_bootstrap_data()
    except Exception:
        return


def ensure_league_schema(conn) -> None:
    del conn
    try:
        django_orm, _m = _orm_modules()
        django_orm.ensure_schema()
    except Exception:
        return


def ensure_league(
    conn,
    name: str,
    owner_user_id: int,
    is_shared: Optional[bool],
    source: Optional[str],
    external_key: Optional[str],
) -> int:
    del conn
    ensure_league_schema(None)
    _django_orm, m = _orm_modules()
    from django.db import transaction

    now = dt.datetime.now()
    with transaction.atomic():
        league = m.League.objects.filter(name=str(name)).first()
        if league:
            updates: dict[str, Any] = {
                "source": source,
                "external_key": external_key,
                "updated_at": now,
            }
            if is_shared is not None:
                updates["is_shared"] = bool(is_shared)
            m.League.objects.filter(id=league.id).update(**updates)
            return int(league.id)

        if is_shared is None:
            # Default for TimeToScore imports: shared unless explicitly disabled.
            is_shared = True
        league = m.League.objects.create(
            name=str(name),
            owner_user_id=int(owner_user_id),
            is_shared=bool(is_shared),
            source=source,
            external_key=external_key,
            created_at=now,
        )
        return int(league.id)


def ensure_league_member(conn, league_id: int, user_id: int, role: str = "viewer") -> None:
    del conn
    _django_orm, m = _orm_modules()
    now = dt.datetime.now()
    lm, created = m.LeagueMember.objects.get_or_create(
        league_id=int(league_id),
        user_id=int(user_id),
        defaults={"role": str(role or "viewer"), "created_at": now},
    )
    if not created and role and str(lm.role or "") != str(role):
        m.LeagueMember.objects.filter(id=lm.id).update(role=str(role))


def map_team_to_league(conn, league_id: int, team_id: int) -> None:
    del conn
    _django_orm, m = _orm_modules()
    m.LeagueTeam.objects.get_or_create(league_id=int(league_id), team_id=int(team_id))


def map_team_to_league_with_division(
    conn,
    *,
    league_id: int,
    team_id: int,
    division_name: Optional[str],
    division_id: Optional[int],
    conference_id: Optional[int],
) -> None:
    dn = (division_name or "").strip() or None
    del conn
    _django_orm, m = _orm_modules()
    lt, created = m.LeagueTeam.objects.get_or_create(
        league_id=int(league_id),
        team_id=int(team_id),
        defaults={"division_name": dn, "division_id": division_id, "conference_id": conference_id},
    )
    if created:
        return
    updates: dict[str, Any] = {}
    if dn is not None:
        updates["division_name"] = dn
    if division_id is not None:
        updates["division_id"] = int(division_id)
    if conference_id is not None:
        updates["conference_id"] = int(conference_id)
    if updates:
        m.LeagueTeam.objects.filter(id=lt.id).update(**updates)


def map_game_to_league(conn, league_id: int, game_id: int) -> None:
    del conn
    _django_orm, m = _orm_modules()
    m.LeagueGame.objects.get_or_create(league_id=int(league_id), game_id=int(game_id))


def map_game_to_league_with_division(
    conn,
    *,
    league_id: int,
    game_id: int,
    division_name: Optional[str],
    division_id: Optional[int],
    conference_id: Optional[int],
    sort_order: Optional[int] = None,
) -> None:
    dn = (division_name or "").strip() or None
    del conn
    _django_orm, m = _orm_modules()
    lg, created = m.LeagueGame.objects.get_or_create(
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
    if dn is not None:
        updates["division_name"] = dn
    if division_id is not None:
        updates["division_id"] = int(division_id)
    if conference_id is not None:
        updates["conference_id"] = int(conference_id)
    if sort_order is not None:
        updates["sort_order"] = int(sort_order)
    if updates:
        m.LeagueGame.objects.filter(id=lg.id).update(**updates)


def ensure_user(conn, email: str, name: str | None = None, password_hash: str | None = None) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm:
        raise RuntimeError("user email is required")
    del conn
    _django_orm, m = _orm_modules()
    user_id = m.User.objects.filter(email=email_norm).values_list("id", flat=True).first()
    if user_id is not None:
        return int(user_id)
    if not password_hash:
        raise RuntimeError(
            f"User {email_norm!r} does not exist; pass --create-user and --password-hash to create it"
        )
    user = m.User.objects.create(
        email=email_norm,
        password_hash=str(password_hash),
        name=(name or email_norm),
        created_at=dt.datetime.now(),
    )
    return int(user.id)


def ensure_team(conn, user_id: int, name: str, *, is_external: bool = True) -> int:
    def _norm_team_name(s: str) -> str:
        # Normalize whitespace and common unicode variants to avoid duplicate teams that render identically in HTML.
        t = str(s or "").replace("\xa0", " ").strip()
        t = (
            t.replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2212", "-")
        )
        t = " ".join(t.split())
        return t

    nm = _norm_team_name(name or "")
    if not nm:
        nm = "UNKNOWN"
    del conn
    _django_orm, m = _orm_modules()
    now = dt.datetime.now()
    team = m.Team.objects.filter(user_id=int(user_id), name=str(nm)).first()
    if team:
        m.Team.objects.filter(id=team.id).update(is_external=bool(is_external), updated_at=now)
        return int(team.id)
    team = m.Team.objects.create(
        user_id=int(user_id),
        name=str(nm),
        is_external=bool(is_external),
        created_at=now,
    )
    return int(team.id)


def _download_logo_bytes(url: str) -> tuple[bytes, Optional[str]]:
    import requests

    resp = requests.get(
        url,
        timeout=float(os.environ.get("HM_T2S_HTTP_TIMEOUT", "30")),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type")


def _guess_ext(url: str, content_type: Optional[str]) -> str:
    path = urlparse(url).path
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
        if path.lower().endswith(ext):
            return ext
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "gif" in ct:
        return ".gif"
    if "webp" in ct:
        return ".webp"
    if "svg" in ct:
        return ".svg"
    return ".jpg"


def _ensure_team_logo(
    conn,
    *,
    team_db_id: int,
    team_owner_user_id: int,
    source: str,
    season_id: int,
    league_id: Optional[int],
    tts_team_id: int,
    logo_dir: Path,
    replace: bool,
    tts_direct,
) -> None:
    del conn
    _django_orm, m = _orm_modules()
    try:
        existing = (
            m.Team.objects.filter(id=int(team_db_id)).values_list("logo_path", flat=True).first()
        )
        existing = str(existing or "")
    except Exception:
        existing = ""

    if existing and not replace:
        return

    try:
        url = tts_direct.scrape_team_logo_url(
            str(source),
            season_id=int(season_id),
            team_id=int(tts_team_id),
            league_id=int(league_id) if league_id is not None else None,
        )
    except Exception:
        url = None
    if not url:
        return

    logo_dir.mkdir(parents=True, exist_ok=True)
    data, content_type = _download_logo_bytes(url)
    ext = _guess_ext(url, content_type)
    dest = logo_dir / f"t2s_{source}_season{int(season_id)}_team{int(tts_team_id)}{ext}"
    if not dest.exists() or replace:
        dest.write_bytes(data)
    m.Team.objects.filter(id=int(team_db_id), user_id=int(team_owner_user_id)).update(
        logo_path=str(dest), updated_at=dt.datetime.now()
    )


def _cleanup_numeric_named_players(conn, *, user_id: int, team_id: int) -> int:
    """Fix bogus players created from numeric scorer ids (e.g. name='88').

    Migrate any game-event rows / game-player links from bogus numeric-name players to the real
    player matching jersey_number, then delete the bogus player records if unused.
    """
    del conn
    _django_orm, m = _orm_modules()
    from django.db import transaction

    moved = 0
    bogus = list(
        m.Player.objects.filter(
            user_id=int(user_id), team_id=int(team_id), name__regex=r"^[0-9]+$"
        ).values_list("id", "name")
    )
    if not bogus:
        return 0

    with transaction.atomic():
        for bogus_id, bogus_name in bogus:
            real_id = (
                m.Player.objects.filter(
                    user_id=int(user_id),
                    team_id=int(team_id),
                    jersey_number=str(bogus_name),
                )
                .exclude(name__regex=r"^[0-9]+$")
                .values_list("id", flat=True)
                .first()
            )
            if real_id is None:
                continue

            # Move event rows to the real player_id (best-effort).
            event_ids = list(
                m.HkyGameEventRow.objects.filter(player_id=int(bogus_id)).values_list(
                    "id", flat=True
                )
            )
            if event_ids:
                moved += int(
                    m.HkyGameEventRow.objects.filter(id__in=[int(x) for x in event_ids]).update(
                        player_id=int(real_id)
                    )
                    or 0
                )

            # Move shift rows to the real player_id (best-effort).
            shift_ids = list(
                m.HkyGameShiftRow.objects.filter(player_id=int(bogus_id)).values_list(
                    "id", flat=True
                )
            )
            if shift_ids:
                moved += int(
                    m.HkyGameShiftRow.objects.filter(id__in=[int(x) for x in shift_ids]).update(
                        player_id=int(real_id)
                    )
                    or 0
                )

            # Move game-player links (dedupe per game).
            bogus_links = list(
                m.HkyGamePlayer.objects.filter(player_id=int(bogus_id)).values_list("id", "game_id")
            )
            for link_id, gid in bogus_links:
                if m.HkyGamePlayer.objects.filter(
                    game_id=int(gid), player_id=int(real_id)
                ).exists():
                    m.HkyGamePlayer.objects.filter(id=int(link_id)).delete()
                else:
                    m.HkyGamePlayer.objects.filter(id=int(link_id)).update(player_id=int(real_id))
                    moved += 1

            if (
                not m.HkyGameEventRow.objects.filter(player_id=int(bogus_id)).exists()
                and not m.HkyGameShiftRow.objects.filter(player_id=int(bogus_id)).exists()
                and not m.HkyGamePlayer.objects.filter(player_id=int(bogus_id)).exists()
            ):
                m.Player.objects.filter(id=int(bogus_id)).delete()

    return moved


def upsert_hky_game(
    conn,
    *,
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[str],
    location: Optional[str],
    team1_score: Optional[int],
    team2_score: Optional[int],
    replace: bool,
    notes: Optional[str] = None,
    timetoscore_game_id: Optional[int] = None,
    timetoscore_season_id: Optional[int] = None,
    timetoscore_source: Optional[str] = None,
) -> int:
    del conn
    _django_orm, m = _orm_modules()
    from django.db import transaction

    # Standardize notes with JSON like the webapp import API, but keep backward
    # compatibility for matching older token formats.
    notes_json_fields: dict[str, Any] = {}
    if timetoscore_game_id is not None:
        notes_json_fields["timetoscore_game_id"] = int(timetoscore_game_id)
    if timetoscore_season_id is not None:
        notes_json_fields["timetoscore_season_id"] = int(timetoscore_season_id)
    if timetoscore_source:
        notes_json_fields["timetoscore_source"] = str(timetoscore_source)
    if notes and str(notes).strip():
        # If caller passed JSON already, merge it; otherwise preserve as raw string under "notes_raw".
        try:
            parsed = json.loads(str(notes))
            if isinstance(parsed, dict):
                notes_json_fields.update(parsed)
            else:
                notes_json_fields["notes_raw"] = str(notes)
        except Exception:
            notes_json_fields["notes_raw"] = str(notes)

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
        return existing

    def _parse_dt(raw: Optional[str]) -> Optional[dt.datetime]:
        s = str(raw or "").strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                return dt.datetime.strptime(s, fmt)
            except Exception:
                continue
        try:
            return dt.datetime.fromisoformat(s)
        except Exception:
            return None

    starts_at_dt = _parse_dt(starts_at)
    now = dt.datetime.now()

    with transaction.atomic():
        game = None
        if starts_at_dt is not None:
            game = m.HkyGame.objects.filter(
                user_id=int(user_id),
                team1_id=int(team1_id),
                team2_id=int(team2_id),
                starts_at=starts_at_dt,
            ).first()

        if game is None and timetoscore_game_id is not None:
            tts_int = int(timetoscore_game_id)
            token_plain = f"game_id={tts_int}"
            token_json_nospace = f'"timetoscore_game_id":{tts_int}'
            token_json_space = f'"timetoscore_game_id": {tts_int}'
            for token in (token_json_nospace, token_json_space, token_plain):
                game = m.HkyGame.objects.filter(
                    user_id=int(user_id), notes__contains=str(token)
                ).first()
                if game is not None:
                    break

        if game is None:
            merged_notes = (
                json.dumps(notes_json_fields, sort_keys=True)
                if notes_json_fields
                else (notes or None)
            )
            g = m.HkyGame.objects.create(
                user_id=int(user_id),
                team1_id=int(team1_id),
                team2_id=int(team2_id),
                game_type_id=(int(game_type_id) if game_type_id is not None else None),
                starts_at=starts_at_dt,
                location=location,
                team1_score=team1_score,
                team2_score=team2_score,
                is_final=bool(team1_score is not None and team2_score is not None),
                notes=merged_notes,
                stats_imported_at=now,
                created_at=now,
            )
            return int(g.id)

        existing_notes = str(game.notes) if game.notes is not None else None
        merged_notes = (
            _merge_notes(existing_notes, notes_json_fields)
            if notes_json_fields
            else (existing_notes or "")
        )

        updates: dict[str, Any] = {
            "notes": merged_notes,
            "stats_imported_at": now,
            "updated_at": now,
        }
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
            if game.team1_score is None and team1_score is not None:
                updates["team1_score"] = team1_score
            if game.team2_score is None and team2_score is not None:
                updates["team2_score"] = team2_score
            if (
                game.team1_score is None
                and game.team2_score is None
                and team1_score is not None
                and team2_score is not None
            ):
                updates["is_final"] = True

        m.HkyGame.objects.filter(id=int(game.id)).update(**updates)
        return int(game.id)


def ensure_player(
    conn, *, user_id: int, team_id: int, name: str, jersey: Optional[str], position: Optional[str]
) -> int:
    def _norm_jersey(val: Any) -> Optional[str]:
        s = str(val or "").strip()
        if not s:
            return None
        m = re.search(r"(\\d+)", s)
        return m.group(1) if m else None

    def _strip_jersey_from_name(raw: str, jersey_number: Optional[str]) -> str:
        nm = str(raw or "").strip()
        if not nm:
            return ""
        jersey_norm = _norm_jersey(jersey_number) if jersey_number else None
        if jersey_norm:
            tail = re.sub(rf"\s*[\(#]?\s*{re.escape(jersey_norm)}\s*\)?\s*$", "", nm).strip()
            if tail:
                nm = tail
            head = re.sub(rf"^#?\s*{re.escape(jersey_norm)}\s+", "", nm).strip()
            if head:
                nm = head
        return nm

    raw_name = str(name or "").strip()
    nm = _strip_jersey_from_name(raw_name, jersey)
    if not nm:
        nm = "UNKNOWN"
    del conn
    _django_orm, m = _orm_modules()
    now = dt.datetime.now()

    p = m.Player.objects.filter(user_id=int(user_id), team_id=int(team_id), name=str(nm)).first()
    if not p and raw_name and raw_name != nm:
        p = m.Player.objects.filter(
            user_id=int(user_id), team_id=int(team_id), name=str(raw_name)
        ).first()
        if p:
            m.Player.objects.filter(id=p.id).update(name=str(nm), updated_at=now)
    if p:
        updates: dict[str, Any] = {}
        if jersey and not str(p.jersey_number or "").strip():
            updates["jersey_number"] = str(jersey)
        if position and not str(p.position or "").strip():
            updates["position"] = str(position)
        if updates:
            updates["updated_at"] = now
            m.Player.objects.filter(id=p.id).update(**updates)
        return int(p.id)

    p = m.Player.objects.create(
        user_id=int(user_id),
        team_id=int(team_id),
        name=str(nm),
        jersey_number=(str(jersey) if jersey else None),
        position=(str(position) if position else None),
        created_at=now,
    )
    return int(p.id)


def apply_games_batch_payload_to_db(conn, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the same payload shape used by the webapp REST endpoint
    `/api/import/hockey/games_batch` directly to a DB connection.

    This is primarily used for regression testing equivalence between direct DB
    import and REST import.
    """
    del conn
    _django_orm, m = _orm_modules()

    league_name = str(payload.get("league_name") or "").strip()
    if not league_name:
        raise ValueError("league_name is required")
    shared: Optional[bool] = bool(payload["shared"]) if "shared" in payload else None
    replace = bool(payload.get("replace", False))
    owner_email = str(payload.get("owner_email") or "").strip().lower()
    owner_name = str(payload.get("owner_name") or owner_email).strip()
    source = str(payload.get("source") or "").strip() or None
    external_key = str(payload.get("external_key") or "").strip() or None
    games = payload.get("games") or []
    if not isinstance(games, list):
        raise ValueError("games must be a list")

    # Best-effort: the deployed webapp DB already has these tables; in unit tests
    # we use a fake DB that doesn't implement DDL queries.
    try:
        ensure_defaults(None)
    except Exception:
        pass
    try:
        ensure_league_schema(None)
    except Exception:
        pass

    # Ensure owner user exists (mirror webapp import behavior; do not require create-user flags here).
    owner_user_id = m.User.objects.filter(email=owner_email).values_list("id", flat=True).first()
    if owner_user_id is None:
        user = m.User.objects.create(
            email=owner_email,
            password_hash="imported",
            name=owner_name or owner_email,
            created_at=dt.datetime.now(),
        )
        owner_user_id = int(user.id)
    else:
        owner_user_id = int(owner_user_id)

    # Ensure league exists.
    now = dt.datetime.now()
    league_row = m.League.objects.filter(name=league_name).values("id", "is_shared").first()
    if league_row:
        league_id = int(league_row["id"])
        if shared is not None and bool(league_row.get("is_shared")) != bool(shared):
            m.League.objects.filter(id=league_id).update(is_shared=bool(shared), updated_at=now)
    else:
        if shared is None:
            shared = True
        league = m.League.objects.create(
            name=league_name,
            owner_user_id=int(owner_user_id),
            is_shared=bool(shared),
            source=source,
            external_key=external_key,
            created_at=now,
        )
        league_id = int(league.id)

    ensure_league_member(None, league_id, owner_user_id, role="admin")

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

    def _ensure_game_type_id(name_any: Any) -> Optional[int]:
        nm = _normalize_import_game_type_name(name_any)
        if not nm:
            return None
        gt, _created = m.GameType.objects.get_or_create(
            name=str(nm), defaults={"is_default": False}
        )
        return int(gt.id)

    def _ensure_player_for_import(
        team_id: int, name: str, jersey_number: Optional[str], position: Optional[str]
    ) -> int:
        nm = (name or "").strip()
        if not nm:
            raise ValueError("player name is required")
        now2 = dt.datetime.now()
        p = m.Player.objects.filter(
            user_id=int(owner_user_id), team_id=int(team_id), name=str(nm)
        ).first()
        if p:
            updates: dict[str, Any] = {}
            if jersey_number is not None:
                updates["jersey_number"] = str(jersey_number)
            if position is not None:
                updates["position"] = str(position)
            if updates:
                updates["updated_at"] = now2
                m.Player.objects.filter(id=p.id).update(**updates)
            return int(p.id)
        p = m.Player.objects.create(
            user_id=int(owner_user_id),
            team_id=int(team_id),
            name=str(nm),
            jersey_number=(str(jersey_number) if jersey_number else None),
            position=(str(position) if position else None),
            created_at=now2,
        )
        return int(p.id)

    results: list[dict[str, Any]] = []

    def _clean_division_name(dn: Any) -> Optional[str]:
        s = str(dn or "").strip()
        if not s:
            return None
        if s.lower() == "external":
            return None
        return s

    def _league_team_div_meta(team_id: int) -> tuple[Optional[str], Optional[int], Optional[int]]:
        r = (
            m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(team_id))
            .values("division_name", "division_id", "conference_id")
            .first()
        )
        if not r:
            return None, None, None
        dn = _clean_division_name(r.get("division_name"))
        did = None
        try:
            did = int(r.get("division_id")) if r.get("division_id") is not None else None
        except Exception:
            did = None
        cid = None
        try:
            cid = int(r.get("conference_id")) if r.get("conference_id") is not None else None
        except Exception:
            cid = None
        return dn, did, cid

    for game in games:
        if not isinstance(game, dict):
            continue
        home_name = str(game.get("home_name") or "").strip()
        away_name = str(game.get("away_name") or "").strip()
        if not home_name or not away_name:
            continue

        team1_id = ensure_team(None, owner_user_id, home_name, is_external=True)
        team2_id = ensure_team(None, owner_user_id, away_name, is_external=True)

        starts_at = str(game.get("starts_at") or "").strip() or None
        location = str(game.get("location") or "").strip() or None
        team1_score = int(game["home_score"]) if game.get("home_score") is not None else None
        team2_score = int(game["away_score"]) if game.get("away_score") is not None else None

        tts_game_id = (
            int(game["timetoscore_game_id"])
            if game.get("timetoscore_game_id") is not None
            else None
        )
        season_id = int(game["season_id"]) if game.get("season_id") is not None else None
        game_type_id = _ensure_game_type_id(
            game.get("game_type_name")
            or game.get("game_type")
            or game.get("timetoscore_type")
            or game.get("type")
        )

        gid = upsert_hky_game(
            None,
            user_id=owner_user_id,
            team1_id=team1_id,
            team2_id=team2_id,
            game_type_id=game_type_id,
            starts_at=starts_at,
            location=location,
            team1_score=team1_score,
            team2_score=team2_score,
            replace=replace,
            notes=None,
            timetoscore_game_id=tts_game_id,
            timetoscore_season_id=season_id,
            timetoscore_source=source,
        )

        # Map teams/games to league with division metadata (best-effort).
        def _int_or_none(x: Any) -> Optional[int]:
            try:
                return int(x) if x is not None else None
            except Exception:
                return None

        division_name = _clean_division_name(game.get("division_name"))
        division_id = _int_or_none(game.get("division_id"))
        conference_id = _int_or_none(game.get("conference_id"))
        sort_order = _int_or_none(game.get("sort_order"))

        home_div_name = _clean_division_name(game.get("home_division_name")) or division_name
        away_div_name = _clean_division_name(game.get("away_division_name")) or division_name

        map_team_to_league_with_division(
            None,
            league_id=league_id,
            team_id=team1_id,
            division_name=home_div_name,
            division_id=_int_or_none(game.get("home_division_id")) or division_id,
            conference_id=_int_or_none(game.get("home_conference_id")) or conference_id,
        )
        map_team_to_league_with_division(
            None,
            league_id=league_id,
            team_id=team2_id,
            division_name=away_div_name,
            division_id=_int_or_none(game.get("away_division_id")) or division_id,
            conference_id=_int_or_none(game.get("away_conference_id")) or conference_id,
        )

        effective_div_name = division_name or home_div_name or away_div_name
        effective_div_id = (
            division_id
            or _int_or_none(game.get("home_division_id"))
            or _int_or_none(game.get("away_division_id"))
        )
        effective_conf_id = (
            conference_id
            or _int_or_none(game.get("home_conference_id"))
            or _int_or_none(game.get("away_conference_id"))
        )
        if not effective_div_name:
            t1_dn, t1_did, t1_cid = _league_team_div_meta(int(team1_id))
            t2_dn, t2_did, t2_cid = _league_team_div_meta(int(team2_id))
            if t1_dn:
                effective_div_name = t1_dn
                effective_div_id = effective_div_id or t1_did
                effective_conf_id = effective_conf_id or t1_cid
            elif t2_dn:
                effective_div_name = t2_dn
                effective_div_id = effective_div_id or t2_did
                effective_conf_id = effective_conf_id or t2_cid

        map_game_to_league_with_division(
            None,
            league_id=league_id,
            game_id=gid,
            division_name=effective_div_name,
            division_id=effective_div_id,
            conference_id=effective_conf_id,
            sort_order=sort_order,
        )

        # Rosters (optional).
        roster_player_ids_by_team: dict[int, set[int]] = {
            int(team1_id): set(),
            int(team2_id): set(),
        }
        for side_key, tid in (("home", team1_id), ("away", team2_id)):
            roster = game.get(f"{side_key}_roster") or []
            if not isinstance(roster, list):
                continue
            for row in roster:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name") or "").strip()
                if not nm:
                    continue
                jersey = str(row.get("number") or "").strip() or None
                pos = str(row.get("position") or "").strip() or None
                pid = _ensure_player_for_import(int(tid), nm, jersey, pos)
                roster_player_ids_by_team[int(tid)].add(int(pid))

        events_csv = game.get("events_csv")
        played = (
            bool(game.get("is_final"))
            or (team1_score is not None and team2_score is not None)
            or (isinstance(events_csv, str) and events_csv.strip())
        )
        if played:
            now = dt.datetime.now()
            to_create = []
            for tid, pids in roster_player_ids_by_team.items():
                for pid in sorted(pids):
                    to_create.append(
                        m.HkyGamePlayer(
                            game_id=int(gid),
                            player_id=int(pid),
                            team_id=int(tid),
                            created_at=now,
                            updated_at=None,
                        )
                    )
            if to_create:
                m.HkyGamePlayer.objects.bulk_create(to_create, ignore_conflicts=True)

        if isinstance(events_csv, str) and events_csv.strip():
            try:
                from tools.webapp.django_app import views as web_views  # type: ignore
            except Exception:  # pragma: no cover
                from django_app import views as web_views  # type: ignore

            web_views._upsert_game_event_rows_from_events_csv(
                game_id=int(gid),
                events_csv=str(events_csv),
                replace=bool(replace),
                create_missing_players=False,
                incoming_source_label=str(payload.get("source") or "timetoscore"),
                prefer_incoming_for_event_types={
                    "goal",
                    "assist",
                    "penalty",
                    "penaltyexpired",
                    "goaliechange",
                },
            )

        results.append({"game_id": int(gid), "team1_id": int(team1_id), "team2_id": int(team2_id)})

    return {
        "ok": True,
        "league_id": int(league_id),
        "owner_user_id": int(owner_user_id),
        "imported": len(results),
        "results": results,
    }


def parse_starts_at(
    source: str, *, stats: dict[str, Any], fallback: Optional[dict[str, Any]]
) -> Optional[str]:
    from hmlib.time2score import util as tutil

    date_s = str(stats.get("date") or "").strip()
    time_s = str(stats.get("time") or "").strip()
    if date_s and time_s:
        try:
            dt_val = tutil.parse_game_time(date_s, time_s, year=None)
            return dt_val.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            if source == "sharksice":
                try:
                    d = dt.datetime.strptime(date_s, "%A, %B %d, %Y").date()
                    t = dt.datetime.strptime(time_s, "%I:%M %p").time()
                    return dt.datetime.combine(d, t).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
    st = (fallback or {}).get("start_time")
    if isinstance(st, dt.datetime):
        return st.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(st, str) and st.strip():
        return st.strip()
    return None


def main(argv: Optional[list[str]] = None) -> int:
    base_dir = Path(__file__).resolve().parents[1]
    default_cfg = os.environ.get("HM_DB_CONFIG") or str(base_dir / "config.json")
    ap = argparse.ArgumentParser(
        description="Import TimeToScore into HockeyMOM webapp DB (no sqlite cache)"
    )
    ap.add_argument("--config", default=default_cfg, help="Path to webapp DB config.json")
    ap.add_argument(
        "--source", choices=("caha", "sharksice"), default="caha", help="TimeToScore site"
    )
    ap.add_argument(
        "--t2s-league-id",
        type=int,
        default=None,
        help=(
            "TimeToScore `league=` id for CAHA imports (default: 3). "
            "Examples: 3 (regular), 5 (tier teams), 18 (tournament-only)."
        ),
    )
    ap.add_argument("--season", type=int, default=0, help="Season id (0 = current/latest)")
    ap.add_argument("--list-seasons", action="store_true", help="List seasons and exit")
    ap.add_argument(
        "--list-divisions", action="store_true", help="List divisions for season and exit"
    )

    ap.add_argument(
        "--user-email", required=True, help="Webapp user email that will own imported data"
    )
    ap.add_argument("--user-name", default=None, help="Name for user creation if missing")
    ap.add_argument(
        "--password-hash", default=None, help="Password hash for creating user if missing"
    )
    ap.add_argument(
        "--create-user",
        action="store_true",
        help="Create user if missing (requires --password-hash)",
    )

    ap.add_argument("--replace", action="store_true", help="Overwrite existing scores/events")
    ap.add_argument(
        "--no-import-logos", action="store_true", help="Skip downloading and saving team logos"
    )
    ap.add_argument(
        "--logo-dir",
        default=None,
        help="Directory to store downloaded logos (default: /opt/hm-webapp/app/instance/uploads/team_logos if present)",
    )
    ap.add_argument(
        "--no-cleanup-bogus-players",
        action="store_true",
        help="Skip cleaning bogus numeric-name players created from score sheets",
    )
    ap.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run DB cleanup (no schedule/game scraping).",
    )
    ap.add_argument(
        "--refresh-team-metadata",
        action="store_true",
        help="Refresh league team divisions/logos from TimeToScore division lists (no game scraping).",
    )
    ap.add_argument(
        "--division",
        dest="divisions",
        action="append",
        default=[],
        help="Division filter token (repeatable)",
    )
    ap.add_argument(
        "--team",
        dest="teams",
        action="append",
        default=[],
        help="Team substring filter (repeatable)",
    )
    ap.add_argument(
        "--game-id",
        dest="game_ids",
        action="append",
        default=[],
        help="Import specific game id (repeatable)",
    )
    ap.add_argument("--games-file", default=None, help="File containing one game id per line")
    ap.add_argument("--limit", type=int, default=None, help="Max games to import (for testing)")
    ap.add_argument(
        "--hockey-db-dir",
        type=Path,
        default=Path.home() / ".cache" / "hockeymom",
        help=(
            "Directory for the local TimeToScore sqlite cache (hockey_league.db). "
            "Used to avoid re-scraping game pages when possible (default: ~/.cache/hockeymom)."
        ),
    )
    ap.add_argument(
        "--scrape",
        action="store_true",
        help="Force re-scraping TimeToScore game pages (refreshed stats are written back to the local cache).",
    )

    ap.add_argument(
        "--league-name",
        default=None,
        help="League name to import into (default: same as --source; created if missing)",
    )
    ap.add_argument(
        "--league-owner-email", default=None, help="Owner of the league (defaults to --user-email)"
    )
    ap.add_argument(
        "--shared",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set whether the league is shared (default: leave unchanged; used for league creation if missing).",
    )
    ap.add_argument(
        "--share-with",
        action="append",
        default=[],
        help="Emails to add as league viewers (repeatable)",
    )

    ap.add_argument(
        "--api-url",
        "--url",
        dest="api_url",
        default=None,
        help="If set, import via the webapp REST API at this base URL (e.g. http://127.0.0.1:8008).",
    )
    ap.add_argument(
        "--api-token",
        "--import-token",
        dest="api_token",
        default=None,
        help="Optional import token for REST API auth (sent as Authorization: Bearer ... and X-HM-Import-Token).",
    )
    ap.add_argument(
        "--api-batch-size",
        type=int,
        default=50,
        help="Games per REST batch request (only with --api-url).",
    )
    ap.add_argument(
        "--api-timeout-s",
        type=float,
        default=180.0,
        help="Requests timeout (seconds) for REST API calls (only with --api-url).",
    )
    ap.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="Max retries for transient REST failures (only with --api-url).",
    )
    ap.add_argument(
        "--api-retry-backoff-s",
        type=float,
        default=2.0,
        help="Initial backoff (seconds) between REST retries (only with --api-url).",
    )
    ap.add_argument(
        "--api-split-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a REST batch request fails, recursively split the batch to isolate the failing game(s). "
            "(only with --api-url)."
        ),
    )
    ap.add_argument(
        "--api-skip-failed-games",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a single-game REST request still fails after retries/splitting, skip it and continue. "
            "(only with --api-url)."
        ),
    )
    ap.add_argument(
        "--api-failure-dir",
        type=Path,
        default=None,
        help=(
            "If set, write failing REST payloads to this directory for debugging (only with --api-url). "
            "Note: payloads may be large."
        ),
    )
    ap.add_argument(
        "--t2s-max-attempts",
        type=int,
        default=6,
        help="Max scrape attempts per game when TimeToScore results look incomplete (throttling/HTML changes).",
    )
    ap.add_argument(
        "--t2s-initial-backoff-s",
        type=float,
        default=1.5,
        help="Initial backoff seconds between TimeToScore scrape attempts.",
    )
    ap.add_argument(
        "--t2s-max-backoff-s",
        type=float,
        default=30.0,
        help="Max backoff seconds between TimeToScore scrape attempts.",
    )
    ap.add_argument(
        "--allow-schedule-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow importing schedule-only games when TimeToScore has a recorded score but the game page has "
            "no usable boxscore (no roster/scoring/penalties)."
        ),
    )

    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from hmlib.time2score import direct as tts_direct
    from hmlib.time2score import normalize as tts_norm
    from hmlib.time2score import database as tts_db

    rest_mode = bool(args.api_url)
    if rest_mode and (args.cleanup_only or args.refresh_team_metadata or args.share_with):
        raise SystemExit(
            "--cleanup-only/--refresh-team-metadata/--share-with are only supported for direct DB imports (omit --api-url)"
        )

    conn = None
    m = None
    if not rest_mode:
        log(f"Initializing ORM via config: {args.config}")
        try:
            from tools.webapp import django_orm  # type: ignore
        except Exception:  # pragma: no cover
            import django_orm  # type: ignore

        django_orm.setup_django(config_path=str(args.config))
        # Ensure the cached module view is based on this config.
        _orm_modules.cache_clear()
        ensure_league_schema(None)
        ensure_defaults(None)
        _django_orm, m = _orm_modules()

    logo_dir = None
    if (not args.no_import_logos) and (not rest_mode):
        if args.logo_dir:
            logo_dir = Path(str(args.logo_dir)).expanduser()
        else:
            preferred = Path("/opt/hm-webapp/app/instance/uploads/team_logos")
            logo_dir = (
                preferred
                if preferred.exists()
                else (base_dir / "instance" / "uploads" / "team_logos")
            )

    user_id = None
    if not rest_mode:
        log(f"Resolving user: {args.user_email}")
        user_id = ensure_user(
            None,
            args.user_email,
            name=args.user_name or args.user_email,
            password_hash=(args.password_hash if args.create_user else None),
        )

    t2s_league_id = int(args.t2s_league_id) if args.t2s_league_id is not None else None
    if str(args.source or "").strip().lower() == "caha" and t2s_league_id is None:
        t2s_league_id = 3

    seasons = tts_direct.list_seasons(args.source, league_id=t2s_league_id)
    if args.list_seasons:
        for name, sid in sorted(seasons.items(), key=lambda kv: int(kv[1])):
            print(f"{sid}\t{name}")
        return 0

    season_id = int(args.season) or tts_direct.pick_current_season_id(
        args.source, league_id=t2s_league_id
    )
    log(f"Using source={args.source} league_id={t2s_league_id} season_id={season_id}")
    hockey_db_dir = Path(args.hockey_db_dir).expanduser()
    cache_source = (
        f"{str(args.source).strip().lower()}:{int(t2s_league_id)}"
        if t2s_league_id is not None
        else str(args.source).strip().lower()
    )

    def _get_cached_stats(game_id: int) -> Optional[dict[str, Any]]:
        try:
            with _working_directory(hockey_db_dir):
                db = tts_db.Database()
                db.create_tables()
                cached = db.get_cached_game_stats(str(cache_source), int(game_id))
                if cached:
                    return cached
                row = db.get_game(int(game_id))
                if row and row.get("stats"):
                    return row.get("stats")
        except Exception:
            return None
        return None

    def _set_cached_stats(game_id: int, stats: dict[str, Any]) -> None:
        try:
            with _working_directory(hockey_db_dir):
                db = tts_db.Database()
                db.create_tables()
                db.set_cached_game_stats(
                    str(cache_source),
                    int(game_id),
                    season_id=int(season_id) if season_id is not None else None,
                    stats=dict(stats or {}),
                )
        except Exception:
            pass

    divs = tts_direct.list_divisions(args.source, season_id=season_id, league_id=t2s_league_id)
    if args.list_divisions:
        for d in sorted(divs, key=lambda x: (int(x.division_id), int(x.conference_id), x.name)):
            print(f"{d.division_id}:{d.conference_id}\t{d.name}\tteams={len(d.teams)}")
        return 0

    # Used to disambiguate team names that appear in multiple divisions (common on CAHA).
    def _clean_team_name(name: str) -> str:
        t = str(name or "").replace("\xa0", " ").strip()
        t = (
            t.replace("\u2010", "-")
            .replace("\u2011", "-")
            .replace("\u2012", "-")
            .replace("\u2013", "-")
            .replace("\u2212", "-")
        )
        t = " ".join(t.split())
        # Normalize spaced dash-index suffixes like "12A - 1" -> "12A-1".
        t = re.sub(r"\s*-\s*(\d+)\s*$", r"-\1", t)
        return t

    def _norm_team_key(name: str) -> str:
        return _clean_team_name(name).casefold()

    def _clean_division_label(name: str) -> str:
        t = str(name or "").replace("\xa0", " ").strip()
        t = " ".join(t.split())
        return t

    def _norm_div_key(name: str) -> str:
        # Normalize common variants:
        #   - "10U B West" == "10 B West"
        #   - "12 AA" == "12AA"
        t = _clean_division_label(name).casefold()
        t = re.sub(r"(?i)(\d)u\b", r"\1", t)  # "10U" -> "10"
        t = re.sub(r"\s+", "", t)
        return t

    def _base_div_token_from_div_key(div_key: str) -> Optional[str]:
        """
        Extract a base token like "12aa" or "10b" from a normalized division key like:
          - "12aa", "12aaa", "10beast", "10bwest", "14bb", ...
        """
        s = str(div_key or "").strip().casefold()
        m = re.match(r"^(\d{1,2})(aaa|aa|bb|a|b)\b", s)
        if not m:
            return None
        try:
            age_i = int(m.group(1))
        except Exception:
            return None
        lvl = str(m.group(2) or "").casefold()
        return f"{age_i}{lvl}"

    def _format_division_name_from_base_token(base_token: str) -> Optional[str]:
        m = re.match(r"^(\d{1,2})(aaa|aa|bb|a|b)$", str(base_token or "").strip().casefold())
        if not m:
            return None
        try:
            age_i = int(m.group(1))
        except Exception:
            return None
        lvl = str(m.group(2) or "").upper()
        return f"{age_i} {lvl}"

    def _is_external_division_name(name: Optional[str]) -> bool:
        return str(name or "").strip().casefold().startswith("external")

    def _is_girls_division_name(name: Optional[str]) -> bool:
        return str(name or "").strip().casefold().startswith("girls")

    # Filter out Girls divisions at the source (we do not import them).
    divs = [
        d
        for d in divs
        if not _is_girls_division_name(_clean_division_label(getattr(d, "name", "") or "").strip())
    ]

    # Pre-discover schedule and per-team division preference so we can:
    #  - ignore divisions with no games scheduled
    #  - choose a stable team division when a team appears in multiple divisions
    explicit_game_ids: list[int] = []
    if args.games_file:
        for line in Path(args.games_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.isdigit():
                explicit_game_ids.append(int(s))
    for s in args.game_ids:
        if str(s).strip().isdigit():
            explicit_game_ids.append(int(s))

    pre_fallback_by_gid: dict[int, dict[str, Any]] = {}
    if not (args.cleanup_only or args.refresh_team_metadata) and not explicit_game_ids:
        log("Discovering game ids from team schedules...")
        pre_fallback_by_gid = tts_direct.iter_season_games(
            args.source,
            season_id=season_id,
            league_id=int(t2s_league_id) if t2s_league_id is not None else None,
            divisions=None,
            team_name_substrings=args.teams,
            progress_cb=log,
            progress_every_teams=10,
            heartbeat_seconds=30.0,
        )

    team_name_example_by_key: dict[str, str] = {}
    div_name_example_by_key: dict[str, str] = {}
    team_div_counts: dict[str, dict[str, int]] = {}
    schedule_div_ids: set[tuple[int, int]] = set()
    schedule_div_name_keys: set[str] = set()
    if pre_fallback_by_gid:
        for fb in pre_fallback_by_gid.values():
            fb_div_name = (
                _clean_division_label((fb or {}).get("division_name") or "").strip() or None
            )
            if _is_girls_division_name(fb_div_name):
                continue
            div_key = _norm_div_key(fb_div_name or "")
            if div_key:
                div_name_example_by_key.setdefault(div_key, fb_div_name or "")
                schedule_div_name_keys.add(div_key)
            try:
                did = int((fb or {}).get("division_id"))
                cid = int((fb or {}).get("conference_id"))
                schedule_div_ids.add((did, cid))
            except Exception:
                pass
            for side_key in ("home", "away"):
                nm_raw = _clean_team_name((fb or {}).get(side_key) or "")
                if not nm_raw:
                    continue
                tkey = _norm_team_key(nm_raw)
                team_name_example_by_key.setdefault(tkey, nm_raw)
                team_div_counts.setdefault(tkey, {})
                team_div_counts[tkey][div_key] = team_div_counts[tkey].get(div_key, 0) + 1

        pre_fallback_by_gid = {
            int(gid): fb
            for gid, fb in (pre_fallback_by_gid or {}).items()
            if not _is_girls_division_name(
                _clean_division_label((fb or {}).get("division_name") or "").strip() or None
            )
        }

    # Filter out divisions with no games scheduled.
    if pre_fallback_by_gid:
        filtered_divs: list[Any] = []
        for d in divs:
            dn = _clean_division_label(getattr(d, "name", "") or "").strip()
            key_id = (int(getattr(d, "division_id", 0)), int(getattr(d, "conference_id", 0)))
            key_name = _norm_div_key(dn)
            if key_id not in schedule_div_ids and key_name not in schedule_div_name_keys:
                continue
            filtered_divs.append(d)
        divs = filtered_divs

    div_meta_by_key: dict[str, tuple[str, Optional[int], Optional[int]]] = {}
    for d in divs:
        dn = _clean_division_label(getattr(d, "name", "") or "").strip()
        if not dn:
            continue
        try:
            did = int(getattr(d, "division_id", 0))
        except Exception:
            did = None
        try:
            cid = int(getattr(d, "conference_id", 0))
        except Exception:
            cid = None
        div_meta_by_key.setdefault(_norm_div_key(dn), (dn, did, cid))

    def _infer_division_key_from_team_name(name: str) -> Optional[str]:
        nm = _clean_team_name(name or "")
        m = re.search(
            r"(?i)(?:^|\s)(\d{1,2})(?:u)?\s*(AAA|AA|BB|A|B)(?:\s*[-–—]?\s*\d+)?\s*$",
            nm,
        )
        if not m:
            return None
        age = m.group(1)
        level = m.group(2).upper()
        return _norm_div_key(f"{age}{level}")

    preferred_base_div_name_by_team_key: dict[str, str] = {}
    if team_div_counts:
        for tkey, counts in team_div_counts.items():
            hint_key = _infer_division_key_from_team_name(team_name_example_by_key.get(tkey, ""))
            chosen_key: Optional[str] = None
            if hint_key:
                # Prefer a division that matches the base token encoded in the team name
                # (e.g. "Tri Valley ... 12AA-1" should stay 12AA even if it plays 12AAA exhibitions).
                candidates = {
                    k
                    for k in (
                        set(counts.keys())
                        | set(div_meta_by_key.keys())
                        | set(div_name_example_by_key.keys())
                    )
                    if _base_div_token_from_div_key(k) == hint_key
                }
                if candidates:
                    chosen_key = max(
                        candidates,
                        key=lambda k: (
                            int(counts.get(k, 0)),
                            1 if k in div_meta_by_key else 0,
                            str(k),
                        ),
                    )
                else:
                    # No known division matches the team's encoded token; still honor the token.
                    chosen_key = str(hint_key)
            elif counts:
                chosen_key = max(counts.items(), key=lambda kv: (int(kv[1]), str(kv[0])))[0]

            if not chosen_key:
                continue
            dn = None
            if chosen_key in div_meta_by_key:
                dn = div_meta_by_key[chosen_key][0]
            elif chosen_key in div_name_example_by_key:
                dn = div_name_example_by_key[chosen_key]
            elif hint_key and chosen_key == hint_key:
                dn = _format_division_name_from_base_token(str(hint_key))
            if dn:
                preferred_base_div_name_by_team_key[tkey] = dn

    # For CAHA, we may merge multiple TimeToScore `league=` ids into a single HockeyMOM league.
    # To keep team-name disambiguation stable across imports (and to classify tournament-only teams),
    # best-effort include additional CAHA leagues when computing duplicates / regular-team membership.
    season_name: Optional[str] = None
    try:
        for nm, sid in (seasons or {}).items():
            if int(sid) == int(season_id):
                season_name = str(nm)
                break
    except Exception:
        season_name = None

    caha_league_id_i: Optional[int] = None
    try:
        caha_league_id_i = int(t2s_league_id) if t2s_league_id is not None else None
    except Exception:
        caha_league_id_i = None

    def _try_list_caha_divisions(league_id: int) -> list[Any]:
        if str(args.source or "").strip().lower() != "caha":
            return []
        try:
            seasons2 = tts_direct.list_seasons("caha", league_id=int(league_id))
        except Exception:
            seasons2 = {}

        def _tokenize(s: str) -> set[str]:
            return {t for t in re.findall(r"[0-9a-z]+", str(s or "").casefold()) if t}

        base_name = str(season_name or "").strip()
        base_tokens = _tokenize(base_name)
        base_years = set(re.findall(r"(20\\d{2})", base_name))

        # NOTE: CAHA season ids/names differ across league=3/5/18, so we try a small set of
        # plausible seasons rather than assuming season_id matches across leagues.
        season_ids_to_try: list[int] = []

        if base_name and base_name in (seasons2 or {}):
            try:
                season_ids_to_try.append(int(seasons2[base_name]))
            except Exception:
                pass

        try:
            for _nm, _sid in (seasons2 or {}).items():
                try:
                    if int(_sid) == int(season_id):
                        season_ids_to_try.append(int(_sid))
                        break
                except Exception:
                    continue
        except Exception:
            pass

        scored: list[tuple[int, int]] = []
        if base_tokens and seasons2:
            for nm, sid in seasons2.items():
                try:
                    sid_i = int(sid)
                except Exception:
                    continue
                tok = _tokenize(str(nm))
                score = len(base_tokens & tok)
                if base_years and any(y in str(nm) for y in base_years):
                    score += 50
                scored.append((int(score), int(sid_i)))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _score, sid_i in scored[:5]:
                season_ids_to_try.append(int(sid_i))

        try:
            season_ids_to_try.append(int(season_id))
        except Exception:
            pass

        seen_sids: set[int] = set()
        ordered_sids: list[int] = []
        for sid_i in season_ids_to_try:
            if int(sid_i) in seen_sids:
                continue
            seen_sids.add(int(sid_i))
            ordered_sids.append(int(sid_i))

        for sid_i in ordered_sids:
            try:
                divs_i = tts_direct.list_divisions(
                    "caha",
                    season_id=int(sid_i),
                    league_id=int(league_id),
                )
            except Exception:
                divs_i = []
            if divs_i:
                return [
                    d
                    for d in divs_i
                    if not _is_girls_division_name(
                        _clean_division_label(getattr(d, "name", "") or "").strip() or None
                    )
                ]
        return []

    extra_divs_by_lid: dict[int, list[Any]] = {}
    if str(args.source or "").strip().lower() == "caha" and caha_league_id_i is not None:
        union_league_ids: list[int]
        if caha_league_id_i in (3, 5):
            union_league_ids = [3, 5]
        elif caha_league_id_i == 18:
            union_league_ids = [3, 5, 18]
        else:
            union_league_ids = [caha_league_id_i]

        for lid in union_league_ids:
            if int(lid) == int(caha_league_id_i):
                extra_divs_by_lid[int(lid)] = list(divs or [])
            else:
                extra_divs_by_lid[int(lid)] = _try_list_caha_divisions(int(lid))

    regular_team_names: set[str] = set()
    if str(args.source or "").strip().lower() == "caha" and caha_league_id_i == 18:
        for lid in (3, 5):
            for d in extra_divs_by_lid.get(int(lid), []):
                dn = _clean_division_label(getattr(d, "name", "") or "")
                if not dn:
                    continue
                for t in getattr(d, "teams", []) or []:
                    nm = _clean_team_name((t or {}).get("name") or "")
                    if nm:
                        regular_team_names.add(_norm_team_key(nm))

    def _effective_division_name_for_team(
        team_name: str, division_name: Optional[str]
    ) -> Optional[str]:
        dn = _clean_division_label(division_name or "").strip() or None
        if not dn:
            return None
        if (
            str(args.source or "").strip().lower() == "caha"
            and caha_league_id_i == 18
            and regular_team_names
        ):
            if _norm_team_key(team_name) not in regular_team_names:
                return f"External {dn}".strip()
        return dn

    # Map normalized team name -> set of division keys where that name appears.
    # Used to disambiguate teams that genuinely share a name across multiple divisions.
    div_keys_by_team_name: dict[str, set[str]] = {}
    divs_for_name_counts: list[Any] = []
    if extra_divs_by_lid:
        for dl in extra_divs_by_lid.values():
            divs_for_name_counts.extend(list(dl or []))
    else:
        divs_for_name_counts = list(divs or [])

    for d in divs_for_name_counts:
        dn = _clean_division_label(getattr(d, "name", "") or "")
        dn_key = _norm_div_key(dn)
        for t in getattr(d, "teams", []) or []:
            nm = _clean_team_name((t or {}).get("name") or "")
            if not nm:
                continue
            div_keys_by_team_name.setdefault(_norm_team_key(nm), set()).add(dn_key)

    def canonical_team_name(name: str, division_name: Optional[str]) -> str:
        nm = _clean_team_name(name or "") or "UNKNOWN"
        dn = _clean_division_label(division_name or "")

        # If the team name already encodes a division token (common on CAHA, e.g. "Sharks 12A-1"),
        # avoid adding redundant "(<division>)" suffixes.
        has_division_token = bool(
            re.search(
                r"(?i)(?:^|\\s)\\d{1,2}(?:u)?\\s*(?:AAA|AA|BB|A|B)(?:\\s*[-–—]?\\s*\\d+)?\\s*$",
                nm,
            )
        )
        if dn and not has_division_token:
            divs_for_name = div_keys_by_team_name.get(_norm_team_key(nm)) or set()
            if len(divs_for_name) > 1:
                return f"{nm} ({dn})"
        return nm

    division_teams: list[dict[str, Any]] = []
    seen_team_names: set[str] = set()
    for d in divs:
        dn_base = str(d.name or "").strip() or None
        try:
            did = int(d.division_id)
        except Exception:
            did = None
        try:
            cid = int(d.conference_id)
        except Exception:
            cid = None
        for t in d.teams:
            nm_raw = str((t or {}).get("name") or "").strip()
            if not nm_raw:
                continue
            dn_pref = preferred_base_div_name_by_team_key.get(_norm_team_key(nm_raw))
            dn_base_eff = dn_pref or dn_base
            dn = _effective_division_name_for_team(nm_raw, dn_base_eff)
            did_eff, cid_eff = did, cid
            if dn_pref:
                meta = div_meta_by_key.get(_norm_div_key(dn_pref))
                if meta:
                    did_eff = int(meta[1]) if meta[1] is not None else did_eff
                    cid_eff = int(meta[2]) if meta[2] is not None else cid_eff
                elif _norm_div_key(dn_pref) != _norm_div_key(dn_base or ""):
                    # Avoid sending mismatched ids (e.g. a 12AA team that appears on a 12AAA exhibition schedule).
                    did_eff = None
                    cid_eff = None
            nm = canonical_team_name(nm_raw, dn)
            key = nm.casefold()
            if key in seen_team_names:
                continue
            seen_team_names.add(key)
            tts_id_raw = (t or {}).get("id")
            try:
                tts_id = int(tts_id_raw) if tts_id_raw is not None else None
            except Exception:
                tts_id = None
            division_teams.append(
                {
                    "name": nm,
                    "division_name": dn,
                    "division_id": did_eff,
                    "conference_id": cid_eff,
                    "tts_team_id": tts_id,
                }
            )

    # Map (team_name_lower, division_name_lower) -> team_id for logo retrieval.
    tts_team_id_by_name_div: dict[tuple[str, str], int] = {}
    tts_team_ids_by_name: dict[str, list[int]] = {}
    tts_team_div_meta_by_id: dict[int, tuple[str, int, int]] = {}
    for d in divs:
        dn = _clean_division_label(d.name or "")
        for t in d.teams:
            nm = _clean_team_name((t or {}).get("name") or "")
            tid = (t or {}).get("id")
            if not nm or tid is None:
                continue
            try:
                tid_i = int(tid)
            except Exception:
                continue
            tts_team_id_by_name_div[(_norm_team_key(nm), _norm_div_key(dn))] = tid_i
            tts_team_ids_by_name.setdefault(_norm_team_key(nm), []).append(tid_i)
            tts_team_div_meta_by_id[tid_i] = (dn, int(d.division_id), int(d.conference_id))

    def resolve_tts_team_id(name: str, division_name: Optional[str]) -> Optional[int]:
        nm = _norm_team_key(name or "")
        dn = _norm_div_key(division_name or "")
        if nm and dn and (nm, dn) in tts_team_id_by_name_div:
            return int(tts_team_id_by_name_div[(nm, dn)])
        ids = tts_team_ids_by_name.get(nm) or []
        if len(ids) == 1:
            return int(ids[0])
        return None

    def resolve_team_division_meta(
        name: str, fallback_division_name: Optional[str]
    ) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """Return (division_name, division_id, conference_id, team_tts_id)."""
        tkey = _norm_team_key(name or "")
        pref_dn = preferred_base_div_name_by_team_key.get(tkey)
        tts_id = resolve_tts_team_id(name, fallback_division_name)
        if tts_id is not None and int(tts_id) in tts_team_div_meta_by_id:
            dn, did, cid = tts_team_div_meta_by_id[int(tts_id)]
            dn_base = pref_dn or dn
            dn_eff = _effective_division_name_for_team(name, dn_base)
            meta = div_meta_by_key.get(_norm_div_key(dn_base or "")) if dn_base else None
            pref_overrides = bool(pref_dn and _norm_div_key(pref_dn) != _norm_div_key(dn))
            if meta is None and pref_overrides:
                did_eff = None
                cid_eff = None
            else:
                did_eff = int(meta[1]) if meta and meta[1] is not None else int(did)
                cid_eff = int(meta[2]) if meta and meta[2] is not None else int(cid)
            return dn_eff, did_eff, cid_eff, int(tts_id)
        # If the team name is unique across all divisions, assign its division without relying on the game.
        ids = tts_team_ids_by_name.get(tkey) or []
        if len(ids) == 1 and int(ids[0]) in tts_team_div_meta_by_id:
            dn, did, cid = tts_team_div_meta_by_id[int(ids[0])]
            dn_base = pref_dn or dn
            dn_eff = _effective_division_name_for_team(name, dn_base)
            meta = div_meta_by_key.get(_norm_div_key(dn_base or "")) if dn_base else None
            pref_overrides = bool(pref_dn and _norm_div_key(pref_dn) != _norm_div_key(dn))
            if meta is None and pref_overrides:
                did_eff = None
                cid_eff = None
            else:
                did_eff = int(meta[1]) if meta and meta[1] is not None else int(did)
                cid_eff = int(meta[2]) if meta and meta[2] is not None else int(cid)
            return dn_eff, did_eff, cid_eff, int(ids[0])
        dn_base = pref_dn or (fallback_division_name or None)
        dn_eff = _effective_division_name_for_team(name, dn_base)
        meta = div_meta_by_key.get(_norm_div_key(dn_base or "")) if dn_base else None
        did_eff = int(meta[1]) if meta and meta[1] is not None else None
        cid_eff = int(meta[2]) if meta and meta[2] is not None else None
        return (dn_eff, did_eff, cid_eff, int(tts_id) if tts_id is not None else None)

    # Division filter resolution
    allowed_divs: Optional[set[tuple[int, int]]] = None
    if args.divisions:
        allowed_divs = set()
        name_map = {(d.division_id, d.conference_id): d.name for d in divs}
        for tok in args.divisions:
            s = str(tok).strip()
            if not s:
                continue
            if ":" in s:
                a, b = s.split(":", 1)
                if a.isdigit() and b.isdigit():
                    allowed_divs.add((int(a), int(b)))
                    continue
            if s.isdigit():
                lvl = int(s)
                for (d_id, conf), _nm in name_map.items():
                    if int(d_id) == lvl:
                        allowed_divs.add((int(d_id), int(conf)))
                continue
            s_low = s.lower()
            for key, nm in name_map.items():
                if s_low in (nm or "").lower():
                    allowed_divs.add(key)

    # Always import into a league (create if needed).
    league_name = str(args.league_name or args.source).strip() or str(args.source)
    owner_email = args.league_owner_email or args.user_email
    owner_id = None
    league_id = None
    if not rest_mode:
        assert user_id is not None
        owner_id = ensure_user(
            None,
            owner_email,
            name=owner_email,
            password_hash=(args.password_hash if args.create_user else None),
        )
        league_id = ensure_league(
            None,
            league_name,
            owner_id,
            args.shared,
            source="timetoscore",
            external_key=f"{args.source}:{season_id}",
        )
        ensure_league_member(None, league_id, owner_id, role="admin")
        ensure_league_member(None, league_id, user_id, role="editor")
        for em in args.share_with:
            try:
                uid = ensure_user(None, em, name=em, password_hash=None)
            except RuntimeError:
                continue
            ensure_league_member(None, league_id, uid, role="viewer")

    api_base = str(args.api_url or "").rstrip("/")
    api_headers: dict[str, str] = {}
    if rest_mode and args.api_token:
        tok = str(args.api_token).strip()
        if tok:
            api_headers["Authorization"] = f"Bearer {tok}"
            api_headers["X-HM-Import-Token"] = tok
    api_batch_size = max(1, int(args.api_batch_size or 1))
    api_timeout_s = max(1.0, float(args.api_timeout_s or 1.0))
    api_max_retries = max(0, int(args.api_max_retries or 0))
    api_retry_backoff_s = max(0.0, float(args.api_retry_backoff_s or 0.0))
    api_split_on_error = bool(args.api_split_on_error)
    api_skip_failed_games = bool(args.api_skip_failed_games)
    api_failure_dir: Optional[Path] = args.api_failure_dir

    logo_url_cache: dict[int, Optional[str]] = {}
    logo_b64_cache: dict[int, Optional[str]] = {}
    logo_ct_cache: dict[int, Optional[str]] = {}
    sent_logo_ids: set[int] = set()

    def _logo_url(tts_id: Optional[int]) -> Optional[str]:
        if (not rest_mode) or args.no_import_logos or tts_id is None:
            return None
        tid_i = int(tts_id)
        if tid_i in logo_url_cache:
            return logo_url_cache[tid_i]
        try:
            u = tts_direct.scrape_team_logo_url(
                str(args.source),
                season_id=int(season_id),
                team_id=tid_i,
                league_id=int(t2s_league_id) if t2s_league_id is not None else None,
            )
        except Exception:
            u = None
        logo_url_cache[tid_i] = u
        return u

    def _logo_b64_and_type(tts_id: Optional[int]) -> tuple[Optional[str], Optional[str]]:
        if (not rest_mode) or args.no_import_logos or tts_id is None:
            return None, None
        tid_i = int(tts_id)
        if tid_i in logo_b64_cache:
            return logo_b64_cache[tid_i], logo_ct_cache.get(tid_i)

        url = _logo_url(tid_i)
        if not url:
            logo_b64_cache[tid_i] = None
            logo_ct_cache[tid_i] = None
            return None, None

        try:
            data, content_type = _download_logo_bytes(url)
        except Exception:
            logo_b64_cache[tid_i] = None
            logo_ct_cache[tid_i] = None
            return None, None

        try:
            b64 = base64.b64encode(data).decode("ascii")
        except Exception:
            b64 = None
        logo_b64_cache[tid_i] = b64
        logo_ct_cache[tid_i] = content_type
        return b64, content_type

    if not (args.cleanup_only or args.refresh_team_metadata):
        if division_teams:
            log(f"Seeding league teams from TimeToScore divisions: {len(division_teams)}")
        if rest_mode and division_teams:
            import requests

            team_payload: list[dict[str, Any]] = []
            for row in division_teams:
                item: dict[str, Any] = {
                    "name": row["name"],
                    "division_name": row["division_name"],
                    "division_id": row["division_id"],
                    "conference_id": row["conference_id"],
                }
                if not args.no_import_logos:
                    tts_id = row.get("tts_team_id")
                    if tts_id is not None:
                        logo_url = _logo_url(int(tts_id))
                        logo_b64, logo_ct = _logo_b64_and_type(int(tts_id))
                        item["logo_url"] = logo_url
                        item["logo_b64"] = logo_b64
                        item["logo_content_type"] = logo_ct
                        if logo_b64:
                            sent_logo_ids.add(int(tts_id))
                team_payload.append(item)
            payload = {
                "league_name": league_name,
                "shared": bool(args.shared),
                "replace": bool(args.replace),
                "owner_email": owner_email,
                "owner_name": owner_email,
                "source": "timetoscore",
                "external_key": f"{args.source}:{season_id}",
                "teams": team_payload,
            }
            r = requests.post(
                f"{api_base}/api/import/hockey/teams",
                json=payload,
                headers=api_headers,
                timeout=180,
            )
            r.raise_for_status()
            out = r.json()
            if not out.get("ok"):
                raise RuntimeError(str(out))
        if (not rest_mode) and division_teams:
            assert league_id is not None
            assert user_id is not None
            for row in division_teams:
                team_db_id = ensure_team(None, user_id, row["name"], is_external=True)
                map_team_to_league_with_division(
                    None,
                    league_id=league_id,
                    team_id=team_db_id,
                    division_name=row["division_name"],
                    division_id=row["division_id"],
                    conference_id=row["conference_id"],
                )
                if logo_dir is not None and row.get("tts_team_id") is not None:
                    try:
                        _ensure_team_logo(
                            None,
                            team_db_id=int(team_db_id),
                            team_owner_user_id=user_id,
                            source=str(args.source),
                            season_id=int(season_id),
                            league_id=int(t2s_league_id) if t2s_league_id is not None else None,
                            tts_team_id=int(row["tts_team_id"]),
                            logo_dir=Path(logo_dir),
                            replace=bool(args.replace),
                            tts_direct=tts_direct,
                        )
                    except Exception as exc:
                        log(
                            f"Warning: failed to ensure team logo for team_id={team_db_id}, "
                            f"tts_team_id={row.get('tts_team_id')}: {exc}"
                        )

    if args.cleanup_only or args.refresh_team_metadata:
        game_ids = []
        fallback_by_gid = {}
    else:
        if explicit_game_ids:
            game_ids = list(explicit_game_ids)
            fallback_by_gid = {}
        else:
            fallback_by_gid = dict(pre_fallback_by_gid or {})
            if allowed_divs is not None and fallback_by_gid:
                filtered = {}
                for gid, fb in fallback_by_gid.items():
                    try:
                        did = int((fb or {}).get("division_id"))
                        cid = int((fb or {}).get("conference_id"))
                    except Exception:
                        continue
                    if (did, cid) in allowed_divs:
                        filtered[int(gid)] = fb
                fallback_by_gid = filtered
            game_ids = sorted(fallback_by_gid.keys())

    total = len(game_ids)
    log(f"Importing games: total={total} replace={bool(args.replace)}")

    count = 0
    skipped = 0
    posted = 0
    schedule_only = 0
    api_games_batch: list[dict[str, Any]] = []
    cleaned_team_ids: set[int] = set()
    started = time.time()
    last_heartbeat = started

    class _RestPostError(RuntimeError):
        def __init__(
            self,
            message: str,
            *,
            status_code: Optional[int] = None,
            response_text: Optional[str] = None,
        ) -> None:
            super().__init__(message)
            self.status_code = status_code
            self.response_text = response_text

    def _t2s_game_ids(batch: list[dict[str, Any]]) -> list[int]:
        out: list[int] = []
        for g in batch:
            try:
                gid = g.get("timetoscore_game_id")
                if gid is None:
                    continue
                out.append(int(gid))
            except Exception:
                continue
        return out

    def _format_t2s_ids(ids: list[int]) -> str:
        if not ids:
            return "[]"
        if len(ids) <= 10:
            return str(ids)
        return f"{ids[:5]}...{ids[-5:]} (n={len(ids)})"

    def _write_failure_payload(batch: list[dict[str, Any]], err: _RestPostError) -> Optional[Path]:
        if not api_failure_dir:
            return None
        try:
            api_failure_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        def _sanitize_game(g: dict[str, Any]) -> dict[str, Any]:
            out = dict(g)
            for k in ("home_logo_b64", "away_logo_b64"):
                if k in out and isinstance(out.get(k), str) and out.get(k):
                    out[k] = f"[omitted {len(str(out[k]))} chars]"
            return out

        payload = {
            "api_url": f"{api_base}/api/import/hockey/games_batch",
            "error": str(err),
            "status_code": err.status_code,
            "timetoscore_game_ids": _t2s_game_ids(batch),
            "request": {
                "league_name": league_name,
                "replace": bool(args.replace),
                "owner_email": owner_email,
                "owner_name": owner_email,
                "source": "timetoscore",
                "external_key": f"{args.source}:{season_id}",
                "shared": bool(args.shared) if args.shared is not None else None,
                "games": [_sanitize_game(g) for g in batch],
            },
        }
        suffix = secrets.token_hex(4)
        p = (
            api_failure_dir
            / f"t2s_games_batch_failed_{dt.datetime.now():%Y%m%d_%H%M%S}_{suffix}.json"
        )
        try:
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return p
        except Exception:
            return None

    def _post_batch() -> None:
        nonlocal posted, skipped, api_games_batch, api_batch_size
        if not rest_mode or not api_games_batch:
            return
        import requests

        def _post_once(batch: list[dict[str, Any]]) -> int:
            payload = {
                "league_name": league_name,
                "replace": bool(args.replace),
                "owner_email": owner_email,
                "owner_name": owner_email,
                "source": "timetoscore",
                "external_key": f"{args.source}:{season_id}",
                "games": batch,
            }
            if args.shared is not None:
                payload["shared"] = bool(args.shared)

            url = f"{api_base}/api/import/hockey/games_batch"
            try:
                r = requests.post(url, json=payload, headers=api_headers, timeout=api_timeout_s)
            except requests.RequestException as e:
                raise _RestPostError(
                    f"REST request failed: {type(e).__name__}: {e}",
                    status_code=None,
                    response_text=None,
                ) from e

            text = ""
            try:
                text = str(r.text or "")
            except Exception:
                text = ""
            out: Any
            try:
                out = r.json()
            except Exception:
                out = None

            if int(r.status_code) >= 400:
                detail = out if isinstance(out, dict) else (text[:1000] if text else "<no body>")
                raise _RestPostError(
                    f"HTTP {int(r.status_code)} from {url}: {detail}",
                    status_code=int(r.status_code),
                    response_text=text,
                )

            if not isinstance(out, dict):
                raise _RestPostError(
                    f"Unexpected non-JSON response from {url}: {text[:1000] if text else '<no body>'}",
                    status_code=int(r.status_code),
                    response_text=text,
                )
            if not out.get("ok"):
                raise _RestPostError(
                    f"REST API returned ok=false: {out}",
                    status_code=int(r.status_code),
                    response_text=text,
                )
            return int(out.get("imported") or 0)

        def _is_retryable(err: _RestPostError) -> bool:
            sc = err.status_code
            if sc is None:
                return True
            return int(sc) in {408, 429, 500, 502, 503, 504}

        def _post_with_fallback(batch: list[dict[str, Any]]) -> None:
            nonlocal posted, skipped, api_batch_size
            ids = _t2s_game_ids(batch)
            ids_s = _format_t2s_ids(ids)
            max_attempts = max(1, api_max_retries)
            last_err: Optional[_RestPostError] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    imported = _post_once(batch)
                    posted += int(imported)
                    return
                except _RestPostError as e:
                    last_err = e
                    # For nginx 504s, retries at the same batch size usually just waste time; split instead.
                    if int(e.status_code or 0) == 504 and api_split_on_error and len(batch) > 1:
                        break
                    if attempt >= max_attempts or not _is_retryable(e):
                        break
                    sleep_s = api_retry_backoff_s * (2.0 ** float(attempt - 1))
                    log(
                        f"Warning: REST post failed (attempt {attempt}/{max_attempts}; "
                        f"batch_size={len(batch)} timetoscore_game_ids={ids_s}): {e}. "
                        f"Retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)

            assert last_err is not None
            failure_path = _write_failure_payload(batch, last_err)
            if failure_path:
                log(f"Saved failing REST payload: {failure_path}")

            if api_split_on_error and len(batch) > 1:
                new_size = max(1, len(batch) // 2)
                if new_size < api_batch_size:
                    api_batch_size = new_size
                    log(
                        f"Reducing --api-batch-size to {api_batch_size} due to REST failures (was larger)."
                    )
                mid = len(batch) // 2
                log(
                    f"REST batch failed; splitting batch_size={len(batch)} timetoscore_game_ids={ids_s} "
                    f"into {mid}+{len(batch) - mid}..."
                )
                _post_with_fallback(batch[:mid])
                _post_with_fallback(batch[mid:])
                return

            if api_skip_failed_games:
                skipped += len(batch)
                log(
                    f"Warning: skipping {len(batch)} game(s) due to REST failure "
                    f"(timetoscore_game_ids={ids_s}): {last_err}"
                )
                return

            raise last_err

        _post_with_fallback(api_games_batch)
        api_games_batch = []

    for gid in game_ids:
        now = time.time()
        if (now - last_heartbeat) >= 30.0:
            pct = (count / total * 100.0) if total else 100.0
            log(f"Working... next game_id={gid} ({count}/{total}, {pct:.1f}%)")
            last_heartbeat = now
        if args.limit is not None and count >= int(args.limit):
            break
        fb = fallback_by_gid.get(int(gid))
        fb_hg = tts_norm.parse_int_or_none((fb or {}).get("homeGoals"))
        fb_ag = tts_norm.parse_int_or_none((fb or {}).get("awayGoals"))
        fb_has_result = (fb_hg is not None) or (fb_ag is not None)
        max_attempts = max(1, int(args.t2s_max_attempts or 1))
        delay_s = max(0.0, float(args.t2s_initial_backoff_s or 0.0))
        max_delay_s = max(0.0, float(args.t2s_max_backoff_s or 0.0))

        schedule_only_game = False
        stats: dict[str, Any] = {}
        attempt = 0
        if not bool(args.scrape):
            cached = _get_cached_stats(int(gid))
            if cached:
                stats = dict(cached or {})

        if not stats:
            for attempt in range(1, max_attempts + 1):
                try:
                    stats = tts_direct.scrape_game_stats(
                        args.source, game_id=int(gid), season_id=season_id
                    )
                    if stats:
                        _set_cached_stats(int(gid), stats)
                except Exception as e:
                    stats = {}
                    # If the game has a recorded score in the schedule, missing stats should be treated as fatal
                    # so we can fix the scraper and backfill correctly.
                    if fb_has_result and attempt < max_attempts:
                        base_sleep_s = min(max_delay_s, delay_s) if max_delay_s else delay_s
                        jitter = random.uniform(0.85, 1.25)
                        sleep_s = (
                            min(max_delay_s, base_sleep_s * jitter)
                            if max_delay_s
                            else base_sleep_s * jitter
                        )
                        log(
                            f"Warning: scrape_game_stats failed (attempt {attempt}/{max_attempts}) "
                            f"for source={args.source} season_id={season_id} game_id={gid} "
                            f"(fallback score homeGoals={fb_hg}, awayGoals={fb_ag}): {type(e).__name__}: {e}. "
                            f"Retrying in {sleep_s:.1f}s..."
                        )
                        time.sleep(sleep_s)
                        delay_s = min(max_delay_s, delay_s * 2.0) if max_delay_s else delay_s * 2.0
                        continue
                    if fb_has_result:
                        if (
                            bool(args.allow_schedule_only)
                            and type(e).__name__ == "MissingStatsError"
                        ):
                            schedule_only_game = True
                            log(
                                f"Warning: importing schedule-only game (missing boxscore after {attempt}/{max_attempts} attempts): "
                                f"source={args.source} season_id={season_id} game_id={gid} "
                                f"(fallback score homeGoals={fb_hg}, awayGoals={fb_ag}): {type(e).__name__}: {e}"
                            )
                            break
                        raise RuntimeError(
                            f"TimeToScore scrape_game_stats failed for a game with a recorded result: "
                            f"source={args.source} season_id={season_id} game_id={gid} "
                            f"(fallback score homeGoals={fb_hg}, awayGoals={fb_ag}): {type(e).__name__}: {e}"
                        ) from e
                    break

                # If the game has a non-zero score, we must be able to attribute per-player goals/assists
                # from the TimeToScore payload. If not, retry (often throttling / partial HTML).
                t1_score_try = tts_norm.parse_int_or_none(stats.get("homeGoals"))
                t2_score_try = tts_norm.parse_int_or_none(stats.get("awayGoals"))
                if t1_score_try is None and fb_hg is not None:
                    t1_score_try = fb_hg
                if t2_score_try is None and fb_ag is not None:
                    t2_score_try = fb_ag
                goal_total_try = int(t1_score_try or 0) + int(t2_score_try or 0)
                ga_rows_try = [
                    agg
                    for agg in tts_norm.aggregate_goals_assists(stats)
                    if str(agg.get("name") or "").strip()
                ]
                ga_sum_try = sum(int(r.get("goals") or 0) for r in ga_rows_try)
                if goal_total_try > 0 and ga_sum_try == 0:
                    has_any_boxscore_rows = False
                    for k in (
                        "homePlayers",
                        "awayPlayers",
                        "homeScoring",
                        "awayScoring",
                        "homePenalties",
                        "awayPenalties",
                        "homeShootout",
                        "awayShootout",
                        "homeSkaters",
                        "awaySkaters",
                        "home_skaters",
                        "away_skaters",
                    ):
                        v = stats.get(k)
                        if isinstance(v, list) and v:
                            has_any_boxscore_rows = True
                            break
                    # If the page has *no* boxscore tables at all, it's often a permanent "schedule-only" game state.
                    # Don't burn full backoff retries in that case.
                    if (
                        bool(args.allow_schedule_only)
                        and not has_any_boxscore_rows
                        and attempt >= min(max_attempts, 2)
                    ):
                        schedule_only_game = True
                        log(
                            f"Warning: importing schedule-only game (scored but no boxscore data after {attempt}/{max_attempts} attempts): "
                            f"source={args.source} season_id={season_id} game_id={gid} "
                            f"(homeGoals={t1_score_try}, awayGoals={t2_score_try})."
                        )
                        break
                    if attempt < max_attempts:
                        base_sleep_s = min(max_delay_s, delay_s) if max_delay_s else delay_s
                        jitter = random.uniform(0.85, 1.25)
                        sleep_s = (
                            min(max_delay_s, base_sleep_s * jitter)
                            if max_delay_s
                            else base_sleep_s * jitter
                        )
                        log(
                            f"Warning: scored game but no scoring attribution (attempt {attempt}/{max_attempts}) "
                            f"for source={args.source} season_id={season_id} game_id={gid} "
                            f"(homeGoals={t1_score_try}, awayGoals={t2_score_try}; "
                            f"homeScoring={len(stats.get('homeScoring') or [])}, awayScoring={len(stats.get('awayScoring') or [])}). "
                            f"Retrying in {sleep_s:.1f}s..."
                        )
                        time.sleep(sleep_s)
                        delay_s = min(max_delay_s, delay_s * 2.0) if max_delay_s else delay_s * 2.0
                        continue
                    raise RuntimeError(
                        f"TimeToScore scrape returned a scored game but no scoring attribution: "
                        f"source={args.source} season_id={season_id} game_id={gid} "
                        f"(homeGoals={t1_score_try}, awayGoals={t2_score_try})."
                    )
                break

        # If we forced a scrape, still persist the result.
        if bool(args.scrape) and stats:
            _set_cached_stats(int(gid), stats)

        if not stats and not fb:
            skipped += 1
            continue
        if schedule_only_game:
            schedule_only += 1

        home_name = (
            str(stats.get("home") or "").strip() or str((fb or {}).get("home") or "").strip()
        )
        away_name = (
            str(stats.get("away") or "").strip() or str((fb or {}).get("away") or "").strip()
        )
        if not home_name or not away_name:
            skipped += 1
            continue

        fb_division_name = str((fb or {}).get("division_name") or "").strip() or None
        fb_division_id = None
        fb_conference_id = None
        try:
            fb_division_id = (
                int((fb or {}).get("division_id"))
                if (fb or {}).get("division_id") is not None
                else None
            )
        except Exception:
            fb_division_id = None
        try:
            fb_conference_id = (
                int((fb or {}).get("conference_id"))
                if (fb or {}).get("conference_id") is not None
                else None
            )
        except Exception:
            fb_conference_id = None

        home_name_raw = str(home_name or "").strip()
        away_name_raw = str(away_name or "").strip()
        home_div_name, home_div_id, home_conf_id, home_tts_id = resolve_team_division_meta(
            home_name_raw, fb_division_name
        )
        away_div_name, away_div_id, away_conf_id, away_tts_id = resolve_team_division_meta(
            away_name_raw, fb_division_name
        )

        # Use per-team division for disambiguation, not the game's division block.
        home_name = canonical_team_name(home_name_raw, home_div_name)
        away_name = canonical_team_name(away_name_raw, away_div_name)

        starts_at = parse_starts_at(args.source, stats=stats, fallback=fb)
        location = (
            str(stats.get("location") or "").strip()
            or str((fb or {}).get("rink") or "").strip()
            or None
        )
        t1_score = tts_norm.parse_int_or_none(stats.get("homeGoals"))
        t2_score = tts_norm.parse_int_or_none(stats.get("awayGoals"))
        if t1_score is None and (fb or {}).get("homeGoals") is not None:
            t1_score = tts_norm.parse_int_or_none((fb or {}).get("homeGoals"))
        if t2_score is None and (fb or {}).get("awayGoals") is not None:
            t2_score = tts_norm.parse_int_or_none((fb or {}).get("awayGoals"))

        # Build a basic events timeline from TimeToScore data (goals + penalties + PP/PK spans)
        # so the webapp can show game events even before any spreadsheets are uploaded.
        def _side_label(side: str) -> str:
            return "Home" if str(side).strip().lower() == "home" else "Away"

        def _norm_jersey(val: Any) -> Optional[str]:
            s = str(val or "").strip()
            if not s:
                return None
            # Bench penalties are often represented as "B" on TimeToScore.
            if s.upper() in {"B", "BENCH"}:
                return "B"
            m = re.search(r"(\d+)", s)
            return m.group(1) if m else None

        roster_home = list(tts_norm.extract_roster(stats, "home"))
        roster_away = list(tts_norm.extract_roster(stats, "away"))
        num_to_name_home = {
            str(m.group(1)): str(p.get("name") or "").strip()
            for p in roster_home
            if (m := re.search(r"(\d+)", str(p.get("number") or "")))
            and str(p.get("name") or "").strip()
        }
        num_to_name_away = {
            str(m.group(1)): str(p.get("name") or "").strip()
            for p in roster_away
            if (m := re.search(r"(\d+)", str(p.get("number") or "")))
            and str(p.get("name") or "").strip()
        }

        def _infer_period_len_s(stats_in: dict[str, Any]) -> int:
            raw = str(stats_in.get("periodLength") or "").strip()
            try:
                v = int(float(raw))
                if v > 0:
                    return int(v) * 60
            except Exception:
                pass
            return 15 * 60

        period_len_s = _infer_period_len_s(stats)

        # Collect raw penalty records per team and compute per-player PIM.
        penalties_by_side: dict[str, list[dict[str, Any]]] = {"home": [], "away": []}
        pim_by_player_name: dict[str, int] = {}
        for side_key, roster_map in (("home", num_to_name_home), ("away", num_to_name_away)):
            pen_rows = stats.get(f"{side_key}Penalties") or []
            if not isinstance(pen_rows, list):
                pen_rows = []
            for prow in pen_rows:
                if not isinstance(prow, dict):
                    continue
                per = _parse_period_token(prow.get("period"))
                if per is None:
                    continue
                jersey = _norm_jersey(prow.get("number"))
                minutes = tts_norm.parse_int_or_none(prow.get("minutes"))
                start_txt = str(prow.get("start") or prow.get("offIce") or "").strip()
                end_txt = str(prow.get("end") or prow.get("onIce") or "").strip()
                start_s = _parse_mmss_to_seconds(start_txt, period_len_s=period_len_s)
                end_s = _parse_mmss_to_seconds(end_txt, period_len_s=period_len_s)
                inf = str(prow.get("infraction") or "").strip()
                rec = {
                    "side": side_key,
                    "period": int(per),
                    "jersey": jersey,
                    "minutes": int(minutes) if minutes is not None else None,
                    "start_txt": start_txt,
                    "end_txt": end_txt,
                    "start_s": start_s,
                    "end_s": end_s,
                    "infraction": inf,
                }
                penalties_by_side[side_key].append(rec)
                if jersey and minutes is not None:
                    nm = roster_map.get(jersey)
                    if nm:
                        pim_by_player_name[nm] = pim_by_player_name.get(nm, 0) + int(minutes)

        # Build basic goal/assist events from scoring tables when available.
        goal_events, assist_events = build_timetoscore_goal_and_assist_events(
            stats=stats,
            period_len_s=period_len_s,
            num_to_name_home=num_to_name_home,
            num_to_name_away=num_to_name_away,
        )

        # Determine per-period time mode and fill missing penalty end times (best-effort).
        penalties_events_rows: list[dict[str, Any]] = []

        def _jersey_detail_prefix(jersey: Any) -> str:
            # Keep import_key stable: penalty import keys include Details, so avoid injecting "#B"
            # for bench penalties.
            j = str(jersey or "").strip()
            return f"#{j}" if j.isdigit() else ""

        for side_key, recs in penalties_by_side.items():
            for rec in recs:
                per = int(rec["period"])
                start_s = rec.get("start_s")
                end_s = rec.get("end_s")
                minutes = rec.get("minutes")
                if start_s is not None and end_s is None and minutes is not None:
                    # Best-effort: assume the sheet uses a running/scoreboard clock and fill end time.
                    # We'll refine direction after inferring elapsed-vs-remaining for the period.
                    rec["end_s_guess_delta"] = int(minutes) * 60
                penalties_events_rows.append(
                    {
                        "Event Type": "Penalty",
                        "Source": "timetoscore",
                        "Team Side": _side_label(side_key),
                        "For/Against": "Against",
                        "Team Rel": _side_label(side_key),
                        "Team Raw": _side_label(side_key),
                        "Period": per,
                        "Game Time": rec.get("start_txt") or "",
                        "Game Seconds": start_s if start_s is not None else "",
                        "Game Seconds End": "",
                        "Details": " ".join(
                            [
                                x
                                for x in [
                                    _jersey_detail_prefix(rec.get("jersey")),
                                    str(rec.get("infraction") or "").strip(),
                                    (
                                        f"{int(rec.get('minutes'))}m"
                                        if rec.get("minutes") is not None
                                        else ""
                                    ),
                                    (
                                        f"(end P{int(rec.get('end_period'))} {rec.get('end_txt')})"
                                        if rec.get("end_txt")
                                        and rec.get("end_period")
                                        and int(rec.get("end_period") or 0) != int(per)
                                        else (
                                            f"(end {rec.get('end_txt')})"
                                            if rec.get("end_txt")
                                            else ""
                                        )
                                    ),
                                ]
                                if x
                            ]
                        ).strip(),
                        "Attributed Players": (
                            num_to_name_home.get(str(rec.get("jersey")))
                            if side_key == "home" and rec.get("jersey")
                            else (
                                num_to_name_away.get(str(rec.get("jersey")))
                                if side_key == "away" and rec.get("jersey")
                                else ""
                            )
                        ),
                        "Attributed Jerseys": rec.get("jersey") or "",
                    }
                )

        # Fill missing end times (and correct cross-period end times) using the TimeToScore
        # scoreboard clock model (counts down each period).
        mode_by_period: dict[int, str] = {}
        for p in range(1, 6):
            if any(int(r.get("Period") or 0) == p for r in penalties_events_rows) or any(
                int(g.get("Period") or 0) == p for g in goal_events
            ):
                mode_by_period[p] = "remaining"

        for side_key, recs in penalties_by_side.items():
            for rec in recs:
                start_s = rec.get("start_s")
                per = int(rec["period"])
                mode = mode_by_period.get(per, "elapsed")

                end_s = rec.get("end_s")
                if end_s is None:
                    if start_s is None:
                        continue
                    delta = rec.get("end_s_guess_delta")
                    if not isinstance(delta, int) or delta <= 0:
                        continue
                    per_end, end_s2 = _compute_end_from_start_and_delta(
                        start_period=int(per),
                        start_within_s=int(start_s),
                        delta_s=int(delta),
                        mode=str(mode),
                        period_len_s=int(period_len_s),
                    )
                    rec["end_s"] = int(end_s2)
                    if not str(rec.get("end_txt") or "").strip():
                        rec["end_txt"] = _format_mmss(int(end_s2))
                    if int(per_end) != int(per):
                        rec["end_period"] = int(per_end)
                else:
                    per_end = _infer_end_period_for_within_times(
                        start_period=int(per),
                        start_within_s=int(start_s) if start_s is not None else None,
                        end_within_s=int(end_s),
                        mode=str(mode),
                        period_len_s=int(period_len_s),
                    )
                    if int(per_end) != int(per):
                        rec["end_period"] = int(per_end)

        # Emit explicit "Penalty Expired" events for each penalty with a known end time.
        penalty_expired_events_rows: list[dict[str, Any]] = []
        for side_key, recs in penalties_by_side.items():
            for rec in recs:
                end_s = rec.get("end_s")
                if end_s is None:
                    continue
                per = int(rec.get("end_period") or rec.get("period") or 0)
                details = " ".join(
                    [
                        x
                        for x in [
                            "Expired",
                            _jersey_detail_prefix(rec.get("jersey")),
                            str(rec.get("infraction") or "").strip(),
                            (
                                f"{int(rec.get('minutes'))}m"
                                if rec.get("minutes") is not None
                                else ""
                            ),
                        ]
                        if x
                    ]
                ).strip()
                penalty_expired_events_rows.append(
                    {
                        "Event Type": "Penalty Expired",
                        "Source": "timetoscore",
                        "Team Side": _side_label(side_key),
                        "For/Against": "For",
                        "Team Rel": _side_label(side_key),
                        "Team Raw": _side_label(side_key),
                        "Period": per,
                        "Game Time": rec.get("end_txt") or _format_mmss(int(end_s)),
                        "Game Seconds": int(end_s),
                        "Game Seconds End": "",
                        "Details": details,
                        "Attributed Players": (
                            num_to_name_home.get(str(rec.get("jersey")))
                            if side_key == "home" and rec.get("jersey")
                            else (
                                num_to_name_away.get(str(rec.get("jersey")))
                                if side_key == "away" and rec.get("jersey")
                                else ""
                            )
                        ),
                        "Attributed Jerseys": rec.get("jersey") or "",
                    }
                )

        # Goalie changes (including starting goalie) are only exposed on some TimeToScore score sheets.
        goalie_change_events_rows: list[dict[str, Any]] = []
        for side_key in ("home", "away"):
            gc_rows = stats.get(f"{side_key}GoalieChanges") or []
            if not isinstance(gc_rows, list):
                gc_rows = []
            for gc in gc_rows:
                if not isinstance(gc, dict):
                    continue
                per = _parse_period_token(gc.get("period"))
                if per is None:
                    continue
                time_txt = str(gc.get("time") or "").strip()
                time_s = _parse_mmss_to_seconds(time_txt, period_len_s=period_len_s)
                details = str(gc.get("details") or "").strip()
                goalie_name = re.sub(r"(?i)\bstarting\b", "", details).strip()
                if re.search(r"(?i)\bempty\s+net\b", goalie_name):
                    goalie_name = ""

                goalie_change_events_rows.append(
                    {
                        "Event Type": "Goalie Change",
                        "Source": "timetoscore",
                        "Team Side": _side_label(side_key),
                        "For/Against": "",
                        "Team Rel": _side_label(side_key),
                        "Team Raw": _side_label(side_key),
                        "Period": int(per),
                        "Game Time": time_txt,
                        "Game Seconds": time_s if time_s is not None else "",
                        "Game Seconds End": "",
                        "Details": details,
                        "Attributed Players": goalie_name,
                        "Attributed Jerseys": "",
                    }
                )

        events_headers = [
            "Event Type",
            "Source",
            "Team Raw",
            "Team Side",
            "For/Against",
            "Team Rel",
            "Period",
            "Game Time",
            "Game Seconds",
            "Game Seconds End",
            "Details",
            "Attributed Players",
            "Attributed Jerseys",
        ]
        events_rows = (
            list(goal_events)
            + list(assist_events)
            + list(goalie_change_events_rows)
            + list(penalties_events_rows)
            + list(penalty_expired_events_rows)
        )
        events_rows.sort(
            key=lambda r: (
                int(r.get("Period") or 0),
                int(r.get("Game Seconds") or 0) if str(r.get("Game Seconds") or "").strip() else 0,
                str(r.get("Event Type") or ""),
            )
        )
        events_csv_text = _to_csv_text(events_headers, events_rows)

        if rest_mode:
            # Prefer mapping tournament games involving an External division to that External division,
            # otherwise keep the team's primary division.
            if _is_external_division_name(home_div_name):
                game_div_name = home_div_name
                game_div_id = home_div_id if home_div_id is not None else fb_division_id
                game_conf_id = home_conf_id if home_conf_id is not None else fb_conference_id
            elif _is_external_division_name(away_div_name):
                game_div_name = away_div_name
                game_div_id = away_div_id if away_div_id is not None else fb_division_id
                game_conf_id = away_conf_id if away_conf_id is not None else fb_conference_id
            else:
                game_div_name = home_div_name or away_div_name or fb_division_name
                game_div_id = (
                    home_div_id
                    if home_div_id is not None
                    else away_div_id if away_div_id is not None else fb_division_id
                )
                game_conf_id = (
                    home_conf_id
                    if home_conf_id is not None
                    else away_conf_id if away_conf_id is not None else fb_conference_id
                )

            raw_t2s_type = str((fb or {}).get("type") or "").strip() or None
            import_game_type_name = raw_t2s_type
            if str(args.source or "").strip().lower() == "caha" and int(t2s_league_id or 0) == 18:
                import_game_type_name = "Tournament"

            home_logo_url = _logo_url(home_tts_id)
            away_logo_url = _logo_url(away_tts_id)
            home_logo_b64 = None
            home_logo_ct = None
            away_logo_b64 = None
            away_logo_ct = None
            if not args.no_import_logos:
                if home_tts_id is not None:
                    tid_i = int(home_tts_id)
                    if bool(args.replace) or tid_i not in sent_logo_ids:
                        home_logo_b64, home_logo_ct = _logo_b64_and_type(tid_i)
                        if home_logo_b64:
                            sent_logo_ids.add(tid_i)
                if away_tts_id is not None:
                    tid_i = int(away_tts_id)
                    if bool(args.replace) or tid_i not in sent_logo_ids:
                        away_logo_b64, away_logo_ct = _logo_b64_and_type(tid_i)
                        if away_logo_b64:
                            sent_logo_ids.add(tid_i)

            api_games_batch.append(
                {
                    "home_name": home_name,
                    "away_name": away_name,
                    "game_type_name": import_game_type_name,
                    "timetoscore_type": raw_t2s_type,
                    "division_name": game_div_name,
                    "division_id": game_div_id,
                    "conference_id": game_conf_id,
                    "home_division_name": home_div_name,
                    "home_division_id": home_div_id,
                    "home_conference_id": home_conf_id,
                    "away_division_name": away_div_name,
                    "away_division_id": away_div_id,
                    "away_conference_id": away_conf_id,
                    "starts_at": starts_at,
                    "location": location,
                    "home_score": t1_score,
                    "away_score": t2_score,
                    "timetoscore_game_id": int(gid),
                    "season_id": int(season_id),
                    "home_logo_url": home_logo_url,
                    "away_logo_url": away_logo_url,
                    "home_logo_b64": home_logo_b64,
                    "home_logo_content_type": home_logo_ct,
                    "away_logo_b64": away_logo_b64,
                    "away_logo_content_type": away_logo_ct,
                    "home_roster": list(tts_norm.extract_roster(stats, "home")),
                    "away_roster": list(tts_norm.extract_roster(stats, "away")),
                    "events_csv": events_csv_text,
                }
            )
            if len(api_games_batch) >= api_batch_size:
                _post_batch()

            count += 1
            if count % 25 == 0 or count == total:
                elapsed = max(0.001, time.time() - started)
                rate = count / elapsed
                pct = (count / total * 100.0) if total else 100.0
                log(
                    f"Progress: scraped {count}/{total} ({pct:.1f}%) games, posted={posted}, skipped={skipped}, {rate:.2f} games/s"
                )
            continue

        assert user_id is not None
        assert league_id is not None

        assert m is not None
        team1_id = ensure_team(None, user_id, home_name, is_external=True)
        team2_id = ensure_team(None, user_id, away_name, is_external=True)
        if logo_dir is not None:
            try:
                if home_tts_id is not None:
                    _ensure_team_logo(
                        None,
                        team_db_id=team1_id,
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        league_id=int(t2s_league_id) if t2s_league_id is not None else None,
                        tts_team_id=int(home_tts_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
                if away_tts_id is not None:
                    _ensure_team_logo(
                        None,
                        team_db_id=team2_id,
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        league_id=int(t2s_league_id) if t2s_league_id is not None else None,
                        tts_team_id=int(away_tts_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
            except Exception:
                pass

        notes = f"Imported from TimeToScore {args.source} game_id={gid}"
        raw_t2s_type = str((fb or {}).get("type") or "").strip() or None
        game_type_name = raw_t2s_type
        if str(args.source or "").strip().lower() == "caha" and int(t2s_league_id or 0) == 18:
            game_type_name = "Tournament"
        game_type_id = None
        if game_type_name:
            # Map TimeToScore schedule Type to our canonical game type names.
            sl = game_type_name.casefold()
            if sl.startswith("regular"):
                nm = "Regular Season"
            elif sl.startswith("preseason"):
                nm = "Preseason"
            elif sl.startswith("exhibition"):
                nm = "Exhibition"
            elif sl.startswith("tournament"):
                nm = "Tournament"
            else:
                nm = game_type_name
            gt, _created = m.GameType.objects.get_or_create(
                name=str(nm), defaults={"is_default": False}
            )
            game_type_id = int(gt.id)
        game_db_id = upsert_hky_game(
            None,
            user_id=user_id,
            team1_id=team1_id,
            team2_id=team2_id,
            game_type_id=game_type_id,
            starts_at=starts_at,
            location=location,
            team1_score=t1_score,
            team2_score=t2_score,
            replace=bool(args.replace),
            notes=notes,
            timetoscore_game_id=int(gid),
        )

        map_team_to_league_with_division(
            None,
            league_id=league_id,
            team_id=team1_id,
            division_name=home_div_name,
            division_id=home_div_id if home_div_id is not None else fb_division_id,
            conference_id=home_conf_id if home_conf_id is not None else fb_conference_id,
        )
        map_team_to_league_with_division(
            None,
            league_id=league_id,
            team_id=team2_id,
            division_name=away_div_name,
            division_id=away_div_id if away_div_id is not None else fb_division_id,
            conference_id=away_conf_id if away_conf_id is not None else fb_conference_id,
        )
        # For a per-division schedule, attribute games to the home team's division when available.
        game_div_name = home_div_name or fb_division_name
        game_div_id = home_div_id if home_div_id is not None else fb_division_id
        game_conf_id = home_conf_id if home_conf_id is not None else fb_conference_id
        map_game_to_league_with_division(
            None,
            league_id=league_id,
            game_id=game_db_id,
            division_name=game_div_name,
            division_id=game_div_id,
            conference_id=game_conf_id,
        )

        # Rosters (optional for schedule-only rows)
        roster_pids_by_team: dict[int, set[int]] = {int(team1_id): set(), int(team2_id): set()}
        for row in tts_norm.extract_roster(stats, "home"):
            pid = ensure_player(
                conn,
                user_id=user_id,
                team_id=team1_id,
                name=row["name"],
                jersey=row["number"],
                position=row["position"],
            )
            roster_pids_by_team[int(team1_id)].add(int(pid))
        for row in tts_norm.extract_roster(stats, "away"):
            pid = ensure_player(
                conn,
                user_id=user_id,
                team_id=team2_id,
                name=row["name"],
                jersey=row["number"],
                position=row["position"],
            )
            roster_pids_by_team[int(team2_id)].add(int(pid))

        # Credit GP (roster presence).
        if roster_pids_by_team:
            now = dt.datetime.now()
            links = []
            for tid, pids in roster_pids_by_team.items():
                for pid in sorted(pids):
                    links.append(
                        m.HkyGamePlayer(
                            game_id=int(game_db_id),
                            player_id=int(pid),
                            team_id=int(tid),
                            created_at=now,
                            updated_at=None,
                        )
                    )
            if links:
                m.HkyGamePlayer.objects.bulk_create(links, ignore_conflicts=True)
        if not args.no_cleanup_bogus_players:
            if int(team1_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(None, user_id=user_id, team_id=int(team1_id))
                cleaned_team_ids.add(int(team1_id))
            if int(team2_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(None, user_id=user_id, team_id=int(team2_id))
                cleaned_team_ids.add(int(team2_id))

        if str(events_csv_text or "").strip():
            try:
                from tools.webapp.django_app import views as web_views  # type: ignore
            except Exception:  # pragma: no cover
                from django_app import views as web_views  # type: ignore

            web_views._upsert_game_event_rows_from_events_csv(
                game_id=int(game_db_id),
                events_csv=str(events_csv_text),
                replace=bool(args.replace),
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

        count += 1
        if count % 25 == 0 or count == total:
            elapsed = max(0.001, time.time() - started)
            rate = count / elapsed
            pct = (count / total * 100.0) if total else 100.0
            log(
                f"Progress: {count}/{total} ({pct:.1f}%) games, skipped={skipped}, {rate:.2f} games/s"
            )

    if rest_mode:
        _post_batch()

    if rest_mode:
        log(
            f"Import complete. Scraped {count} games, posted={posted}, skipped={skipped}, schedule_only={schedule_only}."
        )
    else:
        log(
            f"Import complete. Imported {count} games, skipped={skipped}, schedule_only={schedule_only}."
        )
    if (not rest_mode) and (not args.no_cleanup_bogus_players):
        try:
            assert league_id is not None
            assert user_id is not None
            assert m is not None
            all_team_ids = sorted(
                set(
                    m.LeagueTeam.objects.filter(
                        league_id=int(league_id), team__user_id=int(user_id)
                    )
                    .values_list("team_id", flat=True)
                    .distinct()
                )
            )
            moved_total = 0
            for tid in all_team_ids:
                moved_total += _cleanup_numeric_named_players(
                    None, user_id=user_id, team_id=int(tid)
                )
            if moved_total:
                log(f"Cleaned bogus numeric-name players: migrated {moved_total} stat rows.")
        except Exception:
            pass
    if (not rest_mode) and args.refresh_team_metadata:
        log("Refreshing league team metadata (divisions/logos)...")
        # Update league_teams divisions based on TimeToScore team list, and fill missing logos.
        try:
            assert league_id is not None
            assert user_id is not None
            assert m is not None
            league_team_rows = list(
                m.LeagueTeam.objects.filter(league_id=int(league_id), team__user_id=int(user_id))
                .select_related("team")
                .values_list("team_id", "team__name", "team__logo_path")
            )
            updated_div = 0
            updated_logo = 0
            for team_db_id, team_name, logo_path in league_team_rows:
                nm = str(team_name or "").strip()
                # If canonical disambiguation is present, strip " (Division)" suffix for matching.
                base_name = nm
                if base_name.endswith(")") and "(" in base_name:
                    base_name = base_name.rsplit("(", 1)[0].strip()
                div_name, div_id, conf_id, tts_id = resolve_team_division_meta(base_name, None)
                if div_name:
                    updates: dict[str, Any] = {"division_name": str(div_name)}
                    if div_id is not None:
                        updates["division_id"] = int(div_id)
                    if conf_id is not None:
                        updates["conference_id"] = int(conf_id)
                    m.LeagueTeam.objects.filter(
                        league_id=int(league_id), team_id=int(team_db_id)
                    ).update(**updates)
                    updated_div += 1
                if logo_dir is not None and (not logo_path) and tts_id is not None:
                    _ensure_team_logo(
                        None,
                        team_db_id=int(team_db_id),
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        league_id=int(t2s_league_id) if t2s_league_id is not None else None,
                        tts_team_id=int(tts_id),
                        logo_dir=Path(logo_dir),
                        replace=False,
                        tts_direct=tts_direct,
                    )
                    updated_logo += 1
            # Also align league_games division to the home team division when possible.
            league_team_meta = {
                int(tid): (dn, did, cid)
                for tid, dn, did, cid in m.LeagueTeam.objects.filter(
                    league_id=int(league_id)
                ).values_list("team_id", "division_name", "division_id", "conference_id")
            }
            league_games = list(
                m.LeagueGame.objects.filter(league_id=int(league_id)).select_related("game")
            )
            to_update = []
            for lg in league_games:
                meta = league_team_meta.get(int(lg.game.team1_id))
                if not meta:
                    continue
                dn, did, cid = meta
                changed = False
                if dn is not None:
                    lg.division_name = dn
                    changed = True
                if did is not None:
                    lg.division_id = int(did)
                    changed = True
                if cid is not None:
                    lg.conference_id = int(cid)
                    changed = True
                if changed:
                    to_update.append(lg)
            if to_update:
                m.LeagueGame.objects.bulk_update(
                    to_update, ["division_name", "division_id", "conference_id"], batch_size=500
                )
            log(f"Refreshed teams: divisions_updated={updated_div} logos_checked={updated_logo}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

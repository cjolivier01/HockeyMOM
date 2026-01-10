#!/usr/bin/env python3
"""Import TimeToScore data into the HockeyMOM webapp DB (optionally using a local sqlite cache).

This script scrapes TimeToScore directly via `hmlib.time2score` and upserts:
- teams (as external teams owned by the specified user)
- games (hky_games)
- players + player_stats (goals/assists derived from TimeToScore game pages)

By default it targets CAHA youth (league=3). SharksIce (adult, league=1) is also supported.
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
import re
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
            updates: dict[str, Any] = {"source": source, "external_key": external_key, "updated_at": now}
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
        defaults={"division_name": dn, "division_id": division_id, "conference_id": conference_id, "sort_order": sort_order},
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
        t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
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
        url = tts_direct.scrape_team_logo_url(str(source), season_id=int(season_id), team_id=int(tts_team_id))
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

    Migrate any player_stats rows from bogus numeric-name players to the real player
    matching jersey_number, then delete the bogus player records if unused.
    """
    del conn
    _django_orm, m = _orm_modules()
    from django.db import transaction

    moved = 0
    bogus = list(
        m.Player.objects.filter(user_id=int(user_id), team_id=int(team_id), name__regex=r"^[0-9]+$")
        .values_list("id", "name")
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

            bogus_stats = list(m.PlayerStat.objects.filter(player_id=int(bogus_id)).values_list("id", "game_id"))
            for ps_id, gid in bogus_stats:
                if m.PlayerStat.objects.filter(game_id=int(gid), player_id=int(real_id)).exists():
                    m.PlayerStat.objects.filter(id=int(ps_id)).delete()
                else:
                    m.PlayerStat.objects.filter(id=int(ps_id)).update(player_id=int(real_id))
                    moved += 1

            if not m.PlayerStat.objects.filter(player_id=int(bogus_id)).exists():
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
            token_json_nospace = f"\"timetoscore_game_id\":{tts_int}"
            token_json_space = f"\"timetoscore_game_id\": {tts_int}"
            for token in (token_json_nospace, token_json_space, token_plain):
                game = m.HkyGame.objects.filter(user_id=int(user_id), notes__contains=str(token)).first()
                if game is not None:
                    break

        if game is None:
            merged_notes = json.dumps(notes_json_fields, sort_keys=True) if notes_json_fields else (notes or None)
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
        merged_notes = _merge_notes(existing_notes, notes_json_fields) if notes_json_fields else (existing_notes or "")

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
    nm = (name or "").strip()
    if not nm:
        nm = "UNKNOWN"
    del conn
    _django_orm, m = _orm_modules()
    now = dt.datetime.now()

    p = m.Player.objects.filter(user_id=int(user_id), team_id=int(team_id), name=str(nm)).first()
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
    from django.db import transaction

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
        gt, _created = m.GameType.objects.get_or_create(name=str(nm), defaults={"is_default": False})
        return int(gt.id)

    def _ensure_player_for_import(
        team_id: int, name: str, jersey_number: Optional[str], position: Optional[str]
    ) -> int:
        nm = (name or "").strip()
        if not nm:
            raise ValueError("player name is required")
        now2 = dt.datetime.now()
        p = m.Player.objects.filter(user_id=int(owner_user_id), team_id=int(team_id), name=str(nm)).first()
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

        tts_game_id = int(game["timetoscore_game_id"]) if game.get("timetoscore_game_id") is not None else None
        season_id = int(game["season_id"]) if game.get("season_id") is not None else None
        game_type_id = _ensure_game_type_id(
            game.get("game_type_name") or game.get("game_type") or game.get("timetoscore_type") or game.get("type")
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
        effective_div_id = division_id or _int_or_none(game.get("home_division_id")) or _int_or_none(game.get("away_division_id"))
        effective_conf_id = conference_id or _int_or_none(game.get("home_conference_id")) or _int_or_none(game.get("away_conference_id"))
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
        roster_player_ids_by_team: dict[int, set[int]] = {int(team1_id): set(), int(team2_id): set()}
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

        # Minimal goals/assists stats.
        stats_rows = game.get("player_stats") or []

        played = bool(game.get("is_final")) or (
            team1_score is not None and team2_score is not None
        ) or (isinstance(stats_rows, list) and bool(stats_rows))
        if played:
            to_create = []
            for tid, pids in roster_player_ids_by_team.items():
                for pid in sorted(pids):
                    to_create.append(
                        m.PlayerStat(
                            user_id=int(owner_user_id),
                            team_id=int(tid),
                            game_id=int(gid),
                            player_id=int(pid),
                        )
                    )
            if to_create:
                m.PlayerStat.objects.bulk_create(to_create, ignore_conflicts=True)

        if isinstance(stats_rows, list):
            for srow in stats_rows:
                if not isinstance(srow, dict):
                    continue
                pname = str(srow.get("name") or "").strip()
                if not pname:
                    continue
                goals = int(srow.get("goals") or 0)
                assists = int(srow.get("assists") or 0)
                if goals == 0 and assists == 0:
                    continue

                pid1 = (
                    m.Player.objects.filter(user_id=int(owner_user_id), team_id=int(team1_id), name=str(pname))
                    .values_list("id", flat=True)
                    .first()
                )
                pid2 = (
                    m.Player.objects.filter(user_id=int(owner_user_id), team_id=int(team2_id), name=str(pname))
                    .values_list("id", flat=True)
                    .first()
                )
                team_ref = team1_id
                pid = int(pid1) if pid1 is not None else (int(pid2) if pid2 is not None else None)
                if pid2 is not None and pid1 is None:
                    team_ref = team2_id
                if pid is None:
                    pid = _ensure_player_for_import(int(team_ref), pname, None, None)

                with transaction.atomic():
                    ps, created = m.PlayerStat.objects.get_or_create(
                        game_id=int(gid),
                        player_id=int(pid),
                        defaults={
                            "user_id": int(owner_user_id),
                            "team_id": int(team_ref),
                            "goals": int(goals),
                            "assists": int(assists),
                        },
                    )
                    if not created:
                        if replace:
                            m.PlayerStat.objects.filter(id=ps.id).update(goals=int(goals), assists=int(assists))
                        else:
                            updates3: dict[str, Any] = {}
                            if ps.goals is None:
                                updates3["goals"] = int(goals)
                            if ps.assists is None:
                                updates3["assists"] = int(assists)
                            if updates3:
                                m.PlayerStat.objects.filter(id=ps.id).update(**updates3)

        results.append({"game_id": int(gid), "team1_id": int(team1_id), "team2_id": int(team2_id)})

    return {"ok": True, "league_id": int(league_id), "owner_user_id": int(owner_user_id), "imported": len(results), "results": results}


def parse_starts_at(source: str, *, stats: dict[str, Any], fallback: Optional[dict[str, Any]]) -> Optional[str]:
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
    ap = argparse.ArgumentParser(description="Import TimeToScore into HockeyMOM webapp DB (no sqlite cache)")
    ap.add_argument("--config", default=default_cfg, help="Path to webapp DB config.json")
    ap.add_argument("--source", choices=("caha", "sharksice"), default="caha", help="TimeToScore site")
    ap.add_argument("--season", type=int, default=0, help="Season id (0 = current/latest)")
    ap.add_argument("--list-seasons", action="store_true", help="List seasons and exit")
    ap.add_argument("--list-divisions", action="store_true", help="List divisions for season and exit")

    ap.add_argument("--user-email", required=True, help="Webapp user email that will own imported data")
    ap.add_argument("--user-name", default=None, help="Name for user creation if missing")
    ap.add_argument("--password-hash", default=None, help="Password hash for creating user if missing")
    ap.add_argument("--create-user", action="store_true", help="Create user if missing (requires --password-hash)")

    ap.add_argument("--replace", action="store_true", help="Overwrite existing scores/player_stats")
    ap.add_argument("--no-import-logos", action="store_true", help="Skip downloading and saving team logos")
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
    ap.add_argument("--division", dest="divisions", action="append", default=[], help="Division filter token (repeatable)")
    ap.add_argument("--team", dest="teams", action="append", default=[], help="Team substring filter (repeatable)")
    ap.add_argument("--game-id", dest="game_ids", action="append", default=[], help="Import specific game id (repeatable)")
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
    ap.add_argument("--league-owner-email", default=None, help="Owner of the league (defaults to --user-email)")
    ap.add_argument(
        "--shared",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Set whether the league is shared (default: leave unchanged; used for league creation if missing).",
    )
    ap.add_argument("--share-with", action="append", default=[], help="Emails to add as league viewers (repeatable)")

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
        "--t2s-max-attempts",
        type=int,
        default=4,
        help="Max scrape attempts per game when TimeToScore results look incomplete (throttling/HTML changes).",
    )
    ap.add_argument(
        "--t2s-initial-backoff-s",
        type=float,
        default=1.0,
        help="Initial backoff seconds between TimeToScore scrape attempts.",
    )
    ap.add_argument(
        "--t2s-max-backoff-s",
        type=float,
        default=20.0,
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
            logo_dir = preferred if preferred.exists() else (base_dir / "instance" / "uploads" / "team_logos")

    user_id = None
    if not rest_mode:
        log(f"Resolving user: {args.user_email}")
        user_id = ensure_user(
            None,
            args.user_email,
            name=args.user_name or args.user_email,
            password_hash=(args.password_hash if args.create_user else None),
        )

    seasons = tts_direct.list_seasons(args.source)
    if args.list_seasons:
        for name, sid in sorted(seasons.items(), key=lambda kv: int(kv[1])):
            print(f"{sid}\t{name}")
        return 0

    season_id = int(args.season) or tts_direct.pick_current_season_id(args.source)
    log(f"Using source={args.source} season_id={season_id}")
    hockey_db_dir = Path(args.hockey_db_dir).expanduser()

    def _get_cached_stats(game_id: int) -> Optional[dict[str, Any]]:
        try:
            with _working_directory(hockey_db_dir):
                db = tts_db.Database()
                db.create_tables()
                cached = db.get_cached_game_stats(str(args.source), int(game_id))
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
                    str(args.source),
                    int(game_id),
                    season_id=int(season_id) if season_id is not None else None,
                    stats=dict(stats or {}),
                )
        except Exception:
            pass

    divs = tts_direct.list_divisions(args.source, season_id=season_id)
    if args.list_divisions:
        for d in sorted(divs, key=lambda x: (int(x.division_id), int(x.conference_id), x.name)):
            print(f"{d.division_id}:{d.conference_id}\t{d.name}\tteams={len(d.teams)}")
        return 0

    # Used to disambiguate team names that appear in multiple divisions (common on CAHA).
    def _clean_team_name(name: str) -> str:
        t = str(name or "").replace("\xa0", " ").strip()
        t = t.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2212", "-")
        t = " ".join(t.split())
        return t

    def _norm_team_key(name: str) -> str:
        return _clean_team_name(name).casefold()

    def _clean_division_label(name: str) -> str:
        t = str(name or "").replace("\xa0", " ").strip()
        t = " ".join(t.split())
        return t

    def _norm_div_key(name: str) -> str:
        return _clean_division_label(name).casefold()

    team_name_counts: dict[str, int] = {}
    for d in divs:
        for t in d.teams:
            nm = _clean_team_name((t or {}).get("name") or "")
            if nm:
                k = _norm_team_key(nm)
                team_name_counts[k] = team_name_counts.get(k, 0) + 1

    def canonical_team_name(name: str, division_name: Optional[str]) -> str:
        nm = _clean_team_name(name or "") or "UNKNOWN"
        dn = _clean_division_label(division_name or "")
        if dn and team_name_counts.get(_norm_team_key(nm), 0) > 1:
            return f"{nm} ({dn})"
        return nm

    division_teams: list[dict[str, Any]] = []
    seen_team_names: set[str] = set()
    for d in divs:
        dn = str(d.name or "").strip() or None
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
                    "division_id": did,
                    "conference_id": cid,
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

    def resolve_team_division_meta(name: str, fallback_division_name: Optional[str]) -> tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        """Return (division_name, division_id, conference_id, team_tts_id)."""
        tts_id = resolve_tts_team_id(name, fallback_division_name)
        if tts_id is not None and int(tts_id) in tts_team_div_meta_by_id:
            dn, did, cid = tts_team_div_meta_by_id[int(tts_id)]
            return dn or None, int(did), int(cid), int(tts_id)
        # If the team name is unique across all divisions, assign its division without relying on the game.
        ids = tts_team_ids_by_name.get(_norm_team_key(name or "")) or []
        if len(ids) == 1 and int(ids[0]) in tts_team_div_meta_by_id:
            dn, did, cid = tts_team_div_meta_by_id[int(ids[0])]
            return dn or None, int(did), int(cid), int(ids[0])
        return (fallback_division_name, None, None, int(tts_id) if tts_id is not None else None)

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
            u = tts_direct.scrape_team_logo_url(str(args.source), season_id=int(season_id), team_id=tid_i)
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
            r = requests.post(f"{api_base}/api/import/hockey/teams", json=payload, headers=api_headers, timeout=180)
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
        # Build explicit game list if present
        explicit_game_ids: list[int] = []
        if args.games_file:
            for line in Path(args.games_file).read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.isdigit():
                    explicit_game_ids.append(int(s))
        for s in args.game_ids:
            if str(s).strip().isdigit():
                explicit_game_ids.append(int(s))

        if explicit_game_ids:
            game_ids = list(explicit_game_ids)
            fallback_by_gid = {}
        else:
            log("Discovering game ids from team schedules...")
            fallback_by_gid = tts_direct.iter_season_games(
                args.source,
                season_id=season_id,
                divisions=sorted(allowed_divs) if allowed_divs is not None else None,
                team_name_substrings=args.teams,
                progress_cb=log,
                progress_every_teams=10,
                heartbeat_seconds=30.0,
            )
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

    def _post_batch() -> None:
        nonlocal posted, api_games_batch
        if not rest_mode or not api_games_batch:
            return
        import requests

        payload = {
            "league_name": league_name,
            "replace": bool(args.replace),
            "owner_email": owner_email,
            "owner_name": owner_email,
            "source": "timetoscore",
            "external_key": f"{args.source}:{season_id}",
            "games": api_games_batch,
        }
        if args.shared is not None:
            payload["shared"] = bool(args.shared)
        r = requests.post(f"{api_base}/api/import/hockey/games_batch", json=payload, headers=api_headers, timeout=180)
        r.raise_for_status()
        out = r.json()
        if not out.get("ok"):
            raise RuntimeError(str(out))
        posted += int(out.get("imported") or 0)
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
                    stats = tts_direct.scrape_game_stats(args.source, game_id=int(gid), season_id=season_id)
                    if stats:
                        _set_cached_stats(int(gid), stats)
                except Exception as e:
                    stats = {}
                    # If the game has a recorded score in the schedule, missing stats should be treated as fatal
                    # so we can fix the scraper and backfill correctly.
                    if fb_has_result and attempt < max_attempts:
                        sleep_s = min(max_delay_s, delay_s) if max_delay_s else delay_s
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
                        if bool(args.allow_schedule_only) and type(e).__name__ == "MissingStatsError":
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
                    agg for agg in tts_norm.aggregate_goals_assists(stats) if str(agg.get("name") or "").strip()
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
                        sleep_s = min(max_delay_s, delay_s) if max_delay_s else delay_s
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

        home_name = str(stats.get("home") or "").strip() or str((fb or {}).get("home") or "").strip()
        away_name = str(stats.get("away") or "").strip() or str((fb or {}).get("away") or "").strip()
        if not home_name or not away_name:
            skipped += 1
            continue

        fb_division_name = str((fb or {}).get("division_name") or "").strip() or None
        fb_division_id = None
        fb_conference_id = None
        try:
            fb_division_id = int((fb or {}).get("division_id")) if (fb or {}).get("division_id") is not None else None
        except Exception:
            fb_division_id = None
        try:
            fb_conference_id = int((fb or {}).get("conference_id")) if (fb or {}).get("conference_id") is not None else None
        except Exception:
            fb_conference_id = None

        home_name_raw = str(home_name or "").strip()
        away_name_raw = str(away_name or "").strip()
        home_div_name, home_div_id, home_conf_id, home_tts_id = resolve_team_division_meta(home_name_raw, fb_division_name)
        away_div_name, away_div_id, away_conf_id, away_tts_id = resolve_team_division_meta(away_name_raw, fb_division_name)

        # Use per-team division for disambiguation, not the game's division block.
        home_name = canonical_team_name(home_name_raw, home_div_name)
        away_name = canonical_team_name(away_name_raw, away_div_name)

        starts_at = parse_starts_at(args.source, stats=stats, fallback=fb)
        location = str(stats.get("location") or "").strip() or str((fb or {}).get("rink") or "").strip() or None
        t1_score = tts_norm.parse_int_or_none(stats.get("homeGoals"))
        t2_score = tts_norm.parse_int_or_none(stats.get("awayGoals"))
        if t1_score is None and (fb or {}).get("homeGoals") is not None:
            t1_score = tts_norm.parse_int_or_none((fb or {}).get("homeGoals"))
        if t2_score is None and (fb or {}).get("awayGoals") is not None:
            t2_score = tts_norm.parse_int_or_none((fb or {}).get("awayGoals"))

        ga_rows = [agg for agg in tts_norm.aggregate_goals_assists(stats) if str(agg.get("name") or "").strip()]

        # Build a basic events timeline from TimeToScore data (goals + penalties + PP/PK spans)
        # so the webapp can show game events even before any spreadsheets are uploaded.
        def _side_label(side: str) -> str:
            return "Home" if str(side).strip().lower() == "home" else "Away"

        def _norm_jersey(val: Any) -> Optional[str]:
            s = str(val or "").strip()
            if not s:
                return None
            m = re.search(r"(\d+)", s)
            return m.group(1) if m else None

        roster_home = list(tts_norm.extract_roster(stats, "home"))
        roster_away = list(tts_norm.extract_roster(stats, "away"))
        num_to_name_home = {str(m.group(1)): str(p.get("name") or "").strip() for p in roster_home if (m := re.search(r"(\d+)", str(p.get("number") or ""))) and str(p.get("name") or "").strip()}
        num_to_name_away = {str(m.group(1)): str(p.get("name") or "").strip() for p in roster_away if (m := re.search(r"(\d+)", str(p.get("number") or ""))) and str(p.get("name") or "").strip()}

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

        # Build basic goal events from scoring tables when available.
        goal_events: list[dict[str, Any]] = []
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
                scorer_name = (
                    roster_map.get(scorer_num) if scorer_num and scorer_num in roster_map else str(scorer_raw or "").strip()
                )
                a1_num = _norm_jersey(a1_raw)
                a2_num = _norm_jersey(a2_raw)
                a1_name = roster_map.get(a1_num) if a1_num and a1_num in roster_map else str(a1_raw or "").strip()
                a2_name = roster_map.get(a2_num) if a2_num and a2_num in roster_map else str(a2_raw or "").strip()

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

        # Determine per-period time mode and fill missing penalty end times (best-effort).
        penalties_events_rows: list[dict[str, Any]] = []
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
                                    (f"#{rec.get('jersey')}" if rec.get("jersey") else ""),
                                    str(rec.get("infraction") or "").strip(),
                                    (f"{int(rec.get('minutes'))}m" if rec.get("minutes") is not None else ""),
                                    (f"(end {rec.get('end_txt')})" if rec.get("end_txt") else ""),
                                ]
                                if x
                            ]
                        ).strip(),
                        "Attributed Players": (
                            num_to_name_home.get(str(rec.get("jersey"))) if side_key == "home" and rec.get("jersey") else
                            num_to_name_away.get(str(rec.get("jersey"))) if side_key == "away" and rec.get("jersey") else
                            ""
                        ),
                        "Attributed Jerseys": rec.get("jersey") or "",
                    }
                )

        def _infer_time_mode(period: int) -> str:
            times: list[int] = []
            for ev in goal_events:
                if int(ev.get("Period") or 0) != int(period):
                    continue
                gs = ev.get("Game Seconds")
                if isinstance(gs, int):
                    times.append(int(gs))
            for row in penalties_events_rows:
                if int(row.get("Period") or 0) != int(period):
                    continue
                gs = row.get("Game Seconds")
                ge = row.get("Game Seconds End")
                if isinstance(gs, int):
                    times.append(int(gs))
                if isinstance(ge, int):
                    times.append(int(ge))
            if not times:
                return "elapsed"
            near_zero = sum(1 for t in times if t <= 120)
            near_high = sum(1 for t in times if t >= period_len_s - 120)
            return "remaining" if near_high > near_zero else "elapsed"

        # Fill missing end times using inferred mode.
        mode_by_period: dict[int, str] = {}
        for p in range(1, 6):
            if any(int(r.get("Period") or 0) == p for r in penalties_events_rows) or any(int(g.get("Period") or 0) == p for g in goal_events):
                mode_by_period[p] = _infer_time_mode(p)

        for side_key, recs in penalties_by_side.items():
            for rec in recs:
                if rec.get("end_s") is not None:
                    continue
                start_s = rec.get("start_s")
                if start_s is None:
                    continue
                delta = rec.get("end_s_guess_delta")
                if not isinstance(delta, int) or delta <= 0:
                    continue
                per = int(rec["period"])
                mode = mode_by_period.get(per, "elapsed")
                if mode == "remaining":
                    rec["end_s"] = max(0, int(start_s) - int(delta))
                else:
                    rec["end_s"] = min(int(period_len_s), int(start_s) + int(delta))

        # Emit explicit "Penalty Expired" events for each penalty with a known end time.
        penalty_expired_events_rows: list[dict[str, Any]] = []
        for side_key, recs in penalties_by_side.items():
            for rec in recs:
                end_s = rec.get("end_s")
                if end_s is None:
                    continue
                per = int(rec.get("period") or 0)
                details = " ".join(
                    [
                        x
                        for x in [
                            "Expired",
                            (f"#{rec.get('jersey')}" if rec.get("jersey") else ""),
                            str(rec.get("infraction") or "").strip(),
                            (f"{int(rec.get('minutes'))}m" if rec.get("minutes") is not None else ""),
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
                        "Game Time": rec.get("end_txt") or "",
                        "Game Seconds": int(end_s),
                        "Game Seconds End": "",
                        "Details": details,
                        "Attributed Players": (
                            num_to_name_home.get(str(rec.get("jersey"))) if side_key == "home" and rec.get("jersey") else
                            num_to_name_away.get(str(rec.get("jersey"))) if side_key == "away" and rec.get("jersey") else
                            ""
                        ),
                        "Attributed Jerseys": rec.get("jersey") or "",
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
        events_rows = list(goal_events) + list(penalties_events_rows) + list(penalty_expired_events_rows)
        events_rows.sort(
            key=lambda r: (
                int(r.get("Period") or 0),
                int(r.get("Game Seconds") or 0) if str(r.get("Game Seconds") or "").strip() else 0,
                str(r.get("Event Type") or ""),
            )
        )
        events_csv_text = _to_csv_text(events_headers, events_rows)

        # Merge per-player stats: goals/assists (when present) + PIM (when present).
        stats_by_name: dict[str, dict[str, int]] = {}
        for r in ga_rows:
            nm = str(r.get("name") or "").strip()
            if not nm:
                continue
            stats_by_name.setdefault(nm, {"goals": 0, "assists": 0, "pim": 0})
            stats_by_name[nm]["goals"] = int(r.get("goals") or 0)
            stats_by_name[nm]["assists"] = int(r.get("assists") or 0)
        for nm, pim in pim_by_player_name.items():
            if not nm:
                continue
            stats_by_name.setdefault(nm, {"goals": 0, "assists": 0, "pim": 0})
            stats_by_name[nm]["pim"] = int(pim or 0)
        player_stats_out = [
            {"name": nm, "goals": int(v.get("goals") or 0), "assists": int(v.get("assists") or 0), "pim": int(v.get("pim") or 0)}
            for nm, v in stats_by_name.items()
            if (int(v.get("goals") or 0) + int(v.get("assists") or 0) + int(v.get("pim") or 0)) > 0
        ]
        player_stats_out.sort(key=lambda r: str(r.get("name") or "").casefold())

        def _sum_pim(side_key: str) -> int:
            return sum(int(r.get("minutes") or 0) for r in (penalties_by_side.get(side_key) or []) if r.get("minutes") is not None)

        home_pim_total = _sum_pim("home")
        away_pim_total = _sum_pim("away")

        game_stats_json = {
            "Home Score": t1_score if t1_score is not None else "",
            "Away Score": t2_score if t2_score is not None else "",
            "Home Penalties": str(len([r for r in penalties_by_side.get("home", []) if r.get("start_s") is not None])),
            "Away Penalties": str(len([r for r in penalties_by_side.get("away", []) if r.get("start_s") is not None])),
            "Home PIM": str(home_pim_total) if home_pim_total else "",
            "Away PIM": str(away_pim_total) if away_pim_total else "",
            "TTS Schedule Only": "1" if schedule_only_game else "",
        }

        if rest_mode:
            game_div_name = home_div_name or fb_division_name
            game_div_id = home_div_id if home_div_id is not None else fb_division_id
            game_conf_id = home_conf_id if home_conf_id is not None else fb_conference_id

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
                    "game_type_name": str((fb or {}).get("type") or "").strip() or None,
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
                    "player_stats": player_stats_out,
                    "events_csv": events_csv_text,
                    "game_stats": game_stats_json,
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
                        tts_team_id=int(away_tts_id),
                        logo_dir=Path(logo_dir),
                        replace=bool(args.replace),
                        tts_direct=tts_direct,
                    )
            except Exception:
                pass

        notes = f"Imported from TimeToScore {args.source} game_id={gid}"
        game_type_name = str((fb or {}).get("type") or "").strip() or None
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
            gt, _created = m.GameType.objects.get_or_create(name=str(nm), defaults={"is_default": False})
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
        for row in tts_norm.extract_roster(stats, "home"):
            ensure_player(
                conn,
                user_id=user_id,
                team_id=team1_id,
                name=row["name"],
                jersey=row["number"],
                position=row["position"],
            )
        for row in tts_norm.extract_roster(stats, "away"):
            ensure_player(
                conn,
                user_id=user_id,
                team_id=team2_id,
                name=row["name"],
                jersey=row["number"],
                position=row["position"],
            )
        if not args.no_cleanup_bogus_players:
            if int(team1_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(None, user_id=user_id, team_id=int(team1_id))
                cleaned_team_ids.add(int(team1_id))
            if int(team2_id) not in cleaned_team_ids:
                _cleanup_numeric_named_players(None, user_id=user_id, team_id=int(team2_id))
                cleaned_team_ids.add(int(team2_id))

        # Minimal player stats (goals/assists). If we couldn't load boxscore stats, this will be empty.
        for agg in tts_norm.aggregate_goals_assists(stats):
            pname = str(agg.get("name") or "").strip()
            if not pname:
                continue
            goals = int(agg.get("goals") or 0)
            assists = int(agg.get("assists") or 0)
            if goals == 0 and assists == 0:
                continue

            # Prefer matching by name within each team roster.
            pid1 = (
                m.Player.objects.filter(user_id=int(user_id), team_id=int(team1_id), name=str(pname))
                .values_list("id", flat=True)
                .first()
            )
            pid2 = (
                m.Player.objects.filter(user_id=int(user_id), team_id=int(team2_id), name=str(pname))
                .values_list("id", flat=True)
                .first()
            )
            team_ref = team1_id
            pid = int(pid1) if pid1 is not None else (int(pid2) if pid2 is not None else None)
            if pid2 is not None and pid1 is None:
                team_ref = team2_id
            if pid is None:
                pid = ensure_player(None, user_id=user_id, team_id=team_ref, name=pname, jersey=None, position=None)

            ps, created = m.PlayerStat.objects.get_or_create(
                game_id=int(game_db_id),
                player_id=int(pid),
                defaults={
                    "user_id": int(user_id),
                    "team_id": int(team_ref),
                    "goals": int(goals),
                    "assists": int(assists),
                },
            )
            if not created:
                if args.replace:
                    m.PlayerStat.objects.filter(id=ps.id).update(goals=int(goals), assists=int(assists))
                else:
                    updates4: dict[str, Any] = {}
                    if ps.goals is None:
                        updates4["goals"] = int(goals)
                    if ps.assists is None:
                        updates4["assists"] = int(assists)
                    if updates4:
                        m.PlayerStat.objects.filter(id=ps.id).update(**updates4)

        count += 1
        if count % 25 == 0 or count == total:
            elapsed = max(0.001, time.time() - started)
            rate = count / elapsed
            pct = (count / total * 100.0) if total else 100.0
            log(f"Progress: {count}/{total} ({pct:.1f}%) games, skipped={skipped}, {rate:.2f} games/s")

    if rest_mode:
        _post_batch()

    log(f"Import complete. Imported {count} games, skipped={skipped}, schedule_only={schedule_only}.")
    if (not rest_mode) and (not args.no_cleanup_bogus_players):
        try:
            assert league_id is not None
            assert user_id is not None
            assert m is not None
            all_team_ids = sorted(
                set(
                    m.LeagueTeam.objects.filter(league_id=int(league_id), team__user_id=int(user_id))
                    .values_list("team_id", flat=True)
                    .distinct()
                )
            )
            moved_total = 0
            for tid in all_team_ids:
                moved_total += _cleanup_numeric_named_players(None, user_id=user_id, team_id=int(tid))
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
                    m.LeagueTeam.objects.filter(league_id=int(league_id), team_id=int(team_db_id)).update(**updates)
                    updated_div += 1
                if logo_dir is not None and (not logo_path) and tts_id is not None:
                    _ensure_team_logo(
                        None,
                        team_db_id=int(team_db_id),
                        team_owner_user_id=user_id,
                        source=str(args.source),
                        season_id=int(season_id),
                        tts_team_id=int(tts_id),
                        logo_dir=Path(logo_dir),
                        replace=False,
                        tts_direct=tts_direct,
                    )
                    updated_logo += 1
            # Also align league_games division to the home team division when possible.
            league_team_meta = {
                int(tid): (dn, did, cid)
                for tid, dn, did, cid in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
                    "team_id", "division_name", "division_id", "conference_id"
                )
            }
            league_games = list(m.LeagueGame.objects.filter(league_id=int(league_id)).select_related("game"))
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
                m.LeagueGame.objects.bulk_update(to_update, ["division_name", "division_id", "conference_id"], batch_size=500)
            log(f"Refreshed teams: divisions_updated={updated_div} logos_checked={updated_logo}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

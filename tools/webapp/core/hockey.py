import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Optional

from .orm import _orm_modules
from .utils import to_dt

try:
    from tools.webapp.hockey_rankings import (  # type: ignore
        GameScore,
        compute_mhr_like_ratings,
        filter_games_ignore_cross_age,
        parse_age_from_division_name,
        parse_level_from_division_name,
        scale_ratings_to_0_99_9_by_component,
    )
except Exception:  # pragma: no cover
    from hockey_rankings import (  # type: ignore
        GameScore,
        compute_mhr_like_ratings,
        filter_games_ignore_cross_age,
        parse_age_from_division_name,
        parse_level_from_division_name,
        scale_ratings_to_0_99_9_by_component,
    )


def create_team(user_id: int, name: str, is_external: bool = False) -> int:
    _django_orm, m = _orm_modules()
    t = m.Team.objects.create(
        user_id=int(user_id),
        name=str(name or ""),
        is_external=bool(is_external),
        created_at=dt.datetime.now(),
        updated_at=None,
    )
    return int(t.id)


def get_team(team_id: int, user_id: int) -> Optional[dict]:
    _django_orm, m = _orm_modules()
    return m.Team.objects.filter(id=int(team_id), user_id=int(user_id)).values().first()


def save_team_logo(file_storage, team_id: int, *, instance_dir: Path) -> Path:
    # Save under instance/uploads/team_logos
    uploads = Path(instance_dir) / "uploads" / "team_logos"
    uploads.mkdir(parents=True, exist_ok=True)
    # sanitize filename
    fname = Path(
        str(getattr(file_storage, "filename", None) or getattr(file_storage, "name", "") or "")
    ).name
    if not fname:
        fname = "logo"
    # prefix with team id and timestamp
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    dest = uploads / f"team{team_id}_{ts}_{fname}"
    if hasattr(file_storage, "save"):
        file_storage.save(dest)
    elif hasattr(file_storage, "chunks"):
        with dest.open("wb") as out:
            for chunk in file_storage.chunks():  # type: ignore[attr-defined]
                out.write(chunk)
    else:
        data = file_storage.read() if hasattr(file_storage, "read") else b""
        dest.write_bytes(data or b"")
    return dest


def ensure_external_team(user_id: int, name: str) -> int:
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
        return t

    name = _norm_team_name(name)
    _django_orm, m = _orm_modules()
    t, _created = m.Team.objects.get_or_create(
        user_id=int(user_id),
        name=str(name),
        defaults={
            "is_external": True,
            "logo_path": None,
            "created_at": dt.datetime.now(),
            "updated_at": None,
        },
    )
    return int(t.id)


def create_hky_game(
    user_id: int,
    team1_id: int,
    team2_id: int,
    game_type_id: Optional[int],
    starts_at: Optional[dt.datetime],
    location: Optional[str],
) -> int:
    _django_orm, m = _orm_modules()
    now = dt.datetime.now()
    g = m.HkyGame.objects.create(
        user_id=int(user_id),
        team1_id=int(team1_id),
        team2_id=int(team2_id),
        game_type_id=int(game_type_id) if game_type_id is not None else None,
        starts_at=starts_at,
        location=location,
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    return int(g.id)


def _safe_return_to_url(value: Optional[str], *, default: str) -> str:
    """
    Only allow returning to same-site relative URLs.
    Accepts paths like "/teams/123" or "/schedule?division=..." and rejects external URLs.
    """
    if not value:
        return str(default)
    s = str(value).strip()
    if not s:
        return str(default)
    if not s.startswith("/"):
        return str(default)
    if s.startswith("//"):
        return str(default)
    return s


def _sanitize_http_url(value: Optional[str]) -> Optional[str]:
    """
    Allow only http(s) URLs for external links (prevents javascript: etc).
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    sl = s.lower()
    if sl.startswith("http://") or sl.startswith("https://"):
        return s
    return None


def _extract_game_video_url_from_notes(notes: Optional[str]) -> Optional[str]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            for k in ("game_video_url", "game_video", "video_url"):
                v = d.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass
    m = re.search(r"(?:^|[\\s|,;])game_video_url\\s*=\\s*([^\\s|,;]+)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip()
    m = re.search(r"(?:^|[\\s|,;])game_video\\s*=\\s*([^\\s|,;]+)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip()
    return None


def _extract_game_stats_note_from_notes(notes: Optional[str]) -> Optional[str]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            for k in ("stats_note", "schedule_note"):
                v = d.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass
    m = re.search(r"(?:^|[\r\n])\s*stats_note\s*[:=]\s*([^\r\n]+)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip()
    m = re.search(r"(?:^|[\r\n])\s*schedule_note\s*[:=]\s*([^\r\n]+)", s, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip()
    return None


def _extract_timetoscore_game_id_from_notes(notes: Optional[str]) -> Optional[int]:
    s = str(notes or "").strip()
    if not s:
        return None
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            v = d.get("timetoscore_game_id")
            if v is not None:
                try:
                    return int(v)
                except Exception:
                    return None
    except Exception:
        pass
    m = re.search(r"(?:^|[\\s|,;])game_id\\s*=\\s*(\\d+)", s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"\"timetoscore_game_id\"\\s*:\\s*(\\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def sort_games_schedule_order(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Stable schedule ordering:
      - games with known start datetime first
      - then by start date
      - then by start time
      - then league sort_order (if present)
      - then created_at
    """

    def _int_or_big(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 2147483647

    def _key(idx_game: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        idx, g = idx_game
        sdt = to_dt(g.get("starts_at"))
        has_dt = sdt is not None
        # Put unknown starts_at at the end but keep a deterministic order using sort_order/created_at.
        dt_key = sdt if sdt else dt.datetime.max
        so = _int_or_big(g.get("sort_order"))
        created = to_dt(g.get("created_at")) or dt.datetime.max
        return (0 if has_dt else 1, dt_key.date(), dt_key.time(), so, created, idx)

    return [g for _idx, g in sorted(list(enumerate(games or [])), key=_key)]


def is_external_division_name(name: Any) -> bool:
    return str(name or "").strip().casefold().startswith("external")


def division_sort_key(division_name: Any) -> tuple:
    """
    Sort key for league division names.

    Primary ordering:
      - age (10U/12AA/etc) ascending
      - level ordering: AAA, AA, A, BB, B, (everything else)
      - non-External before External (within the same age/level)
      - then lexicographic as a stable tie-breaker
    """
    raw = str(division_name or "").strip()
    if not raw:
        return (999, 99, 1, "", "")

    external = is_external_division_name(raw)
    base = raw
    if external:
        base = re.sub(r"(?i)^external\s*", "", base).strip()

    age = parse_age_from_division_name(base)
    age_key = int(age) if age is not None else 999

    m = re.search(
        r"(?i)(?:^|\b)\d{1,2}(?:u)?\s*(AAA|AA|BB|A|B)(?=\b|\s|$|[-–—])",
        base,
    )
    level_token = str(m.group(1)).upper() if m else ""
    level_rank = {"AAA": 0, "AA": 1, "A": 2, "BB": 3, "B": 4}.get(level_token, 99)

    return (age_key, int(level_rank), 1 if external else 0, base.casefold(), raw.casefold())


def _league_game_is_cross_division_non_external_row(
    game_division_name: Optional[str],
    team1_division_name: Optional[str],
    team2_division_name: Optional[str],
) -> bool:
    """
    True when both teams have known, non-External league divisions and they differ.
    """
    d1 = str(team1_division_name or "").strip()
    d2 = str(team2_division_name or "").strip()
    if not d1 or not d2:
        return False
    if is_external_division_name(d1) or is_external_division_name(d2):
        return False
    return d1 != d2


def _league_game_is_cross_division_non_external(game_row: dict[str, Any]) -> bool:
    """
    Returns True if both teams have known, non-External league divisions and those divisions differ.
    """
    d1 = str(game_row.get("team1_league_division_name") or "").strip()
    d2 = str(game_row.get("team2_league_division_name") or "").strip()
    if not d1 or not d2:
        return False
    if is_external_division_name(d1) or is_external_division_name(d2):
        return False
    ld = str(game_row.get("division_name") or game_row.get("league_division_name") or "").strip()
    if is_external_division_name(ld):
        return False
    return d1 != d2


def recompute_league_mhr_ratings(
    db_conn, league_id: int, *, max_goal_diff: int = 7, min_games: int = 2
) -> dict[int, dict[str, Any]]:
    """
    Recompute and persist MyHockeyRankings-like ratings for teams in a league.
    Stores values on `league_teams` as:
      - mhr_rating (NULL if games < min_games)
      - mhr_agd, mhr_sched, mhr_games, mhr_updated_at
    """
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db import transaction

    league_team_rows = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id))
        .select_related("team")
        .values("team_id", "division_name", "team__name")
    )

    team_age: dict[int, Optional[int]] = {}
    team_level: dict[int, Optional[str]] = {}
    for r in league_team_rows:
        try:
            tid = int(r.get("team_id"))
        except Exception:
            continue
        dn = str(r.get("division_name") or "").strip()
        age = parse_age_from_division_name(dn)
        if age is None:
            age = parse_age_from_division_name(str(r.get("team__name") or "").strip())
        team_age[tid] = age
        lvl = parse_level_from_division_name(dn)
        if lvl is None:
            lvl = parse_level_from_division_name(str(r.get("team__name") or "").strip())
        team_level[tid] = lvl

    games: list[GameScore] = []
    for team1_id, team2_id, team1_score, team2_score in m.LeagueGame.objects.filter(
        league_id=int(league_id),
        game__team1_score__isnull=False,
        game__team2_score__isnull=False,
    ).values_list(
        "game__team1_id",
        "game__team2_id",
        "game__team1_score",
        "game__team2_score",
    ):
        try:
            games.append(
                GameScore(
                    team1_id=int(team1_id),
                    team2_id=int(team2_id),
                    team1_score=int(team1_score),
                    team2_score=int(team2_score),
                )
            )
        except Exception:
            continue

    # Eligibility rule: A team cannot have an MHR rating unless it has played at least one game
    # against another team in the same age group (age + level when known).
    age_group_games: dict[int, int] = {}
    for g in games:
        a = int(g.team1_id)
        b = int(g.team2_id)
        age_a = team_age.get(a)
        age_b = team_age.get(b)
        if age_a is None or age_b is None:
            continue
        if int(age_a) != int(age_b):
            continue
        lvl_a = str(team_level.get(a) or "").strip().upper() or None
        lvl_b = str(team_level.get(b) or "").strip().upper() or None
        if lvl_a is not None and lvl_b is not None and lvl_a != lvl_b:
            continue
        age_group_games[a] = int(age_group_games.get(a, 0)) + 1
        age_group_games[b] = int(age_group_games.get(b, 0)) + 1

    # Ignore cross-age games when computing ratings (no cross-age coupling).
    games_same_age = filter_games_ignore_cross_age(games, team_age=team_age)

    computed = compute_mhr_like_ratings(
        games=games_same_age,
        max_goal_diff=int(max_goal_diff),
        min_games_for_rating=int(min_games),
    )
    # Normalize per disconnected component: top team in each independent group becomes 99.9.
    computed_norm = scale_ratings_to_0_99_9_by_component(
        computed, games=games_same_age, key="rating"
    )

    now = dt.datetime.now()
    # Persist for all league teams (set NULL when unknown/insufficient).
    league_team_objs = list(m.LeagueTeam.objects.filter(league_id=int(league_id)))
    for lt in league_team_objs:
        tid = int(lt.team_id)
        row_norm = computed_norm.get(tid) or {}
        rating = row_norm.get("rating")
        # Use raw AGD/SCHED/GAMES from the base computation (before shifting).
        row_base = computed.get(tid) or {}
        if int(age_group_games.get(tid, 0)) <= 0:
            lt.mhr_rating = None
            lt.mhr_agd = None
            lt.mhr_sched = None
            lt.mhr_games = 0
        else:
            lt.mhr_rating = float(rating) if rating is not None else None
            lt.mhr_agd = float(row_base.get("agd")) if row_base.get("agd") is not None else None
            lt.mhr_sched = (
                float(row_base.get("sched")) if row_base.get("sched") is not None else None
            )
            lt.mhr_games = int(row_base.get("games")) if row_base.get("games") is not None else 0
        lt.mhr_updated_at = now
    if league_team_objs:
        with transaction.atomic():
            m.LeagueTeam.objects.bulk_update(
                league_team_objs,
                ["mhr_rating", "mhr_agd", "mhr_sched", "mhr_games", "mhr_updated_at"],
                batch_size=500,
            )
    return computed_norm

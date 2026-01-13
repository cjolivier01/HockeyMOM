#!/usr/bin/env python3
"""Import CAHA tier schedules from https://caha.com/schedule.pl into the webapp.

This importer scrapes CAHA's public schedule pages (AA / AAA Major / AAA Minor / 15O AAA) and
upserts games into the HockeyMOM webapp:
  - Local mode: direct Django ORM writes (using the same helpers as the webapp import API)
  - REST mode: POST to `/api/import/hockey/games_batch` (for GCP / remote installs)

The CAHA schedule pages do not expose TimeToScore game ids. We therefore import games using
`external_game_key` (deterministic, derived from year + schedule group + CAHA game number).

To avoid creating duplicates when the same games are already imported from TimeToScore (league 5),
we optionally map CAHA schedule team display names to the corresponding TimeToScore team names for
the inferred division (best-effort).
"""

from __future__ import annotations

import argparse
import datetime as dt
from difflib import SequenceMatcher
from functools import lru_cache
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests


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
        from tools.webapp.django_app import views as web_views  # type: ignore
    except Exception:  # pragma: no cover
        from django_app import views as web_views  # type: ignore

    return django_orm, web_views


def _now_ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _fmt_dt(val: Optional[dt.datetime]) -> Optional[str]:
    if val is None:
        return None
    return val.strftime("%Y-%m-%d %H:%M:%S")


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").casefold())


def _parse_age_num(age_group: str) -> Optional[int]:
    m = re.search(r"(\d+)", str(age_group or ""))
    if not m:
        return None
    try:
        v = int(m.group(1))
        return v if v > 0 else None
    except Exception:
        return None


def _parse_variant_num(name: str) -> Optional[int]:
    # "Bulls (1)" -> 1, "Tri-Valley Bulls 14AA-2" -> 2
    s = str(name or "").strip()
    m = re.search(r"\((\d+)\)\s*$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.search(r"-(\d+)\s*$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _strip_variant_suffix(name: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", str(name or "").strip()).strip()


_DIV_TOKEN_RE = re.compile(r"(?i)\s+\d{1,2}(?:u)?\s*(?:aaa|aa|bb|a|b)\s*(?:-\d+)?\s*$")


def _strip_trailing_division_token(team_name: str) -> str:
    # "Golden State Elite 16AA-1" -> "Golden State Elite"
    # "Tri Valley Blue Devils 16A" -> "Tri Valley Blue Devils"
    return _DIV_TOKEN_RE.sub("", str(team_name or "").strip()).strip()


def _acronym_words(s: str) -> str:
    words = [w for w in re.split(r"\s+", str(s or "").strip()) if w]
    stop = {"jr", "ice", "hockey", "club"}
    letters = [w[0] for w in words if w and w.casefold() not in stop]
    return "".join(letters).upper()


def _map_team_name_to_tts(
    schedule_name: str, *, candidates: list[str]
) -> tuple[str, Optional[str]]:
    """Return (mapped_name, matched_candidate_name).

    If no confident match, returns (schedule_name, None).
    """
    sched_raw = str(schedule_name or "").strip()
    if not sched_raw or not candidates:
        return sched_raw, None

    sched_variant = _parse_variant_num(sched_raw)
    sched_base = _strip_variant_suffix(sched_raw)
    sched_key = _norm_key(sched_base)
    if not sched_key:
        return sched_raw, None

    cand_rows: list[tuple[str, str, Optional[int], str]] = []
    # (candidate_name, cand_key, variant, cand_acronym_key)
    for cand in candidates:
        c = str(cand or "").strip()
        if not c:
            continue
        cand_variant = _parse_variant_num(c)
        cand_club = _strip_trailing_division_token(c)
        cand_key = _norm_key(cand_club)
        if not cand_key:
            continue
        cand_acr = _norm_key(_acronym_words(cand_club))
        cand_rows.append((c, cand_key, cand_variant, cand_acr))

    if not cand_rows:
        return sched_raw, None

    # Prefer matching variant suffixes when present.
    filtered = cand_rows
    if sched_variant is not None:
        same_variant = [r for r in cand_rows if r[2] == sched_variant]
        if same_variant:
            filtered = same_variant

    best_score = -1.0
    best_name: Optional[str] = None
    for cand_name, cand_key, _v, cand_acr in filtered:
        ratio = SequenceMatcher(None, sched_key, cand_key).ratio()
        bonus = 0.0
        if sched_key == cand_key:
            bonus += 1.0
        if sched_key and (sched_key in cand_key or cand_key in sched_key):
            bonus += 0.3
        if cand_acr and sched_key == cand_acr:
            bonus += 0.5
        score = ratio + bonus
        if score > best_score:
            best_score = score
            best_name = cand_name

    if best_name is None:
        return sched_raw, None

    # Heuristic threshold to avoid bad matches.
    if best_score < 0.8:
        return sched_raw, None
    return best_name, best_name


def _infer_tts_division_name_for_group(
    *, age_group: str, group_label: str, known_division_names: set[str]
) -> Optional[str]:
    label = str(group_label or "").strip()
    base_age = _parse_age_num(age_group)
    if base_age is None:
        return None

    if label == "AA":
        return f"{base_age} AA"
    if label == "AAA Major":
        return f"{base_age} AAA"
    if label == "AAA Minor":
        cand = f"{base_age + 1} AAA"
        return cand if cand in known_division_names else f"{base_age} AAA"
    if label == "15O AAA":
        return "15 AAA" if "15 AAA" in known_division_names else f"{base_age} AAA"
    return None


def _external_game_key(*, year: int, d: int, game_number: int) -> str:
    return f"caha:schedule_pl:y{int(year)}:d{int(d)}:gm{int(game_number)}"


def _post_games_batch(
    *,
    api_base: str,
    api_token: Optional[str],
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    tok = str(api_token or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
        headers["X-HM-Import-Token"] = tok
    r = requests.post(
        f"{api_base}/api/import/hockey/games_batch",
        json=payload,
        headers=headers,
        timeout=max(1.0, float(timeout_s)),
    )
    r.raise_for_status()
    out = r.json()
    if not out.get("ok"):
        raise RuntimeError(str(out))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    base_dir = Path(__file__).resolve().parents[1]
    default_cfg = os.environ.get("HM_DB_CONFIG") or str(base_dir / "config.json")

    ap = argparse.ArgumentParser(
        description="Import CAHA tier schedule.pl schedules into the webapp"
    )
    ap.add_argument("--config", default=default_cfg, help="Path to webapp DB config.json")
    ap.add_argument(
        "--year",
        type=int,
        default=0,
        help="CAHA schedule year (e.g. 2025). 0 = use current year selected on schedule.pl",
    )
    ap.add_argument("--limit", type=int, default=None, help="Max games to import (for testing)")
    ap.add_argument("--replace", action="store_true", help="Overwrite existing game scores")

    ap.add_argument("--league-name", default="CAHA", help="League name to import into")
    ap.add_argument("--shared", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--owner-email", required=True, help="Webapp user email that will own imported data"
    )
    ap.add_argument(
        "--owner-name", default=None, help="Display name for owner user creation if missing"
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
    ap.add_argument("--api-batch-size", type=int, default=50, help="Games per REST batch request")
    ap.add_argument(
        "--api-timeout-s", type=float, default=180.0, help="REST request timeout seconds"
    )

    ap.add_argument(
        "--map-team-names-from-tts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map CAHA schedule team display names to TimeToScore league team names (best-effort) to avoid duplicates.",
    )
    ap.add_argument(
        "--tts-league-id",
        type=int,
        default=5,
        help="TimeToScore `league=` id to use for name mapping (default: 5).",
    )
    ap.add_argument(
        "--tts-season",
        type=int,
        default=0,
        help="TimeToScore season id for team name mapping (0 = current/latest for the selected --tts-league-id).",
    )

    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from hmlib.time2score import caha_schedule_pl
    from hmlib.time2score import direct as tts_direct

    rest_mode = bool(args.api_url)

    # Scrape schedule.pl
    schedule_year = int(args.year) if int(args.year) > 0 else None
    effective_year, groups = caha_schedule_pl.scrape_index(year=schedule_year)
    rows: list[caha_schedule_pl.ScheduleGameRow] = []
    for g in groups:
        rows.extend(caha_schedule_pl.scrape_schedule_group(g))

    if args.limit is not None:
        rows = rows[: int(args.limit)]

    # Build TimeToScore division -> team-name map for best-effort team-name mapping (to avoid duplicates).
    tts_team_names_by_div: dict[str, list[str]] = {}
    known_div_names: set[str] = set()
    if bool(args.map_team_names_from_tts):
        tts_season_id = int(args.tts_season)
        if tts_season_id <= 0:
            tts_season_id = tts_direct.pick_current_season_id(
                "caha", league_id=int(args.tts_league_id)
            )
        divs = tts_direct.list_divisions(
            "caha", season_id=int(tts_season_id), league_id=int(args.tts_league_id)
        )
        for d in divs:
            dn = str(d.name or "").strip()
            if not dn:
                continue
            if dn.casefold().startswith(("girls", "high school")):
                continue
            known_div_names.add(dn)
            tts_team_names_by_div[dn] = [str(t.get("name") or "").strip() for t in (d.teams or [])]

    def log(msg: str) -> None:
        print(f"[{_now_ts()}] {msg}", flush=True)

    def build_game_payload(row: caha_schedule_pl.ScheduleGameRow) -> dict[str, Any]:
        div_name = _infer_tts_division_name_for_group(
            age_group=str(row.age_group),
            group_label=str(row.group_label),
            known_division_names=known_div_names,
        )

        home_name = str(row.home or "").strip()
        away_name = str(row.away or "").strip()
        if bool(args.map_team_names_from_tts) and div_name and div_name in tts_team_names_by_div:
            home_name, _ = _map_team_name_to_tts(
                home_name, candidates=tts_team_names_by_div[div_name]
            )
            away_name, _ = _map_team_name_to_tts(
                away_name, candidates=tts_team_names_by_div[div_name]
            )

        starts_at = _fmt_dt(row.starts_at)
        return {
            "home_name": home_name,
            "away_name": away_name,
            "division_name": div_name,
            "starts_at": starts_at,
            "location": row.rink,
            "home_score": row.home_score,
            "away_score": row.away_score,
            "game_type_name": row.game_type,
            "external_game_key": _external_game_key(
                year=int(row.year), d=int(row.d), game_number=int(row.game_number)
            ),
            "caha_schedule_year": int(row.year),
            "caha_schedule_group": f"{row.age_group}:{row.group_label}",
            "caha_schedule_game_number": int(row.game_number),
        }

    games_payload = [build_game_payload(r) for r in rows]

    log(
        f"Scraped CAHA schedule.pl games: year={effective_year} groups={len(groups)} games={len(games_payload)}"
    )

    if rest_mode:
        api_base = str(args.api_url or "").strip().rstrip("/")
        if not api_base:
            raise SystemExit("--api-url must be non-empty when provided")
        api_token = args.api_token or os.environ.get("HM_WEBAPP_IMPORT_TOKEN") or None

        batch_size = max(1, int(args.api_batch_size))
        for i in range(0, len(games_payload), batch_size):
            batch = games_payload[i : i + batch_size]
            payload = {
                "league_name": str(args.league_name),
                "replace": bool(args.replace),
                "owner_email": str(args.owner_email),
                "owner_name": str(args.owner_name or args.owner_email),
                "source": "caha_schedule_pl",
                "external_key": f"caha:schedule_pl:{effective_year}",
                "games": batch,
            }
            if args.shared is not None:
                payload["shared"] = bool(args.shared)
            log(f"Posting CAHA schedule batch: {i}/{len(games_payload)} size={len(batch)} ...")
            _post_games_batch(
                api_base=api_base,
                api_token=api_token,
                payload=payload,
                timeout_s=float(args.api_timeout_s),
            )
            time.sleep(0.05)
        log("Done.")
        return 0

    # Direct DB import mode
    django_orm, web_views = _orm_modules()
    del django_orm

    owner_email = str(args.owner_email).strip().lower()
    owner_name = str(args.owner_name or owner_email).strip() or owner_email
    owner_user_id = web_views._ensure_user_for_import(owner_email, name=owner_name)
    league_id = web_views._ensure_league_for_import(
        league_name=str(args.league_name),
        owner_user_id=int(owner_user_id),
        is_shared=args.shared,
        source="caha_schedule_pl",
        external_key=f"caha:schedule_pl:{effective_year}",
        commit=False,
    )
    web_views._ensure_league_member_for_import(
        int(league_id), int(owner_user_id), role="admin", commit=False
    )

    for idx, g in enumerate(games_payload):
        if idx and idx % 250 == 0:
            log(f"Imported {idx}/{len(games_payload)} games...")

        home_name = str(g.get("home_name") or "").strip()
        away_name = str(g.get("away_name") or "").strip()
        if not home_name or not away_name:
            continue

        team1_id = web_views._ensure_external_team_for_import(
            int(owner_user_id), home_name, commit=False
        )
        team2_id = web_views._ensure_external_team_for_import(
            int(owner_user_id), away_name, commit=False
        )
        web_views._map_team_to_league_for_import(
            int(league_id), int(team1_id), division_name=g.get("division_name"), commit=False
        )
        web_views._map_team_to_league_for_import(
            int(league_id), int(team2_id), division_name=g.get("division_name"), commit=False
        )

        game_type_id = web_views._ensure_game_type_id_for_import(g.get("game_type_name"))
        notes_fields = {
            "external_game_key": str(g.get("external_game_key") or "").strip() or None,
            "source": "caha_schedule_pl",
            "caha_schedule_year": g.get("caha_schedule_year"),
            "caha_schedule_group": g.get("caha_schedule_group"),
            "caha_schedule_game_number": g.get("caha_schedule_game_number"),
        }
        notes_fields = {k: v for k, v in notes_fields.items() if v is not None}

        gid = web_views._upsert_game_for_import(
            owner_user_id=int(owner_user_id),
            team1_id=int(team1_id),
            team2_id=int(team2_id),
            game_type_id=int(game_type_id) if game_type_id is not None else None,
            starts_at=str(g.get("starts_at") or "").strip() or None,
            location=str(g.get("location") or "").strip() or None,
            team1_score=int(g["home_score"]) if g.get("home_score") is not None else None,
            team2_score=int(g["away_score"]) if g.get("away_score") is not None else None,
            replace=bool(args.replace),
            notes_json_fields=notes_fields,
            commit=False,
        )

        web_views._map_game_to_league_for_import(
            int(league_id), int(gid), division_name=g.get("division_name"), commit=False
        )

    log(f"Done. Imported {len(games_payload)} schedule.pl games.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

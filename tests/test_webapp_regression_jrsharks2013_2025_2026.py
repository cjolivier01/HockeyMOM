from __future__ import annotations

import datetime as dt
import difflib
import json
import re
from pathlib import Path
from typing import Any, Optional

import pytest


_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "webapp_regression"
    / "jrsharks2013_2025_2026"
    / "db_fixture_team44.json"
)
_EXPECTED_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "webapp_regression"
    / "jrsharks2013_2025_2026"
    / "expected_public_pages.json"
)


def _parse_dt(v: Any) -> Optional[dt.datetime]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _strip_tags(s: str) -> str:
    # We intentionally keep this simple: the templates are stable and we're snapshotting content,
    # not validating HTML semantics.
    out = re.sub(r"<[^>]*>", " ", str(s or ""), flags=re.DOTALL)
    out = out.replace("\xa0", " ")
    return " ".join(out.split()).strip()


def _extract_table(html: str, *, table_id: str) -> dict[str, Any]:
    """
    Extract a HTML table by its id into a stable dict representation.
    """
    m = re.search(
        rf'<table[^>]*id="{re.escape(table_id)}"[^>]*>(?P<table>.*?)</table>',
        html,
        re.DOTALL,
    )
    assert m is not None, f"table not found: {table_id!r}"
    table_html = m.group("table")

    thead_m = re.search(r"<thead>(?P<thead>.*?)</thead>", table_html, re.DOTALL)
    assert thead_m is not None, f"thead not found: {table_id!r}"
    tbody_m = re.search(r"<tbody>(?P<tbody>.*?)</tbody>", table_html, re.DOTALL)
    assert tbody_m is not None, f"tbody not found: {table_id!r}"

    ths = re.findall(r"<th[^>]*>(?P<th>.*?)</th>", thead_m.group("thead"), re.DOTALL)
    headers: list[dict[str, Any]] = []
    for th in ths:
        label = _strip_tags(th)
        subnote = None
        sub_m = re.search(r'class="th-subnote"[^>]*>\(\s*(\d+)\s*Games?\s*\)</div>', th)
        if sub_m:
            try:
                subnote = int(sub_m.group(1))
            except Exception:
                subnote = None
        headers.append({"label": label, "n_games": subnote})

    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(?P<tr>.*?)</tr>", tbody_m.group("tbody"), re.DOTALL):
        tds = re.findall(r"<td[^>]*>(?P<td>.*?)</td>", tr, re.DOTALL)
        if not tds:
            continue
        rows.append([_strip_tags(td) for td in tds])

    return {"headers": headers, "rows": rows}


def _extract_player_stats_table(html: str) -> dict[str, Any]:
    m = re.search(
        r"<h3>\s*Player Stats\s*\(Skaters\)\s*</h3>.*?"
        r'(?P<table><table[^>]*data-sortable="1"[^>]*>.*?</table>)',
        html,
        re.DOTALL,
    )
    assert m is not None, "Player Stats (Skaters) table not found"
    table_html = m.group("table")

    thead_m = re.search(r"<thead>(?P<thead>.*?)</thead>", table_html, re.DOTALL)
    assert thead_m is not None
    tbody_m = re.search(r"<tbody>(?P<tbody>.*?)</tbody>", table_html, re.DOTALL)
    assert tbody_m is not None

    ths = re.findall(r"<th[^>]*>(?P<th>.*?)</th>", thead_m.group("thead"), re.DOTALL)
    headers: list[dict[str, Any]] = []
    for th in ths:
        # Most columns wrap the label in a <div>, but the first two are plain text.
        label_m = re.search(r"<div>(?P<label>.*?)</div>", th, re.DOTALL)
        label = _strip_tags(label_m.group("label")) if label_m else _strip_tags(th)
        subnote = None
        sub_m = re.search(r'class="th-subnote"[^>]*>\(\s*(\d+)\s*Games?\s*\)</div>', th)
        if sub_m:
            try:
                subnote = int(sub_m.group(1))
            except Exception:
                subnote = None
        headers.append({"label": label, "n_games": subnote})

    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(?P<tr>.*?)</tr>", tbody_m.group("tbody"), re.DOTALL):
        tds = re.findall(r"<td[^>]*>(?P<td>.*?)</td>", tr, re.DOTALL)
        if not tds:
            continue
        rows.append([_strip_tags(td) for td in tds])

    return {"headers": headers, "rows": rows}


def _extract_team_stats_meta(html: str) -> dict[str, Any]:
    # Example:
    #   Stats may be incomplete. This selection has <strong>23</strong> completed games; ...
    total_games = None
    m = re.search(r"This selection has\s*<strong>(\d+)</strong>\s*completed games", html)
    if m:
        try:
            total_games = int(m.group(1))
        except Exception:
            total_games = None
    return {"eligible_completed_games": total_games}


def _load_snapshot_fixture(m, fixture_path: Path) -> None:
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))

    def _rows(key: str) -> list[dict[str, Any]]:
        rows = raw.get(key) or []
        if not isinstance(rows, list):
            raise TypeError(f"{key} must be a list")
        out: list[dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                raise TypeError(f"{key} entries must be dicts")
            out.append(dict(r))
        return out

    def _chunked(seq: list[Any], n: int) -> list[list[Any]]:
        return [seq[i : i + n] for i in range(0, len(seq), n)]

    def _coerce_dt_fields(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
        for r in rows:
            for k in keys:
                if k in r:
                    r[k] = _parse_dt(r.get(k))

    from django.db import transaction

    with transaction.atomic():
        # FK order: User -> League -> Team -> Player -> EventType -> Game -> League mappings -> Game data.
        user_rows = _rows("users")
        _coerce_dt_fields(user_rows, ("created_at",))
        m.User.objects.bulk_create([m.User(**r) for r in user_rows], batch_size=200)

        league_rows = _rows("leagues")
        _coerce_dt_fields(league_rows, ("created_at", "updated_at"))
        m.League.objects.bulk_create([m.League(**r) for r in league_rows], batch_size=50)

        team_rows = _rows("teams")
        _coerce_dt_fields(team_rows, ("created_at", "updated_at"))
        m.Team.objects.bulk_create([m.Team(**r) for r in team_rows], batch_size=200)

        player_rows = _rows("players")
        _coerce_dt_fields(player_rows, ("created_at", "updated_at"))
        m.Player.objects.bulk_create([m.Player(**r) for r in player_rows], batch_size=500)

        et_rows = _rows("hky_event_types")
        _coerce_dt_fields(et_rows, ("created_at",))
        m.HkyEventType.objects.bulk_create([m.HkyEventType(**r) for r in et_rows], batch_size=200)

        game_rows = _rows("hky_games")
        _coerce_dt_fields(game_rows, ("starts_at", "stats_imported_at", "created_at", "updated_at"))
        m.HkyGame.objects.bulk_create([m.HkyGame(**r) for r in game_rows], batch_size=200)

        m.LeagueTeam.objects.bulk_create(
            [m.LeagueTeam(**r) for r in _rows("league_teams")], batch_size=500
        )
        m.LeagueGame.objects.bulk_create(
            [m.LeagueGame(**r) for r in _rows("league_games")], batch_size=500
        )

        gp_rows = _rows("hky_game_players")
        _coerce_dt_fields(gp_rows, ("created_at", "updated_at"))
        m.HkyGamePlayer.objects.bulk_create([m.HkyGamePlayer(**r) for r in gp_rows], batch_size=800)

        sup_rows = _rows("hky_game_event_suppressions")
        _coerce_dt_fields(sup_rows, ("created_at", "updated_at"))
        if sup_rows:
            m.HkyGameEventSuppression.objects.bulk_create(
                [m.HkyGameEventSuppression(**r) for r in sup_rows], batch_size=500
            )

        shift_rows = _rows("hky_game_shift_rows")
        _coerce_dt_fields(shift_rows, ("created_at", "updated_at"))
        for chunk in _chunked(shift_rows, 1500):
            m.HkyGameShiftRow.objects.bulk_create(
                [m.HkyGameShiftRow(**r) for r in chunk], batch_size=1500
            )

        ev_rows = _rows("hky_game_event_rows")
        _coerce_dt_fields(ev_rows, ("created_at", "updated_at"))
        for chunk in _chunked(ev_rows, 1500):
            m.HkyGameEventRow.objects.bulk_create(
                [m.HkyGameEventRow(**r) for r in chunk], batch_size=1500
            )


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    from django.test import Client

    return Client(), m


def should_regression_jrsharks2013_2025_2026_public_pages_match_snapshot(client_and_models):
    client, m = client_and_models
    assert _FIXTURE_PATH.exists(), f"missing fixture: {_FIXTURE_PATH}"
    assert _EXPECTED_PATH.exists(), f"missing expected snapshot: {_EXPECTED_PATH}"

    _load_snapshot_fixture(m, _FIXTURE_PATH)

    team_resp = client.get("/public/leagues/1/teams/44")
    assert team_resp.status_code == 200
    team_html = team_resp.content.decode("utf-8")

    schedule_resp = client.get("/public/leagues/1/schedule?team_id=44")
    assert schedule_resp.status_code == 200
    schedule_html = schedule_resp.content.decode("utf-8")

    snapshot = {
        "team_page": {
            "meta": _extract_team_stats_meta(team_html),
            "player_stats": _extract_player_stats_table(team_html),
            # Stable id-based selectors for schedule: we add an explicit id below via template patching.
            "schedule_table": _extract_table(team_html, table_id="team-schedule-table"),
        },
        "schedule_page": {
            "schedule_table": _extract_table(schedule_html, table_id="league-schedule-table"),
        },
    }

    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    if snapshot != expected:
        expected_s = json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        snapshot_s = json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        diff_lines = list(
            difflib.unified_diff(
                expected_s.splitlines(),
                snapshot_s.splitlines(),
                fromfile=str(_EXPECTED_PATH),
                tofile="actual_snapshot",
                lineterm="",
            )
        )
        max_lines = 400
        truncated = len(diff_lines) > max_lines
        diff_s = "\n".join(diff_lines[:max_lines])
        if truncated:
            diff_s += f"\n... ({len(diff_lines) - max_lines} more lines truncated)"
        raise AssertionError(
            "Golden regression snapshot mismatch.\n"
            "Do NOT update the golden fixture/expected snapshots without explicit user permission; "
            "fix webapp code/UI to restore the prior output.\n\n"
            f"{diff_s}"
        )

from __future__ import annotations

import contextlib
import datetime as dt
import json
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest


def _chrome_executable() -> Optional[str]:
    env_bin = str(os.environ.get("HM_CHROME_BIN") or "").strip()
    if env_bin and Path(env_bin).exists():
        return env_bin
    for p in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        if Path(p).exists():
            return p
    return None


def _maybe_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except Exception:
        return None
    return sync_playwright


def _fake_pairings_payload(*, n_players: int = 40, teammates_per_player: int = 8) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eligible_games = 12
    shift_games = 12
    for pidx in range(n_players):
        pid = 1000 + pidx
        player_jersey = str(10 + pidx)
        player_name = f"Player {pidx + 1}"
        for tidx in range(teammates_per_player):
            tid = 2000 + (pidx * 100) + tidx
            rows.append(
                {
                    "player_id": pid,
                    "player_jersey": player_jersey,
                    "player_name": player_name,
                    "teammate_id": tid,
                    "teammate_jersey": str(80 + tidx),
                    "teammate_name": f"Teammate {tidx + 1}",
                    "shift_games": shift_games,
                    "overlap_pct": float((tidx + 1) * 3.3),
                    "gf_together": tidx % 3,
                    "ga_together": tidx % 2,
                    "player_goals_on_ice_together": tidx % 2,
                    "player_assists_on_ice_together": tidx % 4,
                    "goals_collab_with_teammate": tidx % 2,
                    "assists_collab_with_teammate": tidx % 3,
                    "plus_minus_together": (tidx % 5) - 2,
                    "player_total_plus_minus": 3,
                    "player_shift_games": shift_games,
                    "teammate_total_plus_minus": -1,
                    "teammate_shift_games": shift_games,
                }
            )
    return {"ok": True, "eligible_games": eligible_games, "shift_games": shift_games, "rows": rows}


def _wrap_static_files(
    *, django_app: Callable[..., Any], static_dir: Path
) -> Callable[[dict[str, Any], Callable[..., Any]], list[bytes]]:
    static_root = static_dir.resolve()

    def app(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        path = str(environ.get("PATH_INFO") or "")
        if not path.startswith("/static/"):
            return list(django_app(environ, start_response))

        rel = path[len("/static/") :].lstrip("/")
        if not rel or rel.startswith("..") or "/../" in rel:
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

        p = (static_root / rel).resolve()
        if not str(p).startswith(str(static_root) + os.sep) or not p.exists() or not p.is_file():
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

        content_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        body = p.read_bytes()
        start_response(
            "200 OK",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ],
        )
        return [body]

    return app


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


@contextlib.contextmanager
def _wsgi_server(*, app: Callable[..., Any]) -> Iterator[str]:
    httpd = make_server("127.0.0.1", 0, app, handler_class=_QuietHandler)
    host, port = httpd.server_address[0], int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


@pytest.mark.ui
def should_keep_pairings_modal_header_and_first_two_columns_sticky(monkeypatch, webapp_db):
    sync_playwright = _maybe_playwright()
    if sync_playwright is None:
        pytest.skip("playwright is not installed (pip install -r requirements.txt)")

    chrome_bin = _chrome_executable()
    if chrome_bin is None:
        pytest.skip("No Chrome/Chromium found (set HM_CHROME_BIN or install google-chrome)")

    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")

    now = dt.datetime(2026, 1, 1, 0, 0, 0)
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    league = m.League.objects.create(
        id=1,
        name="Test League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team = m.Team.objects.create(
        id=44,
        user_id=int(owner.id),
        name="Team 44",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    opp = m.Team.objects.create(
        id=45,
        user_id=int(owner.id),
        name="Opp",
        is_external=True,
        logo_path=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(team.id),
        division_name="10 A",
        division_id=1,
        conference_id=0,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(opp.id),
        division_name="10 A",
        division_id=1,
        conference_id=0,
    )
    gt, _created = m.GameType.objects.get_or_create(
        name="Regular Season", defaults={"is_default": True}
    )
    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team.id),
        team2_id=int(opp.id),
        game_type_id=int(gt.id),
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=2,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(game.id),
        division_name="10 A",
        division_id=1,
        conference_id=0,
        sort_order=1,
    )
    p = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team.id),
        name="Skater",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.HkyGameShiftRow.objects.create(
        game_id=int(game.id),
        import_key="s1",
        source="test",
        team_id=int(team.id),
        player_id=int(p.id),
        team_side="home",
        period=1,
        game_seconds=0,
        game_seconds_end=10,
        video_seconds=None,
        video_seconds_end=None,
        created_at=now,
        updated_at=None,
    )

    from django.core.wsgi import get_wsgi_application

    django_app = get_wsgi_application()
    repo_root = Path(__file__).resolve().parents[1]
    static_dir = repo_root / "tools" / "webapp" / "static"
    wsgi_app = _wrap_static_files(django_app=django_app, static_dir=static_dir)

    pairings_payload = _fake_pairings_payload()

    with _wsgi_server(app=wsgi_app) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                executable_path=chrome_bin,
                args=["--no-sandbox", "--disable-gpu"],
            )
            page = browser.new_page(viewport={"width": 860, "height": 640})

            def _route_pairings(route, request):  # type: ignore[no-untyped-def]
                if "/api/hky/teams/" in request.url and "/pair_on_ice" in request.url:
                    route.fulfill(
                        status=200,
                        headers={"Content-Type": "application/json"},
                        body=json.dumps(pairings_payload),
                    )
                    return
                route.continue_()

            page.route("**/api/hky/teams/*/pair_on_ice*", _route_pairings)

            page.goto(f"{base_url}/public/leagues/1/teams/44", wait_until="domcontentloaded")
            page.locator("button.player-pairings-link").first.click()

            page.wait_for_selector("#team-player-pairings-modal[open]")
            page.wait_for_selector(
                "#team-player-pairings-body .table-scroll-y table.team-player-pairings-table"
            )
            page.wait_for_selector(
                "#team-player-pairings-body thead th[data-sticky-col='0'].sticky-col"
            )

            metrics = page.evaluate(
                """() => {
  const wrap = document.querySelector("#team-player-pairings-body .table-scroll-y");
  if (!wrap) return { ok: false, error: "missing wrap" };
  const table = wrap.querySelector("table.team-player-pairings-table");
  if (!table) return { ok: false, error: "missing table" };

  const stickyTh0 = wrap.querySelector("thead th[data-sticky-col='0']");
  const stickyTh1 = wrap.querySelector("thead th[data-sticky-col='1']");
  const thAny = wrap.querySelector("thead th");
  const bodyTd0 = wrap.querySelector("tbody td[data-sticky-col='0']");
  const bodyTd1 = wrap.querySelector("tbody td[data-sticky-col='1']");
  const nonStickyTh = wrap.querySelector("thead th:not([data-sticky-col])");

  if (!stickyTh0 || !stickyTh1 || !thAny || !bodyTd0 || !bodyTd1 || !nonStickyTh) {
    return { ok: false, error: "missing cells" };
  }

  const wrapRect = wrap.getBoundingClientRect();
  const rectRel = (el) => {
    const r = el.getBoundingClientRect();
    return { top: r.top - wrapRect.top, left: r.left - wrapRect.left, right: r.right - wrapRect.left };
  };

  const before = {
    headerTopRel: rectRel(thAny).top,
    th0LeftRel: rectRel(stickyTh0).left,
    th1LeftRel: rectRel(stickyTh1).left,
    td0LeftRel: rectRel(bodyTd0).left,
    td1LeftRel: rectRel(bodyTd1).left,
    stickyTh1Z: Number(getComputedStyle(stickyTh1).zIndex || "0"),
    nonStickyThZ: Number(getComputedStyle(nonStickyTh).zIndex || "0"),
    canScrollY: wrap.scrollHeight > wrap.clientHeight + 2,
    canScrollX: wrap.scrollWidth > wrap.clientWidth + 2,
  };

  wrap.scrollTop = Math.max(0, wrap.scrollHeight - wrap.clientHeight - 1);
  wrap.scrollLeft = Math.max(0, Math.min(900, wrap.scrollWidth - wrap.clientWidth - 1));

  // Force layout.
  wrap.getBoundingClientRect();

  const after = {
    headerTopRel: rectRel(thAny).top,
    th0LeftRel: rectRel(stickyTh0).left,
    th1LeftRel: rectRel(stickyTh1).left,
    td0LeftRel: rectRel(bodyTd0).left,
    td1LeftRel: rectRel(bodyTd1).left,
  };

  // Find an overlap point between sticky header col 1 and any non-sticky header.
  const stickyRect = stickyTh1.getBoundingClientRect();
  const others = Array.from(wrap.querySelectorAll("thead th:not([data-sticky-col])"));
  let overlap = null;
  for (const th of others) {
    const r = th.getBoundingClientRect();
    const left = Math.max(stickyRect.left, r.left);
    const right = Math.min(stickyRect.right, r.right);
    const top = Math.max(stickyRect.top, r.top);
    const bottom = Math.min(stickyRect.bottom, r.bottom);
    if (right - left > 2 && bottom - top > 2) {
      overlap = { x: (left + right) / 2, y: (top + bottom) / 2, otherText: (th.textContent || "").trim() };
      break;
    }
  }
  let overlapHit = null;
  if (overlap) {
    const hit = document.elementFromPoint(overlap.x, overlap.y);
    const hitTh = hit ? hit.closest("th") : null;
    overlapHit = {
      hitText: (hitTh ? hitTh.textContent : (hit ? hit.textContent : "") || "").trim(),
      hitIsSticky: !!hitTh && hitTh === stickyTh1,
    };
  }

  return { ok: true, before, after, overlap, overlapHit };
}"""
            )

            browser.close()

    assert metrics["ok"], metrics
    before = metrics["before"]
    after = metrics["after"]

    assert before["canScrollY"], metrics
    assert before["canScrollX"], metrics

    tol = 1.25
    assert abs(after["headerTopRel"] - before["headerTopRel"]) <= tol, metrics
    assert abs(after["td0LeftRel"] - before["td0LeftRel"]) <= tol, metrics
    assert abs(after["td1LeftRel"] - before["td1LeftRel"]) <= tol, metrics
    assert abs(after["th0LeftRel"] - before["th0LeftRel"]) <= tol, metrics
    assert abs(after["th1LeftRel"] - before["th1LeftRel"]) <= tol, metrics

    assert metrics["overlap"] is not None, metrics
    assert metrics["overlapHit"] is not None, metrics
    assert metrics["overlapHit"]["hitIsSticky"], metrics

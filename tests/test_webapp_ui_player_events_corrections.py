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
def should_render_corrected_player_events_in_red_with_tooltip(monkeypatch, webapp_db):
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
        name="Public League",
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
    notes = json.dumps({"game_video": "https://youtu.be/abc123"}, sort_keys=True)
    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team.id),
        team2_id=int(opp.id),
        game_type_id=int(gt.id),
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=notes,
        team1_score=2,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key="utah-1",
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
        jersey_number="13",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.PlayerStat.objects.create(
        user_id=int(owner.id),
        team_id=int(team.id),
        game_id=int(game.id),
        player_id=int(p.id),
        goals=1,
        assists=0,
        shots=0,
        pim=0,
        plus_minus=0,
        sog=0,
        expected_goals=0,
        completed_passes=0,
        giveaways=0,
        turnovers_forced=0,
        created_turnovers=0,
        takeaways=0,
        controlled_entry_for=0,
        controlled_entry_against=0,
        controlled_exit_for=0,
        controlled_exit_against=0,
        gt_goals=0,
        gw_goals=0,
        ot_goals=0,
        ot_assists=0,
        gf_counted=0,
        ga_counted=0,
    )

    from django.core.wsgi import get_wsgi_application

    django_app = get_wsgi_application()
    repo_root = Path(__file__).resolve().parents[1]
    static_dir = repo_root / "tools" / "webapp" / "static"
    wsgi_app = _wrap_static_files(django_app=django_app, static_dir=static_dir)

    payload = {
        "ok": True,
        "team_id": int(team.id),
        "player_id": int(p.id),
        "player_name": "Skater",
        "jersey_number": "13",
        "position": "F",
        "eligible_games": 1,
        "events": [
            {
                "kind": "attributed",
                "game_id": int(game.id),
                "game_starts_at": "2026-01-02T10:00:00Z",
                "game_type": "Regular Season",
                "opponent": "Opp",
                "game_url": None,
                "video_url": "https://youtu.be/abc123",
                "event_id": None,
                "event_type": "Goal",
                "event_type_key": "goal",
                "team_side": "Home",
                "period": 2,
                "game_time": "03:14",
                "video_time": "10:20",
                "game_seconds": 194,
                "video_seconds": 620,
                "details": "A1: #3\\nA2: #8",
                "correction": {
                    "version": 1,
                    "note": "see video",
                    "reason": "Swap scorer/assist for goal at P2 03:14",
                    "changes": [{"field": "jersey", "from": "3", "to": "13"}],
                },
                "for_against": "For",
                "source": "correction",
            }
        ],
        "on_ice_goals": [],
    }

    with _wsgi_server(app=wsgi_app) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                executable_path=chrome_bin,
                args=["--no-sandbox", "--disable-gpu"],
            )
            page = browser.new_page(viewport={"width": 880, "height": 700})

            def _route_events(route, request):  # type: ignore[no-untyped-def]
                if "/api/hky/teams/" in request.url and "/events" in request.url:
                    route.fulfill(
                        status=200,
                        headers={"Content-Type": "application/json"},
                        body=json.dumps(payload),
                    )
                    return
                route.continue_()

            page.route("**/api/hky/teams/*/players/*/events*", _route_events)
            page.goto(f"{base_url}/public/leagues/1/teams/44", wait_until="domcontentloaded")

            page.locator("button.player-events-link").first.click()
            page.wait_for_selector("#team-player-events-modal[open]")
            page.wait_for_selector("#team-player-events-body table.team-player-events-table")
            page.wait_for_selector(
                "#team-player-events-body table.team-player-events-table tbody tr.event-corrected"
            )

            metrics = page.evaluate(
                """() => {
  const tr = document.querySelector("#team-player-events-body table.team-player-events-table tbody tr.event-corrected");
  if (!tr) return { ok: false, error: "missing corrected row" };
  const td0 = tr.querySelector("td");
  const tds = Array.from(tr.querySelectorAll("td"));
  const detailsTd = tds[6] || null;
  const title = detailsTd ? (detailsTd.getAttribute("title") || "") : "";
  const detailsText = detailsTd ? (detailsTd.textContent || "") : "";
  return {
    ok: true,
    hasClass: tr.classList.contains("event-corrected"),
    color: td0 ? getComputedStyle(td0).color : "",
    title,
    detailsText,
    detailsHasA1: detailsText.indexOf("A1:") >= 0,
    detailsHasA2: detailsText.indexOf("A2:") >= 0,
    detailsWhiteSpace: detailsTd ? getComputedStyle(detailsTd).whiteSpace : "",
  };
}"""
            )
            browser.close()

    assert metrics["ok"], metrics
    assert metrics["hasClass"], metrics
    assert "rgb(255, 138, 138)" in metrics["color"], metrics
    assert "Scorer: #3 → #13" in metrics["title"], metrics
    assert "Note: see video" in metrics["title"], metrics
    assert "Video clip: yes" in metrics["title"], metrics
    assert metrics["detailsHasA1"], metrics
    assert metrics["detailsHasA2"], metrics
    assert metrics["detailsWhiteSpace"] == "pre-line", metrics

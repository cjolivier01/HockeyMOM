from __future__ import annotations

import contextlib
import datetime as dt
import json
import mimetypes
import os
import re
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


def _ui_login(*, page, base_url: str, email: str, password: str) -> None:
    page.goto(f"{base_url}/login", wait_until="domcontentloaded")
    page.fill('input[name="email"]', email)
    page.fill('input[name="password"]', password)
    page.click('button[type="submit"]')
    page.wait_for_load_state("domcontentloaded")


def _first_int(text: str) -> int:
    m = re.search(r"(\d+)", str(text or ""))
    return int(m.group(1)) if m else 0


@pytest.mark.ui
def should_track_player_events_and_clip_views_and_show_to_owner(monkeypatch, webapp_db):
    sync_playwright = _maybe_playwright()
    if sync_playwright is None:
        pytest.skip("playwright is not installed (pip install -r requirements.txt)")

    chrome_bin = _chrome_executable()
    if chrome_bin is None:
        pytest.skip("No Chrome/Chromium found (set HM_CHROME_BIN or install google-chrome)")

    from werkzeug.security import generate_password_hash

    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")

    now = dt.datetime(2026, 1, 1, 0, 0, 0)
    owner_pass = "ownerpw123"
    viewer_pass = "viewerpw123"
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash=generate_password_hash(owner_pass),
        name="Owner",
        created_at=now,
        default_league_id=1,
        video_clip_len_s=None,
    )
    viewer = m.User.objects.create(
        id=20,
        email="viewer@example.com",
        password_hash=generate_password_hash(viewer_pass),
        name="Viewer",
        created_at=now,
        default_league_id=1,
        video_clip_len_s=None,
    )
    league = m.League.objects.create(
        id=1,
        name="Shared League",
        owner_user_id=int(owner.id),
        is_shared=False,
        is_public=False,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueMember.objects.create(
        league_id=int(league.id),
        user_id=int(viewer.id),
        role="viewer",
        created_at=now,
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
        division_name="12AA",
        division_id=1,
        conference_id=0,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id),
        team_id=int(opp.id),
        division_name="12AA",
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
        external_game_key="game-1001",
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(game.id),
        division_name="12AA",
        division_id=1,
        conference_id=0,
        sort_order=1,
    )
    player = m.Player.objects.create(
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

    ev_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    event_row_id = 9001
    m.HkyGameEventRow.objects.create(
        id=int(event_row_id),
        game_id=int(game.id),
        event_type_id=int(ev_goal.id),
        import_key="g-1",
        team_id=int(team.id),
        player_id=int(player.id),
        source="timetoscore",
        event_id=1,
        team_raw="Home",
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="11:50",
        video_time="00:10",
        game_seconds=10,
        game_seconds_end=None,
        video_seconds=10,
        details="Goal",
        attributed_players=str(player.name),
        attributed_jerseys=str(player.jersey_number),
        created_at=now,
        updated_at=None,
    )

    from django.core.wsgi import get_wsgi_application

    django_app = get_wsgi_application()
    repo_root = Path(__file__).resolve().parents[1]
    static_dir = repo_root / "tools" / "webapp" / "static"
    wsgi_app = _wrap_static_files(django_app=django_app, static_dir=static_dir)

    with _wsgi_server(app=wsgi_app) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                executable_path=chrome_bin,
                args=["--no-sandbox", "--disable-gpu"],
            )

            # Viewer opens Player Events + watches one clip (increments both counters).
            ctx_viewer = browser.new_context(viewport={"width": 980, "height": 760})
            page_viewer = ctx_viewer.new_page()
            _ui_login(
                page=page_viewer, base_url=base_url, email=str(viewer.email), password=viewer_pass
            )
            page_viewer.goto(f"{base_url}/hky/games/{int(game.id)}", wait_until="domcontentloaded")
            page_viewer.locator("button.player-events-btn", has_text=str(player.name)).click()
            page_viewer.wait_for_selector("#events-drilldown-modal[open]")
            page_viewer.wait_for_selector("#events-drilldown-body table.events-drilldown-table")
            page_viewer.locator("#events-drilldown-body button .hm-clip-time").first.click()
            page_viewer.wait_for_selector("#events-drilldown-video-pane", state="visible")

            # Owner sees the updated counts in the same modal (from either game or team page).
            ctx_owner = browser.new_context(viewport={"width": 980, "height": 760})
            page_owner = ctx_owner.new_page()
            _ui_login(
                page=page_owner, base_url=base_url, email=str(owner.email), password=owner_pass
            )
            page_owner.goto(f"{base_url}/hky/games/{int(game.id)}", wait_until="domcontentloaded")
            page_owner.locator("button.player-events-btn", has_text=str(player.name)).click()
            page_owner.wait_for_selector("#events-drilldown-modal[open]")

            page_owner.wait_for_function(
                """() => {
  const el = document.querySelector("#events-drilldown-pageviews [data-role='count']");
  if (!el) return false;
  const n = parseInt(String(el.textContent || "").trim() || "0", 10);
  return Number.isFinite(n) && n >= 1;
}"""
            )
            views_txt = page_owner.locator(
                "#events-drilldown-pageviews [data-role='count']"
            ).inner_text()

            clip_sel = (
                "#events-drilldown-body "
                f"[data-hm-clip-views='1'][data-event-row-id='{int(event_row_id)}']"
            )
            page_owner.wait_for_function(
                """(sel) => {
  const el = document.querySelector(sel);
  if (!el) return false;
  const txt = String(el.textContent || "").trim();
  const n = parseInt(txt || "0", 10);
  return txt.length > 0 && Number.isFinite(n);
}""",
                arg=clip_sel,
            )
            clip_txt = page_owner.locator(clip_sel).first.inner_text()

            # Same player_events count when opening from the team page.
            page_owner.goto(f"{base_url}/teams/{int(team.id)}", wait_until="domcontentloaded")
            page_owner.locator("button.player-events-link", has_text=str(player.name)).click()
            page_owner.wait_for_selector("#team-player-events-modal[open]")
            page_owner.wait_for_function(
                """() => {
  const el = document.querySelector("#team-player-events-pageviews [data-role='count']");
  if (!el) return false;
  const n = parseInt(String(el.textContent || "").trim() || "0", 10);
  return Number.isFinite(n) && n >= 1;
}"""
            )
            team_views_txt = page_owner.locator(
                "#team-player-events-pageviews [data-role='count']"
            ).inner_text()

            team_clip_sel = (
                "#team-player-events-body "
                f"[data-hm-clip-views='1'][data-event-row-id='{int(event_row_id)}']"
            )
            page_owner.wait_for_function(
                """(sel) => {
  const el = document.querySelector(sel);
  if (!el) return false;
  const txt = String(el.textContent || "").trim();
  const n = parseInt(txt || "0", 10);
  return txt.length > 0 && Number.isFinite(n);
}""",
                arg=team_clip_sel,
            )
            team_clip_txt = page_owner.locator(team_clip_sel).first.inner_text()

            browser.close()

    assert _first_int(views_txt) >= 1
    assert _first_int(clip_txt) >= 1
    assert _first_int(team_views_txt) == _first_int(views_txt)
    assert _first_int(team_clip_txt) >= 1

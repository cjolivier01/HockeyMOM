from __future__ import annotations

import contextlib
import datetime as dt
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
def should_show_goal_in_game_player_events_when_attributed_fields_missing(monkeypatch, webapp_db):
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
        id=10, email="owner@example.com", password_hash="x", created_at=now
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

    team1 = m.Team.objects.create(
        id=292,
        user_id=int(owner.id),
        name="Home",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=293,
        user_id=int(owner.id),
        name="Away",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team1.id), division_name="12AA"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team2.id), division_name="12AA"
    )

    m.Player.objects.create(
        id=5001,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Ethan L Olivier",
        jersey_number="1",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )

    game = m.HkyGame.objects.create(
        id=2574,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=1,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(game.id),
        division_name="12AA",
        division_id=None,
        conference_id=None,
        sort_order=None,
    )

    et_goal = m.HkyEventType.objects.create(key="goal", name="Goal", created_at=now)
    # Goal event is attributed via player_id, but attributed_players/attributed_jerseys are empty.
    m.HkyGameEventRow.objects.create(
        id=9001,
        game_id=int(game.id),
        event_type=et_goal,
        import_key="goal1",
        team_id=int(team1.id),
        player_id=5001,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="10:00",
        game_seconds=0,
        video_time="1:23",
        video_seconds=83,
        details="",
        attributed_players="",
        attributed_jerseys="",
        on_ice_players_home="Ethan L Olivier (#1)",
        on_ice_players_away="",
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
            page = browser.new_page(viewport={"width": 1100, "height": 820})
            page.goto(f"{base_url}/public/leagues/1/hky/games/2574", wait_until="domcontentloaded")

            # Ensure the per-game player stats table shows a goal for Ethan.
            stats = page.evaluate(
                """() => {
  const btn = Array.from(document.querySelectorAll('button.player-events-btn')).find(b => (b.textContent || '').includes('Ethan L Olivier'));
  if (!btn) return { ok: false, error: 'missing player button' };
  const tr = btn.closest('tr');
  if (!tr) return { ok: false, error: 'missing player row' };
  const cells = Array.from(tr.querySelectorAll('td')).map(td => (td.textContent || '').trim());
  return { ok: true, cells };
}"""
            )
            assert stats["ok"], stats
            # Typically columns are: jersey, name, ... and Goals somewhere later; just ensure a "1" exists.
            assert "1" in stats["cells"], stats

            page.click('button.player-events-btn:has-text("Ethan L Olivier")')
            page.wait_for_selector("#events-drilldown-modal[open]")
            page.wait_for_selector("#events-drilldown-body")

            out = page.evaluate(
                """() => {
  const body = document.getElementById('events-drilldown-body');
  if (!body) return { ok: false, error: 'missing body' };
  const hasGoalsHeading = Array.from(body.querySelectorAll('h4')).some(h => (h.textContent || '').toLowerCase().includes('goals'));
  const text = (body.textContent || '');
  return { ok: true, hasGoalsHeading, hasGoalRow: text.includes('Goal') || text.includes('Goals') };
}"""
            )
            assert out["ok"], out
            assert out["hasGoalsHeading"], out

            browser.close()

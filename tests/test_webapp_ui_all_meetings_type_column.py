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
def should_show_tournament_type_in_all_meetings_when_external_division(monkeypatch, webapp_db):
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

    gt_pre, _ = m.GameType.objects.get_or_create(name="Preseason", defaults={"is_default": False})

    team_a = m.Team.objects.create(
        id=43,
        user_id=int(owner.id),
        name="Home A",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    team_b = m.Team.objects.create(
        id=44,
        user_id=int(owner.id),
        name="Away A",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_a.id), division_name="12AA"
    )
    m.LeagueTeam.objects.create(
        league_id=int(league.id), team_id=int(team_b.id), division_name="12AA"
    )

    g_tour = m.HkyGame.objects.create(
        id=1000,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=None,
        starts_at=dt.datetime(2025, 10, 10, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=1,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(g_tour.id),
        division_name="External Tournament",
        division_id=None,
        conference_id=None,
        sort_order=None,
    )

    g_current = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team_a.id),
        team2_id=int(team_b.id),
        game_type_id=int(gt_pre.id),
        starts_at=dt.datetime(2025, 11, 1, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=3,
        team2_score=4,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=int(league.id),
        game_id=int(g_current.id),
        division_name="12AA",
        division_id=None,
        conference_id=None,
        sort_order=None,
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
            page = browser.new_page(viewport={"width": 980, "height": 760})
            page.goto(f"{base_url}/public/leagues/1/hky/games/1001", wait_until="domcontentloaded")
            page.wait_for_selector("table.meeting-table")

            metrics = page.evaluate(
                """() => {
  const table = document.querySelector("table.meeting-table");
  if (!table) return { ok: false, error: "missing meeting table" };
  const rows = Array.from(table.querySelectorAll("tbody tr"));
  const byDate = {};
  for (const tr of rows) {
    const tds = Array.from(tr.querySelectorAll("td"));
    const date = (tds[0]?.textContent || "").trim();
    const type = (tds[4]?.textContent || "").trim();
    byDate[date] = type;
  }
  return { ok: true, byDate };
}"""
            )
            browser.close()

    assert metrics["ok"], metrics
    assert metrics["byDate"].get("2025-10-10") == "Tournament", metrics
    assert metrics["byDate"].get("2025-11-01") == "Preseason", metrics

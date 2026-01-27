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
def should_hide_player_name_in_assist_details_in_game_player_events_modal(monkeypatch, webapp_db):
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
        id=43,
        user_id=int(owner.id),
        name="San Jose Jr Sharks 12AA-2",
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=44,
        user_id=int(owner.id),
        name="Opponent",
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
        id=501,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Ethan L Olivier",
        jersey_number="1",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=502,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Bob Helper",
        jersey_number="2",
        position="F",
        shoots="L",
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=503,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Other Assist",
        jersey_number="3",
        position="F",
        shoots="L",
        created_at=now,
        updated_at=None,
    )

    game = m.HkyGame.objects.create(
        id=760,
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

    # Create a goal at P1 1:00 by #2, with assists by #1 (Ethan) and #3.
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
        "Video Time",
        "Video Seconds",
        "Details",
        "Attributed Players",
        "Attributed Jerseys",
    ]
    events_rows = [
        {
            "Event Type": "Goal",
            "Source": "t2s",
            "Team Raw": "Home",
            "Team Side": "Home",
            "For/Against": "For",
            "Team Rel": "Home",
            "Period": "1",
            "Game Time": "1:00",
            "Game Seconds": "60",
            "Video Seconds": "120",
            "Attributed Players": "Bob Helper",
            "Attributed Jerseys": "2",
        },
        {
            "Event Type": "Assist",
            "Source": "t2s",
            "Team Raw": "Home",
            "Team Side": "Home",
            "For/Against": "For",
            "Team Rel": "Home",
            "Period": "1",
            "Game Time": "1:00",
            "Game Seconds": "60",
            "Video Seconds": "120",
            "Attributed Players": "Ethan L Olivier",
            "Attributed Jerseys": "1",
        },
        {
            "Event Type": "Assist",
            "Source": "t2s",
            "Team Raw": "Home",
            "Team Side": "Home",
            "For/Against": "For",
            "Team Rel": "Home",
            "Period": "1",
            "Game Time": "1:00",
            "Game Seconds": "60",
            "Video Seconds": "120",
            "Attributed Players": "Other Assist",
            "Attributed Jerseys": "3",
        },
    ]

    from tools.webapp.scripts import import_time2score as importer
    from tools.webapp.django_app import views as v

    events_csv = importer._to_csv_text(events_headers, events_rows)  # type: ignore[attr-defined]
    up = v._upsert_game_event_rows_from_events_csv(  # type: ignore[attr-defined]
        game_id=int(game.id),
        events_csv=str(events_csv),
        replace=True,
        create_missing_players=False,
    )
    assert up["ok"] is True

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
            page = browser.new_page(viewport={"width": 980, "height": 820})
            page.goto(f"{base_url}/public/leagues/1/hky/games/760", wait_until="domcontentloaded")

            # Open Player Events modal by clicking the player's name/button.
            page.click('button.player-events-btn:has-text("Ethan L Olivier")')
            page.wait_for_selector("#events-drilldown-body table.events-drilldown-table")

            out = page.evaluate(
                """() => {
  const body = document.getElementById("events-drilldown-body");
  if (!body) return { ok: false, error: "missing modal body" };
  const hs = Array.from(body.querySelectorAll("h4"));
  const h = hs.find(x => {
    const t = String(x.textContent || "").replace(/\\s+/g, " ").trim().toLowerCase();
    return t.includes("assists");
  });
  if (!h) return { ok: false, error: "missing Assists heading" };
  let t = h.nextElementSibling;
  let tbl = null;
  while (t) {
    if (t.tagName && t.tagName.toLowerCase() === "table") { tbl = t; break; }
    if (t.querySelector) {
      const cand = t.querySelector("table");
      if (cand) { tbl = cand; break; }
    }
    t = t.nextElementSibling;
  }
  if (!tbl) return { ok: false, error: "missing assists table" };
  const firstRow = tbl.querySelector("tbody tr");
  if (!firstRow) return { ok: false, error: "no assist rows" };
  const tds = Array.from(firstRow.querySelectorAll("td"));
  const details = (tds[3]?.textContent || "").trim(); // P, Time, Video, Details, Source
  return { ok: true, details };
}"""
            )
            assert out["ok"] is True, out
            details = str(out["details"] or "")
            assert "Goal: #2 Bob Helper" in details
            assert "Other A: #3 Other Assist" in details
            assert "Ethan" not in details

            browser.close()

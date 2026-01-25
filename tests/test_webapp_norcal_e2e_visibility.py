import os
import re
import time

import pytest


def _maybe_ensure_shared_league(sess, *, base_url: str, league_name: str) -> None:
    """
    Best-effort: ensure the shared league exists on the target instance.

    For local E2E (`HM_WEBAPP_E2E_URL=http://localhost`), import endpoints may be callable without a token.
    For remote instances, set `HM_WEBAPP_E2E_IMPORT_TOKEN` if import endpoints require auth.
    """
    import requests

    token = os.environ.get("HM_WEBAPP_E2E_IMPORT_TOKEN") or ""
    headers = {}
    if token:
        headers["X-HM-Import-Token"] = str(token)
    try:
        r = sess.post(
            f"{base_url}/api/import/hockey/ensure_league",
            json={"league_name": str(league_name), "shared": True},
            headers=headers,
            timeout=60,
        )
    except requests.RequestException:
        return
    if r.status_code != 200:
        return
    return


def _extract_csrf_token(html: str) -> str | None:
    m = re.search(
        r'name="csrfmiddlewaretoken"[^>]*value="(?P<tok>[^"]+)"', html or "", flags=re.IGNORECASE
    )
    return m.group("tok") if m else None


def _get_csrf_token(sess, url: str) -> str:
    r = sess.get(url, timeout=60)
    assert r.status_code == 200
    tok = _extract_csrf_token(r.text) or sess.cookies.get("csrftoken")
    assert tok, f"Expected CSRF token from {url}"
    return str(tok)


@pytest.mark.skipif(
    not os.environ.get("HM_WEBAPP_E2E_URL"),
    reason="Set HM_WEBAPP_E2E_URL to run live webapp E2E checks",
)
def should_show_shared_caha_league_and_stats_for_all_users():
    import requests

    base_url = os.environ["HM_WEBAPP_E2E_URL"].rstrip("/")
    sess = requests.Session()

    # Create a fresh user (register logs you in).
    uniq = f"{int(time.time())}-{os.getpid()}"
    email = f"e2e-caha-{uniq}@example.com"
    password = "test-password-123"
    csrf = _get_csrf_token(sess, f"{base_url}/register")
    r = sess.post(
        f"{base_url}/register",
        data={
            "csrfmiddlewaretoken": csrf,
            "email": email,
            "password": password,
            "name": f"E2E CAHA {uniq}",
        },
        headers={"Referer": f"{base_url}/register"},
        allow_redirects=True,
        timeout=60,
    )
    assert r.status_code == 200

    # Shared league must be visible to a brand-new user.
    _maybe_ensure_shared_league(sess, base_url=base_url, league_name="CAHA")
    leagues_resp = sess.get(f"{base_url}/leagues", timeout=60)
    assert leagues_resp.status_code == 200
    leagues_html = leagues_resp.text
    if "Logout" not in leagues_html:
        pytest.skip("E2E instance did not keep the logged-in session after /register")
    if "CAHA" not in leagues_html:
        pytest.skip(
            "CAHA shared league not present on E2E instance; seed the instance (e.g. ./import_webapp.sh)"
        )

    m = re.search(r'<option value="(?P<id>\d+)"[^>]*>\s*CAHA\s*<', leagues_html)
    if not m:
        pytest.skip("CAHA league not present in the league selector for a new user")
    league_id = m.group("id")

    # Switch context to the shared league.
    csrf = _extract_csrf_token(leagues_html) or sess.cookies.get("csrftoken")
    assert csrf, "Expected CSRF token on /leagues page"
    r = sess.post(
        f"{base_url}/league/select",
        data={"csrfmiddlewaretoken": csrf, "league_id": league_id},
        headers={"Referer": f"{base_url}/leagues"},
        allow_redirects=True,
        timeout=60,
    )
    assert r.status_code == 200

    teams_html = sess.get(f"{base_url}/teams", timeout=60).text
    team_ids = sorted({int(x) for x in re.findall(r'href="/teams/(\d+)"', teams_html)})
    assert len(team_ids) >= 10, f"Expected many teams in CAHA, got {len(team_ids)}"

    # At least one team page should show a roster and indicate read-only.
    found_team_with_players = False
    for tid in team_ids[:20]:
        html = sess.get(f"{base_url}/teams/{tid}", timeout=60).text
        if "Read-only" not in html:
            continue
        if "No players yet." in html:
            continue
        # Roster section is present when the team has players.
        if "Roster" in html and "hm-player-name" in html:
            found_team_with_players = True
            break
    assert (
        found_team_with_players
    ), "Expected at least one CAHA team to have players visible to a non-owner user"

    schedule_html = sess.get(f"{base_url}/schedule", timeout=60).text
    game_ids = sorted(
        {int(x) for x in re.findall(r'href="/hky/games/(\d+)(?:\\?|")', schedule_html)}
    )
    assert len(game_ids) >= 10, f"Expected many games in CAHA schedule, got {len(game_ids)}"

    # Find a game page that renders player stats rows for a non-owner user.
    found_game_with_players = False
    for gid in game_ids[:50]:
        html = sess.get(f"{base_url}/hky/games/{gid}", timeout=60).text
        # Read-only pages don't render form inputs; detect the presence of player stats tables/rows.
        if 'data-player-stats-table="1"' in html and "hm-player-name" in html:
            found_game_with_players = True
            break
    assert (
        found_game_with_players
    ), "Expected at least one CAHA game to show player stats rows to a non-owner user"

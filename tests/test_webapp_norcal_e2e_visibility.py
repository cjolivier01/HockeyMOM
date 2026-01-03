import os
import re
import time

import pytest


@pytest.mark.skipif(not os.environ.get("HM_WEBAPP_E2E_URL"), reason="Set HM_WEBAPP_E2E_URL to run live webapp E2E checks")
def should_show_shared_norcal_league_and_stats_for_all_users():
    import requests

    base_url = os.environ["HM_WEBAPP_E2E_URL"].rstrip("/")
    sess = requests.Session()

    # Create a fresh user (register logs you in).
    uniq = f"{int(time.time())}-{os.getpid()}"
    email = f"e2e-norcal-{uniq}@example.com"
    password = "test-password-123"
    r = sess.post(
        f"{base_url}/register",
        data={"email": email, "password": password, "name": f"E2E Norcal {uniq}"},
        allow_redirects=True,
        timeout=60,
    )
    assert r.status_code == 200

    # Shared league must be visible to a brand-new user.
    leagues_html = sess.get(f"{base_url}/leagues", timeout=60).text
    assert "Norcal" in leagues_html

    m = re.search(r'<option value="(?P<id>\d+)"[^>]*>\s*Norcal\s*<', leagues_html)
    assert m, "Expected Norcal to be present in the league selector for a new user"
    league_id = m.group("id")

    # Switch context to the shared league.
    r = sess.post(f"{base_url}/league/select", data={"league_id": league_id}, allow_redirects=True, timeout=60)
    assert r.status_code == 200

    teams_html = sess.get(f"{base_url}/teams", timeout=60).text
    team_ids = sorted({int(x) for x in re.findall(r'href="/teams/(\d+)"', teams_html)})
    assert len(team_ids) >= 10, f"Expected many teams in Norcal, got {len(team_ids)}"

    # At least one team page should show a roster and indicate read-only.
    found_team_with_players = False
    for tid in team_ids[:20]:
        html = sess.get(f"{base_url}/teams/{tid}", timeout=60).text
        if "Read-only (shared league)" not in html:
            continue
        if "No players yet." in html:
            continue
        if "<h3>Players</h3>" in html:
            found_team_with_players = True
            break
    assert found_team_with_players, "Expected at least one Norcal team to have players visible to a non-owner user"

    schedule_html = sess.get(f"{base_url}/schedule", timeout=60).text
    game_ids = sorted({int(x) for x in re.findall(r'href="/hky/games/(\d+)"', schedule_html)})
    assert len(game_ids) >= 10, f"Expected many games in Norcal schedule, got {len(game_ids)}"

    # Find a game page that renders roster inputs (player list) for a non-owner user.
    found_game_with_players = False
    for gid in game_ids[:50]:
        html = sess.get(f"{base_url}/hky/games/{gid}", timeout=60).text
        if "Read-only (shared league)" not in html:
            continue
        if re.search(r'name="ps_goals_\d+"', html):
            found_game_with_players = True
            break
    assert found_game_with_players, "Expected at least one Norcal game to show player stats rows to a non-owner user"

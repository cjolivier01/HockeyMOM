from __future__ import annotations

import bs4


def should_parse_caha_division_names_from_schedule_headers(monkeypatch):
    from hmlib.time2score import caha_lib

    html = """
    <html><body>
      <table>
        <tr><th>10U B West Schedule</th></tr>
        <tr><th><a href="display-league-stats?stat_class=1&league=3&season=31&level=136&conf=0">Division Player Stats</a></th></tr>
        <tr><th>Team</th></tr>
        <tr><td><a href="display-schedule?team=123&league=3&stat_class=1">Team One</a></td></tr>
        <tr><td><a href="display-schedule?team=124&league=3&stat_class=1">Team Two</a></td></tr>
        <tr><th>12U A Schedule</th></tr>
        <tr><th><a href="display-league-stats?stat_class=1&league=3&season=31&level=4&conf=0">Division Player Stats</a></th></tr>
        <tr><th>Team</th></tr>
        <tr><td><a href="display-schedule?team=200&league=3&stat_class=1">Other Team</a></td></tr>
      </table>
    </body></html>
    """

    def fake_get_html(_url: str, params=None, log=False):  # noqa: ANN001
        return bs4.BeautifulSoup(html, "html5lib")

    monkeypatch.setattr(caha_lib.util, "get_html", fake_get_html)
    divs = caha_lib.scrape_season_divisions(season_id=31)
    names = {d["name"] for d in divs}
    assert "10 B West" in names
    assert any(t["name"] == "Team One" for d in divs for t in d.get("teams", []))


def should_map_numeric_scorers_to_roster_names():
    from hmlib.time2score import normalize

    stats = {
        "homePlayers": [{"number": "88", "position": "F", "name": "Alice"}, {"number": "76", "position": "D", "name": "Bob"}],
        "awayPlayers": [],
        "homeScoring": [{"goal": "88", "assist1": "76", "assist2": ""}],
        "awayScoring": [],
    }
    out = normalize.aggregate_goals_assists(stats)
    out_by_name = {r["name"]: r for r in out}
    assert out_by_name["Alice"]["goals"] == 1
    assert out_by_name["Bob"]["assists"] == 1

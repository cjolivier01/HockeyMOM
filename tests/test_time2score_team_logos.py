from __future__ import annotations

import bs4


def should_scrape_team_logo_url_from_team_page(monkeypatch):
    from hmlib.time2score import direct

    html = """
    <html><body>
      <img src="https://se-portal-production.s3.amazonaws.com/uploads/3848/logo.png" height="150" />
    </body></html>
    """

    from hmlib.time2score import util

    def fake_get_html(_url: str, params=None, log=False):  # noqa: ANN001
        return bs4.BeautifulSoup(html, "html5lib")

    monkeypatch.setattr(util, "get_html", fake_get_html)
    url = direct.scrape_team_logo_url("caha", season_id=31, team_id=58)
    assert url == "https://se-portal-production.s3.amazonaws.com/uploads/3848/logo.png"


def should_ignore_placeholder_team_logo_urls(monkeypatch):
    from hmlib.time2score import direct
    from hmlib.time2score import util

    html = "<html><body><img src=\"https://assets.sharksice.timetoscore.com/\" /></body></html>"

    def fake_get_html(_url: str, params=None, log=False):  # noqa: ANN001
        return bs4.BeautifulSoup(html, "html5lib")

    monkeypatch.setattr(util, "get_html", fake_get_html)
    assert direct.scrape_team_logo_url("sharksice", season_id=72, team_id=123) is None

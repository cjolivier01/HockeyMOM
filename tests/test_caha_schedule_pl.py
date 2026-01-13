from __future__ import annotations

import bs4


def should_return_empty_rows_when_schedule_table_missing(monkeypatch, caplog) -> None:
    from hmlib.time2score import caha_schedule_pl

    html = """
    <html>
      <body>
        <form action="schedule.pl">
          <input type="hidden" name="d" value="33" />
          <select name="t"><option value="0">-- Entire Schedule --</option></select>
          <select name="w"><option value="0">-- All Weeks --</option></select>
        </form>
      </body>
    </html>
    """.strip()
    soup = bs4.BeautifulSoup(html, "html.parser")

    def _fake_get_html(_url: str, *, params=None):  # noqa: ANN001
        return soup

    monkeypatch.setattr(caha_schedule_pl, "_get_html", _fake_get_html)

    group = caha_schedule_pl.ScheduleGroupLink(
        year=2025,
        age_group="12U",
        column="AAA",
        label="AAA Minor",
        d=33,
        url="https://caha.com/schedule.pl?d=33&y=2025&",
    )
    with caplog.at_level("WARNING"):
        out = caha_schedule_pl.scrape_schedule_group(group)
    assert out == []
    assert "no schedule rows found" in caplog.text.lower()


def should_parse_schedule_table_with_expected_header(monkeypatch) -> None:
    from hmlib.time2score import caha_schedule_pl

    html = """
    <html>
      <body>
        <table>
          <tr>
            <th>Date</th><th>Time</th><th>GM</th><th>Home</th><th>Score</th><th>Visitor</th><th>Score</th><th>Rink</th><th>Type</th>
          </tr>
          <tr>
            <td>09/28/25</td><td>12:00p</td><td>12011</td><td>Jr. Ducks</td><td>10</td><td>Jr. Kings</td><td>2</td><td>Great Park</td><td>Reg</td>
          </tr>
        </table>
      </body>
    </html>
    """.strip()
    soup = bs4.BeautifulSoup(html, "html.parser")

    def _fake_get_html(_url: str, *, params=None):  # noqa: ANN001
        return soup

    monkeypatch.setattr(caha_schedule_pl, "_get_html", _fake_get_html)

    group = caha_schedule_pl.ScheduleGroupLink(
        year=2025,
        age_group="12U",
        column="AAA",
        label="AAA Major",
        d=5,
        url="https://caha.com/schedule.pl?d=5&y=2025&",
    )
    out = caha_schedule_pl.scrape_schedule_group(group)
    assert len(out) == 1
    row = out[0]
    assert row.game_number == 12011
    assert row.home == "Jr. Ducks"
    assert row.away == "Jr. Kings"
    assert row.home_score == 10
    assert row.away_score == 2
    assert row.rink == "Great Park"
    assert row.game_type == "Reg"

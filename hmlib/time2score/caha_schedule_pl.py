"""Scraper for CAHA's public tier schedule pages at https://caha.com/schedule.pl.

This is *not* the TimeToScore site. The CAHA schedule pages expose basic schedule + score info
for Tier (AA/AAA) without reliable links back to TimeToScore game ids.

Consumers should treat the scraped "GM" numbers as CAHA schedule game numbers, not TimeToScore ids.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urljoin, urlsplit

import bs4
import requests

logger = logging.getLogger(__name__)


SCHEDULE_INDEX_URL = "https://caha.com/schedule.pl"


_DEFAULT_HEADERS: dict[str, str] = {
    # CAHA blocks naive scrapers (often 406/403); send a browser-ish request.
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://caha.com/",
}


def _http_timeout_s() -> float:
    try:
        return float(os.environ.get("HM_CAHA_SCHEDULE_HTTP_TIMEOUT", "30"))
    except Exception:
        return 30.0


def _get_html(url: str, *, params: dict[str, str] | None = None) -> bs4.BeautifulSoup:
    r = requests.get(url, params=params, headers=_DEFAULT_HEADERS, timeout=_http_timeout_s())
    r.raise_for_status()
    return bs4.BeautifulSoup(r.text, "html.parser")


def _norm(s: str) -> str:
    return " ".join(str(s or "").replace("\xa0", " ").split()).strip()


def _parse_int(v: object) -> Optional[int]:
    s = str(v or "").strip()
    if not s:
        return None
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _parse_caha_time_token(time_s: str) -> Optional[dt.time]:
    s = _norm(time_s).lower()
    if not s or s in {"tbd", "time"}:
        return None
    # Common format: "9:40a" / "12:00p"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})([ap])", s)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    ap = m.group(3)
    if hour < 1 or hour > 12 or minute < 0 or minute > 59:
        return None
    if ap == "a":
        hour_24 = 0 if hour == 12 else hour
    else:
        hour_24 = 12 if hour == 12 else hour + 12
    try:
        return dt.time(hour_24, minute, 0)
    except Exception:
        return None


def _parse_caha_date_token(date_s: str) -> Optional[dt.date]:
    s = _norm(date_s)
    if not s or s.lower() in {"date"}:
        return None
    # Common format: "09/28/25"
    try:
        return dt.datetime.strptime(s, "%m/%d/%y").date()
    except Exception:
        return None


def _parse_starts_at(date_s: str, time_s: str) -> Optional[dt.datetime]:
    d = _parse_caha_date_token(date_s)
    t = _parse_caha_time_token(time_s)
    if d is None or t is None:
        return None
    return dt.datetime.combine(d, t)


@dataclass(frozen=True)
class ScheduleGroupLink:
    year: int
    age_group: str
    column: str  # "AA" or "AAA"
    label: str  # e.g. "AAA Major", "15O AAA"
    d: int
    url: str


@dataclass(frozen=True)
class ScheduleGameRow:
    year: int
    age_group: str
    group_label: str
    d: int
    game_number: int
    date: str
    time: str
    starts_at: Optional[dt.datetime]
    home: str
    away: str
    home_score: Optional[int]
    away_score: Optional[int]
    rink: Optional[str]
    game_type: Optional[str]


def scrape_index(
    *,
    year: Optional[int] = None,
    allowed_link_labels: Optional[Iterable[str]] = None,
) -> tuple[int, list[ScheduleGroupLink]]:
    """Return (effective_year, groups) from the CAHA schedule index.

    This discovers the "AA / AAA Major / AAA Minor / 15O AAA" schedule links for
    each age group, excluding Girls and High School sections.
    """
    allow = {str(s).strip() for s in (allowed_link_labels or []) if str(s).strip()}
    if not allow:
        allow = {"AA", "AAA Major", "AAA Minor", "15O AAA"}

    params = {"y": str(int(year))} if year is not None and int(year) > 0 else None
    soup = _get_html(SCHEDULE_INDEX_URL, params=params)

    # Find the tier schedule selection table.
    tier_table = None
    for table in soup.find_all("table"):
        th_texts = [_norm(th.get_text(" ", strip=True)) for th in table.find_all("th")]
        if not th_texts:
            continue
        if {"Age/Group", "AA", "AAA"}.issubset(set(th_texts)):
            tier_table = table
            break
    if tier_table is None:
        raise RuntimeError("CAHA schedule index: could not locate tier schedule table")

    effective_year = None
    sel = tier_table.find("select", attrs={"name": "y"})
    if sel is not None:
        opt = sel.find("option", selected=True)
        if opt is not None:
            effective_year = _parse_int(opt.get("value") or opt.get_text(" ", strip=True))
    if effective_year is None:
        effective_year = int(year) if year is not None and int(year) > 0 else None
    if effective_year is None:
        raise RuntimeError("CAHA schedule index: could not determine schedule year")

    # Column indices for Boys schedule groups (exclude Girls/High School).
    # The table headers are: Age/Group, AA, AAA, Girls AA, Girls AAA, Senior.
    boys_cols = {"AA": 1, "AAA": 2}

    groups: list[ScheduleGroupLink] = []
    for tr in tier_table.find_all("tr", class_=re.compile(r"\bnormal_center\b")):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 3:
            continue
        age_group = _norm(tds[0].get_text(" ", strip=True))
        if not age_group:
            continue
        if any(tok in age_group.lower() for tok in ("girls", "high school", "women")):
            continue

        for col_name, idx in boys_cols.items():
            if idx >= len(tds):
                continue
            td = tds[idx]
            for a in td.find_all("a", href=True):
                label = _norm(a.get_text(" ", strip=True))
                if label not in allow:
                    continue
                href = str(a.get("href") or "").strip()
                if not href:
                    continue
                d_val = None
                y_val = None
                try:
                    query = urlsplit(href).query
                    q = dict(parse_qsl(query))
                    d_val = _parse_int(q.get("d"))
                    y_val = _parse_int(q.get("y"))
                except Exception:
                    d_val = None
                    y_val = None
                if d_val is None:
                    continue
                if y_val is None:
                    y_val = int(effective_year)
                groups.append(
                    ScheduleGroupLink(
                        year=int(y_val),
                        age_group=age_group,
                        column=col_name,
                        label=label,
                        d=int(d_val),
                        url=urljoin(SCHEDULE_INDEX_URL, href),
                    )
                )

    return int(effective_year), groups


_SCHEDULE_TABLE_HEADER = ("Date", "Time", "GM", "Home", "Score", "Visitor", "Score", "Rink", "Type")


def scrape_schedule_group(group: ScheduleGroupLink) -> list[ScheduleGameRow]:
    """Scrape a schedule.pl division/group page into game rows."""
    soup = _get_html(group.url)

    header_tr = None
    for tr in soup.find_all("tr"):
        cells = [
            _norm(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"], recursive=False)
        ]
        if tuple(cells) == _SCHEDULE_TABLE_HEADER:
            header_tr = tr
            break
        if len(cells) == len(_SCHEDULE_TABLE_HEADER) and [
            re.sub(r"[^a-z0-9]+", "", c.casefold()) for c in cells
        ] == [re.sub(r"[^a-z0-9]+", "", c.casefold()) for c in _SCHEDULE_TABLE_HEADER]:
            header_tr = tr
            break
    if header_tr is None:
        # Some divisions currently render the filter UI but contain no schedule rows yet (no table).
        has_any_data_rows = any(
            len(tr.find_all("td", recursive=False)) == 9 for tr in soup.find_all("tr")
        )
        if not has_any_data_rows:
            logger.warning(
                "CAHA schedule page: no schedule rows found (0 games?) for %s %s %s (d=%s, y=%s): %s",
                str(group.age_group),
                str(group.column),
                str(group.label),
                int(group.d),
                int(group.year),
                group.url,
            )
            return []

        # Schedule rows exist but the header changed; fail loudly so we don't silently miss games.
        raise RuntimeError(f"CAHA schedule page: could not find schedule table header: {group.url}")

    table = header_tr.find_parent("table")
    if table is None:
        raise RuntimeError(f"CAHA schedule page: missing schedule table: {group.url}")

    rows = table.find_all("tr", recursive=False)
    try:
        start_idx = rows.index(header_tr) + 1
    except Exception:
        start_idx = 0

    out: list[ScheduleGameRow] = []
    for tr in rows[start_idx:]:
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 9:
            continue
        date_s = _norm(tds[0].get_text(" ", strip=True))
        time_s = _norm(tds[1].get_text(" ", strip=True))
        gm_s = _norm(tds[2].get_text(" ", strip=True))
        home = _norm(tds[3].get_text(" ", strip=True))
        home_score_s = _norm(tds[4].get_text(" ", strip=True))
        away = _norm(tds[5].get_text(" ", strip=True))
        away_score_s = _norm(tds[6].get_text(" ", strip=True))
        rink = _norm(tds[7].get_text(" ", strip=True)) or None
        game_type = _norm(tds[8].get_text(" ", strip=True)) or None

        gm = _parse_int(gm_s)
        if gm is None:
            continue
        if not home or not away:
            continue

        out.append(
            ScheduleGameRow(
                year=int(group.year),
                age_group=str(group.age_group),
                group_label=str(group.label),
                d=int(group.d),
                game_number=int(gm),
                date=date_s,
                time=time_s,
                starts_at=_parse_starts_at(date_s, time_s),
                home=home,
                away=away,
                home_score=_parse_int(home_score_s),
                away_score=_parse_int(away_score_s),
                rink=rink,
                game_type=game_type,
            )
        )

    if not out:
        logger.warning("CAHA schedule page parsed 0 games: %s", group.url)
    return out


def scrape_tier_schedule_games(
    *, year: Optional[int] = None, allowed_link_labels: Optional[Iterable[str]] = None
) -> list[ScheduleGameRow]:
    """Scrape all Tier (boys) schedule.pl groups (AA/AAA Major/Minor/15O AAA)."""
    _effective_year, groups = scrape_index(year=year, allowed_link_labels=allowed_link_labels)
    rows: list[ScheduleGameRow] = []
    for g in groups:
        rows.extend(scrape_schedule_group(g))
    return rows

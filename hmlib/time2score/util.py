"""Helper functions for sharks scraper."""

import datetime
import json
import logging
import os
import re
from urllib import parse

import bs4
import requests

HEADERS = {
    "Content-Type": "html",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
    ),
}

CACHE = False


logger = logging.getLogger(__name__)


def get_value_from_link(url: str, key: str):
    query = parse.urlsplit(url).query
    query_map = dict(parse.parse_qsl(query))
    return query_map.get(key)


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def infer_season_year_for_date_str(
    date_str: str,
    *,
    season_start_month: int = 10,
    season_end_month: int = 6,
    reference_dt: datetime.datetime | None = None,
) -> int:
    """Infer the year for a season that spans across calendar years.

    For youth hockey leagues, a "season" often spans roughly Oct-Jun.

    The intent here is that when a page only includes month/day (no year),
    we can infer the correct year relative to the current "active season":
      - Oct-Dec => season start year
      - Jan-Jun => following year
      - Jul-Sep (off-season/preseason) => season start year
    """
    ref = reference_dt or datetime.datetime.now()
    # Determine which season we're "in" relative to the reference date.
    # Example with season_start_month=10 and season_end_month=6:
    #   - In Jan-Jun 2026: active season started Oct 2025 (start_year=2025)
    #   - In Oct-Dec 2026: active season started Oct 2026 (start_year=2026)
    #   - In Jul-Sep 2026: treat as last season (start_year=2025)
    ref_y = int(ref.year)
    ref_m = int(ref.month)
    start_m = int(season_start_month)
    end_m = int(season_end_month)
    if start_m < 1 or start_m > 12 or end_m < 1 or end_m > 12:
        season_start_year = ref_y
    else:
        if ref_m >= start_m:
            season_start_year = ref_y
        elif ref_m <= end_m:
            season_start_year = ref_y - 1
        else:
            # Off-season months between end_m and start_m.
            season_start_year = ref_y - 1

    s = str(date_str or "")
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", s, flags=re.IGNORECASE)
    if not m:
        return int(ref.year)
    month = _MONTHS.get(m.group(1).lower()[:3])
    if not month:
        return int(ref.year)
    if start_m < 1 or start_m > 12 or end_m < 1 or end_m > 12:
        return int(ref.year)
    if month >= start_m:
        return int(season_start_year)
    if month <= end_m:
        return int(season_start_year) + 1
    # Off-season / preseason months: keep in season start year.
    return int(season_start_year)


def parse_game_time(date_str, time_str, year=None):
    """Parse a game date/time into a datetime.

    TimeToScore pages use a few different date formats depending on which page is being scraped:
      - Schedule pages often use: "Sun Jan 25"
      - Scoresheet pages may use: "01-25-26" or "01/25/2026"

    This helper accepts both.
    """

    ds = str(date_str or "").strip()
    ts = str(time_str or "").strip()
    ts = ts.replace("12 Noon", "12:00 PM")
    # Normalize "4:45PM" -> "4:45 PM"
    ts = re.sub(r"(?i)(\\d)(am|pm)$", r"\\1 \\2", ts)
    if not ds or not ts:
        raise ValueError(f"Empty game date/time: date={ds!r} time={ts!r}")

    def _parse_time() -> datetime.time:
        for fmt in ("%I:%M %p", "%I:%M:%S %p", "%H:%M", "%H:%M:%S"):
            try:
                return datetime.datetime.strptime(ts, fmt).time()
            except Exception:
                pass
        try:
            return datetime.time.fromisoformat(ts)
        except Exception as e:
            raise ValueError(f"Unrecognized game time {ts!r}") from e

    # Text dates: "Sun Jan 25"
    if re.search(r"[A-Za-z]", ds):
        y = str(year) if year is not None else str(infer_season_year_for_date_str(ds))
        for fmt in ("%Y %a %b %d %I:%M %p", "%Y %b %d %I:%M %p", "%Y %a %b %d %H:%M"):
            try:
                return datetime.datetime.strptime(f"{y} {ds} {ts}", fmt)
            except Exception:
                pass
        raise ValueError(f"Unrecognized game date/time: date={ds!r} time={ts!r} year={y!r}")

    # Numeric dates: "01-25-26" / "01/25/2026" / "2026-01-25"
    ds2 = ds.replace("/", "-")
    d: datetime.date | None = None
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m-%d-%y"):
        try:
            d = datetime.datetime.strptime(ds2, fmt).date()
            break
        except Exception:
            d = None
    if d is None:
        m = re.match(r"^(\\d{1,2})-(\\d{1,2})$", ds2)
        if m:
            mm = int(m.group(1))
            dd = int(m.group(2))
            yy = int(year) if year is not None else int(datetime.datetime.now().year)
            d = datetime.date(yy, mm, dd)
    if d is None:
        raise ValueError(f"Unrecognized game date {ds!r}")
    return datetime.datetime.combine(d, _parse_time())


def get_html(url: str, params: dict[str, str] | None = None, log=False):
    """Read HTML from a given URL."""
    if log:
        logger.info("Reading HTML from %s (%s)...", url, params)
    timeout_s = float(os.environ.get("HM_T2S_HTTP_TIMEOUT", "30"))
    html = requests.get(url, params=params, headers=HEADERS, timeout=timeout_s)
    return bs4.BeautifulSoup(html.text, "html5lib")


def cache_json(
    file_format,
    max_age=datetime.timedelta(days=1),
    reload_kwarg="reload",  # pylint: disable=g-bare-generic
):
    """A function that creates a decorator which will use "cache_json" for caching the results of the decorated function "fn"."""

    def decorator(fn):  # define a decorator for a function "fn"
        # define a wrapper that will finally call "fn" with all arguments
        def wrapped(*args, **kwargs):
            if not CACHE:
                return fn(*args, **kwargs)

            # Format filepath and create intermediate directories
            path = os.path.join("/tmp/__cache__", file_format.format(*args, **kwargs) + ".json")
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            try_load = not kwargs.get(reload_kwarg, False)
            if try_load and os.path.exists(path):
                modify_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                age = datetime.datetime.now() - modify_time
                if max_age is not None and age < max_age:
                    with open(path, "r") as cachehandle:
                        logger.info("Using cached result from '%s'", path)
                        return json.load(cachehandle)
                else:
                    logger.info("Cache is stale. Reloading...")

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)
            # write to cache file
            with open(path, "w") as cachehandle:
                logger.info("Saving result to cache '%s'", path)
                json.dump(res, cachehandle)
            return res

        return wrapped

    return decorator

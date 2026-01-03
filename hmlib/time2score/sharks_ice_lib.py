"""Scraper for SIAHL."""

import re
import logging
from typing import Any

import hashlib
import hmac
import time
from urllib.parse import quote

try:  # pragma: no cover
    from . import database  # type: ignore
    from . import util  # type: ignore
except Exception:  # noqa: BLE001
    # Fallback for direct execution without package context.
    import os as _os
    import sys as _sys

    _here = _os.path.dirname(__file__)
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import database  # type: ignore  # noqa: E402
    import util  # type: ignore  # noqa: E402

TIMETOSCORE_URL = "https://stats.sharksice.timetoscore.com/"
TEAM_URL = TIMETOSCORE_URL + "display-schedule"
GAME_URL = TIMETOSCORE_URL + "oss-scoresheet"
DIVISION_URL = TIMETOSCORE_URL + "display-league-stats"
MAIN_STATS_URL = TIMETOSCORE_URL + "display-stats.php"
CALENDAR = "webcal://stats.sharksice.timetoscore.com/team-cal.php?team={team}&tlev=0&tseq=0&season={season}&format=iCal"

logger = logging.getLogger(__name__)

_API_CONFIG: dict[str, str] | None = None


td_selectors = dict(
    # Game stats
    date=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(1) > td:nth-child(1)"
    ),
    time=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(1) > td:nth-child(2)"
    ),
    league=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(2) > td:nth-child(1)"
    ),
    level=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(3) > td:nth-child(1)"
    ),
    location=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(4) > td:nth-child(1)"
    ),
    scorekeeper=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(1) > td:nth-child(2)"
    ),
    periodLength=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(4) > td:nth-child(2)"
    ),
    referee1=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(2) > td:nth-child(2)"
    ),
    referee2=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(3) > td:nth-child(2)"
    ),
    away=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(2) > td:nth-child(2)"
    ),
    home=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(3) > td:nth-child(2)"
    ),
    # Selectors we'll use to verify that parsing other parts were correct.
    awayGoals=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(2) > td:nth-child(7)"
    ),
    homeGoals=(
        "body > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(3) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(3) > td:nth-child(7)"
    ),
)

tr_selectors = dict(
    awayPlayers=(
        "body > table:nth-child(3) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(1) > table:nth-child(2) > tbody:nth-child(1) >"
        " tr:nth-child(2) > td:nth-child(1) > table:nth-child(1) >"
        " tbody:nth-child(1) > tr:nth-child(n+2)"
    ),
    homePlayers=(
        "body > table:nth-child(3) > tbody:nth-child(1) > tr:nth-child(1) >"
        " td:nth-child(2) > table:nth-child(1) > tbody:nth-child(1) >"
        " tr:nth-child(2) > td:nth-child(1) > table:nth-child(1) >"
        " tbody:nth-child(1) > tr:nth-child(n+2)"
    ),
    awayScoring=(
        "body > div > div.d50l > div.d25l > table:nth-child(1) >"
        " tbody:nth-child(1) > tr:nth-child(n+4)"
    ),
    homeScoring=(
        "body > div > div.d50r > div.d25l > table:nth-child(1) >"
        " tbody:nth-child(1) > tr:nth-child(n+4)"
    ),
    awayPenalties=(
        "body > div > div.d50l > div.d25r > table:nth-child(1) >" " tbody:nth-child(1) > tr"
    ),
    homePenalties=(
        "body > div > div.d50r > div.d25r > table:nth-child(1) >" " tbody:nth-child(1) > tr"
    ),
    awayShootout=(
        "body > div > div.d50l > div.d25l > table:nth-child(2) >"
        " tbody:nth-child(1) > tr:nth-child(n+4)"
    ),
    homeShootout=(
        "body > div > div.d50r > div.d25l > table:nth-child(2) >"
        " tbody:nth-child(1) > tr:nth-child(n+4)"
    ),
)

columns = dict(
    players=["number", "position", "name"],
    penalties=[
        "period",
        "number",
        "infraction",
        "minutes",
        "offIce",
        "start",
        "end",
        "onIce",
    ],
    scoring=["period", "time", "extra", "goal", "assist1", "assist2"],
    shootout=["number", "player", "result"],
)

team_columns_rename = {
    "GP": "gamesPlayed",
    "W": "wins",
    "T": "ties",
    "L": "losses",
    "OTL": "overtimeLosses",
    "PTS": "points",
    "Streak": "streak",
    "Tie Breaker": "tieBreaker",
}

player_columns_rename = {
    "Team": "team",
    "Name": "name",
    "#": "number",
    "GP": "gamesPlayed",
    "Ass.": "assists",
    "Goals": "goals",
    "Pts": "points",
    "Pts/Game": "pointsPerGame",
    "Hat": "hatTricks",
    "Min": "penaltyMinutes",
}

goalie_columns_rename = {
    "Team": "team",
    "Name": "name",
    "GP": "gamesPlayed",
    "Shots": "shots",
    "GA": "goalsAgainst",
    "GAA": "goalsAgainstAverage",
    "Save %": "savePercentage",
    "SO": "shutouts",
}

game_columns_rename = {
    "Game": ("id", lambda g: str(g).replace("*", "").replace("^", "")),
    "Date": "date",
    "Time": "time",
    "Rink": "rink",
    "League": "league",
    "Level": "level",
    "Away": "away",
    "Home": "home",
    "Type": "type",
    "Goals.1": "homeGoals",
    "Goals": "awayGoals",
    "Scoresheet": None,
    "Box Score": None,
    "Game Center": None,
}


class Error(Exception):
    pass


class MissingStatsError(Error):
    pass


def rename(initial: dict[str, str], mapping: dict[str, str]):
    """Renames columns in a dict."""
    new_map = {}
    for key, val in initial.items():
        mapped_key = mapping.get(key, key)
        if mapped_key is None:
            continue
        if isinstance(mapped_key, tuple):
            mapped_key, func = mapped_key
            val = func(val)
        new_map[mapped_key] = val
    return new_map


def parse_td_row(row):
    val = []
    for td in row("td"):
        if td("a"):
            val.append({"text": td.a.text.strip(), "link": td.a["href"]})
        else:
            val.append(td.text.strip())
    return val


def fix_players_rows(rows):
    val = []
    for row in rows:
        val.append(row[:3])
        if len(row) == 6:
            val.append(row[3:])
    return val


@util.cache_json("seasons")
def scrape_seasons():
    """Scrape season data from HTML."""
    soup = util.get_html(MAIN_STATS_URL, params={"league": "1"})
    season_ids: dict[str, int] = {}
    sel = soup.find("select")
    if sel is not None:
        season_ids = {
            o.text.strip(): int(o["value"])
            for o in sel("option")
            if (o.get("value") or "").isdigit() and int(o["value"]) > 0
        }
    else:
        # Newer versions of the site may omit the season dropdown; fall back to links.
        seasons = set()
        for a in soup.find_all("a", href=True):
            m = re.search(r"season=(\d+)", a.get("href") or "")
            if m:
                seasons.add(int(m.group(1)))
        if seasons:
            # Only add a best-effort list; "Current" will be selected by sync logic.
            for sid in sorted(seasons, reverse=True):
                season_ids[f"Season {sid}"] = int(sid)
    current = 0
    for link in soup.find_all("a", href=True):
        current = re.search(r"season=(\d+)", link["href"])
        if current:
            current = int(current.group(1))
            break
    if current > 0:
        season_ids["Current"] = current
    return season_ids


def get_team_id(db: database.Database, team_name: str, season: int) -> str:
    """Get team id from string."""
    if not team_name:
        return None
    teams = db.list_teams('season_id = %s AND name = "%s"' % (season, team_name))
    # This shouldn't happen
    if len(teams) > 1:
        raise KeyError("Duplicate team names: %s" % teams)
    if not teams:
        raise ValueError("No team named %s in season %s" % (team_name, season))
    return teams[0]["team_id"]


def NO_LINK_INT(a):
    return int(a[0]) if a[0] else 0


def NO_LINK(a):
    return a[0] if a[0] else ""


DIVISION_GAME_CONVERTERS = {
    "G": NO_LINK_INT,
    "GP": NO_LINK_INT,
    "W": NO_LINK_INT,
    "L": NO_LINK_INT,
    "T": NO_LINK_INT,
    "OTL": NO_LINK_INT,
    "PTS": NO_LINK_INT,
    "Streak": NO_LINK,
    "Tie Breaker": NO_LINK,
}


def _parse_division_teams(table_str: str):
    """Parse team data from a table."""
    # Avoid pandas.read_html (requires optional lxml); parse with BeautifulSoup instead.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(table_str, "html5lib")
    table = soup.find("table")
    if not table:
        return []

    def _cell_value(td):
        a = td.find("a")
        if a and a.get("href"):
            return {"text": a.get_text(strip=True), "link": a.get("href")}
        return td.get_text(strip=True)

    # Header row (skip any 1-col title rows)
    header_tr = None
    for tr in table.find_all("tr"):
        ths = tr.find_all("th")
        if len(ths) >= 2:
            header_tr = tr
            break
    headers = [th.get_text(strip=True) for th in (header_tr.find_all("th") if header_tr else [])]
    teams: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds or not headers:
            continue
        vals = [_cell_value(td) for td in tds]
        if len(vals) != len(headers):
            continue
        row: dict[str, Any] = dict(zip(headers, vals))
        team_cell = row.get("Team")
        if isinstance(team_cell, dict):
            row["name"] = team_cell.get("text", "")
            link = team_cell.get("link") or ""
            tid = util.get_value_from_link(link, "team")
            row["id"] = int(tid) if (tid or "").isdigit() else None
        else:
            row["name"] = str(team_cell or "")
            row["id"] = None
        row.pop("Team", None)

        # Apply converters for known numeric columns
        for col, conv in DIVISION_GAME_CONVERTERS.items():
            if col in row:
                v = row[col]
                if isinstance(v, dict):
                    v = v.get("text") or ""
                try:
                    row[col] = conv((str(v), "")) if conv in (NO_LINK, NO_LINK_INT) else conv(v)  # type: ignore[arg-type]
                except Exception:
                    row[col] = v

        teams.append(rename(row, team_columns_rename))
    return teams


def scrape_season_divisions(season_id: int):
    """Scrape divisions and teams in a season."""
    soup = util.get_html(MAIN_STATS_URL, params=dict(league=1, season=season_id))
    divisions = []
    division_name = ""
    division_id = 0
    conference_id = 0
    table_rows = []
    for row in soup.table.find_all("tr"):
        # Ignore non-header rows
        if row("th"):
            header = row.th.text.strip()
            if header.startswith("Adult Division") or header.startswith("Senior"):
                division_name = header
                href = row.next_sibling.a["href"].strip()
                division_id = int(util.get_value_from_link(href, "level"))
                conference_id = int(util.get_value_from_link(href, "conf"))
                continue
            if len(table_rows) > 1:
                teams = _parse_division_teams("<table>" + "\n".join(table_rows) + "</table>")
                divisions.append(
                    {
                        "name": division_name,
                        "id": division_id,
                        "conferenceId": conference_id,
                        "seasonId": season_id,
                        "teams": teams,
                    }
                )
            table_rows = [str(row)]
        else:
            table_rows.append(str(row))

    # Add the last division too.
    if len(table_rows) > 1:
        teams = _parse_division_teams("<table>" + "\n".join(table_rows) + "</table>")
        divisions.append(
            {
                "name": division_name,
                "id": division_id,
                "conferenceId": conference_id,
                "seasonId": season_id,
                "teams": teams,
            }
        )
    return divisions


@util.cache_json("seasons/{season_id}/teams/{team_id}")
def get_team(season_id: int, team_id: int, reload=False):
    """Get team info from a season and id."""
    info = {}
    soup = util.get_html(TEAM_URL, params=dict(season=season_id, team=team_id))
    if not soup.table:
        return {}

    games = []
    # Avoid pandas.read_html (requires optional lxml); parse with BeautifulSoup instead.
    def _uniquify(headers: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        out = []
        for h in headers:
            if h not in seen:
                seen[h] = 0
                out.append(h)
            else:
                seen[h] += 1
                out.append(f"{h}.{seen[h]}")
        return out

    header_tr = None
    for tr in soup.table.find_all("tr"):
        ths = tr.find_all("th")
        if len(ths) >= 5 and any(th.get_text(strip=True) == "Game" for th in ths):
            header_tr = tr
            break
    if header_tr is None:
        return {}
    headers_raw = [th.get_text(strip=True) for th in header_tr.find_all("th")]
    headers = _uniquify(headers_raw)
    in_body = False
    for tr in soup.table.find_all("tr"):
        if tr is header_tr:
            in_body = True
            continue
        tds = tr.find_all("td")
        if not in_body or not tds or not headers:
            continue
        if len(tds) != len(headers):
            continue

        cells = []
        for td in tds:
            a = td.find("a")
            if a and a.get("href"):
                cells.append({"text": td.get_text(strip=True), "link": a.get("href")})
            else:
                cells.append(td.get_text(strip=True))

        row_dict: dict[str, Any] = dict(zip(headers, cells))
        # Prefer extracting game_id from any link in the row.
        game_id = None
        for v in cells:
            if isinstance(v, dict):
                gid = util.get_value_from_link(v.get("link") or "", "game_id")
                if gid and str(gid).isdigit():
                    game_id = str(gid)
                    break
        if game_id is not None:
            row_dict["Game"] = game_id
        # Flatten linked cells to their display text for rename mapping.
        for k, v in list(row_dict.items()):
            if isinstance(v, dict):
                row_dict[k] = v.get("text", "")

        row = rename(row_dict, game_columns_rename)
        if row["type"] == "Practice":
            continue
        date = row.pop("date", None)
        time = row.pop("time", None)
        year = None  # Estimate year
        if not date or not time:
            row["start_time"] = None
        else:
            row["start_time"] = util.parse_game_time(date, time, year)
        # Goals can be str, int, or float for some reason.
        # Correct all to string to allow for shootouts (e.g. "4 S")
        if isinstance(row.get("homeGoals"), float):
            row["homeGoals"] = str(int(row["homeGoals"]))
        elif row.get("homeGoals") in ("", None):
            row.pop("homeGoals", None)
        if isinstance(row.get("awayGoals"), float):
            row["awayGoals"] = str(int(row["awayGoals"]))
        elif row.get("awayGoals") in ("", None):
            row.pop("awayGoals", None)
        games.append(row)
    info["games"] = games
    return info


def scrape_game_stats(game_id: int, season_id: int | None = None):
    """Get game stats from an id.

    SharksIce has migrated the game center / scoresheet views to a Flutter app.
    The old HTML tables are no longer present; fetch data via the public API
    used by the widget instead.
    """
    return scrape_game_stats_api(game_id, season_id=season_id)


def _encode_component(val: str) -> str:
    # Dart/JS encodeURIComponent leaves these unescaped.
    return quote(val, safe="-_.!~*'()")


def _api_config_for_game(game_id: int) -> dict[str, str]:
    import requests

    global _API_CONFIG  # noqa: PLW0603
    if _API_CONFIG is not None:
        return _API_CONFIG

    url = "https://react.sharksice.timetoscore.com/game-center.php"
    resp = requests.get(url, params={"game_id": int(game_id)}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    txt = resp.text
    # Extract config values from the inline JS.
    def _find(key: str) -> str:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', txt)
        if not m:
            raise RuntimeError(f"Failed to locate {key} in SharksIce game-center config")
        return m.group(1)

    _API_CONFIG = {
        "username": _find("username"),
        "secret": _find("secret"),
        "api_url": _find("api_url"),
        "league_id": _find("league_id"),
    }
    return _API_CONFIG


def _api_get(endpoint: str, *, game_id: int, season_id: int | None, extra_params: dict[str, Any] | None = None):
    import requests

    cfg = _api_config_for_game(game_id)
    username = cfg["username"]
    secret = cfg["secret"]
    api_url = cfg["api_url"]
    league_id = cfg["league_id"]

    params: list[tuple[str, str]] = []
    params.append(("auth_key", username))
    params.append(("auth_timestamp", str(int(time.time()))))
    params.append(("body_md5", hashlib.md5(b"").hexdigest()))
    params.append(("game_id", str(int(game_id))))
    params.append(("league_id", str(league_id)))
    if season_id is not None:
        params.append(("season_id", str(int(season_id))))
    if extra_params:
        for k, v in extra_params.items():
            if v is None:
                continue
            params.append((str(k), str(v)))

    qs = "&".join(f"{k}={_encode_component(v)}" for k, v in params if k)
    string_to_sign = f"GET\n/{endpoint}\n{qs}"
    signature = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    params.append(("auth_signature", signature))

    url = f"https://{api_url}/{endpoint}"
    resp = requests.get(url, params=dict(params), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    if resp.status_code != 200:
        raise MissingStatsError(f"API error {resp.status_code} for {endpoint} game_id={game_id}")
    return resp.json()


def scrape_game_stats_api(game_id: int, season_id: int | None):
    """Return a scrape_game_stats-compatible dict from the SharksIce game center API."""
    # Default season to current (best-effort).
    if season_id is None:
        try:
            seasons = scrape_seasons()
            season_id = int(seasons.get("Current") or 0) or None
        except Exception:
            season_id = None

    j = _api_get(
        "get_game_center",
        game_id=game_id,
        season_id=season_id,
        extra_params={"widget": "gamecenter"},
    )
    gc = (j or {}).get("game_center") or {}
    info = gc.get("game_info") or {}
    live = gc.get("live") or {}

    def _strip(v: Any) -> str:
        return str(v or "").strip()

    home_name = _strip(info.get("home_name"))
    away_name = _strip(info.get("away_name"))
    data: dict[str, Any] = {
        "date": _strip(info.get("formatted_date")),
        "time": _strip(info.get("time")),
        "league": _strip(info.get("alias")) or _strip(info.get("away_ab")),
        "level": "",
        "location": _strip(info.get("location")),
        "home": home_name,
        "away": away_name,
    }

    goal_summary = live.get("goal_summary") or {}
    try:
        data["homeGoals"] = int((goal_summary.get("home_goals") or {}).get("total"))
    except Exception:
        data["homeGoals"] = None
    try:
        data["awayGoals"] = int((goal_summary.get("away_goals") or {}).get("total"))
    except Exception:
        data["awayGoals"] = None

    def _player_rows(items: list[dict[str, Any]]):
        out = []
        for it in items or []:
            nm = _strip(it.get("name"))
            if not nm:
                continue
            out.append(
                {
                    "number": _strip(it.get("jersey")),
                    "position": _strip(it.get("position")),
                    "name": nm,
                    "goals": it.get("goals"),
                    "assists": it.get("assists"),
                }
            )
        return out

    home_skaters = _player_rows(live.get("home_skaters") or [])
    away_skaters = _player_rows(live.get("away_skaters") or [])
    home_goalies = _player_rows(live.get("home_goalies") or [])
    away_goalies = _player_rows(live.get("away_goalies") or [])

    data["homePlayers"] = [{"number": p["number"], "position": p["position"], "name": p["name"]} for p in home_skaters + home_goalies]
    data["awayPlayers"] = [{"number": p["number"], "position": p["position"], "name": p["name"]} for p in away_skaters + away_goalies]
    # Provide per-player totals for import scripts (not part of the legacy HTML scraper).
    data["homeSkaters"] = home_skaters + home_goalies
    data["awaySkaters"] = away_skaters + away_goalies
    return data


def sync_seasons(db: database.Database):
    seasons = scrape_seasons()
    for name, season_id in seasons.items():
        db.add_season(season_id, name)
    if "Current" not in seasons:
        db.add_season(0, "Current")


def sync_divisions(db: database.Database, season: int):
    """Sync divisions from site."""
    divs = scrape_season_divisions(season_id=season)
    logger.info("Found %d divisions in season %s...", len(divs), season)
    for div in divs:
        db.add_division(
            division_id=div["id"],
            conference_id=div["conferenceId"],
            name=div["name"],
        )
        logger.info("%s teams in %s", len(div["teams"]), div["name"])
        for team in div["teams"]:
            team_id = team.pop("id")
            team_name = team.pop("name")
            db.add_team(
                season_id=season,
                division_id=div["id"],
                conference_id=div["conferenceId"],
                team_id=team_id,
                name=team_name,
                stats=team,
            )


def get_team_or_unknown(db: database.Database, team_name: str, season: int):
    """Get team id from string."""
    try:
        team_id = get_team_id(db, team_name, season)
    except ValueError as e:
        logger.warning("Unknown team '%s' in season %s: %s", team_name, season, e)
        db.add_season(season_id=-1, name="UNKNOWN")
        db.add_division(division_id=-1, conference_id=-1, name="UNKNOWN")
        team_id = -1
        db.add_team(
            season_id=-1,
            division_id=-1,
            conference_id=-1,
            team_id=team_id,
            name="UNKNOWN",
            stats={},
        )
    return team_id


def add_game(
    db: database.Database,
    season: int,
    team: dict[str, Any],
    game: dict[str, Any],
):
    """Add a game to the database."""
    # Clean up dict and translate data.
    game_id = game.pop("id")
    start_time = game.pop("start_time", None)
    rink = game.pop("rink", None)
    # Get Team IDs from names and season.
    home, away = game["home"], game["away"]
    if home == team["name"]:
        home_id = team["team_id"]
        away_id = get_team_or_unknown(db, away, season)
    else:
        home_id = get_team_or_unknown(db, home, season)
        away_id = team["team_id"]
    db.add_game(
        season_id=season,
        division_id=team["division_id"],
        conference_id=team["conference_id"],
        game_id=game_id,
        home_id=home_id,
        away_id=away_id,
        rink=rink,
        start_time=start_time,
        info=game,
    )


def sync_season_teams(db: database.Database, season: int):
    """Sync games from site."""
    teams = db.list_teams("season_id = %d" % season)
    game_ids = set()
    for team in teams:
        logger.info("Syncing %s season %d...", team["name"], season)
        team_info = get_team(season_id=season, team_id=team["team_id"])
        games = team_info.pop("games", [])
        for game in games:
            if game["id"] in game_ids:
                continue
            game_ids.add(game["id"])
            add_game(db, season, team, game)


def sync_game_stats(db: database.Database):
    games = db.list_games()
    for game in games:
        if not game["stats"]:
            try:
                stats = scrape_game_stats(game["game_id"])
            except Exception as e:
                logger.exception("Failed to scrape game stats: %s", e)
                continue
            db.add_game_stats(game["game_id"], stats)


def load_data(db: database.Database):
    db.create_tables()
    sync_seasons(db)
    # print(db.list_seasons())
    seasons = [a["season_id"] for a in db.list_seasons()]
    for season in sorted(seasons, reverse=True):
        if season >= 32:
            continue
        sync_divisions(db, season)
        sync_season_teams(db, season)


if __name__ == "__main__":
    load_data(database.Database())
    # d = scrape_season_divisions(66)

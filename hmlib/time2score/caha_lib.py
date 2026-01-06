"""Scraper for CAHA youth (TimeToScore).

This mirrors the Sharks Ice scraper but targets the CAHA youth site at
https://stats.caha.timetoscore.com and adapts for minor structural differences
such as no season drop-down on the main stats page.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from . import util
from .database import Database

# Base site configuration
TIMETOSCORE_URL = "https://stats.caha.timetoscore.com/"
TEAM_URL = TIMETOSCORE_URL + "display-schedule"
GAME_URL = TIMETOSCORE_URL + "oss-scoresheet"
DIVISION_URL = TIMETOSCORE_URL + "display-league-stats"
MAIN_STATS_URL = TIMETOSCORE_URL + "display-stats"

# CAHA youth league id
CAHA_LEAGUE = 3
STAT_CLASS = 1  # youth


logger = logging.getLogger(__name__)


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


def rename(initial: dict[str, Any], mapping: dict[str, Any]):
    """Renames columns in a dict."""
    new_map: dict[str, Any] = {}
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


def fix_players_rows(rows: list[list[Any]]):
    val = []
    for row in rows:
        val.append(row[:3])
        if len(row) == 6:
            val.append(row[3:])
    return val


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


def _parse_division_teams_table(table_html: str) -> list[dict[str, Any]]:
    """Parse team data from a given table HTML string.

    Returns a list of team dicts with normalized columns.
    """
    table = pd.read_html(
        io.StringIO(table_html), extract_links="body", converters=DIVISION_GAME_CONVERTERS
    )[0].fillna("")
    if "Team" not in table.columns:
        return []
    team = table["Team"].apply(pd.Series)
    table["id"] = team[1].str.extract(r"team=(\d+)")
    table["name"] = team[0]
    del table["Team"]
    teams: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        row_d = rename(row.to_dict(), team_columns_rename)
        teams.append(row_d)
    return teams


def _first_table_with_team_column(soup) -> str | None:
    """Return HTML of the first table that has a Team column, else None."""
    for tbl in soup.find_all("table"):
        try:
            df = pd.read_html(io.StringIO(str(tbl)), extract_links="body")[0]
        except Exception:  # Parsing errors: skip
            continue
        if isinstance(df, pd.DataFrame) and any(col == "Team" for col in df.columns):
            return str(tbl)
    return None


@util.cache_json("seasons_caha")
def scrape_seasons():
    """Scrape season ids for CAHA youth.

    The CAHA site does not expose a season <select> on the main page.
    We discover season ids by scanning links for a `season=` query param
    and mark the max as Current. If none found, return only Current=0.
    """
    soup = util.get_html(
        MAIN_STATS_URL, params={"league": str(CAHA_LEAGUE), "stat_class": str(STAT_CLASS)}
    )
    season_ids: dict[str, int] = {}
    seasons: set[int] = set()
    for a in soup.find_all("a", href=True):
        season = util.get_value_from_link(a["href"], "season")
        if season and season.isdigit() and int(season) > 0:
            seasons.add(int(season))

    if seasons:
        for sid in sorted(seasons):
            season_ids[f"Season {sid}"] = sid
        season_ids["Current"] = max(seasons)
    else:
        season_ids["Current"] = 0
    return season_ids


def _unique_division_links(soup) -> list[tuple[int, int, str]]:
    """Find unique (level, conf, text) tuples from display-league-stats links."""
    seen: set[tuple[int, int]] = set()
    results: list[tuple[int, int, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "display-league-stats" not in href:
            continue
        level = util.get_value_from_link(href, "level")
        conf = util.get_value_from_link(href, "conf")
        if not level or not conf or not level.isdigit() or not conf.isdigit():
            continue
        key = (int(level), int(conf))
        if key in seen:
            continue
        seen.add(key)
        results.append((key[0], key[1], a.text.strip() or f"Level {key[0]} Conf {key[1]}"))
    return results


def scrape_season_divisions(season_id: int):
    """Scrape divisions and teams for a season on CAHA youth.

    CAHA's display-league-stats pages often omit standings tables. Instead,
    the main display-stats page groups team schedule links under a header row
    linking to "Division Player Stats" for each (level, conf). We parse teams
    directly from that page, bounded within the same table section.
    """
    params = {"league": str(CAHA_LEAGUE), "stat_class": str(STAT_CLASS)}
    if season_id and season_id > 0:
        params["season"] = str(season_id)
    soup = util.get_html(MAIN_STATS_URL, params=params)

    divisions: list[dict[str, Any]] = []

    def collect_for_header(a_tag) -> dict[str, Any] | None:
        level = util.get_value_from_link(a_tag.get("href", ""), "level")
        conf = util.get_value_from_link(a_tag.get("href", ""), "conf")
        # Find containing table and starting row (th row)
        th = a_tag.parent
        row = th
        while row is not None and getattr(row, "name", None) != "tr":
            row = row.parent
        if row is None:
            return None
        table = row
        while table is not None and getattr(table, "name", None) != "table":
            table = table.parent
        if table is None:
            return None
        # The "Division Player Stats" row is preceded by a "X Schedule" row which carries the actual division name.
        # On the live site there are sometimes spacer rows in between, so we walk backwards to find the closest
        # sibling row with a non-empty "… Schedule" label.
        div_name = a_tag.get_text(strip=True) or f"Level {level} Conf {conf}"
        try:
            prev = row.find_previous_sibling("tr")
            while prev is not None:
                txt = prev.get_text(" ", strip=True).replace("\xa0", " ").strip()
                if txt and txt.lower().endswith("schedule"):
                    # e.g. "10U B West Schedule" -> "10 B West"
                    txt = txt[:-8].strip()
                    txt = re.sub(r"^(\d+)U\b", r"\1", txt).strip()
                    div_name = txt or div_name
                    break
                if txt and div_name.lower() == "division player stats":
                    # If the schedule label isn't present but we found any non-empty header text,
                    # use it as a better fallback than "Division Player Stats".
                    div_name = txt
                prev = prev.find_previous_sibling("tr")
        except Exception:
            pass
        # Walk subsequent rows in this table until the next header row (with th)
        teams: list[dict[str, Any]] = []
        for sib in row.find_next_siblings():
            if getattr(sib, "name", None) == "tr" and sib.find("th") is not None:
                # If this is the next division header (has link to display-league-stats), stop.
                next_div_link = sib.find("a", href=True)
                if next_div_link and "display-league-stats" in next_div_link.get("href", ""):
                    break
                # Otherwise it's just the column header row; skip and continue.
                continue
            for lnk in sib.find_all("a", href=True):
                href = lnk.get("href", "")
                if "display-schedule" in href and "team=" in href:
                    tid = util.get_value_from_link(href, "team")
                    name = lnk.get_text(strip=True)
                    if tid:
                        teams.append(
                            {
                                "id": int(tid) if tid.isdigit() else tid,
                                "name": name,
                            }
                        )
        if not teams:
            return None
        return {
            "name": div_name,
            "id": int(level) if level and level.isdigit() else 0,
            "conferenceId": int(conf) if conf and conf.isdigit() else 0,
            "seasonId": season_id,
            "teams": teams,
        }

    # Find all headers for divisions and collect teams within each block
    for a in soup.find_all(
        "a", href=True, string=lambda s: isinstance(s, str) and "Division Player Stats" in s
    ):
        div = collect_for_header(a)
        if div:
            divisions.append(div)

    # If nothing found, try fallback of parsing adjacent table containing a Team column
    if not divisions:
        table_html = _first_table_with_team_column(soup)
        if table_html:
            teams = _parse_division_teams_table(table_html)
            if teams:
                divisions.append(
                    {
                        "name": "Division",
                        "id": 0,
                        "conferenceId": 0,
                        "seasonId": season_id,
                        "teams": teams,
                    }
                )
    return divisions


@util.cache_json("seasons/{season_id}/teams/{team_id}")
def get_team(season_id: int, team_id: int, reload: bool = False):
    """Get team info (schedule and results) for CAHA youth.

    CAHA may not require a season param for current schedule; include when >0.
    """
    params = {"team": int(team_id), "league": CAHA_LEAGUE, "stat_class": STAT_CLASS}
    if season_id and season_id > 0:
        params["season"] = int(season_id)
    info: dict[str, Any] = {}
    soup = util.get_html(TEAM_URL, params=params)
    if not soup.table:
        return {}

    games: list[dict[str, Any]] = []
    results = pd.read_html(io.StringIO(str(soup.table)), header=1, flavor="bs4")[0]
    results = results.fillna(np.nan).replace([np.nan], [None])
    for _, row in results.iterrows():
        row = rename(row.to_dict(), game_columns_rename)
        if row.get("type") == "Practice":
            continue
        date = row.pop("date", None)
        time = row.pop("time", None)
        year = None  # Estimate year
        if not date or not time:
            row["start_time"] = None
        else:
            row["start_time"] = util.parse_game_time(date, time, year)
        # Goals can be str, int, or float; normalize for shootouts
        if isinstance(row.get("homeGoals"), float):
            row["homeGoals"] = str(int(row["homeGoals"]))
        elif row.get("homeGoals") is None:
            row.pop("homeGoals", None)
        if isinstance(row.get("awayGoals"), float):
            row["awayGoals"] = str(int(row["awayGoals"]))
        elif row.get("awayGoals") is None:
            row.pop("awayGoals", None)
        games.append(row)
    info["games"] = games
    return info


def scrape_game_stats(game_id: int):
    """Get game stats from an id (box score/players/goals/penalties)."""
    soup = util.get_html(GAME_URL, params=dict(game_id=game_id))
    if not soup.select_one(td_selectors["periodLength"]):
        raise MissingStatsError(f"No game stats for {game_id}")
    data: dict[str, Any] = {}
    for name, selector in td_selectors.items():
        ele = soup.select_one(selector)
        if not ele and name == "scorekeeper":
            raise MissingStatsError("Failed to read data for game. Has it happened yet?")
        val = ele.text.strip()
        if ":" in val:
            val = val.split(":", 1)[1]
        data[name] = val

    for name, selector in tr_selectors.items():
        prefix = "home" if name.startswith("home") else "away"
        suffix = name[len(prefix) :].lower()
        eles = soup.select(selector)
        rows = [parse_td_row(row) for row in eles if row("td")]
        # Hack for players tables.
        if name.endswith("Players"):
            rows = fix_players_rows(rows)
        val = [dict(zip(columns[suffix], row)) for row in rows]
        data[name] = val
    return data


def scrape_league_schedule(season_id: int) -> list[dict[str, Any]]:
    """Scrape the CAHA league schedule page for a full season.

    This page contains all games (including results/scores when available) and links to the game id.
    Example: https://stats.caha.timetoscore.com/display-schedule.php?stat_class=1&league=3&season=31
    """
    params = {"league": str(CAHA_LEAGUE), "stat_class": str(STAT_CLASS)}
    if season_id and season_id > 0:
        params["season"] = str(season_id)
    soup = util.get_html(TIMETOSCORE_URL + "display-schedule.php", params=params)
    tables = soup.find_all("table") if soup else []
    if not tables:
        return []

    def _flatten_col(col: Any) -> str:
        if isinstance(col, tuple):
            parts = [str(p) for p in col if str(p) and str(p).lower() != "nan"]
            return " ".join(parts).strip()
        return str(col or "").strip()

    def _rows_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        df = df.fillna(np.nan).replace([np.nan], [None])
        cols = [_flatten_col(c) for c in df.columns]
        if len(cols) < 8:
            return []
        cols_lower = [c.lower() for c in cols]
        if not cols_lower:
            return []
        if not (cols_lower[0].startswith("game") or any("game results" in c for c in cols_lower)):
            if not any("game" in c for c in cols_lower):
                return []

        rows_out: list[dict[str, Any]] = []
        keys = [
            "id",
            "date",
            "time",
            "rink",
            "league",
            "level",
            "away",
            "awayGoals",
            "home",
            "homeGoals",
            "type",
        ]
        for _, r in df.iterrows():
            row = list(r.values.tolist())
            first = str(row[0] or "").strip() if row else ""
            if first.lower() == "game":
                continue
            out: dict[str, Any] = {}
            for idx, key in enumerate(keys):
                out[key] = row[idx] if idx < len(row) else None
            rows_out.append(out)
        return rows_out

    rows: list[dict[str, Any]] = []
    for tbl in tables:
        try:
            dfs = pd.read_html(io.StringIO(str(tbl)), header=0, flavor="bs4")
        except Exception:
            continue
        for df in dfs:
            rows.extend(_rows_from_df(df))

    return rows


def sync_seasons(db: Database):
    seasons = scrape_seasons()
    for name, season_id in seasons.items():
        db.add_season(season_id, name)
    if "Current" not in seasons:
        db.add_season(0, "Current")


def sync_divisions(db: Database, season: int):
    """Sync divisions from CAHA site."""
    divs = scrape_season_divisions(season_id=season)
    logger.info("Found %d divisions in season %s...", len(divs), season)
    for div in divs:
        db.add_division(division_id=div["id"], conference_id=div["conferenceId"], name=div["name"])
        logger.info("%s teams in %s", len(div["teams"]), div["name"])
        for team in div["teams"]:
            team_id = team.pop("id")
            team_name = team.pop("name")
            db.add_team(
                season_id=season,
                division_id=div["id"],
                conference_id=div["conferenceId"],
                team_id=int(team_id) if isinstance(team_id, str) and team_id.isdigit() else team_id,
                name=team_name,
                stats=team,
            )


def get_team_id(
    db: Database,
    team_name: str,
    season: int,
    division_id: int | None = None,
    conference_id: int | None = None,
) -> int | None:
    """Resolve a team id by name, optionally constrained to a division.

    CAHA reuses team names across levels; constrain by division+conference when known.
    """
    if not team_name:
        return None
    conditions = [f"season_id = {season}", f'name = "{team_name}"']
    if division_id is not None and conference_id is not None:
        conditions.append(f"division_id = {division_id}")
        conditions.append(f"conference_id = {conference_id}")
    teams = db.list_teams(" AND ".join(conditions))
    if len(teams) > 1:
        # Still ambiguous; prefer exact division match if not already used, else first.
        return teams[0]["team_id"]
    if not teams:
        raise ValueError("No team named %s in season %s" % (team_name, season))
    return teams[0]["team_id"]


def get_team_or_unknown(
    db: Database,
    team_name: str,
    season: int,
    division_id: int | None = None,
    conference_id: int | None = None,
):
    """Get team id from string, inserting UNKNOWN if needed."""
    try:
        team_id = get_team_id(db, team_name, season, division_id, conference_id)
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


def add_game(db: Database, season: int, team: dict[str, Any], game: dict[str, Any]):
    """Add a game to the database from a parsed row."""
    # Clean up dict and translate data.
    game_id = game.pop("id")
    start_time = game.pop("start_time", None)
    rink = game.pop("rink", None)
    # Get Team IDs from names and season.
    home, away = game["home"], game["away"]
    if home == team["name"]:
        home_id = team["team_id"]
        away_id = get_team_or_unknown(db, away, season, team["division_id"], team["conference_id"])
    else:
        home_id = get_team_or_unknown(db, home, season, team["division_id"], team["conference_id"])
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


def sync_season_teams(db: Database, season: int):
    """Sync games for all teams in a season."""
    teams = db.list_teams("season_id = %d" % season)
    game_ids: set[int] = set()
    for team in teams:
        logger.info("Syncing %s season %d...", team["name"], season)
        team_info = get_team(season_id=season, team_id=team["team_id"])  # type: ignore[arg-type]
        games = team_info.pop("games", [])
        for game in games:
            if game["id"] in game_ids:
                continue
            game_ids.add(game["id"])  # type: ignore[arg-type]
            add_game(db, season, team, game)


def sync_game_stats(db: Database):
    games = db.list_games()
    for game in games:
        if not game["stats"]:
            try:
                stats = scrape_game_stats(game["game_id"])  # type: ignore[arg-type]
            except Exception as e:  # Broad skip; remote data may be missing
                logger.exception("Failed to scrape game stats: %s", e)
                continue
            db.add_game_stats(game["game_id"], stats)  # type: ignore[arg-type]


def load_data(db: Database):
    db.create_tables()
    sync_seasons(db)
    seasons = [a["season_id"] for a in db.list_seasons()]
    for season in sorted(seasons, reverse=True):
        sync_divisions(db, season)
        sync_season_teams(db, season)


if __name__ == "__main__":
    load_data(Database())

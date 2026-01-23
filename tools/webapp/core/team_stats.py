from __future__ import annotations

from fractions import Fraction
from typing import Any, Optional

from .hockey import _league_game_is_cross_division_non_external, is_external_division_name
from .orm import _get_league_name, _orm_modules
from .player_stats import game_is_eligible_for_stats
from .seed_placeholders import SEED_PLACEHOLDER_TEAM_NAME, is_seed_placeholder_name


def compute_team_stats(db_conn, team_id: int, user_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    rows = list(
        m.HkyGame.objects.filter(
            user_id=int(user_id),
            team1_score__isnull=False,
            team2_score__isnull=False,
        )
        .filter(Q(team1_id=int(team_id)) | Q(team2_id=int(team_id)))
        .select_related("game_type")
        .values("team1_id", "team2_id", "team1_score", "team2_score", "game_type__name")
    )
    wins = losses = ties = gf = ga = 0
    for r in rows:
        # Keep non-league team totals consistent with league/team/player stat behavior:
        # exclude outlier games (MHR-like ratings handle outliers separately via goal-diff capping).
        r2 = dict(r)
        r2["game_type_name"] = r.get("game_type__name")
        if not game_is_eligible_for_stats(r2, team_id=int(team_id), league_name=None):
            continue
        t1 = int(r["team1_id"]) == team_id
        my_score = (
            int(r["team1_score"])
            if t1
            else int(r["team2_score"]) if r["team2_score"] is not None else 0
        )
        op_score = (
            int(r["team2_score"])
            if t1
            else int(r["team1_score"]) if r["team1_score"] is not None else 0
        )
        gf += my_score
        ga += op_score
        if my_score > op_score:
            wins += 1
        elif my_score < op_score:
            losses += 1
        else:
            ties += 1
    points_total = wins * 2 + ties * 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "gf": gf,
        "ga": ga,
        "points": points_total,
        "points_total": points_total,
    }


def compute_team_stats_league(db_conn, team_id: int, league_id: int) -> dict:
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db.models import Q

    league_name = _get_league_name(None, int(league_id))
    league_team_div: dict[int, str] = {
        int(tid): str(dn or "").strip()
        for tid, dn in m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list(
            "team_id", "division_name"
        )
    }
    rows: list[dict[str, Any]] = []
    for lg in (
        m.LeagueGame.objects.filter(
            league_id=int(league_id),
            game__team1_score__isnull=False,
            game__team2_score__isnull=False,
        )
        .filter(Q(game__team1_id=int(team_id)) | Q(game__team2_id=int(team_id)))
        .select_related("game", "game__game_type")
    ):
        g = lg.game
        t1_id = int(g.team1_id)
        t2_id = int(g.team2_id)
        rows.append(
            {
                "team1_id": t1_id,
                "team2_id": t2_id,
                "team1_score": g.team1_score,
                "team2_score": g.team2_score,
                "is_final": bool(g.is_final),
                "league_division_name": lg.division_name,
                "game_type_name": (g.game_type.name if g.game_type else None),
                "team1_league_division_name": league_team_div.get(t1_id),
                "team2_league_division_name": league_team_div.get(t2_id),
            }
        )
    wins = losses = ties = gf = ga = 0
    swins = slosses = sties = sgf = sga = 0

    def _is_regular_game(r: dict) -> bool:
        # Only regular-season games should contribute to standings points/rankings.
        gt = str(r.get("game_type_name") or "").strip()
        if not gt or not gt.lower().startswith("regular"):
            return False
        # Cross-division games are part of the team's overall record, but should not affect
        # division standings points/rankings.
        if _league_game_is_cross_division_non_external(r):
            return False
        # Any game involving an External team, or mapped to the External division, does not count for standings.
        for key in (
            "league_division_name",
            "team1_league_division_name",
            "team2_league_division_name",
        ):
            dn = str(r.get(key) or "").strip()
            if is_external_division_name(dn):
                return False
        return True

    for r in rows:
        if not game_is_eligible_for_stats(r, team_id=int(team_id), league_name=league_name):
            continue
        t1 = int(r["team1_id"]) == team_id
        my_score = (
            int(r["team1_score"])
            if t1
            else int(r["team2_score"]) if r["team2_score"] is not None else 0
        )
        op_score = (
            int(r["team2_score"])
            if t1
            else int(r["team1_score"]) if r["team1_score"] is not None else 0
        )
        gf += my_score
        ga += op_score
        if my_score > op_score:
            wins += 1
        elif my_score < op_score:
            losses += 1
        else:
            ties += 1

        if _is_regular_game(r):
            sgf += my_score
            sga += op_score
            if my_score > op_score:
                swins += 1
            elif my_score < op_score:
                slosses += 1
            else:
                sties += 1

    points = swins * 2 + sties * 1
    points_total = wins * 2 + ties * 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "gf": gf,
        "ga": ga,
        "points": points,
        "points_total": points_total,
        # Used only for sorting/tiebreakers in standings; display fields above include all games.
        "standings_wins": swins,
        "standings_losses": slosses,
        "standings_ties": sties,
        "standings_gf": sgf,
        "standings_ga": sga,
    }


def sort_key_team_standings(team_row: dict, stats: dict) -> tuple:
    """Standard hockey standings sort (points, wins, goal diff, goals for, goals against, name)."""
    pts = int(stats.get("points", 0))
    wins = int(stats.get("standings_wins", stats.get("wins", 0)))
    gf = int(stats.get("standings_gf", stats.get("gf", 0)))
    ga = int(stats.get("standings_ga", stats.get("ga", 0)))
    gd = gf - ga
    name = str(team_row.get("name") or "")
    return (-pts, -wins, -gd, -gf, ga, name.lower())


def _goal_diff_cap_contrib(diff: int, *, cap: int = 8) -> int:
    d = int(diff)
    c = int(cap)
    if c <= 0:
        return d
    if d > c:
        return c
    if d < -c:
        return -c
    return d


def _division_standings_regular_game_ok(r: dict) -> bool:
    gt = str(r.get("game_type_name") or "").strip()
    if not gt or not gt.lower().startswith("regular"):
        return False
    if _league_game_is_cross_division_non_external(r):
        return False
    for key in ("league_division_name", "team1_league_division_name", "team2_league_division_name"):
        dn = str(r.get(key) or "").strip()
        if is_external_division_name(dn):
            return False
    return True


def division_standings_team_ids(league_id: int, division_name: str) -> list[int]:
    """
    Return team ids ranked by division standings (Regular Season only) using NorCal/USA Hockey-like
    procedures:
      - Points percentage (Pts / (GP*2))
      - Head-to-head points among tied teams
      - Most wins
      - Goal differential (per-game diff capped to +/-8)
      - Goals against
      - Goals for

    Notes:
      - Regulation wins / penalty minutes / quickest first goal are not currently available from the
        stored schedule data, so they are not applied here.
    """
    lid = int(league_id)
    div = str(division_name or "").strip()
    if not div:
        return []

    _django_orm, m = _orm_modules()
    league_name = _get_league_name(None, lid)

    # Candidate division teams.
    team_rows = list(
        m.LeagueTeam.objects.filter(league_id=lid, division_name=div)
        .select_related("team")
        .values("team_id", "team__name")
    )
    team_name: dict[int, str] = {}
    team_ids: list[int] = []
    for r0 in team_rows:
        try:
            tid = int(r0.get("team_id") or 0)
        except Exception:
            continue
        if tid <= 0:
            continue
        nm = str(r0.get("team__name") or "").strip()
        if not nm:
            continue
        if nm == SEED_PLACEHOLDER_TEAM_NAME or is_seed_placeholder_name(nm):
            continue
        team_name[tid] = nm
        team_ids.append(tid)
    team_ids = sorted(set(team_ids))
    if not team_ids:
        return []
    team_set = set(team_ids)

    # League team -> division map needed for CAHA preseason eligibility and cross-division checks.
    league_team_div: dict[int, str] = {
        int(tid): str(dn or "").strip()
        for tid, dn in m.LeagueTeam.objects.filter(league_id=lid).values_list(
            "team_id", "division_name"
        )
    }

    # Aggregate regular-season division games.
    gp: dict[int, int] = {tid: 0 for tid in team_ids}
    pts: dict[int, int] = {tid: 0 for tid in team_ids}
    wins: dict[int, int] = {tid: 0 for tid in team_ids}
    gf: dict[int, int] = {tid: 0 for tid in team_ids}
    ga: dict[int, int] = {tid: 0 for tid in team_ids}
    gd_cap: dict[int, int] = {tid: 0 for tid in team_ids}
    h2h_pts: dict[tuple[int, int], int] = {}

    for lg in m.LeagueGame.objects.filter(
        league_id=lid,
        division_name=div,
        game__team1_score__isnull=False,
        game__team2_score__isnull=False,
    ).select_related("game", "game__game_type"):
        g = lg.game
        t1_id = int(g.team1_id)
        t2_id = int(g.team2_id)
        if t1_id not in team_set or t2_id not in team_set:
            continue
        if g.team1_score is None or g.team2_score is None:
            continue
        s1 = int(g.team1_score)
        s2 = int(g.team2_score)

        row = {
            "team1_id": t1_id,
            "team2_id": t2_id,
            "team1_score": s1,
            "team2_score": s2,
            "is_final": bool(getattr(g, "is_final", False)),
            "league_division_name": lg.division_name,
            "division_name": lg.division_name,
            "game_type_name": (g.game_type.name if g.game_type else None),
            "team1_league_division_name": league_team_div.get(t1_id),
            "team2_league_division_name": league_team_div.get(t2_id),
        }
        if not _division_standings_regular_game_ok(row):
            continue
        if not (
            game_is_eligible_for_stats(row, team_id=t1_id, league_name=league_name)
            and game_is_eligible_for_stats(row, team_id=t2_id, league_name=league_name)
        ):
            continue

        gp[t1_id] += 1
        gp[t2_id] += 1
        gf[t1_id] += s1
        ga[t1_id] += s2
        gf[t2_id] += s2
        ga[t2_id] += s1

        diff = s1 - s2
        diff_c = _goal_diff_cap_contrib(diff, cap=8)
        gd_cap[t1_id] += diff_c
        gd_cap[t2_id] -= diff_c

        if s1 > s2:
            pts[t1_id] += 2
            wins[t1_id] += 1
            h2h_pts[(t1_id, t2_id)] = int(h2h_pts.get((t1_id, t2_id), 0)) + 2
            h2h_pts[(t2_id, t1_id)] = int(h2h_pts.get((t2_id, t1_id), 0)) + 0
        elif s2 > s1:
            pts[t2_id] += 2
            wins[t2_id] += 1
            h2h_pts[(t1_id, t2_id)] = int(h2h_pts.get((t1_id, t2_id), 0)) + 0
            h2h_pts[(t2_id, t1_id)] = int(h2h_pts.get((t2_id, t1_id), 0)) + 2
        else:
            pts[t1_id] += 1
            pts[t2_id] += 1
            h2h_pts[(t1_id, t2_id)] = int(h2h_pts.get((t1_id, t2_id), 0)) + 1
            h2h_pts[(t2_id, t1_id)] = int(h2h_pts.get((t2_id, t1_id), 0)) + 1

    pp: dict[int, Fraction] = {}
    for tid in team_ids:
        gpi = int(gp.get(tid, 0))
        p = int(pts.get(tid, 0))
        pp[tid] = Fraction(p, gpi * 2) if gpi > 0 else Fraction(0, 1)

    def _name_key(tid: int) -> str:
        return str(team_name.get(int(tid), "")).casefold()

    def _head_to_head_points(tid: int, group: list[int]) -> int:
        total = 0
        for other in group:
            if int(other) == int(tid):
                continue
            total += int(h2h_pts.get((int(tid), int(other)), 0))
        return int(total)

    criteria: list[tuple[str, int]] = [
        ("h2h", -1),  # head-to-head points among tied teams
        ("wins", -1),  # most wins
        ("gd_cap", -1),  # capped goal diff
        ("ga", 1),  # lowest goals against
        ("gf", -1),  # highest goals for
    ]

    def _metric(tid: int, group: list[int], key: str) -> int:
        if key == "h2h":
            return _head_to_head_points(int(tid), group)
        if key == "wins":
            return int(wins.get(int(tid), 0))
        if key == "gd_cap":
            return int(gd_cap.get(int(tid), 0))
        if key == "ga":
            return int(ga.get(int(tid), 0))
        if key == "gf":
            return int(gf.get(int(tid), 0))
        return 0

    def _rank_group(group: list[int], start_idx: int, *, allow_restart: bool) -> list[int]:
        if len(group) <= 1:
            return list(group)
        if start_idx >= len(criteria):
            return sorted(group, key=_name_key)

        k, direction = criteria[start_idx]
        vals = {int(tid): _metric(int(tid), group, k) for tid in group}
        ordered = sorted(
            group, key=lambda tid: (int(direction) * int(vals[int(tid)]), _name_key(int(tid)))
        )

        out: list[int] = []
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and int(vals[int(ordered[j])]) == int(vals[int(ordered[i])]):
                j += 1
            sub = ordered[i:j]
            if len(sub) <= 1:
                out.extend(sub)
            else:
                # NorCal procedure: when a multi-team tie collapses to a 2-team tie, restart at H2H.
                if len(sub) == 2 and start_idx > 0 and allow_restart:
                    out.extend(_rank_group(list(sub), 0, allow_restart=False))
                else:
                    out.extend(
                        _rank_group(
                            list(sub),
                            start_idx + 1,
                            allow_restart=(len(sub) > 2),
                        )
                    )
            i = j
        return out

    base_sorted = sorted(team_ids, key=lambda tid: (-pp[int(tid)], _name_key(int(tid))))
    out_ids: list[int] = []
    i = 0
    while i < len(base_sorted):
        j = i + 1
        while j < len(base_sorted) and pp[int(base_sorted[j])] == pp[int(base_sorted[i])]:
            j += 1
        tie_group = base_sorted[i:j]
        if len(tie_group) <= 1:
            out_ids.extend(tie_group)
        else:
            out_ids.extend(_rank_group(list(tie_group), 0, allow_restart=(len(tie_group) > 2)))
        i = j

    return out_ids


def division_seed_team_id(league_id: int, division_name: str, seed: int) -> Optional[int]:
    try:
        s = int(seed)
    except Exception:
        return None
    if s <= 0:
        return None
    ranked = division_standings_team_ids(int(league_id), str(division_name or ""))
    if len(ranked) < s:
        return None
    return int(ranked[s - 1])

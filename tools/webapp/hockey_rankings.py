from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GameScore:
    team1_id: int
    team2_id: int
    team1_score: int
    team2_score: int


def clamp_goal_diff(diff: int, *, cap: int = 7) -> int:
    """
    MyHockeyRankings caps the per-game goal differential contribution (default cap: 7).
    """
    c = int(cap)
    if c <= 0:
        return int(diff)
    d = int(diff)
    if d > c:
        return c
    if d < -c:
        return -c
    return d


def compute_mhr_like_ratings(
    *,
    games: list[GameScore],
    max_goal_diff: int = 7,
    min_games_for_rating: int = 5,
    damping: float = 0.85,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> dict[int, dict[str, Any]]:
    """
    Compute team ratings in the spirit of MyHockeyRankings, per their published description:

      - AGD = average per-game (Goals For - Goals Against) with per-game diff capped to +/-max_goal_diff
      - SCHED = average opponent rating across games
      - Rating = AGD + SCHED

    This is a self-consistent system because SCHED depends on the (unknown) ratings.
    We solve it by fixed-point iteration with a mean-centering step each iteration (the system
    is translation-invariant: adding a constant to all ratings preserves the equations).

    Returns per-team dicts with:
      - games: int
      - agd: float
      - sched: float
      - rating: Optional[float] (None when games < min_games_for_rating)
      - rating_raw: float (always computed for teams with at least 1 game)
    """
    cap = int(max_goal_diff)
    min_games = max(1, int(min_games_for_rating))
    damp = float(damping)
    if not (0.0 < damp <= 1.0):
        damp = 0.85
    iters = max(1, int(max_iter))
    eps = float(tol)

    # Build per-team opponent lists (with duplicates) and AGD sums.
    opponents: dict[int, list[int]] = {}
    gd_sum: dict[int, int] = {}
    games_ct: dict[int, int] = {}

    for g in games or []:
        try:
            t1 = int(g.team1_id)
            t2 = int(g.team2_id)
            s1 = int(g.team1_score)
            s2 = int(g.team2_score)
        except Exception:
            continue
        if t1 == t2:
            continue

        d12 = clamp_goal_diff(s1 - s2, cap=cap)
        d21 = -d12

        opponents.setdefault(t1, []).append(t2)
        opponents.setdefault(t2, []).append(t1)
        gd_sum[t1] = int(gd_sum.get(t1, 0)) + int(d12)
        gd_sum[t2] = int(gd_sum.get(t2, 0)) + int(d21)
        games_ct[t1] = int(games_ct.get(t1, 0)) + 1
        games_ct[t2] = int(games_ct.get(t2, 0)) + 1

    team_ids = sorted(games_ct.keys())
    if not team_ids:
        return {}

    agd: dict[int, float] = {}
    for tid in team_ids:
        n = int(games_ct.get(tid, 0))
        agd[tid] = (float(gd_sum.get(tid, 0)) / float(n)) if n > 0 else 0.0

    # Initialize ratings from AGD (helps converge faster than zeros for sparse graphs).
    rating: dict[int, float] = {tid: float(agd.get(tid, 0.0)) for tid in team_ids}

    # Fixed-point iteration: r = AGD + avg_opp(r)
    for _ in range(iters):
        new_rating: dict[int, float] = {}
        for tid in team_ids:
            opps = opponents.get(tid) or []
            if not opps:
                new_rating[tid] = float(agd.get(tid, 0.0))
                continue
            sched = sum(float(rating.get(o, 0.0)) for o in opps) / float(len(opps))
            raw = float(agd.get(tid, 0.0)) + float(sched)
            new_rating[tid] = damp * raw + (1.0 - damp) * float(rating.get(tid, 0.0))

        # Center to mean 0 (translation invariance).
        mean_val = sum(new_rating.values()) / float(len(new_rating))
        for tid in team_ids:
            new_rating[tid] = float(new_rating[tid]) - float(mean_val)

        delta = max(abs(new_rating[tid] - rating[tid]) for tid in team_ids)
        rating = new_rating
        if delta <= eps:
            break

    # Final SCHED using converged ratings.
    out: dict[int, dict[str, Any]] = {}
    for tid in team_ids:
        opps = opponents.get(tid) or []
        sched = (sum(float(rating.get(o, 0.0)) for o in opps) / float(len(opps))) if opps else 0.0
        raw = float(agd.get(tid, 0.0)) + float(sched)
        n = int(games_ct.get(tid, 0))
        out[int(tid)] = {
            "games": n,
            "agd": float(agd.get(tid, 0.0)),
            "sched": float(sched),
            "rating_raw": float(raw),
            "rating": (float(raw) if n >= min_games else None),
        }
    return out


def parse_mhr_config_from_source(*, max_goal_diff: Optional[int] = None) -> int:
    """
    Placeholder helper in case we later want league-specific overrides.
    For now, defaults to the published MHR cap of 7.
    """
    try:
        v = int(max_goal_diff) if max_goal_diff is not None else 7
    except Exception:
        v = 7
    return 7 if v <= 0 else v


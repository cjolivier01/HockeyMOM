from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import re


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


def scale_ratings_to_0_99_9(results: dict[int, dict[str, Any]], *, key: str = "rating") -> dict[int, dict[str, Any]]:
    """
    Return a shallow-copied results dict with `key` shifted into a non-negative range
    with the top rated team being exactly 99.9.

    Important: MyHockeyRankings documents that rating *differences* correspond to the expected
    goal differential between teams. To preserve that property, we only APPLY A CONSTANT SHIFT
    (no min/max rescaling). We clamp at 0.0 only if necessary.

    Teams whose `key` is None are left as None.
    """
    if not results:
        return {}

    rated: list[tuple[int, float]] = []
    for tid, row in results.items():
        try:
            v = row.get(key)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            rated.append((int(tid), float(v)))
        except Exception:
            continue

    if not rated:
        return {int(tid): dict(row) for tid, row in results.items()}

    vmax = max(v for _tid, v in rated)
    offset = 99.9 - float(vmax)

    def _shift(v: float) -> float:
        # Preserve differences: v_new - u_new == v - u (unless clamped at 0).
        out_v = float(v) + float(offset)
        if out_v < 0.0:
            out_v = 0.0
        return round(out_v, 2)

    out: dict[int, dict[str, Any]] = {int(tid): dict(row) for tid, row in results.items()}
    for tid, v in rated:
        out[int(tid)][key] = _shift(float(v))
    return out


def scale_ratings_to_0_99_9_by_component(
    results: dict[int, dict[str, Any]], *, games: list[GameScore], key: str = "rating"
) -> dict[int, dict[str, Any]]:
    """
    Like `scale_ratings_to_0_99_9`, but normalizes each disconnected connected-component of the
    game graph independently, setting the top team in each component to 99.9.

    This is the right behavior when there is no cross-pollination between groups of teams: the
    MHR equations are translation-invariant per component, so absolute offsets between components
    are not identifiable.
    """
    if not results:
        return {}

    # Build adjacency from games.
    adj: dict[int, set[int]] = {}
    for g in games or []:
        try:
            a = int(g.team1_id)
            b = int(g.team2_id)
        except Exception:
            continue
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    all_nodes = set(adj.keys()) | {int(tid) for tid in results.keys()}
    out: dict[int, dict[str, Any]] = {int(tid): dict(row) for tid, row in results.items()}

    seen: set[int] = set()
    for start in sorted(all_nodes):
        if start in seen:
            continue
        # BFS component
        stack = [start]
        comp: list[int] = []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)

        rated_vals: list[float] = []
        for tid in comp:
            row = results.get(int(tid)) or {}
            v = row.get(key)
            if v is None:
                continue
            try:
                rated_vals.append(float(v))
            except Exception:
                continue
        if not rated_vals:
            continue

        vmax = max(rated_vals)
        offset = 99.9 - float(vmax)
        for tid in comp:
            row = results.get(int(tid)) or {}
            v = row.get(key)
            if v is None:
                continue
            try:
                shifted = float(v) + float(offset)
            except Exception:
                continue
            if shifted < 0.0:
                shifted = 0.0
            out[int(tid)][key] = round(float(shifted), 2)

    return out


_AGE_WORDS: list[tuple[str, int]] = [
    ("mite", 8),
    ("squirt", 10),
    ("peewee", 12),
    ("pee wee", 12),
    ("bantam", 14),
    ("midget", 16),
    ("junior", 18),
]

# Match a leading age like "12U", "12AA", "10 B West", etc.
_AGE_RE = re.compile(r"(?i)(?:^|\b)(\d{1,2})(?=\s|$|u\b|[A-Za-z])")


def parse_age_from_division_name(division_name: str) -> Optional[int]:
    """
    Extract the numeric age from a division string like "10U B West", "12AA", "16A", etc.
    Returns None if no plausible age is found.
    """
    s = str(division_name or "").strip()
    if not s:
        return None
    low = s.lower()
    for word, age in _AGE_WORDS:
        if word in low:
            return int(age)
    m = _AGE_RE.search(s)
    if not m:
        return None
    val = m.group(1)
    try:
        age = int(val)
    except Exception:
        return None
    # Youth ages are typically 6..20; keep a broad band.
    if age < 4 or age > 30:
        return None
    return age


def normalize_ratings_to_0_99_9_age_aware(
    results: dict[int, dict[str, Any]],
    *,
    games: list[GameScore],
    team_age: dict[int, Optional[int]],
    key: str = "rating",
    min_cross_teams_each_side: int = 2,
) -> dict[int, dict[str, Any]]:
    """
    Normalize ratings into a [0, 99.9] display range while preserving rating differences
    within the groups we consider comparable.

    Rule:
      - By default, ratings are anchored separately per age group (10U vs 12U, etc), so each age
        group can have a 99.9-rated team.
      - Two age groups are only "comparable" (share the same anchoring) if there is strong
        cross-pollination: at least `min_cross_teams_each_side` distinct teams from each age group
        have played the other age group.
      - Within an age-group anchoring set, if the (filtered) team graph is disconnected, each
        disconnected component is anchored independently (top team in each component is 99.9).

    Important: we ONLY APPLY CONSTANT SHIFTS per component (no min/max rescaling), so differences
    remain in goal-differential units inside each anchored component.
    """
    if not results:
        return {}

    # Track cross-age participation for "strong" links.
    cross: dict[tuple[int, int], tuple[set[int], set[int]]] = {}
    for g in games or []:
        try:
            a = int(g.team1_id)
            b = int(g.team2_id)
        except Exception:
            continue
        if a == b:
            continue
        age_a = team_age.get(a)
        age_b = team_age.get(b)
        if age_a is None or age_b is None:
            continue
        if int(age_a) == int(age_b):
            continue
        lo, hi = (int(age_a), int(age_b)) if int(age_a) < int(age_b) else (int(age_b), int(age_a))
        key_pair = (lo, hi)
        sa, sb = cross.get(key_pair, (set(), set()))
        # Always store as (teams_in_lo_age, teams_in_hi_age)
        if int(age_a) == lo:
            sa.add(a)
            sb.add(b)
        else:
            sa.add(b)
            sb.add(a)
        cross[key_pair] = (sa, sb)

    strong_age_links: set[tuple[int, int]] = set()
    thresh = max(1, int(min_cross_teams_each_side))
    for (lo, hi), (sa, sb) in cross.items():
        if len(sa) >= thresh and len(sb) >= thresh:
            strong_age_links.add((int(lo), int(hi)))

    def _age_id(tid: int) -> Optional[int]:
        v = team_age.get(int(tid))
        return int(v) if v is not None else None

    def _ages_strong(a: Optional[int], b: Optional[int]) -> bool:
        if a is None or b is None:
            return False
        if a == b:
            return True
        lo, hi = (a, b) if a < b else (b, a)
        return (lo, hi) in strong_age_links

    # Build adjacency for anchoring components:
    # - include same-age games
    # - include cross-age games ONLY for strongly-linked age pairs
    adj: dict[int, set[int]] = {}
    for g in games or []:
        try:
            a = int(g.team1_id)
            b = int(g.team2_id)
        except Exception:
            continue
        if a == b:
            continue
        age_a = _age_id(a)
        age_b = _age_id(b)
        if age_a is None or age_b is None:
            continue
        if not _ages_strong(age_a, age_b):
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # Ensure all rated teams are included as nodes (even if they have no edges for anchoring).
    all_nodes = set(adj.keys()) | {int(tid) for tid in results.keys()}

    out: dict[int, dict[str, Any]] = {int(tid): dict(row) for tid, row in results.items()}

    seen: set[int] = set()
    for start in sorted(all_nodes):
        if start in seen:
            continue
        # BFS component
        stack = [start]
        comp: list[int] = []
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, set()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)

        rated_vals: list[float] = []
        for tid in comp:
            row = results.get(int(tid)) or {}
            v = row.get(key)
            if v is None:
                continue
            try:
                rated_vals.append(float(v))
            except Exception:
                continue
        if not rated_vals:
            continue

        vmax = max(rated_vals)
        offset = 99.9 - float(vmax)
        for tid in comp:
            row = results.get(int(tid)) or {}
            v = row.get(key)
            if v is None:
                continue
            try:
                shifted = float(v) + float(offset)
            except Exception:
                continue
            if shifted < 0.0:
                shifted = 0.0
            out[int(tid)][key] = round(float(shifted), 2)

    return out


def filter_games_ignore_cross_age(games: list[GameScore], *, team_age: dict[int, Optional[int]]) -> list[GameScore]:
    """
    Return only games where both teams have a known age and the ages match.
    This implements "ignore games that cross an age boundary" for rating computations.
    """
    out: list[GameScore] = []
    for g in games or []:
        try:
            a = int(g.team1_id)
            b = int(g.team2_id)
        except Exception:
            continue
        age_a = team_age.get(a)
        age_b = team_age.get(b)
        if age_a is None or age_b is None:
            continue
        if int(age_a) != int(age_b):
            continue
        out.append(g)
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

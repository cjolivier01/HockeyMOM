"""Normalization helpers for per-game TimeToScore stats payloads."""

from __future__ import annotations

from typing import Any, Optional


def parse_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s.split()[0])
    except Exception:
        return None


def extract_roster(game_stats: dict[str, Any], side: str) -> list[dict[str, Any]]:
    rows = game_stats.get(f"{side}Players") or []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "number": str(row.get("number") or "").strip() or None,
                "position": str(row.get("position") or "").strip() or None,
            }
        )
    return out


def aggregate_goals_assists(game_stats: dict[str, Any]) -> list[dict[str, Any]]:
    # SharksIce game-center API provides per-player totals directly.
    for key_home, key_away in (("homeSkaters", "awaySkaters"), ("home_skaters", "away_skaters")):
        if isinstance(game_stats.get(key_home), list) or isinstance(game_stats.get(key_away), list):
            out: list[dict[str, Any]] = []
            for row in list(game_stats.get(key_home) or []) + list(game_stats.get(key_away) or []):
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                gval = parse_int_or_none(row.get("goals")) or 0
                aval = parse_int_or_none(row.get("assists")) or 0
                if gval == 0 and aval == 0:
                    continue
                out.append({"name": name, "goals": int(gval), "assists": int(aval)})
            return out

    def incr(d: dict[str, dict[str, int]], who: str, key: str) -> None:
        rec = d.setdefault(who, {"goals": 0, "assists": 0})
        rec[key] = rec.get(key, 0) + 1

    stats_by_player: dict[str, dict[str, int]] = {}
    for side in ("home", "away"):
        scoring = game_stats.get(f"{side}Scoring") or []
        if not isinstance(scoring, list):
            continue
        for srow in scoring:
            if not isinstance(srow, dict):
                continue
            gname = str(srow.get("goal") or "").strip()
            a1 = str(srow.get("assist1") or "").strip()
            a2 = str(srow.get("assist2") or "").strip()
            if gname:
                incr(stats_by_player, gname, "goals")
            if a1:
                incr(stats_by_player, a1, "assists")
            if a2:
                incr(stats_by_player, a2, "assists")

    out: list[dict[str, Any]] = []
    for name, rec in stats_by_player.items():
        out.append({"name": name, "goals": int(rec.get("goals", 0)), "assists": int(rec.get("assists", 0))})
    return out


from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, Optional


def parse_dt_or_none(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        if "T" in s:
            return dt.datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
        return dt.datetime.fromisoformat(s + "T00:00").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def compute_team_stats(rows: Iterable[Dict[str, Any]], team_id: int) -> Dict[str, int]:
    wins = losses = ties = gf = ga = 0
    for r in rows:
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
    points = wins * 2 + ties * 1
    return {"wins": wins, "losses": losses, "ties": ties, "gf": gf, "ga": ga, "points": points}


def aggregate_players_totals(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    for r in rows:
        pid = int(r["player_id"])
        goals = int(r.get("goals") or 0)
        assists = int(r.get("assists") or 0)
        pim = int(r.get("pim") or 0)
        shots = int(r.get("shots") or 0)
        out[pid] = {
            "goals": goals,
            "assists": assists,
            "points": goals + assists,
            "shots": shots,
            "pim": pim,
        }
    return out


def read_dirwatch_state() -> dict:
    from pathlib import Path
    import json

    state_path = Path("/var/lib/dirwatcher/state.json")
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {"processed": {}, "active": {}}


def send_email(to_addr: str, subject: str, body: str, from_addr: Optional[str] = None) -> None:
    import os
    import shutil
    import subprocess

    from_addr = from_addr or ("no-reply@" + os.uname().nodename)
    msg = (
        f"From: {from_addr}\nTo: {to_addr}\nSubject: {subject}\n"
        f"Content-Type: text/plain; charset=utf-8\n\n{body}\n"
    )
    sendmail = shutil.which("sendmail")
    if not sendmail:
        return
    try:
        subprocess.run([sendmail, "-t"], input=msg.encode("utf-8"), check=True)
    except Exception:
        # Best-effort only
        return


from __future__ import annotations

import importlib.util


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_apply_shot_implication_chain_in_game_event_stats(monkeypatch):
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    monkeypatch.setenv("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    mod = _load_app_module()

    rows = []
    rows += [{"Event Type": "Goal", "Team Side": "Home"}] * 2
    rows += [{"Event Type": "xG", "Team Side": "Home"}] * 1
    rows += [{"Event Type": "SOG", "Team Side": "Home"}] * 3
    rows += [{"Event Type": "Shot", "Team Side": "Home"}] * 4

    rows += [{"Event Type": "Goal", "Team Side": "Away"}] * 1
    rows += [{"Event Type": "SOG", "Team Side": "Away"}] * 2

    stats = mod.compute_game_event_stats_by_side(rows)
    by_type = {
        str(r.get("event_type") or ""): (int(r.get("home") or 0), int(r.get("away") or 0))
        for r in stats
    }

    assert by_type["Goal"] == (2, 1)
    assert by_type["xG"] == (3, 1)  # goals + xg
    assert by_type["SOG"] == (6, 3)  # (goals + xg) + sog
    assert by_type["Shot"] == (10, 3)  # ((goals + xg) + sog) + shot

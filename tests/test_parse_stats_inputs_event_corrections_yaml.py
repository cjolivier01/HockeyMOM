from __future__ import annotations

import importlib.util
from pathlib import Path


def should_parse_event_corrections_from_yaml_file_list(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "parse_stats_inputs.py"
    spec = importlib.util.spec_from_file_location("parse_stats_inputs", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _load_input_entries_from_yaml_file_list = mod._load_input_entries_from_yaml_file_list

    yml = tmp_path / "games.yaml"
    yml.write_text(
        """
games:
  - label: utah-1
    path: utah-1/stats
    side: AWAY
    metadata:
      home_team: Texas Warriors 12AA
      away_team: San Jose Jr Sharks 12AA-2
      date: "2026-01-16"
      game_video: "https://youtu.be/abc123"
    event_corrections:
      reason: "Swap scorer/assist"
      patch:
        - match:
            event_type: Goal
            period: 2
            game_time: "03:14"
            team_side: Away
            jersey: "3"
          set:
            jersey: "13"
          note: "see video"
""".lstrip(),
        encoding="utf-8",
    )

    entries = _load_input_entries_from_yaml_file_list(yml, base_dir=yml.parent, use_t2s=True)
    assert entries
    e0 = entries[0]
    assert e0.label == "utah-1"
    assert e0.side == "away"
    assert e0.event_corrections
    assert isinstance(e0.event_corrections, dict)
    assert e0.event_corrections.get("reason") == "Swap scorer/assist"

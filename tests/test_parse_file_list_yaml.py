from __future__ import annotations

from pathlib import Path

import scripts.parse_stats_inputs as pss


def should_parse_yaml_file_list_mapping_metadata_alias(tmp_path: Path) -> None:
    file_list = tmp_path / "games.yaml"
    file_list.write_text(
        """
games:
  - path: some_dir
    side: HOME
    metadata:
      home_team: "Home Team"
      away_team: "Away Team"
    league: CAHA
""".lstrip(),
        encoding="utf-8",
    )

    entries = pss._load_input_entries_from_yaml_file_list(
        file_list, base_dir=tmp_path, use_t2s=True
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.path == (tmp_path / "some_dir").resolve()
    assert e.side == "home"
    assert e.meta.get("home_team") == "Home Team"
    assert e.meta.get("away_team") == "Away Team"
    assert e.meta.get("league") == "CAHA"


def should_parse_yaml_file_list_t2s_only_entry(tmp_path: Path) -> None:
    file_list = tmp_path / "games.yaml"
    file_list.write_text(
        """
games:
  - t2s: 51602
    side: AWAY
    label: stockton-r2
    metadata:
      game_video: "https://youtu.be/example"
""".lstrip(),
        encoding="utf-8",
    )

    entries = pss._load_input_entries_from_yaml_file_list(
        file_list, base_dir=tmp_path, use_t2s=True
    )
    assert len(entries) == 1
    e = entries[0]
    assert e.path is None
    assert e.t2s_id == 51602
    assert e.side == "away"
    assert e.label == "stockton-r2"
    assert e.meta.get("game_video") == "https://youtu.be/example"

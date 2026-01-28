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

    entries, teams = pss._load_input_entries_from_yaml_file_list(
        file_list, base_dir=tmp_path, use_t2s=True
    )
    assert teams == {}
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

    entries, teams = pss._load_input_entries_from_yaml_file_list(
        file_list, base_dir=tmp_path, use_t2s=True
    )
    assert teams == {}
    assert len(entries) == 1
    e = entries[0]
    assert e.path is None
    assert e.t2s_id == 51602
    assert e.side == "away"
    assert e.label == "stockton-r2"
    assert e.meta.get("game_video") == "https://youtu.be/example"


def should_parse_yaml_file_list_sheets_entry(tmp_path: Path) -> None:
    shared_long_dir = tmp_path / "shared_long"
    shared_long_dir.mkdir(parents=True, exist_ok=True)
    long_sheet = shared_long_dir / "tv-12-1-r1-long-51238.xlsx"
    long_sheet.write_bytes(b"")

    file_list = tmp_path / "games.yaml"
    file_list.write_text(
        """
games:
  - label: tv-12-1-r1
    shared_long_path: shared_long
    metadata:
      game_video: "https://youtu.be/example"
    sheets:
      - side: AWAY
        path: away_stats
      - side: HOME
        path: home_stats
""".lstrip(),
        encoding="utf-8",
    )

    entries, teams = pss._load_input_entries_from_yaml_file_list(
        file_list, base_dir=tmp_path, use_t2s=True
    )
    assert teams == {}
    assert len(entries) == 3

    # Shared long sheet entry.
    long_entries = [e for e in entries if e.path == long_sheet.resolve()]
    assert len(long_entries) == 1
    assert long_entries[0].side is None
    assert long_entries[0].label == "tv-12-1-r1"
    assert long_entries[0].meta.get("game_video") == "https://youtu.be/example"

    # Per-side sheet entries.
    sheet_entries = [e for e in entries if e.path != long_sheet.resolve()]
    assert {e.side for e in sheet_entries} == {"home", "away"}
    assert {e.label for e in sheet_entries} == {"tv-12-1-r1"}
    assert all(e.meta.get("game_video") == "https://youtu.be/example" for e in sheet_entries)


def should_parse_yaml_file_list_team_icons_section(tmp_path: Path) -> None:
    file_list = tmp_path / "games.yaml"
    file_list.write_text(
        """
teams:
  - name: Home Team
    icon: home.png
    replace_logo: true
  - name: Away Team
    icon: away.png
games:
  - t2s: 51602
    side: HOME
""".lstrip(),
        encoding="utf-8",
    )

    entries, teams = pss._load_input_entries_from_yaml_file_list(
        file_list, base_dir=tmp_path, use_t2s=True
    )
    assert len(entries) == 1
    assert entries[0].t2s_id == 51602

    import re

    home_key = re.sub(r"\\s+", " ", "Home Team".strip()).casefold()
    away_key = re.sub(r"\\s+", " ", "Away Team".strip()).casefold()
    assert home_key in teams
    assert away_key in teams
    assert teams[home_key].icon == "home.png"
    assert teams[home_key].replace_logo is True

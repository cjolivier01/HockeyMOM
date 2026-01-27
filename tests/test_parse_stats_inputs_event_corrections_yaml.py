from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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


def should_error_if_event_corrections_is_nested_under_metadata(tmp_path: Path) -> None:
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
  - label: sharks-12-1-r4
    path: sharks-12-1-r4/stats
    metadata:
      home_team: San Jose Jr Sharks 12AA-1
      away_team: San Jose Jr Sharks 12AA-2
      game_video: "https://youtu.be/m_8Vj0upEJs"
      event_corrections:
        reason: "Fix goal clip time"
        patch:
          - match:
              event_type: Goal
              period: 3
              game_time: "12:53"
              team_side: Away
              jersey: "1"
            set:
              video_time: "50:30"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"games\[0\]\.metadata contains reserved key\(s\)"):
        _load_input_entries_from_yaml_file_list(yml, base_dir=yml.parent, use_t2s=True)


def should_error_on_common_yaml_patch_typos(tmp_path: Path) -> None:
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
  - label: sharks-12-1-r4
    path: sharks-12-1-r4/stats
    side: AWAY
    metadata:
      home_team: San Jose Jr Sharks 12AA-1
      away_team: San Jose Jr Sharks 12AA-2
      game_video: "https://youtu.be/m_8Vj0upEJs"
    event_corrections:
      reason: "Fix goal clip time"
      patch:
        - -match:
            event_type: Goal
            period: 3
            game_time: "12:53"
            team_side: Away
            jersey: "1"
          set:
            video_time: "50:30"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"did you mean 'match'\\?"):
        _load_input_entries_from_yaml_file_list(yml, base_dir=yml.parent, use_t2s=True)


def should_preserve_event_corrections_when_expanding_directory_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "parse_stats_inputs.py"
    spec = importlib.util.spec_from_file_location("parse_stats_inputs", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stats_dir = tmp_path / "sharks-12-1-r4" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "sharks-12-1-r4-53907.xlsx").write_bytes(b"")

    yml = tmp_path / "games.yaml"
    yml.write_text(
        """
games:
  - label: sharks-12-1-r4
    path: sharks-12-1-r4/stats
    metadata:
      home_team: San Jose Jr Sharks 12AA-1
      away_team: San Jose Jr Sharks 12AA-2
      game_video: "https://youtu.be/m_8Vj0upEJs"
    event_corrections:
      reason: "Fix goal clip time"
      patch:
        - match:
            event_type: Goal
            period: 3
            game_time: "12:53"
            team_side: Away
            jersey: "1"
          set:
            video_time: "50:30"
""".lstrip(),
        encoding="utf-8",
    )

    entries = mod._load_input_entries_from_yaml_file_list(yml, base_dir=yml.parent, use_t2s=True)
    assert len(entries) == 1
    e0 = entries[0]
    assert e0.event_corrections

    discovered = mod._expand_dir_input_to_game_sheets(e0.path)
    assert len(discovered) == 1
    e1 = mod._clone_input_entry_for_path(e0, discovered[0])
    assert e1.label == "sharks-12-1-r4"
    assert e1.event_corrections == e0.event_corrections


def should_accept_home_away_team_icon_aliases_for_logos(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "scripts" / "parse_stats_inputs.py"
    spec = importlib.util.spec_from_file_location("parse_stats_inputs", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    icon_bytes = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 32)
    (tmp_path / "home.png").write_bytes(icon_bytes)
    (tmp_path / "away.png").write_bytes(icon_bytes)

    meta = {
        "home_team_icon": "home.png",
        "away_team_icon": "away.png",
    }
    out = mod._load_logo_fields_from_meta(meta, base_dir=tmp_path, warn_label="utah-1")
    assert out.get("home_logo_b64")
    assert out.get("home_logo_content_type") == "image/png"
    assert out.get("away_logo_b64")
    assert out.get("away_logo_content_type") == "image/png"

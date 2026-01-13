from __future__ import annotations

from typing import Any, Dict, List, Tuple

import hmlib.cli.concat_oriented_videos as concat_cli


def should_concat_both_sides_by_default(monkeypatch, tmp_path):
    game_dir = tmp_path / "Videos" / "game-1"
    game_dir.mkdir(parents=True)

    (game_dir / "l1.mp4").touch()
    (game_dir / "l2.mp4").touch()
    (game_dir / "r1.mp4").touch()

    cfg: Dict[str, Any] = {
        "game": {
            "videos": {
                "left": ["l1.mp4", "l2.mp4"],
                "right": ["r1.mp4"],
            }
        }
    }

    monkeypatch.setattr(concat_cli, "get_game_dir", lambda game_id: str(game_dir))
    monkeypatch.setattr(concat_cli, "get_game_config_private", lambda game_id: cfg)

    calls: List[Tuple[List[str], str]] = []

    def _fake_concat_copy(inputs: List[str], output_path: str) -> None:
        calls.append((list(inputs), output_path))

    monkeypatch.setattr(concat_cli, "_concat_copy", _fake_concat_copy)

    rc = concat_cli.main(["--game-id", "game-1"])
    assert rc == 0
    assert calls == [
        ([str(game_dir / "l1.mp4"), str(game_dir / "l2.mp4")], str(game_dir / "left.mp4")),
        ([str(game_dir / "r1.mp4")], str(game_dir / "right.mp4")),
    ]


def should_concat_left_only(monkeypatch, tmp_path):
    game_dir = tmp_path / "Videos" / "game-2"
    game_dir.mkdir(parents=True)

    (game_dir / "l1.mp4").touch()
    (game_dir / "l2.mp4").touch()
    (game_dir / "r1.mp4").touch()

    cfg: Dict[str, Any] = {
        "game": {
            "videos": {
                "left": ["l1.mp4", "l2.mp4"],
                "right": ["r1.mp4"],
            }
        }
    }

    monkeypatch.setattr(concat_cli, "get_game_dir", lambda game_id: str(game_dir))
    monkeypatch.setattr(concat_cli, "get_game_config_private", lambda game_id: cfg)

    calls: List[Tuple[List[str], str]] = []

    def _fake_concat_copy(inputs: List[str], output_path: str) -> None:
        calls.append((list(inputs), output_path))

    monkeypatch.setattr(concat_cli, "_concat_copy", _fake_concat_copy)

    rc = concat_cli.main(["--game-id", "game-2", "--left"])
    assert rc == 0
    assert calls == [
        ([str(game_dir / "l1.mp4"), str(game_dir / "l2.mp4")], str(game_dir / "left.mp4"))
    ]


def should_skip_when_output_exists(monkeypatch, tmp_path):
    game_dir = tmp_path / "Videos" / "game-3"
    game_dir.mkdir(parents=True)
    (game_dir / "left.mp4").touch()

    cfg: Dict[str, Any] = {"game": {"videos": {"left": ["l1.mp4"], "right": ["r1.mp4"]}}}

    monkeypatch.setattr(concat_cli, "get_game_dir", lambda game_id: str(game_dir))
    monkeypatch.setattr(concat_cli, "get_game_config_private", lambda game_id: cfg)

    calls: List[Tuple[List[str], str]] = []

    def _fake_concat_copy(inputs: List[str], output_path: str) -> None:
        calls.append((list(inputs), output_path))

    monkeypatch.setattr(concat_cli, "_concat_copy", _fake_concat_copy)

    rc = concat_cli.main(["--game-id", "game-3", "--left"])
    assert rc == 0
    assert calls == []

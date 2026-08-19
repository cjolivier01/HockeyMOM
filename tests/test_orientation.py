from pathlib import Path
from typing import Any, Dict

import hmlib.orientation as orientation


def should_configure_cameras_with_different_chapter_counts(monkeypatch, tmp_path):
    game_dir = tmp_path / "game"
    left_files = [game_dir / "cam1" / f"left-{index}.mp4" for index in range(1, 6)]
    right_files = [game_dir / "cam2" / f"right-{index}.mp4" for index in range(1, 5)]
    analyzed_videos = {
        "left": {index: str(path) for index, path in enumerate(left_files, start=1)},
        "right": {index: str(path) for index, path in enumerate(right_files, start=1)},
    }
    saved: Dict[str, Any] = {}

    monkeypatch.setattr(orientation, "get_game_dir", lambda game_id: str(game_dir))
    monkeypatch.setattr(
        orientation, "get_available_videos", lambda dir_name: {"cam1": {}, "cam2": {}}
    )
    monkeypatch.setattr(orientation, "get_game_config_private", lambda game_id: {})
    monkeypatch.setattr(
        orientation,
        "get_game_videos_analysis",
        lambda game_id, device, inference_scale: analyzed_videos,
    )
    monkeypatch.setattr(
        orientation,
        "save_private_config",
        lambda game_id, data: saved.update(game_id=game_id, data=data),
    )

    result = orientation.configure_game_videos(game_id="game")

    assert result == {
        "left": [str(path) for path in left_files],
        "right": [str(path) for path in right_files],
    }
    assert saved == {
        "game_id": "game",
        "data": {
            "game": {
                "videos": {
                    "left": [str(Path("cam1") / path.name) for path in left_files],
                    "right": [str(Path("cam2") / path.name) for path in right_files],
                }
            }
        },
    }

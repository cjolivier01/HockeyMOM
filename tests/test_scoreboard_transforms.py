from __future__ import annotations

from typing import Any, Dict, List

import torch

from hmlib.transforms.scoreboard_transforms import HmConfigureScoreboard


def _scoreboard_game_config() -> Dict[str, Any]:
    return {
        "rink": {
            "scoreboard": {
                "projected_width": "%10",
                "projected_height": "%20",
            }
        }
    }


def should_configure_scoreboard_from_stitched_reference_frame(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def _fake_configure_scoreboard(game_id: str, image=None, **kwargs):
        calls.append({"game_id": game_id, "image": image})
        return [[10, 20], [30, 20], [30, 40], [10, 40]]

    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.configure_scoreboard",
        _fake_configure_scoreboard,
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_config",
        lambda game_id: _scoreboard_game_config(),
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_clip_box",
        lambda game_id: None,
    )

    transform = HmConfigureScoreboard(game_id="test-game")
    results = {"img": torch.zeros((1, 4, 4, 3), dtype=torch.uint8)}
    configured = transform(results)

    assert calls == [{"game_id": "test-game", "image": None}]
    assert configured["scoreboard_cfg"]["scoreboard_points"] == [
        [10, 20],
        [30, 20],
        [30, 40],
        [10, 40],
    ]


def should_fallback_to_current_frame_when_stitched_reference_is_missing(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def _fake_configure_scoreboard(game_id: str, image=None, **kwargs):
        calls.append({"game_id": game_id, "image": image})
        if image is None:
            raise FileNotFoundError("missing s.png")
        return [[1, 2], [3, 2], [3, 4], [1, 4]]

    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.configure_scoreboard",
        _fake_configure_scoreboard,
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_config",
        lambda game_id: _scoreboard_game_config(),
    )
    monkeypatch.setattr(
        "hmlib.transforms.scoreboard_transforms.get_clip_box",
        lambda game_id: None,
    )

    image = torch.zeros((1, 4, 4, 3), dtype=torch.uint8)
    transform = HmConfigureScoreboard(game_id="test-game")
    configured = transform({"img": image})

    assert len(calls) == 2
    assert calls[0] == {"game_id": "test-game", "image": None}
    assert calls[1]["game_id"] == "test-game"
    assert calls[1]["image"] is image
    assert configured["scoreboard_cfg"]["scoreboard_points"] == [
        [1, 2],
        [3, 2],
        [3, 4],
        [1, 4],
    ]

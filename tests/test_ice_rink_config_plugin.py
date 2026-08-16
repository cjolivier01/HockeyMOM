from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

if torch is not None:
    TESTS_DIR = Path(__file__).resolve().parent
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))

    from aspen_plugin_harness import make_track_data_sample
    from hmlib.aspen.plugins.ice_rink_boundaries_plugins import IceRinkSegmConfigPlugin
    from hmlib.utils.gpu import wrap_tensor
else:
    make_track_data_sample = None  # type: ignore[assignment]
    IceRinkSegmConfigPlugin = None  # type: ignore[assignment]
    wrap_tensor = None  # type: ignore[assignment]

requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


@requires_torch
def should_prefer_stitched_frame_shape_over_detector_input_shape(monkeypatch) -> None:
    captured = {}

    def _fake_configure(**kwargs):
        captured.update(kwargs)
        return {"shape": tuple(kwargs["expected_shape"])}

    monkeypatch.setattr(
        "hmlib.segm.ice_rink.configure_ice_rink_mask",
        _fake_configure,
    )

    plugin = IceRinkSegmConfigPlugin()
    result = plugin(
        {
            "data_samples": make_track_data_sample(num_frames=1, ori_shape=(200, 300)),
            "img": wrap_tensor(torch.zeros((1, 20, 30, 3), dtype=torch.float32)),
            "inputs": torch.zeros((1, 3, 736, 1984), dtype=torch.float32),
            "shared": {"game_id": "game-1"},
        }
    )

    assert result["rink_profile"]["shape"] == (20, 30)
    assert tuple(captured["expected_shape"]) == (20, 30)
    assert isinstance(captured["image"], torch.Tensor)
    assert tuple(captured["image"].shape) == (20, 30, 3)

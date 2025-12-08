from __future__ import annotations

import types
from typing import Any, Dict

import torch

from hmlib.tasks import tracking


class _DummyDataloader:
    """Minimal dataloader stub so run_mmtrack can build AspenNet once and exit."""

    def __init__(self, batch_size: int = 1, fps: float = 30.0):
        self.batch_size = batch_size
        self.fps = fps

    def __len__(self) -> int:
        # Zero batches so the main loop exits immediately after setup.
        return 0

    def __iter__(self):
        return iter(())


def should_propagate_camera_ui_into_aspen_shared(monkeypatch):
    captured: Dict[str, Any] = {}
    sentinel_controller: object = object()

    # Stub AspenNet so we can inspect the shared dict passed from run_mmtrack.
    class DummyAspenNet(torch.nn.Module):
        def __init__(self, name: str, graph_cfg: Dict[str, Any], shared: Dict[str, Any] | None = None, **_: Any):  # type: ignore[override]
            super().__init__()
            captured["shared"] = dict(shared or {})

        def to(self, *args: Any, **kwargs: Any):  # pragma: no cover - trivial passthrough
            return self

        def forward(self, context: Dict[str, Any]):  # pragma: no cover - not exercised
            return context

        def finalize(self):  # pragma: no cover - not exercised
            pass

    monkeypatch.setattr(tracking, "AspenNet", DummyAspenNet)
    # Avoid filesystem lookups for precomputed CSVs.
    monkeypatch.setattr(tracking, "find_latest_dataframe_file", lambda *a, **k: None)

    dl = _DummyDataloader()

    cfg: Dict[str, Any] = {
        "aspen": {
            "plugins": {},
            "pipeline": {},
        },
        "initial_args": {
            "camera_ui": 1,
        },
        "camera_ui": 1,
        "game_config": {},
        "stitch_rotation_controller": sentinel_controller,
    }

    tracking.run_mmtrack(
        model=None,
        pose_inferencer=None,
        config=cfg,
        dataloader=dl,
        postprocessor=None,
        progress_bar=None,
        device=torch.device("cpu"),
        input_cache_size=1,
        fp16=False,
        no_cuda_streams=True,
        track_mean_mode=None,
        profiler=None,
    )

    shared = captured.get("shared")
    assert isinstance(shared, dict)
    # Ensure the CLI flag is threaded into Aspen shared context for PlayTrackerPlugin.
    assert shared.get("camera_ui") == 1
    # Stitch rotation controller should also be forwarded untouched.
    assert shared.get("stitch_rotation_controller") is sentinel_controller

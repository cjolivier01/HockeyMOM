from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List

import torch

import hmlib.cli.stitch as stitch_cli


class _DummyAspenNet(torch.nn.Module):
    """Lightweight AspenNet stub to capture graph config and contexts."""

    def __init__(
        self,
        name: str,
        graph_cfg: Dict[str, Any],
        shared: Dict[str, Any] | None = None,
        **_: Any,
    ):  # type: ignore[override]
        super().__init__()
        self.name = name
        self.graph_cfg = graph_cfg
        self.shared = dict(shared or {})
        self.calls: List[Dict[str, Any]] = []

    def to(self, *args: Any, **kwargs: Any):  # pragma: no cover - trivial passthrough
        return self

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Capture the per-frame context passed from stitch_videos.
        self.calls.append(dict(context))
        return context

    def finalize(self):  # pragma: no cover - not exercised deeply here
        pass


class _DummyVideoInfo:
    """Minimal BasicVideoInfo stub to avoid real ffprobe/IO."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.frame_count = 10
        self.fps = 30.0


class _DummyStitchDataset:
    """Small iterable that yields a couple of dummy frames."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - init is trivial
        self.batch_size = int(kwargs.get("batch_size", 1) or 1)
        self._len = 2
        self._closed = False
        self._fps = 30.0

    @property
    def fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        for _ in range(self._len):
            # Simple uint8 frame tensor [B, H, W, C]
            yield torch.zeros((self.batch_size, 4, 4, 3), dtype=torch.uint8)

    def close(self) -> None:
        self._closed = True


def should_build_aspen_pipeline_for_stitching(monkeypatch, tmp_path):
    captured_net: Dict[str, _DummyAspenNet] = {}

    def _make_dummy_aspen(name: str, graph_cfg: Dict[str, Any], shared: Dict[str, Any] | None = None, **kwargs: Any):  # type: ignore[override]
        net = _DummyAspenNet(name=name, graph_cfg=graph_cfg, shared=shared, **kwargs)
        captured_net["net"] = net
        return net

    # Patch heavy dependencies in stitch_videos.
    monkeypatch.setattr(stitch_cli, "AspenNet", _make_dummy_aspen)
    monkeypatch.setattr(stitch_cli, "BasicVideoInfo", _DummyVideoInfo)
    monkeypatch.setattr(stitch_cli, "StitchDataset", _DummyStitchDataset)

    def _fake_configure_video_stitching(
        dir_name: str,
        video_left: str,
        video_right: str,
        project_file_name: str,
        left_frame_offset: int,
        right_frame_offset: int,
        base_frame_offset: int,
        max_control_points: int,
        force: bool,
    ):
        # Return a dummy project path and zero offsets.
        pto_path = str(tmp_path / "dummy.pto")
        Path(pto_path).touch()
        return pto_path, 0, 0

    monkeypatch.setattr(stitch_cli, "configure_video_stitching", _fake_configure_video_stitching)

    args = types.SimpleNamespace(
        profiler=None,
        max_blend_levels=None,
        skip_final_video_save=False,
        save_frame_dir=None,
        no_cuda_streams=True,
        config_overrides=["aspen.stitching.enabled=false"],
    )

    out_path = tmp_path / "stitched.mkv"

    lfo, rfo = stitch_cli.stitch_videos(
        dir_name=str(tmp_path),
        videos={"left": ["left.mp4"], "right": ["right.mp4"]},
        max_control_points=10,
        game_id="test-game",
        output_stitched_video_file=str(out_path),
        args=args,
    )

    # Offsets should come back from the fake configure_video_stitching.
    assert lfo == 0
    assert rfo == 0

    net = captured_net.get("net")
    assert isinstance(net, _DummyAspenNet)

    # Ensure the stitching Aspen graph has a video_out plugin with the
    # CLI-provided output path wired in.
    plugins = net.graph_cfg.get("plugins", {})
    assert "video_out" in plugins
    vo_spec = plugins["video_out"]
    params = vo_spec.get("params", {})
    assert params.get("output_video_path") == str(out_path)

    # --config-override should have been applied to the loaded Aspen config.
    # Disabling aspen.stitching.enabled should disable the StitchingPlugin trunk.
    assert plugins.get("stitching", {}).get("enabled") is False

    # At least one frame should have been forwarded into AspenNet with
    # img, frame_ids, data.fps and game_id populated.
    assert net.calls, "Expected AspenNet.forward to be called at least once"
    ctx0 = net.calls[0]
    assert "img" in ctx0
    assert "frame_ids" in ctx0
    assert ctx0.get("game_id") == "test-game"
    data = ctx0.get("data") or {}
    assert data.get("fps") == 30.0

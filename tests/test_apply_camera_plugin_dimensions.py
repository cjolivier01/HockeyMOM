from __future__ import annotations

import sys
import types

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]


requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


def _install_mmcv_transforms_stub(monkeypatch) -> None:
    transforms_module = types.ModuleType("mmcv.transforms")

    class _Compose:
        def __init__(self, pipeline):
            self._pipeline = pipeline or []

        def __iter__(self):
            return iter(())

        def __call__(self, data):
            return data

    transforms_module.Compose = _Compose

    mmcv_module = sys.modules.get("mmcv")
    if mmcv_module is None:
        mmcv_module = types.ModuleType("mmcv")
    monkeypatch.setattr(mmcv_module, "transforms", transforms_module, raising=False)
    monkeypatch.setitem(sys.modules, "mmcv", mmcv_module)
    monkeypatch.setitem(sys.modules, "mmcv.transforms", transforms_module)


@requires_torch
def should_report_actual_frame_size_when_full_panorama_is_not_cropped(monkeypatch):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin
    from hmlib.utils.gpu import unwrap_tensor

    plugin = ApplyCameraPlugin(
        video_out_pipeline=None,
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)

    out = plugin({"img": img, "shared": {}})

    out_img = unwrap_tensor(out["img"])
    assert tuple(out_img.shape) == (1, 9, 15, 3)
    assert out["video_frame_cfg"]["output_frame_width"] == 15
    assert out["video_frame_cfg"]["output_frame_height"] == 9


@requires_torch
def should_clamp_apply_camera_output_width_from_game_config(monkeypatch):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin

    plugin = ApplyCameraPlugin(
        video_out_pipeline=None,
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)
    context = {"img": img, "shared": {"game_config": {"video_out": {"output_width": 10}}}}

    plugin._ensure_initialized(context)

    assert plugin._video_frame_cfg is not None
    assert plugin._video_frame_cfg["output_frame_width"] == 10
    assert plugin._video_frame_cfg["output_frame_height"] == 6


@requires_torch
def should_clamp_apply_camera_output_height_from_game_config(monkeypatch):
    _install_mmcv_transforms_stub(monkeypatch)
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin

    plugin = ApplyCameraPlugin(
        video_out_pipeline=None,
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)
    context = {"img": img, "shared": {"game_config": {"video_out": {"output_height": 6}}}}

    plugin._ensure_initialized(context)

    assert plugin._video_frame_cfg is not None
    assert plugin._video_frame_cfg["output_frame_width"] == 10
    assert plugin._video_frame_cfg["output_frame_height"] == 6

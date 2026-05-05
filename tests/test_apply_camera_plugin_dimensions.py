from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]


requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


@requires_torch
def should_report_actual_frame_size_when_full_panorama_is_not_cropped():
    import hmlib.transforms  # noqa: F401 - register HmMakeVisibleImage
    from hmlib.camera.apply_camera_plugin import ApplyCameraPlugin
    from hmlib.utils.gpu import unwrap_tensor

    plugin = ApplyCameraPlugin(
        video_out_pipeline=[{"type": "HmMakeVisibleImage"}],
        crop_output_image=False,
        crop_play_box=False,
    )
    img = torch.zeros((1, 9, 15, 3), dtype=torch.uint8)

    out = plugin({"img": img, "shared": {}})

    out_img = unwrap_tensor(out["img"])
    assert tuple(out_img.shape) == (1, 9, 15, 3)
    assert out["video_frame_cfg"]["output_frame_width"] == 15
    assert out["video_frame_cfg"]["output_frame_height"] == 9

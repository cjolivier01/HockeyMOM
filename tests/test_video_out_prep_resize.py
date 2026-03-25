from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]


requires_torch = pytest.mark.skipif(torch is None, reason="requires torch")


@requires_torch
def should_use_bicubic_antialias_when_video_out_prep_downscales(monkeypatch):
    from hmlib.video import video_out_prep as prep_mod

    preparer = prep_mod.VideoOutputPreparer(
        skip_final_save=True,
        device="cpu",
        output_width=10,
        output_height=6,
    )

    img = torch.arange(1 * 12 * 20 * 3, dtype=torch.uint8).reshape(1, 12, 20, 3)
    results = {
        "img": img,
        "video_frame_cfg": {
            "output_frame_width": 20,
            "output_frame_height": 12,
            "output_aspect_ratio": 20.0 / 12.0,
        },
    }

    calls: list[dict[str, object]] = []
    original_resize_image = prep_mod.resize_image

    def _spy_resize_image(*, img, new_width, new_height, mode=None, antialias=True, **kwargs):
        calls.append(
            {
                "new_width": int(new_width),
                "new_height": int(new_height),
                "mode": mode,
                "antialias": bool(antialias),
            }
        )
        return original_resize_image(
            img=img,
            new_width=new_width,
            new_height=new_height,
            mode=mode,
            antialias=antialias,
            **kwargs,
        )

    monkeypatch.setattr(prep_mod, "resize_image", _spy_resize_image)

    prepared = preparer.prepare_results(results)

    assert len(calls) == 1
    assert calls[0]["new_width"] == 10
    assert calls[0]["new_height"] == 6
    assert calls[0]["mode"] == "bicubic"
    assert calls[0]["antialias"] is True
    assert tuple(prepared["img"].shape) == (1, 6, 10, 3)

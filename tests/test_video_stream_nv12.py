import pytest
import torch

from hmlib.video.video_stream import _nv12_to_bgr


def _planes(y_value: int, u_value: int, v_value: int):
    y = torch.full((2, 2, 1), y_value, dtype=torch.uint8)
    uv = torch.tensor([[[u_value, v_value]]], dtype=torch.uint8)
    return [y, uv]


def should_convert_full_range_bt709_nv12_to_bgr():
    result = _nv12_to_bgr(
        _planes(100, 90, 240),
        color_range="pc",
        color_space="bt709",
    )

    assert result.shape == (3, 2, 2)
    assert result[:, 0, 0].tolist() == [29, 55, 255]


def should_convert_limited_range_bt709_nv12_to_bgr():
    result = _nv12_to_bgr(
        _planes(100, 90, 240),
        color_range="tv",
        color_space="bt709",
    )

    assert result[:, 0, 0].tolist() == [18, 46, 255]


def should_infer_full_range_from_yuvj_pixel_format():
    result = _nv12_to_bgr(
        _planes(255, 128, 128),
        pixel_format="yuvj420p",
    )

    assert torch.all(result == 255)


def should_reject_non_nv12_planes():
    with pytest.raises(ValueError, match="expected 2 planes"):
        _nv12_to_bgr([torch.zeros((2, 2), dtype=torch.uint8)])

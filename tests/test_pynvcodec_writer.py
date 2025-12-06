import os
from pathlib import Path

import pytest
import torch

from hmlib.video.video_stream import PyNvVideoCodecWriter


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for PyNvVideoCodec")
def should_write_raw_h265_with_pynvcodec(tmp_path: Path):
    filename = tmp_path / "pynvcodec_raw"
    writer = PyNvVideoCodecWriter(
        filename=str(filename),
        fps=30.0,
        width=640,
        height=360,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=1,
    )
    writer.open()

    frame = torch.randint(
        0,
        256,
        (1, 360, 640, 3),
        dtype=torch.uint8,
        device=writer.device,
    )
    writer.write(frame)
    writer.close()

    raw_path = tmp_path / "pynvcodec_raw.h265"
    assert raw_path.is_file()
    assert raw_path.stat().st_size > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required for PyNvVideoCodec")
def should_optionally_produce_mkv_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    filename = tmp_path / "pynvcodec_mkv"
    monkeypatch.setenv("HM_VIDEO_ENCODER_CONTAINER", "mkv")

    writer = PyNvVideoCodecWriter(
        filename=str(filename),
        fps=30.0,
        width=640,
        height=360,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=1,
    )
    writer.open()

    frame = torch.randint(
        0,
        256,
        (1, 360, 640, 3),
        dtype=torch.uint8,
        device=writer.device,
    )
    writer.write(frame)
    writer.close()

    raw_path = tmp_path / "pynvcodec_mkv.h265"
    mkv_path = tmp_path / "pynvcodec_mkv.mkv"

    assert raw_path.is_file()
    assert raw_path.stat().st_size > 0
    assert mkv_path.is_file()
    assert mkv_path.stat().st_size > 0

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def _require_rocm_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA device not available")
    if not getattr(torch.version, "hip", None):
        pytest.skip("PyAmdVideoCodec tests require a ROCm torch runtime")
    _ensure_repo_on_path()
    return torch


def _require_torch():
    torch = pytest.importorskip("torch")
    _ensure_repo_on_path()
    return torch


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for PyAmdVideoCodec tests")
    return ffmpeg


def _ffprobe_stream_metadata(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,width,height,duration",
        "-of",
        "json",
        str(video_path),
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(raw.decode("utf-8"))
    assert data.get("streams"), f"ffprobe did not return streams for {video_path}"
    return data["streams"][0]


def _write_test_video_cv2(path: Path, *, fps: float, width: int, height: int, frames: int) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        x = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(x, (height, 1))
        frame = np.stack(
            [
                gradient,
                np.full_like(gradient, i * 10 % 255),
                255 - gradient,
            ],
            axis=2,
        )
        writer.write(frame)
    writer.release()


def should_write_h264_with_pyamdencoder(tmp_path: Path):
    torch = _require_rocm_torch()
    from hmlib.video.py_amd_codec import PyAmdVideoCodec, PyAmdVideoEncoder

    if not PyAmdVideoCodec.is_encoder_available("h264"):
        pytest.skip("AMD H.264 encoder backend is not available")

    filename = tmp_path / "pyamdencoder_out.mp4"
    encoder = PyAmdVideoEncoder(
        output_path=str(filename),
        width=320,
        height=240,
        fps=15.0,
        codec="h264",
        device=torch.device("cuda", 0),
        bitrate=int(5e6),
    )
    encoder.open()
    frames = torch.randint(
        0,
        256,
        (4, 240, 320, 3),
        dtype=torch.uint8,
        device="cuda:0",
    )
    encoder.write(frames, frame_ids=torch.tensor([[1], [2], [3], [4]], device="cuda:0"))
    encoder.close()

    assert filename.is_file()
    assert filename.stat().st_size > 0

    meta = _ffprobe_stream_metadata(filename)
    assert int(meta["width"]) == 320
    assert int(meta["height"]) == 240
    assert int(meta.get("nb_frames") or meta.get("nb_read_frames") or 0) == 4


def should_preserve_rtmp_output_url_in_encoder_cmd(monkeypatch):
    torch = _require_torch()
    from hmlib.video import py_amd_codec as amd_codec

    monkeypatch.setattr(amd_codec, "_require_ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        amd_codec.PyAmdVideoCodec,
        "resolve_encoder",
        classmethod(lambda cls, codec, backend="auto": "h264_vaapi"),
    )
    monkeypatch.setattr(
        amd_codec.PyAmdVideoCodec,
        "default_vaapi_device",
        classmethod(lambda cls: "/dev/dri/renderD128"),
    )

    output_url = "rtmp://127.0.0.1:1935/live/test-stream"
    encoder = amd_codec.PyAmdVideoEncoder(
        output_path=output_url,
        width=128,
        height=96,
        fps=10.0,
        codec="h264",
        device=torch.device("cpu"),
        backend="vaapi",
    )

    cmd = encoder._build_cmd()
    assert cmd[-1] == output_url
    assert "rtmp:/127.0.0.1" not in cmd[-1]


def should_report_decoder_unavailable_without_ffmpeg(monkeypatch):
    _ensure_repo_on_path()
    from hmlib.video import py_amd_codec as amd_codec

    monkeypatch.setattr(amd_codec.shutil, "which", lambda name: None)
    monkeypatch.setattr(amd_codec, "_FFMPEG_ENCODERS", None)
    monkeypatch.setattr(amd_codec, "_FFMPEG_HWACCELS", None)

    assert amd_codec._ffmpeg_path() is None
    assert not amd_codec.PyAmdVideoCodec.is_decoder_available("software")
    with pytest.raises(RuntimeError, match="FFmpeg"):
        amd_codec._require_ffmpeg_path()


def should_stream_h264_bitstream_with_pyamdencoder():
    torch = _require_rocm_torch()
    from hmlib.video.py_amd_codec import PyAmdVideoCodec, PyAmdVideoEncoder

    if not PyAmdVideoCodec.is_encoder_available("h264"):
        pytest.skip("AMD H.264 encoder backend is not available")

    payloads: list[bytes] = []
    encoder = PyAmdVideoEncoder(
        output_path=None,
        width=128,
        height=96,
        fps=10.0,
        codec="h264",
        device=torch.device("cuda", 0),
        bitrate=int(2e6),
        bitstream_handler=payloads.append,
    )
    encoder.open()
    frame = torch.randint(
        0,
        256,
        (2, 96, 128, 3),
        dtype=torch.uint8,
        device="cuda:0",
    )
    encoder.write(frame, frame_ids=torch.tensor([1, 2], device="cuda:0"))
    encoder.close()

    assert payloads
    assert sum(len(chunk) for chunk in payloads) > 0


def should_seek_to_exact_frame_with_software_decoder(tmp_path: Path):
    _require_ffmpeg()
    torch = _require_torch()
    cv2 = pytest.importorskip("cv2")
    del cv2

    from hmlib.video.py_amd_codec import PyAmdVideoCodec, PyAmdVideoDecoder

    if not PyAmdVideoCodec.is_decoder_available("software"):
        pytest.skip("FFmpeg software decoder is not available")

    src = tmp_path / "seek_source.mp4"
    _write_test_video_cv2(src, fps=10.0, width=64, height=48, frames=12)

    decoder = PyAmdVideoDecoder(
        src,
        device=torch.device("cpu"),
        backend="software",
        batch_size=1,
    )
    try:
        decoder.seek_to_index(7)
        batch = decoder.read_batch(1)
        assert batch is not None
        frame = batch[0]
        green_mean = float(frame[1].to(torch.float32).mean().item())
        assert green_mean == pytest.approx(70.0, abs=12.0)

        batch = decoder.read_batch(1)
        assert batch is not None
        next_frame = batch[0]
        next_green_mean = float(next_frame[1].to(torch.float32).mean().item())
        assert next_green_mean == pytest.approx(80.0, abs=12.0)
    finally:
        decoder.close()


def should_transcode_with_pyamd_decoder_and_writer(tmp_path: Path):
    torch = _require_rocm_torch()
    from hmlib.utils.gpu import unwrap_tensor
    from hmlib.video.py_amd_codec import PyAmdVideoCodec, PyAmdVideoEncoder
    from hmlib.video.video_stream import VideoStreamReader, create_output_video_stream

    if not PyAmdVideoCodec.is_decoder_available():
        pytest.skip("AMD decoder backend is not available")
    if not PyAmdVideoCodec.is_encoder_available("h264"):
        pytest.skip("AMD H.264 encoder backend is not available")

    fps = 12.0
    frame_count = 6
    width, height = 160, 120
    src = tmp_path / "source.mp4"
    dst = tmp_path / "transcoded.mp4"
    source_frames = torch.randint(
        0,
        256,
        (frame_count, height, width, 3),
        dtype=torch.uint8,
        device="cuda:0",
    )
    source_encoder = PyAmdVideoEncoder(
        output_path=str(src),
        width=width,
        height=height,
        fps=fps,
        codec="h264",
        device=torch.device("cuda", 0),
        bitrate=int(4e6),
    )
    source_encoder.open()
    source_encoder.write(source_frames, frame_ids=torch.arange(1, frame_count + 1, device="cuda:0"))
    source_encoder.close()

    reader = VideoStreamReader(
        str(src),
        type="pyamdcodec",
        batch_size=2,
        device=torch.device("cuda", 0),
    )
    writer = create_output_video_stream(
        filename=str(dst),
        fps=Fraction(int(fps), 1),
        width=width,
        height=height,
        codec="h264_vaapi",
        device=torch.device("cuda", 0),
        bit_rate=int(4e6),
        batch_size=2,
    )
    next_frame_id = 1
    try:
        for batch in reader:
            batch_t = unwrap_tensor(batch)
            batch_hwc = batch_t.permute(0, 2, 3, 1).contiguous()
            batch_size = int(batch_hwc.shape[0])
            frame_ids = torch.arange(
                next_frame_id,
                next_frame_id + batch_size,
                dtype=torch.int64,
                device=batch_hwc.device,
            )
            writer.write(batch_hwc, frame_ids=frame_ids)
            next_frame_id += batch_size
    finally:
        reader.close()
        writer.close()

    src_meta = _ffprobe_stream_metadata(src)
    dst_meta = _ffprobe_stream_metadata(dst)

    def _rate_to_float(rate: str) -> float:
        return float(Fraction(rate)) if rate else 0.0

    def _frame_count(meta: dict) -> int:
        return int(meta.get("nb_frames") or meta.get("nb_read_frames") or 0)

    assert _frame_count(src_meta) == frame_count
    assert _frame_count(dst_meta) == frame_count
    assert int(dst_meta["width"]) == width
    assert int(dst_meta["height"]) == height
    assert _rate_to_float(dst_meta.get("r_frame_rate")) == pytest.approx(fps, rel=0, abs=1e-3)

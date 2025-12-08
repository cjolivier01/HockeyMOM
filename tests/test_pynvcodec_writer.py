import os
import sys
from pathlib import Path

import av  # type: ignore[import-not-found]
import pytest
import torch


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def _ensure_hockeymom_ext(repo_root: Path) -> None:
    """
    When running tests via plain pytest (outside Bazel), prefer the Bazel-built
    hockeymom extension if available so that new symbols are present.
    """
    ext_key = "hockeymom._hockeymom"
    if ext_key in sys.modules and hasattr(sys.modules[ext_key], "bgr_to_i420_cuda"):
        return

    ext_path = repo_root / "bazel-bin" / "hockeymom" / "src" / "_hockeymom.so"
    if not ext_path.is_file():
        return

    import importlib.util

    spec = importlib.util.spec_from_file_location(ext_key, ext_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules[ext_key] = module


_repo_root = _ensure_repo_on_path()
_ensure_hockeymom_ext(_repo_root)

from hmlib.video.video_stream import PyNvVideoEncoderWriter


def should_write_h265_with_pynvencoder(tmp_path: Path):
    """
    Basic smoke test that exercises the GPU-only PyNvVideoEncoderWriter path.
    """
    filename = tmp_path / "pynvencoder_raw.mp4"
    writer = PyNvVideoEncoderWriter(
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

    assert filename.is_file()
    assert filename.stat().st_size > 0


def should_support_mkv_container_with_pynvencoder(tmp_path: Path):
    """
    Verify that writing to a .mkv path produces a valid container.
    """
    filename = tmp_path / "pynvencoder_out.mkv"

    writer = PyNvVideoEncoderWriter(
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

    assert filename.is_file()
    assert filename.stat().st_size > 0


def should_use_pyav_backend_with_pynvencoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Ensure that the PyAV backend is exercised when HM_VIDEO_ENCODER_BACKEND=pyav.

    This test imports av at module level, so it will fail if PyAV is not installed.
    """
    monkeypatch.setenv("HM_VIDEO_ENCODER_BACKEND", "pyav")

    filename = tmp_path / "pynvencoder_pyav.mkv"
    writer = PyNvVideoEncoderWriter(
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

    assert filename.is_file()
    assert filename.stat().st_size > 0

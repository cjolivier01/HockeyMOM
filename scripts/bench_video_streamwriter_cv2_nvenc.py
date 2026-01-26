import sys
import os
import time
from pathlib import Path

import numpy as np

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmlib.video.video_stream import VideoStreamWriterCV2


def _run_cpu(out_path: Path, frames: int, width: int, height: int) -> float:
    cpu_codec = "mp4v"
    if width >= 4096:
        # MPEG-4 Part 2 (mp4v) fails at very large dimensions.
        # Use HEVC for 8K-class sizes (also matches NVENC preference below).
        cpu_codec = "HEVC"
    writer = VideoStreamWriterCV2(
        filename=str(out_path),
        fps=30.0,
        width=width,
        height=height,
        device=torch.device("cpu"),
        codec=cpu_codec,
        bit_rate=int(8e6),
    )
    assert writer.isOpened()
    assert writer.backend == "opencv_video_writer"

    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(frames):
        writer.write(frame)
    writer.close()
    t1 = time.perf_counter()
    return frames / max(1e-9, t1 - t0)


def _run_nvenc(out_path: Path, frames: int, width: int, height: int) -> float:
    nv_codec = "mp4v"
    if width >= 4096:
        # H.264 NVENC can fail at very large dimensions in some OpenCV builds;
        # HEVC is generally supported for 8K-class sizes.
        nv_codec = "HEVC"
    writer = VideoStreamWriterCV2(
        filename=str(out_path),
        fps=30.0,
        width=width,
        height=height,
        device=torch.device("cuda:0"),
        codec=nv_codec,
        bit_rate=int(8e6),
    )
    assert writer.isOpened()
    if writer.backend != "opencv_cudacodec_nvenc":
        raise RuntimeError(f"NVENC backend not selected (backend={writer.backend})")

    frame = torch.randint(0, 255, (height, width, 3), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(frames):
        writer.write(frame)
    writer.close()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return frames / max(1e-9, t1 - t0)


def _run_pynvenc(out_path: Path, frames: int, width: int, height: int) -> float:
    # Use the native PyNvVideoEncoder backend (hmlib.video.video_stream.PyNvVideoEncoderWriter).
    from hmlib.video.video_stream import create_output_video_stream

    stream = create_output_video_stream(
        filename=str(out_path),
        fps=30.0,
        width=width,
        height=height,
        codec="hevc_nvenc",
        device=torch.device("cuda:0"),
        bit_rate=int(8e6),
        batch_size=1,
        profiler=None,
    )
    assert stream.isOpened()

    frame = torch.randint(0, 255, (height, width, 3), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(frames):
        stream.write(frame)
    stream.close()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return frames / max(1e-9, t1 - t0)


def main() -> None:
    frames = int(os.environ.get("HM_BENCH_FRAMES", "300"))
    width = int(os.environ.get("HM_BENCH_WIDTH", "1280"))
    height = int(os.environ.get("HM_BENCH_HEIGHT", "720"))

    print("cv2 version:", cv2.__version__)
    print("opencv cudacodec:", hasattr(cv2, "cudacodec"))
    print("opencv cuda devices:", getattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)())
    print("torch cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    out_dir = Path("/tmp/hm_bench_video_writer")
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_out = out_dir / "cpu.mp4"
    nv_out = out_dir / "nvenc.mp4"
    py_nv_out_mp4 = out_dir / "pynvenc.mp4"
    py_nv_out_mkv = out_dir / "pynvenc.mkv"
    if cpu_out.exists():
        cpu_out.unlink()
    if nv_out.exists():
        nv_out.unlink()
    if py_nv_out_mp4.exists():
        py_nv_out_mp4.unlink()
    if py_nv_out_mkv.exists():
        py_nv_out_mkv.unlink()

    cpu_fps = _run_cpu(cpu_out, frames=frames, width=width, height=height)
    print(f"CPU OpenCV VideoWriter: {cpu_fps:.2f} fps")

    if (
        torch.cuda.is_available()
        and hasattr(cv2, "cudacodec")
        and hasattr(cv2, "cuda")
        and int(cv2.cuda.getCudaEnabledDeviceCount()) > 0
    ):
        nv_fps = _run_nvenc(nv_out, frames=frames, width=width, height=height)
        print(f"OpenCV cudacodec NVENC: {nv_fps:.2f} fps")
        if cpu_fps > 0:
            print(f"Speedup: {nv_fps / cpu_fps:.2f}x")
    else:
        print("NVENC path not available; skipping.")

    try:
        py_fps = None
        try:
            py_fps = _run_pynvenc(py_nv_out_mp4, frames=frames, width=width, height=height)
            print(f"PyNvVideoEncoder (hevc_nvenc) mp4: {py_fps:.2f} fps")
        except Exception:
            py_fps = _run_pynvenc(py_nv_out_mkv, frames=frames, width=width, height=height)
            print(f"PyNvVideoEncoder (hevc_nvenc) mkv: {py_fps:.2f} fps")
        if py_fps is not None and cpu_fps > 0:
            print(f"PyNv speedup vs CPU: {py_fps / cpu_fps:.2f}x")
    except Exception as exc:
        print(f"PyNvVideoEncoder benchmark unavailable: {exc}")


if __name__ == "__main__":
    main()

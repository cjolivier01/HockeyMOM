import os

import pytest

try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_CV2 or not _HAS_TORCH, reason="cv2/torch not available")
def should_use_nvenc_when_available(tmp_path):
    import cv2
    import cupy  # noqa: F401
    import torch

    if not hasattr(cv2, "cudacodec") or not hasattr(cv2, "cuda"):
        pytest.skip("OpenCV cudacodec/cuda not available")
    if int(cv2.cuda.getCudaEnabledDeviceCount()) <= 0:
        pytest.skip("No CUDA devices available to OpenCV")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")

    out_path = tmp_path / "nvenc.mp4"
    from hmlib.video.video_stream import VideoStreamWriterCV2

    writer = VideoStreamWriterCV2(
        filename=str(out_path),
        fps=30.0,
        width=320,
        height=240,
        device=torch.device("cuda:0"),
        codec="mp4v",
        bit_rate=int(2e6),
    )
    assert writer.isOpened()
    assert getattr(writer, "backend") == "opencv_cudacodec_nvenc"

    frame = torch.zeros((240, 320, 3), dtype=torch.uint8, device="cuda")
    writer.write(frame)
    assert getattr(writer, "backend_transfer") in {"cupy_d2d", "cpu_upload"}
    writer.close()

    assert out_path.exists()
    assert os.path.getsize(out_path) > 0


@pytest.mark.skipif(not _HAS_CV2 or not _HAS_TORCH, reason="cv2/torch not available")
def should_use_cupy_d2d_when_available(tmp_path):
    import cv2
    import torch

    try:
        import cupy  # noqa: F401
    except Exception:
        pytest.skip("CuPy not available")

    if not hasattr(cv2, "cudacodec") or not hasattr(cv2, "cuda"):
        pytest.skip("OpenCV cudacodec/cuda not available")
    if int(cv2.cuda.getCudaEnabledDeviceCount()) <= 0:
        pytest.skip("No CUDA devices available to OpenCV")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA not available")

    out_path = tmp_path / "nvenc_cupy.mp4"
    from hmlib.video.video_stream import VideoStreamWriterCV2

    writer = VideoStreamWriterCV2(
        filename=str(out_path),
        fps=30.0,
        width=320,
        height=240,
        device=torch.device("cuda:0"),
        codec="mp4v",
        bit_rate=int(2e6),
    )
    assert writer.isOpened()
    if getattr(writer, "backend") != "opencv_cudacodec_nvenc":
        pytest.skip("NVENC backend not selected")

    frame = torch.zeros((240, 320, 3), dtype=torch.uint8, device="cuda")
    writer.write(frame)
    assert getattr(writer, "backend_transfer") == "cupy_d2d"
    writer.close()


@pytest.mark.skipif(not _HAS_CV2 or not _HAS_TORCH, reason="cv2/torch not available")
def should_fallback_to_cpu_writer(tmp_path):
    import torch

    out_path = tmp_path / "cpu.mp4"
    from hmlib.video.video_stream import VideoStreamWriterCV2

    writer = VideoStreamWriterCV2(
        filename=str(out_path),
        fps=30.0,
        width=160,
        height=120,
        device=torch.device("cpu"),
        codec="mp4v",
        bit_rate=int(1e6),
    )
    assert writer.isOpened()
    assert getattr(writer, "backend") == "opencv_video_writer"

    frame = torch.zeros((120, 160, 3), dtype=torch.uint8, device="cpu")
    writer.write(frame)
    writer.close()

    assert out_path.exists()
    assert os.path.getsize(out_path) > 0

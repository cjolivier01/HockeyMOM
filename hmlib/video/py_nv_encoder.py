from __future__ import annotations

import contextlib
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Union

import torch
from typeguard import typechecked

from hockeymom import bgr_to_i420_cuda

try:
    import PyNvVideoCodec as nvc  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    nvc = None  # type: ignore[assignment]


class _DLPackFrame:
    """
    Lightweight wrapper exposing a torch CUDA tensor via the DLPack protocol.

    This exists to adapt to PyNvEncoder's expectation of calling
    __dlpack__(consumer_stream) with a positional argument, while newer
    PyTorch versions require keyword-only parameters. We ignore the
    consumer stream and delegate to tensor.__dlpack__().
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def cuda(self) -> "_DLPackFrame":
        # PyNvEncoder's Encode() checks for a .cuda() method and calls it;
        # returning self keeps everything on the GPU while still allowing
        # the encoder to discover the DLPack interface on this wrapper.
        return self

    def __dlpack__(self, *args, **kwargs):
        # Ignore consumer_stream argument; rely on PyTorch defaults.
        return self._tensor.__dlpack__()

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


class PyNvVideoEncoder:
    """
    High-level GPU-only video encoder backed by PyNvVideoCodec.

    - Accepts BGR torch.Tensors on CUDA (single frame or batch).
    - Uses jetson-utils (via hockeymom.bgr_to_i420_cuda) to convert
      BGR -> planar YUV420 (I420) entirely on the GPU.
    - Feeds YUV420 frames to PyNvVideoCodec's NVENC bindings using CUDA
      memory through the DLPack interface (no CPU copies of raw frames).
    - Streams the elementary bitstream to ffmpeg for container muxing
      (MP4/MKV/etc., based on the output file extension).
    """
    @typechecked
    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        codec: str = "h265",
        preset: str = "P1",
        device: Optional[torch.device] = None,
        gpu_id: Optional[int] = None,
        cuda_context: Optional[int] = None,
        cuda_stream: Optional[int] = None,
        bitrate: Optional[int] = None,
        use_pyav: Optional[bool] = None,
        profiler: Optional[Any] = None,
    ) -> None:
        if nvc is None:
            raise ImportError(
                "PyNvVideoEncoder requires the PyNvVideoCodec package "
                "(import PyNvVideoCodec failed)."
            )

        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec.lower()
        self.preset = preset.upper()

        if device is None:
            device = torch.device("cuda", 0)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("PyNvVideoEncoder requires a CUDA device.")
        self.device = device

        self.gpu_id = int(gpu_id if gpu_id is not None else (device.index or 0))
        self.cuda_context = cuda_context
        self.cuda_stream = cuda_stream
        self.bitrate = int(bitrate) if bitrate is not None else None
        # Optional HmProfiler instance injected by callers.
        self._profiler: Optional[Any] = profiler

        backend_env = os.environ.get("HM_VIDEO_ENCODER_BACKEND", "").lower()
        if use_pyav is not None:
            # Explicit caller override wins.
            self._use_pyav = bool(use_pyav)
        elif backend_env:
            # Environment variable controls backend when set.
            self._use_pyav = backend_env == "pyav"
        else:
            # Default to ffmpeg CLI backend when no override is provided.
            self._use_pyav = True

        if self.width % 2 or self.height % 2:
            raise ValueError("Width and height must be even for YUV420 encoding.")

        self._encoder: Optional[nvc.PyNvEncoder] = None  # type: ignore[assignment]
        self._ffmpeg_proc: Optional[subprocess.Popen[bytes]] = None
        self._av_container = None
        self._av_stream = None
        self._next_pts: int = 0
        self._opened = False
        self._frames_in_current_bitstream: int = 0
        self._frame_duration_units: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialize NVENC encoder and container muxer."""
        if self._opened:
            return

        self._encoder = self._create_encoder()
        if self._use_pyav:
            self._open_pyav_container()
        else:
            self._ffmpeg_proc = self._spawn_ffmpeg()
        self._opened = True

    def write(self, frames: torch.Tensor) -> None:
        """
        Encode one or more BGR frames and append them to the output stream.

        Args:
            frames: torch.Tensor with shape:
                - [H, W, 3]
                - [N, H, W, 3]
                - [3, H, W]
                - [N, 3, H, W]
              Values are interpreted as 8-bit BGR (0–255) or floats in
              [0, 1] or [0, 255]. All processing stays on CUDA.
        """
        if not self._opened:
            raise RuntimeError("Encoder is not open. Call open() before write().")

        if self._encoder is None:
            raise RuntimeError("Encoder is not properly initialized.")

        batch = self._normalize_frames(frames)
        prof = self._profiler
        batch_ctx = (
            prof.rf("video.nvenc.write_batch")  # type: ignore[union-attr]
            if getattr(prof, "enabled", False)
            else contextlib.nullcontext()
        )
        with batch_ctx:
            for frame in batch:
                frame_ctx = (
                    prof.rf("video.nvenc.encode_frame")  # type: ignore[union-attr]
                    if getattr(prof, "enabled", False)
                    else contextlib.nullcontext()
                )
                with frame_ctx:
                    yuv420 = self._bgr_to_yuv420(frame)
                    # yuv420 is a 2D CUDA tensor with shape [H*3/2, W], uint8.
                    bitstream = self._encoder.Encode(yuv420)  # type: ignore[union-attr]
                    self._frames_in_current_bitstream += 1
                    if bitstream:
                        if self._use_pyav:
                            self._mux_packet_pyav(bitstream)
                            self._frames_in_current_bitstream = 0
                        else:
                            assert (
                                self._ffmpeg_proc is not None
                                and self._ffmpeg_proc.stdin is not None
                            )
                            try:
                                self._ffmpeg_proc.stdin.write(bytearray(bitstream))
                            except BrokenPipeError as exc:
                                rc = None
                                stderr_output = ""
                                try:
                                    rc = self._ffmpeg_proc.poll()
                                    out, err = self._ffmpeg_proc.communicate(timeout=0.1)
                                    if err:
                                        stderr_output = err.decode("utf-8", errors="ignore")
                                except Exception:
                                    pass
                                raise RuntimeError(
                                    "ffmpeg muxer exited unexpectedly while writing NVENC bitstream. "
                                    "Check that codec/format settings are valid and disk space is available. "
                                    "See ffmpeg logs for more details. "
                                    f"(returncode={rc}, stderr={stderr_output!r})"
                                ) from exc

    def close(self) -> None:
        """Flush pending frames, finalize container, and release resources."""
        if not self._opened:
            return

        if self._encoder is not None:
            # Flush encoder
            bitstream = self._encoder.EndEncode()  # type: ignore[union-attr]
            if bitstream:
                if self._use_pyav:
                    self._mux_packet_pyav(bitstream)
                elif self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin is not None:
                    self._ffmpeg_proc.stdin.write(bytearray(bitstream))

        if not self._use_pyav:
            if self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin is not None:
                # Close ffmpeg stdin and wait for it to finish writing the container
                self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait()
        else:
            if self._av_container is not None:
                self._av_container.close()

        self._encoder = None
        self._ffmpeg_proc = None
        self._av_container = None
        self._av_stream = None
        self._opened = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_encoder(self) -> nvc.PyNvEncoder:  # type: ignore[override]
        """
        Create a PyNvEncoder configured for YUV420 (I420) input and the requested codec.

        The encoder expects CUDA-accessible YUV420 surfaces; we provide them
        via torch tensors implementing __dlpack__/__cuda_array_interface__.
        """
        config: dict[str, str] = {
            "codec": self.codec,
            "preset": self.preset,
            "fps": str(self.fps),
            "gpu_id": str(self.gpu_id),
        }

        if self.cuda_context is not None:
            config["cudacontext"] = int(self.cuda_context)
        if self.cuda_stream is not None:
            config["cudastream"] = int(self.cuda_stream)
        if self.bitrate is not None:
            config["bitrate"] = str(self.bitrate)

        # YUV420 (I420) is efficient for NVENC and maps to yuv420p in the container.
        # Set usecpuinputbuffer=False to keep frames on CUDA.
        return nvc.CreateEncoder(self.width, self.height, "YUV420", False, **config)

    def _spawn_ffmpeg(self) -> subprocess.Popen[bytes]:
        """
        Launch ffmpeg to remux an elementary bitstream into a container.

        No re-encoding is done; ffmpeg simply copies the video stream into
        the requested container format based on the output file extension.
        """
        import ctypes
        import signal
        from shutil import which

        ffmpeg = which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for container muxing but was not found in PATH."
            )

        if self.codec == "h264":
            stream_format = "h264"
        elif self.codec in ("hevc", "h265"):
            stream_format = "hevc"
        elif self.codec == "av1":
            stream_format = "av1"
        else:
            raise ValueError(f"Unsupported codec for muxing: {self.codec}")

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-r",
            str(self.fps),
            "-f",
            stream_format,
            "-i",
            "pipe:0",
            "-c:v",
            "libx265" if self.codec in ("hevc", "h265") else "libx264",
            "-preset",
            "medium",
            str(self.output_path),
        ]

        kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        if os.name == "posix":

            def _set_pdeathsig() -> None:
                try:
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    PR_SET_PDEATHSIG = 1
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:
                    # Best-effort; keep encoder usable even if prctl fails.
                    pass

            kwargs["preexec_fn"] = _set_pdeathsig

        proc: subprocess.Popen[bytes] = subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]
        return proc

    def _open_pyav_container(self) -> None:
        import av

        self._av_container = av.open(str(self.output_path), mode="w")

        fps = Fraction(int(round(self.fps * 1001)), 1001)

        codec_name = "hevc" if self.codec in ("h265", "hevc") else self.codec
        self._av_stream = self._av_container.add_stream(codec_name, rate=fps)
        self._av_stream.width = self.width
        self._av_stream.height = self.height
        self._av_stream.pix_fmt = "yuv420p"

        # Give muxer a sensible hint (optional, but helps keep things stable).
        self._av_stream.time_base = Fraction(1, 90000)

        # IMPORTANT: finalize header so time_base is final
        self._av_container.start_encoding()

        tb = self._av_stream.time_base  # may have changed after start_encoding()
        self._ticks_per_frame = int(round((Fraction(1, 1) / fps) / tb))

        self._next_pts = 0  # in tb ticks
        self._frames_in_current_bitstream = 0

    def _mux_packet_pyav(self, bitstream: bytes) -> None:
        import av

        dur = int(self._frames_in_current_bitstream * self._ticks_per_frame)

        packet = av.packet.Packet(bitstream)
        packet.stream = self._av_stream
        packet.time_base = self._av_stream.time_base
        packet.pts = self._next_pts
        packet.dts = self._next_pts
        packet.duration = dur

        self._av_container.mux_one(packet)

        self._next_pts += dur

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to a CUDA tensor with shape [N, H, W, 3] and dtype uint8 (BGR).
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError("frames must be a torch.Tensor")

        # Move to the requested GPU and drop gradients; no CPU round-trip.
        frames = frames.to(device=self.device, non_blocking=True).detach()

        if frames.ndim == 3:
            # HWC or CHW
            if frames.shape[0] == 3 and frames.shape[-1] != 3:
                # CHW -> HWC
                frames = frames.permute(1, 2, 0)
            frames = frames.unsqueeze(0)
        elif frames.ndim == 4:
            # NHWC or NCHW
            if frames.shape[1] == 3 and frames.shape[-1] != 3:
                # NCHW -> NHWC
                frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError("frames must have 3 or 4 dimensions")

        if frames.shape[-1] != 3:
            raise ValueError("Last dimension must be 3 (BGR channels)")

        n, h, w, _c = frames.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"Expected frames of shape (*, {self.height}, {self.width}, 3), "
                f"got {tuple(frames.shape)}"
            )

        # Normalize dtype to uint8 in [0, 255]
        if frames.dtype.is_floating_point:
            max_val = frames.max()
            if float(max_val) <= 1.0:
                frames = frames * 255.0
            frames = frames.clamp(0, 255).to(torch.uint8)
        elif frames.dtype != torch.uint8:
            frames = frames.to(torch.uint8)

        return frames.contiguous()

    def _bgr_to_yuv420(self, frame: torch.Tensor) -> _DLPackFrame:
        """
        Convert a single BGR frame (H, W, 3) in uint8 (CUDA) to planar YUV420 (I420)
        layout suitable for PyNvEncoder when configured with format \"YUV420\".

        The resulting tensor has shape [H * 3 / 2, W] on CUDA.
        """
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Expected BGR frame with shape (H, W, 3)")

        h, w, _ = frame.shape
        if h != self.height or w != self.width:
            raise ValueError(
                f"Frame size mismatch: expected {self.width}x{self.height}, got {w}x{h}"
            )

        if frame.device != self.device:
            frame = frame.to(self.device, non_blocking=True)
        if frame.dtype != torch.uint8:
            frame = frame.clamp(0, 255).to(torch.uint8)

        frame = frame.contiguous()

        # Delegate BGR -> I420 conversion to jetson-utils via hockeymom binding.
        yuv420 = bgr_to_i420_cuda(frame)

        if yuv420.dim() != 2 or yuv420.size(0) != h * 3 // 2 or yuv420.size(1) != w:
            raise RuntimeError(
                "bgr_to_i420_cuda returned unexpected shape "
                f"{tuple(yuv420.shape)} for input {h}x{w}"
            )

        return _DLPackFrame(yuv420.contiguous())

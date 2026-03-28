"""Browser preview and live-stream helpers for headless environments."""

from __future__ import annotations

import html
import io
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from hmlib.ui.display_env import env_flag, has_local_display_env, sanitize_display_env_for_cv2

sanitize_display_env_for_cv2()

import numpy as np
import torch
from PIL import Image

from hmlib.log import get_root_logger
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.utils.image import make_visible_image


def has_local_display() -> bool:
    """Return True when an interactive display is available."""
    if env_flag("HM_FORCE_HEADLESS_PREVIEW"):
        return False
    return has_local_display_env()


def _best_public_host() -> Optional[str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
        finally:
            sock.close()
        if host and not host.startswith("127."):
            return host
    except OSError:
        return None
    return None


def _normalize_preview_frame(
    img: torch.Tensor | np.ndarray | StreamTensorBase,
    *,
    show_scaled: Optional[float] = None,
) -> np.ndarray:
    visible = make_visible_image(
        unwrap_tensor(img),
        enable_resizing=show_scaled,
        force_numpy=True,
    )
    if visible.ndim == 4:
        if visible.shape[0] != 1:
            raise ValueError(
                f"Expected a single frame for preview publishing, got batch shape={visible.shape}"
            )
        visible = visible[0]
    if visible.dtype != np.uint8:
        visible = visible.astype(np.uint8)
    return np.ascontiguousarray(visible)


class _PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, request_handler_class, preview: "BrowserPreviewServer"):
        super().__init__(server_address, request_handler_class)
        self.preview = preview


class _PreviewRequestHandler(BaseHTTPRequestHandler):
    server: _PreviewHttpServer

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("", "/"):
            self._serve_index()
            return
        if self.path == "/stream.mjpg":
            self._serve_stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def _serve_index(self) -> None:
        title = html.escape(self.server.preview.label)
        body = (
            "<!doctype html><html><head>"
            f"<title>{title}</title>"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<style>"
            "body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
            "header{padding:12px 16px;font-size:14px;background:#1a1a1a;}"
            "main{display:flex;justify-content:center;align-items:center;min-height:calc(100vh - 48px);}"
            "img{max-width:100vw;max-height:100vh;object-fit:contain;background:#000;}"
            "a{color:#9cd5ff;}"
            "</style>"
            "</head><body>"
            f'<header>{title} preview | <a href="/stream.mjpg">raw MJPEG</a></header>'
            '<main><img src="/stream.mjpg" alt="Live preview"></main>'
            "</body></html>"
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        boundary = b"frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=" + boundary.decode("ascii"),
        )
        self.end_headers()
        last_sequence = -1
        preview = self.server.preview
        while not preview.closed:
            frame = preview.wait_for_frame(last_sequence, timeout=1.0)
            if frame is None:
                continue
            jpeg_bytes, last_sequence = frame
            try:
                self.wfile.write(b"--" + boundary + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break


class BrowserPreviewServer:
    """Tiny threaded MJPEG server for browser-based live previews."""

    def __init__(
        self,
        label: str,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        jpeg_quality: int = 80,
        logger: Any = None,
    ) -> None:
        self.label = str(label or "Preview")
        self._host = str(host or "0.0.0.0")
        self._port = int(port or 0)
        self._jpeg_quality = int(jpeg_quality)
        self._logger = logger if logger is not None else get_root_logger()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest_frame: Optional[bytes] = None
        self._frame_sequence = 0
        self._server: Optional[_PreviewHttpServer] = None
        self._thread: Optional[threading.Thread] = None
        self._announced = False

    @property
    def closed(self) -> bool:
        return self._server is None

    @property
    def port(self) -> int:
        return 0 if self._server is None else int(self._server.server_port)

    @property
    def urls(self) -> list[str]:
        return self._candidate_urls()

    @property
    def preferred_url(self) -> Optional[str]:
        urls = self.urls
        for url in urls:
            host = urlsplit(url).hostname or ""
            if host not in {"127.0.0.1", "localhost"}:
                return url
        return urls[0] if urls else None

    def _candidate_urls(self) -> list[str]:
        port = self.port
        if port <= 0:
            return []
        urls: list[str] = [f"http://127.0.0.1:{port}/"]
        if self._host not in {"0.0.0.0", "::", ""}:
            urls.append(f"http://{self._host}:{port}/")
        public_host = _best_public_host()
        if public_host:
            urls.append(f"http://{public_host}:{port}/")
        deduped: list[str] = []
        for url in urls:
            if url not in deduped:
                deduped.append(url)
        return deduped

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _PreviewHttpServer((self._host, self._port), _PreviewRequestHandler, self)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if not self._announced:
            urls = self._candidate_urls()
            if urls:
                self._logger.info(
                    "Headless preview for %s is available at: %s",
                    self.label,
                    ", ".join(urls),
                )
            self._announced = True

    def publish(
        self,
        img: torch.Tensor | np.ndarray | StreamTensorBase,
        *,
        show_scaled: Optional[float] = None,
    ) -> None:
        if self._server is None:
            self.start()
        frame = _normalize_preview_frame(img, show_scaled=show_scaled)
        if frame.ndim == 2:
            pil_frame = Image.fromarray(frame)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            pil_frame = Image.fromarray(frame[:, :, ::-1])
        elif frame.ndim == 3 and frame.shape[2] == 4:
            pil_frame = Image.fromarray(frame[:, :, [2, 1, 0, 3]]).convert("RGB")
        else:
            raise ValueError(f"Unsupported preview frame shape={frame.shape}")
        encoded = io.BytesIO()
        pil_frame.save(encoded, format="JPEG", quality=self._jpeg_quality)
        with self._condition:
            self._latest_frame = encoded.getvalue()
            self._frame_sequence += 1
            self._condition.notify_all()

    def wait_for_frame(
        self,
        last_sequence: int,
        timeout: float = 1.0,
    ) -> Optional[tuple[bytes, int]]:
        deadline = time.time() + float(timeout)
        with self._condition:
            while self._latest_frame is None or self._frame_sequence == last_sequence:
                remaining = deadline - time.time()
                if remaining <= 0:
                    if self._latest_frame is None:
                        return None
                    break
                self._condition.wait(timeout=remaining)
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._frame_sequence

    def close(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
        with self._condition:
            self._condition.notify_all()


def mask_stream_url(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path or ""
    if path.count("/") >= 2:
        prefix, _, suffix = path.rpartition("/")
        if suffix:
            suffix = suffix[:4] + "..." if len(suffix) > 4 else "***"
        path = prefix + "/" + suffix
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def resolve_youtube_stream_url(base_url: str, stream_key: Optional[str]) -> str:
    key = (stream_key or os.environ.get("HM_YOUTUBE_STREAM_KEY") or "").strip()
    if not key:
        key = (os.environ.get("YOUTUBE_STREAM_KEY") or "").strip()
    base = (base_url or "").strip()
    if not base:
        base = "rtmps://a.rtmps.youtube.com/live2"
    if key and not base.rstrip("/").endswith(key):
        return base.rstrip("/") + "/" + key
    return base


def validate_youtube_stream_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"rtmp", "rtmps"}:
        raise ValueError(f"YouTube live preview requires an rtmp:// or rtmps:// URL, got {url!r}")
    path = (parsed.path or "").strip("/")
    if not path or path.endswith("live2") or path.endswith("live2/"):
        raise ValueError(
            "YouTube live preview requires a full publish URL with a stream key. "
            "Provide --youtube-stream-key or set HM_YOUTUBE_STREAM_KEY."
        )


class FFmpegLivePublisher:
    """Pipe raw preview frames into ffmpeg for live RTMP(S) publishing."""

    def __init__(
        self,
        output_url: str,
        *,
        label: str,
        fps: float,
        logger: Any = None,
    ) -> None:
        self.output_url = str(output_url)
        self.label = str(label or "Live Preview")
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._logger = logger if logger is not None else get_root_logger()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def _stderr_worker(self, pipe) -> None:
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
        finally:
            pipe.close()

    def _build_cmd(self, width: int, height: int) -> list[str]:
        gop = max(1, int(round(self.fps * 2.0)))
        return [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "flv",
            self.output_url,
        ]

    def _ensure_open(self, width: int, height: int) -> None:
        if self._process is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("ffmpeg is required for live preview streaming")
        self._width = int(width)
        self._height = int(height)
        cmd = self._build_cmd(width=self._width, height=self._height)
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert self._process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._stderr_worker,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()
        self._logger.info(
            "Publishing %s to %s",
            self.label,
            mask_stream_url(self.output_url),
        )

    def write_frame(self, img: torch.Tensor | np.ndarray | StreamTensorBase) -> None:
        frame = _normalize_preview_frame(img, show_scaled=None)
        self._ensure_open(width=frame.shape[1], height=frame.shape[0])
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("ffmpeg live publisher is not open")
        try:
            process.stdin.write(frame.tobytes())
            process.stdin.flush()
        except BrokenPipeError as exc:
            stderr_tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                f"ffmpeg live publisher failed for {mask_stream_url(self.output_url)}\n{stderr_tail}"
            ) from exc

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        if process.returncode not in (0, None):
            stderr_tail = "\n".join(self._stderr_tail)
            if any(
                marker in stderr_tail
                for marker in (
                    "Connection reset by peer",
                    "Broken pipe",
                    "Error writing trailer",
                )
            ):
                self._logger.warning(
                    "ffmpeg live publisher for %s ended after the remote peer closed the stream.\n%s",
                    mask_stream_url(self.output_url),
                    stderr_tail,
                )
                return
            raise RuntimeError(
                f"ffmpeg live publisher exited with code {process.returncode} for "
                f"{mask_stream_url(self.output_url)}\n{stderr_tail}"
            )

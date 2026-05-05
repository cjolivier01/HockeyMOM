import argparse
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch

from hmlib.hm_opts import hm_opts
from hmlib.ui.headless_preview import FFmpegLivePublisher, mask_stream_url
from hmlib.ui.shower import Shower

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class HeadlessPreviewRuntimeTest(unittest.TestCase):
    def _run_cli_smoke(self, script_name: str, *extra_args: str, env: dict[str, str] | None = None):
        full_env = os.environ.copy()
        full_env["PYTHONPATH"] = str(REPO_ROOT) + (
            ":" + full_env["PYTHONPATH"] if full_env.get("PYTHONPATH") else ""
        )
        if env:
            full_env.update(env)
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "hmlib" / "cli" / script_name),
                "--smoke-test",
                "--game-id=smoke-ci",
                *extra_args,
            ],
            cwd=str(REPO_ROOT),
            env=full_env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

    def test_hm_opts_maps_youtube_and_headless_preview_args(self):
        parser = hm_opts.parser(argparse.ArgumentParser())
        args = parser.parse_args(
            [
                "--show-youtube",
                "--youtube-stream-key",
                "test-key",
                "--headless-preview-host",
                "127.0.0.1",
                "--headless-preview-port",
                "9001",
                "--always-stream",
            ]
        )
        game_config = {
            "video_out": {
                "show_youtube": False,
                "youtube_stream_key": None,
                "headless_preview_host": "0.0.0.0",
                "headless_preview_port": 0,
                "always_stream": False,
            }
        }
        hm_opts.apply_arg_config_overrides(game_config, args, parser=parser)
        self.assertTrue(game_config["video_out"]["show_youtube"])
        self.assertEqual(game_config["video_out"]["youtube_stream_key"], "test-key")
        self.assertEqual(game_config["video_out"]["headless_preview_host"], "127.0.0.1")
        self.assertEqual(game_config["video_out"]["headless_preview_port"], 9001)
        self.assertTrue(game_config["video_out"]["always_stream"])

    def test_shower_starts_browser_preview_when_headless(self):
        with mock.patch.dict(
            os.environ,
            {"HM_FORCE_HEADLESS_PREVIEW": "1"},
            clear=False,
        ):
            shower = Shower(
                "headless-preview-test",
                show_scaled=0.5,
                max_size=1,
                headless_preview_host="127.0.0.1",
                headless_preview_port=0,
                always_stream=True,
            )
            try:
                self.assertIsNotNone(shower._headless_preview)
                for _ in range(48):
                    shower.show(torch.full((32, 48, 3), 127, dtype=torch.uint8))
                    time.sleep(0.05)
                port = shower._headless_preview.port
                self.assertGreater(port, 0)
                html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
                self.assertIn(b"stream.m3u8", html)

                manifest = b""
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    try:
                        manifest = urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/stream.m3u8",
                            timeout=5,
                        ).read()
                    except Exception:
                        manifest = b""
                    if b"#EXTM3U" in manifest and b"preview-" in manifest:
                        break
                    time.sleep(0.1)
                self.assertIn(b"#EXTM3U", manifest)
                segment_name = next(
                    line.decode("utf-8")
                    for line in manifest.splitlines()
                    if line and not line.startswith(b"#")
                )
                payload = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/{segment_name}",
                    timeout=5,
                ).read(256)
                self.assertGreater(len(payload), 0)
            finally:
                shower.close()

    def test_shower_headless_preview_waits_for_client_by_default(self):
        with (
            mock.patch.dict(
                os.environ,
                {"HM_FORCE_HEADLESS_PREVIEW": "1"},
                clear=False,
            ),
            mock.patch(
                "hmlib.ui.headless_preview._RawHlsPreviewPublisher", autospec=True
            ) as pub_cls,
        ):
            shower = Shower(
                "headless-preview-demand-test",
                show_scaled=0.5,
                max_size=1,
                headless_preview_host="127.0.0.1",
                headless_preview_port=0,
            )
            try:
                frame = torch.full((32, 48, 3), 127, dtype=torch.uint8)
                shower.show(frame)
                time.sleep(0.2)
                self.assertIsNotNone(shower._headless_preview)
                self.assertGreater(shower._headless_preview.port, 0)
                self.assertFalse(pub_cls.return_value.write_frame.called)

                urllib.request.urlopen(
                    f"http://127.0.0.1:{shower._headless_preview.port}/",
                    timeout=5,
                ).read()
                for _ in range(3):
                    shower.show(frame)
                    time.sleep(0.05)
                time.sleep(0.2)
                self.assertTrue(pub_cls.return_value.write_frame.called)
            finally:
                shower.close()

    def test_shower_exposes_stream_urls_for_progress_bar(self):
        with (
            mock.patch.dict(
                os.environ,
                {"HM_FORCE_HEADLESS_PREVIEW": "1"},
                clear=False,
            ),
            mock.patch("hmlib.ui.shower.FFmpegLivePublisher", autospec=True),
        ):
            shower = Shower(
                "headless-progress-test",
                show_scaled=0.5,
                max_size=1,
                show_youtube=True,
                youtube_stream_url="rtmp://127.0.0.1:1935/live/secret-stream-key",
                headless_preview_host="127.0.0.1",
                headless_preview_port=0,
            )
            try:
                table_map: dict[str, str] = {}
                shower.update_progress_table(table_map)
                self.assertEqual(
                    table_map["Publish URL"],
                    mask_stream_url("rtmp://127.0.0.1:1935/live/secret-stream-key"),
                )
                self.assertNotIn("Preview URL", table_map)

                shower.show(torch.full((32, 48, 3), 127, dtype=torch.uint8))
                time.sleep(0.2)

                shower.update_progress_table(table_map)
                self.assertIn("Preview URL", table_map)
                self.assertTrue(table_map["Preview URL"].startswith("http://"))
                self.assertIn("Publish URL", table_map)
            finally:
                shower.close()

    def test_progress_bar_updates_with_shower_stream_urls(self):
        from hmlib.utils.progress_bar import ProgressBar

        with (
            mock.patch.dict(
                os.environ,
                {"HM_FORCE_HEADLESS_PREVIEW": "1"},
                clear=False,
            ),
            mock.patch("hmlib.ui.shower.FFmpegLivePublisher", autospec=True),
        ):
            progress_bar = ProgressBar(total=1)
            shower = Shower(
                "progress-bar-preview-test",
                show_scaled=0.5,
                max_size=1,
                show_youtube=True,
                youtube_stream_url="rtmp://127.0.0.1:1935/live/secret-stream-key",
                headless_preview_host="127.0.0.1",
                headless_preview_port=0,
            )
            try:
                progress_bar.add_table_callback(shower.update_progress_table)
                shower.show(torch.full((32, 48, 3), 127, dtype=torch.uint8))
                time.sleep(0.2)
                table_map: OrderedDict[str, str] = OrderedDict()
                progress_bar._run_callbacks(table_map)
                self.assertIn("Preview URL", table_map)
                self.assertTrue(table_map["Preview URL"].startswith("http://"))
                self.assertIn("Publish URL", table_map)
            finally:
                shower.close()

    def test_shower_disables_live_publisher_after_publish_failure(self):
        with mock.patch("hmlib.ui.shower.FFmpegLivePublisher", autospec=True) as publisher_cls:
            publisher = publisher_cls.return_value
            publisher.write_frame.side_effect = RuntimeError("broken pipe")
            shower = Shower(
                "publish-failure-test",
                max_size=1,
                enable_local_display=False,
                show_youtube=True,
                youtube_stream_url="rtmp://127.0.0.1:1935/live/test",
            )
            try:
                shower.show(torch.full((32, 48, 3), 127, dtype=torch.uint8))
                time.sleep(0.2)
                self.assertTrue(publisher.close.called)
                self.assertIsNone(shower._youtube_publisher)
            finally:
                shower.close()

    def test_shower_close_stops_fps_worker(self):
        shower = Shower(
            "fps-close-test",
            fps=10.0,
            max_size=1,
            enable_local_display=False,
        )
        shower.show(torch.full((16, 16, 3), 127, dtype=torch.uint8))
        time.sleep(0.1)
        closer = threading.Thread(target=shower.close)
        closer.start()
        closer.join(timeout=2.0)
        self.assertFalse(closer.is_alive(), "Shower.close() should stop fps workers promptly")

    def test_shower_drop_oldest_when_full_keeps_latest_frame(self):
        shower = Shower(
            "drop-oldest-preview-test",
            max_size=1,
            enable_local_display=False,
            skip_frame_when_full=True,
            drop_oldest_when_full=True,
        )
        try:
            shower.close()
            shower._q = queue.Queue()
            shower._thread = object()
            first = torch.full((8, 8, 3), 1, dtype=torch.uint8)
            second = torch.full((8, 8, 3), 2, dtype=torch.uint8)
            shower._q.put(first)
            shower.show(second, clone=False)
            kept = shower._q.get_nowait()
            self.assertTrue(torch.equal(kept, second))
        finally:
            shower._thread = None

    def test_cli_smoke_runs_headless_preview_for_hmtrack_and_stitch(self):
        for script_name in ("hmtrack.py", "stitch.py"):
            result = self._run_cli_smoke(
                script_name,
                "--show",
                "--headless-preview-host=127.0.0.1",
                "--headless-preview-port=0",
                env={"HM_FORCE_HEADLESS_PREVIEW": "1"},
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Headless preview OK. url=http://127.0.0.1:", result.stdout)
            self.assertIn("Smoke test OK.", result.stdout)

    def test_cli_smoke_runs_youtube_publish_for_hmtrack_and_stitch(self):
        for script_name in ("hmtrack.py", "stitch.py"):
            port = _find_free_port()
            with tempfile.TemporaryDirectory(prefix="hm-cli-preview-") as tmpdir:
                output_path = Path(tmpdir) / f"{script_name}.flv"
                stream_url = f"rtmp://127.0.0.1:{port}/live/{script_name.replace('.py', '')}"
                receiver = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-listen",
                        "1",
                        "-timeout",
                        "5",
                        "-i",
                        stream_url,
                        "-t",
                        "2",
                        "-c",
                        "copy",
                        str(output_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    time.sleep(1.0)
                    result = self._run_cli_smoke(
                        script_name,
                        "--show-youtube",
                        f"--youtube-stream-url={stream_url}",
                    )
                    self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
                    self.assertIn("YouTube preview publish OK. url=", result.stdout)
                    self.assertIn("Smoke test OK.", result.stdout)
                    receiver.wait(timeout=10)
                finally:
                    if receiver.poll() is None:
                        receiver.kill()
                        receiver.wait(timeout=5)
                self.assertEqual(receiver.returncode, 0)
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

    def test_ffmpeg_live_publisher_prefers_nvenc_for_cuda_frames(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        packets: list[bytes] = []
        encoder_instances: list[object] = []

        class _FakeMuxer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.closed = False

            def write_packet(self, payload: bytes) -> None:
                packets.append(payload)

            def close(self) -> None:
                self.closed = True

        class _FakeEncoder:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.opened = False
                self.closed = False
                self.written_shapes: list[tuple[int, ...]] = []
                encoder_instances.append(self)

            def open(self) -> None:
                self.opened = True

            def write(self, frame: torch.Tensor) -> None:
                self.written_shapes.append(tuple(int(dim) for dim in frame.shape))
                handler = self.kwargs["bitstream_handler"]
                handler(b"fake-nvenc-packet")

            def close(self) -> None:
                self.closed = True

        with (
            mock.patch("hmlib.ui.headless_preview._BitstreamLiveMuxer", _FakeMuxer),
            mock.patch.dict(
                sys.modules,
                {"hmlib.video.py_nv_encoder": types.SimpleNamespace(PyNvVideoEncoder=_FakeEncoder)},
            ),
        ):
            publisher = FFmpegLivePublisher(
                output_url="rtmp://127.0.0.1:1935/live/test",
                label="nvenc-preview-test",
                fps=10.0,
            )
            frame = torch.zeros((48, 64, 3), dtype=torch.uint8, device="cuda")
            frame[..., 2] = 255
            publisher.write_frame(frame, show_scaled=0.5)
            publisher.close()

        self.assertEqual(packets, [b"fake-nvenc-packet"])
        self.assertEqual(len(encoder_instances), 1)
        self.assertTrue(encoder_instances[0].opened)
        self.assertTrue(encoder_instances[0].closed)
        self.assertEqual(encoder_instances[0].kwargs["codec"], "h264")
        self.assertEqual(encoder_instances[0].written_shapes, [(24, 32, 3)])

    def test_ffmpeg_live_publisher_nvenc_crops_odd_dimensions(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

        encoder_instances: list[object] = []

        class _FakeMuxer:
            def __init__(self, *args, **kwargs):
                self.closed = False

            def write_packet(self, payload: bytes) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        class _FakeEncoder:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                self.written_shapes: list[tuple[int, ...]] = []
                encoder_instances.append(self)

            def open(self) -> None:
                return None

            def write(self, frame: torch.Tensor) -> None:
                self.written_shapes.append(tuple(int(dim) for dim in frame.shape))

            def close(self) -> None:
                return None

        with (
            mock.patch("hmlib.ui.headless_preview._BitstreamLiveMuxer", _FakeMuxer),
            mock.patch.dict(
                sys.modules,
                {"hmlib.video.py_nv_encoder": types.SimpleNamespace(PyNvVideoEncoder=_FakeEncoder)},
            ),
        ):
            publisher = FFmpegLivePublisher(
                output_url="rtmp://127.0.0.1:1935/live/test",
                label="nvenc-odd-dims-test",
                fps=10.0,
            )
            frame = torch.zeros((49, 65, 3), dtype=torch.uint8, device="cuda")
            publisher.write_frame(frame)
            publisher.close()

        self.assertEqual(len(encoder_instances), 1)
        self.assertEqual(encoder_instances[0].kwargs["width"], 64)
        self.assertEqual(encoder_instances[0].kwargs["height"], 48)
        self.assertEqual(encoder_instances[0].written_shapes, [(48, 64, 3)])

    def test_ffmpeg_live_publisher_streams_to_local_rtmp_listener(self):
        port = _find_free_port()
        with tempfile.TemporaryDirectory(prefix="hm-preview-") as tmpdir:
            output_path = Path(tmpdir) / "preview.flv"
            recv_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-listen",
                "1",
                "-timeout",
                "5",
                "-i",
                f"rtmp://127.0.0.1:{port}/live/test",
                "-t",
                "2",
                "-c",
                "copy",
                str(output_path),
            ]
            receiver = subprocess.Popen(
                recv_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                time.sleep(1.0)
                publisher = FFmpegLivePublisher(
                    output_url=f"rtmp://127.0.0.1:{port}/live/test",
                    label="rtmp-preview-test",
                    fps=10.0,
                )
                frame = torch.zeros((48, 64, 3), dtype=torch.uint8)
                frame[..., 2] = 255
                for _ in range(25):
                    publisher.write_frame(frame)
                publisher.close()
                receiver.wait(timeout=10)
            finally:
                if receiver.poll() is None:
                    receiver.kill()
                    receiver.wait(timeout=5)
            self.assertEqual(receiver.returncode, 0)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertNotIn("/live/test", mask_stream_url(f"rtmp://127.0.0.1:{port}/live/test"))


if __name__ == "__main__":
    unittest.main()

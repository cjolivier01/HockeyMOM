import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

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
            ]
        )
        game_config = {
            "video_out": {
                "show_youtube": False,
                "youtube_stream_key": None,
                "headless_preview_host": "0.0.0.0",
                "headless_preview_port": 0,
            }
        }
        hm_opts.apply_arg_config_overrides(game_config, args, parser=parser)
        self.assertTrue(game_config["video_out"]["show_youtube"])
        self.assertEqual(game_config["video_out"]["youtube_stream_key"], "test-key")
        self.assertEqual(game_config["video_out"]["headless_preview_host"], "127.0.0.1")
        self.assertEqual(game_config["video_out"]["headless_preview_port"], 9001)

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
            )
            try:
                self.assertIsNotNone(shower._headless_preview)
                shower.show(np.full((32, 48, 3), 127, dtype=np.uint8))
                time.sleep(0.2)
                port = shower._headless_preview.port
                self.assertGreater(port, 0)
                html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
                self.assertIn(b"stream.mjpg", html)
                stream = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=5)
                try:
                    payload = stream.read(2048)
                finally:
                    stream.close()
                self.assertIn(b"Content-Type: image/jpeg", payload)
                self.assertIn(b"\xff\xd8", payload)
            finally:
                shower.close()

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
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
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

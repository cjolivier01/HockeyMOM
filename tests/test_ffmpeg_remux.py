import subprocess
import tempfile
from pathlib import Path


def should_remux_raw_bitstream_with_timestamps():
    from hmlib.video.py_nv_encoder import PyNvVideoEncoder

    with tempfile.TemporaryDirectory(prefix="hm_ffmpeg_mux_") as td:
        td_path = Path(td)
        raw = td_path / "test.h264"
        out = td_path / "out.mkv"

        # Create a tiny raw H.264 elementary stream (no container timestamps).
        subprocess.check_call(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:r=5",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "h264",
                str(raw),
            ]
        )
        assert raw.is_file() and raw.stat().st_size > 0

        # Instantiate without calling __init__ (avoids requiring PyNvVideoCodec).
        enc = PyNvVideoEncoder.__new__(PyNvVideoEncoder)
        enc.output_path = out
        enc.fps = 5.0
        enc.codec = "h264"
        enc._frames_in_current_bitstream = 5
        enc._ffmpeg_output_handler = None

        enc._mux_bitstream_file_with_ffmpeg(raw)

        assert out.is_file() and out.stat().st_size > 0
        probe = subprocess.check_output(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(out),
            ]
        ).decode("utf-8", errors="ignore")
        duration = float(probe.strip())
        assert duration > 0.5

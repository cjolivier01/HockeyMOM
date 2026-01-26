import shutil
import subprocess

import pytest

from hmlib.audio import has_audio_stream, mux_audio_in_place


def _have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg/ffprobe not available")
def should_mux_audio_into_video(tmp_path):
    audio_src = tmp_path / "audio_src.mp4"
    video_dst = tmp_path / "video_no_audio.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(audio_src),
        ],
        check=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_dst),
        ],
        check=True,
    )

    assert not has_audio_stream(str(video_dst))

    out = mux_audio_in_place(input_audio=str(audio_src), video_path=str(video_dst))
    assert out == str(video_dst)
    assert has_audio_stream(str(video_dst))

from __future__ import annotations

import argparse
from pathlib import Path


def should_sanitize_game_id_for_filename():
    from hmlib.utils.path import sanitize_game_id_for_filename

    assert sanitize_game_id_for_filename("tv-12-1-r3") == "tv-12-1-r3"
    assert sanitize_game_id_for_filename("tv/12/1") == "tv_12_1"
    assert sanitize_game_id_for_filename(r"tv\\12\\1") == "tv_12_1"


def should_prefix_game_id_for_deployed_videos():
    from hmlib.utils.path import add_game_id_prefix_to_filename

    name = "tracking_output-with-audio.mp4"
    game_id = "tv-12-1-r3"

    prefixed = add_game_id_prefix_to_filename(name, game_id, sep="-")
    assert prefixed == "tv-12-1-r3-tracking_output-with-audio.mp4"

    # Idempotent when already prefixed.
    assert add_game_id_prefix_to_filename(prefixed, game_id, sep="-") == prefixed

    # Preserves Path type.
    p = Path("/tmp") / name
    out_p = add_game_id_prefix_to_filename(p, "tv/12", sep="-")
    assert isinstance(out_p, Path)
    assert out_p.name == "tv_12-tracking_output-with-audio.mp4"


def should_build_ffmpeg_mux_command_with_audio():
    from hmlib.video.py_nv_encoder import build_ffmpeg_raw_bitstream_mux_cmd

    cmd = build_ffmpeg_raw_bitstream_mux_cmd(
        ffmpeg="ffmpeg",
        bitstream_path=Path("in.h265"),
        output_path=Path("out.mp4"),
        stream_format="hevc",
        fps=30.0,
        frames_in_bitstream=300,
        muxer="mp4",
        audio_file="audio.mp4",
        audio_stream=0,
        audio_offset_seconds=0.25,
        audio_codec="aac",
        aac_bitrate="192k",
    )

    # Video input.
    assert "-f" in cmd and "hevc" in cmd
    assert str(Path("in.h265")) in cmd

    # Audio input + offset.
    assert "-itsoffset" in cmd and "0.25" in cmd
    assert "audio.mp4" in cmd

    # Stream selection + codec handling.
    assert ["-map", "0:v:0"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["-map", "1:a:0"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["-c:a", "copy"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert "-shortest" in cmd

    # MP4 faststart + iPhone-friendly tag.
    assert ["-movflags", "+faststart"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["-tag:v", "hvc1"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]

    # Non-AAC audio should be re-encoded to AAC.
    cmd_reencode = build_ffmpeg_raw_bitstream_mux_cmd(
        ffmpeg="ffmpeg",
        bitstream_path=Path("in.h265"),
        output_path=Path("out.mp4"),
        stream_format="hevc",
        fps=30.0,
        frames_in_bitstream=300,
        muxer="mp4",
        audio_file="audio.mkv",
        audio_stream=1,
        audio_offset_seconds=0.0,
        audio_codec="mp3",
        aac_bitrate="192k",
    )
    assert ["-c:a", "aac"] in [cmd_reencode[i : i + 2] for i in range(len(cmd_reencode) - 1)]
    assert ["-b:a", "192k"] in [cmd_reencode[i : i + 2] for i in range(len(cmd_reencode) - 1)]


def should_parse_mux_audio_args():
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--output-label",
            "variant_a",
            "--mux-audio-file",
            "/tmp/audio.mp4",
            "--mux-audio-stream",
            "1",
            "--mux-audio-offset-seconds",
            "0.125",
            "--mux-audio-aac-bitrate",
            "256k",
        ]
    )

    assert args.output_label == "variant_a"
    assert args.mux_audio_file == "/tmp/audio.mp4"
    assert args.mux_audio_stream == 1
    assert abs(float(args.mux_audio_offset_seconds) - 0.125) < 1e-9
    assert args.mux_audio_aac_bitrate == "256k"

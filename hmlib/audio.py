"""Utilities for copying, synchronizing and concatenating audio tracks.

These helpers wrap small ``ffmpeg`` pipelines to extract and merge audio
streams between videos.

@see @ref hmlib.game_audio.transfer_audio "transfer_audio" for game-level usage.
@see @ref hmlib.stitching.synchronize.synchronize_by_audio "synchronize_by_audio"
     for audio-based synchronization of multiple cameras.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from hmlib.log import get_logger
from hmlib.stitching.synchronize import synchronize_by_audio


def make_parser():
    """Build an :class:`argparse.ArgumentParser` for the audio utility CLI.

    @return: Configured parser with ``--input-audio`` / ``--input-video`` options.
    @see @ref copy_audio "copy_audio" for the core implementation invoked by the CLI.
    """
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "--input-audio",
        default=None,
        type=str,
        # required=True,
        help="Input audio/videos that have audio",
    )
    parser.add_argument(
        "--input-video",
        default=None,
        type=str,
        # required=True,
        help="Input videos that needs (new) audio",
    )
    parser.add_argument(
        "--output-video",
        "-o",
        "--output",
        default=None,
        type=str,
        # required=True,
        help="Input videos that needs (new) audio",
    )
    return parser


def extract_audio(filename: str, force: bool = False) -> str:
    """Extract an AAC audio track from a video using ``ffmpeg``.

    @param filename: Input video file path.
    @param force: Overwrite an existing ``.aac`` file if ``True``.
    @return: Path to the extracted ``.aac`` file, or ``None`` on failure.
    @see @ref concatenate_audio "concatenate_audio" for merging multiple tracks.
    """
    audio_file, _ = os.path.splitext(filename)
    audio_file += ".aac"
    if not force and os.path.exists(audio_file):
        return audio_file
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",  # Overwrite output files without asking
        "-i",
        filename,
        "-vn",
        "-acodec",
        "copy",
        audio_file,
    ]
    process = None
    try:
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=sys.stdout)
        return audio_file
    except subprocess.CalledProcessError as e:
        get_logger(__name__).error(
            "Failed to extract audio: %s\n%s", e, getattr(process, "stdout", b"")
        )
        return None


def concatenate_audio(files) -> tempfile.TemporaryFile:
    """Concatenate audio tracks from one or more input files.

    Each input file may be a video (audio is extracted first) or an audio
    file that ``ffmpeg`` can decode.

    @param files: Iterable of filenames whose audio streams will be concatenated.
    @return: Temporary file handle pointing at the concatenated AAC audio.
    @see @ref copy_audio "copy_audio" for attaching merged audio to a video.
    """
    audio_files = [extract_audio(f) for f in files]

    # Create a temporary file for the output audio
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".aac")
    concat_list_file = tempfile.NamedTemporaryFile(delete=True, suffix=".txt")

    with open(concat_list_file.name, "w") as f:
        for audio_file in audio_files:
            # Write file paths, ensuring any special characters are escaped
            f.write(f"file '{audio_file}'\n")

    # Prepare the FFmpeg command for concatenating audio streams
    command = [
        "ffmpeg",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-y",  # Overwrite output files without asking
        "-i",
        concat_list_file.name,
        "-c",
        "copy",
        temp_file.name,
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        get_logger(__name__).info("Audio concatenated successfully, saved to %s", temp_file.name)
    except subprocess.CalledProcessError as e:
        get_logger(__name__).error("Failed to concatenate audio: %s", e)
        temp_file.close()
        return None

    return temp_file


def copy_audio(
    input_audio: Union[str, List[str], dict],
    input_video: str,
    output_video: str,
    shortest: bool = True,
    start_seconds: float = 0.0,
    check: bool = True,
):
    """Copy or merge audio from one or more sources into a target video.

    @param input_audio: Single path, list of paths, or dict with ``\"left\"``/``\"right\"`` keys.
    @param input_video: Path to the video that will receive the audio track.
    @param output_video: Output video filename with audio attached.
    @param shortest: If ``True``, stop when the shortest stream ends (ffmpeg ``-shortest``).
    @param start_seconds: Optional offset (in seconds) to seek into the audio source before muxing.
    @param check: If True, raise on ffmpeg failure.
    @see @ref hmlib.game_audio.transfer_audio "transfer_audio" for a higher-level wrapper.
    """
    temp_audio_file = None
    audio_source = None
    if isinstance(input_audio, dict) or (isinstance(input_audio, list) and len(input_audio) == 2):
        if isinstance(input_audio, dict):
            input_sync_audio = [input_audio["left"][0], input_audio["right"][0]]
        else:
            input_sync_audio = input_audio
        assert len(input_audio) == 2
        lfo, rfo = synchronize_by_audio(input_sync_audio[0], input_sync_audio[1])
        if lfo == 0:
            if isinstance(input_audio, dict) and len(input_audio["left"]) > 1:
                temp_audio_file = concatenate_audio(input_audio["left"])
                audio_source = temp_audio_file.name
            else:
                audio_source = input_sync_audio[0]
        elif rfo == 0:
            if isinstance(input_audio, dict) and len(input_audio["right"]) > 1:
                temp_audio_file = concatenate_audio(input_audio["right"])
                audio_source = temp_audio_file.name
            else:
                audio_source = input_sync_audio[1]
        else:
            raise AssertionError(f"Either lfo or rfo should be zero, but were {lfo=} and {rfo=}")
    else:
        if isinstance(input_audio, list):
            if len(input_audio) == 1:
                audio_source = input_audio[0]
            else:
                temp_audio_file = concatenate_audio(input_audio)
                audio_source = temp_audio_file.name if temp_audio_file is not None else None
        else:
            audio_source = input_audio

    if not audio_source:
        raise RuntimeError("No valid audio source provided for muxing.")

    output_suffix = Path(output_video).suffix.lower()
    output_is_mp4 = output_suffix in {".mp4", ".m4v", ".mov"}
    video_codec = None
    if output_is_mp4:
        try:
            video_codec = (
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        input_video,
                    ]
                )
                .decode("utf-8", errors="ignore")
                .strip()
                .lower()
            )
        except Exception:
            video_codec = None
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
    ]
    if start_seconds and start_seconds > 0:
        command += ["-ss", str(float(start_seconds))]
    command += [
        "-i",
        audio_source,
        "-i",
        input_video,
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-map",
        "1:v:0",
        "-map",
        "0:a:0",
    ]
    if output_is_mp4:
        # Improve compatibility with iOS/macOS players and streaming.
        command += ["-movflags", "+faststart"]
        if video_codec in {"hevc", "h265"}:
            command += ["-tag:v", "hvc1"]
    if shortest:
        command.append("-shortest")

    command.append(output_video)
    if check:
        subprocess.run(command, check=True)
    else:
        process = subprocess.Popen(command)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    if temp_audio_file is not None:
        try:
            temp_audio_file.close()
        except Exception:
            pass


def has_audio_stream(path: str) -> bool:
    """Return True if ffprobe detects at least one audio stream in the file."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                path,
            ]
        )
    except Exception:
        return False
    return bool(out.strip())


def mux_audio_in_place(
    input_audio: Union[str, List[str], dict],
    video_path: str,
    *,
    start_seconds: float = 0.0,
    shortest: bool = True,
    keep_original: bool = False,
) -> Optional[str]:
    """Mux audio into an existing video file by writing a temp file and replacing.

    Returns the final video path (same as ``video_path``) on success, or ``None`` on failure.
    """
    if not video_path:
        return None
    if not os.path.exists(video_path):
        return None

    path = Path(video_path)
    suffix = path.suffix or ""
    if suffix:
        tmp_path = str(path.with_name(f".{path.stem}.with-audio{suffix}"))
        bak_path = str(path.with_name(f".{path.stem}.no-audio{suffix}")) if keep_original else None
    else:
        dir_name = os.path.dirname(video_path) or "."
        base = os.path.basename(video_path)
        tmp_path = os.path.join(dir_name, f".{base}.with-audio.tmp")
        bak_path = os.path.join(dir_name, f".{base}.no-audio.bak") if keep_original else None
    try:
        copy_audio(
            input_audio=input_audio,
            input_video=video_path,
            output_video=tmp_path,
            shortest=shortest,
            start_seconds=start_seconds,
            check=True,
        )
        if keep_original and bak_path:
            os.replace(video_path, bak_path)
        os.replace(tmp_path, video_path)
        return video_path
    except Exception as exc:
        get_logger(__name__).warning("Audio mux failed for %s: %s", video_path, exc)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.input_audio and "," in args.input_audio:
        args.input_audio = args.input_audio.split(",")
    home = os.environ["HOME"]
    game_id = "ev-sabercats-3"
    file_list = {
        "left": [
            # "/olivier-pool/Videos/test/left-1.mp4",
            # "/olivier-pool/Videos/test/left-2.mp4",
            # "/olivier-pool/Videos/test/left-3.mp4",
            f"{home}/Videos/{game_id}/GX010084.MP4",
            f"{home}/Videos/{game_id}/GX020084.MP4",
            f"{home}/Videos/{game_id}/GX030084.MP4",
        ],
        "right": [
            # "/olivier-pool/Videos/test/right-1.mp4",
            # "/olivier-pool/Videos/test/right-2.mp4",
            # "/olivier-pool/Videos/test/right-3.mp4",
            f"{home}/Videos/{game_id}/GX010004.MP4",
            f"{home}/Videos/{game_id}/GX020004.MP4",
            f"{home}/Videos/{game_id}/GX030004.MP4",
        ],
    }
    # concatenate_videos(file_list["right"], "/olivier-pool/Videos/test3/right.mp4")
    # args.input_video = f"{home}/Videos/{game_id}/tracking_output.mkv"
    # args.input_video = "/home/colivier/rsrc/hm/test2.mkv"
    args.output_video = "withsound.mp4"
    copy_audio(
        # args.input_audio,
        file_list,
        args.input_video,
        args.output_video,
    )

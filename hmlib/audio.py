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
from typing import List, Union

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
        get_logger(__name__).info(
            "Audio concatenated successfully, saved to %s", temp_file.name
        )
    except subprocess.CalledProcessError as e:
        get_logger(__name__).error("Failed to concatenate audio: %s", e)
        temp_file.close()
        return None

    return temp_file


def copy_audio(
    input_audio: Union[str, List[str]], input_video: str, output_video: str, shortest: bool = True
):
    """Copy or merge audio from one or more sources into a target video.

    @param input_audio: Single path, list of paths, or dict with ``\"left\"``/``\"right\"`` keys.
    @param input_video: Path to the video that will receive the audio track.
    @param output_video: Output video filename with audio attached.
    @param shortest: If ``True``, stop when the shortest stream ends (ffmpeg ``-shortest``).
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
            assert len(input_audio) == 1
            input_audio = input_audio[0]
        audio_source = input_audio
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
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
    if shortest:
        command.append("-shortest")

    command.append(output_video)
    process = subprocess.Popen(command)
    process.wait()


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

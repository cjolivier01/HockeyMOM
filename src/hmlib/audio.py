import argparse
import os
import subprocess
import tempfile
from typing import List, Union

from hmlib.stitching.synchronize import synchronize_by_audio


def make_parser():
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


def concatenate_audio(files) -> tempfile.TemporaryFile:
    # Create a temporary file for the output audio
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".aac")
    concat_list_file = tempfile.NamedTemporaryFile(delete=True, suffix=".txt")

    with open(concat_list_file.name, "w") as f:
        for output_file in files:
            # Write file paths, ensuring any special characters are escaped
            f.write(f"file '{output_file}'\n")

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
    ]

    # Adding input files to the command
    for file in files:
        command.extend(["-i", file])

    command.extend(["-c", "copy", temp_file.name])

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Audio concatenated successfully, saved to {temp_file.name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to concatenate audio: {e}")
        temp_file.close()
        return None

    return temp_file


def copy_audio(
    input_audio: Union[str, List[str]], input_video: str, output_video: str, shortest: bool = True
):
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
    file_list = {
        "left": [
            "/olivier-pool/Videos/test/parts/left-1.mp4",
            "/olivier-pool/Videos/test/parts/left-3.mp4",
            "/olivier-pool/Videos/test/parts/left-2.mp4",
        ],
        "right": [
            "/olivier-pool/Videos/test/parts/right-1.mp4",
            "/olivier-pool/Videos/test/parts/right-3.mp4",
            "/olivier-pool/Videos/test/parts/right-2.mp4",
        ],
    }
    args.input_video = "/home/colivier/src/hm/output_workdirs/test/tracking_output.mkv"
    args.output_video = "withsound.mp4"
    copy_audio(
        # args.input_audio,
        file_list,
        args.input_video,
        args.output_video,
    )

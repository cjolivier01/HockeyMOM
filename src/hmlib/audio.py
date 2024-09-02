import argparse
import subprocess
from typing import List, Union

from hmlib.stitching.synchronize import synchronize_by_audio


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "--input-audio",
        default=None,
        type=str,
        required=True,
        help="Input audio/videos that have audio",
    )
    parser.add_argument(
        "--input-video",
        default=None,
        type=str,
        required=True,
        help="Input videos that needs (new) audio",
    )
    parser.add_argument(
        "--output-video",
        "-o",
        "--output",
        default=None,
        type=str,
        required=True,
        help="Input videos that needs (new) audio",
    )
    return parser


def copy_audio(input_audio: Union[str, List[str]], input_video: str, output_video: str):
    audio_source = None
    if isinstance(input_audio, list):
        assert len(input_audio) == 2
        lfo, rfo = synchronize_by_audio(input_audio[0], input_audio[1])
        if lfo == 0:
            audio_source = input_audio[0]
        elif rfo == 0:
            audio_source = input_audio[1]
        else:
            raise AssertionError(f"Either lfo or rfo should be zero, but were {lfo=} and {rfo=}")
    else:
        audio_source = input_audio
    command = [
        "ffmpeg",
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
        "-shortest",
        output_video,
    ]
    process = subprocess.Popen(command)
    process.wait()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if "," in args.input_audio:
        args.input_audio = args.input_audio.split(",")
    copy_audio(
        args.input_audio,
        args.input_video,
        args.output_video,
    )

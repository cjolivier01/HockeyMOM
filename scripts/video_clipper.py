"""
Given a vide, timestamp file of:

start_hh_mm_hh end_hh_mm_hh
start_hh_mm_hh end_hh_mm_hh
start_hh_mm_hh end_hh_mm_hh
...

.. and a text label...

Create transition text screens to separate and join all the clips into a new
video along with text labeling of the clip number and user-designated label

"""

import argparse
import subprocess
import os
from datetime import datetime
from typing import List
from pathlib import Path
import json


def validate_timestamp(timestamp):
    try:
        datetime.strptime(timestamp, "%H:%M:%S")
        return True
    except ValueError:
        return False


_DEBUG = True

ENCODER_ARGS_LOSSLESS = "-c:v hevc_nvenc -preset slow -qp 0 -pix_fmt yuv444p".split(" ")
ENCODER_ARGS_FAST = "-c:v hevc_nvenc -preset fast -pix_fmt yuv444p".split(" ")
ENCODER_ARGS_HQ = "-c:v hevc_nvenc -preset medium -pix_fmt yuv444p".split(" ")

FFMPEG_CUDA_DECODER = ["-c:v", "hevc_cuvid"]

if not _DEBUG or int(os.environ.get("VIDEO_CLIPPER_HQ", "0")) > 0:
    WORKING_ENCODER_ARGS = ENCODER_ARGS_LOSSLESS
    FINAL_ENCODER_ARGS = ENCODER_ARGS_HQ
else:
    # Debugging, faster, lower quality encoding
    WORKING_ENCODER_ARGS = ENCODER_ARGS_FAST
    FINAL_ENCODER_ARGS = ENCODER_ARGS_FAST

FFMPEG_BASE = ["ffmpeg", "-hide_banner"]
FFMPEG_BASE_HW: List[str] = FFMPEG_BASE + ["-hwaccel", "cuda"]


def get_audio_sample_rate(file_path: str):
    try:
        # Execute the ffprobe command to get the audio stream information
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "json",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Parse the JSON output
        output = json.loads(result.stdout)
        sample_rate = output["streams"][0]["sample_rate"]
        return int(sample_rate)
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as e:
        print(f"Error retrieving sample rate: {e}")
        return None


def hhmmss_to_duration_seconds(time_str: str) -> float:
    # Split the time duration string into components
    h = 0
    m = 0
    s = 0
    tokens = time_str.split(":")
    s = float(tokens[-1])
    if len(tokens) > 1:
        m = int(tokens[-2])
        if len(tokens) > 2:
            assert len(tokens) == 3
            h = int(tokens[0])
    # Extract seconds and milliseconds
    # Convert hours, minutes, seconds, and milliseconds to total seconds
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds


def extract_clip(
    input_video: str, start_time: str, end_time: str, clip_file: str, rate_k: int = 192
) -> None:
    duration = hhmmss_to_duration_seconds(end_time) - hhmmss_to_duration_seconds(start_time)
    cmd = (
        FFMPEG_BASE_HW
        # + FFMPEG_CUDA_DECODER
        + [
            "-ss",
            start_time,
            "-i",
            input_video,
            "-t",
            str(duration),
            "-c:a",
            "aac",
        ]
        + WORKING_ENCODER_ARGS
        + [
            "-y",
            clip_file,
        ]
    )
    subprocess.run(cmd, check=True)


def concat_video_clips(list_file: str, output_file: str) -> None:
    subprocess.run(
        FFMPEG_BASE
        + [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
        ]
        # + FINAL_ENCODER_ARGS
        + [
            "-c:a",
            "copy",
            "-c:v",
            "copy",
        ]
        + [
            "-y",
            output_file,
        ],
        check=True,
    )


def create_text_video(
    text: str,
    duration: int,
    output_file: str,
    width: int,
    height: int,
    fps: float,
    audio_sample_rate: int,
) -> None:
    cmd = (
        FFMPEG_BASE_HW
        + [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s={}x{}:d={}".format(width, height, duration),
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={int(audio_sample_rate)}:cl=stereo",
            "-vf",
            f"drawtext=text='{text}':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-r",
            str(fps),
            "-shortest",
        ]
        + WORKING_ENCODER_ARGS
        + [
            "-y",
            output_file,
        ]
    )
    subprocess.run(cmd, check=True)


def add_clip_number(
    input_file: str, output_file: str, label: str, clip_number: int, width: int, height: int
) -> None:
    text = label + " " + str(clip_number)
    cmd = (
        [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"drawtext=text='{text}':fontsize=52:fontcolor=white:x=w-tw-10:y=10",
            "-codec:a",
            "copy",
        ]
        + WORKING_ENCODER_ARGS
        + [
            "-y",
            output_file,
        ]
    )
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", help="Input video file")
    parser.add_argument("timestamps_file", help="File containing timestamps")
    parser.add_argument("label", help="Text label for transitions")
    args = parser.parse_args()

    # Get video dimensions
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=s=x:p=0",
        args.input_video,
    ]
    fprobe_output = subprocess.check_output(cmd).decode().strip()
    width, height, r_frame_rate = fprobe_output.split("x")
    width = int(width)
    height = int(height)
    frame_rate_num, frame_rate_demon = map(int, r_frame_rate.split("/"))
    fps = float(frame_rate_num) / float(frame_rate_demon)

    audio_sample_rate = get_audio_sample_rate(args.input_video)

    # Create temporary directory
    temp_dir = "temp_clips"
    os.makedirs(temp_dir, exist_ok=True)

    clips: List[str] = []
    with open(args.timestamps_file, "r") as f:
        timestamps = f.readlines()

    # Extract clips and create transition screens
    for i, line in enumerate(timestamps):
        if not line or line[0] == "#":
            continue
        start_time, end_time = line.replace("\t", " ").strip().split()
        if not all(validate_timestamp(t) for t in [start_time, end_time]):
            raise ValueError(f"Invalid timestamp format in line {i+1}")

        # Extract clip
        clip_file = f"{temp_dir}/clip_{i}.mp4"
        extract_clip(args.input_video, start_time, end_time, clip_file)

        # Create transition screen
        transition = f"{temp_dir}/transition_{i}.mp4"
        create_text_video(
            f"{args.label}\nClip {i + 1}", 3.0, transition, width, height, fps, audio_sample_rate
        )
        clips.append(transition)

        # Add clip number overlay
        numbered_clip = f"{temp_dir}/clip_{i}_numbered.mp4"
        add_clip_number(clip_file, numbered_clip, args.label, i + 1, width, height)
        clips.append(numbered_clip)

    # Create file list for concatenation
    with open(f"{temp_dir}/list.txt", "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.realpath(clip)}'\n")
    print("Doing final join quietly...")
    # Concatenate all clips
    concat_video_clips(f"{temp_dir}/list.txt", f"clips-{args.label}.mp4")

    # Cleanup
    # for file in os.listdir(temp_dir):
    #     os.remove(os.path.join(temp_dir, file))
    # os.rmdir(temp_dir)


if __name__ == "__main__":
    main()

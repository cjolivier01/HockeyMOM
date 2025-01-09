import argparse
import subprocess
import os
from datetime import datetime


def validate_timestamp(timestamp):
    try:
        datetime.strptime(timestamp, "%H:%M:%S")
        return True
    except ValueError:
        return False


# encoder_args = "-c:v hevc_nvenc".split(" ")
encoder_args = "-c:v hevc_nvenc -preset slow -qp 0 -pix_fmt yuv444p".split(" ")


def create_text_video(text: str, duration: int, output_file: str, width: int, height: int) -> None:
    global encoder_args
    cmd = (
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s={}x{}:d={}".format(width, height, duration),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-vf",
            f"drawtext=text='{text}':fontsize=50:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-r",
            "30",
            "-shortest",
        ]
        + encoder_args
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
        + encoder_args
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
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        args.input_video,
    ]
    dimensions = subprocess.check_output(cmd).decode().strip()
    width, height = map(int, dimensions.split("x"))

    # Create temporary directory
    temp_dir = "temp_clips"
    os.makedirs(temp_dir, exist_ok=True)

    clips = []
    with open(args.timestamps_file, "r") as f:
        timestamps = f.readlines()

    # Extract clips and create transition screens
    for i, line in enumerate(timestamps):
        start_time, end_time = line.strip().split()
        if not all(validate_timestamp(t) for t in [start_time, end_time]):
            raise ValueError(f"Invalid timestamp format in line {i+1}")

        # Extract clip
        clip_file = f"{temp_dir}/clip_{i}.mp4"
        cmd = (
            [
                "ffmpeg",
                "-ss",
                start_time,
                "-i",
                args.input_video,
                "-to",
                end_time,
                "-c:a",
                "copy",
            ]
            + encoder_args
            + [
                "-copyts",
                "-y",
                clip_file,
            ]
        )
        subprocess.run(cmd, check=True)

        # Create transition screen
        transition = f"{temp_dir}/transition_{i}.mp4"
        create_text_video(f"{args.label}\nClip {i + 1}", 5, transition, width, height)
        clips.append(transition)

        # Add clip number overlay
        numbered_clip = f"{temp_dir}/clip_{i}_numbered.mp4"
        add_clip_number(clip_file, numbered_clip, args.label, i + 1, width, height)
        clips.append(numbered_clip)

    # Create file list for concatenation
    with open(f"{temp_dir}/list.txt", "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.realpath(clip)}'\n")

    # Concatenate all clips
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{temp_dir}/list.txt",
            "-c:a ",
            "copy",
        ]
        + encoder_args
        + [
            "-copyts",
            "-y",
            "output.mp4",
        ],
        check=True,
    )

    # Cleanup
    # for file in os.listdir(temp_dir):
    #     os.remove(os.path.join(temp_dir, file))
    # os.rmdir(temp_dir)


if __name__ == "__main__":
    main()

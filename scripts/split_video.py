#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


def split_video(input_file, split_count):
    # Ensure split_count is at least 1
    if split_count < 1:
        print("Error: --split-count must be at least 1.")
        sys.exit(1)

    # Get the duration of the video using ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        duration = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration of video: {e.stderr}")
        sys.exit(1)
    except ValueError:
        print("Could not parse duration.")
        sys.exit(1)

    # Calculate segment duration
    segment_duration = duration / split_count

    # Prepare output file names
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    extension = os.path.splitext(input_file)[1]
    output_files = []
    for i in range(split_count):
        output_file = f"{base_name}_part{i+1}{extension}"
        output_files.append(output_file)

    # Split the video
    for i, output_file in enumerate(output_files):
        start_time = segment_duration * i
        # For the last segment, ensure it goes till the end
        if i == split_count - 1:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-i",
                input_file,
                "-ss",
                str(start_time),
                "-c",
                "copy",
                output_file,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-i",
                input_file,
                "-ss",
                str(start_time),
                "-t",
                str(segment_duration),
                "-c",
                "copy",
                output_file,
            ]
        print(f"Creating segment {i+1}/{split_count}: {output_file}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error creating segment {output_file}: {result.stderr}")
            sys.exit(1)

    print("\nVideo successfully split into segments.")

    # Provide instructions to concatenate the parts back
    concat_list = "concat_list.txt"
    with open(concat_list, "w") as f:
        for output_file in output_files:
            # Write file paths, ensuring any special characters are escaped
            f.write(f"file '{output_file}'\n")
    print("\nTo concatenate the parts back into the original video, run the following command:")
    print(f"ffmpeg -f concat -safe 0 -i {concat_list} -c copy output_concat{extension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video file into multiple parts.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument(
        "--split-count", type=int, default=3, help="Number of parts to split the video into."
    )

    args = parser.parse_args()

    input_video = args.input_video
    split_count = args.split_count

    if not os.path.isfile(input_video):
        print(f"Error: The file '{input_video}' does not exist.")
        sys.exit(1)

    split_video(input_video, split_count)

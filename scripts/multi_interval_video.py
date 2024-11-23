"""
Creater a video given intenrals in yaml file, i.e.:

intervals:
  - ["00:00:10.000", "00:00:20.000"]
  - ["00:01:00.000", "00:01:15.000"]

Usage:

    python script.py input.mp4 intervals.yaml intermission.png output.mp4

"""

import argparse
import os
from typing import List, Tuple

import yaml
from video.ffmpeg import BasicVideoInfo


def create_intermission(image_file: str, duration: int = 5, fps: float = 29.97) -> str:
    """
    Creates a video of a static image that loops for a specified duration.

    Args:
        image_file (str): Path to the image file to use for intermission.
        duration (int): Duration in seconds for the intermission.

    Returns:
        str: Path to the created intermission video file.
    """
    intermission_file = "intermission.mp4"
    os.system(
        f"ffmpeg -loop 1 -i {image_file} -c:v libx264 -t {duration} -f {fps} {intermission_file}"
    )
    return intermission_file


def extract_video_segments(video_file: str, intervals: List[Tuple[str, str]]) -> List[str]:
    """
    Extracts video segments from the input video based on start and end times.

    Args:
        video_file (str): Path to the input video file.
        intervals (List[Tuple[str, str]]): List of tuples containing start and end times.

    Returns:
        List[str]: A list of filenames of the extracted video segments.
    """
    segment_files = []
    for i, (start, end) in enumerate(intervals):
        segment_file = f"segment{i+1}.mp4"
        os.system(
            f"ffmpeg -i {video_file} -ss {start} -to {end} -c:a copy -c:v copy {segment_file}"
        )
        segment_files.append(segment_file)
    return segment_files


def create_concat_file(segment_files: List[str], intermission_file: str) -> None:
    """
    Creates a concat list file to concatenate segments and intermissions.

    Args:
        segment_files (List[str]): List of segment video file paths.
        intermission_file (str): Path to the intermission video file.
    """
    with open("concat_list.txt", "w") as f:
        for segment in segment_files:
            f.write(f"file '{segment}'\n")
            f.write(f"file '{intermission_file}'\n")  # Intermission after every segment


def concatenate_videos(output_file: str) -> None:
    """
    Concatenates the video segments and intermissions into a final video.

    Args:
        output_file (str): Path to the final output video file.
    """
    os.system(f"ffmpeg -f concat -safe 0 -i concat_list.txt -c:a copy -c:v copy {output_file}")


def read_yaml_intervals(yaml_file: str) -> List[Tuple[str, str]]:
    """
    Reads intervals from a YAML file.

    Args:
        yaml_file (str): Path to the YAML file containing intervals.

    Returns:
        List[Tuple[str, str]]: A list of tuples with start and end times.
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data.get("intervals", [])


def main() -> None:
    """
    Main function to parse command-line arguments and create a video with intermissions.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create a video with intermissions from specific intervals."
    )
    parser.add_argument("input_video", type=str, help="Input video file (mp4 format)")
    parser.add_argument(
        "yaml_file", type=str, help="YAML file containing intervals in HH:MM:SS.ssss format"
    )
    parser.add_argument(
        "intermission_image", type=str, help="Image file to use for the intermission"
    )
    parser.add_argument("output_video", type=str, help="Output video file name")

    args = parser.parse_args()

    video_info = BasicVideoInfo(args.input_video)

    # Read intervals from the YAML file
    intervals: List[Tuple[str, str]] = read_yaml_intervals(args.yaml_file)

    # Create the intermission video
    intermission_file: str = create_intermission(args.intermission_image, fps=video_info.fps)

    # Extract video segments based on the intervals
    segment_files: List[str] = extract_video_segments(args.input_video, intervals)

    # Create the concat file
    create_concat_file(segment_files, intermission_file)

    # Concatenate videos and create the final output
    concatenate_videos(args.output_video)

    print(f"Video created successfully: {args.output_video}")

# ./p scripts/multi_interval_video.py ~/Videos/ev-blackstars-ps/tracking_output-with-audio.mp4 ~/Videos/ev-blackstars-ps/ethan_dom.yaml ~/Videos/ev-blackstars-ps/xor_file.png output.mp4
if __name__ == "__main__":
    main()

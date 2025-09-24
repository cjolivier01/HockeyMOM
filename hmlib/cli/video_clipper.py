"""
Given a video, timestamp file of:

start_hh_mm_ss end_hh_mm_ss
start_hh_mm_ss end_hh_mm_ss
...

.. and a text label...

Create transition text screens to separate and join all the clips into a new
video along with text labeling of the clip number and user-designated label
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def validate_timestamp(timestamp):
    try:
        datetime.strptime(timestamp, "%H:%M:%S")
        return True
    except ValueError:
        return False


_DEBUG = True

# PIXEL_FORMAT = "-pix_fmt yuv444p"
# ENCODER_ARGS_LOSSLESS = "-c:v hevc_nvenc -preset slow -qp 0 -pix_fmt yuv444p".split(" ")
# ENCODER_ARGS_LOSSLESS = "-c:v hevc_nvenc -preset slow -qp 0".split(" ")
# ENCODER_ARGS_LOSSLESS = "-c:v hevc_nvenc -b:v 40M -preset p4".split(" ")
ENCODER_ARGS_LOSSLESS = "-c:v hevc_nvenc -preset p4 -rc constqp -qp 0".split(" ")
# ENCODER_ARGS_FAST = "-c:v hevc_nvenc -preset ultrafast -crf 23".split(" ")
ENCODER_ARGS_FAST = "-c:v mpeg4 -preset slow -crf 2".split(" ")
ENCODER_ARGS_HQ = f"-c:v hevc_nvenc -preset medium -b:v 40M".split(" ")

FFMPEG_CUDA_DECODER = ["-c:v", "hevc_cuvid"]

if not _DEBUG or int(os.environ.get("VIDEO_CLIPPER_HQ", "0")) > 0:
    print("Using lossless encoding for intermediate clips (slow)")
    WORKING_ENCODER_ARGS = ENCODER_ARGS_LOSSLESS
else:
    WORKING_ENCODER_ARGS = ENCODER_ARGS_HQ

FFMPEG_BASE = ["ffmpeg", "-hide_banner"]
FFMPEG_BASE_HW: List[str] = FFMPEG_BASE + ["-hwaccel", "cuda"]


def friendly_label(label: str) -> str:
    return label.replace("_", " ")


def escape_drawtext(text: str) -> str:
    """
    Escapes text for ffmpeg drawtext. Handles quotes, colons, backslashes, and newlines.
    """
    # ffmpeg drawtext likes: escape backslashes, colons, quotes; newlines as \n
    t = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    t = t.replace("\n", "\\n")
    return t


def hhmmss_to_duration_seconds(time_str: str) -> float:
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
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds


def get_audio_sample_rate(file_path: str):
    try:
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
        output = json.loads(result.stdout)
        sample_rate = output["streams"][0]["sample_rate"]
        return int(sample_rate)
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as e:
        print(f"Error retrieving sample rate: {e}")
        return None


def concat_video_clips(list_file: str, output_file: str) -> None:
    subprocess.run(
        FFMPEG_BASE
        + FFMPEG_CUDA_DECODER
        + [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
        ]
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
    duration: float,
    output_file: str,
    width: int,
    height: int,
    fps: float,
    audio_sample_rate: Optional[int],
) -> None:
    text = friendly_label(text)
    etext = escape_drawtext(text)
    cmd = FFMPEG_BASE_HW + [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={int(audio_sample_rate) if audio_sample_rate else 48000}:cl=stereo",
        "-vf",
        f"drawtext=text='{etext}':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
        "-r",
        str(fps),
        "-shortest",
    ]
    cmd += WORKING_ENCODER_ARGS + [
        "-c:a",
        "aac",
        "-y",
        output_file,
    ]
    subprocess.run(cmd, check=True)


def extract_clip_with_overlay(
    *,
    input_video: str,
    start_time: Optional[str],  # None means from start (filelist mode)
    end_time: Optional[str],  # None means till end (filelist mode)
    output_file: str,
    dest_fps: float,
    label: str,
    clip_number: int,
    top_right_margin: int = 10,
) -> None:
    """
    Single-pass: cut (optionally), overlay clip-number text, set fps/format, encode.
    """
    # Compose overlay text
    overlay_text = f"{friendly_label(label)} {clip_number}"
    etext = escape_drawtext(overlay_text)

    # Build the filter chain: fps & format before drawtext to stabilize glyph rasterization
    vf_chain = f"fps={dest_fps},format=nv12,drawtext=text='{etext}':fontsize=52:fontcolor=white:x=w-tw-{top_right_margin}:y={top_right_margin}"

    cmd = list(FFMPEG_BASE_HW)
    # Decoding via cuvid if available (you had it globally as a list; attach per-input)
    if start_time:
        cmd += ["-ss", start_time]
    cmd += FFMPEG_CUDA_DECODER + ["-i", input_video]
    if start_time and end_time:
        duration = hhmmss_to_duration_seconds(end_time) - hhmmss_to_duration_seconds(start_time)
        cmd += ["-t", f"{duration}"]

    cmd += [
        "-vf",
        vf_chain,
    ]
    # Encode video once; use AAC for reliable cuts across containers/codecs
    cmd += WORKING_ENCODER_ARGS + [
        "-c:a",
        "aac",
        "-y",
        output_file,
    ]

    subprocess.run(cmd, check=True)


def _process_clip_from_timestamps(
    idx: int,
    *,
    input_video: str,
    start_time: str,
    end_time: str,
    label: str,
    temp_dir: str,
    width: int,
    height: int,
    fps: float,
    audio_sample_rate: int,
) -> list[str]:
    # Transition screen
    transition = f"{temp_dir}/transition_{idx}.mp4"
    create_text_video(
        f"{label} Clip {idx + 1}",
        3.0,
        transition,
        width,
        height,
        fps,
        audio_sample_rate,
    )

    # Fused extract + overlay
    numbered_clip = f"{temp_dir}/clip_{idx}_numbered.mp4"
    extract_clip_with_overlay(
        input_video=input_video,
        start_time=start_time,
        end_time=end_time if end_time else None,
        output_file=numbered_clip,
        dest_fps=fps,
        label=label,
        clip_number=idx + 1,
    )

    return [transition, numbered_clip]


def _process_clip_from_filelist(
    idx: int,
    *,
    clip_file: str,
    label: str,
    temp_dir: str,
    width: int,
    height: int,
    fps: float,
    audio_sample_rate: int,
) -> list[str]:
    # Transition screen
    transition = f"{temp_dir}/transition_{idx}.mp4"
    create_text_video(
        f"{label} Clip {idx + 1}",
        3.0,
        transition,
        width,
        height,
        fps,
        audio_sample_rate,
    )

    # Fused overlay-only (no trimming)
    numbered_clip = f"{temp_dir}/clip_{idx}_numbered.mp4"
    extract_clip_with_overlay(
        input_video=clip_file,
        start_time=None,
        end_time=None,
        output_file=numbered_clip,
        dest_fps=fps,
        label=label,
        clip_number=idx + 1,
    )

    return [transition, numbered_clip]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=None, help="Input video file")
    parser.add_argument("--timestamps", "-t", default=None, help="File containing timestamps")
    parser.add_argument("--quick", type=int, default=0, help="Quick mode (lower quality)")
    parser.add_argument("--video-file-list", type=str, default=None, help="List of video files")
    parser.add_argument("--temp-dir", type=str, default=None, help="Directory to store intermediate clips")
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        default=1,
        help="Maximum number of clips to process in parallel",
    )
    parser.add_argument("label", help="Text label for transitions")
    args = parser.parse_args()

    # Override encoder selection based on flags/env at runtime
    # global WORKING_ENCODER_ARGS
    # try:
    #     q = int(args.quick) if args.quick is not None else 0
    # except Exception:
    #     q = 0
    # if q > 0:
    #     WORKING_ENCODER_ARGS = ENCODER_ARGS_FAST
    # elif int(os.environ.get("VIDEO_CLIPPER_HQ", "0")) > 0:
    #     WORKING_ENCODER_ARGS = ENCODER_ARGS_LOSSLESS

    if args.video_file_list:
        if os.path.isfile(args.video_file_list):
            with open(args.video_file_list, "r") as f:
                args.video_file_list = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        else:
            args.video_file_list = args.video_file_list.split(",")

    if not args.video_file_list and not (args.input and args.timestamps):
        print("--video-file-list or both --input and --timestamps must be provided")
        exit(1)

    probe_video = args.input if args.input else args.video_file_list[0]

    # Get video dimensions and fps
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
        probe_video,
    ]
    fprobe_output = subprocess.check_output(cmd).decode().strip()
    ffprobe_results = fprobe_output.split("x")
    assert len(ffprobe_results) >= 3
    width = int(ffprobe_results[0])
    height = int(ffprobe_results[1])
    frame_rate_num, frame_rate_demon = map(int, ffprobe_results[2].split("/"))
    fps = float(frame_rate_num) / float(frame_rate_demon)

    audio_sample_rate = get_audio_sample_rate(probe_video)

    # Create temporary directory
    temp_dir = args.temp_dir if args.temp_dir else "temp_clips"
    os.makedirs(temp_dir, exist_ok=True)

    clips: List[str] = []
    max_workers = max(1, int(args.threads)) if args.threads is not None else 1

    if args.video_file_list:
        jobs = [(i, clip_file) for i, clip_file in enumerate(args.video_file_list)]
        results: dict[int, list[str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _process_clip_from_filelist,
                    i,
                    clip_file=clip_file,
                    label=args.label,
                    temp_dir=temp_dir,
                    width=width,
                    height=height,
                    fps=fps,
                    audio_sample_rate=audio_sample_rate,
                ): i
                for i, clip_file in jobs
            }
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                results[idx] = fut.result()
        for i in sorted(results.keys()):
            clips.extend(results[i])
    else:
        with open(args.timestamps, "r") as f:
            raw_lines = f.readlines()

        # Build timestamp jobs in order, skipping comments/blank lines
        ts_jobs: list[tuple[int, str, str]] = []
        for i, line in enumerate(raw_lines):
            line = re.sub(r"\s+", " ", line)
            line = line.strip()
            if not line or line[0] == "#":
                continue
            print(f"{line=}")
            time_tokens = line.replace("\t", " ").strip().split(" ")

            end_time = ""
            start_time = time_tokens[0]
            if len(time_tokens) > 1:
                end_time = time_tokens[1]

            if not all(validate_timestamp(t) for t in time_tokens):
                raise ValueError(f"Invalid timestamp format in line {i+1}: {time_tokens}")

            ts_jobs.append((i, start_time, end_time))

        results: dict[int, list[str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _process_clip_from_timestamps,
                    i,
                    input_video=args.input,
                    start_time=start_time,
                    end_time=end_time,
                    label=args.label,
                    temp_dir=temp_dir,
                    width=width,
                    height=height,
                    fps=fps,
                    audio_sample_rate=audio_sample_rate,
                ): i
                for (i, start_time, end_time) in ts_jobs
            }
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                results[idx] = fut.result()
        for i in sorted(results.keys()):
            clips.extend(results[i])

    # Create file list for concatenation
    with open(f"{temp_dir}/list.txt", "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.realpath(clip)}'\n")
    print("Doing final join quietly...")
    # Concatenate all clips
    label = args.label.replace(" ", "_")
    concat_video_clips(f"{temp_dir}/list.txt", f"clips-{label}.mp4")

    # Cleanup (optional)
    # for file in os.listdir(temp_dir):
    #     os.remove(os.path.join(temp_dir, file))
    # os.rmdir(temp_dir)


if __name__ == "__main__":
    main()

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
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


def validate_timestamp(timestamp):
    try:
        datetime.strptime(timestamp, "%H:%M:%S")
        return True
    except ValueError:
        return False


_DEBUG = True

# Encoder presets for HEVC and H.264. We keep the same quality tiers where possible.
ENCODER_ARGS_LOSSLESS_HEVC = "-c:v hevc_nvenc -preset p4 -rc constqp -qp 0".split(" ")
ENCODER_ARGS_HQ_HEVC = "-c:v hevc_nvenc -preset medium -b:v 40M".split(" ")

# H.264 NVENC: mirror HEVC quality tiers as closely as possible
ENCODER_ARGS_LOSSLESS_H264 = "-c:v h264_nvenc -preset p4 -rc constqp -qp 0".split(" ")
ENCODER_ARGS_HQ_H264 = "-c:v h264_nvenc -preset medium -b:v 40M".split(" ")

# "Fast" mode kept as-is for backwards-compat (intended for quick iteration)
ENCODER_ARGS_FAST = "-c:v mpeg4 -preset slow -crf 2".split(" ")

# Default working encoder (overridden in main based on --codec/flags)
WORKING_ENCODER_ARGS = ENCODER_ARGS_HQ_HEVC

FFMPEG_BASE = ["ffmpeg", "-hide_banner"]
FFMPEG_BASE_HW: List[str] = FFMPEG_BASE + ["-hwaccel", "cuda"]


def get_video_codec(file_path: str, dry_run: bool = False) -> Optional[str]:
    """
    Returns codec_name (e.g., "hevc", "h264") of the first video stream.
    In --dry-run mode, returns None (no assumption).
    """
    if dry_run:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "json",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return data.get("streams", [{}])[0].get("codec_name")
    except Exception:
        return None


def get_decoder_args_for_video(file_path: str, dry_run: bool = False) -> List[str]:
    """
    Choose CUDA decoder based on detected codec. If unknown, return [].
    """
    codec = get_video_codec(file_path, dry_run=dry_run)
    if not codec:
        return []
    codec = codec.lower()
    if codec in ("hevc", "h265"):  # common names for H.265
        return ["-c:v", "hevc_cuvid"]
    if codec in ("h264", "avc1", "avc"):  # common names for H.264
        return ["-c:v", "h264_cuvid"]
    return []


def friendly_label(label: str) -> str:
    return label.replace("_", " ")


def escape_drawtext(text: str) -> str:
    """
    Escapes text for ffmpeg drawtext. Handles quotes, colons, backslashes, and newlines.
    """
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


def format_cmd(cmd: List[str]) -> str:
    # Pretty print a shell-safe string for dry-run output
    return " ".join(shlex.quote(c) for c in cmd)


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    if dry_run:
        print("[DRY-RUN]", format_cmd(cmd))
        return
    subprocess.run(cmd, check=True)


def get_audio_sample_rate(file_path: str, dry_run: bool = False) -> Optional[int]:
    """
    Returns the sample rate (Hz) of the first audio stream, or None if not found.
    In --dry-run mode, returns a reasonable default (48000) without probing.
    """
    if dry_run:
        return 48000
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
            check=True,
        )
        output = json.loads(result.stdout)
        sample_rate = output["streams"][0]["sample_rate"]
        return int(sample_rate)
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError):
        return None


def get_media_duration_seconds(file_path: str, dry_run: bool = False) -> Optional[float]:
    """
    Returns duration (float seconds) using ffprobe format=duration.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        file_path,
    ]
    if dry_run:
        print("[DRY-RUN]", format_cmd(cmd))
        # In dry-run, we cannot know; return None so caller doesn't rely on it.
        return None
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def durations_close(a: Optional[float], b: Optional[float], tol: float = 0.05) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def should_skip_existing(
    output_file: str, expected_duration: Optional[float], cont: bool, dry_run: bool, tol: float = 0.05
) -> bool:
    """
    If --continue is enabled and output exists and duration matches expected, skip.
    """
    if not cont:
        return False
    if not os.path.exists(output_file):
        return False
    actual = get_media_duration_seconds(output_file, dry_run=dry_run)
    if expected_duration is None:
        # Without an expected duration, be conservative: don't skip.
        return False
    if durations_close(actual, expected_duration, tol=tol):
        print(f"[CONTINUE] Skipping existing {output_file} (duration matches ~{expected_duration:.3f}s)")
        return True
    # Otherwise, remake
    print(f"[CONTINUE] Rebuilding {output_file} (duration mismatch: have {actual}, want {expected_duration})")
    return False


def concat_video_clips(
    list_file: str, output_file: str, dry_run: bool, cont: bool, expected_concat_duration: Optional[float]
) -> None:
    if should_skip_existing(output_file, expected_concat_duration, cont, dry_run):
        return
    # Concatenation uses stream copy; no decode needed from list file.
    cmd = (
        FFMPEG_BASE
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
        ]
    )
    run_cmd(cmd, dry_run)


def create_text_video(
    text: str,
    duration: float,
    output_file: str,
    width: int,
    height: int,
    fps: float,
    audio_sample_rate: Optional[int],
    *,
    dry_run: bool,
    cont: bool,
) -> Tuple[str, float]:
    """
    Creates a transition screen. Returns (path, expected_duration).
    Applies --dry-run and --continue skip.
    """
    if should_skip_existing(output_file, duration, cont, dry_run):
        return output_file, duration

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
    run_cmd(cmd, dry_run)
    return output_file, duration


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
    dry_run: bool,
    cont: bool,
    expected_duration: Optional[float],  # caller computes/guesses
) -> Tuple[str, Optional[float]]:
    """
    Single-pass: cut (optionally), overlay clip-number text, set fps/format, encode.
    Returns (path, expected_duration).
    """
    if should_skip_existing(output_file, expected_duration, cont, dry_run):
        return output_file, expected_duration

    overlay_text = f"{friendly_label(label)} {clip_number}"
    etext = escape_drawtext(overlay_text)
    vf_chain = f"fps={dest_fps},format=nv12,drawtext=text='{etext}':fontsize=52:fontcolor=white:x=w-tw-{top_right_margin}:y={top_right_margin}"

    cmd = list(FFMPEG_BASE_HW)
    if start_time:
        cmd += ["-ss", start_time]
    # Detect and select the appropriate CUDA decoder (H.264 or HEVC) per input
    cmd += get_decoder_args_for_video(input_video, dry_run=dry_run) + ["-i", input_video]
    if start_time and end_time:
        dur = hhmmss_to_duration_seconds(end_time) - hhmmss_to_duration_seconds(start_time)
        cmd += ["-t", f"{dur}"]

    cmd += [
        "-vf",
        vf_chain,
    ]
    cmd += WORKING_ENCODER_ARGS + [
        "-c:a",
        "aac",
        "-y",
        output_file,
    ]
    run_cmd(cmd, dry_run)
    return output_file, expected_duration


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
    dry_run: bool,
    cont: bool,
) -> list[str]:
    # Transition screen
    transition = f"{temp_dir}/transition_{idx}.mp4"
    t_path, t_dur = create_text_video(
        f"{label} Clip {idx + 1}",
        3.0,
        transition,
        width,
        height,
        fps,
        audio_sample_rate,
        dry_run=dry_run,
        cont=cont,
    )

    # Expected duration for the numbered clip from timestamps
    exp_dur = None
    if end_time:
        exp_dur = hhmmss_to_duration_seconds(end_time) - hhmmss_to_duration_seconds(start_time)

    # Fused extract + overlay
    numbered_clip = f"{temp_dir}/clip_{idx}_numbered.mp4"
    n_path, n_dur = extract_clip_with_overlay(
        input_video=input_video,
        start_time=start_time,
        end_time=end_time if end_time else None,
        output_file=numbered_clip,
        dest_fps=fps,
        label=label,
        clip_number=idx + 1,
        dry_run=dry_run,
        cont=cont,
        expected_duration=exp_dur,
    )

    return [t_path, n_path]


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
    dry_run: bool,
    cont: bool,
) -> list[str]:
    # Transition screen
    transition = f"{temp_dir}/transition_{idx}.mp4"
    t_path, t_dur = create_text_video(
        f"{label}\nClip {idx + 1}",
        3.0,
        transition,
        width,
        height,
        fps,
        audio_sample_rate,
        dry_run=dry_run,
        cont=cont,
    )

    # Expected duration for overlay-only is duration of source file
    src_dur = get_media_duration_seconds(clip_file, dry_run=dry_run)

    # Fused overlay-only (no trimming)
    numbered_clip = f"{temp_dir}/clip_{idx}_numbered.mp4"
    n_path, n_dur = extract_clip_with_overlay(
        input_video=clip_file,
        start_time=None,
        end_time=None,
        output_file=numbered_clip,
        dest_fps=fps,
        label=label,
        clip_number=idx + 1,
        dry_run=dry_run,
        cont=cont,
        expected_duration=src_dur,
    )

    return [t_path, n_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=None, help="Input video file")
    parser.add_argument("--timestamps", "-t", default=None, help="File containing timestamps")
    parser.add_argument("--quick", type=int, default=0, help="Quick mode (lower quality)")
    parser.add_argument(
        "--codec",
        choices=["hevc", "h264"],
        default="hevc",
        help="Output codec: hevc (H.265) or h264 (H.264). Default: hevc",
    )
    parser.add_argument("--video-file-list", type=str, default=None, help="List of video files")
    parser.add_argument("--temp-dir", type=str, default=None, help="Directory to store intermediate clips")
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        default=1,
        help="Maximum number of clips to process in parallel",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of executing them")
    parser.add_argument(
        "--continue",
        dest="cont",
        action="store_true",
        help="Skip remaking mp4s that already exist when durations match",
    )
    parser.add_argument("label", help="Text label for transitions")
    args = parser.parse_args()

    # Encoder selection based on codec + flags/env
    global WORKING_ENCODER_ARGS
    try:
        q = int(args.quick) if args.quick is not None else 0
    except Exception:
        q = 0
    if q > 0:
        WORKING_ENCODER_ARGS = ENCODER_ARGS_FAST
    else:
        # VIDEO_CLIPPER_HQ>0 means use lossless for intermediates (existing behavior)
        if int(os.environ.get("VIDEO_CLIPPER_HQ", "0")) > 0:
            WORKING_ENCODER_ARGS = (
                ENCODER_ARGS_LOSSLESS_H264 if args.codec == "h264" else ENCODER_ARGS_LOSSLESS_HEVC
            )
        else:
            WORKING_ENCODER_ARGS = ENCODER_ARGS_HQ_H264 if args.codec == "h264" else ENCODER_ARGS_HQ_HEVC

    # Parse video file list
    if args.video_file_list:
        if os.path.isfile(args.video_file_list):
            with open(args.video_file_list, "r") as f:
                args.video_file_list = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        else:
            args.video_file_list = [s.strip() for s in args.video_file_list.split(",") if s.strip()]

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
    if args.dry_run:
        print("[DRY-RUN]", format_cmd(cmd))
        # In dry-run, assume some safe defaults to let us print commands sensibly
        width, height, fps = 1920, 1080, 30.0
    else:
        fprobe_output = subprocess.check_output(cmd).decode().strip()
        ffprobe_results = fprobe_output.split("x")
        assert len(ffprobe_results) >= 3
        width = int(ffprobe_results[0])
        height = int(ffprobe_results[1])
        frame_rate_num, frame_rate_demon = map(int, ffprobe_results[2].split("/"))
        fps = float(frame_rate_num) / float(frame_rate_demon)

    audio_sample_rate = get_audio_sample_rate(probe_video) if not args.dry_run else 48000

    # Create temporary directory
    temp_dir = args.temp_dir if args.temp_dir else "temp_clips"
    os.makedirs(temp_dir, exist_ok=True)

    clips: List[str] = []
    # We also track expected durations to decide concat skip
    clip_expected_durations: List[Optional[float]] = []

    max_workers = max(1, int(args.threads)) if args.threads is not None else 1

    if args.video_file_list:
        jobs = [(i, clip_file) for i, clip_file in enumerate(args.video_file_list)]
        results: dict[int, Tuple[list[str], list[Optional[float]]]] = {}
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
                    dry_run=args.dry_run,
                    cont=args.cont,
                ): i
                for i, clip_file in jobs
            }
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                paths = fut.result()
                results[idx] = (paths, [3.0, get_media_duration_seconds(paths[1], dry_run=args.dry_run)])
        for i in sorted(results.keys()):
            pths, durs = results[i]
            clips.extend(pths)
            clip_expected_durations.extend(durs)
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

        results: dict[int, Tuple[list[str], list[Optional[float]]]] = {}
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
                    dry_run=args.dry_run,
                    cont=args.cont,
                ): i
                for (i, start_time, end_time) in ts_jobs
            }
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                paths = fut.result()
                # durations: transition=3.0, numbered=expected (from timestamps)
                start_time = ts_jobs[idx][1]
                end_time = ts_jobs[idx][2]
                exp_dur = None
                if end_time:
                    exp_dur = hhmmss_to_duration_seconds(end_time) - hhmmss_to_duration_seconds(start_time)
                results[idx] = (paths, [3.0, exp_dur])
        for i in sorted(results.keys()):
            pths, durs = results[i]
            clips.extend(pths)
            clip_expected_durations.extend(durs)

    # Create file list for concatenation
    list_file = f"{temp_dir}/list.txt"
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.realpath(clip)}'\n")
    print("Doing final join quietly...")

    # Expected concat duration = sum of expected durations (best effort)
    # If any unknown (None), try probing each individual file (unless dry-run).
    if any(d is None for d in clip_expected_durations) and not args.dry_run:
        clip_expected_durations = [
            d if d is not None else get_media_duration_seconds(p, dry_run=False)
            for p, d in zip(clips, clip_expected_durations)
        ]
    expected_concat_duration = None
    if all(d is not None for d in clip_expected_durations) and len(clip_expected_durations) > 0:
        expected_concat_duration = float(sum(d for d in clip_expected_durations if d is not None))

    # Concatenate all clips
    label = args.label.replace(" ", "_")
    final_out = f"clips-{label}.mp4"
    concat_video_clips(
        list_file,
        final_out,
        dry_run=args.dry_run,
        cont=args.cont,
        expected_concat_duration=expected_concat_duration,
    )

    # Cleanup (optional)
    # for file in os.listdir(temp_dir):
    #     os.remove(os.path.join(temp_dir, file))
    # os.rmdir(temp_dir)


if __name__ == "__main__":
    main()

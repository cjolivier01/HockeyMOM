import os
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import cv2

from hmlib.analytics.analyze_jerseys import analyze_data
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.config import get_game_dir
from hmlib.hm_opts import hm_opts
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils.image import image_height, image_width
from hmlib.utils.time import format_duration_to_hhmmss


def load_player_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return TrackingDataFrame(input_file=os.path.join(game_dir, "tracking.csv"), input_batch_size=1)


def load_camera_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return CameraTrackingDataFrame(
        input_file=os.path.join(game_dir, "camera.csv"), input_batch_size=1
    )


def get_uncropped_width_height(game_id: str) -> Tuple[int, int]:
    game_dir: str = get_game_dir(game_id=game_id)
    path = os.path.join(game_dir, "s.png")
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return image_width(img), image_height(img)


def merge_intervals(
    intervals: List[Tuple[float, Set[int]]]
) -> Dict[int, List[Tuple[float, float]]]:
    # Dictionary to store the merged intervals for each jersey number
    jersey_times: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    # Process each interval
    for i in range(len(intervals)):
        start, jerseys = intervals[i]
        end = float("inf") if i == len(intervals) - 1 else intervals[i + 1][0]

        for jersey in jerseys:
            if not jersey_times[jersey] or jersey_times[jersey][-1][1] < start:
                # If no interval exists for this jersey, or there's no overlap
                jersey_times[jersey].append([start, end])
            else:
                # Extend the current interval
                jersey_times[jersey][-1][1] = end

    # Convert list of intervals to (start, duration) format and prepare the result
    result: Dict[int, List[Tuple[float, float]]] = {}
    for jersey, times in jersey_times.items():
        merged: List[Tuple[float, float]] = []
        for time in times:
            if merged and merged[-1][1] == time[0]:
                # Merge with the previous interval
                merged[-1] = (merged[-1][0], time[1] - merged[-1][0])
            else:
                # Append a new interval entry
                merged.append((time[0], time[1] - time[0]))
        result[jersey] = merged

    return result


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    if not args.game_id:
        args.game_id = "ev-stockton-1"
    try:
        wh = get_uncropped_width_height(game_id=args.game_id)
        uncropped_width = wh[0] if wh else 8803
        player_tracking_data = load_player_tracking_data(game_id=args.game_id)
        camera_tracking_data = load_camera_tracking_data(game_id=args.game_id)
        roster = {29, 37, 40, 98, 73, 89, 54, 24, 79, 16, 27, 90, 57, 8, 96, 74}
        start_interval_and_jerseys = analyze_data(
            player_tracking_data,
            camera_tracking_data,
            uncropped_width=uncropped_width,
            roster=roster,
        )
        for st, jr in start_interval_and_jerseys:
            start_time_hhmmss = format_duration_to_hhmmss(st, decimals=0)
            jersey_numbers = sorted(list(jr))
            print(
                f"Interval starting at {start_time_hhmmss} finds {len(jersey_numbers)} jerseys: {jersey_numbers}"
            )

        in_file_basename = "tracking_output-with-audio.mp4"
        game_dir = get_game_dir(args.game_id)
        input_file = os.path.join(game_dir, in_file_basename)
        shell_file = os.path.join(game_dir, "make_player_highlights.sh")
        with open(shell_file, "w") as sf:
            sf.write("#!/bin/bash\n")
            sf.write("set +x\n")
            sf.write("set +e\n")
            sf.write("\n")
            player_intervals = merge_intervals(start_interval_and_jerseys)
            for player, intervals in player_intervals.items():
                print(f"Player: {player}")
                player_file: str = f"player_{player}.txt"
                player_file_path: str = os.path.join(game_dir, player_file)
                with open(player_file_path, "w") as f:
                    for i, (st, dur) in enumerate(intervals):
                        f.write(f"file '{input_file}'\n")
                        f.write(f"inpoint {st}\n")
                        if dur == float("inf"):
                            dur = "..."
                        else:
                            f.write(f"outpoint {st + dur}\n")
                        start_time_hhmmss = format_duration_to_hhmmss(st, decimals=0)
                        print(
                            f"\tShift {i}: Starts at {start_time_hhmmss}, shift for {dur} seconds"
                        )
                player_video = f"player_{player}.mp4"
                ffmpeg_command = f'ffmpeg -f concat -safe 0 -segment_time_metadata 1 -i {player_file_path}  -b:v 5M -c:v hevc_nvenc -vf "select=concatdec_select" -af "aselect=concatdec_select,aresample=async=1" {player_video}'
                print(ffmpeg_command)
                sf.write(f"{ffmpeg_command}\n")
        print(f"Saved player highlight creation script to {shell_file}")

    except Exception:
        traceback.print_exc()

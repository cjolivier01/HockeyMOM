import math
import os
import traceback
from typing import Any, Dict, Tuple

import cv2

from hmlib.analytics.analyze_jerseys import (
    analyze_data,
    interval_jerseys_to_merged_jersey_time_intervals,
)
from hmlib.aspen.plugins.load_plugins import LoadCameraPlugin
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.config import get_game_dir
from hmlib.hm_opts import hm_opts
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils.image import image_height, image_width
from hmlib.utils.time import format_duration_to_hhmmss


def load_player_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return TrackingDataFrame(input_file=os.path.join(game_dir, "tracking.csv"), input_batch_size=1)


def load_camera_tracking_data(game_id: str) -> CameraTrackingDataFrame:
    """
    Load camera tracking data for a game.

    Prefer the Aspen LoadCameraPlugin path (to reuse its path discovery logic),
    but fall back to opening camera.csv directly when needed.
    """
    game_dir: str = get_game_dir(game_id=game_id)

    # Try Aspen LoadCameraPlugin first so we share camera.csv discovery logic
    try:
        trunk = LoadCameraPlugin()
        ctx = {"frame_id": 1, "game_dir": game_dir, "shared": {"game_dir": game_dir}}
        out = trunk(ctx)
        df = out.get("camera_dataframe")
        if isinstance(df, CameraTrackingDataFrame):
            return df
    except Exception:
        # Fall back to direct CSV load below
        traceback.print_exc()

    # Direct CSV path fallback
    return CameraTrackingDataFrame(
        input_file=os.path.join(game_dir, "camera.csv"),
        input_batch_size=1,
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


def main():
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
        period_start_interval_and_jerseys, shift_start_interval_and_jerseys = analyze_data(
            player_tracking_data,
            camera_tracking_data,
            uncropped_width=uncropped_width,
            roster=roster,
        )

        for interval_jerseys in period_start_interval_and_jerseys:
            start_time_hhmmss = format_duration_to_hhmmss(interval_jerseys.start_time, decimals=0)
            jersey_numbers = sorted(list(interval_jerseys.jersey_numbers))
            print(
                f"Period Interval starting at {start_time_hhmmss} finds {len(jersey_numbers)} jerseys: {jersey_numbers}"
            )

        for interval_jerseys in shift_start_interval_and_jerseys:
            start_time_hhmmss = format_duration_to_hhmmss(interval_jerseys.start_time, decimals=0)
            jersey_numbers = sorted(list(interval_jerseys.jersey_numbers))
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
            player_shift_intervals = interval_jerseys_to_merged_jersey_time_intervals(
                shift_start_interval_and_jerseys
            )
            for jersey_time_interval in player_shift_intervals:
                player = jersey_time_interval.jersey_number
                print(f"Player: {player}")
                player_file: str = f"player_{player}.txt"
                player_file_path: str = os.path.join(game_dir, player_file)
                with open(player_file_path, "w") as f:
                    for i, time_interval in enumerate(jersey_time_interval.intervals):
                        interval_start_adjust = -3.0
                        st = time_interval.start_time + interval_start_adjust  # Go back 3 seconds
                        st = max(0, st)
                        if True:
                            end = time_interval.start_time + time_interval.duration
                            if math.isfinite(st):
                                from_time: str = format_duration_to_hhmmss(st, decimals=0)
                                if math.isfinite(end):
                                    to_time: str = format_duration_to_hhmmss(end, decimals=0)
                                else:
                                    to_time = ""
                                print(f"{from_time}    {to_time}")
                                f.write(f"{from_time}    {to_time}\n")
                            else:
                                print(f"Bad range: {st=}, {end=}")
                        else:
                            dur = time_interval.duration - interval_start_adjust
                            f.write(f"file '{input_file}'\n")
                            f.write(f"inpoint {st}\n")
                            if dur == float("inf"):
                                dur = "..."
                                end_time_hhmmss = "..."
                            else:
                                f.write(f"outpoint {st + dur}\n")
                                end_time_hhmmss = format_duration_to_hhmmss(st + dur, decimals=0)
                            start_time_hhmmss = format_duration_to_hhmmss(st, decimals=0)
                            print(
                                f"\tShift {i}: From {start_time_hhmmss} to {end_time_hhmmss}, shift for {dur} seconds"
                            )
                # player_video = f"player_{player}.mkv"
                # ffmpeg_command = f'ffmpeg -f concat -safe 0 -segment_time_metadata 1 -i {player_file_path}  -b:v 5M -c:v hevc_nvenc -vf "select=concatdec_select" -af "aselect=concatdec_select,aresample=async=1" {player_video}'
                # print(ffmpeg_command)
                # sf.write(f"{ffmpeg_command}\n")
        print(f"Saved player highlight creation script to {shell_file}")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()

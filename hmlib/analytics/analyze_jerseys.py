"""Analytics utilities for jersey usage and crowded-play detection.

This script operates on tracking dataframes to derive jersey intervals,
crowded periods and related summaries, and is often paired with CLI tools.

@see @ref hmlib.analytics.play_breaks.find_low_velocity_ranges "find_low_velocity_ranges"
     for complementary break detection logic.
"""

import math
import traceback
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

from hmlib.analytics.play_breaks import find_low_velocity_ranges
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.datasets.dataframe import json_to_dataclass
from hmlib.jersey.number_classifier import TrackJerseyInfo
from hmlib.log import get_logger
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils.time import format_duration_to_hhmmss


@dataclass
class IntervalJerseys:
    start_time: float = -1
    jersey_numbers: Set[int] = None


@dataclass
class TimeInterval:
    start_time: float = None
    duration: float = None


@dataclass
class JerseyTimeIntervals:
    jersey_number: int = None
    intervals: List[TimeInterval] = None


@dataclass
class CrowdedPeriod:
    start_time: float  # in seconds
    end_time: float  # in seconds
    avg_track_count: float


def find_crowded_periods(
    frame_tracks: Dict[int, Set[int]],
    fps: float,
    min_tracks: int,
    min_duration: float,
    occupancy_threshold: float = 0.9,
) -> List[CrowdedPeriod]:
    """
    Find periods where there are more than min_tracks active for at least min_duration seconds.

    Args:
        frame_tracks: Dictionary mapping frame_id to set of active track IDs
        fps: Frames per second
        min_tracks: Minimum number of concurrent tracks required
        min_duration: Minimum duration in seconds
        occupancy_threshold: Minimum fraction of frames that must have min_tracks or more tracks
                           within the period (default: 0.9)

    Returns:
        List of CrowdedPeriod objects containing start_time, end_time, and average track count
    """
    if not frame_tracks:
        return []

    # Convert to numpy array for efficient processing
    max_frame = max(frame_tracks.keys())
    track_counts = np.zeros(max_frame + 1, dtype=int)

    # Fill in track counts for each frame
    for frame_id, tracks in frame_tracks.items():
        track_counts[frame_id] = len(tracks)

    # Convert frames to times
    frame_times = np.arange(len(track_counts)) / fps

    # Minimum number of frames for the required duration
    min_frames = int(min_duration * fps)

    if min_frames <= 0:
        raise ValueError("min_duration must result in at least 1 frame")

    crowded_periods = []
    start_frame = 0

    while start_frame < len(track_counts) - min_frames:
        # Look at a window of min_frames
        window = track_counts[start_frame : start_frame + min_frames]

        # Check if enough frames in the window have sufficient tracks
        crowded_frames = np.sum(window >= min_tracks)
        occupancy_rate = crowded_frames / min_frames

        if occupancy_rate >= occupancy_threshold:
            # Found a qualifying window, try to extend it
            end_frame = start_frame + min_frames

            while (
                end_frame < len(track_counts)
                and np.sum(track_counts[start_frame:end_frame] >= min_tracks)
                / (end_frame - start_frame)
                >= occupancy_threshold
            ):
                end_frame += 1

            # Back up one frame since the last extension failed
            end_frame -= 1

            # Calculate average number of tracks in this period
            avg_tracks = np.mean(track_counts[start_frame:end_frame])

            period = CrowdedPeriod(
                start_time=frame_times[start_frame],
                end_time=frame_times[end_frame],
                avg_track_count=float(avg_tracks),
            )
            crowded_periods.append(period)

            # Start looking after this period
            start_frame = end_frame
        else:
            start_frame += 1

    return crowded_periods


def split_by_lg_10(numbers: List[int]) -> Tuple[List[int], List[int]]:
    less_than_10: List[int] = []
    greater_than_10: List[int] = []
    for n in numbers:
        if n < 10:
            less_than_10.append(n)
        else:
            greater_than_10.append(n)
    return less_than_10, greater_than_10


def remove_one_digit_numbers(numbers: List[int]) -> List[int]:
    # Split numbers into one-digit and two-digit groups
    one_digit: List[int] = [num for num in numbers if num < 10]
    two_digit: List[int] = [num for num in numbers if num >= 10]

    # Convert two-digit numbers to strings for digit comparison
    two_digit_str: List[str] = [str(num) for num in two_digit]

    # Filter out one-digit numbers that appear in any two-digit number
    filtered_one_digit: List[int] = [
        num for num in one_digit if not any(str(num) in tds for tds in two_digit_str)
    ]

    # Combine the lists to form the final result
    return filtered_one_digit + two_digit


def remove_low_frequency_numbers(numbers: List[int], min_freq: float = 0.1) -> List[int]:
    """
    Removes numbers from the list that appear less than 10% of the time
    relative to the most frequently-appearing number.

    Args:
    numbers (List[int]): A list of integers, possibly with duplicates.

    Returns:
    List[int]: A list of integers where each number appears at least 10%
               of the times of the most frequently appearing number.
    """
    # Count the frequency of each number in the list
    frequency: Counter = Counter(numbers)

    # Find the maximum frequency
    max_freq: int = max(frequency.values())

    # Calculate 10% of the maximum frequency
    threshold: float = max_freq * min_freq

    # Filter out numbers where the frequency is less than the threshold
    filtered_numbers: List[int] = [
        number for number, count in frequency.items() if count >= threshold
    ]

    return filtered_numbers


def reorder_jerseys_with_frames(
    jersey_numbers: List[int], frame_jersey_map: Dict[int, int]
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Reorders a list of jersey numbers based on the first occurrence of each jersey number by
    increasing frame_id in the provided dictionary and returns a list of frames for each number.

    Args:
    jersey_numbers (List[int]): List of jersey numbers to reorder.
    frame_jersey_map (Dict[int, int]): Dictionary with frame_id as keys and jersey_number as values.

    Returns:
    Tuple[List[int], Dict[int, List[int]]]: A tuple containing the reordered list of jersey numbers
                                           and a dictionary mapping each jersey number to a list of frames.
    """
    # Mapping from jersey numbers to their earliest frame_id
    earliest_frames: Dict[int, int] = {}
    # Mapping from jersey numbers to all their frame_ids
    frames_per_jersey: Dict[int, List[int]] = {}

    for frame_id, jersey_number in frame_jersey_map.items():
        if jersey_number in earliest_frames:
            earliest_frames[jersey_number] = min(earliest_frames[jersey_number], frame_id)
        else:
            earliest_frames[jersey_number] = frame_id

        if jersey_number in frames_per_jersey:
            frames_per_jersey[jersey_number].append(frame_id)
        else:
            frames_per_jersey[jersey_number] = [frame_id]

    # Sort the jersey numbers by the earliest frame_id
    sorted_jerseys: List[Tuple[int, int]] = sorted(
        ((jersey, frame) for jersey, frame in earliest_frames.items()), key=lambda x: x[1]
    )

    # Extract the sorted jersey numbers
    sorted_jerseys_only: List[int] = [jersey for jersey, frame in sorted_jerseys]

    # Filter the original jersey numbers to match the new order, maintaining the original frequency and order
    jersey_order: Dict[int, int] = {jersey: idx for idx, jersey in enumerate(sorted_jerseys_only)}
    final_sorted_jerseys: List[int] = sorted(
        (j for j in jersey_numbers if j in jersey_order), key=lambda x: jersey_order[x]
    )

    return final_sorted_jerseys, {
        jersey: sorted(frames) for jersey, frames in frames_per_jersey.items()
    }


def analyze_track(
    tracking_id: int,
    numbers: Set[int],
    frame_and_numbers: Dict[int, int],
    roster: Set[int],
    track_numbers: Dict[int, int],
):
    numbers = sorted(list(numbers))
    if len(numbers) == 1:
        track_numbers[tracking_id] = numbers[0]
        # print(f"Track {tracking_id} has single number: {numbers[0]}")
        return None
    numbers_on_roster: Set[int] = set()
    for num in numbers:
        if num in roster:
            numbers_on_roster.add(num)
    if not numbers_on_roster:
        # No roster numbers were seen (should we guess here?)
        return None
    if len(numbers_on_roster) == 1:
        num = next(iter(numbers_on_roster))
        track_numbers[tracking_id] = num
        # print(f"Single roster number from track {tracking_id}: {num}")
        return None
    # Check for just picking up one of the two numbers
    trim_digit_list = remove_one_digit_numbers(numbers_on_roster)
    if len(trim_digit_list) != len(numbers_on_roster):
        if len(trim_digit_list) == 1:
            num = trim_digit_list[0]
            # print(f"Trimmed number list {numbers_on_roster} down to {num}")
            track_numbers[tracking_id] = num
            return None
    all_seen_occurrences: List[int] = [n for n in frame_and_numbers.values()]
    high_freq_numbers = list(set(remove_low_frequency_numbers(all_seen_occurrences)))
    if len(high_freq_numbers) == 1:
        num = high_freq_numbers[0]
        # print(f"High freq number for track {tracking_id} is {num}")
        track_numbers[tracking_id] = num
        return None
    assert high_freq_numbers

    #
    # If we get here, the track probably got split and all numbers are valid (hopefully)
    #
    occuring_numbers, jersey_frames = reorder_jerseys_with_frames(
        high_freq_numbers, frame_and_numbers
    )
    #
    new_data: OrderedDict[Union[int, float], Dict[str, Any]] = {}
    for i, num in enumerate(occuring_numbers):
        if i == 0:
            tid = tracking_id
        else:
            tid = tracking_id + i / 10
        # If this fires, need to make the increment smaller because we went into the next tracking id
        assert int(tid) == tracking_id
        frames_seen = jersey_frames[num]
        track_numbers[tid] = num
        new_data[tid] = {"jersey_number": num, "frames": frames_seen}
        # print(f"tracking id: {tid} is number {num}")
    return new_data


def auto_dict(the_dict: Dict[Any, Any], keys: List[Any], final_type: type = dict) -> Dict[Any, Any]:
    if not isinstance(keys, list):
        keys = [keys]
    current_dict = the_dict
    key_count = len(keys)
    for i, key in enumerate(keys):
        if key not in current_dict:
            if i == key_count - 1:
                current_dict[key] = final_type()
            else:
                current_dict[key] = {}
        current_dict = current_dict[key]
    return current_dict


def calc_center(player_tracking_item: TrackingDataFrame) -> float:
    return np.array(
        [
            player_tracking_item.BBox_X + player_tracking_item.BBox_W / 2,
            player_tracking_item.BBox_Y + player_tracking_item.BBox_Y / 2,
        ],
        dtype="float",
    )


def panoramic_distance(image_width, x1, x2, rink_width=200):
    """
    Calculate the real-world distance between two points on a 180-degree panoramic image.

    Args:
    image_width (int): The width of the panoramic image in pixels.
    x1, x2 (int): The x-coordinates of the two points on the image.
    rink_width (float): The width of the hockey rink in feet.

    Returns:
    float: The distance between the two points in feet on the hockey rink.
    """
    # Center of the image (0 degrees)
    center_x = image_width / 2

    # Normalize the x-coordinates to a range of [-1, 1]
    normalized_x1 = (x1 - center_x) / center_x
    normalized_x2 = (x2 - center_x) / center_x

    # Convert normalized coordinates to angles in radians
    angle1_rad = math.pi * normalized_x1 / 2  # -pi/2 to pi/2
    angle2_rad = math.pi * normalized_x2 / 2  # -pi/2 to pi/2

    radius = rink_width / 2
    angle_diff = abs(angle1_rad - angle2_rad)
    distance = radius * angle_diff  # Arc length = radius * angle in radians

    return distance


def merge_intervals(
    intervals: List[Tuple[float, float]], min_difference: float
) -> List[Tuple[float, float]]:

    if not intervals:
        return intervals

    # Sort intervals by start time
    intervals.sort()

    merged = []
    current_start, first_duration = intervals[0]
    current_end = current_start + first_duration

    for start, duration in intervals:
        if start <= current_end + min_difference:
            # If the start of the next interval is within min_difference of the current end,
            # extend the current end to the maximum of the current end or the end of this interval
            current_end = max(current_end, start + duration)
        else:
            # Otherwise, save the current interval and start a new one
            merged.append((current_start, current_end - current_start))
            current_start, current_end = start, start + duration

    # Add the last processed interval to the list
    merged.append((current_start, current_end - current_start))

    return merged


def analyze_data(
    player_tracking_data: TrackingDataFrame,
    camera_tracking_data: CameraTrackingDataFrame,
    uncropped_width: int,
    fps: float = 29.97,
    roster: Set[int] = None,
) -> List[IntervalJerseys]:
    # tracking_id -> [numbers]
    tracking_id_to_numbers: OrderedDict[int, Set[int]] = OrderedDict()
    # tracking_id -> frame_id -> number
    tracking_id_frame_and_numbers: OrderedDict[int, Dict[int, int]] = OrderedDict()
    # frame_id -> tracking_id
    frame_to_tracking_ids: OrderedDict[Union[int, float], Set[int]] = OrderedDict()
    # track_id -> (last seen frame_id, last center point)
    track_id_to_last_frame_id: OrderedDict[int, Tuple[int, Tuple[float, float]]] = OrderedDict()
    # frame_id -> tracking_id -> velocity
    frame_track_velocity: Dict[int, Dict[int, float]] = {}
    player_tracking_iter = iter(player_tracking_data)

    # json strings that we can ignore
    empty_json_set: Set[str] = set()
    seen_numbers: Set[int] = set()
    item_count: int = 0

    max_frame_id = 0
    # max_frame_id = 10000

    try:
        last_frame_id = 0
        while True:

            if max_frame_id > 0 and last_frame_id > max_frame_id:
                get_logger(__name__).info(
                    "Early exit at max frame id %d", last_frame_id
                )
                break

            # Items are frame_id, tracking_id, ...
            # Axis 0 is frame_id
            player_tracking_item = next(player_tracking_iter)
            frame_id = int(player_tracking_item.Frame)
            tracking_id = player_tracking_item.ID

            if player_tracking_item.JerseyInfo in empty_json_set:
                jersey_item = None
            else:
                jersey_item = json_to_dataclass(player_tracking_item.JerseyInfo, TrackJerseyInfo)
                if jersey_item.tracking_id < 0:
                    # Add bad use-case json string for speedy skipping
                    get_logger(__name__).info(
                        "Ignoring jersey record case: %s", player_tracking_item.JerseyInfo
                    )
                    empty_json_set.add(player_tracking_item.JerseyInfo)
                    jersey_item = None
                else:
                    assert jersey_item.tracking_id == tracking_id

            item_count += 1

            row_tracking_id = int(player_tracking_item.ID)
            new_center = calc_center(player_tracking_item)
            prev_frame_stuff = track_id_to_last_frame_id.get(row_tracking_id)
            if prev_frame_stuff is not None:
                prev_frame_id, prev_center = prev_frame_stuff
                if prev_frame_id == frame_id:
                    pass
                assert prev_frame_id <= frame_id
                # assert prev_frame_id != frame_id
                # FIXME: batch size issue when saving frame data (hence the 2)
                if prev_frame_id == frame_id - 1 or prev_frame_id == frame_id - 2:
                    pandist_x = panoramic_distance(
                        image_width=uncropped_width, x1=new_center[0], x2=prev_center[0]
                    )
                    pandist_y = panoramic_distance(
                        image_width=uncropped_width, x1=new_center[1], x2=prev_center[1]
                    )
                    velocity = math.sqrt(pandist_x**2 + pandist_y**2)
                    if frame_id not in frame_track_velocity:
                        frame_track_velocity[frame_id] = {}
                    frame_track_velocity[frame_id][row_tracking_id] = velocity

            track_id_to_last_frame_id[row_tracking_id] = frame_id, new_center

            if frame_id != last_frame_id:
                assert frame_id >= last_frame_id
                last_frame_id = frame_id

            if frame_id not in frame_to_tracking_ids:
                frame_to_tracking_ids[frame_id] = set()
            frame_to_tracking_ids[frame_id].add(tracking_id)
            if jersey_item is not None:
                number = int(jersey_item.number)
                if not roster or number in roster:
                    auto_dict(tracking_id_to_numbers, tracking_id, set).add(number)
                    # tracking_id_to_numbers[tracking_id].add(number)
                    auto_dict(tracking_id_frame_and_numbers, tracking_id)[frame_id] = number
                    if number not in seen_numbers:
                        seen_numbers.add(number)
                        get_logger(__name__).info(
                            "First sighting of number %d at frame %d", number, frame_id
                        )

    except StopIteration:
        get_logger(__name__).info("Finished reading %d items", item_count)
    except Exception:
        traceback.print_exc()
    logger = get_logger(__name__)
    logger.info("Unique player numbers seen: %d", len(seen_numbers))
    logger.info("Tracks seen with numbers: %d", len(tracking_id_to_numbers))

    crowded_periods = find_crowded_periods(
        frame_tracks=frame_to_tracking_ids,
        fps=29.97,
        min_tracks=15,
        min_duration=1.0,
        occupancy_threshold=0.4,
    )
    logger.info("Crowded periods: %s", crowded_periods)

    period_intervals = {
        "min_velocity": 0.2,
        "min_frames": 30,
        "min_slow_track_ratio": 0.9,
    }

    # Period separators
    period_breaks = find_low_velocity_ranges(data=frame_track_velocity, **period_intervals)
    # print("Unmerged period breaks:")
    # show_frame_intervals(period_breaks, fps=fps)
    period_breaks = frames_to_seconds(period_breaks, fps=fps)
    merged_period_breaks = merge_intervals(period_breaks, 300)
    # print(f"Periods: {merged_period_breaks}")
    show_time_intervals("Period breaks", merged_period_breaks)

    faceoff_intervals = {
        # "min_velocity": 0.3,
        "min_velocity": 0.4,
        # "min_frames": 30,
        "min_frames": 20,
        "min_slow_track_ratio": 0.7,
        # "min_slow_track_ratio": 0.8,
    }
    faceoff_breaks = find_low_velocity_ranges(data=frame_track_velocity, **faceoff_intervals)
    # print("Unmerged faceoff breaks:")
    # show_frame_intervals(faceoff_breaks, fps=fps)
    faceoff_breaks = frames_to_seconds(faceoff_breaks, fps=fps)
    merged_faceoff_breaks = merge_intervals(faceoff_breaks, 20.0)
    # print(f"Faceoffs: {merged_faceoff_breaks}")
    show_time_intervals("Merged faceoff breaks", merged_faceoff_breaks)

    # Now analyze tracks
    track_numbers: Dict[int, int] = {}
    massaged_data: Dict[Union[float, int], Any] = {}
    assert len(tracking_id_to_numbers) == len(tracking_id_frame_and_numbers)
    for tracking_id in tracking_id_to_numbers.keys():
        new_data = analyze_track(
            tracking_id,
            tracking_id_to_numbers[tracking_id],
            tracking_id_frame_and_numbers[tracking_id],
            roster=roster,
            track_numbers=track_numbers,
        )
        if new_data:
            massaged_data.update(new_data)

    frame_to_jersey_number: Dict[int, Set[int]] = {}

    for tracking_id, jersey_info in massaged_data.items():
        num = jersey_info["jersey_number"]
        # assert tracking_id not in tracking_id_to_numbers
        tracking_id_to_numbers[tracking_id] = num
        tracking_id_frame_and_numbers[tracking_id] = OrderedDict()
        for frame_id in jersey_info["frames"]:
            if frame_id not in frame_to_jersey_number:
                frame_to_jersey_number[frame_id] = set()
            else:
                pass
            frame_to_jersey_number[frame_id].add(num)
            # This won't be updated correctly for other cases inside analyze_track
            tracking_id_frame_and_numbers[tracking_id][frame_id] = num
            if frame_id not in frame_to_tracking_ids:
                frame_to_tracking_ids[frame_id] = set()
            frame_to_tracking_ids[frame_id].add(tracking_id)
    # print(frame_to_jersey_number)

    period_jerseys: List[float, Set[int]] = calculate_start_time_jerseys(
        start_time_to_jersey_set=merged_period_breaks,
        frame_to_tracking_ids=frame_to_tracking_ids,
        tracking_id_to_numbers=tracking_id_to_numbers,
        fps=fps,
    )

    shift_jerseys: List[float, Set[int]] = calculate_start_time_jerseys(
        start_time_to_jersey_set=merged_faceoff_breaks,
        frame_to_tracking_ids=frame_to_tracking_ids,
        tracking_id_to_numbers=tracking_id_to_numbers,
        fps=fps,
    )

    return period_jerseys, shift_jerseys


def calculate_start_time_jerseys(
    start_time_to_jersey_set: List[Tuple[float, float]],
    frame_to_tracking_ids: Dict[int, Set[int]],
    tracking_id_to_numbers: Dict[int, Set[int]],
    fps: float,
) -> List[IntervalJerseys]:
    """
    Return list of interval start times + jersey seen in that interval
    """
    tracking_id_intervals: List[Set[int]] = _do_assign(
        start_time_to_jersey_set, fps=fps, frame_to_tracking_ids=frame_to_tracking_ids
    )
    start_times_s = [0.0] + [intvl[0] for intvl in start_time_to_jersey_set]
    interval_jersey_numbers: List[Set[int]] = [set() for _ in range(len(tracking_id_intervals))]
    for interval_index, interval_tracking_ids in enumerate(tracking_id_intervals):
        for tid in interval_tracking_ids:
            if tid in tracking_id_to_numbers:
                number = tracking_id_to_numbers[tid]
                if isinstance(number, (list, set)):
                    for n in number:
                        interval_jersey_numbers[interval_index].add(n)
                else:
                    interval_jersey_numbers[interval_index].add(number)

    start_time_jerseys: List[IntervalJerseys] = []
    for start_time, jersey_numbers in zip(start_times_s, interval_jersey_numbers):
        start_time_jerseys.append(
            IntervalJerseys(start_time=start_time, jersey_numbers=jersey_numbers)
        )
    return start_time_jerseys


def _do_assign(
    time_intervals: List[Tuple[float, float]],
    fps: float,
    frame_to_tracking_ids: Dict[int, Set[int]],
) -> List[Set[int]]:
    start_frames = [0] + [int(t * fps) for t, _ in time_intervals]
    assigned_intervals = assign_numbers_to_intervals(start_frames, frame_to_tracking_ids)
    # print(assigned_intervals)
    return assigned_intervals


def assign_numbers_to_intervals(
    frame_starts: List[int],
    frame_to_tracking_ids: Dict[int, Set[int]],
    start_frame_offset_ratio: float = 1.0,  # Only consider detections withing the middle % here
    stop_frame_offset_ratio: float = 0.9,  # Only consider detections withing the middle % here
) -> List[Set[int]]:
    """
    Given a list of frame start points and frame->jersey number detection points,
    return a list of detected numbers during that interval
    """
    # Assure it's sorted
    assert frame_starts == sorted(frame_starts)
    seen_tracking_ids: List[Set[int]] = [set() for _ in range(len(frame_starts))]
    for interval_index in range(len(frame_starts)):

        start_frame = frame_starts[interval_index]
        end_frame = (
            float("inf")
            if interval_index == len(frame_starts) - 1
            else frame_starts[interval_index + 1] - 1
        )
        frame_interval = end_frame - start_frame
        floating_start = start_frame + (1 - start_frame_offset_ratio) * frame_interval
        floating_end = end_frame - (1 - stop_frame_offset_ratio) * frame_interval

        for frame_id, j_numbers in frame_to_tracking_ids.items():
            if frame_id < floating_start or frame_id > floating_end:
                continue
            seen_tracking_ids[interval_index] = seen_tracking_ids[interval_index].union(j_numbers)

    return seen_tracking_ids


def show_frame_intervals(intervals: List[Tuple[int, int]], fps: float) -> None:
    time_ranges: List[Tuple[float, float]] = frames_to_seconds(intervals, fps)

    logger = get_logger(__name__)
    for start_s, duration_s in time_ranges:
        start_hhmmss = format_duration_to_hhmmss(start_s, decimals=0)
        logger.info(
            "%s for %.1f seconds", start_hhmmss, int(duration_s * 10) / 10
        )


def show_time_intervals(label: str, intervals: List[Tuple[float, float]]) -> None:
    logger = get_logger(__name__)
    if intervals and label:
        logger.info("---------------------------------------------")
        logger.info("- %s", label)
        logger.info("---------------------------------------------")
    for start_s, duration_s in intervals:
        start_hhmmss = format_duration_to_hhmmss(start_s, decimals=0)
        logger.info(
            "%s for %.1f seconds", start_hhmmss, int(duration_s * 10) / 10
        )
    logger.info("---------------------------------------------")


def frames_to_seconds(
    frame_tuple_list: List[Tuple[int, int]], fps: float
) -> List[Tuple[float, float]]:
    results: List[Tuple[int, int]] = []
    for start_end_frame in frame_tuple_list:
        start_frame, end_frame = start_end_frame
        assert end_frame > start_frame
        frame_count = end_frame - start_frame
        duration_s = frame_count / fps
        start_s = start_frame / fps
        results.append((start_s, duration_s))
    return results


def interval_jerseys_to_merged_jersey_time_intervals(
    intervals: List[IntervalJerseys],
) -> List[JerseyTimeIntervals]:
    # Dictionary to store the merged intervals for each jersey number
    jersey_times: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    # Process each interval
    for i in range(len(intervals)):
        interval_jerseys = intervals[i]
        jerseys = intervals[i].jersey_numbers
        end = float("inf") if i == len(intervals) - 1 else intervals[i + 1].start_time

        for jersey in jerseys:
            if (
                not jersey_times[jersey]
                or jersey_times[jersey][-1][1] < interval_jerseys.start_time
            ):
                # If no interval exists for this jersey, or there's no overlap
                jersey_times[jersey].append([interval_jerseys.start_time, end])
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

    jt_results: List[JerseyTimeIntervals] = []
    for jn, jinv in result.items():
        jn_intervals = JerseyTimeIntervals(jersey_number=jn, intervals=[])
        for intv in jinv:
            jn_intervals.intervals.append(TimeInterval(start_time=intv[0], duration=intv[1]))
        jt_results.append(jn_intervals)

    return jt_results

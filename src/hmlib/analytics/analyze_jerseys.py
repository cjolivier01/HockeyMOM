import argparse
import json
import math
import os
import traceback
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np

from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame

SHARKS_12_1_ROSTER: Set[int] = {29, 37, 40, 98, 73, 89, 54, 24, 79, 16, 27, 90, 57, 8, 96, 74}


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
        print(f"Single roster number from track {tracking_id}: {num}")
        return None
    # Check for just picking up one of the two numbers
    trim_digit_list = remove_one_digit_numbers(numbers_on_roster)
    if len(trim_digit_list) != len(numbers_on_roster):
        if len(trim_digit_list) == 1:
            num = trim_digit_list[0]
            print(f"Trimmed number list {numbers_on_roster} down to {num}")
            track_numbers[tracking_id] = num
            return None
    all_seen_occurrences: List[int] = [n for n in frame_and_numbers.values()]
    high_freq_numbers = list(set(remove_low_frequency_numbers(all_seen_occurrences)))
    if len(high_freq_numbers) == 1:
        num = high_freq_numbers[0]
        print(f"High freq number for track {tracking_id} is {num}")
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
        print(f"tracking id: {tid} is number {num}")
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


def calc_center(tracking_item: TrackingDataFrame) -> float:
    return np.array(
        [
            tracking_item.BBox_X + tracking_item.BBox_W / 2,
            tracking_item.BBox_Y + tracking_item.BBox_Y / 2,
        ],
        dtype="float",
    )


def analyze_data(tracking_data: TrackingDataFrame) -> None:
    frame_data: OrderedDict[int, Any] = OrderedDict()
    # tracking_id -> [numbers]
    tracking_id_numbers: OrderedDict[int, Set[int]] = OrderedDict()
    # tracking_id -> frame_id -> number
    tracking_id_frame_and_numbers: OrderedDict[int, Dict[int, int]] = OrderedDict()
    # frame_id -> tracking_id
    frame_to_tracking_ids: OrderedDict[Union[int, float], Set[int]] = OrderedDict()
    # track_id -> (last seen frame_id, last center point)
    track_id_to_last_frame_id: OrderedDict[int, Tuple[int, Tuple[float, float]]] = OrderedDict()
    # frame_id -> tracking_id -> velocity (may need to consider bbox size relative to entire width when computing velocity)
    track_id_to_last_frame_id: OrderedDict[int, OrderedDict[int, float]] = OrderedDict()
    # frame_id -> tracking_id -> velocity
    frame_track_velocity: Dict[int, Dict[int, float]] = {}
    tracking_iter = iter(tracking_data)
    # json strings that we can ignore
    empty_json_set: Set[str] = set()
    seen_numbers: Set[int] = set()
    item_count: int = 0
    min_score: float = 0.7
    try:
        last_frame_id = 0
        while True:
            # Items are frame_id, tracking_id, ...
            # Axis 0 is frame_id
            tracking_item = next(tracking_iter)
            item_count += 1
            if not tracking_item.JerseyInfo or tracking_item.JerseyInfo in empty_json_set:
                continue
            jersey_info = json.loads(tracking_item.JerseyInfo)
            if "items" not in jersey_info:
                # quick skip recurrences of this string
                empty_json_set.add(tracking_item.JerseyInfo)
                continue
            jersey_items = jersey_info["items"]
            if not jersey_items:
                # quick skip recurrences of this string
                empty_json_set.add(tracking_item.JerseyInfo)
                continue
            # print(f"items: {jersey_items}")
            frame_id = int(tracking_item.Frame)

            row_tracking_id = int(tracking_item.ID)
            new_center = calc_center(tracking_item)
            prev_frame_stuff = track_id_to_last_frame_id.get(row_tracking_id)
            if prev_frame_stuff is not None:
                prev_frame_id, prev_center = prev_frame_stuff
                assert prev_frame_id != frame_id
                # FIXME: batch size issue when saving frame data (hence the 2)
                if prev_frame_id == frame_id - 1 or prev_frame_id == frame_id - 2:
                    velocity = math.sqrt(
                        abs(new_center[0] - prev_center[0]) ** 2
                        + abs(new_center[1] - prev_center[1]) ** 2
                    )
                    if frame_id not in frame_track_velocity:
                        frame_track_velocity[frame_id] = {}
                    frame_track_velocity[frame_id][row_tracking_id] = velocity

            track_id_to_last_frame_id[row_tracking_id] = frame_id, new_center

            # this_row_tracking_id = tracking_item.
            # # Get velocity if this track was around in the previous frame
            # prev_frame_stuff = track_id_to_last_frame_id.get(tracking_id)
            # new_center = calc_center(tracking_item)
            # if prev_frame_stuff is not None:
            #     # was here previous frame
            #     prev_frame_id, prev_center = prev_frame_stuff
            #     if prev_frame_id == frame_id - 1:
            #         pass
            # # This is how we'd compute velocity, based on the previous frame if this track id was there
            # assert (
            #     tracking_id not in track_id_to_last_frame_id
            #     or track_id_to_last_frame_id[tracking_id] != frame_id
            # )
            # track_id_to_last_frame_id[tracking_id] = frame_id, new_center

            # Make sure frame_id is increasing (for velocity calculation)
            # assert frame_id == last_frame_id or frame_id == last_frame_id + 1
            if frame_id != last_frame_id:
                assert frame_id >= last_frame_id
                last_frame_id = frame_id

            if frame_id not in frame_to_tracking_ids:
                frame_to_tracking_ids[frame_id] = set()

            for j_item in jersey_items:
                tracking_id = j_item["tracking_id"]
                if tracking_id != row_tracking_id:
                    # We'll deal with it when we get to that tracking id for this frame
                    continue
                frame_to_tracking_ids[frame_id].add(tracking_id)
                number = int(j_item["number"])
                score = j_item["score"]
                auto_dict(tracking_id_numbers, tracking_id, set).add(number)
                # tracking_id_numbers[tracking_id].add(number)
                auto_dict(tracking_id_frame_and_numbers, tracking_id)[frame_id] = number
                if number not in seen_numbers:
                    seen_numbers.add(number)
                    # print(f"First sighting of number {number} at frame {frame_id}")

    except StopIteration:
        print(f"Finished reading {item_count} items")
    except Exception:
        traceback.print_exc()
    print(f"Unique player numbers seen: {len(seen_numbers)}")
    print(f"Tracks seen with numbers: {len(tracking_id_numbers)}")
    # Now analyze tracks
    track_numbers: Dict[int, int] = {}
    massaged_data: Dict[Union[float, int], Any] = {}
    assert len(tracking_id_numbers) == len(tracking_id_frame_and_numbers)
    for tracking_id in tracking_id_numbers.keys():
        new_data = analyze_track(
            tracking_id,
            tracking_id_numbers[tracking_id],
            tracking_id_frame_and_numbers[tracking_id],
            roster=SHARKS_12_1_ROSTER,
            track_numbers=track_numbers,
        )
        if new_data:
            massaged_data.update(new_data)
    for tracking_id, jersey_info in massaged_data.items():
        num = jersey_info["jersey_number"]
        tracking_id_numbers[tracking_id] = num
        tracking_id_frame_and_numbers[tracking_id] = OrderedDict()
        for frame_id in jersey_info["frames"]:
            # This won't be updated correctly for other cases inside analyze_track
            tracking_id_frame_and_numbers[tracking_id][frame_id] = num
            if frame_id not in frame_to_tracking_ids:
                frame_to_tracking_ids[frame_id] = set()
            frame_to_tracking_ids[frame_id].add(tracking_id)
        pass
    return

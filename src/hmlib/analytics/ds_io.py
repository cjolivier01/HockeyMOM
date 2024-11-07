import argparse
import json
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple

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
    filtered_one_digit: List[int] = [num for num in one_digit if not any(str(num) in tds for tds in two_digit_str)]

    # Combine the lists to form the final result
    return filtered_one_digit + two_digit


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
        return
    numbers_on_roster: Set[int] = set()
    for num in numbers:
        if num in roster:
            numbers_on_roster.add(num)
    if not numbers_on_roster:
        # No roster numbers were seen (should we guess here?)
        return
    if len(numbers_on_roster) == 1:
        num = next(iter(numbers_on_roster))
        track_numbers[tracking_id] = num
        print(f"Single roster number from track {tracking_id}: {num}")
        return
    # Check for just picking up one of the two numbers
    trim_digit_list = remove_one_digit_numbers(numbers_on_roster)
    if len(trim_digit_list) != len(numbers_on_roster):
        if len(trim_digit_list) == 1:
            num = trim_digit_list[0]
            print(f"Trimmed number list {numbers_on_roster} down to {num}")
            track_numbers[tracking_id] = num
            return
    #print(f"Still confused: {trim_digit_list}")
    return


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


def analyze_data(tracking_data: TrackingDataFrame) -> None:
    frame_data: OrderedDict[int, Any] = OrderedDict()
    # tracking_id -> [numbers]
    tracking_id_numbers: OrderedDict[int, Set[int]] = OrderedDict()
    # tracking_id -> frame_id -> number
    tracking_id_frame_and_numbers: OrderedDict[int, Dict[int, int]] = OrderedDict()

    tracking_iter = iter(tracking_data)
    # json strings that we can ignore
    empty_json_set: Set[str] = set()
    seen_numbers: Set[int] = set()
    item_count: int = 0
    min_score: float = 0.7
    try:
        while True:
            tracking_item = next(tracking_iter)
            item_count += 1
            # if item_count == 1374504:
            #     pass
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
            for j_item in jersey_items:
                tracking_id = j_item["tracking_id"]
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
    assert len(tracking_id_numbers) == len(tracking_id_frame_and_numbers)
    for tracking_id in tracking_id_numbers.keys():
        analyze_track(
            tracking_id,
            tracking_id_numbers[tracking_id],
            tracking_id_frame_and_numbers[tracking_id],
            roster=SHARKS_12_1_ROSTER,
            track_numbers=track_numbers,
        )
    return

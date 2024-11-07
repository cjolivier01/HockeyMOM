import argparse
import json
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Set

from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame


def analyze_track(
    tracking_id_numbers: OrderedDict[int, Set[int]],
    tracking_id_frame_and_numbers: OrderedDict[int, Dict[int, int]],
):
    # These args have duplicate info, but both sent for efficiency
    pass


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
                    print(f"First sighting of number {number} at frame {frame_id}")
    except StopIteration:
        print(f"Finished reading {item_count} items")
    except Exception:
        traceback.print_exc()
    print(f"Unique player numbers seen: {len(seen_numbers)}")
    print(f"Tracks seen with numbers: {len(tracking_id_numbers)}")
    # Now analyze tracks

    return

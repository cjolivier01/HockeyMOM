import argparse
import json
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, Set

from hmlib.config import get_game_dir
from hmlib.hm_opts import hm_opts
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame


def load_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return TrackingDataFrame(input_file=os.path.join(game_dir, "results.csv"))


def analyze_data(tracking_data: TrackingDataFrame) -> None:
    frame_data: OrderedDict[int, Any] = OrderedDict()
    tracking_iter = iter(tracking_data)
    # jsons trings that we can ignore
    empty_json_set: Set[str] = set()
    item_count: int = 0
    while True:
        try:
            tracking_item = next(tracking_iter)
            item_count += 1
        except StopIteration:
            print(f"Finished reading {item_count} items")
            break
        except Exception:
            traceback.print_exc()
    return


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    if not args.game_id:
        args.game_id = "ev-stockton-1"

    tracking_data = load_data(game_id=args.game_id)
    analyze_data(tracking_data)

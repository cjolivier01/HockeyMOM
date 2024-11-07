import os
import traceback
from typing import Any, Dict

from hmlib.analytics.analyze_jerseys import analyze_data
from hmlib.config import get_game_dir
from hmlib.hm_opts import hm_opts
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame


def load_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return TrackingDataFrame(input_file=os.path.join(game_dir, "results.csv"))


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    if not args.game_id:
        args.game_id = "ev-stockton-1"
    try:
        tracking_data = load_data(game_id=args.game_id)
        analyze_data(tracking_data)
    except Exception:
        traceback.print_exc()

import os
import traceback
from typing import Any, Dict

from hmlib.analytics.analyze_jerseys import analyze_data
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.config import get_game_dir
from hmlib.hm_opts import hm_opts
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame


def load_player_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return TrackingDataFrame(input_file=os.path.join(game_dir, "results.csv"))


def load_camera_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return CameraTrackingDataFrame(input_file=os.path.join(game_dir, "camera.csv"))


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    if not args.game_id:
        args.game_id = "ev-stockton-1"
    try:
        player_tracking_data = load_player_tracking_data(game_id=args.game_id)
        camera_tracking_data = load_camera_tracking_data(game_id=args.game_id)
        analyze_data(player_tracking_data, camera_tracking_data)
    except Exception:
        traceback.print_exc()

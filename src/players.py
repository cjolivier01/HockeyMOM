import os
import traceback
from typing import Any, Dict, Tuple

import cv2

from hmlib.analytics.analyze_jerseys import analyze_data
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.config import get_game_dir
from hmlib.hm_opts import hm_opts
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils.image import image_height, image_width


def load_player_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return TrackingDataFrame(input_file=os.path.join(game_dir, "results.csv"))


def load_camera_tracking_data(game_id: str) -> Dict[str, Any]:
    game_dir: str = get_game_dir(game_id=game_id)
    return CameraTrackingDataFrame(input_file=os.path.join(game_dir, "camera.csv"))


def get_uncropped_width_height(game_id: str) -> Tuple[int, int]:
    game_dir: str = get_game_dir(game_id=game_id)
    path = os.path.join(game_dir, "s.png")
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return image_width(img), image_height(img)


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
        analyze_data(player_tracking_data, camera_tracking_data, uncropped_width=uncropped_width)
    except Exception:
        traceback.print_exc()

from typing import Any, Dict, List, Optional, Union

import torch
from mmengine.registry import TRANSFORMS

from hmlib.config import get_clip_box, get_config, get_nested_value
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.scoreboard.selector import configure_scoreboard
from hmlib.utils.image import make_channels_last


def _try_pop(d: Dict[str, Any], k: str) -> Union[Any, None]:
    if k in d:
        return d.pop(k)
    return None


@TRANSFORMS.register_module()
class HmConfigureScoreboard:
    def __init__(
        self,
        game_id: Optional[str] = None,
    ):
        self._game_id = game_id
        self._scoreboard_config = None
        self._configured = False

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self._game_id is None:
            self._game_id = results.get("game_id", None)
        if self._game_id and not self._configured:
            self._configured = True
            scoreboard_points = configure_scoreboard(game_id=self._game_id, image=results["img"])
            if (
                scoreboard_points is not None
                and torch.sum(torch.tensor(scoreboard_points, dtype=torch.float)).item() != 0
            ):
                game_config = get_config(game_id=self._game_id)
                if game_config:
                    clip_box = get_clip_box(game_id=self._game_id)
                    if clip_box:
                        scoreboard_points[0] += clip_box[0]
                        scoreboard_points[2] += clip_box[0]
                        scoreboard_points[1] += clip_box[1]
                        scoreboard_points[3] += clip_box[1]
                    self._scoreboard_config: Dict[str, Any] = dict(
                        scoreboard_points=scoreboard_points,
                        dest_width=get_nested_value(game_config, "rink.scoreboard.projected_width"),
                        dest_height=get_nested_value(
                            game_config, "rink.scoreboard.projected_height"
                        ),
                    )
        if self._scoreboard_config:
            results["scoreboard_cfg"] = self._scoreboard_config

        return results


@TRANSFORMS.register_module()
class HmCaptureScoreboard:
    def __init__(
        self,
        scoreboard_scale: float = 1.0,
    ):
        self._scoreboard = None
        self._scoreboard_scale = scoreboard_scale

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        scoreboard = results.get("scoreboard_cfg")
        if not scoreboard:
            return results
        img = results["img"]
        if self._scoreboard is None:
            dest_width = scoreboard.pop("dest_width")
            if isinstance(dest_width, str) and dest_width.startswith("%"):
                ratio = float(dest_width[1:]) / 100
                if "video_frame_cfg" in results:
                    dw = results["video_frame_cfg"]["output_frame_width"]
                else:
                    dw = results["ori_shape"][-1]
                dest_width = dw * ratio
            dest_height = scoreboard.pop("dest_height")
            if isinstance(dest_height, str) and dest_height.startswith("%"):
                ratio = float(dest_height[1:]) / 100
                if "video_frame_cfg" in results:
                    dh = results["video_frame_cfg"]["output_frame_height"]
                else:
                    dh = results["ori_shape"][-2]
                dest_height = dh * ratio
            self._scoreboard = Scoreboard(
                src_pts=scoreboard["scoreboard_points"],
                dest_width=int(dest_width),
                dest_height=int(dest_height),
                scoreboard_scale=self._scoreboard_scale,
                dtype=torch.float,
                device=img.device,
            )
        # w/h may have been adjusted based upon aspect ratio of the given points, etc.
        scoreboard_img = make_channels_last(self._scoreboard.forward(img))
        results["scoreboard_img"] = scoreboard_img

        return results


@TRANSFORMS.register_module()
class HmRenderScoreboard:
    def __init__(self, image_labels: List[str]):
        self._image_labels = image_labels
        self._scoreboard_width: Optional[int] = None
        self._scoreboard_height: Optional[int] = None

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        scoreboard_img = _try_pop(results, "scoreboard_img")
        if scoreboard_img is None:
            return results

        if self._scoreboard_height is None or self._scoreboard_width is None:
            assert scoreboard_img.ndim == 4
            self._scoreboard_height = int(scoreboard_img.shape[1])
            self._scoreboard_width = int(scoreboard_img.shape[2])

        results.pop("scoreboard_cfg", None)
        for img_label in self._image_labels:
            img = results.get(img_label)
            if img is not None:
                img = make_channels_last(img)
                if torch.is_floating_point(img) and not torch.is_floating_point(scoreboard_img):
                    scoreboard_img = scoreboard_img.to(scoreboard_img.dtype, non_blocking=True)
                assert self._scoreboard_height is not None and self._scoreboard_width is not None
                sh = self._scoreboard_height
                sw = self._scoreboard_width
                img[:, :sh, :sw, :] = scoreboard_img
                results[img_label] = img
        return results

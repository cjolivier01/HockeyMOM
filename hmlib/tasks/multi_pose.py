import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from hmlib.utils.image import make_channels_last


@typechecked
def multi_pose_task(
    pose_inferencer,
    cur_frame: torch.Tensor,
    smooth: bool = False,
    show: bool = False,
):
    inputs = []
    for img in make_channels_last(cur_frame):
        inputs.append(img.cpu().numpy())
    all_pose_results = []
    for pose_results in pose_inferencer(inputs=inputs, visualize=show):
        all_pose_results.append(pose_results)
    return pose_results

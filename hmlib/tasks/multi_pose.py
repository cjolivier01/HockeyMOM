import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from hmlib.ui import show_image
from hmlib.utils.image import make_channels_last
from hmlib.visualization import PytorchPoseLocalVisualizer


@typechecked
def multi_pose_task(
    pose_inferencer,
    cur_frame: torch.Tensor,
    smooth: bool = False,
    show: bool = False,
):
    inputs = []
    for img in make_channels_last(cur_frame):
        inputs.append(img)

    if show and not hasattr(pose_inferencer, "visualizer"):
        pose_inferencer.visualizer = PytorchPoseLocalVisualizer()

    all_pose_results = []
    for pose_results in pose_inferencer(inputs=inputs, visualize=show):
        all_pose_results.append(pose_results)

    # if pose_inferencer.visualizer is not None:
    #     pose_inferencer.visualizer.add_datasample(
    #         name="pose results",
    #         image,
    #         data_sample=
    #     )

    # show_image("pose results", inputs[0], wait=False)
    return all_pose_results

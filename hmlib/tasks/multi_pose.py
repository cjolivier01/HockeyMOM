import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from hmlib.ui import show_image
from hmlib.utils.image import make_channels_last

# from hmlib.visualization import PytorchPoseLocalVisualizer


def multi_pose_task(
    pose_inferencer,
    cur_frame: torch.Tensor,
    smooth: bool = False,
    show: bool = False,
):
    inputs = []
    for img in make_channels_last(cur_frame):
        inputs.append(img)

    all_pose_results = []
    for pose_results in pose_inferencer(
        inputs=inputs, return_datasamples=True, visualize=show, **pose_inferencer.filter_args
    ):
        all_pose_results.append(pose_results)

    if show and pose_inferencer.inferencer.visualizer is not None:
        for img, pose_result in zip(inputs, all_pose_results):
            data_sample = pose_result["predictions"]
            assert len(data_sample) == 1
            pose_inferencer.inferencer.visualizer.add_datasample(
                name="pose results",
                image=img,
                data_sample=data_sample[0],
                clone_image=False,
                draw_gt=False,
                draw_bbox=True,
            )

    # show_image("pose results", inputs[0], wait=False)

    return all_pose_results

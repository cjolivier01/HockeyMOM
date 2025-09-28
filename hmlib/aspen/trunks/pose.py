from typing import Any, Dict

import torch

from hmlib.tasks.multi_pose import multi_pose_task
from hmlib.utils.gpu import StreamTensor

from .base import Trunk


class PoseTrunk(Trunk):
    """
    Runs multi-pose inference using an MMPose inferencer.

    Expects in context:
      - data_to_send: dict containing 'original_images'
      - pose_inferencer: initialized MMPoseInferencer
      - plot_pose: bool (optional)

    Produces in context:
      - data_to_send: updated with 'pose_results'
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        pose_inferencer = context.get("pose_inferencer")
        if pose_inferencer is None:
            return {}

        data_to_send: Dict[str, Any] = context["data_to_send"]
        cur_frame = data_to_send.get("original_images")
        if isinstance(cur_frame, StreamTensor):
            cur_frame = cur_frame.wait()
            data_to_send["original_images"] = cur_frame

        pose_results = multi_pose_task(
            pose_inferencer=pose_inferencer,
            cur_frame=cur_frame,
            show=bool(context.get("plot_pose", False)),
        )
        data_to_send["pose_results"] = pose_results
        return {"data_to_send": data_to_send}


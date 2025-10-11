from typing import Any, Dict, List

import torch

from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_channels_last

from .base import Trunk


class PoseTrunk(Trunk):
    """
    Runs multi-pose inference using an MMPose inferencer.

    Expects in context:
      - data: dict containing 'original_images'
      - pose_inferencer: initialized MMPoseInferencer
      - plot_pose: bool (optional)

    Produces in context:
      - data: updated with 'pose_results'
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        pose_inferencer = context.get("pose_inferencer")
        if pose_inferencer is None:
            return {}

        data: Dict[str, Any] = context["data"]
        cur_frame = data.get("original_images")
        if isinstance(cur_frame, StreamTensor):
            cur_frame = cur_frame.wait()
            data["original_images"] = cur_frame

        # Prepare inputs: iterate per-frame with channels-last layout
        inputs: List[torch.Tensor] = []
        for img in make_channels_last(cur_frame):
            inputs.append(img)

        all_pose_results = []
        show = bool(context.get("plot_pose", False))
        for pose_results in pose_inferencer(
            inputs=inputs, return_datasamples=True, visualize=show, **pose_inferencer.filter_args
        ):
            all_pose_results.append(pose_results)

        if show and getattr(pose_inferencer, "inferencer", None) is not None:
            vis = pose_inferencer.inferencer.visualizer
            if vis is not None:
                for img, pose_result in zip(inputs, all_pose_results):
                    data_sample = pose_result["predictions"]
                    assert len(data_sample) == 1
                    vis.add_datasample(
                        name="pose results",
                        image=img,
                        data_sample=data_sample[0],
                        clone_image=False,
                        draw_gt=False,
                        draw_bbox=False,
                    )

        pose_results = all_pose_results
        data["pose_results"] = pose_results
        return {"data": data}

    def input_keys(self):
        return {"data", "pose_inferencer", "plot_pose"}

    def output_keys(self):
        return {"data"}

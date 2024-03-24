from typing import Any, Union, List

import mmdet
from mmcv.runner import auto_fp16
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from mmtrack.models.mot.byte_track import ByteTrack

from ..builder import MODELS


@MODELS.register_module()
class HmEndToEnd(ByteTrack):
    def __init__(
        self,
        *args,
        post_detection_pipeline: List[Any] = [],
        **kwargs,
    ):
        super(HmEndToEnd, self).__init__(*args, **kwargs)
        self.post_detection_pipeline = post_detection_pipeline
        self.post_detection_composed_pipeline = None

    def __call__(self, *args, **kwargs):
        return super(HmEndToEnd, self).__call__(*args, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        if (
            self.post_detection_pipeline
            and self.post_detection_composed_pipeline is None
        ):
            self.post_detection_composed_pipeline = Compose(
                self.post_detection_pipeline
            )
        results = super(HmEndToEnd, self).forward(
            img, return_loss=return_loss, **kwargs
        )
        if self.post_detection_composed_pipeline is not None:
            results = self.post_detection_composed_pipeline(results)
        return results

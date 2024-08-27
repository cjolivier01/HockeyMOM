from typing import Any, List
import torch

# import mmdet
from mmcv.runner import auto_fp16

# from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose

# from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from mmtrack.models.mot.byte_track import ByteTrack
from mmtrack.core import outs2results, results2outs
from xmodels.SVHNClassifier.model import Model as SVHNClassifier

from ..builder import MODELS


@MODELS.register_module()
class HmNumberClassifier(SVHNClassifier):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(HmNumberClassifier, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return super(HmNumberClassifier, self).__call__(*args, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, img, **kwargs):
        results = super(HmNumberClassifier, self).forward(img)
        return results

    def simple_test(self, data, **kwargs):
        return data

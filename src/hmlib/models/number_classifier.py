from typing import Any, List

# import mmdet
import torch
from mmcv.runner import BaseModule, auto_fp16

# from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmtrack.core import outs2results, results2outs

# from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from mmtrack.models.mot.byte_track import ByteTrack

from xmodels.SVHNClassifier.model import Model as SVHNClassifier

from ..builder import NECKS


@NECKS.register_module()
class HmNumberClassifier(BaseModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(HmNumberClassifier, self).__init__(*args, **kwargs)
        self._classifier = SVHNClassifier(*args, **kwargs)
        self._classifier._params_init_info = []

    def __call__(self, *args, **kwargs):
        return super(HmNumberClassifier, self).__call__(*args, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, img, **kwargs):
        results = super(HmNumberClassifier, self).forward(img)
        return results

    def simple_test(self, data, **kwargs):
        return data

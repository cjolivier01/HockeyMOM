import mmdet
from mmdet.datasets import PIPELINES
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from mmtrack.models.mot.byte_track import ByteTrack

from ..builder import MODELS


@MODELS.register_module()
class HmEndToEnd(ByteTrack):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(HmEndToEnd, self).__init__(*args, **kwargs)

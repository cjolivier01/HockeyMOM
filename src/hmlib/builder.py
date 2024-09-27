import mmdet
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from mmdet.datasets import PIPELINES as MMDET_PIPELINES
from mmdet.models.builder import NECKS as MMTRACK_NECKS
from mmpose.datasets import PIPELINES as MMPOSE_PIPELINES
from mmtrack.models.builder import MODELS as MMTRACK_MODELS

# MODELS = Registry("models", parent=MMTRACK_MODELS)
MODELS = MMTRACK_MODELS
NECKS = MMTRACK_NECKS

HM = Registry("hm")
PIPELINES = MMDET_PIPELINES
POSE_PIPELINES = MMPOSE_PIPELINES

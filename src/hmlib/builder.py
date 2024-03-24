import mmdet
from mmdet.datasets import PIPELINES as MMDET_PIPELINES
from mmtrack.models.builder import MODELS as MMTRACK_MODELS
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version

#MODELS = Registry("models", parent=MMTRACK_MODELS)
MODELS = MMTRACK_MODELS

HM = Registry("hm")
PIPELINES = MMDET_PIPELINES

# def build_model(cfg, train_cfg=None, test_cfg=None):
#     """Build model."""
#     if train_cfg is None and test_cfg is None:
#         return MODELS.build(cfg)
#     else:
#         return MODELS.build(cfg, MODELS,
#                             dict(train_cfg=train_cfg, test_cfg=test_cfg))

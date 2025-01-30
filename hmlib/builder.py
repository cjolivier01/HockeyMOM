import mmdet
from mmengine import Registry

# from mmdet.mmengineimport PIPELINES
# from mmdet.models.builder import NECKS as MMTRACK_NECKS
# from mmpose.datasets import PIPELINES as MMPOSE_PIPELINES
# from mmtrack.models.builder import MODELS as MMTRACK_MODELS

DATASETS = Registry("dataset")
# PIPELINES = Registry("pipeline")
from mmengine import TRANSFORMS as PIPELINES

# MODELS = Registry("models", parent=MMTRACK_MODELS)
# MODELS = MMTRACK_MODELS
# NECKS = MMTRACK_NECKS

HM = Registry("hm")
# PIPELINES = MMDET_PIPELINES
# POSE_PIPELINES = MMPOSE_PIPELINES

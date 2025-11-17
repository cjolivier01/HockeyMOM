"""Registry helpers for hmlib training and inference pipelines.

This module exposes :data:`DATASETS`, :data:`PIPELINES` (an alias for
``mmengine.TRANSFORMS``) and :data:`HM` registries used to register Aspen
trunks and custom components.

@see @ref hmlib.aspen.trunks.base "Aspen trunk base classes" for usage.
"""

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

import mmdet
from mmdet.datasets import PIPELINES
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version


HM = Registry("hm")
PIPELINES = PIPELINES

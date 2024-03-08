#import mmdet.datasets.builder.PIPELINES as PIPELINES
import mmdet
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version

HM = Registry("hm")
PIPELINES = mmdet.datasets.builder.PIPELINES

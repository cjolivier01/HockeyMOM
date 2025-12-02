"""Registry helpers for hmlib training and inference pipelines.

This module exposes :data:`DATASETS`, :data:`PIPELINES` (an alias for
``mmengine.TRANSFORMS``) and :data:`HM` registries used to register Aspen
trunks and custom components.

@see @ref hmlib.aspen.plugins.base_plugin "Aspen trunk base classes" for usage.
"""

from mmengine import Registry

# Prefer the canonical mmengine transforms registry so our custom transforms can
# be used seamlessly alongside upstream components. Fall back to a standalone
# registry if mmengine's alias is unavailable.
try:  # pragma: no cover - import guard
    from mmengine.registry import TRANSFORMS as PIPELINES
except Exception:  # pragma: no cover - optional dependency
    PIPELINES = Registry("pipeline")

DATASETS = Registry("dataset")
HM = Registry("hm")

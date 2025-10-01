from __future__ import annotations

from typing import Any, Dict, Optional

import torch

try:
    from mmengine.config import ConfigDict
    from mmdet.registry import MODELS
except Exception:  # pragma: no cover - optional at runtime
    MODELS = None  # type: ignore
    ConfigDict = dict  # type: ignore

from .base import Trunk


class DetectorFactoryTrunk(Trunk):
    """
    Builds and caches a pure detection model (e.g., YOLOX) from Aspen YAML.

    Params:
      - detector: dict model config (optional)
      - detector_yaml: str path to YAML with a top-level detector definition (optional)
      - data_preprocessor: dict (optional), passed to model as TrackDataPreprocessor-compatible
      - to_device: bool (default True)

    Outputs in context:
      - detector_model: the constructed detection model (eval mode)
    """

    def __init__(
        self,
        detector: Optional[Dict[str, Any]] = None,
        detector_yaml: Optional[str] = None,
        data_preprocessor: Optional[Dict[str, Any]] = None,
        to_device: bool = True,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._detector_dict = detector
        self._detector_yaml = detector_yaml
        self._data_preprocessor = data_preprocessor
        self._to_device = to_device
        self._model = None

    def _load_detector_from_yaml(self, path: str) -> Dict[str, Any]:
        import yaml

        with open(path, "r") as f:
            y = yaml.safe_load(f)
        if isinstance(y, dict) and "detector" in y:
            return y["detector"]
        return y

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if bool(context.get("using_precalculated_detection", False)):
            # Skip building a detector if detections are provided externally
            return {}
        if self._model is None:
            if MODELS is None:
                raise RuntimeError("mmdet MODELS registry is unavailable; cannot build detector.")

            detector_cfg = self._detector_dict
            if detector_cfg is None and self._detector_yaml:
                detector_cfg = self._load_detector_from_yaml(self._detector_yaml)

            # Convert to ConfigDict recursively for mmengine
            def _to_cfg(x):
                if isinstance(x, dict):
                    return ConfigDict({k: _to_cfg(v) for k, v in x.items()})
                if isinstance(x, list):
                    return [_to_cfg(v) for v in x]
                return x

            model_cfg = _to_cfg(detector_cfg)
            if self._data_preprocessor is not None:
                # Attach data_preprocessor to the model config before build
                if not isinstance(model_cfg, dict):
                    # Convert to dict to inject
                    model_cfg = ConfigDict(model_cfg)
                model_cfg["data_preprocessor"] = _to_cfg(self._data_preprocessor)
            model = MODELS.build(model_cfg)

            if hasattr(model, "init_weights"):
                model.init_weights()

            if self._to_device and "device" in context and isinstance(context["device"], torch.device):
                model = model.to(context["device"])  # type: ignore[assignment]
            model.eval()
            self._model = model

        context["detector_model"] = self._model
        return {"detector_model": self._model}

    def input_keys(self):
        return {"device", "using_precalculated_detection"}

    def output_keys(self):
        return {"detector_model"}

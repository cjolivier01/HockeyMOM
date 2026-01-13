from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import torch

from hmlib.config import prepend_root_dir
from hmlib.utils.numpy_pickle_compat import numpy2_pickle_compat

from .base import Plugin

try:
    from mmengine.config import Config, ConfigDict
except Exception:  # pragma: no cover - optional at runtime
    Config = None  # type: ignore
    ConfigDict = dict  # type: ignore


def _import_class(path: str):
    mod_name, _, cls_name = path.rpartition(".")
    if not mod_name:
        raise ValueError(f"Invalid class path: {path}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


class ModelFactoryPlugin(Plugin):
    """Plugin that builds and caches the MMTracking-style end-to-end model.

    Parameters (YAML fields):
      - ``model_class``: str import path (default: ``hmlib.models.end_to_end_plugin.HmEndToEnd``).
      - ``data_preprocessor``: dict (optional).
      - ``detector``: dict (optional).
      - ``detector_mmconfig``: path to an mmengine config with a ``model.detector`` field (optional).
      - ``detector_overrides``: dict to update the detector after loading from mmconfig (optional).
      - ``tracker``: dict (optional).
      - ``post_tracking_pipeline``: List[dict] (optional).
      - ``to_device``: bool (default ``True``) – move model to ``context['device']`` if present.

    Outputs in context:
      - ``model``: the constructed model instance.

    @see @ref hmlib.models.end_to_end_plugin.HmEndToEnd "HmEndToEnd" for the default implementation.
    """

    def __init__(
        self,
        model_class: str = "hmlib.models.end_to_end_plugin.HmEndToEnd",
        data_preprocessor: Optional[Dict[str, Any]] = None,
        detector: Optional[Dict[str, Any]] = None,
        detector_mmconfig: Optional[str] = None,
        detector_yaml: Optional[str] = None,
        detector_overrides: Optional[Dict[str, Any]] = None,
        tracker: Optional[Dict[str, Any]] = None,
        post_tracking_pipeline: Optional[List[Dict[str, Any]]] = None,
        to_device: bool = True,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._model_class = model_class
        self._data_preprocessor = data_preprocessor
        self._detector_dict = detector
        self._detector_mmconfig = detector_mmconfig
        self._detector_overrides = detector_overrides or {}
        self._tracker_dict = tracker
        self._post_track = post_tracking_pipeline
        self._to_device = to_device
        self._model = None
        self._detector_yaml = prepend_root_dir(detector_yaml) if detector_yaml else None
        self._detector_mmconfig = prepend_root_dir(detector_mmconfig) if detector_mmconfig else None

    def _load_detector_from_mmconfig(self, cfg_path: str) -> Dict[str, Any]:
        if Config is None:
            raise RuntimeError("mmengine is not available to load mmconfig.")
        cfg = Config.fromfile(prepend_root_dir(cfg_path))
        # Try common locations for detector
        det = None
        # Prefer a top-level 'detector' variable if present
        if "detector" in cfg:
            cand = cfg.get("detector")
            if isinstance(cand, dict):
                det = cand
        # Otherwise look inside the model dict
        if det is None and isinstance(cfg.get("model"), dict):
            det = cfg.model.get("detector")
        if det is None:
            raise KeyError(f"No detector found in mmconfig at '{cfg_path}'.")
        # Apply overrides
        for k, v in self._detector_overrides.items():
            # dot-notation updates
            cur = det
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        return det

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if self._model is None:
            # Prepare components
            detector = self._detector_dict
            if detector is None and self._detector_yaml:
                import yaml

                with open(self._detector_yaml, "r") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict) and "detector" in loaded:
                    detector = loaded["detector"]
                else:
                    detector = loaded
            if detector is None and self._detector_mmconfig:
                detector = self._load_detector_from_mmconfig(self._detector_mmconfig)

            ModelCls = _import_class(self._model_class)
            kwargs: Dict[str, Any] = {}

            def _to_cfg(x):
                if isinstance(x, dict):
                    return ConfigDict({k: _to_cfg(v) for k, v in x.items()})
                if isinstance(x, list):
                    return [_to_cfg(v) for v in x]
                return x

            # Strip nested data_preprocessor from detector; ByteTrack supplies its own
            if isinstance(detector, dict) and "data_preprocessor" in detector:
                detector = dict(detector)
                detector.pop("data_preprocessor", None)

            if self._data_preprocessor is not None:
                kwargs["data_preprocessor"] = _to_cfg(self._data_preprocessor)
            if detector is not None:
                kwargs["detector"] = _to_cfg(detector)
            if self._tracker_dict is not None:
                kwargs["tracker"] = _to_cfg(self._tracker_dict)
            if self._post_track is not None:
                kwargs["post_tracking_pipeline"] = self._post_track

            model = ModelCls(**kwargs)
            if hasattr(model, "init_weights"):
                with numpy2_pickle_compat():
                    model.init_weights()

            if (
                self._to_device
                and "device" in context
                and isinstance(context["device"], torch.device)
            ):
                model = model.to(context["device"])  # type: ignore[assignment]
            model.eval()
            self._model = model

        # Expose in context
        context["model"] = self._model
        return {"model": self._model}

    def input_keys(self):
        return {"device"}

    def output_keys(self):
        return {"model"}

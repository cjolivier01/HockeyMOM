from typing import Any, Dict, List, Optional

from .base import Trunk


class ModelConfigTrunk(Trunk):
    """
    Configures the MMTracking model instance from Aspen YAML.

    Expects in context:
      - model: instantiated model (e.g., HmEndToEnd)

    Params:
      - post_detection_pipeline: Optional[List[Dict]] (mmcv Compose-style specs)
      - post_tracking_pipeline: Optional[List[Dict]] (mmcv Compose-style specs)
      - other attributes present on model can be set by providing matching keys
        (e.g., `enabled`, `cpp_bytetrack`, etc.)
    """

    def __init__(
        self,
        post_detection_pipeline: Optional[List[Dict[str, Any]]] = None,
        post_tracking_pipeline: Optional[List[Dict[str, Any]]] = None,
        enabled: Optional[bool] = None,
        **extra: Any,
    ):
        super().__init__(enabled=True)
        self._post_det = post_detection_pipeline
        self._post_track = post_tracking_pipeline
        self._enabled_flag = enabled
        self._extra = extra

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        model = context.get("model")
        if model is None:
            return {}

        # Set pipelines if provided
        if self._post_det is not None:
            setattr(model, "post_detection_pipeline", self._post_det)
            # Reset composed pipeline to force rebuild lazily in model
            if hasattr(model, "post_detection_composed_pipeline"):
                setattr(model, "post_detection_composed_pipeline", None)

        if self._post_track is not None:
            setattr(model, "post_tracking_pipeline", self._post_track)
            if hasattr(model, "post_tracking_composed_pipeline"):
                setattr(model, "post_tracking_composed_pipeline", None)

        # Optional enabled flag (for models that support it)
        if self._enabled_flag is not None and hasattr(model, "_enabled"):
            setattr(model, "_enabled", bool(self._enabled_flag))

        # Any extra simple attributes provided
        for k, v in self._extra.items():
            if hasattr(model, k):
                try:
                    setattr(model, k, v)
                except Exception:
                    pass

        return {}

    def input_keys(self):
        return {"model"}

    def output_keys(self):
        return set()


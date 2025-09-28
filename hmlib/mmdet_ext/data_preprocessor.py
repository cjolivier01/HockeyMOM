from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    # If mmdet already provides TrackDataPreprocessor, import it and also
    # ensure it's visible under the mmengine MODELS registry for BaseModel.
    from mmdet.models.data_preprocessors.track_data_preprocessor import TrackDataPreprocessor as _TDPP  # type: ignore
    try:
        from mmengine.registry import MODELS as _E_MODELS  # type: ignore
        try:
            _E_MODELS.register_module(module=_TDPP, name="TrackDataPreprocessor")
        except KeyError:
            # Already registered under this registry scope
            pass
    except Exception:
        pass
except Exception:
    # Fallback: define a thin shim only if needed.
    try:
        from mmengine.registry import MODELS  # type: ignore
        # Prefer the DetDataPreprocessor from mmdet if available for API compatibility
        try:
            from mmdet.models.data_preprocessors import DetDataPreprocessor  # type: ignore
        except Exception:
            from mmdet.models import DetDataPreprocessor  # type: ignore
    except Exception:
        # Extremely old mmengine/mmdet fallback
        from mmdet.registry import MODELS  # type: ignore
        try:
            from mmdet.models import DetDataPreprocessor  # type: ignore
        except Exception:
            DetDataPreprocessor = object  # type: ignore

    @MODELS.register_module(name="TrackDataPreprocessor")  # type: ignore[misc]
    class TrackDataPreprocessor(DetDataPreprocessor):
        """Compatibility shim for TrackDataPreprocessor.

        Registers a minimal subclass under the expected name when the upstream
        implementation is unavailable.
        """

        def __init__(
            self,
            use_det_processor: bool = True,
            batch_augments: Optional[List[Dict[str, Any]]] = None,
            pad_size_divisor: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(batch_augments=batch_augments, pad_size_divisor=pad_size_divisor, **kwargs)  # type: ignore[misc]
            self.use_det_processor = use_det_processor

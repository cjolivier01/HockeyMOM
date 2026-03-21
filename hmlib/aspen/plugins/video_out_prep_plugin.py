from __future__ import annotations

from typing import Any, Dict, Optional

from hmlib.builder import HM
from hmlib.video.video_out_prep import VideoOutputPreparer

from .base import Plugin


@HM.register_module()
class VideoOutPrepPlugin(Plugin):
    """Prepare final video tensors for the sink stage without performing I/O."""

    def __init__(
        self,
        enabled: bool = True,
        video_out_device: Optional[str] = None,
        skip_final_save: bool = False,
        output_width: Optional[str | int] = None,
        output_height: Optional[str | int] = None,
    ) -> None:
        """Construct a pure Aspen video-prep plugin.

        This plugin owns only shape/layout/device preparation and optional
        CUDA-graphable tensor transforms. It intentionally does not open video
        writers, UI windows, or perform any host-side I/O.

        @param enabled: Whether this plugin should execute.
        @param video_out_device: Optional preferred device for prepared tensors.
        @param skip_final_save: When True, preserve the input tensor device and
                                avoid sink-oriented device moves.
        @param output_width: Optional target output width before writing.
        @param output_height: Optional target output height before writing.
        """
        super().__init__(enabled=enabled)
        self._preparer: Optional[VideoOutputPreparer] = None
        self._vo_dev = video_out_device
        self._skip_final_save = bool(skip_final_save)
        self._output_width = output_width
        self._output_height = output_height

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._preparer is not None:
            return
        shared = context.get("shared", {})
        device = self._vo_dev
        if device is None and isinstance(shared, dict):
            # Mirror the sink plugin default: use the shared Aspen device unless
            # the plugin was explicitly pinned elsewhere.
            device = shared.get("device")
        self._preparer = VideoOutputPreparer(
            skip_final_save=self._skip_final_save,
            device=device,
            output_width=self._output_width,
            output_height=self._output_height,
            name="TRACKING",
        )
        self._preparer.set_cuda_graph_enabled(self._cuda_graph_enabled)

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if self._preparer is not None and hasattr(self._preparer, "set_cuda_graph_enabled"):
            self._preparer.set_cuda_graph_enabled(enabled)
        return True

    def input_keys(self):
        if not hasattr(self, "_input_keys"):
            self._input_keys = {
                "img",
                "original_images",
                "shared",
                "video_frame_cfg",
                "end_zone_img",
            }
        return self._input_keys

    def output_keys(self):
        return {"img", "end_zone_img", "video_frame_cfg", "video_out_prepared"}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        self._ensure_initialized(context)
        assert self._preparer is not None
        prepared = self._preparer.prepare_results(context)
        # Only publish the keys that downstream sinks need. The original context
        # stays in AspenNet, so there is no need to echo unrelated values here.
        out = {
            "img": prepared["img"],
            "video_out_prepared": True,
        }
        if "video_frame_cfg" in prepared:
            out["video_frame_cfg"] = prepared["video_frame_cfg"]
        if "end_zone_img" in prepared:
            out["end_zone_img"] = prepared["end_zone_img"]
        return out

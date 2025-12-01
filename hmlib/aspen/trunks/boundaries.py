"""Aspen trunk for managing rink boundary transforms in configs."""

from typing import Any, Dict, Optional

from hmlib.utils.pipeline import update_pipeline_item

from .base import Trunk


class BoundariesTrunk(Trunk):
    """
    Legacy trunk for updating boundary-related transforms on legacy models.

    If a model exposes a legacy post-detection pipeline, this trunk updates it
    with manual boundary lines or auto rink boundaries. Otherwise it is a no-op.

    Expects in context/shared (for legacy path only):
      - model (with optional attribute `post_detection_pipeline`)
      - game_id: str
      - original_clip_box: Optional[list|tuple]
      - top_border_lines: Optional
      - bottom_border_lines: Optional
      - plot_ice_mask: bool (optional)
    """

    def __init__(self, enabled: bool = True, plot_ice_mask: bool = False):
        super().__init__(enabled=enabled)
        self.factory_complete: bool = False
        self._default_plot_ice_mask: bool = bool(plot_ice_mask)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled or self.factory_complete:
            return {}
        model = context.get("model")
        if model is None or not hasattr(model, "post_detection_pipeline"):
            return {}
        pipeline = getattr(model, "post_detection_pipeline", None)
        if not pipeline:
            # No post-detection pipeline to update; nothing to do
            self.factory_complete = True
            return {}
        game_id: Optional[str] = context.get("game_id") or context.get("shared", {}).get("game_id")
        original_clip_box = context.get("original_clip_box") or context.get("shared", {}).get(
            "original_clip_box"
        )
        top_border_lines = context.get("top_border_lines") or context.get("shared", {}).get(
            "top_border_lines"
        )
        bottom_border_lines = context.get("bottom_border_lines") or context.get("shared", {}).get(
            "bottom_border_lines"
        )
        plot_ice_mask: bool = bool(
            context.get("plot_ice_mask")
            or context.get("shared", {}).get("plot_ice_mask", self._default_plot_ice_mask)
        )

        if top_border_lines or bottom_border_lines:
            updated = update_pipeline_item(
                pipeline,
                "BoundaryLines",
                dict(
                    upper_border_lines=top_border_lines,
                    lower_border_lines=bottom_border_lines,
                    original_clip_box=original_clip_box,
                ),
            )
            if updated:
                return {}

        if game_id:
            update_pipeline_item(
                pipeline,
                "IceRinkSegmBoundaries",
                dict(
                    game_id=game_id,
                    original_clip_box=original_clip_box,
                    draw=plot_ice_mask,
                ),
            )
        self.factory_complete = True
        return {}

    def input_keys(self):
        return {
            "model",
            "game_id",
            "original_clip_box",
            "top_border_lines",
            "bottom_border_lines",
            "plot_ice_mask",
        }

    def output_keys(self):
        return set()

"""Draw the ice rink mask and the painted rink landmarks on the stitched frame.

This exists for the stitching graph, which has no detector or tracker trunk and
therefore never builds the ``rink_profile`` that
:class:`~hmlib.aspen.plugins.ice_rink_boundaries_plugins.IceRinkSegmBoundariesPlugin`
relies on. Both models run once, on the first frame, and the result is baked
into a single (color, alpha) layer so every later frame costs one blend.

Placed between ``stitching`` and ``apply_camera`` it rewrites ``img``, so the
overlay reaches the live preview, the camera UI and the encoded output alike.
"""

import logging
import os
from typing import Any, Dict, Optional, Sequence

import torch

from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hmlib.utils.image import image_height, image_width

from .base import Plugin

logger = logging.getLogger(__name__)


class RinkOverlayPlugin(Plugin):
    """
    Overlay the rink mask and/or the painted rink landmarks on the stitched image.

    Expects in context:
      - img: stitched frame tensor (this is what gets annotated)
      - game_id, work_dir (optional, from context or shared)

    Produces in context:
      - img: the same frame with the overlay composited in
    """

    # NOTE: deliberately *not* disable_in_cuda_graph_pipeline. That flag marks a
    # plugin as a deferred sink, and AspenNet._validate_cuda_graph_deferred_nodes
    # requires deferred nodes to be terminal -- this one feeds apply_camera, so
    # setting it would abort the run under --aspen-cuda-graph. Opting out of
    # capture is set_cuda_graph_enabled's job instead.

    def __init__(
        self,
        enabled: bool = True,
        rink_mask: bool = True,
        landmarks: bool = True,
        score_thr: Optional[float] = None,
        inference_scale: Optional[float] = None,
        classes: Optional[Sequence[str]] = None,
        fill_alpha: float = 0.35,
        rink_mask_alpha: float = 0.10,
        outline_thickness: int = 3,
        label_text: bool = False,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None,
        model_config: Optional[str] = None,
        save_debug_frame: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._draw_rink_mask = bool(rink_mask)
        self._draw_landmarks = bool(landmarks)
        self._score_thr = score_thr
        self._inference_scale = inference_scale
        self._classes = list(classes) if classes else None
        self._fill_alpha = float(fill_alpha)
        self._rink_mask_alpha = float(rink_mask_alpha)
        self._outline_thickness = int(outline_thickness)
        self._label_text = bool(label_text)
        self._device = device
        self._checkpoint = checkpoint
        self._model_config = model_config
        self._save_debug_frame = bool(save_debug_frame)

        self._layer: Optional[tuple] = None
        self._layer_shape: Optional[tuple] = None
        self._attempted: bool = False

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        # Mask inference and the first-frame PNG dump cannot be captured.
        self._cuda_graph_enabled = False
        return False

    def _resolve_device(self) -> torch.device:
        if self._device:
            return torch.device(self._device)
        return torch.device("cpu")

    def _build_layer(self, frame: torch.Tensor, context: Dict[str, Any]) -> None:
        """Run both models once and cache the composited overlay for this frame size."""
        from hmlib.segm.rink_landmarks import (
            DEFAULT_SCORE_THRESH,
            build_overlay_layer,
            configure_rink_landmarks,
            filter_landmarks,
            summarize_landmarks,
        )

        shared = context.get("shared") or {}
        game_id = context.get("game_id") or shared.get("game_id")
        if not game_id:
            logger.warning("No game_id available; skipping the rink overlay.")
            return

        height = int(image_height(frame))
        width = int(image_width(frame))
        device = self._resolve_device()

        rink_mask = None
        if self._draw_rink_mask:
            from hmlib.segm.ice_rink import configure_ice_rink_mask

            try:
                rink_profile = configure_ice_rink_mask(
                    game_id=game_id,
                    device=device,
                    expected_shape=torch.Size((height, width)),
                    image=frame,
                )
                rink_mask = (rink_profile or {}).get("combined_mask")
            except Exception as ex:
                # The overlay is decoration. Nothing it can fail at -- a missing
                # game dir (AssertionError), an absent checkpoint, an OOM during
                # inference -- justifies killing a multi-hour encode.
                logger.warning("Rink mask unavailable for the overlay: %s", ex)

        landmarks = None
        if self._draw_landmarks:
            try:
                landmarks = configure_rink_landmarks(
                    game_id=game_id,
                    image=frame,
                    device=device,
                    score_thr=(
                        self._score_thr if self._score_thr is not None else DEFAULT_SCORE_THRESH
                    ),
                    inference_scale=self._inference_scale,
                    checkpoint=self._checkpoint,
                    model_config=self._model_config,
                )
            except Exception as ex:
                logger.warning("Rink landmarks unavailable for the overlay: %s", ex)
            else:
                logger.info("Rink overlay: %s", summarize_landmarks(landmarks))
                if landmarks is not None:
                    landmarks = filter_landmarks(landmarks, include=self._classes)

        self._layer = build_overlay_layer(
            height=height,
            width=width,
            landmarks=landmarks,
            rink_mask=rink_mask,
            rink_mask_alpha=self._rink_mask_alpha,
            fill_alpha=self._fill_alpha,
            outline_thickness=self._outline_thickness,
            label_text=self._label_text,
        )
        if self._layer is None:
            logger.warning("Rink overlay produced nothing to draw.")
            return
        self._layer_shape = (height, width)

        if self._save_debug_frame:
            self._dump_debug_frame(frame, context)

    def _dump_debug_frame(self, frame: torch.Tensor, context: Dict[str, Any]) -> None:
        """Write the first annotated frame so one run always leaves a still to inspect."""
        shared = context.get("shared") or {}
        work_dir = context.get("work_dir") or shared.get("work_dir")
        if not work_dir:
            return
        try:
            import cv2

            from hmlib.utils.image import make_visible_image

            annotated = self._composite(frame)
            if annotated.ndim == 4:
                annotated = annotated[0]
            path = os.path.join(str(work_dir), "rink_overlay_0.png")
            cv2.imwrite(path, make_visible_image(annotated, force_numpy=True))
            logger.info("Wrote rink overlay reference frame to %s", path)
        except Exception as ex:
            logger.warning("Could not write the rink overlay reference frame: %s", ex)

    def _composite(self, frame: torch.Tensor) -> torch.Tensor:
        from hmlib.segm.rink_landmarks import composite_overlay

        color_layer, alpha_layer = self._layer
        return composite_overlay(frame, color_layer, alpha_layer)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        if not self._draw_rink_mask and not self._draw_landmarks:
            return {}

        img = context.get("img")
        if img is None:
            return {}
        frame = unwrap_tensor(img)
        if not torch.is_tensor(frame):
            return {}

        height = int(image_height(frame))
        width = int(image_width(frame))

        if self._layer_shape is not None and self._layer_shape != (height, width):
            # The panorama changed size mid-run (a recalibration, or a different
            # max_output_width). A stale mask would be drawn in the wrong place.
            logger.info(
                "Stitched frame resized to %dx%d; rebuilding the rink overlay.", width, height
            )
            self._layer = None
            self._layer_shape = None
            self._attempted = False

        if self._layer is None:
            if self._attempted:
                return {}
            self._attempted = True
            with self.profile_scope("rink_overlay.build"):
                self._build_layer(frame, context)
            if self._layer is None:
                return {}

        with self.profile_scope("rink_overlay.composite"):
            return {"img": wrap_tensor(self._composite(frame))}

    def input_keys(self):
        return {"img", "game_id", "work_dir", "shared"}

    def output_keys(self):
        return {"img"}


__all__ = ["RinkOverlayPlugin"]

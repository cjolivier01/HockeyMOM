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
        # A str is a Sequence[str], so list("Blue Line") would become an
        # allow-list of single characters. Per-game YAML naturally spells this
        # as a comma-separated string, matching the CLI flag.
        if isinstance(classes, str):
            classes = [name.strip() for name in classes.split(",") if name.strip()]
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
        self._attempted_shape: Optional[tuple] = None
        self._device_layer: Optional[tuple] = None
        self._device_layer_key: Optional[torch.device] = None

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        # Mask inference and the first-frame PNG dump cannot be captured.
        self._cuda_graph_enabled = False
        return False

    def _resolve_device(self, frame: torch.Tensor, context: Dict[str, Any]) -> torch.device:
        """Pick the inference device, preferring the one the frame already lives on."""
        if self._device:
            return torch.device(self._device)
        if torch.is_tensor(frame) and frame.is_cuda:
            return frame.device
        shared = context.get("shared") or {}
        shared_device = shared.get("device")
        if shared_device is not None:
            device = torch.device(shared_device)
            if device.type == "cuda":
                return device
        return torch.device("cpu")

    def _build_layer(self, frame: torch.Tensor, context: Dict[str, Any]) -> None:
        """Run both models once and cache the baked overlay for this frame size."""
        from hmlib.segm.rink_landmarks import (
            DEFAULT_SCORE_THRESH,
            build_overlay_layer,
            configure_rink_landmarks,
            summarize_landmarks,
        )

        shared = context.get("shared") or {}
        game_id = context.get("game_id") or shared.get("game_id")
        if not game_id:
            logger.warning("No game_id available; skipping the rink overlay.")
            return

        height = int(image_height(frame))
        width = int(image_width(frame))
        device = self._resolve_device(frame, context)
        geometry = context.get("camera_input_geometry") or {}
        revision = geometry.get("stitched_geometry_revision")

        rink_mask = None
        if self._draw_rink_mask:
            from hmlib.segm.ice_rink import configure_ice_rink_mask

            rink_profile = self._run_model(
                "rink mask",
                configure_ice_rink_mask,
                device,
                game_id=game_id,
                expected_shape=torch.Size((height, width)),
                image=frame,
                # A preview must never rewrite the game's calibration. persist
                # would overwrite rink_mask_<i>.png and drop
                # rink.ice_contours_geometry_revision, invalidating the
                # tracking pipeline's provenance-checked cache. Passing the
                # revision also stops a mask baked for a different warp of the
                # same size being reused -- the exact thing this draw is for.
                persist=False,
                geometry_revision=revision,
            )
            rink_mask = (rink_profile or {}).get("combined_mask")

        landmarks = None
        if self._draw_landmarks:
            landmarks = self._run_model(
                "rink landmarks",
                configure_rink_landmarks,
                device,
                game_id=game_id,
                image=frame,
                score_thr=(
                    self._score_thr if self._score_thr is not None else DEFAULT_SCORE_THRESH
                ),
                inference_scale=self._inference_scale,
                checkpoint=self._checkpoint,
                model_config=self._model_config,
                include=self._classes,
            )
            logger.info("Rink overlay: %s", summarize_landmarks(landmarks))

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

    def _run_model(self, what: str, call, device: torch.device, **kwargs):
        """Run one calibration model, degrading to no overlay on any failure.

        Retries on the CPU once if the GPU is out of memory, so a tight card
        loses the overlay's speed rather than the overlay itself.
        """
        try:
            return call(device=device, **kwargs)
        except torch.OutOfMemoryError:
            if device.type == "cpu":
                logger.warning("Ran out of memory computing the %s; skipping it.", what)
                return None
            logger.warning("Ran out of GPU memory computing the %s; retrying on the CPU.", what)
        except Exception as ex:
            # Decoration must never abort an encode: a missing game dir raises
            # AssertionError, an absent checkpoint FileNotFoundError, and so on.
            logger.warning("%s unavailable for the overlay: %s", what.capitalize(), ex)
            return None
        try:
            return call(device=torch.device("cpu"), **kwargs)
        except Exception as ex:
            logger.warning("%s unavailable for the overlay: %s", what.capitalize(), ex)
            return None

    def _dump_debug_frame(self, annotated: torch.Tensor, context: Dict[str, Any]) -> None:
        """Write the first annotated frame so a run always leaves a still to inspect.

        Takes the composite ``forward`` already computed rather than blending
        the panorama a second time.
        """
        shared = context.get("shared") or {}
        work_dir = context.get("work_dir") or shared.get("work_dir")
        if not work_dir:
            return
        try:
            import cv2

            from hmlib.utils.image import make_visible_image

            if annotated.ndim == 4:
                annotated = annotated[0]
            path = os.path.join(str(work_dir), "rink_overlay_0.png")
            cv2.imwrite(path, make_visible_image(annotated, force_numpy=True))
            logger.info("Wrote rink overlay reference frame to %s", path)
        except Exception as ex:
            logger.warning("Could not write the rink overlay reference frame: %s", ex)

    def _composite(self, frame: torch.Tensor) -> torch.Tensor:
        from hmlib.segm.rink_landmarks import composite_overlay, overlay_layer_to

        device = frame.device
        if self._device_layer is None or self._device_layer_key != device:
            # Upload once per device. Re-uploading each frame would cost more
            # than the blend: 409 MB on a 12407x4710 panorama.
            self._device_layer = overlay_layer_to(self._layer, device)
            self._device_layer_key = device
        color_layer, alpha_layer = self._device_layer
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
            self._device_layer = None
            self._device_layer_key = None

        if self._layer is None:
            # Key the "already tried and failed" latch on the shape, so a
            # failure at one panorama size does not mute the overlay forever
            # once the stream settles on a size the models can handle.
            if self._attempted_shape == (height, width):
                return {}
            self._attempted_shape = (height, width)
            try:
                with self.profile_scope("rink_overlay.build"):
                    self._build_layer(frame, context)
            except Exception as ex:
                # Everything past the models -- allocating the panorama-sized
                # planes, cv2 contour tracing -- is still only decoration.
                logger.warning("Could not build the rink overlay: %s", ex)
                self._layer = None
            if self._layer is None:
                return {}
            first_build = True
        else:
            first_build = False

        try:
            with self.profile_scope("rink_overlay.composite"):
                annotated = self._composite(frame)
        except Exception as ex:
            # A transient OOM on frame N must cost the overlay, not the encode.
            logger.warning("Rink overlay compositing failed; dropping it: %s", ex)
            self._layer = None
            self._device_layer = None
            self._device_layer_key = None
            return {}

        if first_build and self._save_debug_frame:
            self._dump_debug_frame(annotated, context)
        return {"img": wrap_tensor(annotated)}

    def input_keys(self):
        return {"img", "game_id", "work_dir", "shared", "camera_input_geometry"}

    def output_keys(self):
        return {"img"}


__all__ = ["RinkOverlayPlugin"]

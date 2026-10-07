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


def _is_out_of_memory(error: BaseException) -> bool:
    """Whether ``error`` is a GPU out-of-memory failure in any of its shapes.

    cuDNN and cuBLAS workspace failures surface as a plain RuntimeError rather
    than OutOfMemoryError, and on a card already holding the stitcher that is
    the common one.
    """
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


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
        fill_alpha: Optional[float] = None,
        rink_mask_alpha: Optional[float] = None,
        outline_thickness: Optional[int] = None,
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
        from hmlib.segm.rink_landmarks import (
            DEFAULT_FILL_ALPHA,
            DEFAULT_OUTLINE_THICKNESS,
            DEFAULT_RINK_MASK_ALPHA,
            parse_class_list,
        )

        self._classes = parse_class_list(classes)
        self._fill_alpha = float(DEFAULT_FILL_ALPHA if fill_alpha is None else fill_alpha)
        self._rink_mask_alpha = float(
            DEFAULT_RINK_MASK_ALPHA if rink_mask_alpha is None else rink_mask_alpha
        )
        self._outline_thickness = int(
            DEFAULT_OUTLINE_THICKNESS if outline_thickness is None else outline_thickness
        )
        self._label_text = bool(label_text)
        self._device = device
        self._checkpoint = checkpoint
        self._model_config = model_config
        self._save_debug_frame = bool(save_debug_frame)

        self._layer: Optional[tuple] = None
        # (height, width, stitched_geometry_revision). The revision matters:
        # a recalibration can land on the same output size while moving every
        # pixel, and a mask aligned to the old warp is worse than none.
        self._layer_key: Optional[tuple] = None
        self._attempted_key: Optional[tuple] = None
        self._device_layer: Optional[tuple] = None
        self._composite_failures: int = 0
        self._dumped_debug_frame: bool = False

    # Give up after this many consecutive blend failures rather than retrying
    # a doomed allocation on every frame of a long encode.
    _MAX_COMPOSITE_FAILURES = 3

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        # This plugin never builds a CudaGraphCallable, so there is nothing to
        # capture. Returning False just keeps it out of the supported-plugins
        # report; AspenNet does not use the value to exclude anything.
        self._cuda_graph_enabled = False
        return False

    def _resolve_device(self, frame: torch.Tensor, capped: bool) -> torch.device:
        """Pick the inference device for one model.

        Measured on a 32 GB card: the ice rink model runs at full panorama
        resolution and reliably OOMs next to the stitcher's working set, so it
        stays on the CPU like IceRinkSegmBoundariesPlugin does. The landmark
        model is capped at DEFAULT_MAX_INFERENCE_EDGE and fits, so it gets the
        GPU and the first frame is not stalled for minutes. An explicit
        ``device`` param overrides both.
        """
        if self._device:
            return torch.device(self._device)
        if capped and torch.is_tensor(frame) and frame.is_cuda:
            return frame.device
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
        geometry = context.get("camera_input_geometry") or {}
        revision = geometry.get("stitched_geometry_revision")

        rink_mask = None
        if self._draw_rink_mask:
            from hmlib.segm.ice_rink import configure_ice_rink_mask

            rink_profile = self._run_model(
                "rink mask",
                configure_ice_rink_mask,
                self._resolve_device(frame, capped=False),
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
                self._resolve_device(frame, capped=True),
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
        self._layer_key = (height, width, revision)

    def _run_model(self, what: str, call, device: torch.device, **kwargs):
        """Run one calibration model, degrading to no overlay on any failure.

        Retries on the CPU once if the GPU is out of memory, so a tight card
        loses the overlay's speed rather than the overlay itself.
        """
        try:
            return call(device=device, **kwargs)
        except Exception as ex:
            # Decoration must never abort an encode: a missing game dir raises
            # AssertionError, an absent checkpoint FileNotFoundError, and so on.
            if device.type == "cpu" or not _is_out_of_memory(ex):
                logger.warning("%s unavailable for the overlay: %s", what.capitalize(), ex)
                return None
            logger.warning("Ran out of GPU memory computing the %s; retrying on the CPU.", what)

        # Hand the partially-allocated model's memory back before retrying, so
        # a fragmented allocator does not cascade the OOM into the stitcher.
        torch.cuda.empty_cache()
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
            logger.warning("No work_dir in context; skipping the rink overlay reference frame.")
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
        if self._device_layer is None or self._device_layer[0].device != device:
            # Upload once per device. Re-uploading each frame would cost more
            # than the blend: 409 MB on a 12407x4710 panorama.
            self._device_layer = overlay_layer_to(self._layer, device)
        return composite_overlay(frame, self._device_layer)

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
        geometry = context.get("camera_input_geometry") or {}
        key = (height, width, geometry.get("stitched_geometry_revision"))

        if self._layer_key is not None and self._layer_key != key:
            # The panorama was resized or recalibrated mid-run. A recalibration
            # can keep the output size and still move every pixel, so a stale
            # mask would be drawn in the wrong place either way.
            logger.info("Stitched geometry changed; rebuilding the rink overlay.")
            self._layer = None
            self._layer_key = None
            self._device_layer = None

        if self._layer is None:
            # Key the "already tried and failed" latch on the geometry, so a
            # failure under one geometry does not mute the overlay forever
            # once the stream settles on one the models can handle.
            if self._attempted_key == key:
                return {}
            self._attempted_key = key
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

        try:
            with self.profile_scope("rink_overlay.composite"):
                annotated = self._composite(frame)
        except Exception as ex:
            # A transient OOM here must cost at most the overlay. Keep the
            # baked layer and just drop the device copy so the next frame
            # re-uploads; only give up after this keeps happening.
            self._composite_failures += 1
            self._device_layer = None
            logger.warning(
                "Rink overlay compositing failed (%d/%d): %s",
                self._composite_failures,
                self._MAX_COMPOSITE_FAILURES,
                ex,
            )
            if self._composite_failures >= self._MAX_COMPOSITE_FAILURES:
                logger.warning("Giving up on the rink overlay for this run.")
                self._layer = None
            return {}

        self._composite_failures = 0
        if self._save_debug_frame and not self._dumped_debug_frame:
            self._dump_debug_frame(annotated, context)
            self._dumped_debug_frame = True
        return {"img": wrap_tensor(annotated)}

    def input_keys(self):
        return {"img", "game_id", "work_dir", "shared", "camera_input_geometry"}

    def output_keys(self):
        return {"img"}


__all__ = ["RinkOverlayPlugin"]

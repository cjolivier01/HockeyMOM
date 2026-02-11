from typing import Any, Dict, Optional

import torch
from mmengine.structures import InstanceData

from hmlib.utils.cuda_graph import CudaGraphCallable
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor

from .base import Plugin


class IceRinkSegmBoundariesPlugin(Plugin):
    """
    Prune detections using the ice rink segmentation mask and optionally draw it.

    Runs after the detector trunk and before the tracker trunk. Uses the
    `rink_profile` provided by the inference pipeline (IceRinkSegmConfig)
    via the TrackDataSample metainfo.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - original_images: optional image tensor for drawing
      - game_id, original_clip_box (optional)
      - plot_ice_mask: bool (optional)

    Produces in context:
      - Side-effects: pred_instances filtered by rink mask; optionally updates original_images when drawing
    """

    def __init__(
        self,
        raise_bbox_center_by_height_ratio: float,
        lower_bbox_bottom_by_height_ratio: float,
        enabled: bool = True,
        det_thresh: float = 0.05,
        max_detections_in_mask: Optional[int] = None,
        plot_ice_mask: bool = False,
        cuda_graph: bool = False,
    ):
        super().__init__(enabled=enabled)
        self._det_thresh = float(det_thresh)
        self._max_detections_in_mask: Optional[int] = (
            int(max_detections_in_mask) if max_detections_in_mask is not None else None
        )
        self._segm = None
        self._default_plot_ice_mask: bool = bool(plot_ice_mask)
        self._cuda_graph_enabled = bool(cuda_graph) and not plot_ice_mask
        self._cg: Optional[CudaGraphCallable] = None
        self._cg_device: Optional[torch.device] = None
        self._raise_bbox_center_by_height_ratio = raise_bbox_center_by_height_ratio
        self._lower_bbox_bottom_by_height_ratio = lower_bbox_bottom_by_height_ratio

    def _ensure_pipeline(self, context: Dict[str, Any], draw: bool):
        if self._segm is not None:
            # Update draw flag dynamically if needed
            try:
                self._segm._draw = bool(draw)
            except Exception:
                pass
            return
        # Lazily create the same logic as pipeline-based IceRinkSegmBoundaries
        from hmlib.tracking_utils.ice_rink_segm_boundaries import IceRinkSegmBoundaries

        self._segm = IceRinkSegmBoundaries(
            game_id=context.get("game_id"),
            original_clip_box=context.get("original_clip_box"),
            det_thresh=self._det_thresh,
            max_detections_in_mask=self._max_detections_in_mask,
            draw=bool(draw),
            raise_bbox_center_by_height_ratio=self._raise_bbox_center_by_height_ratio,
            lower_bbox_bottom_by_height_ratio=self._lower_bbox_bottom_by_height_ratio,
        )

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        # Access TrackDataSample list
        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        draw_mask: bool = bool(
            context.get("plot_ice_mask")
            or context.get("shared", {}).get("plot_ice_mask", self._default_plot_ice_mask)
        )
        self._ensure_pipeline(context, draw_mask)

        # Optional original_images for overlay
        original_images = context.get("original_images")
        updated_original_images = None

        # Iterate frames and prune detections using rink mask
        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            det_instances = getattr(img_data_sample, "pred_instances", None)
            if det_instances is None:
                continue

            det_bboxes = unwrap_tensor(det_instances.bboxes)
            det_labels = unwrap_tensor(det_instances.labels)
            det_scores = unwrap_tensor(det_instances.scores)

            # Fast path: CUDA graph the pruning operation (no drawing, fixed max outputs).
            if (
                self._cuda_graph_enabled
                and self._iter_num > 1
                and not draw_mask
                and self._max_detections_in_mask is not None
                and isinstance(det_bboxes, torch.Tensor)
                and det_bboxes.is_cuda
                and det_bboxes.numel() > 0
            ):
                if self._cg is None or self._cg_device != det_bboxes.device:
                    # Build a graph for the current input signature.
                    def _fn(b: torch.Tensor, labels_t: torch.Tensor, s: torch.Tensor):
                        out_b, out_l, out_s, num_valid = self._segm.prune_detections_static(
                            det_bboxes=b, labels=labels_t, scores=s
                        )
                        return out_b, out_l, out_s, num_valid

                    self._cg = CudaGraphCallable(
                        _fn,
                        (det_bboxes, det_labels, det_scores),
                        name="rink_prune",
                    )
                    self._cg_device = det_bboxes.device
                new_bboxes, new_labels, new_scores, num_valid = self._cg(
                    det_bboxes, det_labels, det_scores
                )
                new_inst = InstanceData(
                    bboxes=wrap_tensor(new_bboxes),
                    labels=wrap_tensor(new_labels),
                    scores=wrap_tensor(new_scores),
                )
                if num_valid is not None:
                    new_inst.set_metainfo({"num_detections": num_valid})
                img_data_sample.pred_instances = new_inst
                continue

            pd_input: Dict[str, Any] = {
                "det_bboxes": det_bboxes,
                "labels": det_labels,
                "scores": det_scores,
                "prune_list": ["det_bboxes", "labels", "scores"],
                "data_samples": track_samples,
                "rink_profile": context.get("rink_profile"),
            }
            if original_images is not None:
                pd_input["original_images"] = original_images

            # Reuse the pipeline module's forward for pruning and optional drawing
            out = self._segm(pd_input)

            new_bboxes = out["det_bboxes"]
            new_labels = out["labels"]
            new_scores = out["scores"]

            new_inst = InstanceData(
                bboxes=wrap_tensor(new_bboxes),
                labels=wrap_tensor(new_labels),
                scores=wrap_tensor(new_scores),
            )
            num_detections = out.get("num_detections")
            if num_detections is not None:
                new_inst.set_metainfo({"num_detections": num_detections})
            img_data_sample.pred_instances = new_inst

            # Update original_images if overlay updated
            if draw_mask:
                updated_original_images = wrap_tensor(out["original_images"])

        return (
            {"original_images": updated_original_images}
            if updated_original_images is not None
            else {}
        )

    def input_keys(self):
        return {
            "data_samples",
            "original_images",
            "game_id",
            "original_clip_box",
            "plot_ice_mask",
            "rink_profile",
        }

    def output_keys(self):
        return {"original_images"}


class IceRinkSegmConfigPlugin(Plugin):
    """
    Compute and attach the rink segmentation profile to TrackDataSample metainfo.

    Replaces the old pipeline transform `IceRinkSegmConfig`.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - original_images: optional image tensor
      - game_id: str (from shared or context)
      - device: torch.device

    Produces in context:
      - rink_profile: dict with rink mask + metadata (for downstream pruning/camera seeding)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._rink_profile = None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        if len(track_data_sample) == 0:
            return {}

        # Build rink profile once from the first frame
        if self._rink_profile is None:
            from hmlib.segm.ice_rink import configure_ice_rink_mask

            game_id = context.get("game_id") or context.get("shared", {}).get("game_id")
            # Prefer original_images if available (channels-last), fallback to detection image
            img = context.get("original_images")
            if img is None:
                img = context.get("img") or context.get("inputs")
            # Use first frame
            if isinstance(img, torch.Tensor) and img.ndim >= 3:
                frame0 = img[0]
            else:
                frame0 = None
            # Try to get expected original shape from metainfo
            meta0 = track_data_sample[0].metainfo
            exp_shape = meta0.get("ori_shape") if isinstance(meta0, dict) else None
            if exp_shape is None and isinstance(frame0, torch.Tensor):
                # HxWxC or CxHxW both ok; configure_ice_rink_mask uses expected_shape and image
                if frame0.ndim == 3:
                    if frame0.shape[-1] in (3, 4):
                        exp_shape = torch.Size(frame0.shape[:2])
                    else:
                        exp_shape = torch.Size(frame0.shape[-2:])
            self._rink_profile = configure_ice_rink_mask(
                game_id=game_id,
                # device=device if isinstance(device, torch.device) else torch.device("cpu"),
                device=torch.device("cpu"),
                expected_shape=exp_shape,
                image=frame0,
            )
        if self._rink_profile is None:
            return {}
        return {"rink_profile": self._rink_profile}

    def input_keys(self):
        return {"data_samples", "original_images", "img", "inputs", "game_id"}

    def output_keys(self):
        return {"rink_profile"}

from typing import Any, Dict, Optional

import torch
from mmengine.structures import InstanceData

from hmlib.log import get_logger

from .base import Plugin


class IceRinkSegmBoundariesPlugin(Plugin):
    """
    Prune detections using the ice rink segmentation mask and optionally draw it.

    Runs after the detector trunk and before the tracker trunk. Uses the
    `rink_profile` provided by the inference pipeline (IceRinkSegmConfig)
    via the TrackDataSample metainfo.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample) and optional 'original_images'
      - game_id, original_clip_box (optional)
      - plot_ice_mask: bool (optional)

    Produces in context:
      - data: with per-frame pred_instances filtered by rink mask; optionally overlayed mask
    """

    def __init__(
        self,
        enabled: bool = True,
        det_thresh: float = 0.05,
        max_detections_in_mask: Optional[int] = None,
        plot_ice_mask: bool = False,
    ):
        super().__init__(enabled=enabled)
        self._det_thresh = float(det_thresh)
        self._max_detections_in_mask: Optional[int] = (
            int(max_detections_in_mask) if max_detections_in_mask is not None else None
        )
        self._segm = None
        self._default_plot_ice_mask: bool = bool(plot_ice_mask)

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
        )

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context.get("data") or {}
        if not data:
            return {}

        # Access TrackDataSample list
        track_samples = data.get("data_samples")
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
        original_images = data.get("original_images")

        # Iterate frames and prune detections using rink mask
        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            det_instances = getattr(img_data_sample, "pred_instances", None)
            if det_instances is None:
                continue

            det_bboxes = det_instances.bboxes
            det_labels = det_instances.labels
            det_scores = det_instances.scores
            det_src_pose_idx = getattr(det_instances, "source_pose_index", None)

            pd_input: Dict[str, Any] = {
                "det_bboxes": det_bboxes,
                "labels": det_labels,
                "scores": det_scores,
                "prune_list": ["det_bboxes", "labels", "scores"],
                "data_samples": data.get("data_samples"),
                "rink_profile": context.get("rink_profile"),
            }
            if original_images is not None:
                pd_input["original_images"] = original_images

            # Reuse the pipeline module's forward for pruning and optional drawing
            out = self._segm(pd_input)

            new_bboxes = out["det_bboxes"]
            new_labels = out["labels"]
            new_scores = out["scores"]

            # Try to propagate source pose indices through post-det filtering
            new_src_pose_idx: Optional[torch.Tensor] = None
            try:
                if det_src_pose_idx is not None:
                    ob = det_bboxes
                    nb = new_bboxes
                    if not isinstance(ob, torch.Tensor):
                        ob = torch.as_tensor(ob)
                    if not isinstance(nb, torch.Tensor):
                        nb = torch.as_tensor(nb)
                    if ob.ndim == 1:
                        ob = ob.reshape(-1, 4)
                    if nb.ndim == 1:
                        nb = nb.reshape(-1, 4)
                    if len(nb) == len(ob):
                        new_src_pose_idx = det_src_pose_idx
                    else:
                        new_src_pose_idx = torch.full(
                            (len(nb),), -1, dtype=torch.int64, device=nb.device
                        )
                        if len(ob) and len(nb):
                            # First try exact match with tolerance
                            for j in range(len(nb)):
                                eq = (
                                    torch.isclose(nb[j], ob).all(dim=1)
                                    if len(ob)
                                    else torch.zeros((0,), dtype=torch.bool)
                                )
                                match_idx = torch.nonzero(eq).reshape(-1)
                                if len(match_idx) == 1:
                                    k = int(match_idx[0].item())
                                    try:
                                        new_src_pose_idx[j] = det_src_pose_idx[k]
                                    except Exception:
                                        pass
                            # If unmatched remain, map by IoU best
                            if (new_src_pose_idx < 0).any():
                                try:
                                    from hmlib.tracking_utils.utils import bbox_iou as _bbox_iou
                                except Exception:
                                    from hmlib.utils.utils import bbox_iou as _bbox_iou
                                iou = _bbox_iou(
                                    nb.to(dtype=torch.float32),
                                    ob.to(dtype=torch.float32),
                                    x1y1x2y2=True,
                                )
                                best_iou, best_idx = torch.max(iou, dim=1)
                                for j in range(len(nb)):
                                    if new_src_pose_idx[j] < 0 and best_iou[j] > 0:
                                        try:
                                            new_src_pose_idx[j] = int(
                                                det_src_pose_idx[int(best_idx[j].item())]
                                            )
                                        except Exception:
                                            pass
            except Exception:
                new_src_pose_idx = None

            new_inst = InstanceData(
                bboxes=new_bboxes,
                labels=new_labels,
                scores=new_scores,
            )
            num_detections = out.get("num_detections")
            if num_detections is not None:
                new_inst.set_metainfo({"num_detections": num_detections})
            if new_src_pose_idx is not None:
                new_inst.source_pose_index = new_src_pose_idx.to(dtype=torch.int64)
            img_data_sample.pred_instances = new_inst

            # Update original_images if overlay updated
            if "original_images" in out:
                data["original_images"] = out["original_images"]

        return {"data": data}

    def input_keys(self):
        return {"data", "game_id", "original_clip_box", "plot_ice_mask", "rink_profile"}

    def output_keys(self):
        return {"data"}


class IceRinkSegmConfigPlugin(Plugin):
    """
    Compute and attach the rink segmentation profile to TrackDataSample metainfo.

    Replaces the old pipeline transform `IceRinkSegmConfig`.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample) and optional 'original_images'
      - game_id: str (from shared or context)
      - device: torch.device

    Produces in context:
      - data: unchanged reference (with 'data_samples' metainfo updated)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._rink_profile = None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context.get("data") or {}
        track_samples = data.get("data_samples")
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
            try:
                from hmlib.segm.ice_rink import confgure_ice_rink_mask

                game_id = context.get("game_id") or context.get("shared", {}).get("game_id")
                # Prefer original_images if available (channels-last), fallback to detection image
                img = data.get("original_images")
                if img is None:
                    img = data.get("img")
                # Use first frame
                if isinstance(img, torch.Tensor) and img.ndim >= 3:
                    frame0 = img[0]
                else:
                    frame0 = None
                # Try to get expected original shape from metainfo
                meta0 = track_data_sample[0].metainfo
                exp_shape = meta0.get("ori_shape") if isinstance(meta0, dict) else None
                if exp_shape is None and isinstance(frame0, torch.Tensor):
                    # HxWxC or CxHxW both ok; confgure_ice_rink_mask uses expected_shape and image
                    if frame0.ndim == 3:
                        if frame0.shape[-1] in (3, 4):
                            exp_shape = torch.Size(frame0.shape[:2])
                        else:
                            exp_shape = torch.Size(frame0.shape[-2:])
                self._rink_profile = confgure_ice_rink_mask(
                    game_id=game_id,
                    # device=device if isinstance(device, torch.device) else torch.device("cpu"),
                    device=torch.device("cpu"),
                    expected_shape=exp_shape,
                    image=frame0,
                )
            except Exception as ex:
                get_logger(__name__).exception("Failed to configure ice rink mask: %s", ex)
                self._rink_profile = None
        results = dict(data=data)
        if self._rink_profile is not None:
            results["rink_profile"] = self._rink_profile
        return results

    def input_keys(self):
        return {"data", "game_id"}

    def output_keys(self):
        return {"data", "rink_profile"}

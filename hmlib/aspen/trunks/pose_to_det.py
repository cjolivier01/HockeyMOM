from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.structures import InstanceData

from .base import Trunk


class InitDataToSendTrunk(Trunk):
    """
    Initializes a minimal `data_to_send` dict from `data` so upstream trunks
    (e.g., PoseTrunk) can read `original_images` before tracking occurs.

    Expects in context:
      - data: dict that may contain 'original_images'

    Produces in context:
      - data_to_send: dict with at least 'original_images' populated if available
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        data: Dict[str, Any] = context.get("data", {})
        existing: Dict[str, Any] = context.get("data_to_send", {})
        out: Dict[str, Any] = dict(existing)
        if "original_images" in data and "original_images" not in out:
            out["original_images"] = data["original_images"]
        return {"data_to_send": out}

    def input_keys(self):
        return {"data", "data_to_send"}

    def output_keys(self):
        return {"data_to_send"}


class PoseToDetTrunk(Trunk):
    """
    Converts pose inferencer outputs into detection `pred_instances` for each
    frame, allowing the tracker trunk to run without a separate detector.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample or [TrackDataSample])
      - data_to_send: dict with 'pose_results' produced by PoseTrunk
      - using_precalculated_detection: bool (optional)

    Produces in context:
      - data: updated so each frame's data_sample has `pred_instances` with
              fields: bboxes (Nx4), scores (N), labels (N)
              Also mirrors pose_results into data for downstream preservation.
    """

    def __init__(
        self,
        enabled: bool = True,
        default_label: int = 0,
        score_key: Optional[str] = None,
        score_adder: Optional[float] = None,
    ):
        super().__init__(enabled=enabled)
        self._default_label = int(default_label)
        # Prefer 'bbox_scores'; fall back to provided or common alternatives
        self._score_key = score_key
        self._score_adder = float(score_adder) if score_adder is not None else 0.25

    def _extract_instances(self, pose_result_item: Any):
        """Return InstanceData-like fields (bboxes, scores) from a pose result item.

        pose_result_item is expected to be a dict with key 'predictions' -> list of
        PoseDataSample(s). We use the first one.
        """
        try:
            preds = pose_result_item.get("predictions")
            if isinstance(preds, list) and len(preds) >= 1:
                ds = preds[0]
                inst = getattr(ds, "pred_instances", None)
                return inst
        except Exception:
            pass
        return None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        if bool(context.get("using_precalculated_detection", False)):
            return {}

        data: Dict[str, Any] = context.get("data", {})
        data_to_send: Dict[str, Any] = context.get("data_to_send", {})
        pose_results: Optional[List[Any]] = data_to_send.get("pose_results")
        if not pose_results:
            # Nothing to convert
            return {}

        # Access TrackDataSample list
        track_samples = data.get("data_samples")
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        if track_data_sample is None:
            return {}

        video_len = len(track_data_sample)
        # Pose results are per-frame
        if len(pose_results) != video_len:
            # Best-effort: clip to min length
            video_len = min(video_len, len(pose_results))

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            inst = self._extract_instances(pose_results[frame_index])
            if inst is None:
                # No instances for this frame
                continue

            # bboxes
            bboxes = None
            try:
                if hasattr(inst, "bboxes"):
                    bboxes = inst.bboxes
            except Exception:
                bboxes = None

            # If missing, try to derive boxes from keypoints
            if bboxes is None and hasattr(inst, "keypoints"):
                kpts = inst.keypoints  # (N, K, 2)
                if isinstance(kpts, torch.Tensor) and kpts.ndim >= 3 and kpts.shape[-1] >= 2:
                    x = kpts[..., 0]
                    y = kpts[..., 1]
                    x1 = torch.min(x, dim=1).values
                    y1 = torch.min(y, dim=1).values
                    x2 = torch.max(x, dim=1).values
                    y2 = torch.max(y, dim=1).values
                    bboxes = torch.stack([x1, y1, x2, y2], dim=1)

            if bboxes is None:
                # Nothing to attach for this frame
                continue

            # scores: prefer bbox_scores, then provided score_key, then 'scores',
            # else fallback to mean keypoint score or ones.
            scores = None
            try:
                if hasattr(inst, "bbox_scores"):
                    scores = inst.bbox_scores
            except Exception:
                scores = None
            if scores is None and self._score_key and hasattr(inst, self._score_key):
                try:
                    scores = getattr(inst, self._score_key)
                except Exception:
                    scores = None
            if scores is None and hasattr(inst, "scores"):
                try:
                    scores = inst.scores
                except Exception:
                    scores = None
            if scores is None and hasattr(inst, "keypoint_scores"):
                try:
                    kps = inst.keypoint_scores  # (N, K)
                    if isinstance(kps, torch.Tensor) and kps.ndim >= 2:
                        scores = torch.mean(kps, dim=1)
                except Exception:
                    scores = None
            if scores is None:
                scores = torch.ones((bboxes.shape[0],), dtype=torch.float32, device=bboxes.device)

            # labels: assign default category (e.g., person=0)
            labels = torch.full((bboxes.shape[0],), int(self._default_label), dtype=torch.long, device=bboxes.device)

            if self._score_adder is not None and self._score_adder != 0.0:
                scores += self._score_adder
                scores = (
                    torch.clamp(scores, 0.0, 1.0) if isinstance(scores, torch.Tensor) else np.clip(scores, 0.0, 1.0)
                )

            new_inst = InstanceData()
            new_inst.bboxes = bboxes
            new_inst.scores = scores
            new_inst.labels = labels

            img_data_sample.pred_instances = new_inst

        # Preserve pose_results for downstream consumers by mirroring into `data`.
        # TrackerTrunk will copy `data` into `data_to_send`, so pose_results survive.
        data["pose_results"] = pose_results

        return {"data": data}

    def input_keys(self):
        return {"data", "data_to_send", "using_precalculated_detection"}

    def output_keys(self):
        return {"data"}

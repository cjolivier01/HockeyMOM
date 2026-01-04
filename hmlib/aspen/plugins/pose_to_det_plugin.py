from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.structures import InstanceData

from .base import Plugin


class PoseToDetPlugin(Plugin):
    """
    Converts pose inferencer outputs into detection `pred_instances` for each
    frame, allowing the tracker trunk to run without a separate detector.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample or [TrackDataSample])
      - data: dict with 'pose_results' produced by PosePlugin
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
                if inst is not None:
                    return inst
                # Support simplified dict predictions loaded from PoseDataFrame
                if isinstance(ds, dict):
                    keys = [
                        "bboxes",
                        "scores",
                        "bbox_scores",
                        "labels",
                        "keypoints",
                        "keypoint_scores",
                    ]
                    attrs = {k: ds[k] for k in keys if k in ds}
                    if attrs:
                        return SimpleNamespace(**attrs)
        except Exception:
            pass
        return None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        if bool(context.get("using_precalculated_detection", False)):
            return {}

        data: Dict[str, Any] = context.get("data", {})
        pose_results: Optional[List[Any]] = data.get("pose_results")
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

        def _to_bboxes_2d(x):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.ndim == 1:
                # If empty, reshape to (0, 4); if size==4, make (1,4)
                if x.numel() == 0:
                    return x.reshape(0, 4)
                x = x.unsqueeze(0)
            # Only keep xyxy if extra cols are present
            if x.size(-1) > 4:
                x = x[..., :4]
            return x

        def _to_tensor_1d(x, dtype=None, device=None):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, device=device)
            if x.ndim > 1:
                # squeeze trailing dims like (N,1)->(N)
                x = x.view(x.shape[0])
            if x.ndim == 0:
                x = x.unsqueeze(0)
            if dtype is not None:
                x = x.to(dtype=dtype)
            if device is not None and x.device != device:
                x = x.to(device=device)
            return x

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

            # scores: prefer bbox_scores; if missing, derive from keypoint_scores; fallback to ones.
            scores = None
            try:
                if hasattr(inst, "bbox_scores"):
                    scores = getattr(inst, "bbox_scores")
            except Exception:
                scores = None
            if scores is None and hasattr(inst, "keypoint_scores"):
                try:
                    kps = inst.keypoint_scores  # (N, K)
                    if isinstance(kps, torch.Tensor) and kps.ndim >= 2:
                        scores = torch.mean(kps, dim=1)
                except Exception:
                    scores = None
            # Normalize shapes and lengths
            bboxes = _to_bboxes_2d(bboxes)
            N = int(bboxes.shape[0])
            if scores is None:
                scores = torch.ones((N,), dtype=torch.float32, device=bboxes.device)
            scores = _to_tensor_1d(scores, dtype=torch.float32, device=bboxes.device)
            if len(scores) != N:
                # If a single score is provided, broadcast it; otherwise fallback to ones
                if len(scores) == 1 and N > 1:
                    scores = scores.expand(N).clone()
                else:
                    scores = torch.ones((N,), dtype=torch.float32, device=bboxes.device)

            # labels: assign default category (e.g., person=0)
            labels = torch.full(
                (N,), int(self._default_label), dtype=torch.long, device=bboxes.device
            )

            if self._score_adder is not None and self._score_adder != 0.0:
                scores += self._score_adder
                scores = (
                    torch.clamp(scores, 0.0, 1.0)
                    if isinstance(scores, torch.Tensor)
                    else np.clip(scores, 0.0, 1.0)
                )

            new_inst = InstanceData()
            new_inst.bboxes = bboxes
            new_inst.scores = scores
            new_inst.labels = labels

            img_data_sample.pred_instances = new_inst

        # Preserve pose_results for downstream consumers by mirroring into `data`.
        # TrackerPlugin will copy `data` into `data`, so pose_results survive.
        data["pose_results"] = pose_results

        return {"data": data}

    def input_keys(self):
        return {"data", "using_precalculated_detection"}

    def output_keys(self):
        return {"data"}

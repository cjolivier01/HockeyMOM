from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import numpy as np
import torch
from mmengine.structures import InstanceData

from .base import Trunk


class SaveDetectionsTrunk(Trunk):
    """
    Saves per-frame detections into `detection_dataframe`.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample or [TrackDataSample])
      - frame_id: int for first frame in batch
      - detection_dataframe: DetectionDataFrame
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = context.get("detection_dataframe")
        if df is None:
            return {}

        data: Dict[str, Any] = context.get("data", {})
        track_samples = data.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        frame_id0: int = int(context.get("frame_id", -1))
        video_len = len(track_data_sample)
        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            inst = getattr(img_data_sample, "pred_instances", None)
            if inst is None:
                # No detections: still record an empty frame
                df.add_frame_records(
                    frame_id=int(frame_id0 + i),
                    scores=np.empty((0,), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    bboxes=np.empty((0, 4), dtype=np.float32),
                    pose_indices=np.empty((0,), dtype=np.int64),
                )
                continue
            # Determine frame id
            fid = img_data_sample.metainfo.get("frame_id", None)
            try:
                if isinstance(fid, torch.Tensor):
                    fid = int(fid.reshape([1])[0].item())
            except Exception:
                fid = None
            if fid is None:
                fid = frame_id0 + i

            pose_indices = getattr(inst, "source_pose_index", None)
            df.add_frame_records(
                frame_id=int(fid),
                scores=getattr(inst, "scores", np.empty((0,), dtype=np.float32)),
                labels=getattr(inst, "labels", np.empty((0,), dtype=np.int64)),
                bboxes=getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
                pose_indices=pose_indices,
            )

        return {}

    def input_keys(self):
        return {"data", "frame_id", "detection_dataframe"}

    def output_keys(self):
        return set()


class SaveTrackingTrunk(Trunk):
    """
    Saves per-frame tracking results into `tracking_dataframe`.

    Expects in context:
      - data: dict with 'data_samples'
      - frame_id: int for first frame in batch
      - tracking_dataframe: TrackingDataFrame
      - jersey_results: Optional per-frame jersey info list
    """

    def __init__(self, enabled: bool = True, pose_iou_thresh: float = 0.3):
        super().__init__(enabled=enabled)
        # Default fallback IoU threshold if we must infer mapping
        self._pose_iou_thresh = float(pose_iou_thresh)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = context.get("tracking_dataframe")
        if df is None:
            return {}

        data: Dict[str, Any] = context.get("data", {})
        jersey_results_all = data.get("jersey_results") or context.get("jersey_results")
        frame_id0: int = int(context.get("frame_id", -1))
        pose_results_all = data.get("pose_results")  # mirrored by PoseToDetTrunk

        track_samples = data.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        video_len = len(track_data_sample)
        def _extract_pose_bboxes(pose_item: Any):
            # Borrow the logic from PoseToDetTrunk for deriving bboxes
            try:
                preds = pose_item.get("predictions")
            except Exception:
                preds = None
            if not isinstance(preds, list) or not preds:
                return torch.empty((0, 4), dtype=torch.float32)
            ds = preds[0]
            inst = getattr(ds, "pred_instances", None)
            # Helper to ensure (N,4) xyxy
            def _to_bboxes_2d(x):
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if x.ndim == 1:
                    if x.numel() == 0:
                        return x.reshape(0, 4)
                    x = x.unsqueeze(0)
                if x.size(-1) > 4:
                    x = x[..., :4]
                return x.to(dtype=torch.float32)
            # Prefer explicit bboxes
            if inst is not None and hasattr(inst, "bboxes"):
                try:
                    return _to_bboxes_2d(inst.bboxes)
                except Exception:
                    pass
            # Fallback: compute from keypoints
            kpts = None
            if inst is not None and hasattr(inst, "keypoints"):
                kpts = inst.keypoints
            elif isinstance(ds, dict) and "keypoints" in ds:
                kpts = ds["keypoints"]
            if isinstance(kpts, torch.Tensor) and kpts.ndim >= 3 and kpts.shape[-1] >= 2:
                x = kpts[..., 0]
                y = kpts[..., 1]
                x1 = torch.min(x, dim=1).values
                y1 = torch.min(y, dim=1).values
                x2 = torch.max(x, dim=1).values
                y2 = torch.max(y, dim=1).values
                return torch.stack([x1, y1, x2, y2], dim=1).to(dtype=torch.float32)
            return torch.empty((0, 4), dtype=torch.float32)

        # IoU util expects xyxy if flag set
        try:
            from hmlib.tracking_utils.utils import bbox_iou as _bbox_iou
        except Exception:
            from hmlib.utils.utils import bbox_iou as _bbox_iou

        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            inst = getattr(img_data_sample, "pred_track_instances", None)
            if inst is None:
                # No tracks: still record an empty frame
                df.add_frame_records(
                    frame_id=frame_id0 + i,
                    tracking_ids=np.empty((0,), dtype=np.int64),
                    tlbr=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    jersey_info=None,
                    pose_indices=np.empty((0,), dtype=np.int64),
                )
                continue
            jersey_results = (
                jersey_results_all[i] if isinstance(jersey_results_all, list) and i < len(jersey_results_all) else None
            )

            # Prefer direct propagation if tracker attached source indices
            pose_indices = getattr(inst, "source_pose_index", None)
            if pose_indices is None:
                # Fallback: infer by IoU if pose results exist in context
                try:
                    if isinstance(pose_results_all, list) and i < len(pose_results_all):
                        track_bboxes = getattr(inst, "bboxes", None)
                        if track_bboxes is not None:
                            tb = track_bboxes
                            if not isinstance(tb, torch.Tensor):
                                tb = torch.as_tensor(tb)
                            if tb.ndim == 1:
                                if tb.numel() == 0:
                                    tb = tb.reshape(0, 4)
                                else:
                                    tb = tb.unsqueeze(0)
                            tb = tb.to(dtype=torch.float32)
                            pb = _extract_pose_bboxes(pose_results_all[i])
                            if pb is not None and len(pb) and len(tb):
                                iou = _bbox_iou(tb, pb, x1y1x2y2=True)  # (Nt, Np)
                                best_iou, best_idx = torch.max(iou, dim=1)
                                pose_indices = torch.where(
                                    best_iou >= self._pose_iou_thresh,
                                    best_idx.to(dtype=torch.int64),
                                    torch.full_like(best_idx, fill_value=-1, dtype=torch.int64),
                                )
                            else:
                                pose_indices = torch.full((len(tb),), -1, dtype=torch.int64)
                except Exception:
                    pose_indices = None
            df.add_frame_records(
                frame_id=frame_id0 + i,
                tracking_ids=getattr(inst, "instances_id", np.empty((0,), dtype=np.int64)),
                tlbr=getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
                scores=getattr(inst, "scores", np.empty((0,), dtype=np.float32)),
                labels=getattr(inst, "labels", np.empty((0,), dtype=np.int64)),
                jersey_info=jersey_results,
                pose_indices=pose_indices,
            )
        return {}

    def input_keys(self):
        return {"data", "frame_id", "tracking_dataframe", "jersey_results"}

    def output_keys(self):
        return set()


class SavePoseTrunk(Trunk):
    """
    Saves per-frame pose results from `data_to_send['pose_results']` into `pose_dataframe`.

    We serialize a simplified structure capturing keypoints/bboxes/scores to JSON.

    Expects in context:
      - data_to_send: dict with 'pose_results' list
      - frame_id: int for first frame in batch
      - pose_dataframe: PoseDataFrame
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    @staticmethod
    def _to_list(x):
        try:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass
        return x

    @classmethod
    def _simplify_pose_item(cls, pose_result_item: Any) -> Dict[str, Any]:
        preds = None
        try:
            preds = pose_result_item.get("predictions")
        except Exception:
            preds = None
        out_preds: List[Dict[str, Any]] = []
        if isinstance(preds, list):
            for ds in preds:
                inst = getattr(ds, "pred_instances", None)
                item: Dict[str, Any] = {}
                if inst is not None:
                    for k in ("bboxes", "scores", "bbox_scores", "labels", "keypoints", "keypoint_scores"):
                        if hasattr(inst, k):
                            item[k] = cls._to_list(getattr(inst, k))
                out_preds.append(item)
        return {"predictions": out_preds}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = context.get("pose_dataframe")
        if df is None:
            return {}

        data_to_send: Dict[str, Any] = context.get("data_to_send", {})
        pose_results: Optional[List[Any]] = data_to_send.get("pose_results")

        frame_id0: int = int(context.get("frame_id", -1))
        # If pose_results is missing, write empty entries for each frame
        if not pose_results:
            data: Dict[str, Any] = context.get("data", {})
            track_samples = data.get("data_samples")
            if isinstance(track_samples, list):
                track_data_sample = track_samples[0]
            else:
                track_data_sample = track_samples
            video_len = len(track_data_sample) if track_data_sample is not None else 0
            for i in range(video_len):
                df.add_frame_records(frame_id=frame_id0 + i, pose_json=json.dumps({"predictions": []}))
        else:
            for i, item in enumerate(pose_results):
                simp = self._simplify_pose_item(item)
                df.add_frame_records(frame_id=frame_id0 + i, pose_json=json.dumps(simp))
        return {}

    def input_keys(self):
        return {"data_to_send", "frame_id", "pose_dataframe"}

    def output_keys(self):
        return set()

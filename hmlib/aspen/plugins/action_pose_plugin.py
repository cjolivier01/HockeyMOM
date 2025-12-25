from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData

from .base import Plugin
from hmlib.tracking_utils.utils import get_track_mask


class ActionFromPosePlugin(Plugin):
    """
    Runs skeleton-based action recognition (MMAction2) per tracked player.

    Expects in context:
      - data: dict with keys:
          - data_samples: TrackDataSample or [TrackDataSample]
          - original_images: Tensor [T, C, H, W] or [T, H, W, C]
          - pose_results: List[dict] as produced by PosePlugin/PoseToDetPlugin
      - action_recognizer: mmaction model (from ActionRecognizerFactoryPlugin)
      - action_label_map: Optional[List[str]] label names

    Produces in context:
      - data: with key 'action_results': List[List[Dict]] per frame

    Notes:
      - We map track IDs to pose indices per frame using
        pred_track_instances.source_pose_index when available; otherwise
        we fallback to an IoU-based assignment derived from SaveTrackingPlugin.
      - For each active track across the current clip, we build a per-track
        keypoint sequence and run inference, then attach the action label
        (top-1) per frame for that track.
    """

    def __init__(self, enabled: bool = True, top_k: int = 1, score_threshold: float = 0.0):
        super().__init__(enabled=enabled)
        self._top_k = int(top_k)
        self._score_threshold = float(score_threshold)

    @staticmethod
    def _ensure_mmaction_imported():
        try:
            import mmaction  # noqa: F401

            return
        except Exception:
            pass
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        vendored = os.path.join(root, "openmm", "mmaction2")
        if os.path.isdir(vendored) and vendored not in sys.path:
            sys.path.insert(0, vendored)
        import mmaction  # type: ignore  # noqa: F401

    @staticmethod
    def _extract_pose_arrays(pose_result_item: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return (keypoints [N,K,2], keypoint_scores [N,K]) from a pose result item.

        Supports both MMPose DataSample and simplified dict written by SavePosePlugin.
        Returns empty arrays if not present.
        """
        try:
            preds = pose_result_item.get("predictions")
        except Exception:
            preds = None
        if isinstance(preds, list) and len(preds) >= 1:
            ds = preds[0]
            inst = getattr(ds, "pred_instances", None)
            if inst is not None:
                try:
                    kpts = getattr(inst, "keypoints", None)
                    kps = getattr(inst, "keypoint_scores", None)
                    if kpts is None:
                        return np.empty((0, 17, 2), dtype=np.float32), np.empty(
                            (0, 17), dtype=np.float32
                        )
                    if isinstance(kpts, torch.Tensor):
                        kpts = kpts.detach().cpu().numpy()
                    if isinstance(kps, torch.Tensor):
                        kps = kps.detach().cpu().numpy()
                    if kps is None:
                        kps = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
                    return kpts.astype(np.float32), kps.astype(np.float32)
                except Exception:
                    pass
            if isinstance(ds, dict):
                kpts = ds.get("keypoints")
                kps = ds.get("keypoint_scores")
                if kpts is None:
                    return np.empty((0, 17, 2), dtype=np.float32), np.empty(
                        (0, 17), dtype=np.float32
                    )
                kpts = np.asarray(kpts, dtype=np.float32)
                if kps is None:
                    kps = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
                else:
                    kps = np.asarray(kps, dtype=np.float32)
                return kpts, kps
        return np.empty((0, 17, 2), dtype=np.float32), np.empty((0, 17), dtype=np.float32)

    @staticmethod
    def _image_shape_from_tensor(frames: torch.Tensor) -> Tuple[int, int]:
        # frames is [T, C, H, W] or [T, H, W, C]
        if frames.ndim != 4:
            raise ValueError("original_images must be 4D tensor")
        # try C-first then C-last
        H = int(frames.shape[2] if frames.shape[1] in (1, 3, 4) else frames.shape[1])
        W = int(frames.shape[3] if frames.shape[1] in (1, 3, 4) else frames.shape[2])
        return H, W

    @staticmethod
    def _bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (Na,4), b: (Nb,4), returns (Na,Nb)
        xa1, ya1, xa2, ya2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        xb1, yb1, xb2, yb2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        inter_x1 = torch.maximum(xa1[:, None], xb1[None, :])
        inter_y1 = torch.maximum(ya1[:, None], yb1[None, :])
        inter_x2 = torch.minimum(xa2[:, None], xb2[None, :])
        inter_y2 = torch.minimum(ya2[:, None], yb2[None, :])
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter = inter_w * inter_h
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a[:, None] + area_b[None, :] - inter
        return torch.where(union > 0, inter / union, torch.zeros_like(union))

    def _map_tracks_to_pose_indices(
        self, inst: InstanceData, pose_kpts: np.ndarray, iou_thresh: float = 0.3
    ) -> Optional[torch.Tensor]:
        # Prefer direct mapping if present
        pose_indices = getattr(inst, "source_pose_index", None)
        track_mask = get_track_mask(inst)
        if isinstance(track_mask, torch.Tensor) and isinstance(pose_indices, torch.Tensor):
            pose_indices = pose_indices[track_mask]
        if pose_indices is not None:
            return pose_indices
        # Fallback: map by IoU between track bboxes and pose bbox (derived from keypoints)
        try:
            tb = getattr(inst, "bboxes", None)
            if tb is None or pose_kpts.size == 0:
                return None
            if not isinstance(tb, torch.Tensor):
                tb = torch.as_tensor(tb)
            if tb.ndim == 1:
                tb = tb.reshape(-1, 4)
            if isinstance(track_mask, torch.Tensor):
                tb = tb[track_mask]
            x = torch.as_tensor(pose_kpts[..., 0])
            y = torch.as_tensor(pose_kpts[..., 1])
            x1 = torch.min(x, dim=1).values
            y1 = torch.min(y, dim=1).values
            x2 = torch.max(x, dim=1).values
            y2 = torch.max(y, dim=1).values
            pb = torch.stack([x1, y1, x2, y2], dim=1).to(dtype=torch.float32)
            tb = tb.to(dtype=torch.float32)
            if len(tb) and len(pb):
                iou = self._bbox_iou_xyxy(tb, pb)
                best_iou, best_idx = torch.max(iou, dim=1)
                mapped = torch.where(
                    best_iou >= iou_thresh,
                    best_idx.to(dtype=torch.int64),
                    torch.full_like(best_idx, fill_value=-1, dtype=torch.int64),
                )
                return mapped
        except Exception:
            return None
        return None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        self._ensure_mmaction_imported()
        from mmaction.apis import inference_skeleton

        data: Dict[str, Any] = context.get("data", {})
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

        # Acquire frames and pose results
        original_images = data.get("original_images")
        if original_images is None:
            return {}
        pose_results_all = data.get("pose_results") or context.get("data", {}).get("pose_results")
        if not pose_results_all:
            return {}

        # Ensure CPU copy for shape
        if isinstance(original_images, torch.Tensor) and original_images.is_cuda:
            # lightweight shape only, do not copy full tensor to host if avoidable
            pass
        H, W = self._image_shape_from_tensor(original_images)
        img_shape = (H, W)

        recognizer = context.get("action_recognizer")
        if recognizer is None:
            return {}
        label_map: Optional[List[str]] = context.get("action_label_map")

        # Build per-track sequences across the current clip
        video_len = len(track_data_sample)
        # Collect all track ids present across frames
        all_track_ids: List[int] = []
        per_frame_instances: List[Optional[InstanceData]] = []
        for i in range(video_len):
            img_ds = track_data_sample[i]
            inst = getattr(img_ds, "pred_track_instances", None)
            per_frame_instances.append(inst)
            if inst is None:
                continue
            tids = getattr(inst, "instances_id", None)
            if tids is None:
                continue
            track_mask = get_track_mask(inst)
            if isinstance(track_mask, torch.Tensor) and isinstance(tids, torch.Tensor):
                tids = tids[track_mask]
            ids_np = (
                tids.detach().cpu().numpy() if isinstance(tids, torch.Tensor) else np.asarray(tids)
            )
            for tid in ids_np.tolist():
                if tid not in all_track_ids:
                    all_track_ids.append(int(tid))

        # Prepare output container per frame
        all_action_results: List[List[Dict[str, Any]]] = [[] for _ in range(video_len)]
        if not all_track_ids:
            data["action_results"] = all_action_results
            return {"data": data}

        # For each track id, build a single-person pose sequence across frames
        for tid in all_track_ids:
            per_frame_pose: List[Dict[str, np.ndarray]] = []
            for i in range(video_len):
                pose_kpts, pose_scores = self._extract_pose_arrays(pose_results_all[i])
                inst = per_frame_instances[i]
                if inst is None:
                    # No tracks in this frame; pad with empty
                    per_frame_pose.append(
                        dict(
                            keypoints=np.empty((0, 17, 2), dtype=np.float32),
                            keypoint_scores=np.empty((0, 17), dtype=np.float32),
                        )
                    )
                    continue
                # Map this frame's track boxes to pose indices
                mapped_idx = self._map_tracks_to_pose_indices(inst, pose_kpts)
                sel = None
                if mapped_idx is not None:
                    # find the index for this tid in instances_id
                    inst_ids = getattr(inst, "instances_id", None)
                    if inst_ids is not None:
                        if not isinstance(inst_ids, torch.Tensor):
                            inst_ids = torch.as_tensor(inst_ids)
                        track_mask = get_track_mask(inst)
                        if isinstance(track_mask, torch.Tensor):
                            inst_ids = inst_ids[track_mask]
                        match = torch.nonzero(inst_ids == int(tid)).reshape(-1)
                        if len(match) == 1:
                            pi = int(mapped_idx[int(match[0])].item())
                            if 0 <= pi < pose_kpts.shape[0]:
                                sel = pi
                if sel is None or pose_kpts.size == 0:
                    per_frame_pose.append(
                        dict(
                            keypoints=np.empty((0, 17, 2), dtype=np.float32),
                            keypoint_scores=np.empty((0, 17), dtype=np.float32),
                        )
                    )
                else:
                    per_frame_pose.append(
                        dict(
                            keypoints=pose_kpts[sel : sel + 1],
                            keypoint_scores=pose_scores[sel : sel + 1],
                        )
                    )

            # Run inference for this track id if it has any non-empty keypoints across frames
            has_any = any(item["keypoints"].shape[0] > 0 for item in per_frame_pose)
            if not has_any:
                continue
            result = inference_skeleton(recognizer, per_frame_pose, img_shape)
            # result.pred_score is a 1D tensor of class scores
            scores = getattr(result, "pred_score", None)
            if scores is None:
                continue
            if not isinstance(scores, torch.Tensor):
                try:
                    scores = torch.as_tensor(scores)
                except Exception:
                    continue
            # Top-k indices
            k = max(1, self._top_k)
            topk = torch.topk(scores, k=min(k, len(scores)))
            top_idx = topk.indices.detach().cpu().tolist()
            top_scores = topk.values.detach().cpu().tolist()
            # Choose primary label
            primary_idx = int(top_idx[0]) if len(top_idx) else -1
            primary_score = float(top_scores[0]) if len(top_scores) else 0.0
            if primary_idx < 0:
                continue
            if label_map is not None and 0 <= primary_idx < len(label_map):
                primary_label = label_map[primary_idx]
            else:
                primary_label = str(primary_idx)
            if primary_score < self._score_threshold:
                continue
            # Attach same per-clip result to each frame where this tid is present
            for i in range(video_len):
                inst = per_frame_instances[i]
                if inst is None:
                    continue
                inst_ids = getattr(inst, "instances_id", None)
                if inst_ids is None:
                    continue
                track_mask = get_track_mask(inst)
                if isinstance(track_mask, torch.Tensor) and isinstance(inst_ids, torch.Tensor):
                    inst_ids = inst_ids[track_mask]
                ids_np = (
                    inst_ids.detach().cpu().numpy()
                    if isinstance(inst_ids, torch.Tensor)
                    else np.asarray(inst_ids)
                )
                if int(tid) in ids_np.tolist():
                    all_action_results[i].append(
                        dict(
                            tracking_id=int(tid),
                            label=primary_label,
                            label_index=primary_idx,
                            score=primary_score,
                            topk_indices=top_idx,
                            topk_scores=top_scores,
                        )
                    )

        data["action_results"] = all_action_results
        return {"data": data}

    def input_keys(self):
        return {"data", "action_recognizer", "action_label_map"}

    def output_keys(self):
        return {"data"}

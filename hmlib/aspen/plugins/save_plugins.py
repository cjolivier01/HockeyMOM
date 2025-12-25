from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.tracking_utils.action_dataframe import ActionDataFrame
from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.tracking_utils.utils import get_track_mask

from .base import Plugin


def _ctx_value(context: Dict[str, Any], key: str) -> Optional[Any]:
    if not key:
        return None
    if key in context:
        return context[key]
    shared = context.get("shared")
    if isinstance(shared, dict):
        return shared.get(key)
    return None


def _apply_track_mask(inst, tids, tlbr, scores, labels, pose_indices=None):
    mask = get_track_mask(inst)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
        if isinstance(tids, torch.Tensor):
            tids = tids[mask]
        else:
            tids = np.asarray(tids)[mask_np]
        if isinstance(tlbr, torch.Tensor):
            tlbr = tlbr[mask]
        else:
            tlbr = np.asarray(tlbr)[mask_np]
        if isinstance(scores, torch.Tensor):
            scores = scores[mask]
        else:
            scores = np.asarray(scores)[mask_np]
        if isinstance(labels, torch.Tensor):
            labels = labels[mask]
        else:
            labels = np.asarray(labels)[mask_np]
        if pose_indices is not None:
            if isinstance(pose_indices, torch.Tensor):
                pose_indices = pose_indices[mask]
            else:
                pose_indices = np.asarray(pose_indices)[mask_np]
    return tids, tlbr, scores, labels, pose_indices


class SavePluginBase(Plugin):
    """Base class for save plugins with common utilities."""

    def is_output(self) -> bool:
        """If enabled, this node is an output."""
        return self.enabled


class SaveDetectionsPlugin(SavePluginBase):
    """
    Saves per-frame detections into `detection_dataframe`.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample or [TrackDataSample])
      - frame_id: int for first frame in batch
      - detection_dataframe: DetectionDataFrame
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "detections.csv",
        write_interval: int = 100,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._detection_dataframe: Optional[DetectionDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[DetectionDataFrame]:
        if self._detection_dataframe is not None:
            return self._detection_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, self._output_filename)
        self._detection_dataframe = DetectionDataFrame(
            output_file=output_path, write_interval=self._write_interval
        )
        return self._detection_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
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
                try:
                    df.add_frame_sample(frame_id=int(frame_id0 + i), data_sample=img_data_sample)
                except Exception:
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

            try:
                df.add_frame_sample(frame_id=int(fid), data_sample=img_data_sample)
            except Exception:
                pose_indices = getattr(inst, "source_pose_index", None)
                df.add_frame_records(
                    frame_id=int(fid),
                    scores=getattr(inst, "scores", np.empty((0,), dtype=np.float32)),
                    labels=getattr(inst, "labels", np.empty((0,), dtype=np.int64)),
                    bboxes=getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
                    pose_indices=pose_indices,
                )

        return {"detection_dataframe": df}

    def input_keys(self):
        return {"data", "frame_id"}

    def output_keys(self):
        return {"detection_dataframe"}

    def finalize(self):
        if self._detection_dataframe is not None:
            self._detection_dataframe.close()


class SaveTrackingPlugin(SavePluginBase):
    """
    Saves per-frame tracking results into `tracking_dataframe`.

    Expects in context:
      - data: dict with 'data_samples'
      - frame_id: int for first frame in batch
      - tracking_dataframe: TrackingDataFrame
      - jersey_results: Optional per-frame jersey info list
      - action_results: Optional per-frame action result list (from ActionFromPosePlugin)
    """

    def __init__(
        self,
        enabled: bool = True,
        pose_iou_thresh: float = 0.3,
        work_dir_key: str = "work_dir",
        output_filename: str = "tracking.csv",
        write_interval: int = 100,
    ):
        super().__init__(enabled=enabled)
        # Default fallback IoU threshold if we must infer mapping
        self._pose_iou_thresh = float(pose_iou_thresh)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._tracking_dataframe: Optional[TrackingDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[TrackingDataFrame]:
        if self._tracking_dataframe is not None:
            return self._tracking_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, self._output_filename)
        self._tracking_dataframe = TrackingDataFrame(
            output_file=output_path,
            input_batch_size=1,
            write_interval=self._write_interval,
        )
        return self._tracking_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        if df is None:
            return {}

        data: Dict[str, Any] = context.get("data", {})
        jersey_results_all = data.get("jersey_results") or context.get("jersey_results")
        action_results_all = data.get("action_results") or context.get("action_results")
        frame_id0: int = int(context.get("frame_id", -1))
        pose_results_all = data.get("pose_results")  # mirrored by PoseToDetPlugin

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
            # Borrow the logic from PoseToDetPlugin for deriving bboxes
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
                try:
                    df.add_frame_sample(
                        frame_id=frame_id0 + i,
                        data_sample=img_data_sample,
                        jersey_info=None,
                        pose_indices=None,
                        action_info=None,
                    )
                except Exception:
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
                jersey_results_all[i]
                if isinstance(jersey_results_all, list) and i < len(jersey_results_all)
                else None
            )
            action_results = (
                action_results_all[i]
                if isinstance(action_results_all, list) and i < len(action_results_all)
                else None
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
            try:
                df.add_frame_sample(
                    frame_id=frame_id0 + i,
                    data_sample=img_data_sample,
                    jersey_info=jersey_results,
                    pose_indices=pose_indices,
                    action_info=action_results,
                )
            except Exception:
                tids = getattr(inst, "instances_id", np.empty((0,), dtype=np.int64))
                tlbr = getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32))
                scores = getattr(inst, "scores", np.empty((0,), dtype=np.float32))
                labels = getattr(inst, "labels", np.empty((0,), dtype=np.int64))
                tids, tlbr, scores, labels, pose_indices = _apply_track_mask(
                    inst, tids, tlbr, scores, labels, pose_indices
                )
                df.add_frame_records(
                    frame_id=frame_id0 + i,
                    tracking_ids=tids,
                    tlbr=tlbr,
                    scores=scores,
                    labels=labels,
                    jersey_info=jersey_results,
                    pose_indices=pose_indices,
                    action_info=action_results,
                )
        return {"tracking_dataframe": df}

    def input_keys(self):
        return {"data", "frame_id", "jersey_results", "action_results"}

    def output_keys(self):
        return {"tracking_dataframe"}

    def finalize(self):
        if self._tracking_dataframe is not None:
            self._tracking_dataframe.close()


class SavePosePlugin(SavePluginBase):
    """
    Saves per-frame pose results from `data['pose_results']` into `pose_dataframe`.

    We serialize a simplified structure capturing keypoints/bboxes/scores to JSON.

    Expects in context:
      - data: dict with 'pose_results' list
      - frame_id: int for first frame in batch
      - pose_dataframe: PoseDataFrame
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "pose.csv",
        write_interval: int = 100,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._pose_dataframe: Optional[PoseDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[PoseDataFrame]:
        if self._pose_dataframe is not None:
            return self._pose_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, self._output_filename)
        self._pose_dataframe = PoseDataFrame(
            output_file=output_path, write_interval=self._write_interval
        )
        return self._pose_dataframe

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
                    for k in (
                        "bboxes",
                        "scores",
                        "bbox_scores",
                        "labels",
                        "keypoints",
                        "keypoint_scores",
                    ):
                        if hasattr(inst, k):
                            item[k] = cls._to_list(getattr(inst, k))
                out_preds.append(item)
        return {"predictions": out_preds}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = self._ensure_dataframe(context)
        if df is None:
            return {}

        data: Dict[str, Any] = context.get("data", {})
        pose_results: Optional[List[Any]] = data.get("pose_results")

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
                df.add_frame_records(
                    frame_id=frame_id0 + i, pose_json=json.dumps({"predictions": []})
                )
        else:
            for i, item in enumerate(pose_results):
                # Prefer direct PoseDataSample storage
                try:
                    df.add_frame_sample(frame_id=frame_id0 + i, pose_item=item)
                except Exception:
                    simp = self._simplify_pose_item(item)
                    df.add_frame_records(frame_id=frame_id0 + i, pose_json=json.dumps(simp))
        return {"pose_dataframe": df}

    def input_keys(self):
        return {"data", "frame_id"}

    def output_keys(self):
        return {"pose_dataframe"}

    def finalize(self):
        if self._pose_dataframe is not None:
            self._pose_dataframe.close()


class SaveActionsPlugin(SavePluginBase):
    """
    Saves per-frame action results. By default, writes into the `tracking_dataframe`
    action columns, if a TrackingDataFrame is provided in context. This trunk is
    optional since SaveTrackingPlugin already persists action results when placed
    after the `actions` trunk; include this only if you need a dedicated action
    saving pass.

    Expects in context:
      - data: dict with 'action_results' per frame
      - frame_id: int for first frame in batch
      - tracking_dataframe: TrackingDataFrame (will update action columns)
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "actions.csv",
        write_interval: int = 100,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._action_dataframe: Optional[ActionDataFrame] = None

    def _ensure_action_dataframe(self, context: Dict[str, Any]) -> Optional[ActionDataFrame]:
        if self._action_dataframe is not None:
            return self._action_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, self._output_filename)
        self._action_dataframe = ActionDataFrame(
            output_file=output_path, write_interval=self._write_interval
        )
        return self._action_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = context.get("tracking_dataframe")
        action_df = context.get("action_dataframe") or self._ensure_action_dataframe(context)
        if df is None and action_df is None:
            return {}
        data: Dict[str, Any] = context.get("data", {})
        action_results_all = data.get("action_results") or context.get("action_results")
        if not action_results_all:
            return {}
        frame_id0: int = int(context.get("frame_id", -1))

        track_samples = data.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        video_len = len(track_data_sample)
        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            inst = getattr(img_data_sample, "pred_track_instances", None)
            if inst is None:
                continue
            actions = action_results_all[i] if i < len(action_results_all) else None
            # Update tracking with action columns if tracking df present
            if df is not None:
                tids = getattr(inst, "instances_id", np.empty((0,), dtype=np.int64))
                tlbr = getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32))
                scores = getattr(inst, "scores", np.empty((0,), dtype=np.float32))
                labels = getattr(inst, "labels", np.empty((0,), dtype=np.int64))
                pose_indices = getattr(inst, "source_pose_index", None)
                tids, tlbr, scores, labels, pose_indices = _apply_track_mask(
                    inst, tids, tlbr, scores, labels, pose_indices
                )
                df.add_frame_records(
                    frame_id=frame_id0 + i,
                    tracking_ids=tids,
                    tlbr=tlbr,
                    scores=scores,
                    labels=labels,
                    jersey_info=None,
                    pose_indices=pose_indices,
                    action_info=actions,
                )
            # Optionally write dedicated action dataframe
            if action_df is not None and actions is not None:
                try:
                    # Build list of ActionDataSample-like dicts (tracking_id, label_index/label, score)
                    action_df.add_frame_sample(frame_id=frame_id0 + i, data_samples=actions)
                except Exception:
                    action_df.add_frame_records(
                        frame_id=frame_id0 + i, action_json=json.dumps(actions)
                    )
        return {"action_dataframe": action_df} if action_df is not None else {}

    def input_keys(self):
        return {"data", "frame_id", "tracking_dataframe", "action_results"}

    def output_keys(self):
        return {"action_dataframe"}

    def finalize(self):
        if self._action_dataframe is not None:
            self._action_dataframe.close()


class SaveCameraPlugin(SavePluginBase):
    """
    Saves per-frame camera boxes into `camera_dataframe`.

    Expects in context:
      - frame_id: int first frame in batch
      - current_box: TLBR camera box tensor or array
      - work_dir: output directory for camera.csv
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "camera.csv",
        write_interval: int = 100,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._camera_dataframe: Optional[CameraTrackingDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[CameraTrackingDataFrame]:
        if self._camera_dataframe is not None:
            return self._camera_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, self._output_filename)
        self._camera_dataframe = CameraTrackingDataFrame(
            output_file=output_path,
            input_batch_size=1,
        )
        self._camera_dataframe.write_interval = self._write_interval
        return self._camera_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        if df is None:
            return {}

        frame_id0 = int(context.get("frame_id", -1))
        current_box = context.get("current_box")
        if current_box is None:
            return {"camera_dataframe": df}

        try:
            if hasattr(current_box, "detach"):
                tlbr = current_box.detach().cpu().numpy()
            else:
                tlbr = np.asarray(current_box)
            if tlbr.ndim == 1:
                tlbr = tlbr.reshape(1, -1)
            df.add_frame_records(frame_id=frame_id0, tlbr=tlbr)
        except Exception:
            pass

        return {"camera_dataframe": df}

    def input_keys(self):
        return {"frame_id", "current_box", "shared"}

    def output_keys(self):
        return {"camera_dataframe"}

    def finalize(self):
        if self._camera_dataframe is not None:
            self._camera_dataframe.close()

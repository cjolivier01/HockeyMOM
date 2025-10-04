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

            df.add_frame_records(
                frame_id=int(fid),
                scores=getattr(inst, "scores", np.empty((0,), dtype=np.float32)),
                labels=getattr(inst, "labels", np.empty((0,), dtype=np.int64)),
                bboxes=getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
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

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = context.get("tracking_dataframe")
        if df is None:
            return {}

        data: Dict[str, Any] = context.get("data", {})
        jersey_results_all = data.get("jersey_results") or context.get("jersey_results")
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
                # No tracks: still record an empty frame
                df.add_frame_records(
                    frame_id=frame_id0 + i,
                    tracking_ids=np.empty((0,), dtype=np.int64),
                    tlbr=np.empty((0, 4), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    jersey_info=None,
                )
                continue
            jersey_results = (
                jersey_results_all[i] if isinstance(jersey_results_all, list) and i < len(jersey_results_all) else None
            )
            df.add_frame_records(
                frame_id=frame_id0 + i,
                tracking_ids=getattr(inst, "instances_id", np.empty((0,), dtype=np.int64)),
                tlbr=getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
                scores=getattr(inst, "scores", np.empty((0,), dtype=np.float32)),
                labels=getattr(inst, "labels", np.empty((0,), dtype=np.int64)),
                jersey_info=jersey_results,
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

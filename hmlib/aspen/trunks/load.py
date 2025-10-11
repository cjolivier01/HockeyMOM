from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.structures import InstanceData

from hmlib.bbox.box_functions import tlwh_to_tlbr_multiple

from .base import Trunk


class LoadDetectionsTrunk(Trunk):
    """
    Loads detections from `detection_dataframe` and attaches them to data_samples
    as `pred_instances`, emulating DetectorInferenceTrunk.

    Expects in context:
      - data: dict with 'data_samples'
      - frame_id: first frame id in batch
      - detection_dataframe: DetectionDataFrame (input_file provided)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = context.get("detection_dataframe")
        if df is None or not df.has_input_data():
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
            rec = df.get_data_dict_by_frame(frame_id=frame_id0 + i)
            if rec is None:
                # Attach empty detections to mirror detector behavior
                inst = InstanceData()
                inst.scores = torch.empty((0,), dtype=torch.float32)
                inst.labels = torch.empty((0,), dtype=torch.long)
                inst.bboxes = torch.empty((0, 4), dtype=torch.float32)
                img_data_sample.pred_instances = inst
            else:
                # Convert to torch tensors for consistency
                scores = rec.get("scores", np.empty((0,), dtype=np.float32))
                labels = rec.get("labels", np.empty((0,), dtype=np.int64))
                bboxes = rec.get("bboxes", np.empty((0, 4), dtype=np.float32))
                inst = InstanceData()
                inst.scores = torch.as_tensor(scores)
                inst.labels = torch.as_tensor(labels)
                inst.bboxes = torch.as_tensor(bboxes)
                img_data_sample.pred_instances = inst

        return {"data": data}

    def input_keys(self):
        return {"data", "frame_id", "detection_dataframe"}

    def output_keys(self):
        return {"data"}


class LoadTrackingTrunk(Trunk):
    """
    Loads tracks from `tracking_dataframe` and attaches `pred_track_instances`.

    Produces `data`, `nr_tracks`, and `max_tracking_id` analogous to TrackerTrunk.

    Expects in context:
      - data: dict with 'data_samples'
      - frame_id: first frame in batch
      - tracking_dataframe: TrackingDataFrame (input_file provided)
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = context.get("tracking_dataframe")
        if df is None or not df.has_input_data():
            return {}

        data: Dict[str, Any] = context.get("data", {})
        preserved_original_images = data.get("original_images")
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
        max_tracking_id = 0
        active_track_count = 0

        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            rec = df.get_data_dict_by_frame(frame_id=frame_id0 + i)
            if rec is None:
                # attach empty tracks instance
                inst = InstanceData(
                    instances_id=torch.empty((0,), dtype=torch.long),
                    bboxes=torch.empty((0, 4), dtype=torch.float32),
                    scores=torch.empty((0,), dtype=torch.float32),
                    labels=torch.empty((0,), dtype=torch.long),
                )
                img_data_sample.pred_track_instances = inst
                try:
                    img_data_sample.set_metainfo({"nr_tracks": 0})
                except Exception:
                    pass
                continue
            # Convert tlwh->tlbr for downstream consumers
            tlwh = rec.get("bboxes", np.empty((0, 4), dtype=np.float32))
            if isinstance(tlwh, np.ndarray):
                tlwh_t = torch.as_tensor(tlwh)
                tlbr_t = tlwh_to_tlbr_multiple(tlwh_t)
                bboxes = tlbr_t
            else:
                bboxes = tlwh_to_tlbr_multiple(tlwh)

            ids = rec.get("tracking_ids", np.empty((0,), dtype=np.int64))
            scores = rec.get("scores", np.empty((0,), dtype=np.float32))
            labels = rec.get("labels", np.empty((0,), dtype=np.int64))

            inst = InstanceData(
                instances_id=torch.as_tensor(ids),
                bboxes=bboxes,
                scores=torch.as_tensor(scores),
                labels=torch.as_tensor(labels),
            )
            img_data_sample.pred_track_instances = inst
            try:
                img_data_sample.set_metainfo({"nr_tracks": len(inst.instances_id)})
            except Exception:
                pass

            active_track_count = max(active_track_count, len(inst.instances_id))
            if len(inst.instances_id):
                max_id = int(torch.max(inst.instances_id))
                max_tracking_id = max(max_tracking_id, max_id)

        return {
            "data": data,
            "nr_tracks": active_track_count,
            "max_tracking_id": max_tracking_id,
        }

    def input_keys(self):
        return {"data", "frame_id", "tracking_dataframe"}

    def output_keys(self):
        return {"data", "nr_tracks", "max_tracking_id"}


class LoadPoseTrunk(Trunk):
    """
    Loads per-frame pose JSON from `pose_dataframe` and sets data['pose_results'].

    The stored format is a simplified structure, but PoseToDetTrunk tolerates dict predictions
    if extended accordingly. Downstream postprocess uses pose results only if required.

    Expects in context:
      - pose_dataframe: PoseDataFrame (input_file provided)
      - data: dict with 'data_samples' to infer video_len
      - frame_id: int first frame
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = context.get("pose_dataframe")
        if df is None or not df.has_input_data():
            return {}

        data: Dict[str, Any] = context.get("data", {})
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

        pose_results: List[Any] = []
        for i in range(video_len):
            rec = df.get_data_dict_by_frame(frame_id=frame_id0 + i)
            if rec is None:
                pose_results.append({"predictions": []})
            else:
                pose_results.append(rec.get("pose", {"predictions": []}))

        data["pose_results"] = pose_results
        return {"data": data}

    def input_keys(self):
        return {"data", "frame_id", "pose_dataframe"}

    def output_keys(self):
        return {"data"}

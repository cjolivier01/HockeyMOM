from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from mmengine.structures import InstanceData

from hmlib.datasets.dataframe import HmDataFrameBase

try:
    from mmpose.structures import PoseDataSample
except Exception:  # pragma: no cover
    PoseDataSample = None  # type: ignore

if TYPE_CHECKING:
    from mmpose.structures import PoseDataSample as _PoseDataSample
else:
    _PoseDataSample = Any  # type: ignore


class PoseDataFrame(HmDataFrameBase):
    """
    Stores per-frame pose results as a JSON string.

    Fields:
      - Frame: int
      - PoseJSON: str (JSON dump of a simplified pose structure)
    """

    def __init__(self, *args, **kwargs):
        fields = [
            "Frame",
            "PoseJSON",
        ]
        super().__init__(*args, fields=fields, **kwargs)

    def add_frame_records(self, frame_id: int, pose_json: str):
        frame_id = int(frame_id)
        rec = pd.DataFrame(
            {
                "Frame": [frame_id],
                "PoseJSON": [pose_json],
            }
        )
        self._dataframe_list.append(rec)
        self.counter += 1
        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0

    def add_frame_sample(self, frame_id: int, pose_item: Any):
        """Persist a per-frame PoseDataSample or simplified dict.

        Stores a compact JSON with only keypoints and scores.
        """

        def _to_list(x):
            try:
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().tolist()
                if isinstance(x, np.ndarray):
                    return x.tolist()
            except Exception:
                pass
            return x

        simp: Dict[str, Any] = {"predictions": []}
        inst = getattr(pose_item, "pred_instances", None)
        if inst is not None:
            simp["predictions"].append(
                {
                    k: _to_list(getattr(inst, k))
                    for k in ("bboxes", "scores", "bbox_scores", "labels", "keypoints", "keypoint_scores")
                    if hasattr(inst, k)
                }
            )
        elif isinstance(pose_item, dict):
            try:
                preds = pose_item.get("predictions")
            except Exception:
                preds = None
            if isinstance(preds, list):
                out: List[Dict[str, Any]] = []
                for ds in preds:
                    if isinstance(ds, dict):
                        out.append({k: _to_list(ds[k]) for k in ds.keys()})
                simp["predictions"] = out
        self.add_frame_records(frame_id=frame_id, pose_json=json.dumps(simp))

    def __getitem__(self, idx: int) -> Optional[_PoseDataSample]:
        # Frame IDs start at 1; return a PoseDataSample if possible
        return self.get_sample_by_frame(frame_id=idx + 1)

    def get_data_by_frame(self, frame_number: int):
        if not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        return None

    def get_data_dict_by_frame(self, frame_id: int) -> Optional[Dict[str, Any]]:
        frame_id = int(frame_id)
        if self.data is None or self.data.empty:
            return None
        rows = self.data[self.data["Frame"] == frame_id]
        if rows.empty:
            return None
        # Expect one row per frame
        pose_json = rows.iloc[0]["PoseJSON"]
        try:
            pose_obj = json.loads(pose_json)
        except Exception:
            pose_obj = {"predictions": []}
        return {"pose": pose_obj}

    def get_sample_by_frame(self, frame_id: int) -> Optional[_PoseDataSample]:
        """Reconstruct a PoseDataSample from stored JSON.

        Returns None if no pose was recorded for the frame.
        """
        rec = self.get_data_dict_by_frame(frame_id)
        if rec is None:
            return None
        obj = rec.get("pose", {"predictions": []})
        preds = obj.get("predictions") if isinstance(obj, dict) else None
        if not preds:
            return None
        # We store a single prediction item per frame for now
        ds0 = preds[0]
        inst = InstanceData()
        if isinstance(ds0, dict):
            if "keypoints" in ds0:
                inst.keypoints = torch.as_tensor(ds0["keypoints"])  # (N,K,2)
            if "keypoint_scores" in ds0:
                inst.keypoint_scores = torch.as_tensor(ds0["keypoint_scores"])  # (N,K)
            if "bboxes" in ds0:
                inst.bboxes = torch.as_tensor(ds0["bboxes"])  # (N,4)
            if "scores" in ds0:
                inst.scores = torch.as_tensor(ds0["scores"])  # (N,)
            if "bbox_scores" in ds0:
                inst.bbox_scores = torch.as_tensor(ds0["bbox_scores"])  # (N,)
        if PoseDataSample is not None:
            pose_ds = PoseDataSample()
            pose_ds.pred_instances = inst
            try:
                pose_ds.set_metainfo({"frame_id": int(frame_id)})
            except Exception:
                pass
            return pose_ds
        return inst

    def get_samples(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> List[_PoseDataSample]:
        """Return a list of PoseDataSample for a frame range (inclusive)."""
        if self.data is None or self.data.empty:
            return []
        frames = sorted(set(int(f) for f in self.data["Frame"].tolist()))
        if not frames:
            return []
        if start_frame is not None or end_frame is not None:
            lo = int(start_frame) if start_frame is not None else frames[0]
            hi = int(end_frame) if end_frame is not None else frames[-1]
            frames = [f for f in frames if lo <= f <= hi]
        out: List[_PoseDataSample] = []
        for f in frames:
            ds = self.get_sample_by_frame(f)
            if ds is not None:
                out.append(ds)
        return out

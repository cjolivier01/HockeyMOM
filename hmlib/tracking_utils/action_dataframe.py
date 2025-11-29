"""DataFrame wrapper for per-frame action recognition outputs.

Stores JSON-encoded action results (per tracked player) and exposes helpers
to append frames and convert to/from mmaction ``ActionDataSample`` objects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import torch

from hmlib.datasets.dataframe import HmDataFrameBase

try:
    from mmaction.structures import ActionDataSample
except Exception:  # pragma: no cover
    ActionDataSample = None  # type: ignore

if TYPE_CHECKING:
    from mmaction.structures import ActionDataSample as _ActionDataSample
else:
    _ActionDataSample = Any  # type: ignore


class ActionDataFrame(HmDataFrameBase):
    """
    Stores per-frame action recognition results as JSON.

    Each frame stores a JSON array where each item corresponds to one tracked
    player with fields: tracking_id, label, label_index, score, topk_indices,
    topk_scores.
    """

    def __init__(self, *args, **kwargs) -> None:
        fields = [
            "Frame",
            "ActionJSON",
        ]
        super().__init__(*args, fields=fields, **kwargs)

    def add_frame_records(self, frame_id: int, action_json: str) -> None:
        frame_id = int(frame_id)
        rec = pd.DataFrame(
            {
                "Frame": [frame_id],
                "ActionJSON": [action_json],
            }
        )
        self._dataframe_list.append(rec)
        self.counter += 1
        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0

    def add_frame_sample(self, frame_id: int, data_samples: Any) -> None:
        """Persist a list of ActionDataSample or a list of dicts for a frame."""
        items: List[Dict[str, Any]] = []
        if isinstance(data_samples, list):
            for ds in data_samples:
                if ActionDataSample is not None and isinstance(ds, ActionDataSample):
                    # Serialize minimal fields
                    label_val = getattr(ds, "pred_label", None)
                    if isinstance(label_val, torch.Tensor):
                        label_val = int(label_val.view(-1)[0].item())
                    score_val = getattr(ds, "pred_score", None)
                    if isinstance(score_val, torch.Tensor):
                        # keep as python list for JSON serializability
                        score_val = score_val.detach().cpu().tolist()
                    it: Dict[str, Any] = dict(label_index=label_val)
                    if score_val is not None:
                        it["scores"] = score_val
                    # carry tracking id if present in metainfo
                    try:
                        tid = ds.metainfo.get("tracking_id", None)
                        if tid is not None:
                            it["tracking_id"] = int(tid)
                    except Exception:
                        pass
                    items.append(it)
                elif isinstance(ds, dict):
                    items.append(ds)
        self.add_frame_records(frame_id=int(frame_id), action_json=json.dumps(items))

    def __getitem__(self, idx: int) -> Optional[List[_ActionDataSample]]:
        # Frame IDs start at 1
        return self.get_sample_by_frame(frame_id=idx + 1)

    def get_data_dict_by_frame(self, frame_id: int) -> Optional[Dict[str, Any]]:
        frame_id = int(frame_id)
        if self.data is None or self.data.empty:
            return None
        rows = self.data[self.data["Frame"] == frame_id]
        if rows.empty:
            return None
        raw = rows.iloc[0]["ActionJSON"]
        try:
            arr = json.loads(raw)
        except Exception:
            arr = []
        return {"actions": arr}

    def get_sample_by_frame(self, frame_id: int) -> Optional[List[_ActionDataSample]]:
        rec = self.get_data_dict_by_frame(frame_id)
        if rec is None:
            return None
        items = rec.get("actions", [])
        if not items:
            return []
        out: List[Any] = []
        if ActionDataSample is None:
            return items
        for it in items:
            if not isinstance(it, dict):
                continue
            ds = ActionDataSample()
            idx = it.get("label_index", None)
            if idx is not None:
                try:
                    ds.set_pred_label(int(idx))
                except Exception:
                    pass
            scores = it.get("scores", None)
            if scores is not None:
                try:
                    ds.set_pred_score(torch.as_tensor(scores))
                except Exception:
                    pass
            # propagate tracking id in metainfo if available
            tid = it.get("tracking_id", None)
            try:
                if tid is not None:
                    ds.set_metainfo({"tracking_id": int(tid)})
            except Exception:
                pass
            out.append(ds)
        return out

    def get_samples(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> List[List[_ActionDataSample]]:
        """Return per-frame action samples for a frame range (inclusive)."""
        if self.data is None or self.data.empty:
            return []
        frames = sorted(set(int(f) for f in self.data["Frame"].tolist()))
        if not frames:
            return []
        if start_frame is not None or end_frame is not None:
            lo = int(start_frame) if start_frame is not None else frames[0]
            hi = int(end_frame) if end_frame is not None else frames[-1]
            frames = [f for f in frames if lo <= f <= hi]
        out: List[List[_ActionDataSample]] = []
        for f in frames:
            ds = self.get_sample_by_frame(f)
            out.append(ds or [])
        return out

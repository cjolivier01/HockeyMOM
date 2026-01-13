import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from mmengine.structures import InstanceData

from hmlib.datasets.dataframe import HmDataFrameBase

try:  # Prefer vendored OpenMMLab structures
    from mmdet.structures import DetDataSample
except Exception:  # pragma: no cover
    DetDataSample = None  # type: ignore

if TYPE_CHECKING:
    from mmdet.structures import DetDataSample as _DetDataSample  # precise type for mypy
else:
    _DetDataSample = Any  # type: ignore


class DetectionDataFrame(HmDataFrameBase):
    def __init__(self, *args, **kwargs):
        fields = [
            "Frame",
            "BBox_X1",
            "BBox_Y1",
            "BBox_X2",
            "BBox_Y2",
            "Scores",
            "Labels",
        ]
        super().__init__(*args, fields=fields, **kwargs)

    def read_data(self) -> None:
        """Read detection CSVs supporting both legacy and current schemas."""
        if not self.input_file:
            return
        from hmlib.log import logger

        if not os.path.exists(self.input_file):
            logger.error("Could not open dataframe file: %s", self.input_file)
            self.data = None
            return

        df = pd.read_csv(self.input_file, header=None)
        n = int(df.shape[1])
        cols = [
            "Frame",
            "BBox_X1",
            "BBox_Y1",
            "BBox_X2",
            "BBox_Y2",
            "Scores",
            "Labels",
        ]
        if n == len(cols) + 1:
            # Legacy: + PoseIndex
            df.columns = cols + ["PoseIndex"]
            df = df.drop(columns=["PoseIndex"])
        elif n >= len(cols):
            # Current (or future): ignore extras
            extra = [f"Extra{i}" for i in range(max(0, n - len(cols)))]
            df.columns = cols + extra
            df = df[cols]
        else:
            # Pad missing columns
            for _ in range(len(cols) - n):
                df[df.shape[1]] = np.nan
            df.columns = cols

        self.data = df.reindex(columns=self.fields)

    @staticmethod
    def _make_array(t: Union[np.ndarray, torch.Tensor, List[Any], Tuple[Any, ...]]) -> np.ndarray:
        """Convert tensors (including sequences of tensors) to CPU numpy arrays."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        if isinstance(t, (list, tuple)):
            if not t:
                return np.empty((0,), dtype=np.float32)
            converted = []
            for item in t:
                if isinstance(item, torch.Tensor):
                    converted.append(item.detach().cpu().numpy())
                else:
                    converted.append(item)
            return np.asarray(converted)
        return t

    def add_frame_records(
        self,
        frame_id: int,
        scores: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
    ):
        frame_id = int(frame_id)
        bboxes = self._make_array(bboxes)
        new_record = pd.DataFrame(
            {
                "Frame": [frame_id for _ in range(len(bboxes))],
                "BBox_X1": bboxes[:, 0],
                "BBox_Y1": bboxes[:, 1],
                "BBox_X2": bboxes[:, 2],
                "BBox_Y2": bboxes[:, 3],
                "Scores": self._make_array(scores),
                "Labels": self._make_array(labels),
            }
        )
        self._dataframe_list.append(new_record)
        self.counter += 1

        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0  # Reset the counter after writing

    def __getitem__(self, idx: int) -> Optional[_DetDataSample]:
        # Frame id's start at 1; return a DetDataSample (preferred)
        return self.get_sample_by_frame(frame_id=idx + 1)

    def add_frame_sample(self, frame_id: int, data_sample: Any):
        """Persist a per-frame DetDataSample.

        Accepts any object exposing a ``pred_instances`` with
        ``scores``, ``labels``, and ``bboxes``.
        """
        inst = getattr(data_sample, "pred_instances", None)
        if inst is None:
            # Record an empty frame
            self.add_frame_records(
                frame_id=int(frame_id),
                scores=np.empty((0,), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                bboxes=np.empty((0, 4), dtype=np.float32),
            )
            return
        self.add_frame_records(
            frame_id=int(frame_id),
            scores=self.get_ndarray(inst, "scores", np.empty((0,), dtype=np.float32)),
            labels=self.get_ndarray(inst, "labels", np.empty((0,), dtype=np.int64)),
            bboxes=self.get_ndarray(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
        )

    def get_data_by_frame(self, frame_number: int):
        """Get all tracking data for a specific frame."""
        if self.data is not None and not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        else:
            from hmlib.log import get_logger

            get_logger(__name__).warning("No data loaded.")
            return None

    def _to_outgoing_array(self, array: np.ndarray) -> np.ndarray:
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return array

    # Function to extract detection info by frame
    def get_data_dict_by_frame(self, frame_id: int) -> Optional[Dict[str, Any]]:
        frame_id = int(frame_id)
        if self.data is None or self.data.empty:
            return None
        # Filter the DataFrame for the specified frame
        frame_data = self.data[self.data["Frame"] == frame_id]
        if frame_data.empty:
            return None
        # Extract columns as NumPy arrays
        scores = frame_data["Scores"].to_numpy()
        labels = frame_data["Labels"].to_numpy()
        bboxes = frame_data[["BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]].to_numpy()
        return dict(
            scores=self._to_outgoing_array(scores),
            labels=self._to_outgoing_array(labels),
            bboxes=self._to_outgoing_array(bboxes),
        )

    def get_sample_by_frame(self, frame_id: int) -> Optional[_DetDataSample]:
        """Reconstruct a DetDataSample for a frame (preferred).

        Falls back to returning an InstanceData if DetDataSample is unavailable.
        """
        rec = self.get_data_dict_by_frame(frame_id)
        if rec is None:
            return None
        inst = InstanceData()
        inst.scores = torch.as_tensor(rec.get("scores", np.empty((0,), dtype=np.float32)))
        inst.labels = torch.as_tensor(rec.get("labels", np.empty((0,), dtype=np.int64)))
        inst.bboxes = torch.as_tensor(rec.get("bboxes", np.empty((0, 4), dtype=np.float32)))
        if DetDataSample is not None:
            ds = DetDataSample()
            ds.pred_instances = inst
            try:
                ds.set_metainfo({"frame_id": int(frame_id)})
            except Exception:
                pass
            return ds
        return inst

    def get_samples(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> List[_DetDataSample]:
        """Return a list of DetDataSample objects for a frame range (inclusive)."""
        if self.data is None or self.data.empty:
            return []
        frames = sorted(set(int(f) for f in self.data["Frame"].tolist()))
        if start_frame is not None or end_frame is not None:
            lo = int(start_frame) if start_frame is not None else frames[0]
            hi = int(end_frame) if end_frame is not None else frames[-1]
            frames = [f for f in frames if lo <= f <= hi]
        out: List[_DetDataSample] = []
        for f in frames:
            ds = self.get_sample_by_frame(f)
            if ds is not None:
                out.append(ds)
        return out

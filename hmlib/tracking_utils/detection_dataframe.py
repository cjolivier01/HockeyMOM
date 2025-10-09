from typing import Union, Dict, Any

import numpy as np
import pandas as pd
import torch

from hmlib.datasets.dataframe import HmDataFrameBase


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
            "PoseIndex",
        ]
        super().__init__(*args, fields=fields, **kwargs)

    @staticmethod
    def _make_array(t: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.to("cpu").numpy()
        return t

    def add_frame_records(
        self,
        frame_id: int,
        scores: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        pose_indices: np.ndarray | None = None,
    ):
        frame_id = int(frame_id)
        bboxes = self._make_array(bboxes)
        if pose_indices is None:
            pose_indices = -np.ones((len(bboxes),), dtype=np.int64)
        else:
            if isinstance(pose_indices, torch.Tensor):
                pose_indices = pose_indices.to("cpu").numpy()
            pose_indices = pose_indices.astype(np.int64, copy=False)
        new_record = pd.DataFrame(
            {
                "Frame": [frame_id for _ in range(len(bboxes))],
                "BBox_X1": bboxes[:, 0],
                "BBox_Y1": bboxes[:, 1],
                "BBox_X2": bboxes[:, 2],
                "BBox_Y2": bboxes[:, 3],
                "Scores": self._make_array(scores),
                "Labels": self._make_array(labels),
                "PoseIndex": pose_indices,
            }
        )
        self._dataframe_list.append(new_record)
        self.counter += 1

        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0  # Reset the counter after writing

    def __getitem__(self, idx: int) -> Union[Dict[str, Any], None]:
        # Frame id's start at 1
        return self.get_data_dict_by_frame(frame_id=idx + 1)

    def get_data_by_frame(self, frame_number: int):
        """Get all tracking data for a specific frame."""
        if not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        else:
            print("No data loaded.")
            return None

    def _to_outgoing_array(self, array: np.ndarray) -> np.ndarray:
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return array

    # Function to extract tracking info by frame
    def get_data_dict_by_frame(self, frame_id: int):
        frame_id = int(frame_id)
        # Filter the DataFrame for the specified frame
        frame_data = self.data[self.data["Frame"] == frame_id]
        # Extract columns as NumPy arrays
        scores = frame_data["Scores"].to_numpy()
        labels = frame_data["Labels"].to_numpy()
        bboxes = frame_data[["BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"]].to_numpy()
        pose_indices = (
            frame_data["PoseIndex"].to_numpy() if "PoseIndex" in frame_data.columns else None
        )
        return dict(
            scores=self._to_outgoing_array(scores),
            labels=self._to_outgoing_array(labels),
            bboxes=self._to_outgoing_array(bboxes),
            pose_indices=pose_indices,
        )

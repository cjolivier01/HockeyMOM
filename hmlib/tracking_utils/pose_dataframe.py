from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from hmlib.datasets.dataframe import HmDataFrameBase


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
        rec = pd.DataFrame({
            "Frame": [frame_id],
            "PoseJSON": [pose_json],
        })
        self._dataframe_list.append(rec)
        self.counter += 1
        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        # Frame IDs start at 1
        return self.get_data_dict_by_frame(frame_id=idx + 1)

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

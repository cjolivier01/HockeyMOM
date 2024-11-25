from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hmlib.bbox.box_functions import convert_tlbr_to_tlwh
from hmlib.datasets.dataframe import (
    HmDataFrameBase,
    dataclass_to_json,
    json_to_dataclass,
)
from hmlib.jersey.number_classifier import TrackJerseyInfo

# @dataclass
# class AllTrackJerseyInfo:
#     items: List[TrackJerseyInfo] = None


class TrackingDataFrame(HmDataFrameBase):

    def __init__(self, *args, input_batch_size: int, **kwargs):
        fields = [
            "Frame",
            "ID",
            "BBox_X",
            "BBox_Y",
            "BBox_W",
            "BBox_H",
            "Scores",
            "Labels",
            "Visibility",
            "JerseyInfo",
        ]
        super().__init__(*args, fields=fields, input_batch_size=input_batch_size, **kwargs)

    def add_frame_records(
        self,
        frame_id: int,
        tracking_ids: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        jersey_info: List[TrackJerseyInfo] = None,
        tlbr: Optional[np.ndarray] = None,
        tlwh: Optional[np.ndarray] = None,
    ):
        if tlwh is None:
            assert tlbr is not None
            tlwh = convert_tlbr_to_tlwh(tlbr)

        frame_id = int(frame_id)
        assert frame_id  # frame id's start at 1
        tracking_ids = self._make_array(tracking_ids)
        tlwh = self._make_array(tlwh)
        scores = self._make_array(scores)
        labels = self._make_array(labels)
        jersey_dict: Dict[int, TrackJerseyInfo] = {}
        if jersey_info is not None:
            for j_info in jersey_info:
                j_t_id = j_info.tracking_id
                # assert j_t_id not in jersey_dict
                if j_t_id in jersey_dict:
                    print(f"Ignoriung duplicate jersey tracking id {jersey_dict}")
                jersey_dict[j_t_id] = dataclass_to_json(j_info)

        def _jersey_item(id: int) -> str:
            v = jersey_dict.get(id)
            if v is None:
                return "{}"
            return v

        new_record = pd.DataFrame(
            {
                "Frame": [frame_id for _ in range(len(tracking_ids))],
                "ID": tracking_ids,
                "BBox_X": tlwh[:, 0],
                "BBox_Y": tlwh[:, 1],
                "BBox_W": tlwh[:, 2],
                "BBox_H": tlwh[:, 3],
                "Scores": scores,
                "Labels": labels,
                "Visibility": [-1 for _ in range(len(tracking_ids))],
                "JerseyInfo": [_jersey_item(t_id) for t_id in tracking_ids],
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

    def get_data_by_frame(self, frame_number: int) -> Union[Dict[str, Any], None]:
        """Get all tracking data for a specific frame."""
        if not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        else:
            print("No data loaded.")
            return None

    def get_data_dict_by_frame(self, frame_id: int) -> Dict[str, Any]:
        assert self.batch_size == 1
        frame_id = int(frame_id)
        # Filter the DataFrame for the specified frame
        frame_data = self.data[self.data["Frame"] == frame_id]
        # Extract columns as NumPy arrays
        tracking_ids = frame_data["ID"].to_numpy()
        scores = frame_data["Scores"].to_numpy()
        labels = frame_data["Labels"].to_numpy()
        tlwh = frame_data[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy()
        jersey_info = frame_data["JerseyInfo"]

        all_track_jersey_info: List[Optional[TrackJerseyInfo]] = []
        for tid, jersey in zip(tracking_ids, jersey_info):
            obj = json_to_dataclass(jersey, TrackJerseyInfo)
            if obj.tracking_id < 0:
                obj = None
            all_track_jersey_info.append(obj)

        return dict(
            frame_id=frame_id,
            tracking_ids=tracking_ids,
            scores=scores,
            bboxes=tlwh,
            labels=labels,
            jersey_info=all_track_jersey_info,
        )

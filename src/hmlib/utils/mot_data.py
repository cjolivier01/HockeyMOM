from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

from hmlib.tracking_utils.log import logger


class TrackingDataBase:
    def __init__(self, input_file=None, output_file=None, write_interval: int = 250):
        self.input_file = input_file
        self.output_file = output_file
        self.write_interval = write_interval
        self.first_write = True
        self._dataframe_list: List[pd.DataFrame] = []
        self.counter = 0  # Counter to track number of records since the last write
        if input_file:
            self.read_data()

    def has_input_data(self):
        return self.input_file is not None

    def read_data(self):
        assert False and "Not implemented"

    def write_data(self, output_path=None, header=False):
        if not output_path:
            output_path = self.output_file
        else:
            self.output_file = output_path

        """Write MOT tracking data to a CSV file incrementally."""
        if self.output_file:
            if self._dataframe_list:
                data = pd.concat(self._dataframe_list, ignore_index=True)
                mode = "a" if not self.first_write else "w"
                data.to_csv(output_path, mode=mode, header=header, index=False)
                self._dataframe_list = []
                logger.info(f"Data saved successfully to {output_path}.")
            else:
                logger.info("No data available to save.")

    def flush(self):
        self.write_data()


class MOTTrackingData(TrackingDataBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_data(self):
        """Read MOT tracking data from a CSV file."""
        if self.input_file:
            self.data = pd.read_csv(
                self.input_file,
                header=None,
                names=[
                    "Frame",
                    "ID",
                    "BBox_X",
                    "BBox_Y",
                    "BBox_W",
                    "BBox_H",
                    "Confidence",
                    "Class",
                    "Visibility",
                ],
            )
            print("Data loaded successfully.")

    def add_frame_records(
        self,
        frame_id: int,
        tracking_ids: np.ndarray,
        scores: np.ndarray,
        tlbr: Optional[np.ndarray] = None,
        tlwh: Optional[np.ndarray] = None,
    ):
        if tlwh is None:
            assert tlbr is not None
            tlwh = convert_tlbr_to_tlwh(tlbr)

        frame_id = int(frame_id)
        new_record = pd.DataFrame(
            {
                "Frame": [frame_id for _ in range(len(tracking_ids))],
                "ID": tracking_ids,
                "BBox_X": tlwh[:, 0],
                "BBox_Y": tlwh[:, 1],
                "BBox_W": tlwh[:, 2],
                "BBox_H": tlwh[:, 3],
                "Confidence": scores,
                "Class": [-1 for _ in range(len(tracking_ids))],
                "Visibility": [-1 for _ in range(len(tracking_ids))],
            }
        )
        self._dataframe_list.append(new_record)
        self.counter += 1

        if self.counter >= self.write_interval:
            self.write_data(self.output_file)
            self.first_write = False
            self.counter = 0  # Reset the counter after writing

    def get_data_by_frame(self, frame_number):
        """Get all tracking data for a specific frame."""
        if not self.data.empty:
            return self.data[self.data["Frame"] == frame_number]
        else:
            print("No data loaded.")
            return None

    # Function to extract tracking info by frame
    def get_tracking_info_by_frame(self, frame_id: int):
        frame_id = int(frame_id)
        # Filter the DataFrame for the specified frame
        frame_data = self.data[self.data["Frame"] == frame_id]
        # Extract columns as NumPy arrays
        tracking_ids = frame_data["ID"].to_numpy()
        scores = frame_data["Confidence"].to_numpy()
        tlwh = frame_data[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy()

        return tracking_ids, scores, tlwh


def convert_tlbr_to_tlwh(tlbr: Union[np.ndarray, torch.Tensor]):
    """
    Convert bounding boxes from TLBR format to TLWH format.

    Parameters:
    - tlbr (Tensor): A tensor containing bounding boxes in TLBR format (x1, y1, x2, y2).

    Returns:
    - Tensor: Bounding boxes in TLWH format (x, y, w, h).
    """
    # Ensure tlbr tensor is of the shape [N, 4] where N is the number of boxes
    if tlbr.ndim != 2 or tlbr.shape[1] != 4:
        raise ValueError("Input tensor must be of shape [N, 4]")

    # Top-left corner remains the same
    x = tlbr[:, 0]
    y = tlbr[:, 1]

    # Width and height are calculated as differences
    w = tlbr[:, 2] - tlbr[:, 0]
    h = tlbr[:, 3] - tlbr[:, 1]

    # Stack the results into a new tensor and return
    if isinstance(tlbr, np.ndarray):
        return np.stack([x, y, w, h], axis=1)
    return torch.stack([x, y, w, h], dim=1)

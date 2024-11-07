from typing import Optional

import numpy as np
import pandas as pd

from hmlib.tracking_utils.tracking_dataframe import (
    TrackingDataBase,
    convert_tlbr_to_tlwh,
)


class CameraTrackingDataFrame(TrackingDataBase):
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
                    "BBox_X",
                    "BBox_Y",
                    "BBox_W",
                    "BBox_H",
                ],
            )
            print("Data loaded successfully.")

    def add_frame_records(
        self,
        frame_id: int,
        tlbr: Optional[np.ndarray] = None,
        tlwh: Optional[np.ndarray] = None,
    ):
        if tlwh is None:
            assert tlbr is not None
            tlwh = convert_tlbr_to_tlwh(tlbr)

        frame_id = int(frame_id)
        new_record = pd.DataFrame(
            {
                "Frame": frame_id,
                "BBox_X": tlwh[:, 0],
                "BBox_Y": tlwh[:, 1],
                "BBox_W": tlwh[:, 2],
                "BBox_H": tlwh[:, 3],
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

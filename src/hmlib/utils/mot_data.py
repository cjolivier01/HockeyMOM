from typing import Optional

import pandas as pd


class MOTTrackingData:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = None
        if file_path:
            self.read_data()

    def read_data(self, input_path: Optional[str] = None):
        """Read MOT tracking data from a CSV file."""
        if not input_path:
            input_path = self.file_path
        if input_path:
            self.data = pd.read_csv(
                input_path,
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
        else:
            print("File path not set.")

    def write_data(self, output_path: Optional[str] = None):
        if not output_path:
            output_path = self.file_path

        """Write MOT tracking data to a CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False, header=False)
            print(f"Data saved successfully to {output_path}.")
        else:
            print("No data available to save.")

    def add_record(
        self,
        frame,
        obj_id,
        bbox_x,
        bbox_y,
        bbox_w,
        bbox_h,
        confidence=1,
        obj_class=-1,
        visibility=-1,
    ):
        """Add a record to the MOT tracking data."""
        if self.data is None:
            self.data = pd.DataFrame(
                columns=[
                    "Frame",
                    "ID",
                    "BBox_X",
                    "BBox_Y",
                    "BBox_W",
                    "BBox_H",
                    "Confidence",
                    "Class",
                    "Visibility",
                ]
            )

        new_record = {
            "Frame": frame,
            "ID": obj_id,
            "BBox_X": bbox_x,
            "BBox_Y": bbox_y,
            "BBox_W": bbox_w,
            "BBox_H": bbox_h,
            "Confidence": confidence,
            "Class": obj_class,
            "Visibility": visibility,
        }
        self.data = self.data.append(new_record, ignore_index=True)

    def get_data_by_frame(self, frame_number):
        """Get all tracking data for a specific frame."""
        if self.data is not None:
            return self.data[self.data["Frame"] == frame_number]
        else:
            print("No data loaded.")


if __name__ == "__main__":
    # Example usage:
    tracker = MOTTrackingData()
    tracker.add_record(1, 1, 100, 150, 50, 60)
    tracker.add_record(1, 2, 120, 160, 55, 65)
    tracker.write_data(
        "tracking_data.csv"
    )  # Specify the output path to save the CSV file

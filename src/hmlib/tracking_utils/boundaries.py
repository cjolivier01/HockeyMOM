import time
from typing import Union

import numpy as np
import torch

from hmlib.builder import PIPELINES
from hmlib.tracking_utils import visualization as vis


# @PIPELINES.register_module()
# class PruneDetections:
#     def __init__(self, det_thresh: float = 0.05):
#         self.det_thresh = det_thresh


@PIPELINES.register_module()
class BoundaryLines:
    def __init__(
        self,
        upper_border_lines,
        lower_border_lines,
        original_clip_box=None,
        det_thresh: float = 0.05,
    ):
        self._original_clip_box = original_clip_box
        self.det_thresh = det_thresh
        if self._original_clip_box is None:
            self._original_clip_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        elif not isinstance(self._original_clip_box, torch.Tensor):
            self._original_clip_box = torch.tensor(
                self._original_clip_box, dtype=torch.float
            )
        clip_upper_left = self._original_clip_box[0:2]
        if upper_border_lines:
            self._upper_borders = torch.tensor(upper_border_lines, dtype=torch.float)
            self._upper_borders[:, 0:2] -= clip_upper_left
            self._upper_borders[:, 2:4] -= clip_upper_left
            # self._upper_line_vectors = self.tlbr_to_line_vectors(self._upper_borders)
        else:
            self._upper_borders = None
            self._upper_line_vectors = None
        if lower_border_lines:
            self._lower_borders = torch.tensor(lower_border_lines, dtype=torch.float)
            self._lower_borders[:, 0:2] -= clip_upper_left
            self._lower_borders[:, 2:4] -= clip_upper_left
            # self._lower_line_vectors = self.tlbr_to_line_vectors(self._lower_borders)
        else:
            self._lower_borders = None
            self._lower_line_vectors = None
        self._passes = 0
        self._duration = 0

    def draw(self, img):
        if self._upper_borders is not None:
            for i in range(len(self._upper_borders)):
                img = vis.plot_line(
                    img,
                    self._upper_borders[i][0:2],
                    self._upper_borders[i][2:4],
                    color=(255, 0, 0),
                    thickness=1,
                )
        if self._lower_borders is not None:
            for i in range(len(self._lower_borders)):
                img = vis.plot_line(
                    img,
                    self._lower_borders[i][0:2],
                    self._lower_borders[i][2:4],
                    color=(0, 0, 255),
                    thickness=1,
                )
        return img

    # def tlbr_to_line_vectors(self, tlbr_batch):
    #     # Assuming tlbr_batch shape is (N, 4) with each box as [top, left, bottom, right]

    #     top_vectors = torch.stack(
    #         (tlbr_batch[:, 1], tlbr_batch[:, 0]), dim=1
    #     ) - torch.stack((tlbr_batch[:, 3], tlbr_batch[:, 0]), dim=1)
    #     left_vectors = torch.stack(
    #         (tlbr_batch[:, 1], tlbr_batch[:, 0]), dim=1
    #     ) - torch.stack((tlbr_batch[:, 1], tlbr_batch[:, 2]), dim=1)
    #     bottom_vectors = torch.stack(
    #         (tlbr_batch[:, 3], tlbr_batch[:, 2]), dim=1
    #     ) - torch.stack((tlbr_batch[:, 1], tlbr_batch[:, 2]), dim=1)
    #     right_vectors = torch.stack(
    #         (tlbr_batch[:, 3], tlbr_batch[:, 2]), dim=1
    #     ) - torch.stack((tlbr_batch[:, 3], tlbr_batch[:, 0]), dim=1)

    #     # Concatenating vectors for all sides
    #     line_vectors = torch.stack(
    #         (top_vectors, left_vectors, bottom_vectors, right_vectors), dim=1
    #     )

    #     return line_vectors

    # def is_point_too_high(self, point):
    #     return False

    # def is_point_too_low(self, point):
    #     too_low = self.is_point_below_line(point)
    #     return False

    # def is_point_outside(self, point):
    #     return self.is_point_above_line(point) or self.is_point_below_line(point)

    # def _is_point_above_line(self, point, line_start, line_end):
    #     # Create vectors
    #     line_vec = line_end - line_start
    #     point_vec = point - line_start

    #     # Compute the cross product
    #     cross_product_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]

    #     # Check if the point is above the line
    #     return cross_product_z < 0

    # def is_point_above_line(self, point):
    #     if self._upper_borders is None:
    #         return False
    #     point = point.cpu()
    #     x = point[0]
    #     # y = point[1]
    #     for i, high_border in enumerate(self._upper_borders):
    #         if x >= high_border[0] and x <= high_border[2]:
    #             if self._is_point_above_line(point, high_border[0:2], high_border[2:4]):
    #                 return True
    #     return False

    def point_batch_check_point_above_segments(self, points, line_segments):
        # Step 1: Expand the arrays
        # Create indices to represent each combination
        point_indices, segment_indices = np.meshgrid(
            np.arange(points.shape[0]), np.arange(line_segments.shape[0]), indexing="ij"
        )

        point_indices = torch.from_numpy(point_indices).to(points.device)
        segment_indices = torch.from_numpy(segment_indices).to(points.device)
        if isinstance(line_segments, np.ndarray):
            line_segments = torch.from_numpy(line_segments)
        line_segments = line_segments.to(points.device)

        # Expand points and segments to match each combination
        expanded_points = points[point_indices.ravel()]
        expanded_segments = line_segments[segment_indices.ravel()]

        # Step 2: Calculate slope and intercept for each expanded line segment
        slopes = (expanded_segments[:, 3] - expanded_segments[:, 1]) / (
            expanded_segments[:, 2] - expanded_segments[:, 0]
        )
        intercepts = expanded_segments[:, 1] - slopes * expanded_segments[:, 0]

        # Step 3: Calculate y value of the line at each point's x
        y_values_at_points_x = slopes * expanded_points[:, 0] + intercepts

        # Step 4: Filter for points where x is within the line segment's x range
        within_x_range = (
            expanded_points[:, 0]
            >= torch.min(expanded_segments[:, 0], expanded_segments[:, 2])
        ) & (
            expanded_points[:, 0]
            <= torch.max(expanded_segments[:, 0], expanded_segments[:, 2])
        )

        # Step 5: Determine if the point's y is above the line's y at that x
        points_above_line = (
            expanded_points[:, 1] > y_values_at_points_x
        ) & within_x_range

        # Reshape to original shape for clarity
        comparison_matrix = points_above_line.reshape(
            points.shape[0], line_segments.shape[0]
        )
        comparison = torch.any(comparison_matrix, axis=1)
        return comparison

    def point_batch_check_point_below_segments(self, points, line_segments):
        # Step 1: Expand the arrays
        # Create indices to represent each combination
        point_indices, segment_indices = np.meshgrid(
            np.arange(points.shape[0]), np.arange(line_segments.shape[0]), indexing="ij"
        )

        point_indices = torch.from_numpy(point_indices).to(points.device)
        segment_indices = torch.from_numpy(segment_indices).to(points.device)
        if isinstance(line_segments, np.ndarray):
            line_segments = torch.from_numpy(line_segments)
        line_segments = line_segments.to(points.device)

        # Expand points and segments to match each combination
        expanded_points = points[point_indices.ravel()]
        expanded_segments = line_segments[segment_indices.ravel()]

        # Step 2: Calculate slope and intercept for each expanded line segment
        slopes = (expanded_segments[:, 3] - expanded_segments[:, 1]) / (
            expanded_segments[:, 2] - expanded_segments[:, 0]
        )
        intercepts = expanded_segments[:, 1] - slopes * expanded_segments[:, 0]

        # Step 3: Calculate y value of the line at each point's x
        y_values_at_points_x = slopes * expanded_points[:, 0] + intercepts

        # Step 4: Filter for points where x is within the line segment's x range
        within_x_range = (
            expanded_points[:, 0]
            >= torch.min(expanded_segments[:, 0], expanded_segments[:, 2])
        ) & (
            expanded_points[:, 0]
            <= torch.max(expanded_segments[:, 0], expanded_segments[:, 2])
        )

        # Step 5: Determine if the point's y is below the line's y at that x
        points_above_line = (
            expanded_points[:, 1] < y_values_at_points_x
        ) & within_x_range

        # Reshape to original shape for clarity
        comparison_matrix = points_above_line.reshape(
            points.shape[0], line_segments.shape[0]
        )
        comparison = torch.any(comparison_matrix, axis=1)
        return comparison

    # def is_point_below_line(self, point):
    #     if self._lower_borders is None:
    #         return False
    #     point = point.cpu()
    #     x = point[0]
    #     # y = point[1]
    #     for i, low_border in enumerate(self._lower_borders):
    #         if x >= low_border[0] and x <= low_border[2]:
    #             if self._is_point_below_line(point, low_border[0:2], low_border[2:4]):
    #                 return True
    #     return False

    # def _is_point_below_line(self, point, line_start, line_end):
    #     # Create vectors
    #     line_vec = line_end - line_start
    #     point_vec = point - line_start

    #     # Compute the cross product
    #     cross_product_z = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]

    #     # Check if the point is below the line
    #     return cross_product_z > 0

    def get_centers(self, bbox_tlbr: Union[torch.Tensor, np.ndarray]):
        # Calculate the centers
        # The center x coordinates are calculated by averaging the left and right coordinates
        centers_x = (bbox_tlbr[:, 1] + bbox_tlbr[:, 3]) / 2

        # The center y coordinates are calculated by averaging the top and bottom coordinates
        centers_y = (bbox_tlbr[:, 0] + bbox_tlbr[:, 2]) / 2

        if isinstance(bbox_tlbr, np.ndarray):
            # Combine the x and y center coordinates
            centers = np.vstack((centers_y, centers_x)).T
        else:
            # Combine the x and y center coordinates into a single tensor
            centers = torch.stack((centers_y, centers_x), dim=1)
        return centers

    # def prune(bboxes, inclusion_box):
    #     if len(bboxes) == 0:
    #         # nothing
    #         return bboxes
    #     filtered_online_tlwh = []
    #     filtered_online_ids = []
    #     online_tlwhs_centers = tlwh_centers(tlwhs=online_tlwhs)
    #     for i in range(len(online_tlwhs_centers)):
    #         center = online_tlwhs_centers[i]
    #         if inclusion_box is not None:
    #             if inclusion_box[0] and center[0] < inclusion_box[0]:
    #                 continue
    #             elif inclusion_box[2] and center[0] > inclusion_box[2]:
    #                 continue
    #             elif inclusion_box[1] and center[1] < inclusion_box[1]:
    #                 continue
    #             elif inclusion_box[3] and center[1] > inclusion_box[3]:
    #                 continue
    #         if boundaries is not None:
    #             # TODO: boundaries could be done with the box edges
    #             if boundaries.is_point_outside(center):
    #                 # print(f"ignoring: {center}")
    #                 continue
    #         filtered_online_tlwh.append(online_tlwhs[i])
    #         filtered_online_ids.append(online_ids[i])
    #     if len(filtered_online_tlwh) == 0:
    #         assert len(filtered_online_ids) == 0
    #         return [], []
    #     return torch.stack(filtered_online_tlwh), torch.stack(filtered_online_ids)

    def prune_items_index(self, batch_item_bboxes: Union[torch.Tensor, np.ndarray]):
        centers = self.get_centers(bbox_tlbr=batch_item_bboxes)

        above_line = self.point_batch_check_point_above_segments(
            centers,
            self._lower_borders,
        )
        below_line = self.point_batch_check_point_below_segments(
            centers,
            self._upper_borders,
        )
        above_or_below = torch.logical_or(above_line, below_line)
        return torch.logical_not(above_or_below)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, data, **kwargs):
        start = time.time()
        if "prune_list" not in data:
            return data
        prune_list = data["prune_list"]
        bbox_tensors = data[prune_list[0]]

        if bbox_tensors.shape[1] == 6:
            # Tracking box (index + tlbr + score)
            bboxes = bbox_tensors[:, 1:5]
        elif bbox_tensors.shape[1] == 5:
            # Detection tlbr + score
            bboxes = bbox_tensors[:, :4]
        else:
            assert False
        keep_indexes = self.prune_items_index(batch_item_bboxes=bboxes)
        for name in prune_list:
            data[name] = data[name][keep_indexes]

        self._duration += time.time() - start
        self._passes += 1
        if self._passes % 50 == 0:
            fps = self._passes / self._duration
            if fps > 50 or True:
                print(f"Boundary pruning speed: {self._passes/self._duration} fps")
            self._passes = 0
            self._duration = 0
        return data

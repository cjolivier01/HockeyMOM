import time
from typing import List, Optional, Union

import numpy as np
import torch

import hmlib.vis.pt_visualization as ptv
from hmlib.builder import PIPELINES
from hmlib.constants import WIDTH_NORMALIZATION_SIZE
from hmlib.tracking_utils import visualization as vis
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import image_height, image_width, make_channels_first

from .boundaries import adjust_point_for_clip_box, adjust_tlbr_for_clip_box


@PIPELINES.register_module()
class SegmBoundaries:

    def __init__(
        self,
        segment_mask: Optional[torch.Tensor] = None,
        centroid: Optional[torch.Tensor] = None,
        original_clip_box: Optional[Union[torch.Tensor, List[int]]] = None,
        det_thresh: float = 0.05,
        draw: bool = False,
        raise_bbox_center_by_height_ratio: float = -0.2,  # FIXME: used to be 0.1, Jr Gulls Game 1 Hack
        lower_bbox_bottom_by_height_ratio: float = 0.1,
    ):
        if isinstance(original_clip_box, list) and len(original_clip_box):
            assert len(original_clip_box) == 4
            assert original_clip_box[2] > original_clip_box[0]
            assert original_clip_box[3] > original_clip_box[1]
            original_clip_box = torch.tensor(original_clip_box, dtype=torch.int64)
        self._original_clip_box = original_clip_box
        self.det_thresh = det_thresh
        self._passes = 0
        self._duration = 0
        self._raise_bbox_center_by_height_ratio = raise_bbox_center_by_height_ratio
        self._lower_bbox_bottom_by_height_ratio = lower_bbox_bottom_by_height_ratio
        self._normalization_scale: float | None = None
        self._draw = draw
        self._color_mask = torch.tensor([0, 255, 0], dtype=torch.uint8).reshape(3, 1)
        self.set_segment_mask_and_centroid(segment_mask, centroid)

    def set_segment_mask_and_centroid(self, segment_mask: torch.Tensor, centroid: torch.Tensor):
        self._segment_mask = segment_mask
        self._centroid = centroid
        if (
            self._original_clip_box is not None
            and len(self._original_clip_box)
            and self._segment_mask is not None
        ):
            # clip the mask to this box as well
            x1, y1, x2, y2 = self._original_clip_box
            assert self._segment_mask.ndim == 2
            self._segment_mask = self._segment_mask[y1:y2, x1:x2]

    def draw(self, img):
        if self._segment_mask is not None:
            assert self._segment_mask.ndim == 2
            assert self._segment_mask.shape[0] == image_height(img)
            assert self._segment_mask.shape[1] == image_width(img)
            # alpha = 0.05
            alpha = 0.10
            if isinstance(img, StreamTensor):
                img = img.wait()
            # Make sure we're all compatible tensors
            if self._segment_mask.device != img.device:
                self._segment_mask = self._segment_mask.to(img.device)
            if self._color_mask.device != img.device:
                self._color_mask = self._color_mask.to(img.device)
            if self._color_mask.dtype != img.dtype:
                self._color_mask = self._color_mask.to(img.dtype)
            img = make_channels_first(img)
            if not torch.is_floating_point(img):
                img = img.to(torch.float, non_blocking=True)
            img[:, :, self._segment_mask] = img[:, :, self._segment_mask] * (1 - alpha) + self._color_mask * alpha
        if self._centroid is not None:
            img = ptv.draw_filled_square(
                img,
                center_x=int(self._centroid[0]),
                center_y=int(self._centroid[1]),
                color=(255, 0, 0),
                size=25,
            )
        return img

    def get_centers(self, bbox_tlbr: Union[torch.Tensor, np.ndarray]):
        # FIXME: THIS HAS X AND Y VAR NAMES BACKWARDS, BUT RETURNS (X, Y) CORRECTLY
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

    def get_bottoms(self, bbox_tlbr: Union[torch.Tensor, np.ndarray]):
        # Calculate the centers
        # The center x coordinates are calculated by averaging the left and right coordinates
        centers_x = (bbox_tlbr[:, 0] + bbox_tlbr[:, 2]) / 2

        # The center y coordinates are calculated by averaging the top and bottom coordinates
        bottoms_y = bbox_tlbr[:, 3]

        if isinstance(bbox_tlbr, np.ndarray):
            # Combine the x and y center coordinates
            bottoms = np.vstack((centers_x, bottoms_y)).T
        else:
            # Combine the x and y center coordinates into a single tensor
            bottoms = torch.stack((centers_x, bottoms_y), dim=1)
        return bottoms

    def prune_items_index(self, batch_item_bboxes: Union[torch.Tensor, np.ndarray]):
        if (
            self._raise_bbox_center_by_height_ratio or self._raise_bbox_center_by_height_ratio != 1
        ) or (
            self._lower_bbox_bottom_by_height_ratio or self._lower_bbox_bottom_by_height_ratio != 1
        ):
            bbox_heights = calculate_box_heights(batch_item_bboxes)
        else:
            bbox_heights = None

        all_centers = self.get_centers(bbox_tlbr=batch_item_bboxes)

        # Center points are for the bottom of the rink (usually top of the wall),
        # so we may want to move it up a bit (reduce Y, since Y is up)
        if self._raise_bbox_center_by_height_ratio or self._raise_bbox_center_by_height_ratio != 1:
            all_centers[:, 1] -= bbox_heights * self._raise_bbox_center_by_height_ratio

        all_bottoms = self.get_bottoms(bbox_tlbr=batch_item_bboxes)

        # Bottom points are for the top of the rink, since we can see the ice on the far
        # side at the bottom of the wall, so we may want to move that down a little (larger Y)
        if self._lower_bbox_bottom_by_height_ratio or self._lower_bbox_bottom_by_height_ratio != 1:
            all_bottoms[:, 1] -= bbox_heights * self._lower_bbox_bottom_by_height_ratio

        points = select_points(
            y_threshold=self._centroid[1],
            points_when_above=all_bottoms,
            points_when_below=all_centers,
        )

        valid_x = (points[:, 0] >= 0) & (points[:, 0] < self._segment_mask.shape[1])
        valid_y = (points[:, 1] >= 0) & (points[:, 1] < self._segment_mask.shape[0])
        valid_points = valid_x & valid_y

        # Filter points to keep only valid ones
        valid_points_indices = torch.where(valid_points)[0]
        valid_points_filtered = points[valid_points_indices]

        if self._segment_mask.device != valid_points_filtered.device:
            self._segment_mask = self._segment_mask.to(
                device=valid_points_filtered.device, non_blocking=True
            )

        # Check mask values at these points
        mask_values = self._segment_mask[
            valid_points_filtered[:, 1].to(torch.long, non_blocking=True),
            valid_points_filtered[:, 0].to(torch.long, non_blocking=True),
        ]

        # Get indices of valid points where the mask is also True
        final_indices = valid_points_indices[mask_values]

        return final_indices

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, data, **kwargs):
        if self._segment_mask is None:
            # We don't have any information to go on
            return data

        if self._normalization_scale is None:
            # Adjust thresholds based on image size
            img_width = self._segment_mask.shape[-1]
            self._normalization_scale = img_width / WIDTH_NORMALIZATION_SIZE
            # if self._normalization_scale > 0:
            #     self._raise_bbox_center_by_height_ratio *= self._normalization_scale / 2
            #     self._lower_bbox_bottom_by_height_ratio *= self._normalization_scale / 2

        # Maybe we render on the original image
        if self._draw and "original_images" in data:
            data["original_images"] = self.draw(img=data["original_images"])

        if "prune_list" not in data:
            # We don't have any data to prune
            return data

        start = time.time()
        prune_list = data["prune_list"]
        bbox_tensors = data[prune_list[0]]

        if bbox_tensors.shape[1] == 6:
            # Tracking box (index + tlbr + score)
            bboxes = bbox_tensors[:, 1:5]
        elif bbox_tensors.shape[1] == 5:
            # Detection tlbr + score
            bboxes = bbox_tensors[:, :4]
        elif bbox_tensors.shape[1] == 4:
            # Detection tlbr only
            bboxes = bbox_tensors
        else:
            assert False
        keep_indexes = self.prune_items_index(batch_item_bboxes=bboxes)
        for name in prune_list:
            data[name] = data[name][keep_indexes]

        self._duration += time.time() - start
        self._passes += 1
        if self._passes % 50 == 0:
            fps = self._passes / self._duration
            if fps < 50:
                print(f"Segment Boundary pruning speed: {self._passes/self._duration} fps")
            self._passes = 0
            self._duration = 0
        return data


def calculate_box_heights(bboxes: torch.Tensor) -> torch.Tensor:
    """
    Calculate the heights of bounding boxes.

    Args:
        bboxes (torch.Tensor): Tensor of shape (N, 4) where each row contains [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Tensor of heights for each bounding box.
    """
    # The height of each bounding box is y2 - y1
    heights = bboxes[:, 3] - bboxes[:, 1]
    return heights


def select_points(
    y_threshold: Union[float, torch.Tensor, np.ndarray],
    points_when_below: Union[torch.Tensor, np.ndarray],
    points_when_above: Union[torch.Tensor, np.ndarray],
):
    """
    Select points from two batches based on a y-value condition.

    Parameters:
        y_threshold (float): The threshold value for the y-coordinate.
        center_points (torch.Tensor): Tensor of shape [B, 2] with center points.
        bottom_points (torch.Tensor): Tensor of shape [B, 2] with bottom points.

    Returns:
        torch.Tensor: A tensor of selected points based on the condition.
    """
    # Normalize inputs to torch for consistent downstream ops
    if isinstance(points_when_below, np.ndarray):
        points_when_below = torch.from_numpy(points_when_below)
    if isinstance(points_when_above, np.ndarray):
        points_when_above = torch.from_numpy(points_when_above)
    if isinstance(y_threshold, np.ndarray):
        # expects scalar-like value
        y_threshold = float(y_threshold)

    # Check if the y-values of the center_points are above the threshold
    # Note that increasing y is down, so the comparison is reversed
    mask = points_when_below[:, 1] > y_threshold  # Assumes y-coordinate is at index 1

    # Use the mask to select between center_points and bottom_points
    selected_points = torch.where(mask.unsqueeze(1), points_when_below, points_when_above)

    return selected_points

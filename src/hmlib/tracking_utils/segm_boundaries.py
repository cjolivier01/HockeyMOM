import time
from typing import List, Optional, Union

import numpy as np
import torch

from hmlib.builder import PIPELINES
from hmlib.tracking_utils import visualization as vis
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import image_height, image_width, make_channels_first

from .boundaries import adjust_point_for_clip_box, adjust_tlbr_for_clip_box


@PIPELINES.register_module()
class SegmBoundaries:

    def __init__(
        self,
        rink_mask: Optional[torch.Tensor] = None,
        centroid: Optional[torch.Tensor] = None,
        original_clip_box: Optional[Union[torch.Tensor, List[int]]] = None,
        det_thresh: float = 0.05,
        draw: bool = False,
        raise_bbox_by_height_ratio: float = 0.25,
    ):
        if isinstance(original_clip_box, list) and len(original_clip_box):
            assert len(original_clip_box) == 4
            assert original_clip_box[2] > original_clip_box[0]
            assert original_clip_box[3] > original_clip_box[1]
            original_clip_box = torch.tensor(original_clip_box, dtype=torch.int64)
        self._original_clip_box = original_clip_box
        self._rink_mask = rink_mask
        self._centroid = centroid
        self.det_thresh = det_thresh
        self._passes = 0
        self._duration = 0
        self._raise_bbox_by_height_ratio = raise_bbox_by_height_ratio
        self._draw = draw
        self._color_mask = torch.tensor([0, 255, 0], dtype=torch.uint8).reshape(3, 1)
        if (
            self._original_clip_box is not None
            and len(self._original_clip_box)
            and self._rink_mask is not None
        ):
            # clip the mask to this box as well
            x1, y1, x2, y2 = self._original_clip_box
            assert self._rink_mask.ndim == 2
            self._rink_mask = self._rink_mask[y1:y2, x1:x2]
            pass

    def draw(self, img):
        if self._rink_mask is not None:
            assert self._rink_mask.ndim == 2
            assert self._rink_mask.shape[0] == image_height(img)
            assert self._rink_mask.shape[1] == image_width(img)
            alpha = 0.2
            if isinstance(img, StreamTensor):
                img = img.wait()
            # Make sure we're all compatible tensors
            if self._rink_mask.device != img.device:
                self._rink_mask = self._rink_mask.to(img.device)
            if self._color_mask.device != img.device:
                self._color_mask = self._color_mask.to(img.device)
            if self._color_mask.dtype != img.dtype:
                self._color_mask = self._color_mask.to(img.dtype)
            img = make_channels_first(img)
            if not torch.is_floating_point(img):
                img = img.to(torch.float, non_blocking=True)
            img[:, :, self._rink_mask] = (
                img[:, :, self._rink_mask] * (1 - alpha) + self._color_mask * alpha
            )
        if self._centroid is not None:
            img = pt_draw_square(
                img, center_x=int(self._centroid[0]), center_y=int(self._centroid[1])
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
        # TODO: Not checking center point
        # points = self.get_centers(bbox_tlbr=batch_item_bboxes)
        points = self.get_bottoms(bbox_tlbr=batch_item_bboxes)

        if self._raise_bbox_by_height_ratio is not None and self._raise_bbox_by_height_ratio != 1:
            heights = calculate_box_heights(batch_item_bboxes)
            points[:, 1] += heights * self._raise_bbox_by_height_ratio

        valid_x = (points[:, 0] >= 0) & (points[:, 0] < self._rink_mask.shape[1])
        valid_y = (points[:, 1] >= 0) & (points[:, 1] < self._rink_mask.shape[0])
        valid_points = valid_x & valid_y

        # Filter points to keep only valid ones
        valid_points_indices = torch.where(valid_points)[0]
        valid_points_filtered = points[valid_points_indices]

        # Check mask values at these points
        mask_values = self._rink_mask[
            valid_points_filtered[:, 1].to(torch.long, non_blocking=True),
            valid_points_filtered[:, 0].to(torch.long, non_blocking=True),
        ]

        # Get indices of valid points where the mask is also True
        final_indices = valid_points_indices[mask_values]

        return final_indices

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, data, **kwargs):
        if self._rink_mask is None:
            # We don't have any information to go on
            return data

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


def pt_draw_square(image, center_x: int, center_y: int, size=20, color=(0, 100, 0)):
    """
    Draw a square on the image at specified location using PyTorch.

    Parameters:
        image (torch.Tensor): The image tensor of shape [3, H, W]
        top_left_x (int): The x coordinate of the top left corner of the square
        top_left_y (int): The y coordinate of the top left corner of the square
        size (int): The size of the side of the square
        color (tuple): The RGB color of the square
    """
    # Ensure the square doesn't go out of the image boundaries
    top_left_x = int(center_x - size // 2)
    top_left_y = int(center_y - size // 2)
    H, W = image_height(image), image_width(image)
    if top_left_x + size > W or top_left_y + size > H:
        raise ValueError("Square goes out of image boundaries.")

    # Set the pixel values to the specified color
    for c in range(3):  # Loop over color channels
        image[c, top_left_y : top_left_y + size, top_left_x : top_left_x + size] = (
            color[c] / 255.0
        )  # Normalize if your image is in [0,1]
    return image

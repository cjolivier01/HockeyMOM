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
        original_clip_box: Optional[Union[torch.Tensor, List[int]]] = None,
        det_thresh: float = 0.05,
        draw: bool = False,
    ):
        if isinstance(original_clip_box, list) and len(original_clip_box):
            assert len(original_clip_box) == 4
            assert original_clip_box[2] > original_clip_box[0]
            assert original_clip_box[3] > original_clip_box[1]
            original_clip_box = torch.tensor(original_clip_box, dtype=torch.int64)
        self._original_clip_box = original_clip_box
        self._rink_mask = rink_mask
        self.det_thresh = det_thresh
        self._passes = 0
        self._duration = 0
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

    # def set_boundaries(
    #     self,
    #     upper: Optional[torch.Tensor] = None,
    #     lower: Optional[torch.Tensor] = None,
    #     source_clip_box: Optional[torch.Tensor] = None,
    # ):
    #     if upper is not None:
    #         self._upper_borders = torch.tensor(upper, dtype=torch.float)
    #     else:
    #         self._upper_borders = None
    #     if lower is not None:
    #         self._lower_borders = torch.tensor(lower, dtype=torch.float)
    #     else:
    #         self._lower_borders = None
    #     if source_clip_box is not None:
    #         self.adjust_for_source_clip_box(source_clip_box)

    # def adjust_for_source_clip_box(self, source_clip_box: torch.Tensor):
    #     # clip_upper_left = source_clip_box[0:2]
    #     if self._upper_borders is not None:
    #         self._upper_borders = adjust_tlbr_for_clip_box(self._upper_borders, source_clip_box)
    #         # self._upper_borders[:, 0:2] -= clip_upper_left
    #         # self._upper_borders[:, 2:4] -= clip_upper_left
    #     if self._lower_borders is not None:
    #         self._lower_borders = adjust_tlbr_for_clip_box(self._lower_borders, source_clip_box)
    #         # self._lower_borders[:, 0:2] -= clip_upper_left
    #         # self._lower_borders[:, 2:4] -= clip_upper_left

    def draw(self, img):
        if self._rink_mask is not None:
            assert self._rink_mask.ndim == 2
            assert self._rink_mask.shape[0] == image_height(img)
            assert self._rink_mask.shape[1] == image_width(img)
            color_mask = torch.tensor([0, 255, 0], dtype=img.dtype, device=img.device)
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
            # img[:, :, self._rink_mask] = (
            #     img[:, :, self._rink_mask] * (1 - alpha) + color_mask * alpha
            # )
            img[:, :, self._rink_mask] = (
                img[:, :, self._rink_mask] * (1 - alpha) + self._color_mask * alpha
            )
        return img

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
        # data[""]
        if self._draw and "original_images" in data:
            data["original_images"] = self.draw(img=data["original_images"])
        return data
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
            if fps < 50:
                print(f"Boundary pruning speed: {self._passes/self._duration} fps")
            self._passes = 0
            self._duration = 0
        return data

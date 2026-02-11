import contextlib
import time
from typing import List, Optional, Union

import numpy as np
import torch

import hmlib.vis.pt_visualization as ptv
from hmlib.builder import PIPELINES
from hmlib.constants import WIDTH_NORMALIZATION_SIZE
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.utils.image import image_height, image_width, make_channels_first


@PIPELINES.register_module()
class SegmBoundaries:
    def __init__(
        self,
        segment_mask: Optional[torch.Tensor] = None,
        centroid: Optional[torch.Tensor] = None,
        original_clip_box: Optional[Union[torch.Tensor, List[int]]] = None,
        det_thresh: float = 0.05,
        max_detections_in_mask: Optional[int] = None,
        draw: bool = False,
        # raise_bbox_center_by_height_ratio: float = -0.2,  # FIXME: used to be 0.1, Jr Gulls Game 1 Hack
        raise_bbox_center_by_height_ratio: float = -0.1,  # FIXME: used to be 0.1, Jr Gulls Game 1 Hack
        lower_bbox_bottom_by_height_ratio: float = 0.1,
    ):
        if isinstance(original_clip_box, list) and len(original_clip_box):
            assert len(original_clip_box) == 4
            assert original_clip_box[2] > original_clip_box[0]
            assert original_clip_box[3] > original_clip_box[1]
            original_clip_box = torch.tensor(original_clip_box, dtype=torch.int64)
        self._original_clip_box = original_clip_box
        self.det_thresh = det_thresh
        self._iter_num = 0
        self._duration = 0
        self._raise_bbox_center_by_height_ratio = raise_bbox_center_by_height_ratio
        self._lower_bbox_bottom_by_height_ratio = lower_bbox_bottom_by_height_ratio
        self._max_detections_in_mask: Optional[int] = (
            int(max_detections_in_mask) if max_detections_in_mask is not None else None
        )
        self._normalization_scale: float | None = None
        self._draw = draw
        # shape (3, 1, 1) so it broadcasts over (C, H, W)
        self._color_mask = torch.tensor([0, 255, 0], dtype=torch.uint8).reshape(3, 1, 1)
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
            alpha = 0.10
            if isinstance(img, StreamTensorBase):
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
            # Broadcast mask over channels to avoid boolean indexing with dynamic shapes
            mask = self._segment_mask
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            # (1, H, W) -> (C, H, W) via broadcasting
            mask = mask.unsqueeze(0)
            img = torch.where(mask, img * (1 - alpha) + self._color_mask * alpha, img)
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
        # Ensure bboxes are a torch tensor
        if isinstance(batch_item_bboxes, np.ndarray):
            bboxes = torch.from_numpy(batch_item_bboxes)
        else:
            bboxes = batch_item_bboxes

        if (
            self._raise_bbox_center_by_height_ratio or self._raise_bbox_center_by_height_ratio != 1
        ) or (
            self._lower_bbox_bottom_by_height_ratio or self._lower_bbox_bottom_by_height_ratio != 1
        ):
            bbox_heights = calculate_box_heights(bboxes)
        else:
            bbox_heights = None

        all_centers = self.get_centers(bbox_tlbr=bboxes)

        # Center points are for the bottom of the rink (usually top of the wall),
        # so we may want to move it up a bit (reduce Y, since Y is up)
        if self._raise_bbox_center_by_height_ratio or self._raise_bbox_center_by_height_ratio != 1:
            all_centers[:, 1] -= bbox_heights * self._raise_bbox_center_by_height_ratio

        all_bottoms = self.get_bottoms(bbox_tlbr=bboxes)

        # Bottom points are for the top of the rink, since we can see the ice on the far
        # side at the bottom of the wall, so we may want to move that down a little (larger Y)
        if self._lower_bbox_bottom_by_height_ratio or self._lower_bbox_bottom_by_height_ratio != 1:
            all_bottoms[:, 1] -= bbox_heights * self._lower_bbox_bottom_by_height_ratio

        # Choose center vs bottom points based on centroid
        y_threshold = self._centroid[1]
        if isinstance(y_threshold, torch.Tensor) and y_threshold.numel() == 1:
            if y_threshold.device.type != "cuda":
                y_threshold = float(y_threshold)
        points = select_points(
            y_threshold=y_threshold,
            points_when_above=all_bottoms,
            points_when_below=all_centers,
        )

        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points)

        # Move the segment mask to the same device as the points once
        if self._segment_mask.device != points.device:
            self._segment_mask = self._segment_mask.to(device=points.device, non_blocking=True)

        height = int(self._segment_mask.shape[0])
        width = int(self._segment_mask.shape[1])

        # Valid coordinates inside the mask bounds
        valid_x = (points[:, 0] >= 0) & (points[:, 0] < width)
        valid_y = (points[:, 1] >= 0) & (points[:, 1] < height)
        valid_points = valid_x & valid_y

        # Clamp coordinates to valid range for indexing, then mask out invalid ones later.
        xs = points[:, 0].to(dtype=torch.long)
        ys = points[:, 1].to(dtype=torch.long)
        xs = xs.clamp(0, width - 1)
        ys = ys.clamp(0, height - 1)

        # Sample mask at all candidate points (fixed shape: one value per bbox)
        mask_values = self._segment_mask[ys, xs]
        if mask_values.dtype != torch.bool:
            mask_values = mask_values.to(torch.bool)

        # Keep boxes with valid coordinates that also lie inside the rink mask
        keep_mask = valid_points & mask_values
        return keep_mask

    def _apply_keep_mask(self, tensor: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0 or tensor.shape[0] != keep_mask.shape[0]:
            return tensor
        mask = keep_mask
        if mask.device != tensor.device:
            mask = mask.to(device=tensor.device, non_blocking=True)
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if tensor.ndim > 1:
            mask = mask.view(-1, *([1] * (tensor.ndim - 1)))
        if tensor.dtype == torch.bool:
            return tensor & mask
        return tensor * mask.to(dtype=tensor.dtype)

    def __call__(self, *args, **kwargs):
        self._iter_num += 1
        # do_trace = self._iter_num == 4
        # if do_trace:
        #     pass
        # from cuda_stacktrace import CudaStackTracer

        # with CudaStackTracer(functions=["cudaStreamSynchronize"], enabled=do_trace):
        with contextlib.nullcontext():
            results = self.forward(*args, **kwargs)
        # if do_trace:
        #     pass
        return results

    def forward(self, data, **kwargs):
        if self._segment_mask is None:
            # We don't have any information to go on
            return data

        if self._normalization_scale is None:
            # Adjust thresholds based on image size
            img_width = self._segment_mask.shape[-1]
            self._normalization_scale = img_width / WIDTH_NORMALIZATION_SIZE
            if self._normalization_scale > 0:
                self._raise_bbox_center_by_height_ratio *= self._normalization_scale / 2
                self._lower_bbox_bottom_by_height_ratio *= self._normalization_scale / 2

        # Maybe we render on the original image
        if self._draw and "original_images" in data:
            data["original_images"] = self.draw(img=data["original_images"])

        if "prune_list" not in data:
            # We don't have any data to prune
            return data

        start = time.time()
        prune_list = data["prune_list"]
        bbox_tensors = data[prune_list[0]]
        if isinstance(bbox_tensors, StreamTensorBase):
            bbox_tensors = unwrap_tensor(bbox_tensors)

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
        keep_mask = self.prune_items_index(batch_item_bboxes=bboxes)
        if not isinstance(keep_mask, torch.Tensor):
            keep_mask = torch.as_tensor(keep_mask)
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        keep_count = keep_mask.to(dtype=torch.long).sum()

        if self._max_detections_in_mask is not None:
            max_items = max(int(self._max_detections_in_mask), 0)
            N = int(keep_mask.shape[0])
            score_key = None
            for key in prune_list:
                if "score" in key:
                    score_key = key
                    break
            if score_key is None and "scores" in data:
                score_key = "scores"
            scores_t: Optional[torch.Tensor] = None
            if score_key is not None:
                scores_val = data.get(score_key)
                if isinstance(scores_val, StreamTensorBase):
                    scores_val = unwrap_tensor(scores_val)
                if isinstance(scores_val, torch.Tensor) and scores_val.shape[0] == N:
                    scores_t = scores_val
                    if scores_t.device != keep_mask.device:
                        scores_t = scores_t.to(device=keep_mask.device, non_blocking=True)

            if N and max_items > 0:
                if scores_t is not None:
                    if not torch.is_floating_point(scores_t):
                        scores_t = scores_t.to(dtype=torch.float32)
                    priority = scores_t.clone()
                    priority = priority.masked_fill(~keep_mask, float("-inf"))
                    k = min(max_items, N)
                    order = torch.topk(priority, k=k).indices
                else:
                    idx = torch.arange(N, device=keep_mask.device)
                    order_key = torch.where(keep_mask, idx, idx + N)
                    order = torch.argsort(order_key)
                    if max_items < N:
                        order = order[:max_items]
            else:
                order = keep_mask.new_empty((0,), dtype=torch.long)

            num_valid = keep_count.clamp(max=max_items)
            data["num_detections"] = num_valid
            valid_mask = None
            if max_items > 0:
                valid_mask = torch.arange(max_items, device=keep_mask.device) < num_valid

            for name in prune_list:
                value = data.get(name)
                if value is None:
                    continue
                if isinstance(value, StreamTensorBase):
                    tensor = unwrap_tensor(value)
                else:
                    tensor = value
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.ndim == 0 or tensor.shape[0] != N:
                    data[name] = tensor
                    continue
                sel_order = order
                if sel_order.device != tensor.device:
                    sel_order = sel_order.to(device=tensor.device, non_blocking=True)
                if sel_order.numel():
                    selected = tensor.index_select(0, sel_order)
                else:
                    selected = tensor[:0]
                if selected.shape[0] < max_items:
                    padded = selected.new_zeros((max_items, *selected.shape[1:]))
                    if selected.numel():
                        padded[: selected.shape[0]].copy_(selected)
                    selected = padded
                if valid_mask is not None:
                    mask = valid_mask
                    if mask.device != selected.device:
                        mask = mask.to(device=selected.device, non_blocking=True)
                    selected = self._apply_keep_mask(selected, mask)
                data[name] = selected
        else:
            if torch.is_tensor(keep_mask) and keep_mask.device.type == "cuda":
                # Avoid dynamic boolean indexing on CUDA which can trigger stream syncs.
                for name in prune_list:
                    value = data.get(name)
                    if value is None:
                        continue
                    if isinstance(value, StreamTensorBase):
                        tensor = unwrap_tensor(value)
                    else:
                        tensor = value
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    data[name] = self._apply_keep_mask(tensor, keep_mask)
            else:
                for name in prune_list:
                    data[name] = data[name][keep_mask]

        self._duration += time.time() - start
        if self._iter_num % 50 == 0:
            fps = self._iter_num / self._duration
            if fps < 50:
                from hmlib.log import get_logger

                get_logger(__name__).info(
                    "Segment Boundary pruning speed: %f fps",
                    self._iter_num / self._duration,
                )
            self._iter_num = 0
            self._duration = 0
        return data

    def prune_detections_static(
        self,
        det_bboxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Prune (bboxes, labels, scores) against the segment mask.

        Returns fixed-shape outputs when `max_detections_in_mask` is set:
          - bboxes_out: (K,4)
          - labels_out: (K,)
          - scores_out: (K,)
          - num_valid: int tensor scalar on-device

        When `max_detections_in_mask` is None, returns masked values (same shape as input)
        and num_valid=None.
        """
        if self._segment_mask is None:
            return det_bboxes, labels, scores, None

        if isinstance(det_bboxes, StreamTensorBase):
            det_bboxes = unwrap_tensor(det_bboxes)
        if isinstance(labels, StreamTensorBase):
            labels = unwrap_tensor(labels)
        if isinstance(scores, StreamTensorBase):
            scores = unwrap_tensor(scores)

        det_bboxes = det_bboxes.to(dtype=torch.float32)
        if labels.dtype != torch.long:
            labels = labels.to(dtype=torch.long)
        if scores.dtype != torch.float32:
            scores = scores.to(dtype=torch.float32)

        keep_mask = self.prune_items_index(batch_item_bboxes=det_bboxes)
        if not isinstance(keep_mask, torch.Tensor):
            keep_mask = torch.as_tensor(keep_mask, device=det_bboxes.device)
        if keep_mask.dtype != torch.bool:
            keep_mask = keep_mask.to(dtype=torch.bool)
        keep_count = keep_mask.to(dtype=torch.int32).sum()

        max_items = self._max_detections_in_mask
        if max_items is None:
            # Keep same-shape tensors to avoid dynamic indexing on CUDA.
            if keep_mask.device.type == "cuda":
                return (
                    self._apply_keep_mask(det_bboxes, keep_mask),
                    self._apply_keep_mask(labels, keep_mask),
                    self._apply_keep_mask(scores, keep_mask),
                    None,
                )
            return det_bboxes[keep_mask], labels[keep_mask], scores[keep_mask], None

        max_items = max(int(max_items), 0)
        N = int(keep_mask.shape[0])
        if N == 0 or max_items == 0:
            out_b = det_bboxes.new_zeros((max_items, 4))
            out_l = labels.new_zeros((max_items,))
            out_s = scores.new_zeros((max_items,))
            num_valid = keep_count.clamp(max=max_items)
            return out_b, out_l, out_s, num_valid

        priority = scores.masked_fill(~keep_mask, float("-inf"))
        k = min(max_items, N)
        order = torch.topk(priority, k=k).indices

        out_b = det_bboxes.index_select(0, order)
        out_l = labels.index_select(0, order)
        out_s = scores.index_select(0, order)

        if out_b.shape[0] < max_items:
            pad_b = out_b.new_zeros((max_items, 4))
            pad_l = out_l.new_zeros((max_items,))
            pad_s = out_s.new_zeros((max_items,))
            pad_b[: out_b.shape[0]].copy_(out_b)
            pad_l[: out_l.shape[0]].copy_(out_l)
            pad_s[: out_s.shape[0]].copy_(out_s)
            out_b, out_l, out_s = pad_b, pad_l, pad_s

        num_valid = keep_count.clamp(max=max_items)
        valid_mask = torch.arange(max_items, device=det_bboxes.device) < num_valid
        out_b = self._apply_keep_mask(out_b, valid_mask)
        out_l = self._apply_keep_mask(out_l, valid_mask)
        out_s = self._apply_keep_mask(out_s, valid_mask)
        return out_b, out_l, out_s, num_valid


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

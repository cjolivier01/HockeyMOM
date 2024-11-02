import glob
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmocr.apis.inferencers import MMOCRInferencer
from torchvision.transforms.functional import normalize

from hmlib.builder import PIPELINES as TRANSFORMS
from hmlib.tracking_utils.utils import xyxy2xywh
from hmlib.ui import show_image
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
)
from hmlib.utils.utils import xyxy2xywh

TV_10_1_ROSTER: Set[int] = {19, 9, 87, 7, 98, 78, 43, 10, 11, 39, 66, 92, 15}
SHARES_12_1_ROSTER: Set[int] = {29, 37, 40, 98, 73, 89, 54, 24, 79, 16, 27, 90, 57, 8, 96, 74}


@TRANSFORMS.register_module()
class HmNumberClassifier:

    def __init__(
        self,
        *args,
        roster: Set[int] = set([*TV_10_1_ROSTER, *SHARES_12_1_ROSTER]),
        init_cfg: Optional[dict] = None,
        category: int = 0,
        enabled: bool = True,
        image_label: str = "img",
        **kwargs,
    ):
        # super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self._roster = roster
        self._category = category
        self._enabled = enabled
        self._image_label = image_label
        self._mean = [0.5, 0.5, 0.5]
        self._std = [0.5, 0.5, 0.5]

    def __call__(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.forward(data, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # typing: none
        if not self._enabled:
            return data
        img = data[self._image_label]
        if isinstance(img, StreamTensor):
            img.verbose = True
            img = img.get()
            # img = img.wait()
            data[self._image_label] = img
        batch_numbers: List[torch.Tensor] = []
        jersey_results: Dict[int, int] = {}
        track_data_sample = data["data_samples"]
        for image_item, data_sample in zip(make_channels_first(img), track_data_sample):
            assert image_item.ndim == 3
            bboxes_xyxy = data_sample.pred_track_instances.bboxes
            tracking_ids = data_sample.pred_track_instances.instances_id
            non_obvelapping_bbox_indices = get_non_overlapping_bbox_indices(bboxes_xyxy)
            non_obvelapping_bboxes_xyxy = bboxes_xyxy[non_obvelapping_bbox_indices]
            if not len(non_obvelapping_bboxes_xyxy):
                continue
            packed_image, index_map = pack_bounding_boxes_as_tiles(
                source_image=image_item, bounding_boxes=bboxes_xyxy.to(torch.int64)
            )
            show_image("packed_image", packed_image, wait=False)
            # subimages = extract_and_resize_jerseys(
            #     image=image_item, bboxes=tlwhs, out_width=54, out_height=54
            # )
            # if subimages is not None:
            #     subimages = make_channels_first(subimages)
            #     subimages = normalize(subimages, mean=self._mean, std=self._std)
            #     results = super().forward(subimages)
            #     indexed_jersey_results = process_results(results, tracking_ids, subimages=subimages)
            #     if indexed_jersey_results:
            #         for index, num_and_score in indexed_jersey_results.items():
            #             num = num_and_score[0]
            #             if self._roster and num not in self._roster:
            #                 continue
            #             score = num_and_score[1]
            #             tid = int(tracking_ids[index])
            #             jersey_results[tid] = num_and_score
            #             # print(
            #             #     f"READ NUMBER: {num}, INDEX NUMBER={index}, TRACKING ID: {tid}, MIN SCORE: {score}"
            #             # )
            #             # show_image("SUBIMAGE", subimages[index], wait=True)
            #         pass
        # batch_numbers.append(jersey_results)
        data["jersey_results"] = jersey_results
        # results["batch_numbers"] = batch_numbers
        return data

    def simple_test(self, data, **kwargs):
        assert False  # huh?
        return data


def extract_and_resize_jerseys(
    image,
    bboxes,
    out_width,
    out_height,
    down_from_box_top_ratio: float = 0.2,
    number_height_from_box_size_ratio: float = 0.25,
    # number_height_from_box_size_ratio: float = 0.2,
):
    """
    Extract and resize sub-images containing likely jersey number areas from given bounding boxes.

    Args:
    - image (torch.Tensor): The image tensor (C, H, W).
    - bboxes (torch.Tensor): Tensor of bounding boxes (N, 4) where each box is (x, y, width, height).
    - out_width (int): The desired output width of the cropped images.
    - out_height (int): The desired output height of the cropped images.

    Returns:
    - torch.Tensor: A batch of cropped and resized images (N, C, out_height, out_width).
    """
    crops = []
    image = make_channels_first(image)
    iw, ih = image_width(image), image_height(image)
    for bbox in bboxes:
        x, y, width, height = bbox

        # Calculate new coordinates for the jersey number area
        new_y = int(y + down_from_box_top_ratio * height)
        new_height = int(number_height_from_box_size_ratio * height)
        # new_y = y
        # new_height = int(height)

        new_width = width
        # new_width = int(width * 0.5)
        # new_width = min(width, new_height)
        # new_x = int(x + new_width / 2)
        new_x = int(x + (width - new_width) / 2)

        # Ensure the new box does not exceed image dimensions
        new_y = max(new_y, 0)
        new_height = min(new_height, ih - new_y)

        # Crop the image
        cropped = image[:, new_y : new_y + new_height, new_x : new_x + int(new_width)]
        # cropped = image[:, y : y + height, x : x + width]

        # show_image("crop", cropped, wait=False)
        # show_image("crop", cropped, wait=True)

        # Resize the cropped image
        resized = F.interpolate(
            cropped.unsqueeze(0), size=(out_height, out_width), mode="bilinear", align_corners=False
        ).squeeze(0)
        show_image("number?", resized, wait=True)
        # outimg = make_channels_last(resized).cpu().numpy()
        # did = cv2.imwrite("test-19.png", outimg)
        crops.append(resized)

    batch_crops = torch.stack(crops, dim=0)

    return batch_crops


import math


def get_non_overlapping_bbox_indices(boxes: torch.Tensor) -> torch.Tensor:
    # boxes should be a tensor of shape [N, 4] representing [x1, y1, x2, y2] for each bounding box
    N: int = boxes.size(0)
    batch_indices: torch.Tensor = torch.arange(N, device=boxes.device)

    # Compute pairwise intersection between boxes
    x1: torch.Tensor = torch.max(boxes[:, None, 0], boxes[:, 0])
    y1: torch.Tensor = torch.max(boxes[:, None, 1], boxes[:, 1])
    x2: torch.Tensor = torch.min(boxes[:, None, 2], boxes[:, 2])
    y2: torch.Tensor = torch.min(boxes[:, None, 3], boxes[:, 3])

    # Compute intersection width and height
    inter_width: torch.Tensor = (x2 - x1).clamp(min=0)
    inter_height: torch.Tensor = (y2 - y1).clamp(min=0)
    intersection: torch.Tensor = inter_width * inter_height

    # Compute overlap matrix (N x N) and set diagonal to 0 (to ignore self-overlap)
    overlap: torch.Tensor = (intersection > 0).fill_diagonal_(False)

    # Get non-overlapping indices
    non_overlapping: torch.Tensor = overlap.sum(dim=1) == 0
    non_overlapping_indices: torch.Tensor = batch_indices[non_overlapping]

    return non_overlapping_indices


def pack_bounding_boxes_as_tiles(
    source_image: torch.Tensor, bounding_boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Assume bounding_boxes is of shape (N, 4), where:
    # N: number of bounding boxes
    # 4: coordinates of the bounding box in (x1, y1, x2, y2) format
    # Assume source_image is of shape (3, H, W), where:
    # 3: number of channels (RGB)
    # H, W: height and width of the source image

    N, _ = bounding_boxes.shape

    # Sort bounding boxes by height (descending) for better packing
    heights = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    _, sorted_indices = torch.sort(heights, descending=True)
    bounding_boxes = bounding_boxes[sorted_indices]

    # Initialize packed image dimensions with a roughly 16:9 aspect ratio
    widths = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    total_area = torch.sum(widths * heights)
    target_height = int(torch.sqrt(total_area / (16 / 9)).item())
    target_width = int(target_height * (16 / 9))

    packed_image = (
        torch.zeros((3, target_height, target_width), dtype=source_image.dtype) + 128
    )  # Assume 3 channels for RGB
    index_map = -torch.ones(
        (target_height, target_width), dtype=torch.long
    )  # Map to store indices of original boxes

    # Start packing from the top-left corner
    current_x, current_y = 0, 0
    max_row_height = 0

    for idx in range(N):
        x1, y1, x2, y2 = bounding_boxes[idx]
        w = x2 - x1
        h = y2 - y1
        cropped_region = source_image[:, y1:y2, x1:x2]

        # Check if the bounding box fits in the current row, otherwise move to next row
        if current_x + w > target_width:
            current_x = 0
            current_y += max_row_height
            max_row_height = 0

        # If the current height exceeds the canvas, resize the canvas vertically
        if current_y + h > target_height:
            new_height = current_y + h
            packed_image = torch.nn.functional.pad(
                packed_image, (0, 0, 0, new_height - target_height)
            )
            index_map = torch.nn.functional.pad(
                index_map, (0, 0, 0, new_height - target_height), value=-1
            )
            target_height = new_height

        # Place the cropped region in the packed image
        packed_image[:, current_y : current_y + h, current_x : current_x + w] = cropped_region
        # Update the index map to indicate which bounding box this region came from
        index_map[current_y : current_y + h, current_x : current_x + w] = sorted_indices[idx]

        # Update current position and max row height
        current_x += w
        max_row_height = max(max_row_height, h)

    return packed_image, index_map


def get_original_bbox_index(index_map: torch.Tensor, y: int, x: int) -> int:
    # Given a point (y, x), return the original bounding box index
    return index_map[y, x]


def sample():
    # Example usage
    source_image = torch.randn((3, 256, 256))  # Random source image of shape (3, 256, 256)
    N = 10  # 10 bounding boxes
    bounding_boxes = torch.randint(
        0, 200, (N, 4)
    )  # Random bounding boxes in (x, y, width, height) format

    packed_image, index_map = pack_bounding_boxes_as_tiles(source_image, bounding_boxes)

    # To get the original bounding box index from a point (y, x):
    y, x = 45, 60
    original_index = get_original_bbox_index(index_map, y, x)
    print(f"The point ({y}, {x}) belongs to the original bounding box index: {original_index}")


# Example usage:
# Assuming 'image_tensor' is a CxHxW tensor and 'bounding_boxes' is a Nx4 tensor
# 'desired_width' and 'desired_height' are the target dimensions for each crop
# cropped_images = extract_and_resize_jerseys(image_tensor, bounding_boxes, desired_width, desired_height)

# ARGS: List[str] = [
#     "/olivier-pool/Videos/ev-tv-10-1-2/test_numbers.png",
#     "--out-dir=/home/colivier/src/openmm/results",
#     "--rec=mmocr/configs/textrecog/abinet/abinet-vision_20e_st-an_mj.py",
#     "--rec-weights=https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth",
#     "--det=mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py",
#     "--det-weights=https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth",
#     "--kie=mmocr/configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py",
#     "--kie-weights=https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt/sdmgr_unet16_60e_wildreceipt_20220825_151648-22419f37.pth",
#     "--show",
# ]


def get_inferencer() -> MMOCRInferencer:
    from mmocr.apis.inferencers import MMOCRInferencer

    config = {
        "det": "FCENet",
        "det_weights": None,
        "rec": "/home/colivier/src/hm/openmm/mmocr/configs/textrecog/svtr/svtr-small_20e_st_mj.py",
        "rec_weights": "https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-small_20e_st_mj/svtr-small_20e_st_mj-35d800d6.pth",
        "kie": None,
        "kie_weights": None,
        "device": "cuda",
    }

    # inferencer = MMOCRInferencer(
    #     det="openmm/mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py",
    #     det_weights="https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth",
    #     rec="openmm/mmocr/configs/textrecog/abinet/abinet-vision_20e_st-an_mj.py",
    #     rec_weights="https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth",
    #     kie="openmm/mmocr/configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py",
    #     kie_weights="https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt/sdmgr_unet16_60e_wildreceipt_20220825_151648-22419f37.pth",
    # )
    # return inferencer


def main():
    inferencer = get_inferencer()


if __name__ == "__main__":
    # main()
    print("Done.")

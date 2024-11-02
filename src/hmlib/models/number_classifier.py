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
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
)


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
        # tracking_data = data["category_bboxes"][self._category]
        # if not tracking_data.shape[0]:
        #     return data
        # bboxes = tracking_data[:, 1:5].astype(np.int)
        # tracking_ids = tracking_data[:, 0].astype(np.int64)
        # tlwhs = xyxy2xywh(bboxes)
        img = data[self._image_label]
        if isinstance(img, StreamTensor):
            img.verbose = True
            img = img.get()
            # img = img.wait()
            data[self._image_label] = img
        assert len(img) == 1
        batch_numbers: List[torch.Tensor] = []
        jersey_results: Dict[int, int] = {}
        for image_item in make_channels_first(img):
            assert image_item.ndim == 3
            subimages = extract_and_resize_jerseys(
                image=image_item, bboxes=tlwhs, out_width=54, out_height=54
            )
            if subimages is not None:
                subimages = make_channels_first(subimages)
                subimages = normalize(subimages, mean=self._mean, std=self._std)
                results = super().forward(subimages)
                indexed_jersey_results = process_results(results, tracking_ids, subimages=subimages)
                if indexed_jersey_results:
                    for index, num_and_score in indexed_jersey_results.items():
                        num = num_and_score[0]
                        if self._roster and num not in self._roster:
                            continue
                        score = num_and_score[1]
                        tid = int(tracking_ids[index])
                        jersey_results[tid] = num_and_score
                        # print(
                        #     f"READ NUMBER: {num}, INDEX NUMBER={index}, TRACKING ID: {tid}, MIN SCORE: {score}"
                        # )
                        # show_image("SUBIMAGE", subimages[index], wait=True)
                    pass
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
        # show_image("number?", resized, wait=True)
        # outimg = make_channels_last(resized).cpu().numpy()
        # did = cv2.imwrite("test-19.png", outimg)
        crops.append(resized)

    batch_crops = torch.stack(crops, dim=0)

    return batch_crops


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

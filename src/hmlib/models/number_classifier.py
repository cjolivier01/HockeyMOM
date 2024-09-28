import glob
import os
from typing import Any, Dict, List, Optional

import cv2

# import torch.jit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from torchvision.transforms.functional import normalize

from hmlib.stitching.laplacian_blend import show_image
from hmlib.tracking_utils.utils import xyxy2xywh
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
)

from ..builder import NECKS

# from xmodels.SVHNClassifier.model import SVHNClassifier as SVHNClassifier


class SVHNClassifier(BaseModule):
    CHECKPOINT_FILENAME_PATTERN = "model-{}.pth"

    __constants__ = [
        "_hidden1",
        "_hidden2",
        "_hidden3",
        "_hidden4",
        "_hidden5",
        "_hidden6",
        "_hidden7",
        "_hidden8",
        "_hidden9",
        "_hidden10",
        "_features",
        "_classifier",
        "_digit_length",
        "_digit1",
        "_digit2",
        "_digit3",
        "_digit4",
        "_digit5",
    ]

    def __init__(self, *args, **kwargs):
        super(SVHNClassifier, self).__init__(*args, **kwargs)

        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        self._hidden9 = nn.Sequential(nn.Linear(192 * 7 * 7, 3072), nn.ReLU())
        self._hidden10 = nn.Sequential(nn.Linear(3072, 3072), nn.ReLU())

        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

    # @torch.jit.script_method
    def forward(self, x):
        x = self._hidden1(x)
        x = self._hidden2(x)
        x = self._hidden3(x)
        x = self._hidden4(x)
        x = self._hidden5(x)
        x = self._hidden6(x)
        x = self._hidden7(x)
        x = self._hidden8(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._hidden9(x)
        x = self._hidden10(x)

        length_logits = self._digit_length(x)
        digit1_logits = self._digit1(x)
        digit2_logits = self._digit2(x)
        digit3_logits = self._digit3(x)
        digit4_logits = self._digit4(x)
        digit5_logits = self._digit5(x)

        return (
            length_logits,
            digit1_logits,
            digit2_logits,
            digit3_logits,
            digit4_logits,
            digit5_logits,
        )

    def store(self, path_to_dir, step, maximum=5):
        path_to_models = glob.glob(
            os.path.join(path_to_dir, SVHNClassifier.CHECKPOINT_FILENAME_PATTERN.format("*"))
        )
        if len(path_to_models) == maximum:
            min_step = min(
                [int(path_to_model.split("/")[-1][6:-4]) for path_to_model in path_to_models]
            )
            path_to_min_step_model = os.path.join(
                path_to_dir, SVHNClassifier.CHECKPOINT_FILENAME_PATTERN.format(min_step)
            )
            os.remove(path_to_min_step_model)

        path_to_checkpoint_file = os.path.join(
            path_to_dir, SVHNClassifier.CHECKPOINT_FILENAME_PATTERN.format(step)
        )
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = int(path_to_checkpoint_file.split("/")[-1][6:-4])
        return step


@NECKS.register_module()
class HmNumberClassifier(SVHNClassifier):

    def __init__(
        self,
        *args,
        init_cfg: Optional[dict] = None,
        category: int = 0,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self._category = category
        self._enabled = enabled
        self._mean = [0.5, 0.5, 0.5]
        self._std = [0.5, 0.5, 0.5]

    # @auto_fp16(apply_to=("img",))
    def forward(self, data: Dict[str, Any], **kwargs):  # typing: none
        if not self._enabled:
            return data
        tracking_data = data["category_bboxes"][self._category]
        if not tracking_data.shape[0]:
            return data
        bboxes = tracking_data[:, 1:5].astype(np.int)
        tracking_ids = tracking_data[:, 0].astype(np.int64)
        tlwhs = xyxy2xywh(bboxes)
        img = data["img"]
        if isinstance(img, StreamTensor):
            img = img.get()
            # img = img.wait()
            data["img"] = img
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
                indexed_jersey_results = process_results(results)
                if indexed_jersey_results:
                    for index, num_and_score in indexed_jersey_results.items():
                        tid = int(tracking_ids[index])
                        jersey_results[tid] = num_and_score
                        print(
                            f"READ NUMBER: {num_and_score[0]}, INDEX NUMBER={index}, TRACKING ID: {tid}, MIN SCORE: {num_and_score[1]}"
                        )
                        # show_image("SUBIMAGE", subimages[index], wait=True)
                    pass
        # batch_numbers.append(jersey_results)
        data["jersey_results"] = jersey_results
        # results["batch_numbers"] = batch_numbers
        return data

    def simple_test(self, data, **kwargs):
        assert False  # huh?
        return data


def process_results(number_results: np.ndarray, min_score=15, largest_number=99) -> Dict[int, int]:
    (
        batch_length_logits,
        batch_digit1_logits,
        batch_digit2_logits,
        batch_digit3_logits,
        batch_digit4_logits,
        batch_digit5_logits,
    ) = number_results

    jersey_results: Dict[int, int] = {}
    for batch_index in range(len(batch_length_logits)):
        length_logits = batch_length_logits[batch_index].unsqueeze(0)
        digit1_logits = batch_digit1_logits[batch_index].unsqueeze(0)
        digit2_logits = batch_digit2_logits[batch_index].unsqueeze(0)
        digit3_logits = batch_digit3_logits[batch_index].unsqueeze(0)
        digit4_logits = batch_digit4_logits[batch_index].unsqueeze(0)
        digit5_logits = batch_digit5_logits[batch_index].unsqueeze(0)

        length_value, length_prediction = length_logits.max(1)
        digit1_value, digit1_prediction = digit1_logits.max(1)
        digit2_value, digit2_prediction = digit2_logits.max(1)
        digit3_value, digit3_prediction = digit3_logits.max(1)
        digit4_value, digit4_prediction = digit4_logits.max(1)
        digit5_value, digit5_prediction = digit5_logits.max(1)

        scores = torch.cat(
            [
                length_value,
                digit1_value,
                digit2_value,
                digit3_value,
                digit4_value,
                digit5_value,
            ],
            dim=0,
        )
        scores = scores[: int(length_prediction + 1)]
        this_min_score = torch.min(scores)
        # bad = False
        # for x in range(length_prediction + 1):
        #     if scores[x] < min_score:
        #         bad = True
        # if bad:
        #     continue
        if this_min_score < min_score:
            continue

        # print("length:", length_prediction.item(), "value:", length_value.item())
        # print(
        #     "digits:",
        #     digit1_prediction.item(),
        #     digit2_prediction.item(),
        #     digit3_prediction.item(),
        #     digit4_prediction.item(),
        #     digit5_prediction.item(),
        # )
        # print(
        #     "values:",
        #     digit1_value.item(),
        #     digit2_value.item(),
        #     digit3_value.item(),
        #     digit4_value.item(),
        #     digit5_value.item(),
        # )
        all_digits = [
            digit1_prediction.item(),
            digit2_prediction.item(),
            digit3_prediction.item(),
            digit4_prediction.item(),
            digit5_prediction.item(),
        ]
        running = 0
        for i in range(length_prediction.item()):
            running *= 10
            running += all_digits[i]
        if running <= largest_number:
            print(f"Final prediction: {running}")
            jersey_results[batch_index] = (running, float(this_min_score))
        else:
            print(f"Bad number: {running}")
    # if jersey_results:
    #     print(f"Found {len(jersey_results)} good numbers")
    return jersey_results


def extract_and_resize_jerseys(
    image,
    bboxes,
    out_width,
    out_height,
    down_from_box_top_ratio: float = 0.2,
    number_height_from_box_size_ratio: float = 0.25,
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

        # new_width = width * 0.5
        new_width = min(width, new_height)
        new_x = int(x + new_width / 2)

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

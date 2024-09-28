import glob
import os
from typing import Any, List, Optional

import torch
import torch.jit
import numpy as np
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

# from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmtrack.core import outs2results, results2outs

# from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from hmlib.tracking_utils.utils import xyxy2xywh
from hmlib.utils.image import make_channels_first, make_channels_last

# from xmodels.SVHNClassifier.model import SVHNClassifier as SVHNClassifier

from ..builder import NECKS


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

    # def __call__(self, *args, **kwargs):
    #     return super().__call__(*args, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, data, **kwargs):
        if not self._enabled:
            return None
        img = data["img"]
        tracking_data = data["category_bboxes"][self._category]
        if not tracking_data.shape[0]:
            return None
        bboxes = tracking_data[:, 1:5].astype(np.int)
        tlwhs = xyxy2xywh(bboxes)
        assert img.size(0) == 1
        img = make_channels_first(img.squeeze(0))
        subimages = extract_and_resize_jerseys(image=img, bboxes=tlwhs, out_width=64, out_height=64)
        results = super().forward(subimages)
        return results

    def simple_test(self, data, **kwargs):
        assert False  # huh?
        return data


import torch
import torch.nn.functional as F


def extract_and_resize_jerseys(image, bboxes, out_width, out_height):
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
    for bbox in bboxes:
        x, y, width, height = bbox

        # Calculate new coordinates for the jersey number area
        new_y = int(y + 0.35 * height)
        new_height = int(0.55 * height)
        new_width = width

        # Ensure the new box does not exceed image dimensions
        new_y = max(new_y, 0)
        new_height = min(new_height, image.size(1) - new_y)

        # Crop the image
        cropped = image[:, new_y : new_y + new_height, x : x + new_width]

        # Resize the cropped image
        resized = F.interpolate(
            cropped.unsqueeze(0), size=(out_height, out_width), mode="bilinear", align_corners=False
        )
        crops.append(resized)

    # Concatenate all cropped images into a batch
    batch_crops = torch.cat(crops, dim=0)

    return batch_crops

    # Example usage:
    # Assuming 'image_tensor' is a CxHxW tensor and 'bounding_boxes' is a Nx4 tensor
    # 'desired_width' and 'desired_height' are the target dimensions for each crop
    # cropped_images = extract_and_resize_jerseys(image_tensor, bounding_boxes, desired_width, desired_height)

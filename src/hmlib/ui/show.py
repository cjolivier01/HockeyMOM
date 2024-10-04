from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL.Image import Image

from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_visible_image


def show_image(
    label: str,
    img: Union[str, torch.Tensor],
    wait: bool = True,
    enable_resizing: Union[bool, None] = None,
):
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, Image):
        img = np.array(img)
    elif isinstance(img, StreamTensor):
        img = img.get()
    if img.ndim == 2:
        # grayscale
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    if img.ndim == 4:
        for i in img:
            cv2.imshow(
                label,
                make_visible_image(
                    i,
                    # scale_elements=255.0,
                    enable_resizing=enable_resizing,
                ),
            )
            cv2.waitKey(1 if not wait else 0)
    else:
        cv2.imshow(
            label,
            make_visible_image(
                img,
                # scale_elements=255.0,
                enable_resizing=enable_resizing,
            ),
        )
        cv2.waitKey(1 if not wait else 0)

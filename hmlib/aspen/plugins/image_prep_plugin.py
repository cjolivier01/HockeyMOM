from typing import Any, Dict, Optional

import torch

from hmlib.utils import MeanTracker
from hmlib.utils.gpu import copy_gpu_to_gpu_async, unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_first

from .base import Plugin


class ImagePrepPlugin(Plugin):
    """
    Prepares detection input tensor for MMTracking-like models.

    Expects in context:
      - data: dict with key 'img' (B, H, W, C or similar)
      - original_images: original images tensor
      - device: torch.device
      - cuda_stream: Optional[torch.cuda.Stream]
      - mean_tracker: Optional[MeanTracker]

    Produces in context:
      - detection_image: Tensor of shape (1, T, C, H, W) on device
      - data: updates 'img' to detection_image and sets 'original_images'
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        # Ensure original_images are available in data for downstream trunks
        original_images = context.get("original_images")
        if original_images is not None and "original_images" not in data:
            data["original_images"] = original_images
        device: torch.device = context["device"]

        mean_tracker: Optional[MeanTracker] = context.get("mean_tracker")

        detection_image = data["img"]

        if detection_image.device != device:
            prev_stream = torch.cuda.current_stream(detection_image.device)
            detection_image = unwrap_tensor(data["img"])
            detection_image = make_channels_first(detection_image)
            detection_image, _ = copy_gpu_to_gpu_async(tensor=detection_image, dest_device=device)
            prev_stream.wait_stream(torch.cuda.current_stream(device))
            data["img"] = wrap_tensor(detection_image)

        if mean_tracker is not None:
            data["img"] = wrap_tensor(mean_tracker.forward(unwrap_tensor(data["img"])))

        return {"detection_image": data["img"], "data": data}

    def input_keys(self):
        return {"data", "device", "mean_tracker", "original_images"}

    def output_keys(self):
        return {"data", "detection_image"}

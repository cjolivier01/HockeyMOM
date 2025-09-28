from typing import Any, Dict, Optional

import torch

from hmlib.utils import MeanTracker
from hmlib.utils.gpu import StreamTensor, copy_gpu_to_gpu_async
from hmlib.utils.image import make_channels_first

from .base import Trunk


class ImagePrepTrunk(Trunk):
    """
    Prepares detection input tensor for MMTracking-like models.

    Expects in context:
      - data: dict with key 'img' (B, H, W, C or similar)
      - origin_imgs: original images tensor
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
        origin_imgs: torch.Tensor = context["origin_imgs"]
        device: torch.device = context["device"]
        cuda_stream: Optional[torch.cuda.Stream] = context.get("cuda_stream")
        mean_tracker: Optional[MeanTracker] = context.get("mean_tracker")

        detection_image = data["img"]
        detection_image = make_channels_first(detection_image)
        if isinstance(detection_image, StreamTensor):
            detection_image.verbose = True
            detection_image = detection_image.wait(cuda_stream)

        if detection_image.device != device:
            detection_image, _ = copy_gpu_to_gpu_async(tensor=detection_image, dest_device=device)

        # Batch=1, frames=T
        detection_image = detection_image.unsqueeze(0)
        assert detection_image.ndim == 5

        if mean_tracker is not None:
            detection_image = mean_tracker.forward(detection_image)

        if "original_images" not in data:
            data["original_images"] = origin_imgs

        data["img"] = detection_image
        return {"detection_image": detection_image, "data": data}


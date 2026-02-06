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
      - inputs: model input tensor (T, H, W, C) or (T, C, H, W)
      - original_images: original images tensor (optional)
      - device: torch.device
      - cuda_stream: Optional[torch.cuda.Stream]
      - mean_tracker: Optional[MeanTracker]

    Produces in context:
      - inputs: Tensor of shape (T, C, H, W) on device
      - detection_image: alias of `inputs` for backward compatibility/debug
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        device: torch.device = context["device"]

        mean_tracker: Optional[MeanTracker] = context.get("mean_tracker")

        inputs_any = context.get("inputs")
        if inputs_any is None:
            return {}
        detection_image = inputs_any

        detection_image = unwrap_tensor(detection_image)
        detection_image = make_channels_first(detection_image)

        if isinstance(detection_image, torch.Tensor) and detection_image.device != device:
            if detection_image.is_cuda and device.type == "cuda":
                prev_stream = torch.cuda.current_stream(detection_image.device)
                detection_image, _ = copy_gpu_to_gpu_async(
                    tensor=detection_image, dest_device=device
                )
                prev_stream.wait_stream(torch.cuda.current_stream(device))
            else:
                detection_image = detection_image.to(device=device, non_blocking=True)

        if mean_tracker is not None:
            detection_image = mean_tracker.forward(detection_image)

        wrapped = wrap_tensor(detection_image)
        return {"inputs": wrapped, "detection_image": wrapped}

    def input_keys(self):
        return {"inputs", "device", "mean_tracker"}

    def output_keys(self):
        return {"inputs", "detection_image"}

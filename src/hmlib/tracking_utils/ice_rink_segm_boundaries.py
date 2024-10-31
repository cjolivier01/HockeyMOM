from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from hmlib.builder import PIPELINES
from hmlib.config import get_nested_value
from hmlib.segm.ice_rink import confgure_ice_rink_mask, get_device_to_use_for_rink
from hmlib.utils.gpu import GpuAllocator

from .segm_boundaries import SegmBoundaries


@PIPELINES.register_module()
class IceRinkSegmConfig:

    def __init__(
        self,
        *args,
        game_id: str = None,
        device: torch.device = torch.device("cpu"),
        shape_label: str = "ori_shape",
        image_label: str = "original_images",
        **kwargs,
    ):
        self._game_id = game_id
        self._rink_profile = None
        self._device = device
        self._shape_label: str = shape_label
        self._image_label = image_label

    def maybe_init_rink_segmentation(self, data: Dict[str, Any]):
        if self._rink_profile is None:
            expected_width_height: Optional[Tuple[int, int]] = None
            if self._shape_label and self._shape_label in data:
                image_shape = get_nested_value(data, self._shape_label)[0]
                assert isinstance(image_shape, torch.Size)
            image = data[self._image_label]
            assert isinstance(image, torch.Tensor)
            batch_size = len(image)
            rink_profile = confgure_ice_rink_mask(
                game_id=self._game_id,
                device=self._device,
                expected_shape=image_shape,
                image=image[0],
            )
            self._rink_profile = [rink_profile for _ in range(batch_size)]
        return data

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.forward(*args, **kwargs)

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if "rink_profile" not in data:
            data = self.maybe_init_rink_segmentation(data)
            data["rink_profile"] = self._rink_profile
        return data


@PIPELINES.register_module()
class IceRinkSegmBoundaries(SegmBoundaries):

    def __init__(
        self,
        *args,
        game_id: str = None,
        original_clip_box: Optional[Union[torch.Tensor, List[int]]] = None,
        det_thresh: float = 0.05,
        draw: bool = False,
        device: torch.device = torch.device("cpu"),
        shape_label: str = "ori_shape",
        **kwargs,
    ):
        super().__init__(
            *args, original_clip_box=original_clip_box, det_thresh=det_thresh, draw=draw, **kwargs
        )
        self._game_id = game_id
        # self._rink_profile = None
        self._device = device
        self._shape_label: str = shape_label

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self._rink_mask is None:
            metainfo = data["data_samples"].video_data_samples[0].metainfo
            self._rink_profile = metainfo.get("rink_profile")
            if self._rink_profile is not None:
                self.set_rink_mask_and_centroid(
                    rink_mask=self._rink_profile["combined_mask"],
                    centroid=self._rink_profile["centroid"],
                )
        return super().forward(data, **kwargs)

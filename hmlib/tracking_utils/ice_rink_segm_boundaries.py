from typing import Any, Dict, List, Optional, Union

import torch

from hmlib.builder import PIPELINES
from hmlib.config import get_nested_value
from hmlib.segm.ice_rink import confgure_ice_rink_mask

from .segm_boundaries import SegmBoundaries


@PIPELINES.register_module()
class IceRinkSegmConfig(torch.nn.Module):
    def __init__(
        self,
        *args,
        game_id: str = None,
        device: torch.device = torch.device("cpu"),
        shape_label: str = "ori_shape",
        image_label: str = "original_images",
        clip: bool = False,
        ice_rink_inference_scale: Optional[float] = None,
        **kwargs,
    ):
        self._game_id = game_id
        self._rink_profile = None
        self._device = device
        self._shape_label: str = shape_label
        self._image_label = image_label
        self._clip = clip
        self._clip_box = None
        self._clipped_shape = None
        self._ice_rink_inference_scale = ice_rink_inference_scale

    def maybe_init_rink_segmentation(self, data: Dict[str, Any]):
        if self._rink_profile is None:
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
                scale=self._ice_rink_inference_scale,
            )
            if self._clip:
                bbox = rink_profile["combined_bbox"]
                self._clip_box = [int(i) for i in bbox]
                self._clipped_shape = torch.Size(
                    (self._clip_box[3] - self._clip_box[1], self._clip_box[2] - self._clip_box[0])
                )
            self._rink_profile = [rink_profile for _ in range(batch_size)]
        return data

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return super().__call__(*args, **kwargs)

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if "rink_profile" not in data:
            data = self.maybe_init_rink_segmentation(data)
            data["rink_profile"] = self._rink_profile
        if self._clipped_shape and self._image_label:
            img = data[self._image_label]
            # we expect channels last
            img = img[
                :, self._clip_box[1] : self._clip_box[3], self._clip_box[0] : self._clip_box[2], :
            ]
            bs = len(img)
            data[self._image_label] = img
            data[self._shape_label] = [self._clipped_shape for _ in range(bs)]
            if "img_shape" in data:
                data["img_shape"] = [self._clipped_shape for _ in range(bs)]
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
            *args,
            original_clip_box=original_clip_box,
            det_thresh=det_thresh,
            max_detections_in_mask=kwargs.pop("max_detections_in_mask", None),
            draw=draw,
            **kwargs,
        )
        self._game_id = game_id
        # self._rink_profile = None
        self._device = device
        self._shape_label: str = shape_label

    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self._segment_mask is None:
            self._rink_profile = data.get("rink_profile")
            if self._rink_profile is not None:
                self.set_segment_mask_and_centroid(
                    segment_mask=self._rink_profile["combined_mask"],
                    centroid=self._rink_profile["centroid"],
                )
        data = super().forward(data, **kwargs)

        return data

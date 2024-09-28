from typing import List, Optional, Union

import torch

from hmlib.builder import PIPELINES
from hmlib.segm.ice_rink import confgure_ice_rink_mask, get_device_to_use_for_rink
from hmlib.utils.gpu import GpuAllocator

from .segm_boundaries import SegmBoundaries


@PIPELINES.register_module()
class IceRinkSegmBoundaries(SegmBoundaries):

    def __init__(
        self,
        *args,
        game_id: str = None,
        original_clip_box: Optional[Union[torch.Tensor, List[int]]] = None,
        det_thresh: float = 0.05,
        draw: bool = False,
        gpu_allocator: Optional[GpuAllocator] = None,
        **kwargs,
    ):
        super().__init__(
            *args, original_clip_box=original_clip_box, det_thresh=det_thresh, draw=draw, **kwargs
        )
        self._game_id = game_id
        self._rink_profile = None
        self._gpu_allocator = gpu_allocator

    def maybe_init_rink_segmentation(self):
        if self._rink_profile is None:
            self._rink_profile = confgure_ice_rink_mask(
                game_id=self._game_id,
                device=get_device_to_use_for_rink(self._gpu_allocator),
            )
            self.set_rink_mask_and_centroid(
                rink_mask=self._rink_profile["combined_mask"],
                centroid=self._rink_profile["centroid"],
            )

    def forward(self, data, **kwargs):
        self.maybe_init_rink_segmentation()
        return super().forward(data, **kwargs)

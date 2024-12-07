from typing import Tuple

import torch

from hmlib.tracking_utils import visualization as vis
from hockeymom.core import AllLivingBoxConfig, BBox, LivingBox, WHDims


class PyLivingBox(LivingBox):

    def __init__(self, *args, color: Tuple[int, int, int], thickness: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._color = color
        self._thickness = thickness

    def _draw_resizing_state(self, img: torch.Tensor):
        rstate = self.resizing_state()
        return img

    def _draw_translation_state(self, img: torch.Tensor):
        rstate = self.translation_state()
        draw_box = self.bounding_box()
        img = vis.plot_rectangle(
            img,
            draw_box,
            color=self._color if not rstate.translation_is_frozen else (128, 128, 128),
            thickness=self._thickness,
            label=self.name(),
            text_scale=2,
        )
        return img

    def draw(self, img: torch.Tensor) -> torch.Tensor:
        img = self._draw_resizing_state(img)
        img = self._draw_translation_state(img)
        return img

    def get_size_scale(self) -> Tuple[float, float]:
        size_scale: WHDims = super().get_size_scale()
        return size_scale.width, size_scale.height

import torch

from hockeymom.core import AllLivingBoxConfig, BBox, LivingBox


class PyLivingBox(LivingBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _draw_resizing_state(self, img: torch.Tensor):
        rstate = self.resizing_state()
        return img

    def _draw_translation_state(self, img: torch.Tensor):
        rstate = self.translation_state()
        return img

    def draw(self, img: torch.Tensor) -> torch.Tensor:
        img = self._draw_resizing_state(img)
        img = self._draw_translation_state(img)
        return img

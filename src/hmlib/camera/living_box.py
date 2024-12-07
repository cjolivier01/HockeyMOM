import torch

from hockeymom.core import AllLivingBoxConfig, BBox, LivingBox

class PyLivingBox(LivingBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def draw(self, torch.Tensor) -> torch.Tensor:
        pass

import math
from typing import Optional, Union

import torch


class ImageHorizontalGaussianDistribution:
    def __init__(
        self,
        width: int,
        invert: bool = True,
        show: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self._invert = invert
        self._show = show
        if not isinstance(width, int):
            raise TypeError("width must be provided as a Python int to avoid implicit device syncs")

        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self._dtype = dtype
        self._width = width
        self.setup_gaussian(length=width)

    @property
    def width(self) -> int:
        return self._width

    def get_gaussian_y_from_image_x_position(
        self, xpos: Union[int, torch.Tensor], wide: bool = False
    ):
        xpos_tensor = self._to_index_tensor(xpos)
        if wide:
            return self.gaussian_wide[xpos_tensor]
        return self.gaussian_y[xpos_tensor]

    def setup_gaussian(self, length: int):
        if not isinstance(length, int):
            raise TypeError("length must be an int to avoid implicit device syncs")

        x = torch.linspace(
            -length / 2,
            length / 2,
            steps=length + 1,
            device=self._device,
            dtype=self._dtype,
        )

        narrow_std_dev = float(length) / 8.0
        self.gaussian_y = self._normalize_tensor(self._gaussian_curve(x=x, std_dev=narrow_std_dev))

        wide_std_dev = float(length)
        self.gaussian_wide = self._normalize_tensor(self._gaussian_curve(x=x, std_dev=wide_std_dev))

        if self._invert:
            self.gaussian_y = torch.ones_like(self.gaussian_y) - self.gaussian_y
            self.gaussian_wide = torch.ones_like(self.gaussian_wide) - self.gaussian_wide

        if self._show:
            if x.device.type != "cpu":
                raise RuntimeError(
                    "Plotting the Gaussian curve requires CPU tensors to avoid host transfers."
                )

            # Late import since this usually isn't used
            import time

            import matplotlib.pyplot as plt

            plt.plot(
                x.cpu().numpy(), (torch.ones_like(self.gaussian_wide) - self.gaussian_wide).numpy()
            )
            plt.title("Gaussian Curve")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            time.sleep(1000)

    def _gaussian_curve(self, x: torch.Tensor, std_dev: float) -> torch.Tensor:
        norm = 1.0 / (std_dev * math.sqrt(2 * math.pi))
        mean = 1.0
        return norm * torch.exp(-((x - mean) ** 2) / (2 * std_dev**2)) * 1000.0

    @staticmethod
    def _normalize_tensor(values: torch.Tensor) -> torch.Tensor:
        min_val = torch.amin(values)
        max_val = torch.amax(values)
        denom = torch.clamp(max_val - min_val, min=torch.finfo(values.dtype).eps)
        return (values - min_val) / denom

    def _to_index_tensor(self, value: Union[int, torch.Tensor]) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self._device)
        tensor = torch.clamp(tensor, 0, self._width - 1)
        return tensor.long()

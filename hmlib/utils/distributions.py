from typing import Union

import numpy as np
import torch


class ImageHorizontalGaussianDistribution:
    def __init__(self, width: Union[int, torch.Tensor], invert: bool = True, show: bool = False):
        self._invert = invert
        self._show = show
        self._width = int(width)
        self.setup_gaussian(length=width)

    @property
    def width(self) -> int:
        return self._width

    def get_gaussian_y_from_image_x_position(self, xpos: int, wide: bool = False):
        if xpos < 0:
            xpos = 0
        elif xpos >= self._width:
            xpos = self._width - 1
        xpos = int(xpos)
        if not wide:
            # Steep curve
            return self.gaussian_y[xpos]
        # Much wider curve
        return self.gaussian_wide[xpos]

    def setup_gaussian(self, length: Union[int, torch.Tensor]):
        # Thinner gaussian
        if isinstance(length, torch.Tensor):
            length = int(length.trunc().item())
        else:
            length = int(length)
        std_dev = float(length) / 8.0
        mean = 1.0
        x = np.linspace(-length / 2, length / 2, length + 1)
        self.gaussian_y = (
            (1 / (std_dev * np.sqrt(2 * np.pi)))
            * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))
            * 1000
        )

        # Whiten the data
        min = np.min(self.gaussian_y)
        max = np.max(self.gaussian_y)
        self.gaussian_y = self.gaussian_y - min
        self.gaussian_y = self.gaussian_y / (max - min)

        # Wide gaussian
        std_dev = float(length)
        self.gaussian_wide = (
            (1 / (std_dev * np.sqrt(2 * np.pi)))
            * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))
            * 1000
        )

        # Whiten the data
        min = np.min(self.gaussian_wide)
        max = np.max(self.gaussian_wide)
        self.gaussian_wide = self.gaussian_wide - min
        self.gaussian_wide = self.gaussian_wide / (max - min)

        # Flip upside down
        if self._invert:
            self.gaussian_y = np.ones_like(self.gaussian_y) - self.gaussian_y
            self.gaussian_wide = np.ones_like(self.gaussian_wide) - self.gaussian_wide

        if self._show:
            # Late import since this usually isn't used
            import matplotlib.pyplot as plt

            # Plot the Gaussian curve
            # plt.plot(x, self.gaussian_y)
            plt.plot(x, np.ones_like(self.gaussian_wide) - self.gaussian_wide)
            plt.title("Gaussian Curve")
            plt.xlabel("X")
            plt.ylabel("Y")
            # Show the plot
            plt.show()
            import time

            time.sleep(1000)

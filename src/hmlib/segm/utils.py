from typing import List, Tuple

import matplotlib.path as mplPath
import numpy as np
import torch


def polygon_to_mask(poly: List[Tuple[float, float]], height: int, width: int) -> torch.Tensor:
    """
    Convert a polygon to an image mask

    :param poly:    Polygon as list of (x, y) tuples
    :param height:  Mask height
    :param width:   Mask width
    :return:        Image mask

    # Example usage
    poly = [(1,1), (5,1), (5,5), (1,5)]  # Define polygon as list of (x, y) tuples
    height, width = 10, 10  # Dimensions of the binary mask
    mask = polygon_to_mask(poly, height, width)
    print(mask)

    # Checking if a point is within the polygon using the mask
    point_x, point_y = 3, 3
    is_inside = mask[point_y, point_x]
    print("Point is inside polygon:", is_inside)
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))  # Create a grid of coordinates
    points = np.vstack((x.flatten(), y.flatten())).T
    path = mplPath.Path(np.array(poly))  # Create a path from polygon
    grid = path.contains_points(points)
    mask = grid.reshape((height, width))
    return torch.tensor(mask, dtype=torch.bool)

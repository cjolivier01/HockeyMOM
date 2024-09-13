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


def scale_polygon(polygon: List[Tuple[float, float]], ratio: float) -> List[Tuple[float, float]]:
    """
    Scales a polygon by a given ratio around its centroid.

    Args:
    polygon (List[Tuple[float, float]]): List of (x, y) tuples representing the polygon vertices.
    ratio (float): Scaling ratio.

    Returns:
    List[Tuple[float, float]]: List of scaled (x, y) tuples representing the new polygon vertices.
    """
    # Convert list of tuples to a PyTorch tensor
    was_list: bool = False
    was_numpy: bool = False
    if isinstance(polygon, list):
        points = torch.tensor(polygon, dtype=torch.float32)
        was_list = True
    elif isinstance(polygon, np.ndarray):
        points = torch.from_numpy(polygon).to(torch.float)
        was_numpy = True
    else:
        points = points

    # Calculate the centroid of the polygon
    centroid = torch.mean(points, dim=0)

    # Move the points to the origin (centroid at origin)
    points_centered = points - centroid

    # Scale the points
    scaled_points = points_centered * ratio

    # Move the points back to their original position
    scaled_points = scaled_points + centroid

    if was_list:
        # Convert tensor back to list of tuples
        scaled_polygon = list(map(tuple, scaled_points.tolist()))
    else:
        scaled_polygon = scaled_points
    if was_numpy:
        scaled_polygon = scaled_polygon.numpy()

    return scaled_polygon

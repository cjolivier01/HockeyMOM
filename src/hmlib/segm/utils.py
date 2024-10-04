from typing import List, Tuple, TypeAlias, Union

import matplotlib.path as mplPath
import numpy as np
import torch

ContourPoints: TypeAlias = Union[Tuple[int, int], torch.Tensor, np.ndarray, List[np.ndarray]]

DTYPE_MAP = None


def to_points_tensor(points: ContourPoints, dtype=torch.float) -> Tuple[torch.Tensor, torch.dtype]:
    if isinstance(points, list):
        initial_dtype = points[0].dtype
        points_tensor = torch.tensor(points, dtype=dtype).reshape(-1, 2)
    elif isinstance(points, np.ndarray):
        points_tensor = torch.from_numpy(points)
        initial_dtype = points_tensor.dtype
    else:
        points_tensor = points
        initial_dtype = points_tensor.dtype
    if points_tensor.dtype != dtype:
        points_tensor = points_tensor.to(dtype, non_blocking=True)
    return points_tensor, initial_dtype


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


def calculate_centroid(polygons: ContourPoints) -> torch.Tensor:
    if isinstance(polygons, list) and isinstance(polygons[0], np.ndarray):
        polygons = np.concatenate(polygons, axis=0)
    points, _ = to_points_tensor(polygons)
    assert points.ndim == 2
    assert points.shape[1] == 2
    # Calculate the centroid of the polygon
    centroid = torch.mean(points, dim=0)
    return centroid


def scale_polygon_y(points: torch.Tensor, top_ratio: float, bottom_ratio: float):
    # Convert the flat list into a tensor and reshape it into (n, 2)

    if isinstance(points, list):
        dtype = points[0].dtype
        points_tensor = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
    elif isinstance(points, np.ndarray):
        points_tensor = torch.from_numpy(points)
        dtype = points_tensor.dtype
    else:
        points_tensor = points
        dtype = points_tensor.dtype

    if not torch.is_floating_point(points_tensor):
        points_tensor = points_tensor.to(dtype=torch.float, non_blocking=True)

    # Calculate the centroid in the y-direction
    centroid_y = points_tensor[:, 1].mean()

    # Create a mask for points above and below the centroid
    top_mask = points_tensor[:, 1] > centroid_y
    bottom_mask = ~top_mask  # Everything not in the top_mask is in the bottom_mask

    # Scale y-values according to their position relative to the centroid
    points_tensor[top_mask, 1] = centroid_y + (points_tensor[top_mask, 1] - centroid_y) * top_ratio
    points_tensor[bottom_mask, 1] = (
        centroid_y + (points_tensor[bottom_mask, 1] - centroid_y) * bottom_ratio
    )
    if dtype != points_tensor.dtype:
        points_tensor = points_tensor.to(dtype, non_blocking=True)

    return points_tensor.reshape(-1)

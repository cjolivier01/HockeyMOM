import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

from hmlib.stitching.laplacian_blend import show_image
from hmlib.utils.image import image_height, image_width


def evenly_spaced_indices(n_points, n_samples):
    """Generate indices to pick n_samples evenly spaced from n_points."""
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch, n_samples):
    """
    Selects a subset of points that are most evenly spaced over the Y range.

    Args:
    - batch (torch.Tensor): A tensor of shape (N, 2) where N is the number of points,
                            and the second dimension represents (X, Y) coordinates.
    - n_samples (int): Number of samples to select.

    Returns:
    - torch.Tensor: Indices of the selected points in the original batch.
    """
    # Sort the points based on Y values
    _, sorted_indices = torch.sort(batch[:, 1])

    # Calculate indices that would space the points evenly
    sample_indices = evenly_spaced_indices(batch.size(0), n_samples)

    # Select indices of the original batch
    selected_indices = sorted_indices[sample_indices]

    return selected_indices


def indices_below_y_threshold(batch: torch.Tensor, y_threshold: float):
    """
    Returns indices of points where the Y value is less than the given threshold.

    Args:
    - batch (torch.Tensor): A tensor of shape (N, 2) where N is the number of points,
                            and the second dimension represents (X, Y) coordinates.
    - threshold (float): The Y value threshold.

    Returns:
    - torch.Tensor: Indices of the points where Y values are below the threshold.
    """
    # Find the indices where Y value is below the threshold
    indices = (batch[:, 1] > y_threshold).nonzero().squeeze()

    return indices


def indices_below_x_threshold(batch: torch.Tensor, x_threshold: float):
    """
    Returns indices of points where the Y value is less than the given threshold.

    Args:
    - batch (torch.Tensor): A tensor of shape (N, 2) where N is the number of points,
                            and the second dimension represents (X, Y) coordinates.
    - threshold (float): The Y value threshold.

    Returns:
    - torch.Tensor: Indices of the points where Y values are below the threshold.
    """
    # Find the indices where X value is below the threshold
    indices = (batch[:, 1] < x_threshold).nonzero().squeeze()

    return indices


def indices_above_x_threshold(batch: torch.Tensor, x_threshold: float):
    """
    Returns indices of points where the Y value is less than the given threshold.

    Args:
    - batch (torch.Tensor): A tensor of shape (N, 2) where N is the number of points,
                            and the second dimension represents (X, Y) coordinates.
    - threshold (float): The Y value threshold.

    Returns:
    - torch.Tensor: Indices of the points where Y values are below the threshold.
    """
    # Find the indices where X value is below the threshold
    indices = (batch[:, 1] > x_threshold).nonzero().squeeze()

    return indices


def calculate_control_points(
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    # max_num_keypoints: int = 50,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = (
        SuperPoint(
            max_num_keypoints=max_num_keypoints,
            # detection_threshold=0.1,
        )
        .eval()
        .to(device)
    )  # load the extractor
    matcher = LightGlue(features="superpoint", depth_confidence=0.95, n_layers=9).eval().to(device)

    if not isinstance(image0, torch.Tensor):
        image0 = load_image(image0)
    if not isinstance(image1, torch.Tensor):
        image1 = load_image(image1)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    y_threshold = image_height(image0) / 20
    left_x_threshold = image_width(image0) - image_width(image0) / 25
    right_x_threshold = image_width(image0) / 25

    # Very top will diverge, so don't trust these points
    indices = indices_below_y_threshold(m_kpts0, y_threshold)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    indices = indices_below_y_threshold(m_kpts1, y_threshold)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    indices = indices_below_x_threshold(m_kpts0, left_x_threshold)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    indices = indices_above_x_threshold(m_kpts1, right_x_threshold)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    indices = select_evenly_spaced(m_kpts0, 20)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("/home/colivier/src/hm/matches.png")
    # show_image("matches", "/home/colivier/src/hm/matches.png")

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    viz2d.save_plot("/home/colivier/src/hm/keypoints.png")
    return dict(kpts0=kpts0, m_kpts0=m_kpts0, kpts1=kpts1, m_kpts1=m_kpts1)


if __name__ == "__main__":
    results = calculate_control_points(
        image0=f"{os.environ['HOME']}/Videos/ev-sabercats-1/left.png",
        image1=f"{os.environ['HOME']}/Videos/ev-sabercats-1/right.png",
    )
    print("Done.")

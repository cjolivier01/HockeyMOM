"""Control-point utilities for panorama stitching.

Provides LightGlue-based control point extraction and simple coordinate
filters used when building Hugin projects for two-camera stitching.
"""

import gc
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from hmlib.utils.image import image_height, image_width


def evenly_spaced_indices(n_points, n_samples):
    """Generate indices to pick ``n_samples`` evenly spaced from ``n_points``."""
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch, n_samples):
    """Select a subset of points that are evenly spaced in Y.

    @param batch: Tensor of shape ``(N, 2)`` with ``(x, y)`` coordinates.
    @param n_samples: Number of sample indices to return.
    @return: 1D tensor of indices into ``batch``.
    """
    # Sort the points based on Y values
    _, sorted_indices = torch.sort(batch[:, 1])

    # Calculate indices that would space the points evenly
    sample_indices = evenly_spaced_indices(batch.size(0), n_samples)

    # Select indices of the original batch
    selected_indices = sorted_indices[sample_indices]

    return selected_indices


def indices_below_y_threshold(batch: torch.Tensor, y_threshold: float):
    """Return indices of points whose Y is greater than ``y_threshold``."""
    # Find the indices where Y value is below the threshold
    indices = (batch[:, 1] > y_threshold).nonzero().squeeze()

    return indices


def indices_below_x_threshold(batch: torch.Tensor, x_threshold: float):
    """Return indices of points whose X is less than ``x_threshold``."""
    # Find the indices where X value is below the threshold
    indices = (batch[:, 1] < x_threshold).nonzero().squeeze()

    return indices


def indices_above_x_threshold(batch: torch.Tensor, x_threshold: float):
    """Return indices of points whose X is greater than ``x_threshold``."""
    # Find the indices where X value is below the threshold
    indices = (batch[:, 1] > x_threshold).nonzero().squeeze()

    return indices


def indices_of_min_x(points: torch.Tensor, N: int):
    """Return indices of the ``N`` points with smallest X in a set."""
    # Concatenate the two batches

    # Sort based on the X values and get the indices
    _, indices = torch.sort(points[:, 0])

    # Select the indices of the N smallest X values
    return indices[:N]


def cvtcolor_bgr_to_rgb(image_bgr: torch.Tensor) -> torch.Tensor:
    """Convert BGR image tensor(s) to RGB."""
    if image_bgr.ndim == 3:
        return image_bgr[[2, 1, 0], :, :]
    return image_bgr[:, [2, 1, 0], :, :]


def compute_destination_size_wh(
    img: torch.Tensor, homography_matrix: torch.Tensor
) -> Tuple[int, int]:
    """Compute destination width/height after applying a homography.

    @param img: Source image tensor (H, W or B, C, H, W).
    @param homography_matrix: 3×3 homography matrix (torch or NumPy).
    @return: Tuple ``(new_width, new_height)``.
    """
    width = image_width(img)
    height = image_height(img)
    corners = np.array(
        [
            [0, 0],  # Top-left corner
            [width, 0],  # Top-right corner
            [width, height],  # Bottom-right corner
            [0, height],  # Bottom-left corner
        ],
        dtype="float32",
    )

    # Reshape for perspectiveTransform
    corners = corners.reshape(-1, 1, 2)

    # Apply homography
    if isinstance(homography_matrix, torch.Tensor):
        homography_matrix = homography_matrix.cpu().numpy()
    transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)

    # Calculate the bounding box of the transformed corners
    x_coords = transformed_corners[:, 0, 0]
    y_coords = transformed_corners[:, 0, 1]

    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    # Compute the dimensions of the bounding box
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    return new_width, new_height


def calculate_control_points(
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    max_control_points: int,
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    output_directory: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Compute LightGlue control points for a pair of images.

    @param image0: First image (path or tensor).
    @param image1: Second image (path or tensor).
    @param max_control_points: Maximum number of matched points to keep.
    @param device: Optional PyTorch device for inference.
    @param max_num_keypoints: Maximum raw detector keypoints per image.
    @param output_directory: Optional directory for LightGlue visualizations.
    @return: Dict containing tensors ``m_kpts0`` and ``m_kpts1`` (Nx2).
    """

    from lightglue import LightGlue, SuperPoint, viz2d
    from lightglue.utils import load_image, rbd

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor: SuperPoint = (
        SuperPoint(
            max_num_keypoints=max_num_keypoints,
        )
        .eval()
        .to(device)
    )  # load the extractor
    matcher = (
        LightGlue(
            features="superpoint",
            # depth_confidence=0.95,
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=0.2,
        )
        .eval()
        .to(device)
    )

    if not isinstance(image0, torch.Tensor):
        image0 = load_image(image0)
    if not isinstance(image1, torch.Tensor):
        image1 = load_image(image1)

    if device is not None:
        if image0.device != device:
            image0 = image0.to(device)
        if image1.device != device:
            image1 = image1.to(device)

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # Clean up memory and garbage collect
    del matcher
    del extractor
    gc.collect()

    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    indices = select_evenly_spaced(m_kpts0, max_control_points)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    if output_directory:
        viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        viz2d.save_plot(os.path.join(output_directory, "matches.png"))

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        viz2d.save_plot(os.path.join(output_directory, "keypoints.png"))
    control_points = dict(m_kpts0=m_kpts0, m_kpts1=m_kpts1)
    return control_points


if __name__ == "__main__":
    results = calculate_control_points(
        image0=f"{os.environ['HOME']}/Videos/ev-sabercats-2/left.png",
        image1=f"{os.environ['HOME']}/Videos/ev-sabercats-2/right.png",
        device=torch.device("cuda", 0),
        output_directory=".",
        max_control_points=240,
    )
    print("Done.")

import os
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import kornia
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd

from hmlib.config import get_game_dir
from hmlib.stitching.laplacian_blend import show_image
from hmlib.utils.image import image_height, image_width, make_channels_last


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


def indices_of_min_x(points: torch.Tensor, N: int):
    """
    Returns the indices of the N points with the smallest X values from two batches.

    Args:
    - batch1 (torch.Tensor): A tensor of shape (N1, 2) representing the first batch of points.
    - batch2 (torch.Tensor): A tensor of shape (N2, 2) representing the second batch of points.
    - N (int): The number of points to return.

    Returns:
    - torch.Tensor: Indices of the N points with the smallest X values.
    """
    # Concatenate the two batches

    # Sort based on the X values and get the indices
    _, indices = torch.sort(points[:, 0])

    # Select the indices of the N smallest X values
    return indices[:N]


def calculate_control_points(
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    # max_num_keypoints: int = 50,
    output_directory: Optional[str] = None,
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
    matcher = LightGlue(features="superpoint", depth_confidence=0.95).eval().to(device)

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

    y_threshold = image_height(image0) / 15
    left_x_threshold = image_width(image0) - image_width(image0) / 25
    right_x_threshold = image_width(image0) / 25

    # Very top will diverge, so don't trust these points
    if False:
        # indices = indices_below_y_threshold(m_kpts0, y_threshold)
        # m_kpts0 = m_kpts0[indices]
        # m_kpts1 = m_kpts1[indices]

        # indices = indices_below_y_threshold(m_kpts1, y_threshold)
        # m_kpts0 = m_kpts0[indices]
        # m_kpts1 = m_kpts1[indices]

        # indices = indices_below_x_threshold(m_kpts0, left_x_threshold)
        # m_kpts0 = m_kpts0[indices]
        # m_kpts1 = m_kpts1[indices]

        # indices = indices_above_x_threshold(m_kpts1, right_x_threshold)
        # m_kpts0 = m_kpts0[indices]
        # m_kpts1 = m_kpts1[indices]

        # indices = indices_of_min_x(m_kpts0, 20)

        indices = select_evenly_spaced(m_kpts0, 50)
        # m_kpts0 = m_kpts0[indices]
        # m_kpts1 = m_kpts1[indices]

    def my_matcher(data: Dict[str, torch.Tensor]):
        results = {
            "keypoints1": m_kpts0.cpu(),
            "keypoints0": m_kpts1.cpu(),
            "batch_indexes": [0],
        }
        return results

    # matcher = kornia.feature.LoFTR(None)
    matcher = my_matcher
    stitcher = kornia.contrib.ImageStitcher(matcher)
    # .to(device=device, dtype=image0.dtype)
    torch.manual_seed(1)  # issue kornia#2027
    # out, mask = stitcher.qstitch(
    #     image0.unsqueeze(0).to(torch.float),
    #     image1.unsqueeze(0).to(torch.float),
    # )
    homo = stitcher.qstitch(
        image0.unsqueeze(0).to(torch.float),
        image1.unsqueeze(0).to(torch.float),
    )

    # H = homo.numpy()
    # img = make_channels_last(image1).numpy()
    # warped_image = cv2.warpPerspective(img, H, (img.shape[1] * 2, img.shape[0] * 2))

    stitched = opencv_stitch(
        make_channels_last(image0 * 255).clamp(0, 255).to(torch.uint8).numpy(),
        make_channels_last(image1 * 255).clamp(0, 255).to(torch.uint8).numpy(),
        H=homo.numpy(),
    )

    show_image("stitched", stitched)
    # show_image("warped_image", img)
    # show_image("out", mask * 255)

    if output_directory:
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        viz2d.save_plot(os.path.join(output_directory, "matches.png"))
        # show_image("matches", os.path.join(output_directory, "matches.png"))

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        viz2d.save_plot(os.path.join(output_directory, "keypoints.png"))
    return dict(kpts0=kpts0, m_kpts0=m_kpts0, kpts1=kpts1, m_kpts1=m_kpts1)


def opencv_stitch(image1, image2, H):
    # sift = cv2.SIFT_create()
    # keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    # keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # matcher = cv2.BFMatcher()
    # matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # good_matches = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
    # if not good_matches:
    #     good_matches = matches

    # if len(good_matches) > 10:
    #     src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width, channels = image1.shape
    output_shape = (width * 2, height)
    result = cv2.warpPerspective(image2, H, output_shape)

    mask_left = np.ones_like(image1)
    mask_right = np.ones_like(image2)
    # 'nearest' to ensure no floating points in the mask
    src_mask = cv2.warpPerspective(mask_right, H, output_shape, flags=cv2.INTER_NEAREST)
    # warp_perspective(mask_right, homo, out_shape, mode="nearest")
    dst_mask = np.concatenate([mask_left, np.zeros_like(mask_right)], -1)
    # return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(
    #     src_mask.dtype
    # )

    result[0:height, 0:width] = image1

    # cv2.imshow("Stitched Image", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    results = calculate_control_points(
        image0=f"{os.environ['HOME']}/Videos/ev-sabercats-1/left.png",
        image1=f"{os.environ['HOME']}/Videos/ev-sabercats-1/right.png",
    )
    print("Done.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Optional
import kornia as K
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
from kornia.geometry import RANSAC, find_homography_dlt

# from kornia.feature import LightGlue
# from kornia.feature import SuperPoint, LightGlue


def _permute(t: torch.Tensor, *args) -> torch.Tensor:
    return t.permute(*args)


def make_channels_first(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) == 4:
        if img.shape[-1] in [1, 3, 4]:
            return _permute(img, 0, 3, 1, 2)
    else:
        assert len(img.shape) == 3
        if img.shape[-1] in [1, 3, 4]:
            return _permute(img, 2, 0, 1)
    return img


def make_channels_last(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) == 4:
        if img.shape[1] in [1, 3, 4]:
            return _permute(img, 0, 2, 3, 1)
    else:
        assert len(img.shape) == 3
        if img.shape[0] in [1, 3, 4]:
            return _permute(img, 1, 2, 0)
    return img


class ImageStitcher(nn.Module):
    def __init__(
        self, superpoint_weights: Optional[str] = None, lightglue_weights: Optional[str] = None
    ):
        super().__init__()

        # Initialize feature extractors and matcher
        self.superpoint = SuperPoint(
            max_num_leypoints=2048,
        )
        self.matcher = LightGlue(
            features="superpoint",
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=0.2,
        )

    def create_gaussian_pyramid(self, img: torch.Tensor, levels: int = 4) -> List[torch.Tensor]:
        """Create a Gaussian pyramid of the image."""
        pyramid = []
        current = img
        for _ in range(levels):
            pyramid.append(current)
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
        return pyramid

    def create_laplacian_pyramid(self, img: torch.Tensor, levels: int = 4) -> List[torch.Tensor]:
        """Create a Laplacian pyramid of the image."""
        gaussian_pyramid = self.create_gaussian_pyramid(img, levels)
        laplacian_pyramid = []

        for i in range(len(gaussian_pyramid) - 1):
            curr_gaussian = gaussian_pyramid[i]
            next_gaussian = gaussian_pyramid[i + 1]
            upsampled = F.interpolate(
                next_gaussian, size=curr_gaussian.shape[-2:], mode="bilinear", align_corners=False
            )
            laplacian = curr_gaussian - upsampled
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    def blend_pyramids(
        self, pyr1: List[torch.Tensor], pyr2: List[torch.Tensor], mask_pyramid: List[torch.Tensor]
    ) -> torch.Tensor:
        """Blend two Laplacian pyramids using a mask pyramid."""
        blended_pyramid = []
        for la1, la2, mask in zip(pyr1, pyr2, mask_pyramid):
            blended = la1 * mask + la2 * (1 - mask)
            blended_pyramid.append(blended)

        # Reconstruct image from pyramid
        reconstruction = blended_pyramid[-1]
        for laplacian in reversed(blended_pyramid[:-1]):
            reconstruction = F.interpolate(
                reconstruction, size=laplacian.shape[-2:], mode="bilinear", align_corners=False
            )
            reconstruction = reconstruction + laplacian

        return reconstruction

    def extract_and_match_features(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and match features between two images using SuperPoint and LightGlue."""
        # Convert images to grayscale if they're RGB
        # if img1.shape[1] == 3:
        #     img1_gray = K.color.rgb_to_grayscale(img1)
        #     img2_gray = K.color.rgb_to_grayscale(img2)
        # else:
        #     img1_gray = img1
        #     img2_gray = img2

        # Extract features
        # features1 = self.superpoint({"image": img1_gray})
        # features2 = self.superpoint({"image": img2_gray})

        feats0 = self.superpoint.extract(img1)
        feats1 = self.superpoint.extract(img2)
        matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return m_kpts0, m_kpts1

    def compute_homography(self, kpts1: torch.Tensor, kpts2: torch.Tensor) -> torch.Tensor:
        """Compute homography matrix between two sets of keypoints."""
        # Use RANSAC for robust homography estimation
        # Initialize RANSAC
        ransac = RANSAC(
            model_type="homography",
            # min_samples=4,
            confidence=0.99,
            max_iter=100,
            inl_th=3.0,
        )

        # Estimate homography using RANSAC
        H, inliers = ransac(kpts1, kpts2)
        return H

    def create_blending_mask(
        self, shape: Tuple[int, int], overlap_region: torch.Tensor
    ) -> torch.Tensor:
        """Create a blending mask for the overlapping region."""
        mask = torch.zeros(1, 1, shape[0], shape[1], device=overlap_region.device)

        # Create gradual transition in overlap region
        x = torch.linspace(0, 1, overlap_region.shape[1], device=overlap_region.device)
        mask[:, :, overlap_region[0] : overlap_region[2], overlap_region[1] : overlap_region[3]] = (
            x.view(1, 1, 1, -1)
        )

        return mask

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Stitch multiple images together using feature matching and Laplacian blending.

        Args:
            images: List of RGB images as tensors with shape [B, 3, H, W]

        Returns:
            Stitched image as tensor with shape [B, 3, H', W']
        """
        if len(images) < 2:
            raise ValueError("At least two images are required for stitching")

        result = images[0]

        for i in range(1, len(images)):
            # Extract and match features
            kpts1, kpts2 = self.extract_and_match_features(result, images[i])

            # Compute homography
            H = self.compute_homography(kpts1, kpts2).unsqueeze(0)

            # Warp image
            warped = K.geometry.warp_perspective(images[i].unsqueeze(0), H, result.shape[-2:])

            show(warped)

            # Find overlap region
            overlap_mask = (result.sum(dim=1, keepdim=True) > 0) & (
                warped.sum(dim=1, keepdim=True) > 0
            )
            overlap_coords = torch.where(overlap_mask[0, 0])
            overlap_region = torch.tensor(
                [
                    overlap_coords[0].min(),
                    overlap_coords[1].min(),
                    overlap_coords[0].max(),
                    overlap_coords[1].max(),
                ]
            )

            # Create blending mask
            blend_mask = self.create_blending_mask(result.shape[-2:], overlap_region)

            # Create Laplacian pyramids
            pyr1 = self.create_laplacian_pyramid(result)
            pyr2 = self.create_laplacian_pyramid(warped)
            mask_pyramid = self.create_gaussian_pyramid(blend_mask)

            # Blend pyramids
            result = self.blend_pyramids(pyr1, pyr2, mask_pyramid)

        return result

    @torch.no_grad()
    def stitch_images(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Utility method to stitch images with gradient computation disabled."""
        return self.forward(images)


def show(img):
    img = make_channels_last(img * 255).clamp(0, 255).to(torch.uint8)
    img = img.squeeze(0)
    img = np.ascontiguousarray(img.cpu().numpy())
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    img1 = load_image("/mnt/home/colivier-local/Videos/pdp/left.png")
    img2 = load_image("/mnt/home/colivier-local/Videos/pdp/right.png")
    # show(img1)
    stitcher = ImageStitcher()
    if torch.cuda.is_available() and False:
        stitcher.to("cuda")
        img1 = img1.to("cuda")
        img2 = img2.to("cuda")
    out = stitcher.forward([img1, img2])
    show(out)

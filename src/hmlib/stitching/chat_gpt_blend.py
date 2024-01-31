import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianBlendWithMasks(nn.Module):
    def __init__(self, levels=5):
        super(LaplacianBlendWithMasks, self).__init__()
        self.levels = levels

    def forward(self, image1, image2, seam_mask, xor_mask):
        # Build Laplacian pyramids for both images
        pyramid1 = self.build_laplacian_pyramid(image1)
        pyramid2 = self.build_laplacian_pyramid(image2)

        # Blend the pyramids using the seam mask and XOR mask
        blended_pyramid = []
        for level in range(self.levels):
            blended_level = pyramid1[level] * seam_mask + pyramid2[level] * (1 - seam_mask) + xor_mask * (pyramid1[level] ^ pyramid2[level])
            blended_pyramid.append(blended_level)

        # Reconstruct the blended image from the blended pyramid
        blended_image = self.reconstruct_laplacian_pyramid(blended_pyramid)

        return blended_image

    def build_laplacian_pyramid(self, image):
        pyramid = []
        for _ in range(self.levels):
            blurred = F.avg_pool2d(image, kernel_size=2)
            upsampled = F.interpolate(blurred, scale_factor=2, mode='bilinear', align_corners=False)
            residual = image - upsampled
            pyramid.append(residual)
            image = blurred
        pyramid.append(image)  # Low-pass residual image
        return pyramid

    def reconstruct_laplacian_pyramid(self, pyramid):
        image = pyramid[-1]
        for i in range(self.levels - 1, -1, -1):
            upsampled = F.interpolate(image, scale_factor=2, mode='bilinear', align_corners=False)
            image = upsampled + pyramid[i]
        return image

# Example usage:
image1 = torch.randn(1, 3, 256, 256)  # Replace with your own images
image2 = torch.randn(1, 3, 256, 256)
seam_mask = torch.randn(1, 1, 256, 256)  # Replace with your seam mask
xor_mask = torch.randn(1, 3, 256, 256)  # Replace with your XOR mask
blend_module = LaplacianBlendWithMasks()
blended_image = blend_module(image1, image2, seam_mask, xor_mask)

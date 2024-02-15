import torch
import torch.nn.functional as F
import numpy as np

def find_perspective_transform(src, dst):
    # Construct the matrix A based on the source and destination coordinates
    A = []
    for i in range(0, len(src)):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])

    A = np.array(A)
    B = dst.flatten()

    # Solve for the transformation matrix
    transform_matrix = np.linalg.solve(A, B)
    transform_matrix = np.append(transform_matrix, 1).reshape(3, 3)

    return transform_matrix

def warp_perspective(image_tensor, M, dsize):
    C, H, W = image_tensor.shape
    M = torch.tensor(M, dtype=torch.float32)

    # Generate grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, dsize[1]), torch.linspace(-1, 1, dsize[0]), indexing='ij')
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).unsqueeze(0)  # Shape: (1, H, W, 3)

    # Transform grid
    grid = grid @ M.T
    grid = grid[:,:,:,:2] / grid[:,:,:,2:3]  # Divide by Z to get image coordinates
    grid = grid.permute(0, 3, 1, 2)  # Reorder dimensions to (N, C, H, W)
    grid = grid[:, :2, :, :]  # Drop ones
    grid = (grid - 0.5) * 2  # Scale grid to [-1, 1]

    # Sample pixels using grid
    warped_image = F.grid_sample(image_tensor.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_image.squeeze(0)

# Example usage
# Assume image_tensor is your loaded image in (C, H, W) format
# Define source points (corners of the distorted scoreboard) and destination points (corners of the desired rectangle)
src_points = torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # Replace these points with actual coordinates
dst_points = torch.tensor([[0, 0], [width, 0], [width, height], [0, height]])  # Desired rectangle's corners

# Calculate the perspective transform matrix
M = find_perspective_transform(src_points, dst_points)

# Apply perspective warp
warped_image = warp_perspective(image_tensor, M, (width, height))

# Now, `warped_image` contains the transformed image tensor

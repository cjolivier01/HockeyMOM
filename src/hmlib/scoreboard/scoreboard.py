import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

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
    B, C, H, W = image_tensor.shape
    M = torch.tensor(M, dtype=torch.float32)

    # Generate grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, dsize[1]), torch.linspace(-1, 1, dsize[0]), indexing='ij')
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).unsqueeze(0)  # Shape: (1, H, W, 3)

    # Transform grid
    grid = grid @ M.T
    # grid = grid[:,:,:,:2] / grid[:,:,:,2:3]  # Divide by Z to get image coordinates
    # grid = grid.permute(0, 3, 1, 2)  # Reorder dimensions to (N, C, H, W)
    # grid = grid[:, :2, :, :]  # Drop ones
    grid = (grid - 0.5) * 2  # Scale grid to [-1, 1]
    grid = grid.unsqueeze(0)

    # Sample pixels using grid
    warped_image = F.grid_sample(image_tensor.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return warped_image.squeeze(0)

def example():
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


# Manually create a grid that applies the perspective transformation
# def warp_perspective_pytorch(image_tensor, M, dsize):
#     if isinstance(image_tensor, np.ndarray):
#         image_tensor = torch.from_numpy(image_tensor)
#         # add batch and make channels first
#         image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
#     B, C, H, W = image_tensor.shape
#     # Convert M from numpy to tensor
#     M = torch.tensor(M, dtype=torch.float32)

#     # Create normalized 2D grid
#     xx, yy = torch.meshgrid(torch.linspace(-1, 1, dsize[1]), torch.linspace(-1, 1, dsize[0]))
#     grid = torch.stack([yy, xx], dim=2).unsqueeze(0)  # Shape: (1, H, W, 2)

#     # Adjust grid for perspective transformation
#     # Add ones to make grid homogenous
#     ones = torch.ones(grid.shape[:-1] + (1,))
#     grid_homogenous = torch.cat([grid, ones], dim=3)

#     # Apply perspective transformation
#     grid_transformed = grid_homogenous @ M.T[:3, :3]
#     grid_transformed = grid_transformed[..., :2] / grid_transformed[..., 2:3]  # Divide by Z to normalize

#     # Warp image using grid_sample
#     image_tensor = image_tensor.to(torch.float32) / 255.0
#     warped_image = F.grid_sample(image_tensor, grid_transformed, mode='bilinear', padding_mode='zeros', align_corners=False)
#     warped_image = torch.clamp(warped_image * 255.0, min=0, max=255).to(torch.uint8)

#     return warped_image

def warp_perspective_pytorch(image_tensor, M, dsize):
    # Convert M to a torch tensor and calculate its inverse
    M = torch.tensor(M, dtype=torch.float32)
    M_inverse = torch.inverse(M)

    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor)

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.shape[-1] == 4:
        image_tensor = image_tensor[:, :, :, :3]

    # Generate a grid of points in the output image
    #height, width = dsize
    width, height = dsize
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2)  # Shape: (H, W, 3)

    # Transform the grid using the inverse matrix
    grid = grid.view(-1, 3)  # Flatten grid
    grid_transformed = torch.mm(grid, M_inverse.t())  # Apply inverse transformation matrix
    grid_transformed = grid_transformed / grid_transformed[:, 2].unsqueeze(-1)  # Normalize z to 1
    grid_transformed = grid_transformed[:, :2]  # Remove the z component

    # Reshape the grid to the shape (1, H, W, 2)
    grid_transformed = grid_transformed.view(height, width, 2)
    grid_transformed = grid_transformed.unsqueeze(0)
    # grid_transformed = grid_transformed.permute(0, 3, 1, 2)  # Change to (1, 2, H, W) for grid_sample

    min_grid = torch.min(grid_transformed)
    max_grid = torch.max(grid_transformed)

    # channels first?
    image_tensor = image_tensor.permute(0, 3, 1, 2)

    # Warp the image using grid_sample

    image_tensor = image_tensor.to(torch.float32) / 255.0

    imin = torch.min(image_tensor)
    imax = torch.max(image_tensor)
    print(grid_transformed)

    warped_image = torch.zeros((3, height, width))
    g_int = grid_transformed.to(torch.int64)
    warped_image[:] = image_tensor[:, g_int[2], g_int[2]]

    warped_image = F.grid_sample(image_tensor, grid_transformed, mode='bilinear', padding_mode='zeros', align_corners=True)

    wmin = torch.min(warped_image)
    wmax = torch.max(warped_image)

    warped_image = torch.clamp(warped_image * 255.0, min=0, max=255).to(torch.uint8)

    wmin = torch.min(warped_image)
    wmax = torch.max(warped_image)

    return warped_image

def main():
    # Load your image
    image_path = '/home/colivier/Videos/sharks-bb3-1/panorama.tif'  # Specify the path to your image
    #image = Image.open(image_path)
    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension

    # This function will be called later to perform the perspective warp
    def outer_warp_perspective(image_tensor, src_points, dst_points, dsize):
        # Calculate the perspective transform matrix
        M = find_perspective_transform(src_points, dst_points)

        # Apply perspective warp
        warped_image = warp_perspective(image_tensor, M, dsize)
        return warped_image

    # Prepare for interactive point selection
    fig, ax = plt.subplots()
    original_image = np.array(np.ascontiguousarray(image))
    img_plot = plt.imshow(image)

    selected_points = []

    # selected_points = [[2007.2903225806454, 389.07741935483864], [2400.2741935483873, 639.1580645161289], [2043.0161290322585, 860.6580645161291], [1907.2580645161293, 574.8516129032257]]

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            # Add the point and redraw
            selected_points.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, 'ro')
            if len(selected_points) > 1:
                plt.plot([selected_points[-2][0], selected_points[-1][0]], [selected_points[-2][1], selected_points[-1][1]], 'r-')
            if len(selected_points) == 4:
                plt.plot([selected_points[-1][0], selected_points[0][0]], [selected_points[-1][1], selected_points[0][1]], 'r-')
                fig.canvas.mpl_disconnect(cid)
                plt.draw()
                # Proceed to warp perspective after 4 points have been selected
                proceed_with_warp_cv2()
            plt.draw()


    def proceed_with_warp_cv2():
        nonlocal original_image
        print(selected_points)

        src_pts = np.array(selected_points, dtype=np.float32)

        width, height = image.size
        #width = 40
        #height = 20

        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

        # Calculate the perspective transform matrix and apply the warp
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print(M)
        warped_image = cv2.warpPerspective(np.array(image), M, (width, height))
        # warped_image = warp_perspective_pytorch(np.array(image), M, (width, height))

        #wmin = torch.min(warped_image)
        #wmax = torch.max(warped_image)

        # Display the warped image
        plt.figure()
        if warped_image.ndim == 4:
            assert warped_image.size(0) == 1
            warped_image = warped_image.squeeze(0)

        if warped_image.shape[0] == 4 or warped_image.shape[0] == 3:
            warped_image = warped_image.permute(1, 2, 0)
            warped_image = warped_image.contiguous().numpy()

        # cv2.imshow("online_im", original_image)
        original_image *= 1
        # cv2.imshow("online_im", warped_image)
        # cv2.waitKey(0)
        plt.imshow(warped_image)
        #plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying correctly
        plt.title("Warped Image")
        plt.show()

    def proceed_with_warp_pytorch():
        # Convert selected points to tensor and perform perspective warp
        src_points_tensor = torch.tensor(selected_points, dtype=torch.float32)
        width, height = image.size
        dst_points = torch.tensor([[0, 0], [width, 0], [width, height], [0, height]], dtype=torch.float32)
        warped_image = outer_warp_perspective(image_tensor, src_points_tensor, dst_points, (width, height))

        # Convert warped image tensor to PIL Image for display
        warped_image_pil = TF.to_pil_image(warped_image.squeeze(0))
        plt.figure()
        plt.imshow(warped_image_pil)
        plt.title("Warped Image")
        plt.show()

    if True:
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    else:
        proceed_with_warp_cv2()


if __name__ == "__main__":
    main()

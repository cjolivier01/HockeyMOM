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


def main():
    # Load your image
    image_path = '/mnt/home/colivier-local/Videos/tvbb2/panorama.tif'  # Specify the path to your image
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
    img_plot = plt.imshow(image)
    selected_points = []

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
        print(selected_points)

        src_pts = np.array(selected_points, dtype=np.float32)

        width, height = image.size
        dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

        # Calculate the perspective transform matrix and apply the warp
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped_image = cv2.warpPerspective(np.array(image), M, (width, height))

        # Display the warped image
        plt.figure()
        plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying correctly
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

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    main()

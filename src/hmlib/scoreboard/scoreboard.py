import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List

import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

from hmlib.utils.image import (
    image_width,
    image_height,
    make_channels_first,
    make_channels_last,
    pad_tensor_to_size_batched,
)
from hmlib.video_out import make_showable_type


class Scoreboard:
    def __init__(self, src_pts: torch.Tensor, dest_height: int, dest_width: int):
        pass


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

    src_width = image_width(image_tensor)
    src_height = image_height(image_tensor)

    # Generate a grid of points in the output image
    # height, width = dsize
    width, height = dsize
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij"
    )
    print(torch.min(grid_x))
    print(torch.max(grid_x))
    print(torch.min(grid_y))
    print(torch.max(grid_y))
    grid = torch.stack(
        (grid_x, grid_y, torch.ones_like(grid_x)), dim=2
    )  # Shape: (H, W, 3)

    # Transform the grid using the inverse matrix
    grid = grid.view(-1, 3)  # Flatten grid
    grid_transformed = torch.mm(
        grid,
        M_inverse.t(),
        # M.t()
    )  # Apply inverse transformation matrix
    # print(grid_transformed)
    # grid_transformed = grid

    grid_transformed = torch.mm(grid, M_inverse.t())

    print(
        f"min={torch.min(grid_transformed)}, max={torch.max(grid_transformed)}, unique={torch.unique(grid_transformed)}"
    )

    grid_transformed = grid_transformed / grid_transformed[:, 2].unsqueeze(
        -1
    )  # Normalize z to 1
    grid_transformed = grid_transformed[:, :2]  # Remove the z component

    print(f"min={torch.min(grid_transformed)}, max={torch.max(grid_transformed)}")

    # Reshape the grid to the shape (1, H, W, 2)
    grid_transformed = grid_transformed.view(height, width, 2)
    # grid_transformed = grid_transformed.permute(height, width, 2)
    grid_transformed = grid_transformed.unsqueeze(0)
    # grid_transformed = grid_transformed.permute(0, 3, 1, 2)  # Change to (1, 2, H, W) for grid_sample

    min_grid_x = torch.min(grid_transformed[:, :, :, 0])
    max_grid_x = torch.max(grid_transformed[:, :, :, 0])

    min_grid_y = torch.min(grid_transformed[:, :, :, 1])
    max_grid_y = torch.max(grid_transformed[:, :, :, 1])

    # channels first?
    image_tensor = image_tensor.permute(0, 3, 1, 2)

    # Warp the image using grid_sample

    image_tensor = image_tensor.to(torch.float32)

    imin = torch.min(image_tensor)
    imax = torch.max(image_tensor)
    # print(grid_transformed)

    warped_image = torch.zeros((3, height, width))
    # g_int = grid_transformed.to(torch.int64)
    # warped_image[:] = image_tensor[:, g_int[2], g_int[2]]
    row_map = grid_transformed[0][:, :, 1]
    col_map = grid_transformed[0][:, :, 0]

    row_min = torch.min(row_map)
    row_max = torch.max(row_map)
    col_min = torch.min(col_map)
    col_max = torch.max(col_map)

    # row_map += 1
    # row_map /= 2
    row_map *= height
    row_map = torch.clamp(row_map, min=0, max=height - 1).to(torch.int32)

    # col_map += 1
    # col_map /= 2
    col_map *= width
    col_map = torch.clamp(col_map, min=0, max=width - 1).to(torch.int32)

    # print(row_map)
    # print(col_map)
    src_height = image_tensor.shape[-2]
    src_width = image_tensor.shape[-1]
    row_map_normalized = (2.0 * row_map / (src_height - 1)) - 1  # Normalize to [-1, 1]
    col_map_normalized = (2.0 * col_map / (src_width - 1)) - 1  # Normalize to [-1, 1]

    # row_map_normalized = (
    #     2.0 * row_map / (src_height - 1)
    # ) - 1  # Normalize to [-1, 1]
    # col_map_normalized = (
    #     2.0 * col_map / (src_width - 1)
    # ) - 1  # Normalize to [-1, 1]

    # Create the grid for grid_sample
    grid = torch.stack((col_map_normalized, row_map_normalized), dim=-1).unsqueeze(0)

    row_min = torch.min(row_map)
    row_max = torch.max(row_map)
    col_min = torch.min(col_map)
    col_max = torch.max(col_map)

    row_norm_min = torch.min(row_map_normalized)
    row_norm_max = torch.max(row_map_normalized)
    col_norm_min = torch.min(col_map_normalized)
    col_norm_max = torch.max(col_map_normalized)

    warped_image[:] = image_tensor[:, :, row_map, col_map]
    # warped_image = F.grid_sample(
    #      image_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    # )
    # warped_image = F.grid_sample(image_tensor, grid_transformed, mode='bilinear', padding_mode='zeros', align_corners=False)

    wmin = torch.min(warped_image)
    wmax = torch.max(warped_image)

    warped_image = torch.clamp(warped_image, min=0, max=255).to(torch.uint8)

    wmin = torch.min(warped_image)
    wmax = torch.max(warped_image)

    return warped_image

def _get_perspective_coeffs(startpoints: List[List[int]], endpoints: List[List[int]]) -> List[float]:
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution

    output: List[float] = res.tolist()
    return output

def find_perspective_transform(src, dst):
    """
    Compute the perspective transform matrix that maps src points to dst points using torch.linalg.solve.
    :param src: Source points (4 points, defined as rectangles' corners)
    :param dst: Destination points (4 points, defined as rectangles' corners)
    :return: Transformation matrix
    """
    matrix = []
    for p1, p2 in zip(dst, src):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = torch.tensor(matrix, dtype=torch.float)
    B = torch.tensor(src, dtype=torch.float).view(8)
    transform_matrix = torch.linalg.solve(A, B)

    transform_matrix = torch.cat((transform_matrix, torch.tensor([1.0])), dim=0)
    transform_matrix = transform_matrix.view(3, 3)
    return transform_matrix


def apply_perspective(img, matrix, w, h):
    """
    Apply the perspective transformation to an image using the computed matrix.
    :param img: Input image tensor of shape (C, H, W)
    :param matrix: Transformation matrix
    :param w: Width of the output image
    :param h: Height of the output image
    :return: Transformed image tensor
    """
    # Invert the transformation matrix
    matrix_inv = torch.inverse(matrix)

    # Normalize the pixel coordinates to [-1, 1]
    grid = torch.nn.functional.affine_grid(matrix_inv[:2].unsqueeze(0), torch.Size((1, *img.shape)), align_corners=False)

    # Apply the transformation
    return torch.nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=False).squeeze(0)


def main():
    # Load your image
    image_path = "/home/colivier/Videos/sharks-bb3-1/panorama.tif"  # Specify the path to your image
    # image = Image.open(image_path)
    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension

    # # This function will be called later to perform the perspective warp
    # def outer_warp_perspective(image_tensor, src_points, dst_points, dsize):
    #     # Calculate the perspective transform matrix
    #     M = find_perspective_transform(src_points, dst_points)

    #     # Apply perspective warp
    #     warped_image = warp_perspective(image_tensor, M, dsize)
    #     return warped_image

    # Prepare for interactive point selection
    fig, ax = plt.subplots()
    original_image = np.array(np.ascontiguousarray(image))
    img_plot = plt.imshow(image)

    selected_points = []

    # selected_points = [[2007.2903225806454, 389.07741935483864], [2400.2741935483873, 639.1580645161289], [2043.0161290322585, 860.6580645161291], [1907.2580645161293, 574.8516129032257]]
    selected_points = [
        [5845.921076009106, 911.8827549830662],
        [6032.949003386821, 969.4298095608242],
        [5996.9820942757215, 1120.4908278274388],
        [5790.954166898008, 1048.5570096052415],
    ]

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            # Add the point and redraw
            selected_points.append([event.xdata, event.ydata])
            ax.plot(event.xdata, event.ydata, "ro")
            if len(selected_points) > 1:
                plt.plot(
                    [selected_points[-2][0], selected_points[-1][0]],
                    [selected_points[-2][1], selected_points[-1][1]],
                    "r-",
                )
            if len(selected_points) == 4:
                plt.plot(
                    [selected_points[-1][0], selected_points[0][0]],
                    [selected_points[-1][1], selected_points[0][1]],
                    "r-",
                )
                fig.canvas.mpl_disconnect(cid)
                plt.draw()
                # Proceed to warp perspective after 4 points have been selected
                proceed_with_warp_cv2()
            plt.draw()

    def get_bbox(point_list: List[List[float]]):
        points = torch.tensor(point_list)
        mins = torch.min(points, dim=0)[0]
        maxs = torch.max(points, dim=0)[0]
        return torch.cat((mins, maxs), dim=0)

    def int_bbox(bbox: torch.Tensor):
        bbox[:2] = torch.floor(bbox[:2])
        bbox[2:] = torch.ceil(bbox[2:])
        return bbox.to(torch.int32)

    def proceed_with_warp_cv2():
        nonlocal original_image
        print(selected_points)

        src_pts = np.array(selected_points, dtype=np.float32)

        src_height = original_image.shape[0]
        src_width = original_image.shape[1]
        # ar = src_width/src_height

        # width, height = image.size
        # width = 200
        # height = 75
        width = src_width
        height = src_height

        to_pil = ToPILImage()
        to_tensor = transforms.ToTensor()

        # src_pts = dst_pts
        # ww=width
        # hh=height/2
        # src_pts = np.array(
        #     [[0, 0], [ww - 1, 0], [ww - 1, hh - 1], [0, hh - 1]],
        #     dtype=np.float32,
        # )
        src_pts = torch.tensor(selected_points, dtype=torch.float)

        if True:
            bbox_src = int_bbox(get_bbox(selected_points))
            # bbox_dest = int_bbox(get_bbox(dst_pts))

            to_tensor = transforms.ToTensor()
            # image = to_tensor(image)
            original_image = make_channels_first(
                torch.from_numpy(original_image).unsqueeze(0)
            )

            src_image = original_image[
                :, :, bbox_src[1] : bbox_src[3], bbox_src[0] : bbox_src[2]
            ]
            src_pts[:, 0] -= bbox_src[0]
            src_pts[:, 1] -= bbox_src[1]

            # width = image_width(src_image) * 3
            # height = image_height(src_image) * 2

            width = image_width(src_image)
            height = image_height(src_image)

            # width = image_width(src_image) / 2
            # height = image_height(src_image) / 3

            src_width = image_width(src_image)
            src_height = image_height(src_image)

            totw = max(width, src_width)
            toth = max(height, src_height)
            if totw > src_width or toth > src_height:
                src_image = pad_tensor_to_size_batched(
                    src_image,
                    target_width=totw,
                    target_height=toth,
                    pad_value=0,
                )
                width = totw
                height = toth

            pil_image = to_pil(src_image.squeeze(0))
        else:
            pil_image = to_pil(original_image)

        # src_pts[:,2] -= bbox_src[0]
        # src_pts[:,3] -= bbox_src[1]
        # show_image(src_image)
        # bbox_dest = get_bbox(dst_pts)

        # print(M)
        # Calculate the perspective transform matrix and apply the warp
        # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warped_image = cv2.warpPerspective(np.array(image), M, (width, height))
        # warped_image = warp_perspective_pytorch(np.array(image), M, (width, height))

        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        coeffs = _get_perspective_coeffs(startpoints=src_pts, endpoints=dst_pts)

        # Compute the perspective transform matrix
        # transform_matrix = find_perspective_transform(src_pts, dst_pts)

        # # Apply the transformation
        # warped_image = apply_perspective(src_image.squeeze(0).to(torch.float) / 255.0, transform_matrix, width, height)
        # warped_image = torch.clamp(warped_image * 255, min=0, max=255).to(torch.uint8)

        warped_image = torchvision.transforms.functional.perspective(
            pil_image, startpoints=src_pts, endpoints=dst_pts
        )
        warped_image = to_tensor(warped_image)

        wmin = torch.min(warped_image)
        wmax = torch.max(warped_image)

        # Display the warped image
        plt.figure()
        if warped_image.ndim == 4:
            assert warped_image.size(0) == 1
            warped_image = warped_image.squeeze(0)

        if warped_image.shape[0] == 4 or warped_image.shape[0] == 3:
            warped_image = warped_image.permute(1, 2, 0)
            warped_image = warped_image.contiguous().numpy()

        wi_min = np.min(warped_image)
        wi_max = np.max(warped_image)

        # cv2.imshow("online_im", original_image)
        original_image *= 1
        # cv2.imshow("online_im", warped_image)
        # cv2.waitKey(0)
        #plt.imshow(pil_image)
        plt.imshow(warped_image)
        # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying correctly
        plt.title("Warped Image")
        plt.show()

    def proceed_with_warp_pytorch():
        # Convert selected points to tensor and perform perspective warp
        src_points_tensor = torch.tensor(selected_points, dtype=torch.float32)
        width, height = image.size
        dst_points = torch.tensor(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=torch.float32
        )
        warped_image = outer_warp_perspective(
            image_tensor, src_points_tensor, dst_points, (width, height)
        )

        # Convert warped image tensor to PIL Image for display
        warped_image_pil = TF.to_pil_image(warped_image.squeeze(0))
        plt.figure()
        plt.imshow(warped_image_pil)
        plt.title("Warped Image")
        plt.show()

    if False:
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
    else:
        proceed_with_warp_cv2()


if __name__ == "__main__":
    main()

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad

import torchvision
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
#from torchvision.transforms._functional_tensor import _apply_grid_transform

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


def _get_perspective_coeffs(
    startpoints: List[List[int]], endpoints: List[List[int]]
) -> List[float]:
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
        a_matrix[2 * i, :] = torch.tensor(
            [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]]
        )
        a_matrix[2 * i + 1, :] = torch.tensor(
            [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]]
        )

    b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution

    output: List[float] = res.tolist()
    return output


def _perspective_grid(
    coeffs: List[float], ow: int, oh: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]],
        dtype=dtype,
        device=device,
    )
    theta2 = torch.tensor(
        [[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]],
        dtype=dtype,
        device=device,
    )

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow, device=device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh, device=device).unsqueeze_(
        -1
    )
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor(
        [0.5 * ow, 0.5 * oh], dtype=dtype, device=device
    )
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)

def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def _apply_grid_transform(
    img: Tensor, grid: Tensor, mode: str, fill: Optional[Union[int, float, List[float]]]
) -> Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        mask = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
        img = torch.cat((img, mask), dim=1)

    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        fill_list, len_fill = (fill, len(fill)) if isinstance(fill, (tuple, list)) else ([float(fill)], 1)
        fill_img = torch.tensor(fill_list, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


# def find_perspective_transform(src, dst):
#     """
#     Compute the perspective transform matrix that maps src points to dst points using torch.linalg.solve.
#     :param src: Source points (4 points, defined as rectangles' corners)
#     :param dst: Destination points (4 points, defined as rectangles' corners)
#     :return: Transformation matrix
#     """
#     matrix = []
#     for p1, p2 in zip(dst, src):
#         matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
#         matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

#     A = torch.tensor(matrix, dtype=torch.float)
#     B = torch.tensor(src, dtype=torch.float).view(8)
#     transform_matrix = torch.linalg.solve(A, B)

#     transform_matrix = torch.cat((transform_matrix, torch.tensor([1.0])), dim=0)
#     transform_matrix = transform_matrix.view(3, 3)
#     return transform_matrix


# def apply_perspective(img, matrix, w, h):
#     """
#     Apply the perspective transformation to an image using the computed matrix.
#     :param img: Input image tensor of shape (C, H, W)
#     :param matrix: Transformation matrix
#     :param w: Width of the output image
#     :param h: Height of the output image
#     :return: Transformed image tensor
#     """
#     # Invert the transformation matrix
#     matrix_inv = torch.inverse(matrix)

#     # Normalize the pixel coordinates to [-1, 1]
#     grid = torch.nn.functional.affine_grid(
#         matrix_inv[:2].unsqueeze(0), torch.Size((1, *img.shape)), align_corners=False
#     )

#     # Apply the transformation
#     return torch.nn.functional.grid_sample(
#         img.unsqueeze(0), grid, align_corners=False
#     ).squeeze(0)


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

        perspective_coeffs = _get_perspective_coeffs(
            startpoints=src_pts, endpoints=dst_pts
        )

        # Compute the perspective transform matrix
        # transform_matrix = find_perspective_transform(src_pts, dst_pts)

        # # Apply the transformation
        # warped_image = apply_perspective(src_image.squeeze(0).to(torch.float) / 255.0, transform_matrix, width, height)
        # warped_image = torch.clamp(warped_image * 255, min=0, max=255).to(torch.uint8)

        # warped_image = torchvision.transforms.functional.perspective(
        #     src_image, startpoints=src_pts, endpoints=dst_pts
        # )

        # ow, oh = src_image.shape[-1], img.shape[-2]
        ow = width
        oh = height
        dtype = src_image.dtype if torch.is_floating_point(src_image) else torch.float32
        grid = _perspective_grid(
            perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=src_image.device
        )
        warped_image = _apply_grid_transform(
            src_image, grid, mode="bilinear", fill=None
        )

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
        # plt.imshow(pil_image)
        plt.imshow(warped_image)
        # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying correctly
        plt.title("Warped Image")
        plt.show()

    if False:
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
    else:
        proceed_with_warp_cv2()


if __name__ == "__main__":
    main()

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import functional as TF

import hmlib.tracking_utils.visualization as vis
from hmlib.config import get_game_config, get_nested_value
from hmlib.hm_opts import hm_opts
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
    make_visible_image,
    pad_tensor_to_size_batched,
    resize_image,
)


def point_distance(pt0: torch.Tensor, pt1: torch.Tensor) -> torch.Tensor:
    return torch.norm(pt0 - pt1)


class Scoreboard(torch.nn.Module):

    def __init__(
        self,
        src_pts: torch.Tensor,
        dest_height: int,
        dest_width: int,
        dtype: torch.dtype,
        clip_box: Union[torch.Tensor, None] = None,
        device: Union[torch.device, None] = None,
        auto_aspect: bool = True,
    ):
        if not isinstance(src_pts, torch.Tensor):
            src_pts = torch.tensor(src_pts, dtype=torch.float)
        assert len(src_pts) == 4
        self._src_pts = src_pts.clone()
        if clip_box is not None:
            if not isinstance(clip_box, torch.Tensor):
                clip_box = torch.tensor(
                    clip_box, dtype=self._src_pts.dtype, device=self._src_pts.device
                )
            self._src_pts[:] -= clip_box[0:2]

        self._bbox_src = int_bbox(get_bbox(self._src_pts))

        self._src_pts[:, 0] -= self._bbox_src[0]
        self._src_pts[:, 1] -= self._bbox_src[1]

        src_width = self._bbox_src[2] - self._bbox_src[0]
        src_height = self._bbox_src[3] - self._bbox_src[1]

        if auto_aspect:
            w_across_top = point_distance(self._src_pts[0], self._src_pts[1])
            w_across_bottom = point_distance(self._src_pts[2], self._src_pts[3])
            h_left = point_distance(self._src_pts[3], self._src_pts[0])
            h_right = point_distance(self._src_pts[2], self._src_pts[3])
            w_avg = (w_across_top + w_across_bottom) / 2
            h_avg = (h_left + h_right) / 2
            aspect_ratio = w_avg / h_avg
            dest_width_new = int(float(dest_height) * aspect_ratio)
            dest_height_new = int(float(dest_width) / aspect_ratio)
            # Try whichever one changes the least
            if (
                abs(dest_width - dest_width_new) / dest_width
                < abs(dest_height - dest_height_new) / dest_height
            ):
                dest_width = dest_width_new
            else:
                dest_height = dest_height_new

        self._dest_width = dest_width
        self._dest_height = dest_height

        totw = max(self._dest_width, src_width)
        toth = max(self._dest_height, src_height)
        if totw > src_width or toth > src_height:
            ratio_w = totw / src_width
            ratio_h = toth / src_height
            self._dest_w = totw
            self._dest_h = toth
            self._src_pts[:, 0] *= ratio_w
            self._src_pts[:, 1] *= ratio_h
        else:
            self._dest_w = src_width
            self._dest_h = src_height

        dst_pts = np.array(
            [
                # TL
                [0, 0],
                # TR
                [dest_width - 1, 0],
                # BR
                [dest_width - 1, dest_height - 1],
                # BL
                [0, dest_height - 1],
            ],
            dtype=np.float32,
        )
        # print(src_pts)
        # print(dst_pts)
        perspective_coeffs = _get_perspective_coeffs(
            startpoints=self._src_pts, endpoints=dst_pts
        )

        ow = self._dest_w
        oh = self._dest_h

        self._grid = _perspective_grid(
            perspective_coeffs,
            ow=ow,
            oh=oh,
            dtype=dtype,
            device=torch.device("cpu") if device is None else device,
        )

    def forward(self, input_image: torch.Tensor):
        original_image = make_channels_first(input_image)
        src_image = original_image[
            :,
            :,
            self._bbox_src[1] : self._bbox_src[3],
            self._bbox_src[0] : self._bbox_src[2],
        ]
        src_image = resize_image(
            img=src_image, new_width=self._dest_w, new_height=self._dest_h
        )
        # cv2.imshow("src_image", make_visible_image(src_image[0]))
        # cv2.waitKey(0)

        warped_image = _apply_grid_transform(
            src_image, self._grid, mode="bilinear", fill=None
        )

        # cv2.imshow("src_image", make_visible_image(warped_image[0]))
        # cv2.waitKey(0)

        warped_image = warped_image[:, :, : self._dest_height, : self._dest_width]
        return warped_image

    @property
    def width(self):
        return self._dest_width

    @property
    def height(self):
        return self._dest_height


def get_bbox(point_list: Union[torch.Tensor, List[List[float]]]):
    if isinstance(point_list, list):
        points = torch.tensor(point_list)
    else:
        points = point_list
    mins = torch.min(points, dim=0)[0]
    maxs = torch.max(points, dim=0)[0]
    return torch.cat((mins, maxs), dim=0)


def int_bbox(bbox: torch.Tensor):
    bbox[:2] = torch.floor(bbox[:2])
    bbox[2:] = torch.ceil(bbox[2:])
    return bbox.to(torch.int32)


def _get_perspective_coeffs(
    startpoints: Union[torch.Tensor, List[List[int]]],
    endpoints: Union[torch.Tensor, List[List[int]]],
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
    # b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
    b_matrix = startpoints.view(8)
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


def _cast_squeeze_in(
    img: Tensor, req_dtypes: List[torch.dtype]
) -> Tuple[Tensor, bool, bool, torch.dtype]:
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


def _cast_squeeze_out(
    img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype
) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
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
        mask = torch.ones(
            (img.shape[0], 1, img.shape[2], img.shape[3]),
            dtype=img.dtype,
            device=img.device,
        )
        img = torch.cat((img, mask), dim=1)

    img = F.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        fill_list, len_fill = (
            (fill, len(fill)) if isinstance(fill, (tuple, list)) else ([float(fill)], 1)
        )
        fill_img = (
            torch.tensor(fill_list, dtype=img.dtype, device=img.device)
            .view(1, len_fill, 1, 1)
            .expand_as(img)
        )
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


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

    def proceed_with_warp_cv2():
        nonlocal original_image
        # print(selected_points)

        src_pts = np.array(selected_points, dtype=np.float32)

        src_height = original_image.shape[0]
        src_width = original_image.shape[1]
        # ar = src_width/src_height

        # width, height = image.size
        width = 200
        height = 100
        # width = src_width
        # height = src_height

        # to_pil = ToPILImage()
        # to_tensor = transforms.ToTensor()

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

            # to_tensor = transforms.ToTensor()
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

            # width = image_width(src_image)
            # height = image_height(src_image)

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
                # width = totw
                # height = toth
                dest_w = totw
                dest_h = toth
            else:
                dest_w = src_width
                dest_h = src_height

            # pil_image = to_pil(src_image.squeeze(0))
        else:
            # pil_image = to_pil(original_image)
            pass

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
        ow = dest_w
        oh = dest_h
        dtype = src_image.dtype if torch.is_floating_point(src_image) else torch.float
        grid = _perspective_grid(
            perspective_coeffs, ow=ow, oh=oh, dtype=dtype, device=src_image.device
        )
        warped_image = _apply_grid_transform(
            src_image, grid, mode="bilinear", fill=None
        )

        warped_image = warped_image[:, :, :height, :width]

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


# def sort_points_tl_tr_br_bl(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     # Calculate the centroid of the points
#     centroid = (
#         sum(x for x, _ in points) / len(points),
#         sum(y for _, y in points) / len(points),
#     )

#     # Function to calculate angle from centroid
#     def angle_from_centroid(point):
#         return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

#     # Sort points by angle from centroid
#     sorted_points = sorted(points, key=angle_from_centroid)

#     # To ensure the sorting starts from top-left, find the point with the smallest angle
#     # and make it the starting point
#     top_left_index = sorted_points.index(
#         min(sorted_points, key=lambda point: (angle_from_centroid(point), -point[1]))
#     )
#     sorted_points = sorted_points[top_left_index:] + sorted_points[:top_left_index]

#     return sorted_points


def sb_main(game_id: str):

    image_path = (
        f"{os.environ['HOME']}/Videos/{game_id}/s.png"  # Specify the path to your image
    )
    image = cv2.imread(image_path)
    image_tensor = make_channels_first(torch.from_numpy(image).unsqueeze(0))

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / ".." / ".." / "..")
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    selected_points = get_nested_value(
        game_config, "rink.scoreboard.perspective_polygon"
    )

    if selected_points:
        for pt in selected_points:
            print(pt)
        # colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 255)]
        # for i in range(len(selected_points)):
        #     if i == len(selected_points) - 1:
        #         image = vis.plot_line(
        #             image,
        #             selected_points[-1],
        #             selected_points[0],
        #             color=colors[i],
        #             thickness=i * 2 + 1,
        #         )
        #     else:
        #         image = vis.plot_line(
        #             image,
        #             selected_points[i],
        #             selected_points[i + 1],
        #             color=colors[i],
        #             thickness=i * 2 + 2,
        #         )
        # cv2.imshow("points", image)
        # cv2.waitKey(0)

        scoreboard = Scoreboard(
            src_pts=selected_points,
            dest_width=700,
            dest_height=300,
            dtype=(
                image_tensor.dtype
                if torch.is_floating_point(image_tensor)
                else torch.float
            ),
        )

        warped_image = scoreboard.forward(image_tensor.to(torch.float) / 255)

        warped_image = torch.clamp(warped_image * 255, min=0, max=255).to(torch.uint8)

        cv2.imshow("warped_image", make_channels_last(warped_image)[0].numpy())
        cv2.waitKey(0)


if __name__ == "__main__":
    # main()
    opts = hm_opts()
    args = opts.parse()
    sb_main(args.game_id)

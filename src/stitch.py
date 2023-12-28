"""
Experiments in stitching
"""
import os
import time
import argparse
import yaml
import numpy as np
from typing import Tuple
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from pathlib import Path
from hmlib.opts import opts
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.ui.mousing import draw_box_with_mouse
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.config import get_clip_box, get_game_config

from hmlib.stitch_synchronize import (
    configure_video_stitching,
)

from hmlib.datasets.dataset.stitching import (
    StitchDataset,
)

from hockeymom import core

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--lfo",
        "--left_frame_offset",
        default=None,
        type=float,
        help="Left frame offset",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of stitching workers"
    )
    parser.add_argument(
        "--project-file",
        "--project_file",
        default="autooptimiser_out.pto",
        type=str,
        help="Use project file as input to stitcher",
    )
    parser.add_argument(
        "--rfo",
        "--right_frame_offset",
        default=None,
        type=float,
        help="Right frame offset",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=str,
        help="Video directory to find 'left.mp4' and 'right.mp4'",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        type=str,
        help="Game ID",
    )
    return parser


def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    lfo: int = None,
    rfo: int = None,
    game_id: str = None,
    project_file_name: str = "my_project.pto",
    start_frame_number: int = 0,
    max_frames: int = None,
    output_stitched_video_file: str = os.path.join(".", "stitched_output.avi"),
):
    left_vid = BasicVideoInfo(os.path.join(dir_name, video_left))
    right_vid = BasicVideoInfo(os.path.join(dir_name, video_right))
    total_frames = min(left_vid.frame_count, right_vid.frame_count)
    print(f"Total possible stitched video frames: {total_frames}")

    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left,
        video_right,
        project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
    )

    # nona = core.HmNona(pto_project_file)
    # control_points = nona.get_control_points()
    # print(control_points)

    data_loader = StitchDataset(
        video_file_1=os.path.join(dir_name, video_left),
        video_file_2=os.path.join(dir_name, video_right),
        pto_project_file=pto_project_file,
        video_1_offset_frame=lfo,
        video_2_offset_frame=rfo,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
        num_workers=1,
        # remap_thread_count=10,
        # blend_thread_count=10,
        remap_thread_count=1,
        blend_thread_count=1,
        fork_workers=False,
        image_roi=get_clip_box(game_id=game_id, root_dir=ROOT_DIR),
    )

    frame_count = 0
    start = None

    dataset_timer = Timer()
    for i, _ in enumerate(data_loader):
        if i > 1:
            dataset_timer.toc()
        if i % 20 == 0:
            logger.info(
                "Dataset frame {} ({:.2f} fps)".format(
                    i, 1.0 / max(1e-5, dataset_timer.average_time)
                )
            )
            dataset_timer = Timer()

        frame_count += 1
        if i == 1:
            # draw_box_with_mouse(stitched_image, destroy_all_windows_after=True)
            start = time.time()
        dataset_timer.tic()

    if start is not None:
        duration = time.time() - start
        print(
            f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
        )
    return lfo, rfo


def map_with_interpolation():
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image

    # Load your source image and convert it to a PyTorch tensor
    source_image = Image.open("path_to_your_image.jpg")  # Replace with your image path
    transform = transforms.ToTensor()
    source_tensor = transform(source_image).unsqueeze(0)  # Add batch dimension

    # Assuming row_map and col_map are already defined PyTorch tensors
    # Normalize these tensors to [-1, 1] range as required by grid_sample
    h, w = source_tensor.shape[2], source_tensor.shape[3]
    row_map_normalized = (2.0 * row_map / (h - 1)) - 1  # Normalize to [-1, 1]
    col_map_normalized = (2.0 * col_map / (w - 1)) - 1  # Normalize to [-1, 1]

    # Create the grid for grid_sample
    grid = torch.stack((col_map_normalized, row_map_normalized), dim=-1).unsqueeze(
        0
    )  # Add batch dimension

    # Perform the grid sampling with bicubic interpolation
    remapped_tensor = F.grid_sample(
        source_tensor, grid, mode="bicubic", padding_mode="zeros", align_corners=False
    )

    # Convert destination tensor back to an image
    to_pil_image = transforms.ToPILImage()
    destination_image = to_pil_image(
        remapped_tensor.squeeze(0)
    )  # Remove batch dimension

    # Save or display the destination image
    destination_image.save("remapped_image_bicubic.jpg")
    # destination_image.show()  # Uncomment to display the image


# Function to pad tensor to the target size
def pad_tensor_to_size(tensor, target_width, target_height, pad_value):
    if len(tensor.shape) == 2:
        pad_height = target_height - tensor.size(0)
        pad_width = target_width - tensor.size(1)
    else:
        assert len(tensor.shape) == 3
        pad_height = target_height - tensor.size(1)
        pad_width = target_width - tensor.size(2)
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)
    padding = [0, pad_width, 0, pad_height]
    padded_tensor = F.pad(tensor, padding, "constant", pad_value)
    return padded_tensor


def pad_tensor_to_size_batched(tensor, target_width, target_height, pad_value):
    if len(tensor.shape) == 3:
        pad_height = target_height - tensor.size(1)
        pad_width = target_width - tensor.size(2)
    else:
        assert len(tensor.shape) == 4
        pad_height = target_height - tensor.size(2)
        pad_width = target_width - tensor.size(3)
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)
    padding = [0, pad_width, 0, pad_height]
    padded_tensor = F.pad(tensor, padding, "constant", pad_value)
    return padded_tensor


class ImageRemapper:
    UNMAPPED_PIXEL_VALUE = 65535

    def __init__(
        self,
        dir_name: str,
        basename: str,
        device: torch.device,
        source_hw: Tuple[int],
        interpolation: str = None,
        channels: int = 3,
    ):
        assert len(source_hw) == 2
        self._dir_name = dir_name
        self._basename = basename
        self._device = device
        self._interpolation = interpolation
        self._source_hw = source_hw
        self._channels = channels
        self._initialized = False

    def init(self):
        x_file = os.path.join(self._dir_name, f"{self._basename}_x.tif")
        y_file = os.path.join(self._dir_name, f"{self._basename}_y.tif")
        x_map = cv2.imread(x_file, cv2.IMREAD_ANYDEPTH)
        y_map = cv2.imread(y_file, cv2.IMREAD_ANYDEPTH)
        if x_map is None:
            raise AssertionError(f"Could not read mapping file: {x_file}")
        if y_map is None:
            raise AssertionError(f"Could not read mapping file: {y_file}")
        col_map = torch.from_numpy(x_map.astype(np.int64))
        row_map = torch.from_numpy(y_map.astype(np.int64))

        src_w = self._source_hw[1]
        src_h = self._source_hw[0]
        dest_w = col_map.shape[1]
        dest_h = col_map.shape[0]
        self._working_w = max(src_w, dest_w)
        self._working_h = max(src_h, dest_h)
        print(f"Padding tensors to size w={self._working_w}, h={self._working_h}")

        col_map = pad_tensor_to_size(
            col_map, self._working_w, self._working_h, self.UNMAPPED_PIXEL_VALUE
        )
        row_map = pad_tensor_to_size(
            row_map, self._working_w, self._working_h, self.UNMAPPED_PIXEL_VALUE
        )
        mask = torch.logical_or(
            row_map == self.UNMAPPED_PIXEL_VALUE, col_map == self.UNMAPPED_PIXEL_VALUE
        )
        # The 65536 will result in an invalid index, so set these to 0,0
        # and we'll get rid of them later with the mask after the image is remapped
        col_map[mask] = 0
        row_map[mask] = 0

        # Give the mask a channel dimension if necessary
        mask = mask.expand((self._channels, self._working_h, self._working_w))

        self._col_map = col_map.contiguous().to(self._device)
        self._row_map = row_map.contiguous().to(self._device)
        self._mask = mask.contiguous().to(self._device)

        self._initialized = True

    def remap(self, source_image: torch.tensor):
        assert self._initialized

        # Per frame code
        if source_image.device != self._device:
            source_image = source_image.to(self._device)
        source_tensor = pad_tensor_to_size_batched(
            source_image, self._working_w, self._working_h, 0
        )
        # if self._mask.shape != source_tensor.shape:
        #     self._mask = self._mask.expand_as(source_tensor)
        # Check if source tensor is a single channel or has multiple channels
        if len(source_tensor.shape) == 3:  # Single channel
            assert source_tensor.shape[1] == self._mask.shape[0]
            assert source_tensor.shape[2] == self._mask.shape[1]
            destination_tensor = source_tensor[self._row_map, self._col_map]
        elif len(source_tensor.shape) == 4:  # Multiple channels
            assert source_tensor.shape[2] == self._mask.shape[1]
            assert source_tensor.shape[3] == self._mask.shape[2]
            _, c, _, _ = source_tensor.shape
            destination_tensor = torch.empty_like(source_tensor)
            for i in range(c):
                destination_tensor[:, i] = source_tensor[
                    :, i, self._row_map, self._col_map
                ]

        destination_tensor[:, self._mask] = 0
        return destination_tensor


def read_frame_batch(cap: cv2.VideoCapture, batch_size: int):
    frame_list = []
    res, frame = cap.read()
    if not res or frame is None:
        raise StopIteration()
    assert frame.dtype == np.uint8
    frame_list.append(torch.from_numpy(frame.transpose(2, 0, 1)))
    for i in range(batch_size - 1):
        res, frame = cap.read()
        if not res or frame is None:
            raise StopIteration()
        frame_list.append(torch.from_numpy(frame.transpose(2, 0, 1)))
    tensor = torch.stack(frame_list)
    return tensor


def remap_image(
    video_file: str,
    dir_name: str,
    basename: str,
    interpolation: str = None,
    show: bool = False,
):
    cap = cv2.VideoCapture(os.path.join(dir_name, video_file))
    if not cap or not cap.isOpened():
        raise AssertionError(
            f"Could not open video file: {os.path.join(dir_name, video_file)}"
        )

    device = "cuda"
    batch_size = 8

    source_tensor = read_frame_batch(cap, batch_size=batch_size)

    remapper = ImageRemapper(
        dir_name=dir_name,
        basename=basename,
        device=device,
        source_hw=source_tensor.shape[-2:],
        channels=source_tensor.shape[1],
    )
    remapper.init()

    timer = Timer()
    frame_count = 0
    while True:
        destination_tensor = remapper.remap(source_image=source_tensor)
        destination_tensor = destination_tensor.detach().cpu()

        frame_count += 1
        if frame_count != 1:
            timer.toc()

        if frame_count % 20 == 0:
            logger.info(
                "Remapping: {:.2f} fps".format(
                    batch_size * 1.0 / max(1e-5, timer.average_time)
                )
            )
            if frame_count % 50 == 0:
                timer = Timer()

        if show:
            for i in range(len(destination_tensor)):
                cv2.imshow(
                    "mapped image", destination_tensor[i].permute(1, 2, 0).cpu().numpy()
                )
                cv2.waitKey(1)

        source_tensor = read_frame_batch(cap, batch_size=batch_size)
        timer.tic()


def main(args):
    video_left = "left.mp4"
    video_right = "right.mp4"

    remap_image(video_left, args.video_dir, "mapping_0000", show=True)

    # args.lfo = 15
    # args.rfo = 0
    # lfo, rfo = stitch_videos(
    #     args.video_dir,
    #     video_left,
    #     video_right,
    #     lfo=args.lfo,
    #     rfo=args.rfo,
    #     project_file_name=args.project_file,
    #     game_id=args.game_id,
    # )


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
    print("Done.")

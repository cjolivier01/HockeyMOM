"""
Remap an image given mapping png files (usually produced by hugin's nona,
which is usual;;ly base dupon some homography matrix)
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import hockeymom.core as core
from hmlib.log import get_root_logger
from hmlib.stitching.configure_stitching import get_image_geo_position
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.image import (
    make_channels_first,
    make_visible_image,
    pad_tensor_to_size_batched,
)
from hmlib.video.video_stream import VideoStreamReader

ROOT_DIR = os.getcwd()

logger = get_root_logger()

def make_parser():
    parser = argparse.ArgumentParser("Image Remapper")
    return parser


def read_frame_batch(
    video_iter,
    batch_size: int,
):
    frame_list = []
    frame = next(video_iter)
    assert frame.ndim == 4  # Must have batch dimension
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
        frame = make_channels_first(frame)
    if batch_size == 1:
        return frame
    frame_list.append(frame)
    for i in range(batch_size - 1):
        frame = next(video_iter)
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
            frame = make_channels_first(frame)
        frame_list.append(frame)
    tensor = torch.cat(frame_list, dim=0)
    return tensor


def create_remapper_config(
    dir_name: str,
    basename: str,
    image_index: int,
    batch_size: int,
    source_hw: Tuple[int],
    device: str,
    interpolation: str = "bilinear",
) -> core.RemapperConfig:
    x_file = os.path.join(dir_name, f"{basename}000{image_index}_x.tif")
    y_file = os.path.join(dir_name, f"{basename}000{image_index}_y.tif")
    x_map = cv2.imread(x_file, cv2.IMREAD_ANYDEPTH)
    y_map = cv2.imread(y_file, cv2.IMREAD_ANYDEPTH)
    if x_map is None:
        raise AssertionError(f"Could not read mapping file: {x_file}")
    if y_map is None:
        raise AssertionError(f"Could not read mapping file: {y_file}")
    config = core.RemapperConfig()
    config.src_height = source_hw[0]
    config.src_width = source_hw[1]
    config.x_pos, config.y_pos = get_image_geo_position(
        os.path.join(dir_name, f"{basename}000{image_index}.tif")
    )
    config.device = str(device)
    config.col_map = torch.from_numpy(x_map.astype(np.int64))
    config.row_map = torch.from_numpy(y_map.astype(np.int64))
    config.batch_size = batch_size
    config.interpolation = interpolation
    return config


class RemapImageInfoEx(core.RemapImageInfo):
    def __init__(self, *args, xpos: int = None, ypos: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.xpos: int = xpos
        self.ypos: int = ypos


class ImageRemapper(torch.jit.ScriptModule):
    UNMAPPED_PIXEL_VALUE = 65535

    def __init__(
        self,
        source_hw: Tuple[int] = None,
        dir_name: str = None,
        basename: str = None,
        remap_info: RemapImageInfoEx = None,
        interpolation: str = None,
        channels: int = 3,
        use_cpp_remap_op: bool = False,
        debug: bool = False,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        assert source_hw is None or len(source_hw) == 2
        self._use_cpp_remap_op = use_cpp_remap_op
        self._debug = debug
        self._dir_name = dir_name
        self._basename = basename
        self._interpolation = interpolation
        self._source_hw = source_hw
        self._channels = channels
        self._alpha_channel = None
        self._grid = None
        self._initialized = False
        self._remap_op = None
        self._remap_op_device = "cpu"
        self._remap_info = remap_info
        self.xpos, self.ypos = None, None
        self._dest_w, self._dest_h = None, None
        self._working_w, self._working_h = None, None
        self._col_map, self._row_map, self._mask = None, None, None
        self._batch_size = batch_size
        if self._remap_info is not None and not self._source_hw:
            self._source_hw = [self._remap_info.src_height, self._remap_info.src_width]
        if self._batch_size is not None:
            self.init(batch_size=self._batch_size)

    # @property
    # def device(self) -> torch.device:
    #     elif self._remap_op is not None:
    #         return self._remap_op_device
    #     return self._mask.device

    def init(self, batch_size: int):
        if self._remap_info is not None:
            self.xpos = self._remap_info.xpos
            self.ypos = self._remap_info.ypos
            col_map = self._remap_info.col_map
            row_map = self._remap_info.row_map
        else:
            assert self._dir_name is not None
            assert self._basename
            x_file = os.path.join(self._dir_name, f"{self._basename}_x.tif")
            y_file = os.path.join(self._dir_name, f"{self._basename}_y.tif")
            self.xpos, self.ypos = get_image_geo_position(
                os.path.join(self._dir_name, f"{self._basename}.tif")
            )

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

        if self._use_cpp_remap_op:
            self._remap_op = core.ImageRemapper(
                src_width=src_w,
                src_height=src_h,
                col_map=col_map,
                row_map=row_map,
                dtype=torch.float32,
                interpolation=self._interpolation,
            )
            self._remap_op.init(batch_size=batch_size)
        else:
            assert col_map.shape == row_map.shape
            self._dest_w = col_map.shape[1]
            self._dest_h = col_map.shape[0]

            self._working_w = max(src_w, self._dest_w)
            self._working_h = max(src_h, self._dest_h)
            # print(f"Padding tensors to size w={self._working_w}, h={self._working_h}")

            col_map = pad_tensor_to_size_batched(
                col_map.unsqueeze(0),
                self._working_w,
                self._working_h,
                self.UNMAPPED_PIXEL_VALUE,
            ).squeeze(0)
            row_map = pad_tensor_to_size_batched(
                row_map.unsqueeze(0),
                self._working_w,
                self._working_h,
                self.UNMAPPED_PIXEL_VALUE,
            ).squeeze(0)
            mask = torch.logical_or(
                row_map == self.UNMAPPED_PIXEL_VALUE,
                col_map == self.UNMAPPED_PIXEL_VALUE,
            )

            # The 65536 will result in an invalid index, so set these to 0,0
            # and we'll get rid of them later with the mask after the image is remapped
            col_map[mask] = 0
            row_map[mask] = 0

            if self._interpolation:
                row_map_normalized = (
                    2.0 * row_map / (self._working_h - 1)
                ) - 1  # Normalize to [-1, 1]
                col_map_normalized = (
                    2.0 * col_map / (self._working_w - 1)
                ) - 1  # Normalize to [-1, 1]

                # Create the grid for grid_sample
                grid = torch.stack((col_map_normalized, row_map_normalized), dim=-1)
                self._grid = grid.expand((batch_size, *grid.shape))

            # Give the mask a channel dimension if necessary
            # mask = mask.expand((self._channels, self._working_h, self._working_w))

            self._col_map = col_map.contiguous()
            self._row_map = row_map.contiguous()
            self._mask = mask.contiguous()

        # Done.
        self._initialized = True

    @property
    def width(self):
        assert self._initialized
        return self._dest_w

    @property
    def height(self):
        return self._dest_h

    def to(self, device: torch.device):
        dev = device
        if self._use_cpp_remap_op:
            self._remap_op_device = dev
            self._remap_op.to(dev)
        else:
            self._col_map = self._col_map.to(dev)
            self._row_map = self._row_map.to(dev)
            self._mask = self._mask.to(dev)
            if self._grid is not None:
                self._grid = self._grid.to(dev)

    @torch.jit.script_method
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self._initialized
        source_image: torch.Tensor = data["img"]
        # make sure channel is where we expect it to be
        assert source_image.shape[1] in [3, 4]
        if not isinstance(source_image, torch.Tensor):
            source_image = torch.from_numpy(source_image)

        if self._use_cpp_remap_op:
            assert self._remap_op is not None
            return self._remap_op.remap(source_image)

        # Per frame code
        source_tensor = pad_tensor_to_size_batched(
            source_image, self._working_w, self._working_h, 0
        )
        # Check if source tensor is a single channel or has multiple channels
        if len(source_tensor.shape) == 3:  # Single channel
            destination_tensor = source_tensor[:, self._row_map, self._col_map]
        else:  # Multiple channels
            assert len(source_tensor.shape) == 4
            if not self._interpolation:
                destination_tensor = torch.empty_like(source_tensor)
                destination_tensor[:, :] = source_tensor[:, :, self._row_map, self._col_map]
            else:
                # Perform the grid sampling with bicubic interpolation
                destination_tensor = F.grid_sample(
                    source_tensor.to(torch.float),
                    self._grid,
                    mode=self._interpolation,
                    padding_mode="zeros",
                    align_corners=False,
                )
                destination_tensor = destination_tensor.clamp(min=0, max=255.0).to(torch.uint8)
        destination_tensor[:, :, self._mask] = 0

        # Clip to the original size that was specified
        destination_tensor = destination_tensor[:, :, : self._dest_h, : self._dest_w]
        data["img"] = destination_tensor
        return data


def remap_video(
    opts: argparse.Namespace,
    video_file: str,
    dir_name: str,
    basename: str,
    interpolation: str = None,
    show: bool = False,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
):
    cap = VideoStreamReader(os.path.join(dir_name, video_file), type="cv2")
    if not cap or not cap.isOpened():
        raise AssertionError(f"Could not open video file: {os.path.join(dir_name, video_file)}")
    video_iter = iter(cap)
    source_tensor = read_frame_batch(video_iter, batch_size=batch_size)

    remapper = ImageRemapper(
        dir_name=dir_name,
        basename=basename,
        source_hw=source_tensor.shape[-2:],
        channels=source_tensor.shape[1],
        interpolation=interpolation,
        batch_size=batch_size,
    )
    remapper.to(device=device)

    timer = Timer()
    frame_count = 0
    while True:
        with torch.no_grad():
            destination_tensor = remapper(dict(img=source_tensor))["img"]
            destination_tensor = destination_tensor.detach().contiguous().cpu()

        frame_count += 1
        if frame_count != 1:
            timer.toc()

        if frame_count % 20 == 0:
            logger.info(
                "Remapping: {:.2f} fps".format(batch_size * 1.0 / max(1e-5, timer.average_time))
            )
            if frame_count % 50 == 0:
                timer = Timer()

        if show:
            for this_image in destination_tensor:
                cv2.imshow(
                    "mapped image",
                    make_visible_image(this_image, enable_resizing=opts.show_scaled),
                )
                cv2.waitKey(1)

        source_tensor = read_frame_batch(video_iter, batch_size=batch_size)
        timer.tic()


def main(args):
    remap_video(
        args,
        "GX010094.MP4",
        args.video_dir,
        "mapping_0000",
        interpolation="bilinear",
        show=False,
        device=torch.device("cpu"),
    )


if __name__ == "__main__":
    from hmlib.hm_opts import hm_opts

    args = hm_opts.parser(make_parser()).parse_args()

    if not args.video_dir:
        args.video_dir = "/mnt/ripper-data/Videos/ev-stockton-ss"

    main(args)
    print("Done.")

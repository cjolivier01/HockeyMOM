"""
Experiments in stitching
"""
import os
import time
import argparse
import numpy as np
from typing import Tuple
import cv2

import torch
import torch.nn.functional as F

from hmlib.stitch_synchronize import get_image_geo_position
from hmlib.async_worker import AsyncWorker

import hockeymom.core as core

ROOT_DIR = os.getcwd()


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

        self.duration = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.duration = 0.0


def make_parser():
    parser = argparse.ArgumentParser("Image Remapper")
    parser.add_argument(
        "--project-file",
        "--project_file",
        default="autooptimiser_out.pto",
        type=str,
        help="Use project file as input to stitcher",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=str,
        help="Video directory to find 'left.mp4' and 'right.mp4'",
    )
    return parser


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


def create_remapper_config(
    dir_name: str,
    basename: str,
    image_index: int,
    batch_size: int,
    source_hw: Tuple[int],
    device: str,
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
    return config


class ImageRemapper:
    UNMAPPED_PIXEL_VALUE = 65535

    def __init__(
        self,
        dir_name: str,
        basename: str,
        source_hw: Tuple[int],
        interpolation: str = None,
        channels: int = 3,
        add_alpha_channel: bool = False,
        fake_remapping: bool = False,
        use_cpp_remap_op: bool = True,
    ):
        assert len(source_hw) == 2
        self._use_cpp_remap_op = use_cpp_remap_op
        self._dir_name = dir_name
        self._basename = basename
        self._interpolation = interpolation
        self._source_hw = source_hw
        self._channels = channels
        self._add_alpha_channel = add_alpha_channel
        self._alpha_channel = None
        self._grid = None
        self._fake_remapping = fake_remapping
        self._initialized = False
        self._remap_op = None
        self._remap_op_device = "cpu"

    @property
    def device(self):
        if self._fake_remapping:
            return "cpu"
        elif self._remap_op is not None:
            return self._remap_op_device
        return self._mask.device

    def init(self, batch_size: int):
        x_file = os.path.join(self._dir_name, f"{self._basename}_x.tif")
        y_file = os.path.join(self._dir_name, f"{self._basename}_y.tif")
        self.xpos, self.ypos = get_image_geo_position(
            os.path.join(self._dir_name, f"{self._basename}.tif")
        )

        if self._fake_remapping:
            self._initialized = True
            return

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
                src_w,
                src_h,
                col_map,
                row_map,
                self._add_alpha_channel,
                self._interpolation,
            )
            self._remap_op.init(batch_size=batch_size)
        else:
            self._dest_w = col_map.shape[1]
            self._dest_h = col_map.shape[0]
            self._working_w = max(src_w, self._dest_w)
            self._working_h = max(src_h, self._dest_h)
            print(f"Padding tensors to size w={self._working_w}, h={self._working_h}")

            col_map = pad_tensor_to_size(
                col_map, self._working_w, self._working_h, self.UNMAPPED_PIXEL_VALUE
            )
            row_map = pad_tensor_to_size(
                row_map, self._working_w, self._working_h, self.UNMAPPED_PIXEL_VALUE
            )
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

            if self._add_alpha_channel:
                # Set up the alpha channel
                self._alpha_channel = torch.empty(
                    size=(batch_size, 1, self._working_h, self._working_w),
                    dtype=torch.uint8,
                )
                self._alpha_channel.fill_(255)
                self._alpha_channel[:, :, self._mask] = 0

        # Done.
        self._initialized = True

    def to(self, device: torch.device):
        if self._fake_remapping:
            return
        dev = str(device)
        if self._use_cpp_remap_op:
            self._remap_op_device = dev
            self._remap_op.to(dev)
        else:
            self._col_map = self._col_map.to(dev)
            self._row_map = self._row_map.to(dev)
            self._mask = self._mask.to(dev)
            if self._grid is not None:
                self._grid = self._grid.to(dev)
            if self._add_alpha_channel:
                self._alpha_channel = self._alpha_channel.to(dev)

    def forward(self, source_image: torch.tensor):
        assert self._initialized
        # make sure channel is where we expect it to be
        assert source_image.shape[1] in [3, 4]
        if not isinstance(source_image, torch.Tensor):
            source_image = torch.from_numpy(source_image)
        if self._fake_remapping:
            return source_image.clone()

        if source_image.device != self.device:
            source_image = source_image.to(self.device)

        if self._use_cpp_remap_op:
            assert self._remap_op is not None
            return self._remap_op.remap(source_image)

        # Per frame code
        source_tensor = pad_tensor_to_size_batched(
            source_image, self._working_w, self._working_h, 0
        )
        # Check if source tensor is a single channel or has multiple channels
        if len(source_tensor.shape) == 3:  # Single channel
            destination_tensor[:] = source_tensor[:, self._row_map, self._col_map]
        elif len(source_tensor.shape) == 4:  # Multiple channels
            if not self._interpolation:
                destination_tensor = torch.empty_like(source_tensor)
                destination_tensor[:, :] = source_tensor[
                    :, :, self._row_map, self._col_map
                ]
            else:
                # Perform the grid sampling with bicubic interpolation
                destination_tensor = F.grid_sample(
                    source_tensor.to(torch.float32),
                    self._grid,
                    mode=self._interpolation,
                    padding_mode="zeros",
                    align_corners=False,
                )
                destination_tensor = destination_tensor.clamp(min=0, max=255.0).to(
                    torch.uint8
                )
        destination_tensor[:, :, self._mask] = 0

        # Add an alpha channel if necessary
        if self._add_alpha_channel:
            destination_tensor = torch.cat(
                (destination_tensor, self._alpha_channel), dim=1
            )

        # Clip to the original size that was specified
        destination_tensor = destination_tensor[:, :, : self._dest_h, : self._dest_w]
        return destination_tensor


class RemappedPair:
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.image_1 = None
        self.image_2 = None


class PairCallback:
    def __init__(self, callback: callable):
        self._callback = callback
        self._data = dict()

    def _deliver(self, frame_id: int):
        rp = self._data[frame_id]
        del self._data[frame_id]
        self._callback(rp)

    def aggregate_callback_1(self, item):
        if isinstance(item, tuple):
            frame_id, remapped_image = item
            if frame_id not in self._data:
                rp = RemappedPair(frame_id=frame_id)
                rp.image_1 = remapped_image
                self._data[frame_id] = rp
            else:
                rp = self._data[frame_id]
                assert rp.image_1 is None
                assert rp.frame_id == frame_id
                self._data[frame_id].image_1 = remapped_image
                self._deliver(frame_id=frame_id)
        else:
            assert False and "Implement me"

    def aggregate_callback_2(self, item):
        if isinstance(item, tuple):
            frame_id, remapped_image = item
            if frame_id not in self._data:
                rp = RemappedPair(frame_id=frame_id)
                rp.image_2 = remapped_image
                self._data[frame_id] = rp
            else:
                rp = self._data[frame_id]
                assert rp.image_2 is None
                assert rp.frame_id == frame_id
                self._data[frame_id].image_2 = remapped_image
                self._deliver(frame_id=frame_id)
        else:
            assert False and "Implement me"


class AsyncRemapperWorker(AsyncWorker):
    def __init__(self, image_remapper: ImageRemapper, pair_callback: callable):
        super(AsyncRemapperWorker, self).__init__(
            name="AsyncRemapperWorker", callback=pair_callback
        )
        self._image_remapper = image_remapper
        self.frame_ids = []

    @property
    def xy_pos(self):
        return [self._image_remapper.xpos, self._image_remapper.ypos]

    @property
    def device(self):
        return self._image_remapper.device

    def init(self, batch_size: int):
        self._image_remapper.init(batch_size)

    def to(self, device: torch.device):
        self._image_remapper.to(device)
        return self

    def start(self, batch_size: int):
        self._batch_size = batch_size
        super(AsyncRemapperWorker, self).start()

    def run(self, **kwargs):
        # self._image_remapper.init(self._batch_size)
        try:
            while True:
                msg = self._incoming_queue.get()
                if msg is None:
                    break
                if isinstance(msg, Exception):
                    raise msg
                remapped_image = self._image_remapper.forward(source_image=msg)
                self.deliver_item(remapped_image)
        except Exception as e:
            self.deliver_item(e)
            raise
        finally:
            try:
                self.deliver_item(None)
            except:
                pass

    def send(self, frame_id: int, source_tensor: torch.Tensor):
        self.frame_ids.append(frame_id)
        self._incoming_queue.put(source_tensor)

    def deliver_item(self, remapped_image):
        if remapped_image is None:
            raise StopIteration()
        elif isinstance(remapped_image, Exception):
            raise remapped_image
        assert len(self.frame_ids) != 0
        frame_id = self.frame_ids[0]
        del self.frame_ids[0]
        super(AsyncRemapperWorker, self).deliver_item((frame_id, remapped_image))


def read_frame_batch(cap: cv2.VideoCapture, batch_size: int):
    frame_list = []
    res, frame = cap.read()
    if not res or frame is None:
        raise StopIteration()
    frame_list.append(torch.from_numpy(frame.transpose(2, 0, 1)))
    for i in range(batch_size - 1):
        res, frame = cap.read()
        if not res or frame is None:
            raise StopIteration()
        frame_list.append(torch.from_numpy(frame.transpose(2, 0, 1)))
    tensor = torch.stack(frame_list)
    return tensor


def remap_video(
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
    batch_size = 1

    source_tensor = read_frame_batch(cap, batch_size=batch_size)

    remapper = ImageRemapper(
        dir_name=dir_name,
        basename=basename,
        source_hw=source_tensor.shape[-2:],
        channels=source_tensor.shape[1],
        interpolation=interpolation,
    )
    remapper.init(batch_size=batch_size)
    remapper.to(device=device)

    timer = Timer()
    frame_count = 0
    while True:
        destination_tensor = remapper.forward(source_image=source_tensor)
        destination_tensor = destination_tensor.detach().contiguous().cpu()

        frame_count += 1
        if frame_count != 1:
            timer.toc()

        if frame_count % 20 == 0:
            print(
                "Remapping: {:.2f} fps".format(
                    batch_size * 1.0 / max(1e-5, timer.average_time)
                )
            )
            if frame_count % 50 == 0:
                timer = Timer()

        if show:
            for i in range(len(destination_tensor)):
                cv2.imshow(
                    "mapped image", destination_tensor[i].permute(1, 2, 0).numpy()
                )
                cv2.waitKey(1)

        source_tensor = read_frame_batch(cap, batch_size=batch_size)
        timer.tic()


def main(args):
    remap_video(
        "left.mp4",
        args.video_dir,
        "mapping_0000",
        # interpolation="bilinear",
        interpolation="",
        show=False,
    )


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
    print("Done.")

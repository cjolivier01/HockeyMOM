"""
Experiments in stitching
"""

import argparse
import datetime
import os
from contextlib import nullcontext
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import hockeymom.core as core
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.hm_opts import copy_opts, hm_opts
from hmlib.stitching.laplacian_blend import show_image
from hmlib.stitching.remapper import read_frame_batch
from hmlib.stitching.synchronize import get_image_geo_position, synchronize_by_audio
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.gpu import (
    CachedIterator,
    GpuAllocator,
    StreamCheckpoint,
    StreamTensor,
    StreamTensorToDevice,
    StreamTensorToDtype,
)
from hmlib.utils.image import image_height, image_width
from hmlib.utils.utils import create_queue
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.video_out import (
    ImageProcData,
    VideoOutput,
    optional_with,
    resize_image,
    rotate_image,
)
from hmlib.video_stream import VideoStreamReader, VideoStreamWriter

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("Image Remapper")
    parser = hm_opts.parser(parser)
    parser.add_argument(
        "-b",
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-q",
        "--queue-size",
        dest="queue_size",
        default=1,
        type=int,
        help="Queue size",
    )
    parser.add_argument(
        "--start-frame-number",
        default=0,
        type=int,
        help="Start frame number",
    )
    return parser


class ImageAndPos:
    def __init__(self, image: torch.Tensor, xpos: int, ypos: int):
        self.image = image
        self.xpos = xpos
        self.ypos = ypos


def make_cv_compatible_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        assert tensor.dim() == 3
        if tensor.size(0) == 3 or tensor.size(0) == 4:
            # Need to make channels-last
            tensor = tensor.permute(1, 2, 0)
        return tensor.contiguous().cpu().numpy()
    if tensor.shape[0] == 3 or tensor.shape[0] == 4:
        tensor = tensor.transpose(1, 2, 0)
    return np.ascontiguousarray(tensor)


def get_dims_for_output_video(
    height: int, width: int, max_width: int, allow_resize: bool = True
):
    if allow_resize and max_width and width > max_width:
        hh = float(height)
        ww = float(width)
        ar = ww / hh
        new_h = float(max_width) / ar
        return int(new_h), int(max_width)
    return int(height), int(width)


def gpu_index(want: int = 1):
    return min(torch.cuda.device_count() - 1, want)


def stream_context(stream: Optional[torch.cuda.Stream]):
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


# def to_tensor(t, no_sync: bool = False):
#     if isinstance(t, StreamTensor):
#         if no_sync:
#             return t.ref()
#         return t.get()
#     if isinstance(t, np.ndarray):
#         t = torch.from_numpy(t)
#     if t.device != device:
#         t = t.to(device, non_blocking=False)
#     if t.dtype == torch.uint8:
#         t = t.to(dtype, non_blocking=False)
#     return t


def copy_video(
    video_file: str,
    dir_name: str,
    device: torch.device,
    output_device: torch.device,
    show: bool = False,
    start_frame_number: int = 0,
    output_video: str = None,
    batch_size: int = 8,
    skip_final_video_save: bool = False,
    queue_size: int = 1,
    dtype: torch.dtype = torch.float16,
):
    video_file = os.path.join(dir_name, video_file)

    video_info = BasicVideoInfo(video_file)

    dataloader = MOTLoadVideoWithOrig(
        img_size=None,
        path=video_file,
        original_image_only=True,
        start_frame_number=start_frame_number,
        batch_size=batch_size,
        max_frames=args.max_frames,
        device=device,
        dtype=dtype,
    )

    video_out = VideoOutput(
        name="VideoOutput",
        args=None,
        output_video_path=output_video,
        output_frame_width=video_info.width,
        output_frame_height=video_info.height,
        fps=video_info.fps,
        device=output_device,
        skip_final_save=skip_final_video_save,
        fourcc="auto",
    )

    main_stream = torch.cuda.Stream(device=device)

    v_iter = CachedIterator(
        iterator=iter(dataloader),
        cache_size=queue_size,
    )

    with stream_context(main_stream):
        io_timer = Timer()
        get_timer = Timer()
        batch_count = 0
        frame_id = start_frame_number
        frame_ids = list()
        for i in range(batch_size):
            frame_ids.append(i)
        frame_ids = torch.tensor(frame_ids, dtype=torch.int64, device=device)
        frame_ids = frame_ids + frame_id
        try:

            while True:
                io_timer.tic()
                source_tensor, _, _, _, frame_ids = next(v_iter)
                io_timer.toc()

                get_timer.tic()
                source_tensor = source_tensor.get()
                get_timer.toc()

                imgproc_info = ImageProcData(
                    frame_id=frame_ids, img=source_tensor, current_box=None
                )
                video_out.append(imgproc_info)

                batch_count += 1

                if batch_count % 50 == 0:
                    print(
                        "IO:        {:.2f} fps".format(
                            batch_size * 1.0 / max(1e-5, io_timer.average_time)
                        )
                    )
                    print(
                        "get():     {:.2f} fps".format(
                            batch_size * 1.0 / max(1e-5, get_timer.average_time)
                        )
                    )
                    if True and batch_count % 50 == 0:
                        io_timer = Timer()
                        get_timer = Timer()
        except StopIteration:
            print("Done.")
        finally:
            if video_out is not None:
                if isinstance(video_out, VideoStreamWriter):
                    video_out.flush()
                    video_out.close()
                else:
                    video_out.stop()


#
# Combined FPS= XY/(X+Y)
#


def main(args):
    # opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    gpu_allocator = GpuAllocator(gpus=args.gpus)
    if not args.video_dir and args.game_id:
        args.video_dir = os.path.join(os.environ["HOME"], "Videos", args.game_id)
    video_gpu = torch.device("cuda", gpu_allocator.allocate_modern())
    fast_gpu = torch.device("cuda", gpu_allocator.allocate_fast())
    with torch.no_grad():
        copy_video(
            video_file="left.mp4",
            dir_name=args.video_dir,
            start_frame_number=args.start_frame_number,
            show=args.show_image,
            output_video=args.output_file,
            output_device=video_gpu,
            batch_size=args.batch_size,
            skip_final_video_save=args.skip_final_video_save,
            queue_size=args.queue_size,
            device=fast_gpu,
            dtype=torch.float16 if args.fp16 else torch.float,
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    print("Done.")

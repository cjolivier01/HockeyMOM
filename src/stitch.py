"""
Experiments in stitching
"""

import os
import time
import argparse
import contextlib
import cv2
from typing import Any
from collections import OrderedDict

import torch

from hmlib.ffmpeg import BasicVideoInfo
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.config import get_clip_box
from hmlib.stitching.remapper import ImageRemapper
from hmlib.utils.gpu import StreamTensor, GpuAllocator, CachedIterator
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput
from hmlib.stitching.laplacian_blend import show_image
from hmlib.hm_opts import hm_opts, preferred_arg
from hmlib.stitching.synchronize import (
    configure_video_stitching,
)

if False:
    from hmlib.datasets.dataset.stitching_dataloader1 import (
        StitchDataset,
    )
else:
    from hmlib.datasets.dataset.stitching_dataloader2 import (
        StitchDataset,
    )

from hockeymom import core

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "--num-workers", default=1, type=int, help="Number of stitching workers"
    )
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    return parser


def convert_seconds_to_hms(total_seconds):
    hours = int(total_seconds // 3600)  # Calculate the number of hours
    minutes = int((total_seconds % 3600) // 60)  # Calculate the remaining minutes
    seconds = int(total_seconds % 60)  # Calculate the remaining seconds

    # Format the time in "HH:MM:SS" format
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def stitch_videos(
    dir_name: str,
    video_left: str = "left.mp4",
    video_right: str = "right.mp4",
    lfo: int = None,
    rfo: int = None,
    game_id: str = None,
    project_file_name: str = "my_project.pto",
    blend_mode: str = "multiblend",
    start_frame_number: int = 0,
    max_frames: int = None,
    batch_size: int = 1,
    show: bool = False,
    output_stitched_video_file: str = os.path.join(".", "stitched_output.mkv"),
    remapping_device: torch.device = torch.device("cuda", 0),
    encoder_device: torch.device = torch.device("cpu"),
    remap_on_async_stream: bool = True,
    ignore_clip_box: bool = True,
    cache_size: int = 4,
    dtype: torch.dtype = torch.float,
):
    if dir_name is None and game_id:
        dir_name = os.path.join(os.environ["HOME"], "Videos", game_id)
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

    data_loader = StitchDataset(
        video_file_1=os.path.join(dir_name, video_left),
        video_file_2=os.path.join(dir_name, video_right),
        pto_project_file=pto_project_file,
        video_1_offset_frame=lfo,
        video_2_offset_frame=rfo,
        start_frame_number=start_frame_number,
        output_stitched_video_file=output_stitched_video_file,
        max_frames=max_frames,
        batch_size=batch_size,
        num_workers=1,
        remap_thread_count=1,
        blend_thread_count=1,
        max_input_queue_size=cache_size,
        fork_workers=False,
        image_roi=(
            get_clip_box(game_id=game_id, root_dir=ROOT_DIR)
            if not ignore_clip_box
            else None
        ),
        encoder_device=encoder_device,
        blend_mode=blend_mode,
        remapping_device=remapping_device,
        remap_on_async_stream=remap_on_async_stream,
        dtype=dtype,
    )

    data_loader_iter = CachedIterator(iterator=iter(data_loader), cache_size=cache_size)

    frame_count = 0
    dataset_delivery_fps = 0.0

    use_progress_bar: bool = True
    scroll_output: Optional[ScrollOutput] = None

    if use_progress_bar:
        total_frame_count = len(data_loader)

        def _table_callback(table_map: OrderedDict):
            table_map["Stitching Dataset Delivery FPS"] = "{:.2f}".format(
                dataset_delivery_fps
            )
            if dataset_delivery_fps > 0:
                remaining_secs = (
                    total_frame_count - frame_count
                ) / dataset_delivery_fps
                table_map["Time Remaining"] = convert_seconds_to_hms(remaining_secs)

        scroll_output = ScrollOutput()

        scroll_output.register_logger(logger)

        progress_bar = ProgressBar(
            total=total_frame_count,
            iterator=data_loader_iter,
            scroll_output=scroll_output,
            update_rate=20,
            table_callback=_table_callback,
        )
        data_loader_iter = progress_bar

    try:
        # with progress_bar.stdout_redirect():
        with contextlib.redirect_stdout(scroll_output):
            # with contextlib.nullcontext():
            start = None

            dataset_timer = Timer()
            for i, stitched_image in enumerate(data_loader_iter):
                # if isinstance(stitched_image, StreamTensor):
                #     stitched_image._verbose = False
                #     stitched_image = stitched_image.get()

                if i > 1:
                    dataset_timer.toc()
                if (i + 1) % 20 == 0:
                    assert stitched_image.ndim == 4
                    dataset_delivery_fps = batch_size / max(
                        1e-5, dataset_timer.average_time
                    )
                    logger.info(
                        "Dataset frame {} ({:.2f} fps)".format(
                            i * batch_size,
                            batch_size / max(1e-5, dataset_timer.average_time),
                        )
                    )
                    if i % 100 == 0:
                        dataset_timer = Timer()

                frame_count += batch_size

                if show:
                    show_image("stitched_image", stitched_image, wait=False)

                if i == 1:
                    start = time.time()
                dataset_timer.tic()

        if start is not None:
            duration = time.time() - start
            print(
                f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
            )
    finally:
        data_loader.close()
    return lfo, rfo


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
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    basename_1: str,
    basename_2: str,
    interpolation: str = None,
    show: bool = False,
):
    cap_1 = cv2.VideoCapture(os.path.join(dir_name, video_file_1))
    if not cap_1 or not cap_1.isOpened():
        raise AssertionError(
            f"Could not open video file: {os.path.join(dir_name, video_file_1)}"
        )

    cap_2 = cv2.VideoCapture(os.path.join(dir_name, video_file_2))
    if not cap_2 or not cap_2.isOpened():
        raise AssertionError(
            f"Could not open video file: {os.path.join(dir_name, video_file_2)}"
        )

    blender = core.EnBlender()

    device = "cuda"
    batch_size = 1

    source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
    source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)

    remapper_1 = ImageRemapper(
        dir_name=dir_name,
        basename=basename_1,
        device=device,
        source_hw=source_tensor_1.shape[-2:],
        channels=source_tensor_1.shape[1],
        interpolation=interpolation,
    )
    remapper_1.init(batch_size=batch_size)

    remapper_2 = ImageRemapper(
        dir_name=dir_name,
        basename=basename_2,
        device=device,
        source_hw=source_tensor_2.shape[-2:],
        channels=source_tensor_2.shape[1],
        interpolation=interpolation,
    )
    remapper_2.init(batch_size=batch_size)

    timer = Timer()
    blend_timer = Timer()
    frame_count = 0
    while True:
        destination_tensor_1 = remapper_1.remap(source_image=source_tensor_1)
        destination_tensor_2 = remapper_2.remap(source_image=source_tensor_2)

        destination_tensor_1 = (
            destination_tensor_1.detach().permute(0, 2, 3, 1).contiguous().cpu()
        )
        destination_tensor_2 = (
            destination_tensor_2.detach().permute(0, 2, 3, 1).contiguous().cpu()
        )

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

        for i in range(len(destination_tensor_1)):
            blend_timer.tic()
            blended = blender.blend_images(
                left_image=destination_tensor_1[i],
                left_xy_pos=[remapper_1.xpos, remapper_1.ypos],
                right_image=destination_tensor_2[i],
                right_xy_pos=[remapper_2.xpos, remapper_2.ypos],
            )
            blend_timer.toc()

            if show:
                show_image("blended", blended, wait=False)

        if frame_count % 20 == 0:
            print(
                "Blending: {:.2f} fps".format(1.0 / max(1e-5, blend_timer.average_time))
            )
            if frame_count % 50 == 0:
                timer = Timer()

        source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
        source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)
        timer.tic()


def main(args):
    video_left = "left.mp4"
    video_right = "right.mp4"
    gpu_allocator = GpuAllocator(gpus=args.gpus.split(","))
    with torch.no_grad():
        if False:
            remap_video(
                video_left,
                video_right,
                args.video_dir,
                "mapping_0000",
                "mapping_0001",
                # interpolation="bicubic",
                show=True,
            )
        else:
            stitch_videos(
                args.video_dir,
                video_left,
                video_right,
                lfo=args.lfo,
                rfo=args.rfo,
                batch_size=args.batch_size,
                project_file_name=args.project_file,
                game_id=args.game_id,
                show=args.show_image,
                max_frames=args.max_frames,
                output_stitched_video_file=args.output_file,
                blend_mode=args.blend_mode,
                remap_on_async_stream=False,
                ignore_clip_box=True,
                cache_size=preferred_arg(args.stitch_cache_size, args.cache_size),
                remapping_device=torch.device("cuda", gpu_allocator.allocate_fast()),
                encoder_device=torch.device("cuda", gpu_allocator.allocate_modern()),
                dtype=torch.half if args.fp16 else torch.float,
            )


if __name__ == "__main__":
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()

    main(args)
    print("Done.")

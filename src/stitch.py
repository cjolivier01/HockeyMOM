"""
Experiments in stitching
"""

import argparse
import os
import time
from collections import OrderedDict
from typing import Any, Optional

import torch

from hmlib.config import get_clip_box
from hmlib.datasets.dataset.stitching_dataloader2 import StitchDataset
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.hm_opts import hm_opts, preferred_arg
from hmlib.stitching.laplacian_blend import show_image
from hmlib.stitching.synchronize import configure_video_stitching
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.gpu import GpuAllocator, StreamTensor
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput, convert_hms_to_seconds
from hockeymom import core

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "--num-workers", default=1, type=int, help="Number of stitching workers"
    )
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    parser.add_argument("--force", action="store_true", help="Force all recalcs")
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
    project_file_name: str = "hm_project.pto",
    blend_mode: str = "multiblend",
    start_frame_number: int = 0,
    max_frames: int = None,
    batch_size: int = 1,
    show: bool = False,
    output_stitched_video_file: str = os.path.join(".", "stitched_output.mkv"),
    decoder_device: Optional[torch.device] = None,
    remapping_device: torch.device = torch.device("cuda", 0),
    encoder_device: torch.device = torch.device("cpu"),
    remap_on_async_stream: bool = True,
    ignore_clip_box: bool = True,
    cache_size: int = 4,
    dtype: torch.dtype = torch.float,
    start_frame_time: Optional[str] = None,
    stitch_frame_time: Optional[str] = None,
    force: Optional[bool] = False,
    auto_adjust_exposure: Optional[bool] = False,
):
    if dir_name is None and game_id:
        dir_name = os.path.join(os.environ["HOME"], "Videos", game_id)
    left_vid = BasicVideoInfo(os.path.join(dir_name, video_left))
    right_vid = BasicVideoInfo(os.path.join(dir_name, video_right))
    total_frames = min(left_vid.frame_count, right_vid.frame_count)
    print(f"Total possible stitched video frames: {total_frames}")

    stitch_frame_number = 0
    if start_frame_time and not stitch_frame_time:
        stitch_frame_time = start_frame_time
    if stitch_frame_time:
        seconds = convert_hms_to_seconds(stitch_frame_time)
        if seconds > 0:
            stitch_frame_number = seconds * left_vid.fps

    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left,
        video_right,
        project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
        base_frame_offset=stitch_frame_number,
        force=force,
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
            get_clip_box(game_id=game_id, root_dir=ROOT_DIR) if not ignore_clip_box else None
        ),
        encoder_device=encoder_device,
        decoder_device=decoder_device,
        blend_mode=blend_mode,
        remapping_device=remapping_device,
        remap_on_async_stream=remap_on_async_stream,
        dtype=dtype,
        auto_adjust_exposure=auto_adjust_exposure,
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
        start = None

        dataset_timer = Timer()
        for i, stitched_image in enumerate(data_loader_iter):
            if not output_stitched_video_file and isinstance(
                stitched_image, StreamTensor
            ):
                stitched_image._verbose = False
                stitched_image = stitched_image.get()

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
    except StopIteration:
        pass
    finally:
        data_loader.close()
    return lfo, rfo


def main(args):
    video_left = "left.mp4"
    video_right = "right.mp4"
    gpu_allocator = GpuAllocator(gpus=args.gpus.split(","))
    assert not args.start_frame_offset
    with torch.no_grad():
        stitch_videos(
            args.video_dir,
            video_left,
            video_right,
            lfo=args.lfo,
            rfo=args.rfo,
            start_frame_time=args.start_frame_time,
            stitch_frame_time=args.stitch_frame_time,
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
            decoder_device=(torch.device(args.decoder_device) if args.decoder_device else None),
            dtype=torch.half if args.fp16 else torch.float,
            force=args.force,
            auto_adjust_exposure=args.stitch_auto_adjust_exposure,
        )


if __name__ == "__main__":
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()

    main(args)
    print("Done.")

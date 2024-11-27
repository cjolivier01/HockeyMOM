"""
Experiments in stitching
"""

import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from hmlib.config import get_clip_box
from hmlib.datasets.dataset.stitching_dataloader2 import StitchDataset
from hmlib.hm_opts import hm_opts, preferred_arg
from hmlib.log import get_root_logger
from hmlib.orientation import configure_game_videos
from hmlib.stitching.configure_stitching import configure_video_stitching
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import Shower
from hmlib.utils.gpu import GpuAllocator, StreamTensor
from hmlib.utils.image import image_height, image_width
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput, convert_hms_to_seconds
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_out import VideoOutput

ROOT_DIR = os.getcwd()

logger = get_root_logger()

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "--num-workers", default=1, type=int, help="Number of stitching workers"
    )
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    parser.add_argument("--force", action="store_true", help="Force all recalcs")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use multiple GPUs (probably slower, but if memory issues)",
    )
    return parser


def convert_seconds_to_hms(total_seconds):
    hours = int(total_seconds // 3600)  # Calculate the number of hours
    minutes = int((total_seconds % 3600) // 60)  # Calculate the remaining minutes
    seconds = int(total_seconds % 60)  # Calculate the remaining seconds

    # Format the time in "HH:MM:SS" format
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def stitch_videos(
    dir_name: str,
    videos: Dict[str, List[Path]],
    max_control_points: int,
    lfo: int = None,
    rfo: int = None,
    game_id: str = None,
    project_file_name: str = "hm_project.pto",
    blend_mode: str = "multiblend",
    start_frame_number: int = 0,
    max_frames: int = None,
    batch_size: int = 1,
    show: bool = False,
    show_scaled: Optional[float] = None,
    output_stitched_video_file: str = os.path.join(".", "stitched_output.mkv"),
    decoder_device: Optional[torch.device] = None,
    remapping_device: torch.device = torch.device("cuda", 0),
    encoder_device: torch.device = torch.device("cpu"),
    ignore_clip_box: bool = True,
    cache_size: int = 4,
    dtype: torch.dtype = torch.float,
    start_frame_time: Optional[str] = None,
    stitch_frame_time: Optional[str] = None,
    force: Optional[bool] = False,
    auto_adjust_exposure: Optional[bool] = False,
    minimize_blend: bool = True,
    python_blender: bool = False,
    async_output: bool = False,
):
    if dir_name is None and game_id:
        dir_name = os.path.join(os.environ["HOME"], "Videos", game_id)
    left_vid = BasicVideoInfo(",".join(videos["left"]))
    right_vid = BasicVideoInfo(",".join(videos["right"]))
    total_frames = min(left_vid.frame_count, right_vid.frame_count)
    print(f"Total possible stitched video frames: {total_frames}")

    stitch_frame_number = 0
    if start_frame_time and not stitch_frame_time:
        stitch_frame_time = start_frame_time
    if stitch_frame_time or start_frame_time:
        seconds = convert_hms_to_seconds(stitch_frame_time)
        if seconds > 0:
            stitch_frame_number = seconds * left_vid.fps
    if start_frame_time:
        assert not start_frame_number
        seconds = convert_hms_to_seconds(start_frame_time)
        if seconds > 0:
            start_frame_number = seconds * left_vid.fps

    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name,
        video_left=str(videos["left"][0]),
        video_right=str(videos["right"][0]),
        project_file_name=project_file_name,
        left_frame_offset=lfo,
        right_frame_offset=rfo,
        base_frame_offset=stitch_frame_number,
        max_control_points=max_control_points,
        force=force,
    )

    stitch_videos = {
        "left": {
            "files": videos["left"],
            "frame_offset": lfo,
        },
        "right": {
            "files": videos["right"],
            "frame_offset": rfo,
        },
    }

    data_loader = StitchDataset(
        pto_project_file=pto_project_file,
        videos=stitch_videos,
        start_frame_number=start_frame_number,
        max_frames=max_frames,
        batch_size=batch_size,
        num_workers=1,
        max_input_queue_size=cache_size,
        image_roi=(
            get_clip_box(game_id=game_id, root_dir=ROOT_DIR) if not ignore_clip_box else None
        ),
        decoder_device=decoder_device,
        blend_mode=blend_mode,
        remapping_device=remapping_device,
        dtype=dtype,
        auto_adjust_exposure=auto_adjust_exposure,
        minimize_blend=minimize_blend,
        python_blender=python_blender,
    )

    data_loader_iter = CachedIterator(iterator=iter(data_loader), cache_size=cache_size)

    frame_count = 0
    dataset_delivery_fps = 0.0

    use_progress_bar: bool = True
    scroll_output: Optional[ScrollOutput] = None

    shower = None
    if show:
        shower = Shower(
            label="stitched_image",
            show_scaled=show_scaled,
        )

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

    video_out = None

    def _maybe_save_frame(frame: torch.Tensor) -> None:
        nonlocal video_out
        if output_stitched_video_file and video_out is None:
            video_out = VideoOutput(
                args=None,
                output_video_path=output_stitched_video_file,
                output_frame_width=image_width(frame),
                output_frame_height=image_height(frame),
                fps=data_loader.fps,
                video_out_pipeline=None,
                max_queue_backlog=cache_size,
                device=encoder_device,
                simple_save=True,
                skip_final_save=False,
                original_clip_box=None,
                progress_bar=progress_bar,
                cache_size=cache_size,
                async_output=async_output,
            )
        if video_out is not None:
            video_out.append(dict(img=frame))

    try:
        start = None

        dataset_timer = Timer()
        for i, stitched_image in enumerate(data_loader_iter):
            if not output_stitched_video_file and isinstance(
                stitched_image, StreamTensor
            ):
                stitched_image._verbose = False
                stitched_image = stitched_image.get()

            _maybe_save_frame(frame=stitched_image)

            if shower is not None:
                shower.show(stitched_image)

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

            if i == 1:
                start = time.time()
            dataset_timer.tic()

            del stitched_image

        if start is not None:
            duration = time.time() - start
            print(
                f"{frame_count} frames in {duration} seconds ({(frame_count)/duration} fps)"
            )
    except StopIteration:
        pass
    finally:
        data_loader.close()
        if shower is not None:
            shower.close()
        if video_out is not None:
            video_out.stop()
    return lfo, rfo


def main(args):
    game_videos = configure_game_videos(game_id=args.game_id, force=args.force)
    gpu_allocator = GpuAllocator(gpus=args.gpus.split(","))
    assert not args.start_frame_offset
    remapping_device = torch.device("cuda", gpu_allocator.allocate_fast())
    if args.multi_gpu:
        encoder_device = torch.device("cuda", gpu_allocator.allocate_modern())
        decoder_device = (
            torch.device(args.decoder_device) if args.decoder_device else remapping_device
        )
    else:
        encoder_device, decoder_device = remapping_device, remapping_device
    if args.encoder_device:
        encoder_device = torch.device(args.encoder_device)
    with torch.no_grad():
        stitch_videos(
            args.video_dir,
            videos=game_videos,
            lfo=args.lfo,
            rfo=args.rfo,
            start_frame_time=args.start_frame_time,
            stitch_frame_time=args.stitch_frame_time,
            batch_size=args.batch_size,
            project_file_name=args.project_file,
            game_id=args.game_id,
            show=args.show_image,
            show_scaled=args.show_scaled,
            max_frames=args.max_frames,
            output_stitched_video_file=args.output_file,
            blend_mode=args.blend_mode,
            ignore_clip_box=True,
            cache_size=preferred_arg(args.stitch_cache_size, args.cache_size),
            remapping_device=remapping_device,
            decoder_device=decoder_device,
            encoder_device=encoder_device,
            dtype=torch.half if args.fp16 else torch.float,
            force=args.force,
            auto_adjust_exposure=args.stitch_auto_adjust_exposure,
            minimize_blend=not args.no_minimize_blend,
            python_blender=args.python_blender,
            max_control_points=args.max_control_points,
        )


if __name__ == "__main__":
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()

    main(args)
    print("Done.")

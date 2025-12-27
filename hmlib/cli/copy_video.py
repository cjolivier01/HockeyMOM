"""
For performance reasons, experiments in streaming and pipelining a simple copy
"""

import argparse
import contextlib
import os
import time
from typing import Any, Dict, List, Union

import numpy as np
import torch

from hmlib.config import get_game_dir
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.hm_opts import hm_opts
from hmlib.log import get_logger
from hmlib.orientation import configure_game_videos
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import Shower
from hmlib.utils.gpu import GpuAllocator, cuda_stream_scope, unwrap_tensor
from hmlib.utils.image import image_height, image_width
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.path import add_suffix_to_filename
from hmlib.utils.utils import calc_combined_fps
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_out import VideoOutput
from hmlib.video.video_stream import (
    VideoStreamWriterInterface,
    create_output_video_stream,
    time_to_frame,
)

ROOT_DIR = os.getcwd()

logger = get_logger(__name__)


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
        "--use-video-out",
        action="store_true",
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
        "--num-splits",
        default=1,
        type=int,
        help="Number of splits",
    )
    parser.add_argument(
        "--async-mode",
        default=1,
        type=int,
        help="Whether to load video frames asynchronously 0|1",
    )
    parser.add_argument(
        "--start-frame-number",
        default=0,
        type=int,
        help="Start frame number",
    )
    return parser


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


def get_dims_for_output_video(height: int, width: int, max_width: int, allow_resize: bool = True):
    if allow_resize and max_width and width > max_width:
        hh = float(height)
        ww = float(width)
        ar = ww / hh
        new_h = float(max_width) / ar
        return int(new_h), int(max_width)
    return int(height), int(width)


def gpu_index(want: int = 1):
    return min(torch.cuda.device_count() - 1, want)


def get_frame_split_points(total_number_of_frames: int, num_splits: int) -> List[int]:
    each = int(total_number_of_frames // num_splits)
    split_frames: List[int] = []
    for i in range(num_splits - 1):
        next_split = (i + 1) * each
        split_frames.append(next_split)
    return split_frames


def copy_video(
    video_file: Union[str, List[str]],
    device: torch.device,
    output_device: torch.device,
    show: bool = False,
    start_frame_number: int = 0,
    output_video: str = None,
    batch_size: int = 8,
    skip_final_video_save: bool = False,
    queue_size: int = 1,
    dtype: torch.dtype = torch.float16,
    use_video_out: bool = False,
    num_splits: int = 1,
    max_frames: int = 0,
    no_cuda_streams: bool = False,
    async_mode: bool = True,
    profiler: Any = None,
):
    main_stream = torch.cuda.Stream(device=device) if "cuda" in str(device) else None
    with cuda_stream_scope(main_stream):
        if isinstance(video_file, list):
            video_file = ",".join(video_file)
        video_info = BasicVideoInfo(video_file)

        dataloader = MOTLoadVideoWithOrig(
            path=video_file,
            original_image_only=True,
            start_frame_number=start_frame_number,
            batch_size=batch_size,
            max_frames=max_frames,
            device=device,
            decoder_device=device,
            dtype=dtype,
            no_cuda_streams=no_cuda_streams or not async_mode,
            async_mode=async_mode,
        )
        if profiler is not None and hasattr(dataloader, "set_profiler"):
            dataloader.set_profiler(profiler)

        split_number: int = 1
        split_frames = get_frame_split_points(len(dataloader), num_splits=num_splits)
        next_split_frame = split_frames[split_number - 1] if split_frames else float("inf")

        def _open_output_video(path: str) -> Union[VideoStreamWriterInterface, VideoOutput]:
            with cuda_stream_scope(main_stream):
                if not path:
                    return None
                if not use_video_out:
                    video_out = create_output_video_stream(
                        filename=path,
                        fps=video_info.fps,
                        width=video_info.width,
                        height=video_info.height,
                        codec="hevc_nvenc",
                        device=device,
                        bit_rate=video_info.bit_rate,
                        batch_size=batch_size,
                        profiler=profiler,
                    )
                    video_out.open()
                else:
                    video_out = VideoOutput(
                        name="VideoOutput",
                        output_video_path=path,
                        output_frame_width=video_info.width,
                        output_frame_height=video_info.height,
                        fps=video_info.fps,
                        skip_final_save=skip_final_video_save,
                        fourcc="auto",
                        device=output_device,
                        profiler=profiler,
                    )
                return video_out

        def _output_file_name(split_number: int) -> str:
            if not output_video:
                return None
            return (
                output_video
                if num_splits <= 1
                else str(add_suffix_to_filename(output_video, f"-{split_number}"))
            )

        video_out = _open_output_video(_output_file_name(split_number))
        if video_out is None:
            logger.info("Not saving the output video (no output path configured).")

        v_iter = CachedIterator(
            iterator=iter(dataloader),
            cache_size=queue_size,
        )

        io_timer = Timer()
        get_timer = Timer()
        batch_count = 0
        processed_frame_count: int = 0
        frame_id = start_frame_number
        frame_ids = list()
        for i in range(batch_size):
            frame_ids.append(i)
        frame_ids = torch.tensor(frame_ids, dtype=torch.int64, device=device)
        frame_ids = frame_ids + frame_id
        shower = (
            Shower(
                "copy_video",
                show_scaled=None if isinstance(show, bool) else show,
                profiler=profiler,
            )
            if show
            else None
        )
        prof = profiler
        prof_enabled = bool(prof is not None and getattr(prof, "enabled", False))
        run_ctx = prof if prof_enabled else contextlib.nullcontext()
        try:
            with run_ctx:
                while True:
                    iter_ctx = (
                        prof.rf("copy_video.iter") if prof_enabled else contextlib.nullcontext()
                    )
                    with iter_ctx:
                        all_fps = []
                        io_timer.tic()
                        inputs: Dict[str, Any] = next(v_iter)
                        io_timer.toc()

                        source_tensor = inputs["img"]
                        frame_id = inputs["frame_ids"]

                        get_timer.tic()
                        source_tensor = unwrap_tensor(source_tensor)
                        get_timer.toc()

                        batch_size = 1 if source_tensor.ndim == 3 else source_tensor.shape[0]

                        if video_out is not None:
                            if not use_video_out:
                                if processed_frame_count >= next_split_frame:
                                    if split_number >= len(split_frames):
                                        next_split_frame = float("inf")
                                    else:
                                        next_split_frame = split_frames[split_number]
                                    split_number += 1
                                    time.sleep(2)
                                    video_out.close()
                                    video_out = _open_output_video(
                                        _output_file_name(split_number)
                                    )

                                if torch.is_floating_point(source_tensor):
                                    source_tensor.clamp_(0, 255)
                                    source_tensor = source_tensor.to(torch.uint8, non_blocking=True)

                                if shower:
                                    shower.show(img=source_tensor)

                                # Raw VideoStreamWriter path (non-VideoOutput) keeps append().
                                video_out.append(source_tensor)
                            else:
                                # For VideoOutput, call the module directly to trigger forward().
                                if shower:
                                    shower.show(img=source_tensor)
                                video_out(
                                    {
                                        "frame_id": frame_ids,
                                        "img": source_tensor,
                                        "current_box": torch.tensor(
                                            [
                                                0,
                                                0,
                                                image_width(source_tensor),
                                                image_height(source_tensor),
                                            ],
                                            dtype=torch.float,
                                        ),
                                    }
                                )
                        else:
                            if shower:
                                shower.show(img=source_tensor)
                            # Synchronize the stream so that we report realistic frame rates
                            if source_tensor.device.type == "cuda":
                                torch.cuda.current_stream(source_tensor.device).synchronize()

                        batch_count += 1
                        processed_frame_count += batch_size

                        if batch_count % 50 == 0:
                            fps = batch_size * 1.0 / max(1e-5, io_timer.average_time)
                            all_fps.append(fps)
                            logger.info("IO:        %.2f fps", fps)
                            fps = batch_size * 1.0 / max(1e-5, get_timer.average_time)
                            all_fps.append(fps)
                            logger.info("get():     %.2f fps", fps)
                            if True and batch_count % 50 == 0:
                                io_timer = Timer()
                                get_timer = Timer()
                            logger.info("Combined fps: %.2f", calc_combined_fps(all_fps))
                    if prof_enabled:
                        prof.step()
        except StopIteration:
            logger.info("Done.")
        finally:
            if video_out is not None:
                if isinstance(video_out, VideoStreamWriterInterface):
                    time.sleep(2)
                    video_out.close()
                else:
                    video_out.stop()
            if shower is not None:
                shower.close()


#
# Combined FPS= XY/(X+Y)
#


def main():
    args = make_parser().parse_args()

    gpu_allocator = GpuAllocator(gpus=args.gpus)
    if not args.video_dir and args.game_id:
        args.video_dir = get_game_dir(args.game_id)
    else:
        args.video_dir = "."
    video_gpu = torch.device("cuda", gpu_allocator.allocate_modern())
    fast_gpu = torch.device("cuda", gpu_allocator.allocate_fast())

    if args.game_id:
        file_dict = configure_game_videos(
            game_id=args.game_id,
            force=False,
            write_results=False,
            inference_scale=getattr(args, "ice_rink_inference_scale", None),
        )
        if "left" in file_dict:
            video_files = file_dict["left"]
        elif "right" in file_dict:
            video_files = file_dict["right"]
    else:
        video_files = args.input_video
    if args.show_scaled:
        args.show_image = args.show_scaled
    if (getattr(args, "max_frames", None) in (None, 0)) and getattr(args, "max_time", None):
        try:
            info_path = ",".join(video_files) if isinstance(video_files, list) else video_files
            vid_info = BasicVideoInfo(info_path)
            if vid_info.fps and vid_info.fps > 0:
                args.max_frames = time_to_frame(time_str=args.max_time, fps=vid_info.fps)
                logger.info(
                    "Limiting processing to %s seconds -> %d frames (fps=%.3f)",
                    args.max_time,
                    args.max_frames,
                    vid_info.fps,
                )
        except Exception as exc:
            logger.warning("Failed converting max-time to frames: %s", exc)
    profiler = None
    try:
        from hmlib.utils.profiler import build_profiler_from_args

        results_folder = os.path.join(".", "output_workdirs", args.game_id or "copy_video")
        os.makedirs(results_folder, exist_ok=True)
        default_prof_dir = os.path.join(results_folder, "profiler")
        profiler = build_profiler_from_args(args, save_dir_fallback=default_prof_dir)
    except Exception:
        profiler = None
    setattr(args, "profiler", profiler)
    with torch.no_grad():
        copy_video(
            video_file=video_files,
            start_frame_number=args.start_frame_number,
            show=args.show_image,
            output_video=args.output_file,
            output_device=video_gpu,
            batch_size=args.batch_size,
            skip_final_video_save=args.skip_final_video_save,
            queue_size=args.queue_size,
            device=fast_gpu,
            dtype=torch.float16 if args.fp16 else torch.float,
            use_video_out=args.use_video_out,
            num_splits=args.num_splits,
            max_frames=args.max_frames,
            no_cuda_streams=getattr(args, "no_cuda_streams", False),
            async_mode=args.async_mode,
            profiler=profiler,
        )


if __name__ == "__main__":
    main()
    logger.info("Done.")

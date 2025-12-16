"""
Experiments in stitching
"""

import argparse
import contextlib
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from hmlib.aspen import AspenNet
import hmlib.transforms  # Register custom transforms for Aspen pipelines
from hmlib.config import get_clip_box
from hmlib.datasets.dataset.stitching_dataloader2 import StitchDataset
from hmlib.hm_opts import hm_opts, preferred_arg
from hmlib.log import get_root_logger
from hmlib.orientation import configure_game_videos
from hmlib.segm.ice_rink import main as ice_rink_main
from hmlib.stitching.configure_stitching import configure_video_stitching
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import Shower
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.image import image_height, image_width, resize_image
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput, convert_hms_to_seconds
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH
from hockeymom.core import show_cuda_tensor

ROOT_DIR = os.getcwd()

logger = get_root_logger()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("--num-workers", default=1, type=int, help="Number of stitching workers")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    parser.add_argument("--force", action="store_true", help="Force all recalcs")
    parser.add_argument(
        "--configure-only", action="store_true", help="Run stitching configuration only"
    )
    parser.add_argument(
        "--single-file",
        default=0,
        type=int,
        help="Only use a single video file from each perspective",
    )
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
    configure_only: bool = False,
    lowmem: bool = False,
    post_stitch_rotate_degrees: Optional[float] = None,
    args: Optional[argparse.Namespace] = None,
):
    from hmlib.config import load_yaml_files_ordered

    cuda_stream = torch.cuda.Stream(remapping_device)
    torch.cuda.synchronize()
    with torch.cuda.stream(cuda_stream):
        if configure_only:
            cache_size = 0
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

        profiler = getattr(args, "profiler", None)

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
            post_stitch_rotate_degrees=post_stitch_rotate_degrees,
            profiler=profiler,
            # no_cuda_streams=args.no_cuda_streams,
            no_cuda_streams=True,
            async_mode=False,
            max_blend_levels=getattr(args, "max_blend_levels", None),
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
                cache_on_cpu=lowmem,
            )

        if use_progress_bar and not configure_only:
            total_frame_count = len(data_loader)

            def _table_callback(table_map: OrderedDict):
                processed = frame_count
                remaining = max(0, total_frame_count - processed)
                table_map["Frames"] = f"{processed}/{total_frame_count}"
                if dataset_delivery_fps > 0:
                    remaining_secs = remaining / dataset_delivery_fps
                    eta = convert_seconds_to_hms(remaining_secs)
                    table_map["Stitch FPS"] = f"{dataset_delivery_fps:.2f}"
                    table_map["ETA"] = eta
                else:
                    table_map["Stitch FPS"] = "warming up"
                    table_map["ETA"] = "--:--:--"

            scroll_output = ScrollOutput()

            scroll_output.register_logger(logger)

            progress_bar = ProgressBar(
                total=total_frame_count,
                iterator=data_loader_iter,
                scroll_output=scroll_output,
                update_rate=20,
                table_callback=_table_callback,
                use_curses=True,
            )
            data_loader_iter = progress_bar

        # Build AspenNet-based video-output pipeline for stitching.
        # Load the stitching Aspen graph from YAML and wire in CLI-specific
        # parameters (output path, skip-final-save, frame dumping).
        aspen_cfg_all: Dict[str, Any] = load_yaml_files_ordered(["config/aspen/stitching.yaml"])
        aspen_graph_cfg: Dict[str, Any] = aspen_cfg_all.get("aspen", {}) or {}
        plugins_cfg: Dict[str, Any] = aspen_graph_cfg.get("plugins", {}) or {}
        video_out_spec: Dict[str, Any] = plugins_cfg.get("video_out", {}) or {}
        video_out_params: Dict[str, Any] = video_out_spec.get("params", {}) or {}
        if output_stitched_video_file:
            video_out_params.setdefault("output_video_path", output_stitched_video_file)
        if args is not None:
            if getattr(args, "skip_final_video_save", None):
                video_out_params["skip_final_save"] = True
            save_frame_dir = getattr(args, "save_frame_dir", None)
            if save_frame_dir:
                video_out_params.setdefault("save_frame_dir", save_frame_dir)
        video_out_spec["params"] = video_out_params
        plugins_cfg["video_out"] = video_out_spec
        aspen_graph_cfg["plugins"] = plugins_cfg

        # For stitching we want to preserve the full panorama resolution, so
        # disable cropping in the camera pipeline by default.
        cam_args = argparse.Namespace(
            crop_output_image=False,
            crop_play_box=False,
        )
        if args is not None:
            # Thread through basic display flags so VideoOutPlugin can honor them.
            setattr(cam_args, "show_image", bool(getattr(args, "show_image", False)))
            setattr(cam_args, "show_scaled", getattr(args, "show_scaled", None))
        aspen_shared: Dict[str, Any] = {"device": encoder_device, "cam_args": cam_args}
        if profiler is not None:
            aspen_shared["profiler"] = profiler
        aspen_name = game_id or "stitch"
        aspen_net = AspenNet(aspen_name, aspen_graph_cfg, shared=aspen_shared)
        aspen_net = aspen_net.to(encoder_device)

        try:
            start = None

            dataset_timer = Timer()
            with (
                (
                    profiler
                    if (profiler is not None and profiler.enabled)
                    else contextlib.nullcontext()
                ),
                torch.no_grad(),
            ):
                for i, stitched_image in enumerate(data_loader_iter):
                    if configure_only:
                        break
                    frame_ids = torch.arange(i * batch_size, (i + 1) * batch_size)

                    cuda_stream.synchronize()

                    # Downscale oversized panoramas to stay within encoder
                    # limits while preserving aspect ratio.
                    width = int(image_width(stitched_image))
                    height = int(image_height(stitched_image))
                    if width > MAX_NEVC_VIDEO_WIDTH:
                        scale = float(MAX_NEVC_VIDEO_WIDTH) / float(width)
                        new_w = MAX_NEVC_VIDEO_WIDTH
                        new_h = int(height * scale)
                        # Ensure even dimensions for encoders
                        if new_w % 2 != 0:
                            new_w -= 1
                        if new_h % 2 != 0:
                            new_h -= 1
                        stitched_image = resize_image(
                            stitched_image, new_width=new_w, new_height=new_h
                        )

                    # Execute the Aspen graph to handle camera cropping and
                    # video encoding via VideoOutPlugin.
                    context: Dict[str, Any] = {
                        "img": stitched_image,
                        "frame_ids": frame_ids,
                        "data": {"fps": data_loader.fps},
                        "game_id": game_id,
                    }
                    aspen_net(context)

                    # Per-iteration profiler step for gated profiling windows
                    if profiler is not None and getattr(profiler, "enabled", False):
                        profiler.step()

                    if shower is not None:
                        if False and stitched_image.device.type == "cuda":
                            for stitched_img in stitched_image:
                                show_cuda_tensor(
                                    "Stitched Image",
                                    stitched_img.clamp(min=0, max=255).to(torch.uint8),
                                    False,
                                    None,
                                )
                        else:
                            shower.show(stitched_image)

                    if i > 1:
                        dataset_timer.toc()
                    if (i + 1) % 20 == 0:
                        assert stitched_image.ndim == 4
                        dataset_delivery_fps = batch_size / max(1e-5, dataset_timer.average_time)
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
            try:
                aspen_net.finalize()
            except Exception:
                pass
    return lfo, rfo


def _main(args) -> None:
    game_videos = configure_game_videos(
        game_id=args.game_id,
        write_results=not args.single_file,
        force=args.force,
        inference_scale=getattr(args, "ice_rink_inference_scale", None),
    )

    HalfFloatType = torch.float16

    if args.fp16:
        torch.set_default_dtype(HalfFloatType)

    if args.single_file or args.configure_only:
        if "left" in game_videos and game_videos["left"]:
            game_videos["left"] = game_videos["left"][:1]
        if "right" in game_videos and game_videos["right"]:
            game_videos["right"] = game_videos["right"][:1]

    # If user specified max processing time (-t/--max-time), convert to frames
    # once FPS is known from input videos. Prefer explicit --max-frames when set.
    try:
        if (getattr(args, "max_frames", None) in (None, 0)) and getattr(args, "max_time", None):
            # Use left video FPS as reference for stitched stream
            left_vid = BasicVideoInfo(",".join(game_videos["left"]))
            seconds = convert_hms_to_seconds(args.max_time)
            if seconds > 0 and left_vid.fps > 0:
                args.max_frames = int(seconds * left_vid.fps)
                logger.info(
                    "Limiting processing to %s seconds -> %d frames (fps=%.3f)",
                    args.max_time,
                    args.max_frames,
                    left_vid.fps,
                )
    except Exception as e:
        logger.warning("Failed converting max-time to frames: %s", e)
    # Initialize lightweight profiler and attach to args for downstream use (same pattern as hmtrack.py)
    profiler = None
    try:
        from hmlib.utils.profiler import build_profiler_from_args

        # Use a per-game profiler directory under output_workdirs/<game_id>/profiler
        results_folder = os.path.join(".", "output_workdirs", args.game_id)
        os.makedirs(results_folder, exist_ok=True)
        default_prof_dir = os.path.join(results_folder, "profiler")
        profiler = build_profiler_from_args(args, save_dir_fallback=default_prof_dir)
    except Exception:
        profiler = None
    setattr(args, "profiler", profiler)

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
    if args.decoder_device:
        decoder_device = torch.device(args.decoder_device)
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
            dtype=HalfFloatType if args.fp16 else torch.float,
            force=args.force,
            auto_adjust_exposure=args.stitch_auto_adjust_exposure,
            minimize_blend=not args.no_minimize_blend,
            python_blender=args.python_blender,
            max_control_points=args.max_control_points,
            configure_only=args.configure_only,
            lowmem=gpu_allocator.is_single_lowmem_gpu(),
            post_stitch_rotate_degrees=getattr(args, "stitch_rotate_degrees", None),
            args=args,
        )

    if args.configure_only:
        # Configure the rink mask as well
        ice_rink_main(
            args,
            device=(
                decoder_device if not gpu_allocator.is_single_lowmem_gpu() else torch.device("cpu")
            ),
        )


def main() -> None:
    parser = hm_opts.parser(parser=make_parser())
    args = parser.parse_args()
    args = hm_opts.init(args, parser=parser)
    _main(args)


if __name__ == "__main__":
    main()
    print("Done.")

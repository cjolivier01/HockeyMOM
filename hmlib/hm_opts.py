from __future__ import absolute_import, division, print_function

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from hmlib.config import get_nested_value, set_nested_value


def copy_opts(src: object, dest: object, parser: argparse.ArgumentParser):
    fake_parsed = parser.parse_known_args()
    item_keys = sorted(fake_parsed[0].__dict__.keys())
    for item_name in item_keys:
        if hasattr(src, item_name):
            setattr(dest, item_name, getattr(src, item_name))
    return dest


class hm_opts(object):
    def __init__(self, parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        self._parser: argparse.ArgumentParser = self.parser(parser)

    @staticmethod
    def parser(parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cameraname",
            "--camera",
            dest="camera_name",
            default=None,
            type=str,
            help="Cameraname",
        )
        parser.add_argument(
            "--gpus", default="0,1,2", help="-1 for CPU, use comma for multiple gpus"
        )
        parser.add_argument("--debug", default=0, type=int, help="debug level")
        parser.add_argument(
            "--crop-play-box",
            default=None,
            type=int,
            help="Crop to play area only",
        )
        parser.add_argument(
            "--end-zones",
            action="store_true",
            help="Enable end-zone camera usage when available",
        )
        parser.add_argument(
            "--unsharp-mask",
            default=None,
            type=float,
            help="Apply unsharp masking to frame (good for blurry LiveBarn footage)",
        )
        # Input color adjustments (applied in inference pipeline via HmImageColorAdjust)
        parser.add_argument(
            "--white-balance",
            dest="white_balance",
            nargs=3,
            type=float,
            default=None,
            metavar=("R_GAIN", "G_GAIN", "B_GAIN"),
            help="Per-channel RGB gains for white balance (e.g., 1.05 1.0 0.95)",
        )
        parser.add_argument(
            "--color-brightness",
            dest="color_brightness",
            type=float,
            default=None,
            help="Brightness multiplier (>1 brighter). No-op if omitted.",
        )
        parser.add_argument(
            "--color-contrast",
            dest="color_contrast",
            type=float,
            default=None,
            help="Contrast factor (>1 more contrast). No-op if omitted.",
        )
        parser.add_argument(
            "--color-gamma",
            dest="color_gamma",
            type=float,
            default=None,
            help="Gamma exponent (>1 darker). No-op if omitted.",
        )
        parser.add_argument(
            "--deterministic",
            default=0,
            type=int,
            help="Whether we should try to be deterministic",
        )
        # Identity
        parser.add_argument(
            "--team",
            default=None,
            type=str,
            help="The primary team that represents the configuration file",
        )
        parser.add_argument(
            "--season",
            default=None,
            type=str,
            help="Season (if not the current)",
        )
        parser.add_argument(
            "--game-id",
            default=None,
            type=str,
            help="Game ID",
        )
        parser.add_argument(
            "--serial",
            default=0,
            type=int,
            help="Serial execution of entire pipeline",
        )
        # stitching
        parser.add_argument(
            "--cache-size",
            type=int,
            default=2,
            help="cache size for GPU stream async operations",
        )
        parser.add_argument(
            "--no-cuda-streams",
            action="store_true",
            help="Don't use CUDA streams",
        )
        aspen_thread_group = parser.add_mutually_exclusive_group()
        aspen_thread_group.add_argument(
            "--aspen-threaded",
            dest="aspen_threaded",
            action="store_true",
            help="Run Aspen trunks in threaded pipeline mode",
        )
        aspen_thread_group.add_argument(
            "--no-aspen-threaded",
            dest="aspen_threaded",
            action="store_false",
            help="Disable threaded Aspen pipeline mode",
        )
        parser.set_defaults(aspen_threaded=None)
        parser.add_argument(
            "--aspen-thread-queue-size",
            dest="aspen_thread_queue_size",
            type=int,
            default=None,
            help="Queue size between threaded Aspen trunks (defaults to 1)",
        )
        aspen_stream_group = parser.add_mutually_exclusive_group()
        aspen_stream_group.add_argument(
            "--aspen-thread-cuda-streams",
            dest="aspen_thread_cuda_streams",
            action="store_true",
            help="Give each threaded Aspen trunk its own CUDA stream",
        )
        aspen_stream_group.add_argument(
            "--no-aspen-thread-cuda-streams",
            dest="aspen_thread_cuda_streams",
            action="store_false",
            help="Disable per-trunk CUDA streams in threaded Aspen mode",
        )
        parser.set_defaults(aspen_thread_cuda_streams=None)
        parser.add_argument(
            "--stitch-cache-size",
            type=int,
            default=None,
            help="cache size for GPU stitching async operations",
        )
        parser.add_argument(
            "--fp16",
            default=False,
            action="store_true",
            help="show as processing",
        )
        parser.add_argument(
            "--fp16-stitch",
            default=False,
            action="store_true",
            help="Stitch images using fp16 (lower mem, but lower quality image output)",
        )
        parser.add_argument(
            "--show-image",
            "--show",
            dest="show_image",
            default=False,
            action="store_true",
            help="show as processing",
        )
        parser.add_argument(
            "--show-image-name",
            default="default",
            type=str,
            help="Name of the image to show, i.e. 'default', 'end_zones'",
        )
        parser.add_argument(
            "--show-scaled",
            type=float,
            default=None,
            help="scale showed image (ignored is --show-image is not specified)",
        )
        parser.add_argument(
            "--ice-rink-inference-scale",
            "--ice-rink-mask-scale",
            dest="ice_rink_inference_scale",
            type=float,
            default=None,
            help="Downscale factor for ice rink segmentation (e.g., 0.5 doubles speed, 1.0 keeps original size)",
        )
        parser.add_argument(
            "--decoder",
            "--video-stream-decode-method",
            dest="video_stream_decode_method",
            # default="ffmpeg",
            default="cv2",
            type=str,
            help="Video stream decode method [cv2, ffmpeg, torchvision, tochaudio]",
        )
        parser.add_argument(
            "--decoder-device",
            default=None,
            type=str,
            help="Video stream decode method [cv2, ffmpeg, torchvision, tochaudio]",
        )
        parser.add_argument(
            "--encoder-device",
            default=None,
            type=str,
            help="Video stream encode device [cpu, cude, cuda:0, etc.]",
        )
        # parser.add_argument(
        #     "--encoder",
        #     "--video-stream-encode-method",
        #     dest="video_stream_encode_method",
        #     default="cv2",
        #     type=str,
        #     help="Video stream decode method [cv2, ffmpeg, torchvision, tochaudio]",
        # )
        parser.add_argument(
            "--async-post-processing",
            type=int,
            default=1,
            help="Async post-processing",
        )
        parser.add_argument(
            "--async-video-out",
            type=int,
            default=1,
            help="Async video output",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output_file",
            type=str,
            default=None,
            help="Output file",
        )
        parser.add_argument(
            "--output-fps",
            dest="output_fps",
            type=float,
            default=None,
            help="Output frames per second",
        )
        parser.add_argument(
            "--lfo",
            "--left-frame-offset",
            dest="lfo",
            type=float,
            default=None,
            help="Offset for left video startig point (first supplied video)",
        )
        parser.add_argument(
            "--plot-frame-number",
            type=int,
            default=0,
            help="Plot frame number",
        )
        parser.add_argument(
            "--plot-frame-time",
            type=int,
            default=0,
            help="Plot frame time",
        )
        parser.add_argument(
            "--rfo",
            "--right-frame-offset",
            dest="rfo",
            type=float,
            default=None,
            help="Offset for right video startiog point (second supplied video)",
        )
        parser.add_argument(
            "--start-frame-offset",
            default=0,
            help="General start frame the video reading (after other offsets are applied)",
        )
        parser.add_argument(
            "--project-file",
            "--project_file",
            dest="project_file",
            default="hm_project.pto",
            type=str,
            help="Use project file as input to stitcher",
        )
        parser.add_argument(
            "--start-frame", type=int, default=0, help="first frame number to process"
        )
        parser.add_argument(
            "-s",
            "--start-time",
            "--start-frame-time",
            dest="start_frame_time",
            type=str,
            default=None,
            help="Start at this time in video stream",
        )
        parser.add_argument(
            "--stitch-frame-time",
            type=str,
            default=None,
            help="Use frame at this timestamp for stitching (HH:MM:SS.ssss)",
        )
        parser.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="maximum number of frames to process",
        )
        parser.add_argument(
            "-t",
            "--max-time",
            dest="max_time",
            type=str,
            default=None,
            help="Maximum amount of time to process",
        )
        parser.add_argument(
            "--video_dir",
            default=None,
            type=str,
            help="Video directory to find 'left.mp4' and 'right.mp4'",
        )
        parser.add_argument(
            "--minimize-blend",
            type=int,
            default=True,
            help="Minimize blending compute to only blend (mostly) overlapping portions of frames",
        )
        parser.add_argument(
            "--blend-mode",
            "--blend_mode",
            default="laplacian",
            type=str,
            help="Stitching blend mode (multiblend|laplacian|gpu-hard-seam)",
        )
        parser.add_argument(
            "--skip_final_video_save",
            "--skip-final-video-save",
            action="store_true",
            help="Don't save the output video frames",
        )
        parser.add_argument(
            "--save_stitched",
            "--save-stitched",
            action="store_true",
            help="Don't save the output video",
        )
        parser.add_argument(
            "--stitch-auto-adjust-exposure",
            type=int,
            default=1,
            help="Auto-adjust exposure when stitching",
        )
        parser.add_argument(
            "--no-minimize-blend",
            action="store_true",
            help="Don't minimize blending to the overlapped portions",
        )
        parser.add_argument(
            "--python-blender",
            type=int,
            default=0,
            help="Use the pythonb lending code (should be identical to C++, but may have performance differences)",
        )
        parser.add_argument(
            "--stitch-rotate-degrees",
            dest="stitch_rotate_degrees",
            type=float,
            default=None,
            help=
            "Optional rotation (degrees) applied after stitching, about image center; keeps same dimensions.",
        )
        parser.add_argument(
            "--max-control-points",
            type=int,
            default=240,
            help="Maximum number of control points used to calculate the homography matrices",
        )
        parser.add_argument(
            "--track-ids",
            type=str,
            default=None,
            help="Comma-separated list of tracking IDs to track specifically (when online)",
        )
        #
        # Progress Bar
        #
        parser.add_argument(
            "--no-progress-bar",
            action="store_true",
            help="Don't use the progress bar",
        )
        parser.add_argument(
            "--curses-progress",
            "--curses",
            action="store_true",
            help="Disable curses-based progress UI (use legacy printing)",
        )
        parser.add_argument(
            "--progress-bar-lines",
            type=int,
            default=4,
            help="Number of logging lines in the progrsss bar",
        )
        parser.add_argument(
            "--print-interval",
            type=int,
            default=20,
            help="How many iterations between log progress printing",
        )
        parser.add_argument(
            "----output-video-bit-rate",
            type=int,
            default=None,
            help="Output video bit-rate",
        )

        # Jersey framework toggles (Koshkina trunk) for reuse across CLIs
        parser.add_argument("--jersey-roi-mode", type=str, choices=["bbox", "pose", "sam"], default=None,
                            help="ROI mode for jersey trunk: bbox|pose|sam")
        parser.add_argument("--jersey-str-backend", type=str, choices=["mmocr", "parseq"], default=None,
                            help="STR backend: mmocr (default) or parseq")
        parser.add_argument("--jersey-parseq-weights", type=str, default=None, help="PARSeq weights path")
        parser.add_argument("--jersey-parseq-device", type=str, default=None, help="PARSeq device (e.g., cuda)")
        parser.add_argument("--jersey-legibility-enabled", action="store_true", help="Enable legibility filter")
        parser.add_argument("--jersey-legibility-weights", type=str, default=None, help="Legibility weights path")
        parser.add_argument("--jersey-legibility-threshold", type=float, default=None, help="Legibility score threshold")
        parser.add_argument("--jersey-reid-enabled", action="store_true", help="Enable ReID outlier removal")
        parser.add_argument("--jersey-reid-backend", type=str, choices=["resnet", "centroid"], default=None,
                            help="ReID backend: resnet (default) or centroid")
        parser.add_argument("--jersey-reid-backbone", type=str, choices=["resnet18", "resnet34"], default=None,
                            help="ReID resnet backbone")
        parser.add_argument("--jersey-reid-threshold", type=float, default=None, help="ReID Mahalanobis threshold")
        parser.add_argument("--jersey-centroid-reid-path", type=str, default=None, help="Path to centroid-reid repo/model")
        parser.add_argument("--jersey-centroid-reid-device", type=str, default=None, help="Device for centroid-reid")
        parser.add_argument("--jersey-sam-enabled", action="store_true", help="Enable SAM ROI refinement")
        parser.add_argument("--jersey-sam-checkpoint", type=str, default=None, help="Path to SAM checkpoint")
        parser.add_argument("--jersey-sam-model-type", type=str, default=None, help="SAM model type (e.g., vit_b)")
        parser.add_argument("--jersey-sam-device", type=str, default=None, help="SAM device")

        #
        # Camera braking / stop-dampening controls
        #
        braking = parser.add_argument_group(
            "camera_braking",
            "Camera movement braking and stop dampening controls",
        )
        braking.add_argument(
            "--stop-on-dir-change-delay",
            default=10,
            type=int,
            help="Frames to brake to a stop on direction change (camera tracking)",
        )
        braking.add_argument(
            "--cancel-stop-on-opposite-dir",
            default=1,
            type=int,
            help="Cancel braking when inputs flip opposite (0/1)",
        )
        braking.add_argument(
            "--stop-cancel-hysteresis-frames",
            default=2,
            type=int,
            help="Consecutive opposite-direction frames required to cancel braking",
        )
        braking.add_argument(
            "--stop-delay-cooldown-frames",
            default=2,
            type=int,
            help="Cooldown frames after stop-delay finishes/cancels before another can start",
        )
        # Breakaway quick-stop knobs via CLI
        braking.add_argument(
            "--overshoot-stop-delay-count",
            default=6,
            type=int,
            help="When overshooting breakaway, brake to stop over N frames",
        )
        braking.add_argument(
            "--post-nonstop-stop-delay-count",
            default=6,
            type=int,
            help="After nonstop ends, brake to stop over N frames",
        )
        braking.add_argument(
            "--time-to-dest-speed-limit-frames",
            default=10,
            type=int,
            help="Minimum frames to reach destination along an axis when speeding up (0 disables)",
        )

        # Generic YAML overrides: --config-override rink.camera.foo.bar=VALUE (repeatable)
        overrides = parser.add_argument_group(
            "config_overrides",
            "Override any YAML config key with --config-override key=value",
        )
        overrides.add_argument(
            "--config-override",
            dest="config_overrides",
            action="append",
            default=[],
            help="Override a YAML key path (dot.notation) with a value (repeatable)",
        )

        #
        # UI controls
        #
        ui = parser.add_argument_group(
            "ui",
            "Runtime UI controls",
        )
        ui.add_argument(
            "--camera-ui",
            default=0,
            type=int,
            help="Enable runtime camera braking UI (OpenCV trackbars)",
        )

        # Progress bar instrumentation toggles
        ui.add_argument(
            "--progress-gpu-metrics",
            dest="progress_gpu_metrics",
            action="store_true",
            help="Show GPU utilization metrics in progress UI (if available)",
        )
        ui.add_argument(
            "--no-progress-gpu-metrics",
            dest="progress_gpu_metrics",
            action="store_false",
            help="Disable GPU utilization metrics in progress UI",
        )
        parser.set_defaults(progress_gpu_metrics=True)
        ui.add_argument(
            "--progress-cuda-sync-counter",
            dest="progress_cuda_sync_counter",
            action="store_true",
            help="Show CUDA stream sync count in progress UI (uses cuda_stacktrace if available)",
        )
        ui.add_argument(
            "--no-progress-cuda-sync-counter",
            dest="progress_cuda_sync_counter",
            action="store_false",
            help="Disable CUDA stream sync count in progress UI",
        )
        parser.set_defaults(progress_cuda_sync_counter=True)

        return parser

    def parse(self, args=""):
        if args == "":
            opt = self._parser.parse_args()
        else:
            opt = self._parser.parse_args(args)
        return self.init(opt)

    # TODO: How can this be generalized with the nesting in the yaml?
    CONFIG_TO_ARGS = [
        # "model.tracker.pre_hm": "pre_hm",
        "model.tracker",
    ]

    @staticmethod
    def init(opt, parser: Optional[argparse.ArgumentParser] = None):
        # Normalize some conflicting arguments
        if opt.serial:
            opt.async_post_processing = 0
            opt.async_video_out = 0
            opt.cache_size = 0
            opt.stitch_cache_size = 0

        for key in hm_opts.CONFIG_TO_ARGS:
            nested_item = get_nested_value(getattr(opt, "game_config", {}), key, None)
            if nested_item is None:
                continue
            if isinstance(nested_item, dict):
                for k, v in nested_item.items():
                    if hasattr(opt, k):
                        current_val = getattr(opt, k)
                        if current_val is None or (
                            parser is not None and current_val == parser.get_default(k)
                        ):
                            print(f"Setting attribute {k} to {v}")
                            setattr(opt, k, v)

        # Map select CLI camera options into YAML-style config (if not already present)
        # This lets downstream code read from args.game_config['rink']['camera']
        game_cfg = getattr(opt, "game_config", None)
        if isinstance(game_cfg, dict):
            # Helper to know if a CLI option was explicitly provided (not just default)
            def _cli_spec(name: str) -> bool:
                try:
                    return parser is not None and getattr(opt, name) != parser.get_default(name)
                except Exception:
                    return False
            # stop_on_dir_change_delay
            try:
                if _cli_spec("stop_on_dir_change_delay") or get_nested_value(game_cfg, "rink.camera.stop_on_dir_change_delay", None) is None:
                    set_nested_value(game_cfg, "rink.camera.stop_on_dir_change_delay", int(opt.stop_on_dir_change_delay))
            except Exception:
                pass
            # cancel_stop_on_opposite_dir (store as bool in YAML)
            try:
                if _cli_spec("cancel_stop_on_opposite_dir") or get_nested_value(game_cfg, "rink.camera.cancel_stop_on_opposite_dir", None) is None:
                    set_nested_value(
                        game_cfg,
                        "rink.camera.cancel_stop_on_opposite_dir",
                        bool(int(opt.cancel_stop_on_opposite_dir)),
                    )
            except Exception:
                pass
            # cancel hysteresis frames
            try:
                if _cli_spec("stop_cancel_hysteresis_frames") or get_nested_value(game_cfg, "rink.camera.stop_cancel_hysteresis_frames", None) is None:
                    set_nested_value(
                        game_cfg,
                        "rink.camera.stop_cancel_hysteresis_frames",
                        int(opt.stop_cancel_hysteresis_frames),
                    )
            except Exception:
                pass
            # stop delay cooldown frames
            try:
                if _cli_spec("stop_delay_cooldown_frames") or get_nested_value(game_cfg, "rink.camera.stop_delay_cooldown_frames", None) is None:
                    set_nested_value(
                        game_cfg,
                        "rink.camera.stop_delay_cooldown_frames",
                        int(opt.stop_delay_cooldown_frames),
                    )
            except Exception:
                pass
            # time to dest speed limit frames
            try:
                if _cli_spec("time_to_dest_speed_limit_frames") or get_nested_value(game_cfg, "rink.camera.time_to_dest_speed_limit_frames", None) is None:
                    set_nested_value(
                        game_cfg,
                        "rink.camera.time_to_dest_speed_limit_frames",
                        int(opt.time_to_dest_speed_limit_frames),
                    )
            except Exception:
                pass
            # Breakaway: overshoot/post-nonstop delays
            try:
                if _cli_spec("overshoot_stop_delay_count") or (
                    get_nested_value(
                        game_cfg, "rink.camera.breakaway_detection.overshoot_stop_delay_count", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.breakaway_detection.overshoot_stop_delay_count",
                        int(opt.overshoot_stop_delay_count),
                    )
            except Exception:
                pass
            try:
                if _cli_spec("post_nonstop_stop_delay_count") or (
                    get_nested_value(
                        game_cfg, "rink.camera.breakaway_detection.post_nonstop_stop_delay_count", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.breakaway_detection.post_nonstop_stop_delay_count",
                        int(opt.post_nonstop_stop_delay_count),
                    )
            except Exception:
                pass

            # Apply generic overrides: key=value, with simple type inference
            for ov in (opt.config_overrides or []):
                if not isinstance(ov, str) or "=" not in ov:
                    continue
                key, val = ov.split("=", 1)
                sval = val.strip()
                # Try to infer type: null/None, bool, int, float; else keep string
                lval = sval.lower()
                if lval in ("null", "none"):
                    pval: Any = None
                elif lval in ("true", "false"):
                    pval = lval == "true"
                else:
                    try:
                        if "." in sval:
                            pval = float(sval)
                        else:
                            pval = int(sval)
                    except Exception:
                        pval = sval
                try:
                    set_nested_value(game_cfg, key.strip(), pval)
                except Exception:
                    pass
        else:
            # If there's no game_config yet, create one to hold overrides
            if getattr(opt, "config_overrides", []):
                opt.game_config = {}
                for ov in (opt.config_overrides or []):
                    if not isinstance(ov, str) or "=" not in ov:
                        continue
                    key, val = ov.split("=", 1)
                    sval = val.strip()
                    lval = sval.lower()
                    if lval in ("null", "none"):
                        pval = None
                    elif lval in ("true", "false"):
                        pval = lval == "true"
                    else:
                        try:
                            if "." in sval:
                                pval = float(sval)
                            else:
                                pval = int(sval)
                        except Exception:
                            pval = sval
                    try:
                        set_nested_value(opt.game_config, key.strip(), pval)
                    except Exception:
                        pass

        return opt


def preferred_arg(preferred_arg: Any, backup_arg: Any):
    if preferred_arg is not None:
        return preferred_arg
    return backup_arg


def add_remaining_autogenerated(parser: argparse.ArgumentParser):
    # parser.add_argument("--cameramake", default=None, type=str, help="Cameramake")
    # parser.add_argument("--cameramodel", default=None, type=str, help="Cameramodel")
    parser.add_argument("--cameraoutput-fps", default=None, type=str, help="Cameraoutput Fps")
    parser.add_argument(
        "--cameramount",
        default=[{"offset_80": None, "resolution": None}, {"offset_90": None, "resolution": None}],
        type=str,
        help="Cameramount",
    )
    parser.add_argument("--rinkname", default=None, type=str, help="Rinkname")
    parser.add_argument("--rinklocation-city", default=None, type=str, help="Rinklocation City")
    parser.add_argument("--rinklocation-state", default=None, type=str, help="Rinklocation State")
    parser.add_argument(
        "--rinklocation-country", default=None, type=str, help="Rinklocation Country"
    )
    parser.add_argument(
        "--rinkdimensions-length", default=None, type=str, help="Rinkdimensions Length"
    )
    parser.add_argument(
        "--rinkdimensions-width", default=None, type=str, help="Rinkdimensions Width"
    )
    parser.add_argument(
        "--rinkdimensions-corner-radius",
        default=None,
        type=str,
        help="Rinkdimensions Corner Radius",
    )
    parser.add_argument(
        "--rinkseating-capacity", default=None, type=str, help="Rinkseating Capacity"
    )
    parser.add_argument(
        "--rinkteams-home-team-name", default=None, type=str, help="Rinkteams Home Team Name"
    )
    parser.add_argument(
        "--rinkteams-home-team-colors", default=None, type=str, help="Rinkteams Home Team Colors"
    )
    parser.add_argument(
        "--rinkfacilities-locker-rooms", default=None, type=str, help="Rinkfacilities Locker Rooms"
    )
    parser.add_argument(
        "--rinkfacilities-concession-stands",
        default=None,
        type=str,
        help="Rinkfacilities Concession Stands",
    )
    parser.add_argument(
        "--rinkfacilities-restrooms", default=None, type=str, help="Rinkfacilities Restrooms"
    )
    parser.add_argument(
        "--rinkparking-capacity", default=None, type=str, help="Rinkparking Capacity"
    )
    parser.add_argument("--rinkparking-price", default=None, type=str, help="Rinkparking Price")
    parser.add_argument(
        "--rinkscoreboard-perspective-polygon",
        default=None,
        type=str,
        help="Rinkscoreboard Perspective Polygon",
    )
    parser.add_argument(
        "--rinkscoreboard-projected-height",
        default="%20",
        type=str,
        help="Rinkscoreboard Projected Height",
    )
    parser.add_argument(
        "--rinkscoreboard-projected-width",
        default="%10",
        type=str,
        help="Rinkscoreboard Projected Width",
    )
    parser.add_argument(
        "--rinkend-zones-left-start", default=None, type=str, help="Rinkend Zones Left Start"
    )
    parser.add_argument(
        "--rinkend-zones-left-stop", default=None, type=str, help="Rinkend Zones Left Stop"
    )
    parser.add_argument(
        "--rinkend-zones-right-start", default=None, type=str, help="Rinkend Zones Right Start"
    )
    parser.add_argument(
        "--rinkend-zones-right-stop", default=None, type=str, help="Rinkend Zones Right Stop"
    )
    parser.add_argument(
        "--rinktracking-cam-ignore-largest",
        default=True,
        type=int,
        help="Rinktracking Cam Ignore Largest",
    )
    parser.add_argument(
        "--rinkcamera-fixed-edge-scaling-factor",
        default=0.8,
        type=str,
        help="Rinkcamera Fixed Edge Scaling Factor",
    )
    parser.add_argument(
        "--rinkcamera-fixed-edge-rotation-angle",
        default=30,
        type=str,
        help="Rinkcamera Fixed Edge Rotation Angle",
    )
    parser.add_argument(
        "--rinkcamera-image-channel-adjustment",
        default=None,
        type=str,
        help="Rinkcamera Image Channel Adjustment",
    )
    parser.add_argument(
        "--rinkcamera-follower-box-scale-width",
        default=1.25,
        type=str,
        help="Rinkcamera Follower Box Scale Width",
    )
    parser.add_argument(
        "--rinkcamera-follower-box-scale-height",
        default=1.25,
        type=str,
        help="Rinkcamera Follower Box Scale Height",
    )
    parser.add_argument(
        "--rinkcamera-sticky-size-ratio-to-frame-width",
        default=10.0,
        type=str,
        help="Rinkcamera Sticky Size Ratio To Frame Width",
    )
    parser.add_argument(
        "--rinkcamera-sticky-translation-gaussian-mult",
        default=5.0,
        type=str,
        help="Rinkcamera Sticky Translation Gaussian Mult",
    )
    parser.add_argument(
        "--rinkcamera-unsticky-translation-size-ratio",
        default=0.75,
        type=str,
        help="Rinkcamera Unsticky Translation Size Ratio",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-min-considered-group-velocity",
        default=3.0,
        type=str,
        help="Rinkcamera Breakaway Detection Min Considered Group Velocity",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-group-ratio-threshold",
        default=0.5,
        type=str,
        help="Rinkcamera Breakaway Detection Group Ratio Threshold",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-group-velocity-speed-ratio",
        default=0.3,
        type=str,
        help="Rinkcamera Breakaway Detection Group Velocity Speed Ratio",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-scale-speed-constraints",
        default=2.0,
        type=str,
        help="Rinkcamera Breakaway Detection Scale Speed Constraints",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-nonstop-delay-count",
        default=2,
        type=str,
        help="Rinkcamera Breakaway Detection Nonstop Delay Count",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-overshoot-scale-speed-ratio",
        default=0.7,
        type=str,
        help="Rinkcamera Breakaway Detection Overshoot Scale Speed Ratio",
    )
    parser.add_argument("--gamename", default=None, type=str, help="Gamename")
    parser.add_argument("--gamerink", default=None, type=str, help="Gamerink")
    parser.add_argument("--gamehome", default=None, type=str, help="Gamehome")
    parser.add_argument("--gameaway", default=None, type=str, help="Gameaway")
    parser.add_argument(
        "--gamestitching-frame-offsets-left",
        default=None,
        type=str,
        help="Gamestitching Frame Offsets Left",
    )
    parser.add_argument(
        "--gamestitching-frame-offsets-right",
        default=None,
        type=str,
        help="Gamestitching Frame Offsets Right",
    )
    parser.add_argument(
        "--gamestitching-control-points-m-kpts0",
        default=None,
        type=str,
        help="Gamestitching Control Points M Kpts0",
    )
    parser.add_argument(
        "--gamestitching-control-points-m-kpts1",
        default=None,
        type=str,
        help="Gamestitching Control Points M Kpts1",
    )
    parser.add_argument(
        "--gamestitching-offsets", default=None, type=str, help="Gamestitching Offsets"
    )
    parser.add_argument("--gameclip-box", default=None, type=str, help="Gameclip Box")
    parser.add_argument(
        "--gameboundaries-upper", default=None, type=str, help="Gameboundaries Upper"
    )
    parser.add_argument(
        "--gameboundaries-upper-tune-position",
        default=None,
        type=str,
        help="Gameboundaries Upper Tune Position",
    )
    parser.add_argument(
        "--gameboundaries-lower", default=None, type=str, help="Gameboundaries Lower"
    )
    parser.add_argument(
        "--gameboundaries-lower-tune-position",
        default=None,
        type=str,
        help="Gameboundaries Lower Tune Position",
    )
    parser.add_argument(
        "--gameboundaries-scale-width", default=None, type=str, help="Gameboundaries Scale Width"
    )
    parser.add_argument(
        "--gameboundaries-scale-height", default=None, type=str, help="Gameboundaries Scale Height"
    )
    parser.add_argument(
        "--modelend-to-end-config",
        default="config/models/hm2/hm_end_to_end.py",
        type=str,
        help="Modelend To End Config",
    )
    parser.add_argument(
        "--modelend-to-end-checkpoint",
        default="pretrained/mmdetection/yolox_s_8x8_300e_coco_80e_ch_with_detector_prefix.pth",
        type=str,
        help="Modelend To End Checkpoint",
    )
    parser.add_argument(
        "--modelend-to-end-checkpoint-local",
        default="pretrained/mmdetection/yolox_s_8x8_300e_coco_80e_ch_with_detector_prefix.pth",
        type=str,
        help="Modelend To End Checkpoint Local",
    )
    parser.add_argument(
        "--modelend-to-end-checkpoint-remote",
        default="https://drive.google.com/file/d/1WgJ-u2aL1Yv6VNXF5w-0DtsxCCDXAEe7/view?usp=drive_link",
        type=str,
        help="Modelend To End Checkpoint Remote",
    )
    parser.add_argument(
        "--modelpose-config",
        default="openmm/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py",
        type=str,
        help="Modelpose Config",
    )
    parser.add_argument(
        "--modelpose-checkpoint",
        default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth",
        type=str,
        help="Modelpose Checkpoint",
    )
    parser.add_argument(
        "--modelpose-checkpoint-remote",
        default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth",
        type=str,
        help="Modelpose Checkpoint Remote",
    )
    parser.add_argument(
        "--modelice-rink-segm-config",
        default="config/models/ice_rink/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py",
        type=str,
        help="Modelice Rink Segm Config",
    )
    parser.add_argument(
        "--modelice-rink-segm-checkpoint-local",
        default="pretrained/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/ice_rink_iter_30000.pth",
        type=str,
        help="Modelice Rink Segm Checkpoint Local",
    )
    parser.add_argument(
        "--modelice-rink-segm-checkpoint",
        default="pretrained/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/ice_rink_iter_30000.pth",
        type=str,
        help="Modelice Rink Segm Checkpoint",
    )
    parser.add_argument(
        "--modelsvnh-classifier-config", default=None, type=str, help="Modelsvnh Classifier Config"
    )
    parser.add_argument(
        "--modelsvnh-classifier-checkpoint",
        default="pretrained/svhnc/model-65000.pth",
        type=str,
        help="Modelsvnh Classifier Checkpoint",
    )


# TODO: FIXME, doesn;t properly handle nested item names converted to arg names (no dash)
def generate_yaml_args_code(parser: argparse.ArgumentParser, yaml_file_path: Path) -> str:
    """
    Generates Python code to add arguments from a YAML file to an argparse parser object, including nested YAML items.

    Args:
        parser (argparse.ArgumentParser): An argparse parser object.
        yaml_file_path (Path): The path to the YAML file containing the arguments.

    Returns:
        str: A string containing the Python code to add the arguments to the parser.
    """
    # Read the YAML file
    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    code_lines = []

    def to_camel_case(snake_str: str) -> str:
        components = snake_str.split("-")
        return " ".join(x.title() for x in components)

    def process_yaml_items(prefix: str, data: Union[Dict[str, Any], Any]) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if key.endswith("_description") or key.endswith("_type"):
                    # Skip _description and _type entries for now
                    continue

                new_prefix = f"{prefix}-{key}-" if prefix else key
                process_yaml_items(new_prefix, value)
        else:
            # Replace underscores with dashes for the argument name
            arg_name = prefix.rstrip("-").replace("_", "-")

            # Determine help description and type
            description = yaml_data.get(
                f'{prefix.rstrip("-")}_description', to_camel_case(arg_name)
            )
            value_type = yaml_data.get(f'{prefix.rstrip("-")}_type', str)

            if isinstance(data, bool):
                value_type = int  # Change boolean type to int for argparse

            # Check if the argument already exists in the parser
            if not any(arg_name == action.dest for action in parser._actions):
                # Generate the code to add the argument to the parser
                code_line = f"parser.add_argument('--{arg_name}', default={repr(data)}, type={value_type.__name__}, help='{description}')"
                code_lines.append(code_line)

    process_yaml_items("", yaml_data)
    return "\n    ".join(code_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Python code to add YAML configuration to argparse parser"
    )
    parser.add_argument("yaml_file_path", type=Path, help="Path to the YAML file")

    args = parser.parse_args()
    generated_code = generate_yaml_args_code(hm_opts.parser(), args.yaml_file_path)

    # Print the generated code
    # TODO: Generate the
    print("def add_remaining_autogenerated(parser: argparse.ArgumentParser):\n    ")
    print(generated_code)

import argparse
import copy
import math
import os
import shutil

# import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# We need this to get registered
import torch
import torch.backends.cudnn as cudnn
from mmcv.transforms import Compose

# from mmdet.apis import init_track_model
# from mmengine.config import Config
# from torch.nn.parallel import DistributedDataParallel as DDP
import hmlib
import hmlib.tracking_utils.segm_boundaries
import hmlib.transforms
from hmlib.camera.camera import should_unsharp_mask_camera
from hmlib.config import (
    get_clip_box,
    get_config,
    get_game_dir,
    get_nested_value,
    resolve_global_refs,
    set_nested_value,
)
from hmlib.datasets.dataframe import find_latest_dataframe_file
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.datasets.dataset.multi_dataset import MultiDatasetWrapper
from hmlib.datasets.dataset.stitching_dataloader2 import MultiDataLoaderWrapper, StitchDataset
from hmlib.audio import has_audio_stream, mux_audio_in_place
from hmlib.hm_opts import copy_opts, hm_opts, preferred_arg

# from hmlib.hm_transforms import update_data_pipeline
from hmlib.log import get_root_logger, logger
from hmlib.orientation import configure_game_videos
from hmlib.stitching.configure_stitching import configure_video_stitching
from hmlib.tasks.tracking import run_mmtrack

# from hmlib.utils.checkpoint import load_checkpoint_to_model
from hmlib.utils.gpu import select_gpus
from hmlib.utils.pipeline import get_pipeline_item, update_pipeline_item
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import time_to_frame

ROOT_DIR = os.path.dirname(os.path.abspath(hmlib.__file__))


def make_parser(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser("HockeyMOM Tracking")
    parser = hm_opts.parser(parser)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Validate that key dependencies can be imported and that the game directory can be "
            "resolved, then exit (does not require a working CUDA runtime)."
        ),
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help="Root directory")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--no-wide-start",
        default=False,
        action="store_true",
        help="Don't start with a tracking box of the entire input frame. Immediately track to player detections.",
    )
    parser.add_argument(
        "--no-rink-rotation",
        default=False,
        action="store_true",
        help="Don't do rink rotation.",
    )
    parser.add_argument(
        "--no-play-tracking",
        default=False,
        action="store_true",
        help="Don't do any postprocessing (i.e. play tracking) after basic player tracking.",
    )
    # Output video flag moved to hm_opts.parser
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # cam args
    parser.add_argument(
        "--adjust-exposure",
        default=None,
        type=float,
        help="Adjust overall exposure of all input images",
    )
    parser.add_argument(
        "--cam-ignore-largest",
        default=False,
        action="store_true",
        help="Remove the largest tracking box from the camera set (i.e. at Vallco, a ref is "
        "often right in front of the camera, but not enough of the ref is "
        "visible to note it as a ref)",
    )
    parser.add_argument(
        "--rink",
        default=None,
        type=str,
        help="rink name",
    )
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--track_thresh_low",
        type=float,
        default=0.1,
        help="tracking confidence threshold lower bound",
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--cvat-output",
        action="store_true",
        help="generate dataset data importable by cvat",
    )
    parser.add_argument(
        "--no-stitch",
        "--no-force-stitching",
        "--no_force_stitching",
        dest="no_force_stitching",
        action="store_true",
        help="force video stitching",
    )
    # Plotting, Profiling, and Camera Controller options moved to hm_opts.parser
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Additional YAML config file(s) to merge in order. "
            "Repeat --config to provide multiple files; later ones override earlier ones."
        ),
    )
    parser.add_argument(
        "--test-size", type=str, default=None, help="WxH of test box size (format WxH)"
    )
    # Save frame dir moved to hm_opts.parser
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        type=str,
        default="tracking",
        help="Comma-separated task list (tracking)",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--min-box-area", type=float, default=100, help="filter out tiny boxes")
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    # Data I/O flags moved to hm_opts.parser
    # ONNX detector export/inference options moved to hm_opts.parser
    # TensorRT detector options moved to hm_opts.parser
    # ONNX detector options moved to hm_opts.parser
    # ONNX pose export/inference options moved to hm_opts.parser
    # TensorRT pose options moved to hm_opts.parser
    # ONNX pose options moved to hm_opts.parser
    # Audio-only and output-video moved to hm_opts.parser
    parser.add_argument("--checkpoint", type=str, default=None, help="Tracking checkpoint file")
    parser.add_argument("--detector", help="det checkpoint file")
    parser.add_argument("--reid", help="reid checkpoint file")

    # Pose args
    parser.add_argument("--pose-config", type=str, default=None, help="Pose config file")
    parser.add_argument("--pose-checkpoint", type=str, default=None, help="Pose checkpoint file")
    # Pose visualization args moved to hm_opts.parser
    parser.add_argument(
        "--debug-play-tracker", action="store_true", help="Print per-frame play boxes and counts"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply a temporal filter to smooth the pose estimation results. "
        "See also --smooth-filter-cfg.",
    )
    return parser


def set_torch_multiprocessing_use_filesystem():
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")


def to_32bit_mul(val):
    return int((val + 31)) & ~31


MAP_ARGS_TO_YAML = {
    "tracker": "model.tracker.type",
}


def update_from_args(args, arg_name, config, noset_value: any = None):
    if not hasattr(args, arg_name):
        return
    set_nested_value(
        dct=config,
        key_str="model.tracker.type",
        set_to=args.tracker,
        noset_value=noset_value,
    )


def configure_model(config: dict, args: argparse.Namespace):
    update_from_args(args, "tracker", config)
    return args


def find_stitched_file(dir_name: str, game_id: str):
    exts = ["mp4", "mkv", "avi"]
    basenames = [
        # "stitched_output",
        "stitched_output-with-audio",
        # "stitched_output-" + game_id,
        # "stitched_output-with-audio-" + game_id,
    ]
    for basename in basenames:
        for ext in exts:
            path = os.path.join(dir_name, basename + "." + ext)
            if os.path.exists(path):
                return path
    return None


def is_stitching(input_video: str) -> bool:
    if not input_video:
        raise AttributeError("No valid input video specified")
    input_video_files = input_video.split(",")
    return len(input_video_files) == 2 or os.path.isdir(input_video)


class _StitchRotationController:
    def __init__(self, game_config: Optional[Dict[str, Any]] = None) -> None:
        self._config = game_config
        self._value: Optional[float] = None

    def get_post_stitch_rotate_degrees(self) -> Optional[float]:
        if isinstance(self._config, dict):
            try:
                val = get_nested_value(self._config, "game.stitching.stitch-rotate-degrees", None)
                if val is None:
                    val = get_nested_value(
                        self._config, "game.stitching.stitch_rotate_degrees", None
                    )
                if val is not None:
                    return float(val)
            except Exception:
                pass
        return self._value

    def set_post_stitch_rotate_degrees(self, degrees: Optional[float]) -> None:
        self._value = degrees
        if isinstance(self._config, dict):
            try:
                set_nested_value(self._config, "game.stitching.stitch-rotate-degrees", degrees)
            except Exception:
                try:
                    set_nested_value(self._config, "game.stitching.stitch_rotate_degrees", degrees)
                except Exception:
                    pass


def _main(args, num_gpu):
    dataloader = None
    postprocessor = None
    opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    try:

        if args.gpus and isinstance(args.gpus, str):
            args.gpus = [int(i) for i in args.gpus.split(",")]

        # set environment variables for distributed training
        cudnn.benchmark = True

        game_config = args.game_config

        dataloader = MultiDatasetWrapper(forgive_missing_attributes=["fps"])

        if args.output_fps is None:
            args.output_fps = get_nested_value(game_config, "camera.output-fps")

        if args.lfo is None and args.rfo is None:
            if "stitching" in game_config["game"] and "offsets" in game_config["game"]["stitching"]:
                offsets = game_config["game"]["stitching"]["offsets"]
                if offsets:
                    args.lfo = offsets[0]
                    if len(offsets) == 1:
                        args.rfo = 0.0
                    else:
                        assert len(offsets) == 2
                        args.rfo = offsets[1]
                    if args.lfo < 0:
                        args.rfo += -args.lfo
                        args.lfo = 0.0
                    assert args.lfo >= 0 and args.rfo >= 0

        model = None

        # cmdline overrides
        if args.camera_name:
            set_nested_value(game_config, "camera.name", args.camera_name)
        else:
            args.camera_name = get_nested_value(game_config, "camera.name")
        if args.unsharp_mask is None and should_unsharp_mask_camera(args.camera_name):
            args.unsharp_mask = 1

        # Derived camera args (former DefaultArguments)
        # Crop output image unless explicitly disabled via CLI.
        # Prefer rink.tracking.cam_ignore_largest when CLI did not override.
        if not getattr(args, "cam_ignore_largest", False):
            args.cam_ignore_largest = get_nested_value(
                game_config, "rink.tracking.cam_ignore_largest", True
            )
        # Map plotting convenience flag to per-frame tracking overlays.
        args.plot_individual_player_tracking = bool(getattr(args, "plot_tracking", False))
        if args.plot_individual_player_tracking:
            args.plot_boundaries = True

        # See if gameid is in videos
        if not args.input_video and args.game_id:
            game_video_dir = get_game_dir(args.game_id)
            if game_video_dir:
                # TODO: also look for avi and mp4 files
                if not args.no_force_stitching:
                    args.input_video = game_video_dir
                else:
                    pre_stitched_file_name = find_stitched_file(
                        dir_name=game_video_dir, game_id=args.game_id
                    )
                    if pre_stitched_file_name and os.path.exists(pre_stitched_file_name):
                        args.input_video = pre_stitched_file_name
                    else:
                        args.input_video = game_video_dir

        results_folder = os.path.join(".", "output_workdirs", args.game_id)
        os.makedirs(results_folder, exist_ok=True)
        args.work_dir = results_folder
        try:
            args.game_dir = get_game_dir(args.game_id, assert_exists=False)
        except Exception:
            args.game_dir = None

        tracking_data_path = getattr(
            args, "input_tracking_data", None
        ) or find_latest_dataframe_file(args.game_dir, "tracking")
        detection_data_path = getattr(
            args, "input_detection_data", None
        ) or find_latest_dataframe_file(args.game_dir, "detections")
        pose_data_path = getattr(args, "input_pose_data", None) or find_latest_dataframe_file(
            args.game_dir, "pose"
        )
        action_data_path = find_latest_dataframe_file(args.game_dir, "actions")
        args.tracking_data_path = tracking_data_path
        args.detection_data_path = detection_data_path
        args.pose_data_path = pose_data_path
        args.action_data_path = action_data_path

        # Initialize lightweight profiler and attach to args for downstream use
        try:
            from hmlib.utils.profiler import build_profiler_from_args

            default_prof_dir = os.path.join(results_folder, "profiler")
            profiler = build_profiler_from_args(args, save_dir_fallback=default_prof_dir)
        except Exception:
            profiler = None
        setattr(args, "profiler", profiler)

        using_precalculated_tracking = bool(tracking_data_path)
        using_precalculated_detections = bool(detection_data_path)
        # using_precalculated_pose = bool(pose_data_path)

        actual_device_count = torch.cuda.device_count()
        if not actual_device_count:
            raise Exception("At leats one GPU is required for this application")
        while len(args.gpus) > actual_device_count:
            del args.gpus[-1]

        gpus, is_single_lowmem_gpu, gpu_allocator = select_gpus(
            allowed_gpus=args.gpus,
            is_stitching=is_stitching(args.input_video),
            # is_multipose=args.multi_pose,
            is_detecting=not using_precalculated_tracking and not using_precalculated_detections,
            stitch_with_fastest=not args.detect_jersey_numbers,
        )
        # Expose per-role devices to downstream components (Aspen plugins, postprocessor)
        args.camera_device = gpus.get("camera")
        args.encoder_device = gpus.get("encoder")
        if is_single_lowmem_gpu:
            print("Adjusting configuration for a single low-memory GPU environment...")
            args.cache_size = 0
            # args.batch_size = 1

        # This would be way too slow on CPU
        assert torch.cuda.is_available()
        main_device = torch.device("cuda")
        for name in ["detection", "stitching", "encoder"]:
            if name in gpus:
                main_device = gpus[name]
                torch.cuda.set_device(main_device)
                break

        data_pipeline = None

        # Prefer unified Aspen config (namespaced under 'aspen') for model + pipeline
        aspen_cfg_for_pipeline = game_config.get("aspen") if isinstance(game_config, dict) else None
        # Expose to downstream run_mmtrack() via args dict
        args.aspen = aspen_cfg_for_pipeline

        # If ONNX/TRT detector flags are provided, thread them into Aspen plugins.detector_factory.params
        if args.aspen and isinstance(args.aspen, dict):
            trunks_cfg = args.aspen.setdefault("plugins", {}) or {}
            # When loading precomputed tracking or detection CSVs, skip writing
            # detections to CSV to avoid unnecessary work and potential dtype issues.
            if getattr(args, "input_tracking_data", None) or getattr(
                args, "input_detection_data", None
            ):
                save_det = trunks_cfg.get("save_detections")
                if isinstance(save_det, dict):
                    save_det["enabled"] = False
            try:
                onnx_enable = bool(
                    args.detector_onnx_enable
                    or args.detector_onnx_path
                    or args.detector_onnx_quantize_int8
                )
                # Optional static detection outputs (fixed-shape top-k)
                static_det_enable = bool(getattr(args, "detector_static_detections", False))
                static_det_max = int(getattr(args, "detector_static_max_detections", 0) or 0)
                # Configure static-shape detection outputs by default whenever the
                # detector supports them (e.g., YOLOXHead). The CLI flag controls
                # additional overrides such as max_detections.
                if "detector" in trunks_cfg:
                    df = trunks_cfg.setdefault(
                        "detector_factory",
                        {
                            "class": "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    df_params = df.setdefault("params", {}) or {}
                    static_cfg = df_params.setdefault("static_detections", {}) or {}
                    # Always enable static detections when supported; allow CLI
                    # to override max_detections when provided.
                    static_cfg.setdefault("enable", True)
                    if static_det_enable and static_det_max > 0:
                        static_cfg["max_detections"] = static_det_max
                    df_params["static_detections"] = static_cfg
                    df["params"] = df_params
                    trunks_cfg["detector_factory"] = df
                if onnx_enable and "detector" in trunks_cfg:
                    df = trunks_cfg.setdefault(
                        "detector_factory",
                        {
                            "class": "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    df_params = df.setdefault("params", {}) or {}
                    onnx_cfg = df_params.setdefault("onnx", {}) or {}
                    # Determine if ONNX should be enabled
                    onnx_cfg["enable"] = True
                    # Default path under results folder if not provided
                    default_onnx_path = os.path.join(results_folder, "detector.onnx")
                    onnx_cfg["path"] = args.detector_onnx_path or default_onnx_path
                    onnx_cfg["force_export"] = bool(args.detector_onnx_force_export)
                    onnx_cfg["quantize_int8"] = bool(args.detector_onnx_quantize_int8)
                    onnx_cfg["calib_frames"] = int(args.detector_onnx_calib_frames or 0)
                    # Mirror NMS configuration for ONNX-backed detectors so the
                    # same DetectorNMS path can be used.
                    onnx_cfg["nms_backend"] = getattr(args, "detector_nms_backend", "trt")
                    onnx_cfg["nms_test"] = bool(getattr(args, "detector_nms_test", False))
                    onnx_cfg["nms_plugin"] = getattr(args, "detector_trt_nms_plugin", "batched")
                    df_params["onnx"] = onnx_cfg
                    df["params"] = df_params
                    trunks_cfg["detector_factory"] = df
                # TensorRT detector integration
                trt_enable = bool(args.detector_trt_enable or args.detector_trt_engine)
                if trt_enable and "detector" in trunks_cfg:
                    df = trunks_cfg.setdefault(
                        "detector_factory",
                        {
                            "class": "hmlib.aspen.plugins.detector_factory_plugin.DetectorFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    df_params = df.setdefault("params", {}) or {}
                    trt_cfg = df_params.setdefault("trt", {}) or {}
                    trt_cfg["enable"] = True
                    default_engine_path = os.path.join(results_folder, "detector.engine")
                    trt_cfg["engine"] = args.detector_trt_engine or default_engine_path
                    trt_cfg["force_build"] = bool(args.detector_trt_force_build)
                    trt_cfg["fp16"] = bool(args.detector_trt_fp16)
                    # INT8 options
                    trt_cfg["int8"] = bool(getattr(args, "detector_trt_int8", False))
                    trt_cfg["calib_frames"] = int(
                        getattr(args, "detector_trt_calib_frames", 0) or 0
                    )
                    # NMS backend selection for TensorRT detector
                    trt_cfg["nms_backend"] = getattr(args, "detector_nms_backend", "trt")
                    trt_cfg["nms_test"] = bool(getattr(args, "detector_nms_test", False))
                    trt_cfg["nms_plugin"] = getattr(args, "detector_trt_nms_plugin", "batched")
                    df_params["trt"] = trt_cfg
                    df["params"] = df_params
                    trunks_cfg["detector_factory"] = df
                # Pose ONNX integration (pose_factory)
                pose_onnx_enable = bool(
                    args.pose_onnx_enable or args.pose_onnx_path or args.pose_onnx_quantize_int8
                )
                if pose_onnx_enable and "pose" in trunks_cfg:
                    pf = trunks_cfg.setdefault(
                        "pose_factory",
                        {
                            "class": "hmlib.aspen.plugins.pose_factory_plugin.PoseInferencerFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    pf_params = pf.setdefault("params", {}) or {}
                    ponnx_cfg = pf_params.setdefault("onnx", {}) or {}
                    ponnx_cfg["enable"] = True
                    default_pose_onnx = os.path.join(results_folder, "pose.onnx")
                    ponnx_cfg["path"] = args.pose_onnx_path or default_pose_onnx
                    ponnx_cfg["force_export"] = bool(args.pose_onnx_force_export)
                    ponnx_cfg["quantize_int8"] = bool(args.pose_onnx_quantize_int8)
                    ponnx_cfg["calib_frames"] = int(args.pose_onnx_calib_frames or 0)
                    pf_params["onnx"] = ponnx_cfg
                    pf["params"] = pf_params
                    trunks_cfg["pose_factory"] = pf
                # Pose TensorRT integration (pose_factory)
                pose_trt_enable = bool(args.pose_trt_enable or args.pose_trt_engine)
                if pose_trt_enable and "pose" in trunks_cfg:
                    pf = trunks_cfg.setdefault(
                        "pose_factory",
                        {
                            "class": "hmlib.aspen.plugins.pose_factory_plugin.PoseInferencerFactoryPlugin",
                            "depends": [],
                            "params": {},
                        },
                    )
                    pf_params = pf.setdefault("params", {}) or {}
                    ptrt_cfg = pf_params.setdefault("trt", {}) or {}
                    ptrt_cfg["enable"] = True
                    default_pose_engine = os.path.join(results_folder, "pose.engine")
                    ptrt_cfg["engine"] = args.pose_trt_engine or default_pose_engine
                    ptrt_cfg["force_build"] = bool(args.pose_trt_force_build)
                    ptrt_cfg["fp16"] = bool(args.pose_trt_fp16)
                    # INT8 options
                    ptrt_cfg["int8"] = bool(getattr(args, "pose_trt_int8", False))
                    ptrt_cfg["calib_frames"] = int(getattr(args, "pose_trt_calib_frames", 0) or 0)
                    pf_params["trt"] = ptrt_cfg
                    pf["params"] = pf_params
                    trunks_cfg["pose_factory"] = pf
                # Tracker backend selection (HmTracker vs static CUDA ByteTrack)
                tracker_backend = getattr(args, "tracker_backend", None)
                if tracker_backend is not None and "tracker" in trunks_cfg:
                    tracker_cfg = trunks_cfg.setdefault(
                        "tracker",
                        {
                            "class": "hmlib.aspen.plugins.tracker_plugin.TrackerPlugin",
                            "depends": [
                                "detector",
                                "ice_boundaries",
                                "model_factory",
                                "boundaries",
                            ],
                            "params": {},
                        },
                    )
                    tracker_params = tracker_cfg.setdefault("params", {}) or {}
                    if tracker_backend == "hm":
                        # Default HmTracker backend; clear any explicit overrides.
                        tracker_params.pop("tracker_class", None)
                        tracker_params.pop("tracker_kwargs", None)
                    elif tracker_backend == "static_bytetrack":
                        tracker_params["tracker_class"] = (
                            "hmlib.tracking_utils.bytetrack.HmByteTrackerCudaStatic"
                        )
                        tracker_kwargs = tracker_params.setdefault("tracker_kwargs", {}) or {}
                        max_det = getattr(args, "tracker_max_detections", 256)
                        max_tracks = getattr(args, "tracker_max_tracks", 256)
                        if max_det is not None:
                            tracker_kwargs["max_detections"] = int(max_det)
                        if max_tracks is not None:
                            tracker_kwargs["max_tracks"] = int(max_tracks)
                        tracker_device = getattr(args, "tracker_device", None)
                        if tracker_device:
                            tracker_kwargs["device"] = tracker_device
                        tracker_params["tracker_kwargs"] = tracker_kwargs
                    tracker_cfg["params"] = tracker_params
                    trunks_cfg["tracker"] = tracker_cfg
                args.aspen["plugins"] = trunks_cfg
            except Exception:
                traceback.print_exc()

        if args.tracking:
            model = None  # Built by Aspen ModelFactoryPlugin

            # Build inference pipeline from Aspen YAML if provided
            pipeline = None
            if aspen_cfg_for_pipeline and "inference_pipeline" in aspen_cfg_for_pipeline:
                pipeline = aspen_cfg_for_pipeline["inference_pipeline"]
                # first transform should be HmLoadImageFromWebcam in streaming
                if pipeline and isinstance(pipeline[0], dict):
                    pipeline[0]["type"] = "HmLoadImageFromWebcam"
                # Coerce types not representable in YAML (e.g., tuple for meta_keys)
                for step in pipeline:
                    if not isinstance(step, dict):
                        continue
                    t = step.get("type")
                    if t in ("mmdet.PackTrackInputs", "PackTrackInputs"):
                        mk = step.get("meta_keys")
                        if isinstance(mk, list):
                            step["meta_keys"] = tuple(mk)
                    update_pipeline_item(
                        pipeline,
                        "IceRinkSegmConfig",
                        dict(
                            game_id=args.game_id,
                            ice_rink_inference_scale=getattr(
                                args, "ice_rink_inference_scale", None
                            ),
                        ),
                    )
                # Apply clip box if present
                orig_clip_box = get_clip_box(game_id=args.game_id, root_dir=args.root_dir)
                if orig_clip_box:
                    hm_crop = get_pipeline_item(pipeline, "HmCrop")
                    if hm_crop is not None:
                        hm_crop["rectangle"] = orig_clip_box
                data_pipeline = Compose(pipeline)
            else:
                data_pipeline = None

            #
            # post-detection pipeline updates
            #
            # For Aspen-built model, boundaries will be applied by BoundariesPlugin.
            # Put boundary inputs into config dict so run_mmtrack can pass to Aspen shared.
            # Recompute tuned boundary lines from game_config (legacy behavior of DefaultArguments).
            game_bound_cfg = (
                game_config.get("game", {}).get("boundaries", {})
                if isinstance(game_config, dict)
                else {}
            )
            top_border_lines = game_bound_cfg.get("upper", []) or []
            bottom_border_lines = game_bound_cfg.get("lower", []) or []
            upper_tune_position = game_bound_cfg.get("upper_tune_position", []) or []
            lower_tune_position = game_bound_cfg.get("lower_tune_position", []) or []
            boundary_scale_width = game_bound_cfg.get("scale_width", 1.0)
            boundary_scale_height = game_bound_cfg.get("scale_height", 1.0)

            def _tune_lines(lines, tune_pos):
                if not lines or not tune_pos:
                    return lines
                tuned = []
                for x1, y1, x2, y2 in lines:
                    if boundary_scale_width:
                        x1 *= boundary_scale_width
                        x2 *= boundary_scale_width
                    if boundary_scale_height:
                        y2 *= boundary_scale_height
                        y1 *= boundary_scale_height
                    x1 += tune_pos[0]
                    x2 += tune_pos[0]
                    y1 += tune_pos[1]
                    y2 += tune_pos[1]
                    tuned.append([x1, y1, x2, y2])
                return tuned

            top_border_lines = _tune_lines(top_border_lines, upper_tune_position)
            bottom_border_lines = _tune_lines(bottom_border_lines, lower_tune_position)

            args.initial_args = vars(args)
            args.initial_args["top_border_lines"] = top_border_lines
            args.initial_args["bottom_border_lines"] = bottom_border_lines
            args.initial_args["original_clip_box"] = get_clip_box(
                game_id=args.game_id, root_dir=args.root_dir
            )
            # Keep a copy under game_config for Aspen plugins that read from game_config.initial_args
            if hasattr(args, "game_config") and isinstance(args.game_config, dict):
                args.game_config["initial_args"] = args.initial_args

        postprocessor = None
        if args.input_video:
            input_video_files = args.input_video.split(",")
            aspen_stitching_cli = getattr(args, "aspen_stitching", None)
            if aspen_stitching_cli is None and isinstance(aspen_cfg_for_pipeline, dict):
                use_aspen_stitching = bool(
                    get_nested_value(aspen_cfg_for_pipeline, "stitching.enabled", False)
                )
            else:
                use_aspen_stitching = bool(aspen_stitching_cli)
            if is_stitching(args.input_video):
                project_file_name = "hm_project.pto"

                game_videos = {}

                if len(input_video_files) == 2:
                    vl = input_video_files[0]
                    vr = input_video_files[1]
                    dir_name = os.path.dirname(vl)
                    assert dir_name == os.path.dirname(vr)
                elif os.path.isdir(args.input_video):
                    game_videos = configure_game_videos(
                        game_id=args.game_id,
                        inference_scale=getattr(args, "ice_rink_inference_scale", None),
                    )
                    dir_name = args.input_video
                    assert dir_name
                    input_video_files = game_videos

                left_vid = BasicVideoInfo(",".join(game_videos["left"]))
                right_vid = BasicVideoInfo(",".join(game_videos["right"]))

                total_frames = min(left_vid.frame_count, right_vid.frame_count)
                print(f"Total possible stitched video frames: {total_frames}")

                assert not args.start_frame or not args.start_frame_time
                if not args.start_frame and args.start_frame_time:
                    args.start_frame = time_to_frame(
                        time_str=args.start_frame_time, fps=left_vid.fps
                    )

                assert not args.max_frames or not args.max_time
                if not args.max_frames and args.max_time:
                    args.max_frames = time_to_frame(time_str=args.max_time, fps=left_vid.fps)

                pto_project_file, lfo, rfo = configure_video_stitching(
                    dir_name=dir_name,
                    video_left=str(game_videos["left"][0]),
                    video_right=str(game_videos["right"][0]),
                    max_control_points=args.max_control_points,
                    project_file_name=project_file_name,
                    left_frame_offset=args.lfo,
                    right_frame_offset=args.rfo,
                )
                stitch_videos = {
                    "left": {
                        "files": game_videos["left"],
                        "frame_offset": lfo,
                    },
                    "right": {
                        "files": game_videos["right"],
                        "frame_offset": rfo,
                    },
                }
                # Prefer audio from the baseline camera (frame offset == 0) for stitched output.
                try:
                    baseline = None
                    if float(lfo or 0) == 0.0:
                        baseline = "left"
                    elif float(rfo or 0) == 0.0:
                        baseline = "right"
                    else:
                        baseline = "left" if float(lfo or 0) <= float(rfo or 0) else "right"
                        logger.warning(
                            "Neither stitching frame offset is 0 (lfo=%s rfo=%s); using %s audio.",
                            lfo,
                            rfo,
                            baseline,
                        )
                    if baseline == "left":
                        args.audio_sources = list(game_videos.get("left") or [])
                    else:
                        args.audio_sources = list(game_videos.get("right") or [])
                except Exception:
                    args.audio_sources = None

                def _set_runtime_arg(name: str, value: Any) -> None:
                    setattr(args, name, value)
                    if hasattr(args, "initial_args") and isinstance(args.initial_args, dict):
                        args.initial_args[name] = value
                    if isinstance(args.game_config, dict):
                        init_args = args.game_config.get("initial_args")
                        if isinstance(init_args, dict):
                            init_args[name] = value

                _set_runtime_arg("stitch_pto_project_file", str(pto_project_file))
                args.stitch_data_pipeline = data_pipeline

                if use_aspen_stitching:
                    # Enable the UI slider without a StitchDataset instance.
                    args.stitch_rotation_controller = _StitchRotationController(args.game_config)

                    frame_step_left = 1
                    frame_step_right = 1
                    if left_vid.fps > right_vid.fps:
                        int_ratio = int(left_vid.fps // right_vid.fps)
                        float_ratio = float(left_vid.fps / right_vid.fps)
                        if math.isclose(float(int_ratio), float_ratio) and int_ratio != 1:
                            frame_step_left = int_ratio
                    elif right_vid.fps > left_vid.fps:
                        int_ratio = int(right_vid.fps // left_vid.fps)
                        float_ratio = float(right_vid.fps / left_vid.fps)
                        if math.isclose(float(int_ratio), float_ratio) and int_ratio != 1:
                            frame_step_right = int_ratio

                    game_id = os.path.basename(str(dir_name))
                    left_loader = MOTLoadVideoWithOrig(
                        path=game_videos["left"],
                        game_id=game_id,
                        max_frames=args.max_frames,
                        batch_size=args.batch_size,
                        start_frame_number=args.start_frame + lfo,
                        original_image_only=True,
                        dtype=torch.uint8,
                        device=gpus["stitching"],
                        decoder_device=(
                            torch.device(args.decoder_device) if args.decoder_device else None
                        ),
                        frame_step=frame_step_left,
                        no_cuda_streams=args.no_cuda_streams,
                        image_channel_adders=None,
                        checkerboard_input=args.checkerboard_input,
                    )
                    right_loader = MOTLoadVideoWithOrig(
                        path=game_videos["right"],
                        game_id=game_id,
                        max_frames=args.max_frames,
                        batch_size=args.batch_size,
                        start_frame_number=args.start_frame + rfo,
                        original_image_only=True,
                        dtype=torch.uint8,
                        device=gpus["stitching"],
                        decoder_device=(
                            torch.device(args.decoder_device) if args.decoder_device else None
                        ),
                        frame_step=frame_step_right,
                        no_cuda_streams=args.no_cuda_streams,
                        image_channel_adders=None,
                        checkerboard_input=args.checkerboard_input,
                    )
                    stitch_inputs = MultiDataLoaderWrapper(
                        dataloaders=[left_loader, right_loader],
                    )
                    dataloader.append_dataset("stitch_inputs", stitch_inputs)
                else:
                    # Optional per-camera stitching color pipelines from Aspen config
                    left_stitch_pipeline_cfg = None
                    right_stitch_pipeline_cfg = None
                    if aspen_cfg_for_pipeline and isinstance(aspen_cfg_for_pipeline, dict):
                        left_stitch_pipeline_cfg = aspen_cfg_for_pipeline.get(
                            "left_stitch_pipeline"
                        )
                        right_stitch_pipeline_cfg = aspen_cfg_for_pipeline.get(
                            "right_stitch_pipeline"
                        )
                    stitched_dataset = StitchDataset(
                        videos=stitch_videos,
                        pto_project_file=pto_project_file,
                        start_frame_number=args.start_frame,
                        max_frames=args.max_frames,
                        image_roi=None,
                        batch_size=args.batch_size,
                        remapping_device=gpus["stitching"],
                        decoder_device=(
                            torch.device(args.decoder_device) if args.decoder_device else None
                        ),
                        blend_mode=opts.blend_mode,
                        dtype=torch.float if not args.fp16_stitch else torch.half,
                        auto_adjust_exposure=args.stitch_auto_adjust_exposure,
                        python_blender=args.python_blender,
                        minimize_blend=preferred_arg(
                            getattr(args, "minimize_blend", None), not args.no_minimize_blend
                        ),
                        no_cuda_streams=args.no_cuda_streams,
                        post_stitch_rotate_degrees=getattr(args, "stitch_rotate_degrees", None),
                        profiler=getattr(args, "profiler", None),
                        config_ref=args.game_config,
                        left_color_pipeline=left_stitch_pipeline_cfg,
                        right_color_pipeline=right_stitch_pipeline_cfg,
                        capture_rgb_stats=bool(getattr(args, "checkerboard_input", False)),
                        checkerboard_input=args.checkerboard_input,
                    )
                    # Expose the StitchDataset instance so PlayTracker can control
                    # post-stitch rotation via the UI slider.
                    args.stitch_rotation_controller = stitched_dataset
                    # Create the MOT video data loader, passing it the
                    # stitching data loader as its image source
                    mot_dataloader = MOTLoadVideoWithOrig(
                        path=None,
                        game_id=dir_name,
                        start_frame_number=args.start_frame,
                        batch_size=1,  # This batch will contain one batch of whatever the stitcher's batch size is
                        embedded_data_loader=stitched_dataset,
                        data_pipeline=data_pipeline,
                        dtype=torch.float if not args.fp16 else torch.half,
                        device=gpus["stitching"],
                        original_image_only=False,
                        adjust_exposure=args.adjust_exposure,
                        no_cuda_streams=args.no_cuda_streams,
                        checkerboard_input=args.checkerboard_input,
                    )
                    try:
                        mot_dataloader.set_profiler(getattr(args, "profiler", None))
                    except Exception:
                        pass
                    dataloader.append_dataset("pano", mot_dataloader)
            else:
                assert len(input_video_files) == 1
                if isinstance(aspen_cfg_for_pipeline, dict):
                    stitching_cfg = aspen_cfg_for_pipeline.get("stitching")
                    if isinstance(stitching_cfg, dict):
                        stitching_cfg["enabled"] = False
                    plugins_cfg = aspen_cfg_for_pipeline.get("plugins")
                    if isinstance(plugins_cfg, dict):
                        stitching_plugin = plugins_cfg.get("stitching")
                        if isinstance(stitching_plugin, dict):
                            stitching_plugin["enabled"] = False
                if os.path.isdir(input_video_files[0]):
                    dir_name = input_video_files[0]
                else:
                    dir_name = Path(input_video_files[0]).parent
                assert not args.start_frame or not args.start_frame_time
                if not args.start_frame and args.start_frame_time:
                    vid_info = BasicVideoInfo(input_video_files[0])
                    args.start_frame = time_to_frame(
                        time_str=args.start_frame_time, fps=vid_info.fps
                    )

                assert not args.max_frames or not args.max_time
                if not args.max_frames and args.max_time:
                    vid_info = BasicVideoInfo(input_video_files[0])
                    args.max_frames = time_to_frame(time_str=args.max_time, fps=vid_info.fps)
                pano_dataloader = MOTLoadVideoWithOrig(
                    path=input_video_files[0],
                    start_frame_number=args.start_frame,
                    batch_size=args.batch_size,
                    max_frames=args.max_frames,
                    device=main_device,
                    decoder_device=(
                        torch.device(args.decoder_device) if args.decoder_device else None
                    ),
                    data_pipeline=data_pipeline,
                    dtype=torch.float if not args.fp16 else torch.half,
                    # When a data_pipeline is provided, we must deliver both
                    # the preprocessed pano and original_images; disable the
                    # original_image_only fast path in this mode.
                    original_image_only=False,
                    adjust_exposure=args.adjust_exposure,
                    no_cuda_streams=args.no_cuda_streams,
                    async_mode=not args.no_async_dataset,
                    checkerboard_input=bool(getattr(args, "checkerboard_input", False)),
                )
                try:
                    pano_dataloader.set_profiler(getattr(args, "profiler", None))
                except Exception:
                    pass
                dataloader.append_dataset("pano", pano_dataloader)

            if args.end_zones:
                # Try far_left and far_right videos if they exist
                other_videos: List[Tuple[str, str]] = [
                    ("far_left", os.path.join(dir_name, "far_left.mp4")),
                    ("far_right", os.path.join(dir_name, "far_right.mp4")),
                ]
                ez_count = 0
                for vid_name, vid_path in other_videos:
                    if os.path.exists(vid_path):
                        extra_dataloader = MOTLoadVideoWithOrig(
                            path=vid_path,
                            start_frame_number=args.start_frame,
                            batch_size=args.batch_size,
                            dtype=torch.float if not args.fp16 else torch.half,
                            device=gpus["encoder"],
                            original_image_only=True,
                            no_cuda_streams=args.no_cuda_streams,
                            async_mode=args.no_async_dataset,
                            checkerboard_input=bool(getattr(args, "checkerboard_input", False)),
                        )
                    try:
                        extra_dataloader.set_profiler(getattr(args, "profiler", None))
                    except Exception:
                        pass
                    dataloader.append_dataset(vid_name, extra_dataloader)
                    ez_count += 1
                if not ez_count:
                    raise ValueError("--end-zones specified, but no end-zone videos found")

        if dataloader is None:
            raise ValueError("Dataloader could not be constructed")

        if not args.no_progress_bar:
            table_map = OrderedDict()
            if is_stitching(args.input_video):
                table_map["Stitching"] = "ENABLED"

            progress_bar = ProgressBar(
                total=len(dataloader),
                scroll_output=ScrollOutput(lines=args.progress_bar_lines).register_logger(logger),
                update_rate=args.print_interval,
                table_map=table_map,
                title=args.game_id,
                use_curses=getattr(args, "curses_progress", False),
            )
        else:
            progress_bar = None

        output_video_path = None
        if not args.no_save_video:
            output_video_path = args.output_video or os.path.join(
                results_folder, "tracking_output.mkv"
            )
            try:
                out_dir = os.path.dirname(str(output_video_path))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass
        args.output_video_path = output_video_path

        #
        # Audio passthrough: provide input audio sources + start offset to Aspen VideoOutPlugin.
        #
        try:
            audio_sources = getattr(args, "audio_sources", None)
            if audio_sources is None and args.input_video:
                audio_sources = input_video_files
                if isinstance(audio_sources, list) and len(audio_sources) == 1:
                    audio_sources = audio_sources[0]
                args.audio_sources = audio_sources
            args.audio_start_seconds = 0.0
            if audio_sources and args.start_frame:
                ref_path = None
                if isinstance(audio_sources, dict):
                    left_files = audio_sources.get("left") or []
                    if left_files:
                        ref_path = left_files[0]
                elif isinstance(audio_sources, list):
                    ref_path = audio_sources[0] if audio_sources else None
                else:
                    ref_path = str(audio_sources)
                if ref_path:
                    ref_info = BasicVideoInfo(str(ref_path))
                    fps_val = float(ref_info.fps) if float(ref_info.fps) > 0 else 0.0
                    if fps_val > 0:
                        args.audio_start_seconds = float(args.start_frame) / fps_val
        except Exception:
            logger.exception("Failed to configure audio passthrough; continuing without audio.")
            args.audio_sources = None
            args.audio_start_seconds = 0.0

        if args.audio_only:
            if args.no_audio:
                logger.info("--audio-only specified with --no-audio; skipping audio mux.")
                return
            target_video = args.output_video or output_video_path
            if not target_video:
                raise ValueError(
                    "--audio-only requires --output-video (or an existing output path)."
                )
            if not os.path.exists(target_video):
                raise FileNotFoundError(f"--audio-only target video does not exist: {target_video}")
            if not args.audio_sources:
                raise ValueError("--audio-only requires --input-video with an audio track.")
            if has_audio_stream(target_video):
                logger.info("Target video already has audio: %s", target_video)
                return
            result = mux_audio_in_place(
                input_audio=args.audio_sources,
                video_path=target_video,
                start_seconds=float(args.audio_start_seconds or 0.0),
                shortest=True,
                keep_original=False,
            )
            if not result:
                raise RuntimeError(f"Failed to mux audio into: {target_video}")
            logger.info("Saved video with audio: %s", result)
            return

        if not args.audio_only:

            if not args.output_video_bit_rate:
                args.output_video_bit_rate = dataloader.get_max_attribute("bit_rate")

            if not args.no_play_tracking:

                #
                # Video output pipeline
                #
                video_out_pipeline = None
                if model is not None and hasattr(model, "cfg"):
                    video_out_pipeline = getattr(model.cfg, "video_out_pipeline")
                else:
                    # Pull from unified Aspen config if available
                    if aspen_cfg_for_pipeline:
                        video_out_pipeline = aspen_cfg_for_pipeline.get("video_out_pipeline")
                if video_out_pipeline:
                    video_out_pipeline = copy.deepcopy(video_out_pipeline)
                # Make video_out_pipeline available to Aspen plugins via args
                args.video_out_pipeline = video_out_pipeline
            postprocessor = None

            other_kwargs = {
                "dataloader": dataloader,
                "postprocessor": postprocessor,
            }

            run_mmtrack(
                model=model,
                config=vars(args),
                device=main_device,
                fp16=args.fp16,
                input_cache_size=args.cache_size,
                progress_bar=progress_bar,
                no_cuda_streams=args.no_cuda_streams,
                profiler=getattr(args, "profiler", None),
                **other_kwargs,
            )

        #
        # Deploy CSV artifacts to the game directory (full run) or --deploy-dir (explicit).
        #
        deploy_dir = getattr(args, "deploy_dir", None)
        target_deploy_dir = None
        if deploy_dir:
            target_deploy_dir = deploy_dir
        elif not bool(args.max_time or args.max_frames):
            target_deploy_dir = (
                args.game_dir if args.game_dir and os.path.isdir(args.game_dir) else None
            )

        if target_deploy_dir:
            os.makedirs(target_deploy_dir, exist_ok=True)
            deployed_video_path = None
            if output_video_path and os.path.exists(output_video_path):
                try:
                    out_abs = os.path.abspath(output_video_path)
                    deploy_abs = os.path.abspath(target_deploy_dir)
                    if os.path.dirname(out_abs) == deploy_abs:
                        deployed_video_path = out_abs
                    else:
                        base_name = os.path.basename(output_video_path)
                        base_root, base_ext = os.path.splitext(base_name)
                        candidate = None
                        for i in range(0, 1000):
                            if i == 0:
                                name_i = f"{base_root}{base_ext}"
                            else:
                                name_i = f"{base_root}-{i}{base_ext}"
                            cand = os.path.join(target_deploy_dir, name_i)
                            if not os.path.exists(cand):
                                candidate = cand
                                break
                        if candidate is None:
                            raise RuntimeError(
                                "Could not find a free deploy filename for output video"
                            )
                        shutil.copy2(output_video_path, candidate)
                        deployed_video_path = candidate
                except Exception:
                    logger.exception("Failed to deploy output video; continuing.")
                    deployed_video_path = None
            csv_names = []
            try:
                for name in os.listdir(results_folder):
                    if not name.endswith(".csv"):
                        continue
                    src_path = os.path.join(results_folder, name)
                    if os.path.isfile(src_path):
                        csv_names.append(name)
            except Exception:
                traceback.print_exc()
                csv_names = []

            def extract_suffix_num(path: Optional[os.PathLike | str]) -> Optional[int]:
                if not path:
                    return None
                base = os.path.splitext(os.path.basename(str(path)))[0]
                dash_idx = base.rfind("-")
                if dash_idx == -1:
                    return None
                tail = base[dash_idx + 1 :]
                if tail.isdigit():
                    return int(tail)
                return None

            def with_index(name: str, suffix_num: int) -> str:
                root, ext = os.path.splitext(name)
                if suffix_num <= 0:
                    return f"{root}{ext}"
                return f"{root}-{suffix_num}{ext}"

            def choose_free_suffix(names: List[str]) -> int:
                for i in range(0, 1000):
                    collision = False
                    for name in names:
                        if os.path.exists(os.path.join(target_deploy_dir, with_index(name, i))):
                            collision = True
                            break
                    if not collision:
                        return i
                raise RuntimeError("Could not find a free suffix for CSV deployment")

            suffix_num = extract_suffix_num(
                deployed_video_path or args.output_video or output_video_path
            )
            if suffix_num is not None and csv_names:
                for name in csv_names:
                    if os.path.exists(
                        os.path.join(target_deploy_dir, with_index(name, suffix_num))
                    ):
                        suffix_num = None
                        break
            if suffix_num is None and csv_names:
                suffix_num = choose_free_suffix(csv_names)

            for name in csv_names:
                src_path = os.path.join(results_folder, name)
                dst_path = os.path.join(target_deploy_dir, with_index(name, int(suffix_num or 0)))
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception:
                    traceback.print_exc()
        logger.info("Completed")
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        try:
            if postprocessor is not None:
                try:
                    postprocessor.stop()
                except Exception:
                    traceback.print_exc()
            if dataloader is not None and hasattr(dataloader, "close"):
                try:
                    dataloader.close()
                except Exception:
                    traceback.print_exc()
        except Exception as ex:
            print(f"Exception while shutting down: {ex}")


def setup_logging():
    root_logger = get_root_logger()
    root_logger.setLevel(20)


def main():
    setup_logging()

    # Prefer CUDA, but don't hard-fail if the runtime isn't detected so CPU-only
    # runs can still proceed (albeit slowly).
    if not torch.cuda.is_available():
        logger.warning("CUDA not detected; running hmtrack on CPU will be very slow.")
    elif not torch.backends.cudnn.is_available():
        logger.warning("cuDNN not detected; performance may be degraded.")

    parser = make_parser()
    args = parser.parse_args()

    game_config = get_config(
        game_id=args.game_id, rink=args.rink, camera=args.camera_name, root_dir=args.root_dir
    )

    # Merge user-provided YAML configs in order (--config can be repeated).
    # Later files override earlier values.
    from hmlib.config import load_yaml_files_ordered, recursive_update

    def _split_and_strip(items):
        paths = []
        if not items:
            return paths
        for it in items:
            if not it:
                continue
            parts = [p.strip() for p in str(it).split(",") if p.strip()]
            paths.extend(parts)
        return paths

    additional_cfg_paths = _split_and_strip(args.config)
    if not additional_cfg_paths:
        default_aspen = os.path.join(ROOT_DIR, "config", "aspen", "tracking.yaml")
        if os.path.exists(default_aspen):
            additional_cfg_paths.append(default_aspen)
    if additional_cfg_paths:
        merged_extra = load_yaml_files_ordered(additional_cfg_paths)
        if merged_extra:
            game_config = recursive_update(game_config, merged_extra)

    # Apply CLI-driven config overrides before resolving GLOBAL.* references so
    # Aspen plugins see the updated values via GLOBAL.*.
    try:
        hm_opts.apply_arg_config_overrides(game_config, args)
    except Exception:
        # Config overrides are non-fatal; fall back to config defaults on error.
        pass

    # Let hm_opts apply --config-override before resolving GLOBAL.* refs.
    args.game_config = game_config
    args = hm_opts.init(args, parser)
    game_config = resolve_global_refs(args.game_config)
    args.game_config = game_config

    # Set up the task flags
    args.tracking = False
    tokens = args.tasks.split(",")
    for t in tokens:
        setattr(args, t, True)

    game_config["initial_args"] = vars(args)
    args.game_config = game_config

    args = configure_model(config=args.game_config, args=args)

    if getattr(args, "smoke_test", False):
        game_dir = None
        if getattr(args, "game_id", None):
            try:
                game_dir = get_game_dir(args.game_id, assert_exists=False)
            except Exception:
                game_dir = None

        # Validate imports for the pieces hmtrack expects to have available.
        import hockeymom._hockeymom  # noqa: F401
        import lightglue  # noqa: F401
        import mmcv  # noqa: F401
        import mmdet  # noqa: F401
        import mmengine  # noqa: F401
        import mmpose  # noqa: F401
        import mmyolo  # noqa: F401

        print(f"Smoke test OK. game_id={getattr(args, 'game_id', None)} game_dir={game_dir}")
        return 0

    if args.game_id:
        num_gpus = 1
    else:
        if isinstance(args.gpus, str):
            args.gpus = [int(g) for g in args.gpus.split(",")]
        num_gpus = len(args.gpus) if args.gpus else 0
        num_gpus = min(num_gpus, torch.cuda.device_count())

    # Optional Python cProfile
    if getattr(args, "py_trace_out", None):
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        try:
            _main(args, num_gpus)
        finally:
            pr.disable()
            out_path = args.py_trace_out
            try:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            except Exception:
                pass
            if out_path.endswith(".txt"):
                with open(out_path, "w") as f:
                    ps = pstats.Stats(pr, stream=f)
                    ps.sort_stats("cumulative").print_stats()
            else:
                pr.dump_stats(out_path)
    else:
        _main(args, num_gpus)
    print("Done.")


if __name__ == "__main__":
    try:
        # Prefer a dedicated CUDA stream when available; otherwise fall back to CPU-friendly start.
        if torch.cuda.is_available():
            with torch.cuda.stream(torch.cuda.Stream(torch.device("cuda"))):
                main()
        else:
            main()
    except Exception as e:
        print(f"Exception during processing: {e}")
        traceback.print_exc()
        # Debug: list live threads to help diagnose hangs where an error
        # has been raised but the process does not exit promptly.
        try:
            import threading

            print("Live threads after exception:")
            for t in threading.enumerate():
                try:
                    print(f" - {t.name} (daemon={t.daemon})")
                except Exception:
                    print(f" - {t}")
        except Exception:
            pass
        raise SystemExit(1)

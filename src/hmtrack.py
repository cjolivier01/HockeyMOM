import argparse
import copy
import logging
import os
import traceback
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.backends.cudnn as cudnn
from mmcv.transforms import Compose
from mmdet.apis import init_track_model
from torch.nn.parallel import DistributedDataParallel as DDP

import hmlib.tracking_utils.ice_rink_segm_boundaries
import hmlib.tracking_utils.segm_boundaries
from hmlib.camera.cam_post_process import DefaultArguments
from hmlib.camera.camera_head import CamTrackHead
from hmlib.config import (
    get_clip_box,
    get_config,
    get_game_dir,
    get_nested_value,
    set_nested_value,
    update_config,
)
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.datasets.dataset.multi_dataset import MultiDatasetWrapper
from hmlib.datasets.dataset.stitching_dataloader2 import StitchDataset
from hmlib.game_audio import transfer_audio
from hmlib.hm_opts import copy_opts, hm_opts
from hmlib.hm_transforms import update_data_pipeline
from hmlib.log import get_root_logger, logger
from hmlib.orientation import configure_game_videos
from hmlib.stitching.configure_stitching import configure_video_stitching
from hmlib.tasks.tracking import run_mmtrack
from hmlib.utils.checkpoint import load_checkpoint_to_model
from hmlib.utils.gpu import GpuAllocator, select_gpus
from hmlib.utils.mot_data import MOTTrackingData
from hmlib.utils.pipeline import get_pipeline_item, update_pipeline_item
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import time_to_frame

ROOT_DIR = os.getcwd()


def make_parser(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser("HockeyMOM Tracking")
    parser = hm_opts.parser(parser)
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
    parser.add_argument(
        "--infer",
        default=False,
        action="store_true",
        help="Run inference instead of validation",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--tracker",
        default="mmtrack",
        type=str,
        help="Use tracker type [hm|fair|mixsort|micsort_oc|sort|ocsort|byte|deepsort|motdt]",
    )
    parser.add_argument(
        "--no_save_video",
        "--no-save-video",
        dest="no_save_video",
        action="store_true",
        help="Don't save the output video",
    )
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
    #
    # GPUs/Devices
    #
    # parser.add_argument(
    #     "--detection-gpu",
    #     help="GPU used for detections",
    #     type=int,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--decoder-gpu",
    #     help="GPU used for video decoding",
    #     type=int,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--encoder-gpu",
    #     help="GPU used for video encoding",
    #     type=int,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--cam-tracking-gpu",
    #     help="GPU used for camera tracking trunk (default us CPU)",
    #     type=int,
    #     default=None,
    # )
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
    parser.add_argument(
        "--camera",
        default="GoPro",
        type=str,
        help="Camera name",
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
        "--stitch",
        "--force-stitching",
        "--force_stitching",
        dest="force_stitching",
        action="store_true",
        help="force video stitching",
    )
    parser.add_argument(
        "--plot-tracking", action="store_true", help="plot individual tracking boxes"
    )
    parser.add_argument(
        "--detect-jersey-numbers", action="store_true", help="Detect jersey numbers"
    )
    parser.add_argument(
        "--plot-jersey-numbers", action="store_true", help="plot individual jersey numbers"
    )
    parser.add_argument("--plot-pose", action="store_true", help="plot individual pose skeletons")
    parser.add_argument(
        "--plot-all-detections",
        type=float,
        default=None,
        help="plot all detections above this given accuracy",
    )
    parser.add_argument(
        "--plot-moving-boxes",
        action="store_true",
        help="plot moving camera tracking boxes",
    )
    parser.add_argument(
        "--test-size", type=str, default=None, help="WxH of test box size (format WxH)"
    )
    parser.add_argument("--no-crop", action="store_true", help="Don't crop output image")
    parser.add_argument(
        "--save-frame-dir",
        type=str,
        default=None,
        help="directory to save the output video frases as png files",
    )
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        type=str,
        default="tracking",
        help="Comma-separated task list (tracking, multi_pose)",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--min-box-area", type=float, default=100, help="filter out tiny boxes")
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    parser.add_argument(
        "--input-video",
        type=str,
        default=None,
        help="Input video file(s)",
    )
    parser.add_argument(
        "--input-tracking-data",
        type=str,
        default=None,
        help="Input tracking data file and use instead of AI calling tracker",
    )
    parser.add_argument(
        "--save-tracking-data",
        action="store_true",
        help="Save tracking data to results.csv",
    )
    parser.add_argument(
        "--save-camera-data",
        action="store_true",
        help="Save tracking data to camera.csv",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only transfer the audio",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="The output video file name",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Tracking checkpoint file")
    parser.add_argument("--detector", help="det checkpoint file")
    parser.add_argument("--reid", help="reid checkpoint file")

    # Pose args
    parser.add_argument("--pose-config", type=str, default=None, help="Pose config file")
    parser.add_argument("--pose-checkpoint", type=str, default=None, help="Pose checkpoint file")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint score threshold")
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Bounding box score threshold")
    parser.add_argument("--radius", type=int, default=4, help="Keypoint radius for visualization")
    parser.add_argument("--thickness", type=int, default=1, help="Link thickness for visualization")
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
    exts = ["mkv", "avi", "mp4"]
    basenames = [
        "stitched_output",
        "stitched_output-with-audio",
        "stitched_output-" + game_id,
        "stitched_output-with-audio-" + game_id,
    ]
    for basename in basenames:
        for ext in exts:
            path = os.path.join(dir_name, basename + "." + ext)
            if os.path.exists(path):
                return path
    return None


class FakeExp:
    def __init__(self):
        self.test_size = None


def is_stitching(input_video: str) -> bool:
    if not input_video:
        raise AttributeError("No valid input video specified")
    input_video_files = input_video.split(",")
    return len(input_video_files) == 2 or os.path.isdir(args.input_video)


def configure_boundaries(
    game_id: str,
    model: torch.nn.Module,
    top_border_lines,
    bottom_border_lines,
    original_clip_box,
    gpu_allocator: GpuAllocator,
    plot_tracking: bool,
):
    if hasattr(model, "post_detection_pipeline"):
        has_boundaries = False
        if top_border_lines or bottom_border_lines:
            # Manual boundaries
            has_boundaries = update_pipeline_item(
                model.post_detection_pipeline,
                "BoundaryLines",
                dict(
                    upper_border_lines=top_border_lines,
                    lower_border_lines=bottom_border_lines,
                    original_clip_box=original_clip_box,
                ),
            )
        if not has_boundaries:
            # Try auto-boundaries
            has_boundaries = update_pipeline_item(
                model.post_detection_pipeline,
                "IceRinkSegmBoundaries",
                dict(
                    game_id=game_id,
                    original_clip_box=original_clip_box,
                    draw=plot_tracking,
                    # gpu_allocator=gpu_allocator,
                ),
            )


def main(args, num_gpu):
    dataloader = None
    tracking_data = None

    opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    try:

        is_distributed = num_gpu > 1
        if args.gpus and isinstance(args.gpus, str):
            args.gpus = [int(i) for i in args.gpus.split(",")]

        # set environment variables for distributed training
        cudnn.benchmark = True

        rank = args.local_rank

        game_config = args.game_config

        tracker = get_nested_value(game_config, "model.tracker.type")
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

        args.multi_pose |= args.plot_pose

        cam_args = DefaultArguments(
            game_config=game_config,
            basic_debugging=args.debug,
            output_video_path=args.output_file,
            opts=args,
        )
        cam_args.show_image = args.show_image
        cam_args.crop_output_image = not args.no_crop

        if args.cvat_output:
            cam_args.crop_output_image = False
            cam_args.fixed_edge_rotation = False
            cam_args.apply_fixed_edge_scaling = False

        cam_args.plot_individual_player_tracking = args.plot_tracking
        if cam_args.plot_individual_player_tracking:
            cam_args.plot_boundaries = True

        # See if gameid is in videos
        if not args.input_video and args.game_id:
            game_video_dir = get_game_dir(args.game_id)
            if game_video_dir:
                # TODO: also look for avi and mp4 files
                if args.force_stitching:
                    args.input_video = game_video_dir
                else:
                    pre_stitched_file_name = find_stitched_file(
                        dir_name=game_video_dir, game_id=args.game_id
                    )
                    if pre_stitched_file_name and os.path.exists(pre_stitched_file_name):
                        args.input_video = pre_stitched_file_name
                    else:
                        args.input_video = game_video_dir

        # TODO: get rid of this, set in cfg (detector.input_size, etc)
        exp = None
        if exp is None:
            exp = FakeExp()
            test_size = getattr(args, "test_size", None)
            if not test_size:
                test_size = (1088, 608)
            elif isinstance(test_size, str):
                tokens = test_size.split("x")
                if len(tokens) == 1:
                    tokens = args.test_size.split("X")
                assert len(tokens) == 2
                test_size = (
                    to_32bit_mul(int(tokens[0])),
                    to_32bit_mul(int(tokens[1])),
                )
            exp.test_size = test_size
            args.test_size = test_size
            results_folder = os.path.join(".", "output_workdirs", args.game_id)
            os.makedirs(results_folder, exist_ok=True)

        if args.save_tracking_data or args.input_tracking_data:
            tracking_data = MOTTrackingData(
                input_file=args.input_tracking_data,
                output_file=(
                    os.path.join(results_folder, "results.csv")
                    if args.input_tracking_data is None
                    else None
                ),
                write_interval=100,
            )

        using_precalculated_tracking = tracking_data is not None and tracking_data.has_input_data()

        actual_device_count = torch.cuda.device_count()
        if not actual_device_count:
            raise Exception("At leats one GPU is required for this application")
        while len(args.gpus) > actual_device_count:
            del args.gpus[-1]

        gpus, is_single_lowmem_gpu, gpu_allocator = select_gpus(
            allowed_gpus=args.gpus,
            is_stitching=is_stitching(args.input_video),
            is_multipose=args.multi_pose,
            is_detecting=not using_precalculated_tracking,
        )
        if is_single_lowmem_gpu:
            print("Adjusting configuration for a single low-memory GPU environment...")
            args.cache_size = 0
            args.stitch_cache_size = 0
            # args.batch_size = 1

        # This would be way too slow on CPU
        assert torch.cuda.is_available()
        main_device = torch.device("cuda")
        for name in ["detection", "stitching", "encoder"]:
            if name in gpus:
                main_device = gpus[name]
                torch.cuda.set_device(main_device)
                break

        # Set up for pose
        pose_model = None
        pose_dataset = None
        pose_dataset_info = None

        data_pipeline = None

        if not args.exp_file:
            args.exp_file = get_nested_value(game_config, "model.end_to_end.config")
        if not args.checkpoint:
            args.checkpoint = get_nested_value(game_config, "model.end_to_end.checkpoint")

        args.config = args.exp_file
        if not using_precalculated_tracking:
            if args.tracking or args.multi_pose:

                # build the model from a config file and a checkpoint file
                model = init_track_model(
                    args.config,
                    args.checkpoint,
                    args.detector,
                    args.reid,
                    device=main_device,
                )

                # Maybe apply a clip box in the data pipeline
                orig_clip_box = get_clip_box(game_id=args.game_id, root_dir=args.root_dir)
                if orig_clip_box:
                    hm_crop = get_pipeline_item(model.cfg.inference_pipeline, "HmCrop")
                    if hm_crop is not None:
                        hm_crop["rectangle"] = orig_clip_box

                if args.checkpoint:
                    load_checkpoint_to_model(model, args.checkpoint)
                cfg = model.cfg.copy()
                pipeline = cfg.inference_pipeline
                pipeline[0].type = "HmLoadImageFromWebcam"

                update_pipeline_item(
                    pipeline,
                    "IceRinkSegmConfig",
                    dict(
                        game_id=args.game_id,
                    ),
                )

                data_pipeline = Compose(pipeline)

            #
            # post-detection pipeline updates
            #
            configure_boundaries(
                game_id=args.game_id,
                model=model,
                top_border_lines=cam_args.top_border_lines,
                bottom_border_lines=cam_args.bottom_border_lines,
                original_clip_box=get_clip_box(game_id=args.game_id, root_dir=args.root_dir),
                gpu_allocator=gpu_allocator,
                plot_tracking=args.plot_tracking,
            )
        if args.multi_pose:
            from mmpose.apis import init_pose_model
            from mmpose.datasets import DatasetInfo

            if not args.pose_config:
                args.pose_config = get_nested_value(game_config, "model.pose.config")
            if not args.pose_checkpoint:
                args.pose_config = get_nested_value(game_config, "model.pose.checkpoint")

            pose_model = init_pose_model(
                args.pose_config, args.pose_checkpoint, device=gpus["multipose"]
            )
            pose_model.cfg.test_pipeline = update_data_pipeline(pose_model.cfg.test_pipeline)
            pose_dataset = pose_model.cfg.data["test"]["type"]
            pose_dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
            if pose_dataset_info is None:
                warnings.warn(
                    "Please set `dataset_info` in the config."
                    "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
                    DeprecationWarning,
                )
            else:
                pose_dataset_info = DatasetInfo(pose_dataset_info)

        dataloader = MultiDatasetWrapper()
        postprocessor = None
        if args.input_video:
            input_video_files = args.input_video.split(",")
            if is_stitching(args.input_video):
                project_file_name = "hm_project.pto"

                game_videos = {}

                if len(input_video_files) == 2:
                    vl = input_video_files[0]
                    vr = input_video_files[1]
                    dir_name = os.path.dirname(vl)
                    file_name, file_extension = os.path.splitext(os.path.basename(vl))
                    video_left = file_name + file_extension
                    file_name, file_extension = os.path.splitext(os.path.basename(vr))
                    video_right = file_name + file_extension
                    assert dir_name == os.path.dirname(vr)
                elif os.path.isdir(args.input_video):
                    game_videos = configure_game_videos(game_id=args.game_id)
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
                # Create the stitcher data loader
                output_stitched_video_file = (
                    os.path.join(".", f"stitched_output-{args.game_id}.mkv")
                    if args.save_stitched
                    else None
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

                stitched_dataset = StitchDataset(
                    videos=stitch_videos,
                    pto_project_file=pto_project_file,
                    start_frame_number=args.start_frame,
                    output_stitched_video_file=(
                        output_stitched_video_file if args.save_stitched else None
                    ),
                    max_frames=args.max_frames,
                    max_input_queue_size=args.cache_size,
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
                    minimize_blend=not args.no_minimize_blend,
                )
                # Create the MOT video data loader, passing it the
                # stitching data loader as its image source
                mot_dataloader = MOTLoadVideoWithOrig(
                    path=None,
                    game_id=dir_name,
                    img_size=exp.test_size,
                    start_frame_number=args.start_frame,
                    batch_size=1,
                    embedded_data_loader=stitched_dataset,
                    embedded_data_loader_cache_size=(
                        args.cache_size
                        if args.stitch_cache_size is None
                        else args.stitch_cache_size
                    ),
                    data_pipeline=data_pipeline,
                    stream_tensors=tracker == "mmtrack",
                    dtype=torch.float if not args.fp16 else torch.half,
                    device=gpus["stitching"],
                    original_image_only=tracking_data is not None,
                    adjust_exposure=args.adjust_exposure,
                )
                dataloader.append_dataset("pano", mot_dataloader)
            else:
                assert len(input_video_files) == 1
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
                    img_size=exp.test_size,
                    start_frame_number=args.start_frame,
                    batch_size=args.batch_size,
                    max_frames=args.max_frames,
                    device=main_device,
                    decoder_device=(
                        torch.device(args.decoder_device) if args.decoder_device else None
                    ),
                    data_pipeline=data_pipeline,
                    dtype=torch.float if not args.fp16 else torch.half,
                    original_image_only=tracking_data is not None,
                    adjust_exposure=args.adjust_exposure,
                )
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
                            img_size=None,
                            start_frame_number=args.start_frame,
                            batch_size=args.batch_size,
                            stream_tensors=tracker == "mmtrack",
                            dtype=torch.float if not args.fp16 else torch.half,
                            device=gpus["encoder"],
                            original_image_only=True,
                        )
                        dataloader.append_dataset(vid_name, extra_dataloader)
                        ez_count += 1
                if not ez_count:
                    raise ValueError("--end-zones specified, but no end-zone videos found")

        if dataloader is None:
            dataloader = exp.get_eval_loader(
                args.batch_size, is_distributed, args.test, return_origin_img=True
            )

        if not args.no_progress_bar:
            table_map = OrderedDict()
            if is_stitching(args.input_video):
                table_map["Stitching"] = "ENABLED"

            progress_bar = ProgressBar(
                total=len(dataloader),
                scroll_output=ScrollOutput(lines=args.progress_bar_lines).register_logger(logger),
                update_rate=args.print_interval,
                table_map=table_map,
            )
        else:
            progress_bar = None

        save_dir = None
        output_video_path = None
        if not args.no_save_video:
            save_dir = results_folder
            output_video_path = os.path.join(results_folder, "tracking_output.mkv")

        if not args.audio_only:
            if not args.no_play_tracking:

                #
                # Video output pipeline
                #
                video_out_pipeline = getattr(model.cfg, "video_out_pipeline")
                if video_out_pipeline:
                    video_out_pipeline = copy.deepcopy(video_out_pipeline)
                    fixed_edge_rotation_angle = (
                        get_nested_value(game_config, "rink.camera.fixed_edge_rotation_angle", None)
                        if not args.no_rink_rotation
                        else 0
                    )
                    update_pipeline_item(
                        video_out_pipeline,
                        "HmPerspectiveRotation",
                        dict(
                            fixed_edge_rotation_angle=fixed_edge_rotation_angle,
                            fixed_edge_rotation=(
                                fixed_edge_rotation_angle is not None
                                and fixed_edge_rotation_angle != 0
                            ),
                            pre_clip=cam_args.crop_output_image,
                            dtype=torch.float,
                        ),
                    )
                    update_pipeline_item(
                        video_out_pipeline,
                        "HmConfigureScoreboard",
                        dict(
                            game_id=args.game_id,
                        ),
                    )
                    update_pipeline_item(
                        video_out_pipeline,
                        "HmCropToVideoFrame",
                        dict(
                            crop_image=cam_args.crop_output_image,
                        ),
                    )
                postprocessor = CamTrackHead(
                    opt=args,
                    args=cam_args,
                    fps=dataloader.fps if args.output_fps is None else args.output_fps,
                    save_dir=save_dir,
                    output_video_path=output_video_path,
                    save_frame_dir=args.save_frame_dir,
                    original_clip_box=get_clip_box(game_id=args.game_id, root_dir=args.root_dir),
                    device=gpus["camera"],
                    video_out_device=gpus["encoder"],
                    video_out_cache_size=args.cache_size,
                    data_type="mot",
                    camera_name=get_nested_value(game_config, "camera.name"),
                    video_out_pipeline=video_out_pipeline,
                    async_post_processing=True,
                    # async_post_processing=False,
                    # async_video_out=False,
                    async_video_out=True,
                )
                postprocessor._args.skip_final_video_save = args.skip_final_video_save
            else:
                postprocessor = None

            other_kwargs = {
                "dataloader": dataloader,
                "postprocessor": postprocessor,
            }

            run_mmtrack(
                model=model,
                pose_model=pose_model,
                pose_dataset_type=pose_dataset,
                pose_dataset_info=pose_dataset_info,
                config=vars(args),
                device=main_device,
                tracking_data=tracking_data,
                fp16=args.fp16,
                input_cache_size=args.cache_size,
                progress_bar=progress_bar,
                **other_kwargs,
            )

        #
        # Now add the audio
        #
        if output_video_path and os.path.exists(output_video_path):
            transfer_audio(
                game_id=args.game_id,
                input_av_files=input_video_files,
                video_source_file=output_video_path,
                output_av_path=args.output_video,
            )
        logger.info("Completed")
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        try:
            if postprocessor is not None:
                postprocessor.stop()
            if dataloader is not None and hasattr(dataloader, "close"):
                dataloader.close()
            if tracking_data is not None:
                tracking_data.flush()
        except Exception as ex:
            print(f"Exception while shutting down: {ex}")


def tensor_to_image(tensor: torch.Tensor):  ##
    if torch.is_floating_point(tensor):
        tensor = torch.clamp(tensor * 255, min=0, max=255).to(torch.uint8, non_blocking=True)
    return tensor


def setup_logging():
    mm_logger = get_root_logger(level=logging.INFO)
    mm_logger.setLevel(logging.INFO)


if __name__ == "__main__":
    setup_logging()

    parser = make_parser()
    args = parser.parse_args()

    game_config = get_config(
        game_id=args.game_id, rink=args.rink, camera=args.camera, root_dir=args.root_dir
    )

    # Set up the task flags
    args.tracking = False
    args.multi_pose = False
    tokens = args.tasks.split(",")
    for t in tokens:
        setattr(args, t, True)

    game_config["initial_args"] = vars(args)
    if args.tracker is None:
        args.tracker = get_nested_value(game_config, "model.tracker.type")
    elif args.tracker != get_nested_value(game_config, "model.tracker.type"):
        game_config = update_config(
            root_dir=ROOT_DIR,
            baseline_config=game_config,
            config_type="models",
            config_name="tracker_" + args.tracker,
        )
    args.game_config = game_config
    args = hm_opts.init(args)

    args = configure_model(config=args.game_config, args=args)

    if args.game_id:
        num_gpus = 1
    else:
        if isinstance(args.gpus, str):
            args.gpus = [int(g) for g in args.gpus.split(",")]
        num_gpus = len(args.gpus) if args.gpus else 0
        assert num_gpus <= torch.cuda.device_count()

    main(args, num_gpus)
    print("Done.")

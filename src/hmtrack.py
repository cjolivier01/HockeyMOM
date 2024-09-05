import argparse
import logging
import os
import time
import traceback
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

# For TopDownGetBboxCenterScale
import mmpose.datasets.pipelines.top_down_transform
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.utils import get_logger as mmcv_get_logger
from mmdet.datasets.pipelines import Compose
from mmtrack.apis import init_model
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from typeguard import typechecked

import hmlib.models.end_to_end  # Registers the model
from hmlib.camera.cam_post_process import DefaultArguments
from hmlib.camera.camera_head import CamTrackHead
from hmlib.config import (
    GAME_DIR_BASE,
    adjusted_config_path,
    get_clip_box,
    get_config,
    get_nested_value,
    set_nested_value,
    update_config,
)
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.datasets.dataset.multi_dataset import MultiDatasetWrapper
from hmlib.datasets.dataset.stitching_dataloader2 import StitchDataset
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.hm_opts import copy_opts, hm_opts
from hmlib.hm_transforms import update_data_pipeline
from hmlib.stitching.synchronize import configure_video_stitching
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.checkpoint import load_checkpoint_to_model
from hmlib.utils.gpu import CachedIterator, StreamTensor, select_gpus
from hmlib.utils.image import make_channels_first, make_channels_last
from hmlib.utils.mot_data import MOTTrackingData
from hmlib.utils.pipeline import get_pipeline_item
from hmlib.utils.progress_bar import ProgressBar, ScrollOutput, convert_seconds_to_hms
from hmlib.video_stream import time_to_frame

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
    parser.add_argument(
        "--detection-gpu",
        help="GPU used for detections",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--decoder-gpu",
        help="GPU used for video decoding",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--encoder-gpu",
        help="GPU used for video encoding",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--cam-tracking-gpu",
        help="GPU used for camera tracking trunk (default us CPU)",
        type=int,
        default=None,
    )
    # cam args
    parser.add_argument(
        "--cam-ignore-largest",
        default=False,
        action="store_true",
        help="Remove the largest tracking box from the camera set (i.e. at Vallco, a ref is "
        "often right in front of the camera, but not enough of the ref is "
        "visible to note it as a ref)",
    )
    parser.add_argument(
        "--end-zones",
        action="store_true",
        help="Enable end-zone camera usage when available",
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
        "--output-video",
        type=str,
        default=None,
        help="The output video file name",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Tracking checkpoint file")

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


def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    input_video_files = input_video.split(",")
    return len(input_video_files) == 2 or os.path.isdir(args.input_video)


def get_game_dir(game_id: str) -> Optional[str]:
    game_video_dir = os.path.join(GAME_DIR_BASE, game_id)
    if os.path.isdir(game_video_dir):
        return game_video_dir
    return None


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

        gpus = select_gpus(
            allowed_gpus=args.gpus,
            is_stitching=is_stitching(args.input_video),
            is_multipose=args.multi_pose,
            is_detecting=not using_precalculated_tracking,
        )

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
        if tracker == "mmtrack":
            args.config = args.exp_file
            if not using_precalculated_tracking:
                if args.tracking or args.multi_pose:
                    model = init_model(
                        args.config,
                        args.checkpoint,
                        device=main_device,
                    )

                    # Maybe apply a clip box in the data pipeline
                    orig_clip_box = get_clip_box(game_id=args.game_id, root_dir=args.root_dir)
                    if orig_clip_box:
                        hm_crop = get_pipeline_item(model.cfg.data.inference.pipeline, "HmCrop")
                        if hm_crop is not None:
                            hm_crop["rectangle"] = orig_clip_box

                    if args.checkpoint:
                        load_checkpoint_to_model(model, args.checkpoint)
                    cfg = model.cfg.copy()
                    pipeline = cfg.data.inference.pipeline
                    pipeline[0].type = "LoadImageFromWebcam"
                    data_pipeline = Compose(pipeline)

                #
                # post-detection pipeline updates
                #
                if hasattr(model, "post_detection_pipeline"):
                    if cam_args.top_border_lines or cam_args.bottom_border_lines:
                        boundaries = get_pipeline_item(
                            model.post_detection_pipeline, "BoundaryLines"
                        )
                        if boundaries is not None:
                            boundaries.update(
                                {
                                    "upper_border_lines": cam_args.top_border_lines,
                                    "lower_border_lines": cam_args.bottom_border_lines,
                                    "original_clip_box": get_clip_box(
                                        game_id=args.game_id, root_dir=args.root_dir
                                    ),
                                }
                            )

            if args.multi_pose:
                from mmpose.apis import init_pose_model
                from mmpose.datasets import DatasetInfo

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
        else:
            assert False and "No longer supported"

        dataloader = MultiDatasetWrapper()
        # dataloader = None
        postprocessor = None
        if args.input_video:
            input_video_files = args.input_video.split(",")
            if is_stitching(args.input_video):
                project_file_name = "autooptimiser_out.pto"

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
                    dir_name = args.input_video
                    video_left = "left.mp4"
                    video_right = "right.mp4"
                    vl = os.path.join(dir_name, video_left)
                    vr = os.path.join(dir_name, video_right)
                    input_video_files = [vl, vr]

                left_vid = BasicVideoInfo(vl)
                right_vid = BasicVideoInfo(vr)
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
                    dir_name,
                    video_left,
                    video_right,
                    project_file_name,
                    left_frame_offset=args.lfo,
                    right_frame_offset=args.rfo,
                )
                # Create the stitcher data loader
                output_stitched_video_file = (
                    os.path.join(".", f"stitched_output-{args.game_id}.mkv")
                    if args.save_stitched
                    else None
                )
                stitched_dataset = StitchDataset(
                    video_file_1=os.path.join(dir_name, video_left),
                    video_file_2=os.path.join(dir_name, video_right),
                    pto_project_file=pto_project_file,
                    video_1_offset_frame=lfo,
                    video_2_offset_frame=rfo,
                    start_frame_number=args.start_frame,
                    output_stitched_video_file=(
                        output_stitched_video_file if args.save_stitched else None
                    ),
                    max_frames=args.max_frames,
                    max_input_queue_size=args.cache_size,
                    num_workers=1,
                    blend_thread_count=1,
                    remap_thread_count=1,
                    fork_workers=False,
                    image_roi=None,
                    batch_size=args.batch_size,
                    remapping_device=gpus["stitching"],
                    decoder_device=(
                        torch.device(args.decoder_device) if args.decoder_device else None
                    ),
                    blend_mode=opts.blend_mode,
                    dtype=torch.float if not args.fp16_stitch else torch.half,
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
                )
                dataloader.append_dataset("pano", mot_dataloader)

                if args.end_zones:
                    # Try far_left and far_right videos if they exist
                    other_videos: List[Tuple[str, str]] = [
                        ("far_left", os.path.join(dir_name, "far_left.mp4")),
                        ("far_right", os.path.join(dir_name, "far_right.mp4")),
                    ]
                    for vid_name, vid_path in other_videos:
                        if os.path.exists(vid_path):
                            extra_dataloader = MOTLoadVideoWithOrig(
                                path=vid_path,
                                # game_id=dir_name,
                                img_size=None,
                                start_frame_number=args.start_frame,
                                batch_size=1,
                                stream_tensors=tracker == "mmtrack",
                                dtype=torch.float if not args.fp16 else torch.half,
                                device=gpus["encoder"],
                                original_image_only=True,
                            )
                            dataloader.append_dataset(vid_name, extra_dataloader)
            else:
                assert len(input_video_files) == 1
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
                )
                dataloader.append_dataset("pano", pano_dataloader)

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

        # TODO: can this be part of the openmm pipeline?
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
            data_type="mot",
            use_fork=False,
            camera_name=get_nested_value(game_config, "camera.name"),
            async_post_processing=True,
        )
        postprocessor._args.skip_final_video_save = args.skip_final_video_save

        if not isinstance(exp, FakeExp):
            trt_file = None
            decoder = None
            if model is not None:
                model = model.to(main_device)
                model.eval()

                if not args.speed and not args.trt:
                    if args.load_model is None:
                        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
                    else:
                        ckpt_file = args.load_model
                    logger.info("loading checkpoint")
                    loc = "cuda:{}".format(rank)
                    ckpt = torch.load(ckpt_file, map_location=loc)
                    # load the model state dict
                    if "model" in ckpt:
                        model.load_state_dict(ckpt["model"])
                    # torch.jit.load()
                    logger.info("loaded checkpoint done.")

                if is_distributed:
                    model = DDP(model, device_ids=[rank])

                if args.trt:
                    assert (
                        not args.fuse and not is_distributed and args.batch_size == 1
                    ), "TensorRT model is not support model fusing and distributed inferencing!"
                    trt_file = os.path.join(file_name, "model_trt.pth")
                    assert os.path.exists(
                        trt_file
                    ), "TensorRT model is not found!\n Run tools/trt.py first!"
                    model.head.decode_in_inference = False
                    decoder = model.head.decode_outputs
            else:
                args.device = f"cuda:{rank}"

            eval_functions = {
                "mmtrack": {"function": run_mmtrack},
            }

            # start evaluate
            args.device = main_device

            *_, summary = eval_functions[tracker]["function"](
                model=model,
                config=args.game_config,
                fp16=args.fp16,
                distributed=is_distributed,
                half=args.fp16,
                trt_file=trt_file,
                decoder=decoder,
                test_size=exp.test_size,
                result_folder=results_folder,
                device=main_device,
            )
        else:
            other_kwargs = {
                "dataloader": dataloader,
                "postprocessor": postprocessor,
            }

            run_mmtrack(
                model=model,
                pose_model=pose_model,
                pose_dataset_type=pose_dataset,
                pose_dataset_info=pose_dataset_info,
                config=args.game_config,
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
        from hmlib.audio import copy_audio

        if output_video_path and os.path.exists(output_video_path):
            video_with_audio = args.output_video
            if not video_with_audio:
                game_video_dir = get_game_dir(args.game_id)
                if not game_video_dir:
                    # Going into results dir
                    dir_tokens = output_video_path.split("/")
                    file_name = dir_tokens[-1]
                    fn_tokens = file_name.split(".")
                    if len(fn_tokens) > 1:
                        if fn_tokens[-1] == "mkv":
                            # There will be audio drift when adding audio
                            # from mkv to mkv due to strange frame rate items
                            # in the mkv that differ from the original
                            fn_tokens[-1] = "mp4"
                        fn_tokens[-2] += "-with-audio"
                    else:
                        fn_tokens[0] += "-with-audio"
                    dir_tokens[-1] = ".".join(fn_tokens)
                    video_with_audio = os.path.join(*dir_tokens)
                else:
                    # Going into game-dir (numbered if pre-existing)
                    file_name = output_video_path.split("/")[-1]
                    base_name, extension = os.path.splitext(file_name)
                    if extension == ".mkv":
                        # There will be audio drift when adding audio
                        # from mkv to mkv due to strange frame rate items
                        # in the mkv that differ from the original
                        extension = ".mp4"
                    video_with_audio = None
                    for i in range(1000):
                        if i:
                            fname = base_name + "-" + str(i) + extension
                        else:
                            fname = base_name + extension
                        fname = os.path.join(game_video_dir, fname)
                        if not os.path.exists(fname):
                            video_with_audio = fname
                            break
            print(f"Saving video with audio to file: {video_with_audio}")
            copy_audio(
                input_audio=input_video_files,
                input_video=output_video_path,
                output_video=video_with_audio,
            )
        logger.info("Completed")
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise
    finally:
        try:
            postprocessor.stop()
            if dataloader is not None and hasattr(dataloader, "close"):
                dataloader.close()
            if tracking_data is not None:
                tracking_data.flush()
        except Exception as ex:
            print(f"Exception while shutting down: {ex}")


def tensor_to_image(tensor: torch.Tensor):
    if torch.is_floating_point(tensor):
        tensor = torch.clamp(tensor * 255, min=0, max=255).to(torch.uint8, non_blocking=True)
    return tensor


def run_mmtrack(
    model,
    pose_model,
    pose_dataset_type,
    pose_dataset_info,
    config,
    dataloader,
    postprocessor,
    progress_bar: Optional[ProgressBar] = None,
    tracking_data: MOTTrackingData = None,
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
):
    try:
        cuda_stream = torch.cuda.Stream(device)
        with torch.cuda.stream(cuda_stream):
            dataloader_iterator = CachedIterator(
                iterator=iter(dataloader), cache_size=input_cache_size
            )
            # print("WARNING: Not cacheing data loader")

            #
            # Calculate some dataset stats for our progress display
            #
            batch_size = dataloader.batch_size
            total_batches_in_video = len(dataloader)
            total_frames_in_video = batch_size * total_batches_in_video
            number_of_batches_processed = 0
            total_duration_str = convert_seconds_to_hms(len(dataloader) / dataloader.fps)

            if model is not None:
                model.eval()

            wraparound_timer = None
            get_timer = Timer()
            detect_timer = None
            last_frame_id = None

            if progress_bar is not None:
                dataloader_iterator = progress_bar.set_iterator(dataloader_iterator)

                #
                # The progress table
                #
                def _table_callback(table_map: OrderedDict[Any, Any]):
                    duration_processed_in_seconds = (
                        number_of_batches_processed * batch_size / dataloader.fps
                    )
                    remaining_frames_to_process = total_frames_in_video - (
                        number_of_batches_processed * batch_size
                    )
                    remaining_seconds_to_process = remaining_frames_to_process / dataloader.fps

                    if wraparound_timer is not None:
                        processing_fps = batch_size / max(1e-5, wraparound_timer.average_time)

                        table_map["HMTrack FPS"] = "{:.2f}".format(processing_fps)
                        table_map["Dataset length"] = total_duration_str
                        table_map["Processed"] = convert_seconds_to_hms(
                            duration_processed_in_seconds
                        )
                        table_map["Remaining"] = convert_seconds_to_hms(
                            remaining_seconds_to_process
                        )
                        table_map["ETA"] = convert_seconds_to_hms(
                            remaining_frames_to_process / processing_fps
                        )

                # Add that table-maker to the progress bar
                progress_bar.add_table_callback(_table_callback)

            using_precalculated_tracking = (
                tracking_data is not None and tracking_data.has_input_data()
            )
            for cur_iter, dataset_results in enumerate(dataloader_iterator):
                origin_imgs, data, _, info_imgs, ids = dataset_results.pop("pano")
                with torch.no_grad():
                    # init tracker
                    frame_id = info_imgs[2][0]

                    # if isinstance(origin_imgs, StreamTensor):
                    #     origin_imgs = origin_imgs.get()

                    batch_size = origin_imgs.shape[0]

                    if last_frame_id is None:
                        last_frame_id = int(frame_id)
                    else:
                        assert int(frame_id) == last_frame_id + batch_size
                        last_frame_id = int(frame_id)

                    batch_size = origin_imgs.shape[0]

                    if not using_precalculated_tracking:
                        if detect_timer is None:
                            detect_timer = Timer()

                        if isinstance(data["img"][0], StreamTensor):
                            get_timer.tic()
                            for i in range(len(data["img"])):
                                data["img"][i] = data["img"][i].get()
                            get_timer.toc()
                        else:
                            get_timer = None

                        data = collate([data], samples_per_gpu=batch_size)
                        if next(model.parameters()).is_cuda:
                            # scatter to specified GPU
                            data = scatter(data, [device])[0]
                        else:
                            for m in model.modules():
                                assert not isinstance(
                                    m, RoIPool
                                ), "CPU inference with RoIPool is not supported currently."
                            # just get the actual data from DataContainer
                            data["img_metas"] = data["img_metas"][0].data

                        for i, img in enumerate(data["img"]):
                            data["img"][i] = make_channels_first(data["img"][i].squeeze(0)).to(
                                torch.float16 if fp16 else torch.float,
                                non_blocking=True,
                            )

                        # forward the model
                        detect_timer.tic()
                        for i in range(1):
                            with torch.no_grad():
                                with autocast() if fp16 else nullcontext():
                                    tracking_results = model(
                                        return_loss=False, rescale=True, **data
                                    )
                        detect_timer.toc()

                        det_bboxes = tracking_results["det_bboxes"]
                        track_bboxes = tracking_results["track_bboxes"]

                    for frame_index in range(len(origin_imgs)):
                        frame_id = info_imgs[2][frame_index]

                        if not using_precalculated_tracking:
                            if pose_model is not None:
                                (
                                    tracking_results,
                                    pose_results,
                                    returned_outputs,
                                    vis_frame,
                                ) = multi_pose_task(
                                    pose_model=pose_model,
                                    cur_frame=make_channels_last(origin_imgs).wait().squeeze(0),
                                    dataset=pose_dataset_type,
                                    dataset_info=pose_dataset_info,
                                    tracking_results=tracking_results,
                                    smooth=args.smooth,
                                    show=args.plot_pose,
                                )
                            else:
                                vis_frame = None

                            if vis_frame is not None:
                                if isinstance(vis_frame, np.ndarray):
                                    vis_frame = torch.from_numpy(vis_frame)
                                if isinstance(origin_imgs, StreamTensor):
                                    origin_imgs = origin_imgs.wait()
                                origin_imgs[frame_index] = vis_frame.to(
                                    device=origin_imgs.device, non_blocking=True
                                )

                            detections = det_bboxes[frame_index]
                            tracking_items = track_bboxes[frame_index]

                            track_ids = tracking_items[:, 0].astype(np.int64)
                            bboxes = tracking_items[:, 1:5]
                            scores = tracking_items[:, -1]

                            if tracking_data is not None:
                                tracking_data.add_frame_records(
                                    frame_id=frame_id,
                                    tracking_ids=track_ids,
                                    tlbr=bboxes,
                                    scores=scores,
                                )

                            online_tlwhs = torch.from_numpy(bboxes)
                            # make boxes tlwh
                            online_tlwhs[:, 2] = (
                                online_tlwhs[:, 2] - online_tlwhs[:, 0]
                            )  # width = x2 - x1
                            online_tlwhs[:, 3] = (
                                online_tlwhs[:, 3] - online_tlwhs[:, 1]
                            )  # height = y2 - y1
                        else:
                            track_ids, scores, bboxes = tracking_data.get_tracking_info_by_frame(
                                frame_id
                            )
                            detections = bboxes
                            online_tlwhs = torch.from_numpy(bboxes)
                            tracking_results = None

                        online_ids = torch.from_numpy(track_ids)
                        online_scores = torch.from_numpy(scores)

                        if postprocessor is not None:
                            if isinstance(origin_imgs, StreamTensor):
                                origin_imgs = origin_imgs.get()
                            tracking_results, detections, online_tlwhs = (
                                postprocessor.process_tracking(
                                    tracking_results=tracking_results,
                                    frame_id=frame_id,
                                    online_tlwhs=online_tlwhs,
                                    online_ids=online_ids,
                                    online_scores=online_scores,
                                    detections=detections,
                                    info_imgs=info_imgs,
                                    letterbox_img=None,
                                    original_img=origin_imgs[frame_index].unsqueeze(0),
                                    data=dataset_results,
                                )
                            )

                        if pose_model is not None and using_precalculated_tracking:
                            if tracking_results is None:
                                # Reconstruct tracking_results
                                tracking_items = torch.from_numpy(
                                    track_ids.astype(bboxes.dtype)
                                ).reshape(track_ids.shape[0], 1)
                                tracking_items = torch.cat(
                                    [
                                        tracking_items,
                                        torch.from_numpy(bboxes),
                                        torch.from_numpy(scores).reshape(scores.shape[0], 1),
                                    ],
                                    dim=1,
                                )
                                tracking_results = {
                                    "track_bboxes": [tracking_items.numpy()],
                                }

                            multi_pose_task(
                                pose_model=pose_model,
                                cur_frame=make_channels_last(origin_imgs[frame_index]),
                                dataset=pose_dataset_type,
                                dataset_info=pose_dataset_info,
                                tracking_results=tracking_results,
                                smooth=args.smooth,
                                show=args.show_image,
                                # show=True,
                            )
                    # end frame loop

                    if detect_timer is not None and cur_iter % 50 == 0:
                        # print(
                        logger.info(
                            "mmtrack tracking, frame {} ({:.2f} fps)".format(
                                frame_id,
                                batch_size * 1.0 / max(1e-5, detect_timer.average_time),
                            )
                        )
                        detect_timer = Timer()

                    if cur_iter % 200 == 0:
                        wraparound_timer = Timer()
                    elif wraparound_timer is None:
                        wraparound_timer = Timer()
                    else:
                        wraparound_timer.toc()

                    number_of_batches_processed += 1
                    wraparound_timer.tic()
    except StopIteration:
        print("run_mmtrack reached end of dataset")


def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    :param mmtracking_results:
    :return: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if "track_bboxes" in mmtracking_results:
        tracking_results = mmtracking_results["track_bboxes"][0]
    elif "track_results" in mmtracking_results:
        tracking_results = mmtracking_results["track_results"][0]

    for track in tracking_results:
        person = {}
        person["track_id"] = int(track[0])
        person["bbox"] = track[1:]  # will also have the score
        person_results.append(person)
    return person_results


@typechecked
def multi_pose_task(
    pose_model,
    cur_frame,
    dataset,
    dataset_info,
    tracking_results: dict,
    smooth: bool = False,
    show: bool = False,
):
    start = time.time()
    # build pose smoother for temporal refinement
    if smooth:
        smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=2)
    else:
        smoother = None

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # keep the person class bounding boxes.
    person_results = process_mmtracking_results(tracking_results)

    from mmpose.apis import inference_top_down_pose_model, vis_pose_tracking_result
    from mmpose.core import Smoother

    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        cur_frame,
        person_results,
        bbox_thr=args.bbox_thr,
        format="xyxy",
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names,
    )
    duration = time.time() - start

    if smoother:
        pose_results = smoother.smooth(pose_results)

    vis_frame = None
    # show the results
    if show:
        # assert cur_frame.size(0) == 1
        vis_frame = vis_pose_tracking_result(
            pose_model,
            cur_frame.squeeze(0).to("cpu").numpy(),
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False,
            # show=True,
        )
        # vis_frame = np.expand_dims(vis_frame, axis=0)
    # duration = time.time() - start
    # print(f"pose took {duration} seconds")
    return tracking_results, pose_results, returned_outputs, vis_frame


def setup_logging():
    mmcv_logger = mmcv_get_logger("mmcv")
    mmcv_logger.setLevel(logging.WARN)
    logger.info("Logger initialized")


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

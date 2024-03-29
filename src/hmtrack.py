import argparse
import copy
import logging
import os
import random
import socket
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmpose.apis import (
    collect_multi_frames,
    inference_top_down_pose_model,
    init_pose_model,
    vis_pose_tracking_result,
)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmtrack.apis import inference_mot, inference_vid, init_model
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from hmlib.datasets import get_yolox_datadir
from hmlib.stitching.synchronize import configure_video_stitching
from hmlib.tracking_utils.boundaries import BoundaryLines
from hmlib.utils.pipeline import replace_pipeline_class
from hmlib.utils.py_utils import find_item_in_module

if False:
    from hmlib.datasets.dataset.stitching_dataloader1 import StitchDataset
else:
    from hmlib.datasets.dataset.stitching_dataloader2 import (
        StitchDataset,
    )

import hmlib.datasets as datasets
from hmlib.camera.cam_post_process import DefaultArguments
from hmlib.camera.camera_head import CamTrackHead
from hmlib.config import (
    get_clip_box,
    get_config,
    get_nested_value,
    set_nested_value,
    update_config,
)
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.ffmpeg import BasicVideoInfo
from hmlib.hm_opts import copy_opts, hm_opts
from hmlib.stitching.laplacian_blend import show_image
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.gpu import (
    CachedIterator,
    StreamTensor,
    select_gpus,
    StreamTensorToDevice,
)
from hmlib.utils.image import make_channels_first, make_channels_last
from hmlib.video_stream import time_to_frame

ROOT_DIR = os.getcwd()


def make_parser(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser("HockeyMOM Tracking")
    parser = hm_opts.parser(parser)
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
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
        "--cache-size", type=int, default=2, help="cache size for GPU stream async operations"
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
    parser.add_argument(
        "--no-crop", action="store_true", help="Don't crop output image"
    )
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
    parser.add_argument(
        "--min-box-area", type=float, default=100, help="filter out tiny boxes"
    )
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    parser.add_argument(
        "--input_video",
        "--input-video",
        type=str,
        default=None,
        help="Input video file(s)",
    )

    # Pose args
    parser.add_argument(
        "--pose-config", type=str, default=None, help="Pose config file"
    )
    parser.add_argument(
        "--pose-checkpoint", type=str, default=None, help="Pose checkpoint file"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Keypoint score threshold"
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--radius", type=int, default=4, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
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


def main(args, num_gpu):
    dataloader = None

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

        if args.lfo is None and args.rfo is None:
            if (
                "stitching" in game_config["game"]
                and "offsets" in game_config["game"]["stitching"]
            ):
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
            game_video_dir = os.path.join(os.environ["HOME"], "Videos", args.game_id)
            if os.path.isdir(game_video_dir):
                # TODO: also look for avi and mp4 files
                if args.force_stitching:
                    args.input_video = game_video_dir
                else:
                    pre_stitched_file_name = find_stitched_file(
                        dir_name=game_video_dir, game_id=args.game_id
                    )
                    if pre_stitched_file_name and os.path.exists(
                        pre_stitched_file_name
                    ):
                        args.input_video = pre_stitched_file_name
                    else:
                        args.input_video = game_video_dir

        actual_device_count = torch.cuda.device_count()
        if not actual_device_count:
            raise Exception("At leats one GPU is required for this application")
        while len(args.gpus) > actual_device_count:
            del args.gpus[-1]

        gpus = select_gpus(allowed_gpus=args.gpus, is_multipose=args.multi_pose)

        if socket.gethostname().startswith("chriso-monster"):
            gpus["stitching"] = torch.device("cuda", 0)
            gpus["detection"] = torch.device("cuda", 0)
            gpus["pose"] = torch.device("cuda", 0)
            gpus["encoder"] = torch.device("cuda", 1)

        torch.cuda.set_device(gpus["detection"].index)

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

        # Set up for pose
        pose_model = None
        pose_dataset = None
        pose_dataset_info = None

        data_pipeline = None
        if tracker == "mmtrack":
            args.config = args.exp_file
            args.checkpoint = None
            if args.tracking or args.multi_pose:
                model = init_model(
                    args.config, args.checkpoint, device=gpus["detection"]
                )
                cfg = model.cfg.copy()
                pipeline = cfg.data.inference.pipeline
                pipeline[0].type = "LoadImageFromWebcam"
                data_pipeline = Compose(pipeline)

            # post-detection pipeline
            if hasattr(model, "post_detection_pipeline"):
                if cam_args.top_border_lines or cam_args.bottom_border_lines:
                    model.post_detection_pipeline.append(
                        BoundaryLines(
                            upper_border_lines=cam_args.top_border_lines,
                            lower_border_lines=cam_args.bottom_border_lines,
                            original_clip_box=get_clip_box(
                                game_id=args.game_id, root_dir=ROOT_DIR
                            ),
                        )
                    )
            #
            # BEGIN Set up multi-pose model, if enabled
            #
            if args.multi_pose:
                pose_config = mmcv.Config.fromfile(args.pose_config)
                test_pipeline = pose_config["test_pipeline"]
                replace_pipeline_class(
                    test_pipeline, "TopDownAffine", "HmTopDownAffine"
                )
                replace_pipeline_class(test_pipeline, "ToTensor", "HmToTensor")
                pose_model = init_pose_model(
                    pose_config, args.pose_checkpoint, device=gpus["multipose"]
                )
                pose_dataset = pose_model.cfg.data["test"]["type"]
                pose_dataset_info = pose_model.cfg.data["test"].get(
                    "dataset_info", None
                )
                if pose_dataset_info is None:
                    warnings.warn(
                        "Please set `dataset_info` in the config."
                        "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
                        DeprecationWarning,
                    )
                else:
                    pose_dataset_info = DatasetInfo(pose_dataset_info)
            #
            # END Set up multi-pose model, if enabled
            #
        else:
            assert False and "No longer supported"

        dataloader = None
        postprocessor = None
        if args.input_video:
            input_video_files = args.input_video.split(",")
            if len(input_video_files) == 2 or os.path.isdir(args.input_video):
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
                    args.max_frames = time_to_frame(
                        time_str=args.max_time, fps=left_vid.fps
                    )

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
                    # batch_size=1,
                    batch_size=args.batch_size,
                    remapping_device=gpus["stitching"],
                    # batch_size=args.batch_size,
                    blend_mode=opts.blend_mode,
                    # dtype=torch.float if not args.fp16 else torch.half,
                )
                # Create the MOT video data loader, passing it the
                # stitching data loader as its image source
                dataloader = MOTLoadVideoWithOrig(
                    path=None,
                    game_id=dir_name,
                    img_size=exp.test_size,
                    return_origin_img=True,
                    start_frame_number=args.start_frame,
                    data_dir=os.path.join(get_yolox_datadir(), "hockeyTraining"),
                    json_file="test.json",
                    # batch_size=args.batch_size,
                    batch_size=1,
                    # clip_original=get_clip_box(game_id=args.game_id, root_dir=ROOT_DIR),
                    name="val",
                    # device=gpus["detection"] if tracker == "mmtrack" else torch.device("cpu"),
                    # device=torch.device("cpu"),
                    # preproc=ValTransform(
                    #     rgb_means=(0.485, 0.456, 0.406),
                    #     std=(0.229, 0.224, 0.225),
                    # ),
                    embedded_data_loader=stitched_dataset,
                    #embedded_data_loader_cache_size=6,
                    embedded_data_loader_cache_size=args.cache_size,
                    # stream_tensors=True,
                    # image_channel_adjustment=game_config["rink"]["camera"][
                    #     "image_channel_adjustment"
                    # ],
                    # device_for_original_image=torch.device("cpu"),
                    data_pipeline=data_pipeline,
                    stream_tensors=tracker == "mmtrack",
                    # dtype=torch.float if not args.fp16 else torch.half,
                )
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
                    args.max_frames = time_to_frame(
                        time_str=args.max_time, fps=vid_info.fps
                    )

                dataloader = MOTLoadVideoWithOrig(
                    path=input_video_files[0],
                    img_size=exp.test_size,
                    return_origin_img=True,
                    start_frame_number=args.start_frame,
                    data_dir=os.path.join(get_yolox_datadir(), "hockeyTraining"),
                    json_file="test.json",
                    batch_size=args.batch_size,
                    # clip_original=get_clip_box(game_id=args.game_id, root_dir=ROOT_DIR),
                    max_frames=args.max_frames,
                    name="val",
                    device=gpus["detection"],
                    original_image_only=tracker == "centertrack",
                    data_pipeline=data_pipeline,
                    dtype=torch.float if not args.fp16 else torch.half,
                )

        if dataloader is None:
            dataloader = exp.get_eval_loader(
                args.batch_size, is_distributed, args.test, return_origin_img=True
            )

        # TODO: can this be part of the openmm pipeline?
        postprocessor = CamTrackHead(
            opt=args,
            args=cam_args,
            fps=dataloader.fps,
            save_dir=results_folder if not args.no_save_video else None,
            save_frame_dir=args.save_frame_dir,
            original_clip_box=get_clip_box(game_id=args.game_id, root_dir=ROOT_DIR),
            device=gpus["camera"],
            video_out_device=gpus["encoder"],
            data_type="mot",
            use_fork=False,
            async_post_processing=True,
        )
        postprocessor._args.skip_final_video_save = args.skip_final_video_save

        if not isinstance(exp, FakeExp):
            trt_file = None
            decoder = None
            if model is not None:
                model = model.to(gpus["detection"])
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
            args.device = gpus["detection"]

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
                device=gpus["detection"],
            )
        else:
            other_kwargs = {
                "dataloader": dataloader,
                "postprocessor": postprocessor,
            }

            *_, summary = run_mmtrack(
                model=model,
                pose_model=pose_model,
                pose_dataset_type=pose_dataset,
                pose_dataset_info=pose_dataset_info,
                config=args.game_config,
                device=gpus["detection"],
                input_cache_size=args.cache_size,
                **other_kwargs,
            )
        if not args.infer:
            logger.info("\n" + str(summary))
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
        except Exception as ex:
            print(f"Exception while shutting down: {ex}")


def tensor_to_image(tensor: torch.Tensor):
    if torch.is_floating_point(tensor):
        tensor = torch.clamp(tensor * 255, min=0, max=255).to(
            torch.uint8, non_blocking=True
        )
    return tensor


def run_mmtrack(
    model,
    pose_model,
    pose_dataset_type,
    pose_dataset_info,
    config,
    dataloader,
    postprocessor,
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
):
    dataloader_iterator = CachedIterator(
        iterator=iter(dataloader), cache_size=input_cache_size
    )
    # print("WARNING: Not cacheing data loader")

    get_timer = Timer()
    detect_timer = None

    last_frame_id = None

    if True:
        # INFO logging
        handler = logging.StreamHandler(sys.stdout)
        logger.setLevel(max(logger.level, logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    for cur_iter, (
        origin_imgs,
        data,
        _,
        info_imgs,
        ids,
    ) in enumerate(dataloader_iterator):
        # info_imgs is 4 scalar tensors: height, width, frame_id, video_id
        with torch.no_grad():
            # init tracker
            frame_id = info_imgs[2][0]

            batch_size = origin_imgs.shape[0]

            if last_frame_id is None:
                last_frame_id = int(frame_id)
            else:
                assert int(frame_id) == last_frame_id + batch_size
                last_frame_id = int(frame_id)

            batch_size = origin_imgs.shape[0]

            if pose_model is not None:
                cpu_original_img = StreamTensorToDevice(
                    tensor=origin_imgs, device=torch.device("cpu"), contiguous=True
                )

            if detect_timer is None:
                detect_timer = Timer()

            # show_image("image", data["img"][0] / 255, wait=False)

            # detect_timer.tic()

            if False:
                img = tensor_to_image(origin_imgs)
                results = my_inference_mot(  # DOTO: return proper data and img_meta items (one for each frame)
                    model,
                    make_channels_last(img.squeeze(0)).cpu().numpy(),
                    frame_id=frame_id,
                )
            else:
                if isinstance(data["img"][0], StreamTensor):
                    get_timer.tic()
                    for i in range(len(data["img"])):
                        data["img"][i] = data["img"][i].get()
                    get_timer.toc()
                else:
                    get_timer = None

                # for i, img in enumerate(data["img"]):
                #     # collate adds a stupid batch size of 1
                #     data["img"][i] = make_channels_first(data["img"][i]).to(
                #         torch.float, non_blocking=True
                #     )
                #     # data["img"][i] = make_channels_first(data["img"][i].squeeze(0)).to(
                #     #     torch.float, non_blocking=True
                #     # )

                # Expand batch tensors into list of tensors
                # if len(data["img"]) == 1:
                #     all_images = []
                #     all_metas = []
                #     for outer_index, img in enumerate(data["img"]):
                #         if img.ndim == 4 and img.shape[0] != 1:
                #             for i in range(img.shape[0]):
                #                 all_images.append(img[i].unsqueeze(0))
                #                 all_metas.append(
                #                     copy.deepcopy(data["img_metas"][outer_index])
                #                 )
                #                 if i:
                #                     all_metas[-1].data["frame_id"] = (
                #                         all_metas[0].data["frame_id"] + i
                #                     )
                #     if all_metas:
                #         data['img'] = all_images
                #         data["img_metas"] = all_metas
                # assert len(data["img"]) == 1
                # data["img"] = data["img"][0]

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
                    # collate adds a stupid batch size of 1
                    # data["img"][i] = make_channels_first(data["img"][i]).to(
                    #     torch.float, non_blocking=True
                    # )
                    data["img"][i] = make_channels_first(data["img"][i].squeeze(0)).to(
                        torch.float16 if fp16 else torch.float, non_blocking=True
                    )

                # forward the model
                detect_timer.tic()
                if fp16:
                    with autocast():
                        tracking_results = model(
                            return_loss=False, rescale=True, **data
                        )
                else:
                    tracking_results = model(return_loss=False, rescale=True, **data)
                detect_timer.toc()

            # detect_timer.toc()

            # del data

            det_bboxes = tracking_results["det_bboxes"]
            track_bboxes = tracking_results["track_bboxes"]


            if pose_model is not None:
                cpu_original_img = cpu_original_img.get()

            for frame_index in range(len(origin_imgs)):
                if pose_model is not None:
                    tracking_results, pose_results, returned_outputs, vis_frame = (
                        multi_pose_task(
                            pose_model=pose_model,
                            cur_frame=make_channels_last(cpu_original_img[frame_index]),
                            # cur_frame=make_channels_last(data["img"][frame_index]),
                            dataset=pose_dataset_type,
                            dataset_info=pose_dataset_info,
                            tracking_results=tracking_results,
                            smooth=args.smooth,
                            show=True,
                        )
                    )
                else:
                    vis_frame = None

                if vis_frame is not None:
                    if isinstance(vis_frame, np.ndarray):
                        vis_frame = torch.from_numpy(vis_frame)
                    origin_imgs[frame_index] = make_channels_first(vis_frame).to(
                        device=origin_imgs.device, non_blocking=True
                    )

                frame_id = info_imgs[2][frame_index]
                detections = det_bboxes[frame_index]
                tracking_items = track_bboxes[frame_index]

                track_ids = tracking_items[:, 0]
                bboxes = tracking_items[:, 1:5]
                scores = tracking_items[:, -1]

                online_tlwhs = torch.from_numpy(bboxes)
                online_ids = torch.from_numpy(track_ids).to(torch.int64)
                online_scores = torch.from_numpy(scores)

                # make boxes tlwh
                online_tlwhs[:, 2] = (
                    online_tlwhs[:, 2] - online_tlwhs[:, 0]
                )  # width = x2 - x1
                online_tlwhs[:, 3] = (
                    online_tlwhs[:, 3] - online_tlwhs[:, 1]
                )  # height = y2 - y1

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
                        )
                    )

                # if pose_model is not None:
                #     multi_pose_task(
                #         pose_model=pose_model,
                #         cur_frame=make_channels_last(origin_imgs[frame_index]),
                #         dataset=pose_dataset_type,
                #         dataset_info=pose_dataset_info,
                #         tracking_results=tracking_results,
                #         smooth=args.smooth,
                #         show=args.show_image,
                #     )
            # end frame loop

            if cur_iter % 50 == 0:
                # print(
                logger.info(
                    "mmtrack tracking, frame {} ({:.2f} fps)".format(
                        frame_id,
                        batch_size * 1.0 / max(1e-5, detect_timer.average_time),
                    )
                )
                detect_timer = Timer()


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
        person["bbox"] = track[1:]
        person_results.append(person)
    return person_results


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

    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        cur_frame.to("cpu").numpy(),
        #cur_frame,
        person_results,
        bbox_thr=args.bbox_thr,
        format="xyxy",
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names,
    )

    if smoother:
        pose_results = smoother.smooth(pose_results)

    vis_frame = None
    # show the results
    if show:
        vis_frame = vis_pose_tracking_result(
            pose_model,
            cur_frame.to("cpu").numpy(),
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False,
        )
    duration = time.time() - start
    print(f"pose took {duration} seconds")
    return tracking_results, pose_results, returned_outputs, vis_frame


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    game_config = get_config(
        game_id=args.game_id, rink=args.rink, camera=args.camera, root_dir=ROOT_DIR
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

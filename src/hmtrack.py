from loguru import logger

import argparse
import random
import warnings

import traceback
import sys, os

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.join(os.path.dirname(__file__), "../MixViT"))
from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger,
)
from yolox.evaluators import MOTEvaluator
from yolox.data import get_yolox_datadir

from hmlib.stitch_synchronize import (
    configure_video_stitching,
)

from hmlib.datasets.dataset.stitching_dataloader import (
    StitchDataset,
)

from hmlib.ffmpeg import BasicVideoInfo
from hmlib.tracking_utils.log import logger
from hmlib.config import get_clip_box, get_config

from hmlib.camera.camera_head import CamTrackHead
from hmlib.camera.cam_post_process import DefaultArguments
import hmlib.datasets as datasets

ROOT_DIR = os.getcwd()


def make_parser(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser("YOLOX Eval")
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
        default="exps/example/mot/yolox_x_sports_mix.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
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
        default=None,
        type=str,
        help="Use tracker type [hm|mixsort|micsort_oc|sort|ocsort|byte|deepsort|motdt]",
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
    # cam args
    parser.add_argument(
        "-s",
        "--show-image",
        "--show",
        dest="show_image",
        default=False,
        action="store_true",
        help="show as processing",
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
    parser.add_argument(
        "--left-file-offset",
        "--lfo",
        dest="lfo",
        default=None,
        type=float,
        help="Left video file offset",
    )
    parser.add_argument(
        "--right-file-offset",
        "--rfo",
        dest="rfo",
        default=None,
        type=float,
        help="Left video file offset",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        type=str,
        help="Game ID",
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
        "--start-frame", type=int, default=0, help="first frame number to process"
    )
    parser.add_argument(
        "--cvat-output",
        action="store_true",
        help="generate dataset data importable by cvat",
    )
    parser.add_argument(
        "--plot-tracking", action="store_true", help="plot individual tracking boxes"
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
        "--max-frames",
        type=int,
        default=None,
        help="maximum number of frames to process",
    )
    parser.add_argument(
        "--save-frame-dir",
        type=str,
        default=None,
        help="directory to save the output video frases as png files",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument(
        "--min-box-area", type=float, default=100, help="filter out tiny boxes"
    )
    parser.add_argument(
        "--mot20", dest="mot20", default=False, action="store_true", help="test mot20."
    )
    # mixformer args
    parser.add_argument("--script", type=str, default="mixformer_deit")
    parser.add_argument("--config", type=str, default="track")
    parser.add_argument("--alpha", type=float, default=0.6, help="fuse parameter")
    parser.add_argument(
        "--radius", type=int, default=0, help="radius for computing similarity"
    )
    parser.add_argument(
        "--input_video", type=str, default=None, help="Input video file(s)"
    )
    parser.add_argument(
        "--iou_only",
        dest="iou_only",
        default=False,
        action="store_true",
        help="only use iou for similarity",
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


def main(exp, args, num_gpu):
    dataloader = None
    try:
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn(
                "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
            )

        #set_deterministic()

        if not args.experiment_name:
            args.experiment_name = args.game_id
        elif args.game_id and args.game_id not in args.experiment_name:
            args.experiment_name = args.experiment_name + "-" + args.game_id

        is_distributed = num_gpu > 1

        if args.gpus and isinstance(args.gpus, str):
            args.gpus = [int(i) for i in args.gpus.split(",")]

        # set environment variables for distributed training
        cudnn.benchmark = True

        rank = args.local_rank
        # rank = get_local_rank()

        file_name = os.path.join(exp.output_dir, args.experiment_name)

        if rank == 0:
            os.makedirs(file_name, exist_ok=True)

        results_folder = os.path.join(file_name, "track_results")
        os.makedirs(results_folder, exist_ok=True)

        setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
        logger.info("Args: {}".format(args))

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms

        if args.test_size:
            assert args.tsize is None
            tokens = args.test_size.split("x")
            if len(tokens) == 1:
                tokens = args.test_size.split("X")
            assert len(tokens) == 2
            exp.test_size = (to_32bit_mul(int(tokens[0])), to_32bit_mul(int(tokens[1])))
        elif args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = None
        if args.tracker not in ["fair", "centertrack"]:
            model = exp.get_model()
            logger.info(
                "Model Summary: {}".format(get_model_info(model, exp.test_size))
            )

        game_config = get_config(
            game_id=args.game_id, rink=args.rink, camera=args.camera, root_dir=ROOT_DIR
        )

        game_config["args"] = vars(args)

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

        cam_args = DefaultArguments(
            game_config=game_config,
            basic_debugging=args.debug,
            show_image=args.show_image,
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
                pre_stitched_file_name = "stitched_output-with-audio.avi"
                pre_stitched_file_path = os.path.join(
                    game_video_dir, pre_stitched_file_name
                )
                if os.path.exists(pre_stitched_file_path):
                    args.input_video = pre_stitched_file_path
                else:
                    args.input_video = game_video_dir

        dataloader = None
        postprocessor = None
        if args.input_video:
            from yolox.data import ValTransform

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

                pto_project_file, lfo, rfo = configure_video_stitching(
                    dir_name,
                    video_left,
                    video_right,
                    project_file_name,
                    left_frame_offset=args.lfo,
                    right_frame_offset=args.rfo,
                )
                # Create the stitcher data loader
                output_stitched_video_file = os.path.join(
                    ".", f"stitched_output-{args.experiment_name}.avi"
                )
                stitched_dataset = StitchDataset(
                    video_file_1=os.path.join(dir_name, video_left),
                    video_file_2=os.path.join(dir_name, video_right),
                    pto_project_file=pto_project_file,
                    video_1_offset_frame=lfo,
                    video_2_offset_frame=rfo,
                    start_frame_number=args.start_frame,
                    output_stitched_video_file=output_stitched_video_file,
                    max_frames=args.max_frames,
                    num_workers=1,
                    blend_thread_count=2,
                    remap_thread_count=6,
                    fork_workers=False,
                    image_roi=None,
                )
                # Create the MOT video data loader, passing it the
                # stitching data loader as its image source
                dataloader = datasets.MOTLoadVideoWithOrig(
                    path=None,
                    img_size=exp.test_size,
                    return_origin_img=True,
                    start_frame_number=args.start_frame,
                    data_dir=os.path.join(get_yolox_datadir(), "hockeyTraining"),
                    json_file="test.json",
                    batch_size=args.batch_size,
                    clip_original=get_clip_box(game_id=args.game_id, root_dir=ROOT_DIR),
                    name="val",
                    preproc=ValTransform(
                        rgb_means=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    embedded_data_loader=stitched_dataset,
                    original_image_only=args.tracker == "centertrack",
                )

            else:
                assert len(input_video_files) == 1
                dataloader = datasets.MOTLoadVideoWithOrig(
                    path=input_video_files[0],
                    img_size=exp.test_size,
                    return_origin_img=True,
                    start_frame_number=args.start_frame,
                    data_dir=os.path.join(get_yolox_datadir(), "hockeyTraining"),
                    json_file="test.json",
                    # json_file="val.json",
                    batch_size=args.batch_size,
                    clip_original=get_clip_box(game_id=args.game_id, root_dir=ROOT_DIR),
                    max_frames=args.max_frames,
                    name="val",
                    preproc=ValTransform(
                        rgb_means=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    original_image_only=args.tracker == "centertrack",
                )

        if dataloader is None:
            dataloader = exp.get_eval_loader(
                args.batch_size, is_distributed, args.test, return_origin_img=True
            )

        if args.game_id:
            detection_device = torch.device("cuda", int(args.gpus[0]))
            track_device = "cpu"
            video_out_device = "cpu"
            if len(args.gpus) > 1:
                video_out_device = torch.device("cuda", int(args.gpus[1]))
            # if len(args.gpus) > 2:
            #     track_device = torch.device("cuda", int(args.gpus[2]))
        else:
            detection_device = torch.device("cuda", int(rank))

        postprocessor = CamTrackHead(
            opt=args,
            args=cam_args,
            fps=dataloader.fps,
            save_dir=results_folder if not args.no_save_video else None,
            save_frame_dir=args.save_frame_dir,
            original_clip_box=dataloader.clip_original,
            device=track_device,
            video_out_device=video_out_device,
            data_type="mot",
            use_fork=False,
            async_post_processing=True,
        )

        evaluator = MOTEvaluator(
            args=args,
            dataloader=dataloader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            postprocessor=postprocessor,
        )

        torch.cuda.set_device(rank)
        trt_file = None
        decoder = None
        if model is not None:
            model = model.to(detection_device)
            # if args.game_id:
            #     model.cuda(int(args.gpus[0]))
            # else:
            #     model.cuda(rank)
            model.eval()

            if not args.speed and not args.trt:
                if args.ckpt is None:
                    ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
                else:
                    ckpt_file = args.ckpt
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

            if args.fuse:
                logger.info("\tFusing model...")
                model = fuse_model(model)

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
        # start evaluate

        eval_functions = {
            "hm": {"function": evaluator.evaluate_hockeymom},
            # "mixsort": {"function": evaluator.evaluate_mixsort},
            "fair": {"function": evaluator.evaluate_fair},
            "centertrack": {"function": evaluator.evaluate_centertrack},
            "mixsort_oc": {"function": evaluator.evaluate_mixsort_oc},
            "sort": {"function": evaluator.evaluate_sort},
            "ocsort": {"function": evaluator.evaluate_ocsort},
            "byte": {"function": evaluator.evaluate_byte},
            "deepsort": {"function": evaluator.evaluate_deepsort},
            "motdt": {"function": evaluator.evaluate_motdt},
        }
        *_, summary = eval_functions[args.tracker]["function"](
            model,
            is_distributed,
            args.fp16,
            trt_file,
            decoder,
            exp.test_size,
            results_folder,
            device=detection_device,
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


if __name__ == "__main__":
    import hmlib.opts_fair as opts_fair
    import lib.opts as centertrack_opts

    parser = make_parser()

    opts_2 = opts_fair.opts(parser=parser)
    parser = opts_2.parser
    args = parser.parse_args()
    if args.tracker == "centertrack":
        opts = centertrack_opts.opts()
        opts.parser = make_parser(opts.parser)
        args = opts.parse()
        args = opts.init()
        exp = get_exp(args.exp_file, args.name)
    elif args.tracker == "fair":
        opts_2.parse(opt=args)
        args = opts_2.init(opt=args)
        exp = get_exp(args.exp_file, args.name)
        # exp.merge(args.opts) # seems to do nothing
    else:
        exp = get_exp(args.exp_file, args.name)
        exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.game_id:
        num_gpus = 1
    else:
        num_gpus = len(args.gpus) if args.gpus else 0
        # num_gpus = torch.cuda.device_count() if args.devices is None else args.devices
        assert num_gpus <= torch.cuda.device_count()
    launch(
        main,
        num_gpus,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpus),
    )
    print("Done.")

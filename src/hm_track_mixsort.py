from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import sys, os
import cv2

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

import argparse
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple
import traceback

from hmtrack import HmPostProcessor
from hmlib.camera.cam_post_process import DefaultArguments
import hmlib.datasets as datasets


def make_parser():
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
        "-d", "--devices", default=1, type=int, help="device for training"
    )
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
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--tracker",
        default="hm",
        type=str,
        help="Use tracker type [hm|mixsort|micsort_oc|sort|ocsort|byte|deepsort|motdt]",
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
    # det args
    parser.add_argument(
        "-c",
        "--ckpt",
        default="pretrained/yolox_x_sportsmix_ch.pth.tar",
        type=str,
        help="ckpt for eval",
    )
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument(
        "--track_thresh", type=float, default=0.6, help="tracking confidence threshold"
    )
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


# def compare_dataframes(gts, ts):
#     accs = []
#     names = []
#     for k, tsacc in ts.items():
#         if k in gts:
#             logger.info("Comparing {}...".format(k))
#             accs.append(
#                 mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5)
#             )
#             names.append(k)
#         else:
#             logger.warning("No ground truth for {}, skipping.".format(k))

#     return accs, names


def set_torch_multiprocessing_use_filesystem():
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")


def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# @logger.catch
def main(exp, args, num_gpu):
    try:
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn(
                "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
            )

        set_deterministic()

        is_distributed = num_gpu > 1

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
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        # logger.info("Model Structure:\n{}".format(str(model)))

        postprocessor = HmPostProcessor(
            opt=args,
            args=DefaultArguments(),
            fps=30,
            save_dir=results_folder,
            #device="cuda",
            device="cpu",
            data_type="mot",
            use_fork=False,
        )

        dataloader = None
        if args.input_video:
            input_video_files = args.input_video.split(",")

            if len(input_video_files) == 2:
                video_1_offset_frame = None
                video_2_offset_frame = None

                video_1_offset_frame = 3
                video_2_offset_frame = 0

                dataloader = datasets.dataset.stitching.StitchDataset(
                    video_file_1=f"{dir_name}/{video_left}",
                    video_file_2=f"{dir_name}/{video_right}",
                    pto_project_file=pto_project_file,
                    video_1_offset_frame=lfo,
                    video_2_offset_frame=rfo,
                    output_stitched_video_file=output_stitched_video_file,
                    max_frames=max_frames,
                    num_workers=1,
                )
            else:
                from yolox.data import ValTransform

                assert len(input_video_files) == 1
                dataloader = datasets.MOTLoadVideoWithOrig(
                    path=input_video_files[0],
                    img_size=exp.test_size,
                    return_origin_img=True,
                    start_frame_number=args.start_frame,
                    # data_dir=os.path.join(get_yolox_datadir(), "SportsMOT"),
                    data_dir=os.path.join(get_yolox_datadir(), "hockeyTraining"),
                    # data_dir=os.path.join(get_yolox_datadir(), "crowdhuman"),
                    json_file="test.json",
                    # json_file="val.json",
                    batch_size=args.batch_size,
                    # batch_size=1,
                    name="val",
                    preproc=ValTransform(
                        rgb_means=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                )

            # from yolox.data import MOTDataset, ValTransform

            # valdataset = MOTDataset(
            #     data_dir=os.path.join(get_yolox_datadir(), "SportsMOT"),
            #     json_file=self.val_ann,
            #     img_size=self.test_size,
            #     name='val', # change to train when running on training set
            #     preproc=ValTransform(
            #         rgb_means=(0.485, 0.456, 0.406),
            #         std=(0.229, 0.224, 0.225),
            #     ),
            #     return_origin_img=return_origin_img,
            # )

            # if is_distributed:
            #     #batch_size = batch_size // world_size
            #     # sampler = torch.utils.data.distributed.DistributedSampler(
            #     #     dataloader, shuffle=False
            #     # )
            #     assert False
            # else:
            #     sampler = torch.utils.data.SequentialSampler(dataloader)

            # dataloader_kwargs = {
            #     "num_workers": 1, # self.data_num_workers,
            #     "pin_memory": True,
            #     "sampler": sampler,
            # }
            # dataloader_kwargs["batch_size"] = args.batch_size
            # dataloader = torch.utils.data.DataLoader(dataloader, **dataloader_kwargs)

        if dataloader is None:
            dataloader = exp.get_eval_loader(
                args.batch_size, is_distributed, args.test, return_origin_img=True
            )

        evaluator = MOTEvaluator(
            args=args,
            dataloader=dataloader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            online_callback=postprocessor.online_callback,
        )

        torch.cuda.set_device(rank)
        model.cuda(rank)
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
            trt_file = None
            decoder = None

        # start evaluate

        eval_functions = {
            "hm": {"function": evaluator.evaluate_hockeymom},
            "mixsort": {"function": evaluator.evaluate_mixsort},
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
        )
        logger.info("\n" + summary)

        logger.info("Completed")
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    os.environ["AUTOGRAPH_VERBOSITY"] = "5"
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )

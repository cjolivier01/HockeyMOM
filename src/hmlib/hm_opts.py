from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from hmlib.config import get_nested_value


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
        self.parser = self.parser(parser)

    @staticmethod
    def parser(parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        # stitching
        parser.add_argument(
            "--game-id",
            default=None,
            type=str,
            help="Game ID",
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
            "--show-scaled",
            type=float,
            default=None,
            help="scale showed image (ignored is --show-image is not specified)",
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
        # parser.add_argument(
        #     "--encoder",
        #     "--video-stream-encode-method",
        #     dest="video_stream_encode_method",
        #     default="cv2",
        #     type=str,
        #     help="Video stream decode method [cv2, ffmpeg, torchvision, tochaudio]",
        # )
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
            default=0.0,
            help="Offset for left video startig point (first supplied video)",
        )
        parser.add_argument(
            "--rfo",
            "--right-frame-offset",
            dest="rfo",
            type=float,
            default=0.0,
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
            default="autooptimiser_out.pto",
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
            "--blend-mode",
            "--blend_mode",
            default="laplacian",
            type=str,
            help="Stitching blend mode (multiblend|laplacian|gpu-hard-seam)",
        )
        parser.add_argument(
            "--skip_final_video_save",
            "--skip-final-video-save",
            dest="skip_final_video_save",
            action="store_true",
            help="Don't save the output video frames",
        )
        parser.add_argument(
            "--save_stitched",
            "--save-stitched",
            dest="save_stitched",
            action="store_true",
            help="Don't save the output video",
        )
        return parser

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return self.init(opt)

    # TODO: How can this be generalized with the nesting in the yaml?
    CONFIG_TO_ARGS = [
        # "model.tracker.pre_hm": "pre_hm",
        "model.tracker",
    ]

    @staticmethod
    def init(opt, parser: argparse.ArgumentParser = None):
        # Normalize some conflicting arguments
        if getattr(opt, "tracker", "") == "centertrack":
            if hasattr(opt, "test_size") and (
                hasattr(opt, "input_w") and hasattr(opt, "input_h")
            ):
                from lib.opts import opts

                assert not opt.test_size or (
                    opt.input_h in [-1, opts.DEFAULT_INPUT_HW]
                    and opt.input_w in [-1, opts.DEFAULT_INPUT_HW]
                )
                if not opt.test_size:
                    opt.test_size = f"{opt.input_h}x{opt.input_w}"
                else:
                    sz = opt.test_size.split("x")
                    opt.input_h = int(sz[0])
                    opt.input_w = int(sz[1]) if len(sz) > 1 else opt.input_h

            print("Have")

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

        return opt

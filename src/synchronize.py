"""
Experiments in stitching
"""

import argparse
import os
import hm_opts

from hmlib.stitching.configure_stitching import configure_video_stitching
from hmlib.video.ffmpeg import BasicVideoInfo

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("--num-workers", default=1, type=int, help="Number of stitching workers")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size")
    parser.add_argument("--left", default="left.mp4", type=str, help="Left file to be stitched")
    parser.add_argument("--right", default="right.mp4", type=str, help="Right file to be stitched")
    return parser


def make_video_path(game_id: str, file_name: str) -> str:
    if os.path.exists(file_name):
        return file_name
    return os.path.join(os.environ["HOME"], "Videos", game_id)


def main(args):
    assert args.game_id and "--gamne-id is required"

    video_left = make_video_path(game_id=args.game_id, file_name=args.left)
    video_right = make_video_path(game_id=args.game_id, file_name=args.right)

    left_vid = BasicVideoInfo(video_left)
    right_vid = BasicVideoInfo(video_right)

    total_frames = min(left_vid.frame_count, right_vid.frame_count)
    print(f"Total possible stitched video frames: {total_frames}")

    pto_project_file, lfo, rfo = configure_video_stitching(
        dir_name=os.path.join(os.environ["HOME"], "Videos", args.game_id),
        video_left=video_left,
        video_right=video_right,
        project_file_name="my_project.pto",
        left_frame_offset=None,
        right_frame_offset=None,
    )
    print(f"{pto_project_file=}")
    print(f"{lfo=}")
    print(f"{rfo=}")
    return


if __name__ == "__main__":
    parser = make_parser()
    parser = hm_opts.parser(parser=parser)
    args = parser.parse_args()

    main(args)
    print("Done.")

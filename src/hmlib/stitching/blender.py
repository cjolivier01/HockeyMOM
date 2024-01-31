"""
Experiments in stitching
"""
import os
import argparse
import numpy as np
from typing import Tuple, List
import cv2

import torch
import torch.nn.functional as F

import hockeymom.core as core
from hmlib.tracking_utils.timer import Timer
from hmlib.video_out import VideoOutput, ImageProcData, resize_image, rotate_image
from hmlib.video_out import make_visible_image
from hmlib.video_stream import VideoStreamWriter, VideoStreamReader
from hmlib.stitching.laplacian_blend import LaplacianBlend
from hmlib.stitching.laplacian_blend import show as show_image
from hmlib.stitching.synchronize import synchronize_by_audio

from hmlib.stitching.remapper import (
    ImageRemapper,
    read_frame_batch,
)

ROOT_DIR = os.getcwd()


def make_parser():
    parser = argparse.ArgumentParser("Image Remapper")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output_video",
        dest="output_video",
        type=str,
        default=None,
        help="Show images",
    )
    parser.add_argument(
        "--project-file",
        "--project_file",
        default="autooptimiser_out.pto",
        type=str,
        help="Use project file as input to stitcher",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        default=1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=str,
        help="Video directory to find 'left.mp4' and 'right.mp4'",
    )
    parser.add_argument(
        "--lfo",
        "--left_frame_offset",
        default=None,
        type=float,
        help="Left frame offset",
    )
    parser.add_argument(
        "--rfo",
        "--right_frame_offset",
        default=None,
        type=float,
        help="Right frame offset",
    )
    parser.add_argument(
        "--skip_final_video_save",
        "--skip-final-video-save",
        dest="skip_final_video_save",
        action="store_true",
        help="Don't save the output video frames",
    )
    parser.add_argument(
        "--rotation_angle",
        default=0,
        type=int,
        help="Rotation angle of final stitched image(s)",
    )
    return parser


class BlendImageInfo:
    def __init__(self, width: int, height: int, xpos: int, ypos: int):
        self.width = width
        self.height = height
        self.xpos = xpos
        self.ypos = ypos


class ImageAndPos:
    def __init__(self, image: torch.Tensor, xpos: int, ypos: int):
        self.image = image
        self.xpos = xpos
        self.ypos = ypos


class ImageBlender:
    def __init__(
        self,
        images_info: List[BlendImageInfo],
        seam_mask: torch.Tensor,
        xor_mask: torch.Tensor,
        laplacian_blend: False,
    ):
        self._images_info = images_info
        self._seam_mask = seam_mask.clone()
        self._xor_mask = xor_mask.clone()
        if laplacian_blend:
            self._laplacian_blend = LaplacianBlend(
                max_levels=4,
                channels=3,
                seam_mask=self._seam_mask,
                xor_mask=self._xor_mask,
            )
        else:
            self._laplacian_blend = None
        assert self._seam_mask.shape[1] == self._xor_mask.shape[1]
        assert self._seam_mask.shape[0] == self._xor_mask.shape[0]

    def init(self):
        # Check some sanity
        print(
            f"Final stitched image size: {self._seam_mask.shape[1]} x {self._seam_mask.shape[0]}"
        )
        self._unique_values = torch.unique(self._seam_mask)
        self._left_value = self._unique_values[0]
        self._right_value = self._unique_values[1]
        assert len(self._unique_values) == 2
        print("Initialized")

    def forward(self, image_1: torch.Tensor, image_2: torch.Tensor):
        # print(
        #     f"1={image_1.shape} @ {self._images_info[0].xpos}, {self._images_info[0].ypos}"
        # )
        # print(
        #     f"2={image_2.shape} @ {self._images_info[1].xpos}, {self._images_info[1].ypos}"
        # )
        batch_size = image_1.shape[0]
        channels = image_1.shape[1]

        if self._laplacian_blend is None:
            canvas = torch.empty(
                size=(
                    batch_size,
                    channels,
                    self._seam_mask.shape[0],
                    self._seam_mask.shape[1],
                ),
                dtype=torch.uint8 if self._laplacian_blend is None else torch.float,
                device=self._seam_mask.device,
            )
            # full_left = torch.zeros_like(canvas)
            # full_right = torch.zeros_like(canvas)
        else:
            # full_left = torch.zeros(
            #     size=(
            #         batch_size,
            #         channels,
            #         self._seam_mask.shape[0],
            #         self._seam_mask.shape[1],
            #     ),
            #     dtype=torch.uint8 if self._laplacian_blend is None else torch.float,
            #     device=self._seam_mask.device,
            # )
            # full_right = torch.zeros(
            #     size=(
            #         batch_size,
            #         channels,
            #         self._seam_mask.shape[0],
            #         self._seam_mask.shape[1],
            #     ),
            #     dtype=torch.uint8 if self._laplacian_blend is None else torch.float,
            #     device=self._seam_mask.device,
            # )
            pass

        h1 = image_1.shape[2]
        w1 = image_1.shape[3]
        x1 = self._images_info[0].xpos
        y1 = self._images_info[0].ypos
        h2 = image_2.shape[2]
        w2 = image_2.shape[3]
        x2 = self._images_info[1].xpos
        y2 = self._images_info[1].ypos

        assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
        if y1 < y2:
            y2 -= y1
            y1 = 0
        elif y2 < y1:
            y1 -= y2
            y2 = 0
        assert x1 == 0 or x2 == 0  # for now this is the case

        # def _make_full(img_1, img_2):
        #     img1 = img_1[:, :, 0:h1, 0:w1]
        #     full_left[:, :, y1 : y1 + h1 + y1, x1 : x1 + w1] = img1

        #     img2 = img_2[:, :, 0:h2, 0:w2]
        #     full_right[:, :, y2 : y2 + h2, x2 : x2 + w2] = img2
        #     return full_left, full_right

        def _make_full(img_1, img_2):
            assert h1 == img_1.shape[2]
            assert w1 == img_1.shape[3]
            # img1 = img_1[:, :, 0:h1, 0:w1]
            # full_left[:, :, y1 : y1 + h1 + y1, x1 : x1 + w1] = img_1

            full_left = torch.nn.functional.pad(
                img_1,
                (
                    x1,
                    self._seam_mask.shape[1] - x1 - w1,
                    y1,
                    self._seam_mask.shape[0] - y1 - h1,
                ),
                mode="constant",
            )

            assert h2 == img_2.shape[2]
            assert w2 == img_2.shape[3]
            # img2 = img_2[:, :, 0:h2, 0:w2]

            # full_right[:, :, y2 : y2 + h2, x2 : x2 + w2] = img_2

            full_right = torch.nn.functional.pad(
                img_2,
                (
                    x2,
                    self._seam_mask.shape[1] - x2 - w2,
                    y2,
                    self._seam_mask.shape[0] - y2 - h2,
                ),
                mode="constant",
            )

            return full_left, full_right

        if self._laplacian_blend is not None:
            # TODO: Can get rid of canvas creation up top for this path
            full_left, full_right = _make_full(image_1, image_2)
            # full_left = full_left.contiguous
            # full_left /= 255.0
            # full_right /= 255.0
            canvas = self._laplacian_blend.forward(left=full_left, right=full_right)
            # canvas = self._laplacian_blend.forward(
            #     left=image_1 / 255.0,
            #     right=image_2 / 255.0,
            #     make_full_fn=_make_full,
            # )
        else:
            full_left, full_right = _make_full(image_1, image_2)
            canvas[:, :, self._seam_mask == self._left_value] = full_left[
                :, :, self._seam_mask == self._left_value
            ]
            canvas[:, :, self._seam_mask == self._right_value] = full_right[
                :, :, self._seam_mask == self._right_value
            ]

        return canvas


def make_cv_compatible_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        assert tensor.dim() == 3
        if tensor.size(0) == 3 or tensor.size(0) == 4:
            # Need to make channels-last
            tensor = tensor.permute(1, 2, 0)
        return tensor.contiguous().cpu().numpy()
    if tensor.shape[0] == 3 or tensor.shape[0] == 4:
        tensor = tensor.transpose(1, 2, 0)
    return np.ascontiguousarray(tensor)


def make_seam_and_xor_masks(
    dir_name: str,
    images_and_positions: List[ImageAndPos],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(images_and_positions) == 2
    seam_filename = os.path.join(dir_name, "seam_file.png")
    xor_filename = os.path.join(dir_name, "xor_file.png")
    blender = core.EnBlender(
        args=[
            f"--save-seams",
            seam_filename,
            f"--save-xor",
            xor_filename,
        ]
    )

    _ = blender.blend_images(
        left_image=make_cv_compatible_tensor(images_and_positions[0].image),
        left_xy_pos=[images_and_positions[0].xpos, images_and_positions[0].ypos],
        right_image=make_cv_compatible_tensor(images_and_positions[1].image),
        right_xy_pos=[images_and_positions[1].xpos, images_and_positions[1].ypos],
    )
    seam_tensor = cv2.imread(seam_filename, cv2.IMREAD_ANYDEPTH)
    xor_tensor = cv2.imread(xor_filename, cv2.IMREAD_ANYDEPTH)
    return seam_tensor, xor_tensor


def get_dims_for_output_video(
    height: int, width: int, max_width: int, allow_resize: bool = True
):
    if allow_resize and max_width and width > max_width:
        hh = float(height)
        ww = float(width)
        ar = ww / hh
        new_h = float(max_width) / ar
        return int(new_h), int(max_width)
    return int(height), int(width)


def blend_video(
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    basename_1: str,
    basename_2: str,
    interpolation: str = None,
    lfo: float = None,
    rfo: float = None,
    show: bool = False,
    start_frame_number: int = 0,
    output_video: str = None,
    #max_width: int = 8192,
    max_width: int = 9999,
    rotation_angle: int = 0,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda"),
    skip_final_video_save: bool = False,
):
    video_file_1 = os.path.join(dir_name, video_file_1)
    video_file_2 = os.path.join(dir_name, video_file_2)

    if lfo is None or rfo is None:
        lfo, rfo = synchronize_by_audio(video_file_1, video_file_2)

    # cap_1 = VideoStreamReader(os.path.join(dir_name, video_file_1), device=device)
    cap_1 = cv2.VideoCapture(video_file_1)
    if not cap_1 or not cap_1.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_1}")
    else:
        if lfo or start_frame_number:
            cap_1.set(cv2.CAP_PROP_POS_FRAMES, lfo + start_frame_number)

    # cap_2 = VideoStreamReader(os.path.join(dir_name, video_file_2))
    cap_2 = cv2.VideoCapture(video_file_2)
    if not cap_2 or not cap_2.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_2}")
    else:
        if rfo or start_frame_number:
            cap_2.set(cv2.CAP_PROP_POS_FRAMES, rfo + start_frame_number)

    source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
    source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)

    remapper_1 = ImageRemapper(
        dir_name=dir_name,
        basename=basename_1,
        source_hw=source_tensor_1.shape[-2:],
        channels=source_tensor_1.shape[1],
        interpolation=interpolation,
        add_alpha_channel=False,
    )
    remapper_1.init(batch_size=batch_size)
    remapper_1.to(device=device)

    remapper_2 = ImageRemapper(
        dir_name=dir_name,
        basename=basename_2,
        source_hw=source_tensor_2.shape[-2:],
        channels=source_tensor_2.shape[1],
        interpolation=interpolation,
        add_alpha_channel=False,
    )
    remapper_2.init(batch_size=batch_size)
    remapper_2.to(device=device)

    video_out = None

    timer = Timer()
    frame_count = 0
    blender = None
    frame_id = start_frame_number
    try:
        while True:
            destination_tensor_1 = remapper_1.forward(source_image=source_tensor_1)
            destination_tensor_2 = remapper_2.forward(source_image=source_tensor_2)

            if frame_count == 0:
                seam_tensor, xor_tensor = make_seam_and_xor_masks(
                    dir_name=dir_name,
                    images_and_positions=[
                        ImageAndPos(
                            image=destination_tensor_1[0],
                            xpos=remapper_1.xpos,
                            ypos=remapper_1.ypos,
                        ),
                        ImageAndPos(
                            image=destination_tensor_2[0],
                            xpos=remapper_2.xpos,
                            ypos=remapper_2.ypos,
                        ),
                    ],
                )

                # show_image("seam_tensor", torch.from_numpy(seam_tensor))
                # show_image("xor_tensor", torch.from_numpy(xor_tensor))

                blender = ImageBlender(
                    images_info=[
                        BlendImageInfo(
                            width=cap_1.get(cv2.CAP_PROP_FRAME_WIDTH),
                            height=cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            xpos=remapper_1.xpos,
                            ypos=remapper_1.ypos,
                        ),
                        BlendImageInfo(
                            width=cap_2.get(cv2.CAP_PROP_FRAME_WIDTH),
                            height=cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            xpos=remapper_2.xpos,
                            ypos=remapper_2.ypos,
                        ),
                    ],
                    seam_mask=torch.from_numpy(seam_tensor).contiguous().to(device),
                    xor_mask=torch.from_numpy(xor_tensor).contiguous().to(device),
                    laplacian_blend=True,
                    # laplacian_blend=False,
                )
                blender.init()

            # blended = destination_tensor_1
            blended = blender.forward(
                image_1=destination_tensor_1,
                image_2=destination_tensor_2,
            )

            # show_image("blended", blended, wait=False)

            if output_video:
                video_dim_height, video_dim_width = get_dims_for_output_video(
                    height=blended.shape[-2],
                    width=blended.shape[-1],
                    max_width=max_width,
                )
                if video_out is None:
                    fps = cap_1.get(cv2.CAP_PROP_FPS)
                    video_out = VideoOutput(
                        name="StitchedOutput",
                        args=None,
                        output_video_path=output_video,
                        output_frame_width=video_dim_width,
                        output_frame_height=video_dim_height,
                        fps=fps,
                        device=blended.device,
                        skip_final_save=skip_final_video_save,
                        fourcc="auto",
                    )
                if (
                    video_dim_height != blended.shape[-2]
                    or video_dim_width != blended.shape[-1]
                ):
                    assert False  # why is this?
                    for i in range(len(blended)):
                        resized = resize_image(
                            img=blended[i].permute(1, 2, 0),
                            new_width=video_dim_width,
                            new_height=video_dim_height,
                        )
                        if isinstance(video_out, VideoStreamWriter):
                            video_out.append(my_blended)
                            frame_id += batch_size
                        else:
                            video_out.append(
                                ImageProcData(
                                    frame_id=frame_id,
                                    img=resized.contiguous().cpu(),
                                    current_box=None,
                                )
                            )
                        frame_id += 1
                else:
                    my_blended = blended.permute(0, 2, 3, 1)
                    if rotation_angle:
                        my_blended = rotate_image(
                            img=my_blended,
                            angle=rotation_angle,
                            rotation_point=(
                                my_blended.shape[-2] // 2,
                                my_blended.shape[-3] // 2,
                            ),
                        )
                    if show:
                        for img in my_blended:
                            show_image("stitched", img, wait=False)
                    for i in range(len(my_blended)):
                        video_out.append(
                            ImageProcData(
                                frame_id=frame_id,
                                img=my_blended[i],
                                current_box=None,
                            )
                        )
                        frame_id += 1
                del my_blended
            else:
                pass

            frame_count += 1

            if frame_count != 1:
                timer.toc()

            if frame_count % 20 == 0:
                print(
                    "Stitching: {:.2f} fps".format(
                        batch_size * 1.0 / max(1e-5, timer.average_time)
                    )
                )
                if frame_count % 50 == 0:
                    timer = Timer()

            if show:
                for i in range(len(blended)):
                    show_image("stitched", blended[i])
                    cv2.waitKey(1)

            source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
            source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)
            timer.tic()
    finally:
        if video_out is not None:
            if isinstance(video_out, VideoStreamWriter):
                video_out.flush()
                video_out.close()
            else:
                video_out.stop()


def main(args):
    with torch.no_grad():
        blend_video(
            "left.mp4",
            "right.mp4",
            args.video_dir,
            "mapping_0000",
            "mapping_0001",
            lfo=args.lfo,
            rfo=args.rfo,
            interpolation="bilinear",
            show=args.show,
            start_frame_number=0,
            # output_video="stitched_output.mkv",
            rotation_angle=args.rotation_angle,
            batch_size=args.batch_size,
            skip_final_video_save=args.skip_final_video_save,
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    print("Done.")

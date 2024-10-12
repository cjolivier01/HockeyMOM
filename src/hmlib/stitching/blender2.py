"""
Experiments in stitching
"""

import argparse
import copy
import datetime
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

import hockeymom.core as core
from hmlib.hm_opts import copy_opts, hm_opts
from hmlib.stitching.configure_stitching import get_image_geo_position
from hmlib.stitching.laplacian_blend import LaplacianBlend
from hmlib.stitching.remapper import ImageRemapper, read_frame_batch
from hmlib.stitching.synchronize import synchronize_by_audio
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import show_image
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
)
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.pt_visualization import draw_box
from hmlib.video_out import VideoOutput, resize_image, rotate_image
from hmlib.video_stream import VideoStreamReader, VideoStreamWriter

ROOT_DIR = os.getcwd()

logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser("Image Remapper")
    parser = hm_opts.parser(parser)
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
        "-q",
        "--queue-size",
        dest="queue_size",
        default=1,
        type=int,
        help="Queue size",
    )
    parser.add_argument(
        "--start-frame-number",
        default=0,
        type=int,
        help="Start frame number",
    )
    parser.add_argument(
        "--python",
        action="store_true",
        help="Python blending path",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw boxes",
    )
    parser.add_argument(
        "--rotation_angle",
        default=0,
        type=int,
        help="Rotation angle of final stitched image(s)",
    )
    return parser


@dataclass
class BlendImageInfo:
    def __init__(self, width: int, height: int, xpos: int, ypos: int):
        self.width = width
        self.height = height
        self.xpos = xpos
        self.ypos = ypos


@dataclass
class ImageAndPos:
    def __init__(self, image: torch.Tensor, xpos: int, ypos: int):
        self.image = image
        self.xpos = xpos
        self.ypos = ypos


class PtImageBlender:
    def __init__(
        self,
        images_info: List[BlendImageInfo],
        seam_mask: torch.Tensor,
        xor_mask: torch.Tensor,
        laplacian_blend: False,
        max_levels: int = 4,
        cuda_stream: torch.cuda.Stream = None,
        dtype: torch.dtype = torch.float,
    ):
        self._images_info = images_info
        self._seam_mask = seam_mask.clone()
        self._xor_mask = xor_mask.clone()
        self._cuda_stream = cuda_stream
        self._dtype = dtype
        self.max_levels = max_levels
        if laplacian_blend:
            self._laplacian_blend = LaplacianBlend(
                max_levels=self.max_levels,
                channels=3,
                seam_mask=self._seam_mask,
                xor_mask=self._xor_mask,
            )
        else:
            self._laplacian_blend = None
        assert self._seam_mask.shape[1] == self._xor_mask.shape[1]
        assert self._seam_mask.shape[0] == self._xor_mask.shape[0]

        # Final misc init
        self._unique_values = None
        self._left_value = None
        self._right_value = None

    def init(self):
        # Check some sanity
        print(f"Final stitched image size: {self._seam_mask.shape[1]} x {self._seam_mask.shape[0]}")
        self._unique_values = torch.unique(self._seam_mask)
        self._left_value = self._unique_values[0]
        self._right_value = self._unique_values[1]
        assert len(self._unique_values) == 2
        print("Initialized")

    def synchronize(self):
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()

    def forward(
        self,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        synchronize: bool = False,
    ):
        if self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                results = self._forward(image_1, image_2)
                if synchronize:
                    self.synchronize()
                return results
        return self._forward(image_1, image_2)

    def _forward(self, image_1: torch.Tensor, image_2: torch.Tensor):
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
                dtype=torch.uint8 if self._laplacian_blend is None else self._dtype,
                device=self._seam_mask.device,
            )

        H1 = 0
        W1 = 1
        X1 = 2
        Y1 = 3

        H2 = 0
        W2 = 1
        X2 = 2
        Y2 = 3

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
        if x1 < x2:
            x2 -= x1
            x1 = 0
        elif x2 < x1:
            x1 -= x2
            x2 = 0

        ainfo_1 = torch.tensor([h1, w1, x1, y1], dtype=torch.int64)
        ainfo_2 = torch.tensor([h2, w2, x2, y2], dtype=torch.int64)
        canvas_dims = torch.tensor(
            [self._seam_mask.shape[0], self._seam_mask.shape[1]], dtype=torch.int64
        )

        level_ainfo_1 = [ainfo_1]
        level_ainfo_2 = [ainfo_2]
        level_canvas_dims = [canvas_dims]

        for _ in range(self.max_levels):
            ainfo_1 = ainfo_1 // 2
            ainfo_2 = ainfo_2 // 2
            canvas_dims = canvas_dims // 2
            level_ainfo_1.append(ainfo_1)
            level_ainfo_2.append(ainfo_2)
            level_canvas_dims.append(canvas_dims)

        def _make_full(img_1, img_2, level):
            ainfo_1 = level_ainfo_1[level]
            ainfo_2 = level_ainfo_2[level]

            h1 = ainfo_1[H1]
            w1 = ainfo_1[W1]
            x1 = ainfo_1[X1]
            y1 = ainfo_1[Y1]
            h2 = ainfo_2[H2]
            w2 = ainfo_2[W2]
            x2 = ainfo_2[X2]
            y2 = ainfo_2[Y2]

            # If these hit, you may have not passed "-s" to autotoptimiser
            assert x1 == 0 or x2 == 0  # for now this is the case
            assert y1 == 0 or y2 == 0  # for now this is the case

            canvas_dims = level_canvas_dims[level]

            full_left = torch.nn.functional.pad(
                img_1,
                (
                    x1,
                    canvas_dims[1] - x1 - w1,
                    y1,
                    canvas_dims[0] - y1 - h1,
                ),
                mode="constant",
            )

            full_right = torch.nn.functional.pad(
                img_2,
                (
                    x2,
                    canvas_dims[1] - x2 - w2,
                    y2,
                    canvas_dims[0] - y2 - h2,
                ),
                mode="constant",
            )

            return full_left, full_right

        if self._laplacian_blend is not None:
            # TODO: Can get rid of canvas creation up top for this path
            canvas = self._laplacian_blend.forward(
                left=image_1,
                right=image_2,
                make_full_fn=_make_full,
            )
        else:
            full_left, full_right = _make_full(image_1, image_2, level=0)
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
    force: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(images_and_positions) == 2
    seam_filename = os.path.join(dir_name, "seam_file.png")
    xor_filename = os.path.join(dir_name, "xor_file.png")
    if not force and os.path.isfile(seam_filename):
        mapping_file = os.path.join(dir_name, "mapping_0000.tif")
        if os.path.exists(mapping_file):
            mapping_file_mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(mapping_file)
            ).isoformat()
            seam_file_mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(seam_filename)
            ).isoformat()
            force = mapping_file_mtime >= seam_file_mtime
            if force:
                print(f"Recreating seam files because mapping file is newer ({mapping_file})")
        else:
            print(f"Warning: no mapping file found: {mapping_file}")
    if force or not os.path.isfile(seam_filename) or not os.path.isfile(xor_filename):
        blender = core.EnBlender(
            args=[
                f"--save-seams",
                seam_filename,
                f"--save-xor",
                xor_filename,
            ]
        )
        # Blend one image to create the seam file
        _ = blender.blend_images(
            left_image=make_cv_compatible_tensor(images_and_positions[0].image),
            left_xy_pos=[images_and_positions[0].xpos, images_and_positions[0].ypos],
            right_image=make_cv_compatible_tensor(images_and_positions[1].image),
            right_xy_pos=[images_and_positions[1].xpos, images_and_positions[1].ypos],
        )
    seam_tensor = cv2.imread(seam_filename, cv2.IMREAD_ANYDEPTH)
    xor_tensor = cv2.imread(xor_filename, cv2.IMREAD_ANYDEPTH)
    return seam_tensor, xor_tensor


def create_blender_config(
    mode: str,
    dir_name: str,
    basename: str,
    device: torch.device,
    levels: int = 10,
    lazy_init: bool = False,
    interpolation: str = "bilinear",
) -> core.RemapperConfig:
    config = core.BlenderConfig()
    if not mode or mode == "multiblend":
        # Nothing needed for legacy multiblend mode
        return config

    left_file = os.path.join(dir_name, f"{basename}0000.tif")
    right_file = os.path.join(dir_name, f"{basename}0001.tif")
    left_pos = get_image_geo_position(tiff_image_file=left_file)
    right_pos = get_image_geo_position(tiff_image_file=right_file)

    left_img = torch.from_numpy(cv2.imread(left_file))
    right_img = torch.from_numpy(cv2.imread(right_file))
    assert left_img is not None and right_img is not None
    config.mode = mode
    config.levels = levels
    config.device = str(device)
    seam, xor_map = make_seam_and_xor_masks(
        dir_name=dir_name,
        images_and_positions=[
            ImageAndPos(image=left_img, xpos=left_pos[0], ypos=left_pos[1]),
            ImageAndPos(image=right_img, xpos=right_pos[0], ypos=right_pos[1]),
        ],
    )
    config.seam = torch.from_numpy(seam)
    config.xor_map = torch.from_numpy(xor_map)
    config.lazy_init = lazy_init
    config.interpolation = interpolation
    return config


def get_dims_for_output_video(height: int, width: int, max_width: int, allow_resize: bool = True):
    if allow_resize and max_width and width > max_width:
        hh = float(height)
        ww = float(width)
        ar = ww / hh
        new_h = float(max_width) / ar
        return int(new_h), int(max_width)
    return int(height), int(width)


def my_draw_box(
    image: torch.Tensor,
    x1: int | None,
    y1: int | None,
    x2: int | None,
    y2: int | None,
    color: Tuple[int, int, int],
    thickness: int = 4,
) -> torch.Tensor:
    if x1 is None:
        x1 = 0
    if y1 is None:
        y1 = 0
    w, h = image_width(image), image_height(image)
    if x2 is None:
        x2 = w - 1
    elif x2 == w:
        x2 -= 1
    if y2 is None:
        y2 = h - 1
    elif y2 == h:
        y2 -= 1
    return draw_box(
        image=image, tlbr=[int(x1), int(y1), int(x2), int(y2)], color=color, thickness=thickness
    )


@dataclass
class Point:
    x: int
    y: int


@dataclass
class CanvasInfo:
    positions: Union[List[Point], None] = None
    width: int = 0
    height: int = 0


def get_canvas_info(
    size_1: List[int], xy_pos_1: List[int], size_2: List[int], xy_pos_2: List[int]
) -> CanvasInfo:
    """
    returns (height, width)
    """
    h1, w1 = size_1[-2:]
    h2, w2 = size_2[-2:]
    x1, y1 = xy_pos_1
    x2, y2 = xy_pos_2
    assert x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0
    if x1 <= x2:
        x2 -= x1
        x1 = 0
    else:
        x1 -= x2
        x2 = 0
    if y1 <= y2:
        y2 -= y1
        y1 = 0
    else:
        y1 -= y2
        y2 = 0
    canvas_w = max(x1 + w1, x2 + w2)
    canvas_h = max(y1 + h1, y2 + h2)
    return CanvasInfo(
        positions=[Point(x=x1, y=y1), Point(x=x2, y=y2)], width=canvas_w, height=canvas_h
    )


class SmartBlender:
    def __init__(
        self,
        remapper_1: ImageRemapper,
        remapper_2: ImageRemapper,
        minimize_blend: bool,
        use_python_blender: bool,
        dtype: torch.dtype,
        overlap_pad: int,
        draw: bool,
    ) -> None:
        self._remapper_1 = copy.deepcopy(remapper_1)
        self._remapper_2 = copy.deepcopy(remapper_2)
        self._use_python_blender = use_python_blender
        self._canvas_info: CanvasInfo = get_canvas_info(
            size_1=[self._remapper_1.height, self._remapper_1.width],
            xy_pos_1=[self._remapper_1.xpos, self._remapper_1.ypos],
            size_2=[self._remapper_2.height, self._remapper_2.width],
            xy_pos_2=[self._remapper_2.xpos, self._remapper_2.ypos],
        )
        self._dtype = dtype
        self._overlap_pad = overlap_pad
        self._draw = draw
        self._minimize_blend = minimize_blend
        self._padded_blended_tlbr = None
        self._init()

    def _init(self) -> None:
        if not self._minimize_blend:
            return
        x1, y1, x2, y2 = (
            self._canvas_info.positions[0].x,
            self._canvas_info.positions[0].y,
            self._canvas_info.positions[1].x,
            self._canvas_info.positions[1].y,
        )

        self._remapper_1.xpos = x1
        self._remapper_2.xpos = x1 + self._overlap_pad  # start overlapping right away
        width_1 = self._remapper_1.width
        overlapping_width = int(width_1 - x2)
        assert width_1 > x2
        # seam tensor box (box we'll be blending)
        self._padded_blended_tlbr = [
            x2 - self._overlap_pad,  # x1
            max(0, min(y1, y2) - self._overlap_pad),  # y1
            width_1 + self._overlap_pad,  # x2
            min(
                self._canvas_info.height,
                max(y1 + self._remapper_1.height, y2 + self._remapper_2.height) + self._overlap_pad,
            ),  # y2
        ]
        assert x2 - self._overlap_pad >= 0
        assert width_1 + self._overlap_pad <= self._canvas_info.width

    def convert_mask_tensor(self, mask: torch.Tensor) -> torch.Tensor:
        # Mask should be the same size as our canvas
        assert image_width(mask) == self._canvas_info.width
        assert image_height(mask) == self._canvas_info.height
        if not self._minimize_blend:
            return mask
        return mask[
            ...,
            self._canvas_info.positions[1].x
            - self._overlap_pad : self._remapper_1.width
            + self._overlap_pad,
        ]

    def draw(self):
        pass

    def forward(
        self, remapped_image_1: torch.Tensor, remapped_image_2: torch.Tensor
    ) -> torch.Tensor:
        pass

    # if self._draw:
    #     self.draw(image)


def blend_video(
    opts: object,
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    basename_1: str,
    basename_2: str,
    device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    interpolation: str = None,
    lfo: float = None,
    rfo: float = None,
    python_blend: bool = False,
    show: bool = False,
    show_scaled: float | None = None,
    start_frame_number: int = 0,
    output_video: str = None,
    max_width: int = 9999,
    rotation_angle: int = 0,
    batch_size: int = 8,
    skip_final_video_save: bool = False,
    blend_mode: str = "laplacian",
    queue_size: int = 1,
    minimize_blend: bool = True,
    overlap_pad: int = 25,
    overlap_pad_value: int = 128,
    draw: bool = False,
):
    video_file_1 = os.path.join(dir_name, video_file_1)
    video_file_2 = os.path.join(dir_name, video_file_2)

    video_file_1 = os.path.join(dir_name, video_file_1)
    video_file_2 = os.path.join(dir_name, video_file_2)

    if lfo is None or rfo is None:
        lfo, rfo = synchronize_by_audio(video_file_1, video_file_2)

    cap_1 = VideoStreamReader(
        os.path.join(dir_name, video_file_1),
        type=opts.video_stream_decode_method,
        device=f"cuda:{gpu_index(want=2)}",
        batch_size=batch_size,
    )
    if not cap_1 or not cap_1.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_1}")
    if lfo or start_frame_number:
        cap_1.seek(frame_number=lfo + start_frame_number)

    cap_2 = VideoStreamReader(
        os.path.join(dir_name, video_file_2),
        type=opts.video_stream_decode_method,
        device=f"cuda:{gpu_index(want=3)}",
        batch_size=batch_size,
    )

    if not cap_2 or not cap_2.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_2}")
    if rfo or start_frame_number:
        cap_2.seek(frame_number=rfo + start_frame_number)

    v1_iter = iter(cap_1)
    v2_iter = iter(cap_2)

    source_tensor_1 = make_channels_first(next(v1_iter))
    source_tensor_2 = make_channels_first(next(v2_iter))

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

    smart_blender = SmartBlender(
        remapper_1=remapper_1,
        remapper_2=remapper_2,
        minimize_blend=minimize_blend,
        overlap_pad=overlap_pad,
        draw=draw,
        use_python_blender=python_blend,
        dtype=dtype,
    )

    video_out = None

    timer = Timer()
    frame_count = 0
    blender = None
    frame_id = start_frame_number
    try:
        width_1, width_2 = remapper_1.width, remapper_2.width
        overlapping_width, x1 = None, None
        cap_1_width = cap_1.width
        cap_2_width = cap_2.width
        canvas_width, canvas_height = None, None
        while True:
            remapped_tensor_1 = remapper_1.forward(source_image=source_tensor_1).to(
                device=device, non_blocking=True
            )
            remapped_tensor_2 = remapper_2.forward(source_image=source_tensor_2).to(
                device=device, non_blocking=True
            )

            if frame_count == 0:
                seam_tensor, xor_tensor = make_seam_and_xor_masks(
                    dir_name=dir_name,
                    images_and_positions=[
                        ImageAndPos(
                            image=remapped_tensor_1[0],
                            xpos=remapper_1.xpos,
                            ypos=remapper_1.ypos,
                        ),
                        ImageAndPos(
                            image=remapped_tensor_2[0],
                            xpos=remapper_2.xpos,
                            ypos=remapper_2.ypos,
                        ),
                    ],
                )
                if minimize_blend:
                    canvas_width = image_width(seam_tensor)
                    canvas_height = image_height(seam_tensor)
                    x1, y1, x2, y2 = (
                        remapper_1.xpos,
                        remapper_1.ypos,
                        remapper_2.xpos,
                        remapper_2.ypos,
                    )
                    if y1 <= y2:
                        y2 -= y1
                        y1 = 0
                    else:
                        y1 -= y2
                        y2 = 0
                    width_1 = image_width(remapped_tensor_1)
                    width_2 = image_width(remapped_tensor_2)
                    assert x1 != x2  # they shouldn;t be starting in the same place
                    if x1 <= x2:
                        x2 -= x1
                        x1 = 0
                    elif x2 <= x1:
                        x1 -= x2
                        x2 = 0
                    remapper_1.xpos = x1
                    remapper_2.xpos = x1 + overlap_pad  # start overlapping right away
                    overlapping_width = int(width_1 - x2)
                    assert width_1 > x2
                    # seam tensor box (box we'll be blending)
                    padded_blended_tlbr = [
                        x2 - overlap_pad,  # x1
                        max(0, min(y1, y2) - overlap_pad),  # y1
                        width_1 + overlap_pad,  # x2
                        min(
                            image_height(seam_tensor),
                            max(y1 + remapper_1.height, y2 + remapper_2.height) + overlap_pad,
                        ),  # y2
                    ]
                    assert x2 - overlap_pad >= 0
                    assert width_1 + overlap_pad <= image_width(seam_tensor)

                    seam_tensor = smart_blender.convert_mask_tensor(seam_tensor)
                    xor_tensor = smart_blender.convert_mask_tensor(xor_tensor)

                    # seam_tensor = seam_tensor[:, x2 - overlap_pad : width_1 + overlap_pad]
                    # xor_tensor = xor_tensor[:, x2 - overlap_pad : width_1 + overlap_pad]
                    cap_1_width = overlapping_width + overlap_pad + overlap_pad
                    cap_2_width = overlapping_width + overlap_pad + overlap_pad

                # show_image("seam_tensor", torch.from_numpy(seam_tensor))
                # show_image("xor_tensor", torch.from_numpy(xor_tensor))
                if not python_blend:
                    # assert False  # Not interested in this path atm
                    blender = core.ImageBlender(
                        mode=(
                            core.ImageBlenderMode.Laplacian
                            if blend_mode == "laplacian"
                            else core.ImageBlenderMode.HardSeam
                        ),
                        half=False,
                        levels=10,
                        seam=torch.from_numpy(seam_tensor),
                        xor_map=torch.from_numpy(xor_tensor),
                        lazy_init=True,
                        interpolation="bilinear",
                    )
                    blender.to(device)
                else:
                    blender = PtImageBlender(
                        images_info=[
                            BlendImageInfo(
                                width=cap_1_width,
                                height=cap_1.height,
                                xpos=remapper_1.xpos,
                                ypos=remapper_1.ypos,
                            ),
                            BlendImageInfo(
                                width=cap_2_width,
                                height=cap_2.height,
                                xpos=remapper_2.xpos,
                                ypos=remapper_2.ypos,
                            ),
                        ],
                        seam_mask=torch.from_numpy(seam_tensor).contiguous().to(device),
                        xor_mask=torch.from_numpy(xor_tensor).contiguous().to(device),
                        laplacian_blend=blend_mode == "laplacian",
                    )

            if overlapping_width:
                assert image_width(remapped_tensor_1) == width_1  # sanity
                canvas = (
                    torch.zeros(
                        size=(
                            remapped_tensor_1.shape[0],
                            remapped_tensor_1.shape[1],
                            canvas_height,
                            canvas_width,
                        ),
                        dtype=remapped_tensor_1.dtype,
                        device=remapped_tensor_1.device,
                    )
                    + overlap_pad_value
                )
                dh1 = image_height(remapped_tensor_1)
                dh2 = image_height(remapped_tensor_2)
                # TODO: can be ... instead of so many colons

                partial_1 = remapped_tensor_1[:, :, :, : x2 + overlap_pad]
                partial_2 = remapped_tensor_2[:, :, :, overlapping_width - overlap_pad :]
                remapped_tensor_1 = remapped_tensor_1[:, :, :, x2 - overlap_pad : width_1]
                remapped_tensor_2 = remapped_tensor_2[:, :, :, : overlapping_width + overlap_pad]

            fwd_args = dict(
                image_1=remapped_tensor_1.to(torch.float, non_blocking=True),
                image_2=remapped_tensor_2.to(torch.float, non_blocking=True),
            )
            if not python_blend:
                fwd_args.update(
                    dict(
                        xy_pos_1=[remapper_1.xpos, remapper_1.ypos],
                        xy_pos_2=[remapper_2.xpos, remapper_2.ypos],
                    )
                )
            blended_img = blender.forward(**fwd_args)

            if overlapping_width:
                canvas[:, :, :, x2 - overlap_pad : x2 + overlapping_width + overlap_pad] = (
                    blended_img.clamp(min=0, max=255).to(dtype=canvas.dtype, non_blocking=True)
                )
                canvas[:, :, y1 : dh1 + y1, : x2 + overlap_pad] = partial_1
                canvas[:, :, y2 : dh2 + y2, x2 + overlapping_width - overlap_pad :] = partial_2
                blended = canvas
                if draw:
                    # left box
                    blended = my_draw_box(
                        blended,
                        x1=None,
                        y1=y1,
                        x2=x2 + overlap_pad - 1,
                        y2=dh1 + y1 - 1,
                        color=(255, 255, 0),
                    )
                    # right box
                    blended = my_draw_box(
                        blended,
                        x1=x2 + overlapping_width - overlap_pad,
                        y1=y2,
                        x2=None,
                        y2=dh2 + y2 - 1,
                        color=(0, 255, 255),
                    )
                    # Blended box
                    blended = my_draw_box(
                        blended,
                        *padded_blended_tlbr,
                        color=(255, 0, 0),
                    )
            else:
                blended = blended_img

            # show_image("blended", blended, wait=False)

            if output_video:
                if video_out is None:
                    video_dim_height, video_dim_width = get_dims_for_output_video(
                        height=blended.shape[-2],
                        width=blended.shape[-1],
                        max_width=max_width,
                    )
                    fps = cap_1.fps
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
                        # batch_size=batch_size,
                        cache_size=queue_size,
                    )
                if video_dim_height != blended.shape[-2] or video_dim_width != blended.shape[-1]:
                    assert False  # why is this?
                    for this_blended in blended:
                        resized = resize_image(
                            img=this_blended.permute(1, 2, 0),
                            new_width=video_dim_width,
                            new_height=video_dim_height,
                        )
                        if isinstance(video_out, VideoStreamWriter):
                            video_out.append(resized)
                            frame_id += batch_size
                        else:
                            video_out.append(
                                dict(
                                    frame_id=frame_id,
                                    img=resized.contiguous().cpu(),
                                    current_box=None,
                                )
                            )
                else:
                    my_blended = make_channels_last(blended)

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
                    if True:
                        video_out.append(
                            {
                                "frame_id": torch.tensor(frame_id, dtype=torch.int64),
                                "img": my_blended,
                                "current_box": None,
                            }
                        )
                        frame_id += len(my_blended)
                    else:
                        for this_blended in my_blended:
                            video_out.append(
                                dict(
                                    frame_id=frame_id,
                                    img=this_blended,
                                    current_box=None,
                                )
                            )
                del my_blended
            else:
                pass

            frame_id += 1
            frame_count += 1

            if frame_count != 1:
                timer.toc()

            if frame_count % 20 == 0:
                print(
                    "Stitching: {:.2f} fps".format(batch_size * 1.0 / max(1e-5, timer.average_time))
                )
                if frame_count % 50 == 0:
                    timer = Timer()

            if show:
                for this_blended in blended:
                    show_image(
                        "this_blended",
                        this_blended,
                        wait=False,
                        enable_resizing=show_scaled,
                    )

            source_tensor_1 = make_channels_first(next(v1_iter))
            source_tensor_2 = make_channels_first(next(v2_iter))
            timer.tic()
    except StopIteration:
        # All done.
        pass
    finally:
        if video_out is not None:
            if isinstance(video_out, VideoStreamWriter):
                video_out.flush()
                video_out.close()
            else:
                video_out.stop()


def get_mapping(dir_name: str, basename: str):
    x_file = os.path.join(dir_name, f"{basename}_x.tif")
    y_file = os.path.join(dir_name, f"{basename}_y.tif")
    xpos, ypos = get_image_geo_position(os.path.join(dir_name, f"{basename}.tif"))

    x_map = cv2.imread(x_file, cv2.IMREAD_ANYDEPTH)
    y_map = cv2.imread(y_file, cv2.IMREAD_ANYDEPTH)
    if x_map is None:
        raise AssertionError(f"Could not read mapping file: {x_file}")
    if y_map is None:
        raise AssertionError(f"Could not read mapping file: {y_file}")
    col_map = torch.from_numpy(x_map.astype(np.int64))
    row_map = torch.from_numpy(y_map.astype(np.int64))
    return xpos, ypos, col_map, row_map


def create_stitcher(
    dir_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    left_image_size_wh: Tuple[int, int],
    right_image_size_wh: Tuple[int, int],
    mapping_basename_1: str = "mapping_0000",
    mapping_basename_2: str = "mapping_0001",
    remapped_basename: str = "nona",
    blend_mode: str = "laplacian",
    interpolation: str = "bilinear",
    remap_on_async_stream: bool = False,
    levels: int = 6,
):
    blender_config = create_blender_config(
        mode=blend_mode,
        dir_name=dir_name,
        basename=remapped_basename,
        device=device,
        levels=levels,
        lazy_init=False,
        interpolation=interpolation,
    )

    xpos_1, ypos_1, col_map_1, row_map_1 = get_mapping(dir_name, mapping_basename_1)
    xpos_2, ypos_2, col_map_2, row_map_2 = get_mapping(dir_name, mapping_basename_2)

    remap_info_1 = core.RemapImageInfo()
    remap_info_1.src_width = int(left_image_size_wh[0])
    remap_info_1.src_height = int(left_image_size_wh[1])
    remap_info_1.col_map = col_map_1
    remap_info_1.row_map = row_map_1

    remap_info_2 = core.RemapImageInfo()
    remap_info_2.src_width = int(right_image_size_wh[0])
    remap_info_2.src_height = int(right_image_size_wh[1])
    remap_info_2.col_map = col_map_2
    remap_info_2.row_map = row_map_2

    stitcher = core.ImageStitcher(
        batch_size=batch_size,
        remap_image_info=[remap_info_1, remap_info_2],
        blender_mode=core.ImageBlenderMode.Laplacian,
        half=dtype == torch.float16,
        levels=blender_config.levels,
        remap_on_async_stream=remap_on_async_stream,
        seam=blender_config.seam,
        xor_map=blender_config.xor_map,
        lazy_init=False,
    )
    return stitcher, [xpos_1, ypos_1], [xpos_2, ypos_2]


def gpu_index(want: int = 1):
    return min(torch.cuda.device_count() - 1, want)


#
# Combined FPS= XY/(X+Y)
#


def main(args):
    opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    gpu_allocator = GpuAllocator(gpus=args.gpus)
    if not args.video_dir and args.game_id:
        args.video_dir = os.path.join(os.environ["HOME"], "Videos", args.game_id)
    video_gpu = torch.device("cuda", gpu_allocator.allocate_modern())
    fast_gpu = torch.device("cuda", gpu_allocator.allocate_fast())
    with torch.no_grad():
        blend_video(
            opts,
            video_file_1="left.mp4",
            video_file_2="right.mp4",
            dir_name=args.video_dir,
            basename_1="mapping_0000",
            basename_2="mapping_0001",
            lfo=args.lfo,
            rfo=args.rfo,
            python_blend=args.python,
            start_frame_number=args.start_frame_number,
            interpolation="bilinear",
            show=args.show_image,
            show_scaled=args.show_scaled,
            output_video=args.output_file,
            output_device=video_gpu,
            rotation_angle=args.rotation_angle,
            batch_size=args.batch_size,
            skip_final_video_save=args.skip_final_video_save,
            queue_size=args.queue_size,
            device=fast_gpu,
            dtype=torch.float16 if args.fp16 else torch.float,
            minimize_blend=True,
            draw=args.draw,
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    print("Done.")

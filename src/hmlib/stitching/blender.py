"""
Experiments in stitching
"""

import os
import argparse
import datetime
import numpy as np
from typing import Tuple, List, Optional
import cv2

import torch
import torch.nn.functional as F

import hockeymom.core as core
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.image import image_width, image_height
from hmlib.video_out import (
    VideoOutput,
    ImageProcData,
    resize_image,
    rotate_image,
)
from hmlib.video_stream import VideoStreamWriter, VideoStreamReader
from hmlib.stitching.laplacian_blend import LaplacianBlend, show_image
from hmlib.stitching.synchronize import synchronize_by_audio, get_image_geo_position
from hmlib.utils.gpu import (
    StreamTensorToDevice,
    StreamTensorToDtype,
    CachedIterator,
    StreamTensor,
    StreamCheckpoint,
    GpuAllocator,
    cuda_stream_scope,
)
from hmlib.hm_opts import hm_opts, copy_opts

from hmlib.stitching.remapper import (
    ImageRemapper,
    read_frame_batch,
)

ROOT_DIR = os.getcwd()


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

    def synchronize(self):
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()

    def forward(
        self,
        image_1: torch.Tensor,
        xy_pos_1: List[int],
        image_2: torch.Tensor,
        xy_pos_2: List[int],
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
                dtype=torch.uint8 if self._laplacian_blend is None else self._dtype,
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

        # def _make_full(img_1, img_2):
        #     img1 = img_1[:, :, 0:h1, 0:w1]
        #     full_left[:, :, y1 : y1 + h1 + y1, x1 : x1 + w1] = img1

        #     img2 = img_2[:, :, 0:h2, 0:w2]
        #     full_right[:, :, y2 : y2 + h2, x2 : x2 + w2] = img2
        #     return full_left, full_right

        def _make_full(img_1, img_2, level):
            # assert h1 == img_1.shape[2]
            # assert w1 == img_1.shape[3]
            # img1 = img_1[:, :, 0:h1, 0:w1]
            # full_left[:, :, y1 : y1 + h1 + y1, x1 : x1 + w1] = img_1

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

            # assert h2 == img_2.shape[2]
            # assert w2 == img_2.shape[3]
            # img2 = img_2[:, :, 0:h2, 0:w2]

            # full_right[:, :, y2 : y2 + h2, x2 : x2 + w2] = img_2

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

            # full_left, full_right = _make_full(image_1, image_2, level=0)
            # canvas = self._laplacian_blend.forward(left=full_left, right=full_right)

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
                print(
                    f"Recreating seam files because mapping file is newer ({mapping_file})"
                )
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
    opts: object,
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    basename_1: str,
    basename_2: str,
    interpolation: str = None,
    lfo: float = None,
    rfo: float = None,
    python_blend: bool = False,
    show: bool = False,
    start_frame_number: int = 0,
    output_video: str = None,
    # max_width: int = 8192,
    max_width: int = 9999,
    rotation_angle: int = 0,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda"),
    skip_final_video_save: bool = False,
    blend_mode: str = "laplacian",
):
    video_file_1 = os.path.join(dir_name, video_file_1)
    video_file_2 = os.path.join(dir_name, video_file_2)

    video_file_1 = os.path.join(dir_name, video_file_1)
    video_file_2 = os.path.join(dir_name, video_file_2)
    torch.float32
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
    else:
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
    else:
        if rfo or start_frame_number:
            cap_2.seek(frame_number=rfo + start_frame_number)

    v1_iter = iter(cap_1)
    v2_iter = iter(cap_2)

    # source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size).to(device)
    # source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size).to(device)

    source_tensor_1 = next(v1_iter)
    source_tensor_2 = next(v2_iter)

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
            destination_tensor_1 = remapper_1.forward(source_image=source_tensor_1).to(
                device
            )
            destination_tensor_2 = remapper_2.forward(source_image=source_tensor_2).to(
                device
            )

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
                if not python_blend:
                    blender = core.ImageBlender(
                        mode=(
                            core.ImageBlenderMode.Laplacian
                            if blend_mode == "laplacian"
                            else core.ImageBlenderMode.HardSeam
                        ),
                        levels=10,
                        seam=torch.from_numpy(seam_tensor),
                        xor_map=torch.from_numpy(xor_tensor),
                        lazy_init=True,
                        # lazy_init=False,
                        interpolation="bilinear",
                    )
                    blender.to(device)
                else:
                    blender = PtImageBlender(
                        images_info=[
                            BlendImageInfo(
                                width=cap_1.width,
                                height=cap_1.height,
                                xpos=remapper_1.xpos,
                                ypos=remapper_1.ypos,
                            ),
                            BlendImageInfo(
                                width=cap_2.width,
                                height=cap_2.height,
                                xpos=remapper_2.xpos,
                                ypos=remapper_2.ypos,
                            ),
                        ],
                        seam_mask=torch.from_numpy(seam_tensor).contiguous().to(device),
                        xor_mask=torch.from_numpy(xor_tensor).contiguous().to(device),
                        laplacian_blend=blend_mode == "laplacian",
                    )
                # blender.init()

            blended = blender.forward(
                image_1=destination_tensor_1,
                xy_pos_1=[remapper_1.xpos, remapper_1.ypos],
                image_2=destination_tensor_2,
                xy_pos_2=[remapper_2.xpos, remapper_2.ypos],
            )

            # show_image("blended", blended, wait=False)

            if output_video:
                video_dim_height, video_dim_width = get_dims_for_output_video(
                    height=blended.shape[-2],
                    width=blended.shape[-1],
                    max_width=max_width,
                )
                if video_out is None:
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
                    show_image("stitched", blended[i], wait=False)

            # source_tensor_1 = read_frame_batch(cap_1, batch_size=batch_size)
            # source_tensor_2 = read_frame_batch(cap_2, batch_size=batch_size)
            source_tensor_1 = next(v1_iter)
            source_tensor_2 = next(v2_iter)
            timer.tic()
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

    source_tensor_1 = cv2.imread(os.path.join(dir_name, f"left.png"))
    source_tensor_2 = cv2.imread(os.path.join(dir_name, f"right.png"))

    remap_info_1 = core.RemapImageInfo()
    remap_info_1.src_width = int(image_width(source_tensor_1))
    remap_info_1.src_height = int(image_height(source_tensor_1))
    remap_info_1.col_map = col_map_1
    remap_info_1.row_map = row_map_1

    remap_info_2 = core.RemapImageInfo()
    remap_info_2.src_width = int(image_width(source_tensor_2))
    remap_info_2.src_height = int(image_height(source_tensor_2))
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
        interpolation=interpolation,
    )
    return stitcher, [xpos_1, ypos_1], [xpos_2, ypos_2]


def gpu_index(want: int = 1):
    return min(torch.cuda.device_count() - 1, want)


def stitch_video(
    opts: object,
    video_file_1: str,
    video_file_2: str,
    dir_name: str,
    basename_1: str,
    basename_2: str,
    output_device: torch.device,
    interpolation: str = None,
    lfo: float = None,
    rfo: float = None,
    python_blend: bool = False,
    show: bool = False,
    start_frame_number: int = 0,
    output_video: str = None,
    max_width: int = 9999,
    rotation_angle: int = 0,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda", 0),
    skip_final_video_save: bool = False,
    queue_size: int = 1,
    remap_on_async_stream: bool = False,
    dtype: torch.dtype = torch.float16,
    perf_only: Optional[bool] = False,
):
    video_file_1 = os.path.join(dir_name, video_file_1)
    video_file_2 = os.path.join(dir_name, video_file_2)

    if lfo is None or rfo is None:
        lfo, rfo = synchronize_by_audio(video_file_1, video_file_2)

    cap_1 = VideoStreamReader(
        os.path.join(dir_name, video_file_1),
        type=opts.video_stream_decode_method,
        device=f"cuda:{gpu_index(want=1)}",
        batch_size=batch_size,
    )
    if not cap_1 or not cap_1.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_1}")
    else:
        if lfo or start_frame_number:
            cap_1.seek(frame_number=lfo + start_frame_number)

    cap_2 = VideoStreamReader(
        os.path.join(dir_name, video_file_2),
        type=opts.video_stream_decode_method,
        device=f"cuda:{gpu_index(want=2)}",
        batch_size=batch_size,
    )

    if not cap_2 or not cap_2.isOpened():
        raise AssertionError(f"Could not open video file: {video_file_2}")
    else:
        if rfo or start_frame_number:
            cap_2.seek(frame_number=rfo + start_frame_number)

    main_stream = torch.cuda.Stream(device=device)

    stitcher, xy_pos_1, xy_pos_2 = create_stitcher(
        dir_name=dir_name,
        batch_size=batch_size,
        dtype=dtype,
        mapping_basename_1=basename_1,
        mapping_basename_2=basename_2,
        device=device,
        blend_mode=opts.blend_mode,
        remap_on_async_stream=remap_on_async_stream,
        interpolation=interpolation,
    )

    v1_iter = iter(cap_1)
    v2_iter = iter(cap_2)

    v1_iter = CachedIterator(
        iterator=v1_iter,
        cache_size=queue_size,
        pre_callback_fn=lambda source_tensor: StreamTensorToDtype(
            tensor=StreamTensorToDevice(tensor=source_tensor, device=device),
            dtype=dtype,
        ),
    )
    v2_iter = CachedIterator(
        iterator=v2_iter,
        cache_size=queue_size,
        pre_callback_fn=lambda source_tensor: StreamTensorToDtype(
            tensor=StreamTensorToDevice(tensor=source_tensor, device=device),
            dtype=dtype,
        ),
    )

    def to_tensor(t, no_sync: bool = False):
        if isinstance(t, StreamTensor):
            if no_sync:
                return t.ref()
            return t.get()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        if t.device != device:
            t = t.to(device, non_blocking=False)
        if t.dtype == torch.uint8:
            t = t.to(dtype, non_blocking=False)
        return t

    with cuda_stream_scope(main_stream):
        stitcher.to(device)

        video_out = None

        stitch_timer = Timer()
        io_timer = Timer()
        get_timer = Timer()
        all_timer = Timer()
        batch_count = 0
        frame_id = start_frame_number
        frame_ids = list()
        for i in range(batch_size):
            frame_ids.append(i)
        frame_ids = torch.tensor(frame_ids, dtype=torch.int64, device=device)
        frame_ids = frame_ids + frame_id
        try:

            stitched_frames = []

            while True:
                io_timer.tic()
                source_tensor_1 = next(v1_iter)
                source_tensor_2 = next(v2_iter)
                io_timer.toc()

                get_timer.tic()
                sinfo_1 = core.StitchImageInfo()
                sinfo_1.image = to_tensor(source_tensor_1, no_sync=True)
                sinfo_1.xy_pos = xy_pos_1

                sinfo_2 = core.StitchImageInfo()
                sinfo_2.image = to_tensor(source_tensor_2, no_sync=True)
                sinfo_2.xy_pos = xy_pos_2
                get_timer.toc()

                stitch_timer.tic()
                blended_stream_tensor = stitcher.forward(inputs=[sinfo_1, sinfo_2])
                blended_stream_tensor = StreamCheckpoint(
                    tensor=blended_stream_tensor,
                    stream=main_stream,
                    owns_stream=False,
                    verbose=False,
                )
                blended = blended_stream_tensor

                if not output_video and perf_only:
                    # We will lose frames here, but just in order to give
                    # the GPU a little time
                    stitched_frames.append(blended)
                    if len(stitched_frames) >= queue_size + 2:
                        blended = stitched_frames[-1]
                        del stitched_frames[-1]
                        blended = blended.get()
                    stitch_timer.toc()
                elif not output_video:
                    main_stream.synchronize()
                    stitch_timer.toc()
                else:
                    main_stream.synchronize()
                    stitch_timer.toc()
                    video_dim_height, video_dim_width = get_dims_for_output_video(
                        height=blended.shape[-2],
                        width=blended.shape[-1],
                        max_width=max_width,
                    )
                    if video_out is None:
                        fps = cap_1.fps
                        video_out = VideoOutput(
                            name="StitchedOutput",
                            args=None,
                            output_video_path=output_video,
                            output_frame_width=video_dim_width,
                            output_frame_height=video_dim_height,
                            fps=fps,
                            device=output_device,
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
                        # main_stream.synchronize()
                        video_out.append(
                            ImageProcData(
                                frame_id=frame_ids,
                                img=my_blended,
                                current_box=None,
                            )
                        )
                        frame_ids = frame_ids + batch_size
                        # for i in range(len(my_blended)):
                        #     video_out.append(
                        #         ImageProcData(
                        #             frame_id=frame_ids,
                        #             img=my_blended[i],
                        #             current_box=None,
                        #         )
                        #     )
                        #     frame_ids += 1
                    del my_blended

                if batch_count:
                    all_timer.toc()

                batch_count += 1

                if batch_count % 50 == 0:
                    print(
                        "\nStitching: {:.2f} fps".format(
                            batch_size * 1.0 / max(1e-5, stitch_timer.average_time)
                        )
                    )
                    print(
                        "IO:        {:.2f} fps".format(
                            batch_size * 1.0 / max(1e-5, io_timer.average_time)
                        )
                    )
                    print(
                        "get():     {:.2f} fps".format(
                            batch_size * 1.0 / max(1e-5, get_timer.average_time)
                        )
                    )
                    print(
                        "Overall:   {:.2f} fps".format(
                            batch_size * 1.0 / max(1e-5, all_timer.average_time)
                        )
                    )
                    if True and batch_count % 50 == 0:
                        stitch_timer = Timer()
                        io_timer = Timer()
                        get_timer = Timer()
                        all_timer = Timer()

                if show:
                    blended = to_tensor(blended)
                    for i in range(len(blended)):
                        show_image("stitched", blended[i], wait=False)
                all_timer.tic()

        except StopIteration:
            print("Done.")
        finally:
            if video_out is not None:
                if isinstance(video_out, VideoStreamWriter):
                    video_out.flush()
                    video_out.close()
                else:
                    video_out.stop()


#
# Combined FPS= XY/(X+Y)
#


def main(args):
    opts = copy_opts(src=args, dest=argparse.Namespace(), parser=hm_opts.parser())
    gpu_allocator = GpuAllocator(gpus=None)
    if not args.video_dir and args.game_id:
        args.video_dir = os.path.join(os.environ["HOME"], "Videos", args.game_id)
    video_gpu = torch.device("cuda", gpu_allocator.allocate_modern())
    fast_gpu = torch.device("cuda", gpu_allocator.allocate_fast())
    with torch.no_grad():
        stitch_video(
            # blend_video(
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
            output_video=args.output_file,
            output_device=video_gpu,
            rotation_angle=args.rotation_angle,
            batch_size=args.batch_size,
            skip_final_video_save=args.skip_final_video_save,
            queue_size=args.queue_size,
            remap_on_async_stream=False,
            device=fast_gpu,
            dtype=torch.float16 if args.fp16 else torch.float,
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
    print("Done.")

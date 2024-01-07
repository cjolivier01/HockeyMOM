from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from numba import njit

import time
import os
import cv2
import numpy as np
import traceback
import multiprocessing
import queue
import PIL
from typing import List, Tuple

from pathlib import Path
import collections

import torch
import torchvision as tv

from torchvision.transforms import functional as F
from PIL import Image

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.utils.utils import create_queue
from hmlib.utils.image import ImageHorizontalGaussianDistribution
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.tracker.multitracker import torch_device

from hmlib.utils.box_functions import (
    center,
    width,
    height,
)


def image_width(img):
    if isinstance(img, torch.Tensor):
        assert img.shape[-1] == 3
        if len(img.shape) == 4:
            return img.shape[2]
        return img.shape[1]
    elif isinstance(img, PIL.Image.Image):
        return img.size[0]
    assert img.shape[-1] == 3
    if len(img.shape) == 4:
        return img.shape[2]
    return img.shape[1]


def image_height(img):
    if isinstance(img, torch.Tensor):
        assert img.shape[-1] == 3
        if len(img.shape) == 4:
            return img.shape[1]
        return img.shape[0]
    elif isinstance(img, PIL.Image.Image):
        return img.size[1]
    assert img.shape[-1] == 3
    if len(img.shape) == 4:
        return img.shape[1]
    return img.shape[0]


_ANGLE = 0.0


def rotate_image(img, angle: float, rotation_point: List[int]):
    rotation_point = [int(i) for i in rotation_point]
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            # H, W, C -> C, W, H
            img = img.permute(0, 3, 2, 1)
            angle = -angle
            img = F.rotate(
                img=img,
                angle=angle,
                center=(rotation_point[1], rotation_point[0]),
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=None,
            )
            # W, H, C -> C, H, W
            img = img.permute(0, 3, 2, 1)
        else:
            # H, W, C -> C, W, H
            img = img.permute(2, 1, 0)
            angle = -angle
            img = F.rotate(
                img=img,
                angle=angle,
                center=(rotation_point[1], rotation_point[0]),
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=None,
            )
            # W, H, C -> C, H, W
            img = img.permute(2, 1, 0)
    elif isinstance(img, PIL.Image.Image):
        img = img.rotate(
            angle, resample=Image.BICUBIC, center=(rotation_point[0], rotation_point[1])
        )
    else:
        rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
        img = cv2.warpAffine(
            img, rotation_matrix, (image_width(img), image_height(img))
        )
    return img


def crop_image(img, left, top, right, bottom):
    if isinstance(img, PIL.Image.Image):
        return img.crop((left, top, right, bottom))
    return img[top : bottom + 1, left : right + 1, 0:3]


def resize_image(
    img, new_width: int, new_height: int, mode=tv.transforms.InterpolationMode.BILINEAR
):
    w = int(new_width)
    h = int(new_height)
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            # Probably doesn't work
            permuted = img.shape[-1] == 3 or img.shape[-1] == 4
            if permuted:
                # H, W, C -> C, W, H
                img = img.permute(0, 3, 2, 1)
            assert img.shape[1] == 2 or img.shape[1] == 3
            img = F.resize(
                img=img,
                size=(w, h),
                interpolation=mode,
            )
            if permuted:
                # C, W, H -> H, W, C
                img = img.permute(0, 3, 2, 1)
        else:
            permuted = img.shape[-1] == 3 or img.shape[-1] == 4
            if permuted:
                # H, W, C -> C, W, H
                img = img.permute(2, 1, 0)
            img = F.resize(
                img=img,
                size=(w, h),
                interpolation=mode,
            )
            if permuted:
                # C, W, H -> H, W, C
                img = img.permute(2, 1, 0)
        return img
    elif isinstance(img, PIL.Image.Image):
        return img.resize((w, h))
    return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)


def paste_watermark_at_position(
    dest_image, watermark_rgb_channels, watermark_mask, x: int, y: int
):
    watermark_height = image_height(watermark_rgb_channels)
    watermark_width = image_width(watermark_rgb_channels)
    dest_image[y : y + watermark_height, x : x + watermark_width] = dest_image[
        y : y + watermark_height, x : x + watermark_width
    ] * (1 - watermark_mask / 255.0) + watermark_rgb_channels * (watermark_mask / 255.0)
    return dest_image


class ImageProcData:
    def __init__(self, frame_id: int, img, current_box: torch.Tensor):
        self.frame_id = frame_id
        self.img = img
        if current_box is None:
            self.current_box = torch.tensor(
                (0, 0, img.shape[-1], img.shape[-2]),
                dtype=torch.int64,
                device=img.device,
            )
        else:
            self.current_box = current_box.clone()

    def dump(self):
        print(f"frame_id={self.frame_id}, current_box={self.current_box}")

    def dump(self):
        print(f"frame_id={self.frame_id}, current_box={self.current_box}")


def _to_float(tensor: torch.Tensor):
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32) / 255.0
    return tensor


def _to_uint8(tensor: torch.Tensor):
    if tensor.dtype == torch.float32:
        return (tensor * 255).clamp(min=0, max=255.0).to(torch.uint8)
    return tensor


def image_wh(image: torch.Tensor):
    if image.shape[-1] in [3, 4]:
        return torch.tensor(
            [image.shape[-2], image.shape[-3]], dtype=torch.float32, device=image.device
        )
    assert image.shape[1] in [3, 4]
    return torch.tensor(
        [image.shape[-1], image.shape[-2]], dtype=torch.float32, device=image.device
    )


class VideoOutput:
    def __init__(
        self,
        args,
        output_video_path: str,
        output_frame_width: int,
        output_frame_height: int,
        fps: float,
        fourcc="XVID",
        # fourcc="HEVC",
        # fourcc="X264",
        # fourcc="H264",
        # fourcc = "HFYU",
        save_frame_dir: str = None,
        use_fork: bool = False,
        start: bool = True,
        max_queue_backlog: int = 25,
        watermark_image_path: str = None,
        device: str = "cuda:1",
        name: str = "",
    ):
        self._args = args
        self._device = device
        self._name = name
        self._fps = fps
        self._output_frame_width = output_frame_width
        self._output_frame_height = output_frame_height
        self._output_aspect_ratio = self._output_frame_width / self._output_frame_height
        self._output_frame_width_int = int(self._output_frame_width)
        self._output_frame_height_int = int(self._output_frame_height)
        self._use_fork = use_fork
        self._max_queue_backlog = max_queue_backlog
        self._imgproc_thread = None
        self._imgproc_queue = create_queue(mp=use_fork)
        self._imgproc_thread = None
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._output_video = None
        self._fourcc = fourcc
        self._horizontal_image_gaussian_distribution = None
        self._zero_f32 = torch.tensor(0, dtype=torch.float32, device=device)
        self._zero_uint8 = torch.tensor(0, dtype=torch.uint8, device=device)

        self._send_to_video_out_timer = Timer()

        if watermark_image_path:
            self.watermark = cv2.imread(
                watermark_image_path,
                cv2.IMREAD_UNCHANGED,
            )
            self.watermark_height = self.watermark.shape[0]
            self.watermark_width = self.watermark.shape[1]
            self.watermark_rgb_channels = self.watermark[:, :, :3]
            self.watermark_alpha_channel = self.watermark[:, :, 3]
            self.watermark_mask = cv2.merge(
                [
                    self.watermark_alpha_channel,
                    self.watermark_alpha_channel,
                    self.watermark_alpha_channel,
                ]
            )

            if self._device is not None:
                self.watermark_rgb_channels = torch.from_numpy(
                    self.watermark_rgb_channels
                ).to(self._device)
                self.watermark_alpha_channel = torch.from_numpy(
                    self.watermark_alpha_channel
                ).to(self._device)
                self.watermark_mask = torch.from_numpy(self.watermark_mask).to(
                    self._device
                )
        else:
            self.watermark = None

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        if start:
            self.start()

    def start(self):
        if self._use_fork:
            self._child_pid = os.fork()
            if not self._child_pid:
                self.final_image_processing()
        else:
            assert (
                self._imgproc_thread is None
                and "Video output thread was already started"
            )
            self._imgproc_thread = Thread(
                target=self._final_image_processing_wrapper,
                name="VideoOutput",
            )
            self._imgproc_thread.start()

    def stop(self):
        self._imgproc_queue.put(None)
        self._imgproc_thread.join()
        self._imgproc_thread = None

    def append(self, img_proc_data: ImageProcData):
        with TimeTracker(
            "Send to Video-Out queue", self._send_to_video_out_timer, print_interval=50
        ):
            while self._imgproc_queue.qsize() > self._max_queue_backlog:
                # print(f"Video out queue too large: {self._imgproc_queue.qsize()}")
                time.sleep(0.01)
            self._imgproc_queue.put(img_proc_data)

    def _final_image_processing_wrapper(self):
        try:
            self._final_image_processing()
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise

    def _get_gaussian(self, image_width: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = (
                ImageHorizontalGaussianDistribution(image_width)
            )
        return self._horizontal_image_gaussian_distribution

    def has_args(self):
        return self._args is not None

    def crop_working_image_width(
        self, image: torch.Tensor, current_box: torch.Tensor, scale: torch.Tensor
    ):
        """
        We try to only retain enough image to supply an arbitrary rotation
        about the center of the given bounding box with pixels,
        and offset that bounding box to be relative to the new (hopefully smaller)
        image
        """
        bbox_w = width(current_box)
        bbox_h = height(current_box)
        bbox_c = center(current_box)
        assert image.ndim == 3  # No batch dimension
        # make sure we're the expected (albeit arbitrary) channels first
        assert image.shape[-1] in [3, 4]
        img_wh = image_wh(image)
        min_width_per_side = torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2
        # min_width_per_side = bbox_w * scale / 2
        clip_left = torch.max(self._zero_uint8, bbox_c[0] - min_width_per_side)
        clip_right = torch.min(img_wh[0] - 1, bbox_c[0] + min_width_per_side)
        image = image[:, int(clip_left) : int(clip_right), :]
        current_box[0] -= clip_left.to(current_box.device)
        current_box[2] -= clip_left.to(current_box.device)

        # Adjust our output frame size if necessary
        if (
            not self._args.crop_output_image
            and image.shape[1] != self._output_frame_width_int
        ):
            self._output_frame_width = torch.tensor(
                image.shape[1], dtype=img_wh.dtype, device=img_wh.device
            )
            self._output_frame_height = torch.tensor(
                image.shape[0], dtype=img_wh.dtype, device=img_wh.device
            )
            self._output_aspect_ratio = (
                self._output_frame_width / self._output_frame_height
            )
            self._output_frame_width_int = int(self._output_frame_width)
            self._output_frame_height_int = int(self._output_frame_height)

        return image, current_box

    def _final_image_processing(self):
        print("VideoOutput thread started.")
        plot_interias = False
        show_image_interval = 1
        skip_frames_before_show = 0
        timer = Timer()
        tv_resizer = None
        # The timer that reocrds the overall throughput
        final_all_timer = None
        if self._output_video_path and self._output_video is None:
            # is_cuda = str(self._device).startswith("cuda")
            # I think it crashes if the size is off by even one pixed between frames?
            is_cuda = False
            fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
            if not is_cuda:
                # def __init__(self, filename: str, apiPreference: int, fourcc: int, fps: float, frameSize: cv2.typing.Size, params: _typing.Sequence[int]) -> None: ...
                # params = Sequence()
                self._output_video = cv2.VideoWriter(
                    filename=self._output_video_path,
                    # apiPreference=cv2.CAP_FFMPEG,
                    # apiPreference=cv2.CAP_GSTREAMER,
                    fourcc=fourcc,
                    fps=self._fps,
                    frameSize=(
                        int(self._output_frame_width),
                        int(self._output_frame_height),
                    ),
                    # params=[
                    #     cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
                    #     #cv2.VIDEOWRITER_PROP_HW_DEVICE, 1,
                    # ],
                )
                assert self._output_video.isOpened()
                self._output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)
            else:
                self._output_video = cv2.cudacodec.VideoWriter(
                    filename=self._output_video_path,
                    fourcc=fourcc,
                    fps=self._fps,
                    frameSize=(
                        int(self._output_frame_width),
                        int(self._output_frame_height),
                    ),
                )

        seen_frames = set()
        while True:
            imgproc_data = self._imgproc_queue.get()
            if imgproc_data is None:
                if self._output_video is not None:
                    self._output_video.release()
                break
            frame_id = imgproc_data.frame_id
            assert frame_id not in seen_frames
            seen_frames.add(frame_id)

            timer.tic()

            current_box = imgproc_data.current_box
            online_im = imgproc_data.img

            if self._device is not None:
                if not isinstance(online_im, torch.Tensor):
                    if online_im.shape[-1] not in [3, 4]:
                        online_im = online_im.transpose(1, 2, 0)
                    online_im = torch.from_numpy(online_im)
                if online_im.device != self._device:
                    online_im = online_im.to(self._device, non_blocking=True)

            src_image_width = image_width(online_im)
            src_image_height = image_height(online_im)

            #
            # Perspective rotation
            #
            if self.has_args() and self._args.fixed_edge_rotation:
                online_im = _to_float(online_im)
                # start = time.time()
                rotation_point = [int(i) for i in center(current_box)]
                width_center = src_image_width / 2
                if rotation_point[0] < width_center:
                    mult = -1
                else:
                    mult = 1

                gaussian = 1 - self._get_gaussian(
                    src_image_width
                ).get_gaussian_y_from_image_x_position(rotation_point[0], wide=True)

                fixed_edge_rotation_angle = self._args.fixed_edge_rotation_angle
                if isinstance(fixed_edge_rotation_angle, (list, tuple)):
                    assert len(fixed_edge_rotation_angle) == 2
                    if rotation_point[0] < src_image_width // 2:
                        fixed_edge_rotation_angle = int(
                            self._args.fixed_edge_rotation_angle[0]
                        )
                    else:
                        fixed_edge_rotation_angle = int(
                            self._args.fixed_edge_rotation_angle[1]
                        )

                # print(f"gaussian={gaussian}")
                angle = fixed_edge_rotation_angle - fixed_edge_rotation_angle * gaussian
                angle *= mult

                # BEGIN PERFORMANCE HACK
                #
                # Chop off edges of image that won't be visible after a final crop
                # before we rotate in order to reduce the computation necessary
                # for the rotation (as well as other subsequent operations)
                #
                if True and self._args.crop_output_image:
                    online_im, current_box = self.crop_working_image_width(
                        image=online_im, current_box=current_box, scale=1.2
                    )
                    src_image_width = image_width(online_im)
                    src_image_height = image_height(online_im)
                    rotation_point = center(current_box)
                #
                # END PERFORMANCE HACK
                #

                # print(f"angle={angle}")
                online_im = rotate_image(
                    img=online_im, angle=angle, rotation_point=rotation_point
                )
                # duration = time.time() - start
                # print(f"rotate image took {duration} seconds")

            #
            # Crop to output video frame image
            #
            if self.has_args() and self._args.crop_output_image:
                online_im = _to_float(online_im)
                # assert torch.isclose(
                #     aspect_ratio(current_box), self._output_aspect_ratio,
                # )
                # print(f"crop ar={aspect_ratio(current_box)}")
                intbox = [int(i) for i in current_box]
                x1 = intbox[0]
                y1 = intbox[1]
                y2 = intbox[3]
                x2 = int(x1 + int(float(y2 - y1) * self._output_aspect_ratio))
                assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
                if y1 >= src_image_height or y2 >= src_image_height:
                    print(
                        f"y1 ({y1}) or y2 ({y2}) is too large, should be < {src_image_height}"
                    )
                    # assert y1 < online_im.shape[0] and y2 < online_im.shape[0]
                    y1 = min(y1, src_image_height)
                    y2 = min(y2, src_image_height)
                if x1 >= src_image_width or x2 >= src_image_width:
                    print(
                        f"x1 {x1} or x2 {x2} is too large, should be < {src_image_width}"
                    )
                    # assert x1 < online_im.shape[1] and x2 < online_im.shape[1]
                    x1 = min(x1, src_image_width)
                    x2 = min(x2, src_image_width)

                online_im = crop_image(online_im, x1, y1, x2, y2)
                if image_height(online_im) != int(
                    self._output_frame_height
                ) or image_width(online_im) != int(self._output_frame_width):
                    online_im = resize_image(
                        img=online_im,
                        new_width=self._output_frame_width,
                        new_height=self._output_frame_height,
                    )
                assert image_height(online_im) == self._output_frame_height_int
                assert image_width(online_im) == self._output_frame_width_int

            # Numpy array after here
            if isinstance(online_im, PIL.Image.Image):
                online_im = np.array(online_im)

            #
            # Watermark
            #
            online_im = _to_uint8(online_im)
            if self.has_args() and self._args.use_watermark:
                y = int(online_im.shape[0] - self.watermark_height)
                x = int(
                    online_im.shape[1]
                    - self.watermark_width
                    - self.watermark_width / 10
                )
                online_im = paste_watermark_at_position(
                    online_im,
                    watermark_rgb_channels=self.watermark_rgb_channels,
                    watermark_mask=self.watermark_mask,
                    x=x,
                    y=y,
                )

            # Make a numpy image array
            if isinstance(online_im, torch.Tensor):
                online_im = online_im.cpu().detach().numpy()

            #
            # Frame Number
            #
            if self.has_args() and self._args.plot_frame_number:
                online_im = vis.plot_frame_number(
                    online_im,
                    frame_id=frame_id,
                )

            # Output (and maybe show) the final image
            if (
                self.has_args()
                and self._args.show_image
                and imgproc_data.frame_id >= skip_frames_before_show
            ):
                if imgproc_data.frame_id % show_image_interval == 0:
                    cv2.imshow("online_im", online_im)
                    cv2.waitKey(1)

            if plot_interias:
                vis.plot_kmeans_intertias(hockey_mom=self._hockey_mom)

            assert int(self._output_frame_width) == online_im.shape[-2]
            assert int(self._output_frame_height) == online_im.shape[-3]
            if self._output_video is not None:
                self._output_video.write(online_im)
            if self._save_frame_dir:
                # frame_id should start with 1
                assert imgproc_data.frame_id
                cv2.imwrite(
                    os.path.join(
                        self._save_frame_dir,
                        "frame_{:06d}.png".format(int(imgproc_data.frame_id) - 1),
                    ),
                    online_im,
                )
            timer.toc()

            if imgproc_data.frame_id % 20 == 0:
                logger.info(
                    "Image Post-Processing {} frame {} ({:.2f} fps)".format(self._name,
                        imgproc_data.frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )
                # timer = Timer()

            if True:
                # Overall FPS
                if final_all_timer is None:
                    final_all_timer = Timer()
                    final_all_timer.tic()
                else:
                    final_all_timer.toc()
                    final_all_timer.tic()

                if imgproc_data.frame_id % 100 == 0:
                    logger.info(
                        "*** Overall performance, frame {} ({:.2f} fps)  -- open files count: {}".format(
                            imgproc_data.frame_id,
                            1.0 / max(1e-5, final_all_timer.average_time),
                            get_open_files_count(),
                        )
                    )


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

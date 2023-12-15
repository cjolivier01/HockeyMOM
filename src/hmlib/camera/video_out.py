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

import torch
import torchvision as tv

from torchvision.transforms import functional as F
from PIL import Image

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.utils.utils import create_queue
from hmlib.utils.image import ImageHorizontalGaussianDistribution
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.tracker.multitracker import torch_device

from hmlib.utils.box_functions import (
    center,
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


def resize_image(img, new_width: int, new_height: int):
    w = int(new_width)
    h = int(new_height)
    if isinstance(img, torch.Tensor):
        # H, W, C -> C, W, H
        img = img.permute(2, 1, 0)
        img = F.resize(
            img=img,
            size=(w, h),
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
        )
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
        self.current_box = current_box.clone()

    def dump(self):
        print(f"frame_id={self.frame_id}, current_box={self.current_box}")

    def dump(self):
        print(f"frame_id={self.frame_id}, current_box={self.current_box}")


class VideoOutput:
    def __init__(
        self,
        args,
        output_video_path: str,
        output_frame_width: int,
        output_frame_height: int,
        fps: float,
        #fourcc="XVID",
        #fourcc="HEVC",
        fourcc="X264",
        #fourcc = "HFYU",
        save_frame_dir: str = None,
        use_fork: bool = False,
        start: bool = True,
        max_queue_backlog: int = 25,
        watermark_image_path: str = None,
        device: str = None,
        # device: str = "cuda:1",
    ):
        self._args = args
        self._device = device
        self._fps = fps
        self._output_frame_width = output_frame_width
        self._output_frame_height = output_frame_height
        self._output_aspect_ratio = self._output_frame_width / self._output_frame_height
        self._output_frame_width_int = int(self._output_frame_width.item())
        self._output_frame_height_int = int(self._output_frame_height.item())
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
        while self._imgproc_queue.qsize() > self._max_queue_backlog:
            time.sleep(0.001)
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
            fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
            self._output_video = cv2.VideoWriter(
                filename=self._output_video_path,
                fourcc=fourcc,
                fps=self._fps,
                frameSize=(
                    int(self._output_frame_width.item()),
                    int(self._output_frame_height.item()),
                ),
                isColor=True,
            )
            assert self._output_video.isOpened()
            self._output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)
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
                online_im = torch.from_numpy(online_im).to(self._device)

            src_image_width = image_width(online_im)
            src_image_height = image_height(online_im)

            #
            # Perspective rotation
            #
            if self._args.fixed_edge_rotation:
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
                # print(f"gaussian={gaussian}")
                angle = (
                    self._args.fixed_edge_rotation_angle
                    - self._args.fixed_edge_rotation_angle * gaussian
                )
                angle *= mult
                # print(f"angle={angle}")
                online_im = rotate_image(
                    img=online_im, angle=angle, rotation_point=rotation_point
                )
                # duration = time.time() - start
                # print(f"rotate image took {duration} seconds")

            #
            # Crop to output video frame image
            #
            if self._args.crop_output_image:
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
                    if (
                        False
                        and isinstance(online_im, torch.Tensor)
                        and online_im.device.type == "cuda"
                    ):
                        if tv_resizer is None:
                            tv_resizer = tv.transforms.Resize(
                                size=(
                                    int(self._output_frame_width),
                                    int(self._output_frame_height),
                                )
                            )
                        gpu_image = tv_resizer.forward(online_im)
                    else:
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
            if self._args.use_watermark:
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
            if self._args.plot_frame_number:
                online_im = vis.plot_frame_number(
                    online_im,
                    frame_id=frame_id,
                )

            # Output (and maybe show) the final image
            if (
                self._args.show_image
                and imgproc_data.frame_id >= skip_frames_before_show
            ):
                if imgproc_data.frame_id % show_image_interval == 0:
                    cv2.imshow("online_im", online_im)
                    cv2.waitKey(1)

            if plot_interias:
                vis.plot_kmeans_intertias(hockey_mom=self._hockey_mom)

            if self._output_video is not None:
                #self._output_video.write(online_im)
                pass
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
                    "Image Post-Processing frame {} ({:.2f} fps)".format(
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

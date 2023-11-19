from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from numba import njit

import time
import os
import cv2
import argparse
import numpy as np
import traceback
import multiprocessing
import queue

from pathlib import Path

import torch
import torchvision as tv

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.utils.utils import create_queue
from hmlib.utils.image import ImageHorizontalGaussianDistribution
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.tracker.multitracker import torch_device

from hmlib.utils.box_functions import (
    width,
    height,
    center,
)


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
        fourcc="XVID",
        # fourcc = "HFYU",
        output_image_dir: str = None,
        use_fork: bool = False,
        start: bool = True,
        max_queue_backlog: int = 25,
        watermark_image_path: str = None,
    ):
        self._args = args
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
        self._output_image_dir = output_image_dir
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
        else:
            self.watermark = None

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
            if imgproc_data.frame_id % 20 == 0:
                logger.info(
                    "Image Post-Processing frame {} ({:.2f} fps)".format(
                        imgproc_data.frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )
                timer = Timer()
            timer.tic()

            current_box = imgproc_data.current_box
            online_im = imgproc_data.img

            if self._args.fixed_edge_rotation:
                rotation_point = [int(i) for i in center(current_box)]
                width_center = online_im.shape[1] / 2
                if rotation_point[0] < width_center:
                    #     dist_from_center_pct = (width_center - rotation_point[0])/width_center
                    mult = -1
                else:
                    #     dist_from_center_pct = (rotation_point[0] - width_center)/width_center
                    mult = 1
                # angle = float(self._args.fixed_edge_rotation_angle)* dist_from_center_pct * mult

                gaussian = 1 - self._get_gaussian(
                    online_im.shape[1]
                ).get_gaussian_y_from_image_x_position(rotation_point[0], wide=True)
                # print(f"gaussian={gaussian}")
                angle = (
                    self._args.fixed_edge_rotation_angle
                    - self._args.fixed_edge_rotation_angle * gaussian
                )
                angle *= mult
                # print(f"angle={angle}")
                rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
                online_im = cv2.warpAffine(
                    online_im, rotation_matrix, (online_im.shape[1], online_im.shape[0])
                )

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
                if y1 >= online_im.shape[0] or y2 >= online_im.shape[0]:
                    print(
                        f"y1 ({y1}) or y2 ({y2}) is too large, should be < {online_im.shape[0]}"
                    )
                    # assert y1 < online_im.shape[0] and y2 < online_im.shape[0]
                    y1 = min(y1, online_im.shape[0])
                    y2 = min(y2, online_im.shape[0])
                if x1 >= online_im.shape[1] or x2 >= online_im.shape[1]:
                    print(
                        f"x1 {x1} or x2 {x2} is too large, should be < {online_im.shape[1]}"
                    )
                    # assert x1 < online_im.shape[1] and x2 < online_im.shape[1]
                    x1 = min(x1, online_im.shape[1])
                    x2 = min(x2, online_im.shape[1])

                if not self._args.fake_crop_output_image:
                    if self._args.use_cuda:
                        gpu_image = torch.Tensor(online_im)[
                            y1 : y2 + 1, x1 : x2 + 1, 0:3
                        ].to(self._device)
                        # gpu_image = torch.Tensor(online_im).to("cuda:1")
                        # gpu_image = gpu_image[y1:y2,x1:x2,0:3]
                        # gpu_image = cv2.cuda_GpuMat(online_im)
                        # gpu_image = cv2.cuda_GpuMat(gpu_image, (x1, y1, x2, y2))
                    else:
                        online_im = online_im[y1 : y2 + 1, x1 : x2 + 1, 0:3]
                if not self._args.fake_crop_output_image and (
                    online_im.shape[0] != self._output_frame_height
                    or online_im.shape[1] != self._output_frame_width
                ):
                    if self._args.use_cuda:
                        if tv_resizer is None:
                            tv_resizer = tv.transforms.Resize(
                                size=(
                                    int(self._output_frame_width),
                                    int(self._output_frame_height),
                                )
                            )
                        gpu_image = tv_resizer.forward(gpu_image)
                    else:
                        online_im = cv2.resize(
                            online_im,
                            dsize=(
                                int(self._output_frame_width),
                                int(self._output_frame_height),
                            ),
                            interpolation=cv2.INTER_CUBIC,
                        )
                assert online_im.shape[0] == self._output_frame_height_int
                assert online_im.shape[1] == self._output_frame_width_int
                if self._args.use_cuda:
                    # online_im = gpu_image.download()
                    online_im = np.array(gpu_image.cpu().numpy(), np.uint8)
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
                online_im[
                    y : y + self.watermark_height, x : x + self.watermark_width
                ] = online_im[
                    y : y + self.watermark_height, x : x + self.watermark_width
                ] * (
                    1 - self.watermark_mask / 255.0
                ) + self.watermark_rgb_channels * (
                    self.watermark_mask / 255.0
                )

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
                self._output_video.write(online_im)
            if self._output_image_dir:
                cv2.imwrite(
                    os.path.join(
                        self._output_image_dir,
                        "{:05d}.png".format(imgproc_data.frame_id),
                    ),
                    online_im,
                )
            timer.toc()

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
                        "*** Overall performance, frame {} ({:.2f} fps)".format(
                            imgproc_data.frame_id,
                            1.0 / max(1e-5, final_all_timer.average_time),
                        )
                    )


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

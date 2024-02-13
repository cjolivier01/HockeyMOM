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
from contextlib import contextmanager
import PIL
from typing import Dict, List, Tuple

import torch
import torchvision as tv

from torchvision.transforms import functional as F
from PIL import Image

from threading import Thread

from hmlib.tracking_utils import visualization as vis
from hmlib.utils.utils import create_queue
from hmlib.tracking_utils.visualization import get_complete_monitor_width
from hmlib.utils.image import ImageHorizontalGaussianDistribution, ImageColorScaler
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.utils.gpu import (
    get_gpu_with_highest_compute_capability,
    get_gpu_capabilities,
)
from hmlib.utils.image import (
    make_channels_last,
    image_width,
    image_height,
    resize_image,
    crop_image,
)
from .video_stream import VideoStreamWriter

from torchvision.io import write_video

from hmlib.utils.box_functions import (
    center,
    width,
    height,
)


@contextmanager
def optional_with(resource):
    """A context manager that works even if the resource is None."""
    if resource is None:
        # If the resource is None, yield nothing but still enter the with block
        yield None
    else:
        # If the resource is not None, use it as a normal context manager
        with resource as r:
            yield r


def get_best_codec(gpu_number: int, width: int, height: int):
    caps = get_gpu_capabilities()
    compute = float(caps[gpu_number]["compute_capability"])
    if (
        compute >= 7 and width <= 9900
    ):  # FIXME: I forget what the max is? 99-thousand-something
        return "hevc_nvenc", True
    elif compute >= 6 and width <= 4096:
        return "hevc_nvenc", True
    else:
        return "XVID", False


def make_showable_type(img: torch.Tensor, scale_elements: float = 255.0):
    if isinstance(img, torch.Tensor):
        if img.ndim == 2:
            # 2D grayscale
            img = img.unsqueeze(0).repeat(3, 1, 1)
        assert len(img.shape) == 3
        img = make_channels_last(img)
        if img.dtype in [torch.float16, torch.float32, torch.float64]:
            # max = torch.max(img)
            if scale_elements and scale_elements != 1:
                img = img * 255.0
            img = torch.clamp(img, min=0, max=255.0).to(torch.uint8)
        img = img.contiguous().cpu().numpy()
    return img


def make_visible_image(
    img, enable_resizing: bool = False, scale_elements: float = 255.0
):
    if not enable_resizing:
        if isinstance(img, torch.Tensor):
            img = make_showable_type(img, scale_elements)
        return img
    width = image_width(img)
    vis_w = get_complete_monitor_width()
    if vis_w and width and width > vis_w:
        height = image_height(img)
        ar = width / height
        new_w = vis_w * 0.7
        new_h = new_w / ar
        img = resize_image(img, new_width=int(new_w), new_height=int(new_h))
    return make_showable_type(img)


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


def paste_watermark_at_position(
    dest_image, watermark_rgb_channels, watermark_mask, x: int, y: int
):
    assert dest_image.ndim == 4
    watermark_height = image_height(watermark_rgb_channels)
    watermark_width = image_width(watermark_rgb_channels)
    dest_image[:, y : y + watermark_height, x : x + watermark_width] = dest_image[
        :, y : y + watermark_height, x : x + watermark_width
    ] * (1 - watermark_mask / 255.0) + watermark_rgb_channels * (watermark_mask / 255.0)
    return dest_image


class ImageProcData:
    def __init__(self, frame_id: int, img, current_box: torch.Tensor):
        self.frame_id = frame_id
        self.img = img
        if current_box is None:
            self.current_box = torch.tensor(
                (0, 0, image_width(img), image_height(img)),
                dtype=torch.int64,
                device=img.device,
            )
            if img.ndim == 4:
                # batched
                assert self.current_box.ndim == 1
                self.current_box = self.current_box.unsqueeze(0).repeat(img.size(0), 1)
        else:
            self.current_box = current_box.clone()
        if not isinstance(self.frame_id, torch.Tensor):
            assert img.ndim == 3 or img.size(0) == 1  # single image only
            self.frame_id = torch.tensor([self.frame_id], dtype=torch.int64)

    def dump(self):
        print(f"frame_id={self.frame_id}, current_box={self.current_box}")

    def dump(self):
        print(f"frame_id={self.frame_id}, current_box={self.current_box}")


def _to_float(
    tensor: torch.Tensor, apply_scale: bool = True, non_blocking: bool = False
):
    if tensor.dtype == torch.uint8:
        if apply_scale:
            return tensor.to(torch.float32, non_blocking=non_blocking) / 255.0
        else:
            return tensor.to(torch.float32, non_blocking=non_blocking)
    else:
        assert torch.is_floating_point(tensor)
    return tensor


def _to_uint8(
    tensor: torch.Tensor, apply_scale: bool = True, non_blocking: bool = False
):
    if not isinstance(tensor, torch.Tensor):
        assert tensor.dtype == np.uint8
        return tensor
    if tensor.dtype != torch.uint8:
        if apply_scale:
            assert torch.is_floating_point(tensor)
            return (
                (tensor * 255)
                .clamp(min=0, max=255.0)
                .to(torch.uint8, non_blocking=non_blocking)
            )
        else:
            return tensor.clamp(min=0, max=255.0).to(
                torch.uint8, non_blocking=non_blocking
            )
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
        fourcc="auto",
        save_frame_dir: str = None,
        use_fork: bool = False,
        start: bool = True,
        max_queue_backlog: int = 1,
        watermark_image_path: str = None,
        device: torch.device = None,
        name: str = "",
        simple_save: bool = False,
        skip_final_save: bool = False,
        image_channel_adjustment: List[float] = None,
        print_interval: int = 50,
    ):
        if device is not None:
            print(
                "Video output {output_frame_width}x{output_frame_height} using device: {device} ({output_video_path})"
            )
        self._args = args
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self._name = name
        self._simple_save = simple_save
        self._fps = fps
        self._skip_final_save = skip_final_save
        assert output_frame_width > 4 and output_frame_height > 4
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
        self._print_interval = print_interval
        self._output_video = None

        if fourcc == "auto":
            if self._device.type == "cuda":
                self._fourcc, is_gpu = get_best_codec(
                    device.index,
                    width=int(output_frame_width),
                    height=int(output_frame_height),
                )
                if not is_gpu:
                    print(f"Can't use GPU for output video {output_video_path}")
                    self._device = torch.device("cpu")
            else:
                self._fourcc = "XVID"
            print(
                f"Output video {self._name} {int(self._output_frame_width)}x{int(self._output_frame_height)} will use codec: {self._fourcc}"
            )
        else:
            self._fourcc = fourcc

        self._horizontal_image_gaussian_distribution = None
        self._zero_f32 = torch.tensor(0, dtype=torch.float32, device=device)
        self._zero_uint8 = torch.tensor(0, dtype=torch.uint8, device=device)

        self._send_to_video_out_timer = Timer()

        self._image_color_scaler = None
        if image_channel_adjustment:
            assert len(image_channel_adjustment) == 3
            self._image_color_scaler = ImageColorScaler(image_channel_adjustment)

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
            counter = 0
            while self._imgproc_queue.qsize() > self._max_queue_backlog:
                counter += 1
                if counter % 100 == 0:
                    print(f"Video out queue too large: {self._imgproc_queue.qsize()}")
                time.sleep(0.01)
            # torch.cuda.current_stream(img_proc_data.img.device).synchronize()
            self._imgproc_queue.put(img_proc_data)

    def _final_image_processing_wrapper(self):
        try:
            self._final_image_processing()
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise
        finally:
            if self._output_video is not None:
                if isinstance(self._output_video, VideoStreamWriter):
                    self._output_video.flush()
                    self._output_video.close()
                else:
                    self._output_video.release()
                self._output_video = None

    def _get_gaussian(self, image_width: int):
        if self._horizontal_image_gaussian_distribution is None:
            self._horizontal_image_gaussian_distribution = (
                ImageHorizontalGaussianDistribution(image_width)
            )
        return self._horizontal_image_gaussian_distribution

    def has_args(self):
        return self._args is not None

    def calculate_desired_bitrate(self, width: int, height: int):
        # 4K @ 55M
        desired_bit_rate_per_pixel = 55e6 / (3840 * 2160)
        desired_bit_rate = int(desired_bit_rate_per_pixel * width * height)
        print(
            f"Desired bit rate for output video ({int(width)} x {int(height)}): {desired_bit_rate//1000} kb/s"
        )
        return desired_bit_rate

    def crop_working_image_width(self, image: torch.Tensor, current_box: torch.Tensor):
        """
        We try to only retain enough image to supply an arbitrary rotation
        about the center of the given bounding box with pixels,
        and offset that bounding box to be relative to the new (hopefully smaller)
        image
        """
        bbox_w = width(current_box)
        bbox_h = height(current_box)
        bbox_c = center(current_box)
        assert bbox_w > 10  # Sanity
        assert bbox_h > 10  # Sanity
        # assert image.ndim == 3  # No batch dimension
        # make sure we're the expected (albeit arbitrary) channels first
        assert image.shape[-1] in [3, 4]
        img_wh = image_wh(image)
        min_width_per_side = torch.sqrt(torch.square(bbox_w) + torch.square(bbox_h)) / 2
        clip_left = torch.max(self._zero_uint8, bbox_c[0] - min_width_per_side)
        clip_right = torch.min(img_wh[0] - 1, bbox_c[0] + min_width_per_side)
        if image.ndim == 3:
            # no batch dimension
            image = image[:, int(clip_left) : int(clip_right), :]
        else:
            # with batch dimension
            image = image[:, :, int(clip_left) : int(clip_right), :]
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
        # The timer that reocrds the overall throughput
        final_all_timer = None
        if (
            self._output_video_path
            and self._output_video is None
            and not self._skip_final_save
        ):
            if "_nvenc" in self._fourcc:
                self._output_video = VideoStreamWriter(
                    filename=self._output_video_path,
                    fps=self._fps,
                    height=int(self._output_frame_height),
                    width=int(self._output_frame_width),
                    codec="hevc_nvenc",
                    device=self._device,
                    batch_size=1,
                )
                self._output_video.open()
            else:
                fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
                self._output_video = cv2.VideoWriter(
                    filename=self._output_video_path,
                    fourcc=fourcc,
                    fps=self._fps,
                    frameSize=(
                        int(self._output_frame_width),
                        int(self._output_frame_height),
                    ),
                )
                # self._output_video.set(cv2.CAP_PROP_BITRATE, 52000 * 1024)
                # self._output_video.set(cv2.CAP_PROP_BITRATE, 80000 * 1024)
                self._output_video.set(
                    cv2.CAP_PROP_BITRATE,
                    self.calculate_desired_bitrate(
                        width=self._output_frame_width, height=self._output_frame_height
                    ),
                )
            assert self._output_video.isOpened()

        cuda_stream = None

        batch_count = 0

        # seen_frames = set()
        while True:
            batch_count += 1
            imgproc_data = self._imgproc_queue.get()
            if imgproc_data is None:
                break
            # frame_id = imgproc_data.frame_id
            # assert frame_id not in seen_frames
            # seen_frames.add(frame_id)

            timer.tic()

            # torch.cuda.synchronize()

            current_box = imgproc_data.current_box
            online_im = imgproc_data.img
            frame_id = imgproc_data.frame_id

            # torch.cuda.synchronize()

            if isinstance(online_im, np.ndarray):
                online_im = torch.from_numpy(online_im)

            # assert online_im.device.type == "cpu" or online_im.device == self._device
            if online_im.device.type != "cpu" and self._device.type != "cpu":
                online_im = online_im.cpu()

            if online_im.ndim == 3:
                online_im = online_im.unsqueeze(0)
                current_box = current_box.unsqueeze(0)

            # batch_size = 1 if online_im.ndim != 4 else online_im.size(0)
            batch_size = online_im.size(0)

            # torch.cuda.synchronize()
            # if cuda_stream is None and (
            #     online_im.device.type == "cuda"
            #     or self._device.type == "cuda"
            #     or "nvenc" in self._fourcc
            # ):
            #     cuda_stream = torch.cuda.Stream(device=self._device)

            with optional_with(
                torch.cuda.stream(cuda_stream) if cuda_stream is not None else None
            ):
                if self._device is not None and (
                    not self._simple_save or "nvenc" in self._fourcc
                ):
                    if not isinstance(online_im, torch.Tensor):
                        # if online_im.shape[-1] not in [3, 4]:
                        #     online_im = online_im.transpose(1, 2, 0)
                        online_im = torch.from_numpy(online_im)
                    if online_im.ndim == 4:
                        # assert online_im.shape[0] == 1  # batch size of 1 here only atm
                        # online_im = online_im.squeeze(0)
                        pass
                    online_im = make_channels_last(online_im)
                    if str(online_im.device) != str(self._device):
                        online_im = online_im.to(self._device, non_blocking=True)

                src_image_width = image_width(online_im)
                src_image_height = image_height(online_im)

                #
                # Perspective rotation
                #
                if self.has_args() and self._args.fixed_edge_rotation:
                    online_im = _to_float(online_im, non_blocking=True)
                    rotated_images = []
                    current_boxes = []
                    for img, bbox in zip(online_im, current_box):
                        # start = time.time()
                        rotation_point = [int(i) for i in center(bbox)]
                        width_center = src_image_width / 2
                        if rotation_point[0] < width_center:
                            mult = -1
                        else:
                            mult = 1

                        gaussian = 1 - self._get_gaussian(
                            src_image_width
                        ).get_gaussian_y_from_image_x_position(
                            rotation_point[0], wide=True
                        )

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
                        angle = (
                            fixed_edge_rotation_angle
                            - fixed_edge_rotation_angle * gaussian
                        )
                        angle *= mult

                        # BEGIN PERFORMANCE HACK
                        #
                        # Chop off edges of image that won't be visible after a final crop
                        # before we rotate in order to reduce the computation necessary
                        # for the rotation (as well as other subsequent operations)
                        #
                        if True and self._args.crop_output_image:
                            img, bbox = self.crop_working_image_width(
                                image=img, current_box=bbox
                            )
                            src_image_width = image_width(img)
                            src_image_height = image_height(img)
                            rotation_point = center(bbox)
                        #
                        # END PERFORMANCE HACK
                        #

                        # print(f"angle={angle}")
                        img = rotate_image(
                            img=img,
                            angle=angle,
                            rotation_point=rotation_point,
                        )
                        rotated_images.append(img)
                        current_boxes.append(bbox)
                    # duration = time.time() - start
                    # print(f"rotate image took {duration} seconds")
                    online_im = torch.stack(rotated_images)
                    current_box = torch.stack(current_boxes)

                #
                # Crop to output video frame image
                #
                if self.has_args() and self._args.crop_output_image:
                    online_im = _to_float(online_im, non_blocking=True)
                    # assert torch.isclose(
                    #     aspect_ratio(current_box), self._output_aspect_ratio,
                    # )
                    # print(f"crop ar={aspect_ratio(current_box)}")
                    cropped_images = []
                    for img, bbox in zip(online_im, current_box):
                        intbox = [int(i) for i in bbox]
                        x1 = intbox[0]
                        y1 = intbox[1]
                        y2 = intbox[3]
                        x2 = int(x1 + int(float(y2 - y1) * self._output_aspect_ratio))
                        assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
                        if y1 >= src_image_height or y2 >= src_image_height:
                            print(
                                f"y1 ({y1}) or y2 ({y2}) is too large, should be < {src_image_height}"
                            )
                            # assert y1 < img.shape[0] and y2 < img.shape[0]
                            y1 = min(y1, src_image_height)
                            y2 = min(y2, src_image_height)
                        if x1 >= src_image_width or x2 >= src_image_width:
                            print(
                                f"x1 {x1} or x2 {x2} is too large, should be < {src_image_width}"
                            )
                            # assert x1 < img.shape[1] and x2 < img.shape[1]
                            x1 = min(x1, src_image_width)
                            x2 = min(x2, src_image_width)

                        img = crop_image(img, x1, y1, x2, y2)
                        if image_height(img) != int(
                            self._output_frame_height
                        ) or image_width(img) != int(self._output_frame_width):
                            img = resize_image(
                                img=img,
                                new_width=self._output_frame_width,
                                new_height=self._output_frame_height,
                            )
                        assert image_height(img) == self._output_frame_height_int
                        assert image_width(img) == self._output_frame_width_int
                        cropped_images.append(img)
                    online_im = torch.stack(cropped_images)

                # Numpy array after here
                if isinstance(online_im, PIL.Image.Image):
                    online_im = np.array(online_im)

                #
                # Watermark
                #
                online_im = _to_uint8(online_im, non_blocking=True)
                if self.has_args() and self._args.use_watermark:
                    y = int(image_height(online_im) - self.watermark_height)
                    x = int(
                        image_width(online_im)
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

                if self._image_color_scaler is not None:
                    online_im = self._image_color_scaler.maybe_scale_image_colors(
                        image=online_im
                    )

                if not isinstance(self._output_video, VideoStreamWriter):
                    if isinstance(online_im, torch.Tensor) and (
                        # If we're actually going to do something with it
                        not self._skip_final_save
                        and (
                            (self.has_args() and self._args.plot_frame_number)
                            or self._output_video is not None
                            or self._save_frame_dir
                        )
                    ):
                        online_im = online_im.detach().contiguous().cpu().numpy()

                #
                # Frame Number
                #
                if self.has_args() and self._args.plot_frame_number:
                    prev_device = online_im.device
                    if cuda_stream is not None:
                        cuda_stream.synchronize()
                    online_im = vis.plot_frame_number(
                        online_im,
                        frame_id=frame_id,
                    )
                    online_im = torch.from_numpy(online_im).to(
                        prev_device, non_blocking=True
                    )

                # if plot_interias:
                #     vis.plot_kmeans_intertias(hockey_mom=self._hockey_mom)

                # Output (and maybe show) the final image
                if (
                    self.has_args()
                    and self._args.show_image
                    and imgproc_data.frame_id >= skip_frames_before_show
                ):
                    if imgproc_data.frame_id % show_image_interval == 0:
                        if cuda_stream is not None:
                            cuda_stream.synchronize()
                        show_img = online_im
                        if show_img.ndim == 3:
                            show_img = show_img.unsqueeze(0)
                        for s_img in show_img:
                            cv2.imshow("online_im", make_visible_image(s_img))
                            cv2.waitKey(1)

                # Synchronzie at the end whether we are saving or not, or else perf numbers aren't real
                if cuda_stream is not None:
                    cuda_stream.synchronize()

                # torch.cuda.synchronize()

                assert int(self._output_frame_width) == online_im.shape[-2]
                assert int(self._output_frame_height) == online_im.shape[-3]
                if self._output_video is not None and not self._skip_final_save:
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

                if self._print_interval and batch_count % self._print_interval == 0:
                    logger.info(
                        "Image Post-Processing {} frame {} ({:.2f} fps)".format(
                            self._name,
                            imgproc_data.frame_id[0],
                            batch_size * 1.0 / max(1e-5, timer.average_time),
                        )
                    )
                    timer = Timer()

                # if frame_id > 300:
                #     print("DONE AT LEAST FOR WRITER")
                #     break

                if True:
                    # Overall FPS
                    if final_all_timer is None:
                        final_all_timer = Timer()
                        final_all_timer.tic()
                    else:
                        final_all_timer.toc()
                        final_all_timer.tic()

                    if (
                        self._print_interval
                        and batch_count % (self._print_interval * 4) == 0
                    ):
                        logger.info(
                            "*** Overall performance, frame {} ({:.2f} fps)  -- open files count: {}".format(
                                imgproc_data.frame_id[0],
                                batch_size
                                * 1.0
                                / max(1e-5, final_all_timer.average_time),
                                get_open_files_count(),
                            )
                        )


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

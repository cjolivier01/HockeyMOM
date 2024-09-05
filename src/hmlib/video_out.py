from __future__ import absolute_import, division, print_function

import os
import time
import traceback
from threading import Thread
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
import torchvision as tv
from PIL import Image
from torchvision.transforms import functional as F

from hmlib.config import get_nested_value
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.log import logger
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.ui.shower import Shower
from hmlib.utils.box_functions import center, height, width
from hmlib.utils.containers import IterableQueue, SidebandQueue, create_queue
from hmlib.utils.gpu import (
    CachedIterator,
    StreamCheckpoint,
    StreamTensor,
    cuda_stream_scope,
    get_gpu_capabilities,
)
from hmlib.utils.image import (
    ImageColorScaler,
    ImageHorizontalGaussianDistribution,
    crop_image,
    image_height,
    image_width,
    make_channels_last,
    make_visible_image,
    resize_image,
)
from hmlib.utils.letterbox import py_letterbox
from hmlib.utils.progress_bar import ProgressBar

from .video_stream import VideoStreamWriter


def slow_to_tensor(tensor: Union[torch.Tensor, StreamTensor]) -> torch.Tensor:
    """
    Give up on the stream and get the sync'd tensor
    """
    if isinstance(tensor, StreamTensor):
        tensor._verbose = True
        # return tensor.get()
        return tensor.wait()
    return tensor


def quick_show(img: torch.Tensor, wait: bool = False):
    if img.ndim == 4:
        for s_img in img:
            cv2.imshow(
                "online_im",
                make_visible_image(
                    s_img,
                ),
            )
            cv2.waitKey(1)
    else:
        assert img.ndim == 3
        cv2.imshow(
            "online_im",
            make_visible_image(
                img,
            ),
        )
        cv2.waitKey(1 if not wait else 0)


def get_best_codec(gpu_number: int, width: int, height: int):
    caps = get_gpu_capabilities()
    compute = float(caps[gpu_number]["compute_capability"])
    if compute >= 7 and width <= 9900:  # FIXME: I forget what the max is? 99-thousand-something
        return "hevc_nvenc", True
    elif compute >= 6 and width <= 4096:
        return "hevc_nvenc", True
    else:
        return "XVID", False
    # return "XVID", False


def rotate_image(img, angle: float, rotation_point: List[int]):
    rotation_point = [int(i) for i in rotation_point]
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            # H, W, C -> C, W, H
            img = img.permute(0, 3, 2, 1)
            angle = -angle
            if current_dtype == torch.half:
                img = img.to(torch.float32, non_blocking=True)
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
            current_dtype = img.dtype
            if current_dtype == torch.half:
                img = img.to(torch.float32, non_blocking=True)
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
        img = cv2.warpAffine(img, rotation_matrix, (image_width(img), image_height(img)))
    return img


def paste_watermark_at_position(dest_image, watermark_rgb_channels, watermark_mask, x: int, y: int):
    assert dest_image.ndim == 4
    assert dest_image.device == watermark_rgb_channels.device
    assert dest_image.device == watermark_mask.device
    watermark_height = image_height(watermark_rgb_channels)
    watermark_width = image_width(watermark_rgb_channels)
    dest_image[:, y : y + watermark_height, x : x + watermark_width] = (
        dest_image[:, y : y + watermark_height, x : x + watermark_width] * (1 - watermark_mask)
        + watermark_rgb_channels * watermark_mask
    )
    return dest_image


# class ImageProcData:
#     def __init__(self, frame_id: int, img, current_box: torch.Tensor):
#         self.frame_id = frame_id
#         self.img = img
#         if current_box is None:
#             self.current_box = torch.tensor(
#                 (0, 0, image_width(img), image_height(img)),
#                 dtype=torch.int64,
#                 device=img.device,
#             )
#             if img.ndim == 4:
#                 # batched
#                 assert self.current_box.ndim == 1
#                 self.current_box = self.current_box.unsqueeze(0).repeat(img.size(0), 1)
#         else:
#             self.current_box = current_box.clone()
#         if not isinstance(self.frame_id, torch.Tensor):
#             frame_count = 1
#             if img.ndim == 4:
#                 frame_count = img.shape[0]
#             # assert img.ndim == 3 or img.size(0) == 1  # single image only
#             self.frame_id = torch.tensor(
#                 [self.frame_id + i for i in range(frame_count)], dtype=torch.int64
#             )

#     def dump(self):
#         logger.info(f"frame_id={self.frame_id}, current_box={self.current_box}")

#     def dump(self):
#         logger.info(f"frame_id={self.frame_id}, current_box={self.current_box}")


def _to_float(
    tensor: torch.Tensor,
    apply_scale: bool = False,
    non_blocking: bool = False,
    dtype: torch.dtype = torch.float,
):
    assert not apply_scale
    if tensor.dtype == torch.uint8:
        if apply_scale:
            assert False
            return tensor.to(dtype, non_blocking=non_blocking) / 255.0
        else:
            return tensor.to(dtype, non_blocking=non_blocking)
    else:
        assert torch.is_floating_point(tensor)
    return tensor


def _to_uint8(tensor: torch.Tensor, apply_scale: bool = False, non_blocking: bool = False):
    assert not apply_scale
    if isinstance(tensor, np.ndarray):
        assert tensor.dtype == np.uint8
        return tensor
    if tensor.dtype != torch.uint8:
        if apply_scale:
            assert False
            assert torch.is_floating_point(tensor)
            return (
                # note, no scale applied here (I removed before adding assert)
                tensor.clamp(min=0, max=255.0).to(torch.uint8, non_blocking=non_blocking)
            )
        else:
            # There has got to be a more elegant way to do this with reflection
            def _clamp(t, *args, **kwargs):
                return t.clamp(*args, **kwargs).to(torch.uint8, non_blocking=non_blocking)

            if isinstance(tensor, StreamTensor):
                assert False
                return tensor.call_with_checkpoint(_clamp, min=0, max=255.0)
            else:
                return _clamp(tensor, min=0, max=255.0)
    return tensor


def image_wh(image: torch.Tensor):
    if image.shape[-1] in [3, 4]:
        return torch.tensor(
            [image.shape[-2], image.shape[-3]], dtype=torch.float, device=image.device
        )
    assert image.shape[1] in [3, 4]
    return torch.tensor([image.shape[-1], image.shape[-2]], dtype=torch.float, device=image.device)


def tensor_ref(tensor: Union[torch.Tensor, StreamTensor]) -> torch.Tensor:
    if isinstance(tensor, StreamTensor):
        return tensor.ref()
    return tensor


def tensor_checkpoint(
    tensor: Union[torch.Tensor, StreamTensor]
) -> Union[torch.Tensor, StreamTensor]:
    if isinstance(tensor, StreamTensor):
        tensor.new_checkpoint()
        return tensor
    return tensor


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
        original_clip_box: torch.Tensor = None,
        progress_bar: ProgressBar | None = None,
        cache_size: int = 2,
    ):
        if device is not None:
            logger.info(
                f"Video output {output_frame_width}x{output_frame_height} "
                f"using device: {device} ({output_video_path})"
            )
        self._args = args

        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._name = name
        self._simple_save = simple_save
        self._fps = fps
        self._cache_size = cache_size
        self._skip_final_save = skip_final_save
        self._progress_bar = progress_bar
        assert output_frame_width > 4 and output_frame_height > 4
        self._output_frame_width = output_frame_width
        self._output_frame_height = output_frame_height
        self._output_aspect_ratio = self._output_frame_width / self._output_frame_height
        self._output_frame_width_int = int(self._output_frame_width)
        self._output_frame_height_int = int(self._output_frame_height)
        self._original_clip_box = original_clip_box
        self._use_fork = use_fork
        self._max_queue_backlog = max_queue_backlog
        self._imgproc_thread = None
        self._imgproc_queue = create_queue(mp=use_fork)
        assert isinstance(self._imgproc_queue, SidebandQueue)
        self._imgproc_thread = None
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._print_interval = print_interval
        self._output_video = None

        self._scoreboard = None
        self._scoreboard_points = None
        if hasattr(args, "game_config"):
            self._scoreboard_points = get_nested_value(
                args.game_config, "rink.scoreboard.perspective_polygon", None
            )

        if fourcc == "auto":
            if self._device.type == "cuda":
                self._fourcc, is_gpu = get_best_codec(
                    device.index,
                    width=int(output_frame_width),
                    height=int(output_frame_height),
                )
                if not is_gpu:
                    logger.info(f"Can't use GPU for output video {output_video_path}")
                    # self._device = torch.device("cpu")
            else:
                self._fourcc = "XVID"
            logger.info(
                f"Output video {self._name} {int(self._output_frame_width)}x"
                f"{int(self._output_frame_height)} will use codec: {self._fourcc}"
            )
        else:
            self._fourcc = fourcc

        self._horizontal_image_gaussian_distribution = None
        self._zero_f32 = torch.tensor(0, dtype=torch.float, device=device)
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
            self.watermark_height = image_height(self.watermark)
            self.watermark_width = image_width(self.watermark)
            self.watermark_rgb_channels = self.watermark[:, :, :3]
            watermark_alpha_channel = self.watermark[:, :, 3]
            self.watermark_mask = cv2.merge(
                [
                    watermark_alpha_channel,
                    watermark_alpha_channel,
                    watermark_alpha_channel,
                ]
            )

            if self._device is not None:
                self.watermark_rgb_channels = torch.from_numpy(self.watermark_rgb_channels).to(
                    self._device
                )
                self.watermark_mask = (
                    torch.from_numpy(self.watermark_mask).to(self._device).to(torch.half)
                )
                # Scale mask to [0, 1]
                self.watermark_mask = self.watermark_mask / torch.max(self.watermark_mask)
        else:
            self.watermark = None

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        if self._args.show_image:
            self._shower = Shower(show_scaled=self._args.show_scaled)
        else:
            self._shower = None

        if start:
            self.start()

    def set_progress_bar(self, progress_bar: ProgressBar):
        # TODO: hook any callbacks here?
        self._progress_bar = progress_bar

    def start(self):
        if self._use_fork:
            self._child_pid = os.fork()
            if not self._child_pid:
                self.final_image_processing()
        else:
            assert self._imgproc_thread is None and "Video output thread was already started"
            self._imgproc_thread = Thread(
                target=self._final_image_processing_wrapper,
                name="VideoOutput",
            )
            self._imgproc_thread.start()

    def stop(self):
        self._imgproc_queue.put(None)
        if self._imgproc_thread is not None:
            self._imgproc_thread.join()
            self._imgproc_thread = None

    def is_cuda_encoder(self):
        return "nvenc" in self._fourcc

    def append(self, img_proc_data: Dict[str, Any]):
        with TimeTracker(
            "Send to Video-Out queue", self._send_to_video_out_timer, print_interval=50
        ):
            counter = 0
            while self._imgproc_queue.qsize() > self._max_queue_backlog:
                counter += 1
                if (
                    not self.has_args()
                    or (
                        not self._args.show_image
                        and (not hasattr(self._args, "debug") or not self._args.debug)
                    )
                ) and counter % 10 == 0:
                    logger.info(f"Video out queue too large: {self._imgproc_queue.qsize()}")
                time.sleep(0.001)

            # Maybe move devices on a different stream
            # if (
            #     not isinstance(img_proc_data.img, np.ndarray)
            #     and img_proc_data.img.device != self._device
            # ):
            #     if isinstance(img_proc_data.img, torch.Tensor):
            #         # img_proc_data.img = img_proc_data.img.to(
            #         #     device=self._device, non_blocking=True
            #         # )
            #         # img_proc_data.img = StreamCheckpoint(tensor=img_proc_data.img)
            #         # img_proc_data.img = StreamTensorToDevice(
            #         #     tensor=img_proc_data.img,
            #         #     device=self._device,
            #         #     verbose=False,
            #         # )
            #         pass
            self._imgproc_queue.put(img_proc_data)

    def _final_image_processing_wrapper(self):
        try:
            self._final_image_processing_worker()
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
            self._horizontal_image_gaussian_distribution = ImageHorizontalGaussianDistribution(
                image_width
            )
        return self._horizontal_image_gaussian_distribution

    def has_args(self):
        return self._args is not None

    def calculate_desired_bitrate(self, width: int, height: int):
        # 4K @ 55M
        desired_bit_rate_per_pixel = 55e6 / (3840 * 2160)
        desired_bit_rate = int(desired_bit_rate_per_pixel * width * height)
        logger.info(
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
        if not self._args.crop_output_image and image.shape[1] != self._output_frame_width_int:
            self._output_frame_width = torch.tensor(
                image.shape[1], dtype=img_wh.dtype, device=img_wh.device
            )
            self._output_frame_height = torch.tensor(
                image.shape[0], dtype=img_wh.dtype, device=img_wh.device
            )
            self._output_aspect_ratio = self._output_frame_width / self._output_frame_height
            self._output_frame_width_int = int(self._output_frame_width)
            self._output_frame_height_int = int(self._output_frame_height)

        return image, current_box

    def _float_type(self):
        return torch.float16 if self._args.fp16 else torch.float

    def _final_image_processing_worker(self):
        logger.info("VideoOutput thread started.")
        plot_interias = False
        show_image_interval = 1
        skip_frames_before_show = 0
        timer = Timer()
        # The timer that reocrds the overall throughput
        final_all_timer = None
        if self._output_video_path and self._output_video is None and not self._skip_final_save:
            if "_nvenc" in self._fourcc or self._output_video_path.startswith("rtmp://"):
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
                self._output_video.set(
                    cv2.CAP_PROP_BITRATE,
                    self.calculate_desired_bitrate(
                        width=self._output_frame_width, height=self._output_frame_height
                    ),
                )
            assert self._output_video.isOpened()

        scoreboard = None

        if self._scoreboard_points:
            scoreboard = Scoreboard(
                src_pts=self._scoreboard_points,
                dest_width=get_nested_value(
                    self._args.game_config, "rink.scoreboard.projected_width"
                ),
                dest_height=get_nested_value(
                    self._args.game_config, "rink.scoreboard.projected_height"
                ),
                clip_box=self._original_clip_box,
                dtype=torch.float,
                device=self._device,
            )

        batch_count = 0

        last_frame_id = None

        default_cuda_stream = None
        cuda_stream = None

        if self._device.type == "cuda":
            default_cuda_stream = torch.cuda.current_stream(self._device)
            cuda_stream = torch.cuda.Stream(self._device)

        with cuda_stream_scope(cuda_stream):
            iqueue = IterableQueue(self._imgproc_queue)
            imgproc_iter = iter(iqueue)

            imgproc_iter = CachedIterator(iterator=imgproc_iter, cache_size=self._cache_size)

            while True:
                batch_count += 1
                try:
                    imgproc_data = next(imgproc_iter)
                except StopIteration:
                    break

                timer.tic()

                current_box = imgproc_data["current_box"]
                online_im = imgproc_data["img"]

                #
                # BEGIN END-ZONE
                #
                replacement_image = None
                pano_width = image_width(online_im)
                current_box_x = center(current_box)[0]
                current_box_left = current_box[0]
                current_box_right = current_box[2]
                if current_box_right - current_box_left < pano_width / 1.5:
                    other_data = imgproc_data["data"]
                    # print(int(current_box_x))
                    width_ratio = 4
                    if current_box_x <= pano_width / width_ratio and "far_left" in other_data:
                        print("LEFT")
                        replacement_image = other_data["far_left"]
                        if replacement_image is not None:
                            replacement_image = replacement_image[0]
                    elif (
                        pano_width - current_box_x <= pano_width / width_ratio
                        and "far_right" in other_data
                    ):
                        print("RIGHT")
                        replacement_image = other_data["far_right"]
                        if replacement_image is not None:
                            replacement_image = replacement_image[0]
                else:
                    print("too large")
                #
                # END END-ZONE
                #

                if isinstance(online_im, StreamTensor):
                    # assert not online_im.owns_stream
                    online_im._verbose = True
                    # online_im = online_im.get()
                    online_im = online_im.wait(cuda_stream)

                frame_id = imgproc_data["frame_id"]
                if frame_id.ndim == 0:
                    frame_id = frame_id.unsqueeze(0)

                if last_frame_id is None:
                    last_frame_id = frame_id[-1]
                else:
                    assert frame_id[0] == last_frame_id + 1
                    last_frame_id = frame_id[-1]

                if isinstance(online_im, np.ndarray):
                    online_im = torch.from_numpy(online_im)

                # assert online_im.device.type == "cpu" or online_im.device == self._device
                if online_im.device.type != "cpu" and self._device.type == "cpu":
                    online_im = online_im.cpu()

                if online_im.ndim == 3:
                    online_im = online_im.unsqueeze(0)
                    current_box = current_box.unsqueeze(0)

                # batch_size = 1 if online_im.ndim != 4 else online_im.size(0)
                batch_size = online_im.shape[0]

                # torch.cuda.synchronize()

                if self._device is not None and (not self._simple_save or "nvenc" in self._fourcc):
                    if isinstance(online_im, np.ndarray):
                        online_im = torch.from_numpy(online_im)
                    online_im = make_channels_last(online_im)
                    if str(online_im.device) != str(self._device):
                        online_im = online_im.to(self._device, non_blocking=True)

                src_image_width = image_width(online_im)
                src_image_height = image_height(online_im)

                #
                # Extract the scoreboard before we do any cropping or rotation
                #
                scoreboard_img = None
                if scoreboard is not None:
                    online_im = slow_to_tensor(online_im)
                    scoreboard_img = make_channels_last(scoreboard.forward(online_im))

                #
                # Perspective rotation
                #
                if (
                    self.has_args()
                    and self._args.fixed_edge_rotation
                    and self._args.fixed_edge_rotation_angle
                ):
                    online_im = slow_to_tensor(online_im)
                    online_im = _to_float(
                        online_im,
                        non_blocking=True,
                        dtype=self._float_type(),
                    )
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

                        # logger.info(f"gaussian={gaussian}")
                        angle = fixed_edge_rotation_angle - fixed_edge_rotation_angle * gaussian
                        angle *= mult

                        # BEGIN PERFORMANCE HACK
                        #
                        # Chop off edges of image that won't be visible after a final crop
                        # before we rotate in order to reduce the computation necessary
                        # for the rotation (as well as other subsequent operations)
                        #
                        if True and self._args.crop_output_image:
                            img, bbox = self.crop_working_image_width(image=img, current_box=bbox)
                            src_image_width = image_width(img)
                            src_image_height = image_height(img)
                            rotation_point = center(bbox)
                        #
                        # END PERFORMANCE HACK
                        #

                        # logger.info(f"angle={angle}")
                        img = rotate_image(
                            img=img,
                            angle=angle,
                            rotation_point=rotation_point,
                        )
                        rotated_images.append(img)
                        current_boxes.append(bbox)
                    # duration = time.time() - start
                    # logger.info(f"rotate image took {duration} seconds")
                    online_im = torch.stack(rotated_images)
                    current_box = torch.stack(current_boxes)

                #
                # Crop to output video frame image
                #
                if self.has_args() and self._args.crop_output_image:
                    online_im = slow_to_tensor(online_im)
                    online_im = _to_float(
                        online_im,
                        non_blocking=True,
                        dtype=self._float_type(),
                    )
                    # assert torch.isclose(
                    #     aspect_ratio(current_box), self._output_aspect_ratio,
                    # )
                    # logger.info(f"crop ar={aspect_ratio(current_box)}")
                    cropped_images = []
                    for img, bbox in zip(online_im, current_box):
                        intbox = [int(i) for i in bbox]
                        x1 = intbox[0]
                        y1 = intbox[1]
                        y2 = intbox[3]
                        x2 = int(x1 + int(float(y2 - y1) * self._output_aspect_ratio))
                        assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
                        if y1 >= src_image_height or y2 >= src_image_height:
                            logger.info(
                                f"y1 ({y1}) or y2 ({y2}) is too large, should be < {src_image_height}"
                            )
                            # assert y1 < img.shape[0] and y2 < img.shape[0]
                            y1 = min(y1, src_image_height)
                            y2 = min(y2, src_image_height)
                        if x1 >= src_image_width or x2 >= src_image_width:
                            logger.info(
                                f"x1 {x1} or x2 {x2} is too large, should be < {src_image_width}"
                            )
                            # assert x1 < img.shape[1] and x2 < img.shape[1]
                            x1 = min(x1, src_image_width)
                            x2 = min(x2, src_image_width)

                        img = crop_image(img, x1, y1, x2, y2)
                        if image_height(img) != int(self._output_frame_height) or image_width(
                            img
                        ) != int(self._output_frame_width):
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
                # Scoreboard
                #
                if scoreboard_img is not None:
                    if torch.is_floating_point(online_im) and not torch.is_floating_point(
                        scoreboard_img
                    ):
                        scoreboard_img = scoreboard_img.to(torch.float, non_blocking=True)
                    online_im[:, : scoreboard.height, : scoreboard.width, :] = scoreboard_img

                #
                # Watermark
                #
                if self.has_args() and self._args.use_watermark:
                    y = int(image_height(online_im) - self.watermark_height)
                    x = int(
                        image_width(online_im) - self.watermark_width - self.watermark_width / 10
                    )
                    online_im = paste_watermark_at_position(
                        online_im,
                        watermark_rgb_channels=self.watermark_rgb_channels,
                        watermark_mask=self.watermark_mask,
                        x=x,
                        y=y,
                    )

                online_im = _to_uint8(online_im, non_blocking=True)

                if self._image_color_scaler is not None:
                    online_im = self._image_color_scaler.maybe_scale_image_colors(image=online_im)

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
                        online_im = np.ascontiguousarray(online_im.detach().cpu().numpy())

                #
                # Frame Number
                #
                if self.has_args() and self._args.plot_frame_number:
                    prev_device = None
                    if isinstance(online_im, torch.Tensor):
                        prev_device = online_im.device
                    if cuda_stream is not None:
                        cuda_stream.synchronize()
                    online_im = vis.plot_frame_number(
                        online_im,
                        frame_id=frame_id,
                    )
                    if prev_device is not None and online_im.device != prev_device:
                        online_im = online_im.to(prev_device, non_blocking=True)

                # Output (and maybe show) the final image
                if (
                    self.has_args()
                    and self._args.show_image
                    and imgproc_data["frame_id"] >= skip_frames_before_show
                ):
                    if imgproc_data["frame_id"] % show_image_interval == 0:
                        if cuda_stream is not None:
                            cuda_stream.synchronize()
                        show_img = online_im
                        if self._shower is not None:
                            # Async
                            self._shower.show(show_img)
                        else:
                            # Sync
                            if show_img.ndim == 3:
                                show_img = show_img.unsqueeze(0)
                            for s_img in show_img:
                                cv2.imshow(
                                    "online_im",
                                    make_visible_image(
                                        s_img, enable_resizing=self._args.show_scaled
                                    ),
                                )
                                cv2.waitKey(1)

                online_im = make_channels_last(online_im)
                assert int(self._output_frame_width) == online_im.shape[-2]
                assert int(self._output_frame_height) == online_im.shape[-3]
                if self._output_video is not None and not self._skip_final_save:
                    if isinstance(self._output_video, cv2.VideoWriter):
                        assert online_im.ndim == 4
                        for img in online_im:
                            self._output_video.write(img)
                    else:
                        online_im = StreamCheckpoint(tensor=online_im)
                        with cuda_stream_scope(default_cuda_stream):
                            # IMPORTANT:
                            # The encode is going to use thedefault stream,
                            # so call write() under that stream so that any actions
                            # taken while pushing occur on the same stream as the
                            # ultimate encoding
                            self._output_video.write(online_im)
                if self._save_frame_dir:
                    # frame_id should start with 1
                    assert imgproc_data["frame_id"]
                    cv2.imwrite(
                        os.path.join(
                            self._save_frame_dir,
                            "frame_{:06d}.png".format(int(imgproc_data["frame_id"]) - 1),
                        ),
                        online_im,
                    )
                timer.toc()

                if self._print_interval and batch_count % self._print_interval == 0:
                    logger.info(
                        "Image Post-Processing {} frame {} ({:.2f} fps)".format(
                            self._name,
                            frame_id[0],
                            batch_size * 1.0 / max(1e-5, timer.average_time),
                        )
                    )
                    timer = Timer()

                if True:
                    # Overall FPS
                    if final_all_timer is None:
                        final_all_timer = Timer()
                    else:
                        final_all_timer.toc()

                    if self._print_interval and batch_count % (self._print_interval * 4) == 0:
                        logger.info(
                            "*** Overall performance, frame {} ({:.2f} fps)  -- open files count: {}".format(
                                frame_id[0],
                                batch_size * 1.0 / max(1e-5, final_all_timer.average_time),
                                get_open_files_count(),
                            )
                        )
                        final_all_timer = Timer()
                    final_all_timer.tic()


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

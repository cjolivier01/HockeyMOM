from __future__ import absolute_import, division, print_function

import os
import time
import traceback
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL
import torch
import torchvision as tv
from PIL import Image
from torchvision.transforms import functional as F

from hmlib.camera.end_zones import EndZones, load_lines_from_config
from hmlib.config import get_nested_value
from hmlib.log import logger
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.scoreboard.selector import configure_scoreboard
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.boundaries import adjust_point_for_clip_box
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.transforms import HmPerspectiveRotation  # TODO: pipeline this
from hmlib.ui.show import show_image
from hmlib.ui.shower import Shower
from hmlib.utils.box_functions import center, height, width
from hmlib.utils.containers import IterableQueue, SidebandQueue, create_queue
from hmlib.utils.gpu import (
    StreamCheckpoint,
    StreamTensor,
    cuda_stream_scope,
    get_gpu_capabilities,
)
from hmlib.utils.image import (
    ImageColorScaler,
    crop_image,
    image_height,
    image_width,
    make_channels_last,
    make_visible_image,
    resize_image,
    to_float_image,
    to_uint8_image,
)
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.path import add_suffix_to_filename
from hmlib.utils.progress_bar import ProgressBar
from hmlib.vis.pt_text import draw_text

from .video_stream import (
    VideoStreamWriterInterface,
    clamp_max_video_dimensions,
    create_output_video_stream,
)


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

    VIDEO_DEFAULT: str = "default"
    VIDEO_END_ZONES: str = "end_zones"

    def __init__(
        self,
        args,
        output_video_path: str,
        output_frame_width: int,
        output_frame_height: int,
        fps: float,
        fourcc: str = "auto",
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
        self._args = args
        self._allow_scaling = False
        if simple_save:
            pre_area = output_frame_width * output_frame_height
            output_frame_width, output_frame_height = clamp_max_video_dimensions(
                output_frame_width, output_frame_height
            )
            post_area = output_frame_width * output_frame_height
            if pre_area != post_area:
                # We had to scale down
                self._allow_scaling = True

        if device is not None:
            logger.info(
                f"Video output {output_frame_width}x{output_frame_height} "
                f"using device: {device} ({output_video_path})"
            )

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
        self._output_videos: Dict[str, VideoStreamWriterInterface] = {}
        self._scoreboard = None

        self._end_zones = None
        if args is not None and args.end_zones:
            lines: Dict[str, List[Tuple[int, int]]] = load_lines_from_config(args.game_config)
            if lines:
                # Adjust for clipb ox, if any
                if original_clip_box is not None:
                    orig_lines = lines
                    lines: Dict[str, List[Tuple[int, int]]] = {}
                    for key, line in orig_lines.items():
                        line[0] = adjust_point_for_clip_box(line[0], original_clip_box)
                        line[1] = adjust_point_for_clip_box(line[1], original_clip_box)
                        lines[key] = line

                self._end_zones = EndZones(
                    lines=lines,
                    output_width=self._output_frame_width_int,
                    output_height=self._output_frame_height_int,
                )

        self._scoreboard = None
        self._scoreboard_points = None
        if hasattr(args, "game_id"):
            self._scoreboard_points = configure_scoreboard(game_id=args.game_id)

        if fourcc == "auto":
            if self._device.type == "cuda":
                self._fourcc, is_gpu = get_best_codec(
                    device.index,
                    width=int(output_frame_width),
                    height=int(output_frame_height),
                )
                if not is_gpu:
                    logger.info(f"Can't use GPU for output video {output_video_path}")
                    self._device = torch.device("cpu")
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

        if (
            not self._simple_save
            and self.has_args()
            and self._args.fixed_edge_rotation
            and self._args.fixed_edge_rotation_angle
        ):
            self._perspective_rotation = HmPerspectiveRotation(
                fixed_edge_rotation=self._args.fixed_edge_rotation,
                fixed_edge_rotation_angle=self._args.fixed_edge_rotation_angle,
                pre_clip=self._args.crop_output_image,
                dtype=self._float_type(),
            )
        else:
            self._perspective_rotation = None

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        if self.has_args() and self._args.show_image:
            self._shower = Shower(label="Video Out", show_scaled=self._args.show_scaled)
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
            self._imgproc_queue.put(img_proc_data)

    def _final_image_processing_wrapper(self):
        try:
            self._final_image_processing_worker()
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise
        finally:
            for _, video_out in self._output_videos.items():
                video_out.close()
            self._output_videos.clear()

    def has_args(self):
        return self._args is not None

    def _float_type(self):
        return torch.float16 if self._args.fp16 else torch.float

    def create_output_videos(self):
        if self._output_video_path and not self._skip_final_save:
            if self.VIDEO_DEFAULT not in self._output_videos:
                self._output_videos[self.VIDEO_DEFAULT] = create_output_video_stream(
                    filename=self._output_video_path,
                    fps=self._fps,
                    height=int(self._output_frame_height),
                    width=int(self._output_frame_width),
                    codec=self._fourcc,
                    device=self._device,
                    batch_size=1,
                )
                assert self._output_videos[self.VIDEO_DEFAULT].isOpened()

            if self._end_zones is not None and self.VIDEO_END_ZONES not in self._output_videos:
                self._output_videos[self.VIDEO_END_ZONES] = create_output_video_stream(
                    filename=str(add_suffix_to_filename(self._output_video_path, "-end-zones")),
                    fps=self._fps,
                    height=int(self._output_frame_height),
                    width=int(self._output_frame_width),
                    codec=self._fourcc,
                    device=self._device,
                    batch_size=1,
                )
                assert self._output_videos[self.VIDEO_END_ZONES].isOpened()

            if self._scoreboard_points:
                # Check for valid scoreboard points
                if torch.sum(torch.tensor(self._scoreboard_points, dtype=torch.int64)) != 0:
                    self._scoreboard = Scoreboard(
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

    def _final_image_processing_worker(self):
        logger.info("VideoOutput thread started.")

        # For opencv, needs to be in the same thread as what writes to it
        self.create_output_videos()

        # plot_interias = False
        show_image_interval = 1
        timer = Timer()
        # The timer that reocrds the overall throughput
        final_all_timer = None

        batch_count = 0

        # last_frame_id = None

        default_cuda_stream = None
        cuda_stream = None

        if self._device.type == "cuda":
            default_cuda_stream = torch.cuda.current_stream(self._device)
            cuda_stream = torch.cuda.Stream(self._device)

        with cuda_stream_scope(cuda_stream):
            iqueue = IterableQueue(self._imgproc_queue)
            imgproc_iter = iter(iqueue)

            imgproc_iter = CachedIterator(iterator=imgproc_iter, cache_size=self._cache_size)

            try:
                while True:
                    batch_count += 1
                    try:
                        results = next(imgproc_iter)
                    except StopIteration:
                        break

                    timer.tic()

                    results = self.forward(results)

                    online_im = results.pop("img")
                    image_w = image_width(online_im)
                    image_h = image_height(online_im)
                    assert online_im.ndim == 4  # Should have a batch dimension
                    batch_size = online_im.size(0)

                    if self._allow_scaling and int(self._output_frame_width) != image_w:
                        online_im = resize_image(
                            img=online_im,
                            new_width=self._output_frame_width,
                            new_height=self._output_frame_height,
                        )
                        image_w = image_width(online_im)
                        image_h = image_height(online_im)

                    # Output (and maybe show) the final image
                    online_im = make_channels_last(online_im)
                    assert int(self._output_frame_width) == image_w
                    assert int(self._output_frame_height) == image_h
                    if not self._skip_final_save:
                        if self.VIDEO_DEFAULT in self._output_videos:
                            if not isinstance(online_im, StreamTensor):
                                online_im = StreamCheckpoint(tensor=online_im)
                            with cuda_stream_scope(default_cuda_stream):
                                # IMPORTANT:
                                # The encode is going to use the default stream,
                                # so call write() under that stream so that any actions
                                # taken while pushing occur on the same stream as the
                                # ultimate encoding
                                self._output_videos[self.VIDEO_DEFAULT].write(online_im)

                        if self.VIDEO_END_ZONES in self._output_videos:
                            ez_img = self._end_zones.get_ez_image(results, dtype=online_im.dtype)
                            if ez_img is None:
                                ez_img = online_im
                            if not isinstance(ez_img, StreamTensor):
                                ez_img = StreamCheckpoint(tensor=ez_img)
                            with cuda_stream_scope(default_cuda_stream):
                                # IMPORTANT:
                                # The encode is going to use the default stream,
                                # so call write() under that stream so that any actions
                                # taken while pushing occur on the same stream as the
                                # ultimate encoding
                                self._output_videos[self.VIDEO_END_ZONES].write(ez_img)
                    if self.has_args() and self._args.show_image:
                        for i, frame_id in enumerate(results["frame_ids"]):
                            if int(frame_id) % show_image_interval == 0:
                                if cuda_stream is not None:
                                    cuda_stream.synchronize()
                                show_img = online_im[i]
                                # show_img = ez_img
                                self._shower.show(show_img)

                    # Save frames as individual frames
                    if self._save_frame_dir:
                        # frame_id should start with 1
                        assert results["frame_ids"][0] != 0
                        cv2.imwrite(
                            os.path.join(
                                self._save_frame_dir,
                                "frame_{:06d}.png".format(int(results["frame_id"]) - 1),
                            ),
                            online_im,
                        )
                    timer.toc()

                    if self._print_interval and batch_count % self._print_interval == 0:
                        logger.info(
                            "Image Post-Processing {} frame {} ({:.2f} fps)".format(
                                self._name,
                                results["frame_ids"][0],
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
                                    results["frame_ids"][0],
                                    batch_size * 1.0 / max(1e-5, final_all_timer.average_time),
                                    get_open_files_count(),
                                )
                            )
                            final_all_timer = Timer()
                        final_all_timer.tic()
            except:
                traceback.print_exc()

    def forward(self, results) -> Dict[str, Any]:
        track_data_sample = results.pop("data_samples")
        online_images = results.pop("img")
        frame_ids = results["frame_ids"]
        current_boxes = results.pop("current_box")

        results["pano_size_wh"] = [image_width(online_images), image_height(online_images)]

        # for frame_index, frame_id in enumerate(frame_ids):
        if True:
            # online_im = online_images[frame_index]
            online_im = online_images
            # current_box = current_boxes[frame_index]

            # We clone, since it gets modified sometimes wrt rotation optimizations
            if current_boxes is None:
                assert False  # how does this happen?
                current_box = torch.tensor(
                    [0, 0, image_width(online_im), image_height(online_im)], dtype=torch.float
                )
            else:
                current_boxes = current_boxes.clone()

            if isinstance(online_im, StreamTensor):
                online_im._verbose = True
                # online_im = online_im.get()
                online_im = online_im.wait(torch.cuda.current_stream())

            # if self._end_zones is not None:
            #     online_im = self._end_zones.draw(online_im)

            # if frame_id.ndim == 0:
            #     frame_id = frame_id.unsqueeze(0)

            if isinstance(online_im, np.ndarray):
                online_im = torch.from_numpy(online_im)

            if online_im.device.type != "cpu" and self._device.type == "cpu":
                assert False  # ?
                online_im = online_im.cpu()

            if online_im.ndim == 3:
                online_im = online_im.unsqueeze(0)
                current_box = current_box.unsqueeze(0)

            if self._device is not None and (not self._simple_save or "nvenc" in self._fourcc):
                if isinstance(online_im, np.ndarray):
                    online_im = torch.from_numpy(online_im)
                online_im = make_channels_last(online_im)
                if str(online_im.device) != str(self._device):
                    online_im = online_im.to(self._device, non_blocking=True)

            #
            # Extract the scoreboard before we do any cropping or rotation
            #
            scoreboard_img = None
            if self._scoreboard is not None:
                # online_im = slow_to_tensor(online_im)
                scoreboard_img = make_channels_last(self._scoreboard.forward(online_images))

            #
            # Perspective rotation
            #
            if self._perspective_rotation is not None:
                results["img"] = online_im
                results["camera_box"] = current_boxes
                results = self._perspective_rotation(results=results)
                # online_im may come back as a list of images, since the post-rotation clip
                # optimization may generate differently-sized images based on the size
                # of the current_box
                online_im = results.pop("img")
                current_boxes = results.pop("camera_box")

            src_image_width = image_width(online_im[0])
            src_image_height = image_height(online_im[0])

            #
            # Crop to output video frame image
            #
            if not self._simple_save and self.has_args() and self._args.crop_output_image:
                cropped_images = []
                for img, bbox in zip(online_im, current_boxes):
                    img = slow_to_tensor(img)
                    img = to_float_image(img, non_blocking=True, dtype=self._float_type())
                    intbox = [int(i) for i in bbox]
                    # print(intbox)
                    x1 = intbox[0]
                    y1 = intbox[1]
                    y2 = intbox[3]
                    x2 = int(x1 + int(float(y2 - y1) * self._output_aspect_ratio))
                    assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
                    if y1 >= src_image_height or y2 >= src_image_height:
                        logger.info(
                            f"y1 ({y1}) or y2 ({y2}) is too large, should be < {src_image_height}"
                        )
                        y1 = min(y1, src_image_height)
                        y2 = min(y2, src_image_height)
                    if x1 >= src_image_width or x2 >= src_image_width:
                        logger.info(
                            f"x1 {x1} or x2 {x2} is too large, should be < {src_image_width}"
                        )
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

            if not self._simple_save:
                #
                # BEGIN END-ZONE
                #
                if self._end_zones is not None:
                    results = self._end_zones(results)
                    ez_image = self._end_zones.get_ez_image(results, dtype=online_im.dtype)
                    if ez_image is not None:
                        # Apply the final overlays to the end-zone image
                        results = self._end_zones.put_ez_image(
                            data=results,
                            img=self.draw_final_overlays(
                                img=ez_image, frame_ids=frame_ids, scoreboard_img=scoreboard_img
                            ),
                        )
                #
                # END END-ZONE
                #
                online_im = self.draw_final_overlays(
                    img=online_im, frame_ids=frame_ids, scoreboard_img=scoreboard_img
                )

            results["img"] = online_im
        return results

    def draw_final_overlays(
        self, img: torch.Tensor, frame_ids: torch.Tensor, scoreboard_img: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Just make sure we're channels-last
        online_im = make_channels_last(img)

        #
        # Scoreboard
        #
        if scoreboard_img is not None:
            if torch.is_floating_point(online_im) and not torch.is_floating_point(scoreboard_img):
                scoreboard_img = scoreboard_img.to(torch.float, non_blocking=True)
            online_im[:, : self._scoreboard.height, : self._scoreboard.width, :] = scoreboard_img

        #
        # Watermark
        #
        if self.has_args() and self._args.use_watermark:
            y = int(image_height(online_im) - self.watermark_height)
            x = int(image_width(online_im) - self.watermark_width - self.watermark_width / 10)
            online_im = paste_watermark_at_position(
                online_im,
                watermark_rgb_channels=self.watermark_rgb_channels,
                watermark_mask=self.watermark_mask,
                x=x,
                y=y,
            )

        #
        # Frame Number
        #
        if self.has_args() and self._args.plot_frame_number:
            for i, frame_id in enumerate(frame_ids):
                online_im[i] = vis.plot_frame_number(
                    online_im[i],
                    frame_id=frame_id,
                )

        online_im = to_uint8_image(online_im, non_blocking=True)

        if self._image_color_scaler is not None:
            online_im = self._image_color_scaler.maybe_scale_image_colors(image=online_im)

        return online_im


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

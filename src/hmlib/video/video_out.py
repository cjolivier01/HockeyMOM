from __future__ import absolute_import, division, print_function

import os
import time
import traceback
from threading import Thread
from typing import Any, Dict, List, Set, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from mmcv.transforms import Compose
from torchvision.transforms import functional as F

from hmlib.bbox.box_functions import scale_box
from hmlib.camera.end_zones import EndZones, load_lines_from_config
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.boundaries import adjust_point_for_clip_box
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.ui.show import show_image
from hmlib.ui.shower import Shower
from hmlib.utils import MeanTracker
from hmlib.utils.containers import IterableQueue, SidebandQueue, create_queue
from hmlib.utils.exceptions import raise_exception_in_thread
from hmlib.utils.gpu import (
    StreamCheckpoint,
    StreamTensor,
    cuda_stream_scope,
    get_gpu_capabilities,
)
from hmlib.utils.image import (
    ImageColorScaler,
    image_height,
    image_width,
    make_channels_last,
    make_visible_image,
    resize_image,
    to_uint8_image,
)
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.path import add_suffix_to_filename
from hmlib.utils.progress_bar import ProgressBar
from hmlib.utils.tensor import to_tensor_scalar
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH

from .video_stream import (
    VideoStreamWriterInterface,
    clamp_max_video_dimensions,
    create_output_video_stream,
)


def get_and_pop(map: Dict[str, Any], key: str) -> Any:
    result = map.get(key, None)
    if result is not None:
        del map[key]
    return result


def slow_to_tensor(
    tensor: Union[torch.Tensor, StreamTensor], stream_wait: bool = True
) -> torch.Tensor:
    """
    Give up on the stream and get the sync'd tensor
    """
    if isinstance(tensor, StreamTensor):
        tensor._verbose = True
        if stream_wait:
            return tensor.wait()
        return tensor.get()
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


def get_best_codec(
    gpu_number: int, width: int, height: int, allow_scaling: bool = False
) -> Tuple[Literal["hevc_nvenc"] | Literal[True]] | Tuple[Literal["XVID"] | Literal[False]]:
    caps = get_gpu_capabilities()
    compute = float(caps[gpu_number]["compute_capability"])
    if compute >= 7 and (width <= MAX_NEVC_VIDEO_WIDTH or allow_scaling):
        return "hevc_nvenc", True
    elif compute >= 6 and width <= 4096:
        return "hevc_nvenc", True
    else:
        return "XVID", False
    # return "XVID", False


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


_FP_TYPES: Set[torch.dtype] = {
    torch.float,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.half,
}


class VideoOutput:

    VIDEO_DEFAULT: str = "default"
    VIDEO_END_ZONES: str = "end_zones"

    def __init__(
        self,
        args,
        output_video_path: str,
        output_frame_width: Union[int, float, torch.Tensor],
        output_frame_height: Union[int, float, torch.Tensor],
        fps: float,
        video_out_pipeline: Dict[str, Any],
        fourcc: str = "auto",
        bit_rate: int = int(55e6),
        save_frame_dir: str = None,
        start: bool = True,
        max_queue_backlog: int = 1,
        device: torch.device = None,
        name: str = "",
        simple_save: bool = False,
        skip_final_save: bool = False,
        image_channel_adjustment: List[float] = None,
        print_interval: int = 50,
        original_clip_box: torch.Tensor = None,
        progress_bar: ProgressBar | None = None,
        cache_size: int = 2,
        clip_to_max_dimensions: bool = True,
        async_output: bool = False,
        visualization_config: Dict[str, Any] = None,
        no_cuda_streams: bool = False,
        dtype: torch.dtype = None,
    ):
        self._args = args
        self._allow_scaling = False
        self._async_output = async_output
        self._clip_to_max_dimensions = clip_to_max_dimensions
        self._visualization_config = visualization_config
        self._no_cuda_streams = no_cuda_streams
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        assert self._dtype in _FP_TYPES

        output_frame_width = to_tensor_scalar(output_frame_width, device=device)
        output_frame_height = to_tensor_scalar(output_frame_height, device=device)

        if fourcc == "auto" and device.type == "cuda":
            fourcc = "hevc_nvenc"

        if simple_save and self._clip_to_max_dimensions:
            original_width = int(output_frame_width)
            output_frame_width, output_frame_height = clamp_max_video_dimensions(
                output_frame_width,
                output_frame_height,
                codec=fourcc,
            )
            self._allow_scaling = original_width != int(output_frame_width)

        if device is not None:
            logger.info(
                f"Video output {output_frame_width}x{output_frame_height} "
                f"using device: {device} ({output_video_path})"
            )
        assert output_frame_width > 4 and output_frame_height > 4
        self._output_frame_width = output_frame_width
        self._output_frame_height = output_frame_height
        self._output_frame_width_int = int(self._output_frame_width)
        self._output_frame_height_int = int(self._output_frame_height)
        self._output_aspect_ratio = self._output_frame_width / self._output_frame_height
        self._video_frame_config = {
            "output_frame_width": int(self._output_frame_width_int),
            "output_frame_height": int(self._output_frame_height_int),
            "output_aspect_ratio": self._output_aspect_ratio,
        }

        # -----------

        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._name = name
        self._simple_save = simple_save
        self._fps = fps
        self._cache_size = cache_size
        self._skip_final_save = skip_final_save
        self._progress_bar = progress_bar
        self._original_clip_box = original_clip_box
        self._max_queue_backlog = max_queue_backlog
        self._imgproc_thread = None
        self._imgproc_queue = create_queue(mp=False)
        assert isinstance(self._imgproc_queue, SidebandQueue)
        self._imgproc_thread = None
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._print_interval = print_interval
        self._output_videos: Dict[str, VideoStreamWriterInterface] = {}
        self._cuda_stream = torch.cuda.Stream(self._device) if self._device.type == "cuda" else None
        self._default_cuda_stream = None

        self._bit_rate = bit_rate
        self._end_zones = None
        if args is not None and args.end_zones:
            lines: Dict[str, List[Tuple[int, int]]] = load_lines_from_config(args.game_config)
            if lines:
                # Adjust for clip box, if any
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

        if fourcc == "auto":
            if self._device.type == "cuda":
                self._fourcc, is_gpu = get_best_codec(
                    device.index,
                    width=int(output_frame_width),
                    height=int(output_frame_height),
                    allow_scaling=self._allow_scaling,
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

        self._video_out_pipeline = video_out_pipeline
        if self._video_out_pipeline is not None:
            self._video_out_pipeline = Compose(self._video_out_pipeline)

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        if self.has_args() and self._args.show_image:
            self._shower = Shower(
                label="Video Out", show_scaled=self._args.show_scaled, max_size=self._cache_size
            )
        else:
            self._shower = None

        self._mean_tracker: Optional[MeanTracker] = None

        if start:
            self.start()

    def set_progress_bar(self, progress_bar: ProgressBar):
        # Should we hook any callbacks here for adding displayed fields?
        self._progress_bar = progress_bar

    def start(self):
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
        if self._shower is not None:
            self._shower.close()
            self._shower = None

    # def is_cuda_encoder(self):
    #     return "nvenc" in self._fourcc

    def append(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if not self._async_output:
            with cuda_stream_scope(self._cuda_stream):
                results = self.forward(results)
                assert results["img"].device == self._device
                results = self.save_frame(
                    results,
                    cuda_stream=self._cuda_stream,
                    default_cuda_stream=self._default_cuda_stream,
                )
            return results
        else:
            with TimeTracker(
                "Send to Video-Out queue", self._send_to_video_out_timer, print_interval=50
            ):
                counter = 0
                assert self._max_queue_backlog > 0
                while self._imgproc_queue.qsize() >= self._max_queue_backlog:
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
                self._imgproc_queue.put(results)

    def _final_image_processing_wrapper(self):
        self._default_cuda_stream = (
            torch.cuda.Stream(self._device) if self._device.type == "cuda" else None
        )
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
                    bit_rate=self._bit_rate,
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
                    bit_rate=self._bit_rate,
                    device=self._device,
                    batch_size=1,
                )
                assert self._output_videos[self.VIDEO_END_ZONES].isOpened()

    def _final_image_processing_worker(self):
        logger.info("VideoOutput thread started.")

        # For opencv, needs to be in the same thread as what writes to it
        self.create_output_videos()

        # plot_interias = False
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

        mean_track_mode = None
        if mean_track_mode and self._mean_tracker is None:
            self._mean_tracker = MeanTracker(file_path="video_out.txt", mode=mean_track_mode)

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

                    batch_size = results["img"].size(0)

                    results = self.forward(results)
                    results = self.save_frame(
                        results, cuda_stream=cuda_stream, default_cuda_stream=default_cuda_stream
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
            except Exception as ex:
                traceback.print_exc()
                raise_exception_in_thread(exception=ex)

    def save_frame(
        self,
        results: Dict[str, Any],
        cuda_stream: torch.cuda.Stream,
        default_cuda_stream: torch.cuda.Stream,
    ) -> Dict[str, Any]:
        online_im = results.pop("img")
        image_w = image_width(online_im)
        image_h = image_height(online_im)
        assert online_im.ndim == 4  # Should have a batch dimension

        # Output (and maybe show) the final image
        online_im = make_channels_last(online_im)
        assert int(self._output_frame_width) == image_w
        assert int(self._output_frame_height) == image_h

        if self._mean_tracker is not None:
            img = online_im
            self._mean_tracker(img)

        # torch.cuda.synchronize()
        # if cuda_stream is not None:
        #     cuda_stream.synchronize()

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
                    # online_im = online_im.get()
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
            online_im = slow_to_tensor(online_im)
            for show_img in online_im:
                if cuda_stream is not None:
                    cuda_stream.synchronize()
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
        return results

    def forward(self, results) -> Dict[str, Any]:

        # visualization_config = self._visualization_config

        online_im = results.pop("img")
        frame_ids = results.get("frame_ids")
        current_boxes = get_and_pop(results, "current_box")

        results["pano_size_wh"] = [image_width(online_im), image_height(online_im)]

        if current_boxes is None:
            assert self._simple_save
            assert online_im.ndim == 4
            batch_size: int = online_im.size(0)
            whole_box = torch.tensor(
                [0, 0, image_width(online_im), image_height(online_im)], dtype=torch.float
            )
            current_boxes = whole_box.repeat(batch_size, 1)
        else:
            current_boxes = current_boxes.clone()

        if isinstance(online_im, StreamTensor):
            online_im._verbose = True
            online_im = online_im.get()
            # online_im = online_im.wait(torch.cuda.current_stream())

        # if self._end_zones is not None:
        #     online_im = self._end_zones.draw(online_im)

        if isinstance(online_im, np.ndarray):
            online_im = torch.from_numpy(online_im)

        if online_im.ndim == 3:
            online_im = online_im.unsqueeze(0)
            current_box = current_box.unsqueeze(0)

        if self._device is not None and (not self._simple_save or "nvenc" in self._fourcc):
            if isinstance(online_im, np.ndarray):
                online_im = torch.from_numpy(online_im)
            online_im = make_channels_last(online_im)
            if str(online_im.device) != str(self._device):
                online_im = online_im.to(self._device, non_blocking=True)

        if not self._simple_save:
            #
            # BEGIN END-ZONE
            #
            if self._end_zones is not None:
                # EZ needs an image only for matching the lighting
                results["img"] = online_im
                results = self._end_zones(results)
                online_im = results.pop("img")
        #
        # END END-ZONE
        #

        #
        # Video-out pipeline
        #
        if self._video_out_pipeline is not None:
            results["img"] = online_im
            results["camera_box"] = current_boxes
            results["video_frame_cfg"] = self._video_frame_config
            results = self._video_out_pipeline(results)
            online_im = results.pop("img")
            current_boxes = results.pop("camera_box")

        if not self._simple_save:
            if self._end_zones is not None:
                ez_image = self._end_zones.get_ez_image(results, dtype=online_im.dtype)
                if ez_image is not None:
                    self._end_zones.put_ez_image(
                        data=results,
                        img=self.draw_final_overlays(img=ez_image, frame_ids=frame_ids),
                    )

        if self._allow_scaling and int(self._output_frame_width) != image_width(online_im):
            online_im = resize_image(
                img=online_im,
                new_width=self._output_frame_width,
                new_height=self._output_frame_height,
            )

        online_im = to_uint8_image(online_im, non_blocking=True)

        # Move to CPU last (if necessary)
        if online_im.device.type != "cpu" and self._device.type == "cpu":
            online_im = online_im.to("cpu", non_blocking=True)
            online_im = StreamCheckpoint(online_im)

        results["img"] = online_im
        return results


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

"""High-level stitched video writer and visualization utilities.

This module coordinates GPU streams, color transforms, overlays and IO to
produce the final rendered videos used by many CLIs.

@see @ref hmlib.video.video_stream "video_stream" for the underlying encoder.
"""

from __future__ import absolute_import, division, print_function

import contextlib
import math
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from mmcv.transforms import Compose

from hmlib.camera.end_zones import EndZones, load_lines_from_config
from hmlib.log import logger
from hmlib.tracking_utils.boundaries import adjust_point_for_clip_box
from hmlib.ui.shower import Shower
from hmlib.utils import MeanTracker
from hmlib.utils.gpu import (
    StreamCheckpoint,
    StreamTensorBase,
    cuda_stream_scope,
    get_gpu_capabilities,
)
from hmlib.utils.image import (
    ImageColorScaler,
    image_height,
    image_width,
    make_channels_last,
    resize_image,
    to_uint8_image,
)
from hmlib.utils.path import add_suffix_to_filename
from hmlib.utils.progress_bar import ProgressBar
from hmlib.utils.tensor import make_const_tensor
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH

from .video_stream import (
    VideoStreamWriterInterface,
    clamp_max_video_dimensions,
    create_output_video_stream,
)

standard_8k_width: int = 7680
standard_8k_height: int = 4320


def get_and_pop(map: Dict[str, Any], key: str) -> Any:
    result = map.get(key, None)
    if result is not None:
        del map[key]
    return result


def slow_to_tensor(
    tensor: Union[torch.Tensor, StreamTensorBase], stream_wait: bool = True
) -> torch.Tensor:
    """
    Give up on the stream and get the sync'd tensor
    """
    if isinstance(tensor, StreamTensorBase):
        tensor._verbose = True
        if stream_wait:
            return tensor.wait()
        return tensor.get()
    return tensor


def get_best_codec(
    gpu_number: int, width: int, height: int, allow_scaling: bool = False
) -> Tuple[Literal["hevc_nvenc"] | Literal[True]] | Tuple[Literal["XVID"] | Literal[False]]:
    caps = get_gpu_capabilities()
    compute = float(caps[gpu_number]["compute_capability"])
    if compute >= 7 and (width <= MAX_NEVC_VIDEO_WIDTH or allow_scaling):
        return "hevc_nvenc", True
        # return "h264_nvenc", True
    elif compute >= 6 and width <= 4096:
        return "hevc_nvenc", True
        # return "h264_nvenc", True
    else:
        return "XVID", False
    # return "XVID", False


def tensor_ref(tensor: Union[torch.Tensor, StreamTensorBase]) -> torch.Tensor:
    if isinstance(tensor, StreamTensorBase):
        return tensor.ref()
    return tensor


def tensor_checkpoint(
    tensor: Union[torch.Tensor, StreamTensorBase],
) -> Union[torch.Tensor, StreamTensorBase]:
    if isinstance(tensor, StreamTensorBase):
        tensor.new_checkpoint()
        return tensor
    return tensor


def is_nearly_8k(width, height, size_tolerance=0.10, aspect_ratio_tolerance=0.01):
    """
    Checks if a given width and height are within a configurable tolerance of 8K
    resolution and have a very similar aspect ratio.

    Args:
        width (int): The width dimension to check.
        height (int): The height dimension to check.
        size_tolerance (float): The maximum allowed percentage difference from 8K
                                dimensions (e.g., 0.10 for 10%).
        aspect_ratio_tolerance (float): The maximum allowed absolute difference
                                       in aspect ratio.

    Returns:
        tuple: A boolean indicating if the dimensions and aspect ratio match,
               and a string with details on the outcome.
    """
    # Define the reference 8K dimensions and aspect ratio
    ref_8k_width, ref_8k_height = standard_8k_width, standard_8k_height
    ref_aspect_ratio = ref_8k_width / ref_8k_height

    # Check if dimensions are within 10% of 8K
    width_ok = ref_8k_width * (1 - size_tolerance) <= width <= ref_8k_width * (1 + size_tolerance)
    height_ok = (
        ref_8k_height * (1 - size_tolerance) <= height <= ref_8k_height * (1 + size_tolerance)
    )

    # Check if the aspect ratio is very close
    try:
        current_aspect_ratio = width / height
        aspect_ratio_ok = math.isclose(
            current_aspect_ratio, ref_aspect_ratio, rel_tol=aspect_ratio_tolerance
        )
    except ZeroDivisionError:
        return False, "Height cannot be zero."

    # Return the combined result
    if width_ok and height_ok and aspect_ratio_ok:
        return True, "Dimensions are within 10% of 8K and have a very close aspect ratio."
    else:
        details = []
        if not width_ok:
            details.append(
                f"Width ({width}) is not within {size_tolerance*100}% of 8K width ({ref_8k_width})."
            )
        if not height_ok:
            details.append(
                f"Height ({height}) is not within {size_tolerance*100}% of 8K height ({ref_8k_height})."
            )
        if not aspect_ratio_ok:
            details.append(
                f"Aspect ratio ({current_aspect_ratio:.4f}) is not very close to 8K ({ref_aspect_ratio:.4f})."
            )

        return False, " and ".join(details)


_FP_TYPES: Set[torch.dtype] = {
    torch.float,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.half,
}


class VideoOutput(torch.nn.ModuleDict):

    VIDEO_DEFAULT: str = "default"
    VIDEO_END_ZONES: str = "end_zones"

    def __init__(
        self,
        output_video_path: str,
        output_frame_width: Union[int, float, torch.Tensor],
        output_frame_height: Union[int, float, torch.Tensor],
        fps: float,
        video_out_pipeline: Dict[str, Any] | None = None,
        fourcc: str = "auto",
        bit_rate: int = int(55e6),
        save_frame_dir: str | None = None,
        name: str = "",
        simple_save: bool = False,
        skip_final_save: bool = False,
        image_channel_adjustment: List[float] | None = None,
        print_interval: int = 50,
        original_clip_box: torch.Tensor | None = None,
        progress_bar: ProgressBar | None = None,
        cache_size: int = 2,
        clip_to_max_dimensions: bool = True,
        visualization_config: Dict[str, Any] | None = None,
        no_cuda_streams: bool = False,
        dtype: torch.dtype | None = None,
        device: Union[torch.device, str, None] = None,
        show_image: bool = False,
        show_scaled: Optional[float] = None,
        profiler: Any = None,
        game_config: Optional[Dict[str, Any]] = None,
        enable_end_zones: bool = False,
    ):
        super().__init__()
        self._allow_scaling = False
        self._clip_to_max_dimensions = clip_to_max_dimensions
        self._visualization_config = visualization_config
        self._no_cuda_streams = no_cuda_streams
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        assert self._dtype in _FP_TYPES

        output_frame_width = torch.tensor(output_frame_width, dtype=torch.int64)
        output_frame_height = torch.tensor(output_frame_height, dtype=torch.int64)
        self._fourcc = fourcc
        # if fourcc == "auto" and device.type == "cuda":
        #     fourcc = "hevc_nvenc"
        # fourcc = "h264_nvenc"

        if simple_save and self._clip_to_max_dimensions:
            original_width = int(output_frame_width)
            output_frame_width, output_frame_height = clamp_max_video_dimensions(
                output_frame_width,
                output_frame_height,
                codec=self._fourcc,
            )
            self._allow_scaling = original_width != int(output_frame_width)
        elif is_nearly_8k(output_frame_width, output_frame_height)[0]:
            # Check if close to standard 8k dimensions, in which case, make that the output
            original_width = int(output_frame_width)
            output_frame_width = torch.tensor(standard_8k_width)
            output_frame_height = torch.tensor(standard_8k_height)
            self._allow_scaling = original_width != int(output_frame_width)

        # if device is not None:
        #     logger.info(
        #         f"Video output {output_frame_width}x{output_frame_height} "
        #         f"using device: {device} ({output_video_path})"
        #     )
        assert output_frame_width > 4 and output_frame_height > 4
        self.register_buffer("_output_frame_width", output_frame_width, persistent=False)
        self.register_buffer("_output_frame_height", output_frame_height, persistent=False)
        self._output_frame_width_int = int(self._output_frame_width)
        self._output_frame_height_int = int(self._output_frame_height)
        self.register_buffer(
            "_output_aspect_ratio",
            self._output_frame_width / self._output_frame_height,
            persistent=False,
        )
        self._video_frame_config = {
            "output_frame_width": int(self._output_frame_width_int),
            "output_frame_height": int(self._output_frame_height_int),
            "output_aspect_ratio": self._output_aspect_ratio,
        }

        # -----------

        if device is not None:
            self._device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self._device = None
        self._name = name
        self._simple_save = simple_save
        self._fps = fps
        self._cache_size = cache_size
        self._skip_final_save = skip_final_save
        self._progress_bar = progress_bar
        self._original_clip_box = original_clip_box
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._print_interval = print_interval
        self._output_videos: Dict[str, VideoStreamWriterInterface] = {}
        self._cuda_stream = None

        self._bit_rate = bit_rate
        self._game_config = game_config
        self._end_zones = None
        if enable_end_zones and game_config is not None:
            lines: Dict[str, List[Tuple[int, int]]] = load_lines_from_config(game_config)
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

        self._fourcc = fourcc
        # if fourcc == "auto":
        #     if self._device.type == "cuda":
        #         self._fourcc, is_gpu = get_best_codec(
        #             device.index,
        #             width=int(output_frame_width),
        #             height=int(output_frame_height),
        #             allow_scaling=self._allow_scaling,
        #         )
        #         if not is_gpu:
        #             logger.info(f"Can't use GPU for output video {output_video_path}")
        #             self._device = torch.device("cpu")
        #     else:
        #         self._fourcc = "XVID"
        #     logger.info(
        #         f"Output video {self._name} {int(self._output_frame_width)}x"
        #         f"{int(self._output_frame_height)} will use codec: {self._fourcc}"
        #     )
        # else:
        #     self._fourcc = fourcc

        self._horizontal_image_gaussian_distribution = None
        # self._zero_f32 = torch.tensor(0, dtype=torch.float, device=device)
        # self._zero_uint8 = torch.tensor(0, dtype=torch.uint8, device=device)

        self._image_color_scaler = None
        if image_channel_adjustment:
            assert len(image_channel_adjustment) == 3
            self._image_color_scaler = ImageColorScaler(image_channel_adjustment)

        self._prof = profiler
        self._fctx = (
            self._prof.rf("video_out.forward")
            if getattr(self._prof, "enabled", False)
            else contextlib.nullcontext()
        )
        self._sctx = (
            self._prof.rf("video_out.save_frame")
            if getattr(self._prof, "enabled", False)
            else contextlib.nullcontext()
        )

        self._video_out_pipeline = self.compose_pipeline(video_out_pipeline)

        # Cache pointer to color adjust transform (if present)
        self._color_adjust_tf = None

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        self._show_image = bool(show_image)
        self._show_scaled = show_scaled
        self._shower = (
            Shower(label="Video Out", show_scaled=self._show_scaled, max_size=self._cache_size)
            if self._show_image
            else None
        )

        self._mean_tracker: Optional[MeanTracker] = None

    def compose_pipeline(self, pipeline):
        if pipeline is None or not pipeline:
            return None
        composed_pipeline = Compose(pipeline)
        for mod in composed_pipeline:
            if isinstance(mod, torch.nn.Module):
                name = str(mod)
                self[name] = mod
        return composed_pipeline

    def _ensure_initialized(self, context: Dict[str, Any]):
        if self._device is None:
            self._device = self._output_aspect_ratio.device
        if self._fourcc == "auto":
            if self._device.type == "cuda":
                self._fourcc, is_gpu = get_best_codec(
                    self._device.index,
                    width=self._output_frame_width_int,
                    height=self._output_frame_height_int,
                    allow_scaling=self._allow_scaling,
                )
                if not is_gpu:
                    logger.info(
                        f"Can't use GPU for output video {self._output_video_path}"
                    )
                    self._device = torch.device("cpu")
            else:
                self._fourcc = "XVID"
            logger.info(
                f"Output video {self._name} {self._output_frame_width_int}x"
                f"{self._output_frame_height_int} will use codec: {self._fourcc}"
            )

    def set_progress_bar(self, progress_bar: ProgressBar):
        # Should we hook any callbacks here for adding displayed fields?
        self._progress_bar = progress_bar

    def start(self):
        # VideoOutput now runs synchronously; start is a no-op kept for API compatibility.
        return None

    def stop(self):
        # Close UI resources; video streams are closed by their own lifecycle.
        if self._shower is not None:
            self._shower.close()
            self._shower = None

    def append(self, results: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized(results)
        with cuda_stream_scope(self._cuda_stream):
            if not self._output_videos:
                self.create_output_videos()
            if isinstance(results["img"], StreamTensorBase):
                results["img"] = results["img"].wait()
            with self._fctx:
                results = self.forward(results)
            assert results["img"].device == self._device
            with self._sctx:
                results = self.save_frame(results)
        return results

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

    def save_frame(
        self,
        results: Dict[str, Any],
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

        if self._show_image and self._shower is not None:
            online_im = slow_to_tensor(online_im)
            for show_img in online_im:
                self._shower.show(show_img)

        with torch.cuda.stream(torch.cuda.default_stream(online_im.device)):

            if online_im.is_cuda:
                # torch.cuda.current_stream(online_im.device).synchronize()
                torch.cuda.synchronize()

            if not self._skip_final_save:
                if self.VIDEO_DEFAULT in self._output_videos:
                    if not isinstance(online_im, StreamTensorBase):
                        online_im = StreamCheckpoint(tensor=online_im)
                    self._output_videos[self.VIDEO_DEFAULT].write(online_im)

                if self.VIDEO_END_ZONES in self._output_videos:
                    ez_img = self._end_zones.get_ez_image(results, dtype=online_im.dtype)
                    if ez_img is None:
                        ez_img = online_im
                    if not isinstance(ez_img, StreamTensorBase):
                        ez_img = StreamCheckpoint(tensor=ez_img)
                    self._output_videos[self.VIDEO_END_ZONES].write(ez_img)

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

        current_boxes = current_boxes.to(online_im.device, non_blocking=True)

        if isinstance(online_im, StreamTensorBase):
            online_im._verbose = True
            online_im = online_im.get()

        # if self._end_zones is not None:
        #     online_im = self._end_zones.draw(online_im)

        if isinstance(online_im, np.ndarray):
            online_im = torch.from_numpy(online_im)

        if online_im.ndim == 3:
            online_im = online_im.unsqueeze(0)
            # current_box = current_box.unsqueeze(0)

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
            # Update color adjust transform at runtime from YAML-like args config
            try:
                if self._color_adjust_tf is None:
                    for tf in getattr(self._video_out_pipeline, "transforms", []):
                        if tf.__class__.__name__ == "HmImageColorAdjust":
                            self._color_adjust_tf = tf
                            break
                if self._color_adjust_tf is not None and self._game_config is not None:
                    cam = None
                    try:
                        cam = self._game_config.get("rink", {}).get("camera", {})
                    except Exception:
                        cam = None
                    if isinstance(cam, dict):
                        color = cam.get("color", {}) or {}
                        # Allow fallback to flat camera keys too
                        wb = color.get("white_balance", cam.get("white_balance"))
                        wbk = color.get(
                            "white_balance_temp",
                            cam.get("white_balance_k", cam.get("white_balance_temp")),
                        )
                        bright = color.get("brightness", cam.get("color_brightness"))
                        contr = color.get("contrast", cam.get("color_contrast"))
                        gamma = color.get("gamma", cam.get("color_gamma"))
                        if wbk is not None and wb is None:
                            try:
                                # Kelvin can be numeric or string like '3500k'
                                self._color_adjust_tf.white_balance = (
                                    self._color_adjust_tf._gains_from_kelvin(wbk)
                                )
                            except Exception:
                                pass
                        elif wb is not None:
                            try:
                                if isinstance(wb, (list, tuple)) and len(wb) == 3:
                                    self._color_adjust_tf.white_balance = [float(x) for x in wb]
                            except Exception:
                                pass
                        # Scalars
                        if bright is not None:
                            try:
                                self._color_adjust_tf.brightness = float(bright)
                            except Exception:
                                pass
                        if contr is not None:
                            try:
                                self._color_adjust_tf.contrast = float(contr)
                            except Exception:
                                pass
                        if gamma is not None:
                            try:
                                self._color_adjust_tf.gamma = float(gamma)
                            except Exception:
                                pass
            except Exception:
                # Non-fatal if color transform not found
                pass
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

        online_im = to_uint8_image(online_im)

        # Move to CPU last (if necessary)
        if online_im.device.type != "cpu" and self._device.type == "cpu":
            online_im = online_im.to("cpu", non_blocking=True)
            online_im = StreamCheckpoint(online_im)

        results["img"] = online_im
        return results


def get_open_files_count():
    pid = os.getpid()
    return len(os.listdir(f"/proc/{pid}/fd"))

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

from hmlib.log import logger
from hmlib.ui.shower import Shower
from hmlib.utils import MeanTracker
from hmlib.utils.gpu import (
    StreamCheckpoint,
    StreamTensorBase,
    get_gpu_capabilities,
    unwrap_tensor,
    wrap_tensor,
)
from hmlib.utils.image import image_height, image_width
from hmlib.utils.path import add_suffix_to_filename
from hmlib.utils.progress_bar import ProgressBar
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH

from .video_stream import VideoStreamWriterInterface, create_output_video_stream

standard_8k_width: int = 7680
standard_8k_height: int = 4320


def get_and_pop(map: Dict[str, Any], key: str) -> Any:
    result = map.get(key, None)
    if result is not None:
        del map[key]
    return result


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
    """Synchronous video writer for final HockeyMOM frames.

    This module owns the lifecycle of one or more output video streams and
    encapsulates tensor-to-video conversion and IO. It does **not** perform
    camera logic, cropping, or overlays; those are handled upstream
    (e.g., :class:`hmlib.camera.apply_camera_plugin.ApplyCameraPlugin`).

    Typical usage::

        video_out = VideoOutput(...)
        video_out = video_out.to(device)
        for results in dataset:
            # results must include:
            #   - "img": tensor[B, H, W, C] or StreamTensorBase
            #   - "frame_ids": tensor[B] (1-based ids)
            # optionally:
            #   - "end_zone_img": tensor[B, H, W, C]
            video_out(results)

    The return value is the updated ``results`` dict; most callers ignore it.
    """

    VIDEO_DEFAULT: str = "default"
    VIDEO_END_ZONES: str = "end_zones"

    def __init__(
        self,
        output_video_path: str,
        fps: float,
        fourcc: str = "auto",
        bit_rate: int = int(55e6),
        save_frame_dir: str | None = None,
        name: str = "",
        simple_save: bool = False,
        skip_final_save: bool = False,
        progress_bar: ProgressBar | None = None,
        cache_size: int = 2,
        clip_to_max_dimensions: bool = True,
        visualization_config: Dict[str, Any] | None = None,
        dtype: torch.dtype | None = None,
        device: Union[torch.device, str, None] = None,
        show_image: bool = False,
        show_scaled: Optional[float] = None,
        profiler: Any = None,
        enable_end_zones: bool = False,
    ):
        """Construct a synchronous video writer.

        @param output_video_path: Destination filename for the main output video
                                  (e.g., ``tracking_output.mkv``). When
                                  ``skip_final_save`` is True, no file is written.
        @param fps: Frames per second to encode the output stream with.
        @param fourcc: Codec identifier. Use ``"auto"`` to pick a codec based
                       on GPU capabilities (e.g., ``"hevc_nvenc"`` or ``"XVID"``).
        @param bit_rate: Target bitrate in bits per second for the encoder.
        @param save_frame_dir: Optional directory for saving individual frames
                               as PNGs alongside the encoded video.
        @param name: Human-readable name used in logs (e.g., ``"TRACKING"``).
        @param simple_save: When True, disables some dynamic scaling logic and
                            assumes input frames are already at output size.
        @param skip_final_save: When True, the underlying writer is not asked to
                                finalize/flush the container at shutdown.
        @param image_channel_adjustment: Deprecated per-channel adjustment; kept
                                         for API compatibility but not used.
        @param print_interval: Interval (in frames) at which throughput logs can
                               be emitted (currently used only via callers).
        @param original_clip_box: Optional crop box applied earlier in the pipeline;
                                  used only for naming/logging, not for IO here.
        @param progress_bar: Optional :class:`ProgressBar` instance used by callers
                             to surface IO/throughput metrics.
        @param cache_size: Reserved for historical async batching; currently only
                           used to size the optional UI shower cache.
        @param clip_to_max_dimensions: When ``simple_save`` is True, scales video
                                      down to encoder-specific maximums (e.g., 8K).
        @param visualization_config: Reserved for future visualization options
                                     (not currently consumed).
        @param dtype: Preferred floating-point dtype for any internal tensors that
                      may be created (defaults to ``torch.get_default_dtype()``).
        @param device: Torch device or device string (e.g., ``"cuda:0"`` or
                       ``"cpu"``) on which encoding should run. When ``None``,
                       the device is inferred from input tensors at first call.
        @param show_image: When True, enables an interactive OpenCV window that
                           displays frames as they are written.
        @param show_scaled: Optional scale factor for the interactive viewer.
        @param profiler: Optional profiler object exposing ``rf(label)`` for
                         scoped timings; see :mod:`hmlib.utils.profiler`.
        @param game_config: Optional game configuration dict; kept for parity with
                            older APIs but not consumed directly by this class.
        @param enable_end_zones: When True, a secondary ``VIDEO_END_ZONES`` stream
                                 will be opened and any ``"end_zone_img"`` tensors
                                 present in ``results`` will be written there.
        """
        super().__init__()
        self._allow_scaling = False
        self._clip_to_max_dimensions = clip_to_max_dimensions
        self._visualization_config = visualization_config
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        assert self._dtype in _FP_TYPES

        if device is not None:
            self._device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self._device = None
        self._name = name
        self._simple_save = simple_save
        # Normalize to float even if upstream passes a Fraction
        self._fps = fps
        self._skip_final_save = skip_final_save
        self._progress_bar = progress_bar
        self._output_video_path = output_video_path
        self._save_frame_dir = save_frame_dir
        self._output_videos: Dict[str, VideoStreamWriterInterface] = {}

        self._bit_rate = bit_rate
        self._enable_end_zones: bool = bool(enable_end_zones)

        self._fourcc = fourcc

        self._prof = profiler
        self._fctx = (
            self._prof.rf("video_out.forward")
            if getattr(self._prof, "enabled", False)
            else contextlib.nullcontext()
        )
        self._sctx = (
            self._prof.rf("video_out._save_frame")
            if getattr(self._prof, "enabled", False)
            else contextlib.nullcontext()
        )

        if self._save_frame_dir and not os.path.isdir(self._save_frame_dir):
            os.makedirs(self._save_frame_dir)

        self._show_image = bool(show_image)
        self._show_scaled = show_scaled
        self._shower = (
            Shower(
                label="Video Out",
                show_scaled=self._show_scaled,
                max_size=cache_size,
                profiler=self._prof,
            )
            if self._show_image
            else None
        )

        self._mean_tracker: Optional[MeanTracker] = None

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        """Resolve device and codec configuration before the first write.

        This method is intentionally idempotent and cheap to call. It:
          - Infers the writer device from the registered output-size buffers
            when no explicit device was provided.
          - Resolves ``"auto"`` codecs to concrete GPU/CPU codecs using
            :func:`get_best_codec`.
        """
        if self._device is None:
            self._device = self._output_aspect_ratio.device
        if self._fourcc == "auto":
            video_frame_cfg = context["video_frame_cfg"]
            output_frame_width = int(video_frame_cfg["output_frame_width"])
            output_frame_height = int(video_frame_cfg["output_frame_height"])
            if self._device.type == "cuda":
                self._fourcc, is_gpu = get_best_codec(
                    self._device.index,
                    width=output_frame_width,
                    height=output_frame_height,
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
                f"Output video {self._name} {output_frame_width}x"
                f"{output_frame_height} will use codec: {self._fourcc}"
            )

    def set_progress_bar(self, progress_bar: ProgressBar):
        """Attach a progress bar instance used for external UI updates."""
        self._progress_bar = progress_bar

    def start(self):
        """Legacy no-op; synchronous VideoOutput does not require start()."""
        return None

    def stop(self):
        """Close any interactive UI resources (e.g., OpenCV shower)."""
        for stream in self._output_videos.values():
            stream.close()
        self._output_videos.clear()
        if self._shower is not None:
            self._shower.close()
            self._shower = None

    def create_output_videos(self, context: Dict[str, Any]) -> None:
        """Create underlying VideoStreamWriter instances if not already open."""
        if self._output_video_path and not self._skip_final_save:
            video_frame_cfg = context["video_frame_cfg"]
            output_frame_width = int(video_frame_cfg["output_frame_width"])
            output_frame_height = int(video_frame_cfg["output_frame_height"])
            if self.VIDEO_DEFAULT not in self._output_videos:
                self._output_videos[self.VIDEO_DEFAULT] = create_output_video_stream(
                    filename=self._output_video_path,
                    fps=self._fps,
                    height=output_frame_height,
                    width=output_frame_width,
                    codec=self._fourcc,
                    bit_rate=self._bit_rate,
                    device=self._device,
                    batch_size=1,
                    profiler=self._prof,
                )
                assert self._output_videos[self.VIDEO_DEFAULT].isOpened()

            if self._enable_end_zones and self.VIDEO_END_ZONES not in self._output_videos:
                self._output_videos[self.VIDEO_END_ZONES] = create_output_video_stream(
                    filename=str(add_suffix_to_filename(self._output_video_path, "-end-zones")),
                    fps=self._fps,
                    height=output_frame_height,
                    width=output_frame_width,
                    codec=self._fourcc,
                    bit_rate=self._bit_rate,
                    device=self._device,
                    batch_size=1,
                    profiler=self._prof,
                )
                assert self._output_videos[self.VIDEO_END_ZONES].isOpened()

    def _save_frame(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Write a batch of frames to the configured video streams.

        Expected keys in ``context`` on entry:
          - ``"img"``: tensor[B, H, W, C] or StreamTensorBase
          - ``"frame_ids"``: tensor[B] (used for logging and PNG naming)
          - ``"end_zone_img"``: optional tensor[B, H, W, C] for far-end output

        The ``"img"`` key is consumed and reattached in-place.
        """

        # Consume the image tensor
        online_im = context.pop("img")

        image_w = image_width(online_im)
        image_h = image_height(online_im)
        assert online_im.ndim == 4  # Should have a batch dimension

        assert online_im.dtype == torch.uint8
        assert online_im.is_contiguous

        video_frame_cfg = context["video_frame_cfg"]
        output_frame_width = int(video_frame_cfg["output_frame_width"])
        output_frame_height = int(video_frame_cfg["output_frame_height"])

        # Output (and maybe show) the final image
        assert int(output_frame_width) == image_w
        assert int(output_frame_height) == image_h

        online_im = unwrap_tensor(online_im, verbose=True)

        if self._mean_tracker is not None:
            self._mean_tracker(online_im)

        if self._show_image and self._shower is not None:
            for show_img in online_im:
                self._shower.show(show_img, clone=True)

        # torch.cuda.synchronize()
        # torch.cuda.current_stream(online_im.device).synchronize()

        if not self._skip_final_save:
            if self.VIDEO_DEFAULT in self._output_videos:
                self._output_videos[self.VIDEO_DEFAULT].write(online_im)

            if self.VIDEO_END_ZONES in self._output_videos:
                ez_img = context.get("end_zone_img")
                if ez_img is None:
                    ez_img = online_im
                self._output_videos[self.VIDEO_END_ZONES].write(ez_img)
        else:
            # Sync the stream if skipping final save
            online_im = wrap_tensor(online_im, verbose=True).get()

        # Save frames as individual frames
        if self._save_frame_dir:
            # frame_id should start with 1
            assert context["frame_ids"][0] != 0
            cv2.imwrite(
                os.path.join(
                    self._save_frame_dir,
                    "frame_{:06d}.png".format(int(context["frame_ids"][0]) - 1),
                ),
                online_im,
            )
        return context

    def forward(self, results: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        """Normalize input images and synchronously write them to disk.

        This is the primary public API. It:
          1. Lazily initializes device/codec/streams.
          2. Opens output video writers on first use.
          3. Writes the frames via :meth:`_save_frame`.

        @param results: Dict containing at least ``"img"`` and ``"frame_ids"``.
        @return: The updated ``results`` dict (for chaining if desired).
        """
        with self._fctx:
            # Step 1: Lazy initialization of device + codec
            self._ensure_initialized(results)

            # Step 2: Ensure underlying video streams are open
            if not self._output_videos:
                self.create_output_videos(results)

            # Step 3: Normalize image tensors onto the writer device
            # online_im = unwrap_tensor(results.get("img"))
            online_im = results["img"]

            if isinstance(online_im, np.ndarray):
                online_im = torch.from_numpy(online_im)

            if online_im.ndim == 3:
                # Ensure a batch dimension is present: [H, W, C] -> [1, H, W, C]
                online_im = online_im.unsqueeze(0)

            # online_im = self._make_visible_image(online_im)

            if not self._skip_final_save:
                if self._device is not None:
                    # Move to writer device and ensure channels-last layout
                    if str(online_im.device) != str(self._device):
                        online_im = unwrap_tensor(online_im).to(self._device)

                # Optional final move to CPU for CPU-only writers
                if (
                    online_im.device.type != "cpu"
                    and self._device is not None
                    and self._device.type == "cpu"
                ):
                    online_im = online_im.to("cpu", non_blocking=True)
                    online_im = wrap_tensor(online_im)

                assert self._device is None or results["img"].device == self._device

            results["img"] = online_im

            # Step 4: Persist frames to disk under profiling scopes
            with self._sctx:
                results = self._save_frame(results)

            assert "img" not in results

            return results

"""Data loader wiring for multi-camera stitching experiments (v2).

Connects MOT-style datasets, GPU stitching kernels and optional caching to
produce input batches for the main stitching CLIs.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import Compose

from hmlib.config import get_game_config, get_nested_value
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.log import logger
from hmlib.stitching.configure_stitching import configure_video_stitching
from hmlib.tracking_utils.timer import Timer
from hmlib.ui import Shower, show_image
from hmlib.utils import MeanTracker
from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import StreamCheckpoint, StreamTensorBase, cuda_stream_scope
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
    make_visible_image,
)
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.persist_cache_mixin import PersistCacheMixin
from hmlib.utils.tensor import make_const_tensor
from hmlib.video.ffmpeg import BasicVideoInfo
from hockeymom.core import CudaStitchPanoF32, CudaStitchPanoU8


def _get_dir_name(path: Any) -> Any:
    if os.path.isdir(str(path)):
        return path
    return Path(path).parent


_VERBOSE: bool = True


def INFO(*args: Any, **kwargs: Any) -> None:
    if not _VERBOSE:
        return
    print(*args, **kwargs)


def safe_put_queue(queue: Any, object: Any) -> None:
    try:
        queue.put(object)
    except BrokenPipeError:
        # Ignore broken pipe error
        return
    return


_LARGE_NUMBER_OF_FRAMES: float = 1e128


def to_tensor(tensor: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, StreamTensorBase):
        tensor.verbose = True
        return tensor.wait()
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    else:
        assert False


def sync_required(tensor: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> bool:
    if isinstance(tensor, (torch.Tensor, np.ndarray)):
        return False
    return True


def distribute_items_detailed(total_item_count: int, worker_count: int) -> List[int]:
    base_items_per_worker: int = total_item_count // worker_count
    remainder: int = total_item_count % worker_count

    distribution: List[int] = []
    for i in range(worker_count):
        if i < remainder:
            distribution.append(base_items_per_worker + 1)
        else:
            distribution.append(base_items_per_worker)

    return distribution


class MultiDataLoaderWrapper:
    def __init__(
        self, dataloaders: List[MOTLoadVideoWithOrig], input_queueue_size: int = 0
    ) -> None:
        self._dataloaders: List[MOTLoadVideoWithOrig] = dataloaders
        self._iters: List[Any] = []
        self._input_queueue_size: int = input_queueue_size
        self._len: Optional[int] = None

    def __iter__(self) -> "MultiDataLoaderWrapper":
        self._iters = []
        for dl in self._dataloaders:
            if self._input_queueue_size:
                self._iters.append(
                    CachedIterator(iterator=iter(dl), cache_size=self._input_queueue_size)
                )
            else:
                self._iters.append(iter(dl))
        return self

    def close(self) -> None:
        for dl in self._dataloaders:
            dl.close()

    def __next__(self) -> Any:
        result: List[Any] = []
        for it in self._iters:
            item = next(it)
            assert item is not None
            result.append(item)
        if not result:
            return None
        elif len(result) == 1:
            return result[0]
        else:
            return result

    def __len__(self) -> int:
        if self._len is None:
            self._len = -1
            for dl in self._dataloaders:
                this_len = len(dl)
                self._len = this_len if self._len is None else min(self._len, this_len)
        return self._len


def as_torch_device(device: Any) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    return device


##
#   _____ _   _  _        _     _____        _                     _
#  / ____| | (_)| |      | |   |  __ \      | |                   | |
# | (___ | |_ _ | |_  ___| |__ | |  | | __ _| |_  __ _  ___   ___ | |_
#  \___ \| __| || __|/ __| '_ \| |  | |/ _` | __|/ _` |/ __| / _ \| __|
#  ____) | |_| || |_| (__| | | | |__| | (_| | |_| (_| |\__ \|  __/| |_
# |_____/ \__|_| \__|\___|_| |_|_____/ \__,_|\__|\__,_||___/ \___| \__|
#
#
class StitchDataset(PersistCacheMixin, torch.utils.data.IterableDataset):

    def __init__(
        self,
        videos: Dict[str, List[Path]],
        pto_project_file: str = None,
        start_frame_number: int = 0,
        max_input_queue_size: int = 2,
        batch_size: int = 1,
        max_frames: int = None,
        auto_configure: bool = True,
        num_workers: int = 1,
        image_roi: List[int] = None,
        blend_mode: str = "laplacian",
        remapping_device: torch.device = None,
        decoder_device: torch.device = None,
        dtype: torch.dtype = torch.float,
        verbose: bool = False,
        auto_adjust_exposure: bool = False,
        on_first_stitched_image_callback: Optional[Callable] = None,
        minimize_blend: bool = True,
        python_blender: bool = True,
        no_cuda_streams: bool = False,
        async_mode: bool = True,
        show_image_components: bool = False,
        post_stitch_rotate_degrees: Optional[float] = None,
        profiler: Any = None,
        # Optional live config and per-camera color pipelines (from Aspen YAML).
        config_ref: Optional[Dict[str, Any]] = None,
        left_color_pipeline: Optional[List[Dict[str, Any]]] = None,
        right_color_pipeline: Optional[List[Dict[str, Any]]] = None,
        max_blend_levels: Optional[int] = None,
        capture_rgb_stats: bool = False,
        checkerboard_input: bool = False,
    ):
        super().__init__()
        if not async_mode:
            no_cuda_streams = True
        # max_input_queue_size = max(1, max_input_queue_size)
        self._start_frame_number = start_frame_number
        self._no_cuda_streams = no_cuda_streams or not async_mode
        self._checkerboard_input = checkerboard_input
        self._dtype = dtype
        self._verbose = verbose
        self._batch_size = batch_size
        self._remapping_device = as_torch_device(remapping_device)
        self._decoder_device = decoder_device
        self._video_left_offset_frame = videos["left"]["frame_offset"]
        self._video_right_offset_frame = videos["right"]["frame_offset"]
        self._videos = videos
        self._pto_project_file = pto_project_file
        self._max_input_queue_size = max_input_queue_size
        self._blend_mode = blend_mode
        self._auto_adjust_exposure = auto_adjust_exposure
        self._exposure_adjustment: List[float] = None
        self._max_frames = max_frames if max_frames is not None else _LARGE_NUMBER_OF_FRAMES
        self._to_coordinator_queue = create_queue(mp=False, name="stitchinbg-to-coordinator")
        self._from_coordinator_queue = create_queue(mp=False, name="stitching-from-coordinator")
        self._current_frame = start_frame_number
        self._next_requested_frame = start_frame_number
        self._on_first_stitched_image_callback = on_first_stitched_image_callback
        self._xy_pos_1, self._xy_pos_2 = None, None
        self._python_blender = python_blender
        self._minimize_blend = minimize_blend
        self._show_image_components = show_image_components
        self._async_mode = bool(async_mode)
        self._remapping_stream = None
        # Optional rotation after stitching (degrees, about image center)
        self._post_stitch_rotate_degrees: Optional[float] = post_stitch_rotate_degrees
        self._profiler = profiler
        self._config_ref: Optional[Dict[str, Any]] = config_ref
        self._left_color_pipeline_cfg: Optional[List[Dict[str, Any]]] = left_color_pipeline
        self._right_color_pipeline_cfg: Optional[List[Dict[str, Any]]] = right_color_pipeline
        self._left_color_pipeline: Optional[Compose] = None
        self._right_color_pipeline: Optional[Compose] = None
        self._max_blend_levels: Optional[int] = max_blend_levels
        self._capture_rgb_stats: bool = bool(capture_rgb_stats)

        # Optimize the roi box
        if image_roi is not None:
            if isinstance(image_roi, (list, tuple)):
                if not any(item is not None for item in image_roi):
                    image_roi = None
        self._image_roi = image_roi

        self._fps = None
        self._bitrate = None
        self._auto_configure = auto_configure
        self._num_workers = num_workers
        self._stitching_workers = {}
        self._batch_count = 0
        # Temporary until we get the middle-man (StitchingWorkersIterator)
        self._current_worker = 0
        self._ordering_queue = create_queue(mp=False, name="stitching-ordering-queue")
        self._coordinator_thread = None

        self._next_frame_timer = Timer()
        self._next_frame_counter = 0

        self._next_timer = Timer()

        self._prepare_next_frame_timer = Timer()

        # Optional per-camera channel adders (R,G,B) loaded from game config
        self._channel_add_left: Optional[List[float]] = None
        self._channel_add_right: Optional[List[float]] = None

        self._video_left_info = BasicVideoInfo(",".join(videos["left"]["files"]))
        self._video_right_info = BasicVideoInfo(",".join(videos["right"]["files"]))
        # Prefer the project directory for all stitching artifacts (mappings, masks)
        # so it stays consistent with configure_video_stitching outputs.
        if self._pto_project_file:
            self._dir_name = Path(self._pto_project_file).parent
        else:
            self._dir_name = _get_dir_name(str(videos["left"]["files"][0]))
        # This would affect number of frames, but actually it's supported
        # for stitching later if one os a modulus of the other
        assert np.isclose(
            float(self._video_left_info.fps),
            float(self._video_right_info.fps),
        )

        v1o = 0 if self._video_left_offset_frame is None else self._video_left_offset_frame
        v2o = 0 if self._video_right_offset_frame is None else self._video_right_offset_frame
        self._total_number_of_frames = int(
            min(
                self._video_left_info.frame_count - v1o,
                self._video_right_info.frame_count - v2o,
            )
        )
        self._stitcher = None
        self._video_output = None
        # In case you're debugging...
        self._shower: Optional[Shower] = None
        self._mean_tracker: Optional[MeanTracker] = None

        # Load per-side RGB adders from config (if present)
        try:
            game_id = os.path.basename(str(self._dir_name))
            cfg = get_game_config(game_id=game_id)

            def _get_adders(side: str) -> Optional[List[float]]:
                # Preferred structure:
                # game.stitching.color_adjustment.left: {r: 45, g: 35, b: 56}
                node = get_nested_value(cfg, f"game.stitching.color_adjustment.{side}")
                if isinstance(node, dict):
                    try:
                        r = float(node.get("r"))
                        g = float(node.get("g"))
                        b = float(node.get("b"))
                        return [r, g, b]
                    except Exception:
                        pass
                # Legacy fallbacks (arrays or other key names)
                candidate_keys = [
                    f"game.stitching.rgb_add.{side}",
                    f"game.stitching.channel_add.{side}",
                    f"game.stitching.image_channel_add.{side}",
                    f"game.stitching.image_channel_adders.{side}",
                    f"game.rgb_add.{side}",
                    f"game.color_add.{side}",
                ]
                for k in candidate_keys:
                    v = get_nested_value(cfg, k)
                    if v is not None:
                        try:
                            if isinstance(v, (list, tuple)) and len(v) >= 3:
                                return [float(v[0]), float(v[1]), float(v[2])]
                        except Exception:
                            pass
                # Global fallback without side (applies to both)
                for k in [
                    "game.stitching.rgb_add",
                    "game.stitching.channel_add",
                    "game.stitching.image_channel_add",
                    "game.stitching.image_channel_adders",
                    "game.rgb_add",
                    "game.color_add",
                ]:
                    v = get_nested_value(cfg, k)
                    if v is not None:
                        try:
                            if isinstance(v, (list, tuple)) and len(v) >= 3:
                                return [float(v[0]), float(v[1]), float(v[2])]
                        except Exception:
                            pass
                return None

            self._channel_add_left = _get_adders("left")
            self._channel_add_right = _get_adders("right")
            pass
        except Exception:
            # Non-fatal if config missing/malformed
            self._channel_add_left = None
            self._channel_add_right = None

        # Optional per-camera color adjustment pipelines (left/right) applied
        # before stitching. These mirror the standard inference/video_out
        # HmImageColorAdjust transform but operate on the individual streams.
        self._build_color_pipelines()

    def __delete__(self):
        if hasattr(self, "close"):
            self.close()

    @property
    def lfo(self):
        assert self._video_left_offset_frame is not None
        return self._video_left_offset_frame

    @property
    def rfo(self):
        assert self._video_right_offset_frame is not None
        return self._video_right_offset_frame

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_post_stitch_rotate_degrees(self) -> Optional[float]:
        return self._post_stitch_rotate_degrees

    def set_post_stitch_rotate_degrees(self, degrees: Optional[float]) -> None:
        self._post_stitch_rotate_degrees = degrees

    def _build_color_pipelines(self) -> None:
        """Instantiate optional left/right color pipelines from config specs."""
        self._left_color_pipeline = None
        self._right_color_pipeline = None

        def _make_pipeline(spec: Optional[List[Dict[str, Any]]]) -> Optional[Compose]:
            if not spec:
                return None
            try:
                pipeline = Compose(copy.deepcopy(spec))
                if self._config_ref is not None:
                    for tf in getattr(pipeline, "transforms", []):
                        # Bind live config so HmImageColorAdjust picks up runtime changes.
                        if tf.__class__.__name__ == "HmImageColorAdjust":
                            setattr(tf, "config_ref", self._config_ref)
                return pipeline
            except Exception:
                return None

        self._left_color_pipeline = _make_pipeline(self._left_color_pipeline_cfg)
        self._right_color_pipeline = _make_pipeline(self._right_color_pipeline_cfg)

    def stitching_worker(self, worker_number: int):
        return self._stitching_workers[worker_number]

    def create_stitching_worker(
        self,
        rank: int,
        start_frame_number: int,
        frame_stride_count: int,
        max_frames: int,
        max_input_queue_size: int,
        remapping_device: torch.device,
        dataset_name: str = "crowdhuman",
    ):
        #
        # Handle when one of the cameras is 60 fps and the other is 30 (oops)
        #
        frame_step_1 = 1
        frame_step_2 = 1

        if self._video_left_info.fps > self._video_right_info.fps:
            int_ratio = self._video_left_info.fps // self._video_right_info.fps
            float_ratio = self._video_left_info.fps / self._video_right_info.fps
            if np.isclose(float(int_ratio), float_ratio) and int_ratio != 1:
                frame_step_1 = int(int_ratio)
        elif self._video_right_info.fps > self._video_left_info.fps:
            int_ratio = self._video_right_info.fps // self._video_left_info.fps
            float_ratio = self._video_right_info.fps / self._video_left_info.fps
            if np.isclose(float(int_ratio), float_ratio) and int_ratio != 1:
                frame_step_2 = int(int_ratio)

        # TODO: must correct for lfo, which is generally calculated based upon
        # one video or the other's frame number.
        # Should turn this into a time seek instead.
        dataloaders = []
        dataloaders.append(
            MOTLoadVideoWithOrig(
                path=self._videos["left"]["files"],
                game_id=os.path.basename(str(self._dir_name)),
                max_frames=max_frames,
                batch_size=self._batch_size,
                start_frame_number=start_frame_number + self._video_left_offset_frame,
                original_image_only=True,
                dtype=torch.uint8,
                device=remapping_device,
                decoder_device=self._decoder_device,
                frame_step=frame_step_1,
                no_cuda_streams=self._no_cuda_streams,
                image_channel_adders=self._channel_add_left,
                async_mode=self._async_mode,
                checkerboard_input=self._checkerboard_input,
            )
        )
        dataloaders.append(
            MOTLoadVideoWithOrig(
                path=self._videos["right"]["files"],
                game_id=os.path.basename(str(self._dir_name)),
                max_frames=max_frames,
                batch_size=self._batch_size,
                start_frame_number=start_frame_number + self._video_right_offset_frame,
                original_image_only=True,
                dtype=torch.uint8,
                device=remapping_device,
                decoder_device=self._decoder_device,
                frame_step=frame_step_2,
                no_cuda_streams=self._no_cuda_streams,
                image_channel_adders=self._channel_add_right,
                async_mode=self._async_mode,
                checkerboard_input=self._checkerboard_input,
            )
        )
        stitching_worker = MultiDataLoaderWrapper(
            dataloaders=dataloaders, input_queueue_size=max_input_queue_size
        )
        return stitching_worker

    def configure_stitching(self):
        if self._video_left_offset_frame is None or self._video_right_offset_frame is None:
            self._pto_project_file, lfo, rfo = configure_video_stitching(
                self._dir_name,
                video_left=self._videos["left"]["files"][0],
                video_right=self._videos["right"]["files"][0],
                left_frame_offset=self._video_left_offset_frame,
                right_frame_offset=self._video_right_offset_frame,
            )
            self._video_left_offset_frame = lfo
            self._video_right_offset_frame = rfo

    def initialize(self):
        if self._auto_configure:
            self.configure_stitching()

    def _load_video_props(self):
        info = BasicVideoInfo(",".join(self._videos["left"]["files"]))
        self._fps = info.fps
        self._bitrate = info.bit_rate

    @property
    def fps(self):
        if self._fps is None:
            self._load_video_props()
        return self._fps

    @property
    def bit_rate(self):
        if self._bitrate is None:
            self._load_video_props()
        return self._bitrate

    def close(self):
        for stitching_worker in self._stitching_workers.values():
            stitching_worker.close()
        if self._async_mode:
            self._stop_coordinator_thread()
        self._stitching_workers.clear()
        if self._video_output is not None:
            self._video_output.stop()
            self._video_output = None
        if self._mean_tracker is not None:
            self._mean_tracker.close()

    @staticmethod
    def ensure_rgba(tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensures that an input image tensor is in RGBA format.

        Supports both:
        - Single image: (C, H, W)
        - Batched images: (B, C, H, W)

        Adds an alpha channel with full opacity if tensor is RGB.

        Args:
            tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            torch.Tensor: Image tensor with 4 channels (RGBA)
        """
        if tensor.ndim == 3:
            # Single image case
            tensor = tensor.unsqueeze(0)  # Convert to (1, C, H, W)
            squeezed = True
        elif tensor.ndim == 4:
            squeezed = False
        else:
            raise ValueError(
                f"Expected tensor of shape (C, H, W) or (B, C, H, W), got {tensor.shape}"
            )
        tensor = make_channels_first(tensor)
        B, C, H, W = tensor.shape
        if C == 4:
            return tensor[0] if squeezed else tensor  # Already RGBA
        elif C == 3:
            alpha = torch.empty((B, 1, H, W), dtype=tensor.dtype, device=tensor.device)
            alpha.fill_(255)
            out = torch.cat([tensor, alpha], dim=1)
            return out[0] if squeezed else out
        else:
            raise ValueError(f"Expected 3 (RGB) or 4 (RGBA) channels, got {C}")

    def _prepare_next_frame(self, frame_id: int):
        try:
            self._prepare_next_frame_timer.tic()

            stitching_worker = self._stitching_workers[self._current_worker]
            image_data_1, image_data_2 = next(stitching_worker)

            imgs_1 = image_data_1["img"]
            ids_1 = image_data_1["frame_ids"]

            imgs_2 = image_data_2["img"]
            rgb_stats: Optional[Dict[str, Any]] = None

            with torch.no_grad():

                if not self._no_cuda_streams and self._remapping_stream is None:
                    self._remapping_stream = torch.cuda.Stream(device=self._remapping_device)

                stream = self._remapping_stream if self._async_mode else None
                with cuda_stream_scope(stream), torch.no_grad():
                    imgs_1 = to_tensor(imgs_1)
                    imgs_2 = to_tensor(imgs_2)

                    pre_stats_left: Optional[Dict[str, Tuple[float, float, float]]] = None
                    pre_stats_right: Optional[Dict[str, Tuple[float, float, float]]] = None
                    if self._capture_rgb_stats:
                        try:
                            pre_stats_left = MOTLoadVideoWithOrig.compute_rgb_stats(imgs_1)
                            pre_stats_right = MOTLoadVideoWithOrig.compute_rgb_stats(imgs_2)
                        except Exception:
                            pre_stats_left = None
                            pre_stats_right = None
                    # Optional per-camera color pipelines for left/right inputs.
                    if self._left_color_pipeline is not None:
                        try:
                            data_1 = {"img": imgs_1}
                            data_1 = self._left_color_pipeline(data_1)
                            if "img" in data_1:
                                imgs_1 = data_1["img"]
                        except Exception:
                            pass
                    if self._right_color_pipeline is not None:
                        try:
                            data_2 = {"img": imgs_2}
                            data_2 = self._right_color_pipeline(data_2)
                            if "img" in data_2:
                                imgs_2 = data_2["img"]
                        except Exception:
                            pass
                    if self._stitcher is None:
                        # Lazily construct the stitcher on first use (async or sync).
                        self._create_stitcher()

                    if isinstance(self._stitcher, CudaStitchPanoF32 | CudaStitchPanoU8):

                        if self._show_image_components:
                            for img1, img2 in zip(imgs_1, imgs_2):
                                t1 = img1.clamp(min=0, max=255).to(torch.uint8).contiguous()
                                t2 = img2.clamp(min=0, max=255).to(torch.uint8).contiguous()
                                show_image(
                                    "img-1",
                                    make_visible_image(t1.clone()),
                                    wait=False,
                                    enable_resizing=0.2,
                                )
                                show_image(
                                    "img-2",
                                    make_visible_image(t2.clone()),
                                    wait=False,
                                    enable_resizing=0.2,
                                )

                        imgs_1 = make_channels_last(self.ensure_rgba(imgs_1))
                        imgs_2 = make_channels_last(self.ensure_rgba(imgs_2))
                        assert imgs_1.dtype == torch.uint8

                        blended_stream_tensor = torch.empty(
                            [
                                imgs_1.shape[0],
                                self._stitcher.canvas_height(),
                                self._stitcher.canvas_width(),
                                imgs_1.shape[-1],
                            ],
                            dtype=imgs_1.dtype,
                            device=imgs_1.device,
                        )
                        stream_handle = (
                            stream.cuda_stream
                            if stream is not None
                            else torch.cuda.current_stream(imgs_1.device).cuda_stream
                        )
                        self._stitcher.process(
                            imgs_1.contiguous(),
                            imgs_2.contiguous(),
                            blended_stream_tensor,
                            stream_handle,
                        )
                        # Optional rotation (keep same size)
                        if (
                            self._post_stitch_rotate_degrees is not None
                            and abs(self._post_stitch_rotate_degrees) > 1e-6
                        ):
                            blended_stream_tensor = self._rotate_tensor_keep_size(
                                blended_stream_tensor,
                                self._post_stitch_rotate_degrees,
                                use_cache=True,
                            )

                        if self._capture_rgb_stats:
                            try:
                                post_stats = MOTLoadVideoWithOrig.compute_rgb_stats(
                                    blended_stream_tensor
                                )
                            except Exception:
                                post_stats = None
                            rgb_stats = {
                                "left": pre_stats_left,
                                "right": pre_stats_right,
                                "stitched": post_stats,
                            }
                    else:
                        blended_stream_tensor = self._stitcher.forward(inputs=[imgs_1, imgs_2])
                        # Optional rotation (keep same size)
                        if (
                            self._post_stitch_rotate_degrees is not None
                            and abs(self._post_stitch_rotate_degrees) > 1e-6
                        ):
                            blended_stream_tensor = self._rotate_tensor_keep_size(
                                blended_stream_tensor,
                                self._post_stitch_rotate_degrees,
                                use_cache=True,
                            )

                        if self._capture_rgb_stats:
                            try:
                                post_stats = MOTLoadVideoWithOrig.compute_rgb_stats(
                                    blended_stream_tensor
                                )
                            except Exception:
                                post_stats = None
                            rgb_stats = {
                                "left": pre_stats_left,
                                "right": pre_stats_right,
                                "stitched": post_stats,
                            }

                    if self._show_image_components:
                        for blended_image in blended_stream_tensor:
                            show_image(
                                "blended",
                                make_visible_image(blended_image),
                                wait=False,
                                enable_resizing=0.2,
                            )
                    blended_stream_tensor = StreamCheckpoint(blended_stream_tensor)

            self._current_worker = (self._current_worker + 1) % len(self._stitching_workers)
            if self._capture_rgb_stats and rgb_stats is not None:
                self._ordering_queue.put((ids_1, blended_stream_tensor, rgb_stats))
            else:
                self._ordering_queue.put((ids_1, blended_stream_tensor))
            self._prepare_next_frame_timer.toc()
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                print(ex)
                traceback.print_exc()
            else:
                ex = None
            self._ordering_queue.put((ex, None))

    def _start_coordinator_thread(self):
        assert self._async_mode
        assert self._coordinator_thread is None
        for _ in range(max(1, min(self._max_input_queue_size, self._max_frames))):
            # INFO(f"putting _to_coordinator_queue.put({self._next_requested_frame})")
            self._to_coordinator_queue.put(self._next_requested_frame)
            self._from_coordinator_queue.put(("ok", self._next_requested_frame))
            self._next_requested_frame += self._batch_size
        self._coordinator_thread = threading.Thread(
            name="StitchCoordinator",
            target=self._coordinator_thread_worker,
            args=(self._next_requested_frame,),
        )
        self._coordinator_thread.start()

    def _stop_coordinator_thread(self):
        if self._coordinator_thread is not None:
            self._to_coordinator_queue.put("stop")
            self._coordinator_thread.join()
            self._coordinator_thread = None

    def _create_stitcher(self) -> None:
        if self._stitcher is not None:
            return
        assert self._remapping_device.type != "cpu"
        from hmlib.stitching.blender2 import create_stitcher

        if self._blend_mode == "laplacian":
            levels_arg = (
                int(self._max_blend_levels)
                if self._max_blend_levels is not None and self._max_blend_levels > 0
                else 11
            )
        else:
            levels_arg = 0

        self._stitcher = create_stitcher(
            dir_name=self._dir_name,
            batch_size=self._batch_size,
            left_image_size_wh=(self._video_left_info.width, self._video_left_info.height),
            right_image_size_wh=(
                self._video_right_info.width,
                self._video_right_info.height,
            ),
            device=self._remapping_device,
            dtype=self._dtype if self._python_blender else torch.uint8,
            python_blender=self._python_blender,
            minimize_blend=self._minimize_blend,
            blend_mode=self._blend_mode,
            add_alpha_channel=False,
            levels=levels_arg,
            auto_adjust_exposure=self._auto_adjust_exposure,
        )

    def _coordinator_thread_worker(self, next_requested_frame, *args, **kwargs):
        try:
            #
            # Create the stitcher
            #
            assert self._stitcher is None
            self._create_stitcher()

            frame_count = 0
            while frame_count < self._max_frames:
                command = self._to_coordinator_queue.get()
                if isinstance(command, str) and command == "stop":
                    break
                frame_id = int(command)
                self._prepare_next_frame(frame_id)
                if next_requested_frame < self._start_frame_number + self._max_frames:
                    self._from_coordinator_queue.put(("ok", next_requested_frame))
                else:
                    pass
                frame_count += self._batch_size
                next_requested_frame += self._batch_size
            safe_put_queue(self._from_coordinator_queue, StopIteration())
        except Exception as ex:
            if not isinstance(ex, StopIteration):
                print(ex)
                traceback.print_exc()
            safe_put_queue(self._from_coordinator_queue, ex)
        finally:
            pass

    @staticmethod
    def prepare_frame_for_video(image: np.array, image_roi: np.array):
        if not image_roi:
            if image.shape[-1] == 4:
                if isinstance(image, StreamTensorBase):
                    image = image.wait()
                if len(image.shape) == 4:
                    image = make_channels_last(image)[:, :, :, :3]
                else:
                    image = make_channels_last(image)[:, :, :3]
        else:
            image_roi = fix_clip_box(image_roi, [image_height(image), image_width(image)])
            if len(image.shape) == 4:
                image = make_channels_last(image)[
                    :, image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
            else:
                assert len(image.shape) == 3
                image = make_channels_last(image)[
                    image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
        return image

    def __iter__(self):
        if not self._stitching_workers:
            self.initialize()
            # Openend close to validate existance as well as get some stats, such as fps
            for worker_number in range(self._num_workers):
                max_for_worker = self._max_frames
                if max_for_worker is not None:
                    max_for_worker = distribute_items_detailed(self._max_frames, self._num_workers)[
                        worker_number
                    ]  # TODO: call just once
                worker_iter = iter(
                    self.create_stitching_worker(
                        rank=worker_number,
                        start_frame_number=self._start_frame_number,
                        frame_stride_count=self._num_workers,
                        max_frames=max_for_worker,
                        max_input_queue_size=self._max_input_queue_size,
                        remapping_device=self._remapping_device,
                    )
                )
                self._stitching_workers[worker_number] = worker_iter
            if self._async_mode:
                self._start_coordinator_thread()
        return self

    def get_next_frame(self, frame_id: int):
        self._next_frame_timer.tic()
        assert frame_id == self._current_frame
        # INFO(f"Dequeing frame id: {self._current_frame}...")
        # stitched_frame = self._ordering_queue.dequeue_key(self._current_frame)
        if self._async_mode:
            rctx = (
                self._profiler.rf("stitch.dequeue")
                if getattr(self._profiler, "enabled", False)
                else contextlib.nullcontext()
            )
            with rctx:
                payload = self._ordering_queue.get()
        else:
            # Synchronous mode: directly compute the frame on the current thread,
            # reusing the same queue-based contract via a temporary queue.
            rctx = (
                self._profiler.rf("stitch.dequeue")
                if getattr(self._profiler, "enabled", False)
                else contextlib.nullcontext()
            )
            # tmp_queue = create_queue(mp=False)
            # orig_queue = self._ordering_queue
            # self._ordering_queue = tmp_queue
            try:
                with rctx:
                    self._prepare_next_frame(frame_id)
                payload = self._ordering_queue.get()
            finally:
                # self._ordering_queue = orig_queue
                pass

        # Unpack queue payload: (ids, frame) or (ids, frame, rgb_stats)
        rgb_stats: Optional[Dict[str, Any]] = None
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, tuple):
            if len(payload) == 2:
                frame_ids, stitched_frame = payload
            elif len(payload) == 3:
                frame_ids, stitched_frame, rgb_stats = payload
            else:
                raise ValueError(f"Unexpected payload from stitching worker: {payload!r}")
            frame_id = frame_ids
        else:
            raise ValueError(f"Unexpected payload type from stitching worker: {type(payload)}")

        if stitched_frame is not None:
            if self._async_mode:
                # INFO(f"Locally dequeued frame id: {self._current_frame}")
                if (
                    not self._max_frames
                    or self._next_requested_frame < self._start_frame_number + self._max_frames
                ):
                    # INFO(f"putting _to_coordinator_queue.put({self._next_requested_frame})")
                    self._to_coordinator_queue.put(self._next_requested_frame)
                    self._next_requested_frame += self._batch_size
                else:
                    # We were pre-requesting future frames, but we're past the
                    # frames we want, so don't ask for anymore and just return these
                    # (running out what's in the queue)
                    # INFO(
                    #     f"Next frame {self._next_requested_frame} would be above the max allowed frames, so not queueing"
                    # )
                    pass

            self._next_frame_timer.toc()
            self._next_frame_counter += 1
        else:
            # No more frames
            pass
        if self._capture_rgb_stats:
            return stitched_frame, rgb_stats
        return stitched_frame

    def __next__(self):
        # INFO(f"\nBEGIN next() self._from_coordinator_queue.get() {self._current_frame}")
        # print(f"self._from_coordinator_queue size: {self._from_coordinator_queue.qsize()}")
        self._next_timer.tic()
        if self._async_mode:
            qctx = (
                self._profiler.rf("stitch.queue_receive")
                if getattr(self._profiler, "enabled", False)
                else contextlib.nullcontext()
            )
            with qctx:
                status = self._from_coordinator_queue.get()
            # INFO(f"END next() self._from_coordinator_queue.get( {self._current_frame})\n")
            if isinstance(status, Exception):
                self.close()
                raise status
            else:
                status, frame_id = status
                assert status == "ok"
                # print(f"self._from_coordinator_queue.get() = {frame_id}, self._current_frame = {self._current_frame} ")
                assert frame_id == self._current_frame
        else:
            # In synchronous mode, we generate the next frame directly.
            frame_id = self._current_frame

        # self._next_timer.tic()
        nctx = (
            self._profiler.rf("stitch.get_next_frame")
            if getattr(self._profiler, "enabled", False)
            else contextlib.nullcontext()
        )
        with nctx:
            next_result = self.get_next_frame(frame_id=frame_id)

        if self._capture_rgb_stats:
            stitched_frame, rgb_stats = next_result
        else:
            stitched_frame = next_result
            rgb_stats = None

        # show_image("stitched_frame", stitched_frame.get(), wait=True)
        if stitched_frame is None:
            self.close()
            raise StopIteration()

        self._batch_count += 1

        # Code doesn't handle strided channels efficiently
        pctx = (
            self._profiler.rf("stitch.prepare_frame")
            if getattr(self._profiler, "enabled", False)
            else contextlib.nullcontext()
        )
        with pctx:
            stitched_frame = self.prepare_frame_for_video(
                stitched_frame,
                image_roi=self._image_roi,
            )

        if self._batch_count == 1:
            frame_path = os.path.join(self._dir_name, "s.png")
            print(
                f"Stitched frame resolution: {image_width(stitched_frame)} x {image_height(stitched_frame)}"
            )
            print(f"Saving first stitched frame to {frame_path}")
            if isinstance(stitched_frame, StreamTensorBase):
                stitched_frame = stitched_frame.get()
            cv2.imwrite(frame_path, make_visible_image(stitched_frame[0], force_numpy=True))
            if self._on_first_stitched_image_callback is not None:
                self._on_first_stitched_image_callback(stitched_frame[0])

        assert stitched_frame.ndim == 4
        # maybe nested batches can be some multiple of, so can remove this check if necessary
        assert self._batch_size == stitched_frame.shape[0]
        self._current_frame += stitched_frame.shape[0]
        self._next_timer.toc()

        if self._verbose and self._batch_count % 50 == 0:
            logger.info(
                "Stitching dataset __next__ wait speed {} ({:.2f} fps)".format(
                    self._current_frame,
                    self._batch_size * 1.0 / max(1e-5, self._next_timer.average_time),
                )
            )

        # show_image("stitched_frame", stitched_frame.get(), wait=False)
        # for img in stitched_frame:
        #     show_cuda_tensor("stitched_frame", make_channels_last(img), wait=False)

        if self._capture_rgb_stats:
            return stitched_frame, rgb_stats
        return stitched_frame

    def __len__(self):
        return self._total_number_of_frames // self._batch_size

    def _rotate_tensor_keep_size(
        self,
        tensor: torch.Tensor,
        degrees: float,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Rotate a batched image tensor about its center by `degrees` while keeping dimensions.

        Args:
            tensor: 4D image tensor of shape (B, H, W, C) or (B, C, H, W).
            degrees: Rotation in degrees about the image center.
            use_cache: If True, cache small shape-dependent tensors (center, S, S_inv)
                across calls. Large per-frame tensors (like the sampling grid) are
                intentionally *not* cached to avoid excessive memory usage.

        Notes:
            - Works on CUDA or CPU tensors without host roundtrips.
            - All dtype/device conversions use non-blocking transfers where applicable.
        """

        if tensor is None:
            return tensor
        if degrees is None or abs(degrees) < 1e-6:
            return tensor

        # --- Determine layout ---
        assert tensor.ndim == 4, f"Expected 4D tensor, got {tensor.shape}"
        was_channels_last = tensor.shape[-1] in (1, 3, 4)
        orig_dtype = tensor.dtype
        device = tensor.device

        # Convert to NCHW
        x = tensor.permute(0, 3, 1, 2) if was_channels_last else tensor
        B, C, H, W = x.shape
        x_work = x.to(dtype=torch.float32, non_blocking=True)

        # --- Setup persistent cache fingerprint ---
        extras = {"op": "rotate_tensor_keep_size"}
        if use_cache and hasattr(self, "_persist_init_or_assert"):
            self._persist_init_or_assert(True, x_work, extras)

        # --- Angle-dependent tensors ---
        angle = make_const_tensor(-degrees * math.pi / 180.0, device=device, dtype=torch.float32)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # --- Cached tensors that only depend on shape/type ---
        def make_center():
            # Single tensor so PersistCacheMixin can safely detach/cache
            return torch.tensor(
                [(W - 1) / 2.0, (H - 1) / 2.0],
                device=device,
                dtype=torch.float32,
            )

        if use_cache and hasattr(self, "_persist_get"):
            center = self._persist_get("center", make_center, True)
        else:
            center = make_center()
        cx, cy = center[0], center[1]

        def make_S() -> torch.Tensor:
            return torch.tensor(
                [
                    [(W - 1) / 2.0, 0.0, (W - 1) / 2.0],
                    [0.0, (H - 1) / 2.0, (H - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )

        def make_S_inv() -> torch.Tensor:
            return torch.tensor(
                [
                    [2.0 / (W - 1), 0.0, -1.0],
                    [0.0, 2.0 / (H - 1), -1.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )

        def make_001() -> torch.Tensor:
            return torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

        if use_cache and hasattr(self, "_persist_get"):
            S = self._persist_get("S", make_S, True)
            S_inv = self._persist_get("S_inv", make_S_inv, True)
            S_001 = self._persist_get("S_001", make_001, True)
        else:
            S = make_S()
            S_inv = make_S_inv()
            S_001 = make_001()

        # --- Compute transform matrices ---
        tx = (1.0 - cos_a) * cx - sin_a * cy
        ty = sin_a * cx + (1.0 - cos_a) * cy
        M_inv = torch.stack(
            [
                torch.stack([cos_a, sin_a, tx]),
                torch.stack([-sin_a, cos_a, ty]),
                S_001,
            ]
        )

        A = S_inv @ M_inv @ S  # 3x3
        theta = A[:2, :].unsqueeze(0).repeat(B, 1, 1)  # Bx2x3

        # --- Cached affine grid ---
        def make_grid():
            # Do not cache the full sampling grid: it scales with H*W and can
            # consume significant memory for large panoramas. Recomputing it
            # per call is a reasonable trade-off for memory usage.
            return F.affine_grid(theta, size=(B, C, H, W), align_corners=True)

        grid = make_grid()

        # --- Rotate using grid_sample ---
        y = F.grid_sample(x_work, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        # --- Restore dtype/layout ---
        if orig_dtype == torch.uint8:
            y = y.clamp(min=0.0, max=255.0).to(dtype=torch.uint8)
        else:
            y = y.to(dtype=orig_dtype)

        if was_channels_last:
            y = y.permute(0, 2, 3, 1)

        return y


def is_none(val: Any) -> bool:
    if isinstance(val, str) and val == "None":
        return True
    return val is None


def fix_clip_box(clip_box: Any, hw: List[int]) -> Any:
    if isinstance(clip_box, list):
        if is_none(clip_box[0]):
            clip_box[0] = 0
        if is_none(clip_box[1]):
            clip_box[1] = 0
        if is_none(clip_box[2]):
            clip_box[2] = hw[1]
        if is_none(clip_box[3]):
            clip_box[3] = hw[0]
        clip_box = np.array(clip_box, dtype=np.int64)
    return clip_box

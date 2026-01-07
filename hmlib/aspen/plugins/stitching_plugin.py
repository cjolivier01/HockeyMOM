"""Aspen plugin that stitches multi-camera inputs into a panorama frame."""

from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import Compose

from hmlib.config import get_game_config, get_nested_value
from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.log import logger
from hmlib.stitching.blender2 import create_stitcher
from hmlib.utils.gpu import (
    StreamCheckpoint,
    StreamTensorBase,
    cuda_stream_scope,
    unwrap_tensor,
    wrap_tensor,
)
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
    resize_image,
)
from hockeymom.core import CudaStitchPanoF32, CudaStitchPanoU8

from .base import Plugin


def _to_tensor(tensor: Union[torch.Tensor, StreamTensorBase, np.ndarray]) -> torch.Tensor:
    tensor = unwrap_tensor(tensor)
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    raise TypeError(f"Unexpected tensor type: {type(tensor)}")


def _parse_dtype(dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in ("float", "float32", "fp32"):
            return torch.float32
        if key in ("float16", "fp16", "half"):
            return torch.float16
        if key in ("uint8", "u8"):
            return torch.uint8
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


class StitchingPlugin(Plugin):
    """Aspen plugin that stitches left/right camera frames into a panorama.

    Expects in context:
      - stitch_inputs: list/tuple of two dicts (left/right) from MultiDataLoaderWrapper
      - stitch_data_pipeline: optional mmcv.Compose pipeline to build detection inputs

    Produces in context:
      - data: dict prepared for downstream detection/tracking
      - original_images: stitched frame tensor (channels-first)
      - ids, frame_ids, info_imgs, frame_id
      - img: stitched frame (channels-last) for video output plugins
    """

    def __init__(
        self,
        enabled: bool = True,
        pto_project_file: Optional[str] = None,
        dir_name: Optional[str] = None,
        blend_mode: str = "laplacian",
        auto_adjust_exposure: bool = False,
        python_blender: bool = False,
        minimize_blend: bool = True,
        dtype: Optional[Union[str, torch.dtype]] = None,
        max_blend_levels: Optional[int] = None,
        no_cuda_streams: bool = False,
        post_stitch_rotate_degrees: Optional[float] = None,
        left_color_pipeline: Optional[List[Dict[str, Any]]] = None,
        right_color_pipeline: Optional[List[Dict[str, Any]]] = None,
        capture_rgb_stats: bool = False,
        max_output_width: Optional[int] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self._pto_project_file = pto_project_file
        self._dir_name = Path(dir_name) if dir_name else None
        if self._pto_project_file and not self._dir_name:
            try:
                self._dir_name = Path(self._pto_project_file).parent
            except Exception:
                self._dir_name = None
        self._blend_mode = str(blend_mode)
        self._auto_adjust_exposure = bool(auto_adjust_exposure)
        self._python_blender = bool(python_blender)
        self._minimize_blend = bool(minimize_blend)
        self._dtype = _parse_dtype(dtype)
        self._max_blend_levels = max_blend_levels
        self._no_cuda_streams = bool(no_cuda_streams)
        self._post_stitch_rotate_degrees = post_stitch_rotate_degrees
        self._left_color_pipeline_cfg = left_color_pipeline
        self._right_color_pipeline_cfg = right_color_pipeline
        self._capture_rgb_stats = bool(capture_rgb_stats)
        self._max_output_width = int(max_output_width) if max_output_width else None

        self._left_color_pipeline: Optional[Compose] = None
        self._right_color_pipeline: Optional[Compose] = None
        self._config_ref: Optional[Dict[str, Any]] = None
        self._initialized: bool = False
        self._stitcher = None
        self._rotate_cache: Dict[Tuple[int, int, torch.device], Dict[str, torch.Tensor]] = {}
        self._width_t: Optional[torch.Tensor] = None
        self._height_t: Optional[torch.Tensor] = None
        self._channel_add_left: Optional[List[float]] = None
        self._channel_add_right: Optional[List[float]] = None
        self._iter_num: int = 0

    def _ensure_initialized(self, context: Dict[str, Any]) -> None:
        if self._initialized:
            return
        shared = context.get("shared", {}) if isinstance(context, dict) else {}
        cfg = shared.get("game_config")
        if cfg is None:
            game_id = shared.get("game_id")
            if game_id:
                try:
                    cfg = get_game_config(game_id=game_id)
                except Exception:
                    cfg = None
        if isinstance(cfg, dict):
            self._config_ref = cfg
        self._build_color_pipelines()
        self._resolve_channel_adders()
        self._initialized = True

    def _build_color_pipelines(self) -> None:
        self._left_color_pipeline = None
        self._right_color_pipeline = None

        def _make_pipeline(spec: Optional[List[Dict[str, Any]]]) -> Optional[Compose]:
            if not spec:
                return None
            try:
                pipeline = Compose(copy.deepcopy(spec))
                if self._config_ref is not None:
                    for tf in getattr(pipeline, "transforms", []):
                        if tf.__class__.__name__ == "HmImageColorAdjust":
                            setattr(tf, "config_ref", self._config_ref)
                return pipeline
            except Exception:
                logger.debug("StitchingPlugin failed to build color pipeline", exc_info=True)
                return None

        self._left_color_pipeline = _make_pipeline(self._left_color_pipeline_cfg)
        self._right_color_pipeline = _make_pipeline(self._right_color_pipeline_cfg)

    def _resolve_channel_adders(self) -> None:
        cfg = self._config_ref
        if not isinstance(cfg, dict):
            return

        def _get_adders(side: str) -> Optional[List[float]]:
            node = get_nested_value(cfg, f"game.stitching.color_adjustment.{side}")
            if isinstance(node, dict):
                try:
                    return [float(node.get("r")), float(node.get("g")), float(node.get("b"))]
                except Exception:
                    pass
            candidate_keys = [
                f"game.stitching.rgb_add.{side}",
                f"game.stitching.channel_add.{side}",
                f"game.stitching.image_channel_add.{side}",
                f"game.stitching.image_channel_adders.{side}",
                f"game.rgb_add.{side}",
                f"game.color_add.{side}",
            ]
            for key in candidate_keys:
                val = get_nested_value(cfg, key)
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    try:
                        return [float(val[0]), float(val[1]), float(val[2])]
                    except Exception:
                        pass
            for key in [
                "game.stitching.rgb_add",
                "game.stitching.channel_add",
                "game.stitching.image_channel_add",
                "game.stitching.image_channel_adders",
                "game.rgb_add",
                "game.color_add",
            ]:
                val = get_nested_value(cfg, key)
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    try:
                        return [float(val[0]), float(val[1]), float(val[2])]
                    except Exception:
                        pass
            return None

        def _to_bgr(adders: Optional[List[float]]) -> Optional[List[float]]:
            if not adders:
                return None
            return [adders[2], adders[1], adders[0]]

        self._channel_add_left = _to_bgr(_get_adders("left"))
        self._channel_add_right = _to_bgr(_get_adders("right"))

    def _apply_channel_adders(
        self, img: torch.Tensor, adders: Optional[List[float]]
    ) -> torch.Tensor:
        if not adders:
            return img
        was_channels_last = img.ndim in (3, 4) and img.shape[-1] in (1, 3, 4)
        t = make_channels_first(img)
        orig_dtype = t.dtype
        if not torch.is_floating_point(t):
            t = t.to(torch.float16, non_blocking=True)
        add = torch.tensor(adders, dtype=t.dtype, device=t.device)
        if t.ndim == 4:
            add = add.view(1, 3, 1, 1)
        else:
            add = add.view(3, 1, 1)
        if t.ndim == 4:
            t[:, 0:3, :, :] += add
        else:
            t[0:3, :, :] += add
        t.clamp_(0.0, 255.0)
        if orig_dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            t = t.to(dtype=orig_dtype)
        if was_channels_last:
            return make_channels_last(t)
        return t

    def _resolve_dir_name(self, context: Dict[str, Any]) -> Path:
        if self._dir_name is not None:
            return self._dir_name
        if self._pto_project_file:
            try:
                self._dir_name = Path(self._pto_project_file).parent
                return self._dir_name
            except Exception:
                pass
        shared = context.get("shared", {})
        game_dir = shared.get("game_dir")
        if game_dir:
            self._dir_name = Path(game_dir)
            return self._dir_name
        raise RuntimeError("StitchingPlugin needs dir_name or pto_project_file")

    def _ensure_rgba(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = make_channels_first(tensor)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        b, c, h, w = tensor.shape
        if c == 4:
            return tensor[0] if squeezed else tensor
        if c == 3:
            alpha = torch.empty((b, 1, h, w), dtype=tensor.dtype, device=tensor.device)
            alpha.fill_(255)
            out = torch.cat([tensor, alpha], dim=1)
            return out[0] if squeezed else out
        raise ValueError(f"Expected 3 or 4 channels, got {c}")

    def _create_stitcher(
        self, context: Dict[str, Any], imgs_1: torch.Tensor, imgs_2: torch.Tensor, device: torch.device
    ) -> None:
        if self._stitcher is not None:
            return
        if device.type == "cpu":
            raise RuntimeError("StitchingPlugin requires a CUDA device")
        dir_name = self._resolve_dir_name(context)
        if self._blend_mode == "laplacian":
            levels_arg = (
                int(self._max_blend_levels)
                if self._max_blend_levels is not None and self._max_blend_levels > 0
                else 11
            )
        else:
            levels_arg = 0
        left_size_wh = (image_width(imgs_1), image_height(imgs_1))
        right_size_wh = (image_width(imgs_2), image_height(imgs_2))
        dtype = self._dtype if self._python_blender else torch.uint8
        self._stitcher = create_stitcher(
            dir_name=str(dir_name),
            batch_size=int(imgs_1.shape[0]),
            left_image_size_wh=left_size_wh,
            right_image_size_wh=right_size_wh,
            device=device,
            dtype=dtype,
            python_blender=self._python_blender,
            use_cuda_pano=not self._python_blender,
            minimize_blend=self._minimize_blend,
            blend_mode=self._blend_mode,
            add_alpha_channel=False,
            levels=levels_arg,
            auto_adjust_exposure=self._auto_adjust_exposure,
        )

    def _resolve_rotation_degrees(self, context: Dict[str, Any]) -> Optional[float]:
        shared = context.get("shared", {})
        cfg = shared.get("game_config") or {}
        try:
            val = get_nested_value(cfg, "game.stitching.stitch-rotate-degrees", None)
            if val is None:
                val = get_nested_value(cfg, "game.stitching.stitch_rotate_degrees", None)
            if val is not None:
                return float(val)
        except Exception:
            pass
        return self._post_stitch_rotate_degrees

    def _prepare_frame_for_video(
        self, image: torch.Tensor, image_roi: Optional[List[int]] = None
    ) -> torch.Tensor:
        if not image_roi:
            if image.shape[-1] == 4:
                image = make_channels_last(image)[:, :, :, :3]
        else:
            image_roi = _fix_clip_box(image_roi, [image_height(image), image_width(image)])
            if image.ndim == 4:
                image = make_channels_last(image)[
                    :, image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
            else:
                image = make_channels_last(image)[
                    image_roi[1] : image_roi[3], image_roi[0] : image_roi[2], :3
                ]
        return image

    def _rotate_tensor_keep_size(
        self,
        tensor: torch.Tensor,
        degrees: float,
    ) -> torch.Tensor:
        if tensor is None or degrees is None or abs(degrees) < 1e-6:
            return tensor
        assert tensor.ndim == 4
        was_channels_last = tensor.shape[-1] in (1, 3, 4)
        orig_dtype = tensor.dtype
        device = tensor.device

        x = tensor.permute(0, 3, 1, 2) if was_channels_last else tensor
        b, c, h, w = x.shape
        x_work = x.to(dtype=torch.float32, non_blocking=True)

        cache_key = (int(h), int(w), device)
        cache = self._rotate_cache.get(cache_key)
        if cache is None:
            center = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], device=device, dtype=torch.float32)
            s = torch.tensor(
                [
                    [(w - 1) / 2.0, 0.0, (w - 1) / 2.0],
                    [0.0, (h - 1) / 2.0, (h - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            s_inv = torch.tensor(
                [
                    [2.0 / (w - 1), 0.0, -1.0],
                    [0.0, 2.0 / (h - 1), -1.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            s_001 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            cache = {"center": center, "s": s, "s_inv": s_inv, "s_001": s_001}
            self._rotate_cache[cache_key] = cache

        angle = torch.zeros((), device=device, dtype=torch.float32) + (-degrees * math.pi / 180.0)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        cx, cy = cache["center"][0], cache["center"][1]
        tx = (1.0 - cos_a) * cx - sin_a * cy
        ty = sin_a * cx + (1.0 - cos_a) * cy
        m_inv = torch.stack(
            [
                torch.stack([cos_a, sin_a, tx]),
                torch.stack([-sin_a, cos_a, ty]),
                cache["s_001"],
            ]
        )
        a = cache["s_inv"] @ m_inv @ cache["s"]
        theta = a[:2, :].unsqueeze(0).repeat(b, 1, 1)
        grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
        y = F.grid_sample(x_work, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        if orig_dtype == torch.uint8:
            y = y.clamp(min=0.0, max=255.0).to(dtype=torch.uint8)
        else:
            y = y.to(dtype=orig_dtype)
        if was_channels_last:
            y = y.permute(0, 2, 3, 1)
        return y

    def _maybe_resize_output(self, img: torch.Tensor) -> torch.Tensor:
        max_w = self._max_output_width
        if not max_w or max_w <= 0:
            return img
        width = int(image_width(img))
        height = int(image_height(img))
        if width <= max_w:
            return img
        scale = float(max_w) / float(width)
        new_w = max_w
        new_h = int(height * scale)
        if new_w % 2 != 0:
            new_w -= 1
        if new_h % 2 != 0:
            new_h -= 1
        return resize_image(img, new_width=new_w, new_height=new_h)

    def __call__(self, *args, **kwds):
        self._iter_num += 1
        # do_trace = self._iter_num == 4
        # if do_trace:
        #     pass
        # from cuda_stacktrace import CudaStackTracer

        # with CudaStackTracer(functions="cudaStreamSynchronize", enabled=do_trace):
        with contextlib.nullcontext():
            results = super().__call__(*args, **kwds)
        # if do_trace:
        #     pass
        return results

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        stitch_inputs = context.get("stitch_inputs")
        if stitch_inputs is None:
            raise RuntimeError("StitchingPlugin requires 'stitch_inputs' in context")
        if isinstance(stitch_inputs, dict):
            image_data_1 = stitch_inputs.get("left")
            image_data_2 = stitch_inputs.get("right")
        else:
            image_data_1, image_data_2 = stitch_inputs
        if image_data_1 is None or image_data_2 is None:
            raise RuntimeError("StitchingPlugin needs left/right input data")

        self._ensure_initialized(context)

        imgs_1 = _to_tensor(image_data_1["img"])
        imgs_2 = _to_tensor(image_data_2["img"])
        ids = _first_not_none(
            image_data_1.get("frame_ids"),
            image_data_1.get("ids"),
            image_data_1.get("img_id"),
        )
        if isinstance(ids, (list, tuple)):
            ids = torch.tensor(ids, dtype=torch.int64)
        if isinstance(ids, torch.Tensor) and ids.is_cuda:
            ids = ids.detach().cpu()
        if ids is None:
            raise RuntimeError("StitchingPlugin could not resolve frame ids")

        imgs_1 = self._apply_channel_adders(imgs_1, self._channel_add_left)
        imgs_2 = self._apply_channel_adders(imgs_2, self._channel_add_right)

        pre_stats_left: Optional[Dict[str, Tuple[float, float, float]]] = None
        pre_stats_right: Optional[Dict[str, Tuple[float, float, float]]] = None
        if self._capture_rgb_stats:
            try:
                pre_stats_left = MOTLoadVideoWithOrig.compute_rgb_stats(imgs_1)
                pre_stats_right = MOTLoadVideoWithOrig.compute_rgb_stats(imgs_2)
            except Exception:
                pre_stats_left = None
                pre_stats_right = None

        if self._left_color_pipeline is not None:
            try:
                data_1 = {"img": imgs_1}
                data_1 = self._left_color_pipeline(data_1)
                if "img" in data_1:
                    imgs_1 = data_1["img"]
            except Exception:
                logger.debug("Left stitch color pipeline failed", exc_info=True)
        if self._right_color_pipeline is not None:
            try:
                data_2 = {"img": imgs_2}
                data_2 = self._right_color_pipeline(data_2)
                if "img" in data_2:
                    imgs_2 = data_2["img"]
            except Exception:
                logger.debug("Right stitch color pipeline failed", exc_info=True)

        device = imgs_1.device
        self._create_stitcher(context=context, imgs_1=imgs_1, imgs_2=imgs_2, device=device)
        if isinstance(self._stitcher, (CudaStitchPanoF32, CudaStitchPanoU8)):
            imgs_1 = make_channels_last(self._ensure_rgba(imgs_1))
            imgs_2 = make_channels_last(self._ensure_rgba(imgs_2))
            blended = torch.empty(
                [
                    imgs_1.shape[0],
                    self._stitcher.canvas_height(),
                    self._stitcher.canvas_width(),
                    imgs_1.shape[-1],
                ],
                dtype=imgs_1.dtype,
                device=imgs_1.device,
            )
            stream_handle = torch.cuda.current_stream(imgs_1.device).cuda_stream
            self._stitcher.process(
                imgs_1.contiguous(),
                imgs_2.contiguous(),
                blended,
                stream_handle,
            )
        else:
            blended = self._stitcher.forward(inputs=[imgs_1, imgs_2])

        rotate_degrees = self._resolve_rotation_degrees(context)
        if rotate_degrees is not None and abs(rotate_degrees) > 1e-6:
            blended = self._rotate_tensor_keep_size(blended, rotate_degrees)

        rgb_stats: Optional[Dict[str, Any]] = None
        if self._capture_rgb_stats:
            try:
                post_stats = MOTLoadVideoWithOrig.compute_rgb_stats(blended)
            except Exception:
                post_stats = None
            rgb_stats = {"left": pre_stats_left, "right": pre_stats_right, "stitched": post_stats}

        blended = self._prepare_frame_for_video(blended, image_roi=None)
        blended = make_channels_last(blended)

        stitched_for_output = self._maybe_resize_output(blended)

        if self._width_t is None or self._height_t is None:
            self._width_t = torch.tensor([image_width(blended)], dtype=torch.int64)
            self._height_t = torch.tensor([image_height(blended)], dtype=torch.int64)

        data_item: Dict[str, Any] = {}
        existing_data = context.get("data")
        if isinstance(existing_data, dict):
            data_item.update(existing_data)
        fps_value = context.get("stitch_fps") or context.get("fps")
        if fps_value is None and isinstance(existing_data, dict):
            fps_value = existing_data.get("fps")
        if fps_value:
            try:
                fps_scalar = float(fps_value)
                data_item["hm_real_time_fps"] = [fps_scalar for _ in range(len(ids))]
                data_item["fps"] = fps_scalar
            except Exception:
                pass

        data_item.update(
            dict(
                img=blended,
                img_info=dict(frame_id=int(ids[0].item()) if isinstance(ids, torch.Tensor) else int(ids[0])),
                img_prefix=None,
                img_id=ids,
            )
        )
        if rgb_stats is not None:
            data_item.setdefault("debug_rgb_stats", {})["stitch"] = rgb_stats

        pipeline = context.get("stitch_data_pipeline")
        if pipeline is not None:
            data_item = data_item.copy()
            pipeline_input = data_item.copy()
            pipeline_input.pop("dataset_results", None)
            pipeline_result = pipeline(pipeline_input)
            if "img" in pipeline_result:
                raise RuntimeError("StitchingPlugin pipeline returned unexpected 'img' key")
            data_item["img"] = pipeline_result.pop("inputs")
            data_item.update(pipeline_result)
            data_item["img"] = wrap_tensor(tensor=data_item["img"])

        original_images = make_channels_first(blended)
        original_images = wrap_tensor(original_images)
        data_item.update(
            dict(
                original_images=original_images,
                img=wrap_tensor(data_item["img"]),
                imgs_info=[
                    self._height_t.repeat(len(ids)),
                    self._width_t.repeat(len(ids)),
                    ids,
                    torch.tensor([1], dtype=torch.int32).repeat(len(ids)),
                    [str(self._resolve_dir_name(context))],
                ],
                ids=ids,
            )
        )

        return {
            "data": data_item,
            "original_images": original_images,
            "ids": ids,
            "frame_ids": ids,
            "info_imgs": data_item["img_info"],
            "frame_id": int(ids[0].item()) if isinstance(ids, torch.Tensor) else int(ids[0]),
            "img": wrap_tensor(stitched_for_output),
        }

    def input_keys(self) -> set:
        return {"stitch_inputs", "stitch_data_pipeline", "stitch_fps"}

    def output_keys(self) -> set:
        return {
            "data",
            "original_images",
            "ids",
            "frame_ids",
            "info_imgs",
            "frame_id",
            "img",
        }

    def get_post_stitch_rotate_degrees(self) -> Optional[float]:
        return self._post_stitch_rotate_degrees

    def set_post_stitch_rotate_degrees(self, degrees: Optional[float]) -> None:
        self._post_stitch_rotate_degrees = degrees


def _is_none(val: Any) -> bool:
    if isinstance(val, str) and val == "None":
        return True
    return val is None


def _fix_clip_box(clip_box: Any, hw: List[int]) -> Any:
    if isinstance(clip_box, list):
        if _is_none(clip_box[0]):
            clip_box[0] = 0
        if _is_none(clip_box[1]):
            clip_box[1] = 0
        if _is_none(clip_box[2]):
            clip_box[2] = hw[1]
        if _is_none(clip_box[3]):
            clip_box[3] = hw[0]
        clip_box = np.array(clip_box, dtype=np.int64)
    return clip_box

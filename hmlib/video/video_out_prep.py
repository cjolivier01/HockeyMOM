"""Pure video-output preparation utilities without sink-side I/O ownership."""

from __future__ import absolute_import, division, print_function

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from hmlib.log import logger
from hmlib.utils.cuda_graph import CudaGraphCallable
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hmlib.utils.image import (
    image_height,
    image_width,
    make_channels_first,
    make_channels_last,
    resize_image,
    to_uint8_image,
)
from hmlib.video.video_stream import MAX_NEVC_VIDEO_WIDTH

from .video_out import _FP_TYPES, get_best_codec, is_number


class VideoOutputPreparer(torch.nn.Module):
    """Prepare final video tensors for writing or display without performing I/O."""

    VIDEO_DEFAULT: str = "default"
    VIDEO_END_ZONES: str = "end_zones"

    def __init__(
        self,
        *,
        fourcc: str = "auto",
        skip_final_save: bool = False,
        clip_to_max_dimensions: bool = True,
        dtype: torch.dtype | None = None,
        device: Union[torch.device, str, None] = None,
        output_width: int | str | None = None,
        output_height: int | str | None = None,
        name: str = "",
    ) -> None:
        """Construct a reusable pure frame-preparation helper.

        Unlike :class:`hmlib.video.video_out.VideoOutput`, this class owns no
        writers, UI windows, muxing state, or frame-id bookkeeping. It only
        handles output geometry resolution, device placement, and optional
        CUDA-graphable tensor transforms.

        @param fourcc: Preferred codec hint used only for device/codec
                       compatibility decisions during prep initialization.
        @param skip_final_save: When True, avoid sink-oriented device moves and
                                preserve the incoming tensor device when possible.
        @param clip_to_max_dimensions: Reserved compatibility flag kept in sync
                                       with :class:`VideoOutput`.
        @param dtype: Preferred floating-point dtype for internal tensors.
        @param device: Preferred prep/output device.
        @param output_width: Optional target output width.
        @param output_height: Optional target output height.
        @param name: Human-readable label used in logs.
        """
        super().__init__()
        self._allow_scaling = False
        self._clip_to_max_dimensions = clip_to_max_dimensions
        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        assert self._dtype in _FP_TYPES
        self._output_width = output_width
        self._output_height = output_height
        self._output_resize_wh: Optional[Tuple[int, int]] = None
        self._output_canvas_wh: Optional[Tuple[int, int]] = None
        if device is not None:
            self._device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self._device = None
        self._name = name
        self._skip_final_save = bool(skip_final_save)
        self._fourcc = fourcc
        self._cuda_graph_enabled: bool = False
        self._img_prepare_cg: Optional[CudaGraphCallable] = None
        self._img_prepare_cg_signature: Optional[
            Tuple[torch.device, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]
        ] = None
        self._end_zone_prepare_cg: Optional[CudaGraphCallable] = None
        self._end_zone_prepare_cg_signature: Optional[
            Tuple[torch.device, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]
        ] = None

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    @property
    def fourcc(self) -> str:
        return str(self._fourcc)

    def _parse_output_dim(self, value: Any, dim_label: str, natural: int) -> Optional[int]:
        if value is None:
            return None
        if not is_number(value):
            key = str(value).strip().lower()
            if key == "auto":
                return None
            if dim_label == "width":
                if key == "4k":
                    return 4096
                if key == "8k":
                    return 8192
            if dim_label == "height":
                if key == "4k":
                    return 2160
                if key == "8k":
                    return 4320
            raise ValueError(f'Invalid output {dim_label}: "{value}"')
        return int(value)

    def _coerce_even(self, value: int, label: str) -> int:
        if value % 2 != 0:
            adjusted = value + 1
            logger.warning(
                "output_%s=%d is not even; adjusting to %d for YUV420 encoders",
                label,
                value,
                adjusted,
            )
            return adjusted
        return value

    def _coerce_even_down(self, value: int, label: str) -> int:
        if value % 2 == 0:
            return value
        adjusted = value - 1 if value > 2 else value + 1
        logger.warning(
            "output_%s=%d is not even; adjusting to %d for YUV420 encoders",
            label,
            value,
            adjusted,
        )
        return adjusted

    @staticmethod
    def _is_auto_output_dim(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() == "auto"
        return False

    def _get_auto_resize_and_canvas(
        self, width: int, height: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        if self._skip_final_save or not self._clip_to_max_dimensions:
            return None, None
        if not self._is_auto_output_dim(self._output_width) or not self._is_auto_output_dim(
            self._output_height
        ):
            return None, None

        resize_w = int(width)
        resize_h = int(height)
        if resize_w > MAX_NEVC_VIDEO_WIDTH:
            scale = float(MAX_NEVC_VIDEO_WIDTH) / float(resize_w)
            resize_w = MAX_NEVC_VIDEO_WIDTH
            resize_h = int(float(resize_h) * scale)

        resize_w = self._coerce_even_down(resize_w, "width")
        resize_h = self._coerce_even_down(resize_h, "height")
        if resize_w == int(width) and resize_h == int(height):
            return None, None

        logger.info(
            "Auto-resizing output from %dx%d to %dx%d for encoder compatibility",
            int(width),
            int(height),
            resize_w,
            resize_h,
        )
        return (resize_w, resize_h), None

    def _get_output_resize_and_canvas(
        self, width: int, height: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        target_w = self._parse_output_dim(self._output_width, "width", width)
        target_h = self._parse_output_dim(self._output_height, "height", height)

        if target_w is None and target_h is None:
            return self._get_auto_resize_and_canvas(width, height)

        if target_w is None:
            if target_h is None or target_h <= 0:
                raise ValueError(f"output_height must be positive; got {target_h}")
            scale = float(target_h) / float(height)
            resize_w = int(round(float(width) * scale))
            resize_h = int(target_h)
            resize_w = self._coerce_even(resize_w, "width")
            resize_h = self._coerce_even(resize_h, "height")
            if resize_w == int(width) and resize_h == int(height):
                return None, None
            return (resize_w, resize_h), None

        if target_h is None:
            if target_w <= 0:
                raise ValueError(f"output_width must be positive; got {target_w}")
            target_w = self._coerce_even(int(target_w), "width")
            scale = float(target_w) / float(width)
            resize_h = int(round(float(height) * scale))
            resize_h = self._coerce_even(resize_h, "height")
            if target_w == int(width) and resize_h == int(height):
                return None, None
            return (target_w, resize_h), None

        target_w = self._coerce_even(int(target_w), "width")
        target_h = self._coerce_even(int(target_h), "height")
        if target_w <= 0 or target_h <= 0:
            raise ValueError(
                f"output_width/output_height must be positive; got {target_w}x{target_h}"
            )
        scale = min(float(target_w) / float(width), float(target_h) / float(height))
        resize_w = int(round(float(width) * scale))
        resize_h = int(round(float(height) * scale))
        resize_w = self._coerce_even(max(1, resize_w), "width")
        resize_h = self._coerce_even(max(1, resize_h), "height")
        resize_w = min(resize_w, target_w)
        resize_h = min(resize_h, target_h)
        if resize_w == target_w and resize_h == target_h:
            if resize_w == int(width) and resize_h == int(height):
                return None, None
            return (resize_w, resize_h), None
        return (resize_w, resize_h), (target_w, target_h)

    @staticmethod
    def _letterbox_tensor(img: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
        w = image_width(img)
        h = image_height(img)
        pad_w = max(0, int(target_w) - int(w))
        pad_h = max(0, int(target_h) - int(h))
        if pad_w == 0 and pad_h == 0:
            return img
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        x = make_channels_first(img)
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], "constant", 0)
        return make_channels_last(x)

    def _reset_prepare_cuda_graphs(self) -> None:
        self._img_prepare_cg = None
        self._img_prepare_cg_signature = None
        self._end_zone_prepare_cg = None
        self._end_zone_prepare_cg_signature = None

    def set_cuda_graph_enabled(self, enabled: bool) -> bool:
        self._cuda_graph_enabled = bool(enabled)
        if not self._cuda_graph_enabled:
            self._reset_prepare_cuda_graphs()
        return True

    @staticmethod
    def _normalize_input_image(img: Any) -> Any:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if isinstance(img, torch.Tensor) and img.ndim == 3:
            img = img.unsqueeze(0)
        return img

    def _prepare_output_layout(
        self, results: dict[str, Any], online_im: Any
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        resize_wh = self._output_resize_wh
        canvas_wh = self._output_canvas_wh
        if self._output_width is None and self._output_height is None:
            return resize_wh, canvas_wh
        if resize_wh is None and canvas_wh is None:
            # The output shape is stable once inferred from the first frame, so
            # cache it and reset any graph capture tied to the previous layout.
            resize_wh, canvas_wh = self._get_output_resize_and_canvas(
                width=image_width(online_im),
                height=image_height(online_im),
            )
            self._output_resize_wh = resize_wh
            self._output_canvas_wh = canvas_wh
            self._reset_prepare_cuda_graphs()
        output_wh = canvas_wh or resize_wh
        if output_wh is not None:
            video_frame_cfg = results.get("video_frame_cfg")
            if isinstance(video_frame_cfg, dict):
                target_w, target_h = output_wh
                video_frame_cfg_local = dict(video_frame_cfg)
                video_frame_cfg_local["output_frame_width"] = target_w
                video_frame_cfg_local["output_frame_height"] = target_h
                video_frame_cfg_local["output_aspect_ratio"] = float(target_w) / float(target_h)
                results["video_frame_cfg"] = video_frame_cfg_local
        return resize_wh, canvas_wh

    def _apply_output_transforms(
        self,
        img: Any,
        resize_wh: Optional[Tuple[int, int]],
        canvas_wh: Optional[Tuple[int, int]],
    ) -> Any:
        out = img
        if resize_wh is not None:
            target_w, target_h = resize_wh
            if image_width(out) != target_w or image_height(out) != target_h:
                out_t = unwrap_tensor(out)
                src_w = image_width(out_t)
                src_h = image_height(out_t)
                resize_mode = None
                if target_w < src_w or target_h < src_h:
                    # Use higher-quality bicubic filtering when downscaling final
                    # output frames in the prep path.
                    resize_mode = "bicubic"
                out_t = resize_image(
                    img=out_t,
                    new_width=target_w,
                    new_height=target_h,
                    mode=resize_mode,
                    antialias=True,
                )
                out = to_uint8_image(out_t).contiguous()
        if canvas_wh is not None:
            target_w, target_h = canvas_wh
            out = self._letterbox_tensor(unwrap_tensor(out), target_w, target_h)
            out = to_uint8_image(out).contiguous()
        return out

    def _maybe_move_to_output_device(self, img: Any) -> Any:
        out = img
        if not self._skip_final_save and self._device is not None:
            # Match the sink path: if we know the eventual writer device, move
            # once here so the prep graph captures the final device placement.
            if str(out.device) != str(self._device):
                out = unwrap_tensor(out).to(self._device)
            if out.device.type != "cpu" and self._device.type == "cpu":
                out = out.to("cpu", non_blocking=True)
                out = wrap_tensor(out)
            else:
                assert self._device is None or out.device == self._device
        return out

    def _graph_prepare_tensor(
        self,
        img: Any,
        *,
        resize_wh: Optional[Tuple[int, int]],
        canvas_wh: Optional[Tuple[int, int]],
        slot: str,
    ) -> Any:
        if resize_wh is None and canvas_wh is None:
            return img
        if not self._cuda_graph_enabled:
            return self._apply_output_transforms(img, resize_wh, canvas_wh)
        if self._device is None or self._device.type != "cuda":
            return self._apply_output_transforms(img, resize_wh, canvas_wh)
        img_t = unwrap_tensor(img)
        if not isinstance(img_t, torch.Tensor) or img_t.device.type != "cuda":
            return self._apply_output_transforms(img, resize_wh, canvas_wh)
        # Capture is keyed by output layout and device so that a new graph is
        # only created when the prep topology actually changes.
        signature = (img_t.device, resize_wh, canvas_wh)
        if slot == self.VIDEO_DEFAULT:
            cg = self._img_prepare_cg
            cg_sig = self._img_prepare_cg_signature
        else:
            cg = self._end_zone_prepare_cg
            cg_sig = self._end_zone_prepare_cg_signature
        if cg is None or cg_sig != signature:
            cg = CudaGraphCallable(
                lambda t: self._apply_output_transforms(t, resize_wh, canvas_wh),
                (img_t,),
                warmup=0,
                name=f"video_out_{slot}_prep",
            )
            if slot == self.VIDEO_DEFAULT:
                self._img_prepare_cg = cg
                self._img_prepare_cg_signature = signature
            else:
                self._end_zone_prepare_cg = cg
                self._end_zone_prepare_cg_signature = signature
        return cg(img_t)

    def _prepare_single_image(
        self,
        img: Any,
        *,
        resize_wh: Optional[Tuple[int, int]],
        canvas_wh: Optional[Tuple[int, int]],
        slot: str,
    ) -> Any:
        out = self._normalize_input_image(img)
        out = self._maybe_move_to_output_device(out)
        return self._graph_prepare_tensor(out, resize_wh=resize_wh, canvas_wh=canvas_wh, slot=slot)

    def _ensure_initialized(self, context: dict[str, Any]) -> None:
        if self._device is None:
            img = context.get("img")
            if img is None:
                img = context.get("end_zone_img")
            img = self._normalize_input_image(img)
            img_t = unwrap_tensor(img) if img is not None else None
            if isinstance(img_t, torch.Tensor):
                self._device = img_t.device
            else:
                self._device = torch.device("cpu")
        if self._fourcc == "auto":
            # Prep does not create writers, but codec selection still matters
            # because GPU-only codecs determine whether the final sink expects
            # CUDA or CPU-ready tensors.
            video_frame_cfg = context["video_frame_cfg"]
            output_frame_width = int(video_frame_cfg["output_frame_width"])
            output_frame_height = int(video_frame_cfg["output_frame_height"])
            if self._device.type == "cuda":
                device_index = self._device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                self._fourcc, is_gpu = get_best_codec(
                    device_index,
                    width=output_frame_width,
                    height=output_frame_height,
                    allow_scaling=self._allow_scaling,
                )
                if not is_gpu:
                    logger.info("Can't use GPU for video output prep; falling back to CPU")
                    self._device = torch.device("cpu")
                    self._reset_prepare_cuda_graphs()
            else:
                self._fourcc = "XVID"
            logger.info(
                "Video output prep %s%sx%s will use codec: %s",
                f"{self._name} " if self._name else "",
                output_frame_width,
                output_frame_height,
                self._fourcc,
            )

    def prepare_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of ``results`` with final video tensors prepared.

        @param results: Aspen context carrying at least ``img`` and
                        ``video_frame_cfg``.
        @return: A copy of ``results`` with prepared ``img`` and optional
                 ``end_zone_img`` tensors.
        """
        prepared = dict(results)
        online_im = prepared["img"]
        online_im = self._normalize_input_image(online_im)

        resize_wh, canvas_wh = self._prepare_output_layout(prepared, online_im)

        self._ensure_initialized(prepared)

        prepared["img"] = self._prepare_single_image(
            online_im,
            resize_wh=resize_wh,
            canvas_wh=canvas_wh,
            slot=self.VIDEO_DEFAULT,
        )

        ez_img = prepared.get("end_zone_img")
        if ez_img is not None:
            prepared["end_zone_img"] = self._prepare_single_image(
                ez_img,
                resize_wh=resize_wh,
                canvas_wh=canvas_wh,
                slot=self.VIDEO_END_ZONES,
            )
        return prepared

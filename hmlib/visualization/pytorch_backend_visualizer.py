# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from hmlib.ui.show import show_image
from hmlib.vis.pt_text import draw_text as torch_draw_text
from hmlib.vis.pt_visualization import draw_box as torch_draw_box
from hmlib.vis.pt_visualization import draw_circle as torch_draw_circle
from hmlib.vis.pt_visualization import draw_line as torch_draw_line

ColorArg = Union[str, Tuple[int, ...], List[int], torch.Tensor, None]
PolygonArg = Union[torch.Tensor, np.ndarray, List[Tuple[float, float]]]


class PytorchBackendVisualizer(Visualizer):
    """Visualizer implementation that keeps drawing operations on PyTorch tensors."""

    def __init__(self, name: str = "visualizer", backend: str = "pytorch", *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        assert backend == "pytorch", "Only the PyTorch backend is supported in this implementation."
        self.backend = backend
        self._image_tensor: Optional[torch.Tensor] = None
        self._image_cpu_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invalidate_cpu_cache(self) -> None:
        self._image_cpu_cache = None

    def _ensure_image_tensor(self) -> torch.Tensor:
        if self._image_tensor is None:
            raise AssertionError("Image tensor has not been set")
        return self._image_tensor

    def _normalize_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Only a single image tensor is supported (batch dimension must be 1).")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, but got shape {tuple(image.shape)}.")
        if image.shape[0] not in (1, 3, 4):
            if image.shape[-1] in (1, 3, 4):
                image = image.permute(2, 0, 1)
            else:
                raise ValueError("Image tensor must be channel-first or channel-last with 1/3/4 channels.")
        if not image.is_contiguous():
            image = image.contiguous()

        if torch.is_floating_point(image):
            max_val = image.max()
            if max_val <= 1.0:
                image = (image * 255.0).round()
            else:
                image = image.round()
            image = image.clamp_(0, 255).to(torch.uint8)
        else:
            image = image.to(torch.uint8)
        return image

    def _tensor_to_numpy(self, image: torch.Tensor) -> np.ndarray:
        return image.detach().cpu().permute(1, 2, 0).numpy()

    def _color_to_tuple(self, color: ColorArg, channels: int) -> Tuple[int, ...]:
        if color is None:
            raise ValueError("Color cannot be None when drawing primitives.")
        if isinstance(color, torch.Tensor):
            color = color.detach().cpu().tolist()
        if isinstance(color, str):
            bgr = mmcv.color_val(color)
            values: Tuple[int, ...] = tuple(int(v) for v in bgr[::-1])
        elif isinstance(color, (tuple, list)):
            if len(color) == 3:
                values = (int(color[2]), int(color[1]), int(color[0]))
            elif len(color) == 4:
                values = (int(color[2]), int(color[1]), int(color[0]), int(color[3]))
            elif len(color) == 1:
                values = (int(color[0]),)
            else:
                raise ValueError(f"Unsupported color format with length {len(color)}: {color}")
        else:
            raise ValueError(f"Unsupported color type: {type(color)}")

        if channels == 4 and len(values) == 3:
            values = (*values, 255)
        elif channels == 3 and len(values) == 4:
            values = values[:3]
        elif channels == 1:
            values = (values[0],)
        elif len(values) != channels:
            raise ValueError(f"Color length {len(values)} does not match required channels {channels}.")
        return tuple(int(v) for v in values)

    def _expand_to_list(self, value: Union[Sequence, ColorArg], count: int) -> List:
        if isinstance(value, (list, tuple)) and len(value) == count:
            return list(value)
        return [value for _ in range(count)]

    def _expand_float(self, value: Union[Sequence[float], float], count: int) -> List[float]:
        if isinstance(value, (list, tuple)):
            if len(value) == count:
                return [float(v) for v in value]
            raise ValueError(f"Expected {count} alpha values, but got {len(value)}")
        return [float(value) for _ in range(count)]

    def _apply_draw(self, image: torch.Tensor, draw_fn, alpha: float) -> torch.Tensor:
        overlay = draw_fn(image.clone())
        if alpha >= 1.0:
            return overlay
        diff_mask = (overlay != image).any(dim=0, keepdim=True)
        if not diff_mask.any():
            return image
        base_f = image.to(torch.float32)
        overlay_f = overlay.to(torch.float32)
        blended = torch.where(
            diff_mask.expand_as(base_f),
            torch.lerp(base_f, overlay_f, alpha),
            base_f,
        )
        return blended.to(image.dtype)

    def _fill_polygon(
        self, image: torch.Tensor, polygon: PolygonArg, color: Tuple[int, ...], alpha: float
    ) -> torch.Tensor:
        if isinstance(polygon, torch.Tensor):
            poly = polygon.to(dtype=torch.float32, device=image.device)
        else:
            poly = torch.as_tensor(polygon, dtype=torch.float32, device=image.device)
        if poly.ndim == 1:
            poly = poly.view(-1, 2)
        if poly.shape[0] < 3:
            return image
        if not torch.allclose(poly[0], poly[-1]):
            poly = torch.cat([poly, poly[0:1]], dim=0)

        xs = poly[:, 0]
        ys = poly[:, 1]
        x_min = int(torch.clamp(xs.min(), 0, image.shape[2] - 1).item())
        x_max = int(torch.clamp(xs.max(), 0, image.shape[2] - 1).item())
        y_min = int(torch.clamp(ys.min(), 0, image.shape[1] - 1).item())
        y_max = int(torch.clamp(ys.max(), 0, image.shape[1] - 1).item())
        if x_max < x_min or y_max < y_min:
            return image

        grid_x = torch.arange(x_min, x_max + 1, device=image.device, dtype=torch.float32)
        grid_y = torch.arange(y_min, y_max + 1, device=image.device, dtype=torch.float32)
        Y, X = torch.meshgrid(grid_y, grid_x, indexing="ij")

        x1 = poly[:-1, 0].view(-1, 1, 1)
        y1 = poly[:-1, 1].view(-1, 1, 1)
        x2 = poly[1:, 0].view(-1, 1, 1)
        y2 = poly[1:, 1].view(-1, 1, 1)

        cond = ((y1 <= Y) & (y2 > Y)) | ((y1 > Y) & (y2 <= Y))
        xints = (Y - y1) * (x2 - x1) / (y2 - y1 + 1e-6) + x1
        crossings = cond & (X < xints)
        mask = crossings.sum(dim=0) % 2 == 1
        if not mask.any():
            return image

        mask = mask.unsqueeze(0).expand(image.shape[0], -1, -1)
        base_region = image[:, y_min : y_max + 1, x_min : x_max + 1].to(torch.float32)
        color_tensor = torch.tensor(color, dtype=torch.float32, device=image.device).view(-1, 1, 1)
        if color_tensor.shape[0] != base_region.shape[0]:
            color_tensor = color_tensor.expand(base_region.shape[0], -1, -1)
        overlay_region = color_tensor.expand_as(base_region)
        blended_region = torch.lerp(base_region, overlay_region, alpha) if alpha < 1.0 else overlay_region

        updated = image.clone()
        updated[:, y_min : y_max + 1, x_min : x_max + 1] = torch.where(
            mask,
            blended_region.to(image.dtype),
            base_region.to(image.dtype),
        )
        return updated

    # ------------------------------------------------------------------
    # Visualizer overrides
    # ------------------------------------------------------------------
    @master_only
    def set_image(self, image: Union[torch.Tensor, np.ndarray]) -> None:
        if isinstance(image, np.ndarray):
            tensor_image = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor_image = image
        else:
            raise TypeError(f"Unsupported image type {type(image)}")
        tensor_image = self._normalize_image_tensor(tensor_image)
        self._image_tensor = tensor_image
        self.height = tensor_image.shape[1]
        self.width = tensor_image.shape[2]
        self._default_font_size = max(int(math.sqrt(self.height * self.width) // 90), 10)
        self._invalidate_cpu_cache()

    @master_only
    def get_image(self) -> torch.Tensor:
        return self._ensure_image_tensor()

    @master_only
    def get_image_numpy(self) -> np.ndarray:
        if self._image_cpu_cache is None:
            self._image_cpu_cache = self._tensor_to_numpy(self._ensure_image_tensor())
        return self._image_cpu_cache

    @master_only
    def draw_circles(
        self,
        center: Union[torch.Tensor, Sequence[Sequence[float]]],
        radius: Union[torch.Tensor, Sequence[float], float],
        face_colors: Union[str, Sequence[str], Sequence[Tuple[int, int, int]], None] = "none",
        alpha: Union[float, Sequence[float]] = 1.0,
        **kwargs,
    ) -> "Visualizer":
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported backend {self.backend}")

        image = self._ensure_image_tensor()
        centers = torch.as_tensor(center, dtype=torch.float32, device=image.device)
        if centers.ndim == 1:
            centers = centers.unsqueeze(0)
        radii = torch.as_tensor(radius, dtype=torch.float32, device=image.device)
        if radii.ndim == 0:
            radii = radii.repeat(centers.shape[0])
        colors = self._expand_to_list(face_colors, centers.shape[0])
        alphas = self._expand_float(alpha, centers.shape[0])
        for (cx, cy), r, color, a in zip(centers, radii, colors, alphas):
            if color in (None, "none"):
                continue
            color_tuple = self._color_to_tuple(color, image.shape[0])

            def _draw(img: torch.Tensor) -> torch.Tensor:
                return torch_draw_circle(
                    image=img,
                    center_x=float(cx.item()),
                    center_y=float(cy.item()),
                    radius=float(r.item()),
                    color=color_tuple,
                    thickness=1,
                    fill=True,
                )

            image = self._apply_draw(image, _draw, float(a))
        self._image_tensor = image
        self._invalidate_cpu_cache()
        return self

    @master_only
    def draw_texts(
        self,
        texts: Union[str, Sequence[str]],
        positions: Union[torch.Tensor, Sequence[Sequence[float]]],
        font_sizes: Optional[Union[int, Sequence[int]]] = None,
        colors: Union[str, Sequence[str], Sequence[Tuple[int, int, int]]] = "g",
        vertical_alignments: Union[str, Sequence[str]] = "top",
        horizontal_alignments: Union[str, Sequence[str]] = "left",
        bboxes: Optional[Union[dict, Sequence[dict]]] = None,
        **kwargs,
    ) -> "Visualizer":
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported backend {self.backend}")

        image = self._ensure_image_tensor()
        if isinstance(texts, str):
            texts = [texts]
        positions_tensor = torch.as_tensor(positions, dtype=torch.float32, device=image.device)
        if positions_tensor.ndim == 1:
            positions_tensor = positions_tensor.unsqueeze(0)
        if font_sizes is None:
            font_sizes_list = [self._default_font_size for _ in texts]
        elif isinstance(font_sizes, (list, tuple)):
            font_sizes_list = list(font_sizes)
        else:
            font_sizes_list = [font_sizes for _ in texts]
        color_list = self._expand_to_list(colors, len(texts))
        va_list = self._expand_to_list(vertical_alignments, len(texts))
        ha_list = self._expand_to_list(horizontal_alignments, len(texts))
        bbox_list: Optional[List[Optional[dict]]] = None
        if bboxes is not None:
            if isinstance(bboxes, dict):
                bbox_list = [bboxes for _ in texts]
            else:
                bbox_list = list(bboxes)

        for idx, text in enumerate(texts):
            pos = positions_tensor[idx]
            font_size = int(font_sizes_list[idx])
            color_tuple = self._color_to_tuple(color_list[idx], image.shape[0])

            x = float(pos[0].item())
            y = float(pos[1].item())
            if ha_list[idx] == "right":
                x = max(0.0, x - len(text) * font_size * 0.6)
            elif ha_list[idx] == "center":
                x = max(0.0, x - len(text) * font_size * 0.3)
            if va_list[idx] == "top":
                y = min(float(self.height), y + font_size)
            elif va_list[idx] == "center":
                y = min(float(self.height), y + font_size * 0.5)

            if bbox_list is not None and bbox_list[idx] is not None:
                bbox_cfg = bbox_list[idx]
                face_color = bbox_cfg.get("facecolor", color_list[idx]) if bbox_cfg else color_list[idx]
                bbox_color_tuple = self._color_to_tuple(face_color, image.shape[0])
                text_width = len(text) * font_size
                top_left = (x, y - font_size * 1.2)
                bottom_right = (x + text_width, y + font_size * 0.2)

                def _draw_bbox(img: torch.Tensor) -> torch.Tensor:
                    return torch_draw_box(
                        image=img,
                        tlbr=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                        color=bbox_color_tuple,
                        thickness=1,
                        alpha=255,
                        filled=True,
                    )

                image = self._apply_draw(image, _draw_bbox, 1.0)

            def _draw_text(img: torch.Tensor) -> torch.Tensor:
                return torch_draw_text(
                    image=img,
                    x=int(x),
                    y=int(y),
                    text=text,
                    font_size=max(1, font_size // 2),
                    color=color_tuple,
                    position_is_text_bottom=True,
                )

            image = self._apply_draw(image, _draw_text, 1.0)

        self._image_tensor = image
        self._invalidate_cpu_cache()
        return self

    @master_only
    def draw_bboxes(
        self,
        bboxes: Union[torch.Tensor, Sequence[Sequence[float]]],
        edge_colors: Union[str, Sequence[str], Sequence[Tuple[int, int, int]]] = "g",
        line_widths: Union[int, float, Sequence[Union[int, float]]] = 2,
        alpha: Union[float, Sequence[float]] = 1.0,
        filled: bool = False,
        **kwargs,
    ) -> "Visualizer":
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported backend {self.backend}")

        image = self._ensure_image_tensor()
        bbox_tensor = torch.as_tensor(bboxes, dtype=torch.float32, device=image.device)
        if bbox_tensor.ndim == 1:
            bbox_tensor = bbox_tensor.unsqueeze(0)
        colors = self._expand_to_list(edge_colors, bbox_tensor.shape[0])
        line_width_list = self._expand_float(line_widths, bbox_tensor.shape[0])
        alpha_list = self._expand_float(alpha, bbox_tensor.shape[0])
        for box, color, thickness, a in zip(bbox_tensor, colors, line_width_list, alpha_list):
            color_tuple = self._color_to_tuple(color, image.shape[0])
            image = torch_draw_box(
                image=image,
                tlbr=box,
                color=color_tuple,
                thickness=max(1, int(thickness)),
                alpha=int(max(0, min(1, a)) * 255),
                filled=filled,
            )
        self._image_tensor = image
        self._invalidate_cpu_cache()
        return self

    @master_only
    def draw_lines(
        self,
        x_datas: Union[torch.Tensor, Sequence[Sequence[float]]],
        y_datas: Union[torch.Tensor, Sequence[Sequence[float]]],
        colors: Union[str, Sequence[str], Sequence[Tuple[int, int, int]]] = "g",
        line_widths: Union[int, float, Sequence[Union[int, float]]] = 2,
        alpha: Union[float, Sequence[float]] = 1.0,
        **kwargs,
    ) -> "Visualizer":
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported backend {self.backend}")

        image = self._ensure_image_tensor()
        x_tensor = torch.as_tensor(x_datas, dtype=torch.float32, device=image.device)
        y_tensor = torch.as_tensor(y_datas, dtype=torch.float32, device=image.device)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
            y_tensor = y_tensor.unsqueeze(0)
        colors_list = self._expand_to_list(colors, x_tensor.shape[0])
        width_list = self._expand_float(line_widths, x_tensor.shape[0])
        alpha_list = self._expand_float(alpha, x_tensor.shape[0])

        for x_pair, y_pair, color, thickness, a in zip(x_tensor, y_tensor, colors_list, width_list, alpha_list):
            color_tuple = self._color_to_tuple(color, image.shape[0])
            x1, x2 = int(x_pair[0].item()), int(x_pair[1].item())
            y1, y2 = int(y_pair[0].item()), int(y_pair[1].item())
            line_thickness = max(1, int(thickness))

            def _draw(img: torch.Tensor) -> torch.Tensor:
                return torch_draw_line(
                    image=img,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    color=color_tuple,
                    thickness=line_thickness,
                )

            image = self._apply_draw(image, _draw, float(a))

        self._image_tensor = image
        self._invalidate_cpu_cache()
        return self

    @master_only
    def draw_polygons(
        self,
        polygons: Union[PolygonArg, Sequence[PolygonArg]],
        edge_colors: Union[str, Sequence[str], Sequence[Tuple[int, int, int]]] = "g",
        alpha: Union[float, Sequence[float]] = 1.0,
        face_colors: Union[str, Sequence[str], Sequence[Tuple[int, int, int]], None] = None,
        **kwargs,
    ) -> "Visualizer":
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported backend {self.backend}")

        image = self._ensure_image_tensor()
        if isinstance(polygons, (np.ndarray, torch.Tensor)):
            polygons_list: List[PolygonArg] = [polygons]
        else:
            polygons_list = list(polygons)
        colors_list = self._expand_to_list(face_colors if face_colors is not None else edge_colors, len(polygons_list))
        alpha_list = self._expand_float(alpha, len(polygons_list))

        for poly, color, a in zip(polygons_list, colors_list, alpha_list):
            if color in (None, "none"):
                continue
            color_tuple = self._color_to_tuple(color, image.shape[0])
            image = self._fill_polygon(image, poly, color_tuple, float(a))

        self._image_tensor = image
        self._invalidate_cpu_cache()
        return self

    @master_only
    def show(
        self,
        drawn_img: Optional[torch.Tensor] = None,
        win_name: str = "image",
        wait_time: float = 0.0,
        continue_key: str = " ",
    ) -> None:
        if self.backend != "pytorch":
            raise ValueError(f"Unsupported backend {self.backend}")

        tensor_to_show = drawn_img if drawn_img is not None else self.get_image()
        show_image(win_name, tensor_to_show, wait=wait_time <= 0)
        if wait_time > 0:
            time.sleep(wait_time)

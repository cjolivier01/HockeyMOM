from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from mmengine.registry import TRANSFORMS

import hmlib.vis.pt_visualization as ptv
from hmlib.config import prepend_root_dir
from hmlib.segm.ice_rink import MaskEdgeDistances, find_extreme_points
from hmlib.tracking_utils import visualization as vis
from hmlib.utils.image import (
    image_height,
    image_width,
    is_channels_first,
    make_channels_first,
    make_channels_last,
)
from hmlib.utils.time import format_duration_to_hhmmss


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


@TRANSFORMS.register_module()
class HmImageOverlays(torch.nn.Module):
    def __init__(
        self,
        frame_number: bool = False,
        frame_time: bool = False,
        watermark_config: Dict[str, Any] = None,
        colors: Dict[str, Tuple[int, int, int]] = None,
        device: Optional[torch.device] = None,
        overlay_text: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        overlay_text_color: Optional[Tuple[int, int, int]] = None,
        overlay_text_origin: Tuple[int, int] = (40, 40),
        overlay_text_scale: Optional[float] = None,
        overlay_text_thickness: int = 4,
        overlay_text_max_lines: Optional[int] = None,
        overhead_rink: bool = False,
        overhead_margin_xy: Tuple[int, int] = (20, 20),
        overhead_min_height_px: int = 120,
        overhead_max_height_ratio: float = 0.25,
    ):
        super().__init__()
        self._draw_frame_number = frame_number
        self._draw_frame_time = frame_time
        self._watermark_config = watermark_config
        self._colors = colors if colors is not None else {}
        self._image_height_percent: float = 0.001
        self._watermark = None
        self._device = device
        self._overlay_text = self._normalize_overlay_text(overlay_text)
        self._overlay_text_color = overlay_text_color
        self._overlay_text_origin = overlay_text_origin
        self._overlay_text_scale = overlay_text_scale
        self._overlay_text_thickness = int(overlay_text_thickness)
        self._overlay_text_max_lines = overlay_text_max_lines
        # Overhead rink minimap config
        self._overhead_enabled = bool(overhead_rink)
        self._overhead_margin_xy = overhead_margin_xy
        self._overhead_min_h = int(overhead_min_height_px)
        self._overhead_max_h_ratio = float(overhead_max_height_ratio)
        # Homography from image -> rink canonical units
        self._H: Optional[torch.Tensor] = None
        self._rink_dims_ft: Tuple[float, float] = (200.0, 85.0)
        self._homog_ready: bool = False
        if self._watermark_config:
            self._load_watermark()

    @staticmethod
    def _normalize_overlay_text(
        overlay_text: Optional[Union[str, List[str], Tuple[str, ...]]]
    ) -> str:
        if overlay_text is None:
            return ""
        if isinstance(overlay_text, (list, tuple)):
            parts = [str(p) for p in overlay_text if p is not None and str(p).strip()]
            return "\n".join(parts)
        return str(overlay_text)

    def _load_watermark(self):
        # TODO: Make watermark separate
        if self._watermark_config and "image" in self._watermark_config:
            self._watermark = self._watermark_config["image"]
            if self._watermark is None:
                return
            if isinstance(self._watermark, str):
                self._watermark = cv2.imread(
                    prepend_root_dir(self._watermark),
                    cv2.IMREAD_UNCHANGED,
                )
            if self._watermark is None:
                raise ValueError(f"Could not load watermark image: {self._watermark_image}")
            self._watermark_height = image_height(self._watermark)
            self._watermark_width = image_width(self._watermark)
            watermark_rgb_channels = self._watermark[:, :, :3]
            watermark_alpha_channel = self._watermark[:, :, 3]
            watermark_mask = cv2.merge(
                [
                    watermark_alpha_channel,
                    watermark_alpha_channel,
                    watermark_alpha_channel,
                ]
            )
            # Scale mask to [0, 1]
            mask_dtype = watermark_mask.dtype
            watermark_mask = (watermark_mask / np.max(watermark_mask)).astype(mask_dtype)
            self.register_buffer(
                "_watermark_rgb_channels",
                torch.from_numpy(watermark_rgb_channels),
                persistent=False,
            )
            self.register_buffer(
                "_watermark_mask",
                torch.from_numpy(watermark_mask),
                persistent=False,
            )
        else:
            self._watermark = None

    def _draw_watermark(self, img: torch.Tensor) -> torch.Tensor:
        if self._watermark is not None:
            y = int(image_height(img) - self._watermark_height)
            x = int(image_width(img) - self._watermark_width - self._watermark_width / 10)
            img = paste_watermark_at_position(
                img,
                watermark_rgb_channels=self._watermark_rgb_channels,
                watermark_mask=self._watermark_mask,
                x=x,
                y=y,
            )
        return img

    @torch.no_grad()
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self._watermark is not None:
            results["img"] = self._draw_watermark(results["img"])

        draw_msg: str = ""
        frame_id: Optional[int] = None
        if self._draw_frame_number:
            frame_ids = results.get("frame_ids")
            if frame_ids is not None:
                # Only first frame id is drawn
                frame_id = int(frame_ids[0])
                draw_msg += f"F: {frame_id}\n"
        if self._draw_frame_time and frame_id is not None:
            assert frame_id > 0  # Should start at 1
            fps = results.get("fps")
            if fps:
                frame_time = frame_id / fps
                draw_msg += f"{format_duration_to_hhmmss(frame_time, decimals=2)}\n"
        if draw_msg:
            img = results["img"]
            icf = is_channels_first(img)
            if not icf:
                img = make_channels_first(img)
            h = image_height(img)
            font_scale = max(h * self._image_height_percent, 2)
            img = vis.plot_text(
                img=make_channels_first(img),
                text=draw_msg,
                org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_scale,
                color=self._colors.get("frame_number", (0, 0, 255)),
                thickness=10,
            )
            if not icf:
                img = make_channels_last(img)
            results["img"] = img
        if self._overlay_text:
            img = results["img"]
            icf = is_channels_first(img)
            if not icf:
                img = make_channels_first(img)
            h = image_height(img)
            if self._overlay_text_scale is None:
                font_scale = max(h * self._image_height_percent, 2)
            else:
                font_scale = float(self._overlay_text_scale)
            text = self._overlay_text
            if self._overlay_text_max_lines is not None:
                try:
                    max_lines = int(self._overlay_text_max_lines)
                    if max_lines > 0:
                        lines = text.splitlines()
                        if len(lines) > max_lines:
                            text = "\n".join(lines[:max_lines])
                except Exception:
                    pass
            img = vis.plot_text(
                img=make_channels_first(img),
                text=text,
                org=self._overlay_text_origin,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_scale,
                color=(
                    self._overlay_text_color
                    if self._overlay_text_color is not None
                    else self._colors.get("overlay_text", (255, 255, 255))
                ),
                thickness=self._overlay_text_thickness,
            )
            if not icf:
                img = make_channels_last(img)
            results["img"] = img
        # Overhead rink minimap
        if self._overhead_enabled:
            self._maybe_draw_overhead_minimap(results)
        return results

    def _maybe_draw_overhead_minimap(self, results: Dict[str, Any]) -> None:
        img = results.get("img")
        if img is None:
            return
        # Initialize homography once using rink_profile
        if not self._homog_ready:
            rp = results.get("rink_profile")
            if rp is None:
                # No rink profile yet (early frames or disabled); skip
                return
            mask = rp.get("combined_mask")
            if isinstance(mask, torch.Tensor):
                mask_bool = mask.to(torch.bool).cpu()
            else:
                mask_bool = torch.from_numpy(mask).to(torch.bool)
            # Use mid-edge intersections at rink centroid for a more stable H
            centroid = rp.get("centroid")
            if isinstance(centroid, torch.Tensor):
                cx, cy = int(centroid[0].item()), int(centroid[1].item())
            else:
                cx, cy = int(centroid[0]), int(centroid[1])
            medges = MaskEdgeDistances.from_mask(mask_bool)
            # Clamp within bounds
            cy_cl = max(0, min(cy, mask_bool.shape[0] - 1))
            cx_cl = max(0, min(cx, mask_bool.shape[1] - 1))
            left_x = int(medges.left_edges[cy_cl].item())
            right_x = int(medges.right_edges[cy_cl].item())
            top_y = int(medges.top_edges[cx_cl].item())
            bot_y = int(medges.bottom_edges[cx_cl].item())
            # Fallback to extremes if any are invalid
            if left_x < 0 or right_x < 0 or top_y < 0 or bot_y < 0:
                min_x_pos, max_x_pos, min_y_pos, max_y_pos = find_extreme_points(mask_bool)
                left_x, right_x = int(min_x_pos[1]), int(max_x_pos[1])
                top_y, bot_y = int(min_y_pos[0]), int(max_y_pos[0])
                cx_cl = int((left_x + right_x) / 2)
                cy_cl = int((top_y + bot_y) / 2)
            src_pts = np.float32(
                [
                    [left_x, cy_cl],  # left mid
                    [right_x, cy_cl],  # right mid
                    [cx_cl, top_y],  # top mid
                    [cx_cl, bot_y],  # bottom mid
                ]
            )
            # Destination canonical rink coords (feet)
            Rw, Rh = self._rink_dims_ft
            dst_pts = np.float32(
                [
                    [0.0, Rh / 2.0],  # left mid
                    [Rw, Rh / 2.0],  # right mid
                    [Rw / 2.0, 0.0],  # top mid
                    [Rw / 2.0, Rh],  # bottom mid
                ]
            )
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self._H = torch.from_numpy(H).to(torch.float32)
            self._homog_ready = True
        # Prepare canvas dims
        B = img.shape[0] if img.ndim == 4 else 1
        H_out = image_height(img)
        # Compute minimap size
        mini_h = max(self._overhead_min_h, int(H_out * self._overhead_max_h_ratio))
        scale = mini_h / self._rink_dims_ft[1]
        mini_w = int(self._rink_dims_ft[0] * scale + 0.5)
        # Position (top-right)
        margin_x, margin_y = self._overhead_margin_xy
        # y is from top
        y0_default = margin_y
        # Access per-frame points and ids
        points_list: List[torch.Tensor] = results.get("player_bottom_points", [])
        ids_list: List[torch.Tensor] = results.get("player_ids", [])
        if not isinstance(points_list, list) or len(points_list) == 0:
            return
        # Ensure channels-first for slice paste
        icf = is_channels_first(img)
        if not icf:
            img = make_channels_first(img)
        # Ensure batch
        sq = img.ndim == 3
        if sq:
            img = img.unsqueeze(0)
        # For each frame in batch, draw minimap
        for bi in range(min(B, len(points_list))):
            frame_img = img[bi]
            # Compute top-right anchor for this frame
            fw = image_width(frame_img)
            x0 = fw - margin_x - mini_w
            y0 = y0_default
            # Create canvas
            dtype = frame_img.dtype
            device = frame_img.device
            rink_canvas = torch.full((3, mini_h, mini_w), 235.0, dtype=dtype, device=device)
            # Draw rink silhouette: rectangle + semicircles (rounded ends)
            radius = mini_h // 2
            rect_left = radius
            rect_right = max(rect_left + 1, mini_w - radius)
            # Fill central rectangle white using alpha-filled rectangle path
            rink_canvas = vis.plot_alpha_rectangle(
                rink_canvas,
                box=[rect_left, 0, rect_right, mini_h - 1],
                color=(255, 255, 255),
                thickness=1,
                opacity_percent=99,
            )
            # Fill semicircles at ends
            rink_canvas = ptv.draw_circle(
                rink_canvas,
                center_x=rect_left,
                center_y=radius,
                radius=radius,
                color=(255, 255, 255),
                thickness=1,
                fill=True,
            )
            rink_canvas = ptv.draw_circle(
                rink_canvas,
                center_x=rect_right,
                center_y=radius,
                radius=radius,
                color=(255, 255, 255),
                thickness=1,
                fill=True,
            )
            # Border outline (approximate): vertical edges + circle outlines
            rink_canvas = vis.plot_rectangle(
                rink_canvas,
                box=[rect_left, 0, rect_right, mini_h - 1],
                color=(0, 0, 0),
                thickness=2,
            )
            rink_canvas = ptv.draw_circle(
                rink_canvas,
                center_x=rect_left,
                center_y=radius,
                radius=radius,
                color=(0, 0, 0),
                thickness=2,
                fill=False,
            )
            rink_canvas = ptv.draw_circle(
                rink_canvas,
                center_x=rect_right,
                center_y=radius,
                radius=radius,
                color=(0, 0, 0),
                thickness=2,
                fill=False,
            )
            # Center and blue lines
            cx_px = int((self._rink_dims_ft[0] / 2.0) * scale)
            rink_canvas = ptv.draw_vertical_line(
                rink_canvas,
                start_x=cx_px,
                start_y=0,
                length=mini_h,
                color=(32, 0, 255),
                thickness=2,
            )
            blue_off = int(25.0 * scale)
            for off in (-blue_off, blue_off):
                x_bl = int(cx_px + off)
                rink_canvas = ptv.draw_vertical_line(
                    rink_canvas,
                    start_x=x_bl,
                    start_y=0,
                    length=mini_h,
                    color=(255, 0, 0),
                    thickness=2,
                )
            # Map and draw players
            pts = points_list[bi]
            ids = ids_list[bi] if bi < len(ids_list) else None
            if isinstance(pts, torch.Tensor) and pts.numel() > 0:
                # Filter points to be on-ice only, if mask present
                rp = results.get("rink_profile")
                mask = rp.get("combined_mask") if rp is not None else None
                if isinstance(mask, torch.Tensor):
                    m = mask.to(torch.bool)
                    keep = []
                    for i in range(pts.shape[0]):
                        x = int(max(0, min(m.shape[1] - 1, int(pts[i, 0].item()))))
                        y = int(max(0, min(m.shape[0] - 1, int(pts[i, 1].item()))))
                        if y < m.shape[0] and x < m.shape[1] and bool(m[y, x].item()):
                            keep.append(True)
                        else:
                            keep.append(False)
                    if any(keep):
                        pts = pts[torch.tensor(keep, dtype=torch.bool, device=pts.device)]
                        if isinstance(ids, torch.Tensor) and ids.numel() == len(keep):
                            ids = ids[torch.tensor(keep, dtype=torch.bool, device=ids.device)]
                # Homogeneous projection
                ones = torch.ones((pts.shape[0], 1), dtype=torch.float32)
                xy1 = torch.cat([pts.to(torch.float32), ones], dim=1).t()  # 3xN
                H = self._H.to(xy1.device)
                uvw = H @ xy1  # 3xN
                w = uvw[2:3, :]
                w = torch.where(w == 0, torch.ones_like(w), w)
                uv = uvw[:2, :] / w
                uv = uv.t()  # Nx2 in rink feet
                # Scale to canvas px
                uv_px = torch.stack([uv[:, 0] * scale, uv[:, 1] * scale], dim=1)  # Nx2
                # Draw circles
                for i in range(uv_px.shape[0]):
                    px = int(uv_px[i, 0].item())
                    py = int(uv_px[i, 1].item())
                    if 0 <= px < mini_w and 0 <= py < mini_h:
                        color = (0, 0, 0)
                        if isinstance(ids, torch.Tensor) and ids.numel() > i:
                            color = vis.get_color(int(ids[i].item()))
                        rink_canvas = ptv.draw_circle(
                            rink_canvas,
                            center_x=px,
                            center_y=py,
                            radius=5,
                            color=color,
                            thickness=2,
                            fill=True,
                        )
            # Paste into frame with boundary checks and source offsets
            fh = image_height(frame_img)
            fw = image_width(frame_img)
            dx = int(x0)
            dy = int(y0)
            sx = 0
            sy = 0
            if dx < 0:
                sx = -dx
                dx = 0
            if dy < 0:
                sy = -dy
                dy = 0
            paste_w = min(mini_w - sx, fw - dx)
            paste_h = min(mini_h - sy, fh - dy)
            if paste_w > 1 and paste_h > 1:
                frame_img[:, dy : dy + paste_h, dx : dx + paste_w] = rink_canvas[
                    :, sy : sy + paste_h, sx : sx + paste_w
                ]
            img[bi] = frame_img
        # Restore original dims
        if sq:
            img = img.squeeze(0)
        if not icf:
            img = make_channels_last(img)
        results["img"] = img

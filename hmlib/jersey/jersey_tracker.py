from __future__ import absolute_import, division, print_function

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

import hmlib.tracking_utils.visualization as vis
import hmlib.vis.pt_text as ptt
import hmlib.vis.pt_visualization as ptv
from hmlib.log import logger
from hmlib.utils.image import is_channels_last, make_channels_first, make_channels_last

from .number_classifier import TrackJerseyInfo


@dataclass
class TrackingIdNumberInfo:
    tracking_id: int = -1
    current_number: int = -1
    current_score: float = 0.0
    last_frame_id: int = -1
    # number -> decayed evidence score
    number_scores: Dict[int, float] = field(default_factory=dict)
    # smoothed anchor point for drawing (x, y) in image coordinates
    anchor_ema: Optional[Tuple[float, float]] = None


class JerseyTracker:
    def __init__(
        self,
        show: bool = False,
        evidence_decay: float = 0.92,
        min_display_evidence: float = 1.0,
        switch_ratio: float = 1.35,
        anchor_ema_alpha: float = 0.55,
    ) -> None:
        self._show = show
        self._evidence_decay = float(evidence_decay)
        self._min_display_evidence = float(min_display_evidence)
        self._switch_ratio = float(switch_ratio)
        self._anchor_ema_alpha = float(anchor_ema_alpha)
        self._tracking_id_jersey: Dict[int, TrackingIdNumberInfo] = {}
        self._jersey_number_to_tracking_id: Dict[int, int] = {}

    def _update_anchor(self, tracking_id: int, anchor_xy: Tuple[float, float]) -> None:
        info = self._tracking_id_jersey.get(tracking_id)
        if info is None:
            return
        if info.anchor_ema is None:
            info.anchor_ema = (float(anchor_xy[0]), float(anchor_xy[1]))
            return
        ax, ay = info.anchor_ema
        alpha = self._anchor_ema_alpha
        info.anchor_ema = (
            float(alpha * anchor_xy[0] + (1.0 - alpha) * ax),
            float(alpha * anchor_xy[1] + (1.0 - alpha) * ay),
        )

    def _decay_scores(self, info: TrackingIdNumberInfo, frame_id: int) -> None:
        if info.last_frame_id < 0:
            return
        dt = max(1, int(frame_id) - int(info.last_frame_id))
        decay = float(self._evidence_decay**dt)
        if decay >= 0.999:
            return
        for k in list(info.number_scores.keys()):
            v = float(info.number_scores.get(k, 0.0)) * decay
            if v < 1e-4:
                info.number_scores.pop(k, None)
            else:
                info.number_scores[k] = v

    def _select_number(self, info: TrackingIdNumberInfo) -> None:
        if not info.number_scores:
            info.current_number = -1
            info.current_score = 0.0
            return
        sorted_items = sorted(info.number_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_num, best_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        current_score = float(info.number_scores.get(info.current_number, 0.0))

        # Initial acquisition.
        if info.current_number < 0:
            if best_score >= self._min_display_evidence:
                info.current_number = int(best_num)
                info.current_score = float(best_score)
            return

        # Keep current unless a contender wins decisively.
        if best_num == info.current_number:
            info.current_score = float(best_score)
            return

        should_switch = (
            best_score >= self._min_display_evidence
            and best_score >= (current_score * self._switch_ratio)
            and best_score >= (second_score * 1.05)
        )
        if should_switch:
            logger.info(
                "Jersey switch: ID %s #%s -> #%s (%.3f -> %.3f)",
                info.tracking_id,
                info.current_number,
                int(best_num),
                current_score,
                best_score,
            )
            info.current_number = int(best_num)
            info.current_score = float(best_score)

    @staticmethod
    def _draw_label(
        image: torch.Tensor | np.ndarray,
        text: str,
        center_x: int,
        bottom_y: int,
        font_scale: float,
        fg: Tuple[int, int, int],
        bg: Tuple[int, int, int],
        bg_alpha: int,
    ) -> torch.Tensor | np.ndarray:
        if isinstance(image, torch.Tensor):
            was_channels_last = is_channels_last(image)
            if was_channels_last:
                image = make_channels_first(image)
            font_size = max(1, int(font_scale * 2))
            text_w, text_h = ptt.measure_text(text=text, font_size=font_size)
            pad_x = max(3, int(font_scale * 6))
            pad_y = max(2, int(font_scale * 4))
            x1 = int(center_x - (text_w // 2) - pad_x)
            x2 = int(center_x + (text_w // 2) + pad_x)
            y2 = int(bottom_y + pad_y)
            y1 = int(bottom_y - text_h - pad_y)
            image = ptv.draw_box(
                image=image, tlbr=(x1, y1, x2, y2), color=bg, filled=True, alpha=bg_alpha
            )
            image = ptv.draw_box(
                image=image,
                tlbr=(x1, y1, x2, y2),
                color=fg,
                filled=False,
                alpha=220,
                thickness=max(1, int(font_scale)),
            )
            image = ptt.draw_text(
                image=image,
                x=int(center_x - text_w // 2),
                y=int(bottom_y),
                text=text,
                font_size=font_size,
                color=fg,
                position_is_text_bottom=True,
            )
            if was_channels_last:
                image = make_channels_last(image)
            return image

        # numpy/cv2 path
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        pad_x = max(3, int(font_scale * 6))
        pad_y = max(2, int(font_scale * 4))
        x1 = int(center_x - (tw // 2) - pad_x)
        x2 = int(center_x + (tw // 2) + pad_x)
        y2 = int(bottom_y + pad_y)
        y1 = int(bottom_y - th - baseline - pad_y)
        img = vis.to_cv2(image)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, thickness=-1)
        alpha = float(max(0, min(255, bg_alpha))) / 255.0
        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), fg, thickness=max(1, int(font_scale)))
        cv2.putText(
            img,
            text,
            (int(center_x - tw // 2), int(bottom_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            fg,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return img

    def draw(
        self, image: torch.Tensor, tracking_ids: torch.Tensor, bboxes: torch.Tensor
    ) -> torch.Tensor:
        if not self._show:
            return image
        assert len(tracking_ids) == len(bboxes)
        # `image` is usually CHW torch, but handle HWC for robustness.
        if isinstance(image, torch.Tensor):
            img_w = int(image.shape[-1] if not is_channels_last(image) else image.shape[1])
        else:
            img_w = int(image.shape[1])
        font_scale = max(0.6, float(img_w) / 1800.0)

        for (x1, y1, w, _), tracking_id in zip(bboxes, tracking_ids):
            tracking_id = int(tracking_id)
            info: TrackingIdNumberInfo = self._tracking_id_jersey.get(tracking_id)
            if info is None or info.current_number <= 0:
                continue
            anchor = (float(x1 + (w * 0.5)), float(y1))
            self._update_anchor(tracking_id=tracking_id, anchor_xy=anchor)
            ax, ay = info.anchor_ema if info.anchor_ema is not None else anchor

            # Fade background with evidence (keeps display stable but less noisy).
            ev = float(info.current_score)
            bg_alpha = int(max(80, min(200, 80 + (ev * 60.0))))
            image = self._draw_label(
                image=image,
                text=str(info.current_number),
                center_x=int(ax),
                bottom_y=int(ay),
                font_scale=font_scale,
                fg=(255, 255, 255),
                bg=(0, 0, 0),
                bg_alpha=bg_alpha,
            )
        return image

    def observe_tracking_id_number_info(self, frame_id: int, info: TrackJerseyInfo) -> None:
        """Consume per-frame jersey predictions and keep a stable per-track display state."""
        tracking_id = int(info.tracking_id)
        number = int(info.number)
        score = float(info.score)
        if tracking_id < 0 or number <= 0:
            return

        prev_info = self._tracking_id_jersey.get(tracking_id)
        if prev_info is None:
            prev_info = TrackingIdNumberInfo(tracking_id=tracking_id)
            self._tracking_id_jersey[tracking_id] = prev_info

        self._decay_scores(prev_info, frame_id=frame_id)
        prev_info.last_frame_id = int(frame_id)

        prev_info.number_scores[number] = float(
            prev_info.number_scores.get(number, 0.0) + max(score, 0.0)
        )
        self._select_number(prev_info)

        # Best-effort mapping (not used for control, but handy for debugging).
        if prev_info.current_number > 0:
            self._jersey_number_to_tracking_id[prev_info.current_number] = tracking_id

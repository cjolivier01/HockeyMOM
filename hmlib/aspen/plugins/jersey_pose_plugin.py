from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData

from hmlib.aspen.plugins.base import Plugin
from hmlib.bbox.tiling import (
    clamp_boxes_to_image,
    get_original_bbox_index_from_tiled_image,
    pack_bounding_boxes_as_tiles,
)
from hmlib.jersey.number_classifier import TrackJerseyInfo
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.ui import show_image  # noqa: F401 (for debugging
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, make_channels_first, make_channels_last

_PADDING = 5
_CONFIDENCE_THRESHOLD = 0.4


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, StreamTensorBase):
        return x.wait()
    assert isinstance(x, np.ndarray)
    return torch.from_numpy(x)


def _is_valid_number(s: str) -> bool:
    if not s or len(s) > 2 or not s.isdigit():
        return False
    try:
        v = int(s)
    except Exception:
        return False
    return 0 < v < 100


@dataclass(frozen=True)
class _RoiSpec:
    track_index: int
    roi: torch.Tensor  # xyxy int64
    vote_scale: float = 1.0


class JerseyNumberFromPosePlugin(Plugin):
    """
    Recognizes jersey numbers per tracked person using pose-guided ROIs and MMOCR.

    Expects in context:
      - data: dict with keys:
          - data_samples: TrackDataSample or [TrackDataSample]
          - original_images: Tensor [T, C, H, W] or [T, H, W, C]
          - pose_results: optional (mirrored by PoseToDetPlugin); if absent, falls back to bbox heuristic

    Produces in context:
      - data: with key 'jersey_results': List[List[TrackJerseyInfo]] per frame
    """

    def __init__(
        self,
        enabled: bool = True,
        roi_mode: str = "pose",  # 'pose' or 'bbox'
        det_thresh: float = 0.5,
        rec_thresh: float = 0.8,
        # Performance controls
        frame_stride: int = 1,
        max_tracks_per_frame: Optional[int] = 20,
        min_track_height: int = 80,
        # BBox-based ROI fallback (if pose missing)
        roi_top: float = 0.25,
        roi_bottom: float = 0.95,
        roi_side: float = 0.2,
        # Side-on sleeve ROI (when player is orthogonal to camera)
        side_view_enabled: bool = False,
        side_view_shoulder_ratio_thresh: float = 0.22,
        side_view_vote_scale: float = 1.25,
    ):
        super().__init__(enabled=enabled)
        self._inferencer = None
        self._roi_mode = str(roi_mode)
        self._det_thresh = float(det_thresh)
        self._rec_thresh = float(rec_thresh)
        self._frame_stride = max(1, int(frame_stride))
        self._max_tracks_per_frame = (
            None if max_tracks_per_frame is None else int(max_tracks_per_frame)
        )
        self._min_track_height = int(min_track_height)
        self._roi_top = float(roi_top)
        self._roi_bottom = float(roi_bottom)
        self._roi_side = float(roi_side)
        self._side_view_enabled = bool(side_view_enabled)
        self._side_view_shoulder_ratio_thresh = float(side_view_shoulder_ratio_thresh)
        self._side_view_vote_scale = float(side_view_vote_scale)

    def _ensure_inferencer(self):
        if self._inferencer is not None:
            return
        # Reuse the configuration from HmNumberClassifier to avoid extra deps
        from hmlib.jersey.number_classifier import HmNumberClassifier

        self._inferencer = HmNumberClassifier.create_inferencer()

    @staticmethod
    def _to_numpy_uint8(img: torch.Tensor) -> np.ndarray:
        img = make_channels_last(img)
        if img.dtype != torch.uint8:
            # assume 0..1 or 0..255 float
            if torch.max(img) <= 1.0:
                img = (img * 255.0).clamp(0, 255)
            img = img.to(dtype=torch.uint8, non_blocking=True)
        return img.cpu().numpy()

    def _roi_from_box(self, box: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rr_top = y1 + self._roi_top * h
        rr_bot = y1 + self._roi_bottom * h
        rr_left = x1 + self._roi_side * w
        rr_right = x2 - self._roi_side * w
        coords = torch.tensor(
            [rr_left, rr_top, rr_right, rr_bot], dtype=torch.float32, device=box.device
        )
        out = torch.round(coords).to(dtype=torch.int64)
        return out

    @staticmethod
    def _bbox_from_keypoints(kpts: torch.Tensor) -> Optional[torch.Tensor]:
        if kpts is None or not torch.is_tensor(kpts) or kpts.numel() == 0:
            return None
        x = kpts[..., 0]
        y = kpts[..., 1]
        if x.numel() == 0 or y.numel() == 0:
            return None
        x1 = torch.min(x)
        y1 = torch.min(y)
        x2 = torch.max(x)
        y2 = torch.max(y)
        return torch.stack([x1, y1, x2, y2]).to(dtype=torch.int64)

    @staticmethod
    def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
        ax1, ay1, ax2, ay2 = a.float()
        bx1, by1, bx2, by2 = b.float()
        ix1 = torch.max(ax1, bx1)
        iy1 = torch.max(ay1, by1)
        ix2 = torch.min(ax2, bx2)
        iy2 = torch.min(ay2, by2)
        iw = torch.clamp(ix2 - ix1, min=0)
        ih = torch.clamp(iy2 - iy1, min=0)
        inter = iw * ih
        area_a = torch.clamp(ax2 - ax1, min=0) * torch.clamp(ay2 - ay1, min=0)
        area_b = torch.clamp(bx2 - bx1, min=0) * torch.clamp(by2 - by1, min=0)
        union = area_a + area_b - inter + 1e-6
        return float((inter / union).item())

    @staticmethod
    def _extract_pose_instances(pose_result_item: Any) -> Optional[SimpleNamespace]:
        try:
            preds = pose_result_item.get("predictions")
            if isinstance(preds, list) and preds:
                ds = preds[0]
                inst = getattr(ds, "pred_instances", None)
                if inst is not None:
                    return inst
                if isinstance(ds, dict):
                    keys = ["bboxes", "scores", "keypoints", "keypoint_scores"]
                    attrs = {k: ds[k] for k in keys if k in ds}
                    if attrs:
                        return SimpleNamespace(**attrs)
        except Exception:
            pass
        return None

    @staticmethod
    def _torso_roi_from_pose(
        kpts: torch.Tensor, kps: Optional[torch.Tensor], img_w: int, img_h: int
    ) -> Optional[torch.Tensor]:
        # COCO indices: 5=LS, 6=RS, 11=LH, 12=RH
        try:
            scores = None
            if kps is not None and torch.is_tensor(kps) and kps.shape[0] == kpts.shape[0]:
                scores = kps
            idxs = [6, 5, 11, 12]
            pts: List[Tuple[float, float, float]] = []
            for i in idxs:
                x = float(kpts[i, 0].item())
                y = float(kpts[i, 1].item())
                s = float(scores[i].item()) if scores is not None else 1.0
                pts.append((x, y, s))
            if any(p[2] < _CONFIDENCE_THRESHOLD for p in pts):
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x_min = int(max(0, min(xs) - _PADDING))
            x_max = int(min(img_w - 1, max(xs) + _PADDING))
            y_min = int(max(0, min(ys) - _PADDING))
            y_max = int(min(img_h - 1, max(ys)))
            if x_max <= x_min or y_max <= y_min:
                return None
            return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int64, device=kpts.device)
        except Exception:
            return None

    def _is_side_view_pose(
        self, kpts: torch.Tensor, kps: Optional[torch.Tensor], bbox_w: float
    ) -> Tuple[bool, Optional[str]]:
        if bbox_w <= 1:
            return False, None
        try:
            scores = None
            if kps is not None and torch.is_tensor(kps) and kps.shape[0] == kpts.shape[0]:
                scores = kps

            def kp(i: int) -> Tuple[float, float, float]:
                x = float(kpts[i, 0].item())
                y = float(kpts[i, 1].item())
                s = float(scores[i].item()) if scores is not None else 1.0
                return x, y, s

            lsx, lsy, lss = kp(5)
            rsx, rsy, rss = kp(6)
            shoulder_ok = (lss >= _CONFIDENCE_THRESHOLD) and (rss >= _CONFIDENCE_THRESHOLD)
            if shoulder_ok:
                dist = float(np.hypot(lsx - rsx, lsy - rsy))
                if (dist / float(bbox_w)) < self._side_view_shoulder_ratio_thresh:
                    side = "left" if lss >= rss else "right"
                    return True, side
                return False, None

            if (lss >= _CONFIDENCE_THRESHOLD) != (rss >= _CONFIDENCE_THRESHOLD):
                side = "left" if lss >= rss else "right"
                return True, side

            return False, None
        except Exception:
            return False, None

    @staticmethod
    def _upper_arm_roi_from_pose(
        kpts: torch.Tensor,
        kps: Optional[torch.Tensor],
        img_w: int,
        img_h: int,
        side: str,
    ) -> Optional[torch.Tensor]:
        # COCO indices: left (5=shoulder,7=elbow,9=wrist) right (6,8,10)
        if side not in ("left", "right"):
            return None
        try:
            scores = None
            if kps is not None and torch.is_tensor(kps) and kps.shape[0] == kpts.shape[0]:
                scores = kps

            ids = (5, 7, 9) if side == "left" else (6, 8, 10)
            pts: List[Tuple[float, float, float]] = []
            for i in ids:
                x = float(kpts[i, 0].item())
                y = float(kpts[i, 1].item())
                s = float(scores[i].item()) if scores is not None else 1.0
                pts.append((x, y, s))

            if pts[0][2] < _CONFIDENCE_THRESHOLD or pts[1][2] < _CONFIDENCE_THRESHOLD:
                return None
            use_pts = [pts[0], pts[1]]
            if pts[2][2] >= _CONFIDENCE_THRESHOLD:
                use_pts.append(pts[2])

            xs = [p[0] for p in use_pts]
            ys = [p[1] for p in use_pts]
            arm_len = float(np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
            pad = int(max(_PADDING, 0.45 * arm_len))
            x_min = int(max(0, min(xs) - pad))
            x_max = int(min(img_w - 1, max(xs) + pad))
            y_min = int(max(0, min(ys) - int(0.25 * pad)))
            y_max = int(min(img_h - 1, max(ys) + int(0.75 * pad)))
            if x_max <= x_min or y_max <= y_min:
                return None
            if (x_max - x_min) < 4 or (y_max - y_min) < 4:
                return None
            return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int64, device=kpts.device)
        except Exception:
            return None

    @staticmethod
    def _process_mmocr_results(
        ocr_results: Dict[str, Any], det_thresh: float = 0.5, rec_thresh: float = 0.8
    ):
        from mmocr.utils import poly2bbox

        predictions = ocr_results["predictions"]
        if isinstance(predictions, list) and len(predictions) >= 1:
            predictions = predictions[0]
        rec_texts = predictions.get("rec_texts", [])
        rec_scores = predictions.get("rec_scores", [])
        det_scores = predictions.get("det_scores", [])
        det_polygons = predictions.get("det_polygons", [])
        nr_items = len(rec_texts)
        centers: List[Tuple[str, int, int, int, float]] = []
        for index in range(nr_items):
            rec_text = rec_texts[index]
            if not rec_text or not str(rec_text).isdigit():
                continue
            if index < len(det_scores) and det_scores[index] < det_thresh:
                continue
            if index < len(rec_scores) and rec_scores[index] < rec_thresh:
                continue
            try:
                print(rec_texts)
                bbox = poly2bbox(det_polygons[index])
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                bw = int(bbox[2] - bbox[0])
            except Exception:
                continue
            centers.append(
                (
                    str(rec_text),
                    cx,
                    cy,
                    bw,
                    float(rec_scores[index] if index < len(rec_scores) else 0.0),
                )
            )
        return centers

    def _build_rois_for_frame(
        self,
        bboxes_xyxy: torch.Tensor,
        pose_inst: Optional[Any],
        img_w: int,
        img_h: int,
    ) -> List[_RoiSpec]:
        device = bboxes_xyxy.device if isinstance(bboxes_xyxy, torch.Tensor) else None
        rois: List[_RoiSpec] = []
        if (
            self._roi_mode == "pose"
            and pose_inst is not None
            and hasattr(pose_inst, "keypoints")
            and pose_inst.keypoints is not None
        ):
            kpts_all = pose_inst.keypoints  # (N,K,2)
            kps_all = getattr(pose_inst, "keypoint_scores", None)
            pose_bboxes: List[Optional[torch.Tensor]] = [
                self._bbox_from_keypoints(kpts_all[i]) for i in range(kpts_all.shape[0])
            ]
            for ti in range(bboxes_xyxy.shape[0]):
                tb = bboxes_xyxy[ti]
                best_iou = 0.0
                best_j = -1
                for pj, pb in enumerate(pose_bboxes):
                    if pb is None:
                        continue
                    iou = self._iou(tb, pb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = pj
                pose_j = best_j if best_iou > 0.1 else -1
                roi = None
                if pose_j >= 0:
                    roi = self._torso_roi_from_pose(
                        kpts_all[pose_j],
                        None if kps_all is None else kps_all[pose_j],
                        img_w,
                        img_h,
                    )
                if roi is None:
                    roi = self._roi_from_box(tb.to(dtype=torch.float32))
                roi = roi.to(dtype=torch.int64)
                if device is not None:
                    roi = roi.to(device=device)
                rois.append(_RoiSpec(track_index=ti, roi=roi, vote_scale=1.0))

                if self._side_view_enabled and pose_j >= 0:
                    bbox_w = float(max(1.0, float(tb[2] - tb[0])))
                    side_view, side = self._is_side_view_pose(
                        kpts_all[pose_j],
                        None if kps_all is None else kps_all[pose_j],
                        bbox_w=bbox_w,
                    )
                    if side_view and side is not None:
                        arm_roi = self._upper_arm_roi_from_pose(
                            kpts_all[pose_j],
                            None if kps_all is None else kps_all[pose_j],
                            img_w,
                            img_h,
                            side=side,
                        )
                        if arm_roi is not None:
                            arm_roi = arm_roi.to(dtype=torch.int64)
                            if device is not None:
                                arm_roi = arm_roi.to(device=device)
                            rois.append(
                                _RoiSpec(
                                    track_index=ti,
                                    roi=arm_roi,
                                    vote_scale=float(self._side_view_vote_scale),
                                )
                            )
            return rois

        # Bbox-only ROIs
        for ti in range(bboxes_xyxy.shape[0]):
            roi = self._roi_from_box(bboxes_xyxy[ti].to(dtype=torch.float32))
            roi = roi.to(dtype=torch.int64)
            if device is not None:
                roi = roi.to(device=device)
            rois.append(_RoiSpec(track_index=ti, roi=roi, vote_scale=1.0))
        return rois

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        # Allow injecting an inferencer for tests
        self._inferencer = context.get("mmocr_inferencer", self._inferencer)
        self._ensure_inferencer()
        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        original_images = context.get("original_images")
        if original_images is None:
            return {}
        original_images = make_channels_first(original_images)
        pose_results: Optional[List[Any]] = context.get("pose_results")

        all_jersey_results: List[List[TrackJerseyInfo]] = []
        W = int(image_width(original_images))
        H = int(image_height(original_images))
        img_size = torch.tensor([W, H], dtype=torch.int64, device=original_images.device)
        original_images = _to_tensor(original_images)

        for frame_index, img_data_sample in enumerate(track_data_sample.video_data_samples):
            if self._frame_stride > 1 and (frame_index % self._frame_stride) != 0:
                all_jersey_results.append([])
                continue

            pred_tracks: Optional[InstanceData] = getattr(
                img_data_sample, "pred_track_instances", None
            )
            if pred_tracks is None:
                all_jersey_results.append([])
                continue

            # Build ROIs from track boxes
            bboxes_xyxy = pred_tracks.bboxes
            if not isinstance(bboxes_xyxy, torch.Tensor):
                bboxes_xyxy = torch.as_tensor(bboxes_xyxy)
            tracking_ids = pred_tracks.instances_id
            if not isinstance(tracking_ids, torch.Tensor):
                tracking_ids = torch.as_tensor(tracking_ids)
            track_mask = get_track_mask(pred_tracks)
            if isinstance(track_mask, torch.Tensor):
                bboxes_xyxy = bboxes_xyxy[track_mask]
                tracking_ids = tracking_ids[track_mask]
            if bboxes_xyxy.numel() == 0:
                all_jersey_results.append([])
                continue

            # Reduce OCR load: keep largest/closest tracks only, and skip tiny tracks.
            if bboxes_xyxy.ndim == 2 and bboxes_xyxy.shape[1] == 4:
                heights = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]).to(torch.float32)
                keep = heights >= float(self._min_track_height)
                if keep.any():
                    bboxes_xyxy = bboxes_xyxy[keep]
                    tracking_ids = tracking_ids[keep]
                if (
                    self._max_tracks_per_frame is not None
                    and bboxes_xyxy.shape[0] > self._max_tracks_per_frame
                ):
                    # Prefer large boxes (better chance of legible numbers).
                    topk = int(self._max_tracks_per_frame)
                    heights2 = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]).to(torch.float32)
                    _, idx = torch.topk(heights2, k=topk)
                    bboxes_xyxy = bboxes_xyxy[idx]
                    tracking_ids = tracking_ids[idx]

            pose_inst = None
            if pose_results and frame_index < len(pose_results):
                pose_inst = self._extract_pose_instances(pose_results[frame_index])

            roi_specs = self._build_rois_for_frame(
                bboxes_xyxy=bboxes_xyxy, pose_inst=pose_inst, img_w=W, img_h=H
            )
            rois_t = torch.stack([r.roi for r in roi_specs], dim=0).to(dtype=torch.int64)
            rois_t = clamp_boxes_to_image(rois_t, image_size=img_size)

            # Pack ROIs into a tiled image
            frame_img = original_images[frame_index]
            packed_image, index_map = pack_bounding_boxes_as_tiles(frame_img, rois_t)
            # show_image("packed_image", packed_image)
            np_image = self._to_numpy_uint8(packed_image)

            # Run OCR
            # results = self._inferencer(packed_image, progress_bar=False)
            results = self._inferencer(np_image, progress_bar=False)
            text_and_centers = self._process_mmocr_results(
                results, det_thresh=self._det_thresh, rec_thresh=self._rec_thresh
            )

            jersey_results: List[TrackJerseyInfo] = []

            best_by_tid: Dict[int, TrackJerseyInfo] = {}
            for text, x, y, det_w, score in text_and_centers:
                if not _is_valid_number(text):
                    continue
                roi_idx = int(get_original_bbox_index_from_tiled_image(index_map, y=y, x=x))
                if roi_idx < 0 or roi_idx >= len(roi_specs):
                    continue
                spec = roi_specs[roi_idx]
                tid = int(tracking_ids[spec.track_index])
                roi_w = max(1, int(spec.roi[2] - spec.roi[0]))
                vote = float(score) * float(det_w) / float(roi_w) * float(spec.vote_scale)
                prev = best_by_tid.get(tid)
                if prev is None or vote > float(prev.score):
                    best_by_tid[tid] = TrackJerseyInfo(
                        tracking_id=tid, number=int(text), score=float(vote)
                    )

            jersey_results = list(best_by_tid.values())

            all_jersey_results.append(jersey_results)

        return {"jersey_results": all_jersey_results}

    def input_keys(self):
        return {"data_samples", "original_images", "pose_results", "mmocr_inferencer"}

    def output_keys(self):
        return {"jersey_results"}

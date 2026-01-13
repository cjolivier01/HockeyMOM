from __future__ import annotations

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


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, StreamTensorBase):
        return x.wait()
    assert isinstance(x, np.ndarray)
    return torch.from_numpy(x)


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
        roi_top: float = 0.25,
        roi_bottom: float = 0.95,
        roi_side: float = 0.2,
    ):
        super().__init__(enabled=enabled)
        self._inferencer = None
        self._roi_top = float(roi_top)
        self._roi_bottom = float(roi_bottom)
        self._roi_side = float(roi_side)

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

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        # Allow injecting an inferencer for tests
        self._inferencer = context.get("mmocr_inferencer", self._inferencer)
        self._ensure_inferencer()
        data: Dict[str, Any] = context.get("data", {})
        track_samples = data.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        original_images = data.get("original_images")
        if original_images is None:
            original_images = context.get("data", {}).get("original_images")
        if original_images is None:
            return {}
        original_images = make_channels_first(original_images)

        all_jersey_results: List[List[TrackJerseyInfo]] = []
        W = int(image_width(original_images))
        H = int(image_height(original_images))
        img_size = torch.tensor([W, H], dtype=torch.int64)
        original_images = _to_tensor(original_images)

        for frame_index, img_data_sample in enumerate(track_data_sample.video_data_samples):
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
            track_mask = get_track_mask(pred_tracks)
            if isinstance(track_mask, torch.Tensor):
                bboxes_xyxy = bboxes_xyxy[track_mask]
            if bboxes_xyxy.numel() == 0:
                all_jersey_results.append([])
                continue
            rois = []
            for bb in bboxes_xyxy:
                roi = self._roi_from_box(bb)
                rois.append(roi)
            rois_t = torch.stack(rois, dim=0).to(dtype=torch.int64)
            rois_t = clamp_boxes_to_image(rois_t, image_size=img_size)

            # Pack ROIs into a tiled image
            frame_img = original_images[frame_index]
            packed_image, index_map = pack_bounding_boxes_as_tiles(frame_img, rois_t)
            # show_image("packed_image", packed_image)
            np_image = self._to_numpy_uint8(packed_image)

            # Run OCR
            # results = self._inferencer(packed_image, progress_bar=False)
            results = self._inferencer(np_image, progress_bar=False)
            text_and_centers = self._process_mmocr_results(results)

            jersey_results: List[TrackJerseyInfo] = []
            seen_ids = set()
            tracking_ids = pred_tracks.instances_id
            if isinstance(track_mask, torch.Tensor):
                tracking_ids = tracking_ids[track_mask]
            for text, x, y, w, score in text_and_centers:
                bbox_idx = int(get_original_bbox_index_from_tiled_image(index_map, y=y, x=x))
                if bbox_idx < 0 or bbox_idx >= len(tracking_ids):
                    continue
                tid = int(tracking_ids[bbox_idx])
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                jersey_results.append(
                    TrackJerseyInfo(tracking_id=tid, number=int(text), score=float(score))
                )

            all_jersey_results.append(jersey_results)

        data["jersey_results"] = all_jersey_results
        return {"data": data}

    def input_keys(self):
        return {"data"}

    def output_keys(self):
        return {"data"}

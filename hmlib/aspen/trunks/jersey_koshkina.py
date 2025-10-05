from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmengine.structures import InstanceData

from hmlib.aspen.trunks.base import Trunk
from hmlib.bbox.tiling import (
    clamp_boxes_to_image,
    get_original_bbox_index_from_tiled_image,
    pack_bounding_boxes_as_tiles,
)
from hmlib.jersey.number_classifier import TrackJerseyInfo
from hmlib.log import logger
from hmlib.ui import show_image  # noqa: F401 (for debugging
from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import image_height, image_width, make_channels_first, make_channels_last


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, StreamTensor):
        return x.wait()
    assert isinstance(x, np.ndarray)
    return torch.from_numpy(x)


class _IdentityLegibility(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns prob legible
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # everything legible by default
        return torch.ones((x.size(0),), dtype=torch.float32, device=x.device)


class KoshkinaJerseyNumberTrunk(Trunk):
    """
    Jersey number recognition trunk inspired by Koshkina et al.'s framework.

    Pipeline:
      1) Pose-guided ROI cropping using our pose/tracking outputs
      2) Optional legibility classifier to filter poor crops
      3) OCR recognition using MMOCR (det+rec) over tiled ROIs
      4) Tracklet-level consolidation: majority vote with score weighting

    Inputs in context:
      - data: dict with 'data_samples' (TrackDataSample) and 'original_images' (T,C,H,W or T,H,W,C)
      - data_to_send may also carry 'original_images'

    Outputs in context:
      - data['jersey_results']: List[List[TrackJerseyInfo]] per frame
    """

    def __init__(
        self,
        enabled: bool = True,
        legibility_enabled: bool = False,
        legibility_weights: Optional[str] = None,
        legibility_threshold: float = 0.5,
        roi_top: float = 0.25,
        roi_bottom: float = 0.95,
        roi_side: float = 0.2,
        det_thresh: float = 0.5,
        rec_thresh: float = 0.8,
        reid_enabled: bool = False,
        reid_threshold: float = 3.0,
        reid_backbone: str = "resnet18",
        reid_backend: str = "resnet",  # 'resnet' (default) or 'centroid'
        centroid_reid_path: Optional[str] = None,  # optional path to centroid-reid model or package
        centroid_reid_device: Optional[str] = None,
        roi_mode: str = "bbox",  # one of: bbox, pose, sam
        # STR backend: 'mmocr' (default) or 'parseq'
        str_backend: str = "mmocr",
        parseq_weights: Optional[str] = None,
        parseq_device: Optional[str] = None,
        sam_enabled: bool = False,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        sam_device: Optional[str] = None,
    ):
        super().__init__(enabled=enabled)
        self._legibility_enabled = bool(legibility_enabled)
        self._legibility_weights = legibility_weights
        self._legibility_threshold = float(legibility_threshold)
        self._roi_top = float(roi_top)
        self._roi_bottom = float(roi_bottom)
        self._roi_side = float(roi_side)
        self._det_thresh = float(det_thresh)
        self._rec_thresh = float(rec_thresh)

        self._mmocr = None
        self._legibility_model: nn.Module = _IdentityLegibility()
        self._reid_model: Optional[nn.Module] = None
        self._reid_enabled = bool(reid_enabled)
        self._reid_threshold = float(reid_threshold)
        self._reid_backbone = reid_backbone
        self._reid_backend = reid_backend
        self._centroid_reid_path = centroid_reid_path
        self._centroid_reid_device = centroid_reid_device
        self._roi_mode = roi_mode
        self._sam_enabled = bool(sam_enabled or roi_mode == "sam")
        self._sam_checkpoint = sam_checkpoint
        self._sam_model_type = sam_model_type
        self._sam_device = sam_device
        self._str_backend = str_backend
        self._parseq_weights = parseq_weights
        self._parseq_device = parseq_device
        self._parseq_model = None
        self._agg_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._agg_max_score: Dict[int, float] = defaultdict(float)
        self._current_number: Dict[int, int] = {}
        # Per-track embedding stats for Gaussian outlier removal (diagonal covariance)
        self._reid_count: Dict[int, int] = defaultdict(int)
        self._reid_mean: Dict[int, torch.Tensor] = {}
        self._reid_M2: Dict[int, torch.Tensor] = {}
        self._sam_predictor = None

    def _ensure_mmocr(self):
        if self._mmocr is not None:
            return
        from hmlib.jersey.number_classifier import HmNumberClassifier

        self._mmocr = HmNumberClassifier.create_inferencer()

    def _ensure_legibility(self, device: torch.device):
        if not self._legibility_enabled:
            return
        if isinstance(self._legibility_model, _IdentityLegibility):
            try:
                from torchvision.models import resnet34
                m = resnet34(weights=None)
                m.fc = nn.Linear(m.fc.in_features, 1)
                if self._legibility_weights and torch.cuda.is_available():
                    state = torch.load(self._legibility_weights, map_location=device)
                    m.load_state_dict(state, strict=False)
                self._legibility_model = m.to(device=device)
                self._legibility_model.eval()
            except Exception:
                # fallback to identity
                self._legibility_model = _IdentityLegibility().to(device=device)

    def _ensure_reid(self, device: torch.device):
        if not self._reid_enabled or self._reid_model is not None:
            return
        if self._reid_backend == "centroid":
            # Try to initialize centroid-reid from a user-provided path or env var
            import os
            import sys

            path = self._centroid_reid_path or os.environ.get("HM_CENTROID_REID_PATH")
            try:
                if path and path not in sys.path:
                    sys.path.insert(0, path)
                # Try common entry points
                try:
                    import centroid_reid as cr  # type: ignore
                except Exception:
                    try:
                        import centroids_reid as cr  # type: ignore
                    except Exception:
                        cr = None
                model = None
                if cr is not None and hasattr(cr, "load_model"):
                    model = cr.load_model(path=path)
                elif cr is not None and hasattr(cr, "build_model"):
                    model = cr.build_model()
                if model is not None and isinstance(model, nn.Module):
                    dev = self._centroid_reid_device or str(device)
                    model.to(dev)
                    model.eval()
                    self._reid_model = model
                    logger.info("Using centroid-reid backend for embeddings")
                    return
            except Exception as ex:
                logger.info(f"Failed to initialize centroid-reid backend ({ex}); falling back to resnet embeddings")
            # Fall through to resnet
        try:
            from torchvision import models
            if self._reid_backbone == "resnet34":
                m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            m.fc = nn.Identity()
            m.eval()
            self._reid_model = m.to(device=device)
            logger.info("Using resnet embeddings for re-id")
        except Exception as ex:
            logger.info(f"Re-id embeddings not available ({ex}); disabling re-id filtering")
            self._reid_model = None

    def _ensure_parseq(self, device_str: str):
        if self._str_backend != "parseq" or self._parseq_model is not None:
            return
        try:
            # Attempt to import PARSeq from strhub if installed
            from strhub.models.parseq.system import parseq  # type: ignore

            # Some distributions provide a convenience builder
            if self._parseq_weights:
                # Expecting a lightning checkpoint; fall back if load fails
                try:
                    self._parseq_model = parseq.load_from_checkpoint(self._parseq_weights)
                except Exception as ex:
                    logger.info(f"PARSeq load_from_checkpoint failed ({ex}); using default builder")
                    self._parseq_model = parseq(pretrained=True)
            else:
                self._parseq_model = parseq(pretrained=True)
            dev = self._parseq_device or device_str
            self._parseq_model.to(dev)
            self._parseq_model.eval()
        except Exception as ex:
            logger.info(f"PARSeq not available ({ex}); falling back to MMOCR rec backend")
            self._parseq_model = None

    @staticmethod
    def _preprocess_imagenet(x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W) in [0,255] uint8 or float
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device).view(3,1,1)
        x = (x - mean) / std
        return x

    def _extract_reid_embeddings(self, frame_img: torch.Tensor, rois: torch.Tensor) -> Optional[List[torch.Tensor]]:
        if not self._reid_enabled or self._reid_model is None or rois.numel() == 0:
            return None
        embs: List[torch.Tensor] = []
        for r in rois:
            x1,y1,x2,y2 = r.tolist()
            crop = frame_img[:, y1:y2, x1:x2]
            if crop.numel() == 0:
                embs.append(None)
                continue
            # resize to 224x224
            crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False).squeeze(0)
            crop = self._preprocess_imagenet(crop)
            with torch.no_grad():
                feat = self._reid_model(crop.unsqueeze(0)).squeeze(0)
            embs.append(feat)
        return embs

    def _reid_accept(self, tid: int, emb: Optional[torch.Tensor]) -> bool:
        if not self._reid_enabled or emb is None:
            return True
        # Welford online update for mean/variance (diagonal)
        n = self._reid_count[tid]
        if n == 0:
            self._reid_mean[tid] = emb.detach()
            self._reid_M2[tid] = torch.zeros_like(emb)
            self._reid_count[tid] = 1
            return True
        mean = self._reid_mean[tid]
        M2 = self._reid_M2[tid]
        delta = emb - mean
        new_mean = mean + delta / (n + 1)
        delta2 = emb - new_mean
        new_M2 = M2 + delta * delta2
        # Compute diagonal std (add epsilon)
        var = (new_M2 / max(1, n)).clamp_min(1e-6)
        z = torch.sqrt(((emb - new_mean) ** 2 / var).sum())
        accept = bool(z.item() <= self._reid_threshold)
        # Update only on accept
        if accept:
            self._reid_mean[tid] = new_mean
            self._reid_M2[tid] = new_M2
            self._reid_count[tid] = n + 1
        return accept

    def _ensure_sam(self, device_str: str):
        if not self._sam_enabled or self._sam_predictor is not None:
            return
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore

            if not self._sam_checkpoint:
                logger.info("SAM enabled but no checkpoint provided; skipping SAM and using bbox/pose ROIs")
                return
            sam = sam_model_registry[self._sam_model_type](checkpoint=self._sam_checkpoint)
            dev = self._sam_device or device_str
            sam.to(device=dev)
            self._sam_predictor = SamPredictor(sam)
        except Exception as ex:
            logger.info(f"SAM not available ({ex}); using bbox/pose ROIs")
            self._sam_predictor = None

    def _sam_refine_rois(self, frame_np: np.ndarray, rois: torch.Tensor) -> torch.Tensor:
        # frame_np: HxWxC uint8 RGB
        if not self._sam_enabled or self._sam_predictor is None or rois.numel() == 0:
            return rois
        try:
            self._sam_predictor.set_image(frame_np)
            refined: List[torch.Tensor] = []
            H, W, _ = frame_np.shape
            for r in rois:
                x1, y1, x2, y2 = [int(v) for v in r.tolist()]
                box_np = np.array([x1, y1, x2, y2])
                masks, _, _ = self._sam_predictor.predict(point_coords=None, point_labels=None, box=box_np, multimask_output=False)
                if masks is None or len(masks) == 0:
                    refined.append(r)
                    continue
                mask = masks[0]
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    refined.append(r)
                    continue
                rx1, ry1, rx2, ry2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                # Slightly expand within original box for OCR context
                pad = 2
                rx1 = max(0, rx1 - pad)
                ry1 = max(0, ry1 - pad)
                rx2 = min(W - 1, rx2 + pad)
                ry2 = min(H - 1, ry2 + pad)
                refined.append(torch.tensor([rx1, ry1, rx2, ry2], dtype=torch.int64, device=rois.device))
            return torch.stack(refined, dim=0)
        except Exception as ex:
            logger.info(f"SAM refine error: {ex}; using original ROIs")
            return rois

    def _to_numpy_uint8(self, img: torch.Tensor) -> np.ndarray:
        img = make_channels_last(img)
        if img.dtype != torch.uint8:
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
        coords = torch.tensor([rr_left, rr_top, rr_right, rr_bot], dtype=torch.float32, device=box.device)
        return torch.round(coords).to(dtype=torch.int64)

    @staticmethod
    def _bbox_from_keypoints(kpts: torch.Tensor) -> Optional[torch.Tensor]:
        # kpts: (K,2) or (K, >=2) [x,y,(score...)]
        if kpts is None or (isinstance(kpts, torch.Tensor) and kpts.numel() == 0):
            return None
        if not isinstance(kpts, torch.Tensor):
            kpts = torch.as_tensor(kpts)
        xs = kpts[..., 0]
        ys = kpts[..., 1]
        if xs.numel() == 0 or ys.numel() == 0:
            return None
        x1, y1 = torch.min(xs), torch.min(ys)
        x2, y2 = torch.max(xs), torch.max(ys)
        return torch.stack([x1, y1, x2, y2], dim=0)

    def _roi_from_keypoints(self, kpts: torch.Tensor) -> Optional[torch.Tensor]:
        # Prefer shoulders+hips if present (COCO indices: LShoulder=5, RShoulder=6, LHip=11, RHip=12)
        try:
            if isinstance(kpts, np.ndarray):
                k = torch.from_numpy(kpts)
            else:
                k = kpts
            sel_idx = []
            for idx in (5, 6, 11, 12):
                if idx < k.shape[-2]:
                    sel_idx.append(idx)
            if sel_idx:
                subset = k[sel_idx]
                bb = self._bbox_from_keypoints(subset)
                if bb is not None:
                    return self._roi_from_box(bb)
        except Exception:
            pass
        # Fallback: use all keypoints bbox
        bb = self._bbox_from_keypoints(kpts)
        return self._roi_from_box(bb) if bb is not None else None

    @staticmethod
    def _extract_pose_instances(pose_item: Any) -> List[Dict[str, Any]]:
        """Return list of {keypoints: Tensor} from a pose result item."""
        out: List[Dict[str, Any]] = []
        try:
            preds = pose_item.get("predictions")
            if isinstance(preds, list) and preds:
                ds = preds[0]
                inst = getattr(ds, "pred_instances", None)
                if inst is not None and hasattr(inst, "keypoints"):
                    kpts = inst.keypoints
                    if isinstance(kpts, torch.Tensor):
                        for i in range(kpts.shape[0]):
                            out.append({"keypoints": kpts[i]})
                    else:
                        # array/list
                        for kp in kpts:
                            out.append({"keypoints": torch.as_tensor(kp)})
                elif isinstance(ds, dict) and "keypoints" in ds:
                    kpts = ds["keypoints"]
                    if isinstance(kpts, list):
                        for kp in kpts:
                            out.append({"keypoints": torch.as_tensor(kp)})
        except Exception:
            pass
        return out

    def _rois_from_pose_matching(self, frame_pose: Any, track_boxes: torch.Tensor) -> Optional[torch.Tensor]:
        # Build pose bboxes
        pose_insts = self._extract_pose_instances(frame_pose)
        if not pose_insts:
            return None
        pose_bboxes: List[torch.Tensor] = []
        pose_rois: List[Optional[torch.Tensor]] = []
        for p in pose_insts:
            bb = self._bbox_from_keypoints(p["keypoints"]) or torch.tensor([0,0,0,0], dtype=torch.float32)
            pose_bboxes.append(bb)
            pose_rois.append(self._roi_from_keypoints(p["keypoints"]))

        if not pose_bboxes:
            return None
        PB = torch.stack(pose_bboxes, dim=0)
        # IoU match track_boxes (N,4) to pose PB (M,4)
        N = track_boxes.size(0)
        M = PB.size(0)
        rois_out: List[torch.Tensor] = []
        for i in range(N):
            tb = track_boxes[i]
            # Compute IoU with all pose bboxes
            x1 = torch.max(tb[0], PB[:,0]); y1 = torch.max(tb[1], PB[:,1])
            x2 = torch.min(tb[2], PB[:,2]); y2 = torch.min(tb[3], PB[:,3])
            inter = (x2 - x1).clamp_min(0)*(y2 - y1).clamp_min(0)
            area_tb = (tb[2]-tb[0])*(tb[3]-tb[1])
            area_pb = (PB[:,2]-PB[:,0])*(PB[:,3]-PB[:,1])
            union = area_tb + area_pb - inter + 1e-6
            iou = inter/union
            j = int(torch.argmax(iou).item())
            roi = pose_rois[j] if iou[j] > 0.1 and pose_rois[j] is not None else None
            if roi is None:
                roi = self._roi_from_box(tb)
            rois_out.append(roi)
        return torch.stack(rois_out, dim=0)

    @staticmethod
    def _parse_mmocr(results: Dict[str, Any], det_thresh: float, rec_thresh: float) -> List[Tuple[str, int, int, int, float]]:
        from mmocr.utils import poly2bbox
        predictions = results.get("predictions")
        if isinstance(predictions, list) and predictions:
            predictions = predictions[0]
        if not isinstance(predictions, dict):
            return []
        rec_texts = predictions.get("rec_texts", [])
        rec_scores = predictions.get("rec_scores", [])
        det_scores = predictions.get("det_scores", [])
        det_polygons = predictions.get("det_polygons", [])
        out: List[Tuple[str, int, int, int, float]] = []
        N = len(rec_texts)
        for i in range(N):
            t = rec_texts[i]
            if not t or not str(t).isdigit():
                continue
            ds = det_scores[i] if i < len(det_scores) else 1.0
            rs = rec_scores[i] if i < len(rec_scores) else 0.0
            if ds < det_thresh or rs < rec_thresh:
                continue
            try:
                bbox = poly2bbox(det_polygons[i])
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                bw = int(bbox[2] - bbox[0])
            except Exception:
                continue
            out.append((str(t), cx, cy, bw, float(rs)))
        return out

    def _recognize_with_parseq(self, image_np: np.ndarray, det_polygons: List[Any]) -> Optional[List[Tuple[str, float]]]:
        # Minimal wrapper: attempt to crop bbox from polygon, apply parseq model, greedy decode
        if self._parseq_model is None:
            return None
        try:
            import cv2  # for resize
        except Exception:
            return None
        results: List[Tuple[str, float]] = []
        for poly in det_polygons:
            try:
                xs = poly[0::2]
                ys = poly[1::2]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                crop = image_np[max(0,y1):max(0,y2), max(0,x1):max(0,x2), :]
                if crop.size == 0:
                    results.append(("", 0.0))
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                # Resize to a common STR size (H=32, keep aspect)
                target_h = 32
                h, w = gray.shape[:2]
                if h <= 1 or w <= 1:
                    results.append(("", 0.0))
                    continue
                new_w = int(w * (target_h / float(h)))
                gray = cv2.resize(gray, (new_w, target_h))
                # To 3-channel and tensor
                img_t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
                # Some PARSeq cuts expect 3 channels; replicate
                img_t = img_t.repeat(1, 3, 1, 1)
                dev = next(self._parseq_model.parameters()).device
                with torch.no_grad():
                    pred = self._parseq_model(img_t.to(dev))
                # Best-effort decoding: if pred is logits [B,T,C], take argmax per step and filter non-digits
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                if torch.is_tensor(pred):
                    seq = pred.argmax(dim=-1).squeeze(0).tolist()
                    # Map to digits if model provides mapping; otherwise fallback
                    text = ''.join(str(min(9, max(0, int(i)))) for i in seq)
                    score = 0.9
                else:
                    text, score = "", 0.0
                # Keep only digit strings per the jersey task
                text_digits = ''.join(ch for ch in text if ch.isdigit())
                results.append((text_digits, score))
            except Exception:
                results.append(("", 0.0))
        return results

    def _consolidate(self, tid: int, number: int, score: float) -> Tuple[int, float]:
        self._agg_counts[tid][number] += 1
        if score > self._agg_max_score[tid]:
            self._agg_max_score[tid] = score
        # choose current by highest count; break ties by max score
        counts = self._agg_counts[tid]
        best_num = max(counts.items(), key=lambda kv: (kv[1], self._agg_max_score[tid] if kv[0] == self._current_number.get(tid, -1) else 0.0))[0]
        self._current_number[tid] = best_num
        return best_num, self._agg_max_score[tid]

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        # Allow dependency injection for tests
        self._mmocr = context.get("mmocr_inferencer", self._mmocr)
        self._ensure_mmocr()

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
            original_images = context.get("data_to_send", {}).get("original_images")
        if original_images is None:
            return {}
        original_images = make_channels_first(original_images)
        device = original_images.device

        original_images = _to_tensor(original_images)

        self._ensure_legibility(device=device)
        self._ensure_reid(device=device)
        self._ensure_sam(device_str=str(device))
        self._ensure_parseq(device_str=str(device))

        W = int(image_width(original_images))
        H = int(image_height(original_images))
        img_size = torch.tensor([W, H], dtype=torch.int64, device=device)

        all_jersey_results: List[List[TrackJerseyInfo]] = []
        pose_results: Optional[List[Any]] = data.get("pose_results") or context.get("data_to_send", {}).get("pose_results")
        for frame_index, img_data_sample in enumerate(track_data_sample.video_data_samples):
            pred_tracks: Optional[InstanceData] = getattr(img_data_sample, "pred_track_instances", None)
            if pred_tracks is None or (hasattr(pred_tracks, 'bboxes') and len(pred_tracks.bboxes) == 0):
                all_jersey_results.append([])
                continue

            bboxes_xyxy = pred_tracks.bboxes
            if not isinstance(bboxes_xyxy, torch.Tensor):
                bboxes_xyxy = torch.as_tensor(bboxes_xyxy, device=device)
            rois = None
            if self._roi_mode == "pose" and pose_results and frame_index < len(pose_results):
                try:
                    rois = self._rois_from_pose_matching(pose_results[frame_index], bboxes_xyxy)
                except Exception as ex:
                    logger.info(f"pose ROI matching failed, fallback to bbox: {ex}")
            elif self._roi_mode == "sam":
                # Optional SAM-based ROI (not bundled); gracefully fallback
                logger.info("SAM ROI mode requested but not available; falling back to bbox")
            if rois is None:
                rois = torch.stack([self._roi_from_box(bb) for bb in bboxes_xyxy], dim=0)
            rois = clamp_boxes_to_image(rois, image_size=img_size)

            # Optional SAM refinement
            if self._sam_enabled and self._sam_predictor is not None:
                frame_np = self._to_numpy_uint8(original_images[frame_index])
                rois = self._sam_refine_rois(frame_np, rois)

            # Optional legibility filter on individual crops
            if self._legibility_enabled and not isinstance(self._legibility_model, _IdentityLegibility):
                crops: List[torch.Tensor] = []
                frame_img = original_images[frame_index]
                for r in rois:
                    x1, y1, x2, y2 = r.tolist()
                    crops.append(frame_img[:, y1:y2, x1:x2])
                # Simple resize/pad as needed could be added here
                scores = self._legibility_model.forward(torch.stack([c for c in crops if c.numel() > 0], dim=0))
                mask = scores.squeeze(-1) >= self._legibility_threshold
                if mask.numel() and not torch.all(mask):
                    idxs = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
                    rois = rois[idxs] if len(idxs) else rois[:0]
                    if rois.numel() == 0:
                        all_jersey_results.append([])
                        continue

            # (Optional) REID embeddings for occlusion/outlier removal
            reid_embs = self._extract_reid_embeddings(original_images[frame_index], rois)

            # Pack ROIs into a single image and OCR
            frame_img = original_images[frame_index]
            packed_image, index_map = pack_bounding_boxes_as_tiles(frame_img, rois)
            np_image = self._to_numpy_uint8(packed_image)
            results = self._mmocr(np_image, progress_bar=False)
            # results = self._mmocr(packed_image)
            centers = self._parse_mmocr(results, det_thresh=self._det_thresh, rec_thresh=self._rec_thresh)
            # Optionally swap recognition with PARSeq: reuse detected polygons and replace rec_texts/scores
            if self._str_backend == "parseq" and self._parseq_model is not None:
                try:
                    preds = results.get("predictions")
                    if isinstance(preds, list) and preds:
                        pred0 = preds[0]
                        det_polys = pred0.get("det_polygons", [])
                        parseq_recs = self._recognize_with_parseq(np_image, det_polys) or []
                        # Rebuild centers with PARSeq texts/scores where available
                        from mmocr.utils import poly2bbox
                        centers = []
                        for i, poly in enumerate(det_polys):
                            try:
                                bbox = poly2bbox(poly)
                                cx = int((bbox[0] + bbox[2]) / 2)
                                cy = int((bbox[1] + bbox[3]) / 2)
                                bw = int(bbox[2] - bbox[0])
                            except Exception:
                                continue
                            text, score = ("", 0.0)
                            if i < len(parseq_recs):
                                text, score = parseq_recs[i]
                            if text and all(ch.isdigit() for ch in text):
                                centers.append((text, cx, cy, bw, float(score)))
                    # else: keep MMOCR centers
                except Exception as ex:
                    logger.info(f"PARSeq substitution failed ({ex}); using MMOCR rec results")

            jersey_results: List[TrackJerseyInfo] = []
            seen = set()
            tracking_ids = pred_tracks.instances_id
            for text, x, y, w, score in centers:
                roi_idx = int(get_original_bbox_index_from_tiled_image(index_map, y=y, x=x))
                if roi_idx < 0 or roi_idx >= len(tracking_ids):
                    continue
                tid = int(tracking_ids[roi_idx])
                if tid in seen:
                    continue
                seen.add(tid)
                number = int(text)
                emb = reid_embs[roi_idx] if reid_embs is not None and roi_idx < len(reid_embs) else None
                if self._reid_accept(tid, emb):
                    consolidated_num, best_score = self._consolidate(tid, number, float(score))
                    jersey_results.append(TrackJerseyInfo(tracking_id=tid, number=consolidated_num, score=best_score))

            all_jersey_results.append(jersey_results)

        data["jersey_results"] = all_jersey_results
        return {"data": data}

    def input_keys(self):
        return {"data"}

    def output_keys(self):
        return {"data"}

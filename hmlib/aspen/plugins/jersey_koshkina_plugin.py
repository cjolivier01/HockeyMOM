from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from hmlib.aspen.plugins.base import Plugin
from hmlib.jersey.number_classifier import TrackJerseyInfo
from hmlib.log import logger
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, make_channels_first

# Jersey-number-pipeline inspired torso crop parameters
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


class _IdentityLegibility(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns prob legible
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return torch.ones((x.size(0),), dtype=torch.float32, device=x.device)


class KoshkinaJerseyNumberPlugin(Plugin):
    """
    Jersey number recognition trunk using the jersey-number-pipeline approach:
      - Pose-guided torso crops (shoulders + hips)
      - Optional legibility gating
      - PARSeq STR inference (via local xmodels/str/parseq)
      - Track-level aggregation with double-digit bias

    Inputs in context:
      - data: dict with 'data_samples' (TrackDataSample) and 'original_images' (T,C,H,W or T,H,W,C)
      - data may also carry 'original_images' and 'pose_results'

    Outputs in context:
      - data['jersey_results']: List[List[TrackJerseyInfo]] per frame
    """

    def __init__(
        self,
        enabled: bool = True,
        # Legibility
        legibility_enabled: bool = False,
        legibility_weights: Optional[str] = None,
        legibility_threshold: float = 0.5,
        # ROI selection
        roi_mode: str = "pose",  # 'pose' (default) or 'bbox'
        # STR / PARSeq
        parseq_weights: Optional[str] = None,
        parseq_device: Optional[str] = None,
        # Accept and ignore additional params for compatibility with legacy config
        **kwargs: Any,
    ):
        super().__init__(enabled=enabled)
        # Legibility
        self._legibility_enabled = bool(legibility_enabled)
        self._legibility_weights = legibility_weights
        self._legibility_threshold = float(legibility_threshold)
        self._legibility_model: nn.Module = _IdentityLegibility()

        # ROI
        self._roi_mode = str(roi_mode)

        # STR
        self._parseq_weights = parseq_weights
        self._parseq_device = parseq_device
        self._parseq_model = None
        self._str_transform = None  # filled by SceneTextDataModule.get_transform

        # Track-level accumulator: track_id -> List[(number, weight)]
        self._votes: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    # ----------------------- Model setup -----------------------
    def _ensure_legibility(self, device: torch.device):
        if not self._legibility_enabled:
            return
        if isinstance(self._legibility_model, _IdentityLegibility):
            try:
                # Prefer jersey-number-pipeline's LegibilityClassifier34 wrapper for state_dict compatibility
                import sys as _sys

                pipeline_path = os.environ.get(
                    "HM_JERSEY_PIPELINE_PATH",
                    "/mnt/monster-data/colivier/src/jersey-number-pipeline",
                )
                if os.path.isdir(pipeline_path) and pipeline_path not in _sys.path:
                    _sys.path.insert(0, pipeline_path)
                try:
                    from networks import LegibilityClassifier34  # type: ignore

                    leg_model = LegibilityClassifier34()
                except Exception:
                    # Fallback to torchvision if import fails
                    from torchvision.models import resnet34

                    leg_model = resnet34(weights=None)
                    leg_model.fc = nn.Linear(leg_model.fc.in_features, 1)

                if self._legibility_weights:
                    state = None
                    try:
                        state = torch.load(self._legibility_weights, map_location=device)
                    except Exception:
                        try:
                            state = torch.load(self._legibility_weights, map_location=device, weights_only=False)  # type: ignore[call-arg]
                        except Exception:
                            state = None
                    if state is not None:
                        try:
                            leg_model.load_state_dict(state, strict=False)
                        except Exception:
                            pass
                self._legibility_model = leg_model.to(device=device).eval()
            except Exception as ex:
                logger.info("Legibility model unavailable (%s); using identity.", ex)
                self._legibility_model = _IdentityLegibility().to(device=device)

    def _ensure_parseq(self, device_str: str):
        if self._parseq_model is not None:
            return
        # Prefer local copy at xmodels/str/parseq (contains strhub)
        try:
            here = Path(__file__).resolve()
            repo_root = here.parents[3]
            parseq_root = repo_root / "xmodels" / "str" / "parseq"
            if parseq_root.exists() and str(parseq_root) not in sys.path:
                sys.path.insert(0, str(parseq_root))
        except Exception:
            pass
        try:
            import string as _string

            from strhub.data.module import SceneTextDataModule  # type: ignore
            from strhub.models.utils import create_model  # type: ignore

            dev = self._parseq_device or device_str
            charset = _string.digits  # restrict to digits
            weights = self._parseq_weights
            model = None
            if weights:
                # Attempt manual state_dict load to avoid Lightning dependency
                state = None
                # First try default torch.load (may use weights_only=True on torch>=2.6 and fail)
                try:
                    state_raw = torch.load(weights, map_location="cpu")
                    state = (
                        state_raw.get("state_dict", state_raw)
                        if isinstance(state_raw, dict)
                        else state_raw
                    )
                except Exception as ex_default:
                    # Retry with weights_only=False (trusted local ckpt)
                    try:
                        state_raw = torch.load(weights, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
                        state = (
                            state_raw.get("state_dict", state_raw)
                            if isinstance(state_raw, dict)
                            else state_raw
                        )
                        logger.info("PARSeq checkpoint loaded with weights_only=False.")
                    except TypeError:
                        # Older torch may not accept weights_only; rethrow original
                        raise ex_default
                    except Exception as ex2:
                        logger.info(
                            f"PARSeq checkpoint manual load failed ({ex2}); using uninitialized model."
                        )
                        state = None
                if state is not None:
                    try:
                        # Derive max_label_length from checkpoint if possible (pos_queries shape = [1, L, C])
                        ml = 25
                        try:
                            for k, v in state.items():
                                if (
                                    isinstance(k, str)
                                    and k.endswith("pos_queries")
                                    and hasattr(v, "shape")
                                ):
                                    if len(v.shape) >= 2:
                                        ml = int(v.shape[1] - 1)
                                        break
                        except Exception:
                            pass
                        model = create_model(
                            "parseq", pretrained=False, charset_test=charset, max_label_length=ml
                        )
                        missing, unexpected = model.load_state_dict(state, strict=False)
                        if missing:
                            logger.info(
                                "PARSeq load: missing keys: %d", len(missing)
                            )
                        if unexpected:
                            logger.info(
                                "PARSeq load: unexpected keys: %d", len(unexpected)
                            )
                    except Exception as ex_load:
                        logger.info(
                            "PARSeq load_state_dict failed (%s); proceeding with uninitialized model.",
                            ex_load,
                        )
            if model is None:
                # Use a default max_label_length compatible with typical PARSeq checkpoints
                model = create_model(
                    "parseq", pretrained=False, charset_test=charset, max_label_length=25
                )
            self._parseq_model = model.eval().to(dev)
            hp = getattr(self._parseq_model, "hparams", None)
            img_size = hp.img_size if hp is not None else (32, 128)
            self._str_transform = SceneTextDataModule.get_transform(img_size)
        except Exception as ex:
            self._parseq_model = None
            self._str_transform = None
            logger.error("Failed to initialize PARSeq STR (%s).", ex)

    # ----------------------- Pose/ROI helpers -----------------------
    @staticmethod
    def _bbox_from_keypoints(kpts: torch.Tensor) -> Optional[torch.Tensor]:
        # kpts: (K,2)
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
        # a,b: (4,) xyxy
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
        val = float((inter / union).item())
        return val

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
                    keys = [
                        "bboxes",
                        "scores",
                        "bbox_scores",
                        "labels",
                        "keypoints",
                        "keypoint_scores",
                    ]
                    attrs = {k: ds[k] for k in keys if k in ds}
                    if attrs:
                        return SimpleNamespace(**attrs)
        except Exception:
            pass
        return None

    @staticmethod
    def _torso_roi_from_pose(
        kpts: torch.Tensor, kps: Optional[torch.Tensor], img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        # Use COCO indices: 5=LS, 6=RS, 11=LH, 12=RH
        try:
            px = kpts.clone().detach()
            scores = None
            if kps is not None and torch.is_tensor(kps) and kps.shape[0] == kpts.shape[0]:
                scores = kps
            idxs = [6, 5, 11, 12]
            pts: List[Tuple[float, float, float]] = []
            for i in idxs:
                x = float(px[i, 0].item())
                y = float(px[i, 1].item())
                s = float(scores[i].item()) if scores is not None else 1.0
                pts.append((x, y, s))
            # Confidence gate
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
            return (x_min, y_min, x_max, y_max)
        except Exception:
            return None

    @staticmethod
    def _bbox_torso_fallback(
        box: torch.Tensor, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        # Fallback: heuristic torso crop from bbox (top->bottom and slight side margin)
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        w = x2 - x1
        # Keep middle vertical band [25%, 95%] and shrink sides by 20%
        new_x1 = max(0, x1 + int(0.2 * w))
        new_x2 = min(img_w - 1, x2 - int(0.2 * w))
        new_y1 = max(0, y1 + int(0.25 * (y2 - y1)))
        new_y2 = min(img_h - 1, y1 + int(0.95 * (y2 - y1)))
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            return (x1, y1, x2, y2)
        return (new_x1, new_y1, new_x2, new_y2)

    # ----------------------- STR helpers -----------------------
    @staticmethod
    def _preprocess_imagenet(x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W) in [0,255]
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=x.device).view(
            3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=x.device).view(
            3, 1, 1
        )
        x = (x - mean) / std
        return x

    def _run_parseq_on_crop(self, crop_chw: torch.Tensor) -> Optional[Tuple[str, List[float]]]:
        # crop_chw: (3,H,W) uint8 on any device
        if self._parseq_model is None:
            return None
        try:
            from PIL import Image

            # Move to CPU for PIL transform if needed
            img = crop_chw.detach().to(device="cpu").permute(1, 2, 0).numpy().astype(np.uint8)
            pil = Image.fromarray(img, mode="RGB")
            if self._str_transform is None:
                # Minimal fallback: resize height to 32 while preserving aspect
                pil = pil.resize((max(16, int(pil.width * 32 / max(1, pil.height))), 32))
                t = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0
                t = t.unsqueeze(0)
            else:
                t = self._str_transform(pil).unsqueeze(0)
            dev = next(self._parseq_model.parameters()).device
            with torch.no_grad():
                logits = self._parseq_model.forward(t.to(dev))
            # Limit to first 3 positions, tokens up to 10 (E + digits)
            probs_full = logits[:, :3, :11].softmax(-1)
            preds, probs = self._parseq_model.tokenizer.decode(probs_full)
            label = preds[0]
            conf = probs[0].detach().cpu().numpy().squeeze().tolist()
            if not isinstance(conf, list):
                conf = [float(conf)]
            return str(label), [float(c) for c in conf]
        except Exception as ex:
            logger.info("PARSeq decode failed: %s", ex)
            return None

    # ----------------------- Aggregation -----------------------
    @staticmethod
    def _double_digit_bias(v: int) -> float:
        return 0.61 if v > 9 else 0.39

    @staticmethod
    def _aggregate_track(
        votes: List[Tuple[int, float]], filter_thresh: float = 0.2, sum_thresh: float = 1.0
    ) -> Tuple[int, float]:
        if not votes:
            return -1, 0.0
        # Filter weak
        filtered = [(v, w if w >= filter_thresh else 0.0) for (v, w) in votes]
        # Sum with bias
        totals: Dict[int, float] = defaultdict(float)
        for v, w in filtered:
            totals[v] += w * KoshkinaJerseyNumberPlugin._double_digit_bias(v)
        best_v = -1
        best_w = 0.0
        for k, tot in totals.items():
            if tot > best_w:
                best_w = tot
                best_v = k
        if best_w < sum_thresh:
            return -1, best_w
        return best_v, best_w

    # ----------------------- Main -----------------------
    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

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
        original_images = make_channels_first(_to_tensor(original_images))
        device = original_images.device

        self._ensure_legibility(device=device)
        self._ensure_parseq(device_str=str(device))

        W = int(image_width(original_images))
        H = int(image_height(original_images))

        pose_results: Optional[List[Any]] = data.get("pose_results") or context.get("data", {}).get(
            "pose_results"
        )

        all_jersey_results: List[List[TrackJerseyInfo]] = []
        for frame_index, img_data_sample in enumerate(track_data_sample.video_data_samples):
            pred_tracks: Optional[InstanceData] = getattr(
                img_data_sample, "pred_track_instances", None
            )
            if pred_tracks is None or (
                hasattr(pred_tracks, "bboxes") and len(pred_tracks.bboxes) == 0
            ):
                all_jersey_results.append([])
                continue

            # Prepare per-frame inputs
            bboxes_xyxy = pred_tracks.bboxes
            if not isinstance(bboxes_xyxy, torch.Tensor):
                bboxes_xyxy = torch.as_tensor(bboxes_xyxy, device=device)
            tracking_ids = pred_tracks.instances_id

            # Pose matching if requested
            pose_inst = None
            if self._roi_mode == "pose" and pose_results and frame_index < len(pose_results):
                pose_inst = self._extract_pose_instances(pose_results[frame_index])

            frame_img = original_images[frame_index]  # (C,H,W)

            # Build ROIs per track
            rois: List[Tuple[int, Tuple[int, int, int, int]]] = (
                []
            )  # (index in tracks, (x1,y1,x2,y2))
            if pose_inst is not None and hasattr(pose_inst, "keypoints"):
                kpts_all = pose_inst.keypoints  # (N,K,2)
                kps_all = getattr(pose_inst, "keypoint_scores", None)
                # Build pose-derived bboxes for IoU
                pose_bboxes: List[Optional[torch.Tensor]] = []
                for i in range(kpts_all.shape[0]):
                    bb = self._bbox_from_keypoints(kpts_all[i])
                    pose_bboxes.append(bb)
                # Match each track bbox to best pose
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
                    roi: Optional[Tuple[int, int, int, int]] = None
                    if best_j >= 0 and best_iou > 0.1:
                        roi = self._torso_roi_from_pose(
                            kpts_all[best_j], None if kps_all is None else kps_all[best_j], W, H
                        )
                    if roi is None:
                        roi = self._bbox_torso_fallback(tb.to(dtype=torch.int64), W, H)
                    rois.append((ti, roi))
            else:
                # Fallback: bbox-based torso crops
                for ti in range(bboxes_xyxy.shape[0]):
                    bb = bboxes_xyxy[ti].to(dtype=torch.int64)
                    rois.append((ti, self._bbox_torso_fallback(bb, W, H)))

            # Optional legibility filtering per-crop
            keep_indices = list(range(len(rois)))
            if self._legibility_enabled and not isinstance(
                self._legibility_model, _IdentityLegibility
            ):
                crops_for_leg: List[torch.Tensor] = []
                map_idx: List[int] = []
                for idx, (_, r) in enumerate(rois):
                    x1, y1, x2, y2 = r
                    crop = frame_img[:, y1:y2, x1:x2]
                    if crop.numel() == 0:
                        continue
                    crop = F.interpolate(
                        crop.unsqueeze(0).to(dtype=torch.float32),
                        size=(128, 128),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    crop = self._preprocess_imagenet(crop)
                    crops_for_leg.append(crop)
                    map_idx.append(idx)
                if crops_for_leg:
                    leg_dev = next(self._legibility_model.parameters()).device
                    batch = torch.stack(crops_for_leg, dim=0).to(device=leg_dev)
                    with torch.no_grad():
                        raw = self._legibility_model(batch).flatten()
                        # Convert to probabilities if model returns logits
                        if raw.min() < -1e-3 or raw.max() > 1.0 + 1e-3:
                            raw = torch.sigmoid(raw)
                        leg_scores = raw.detach().cpu()
                    # Log a few sample probabilities for early frames to verify values
                    try:
                        if frame_index < 2 and leg_scores.numel() > 0:
                            vals = leg_scores.tolist()
                            nshow = min(5, len(vals))
                            samples = [(int(map_idx[i]), float(vals[i])) for i in range(nshow)]
                            logger.info(
                                f"Legibility samples (frame {frame_index}): first {nshow} crops idx->prob: {samples}; "
                                f"stats min={min(vals):.3f} max={max(vals):.3f} mean={float(sum(vals)/len(vals)):.3f}"
                            )
                    except Exception:
                        pass
                    keep_mask = leg_scores >= self._legibility_threshold
                    keep_indices = [
                        map_idx[i] for i in keep_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                    ]
                else:
                    keep_indices = []

            # STR inference per kept ROI
            for local_idx in keep_indices:
                ti, r = rois[local_idx]
                x1, y1, x2, y2 = r
                crop = frame_img[:, y1:y2, x1:x2]
                if crop.numel() == 0 or (y2 - y1) < 4 or (x2 - x1) < 4:
                    continue
                # Ensure 3-channel uint8 CHW
                crop_u8 = crop.detach().to(device="cpu")
                if crop_u8.dtype != torch.uint8:
                    crop_u8 = crop_u8.clamp(0, 255).to(torch.uint8)
                if crop_u8.shape[0] == 1:
                    crop_u8 = crop_u8.repeat(3, 1, 1)
                elif crop_u8.shape[0] == 2:
                    crop_u8 = torch.cat([crop_u8, crop_u8[:1]], dim=0)

                res = self._run_parseq_on_crop(crop_u8)
                if res is None:
                    continue
                label, conf_list = res
                if not _is_valid_number(label):
                    continue
                # Multiply token probabilities except the last end token if present
                if len(conf_list) > 1:
                    total_prob = float(np.prod(conf_list[:-1]))
                else:
                    total_prob = float(np.prod(conf_list))
                try:
                    number_val = int(label)
                except Exception:
                    continue
                tid = int(tracking_ids[ti])
                self._votes[tid].append((number_val, total_prob))

            # Consolidate for visible tracks this frame
            jersey_results: List[TrackJerseyInfo] = []
            for ti in range(bboxes_xyxy.shape[0]):
                tid = int(tracking_ids[ti])
                best_num, best_w = self._aggregate_track(self._votes.get(tid, []))
                if best_num > 0:
                    jersey_results.append(
                        TrackJerseyInfo(tracking_id=tid, number=best_num, score=float(best_w))
                    )
            all_jersey_results.append(jersey_results)

        data["jersey_results"] = all_jersey_results
        return {"data": data}

    def input_keys(self):
        return {"data"}

    def output_keys(self):
        return {"data"}

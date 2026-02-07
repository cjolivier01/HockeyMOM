"""Aspen trunk that computes per-frame camera boxes (pan/zoom)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData

from hmlib.bbox.box_functions import center, clamp_box, make_box_at_center
from hmlib.builder import HM
from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, unpack_gpt_checkpoint
from hmlib.camera.camera_transformer import (
    CameraNorm,
    build_frame_base_features_torch,
    build_frame_features_torch,
)
from hmlib.camera.clusters import ClusterMan
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor

from .base import Plugin


@HM.register_module()
class CameraControllerPlugin(Plugin):
    """
    Camera controller trunk that computes per-frame camera boxes (pan/zoom).

    Modes:
      - controller="rule": defer to PlayTracker (no override boxes are produced).
      - controller="gpt": use trained GPT checkpoint.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - frame_id: int for first frame in batch

    Produces in context:
      - camera_boxes: List[torch.Tensor] (TLBR per frame)
      - Side-effects: updates each img_data_sample.pred_cam_box with TLBR tensor
    """

    def __init__(
        self,
        enabled: bool = True,
        controller: str = "rule",
        model_path: Optional[str] = None,
        window: int = 8,
        aspect_ratio: float = 16.0 / 9.0,
    ) -> None:
        super().__init__(enabled=enabled)
        self._controller = str(controller or "rule").lower()
        if self._controller not in ("rule", "gpt"):
            raise ValueError(
                f"Unsupported camera controller={self._controller!r}; expected 'rule' or 'gpt'."
            )
        self._gpt_model: Optional[CameraPanZoomGPT] = None
        self._gpt_cfg: Optional[CameraGPTConfig] = None
        self._norm: Optional[CameraNorm] = None
        self._window = int(window)
        self._feat_buf: deque = deque(maxlen=self._window)
        self._prev_center: Optional[torch.Tensor] = None
        self._prev_h: Optional[torch.Tensor] = None
        self._prev_y: Optional[torch.Tensor] = None
        self._cluster_man: Optional[ClusterMan] = None
        self._ar = float(aspect_ratio)
        self._feat_device: Optional[torch.device] = None

        if self._controller == "gpt":
            if model_path:
                try:
                    ckpt = torch.load(model_path, map_location="cpu")
                    sd, norm, w, cfg = unpack_gpt_checkpoint(ckpt)
                    self._gpt_cfg = cfg
                    self._gpt_model = CameraPanZoomGPT(cfg)
                    self._gpt_model.load_state_dict(sd)
                    self._gpt_model.eval()
                    self._norm = norm
                    self._window = int(w)
                    self._feat_buf = deque(maxlen=self._window)
                except Exception:
                    self._controller = "rule"
            else:
                self._controller = "rule"

    def _ensure_cluster_man(self, sizes: List[int] = [3, 2]):
        if self._cluster_man is None:
            self._cluster_man = ClusterMan(sizes=sizes, device="cpu")

    @staticmethod
    def _default_prev_y(d_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if int(d_out) == 3:
            return torch.tensor([0.5, 0.5, 1.0], device=device, dtype=dtype)
        if int(d_out) == 4:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype)
        if int(d_out) == 8:
            return torch.tensor(
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], device=device, dtype=dtype
            )
        return torch.zeros((int(d_out),), device=device, dtype=dtype)

    @staticmethod
    def _resolve_device(context: Dict[str, Any], fallback: Optional[torch.device] = None):
        # Prefer flattened Aspen context keys.
        for key in ("inputs", "img", "original_images"):
            t = context.get(key)
            if isinstance(t, torch.Tensor):
                return t.device
            dev = getattr(t, "device", None)
            if isinstance(dev, torch.device):
                return dev
        shared = context.get("shared", {}) or {}
        device = shared.get("camera_device") or shared.get("device")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            try:
                return torch.device(device)
            except Exception:
                pass
        if fallback is not None:
            return fallback
        if torch.cuda.is_available():
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except Exception:
                pass
        return torch.device("cpu")

    def _pose_features(
        self, pose_results: Optional[List[Any]], frame_index: int, device: torch.device
    ) -> torch.Tensor:
        """Extract a fixed-length (8) pose feature vector for the current frame."""
        feat = torch.zeros((8,), device=device, dtype=torch.float32)
        if self._norm is None:
            return feat
        try:
            if not isinstance(pose_results, list) or frame_index >= len(pose_results):
                return feat
            pr = pose_results[frame_index]
            preds = pr.get("predictions") if isinstance(pr, dict) else None
            ds0 = preds[0] if isinstance(preds, list) and preds else None
            inst0 = getattr(ds0, "pred_instances", ds0)
            bxs = getattr(inst0, "bboxes", None)
            kps = getattr(inst0, "keypoint_scores", None)
            bbox_scores = getattr(inst0, "bbox_scores", None)
            scores = getattr(inst0, "scores", None)
        except Exception:
            return feat

        if bxs is not None:
            try:
                if torch.is_tensor(bxs):
                    bxs_t = bxs.to(device=device, dtype=torch.float32)
                else:
                    bxs_t = torch.as_tensor(bxs, device=device, dtype=torch.float32)
                bxs_t = bxs_t.reshape(-1, 4)
                if bxs_t.numel() > 0:
                    cxn = (bxs_t[:, 0] + bxs_t[:, 2]) * 0.5 / max(1e-6, float(self._norm.scale_x))
                    cyn = (bxs_t[:, 1] + bxs_t[:, 3]) * 0.5 / max(1e-6, float(self._norm.scale_y))
                    hn = (bxs_t[:, 3] - bxs_t[:, 1]) / max(1e-6, float(self._norm.scale_y))
                    feat[0] = min(float(bxs_t.shape[0]) / max(1, int(self._norm.max_players)), 1.0)
                    feat[1] = torch.clamp(torch.mean(cxn), 0.0, 1.0)
                    feat[2] = torch.clamp(torch.mean(cyn), 0.0, 1.0)
                    feat[3] = torch.clamp(torch.std(cxn, unbiased=False), 0.0, 1.0)
                    feat[4] = torch.clamp(torch.std(cyn, unbiased=False), 0.0, 1.0)
                    feat[5] = torch.clamp(torch.mean(hn), 0.0, 1.0)
            except Exception:
                pass

        score_val = None
        for vv in (kps, bbox_scores, scores):
            if vv is None:
                continue
            try:
                if torch.is_tensor(vv):
                    vv_t = vv.to(device=device, dtype=torch.float32)
                else:
                    vv_t = torch.as_tensor(vv, device=device, dtype=torch.float32)
                if vv_t is not None and vv_t.numel() > 0:
                    score_val = torch.mean(vv_t)
                    break
            except Exception:
                continue
        if score_val is not None:
            feat[6] = torch.clamp(score_val, 0.0, 1.0)

        if kps is not None:
            try:
                if torch.is_tensor(kps):
                    kk = kps.to(device=device, dtype=torch.float32)
                else:
                    kk = torch.as_tensor(kps, device=device, dtype=torch.float32)
                if kk is not None and kk.numel() > 0:
                    feat[7] = torch.mean((kk > 0.5).to(dtype=feat.dtype))
            except Exception:
                pass

        return feat

    def _rink_features(self, context: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Fixed-length rink features (7,) derived from rink_profile or rink_mask_0.png."""
        feat = torch.zeros((7,), device=device, dtype=torch.float32)
        if self._norm is None:
            return feat
        sx = max(1e-6, float(self._norm.scale_x))
        sy = max(1e-6, float(self._norm.scale_y))

        rp = context.get("rink_profile")
        if isinstance(rp, dict):
            try:
                bbox = rp.get("combined_bbox")
                centroid = rp.get("centroid")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
                    feat[0] = torch.clamp(
                        torch.tensor(x1 / sx, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[1] = torch.clamp(
                        torch.tensor(y1 / sy, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[2] = torch.clamp(
                        torch.tensor(x2 / sx, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[3] = torch.clamp(
                        torch.tensor(y2 / sy, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                if centroid is not None and len(centroid) == 2:
                    cx, cy = float(centroid[0]), float(centroid[1])
                    feat[4] = torch.clamp(
                        torch.tensor(cx / sx, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                    feat[5] = torch.clamp(
                        torch.tensor(cy / sy, device=device, dtype=feat.dtype), 0.0, 1.0
                    )
                # area fraction if mask present
                mask = rp.get("combined_mask")
                if mask is not None:
                    try:
                        if torch.is_tensor(mask):
                            m = mask.to(device=device)
                        else:
                            m = torch.as_tensor(mask, device=device)
                        feat[6] = torch.clamp(torch.mean((m > 0).to(dtype=feat.dtype)), 0.0, 1.0)
                    except Exception:
                        pass
                return feat
            except Exception:
                pass

        # Fallback: load rink_mask_0.png from game_dir if present.
        try:
            import cv2

            shared = context.get("shared", {}) or {}
            game_dir = shared.get("game_dir")
            if not game_dir:
                return feat
            p = Path(str(game_dir)) / "rink_mask_0.png"
            if not p.is_file():
                return feat
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                return feat
            mask_t = torch.as_tensor(mask, device=device)
            ys, xs = torch.nonzero(mask_t > 0, as_tuple=True)
            if xs.numel() == 0 or ys.numel() == 0:
                return feat
            x1 = xs.min().to(dtype=feat.dtype)
            y1 = ys.min().to(dtype=feat.dtype)
            x2 = xs.max().to(dtype=feat.dtype)
            y2 = ys.max().to(dtype=feat.dtype)
            cx = xs.to(dtype=feat.dtype).mean()
            cy = ys.to(dtype=feat.dtype).mean()
            area = xs.numel() / float(mask.shape[0] * mask.shape[1])
            feat[0] = torch.clamp(x1 / sx, 0.0, 1.0)
            feat[1] = torch.clamp(y1 / sy, 0.0, 1.0)
            feat[2] = torch.clamp(x2 / sx, 0.0, 1.0)
            feat[3] = torch.clamp(y2 / sy, 0.0, 1.0)
            feat[4] = torch.clamp(cx / sx, 0.0, 1.0)
            feat[5] = torch.clamp(cy / sy, 0.0, 1.0)
            feat[6] = torch.clamp(torch.tensor(area, device=device, dtype=feat.dtype), 0.0, 1.0)
        except Exception:
            pass
        return feat

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        # In rule mode, defer entirely to PlayTracker's native camera controller.
        # This avoids accidentally overriding the default camera behavior on Python-only runs.
        if self._controller == "rule":
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)
        pose_results = context.get("pose_results")

        cam_boxes: List[torch.Tensor] = []
        cam_fast_boxes: List[torch.Tensor] = []
        self._ensure_cluster_man()

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            inst: InstanceData = getattr(img_data_sample, "pred_track_instances", None)
            device = self._resolve_device(context)
            if self._feat_device is None:
                self._feat_device = device
            elif self._feat_device != device:
                self._feat_device = device
                self._feat_buf.clear()
                self._prev_center = None
                self._prev_h = None
                self._prev_y = None

            ori_shape = img_data_sample.metainfo.get("ori_shape")
            H = int(ori_shape[0]) if isinstance(ori_shape, (list, tuple)) else int(1080)
            W = int(ori_shape[1]) if isinstance(ori_shape, (list, tuple)) else int(1920)
            if inst is None or not hasattr(inst, "bboxes"):
                # Default to centered wide shot
                h_px = H * 0.8
                w_px = h_px * self._ar
                cx, cy = W / 2.0, H / 2.0
                box = torch.tensor(
                    [cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2],
                    dtype=torch.float32,
                    device=device,
                )
                cam_boxes.append(box)
                setattr(img_data_sample, "pred_cam_box", box)
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and int(self._gpt_cfg.d_out) == 8
                ):
                    cam_fast_boxes.append(box)
                    setattr(img_data_sample, "pred_cam_fast_box", box)
                try:
                    w_denom = max(1.0, float(W))
                    h_denom = max(1.0, float(H))
                    self._prev_center = torch.stack(
                        [
                            (box[0] + box[2]) / (2.0 * w_denom),
                            (box[1] + box[3]) / (2.0 * h_denom),
                        ],
                        dim=0,
                    )
                    self._prev_h = (box[3] - box[1]) / h_denom
                except Exception:
                    pass
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow"))
                    == "base_prev_y"
                ):
                    # Seed prev_y from the default wide shot.
                    try:
                        w_denom = max(1.0, float(W))
                        h_denom = max(1.0, float(H))
                        slow_tlwh = torch.stack(
                            [
                                torch.clamp(box[0] / w_denom, 0.0, 1.0),
                                torch.clamp(box[1] / h_denom, 0.0, 1.0),
                                torch.clamp((box[2] - box[0]) / w_denom, 0.0, 1.0),
                                torch.clamp((box[3] - box[1]) / h_denom, 0.0, 1.0),
                            ],
                            dim=0,
                        ).to(dtype=torch.float32)
                        d_out = int(self._gpt_cfg.d_out)
                        if d_out == 3:
                            self._prev_y = torch.stack(
                                [
                                    slow_tlwh[0] + slow_tlwh[2] * 0.5,
                                    slow_tlwh[1] + slow_tlwh[3] * 0.5,
                                    slow_tlwh[3],
                                ],
                                dim=0,
                            )
                        elif d_out == 4:
                            self._prev_y = slow_tlwh
                        elif d_out == 8:
                            self._prev_y = torch.cat([slow_tlwh, slow_tlwh], dim=0)
                        else:
                            self._prev_y = self._default_prev_y(
                                d_out, device=device, dtype=torch.float32
                            )
                    except Exception:
                        self._prev_y = self._default_prev_y(
                            int(self._gpt_cfg.d_out), device=device, dtype=torch.float32
                        )
                continue

            det_tlbr = unwrap_tensor(inst.bboxes)
            inst.bboxes = wrap_tensor(det_tlbr)

            if not isinstance(det_tlbr, torch.Tensor):
                det_tlbr = torch.as_tensor(det_tlbr, device=device, dtype=torch.float32)
            device = det_tlbr.device
            if self._feat_device is None:
                self._feat_device = device
            elif self._feat_device != device:
                self._feat_device = device
                self._feat_buf.clear()
                self._prev_center = None
                self._prev_h = None
                self._prev_y = None
            track_mask = get_track_mask(inst)
            if isinstance(track_mask, torch.Tensor):
                det_tlbr = det_tlbr[track_mask]
            if det_tlbr.numel() == 0:
                h_px = H * 0.8
                w_px = h_px * self._ar
                cx, cy = W / 2.0, H / 2.0
                box = torch.tensor(
                    [cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2],
                    dtype=torch.float32,
                    device=device,
                )
                cam_boxes.append(box)
                setattr(img_data_sample, "pred_cam_box", box)
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and int(self._gpt_cfg.d_out) == 8
                ):
                    cam_fast_boxes.append(box)
                    setattr(img_data_sample, "pred_cam_fast_box", box)
                try:
                    w_denom = max(1.0, float(W))
                    h_denom = max(1.0, float(H))
                    self._prev_center = torch.stack(
                        [
                            (box[0] + box[2]) / (2.0 * w_denom),
                            (box[1] + box[3]) / (2.0 * h_denom),
                        ],
                        dim=0,
                    )
                    self._prev_h = (box[3] - box[1]) / h_denom
                except Exception:
                    pass
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow"))
                    == "base_prev_y"
                ):
                    try:
                        w_denom = max(1.0, float(W))
                        h_denom = max(1.0, float(H))
                        slow_tlwh = torch.stack(
                            [
                                torch.clamp(box[0] / w_denom, 0.0, 1.0),
                                torch.clamp(box[1] / h_denom, 0.0, 1.0),
                                torch.clamp((box[2] - box[0]) / w_denom, 0.0, 1.0),
                                torch.clamp((box[3] - box[1]) / h_denom, 0.0, 1.0),
                            ],
                            dim=0,
                        ).to(dtype=torch.float32)
                        d_out = int(self._gpt_cfg.d_out)
                        if d_out == 3:
                            self._prev_y = torch.stack(
                                [
                                    slow_tlwh[0] + slow_tlwh[2] * 0.5,
                                    slow_tlwh[1] + slow_tlwh[3] * 0.5,
                                    slow_tlwh[3],
                                ],
                                dim=0,
                            )
                        elif d_out == 4:
                            self._prev_y = slow_tlwh
                        elif d_out == 8:
                            self._prev_y = torch.cat([slow_tlwh, slow_tlwh], dim=0)
                        else:
                            self._prev_y = self._default_prev_y(
                                d_out, device=device, dtype=torch.float32
                            )
                    except Exception:
                        self._prev_y = self._default_prev_y(
                            int(self._gpt_cfg.d_out), device=device, dtype=torch.float32
                        )
                continue

            # Convert to TLWH for features
            tlwh = det_tlbr.clone()
            tlwh[:, 2] = tlwh[:, 2] - tlwh[:, 0]
            tlwh[:, 3] = tlwh[:, 3] - tlwh[:, 1]

            box_out: Optional[torch.Tensor] = None
            box_fast_out: Optional[torch.Tensor] = None
            if (
                self._controller == "gpt"
                and self._gpt_model is not None
                and self._norm is not None
                and self._gpt_cfg is not None
            ):
                pose_feat = (
                    self._pose_features(pose_results, frame_index, device)
                    if bool(getattr(self._gpt_cfg, "include_pose", False))
                    else None
                )
                rink_feat = (
                    self._rink_features(context, device)
                    if bool(getattr(self._gpt_cfg, "include_rink", False))
                    else None
                )

                if str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow")) == "base_prev_y":
                    base_feat = build_frame_base_features_torch(tlwh=tlwh, norm=self._norm)
                    if pose_feat is not None:
                        base_feat = torch.cat([base_feat, pose_feat], dim=0)
                    if rink_feat is not None:
                        base_feat = torch.cat([base_feat, rink_feat], dim=0)
                    if self._prev_y is None or int(self._prev_y.shape[0]) != int(
                        self._gpt_cfg.d_out
                    ):
                        self._prev_y = self._default_prev_y(
                            int(self._gpt_cfg.d_out), device=device, dtype=base_feat.dtype
                        )
                    feat = torch.cat([base_feat, self._prev_y.to(device=device)], dim=0)
                else:
                    feat = build_frame_features_torch(
                        tlwh=tlwh,
                        norm=self._norm,
                        prev_cam_center=self._prev_center,
                        prev_cam_h=self._prev_h,
                    )
                    if pose_feat is not None:
                        feat = torch.cat([feat, pose_feat], dim=0)
                    if rink_feat is not None:
                        feat = torch.cat([feat, rink_feat], dim=0)

                # Pad/truncate to the model's expected input dimension.
                if int(feat.shape[0]) < int(self._gpt_cfg.d_in):
                    feat = F.pad(
                        feat,
                        (0, int(self._gpt_cfg.d_in) - int(feat.shape[0])),
                        mode="constant",
                        value=0.0,
                    )
                elif int(feat.shape[0]) > int(self._gpt_cfg.d_in):
                    feat = feat[: int(self._gpt_cfg.d_in)]

                gpt_device = next(self._gpt_model.parameters()).device
                if gpt_device != device:
                    self._gpt_model.to(device)
                self._feat_buf.append(feat.unsqueeze(0))
                # GPT supports variable context length; emit a prediction as soon as we have any history.
                x = torch.cat(list(self._feat_buf), dim=0).unsqueeze(0)
                with torch.no_grad():
                    pred_seq = self._gpt_model(x).squeeze(0)
                pred_last = pred_seq[-1]

                if int(self._gpt_cfg.d_out) == 3:
                    cx, cy, hr = pred_last[0], pred_last[1], pred_last[2]
                    w_denom = max(1.0, float(W))
                    h_denom = max(1.0, float(H))
                    h_px = torch.clamp(hr * h_denom, min=1.0)
                    w_px = h_px * self._ar
                    cx_px = cx * w_denom
                    cy_px = cy * h_denom
                    # Keep box fully inside the frame without shrinking (avoid clamp distortion).
                    w_over = w_px > float(W)
                    w_px = torch.where(w_over, det_tlbr.new_tensor(float(W)), w_px)
                    h_px = torch.where(w_over, w_px / self._ar, h_px)
                    h_over = h_px > float(H)
                    h_px = torch.where(h_over, det_tlbr.new_tensor(float(H)), h_px)
                    w_px = torch.where(h_over, h_px * self._ar, w_px)
                    cx_px = torch.clamp(cx_px, w_px * 0.5, float(W) - w_px * 0.5)
                    cy_px = torch.clamp(cy_px, h_px * 0.5, float(H) - h_px * 0.5)
                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.stack([left, top, right, bottom], dim=0).to(
                        dtype=det_tlbr.dtype, device=device
                    )
                    box_out = clamp_box(
                        box_out,
                        torch.tensor([0, 0, W, H], dtype=box_out.dtype, device=device),
                    )
                    try:
                        w_denom = max(1.0, float(W))
                        h_denom = max(1.0, float(H))
                        self._prev_center = torch.stack(
                            [
                                (box_out[0] + box_out[2]) / (2.0 * w_denom),
                                (box_out[1] + box_out[3]) / (2.0 * h_denom),
                            ],
                            dim=0,
                        )
                        self._prev_h = (box_out[3] - box_out[1]) / h_denom
                    except Exception:
                        pass
                elif int(self._gpt_cfg.d_out) in (4, 8):
                    slow_tlwh = pred_last[:4]
                    w_denom = max(1.0, float(W))
                    h_denom = max(1.0, float(H))
                    cx_px = (slow_tlwh[0] + slow_tlwh[2] * 0.5) * w_denom
                    cy_px = (slow_tlwh[1] + slow_tlwh[3] * 0.5) * h_denom
                    h0 = torch.clamp(slow_tlwh[3] * h_denom, min=1.0)

                    # Enforce output aspect ratio (avoid mixed up/downscale in crop pipeline).
                    h_px = h0
                    w_px = h_px * self._ar
                    # Clamp to frame bounds while preserving aspect ratio (shrink only).
                    w_over = w_px > float(W)
                    w_px = torch.where(w_over, det_tlbr.new_tensor(float(W)), w_px)
                    h_px = torch.where(w_over, w_px / self._ar, h_px)
                    h_over = h_px > float(H)
                    h_px = torch.where(h_over, det_tlbr.new_tensor(float(H)), h_px)
                    w_px = torch.where(h_over, h_px * self._ar, w_px)

                    # Clamp center so the box fits without shrinking.
                    cx_px = torch.clamp(cx_px, w_px * 0.5, float(W) - w_px * 0.5)
                    cy_px = torch.clamp(cy_px, h_px * 0.5, float(H) - h_px * 0.5)

                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.stack([left, top, right, bottom], dim=0).to(
                        dtype=det_tlbr.dtype, device=device
                    )
                    box_out = clamp_box(
                        box_out,
                        torch.tensor([0, 0, W, H], dtype=box_out.dtype, device=device),
                    )
                    try:
                        w_denom = max(1.0, float(W))
                        h_denom = max(1.0, float(H))
                        cxn = (box_out[0] + box_out[2]) / (2.0 * w_denom)
                        cyn = (box_out[1] + box_out[3]) / (2.0 * h_denom)
                        hrn = (box_out[3] - box_out[1]) / h_denom
                        self._prev_center = torch.stack([cxn, cyn], dim=0)
                        self._prev_h = hrn
                    except Exception:
                        pass

                    if int(self._gpt_cfg.d_out) == 8:
                        fast_tlwh = pred_last[4:8]
                        xf = fast_tlwh[0] * w_denom
                        yf = fast_tlwh[1] * h_denom
                        wf = torch.clamp(fast_tlwh[2] * w_denom, min=1.0)
                        hf = torch.clamp(fast_tlwh[3] * h_denom, min=1.0)
                        box_fast_out = torch.stack([xf, yf, xf + wf, yf + hf], dim=0).to(
                            dtype=det_tlbr.dtype, device=device
                        )
                        box_fast_out = clamp_box(
                            box_fast_out,
                            torch.tensor([0, 0, W, H], dtype=box_fast_out.dtype, device=device),
                        )

            if box_out is None:
                # Rule-based fallback: union of detections -> fixed height with aspect
                left = (
                    torch.min(det_tlbr[:, 0])
                    if len(det_tlbr)
                    else torch.tensor(0.0).to(device=det_tlbr.device, non_blocking=True)
                )
                right = (
                    torch.max(det_tlbr[:, 2])
                    if len(det_tlbr)
                    else det_tlbr.new_tensor(float(W)).to(device=det_tlbr.device, non_blocking=True)
                )
                top = (
                    torch.min(det_tlbr[:, 1])
                    if len(det_tlbr)
                    else torch.tensor(0.0).to(device=det_tlbr.device, non_blocking=True)
                )
                bottom = (
                    torch.max(det_tlbr[:, 3])
                    if len(det_tlbr)
                    else det_tlbr.new_tensor(float(H)).to(device=det_tlbr.device, non_blocking=True)
                )
                uni = torch.stack([left, top, right, bottom])
                c = center(uni)
                h_px = torch.clamp((bottom - top) * 1.4, min=H * 0.35, max=H * 0.95)
                w_px = h_px * self._ar
                box_out = make_box_at_center(c, w=w_px, h=h_px)
                if not hasattr(self, "_wh_box"):
                    self._wh_box = torch.tensor([0, 0, W, H], dtype=box_out.dtype).to(
                        device=box_out.device, non_blocking=True
                    )
                box_out = clamp_box(box_out, self._wh_box)

            # For base_prev_y checkpoints, feed back the *actual* box used after clamping/aspect enforcement.
            if (
                self._controller == "gpt"
                and self._gpt_cfg is not None
                and str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow")) == "base_prev_y"
            ):
                try:
                    w_denom = max(1.0, float(W))
                    h_denom = max(1.0, float(H))
                    slow_tlwh = torch.stack(
                        [
                            torch.clamp(box_out[0] / w_denom, 0.0, 1.0),
                            torch.clamp(box_out[1] / h_denom, 0.0, 1.0),
                            torch.clamp((box_out[2] - box_out[0]) / w_denom, 0.0, 1.0),
                            torch.clamp((box_out[3] - box_out[1]) / h_denom, 0.0, 1.0),
                        ],
                        dim=0,
                    ).to(dtype=torch.float32)
                    d_out = int(self._gpt_cfg.d_out)
                    if d_out == 3:
                        self._prev_y = torch.stack(
                            [
                                slow_tlwh[0] + slow_tlwh[2] * 0.5,
                                slow_tlwh[1] + slow_tlwh[3] * 0.5,
                                slow_tlwh[3],
                            ],
                            dim=0,
                        )
                    elif d_out == 4:
                        self._prev_y = slow_tlwh
                    elif d_out == 8:
                        fast_box = box_fast_out if box_fast_out is not None else box_out
                        fast_tlwh = torch.stack(
                            [
                                torch.clamp(fast_box[0] / w_denom, 0.0, 1.0),
                                torch.clamp(fast_box[1] / h_denom, 0.0, 1.0),
                                torch.clamp((fast_box[2] - fast_box[0]) / w_denom, 0.0, 1.0),
                                torch.clamp((fast_box[3] - fast_box[1]) / h_denom, 0.0, 1.0),
                            ],
                            dim=0,
                        ).to(dtype=torch.float32)
                        self._prev_y = torch.cat([slow_tlwh, fast_tlwh], dim=0)
                    else:
                        self._prev_y = self._default_prev_y(
                            d_out, device=device, dtype=torch.float32
                        )
                except Exception:
                    self._prev_y = self._default_prev_y(
                        int(self._gpt_cfg.d_out), device=device, dtype=torch.float32
                    )

            cam_boxes.append(box_out)
            if box_fast_out is not None:
                cam_fast_boxes.append(box_fast_out)
            setattr(img_data_sample, "pred_cam_box", wrap_tensor(box_out))
            if box_fast_out is not None:
                setattr(img_data_sample, "pred_cam_fast_box", wrap_tensor(box_fast_out))

        out: Dict[str, Any] = {"camera_boxes": wrap_tensor(torch.stack(cam_boxes, dim=0))}
        if cam_fast_boxes and len(cam_fast_boxes) == len(cam_boxes):
            out["camera_fast_boxes"] = wrap_tensor(torch.stack(cam_fast_boxes, dim=0))
        return out

    def input_keys(self):
        return {
            "data_samples",
            "pose_results",
            "inputs",
            "img",
            "original_images",
            "shared",
            "rink_profile",
        }

    def output_keys(self):
        return {"camera_boxes", "camera_fast_boxes"}

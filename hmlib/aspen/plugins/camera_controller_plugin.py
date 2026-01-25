"""Aspen trunk that computes per-frame camera boxes (pan/zoom)."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import InstanceData

from pathlib import Path

from hmlib.bbox.box_functions import center, clamp_box, make_box_at_center
from hmlib.builder import HM
from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, unpack_gpt_checkpoint
from hmlib.camera.camera_transformer import (
    CameraNorm,
    CameraPanZoomTransformer,
    build_frame_base_features,
    build_frame_features,
    unpack_checkpoint,
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
      - controller="rule": cluster-based heuristic similar to PlayTracker.
      - controller="transformer": use trained transformer checkpoint.
      - controller="gpt": use trained causal transformer checkpoint.

    Expects in context:
      - data: dict with 'data_samples' list[TrackDataSample]
      - frame_id: int for first frame in batch

    Produces in context:
      - camera_boxes: List[torch.Tensor] (TLBR per frame)
      - data: updates each img_data_sample.pred_cam_box with TLBR tensor
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
        self._controller = controller
        self._model: Optional[CameraPanZoomTransformer] = None
        self._gpt_model: Optional[CameraPanZoomGPT] = None
        self._gpt_cfg: Optional[CameraGPTConfig] = None
        self._norm: Optional[CameraNorm] = None
        self._window = int(window)
        self._feat_buf: deque = deque(maxlen=self._window)
        self._prev_center: Optional[Tuple[float, float]] = None
        self._prev_h: Optional[float] = None
        self._prev_y: Optional[np.ndarray] = None
        self._cluster_man: Optional[ClusterMan] = None
        self._ar = float(aspect_ratio)

        if controller == "transformer":
            if model_path:
                try:
                    ckpt = torch.load(model_path, map_location="cpu")
                    sd, norm, w = unpack_checkpoint(ckpt)
                    self._model = CameraPanZoomTransformer(d_in=11)
                    self._model.load_state_dict(sd)
                    self._model.eval()
                    self._norm = norm
                    self._window = int(w)
                    self._feat_buf = deque(maxlen=self._window)
                except Exception:
                    # Fall back to rule-based
                    self._controller = "rule"
            else:
                # No checkpoint -> do not override PlayTracker.
                self._controller = "rule"
        elif controller == "gpt":
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
    def _default_prev_y(d_out: int) -> np.ndarray:
        if int(d_out) == 3:
            return np.asarray([0.5, 0.5, 1.0], dtype=np.float32)
        if int(d_out) == 4:
            return np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        if int(d_out) == 8:
            return np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        return np.zeros((int(d_out),), dtype=np.float32)

    def _pose_features(self, data: Dict[str, Any], frame_index: int) -> np.ndarray:
        """Extract a fixed-length (8) pose feature vector for the current frame."""
        feat = np.zeros((8,), dtype=np.float32)
        if self._norm is None:
            return feat
        try:
            pose_results = data.get("pose_results")
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
                bxs_np = (
                    bxs.detach().cpu().numpy()
                    if hasattr(bxs, "detach")
                    else np.asarray(bxs, dtype=np.float32)
                )
                bxs_np = bxs_np.reshape(-1, 4)
                if bxs_np.size:
                    cxn = (bxs_np[:, 0] + bxs_np[:, 2]) * 0.5 / max(1e-6, float(self._norm.scale_x))
                    cyn = (bxs_np[:, 1] + bxs_np[:, 3]) * 0.5 / max(1e-6, float(self._norm.scale_y))
                    hn = (bxs_np[:, 3] - bxs_np[:, 1]) / max(1e-6, float(self._norm.scale_y))
                    feat[0] = float(min(len(bxs_np) / max(1, int(self._norm.max_players)), 1.0))
                    feat[1] = float(np.clip(np.mean(cxn), 0.0, 1.0))
                    feat[2] = float(np.clip(np.mean(cyn), 0.0, 1.0))
                    feat[3] = float(np.clip(np.std(cxn), 0.0, 1.0))
                    feat[4] = float(np.clip(np.std(cyn), 0.0, 1.0))
                    feat[5] = float(np.clip(np.mean(hn), 0.0, 1.0))
            except Exception:
                pass

        score_val = None
        for vv in (kps, bbox_scores, scores):
            if vv is None:
                continue
            try:
                vv_np = (
                    vv.detach().cpu().numpy()
                    if hasattr(vv, "detach")
                    else np.asarray(vv, dtype=np.float32)
                )
                if vv_np is not None and vv_np.size:
                    score_val = float(np.mean(vv_np))
                    break
            except Exception:
                continue
        if score_val is not None:
            feat[6] = float(np.clip(score_val, 0.0, 1.0))

        if kps is not None:
            try:
                kk = (
                    kps.detach().cpu().numpy()
                    if hasattr(kps, "detach")
                    else np.asarray(kps, dtype=np.float32)
                )
                if kk is not None and kk.size:
                    feat[7] = float(np.mean(kk > 0.5))
            except Exception:
                pass

        return feat

    def _rink_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Fixed-length rink features (7,) derived from rink_profile or rink_mask_0.png."""
        feat = np.zeros((7,), dtype=np.float32)
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
                    feat[0] = float(np.clip(x1 / sx, 0.0, 1.0))
                    feat[1] = float(np.clip(y1 / sy, 0.0, 1.0))
                    feat[2] = float(np.clip(x2 / sx, 0.0, 1.0))
                    feat[3] = float(np.clip(y2 / sy, 0.0, 1.0))
                if centroid is not None and len(centroid) == 2:
                    cx, cy = float(centroid[0]), float(centroid[1])
                    feat[4] = float(np.clip(cx / sx, 0.0, 1.0))
                    feat[5] = float(np.clip(cy / sy, 0.0, 1.0))
                # area fraction if mask present
                mask = rp.get("combined_mask")
                if mask is not None:
                    try:
                        if hasattr(mask, "detach"):
                            m = mask.detach().cpu().numpy()
                        else:
                            m = np.asarray(mask)
                        feat[6] = float(np.clip(float(np.mean(m > 0)), 0.0, 1.0))
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
            ys, xs = np.nonzero(mask > 0)
            if xs.size == 0 or ys.size == 0:
                return feat
            x1 = float(xs.min())
            y1 = float(ys.min())
            x2 = float(xs.max())
            y2 = float(ys.max())
            cx = float(xs.mean())
            cy = float(ys.mean())
            area = float(xs.size) / float(mask.shape[0] * mask.shape[1])
            feat[0] = float(np.clip(x1 / sx, 0.0, 1.0))
            feat[1] = float(np.clip(y1 / sy, 0.0, 1.0))
            feat[2] = float(np.clip(x2 / sx, 0.0, 1.0))
            feat[3] = float(np.clip(y2 / sy, 0.0, 1.0))
            feat[4] = float(np.clip(cx / sx, 0.0, 1.0))
            feat[5] = float(np.clip(cy / sy, 0.0, 1.0))
            feat[6] = float(np.clip(area, 0.0, 1.0))
        except Exception:
            pass
        return feat

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        # In rule mode, defer entirely to PlayTracker's native camera controller.
        # This avoids accidentally overriding the default camera behavior on Python-only runs.
        if self._controller == "rule":
            return {"data": data}
        track_samples = data.get("data_samples")
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        cam_boxes: List[torch.Tensor] = []
        cam_fast_boxes: List[torch.Tensor] = []
        self._ensure_cluster_man()

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]
            inst: InstanceData = getattr(img_data_sample, "pred_track_instances", None)

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
                    self._prev_center = (
                        float((float(box[0]) + float(box[2])) / (2.0 * float(W))),
                        float((float(box[1]) + float(box[3])) / (2.0 * float(H))),
                    )
                    self._prev_h = float((float(box[3]) - float(box[1])) / max(1.0, float(H)))
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
                        slow_tlwh = np.asarray(
                            [
                                float(np.clip(float(box[0]) / max(1.0, float(W)), 0.0, 1.0)),
                                float(np.clip(float(box[1]) / max(1.0, float(H)), 0.0, 1.0)),
                                float(
                                    np.clip(float(box[2] - box[0]) / max(1.0, float(W)), 0.0, 1.0)
                                ),
                                float(
                                    np.clip(float(box[3] - box[1]) / max(1.0, float(H)), 0.0, 1.0)
                                ),
                            ],
                            dtype=np.float32,
                        )
                        d_out = int(self._gpt_cfg.d_out)
                        if d_out == 3:
                            self._prev_y = np.asarray(
                                [
                                    float(slow_tlwh[0] + slow_tlwh[2] * 0.5),
                                    float(slow_tlwh[1] + slow_tlwh[3] * 0.5),
                                    float(slow_tlwh[3]),
                                ],
                                dtype=np.float32,
                            )
                        elif d_out == 4:
                            self._prev_y = slow_tlwh
                        elif d_out == 8:
                            self._prev_y = np.concatenate([slow_tlwh, slow_tlwh], axis=0).astype(
                                np.float32, copy=False
                            )
                        else:
                            self._prev_y = self._default_prev_y(d_out)
                    except Exception:
                        self._prev_y = self._default_prev_y(int(self._gpt_cfg.d_out))
                continue

            det_tlbr = unwrap_tensor(inst.bboxes)
            inst.bboxes = wrap_tensor(det_tlbr)

            if not isinstance(det_tlbr, torch.Tensor):
                det_tlbr = torch.as_tensor(det_tlbr)
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
                    self._prev_center = (
                        float((float(box[0]) + float(box[2])) / (2.0 * float(W))),
                        float((float(box[1]) + float(box[3])) / (2.0 * float(H))),
                    )
                    self._prev_h = float((float(box[3]) - float(box[1])) / max(1.0, float(H)))
                except Exception:
                    pass
                if (
                    self._controller == "gpt"
                    and self._gpt_cfg is not None
                    and str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow"))
                    == "base_prev_y"
                ):
                    try:
                        slow_tlwh = np.asarray(
                            [
                                float(np.clip(float(box[0]) / max(1.0, float(W)), 0.0, 1.0)),
                                float(np.clip(float(box[1]) / max(1.0, float(H)), 0.0, 1.0)),
                                float(
                                    np.clip(float(box[2] - box[0]) / max(1.0, float(W)), 0.0, 1.0)
                                ),
                                float(
                                    np.clip(float(box[3] - box[1]) / max(1.0, float(H)), 0.0, 1.0)
                                ),
                            ],
                            dtype=np.float32,
                        )
                        d_out = int(self._gpt_cfg.d_out)
                        if d_out == 3:
                            self._prev_y = np.asarray(
                                [
                                    float(slow_tlwh[0] + slow_tlwh[2] * 0.5),
                                    float(slow_tlwh[1] + slow_tlwh[3] * 0.5),
                                    float(slow_tlwh[3]),
                                ],
                                dtype=np.float32,
                            )
                        elif d_out == 4:
                            self._prev_y = slow_tlwh
                        elif d_out == 8:
                            self._prev_y = np.concatenate([slow_tlwh, slow_tlwh], axis=0).astype(
                                np.float32, copy=False
                            )
                        else:
                            self._prev_y = self._default_prev_y(d_out)
                    except Exception:
                        self._prev_y = self._default_prev_y(int(self._gpt_cfg.d_out))
                continue

            # Convert to TLWH for features
            tlwh = det_tlbr.clone()
            tlwh[:, 2] = tlwh[:, 2] - tlwh[:, 0]
            tlwh[:, 3] = tlwh[:, 3] - tlwh[:, 1]

            box_out: Optional[torch.Tensor] = None
            box_fast_out: Optional[torch.Tensor] = None
            if (
                self._controller == "transformer"
                and self._model is not None
                and self._norm is not None
            ):
                feat = build_frame_features(
                    tlwh=tlwh.cpu().numpy(),
                    norm=self._norm,
                    prev_cam_center=self._prev_center,
                    prev_cam_h=self._prev_h,
                )
                self._feat_buf.append(torch.from_numpy(feat).unsqueeze(0))
                if len(self._feat_buf) >= self._window:
                    x = torch.cat(list(self._feat_buf), dim=0).unsqueeze(0)
                    try:
                        x = x.to(next(self._model.parameters()).device, non_blocking=True)
                    except Exception:
                        pass
                    with torch.no_grad():
                        pred = self._model(x).squeeze(0).cpu().numpy()
                    cx, cy, hr = float(pred[0]), float(pred[1]), float(pred[2])
                    self._prev_center = (cx, cy)
                    self._prev_h = hr
                    h_px = max(1.0, hr * H)
                    w_px = h_px * self._ar
                    cx_px = cx * W
                    cy_px = cy * H
                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.tensor([left, top, right, bottom], dtype=det_tlbr.dtype)
                    box_out = clamp_box(box_out, torch.tensor([0, 0, W, H], dtype=box_out.dtype))
            elif (
                self._controller == "gpt"
                and self._gpt_model is not None
                and self._norm is not None
                and self._gpt_cfg is not None
            ):
                tlwh_np = tlwh.cpu().numpy()
                pose_feat = (
                    self._pose_features(data, frame_index)
                    if bool(getattr(self._gpt_cfg, "include_pose", False))
                    else None
                )
                rink_feat = (
                    self._rink_features(context)
                    if bool(getattr(self._gpt_cfg, "include_rink", False))
                    else None
                )

                if str(getattr(self._gpt_cfg, "feature_mode", "legacy_prev_slow")) == "base_prev_y":
                    base_feat = build_frame_base_features(tlwh=tlwh_np, norm=self._norm)
                    if pose_feat is not None:
                        base_feat = np.concatenate([base_feat, pose_feat], axis=0).astype(
                            np.float32, copy=False
                        )
                    if rink_feat is not None:
                        base_feat = np.concatenate([base_feat, rink_feat], axis=0).astype(
                            np.float32, copy=False
                        )
                    if self._prev_y is None or int(self._prev_y.shape[0]) != int(
                        self._gpt_cfg.d_out
                    ):
                        self._prev_y = self._default_prev_y(int(self._gpt_cfg.d_out))
                    feat = np.concatenate([base_feat, self._prev_y], axis=0).astype(
                        np.float32, copy=False
                    )
                else:
                    feat = build_frame_features(
                        tlwh=tlwh_np,
                        norm=self._norm,
                        prev_cam_center=self._prev_center,
                        prev_cam_h=self._prev_h,
                    )
                    if pose_feat is not None:
                        feat = np.concatenate([feat, pose_feat], axis=0).astype(
                            np.float32, copy=False
                        )
                    if rink_feat is not None:
                        feat = np.concatenate([feat, rink_feat], axis=0).astype(
                            np.float32, copy=False
                        )

                # Pad/truncate to the model's expected input dimension.
                if int(feat.shape[0]) < int(self._gpt_cfg.d_in):
                    feat = np.pad(
                        feat,
                        (0, int(self._gpt_cfg.d_in) - int(feat.shape[0])),
                        mode="constant",
                        constant_values=0.0,
                    )
                elif int(feat.shape[0]) > int(self._gpt_cfg.d_in):
                    feat = feat[: int(self._gpt_cfg.d_in)]

                self._feat_buf.append(torch.from_numpy(feat).unsqueeze(0))
                # GPT supports variable context length; emit a prediction as soon as we have any history.
                x = torch.cat(list(self._feat_buf), dim=0).unsqueeze(0)
                try:
                    x = x.to(next(self._gpt_model.parameters()).device, non_blocking=True)
                except Exception:
                    pass
                with torch.no_grad():
                    pred_seq = self._gpt_model(x).squeeze(0).cpu().numpy()
                pred_last = pred_seq[-1]

                if int(self._gpt_cfg.d_out) == 3:
                    cx, cy, hr = float(pred_last[0]), float(pred_last[1]), float(pred_last[2])
                    h_px = float(max(1.0, hr * H))
                    w_px = float(h_px * self._ar)
                    cx_px = float(cx * W)
                    cy_px = float(cy * H)
                    # Keep box fully inside the frame without shrinking (avoid clamp distortion).
                    if w_px > float(W):
                        w_px = float(W)
                        h_px = float(w_px / self._ar)
                    if h_px > float(H):
                        h_px = float(H)
                        w_px = float(h_px * self._ar)
                    try:
                        cx_px = float(np.clip(cx_px, w_px * 0.5, float(W) - w_px * 0.5))
                        cy_px = float(np.clip(cy_px, h_px * 0.5, float(H) - h_px * 0.5))
                    except Exception:
                        pass
                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.tensor([left, top, right, bottom], dtype=det_tlbr.dtype)
                    box_out = clamp_box(box_out, torch.tensor([0, 0, W, H], dtype=box_out.dtype))
                    try:
                        self._prev_center = (
                            float((float(box_out[0]) + float(box_out[2])) / (2.0 * float(W))),
                            float((float(box_out[1]) + float(box_out[3])) / (2.0 * float(H))),
                        )
                        self._prev_h = float(
                            (float(box_out[3]) - float(box_out[1])) / max(1.0, float(H))
                        )
                    except Exception:
                        pass
                elif int(self._gpt_cfg.d_out) in (4, 8):
                    slow_tlwh = pred_last[:4]
                    cx_px = float((slow_tlwh[0] + slow_tlwh[2] * 0.5) * W)
                    cy_px = float((slow_tlwh[1] + slow_tlwh[3] * 0.5) * H)
                    h0 = float(max(1.0, slow_tlwh[3] * H))

                    # Enforce output aspect ratio (avoid mixed up/downscale in crop pipeline).
                    h_px = float(h0)
                    w_px = float(h_px * self._ar)
                    # Clamp to frame bounds while preserving aspect ratio (shrink only).
                    if w_px > float(W):
                        w_px = float(W)
                        h_px = float(w_px / self._ar)
                    if h_px > float(H):
                        h_px = float(H)
                        w_px = float(h_px * self._ar)

                    # Clamp center so the box fits without shrinking.
                    try:
                        cx_px = float(np.clip(cx_px, w_px * 0.5, float(W) - w_px * 0.5))
                        cy_px = float(np.clip(cy_px, h_px * 0.5, float(H) - h_px * 0.5))
                    except Exception:
                        pass

                    left = cx_px - w_px / 2.0
                    top = cy_px - h_px / 2.0
                    right = left + w_px
                    bottom = top + h_px
                    box_out = torch.tensor([left, top, right, bottom], dtype=det_tlbr.dtype)
                    box_out = clamp_box(box_out, torch.tensor([0, 0, W, H], dtype=box_out.dtype))
                    try:
                        cxn = float((float(box_out[0]) + float(box_out[2])) / (2.0 * float(W)))
                        cyn = float((float(box_out[1]) + float(box_out[3])) / (2.0 * float(H)))
                        hrn = float((float(box_out[3]) - float(box_out[1])) / max(1.0, float(H)))
                        self._prev_center = (cxn, cyn)
                        self._prev_h = hrn
                    except Exception:
                        pass

                    if int(self._gpt_cfg.d_out) == 8:
                        fast_tlwh = pred_last[4:8]
                        xf = float(fast_tlwh[0] * W)
                        yf = float(fast_tlwh[1] * H)
                        wf = float(max(1.0, fast_tlwh[2] * W))
                        hf = float(max(1.0, fast_tlwh[3] * H))
                        box_fast_out = torch.tensor(
                            [xf, yf, xf + wf, yf + hf], dtype=det_tlbr.dtype
                        )
                        box_fast_out = clamp_box(
                            box_fast_out, torch.tensor([0, 0, W, H], dtype=box_fast_out.dtype)
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
                    else torch.tensor(float(W)).to(device=det_tlbr.device, non_blocking=True)
                )
                top = (
                    torch.min(det_tlbr[:, 1])
                    if len(det_tlbr)
                    else torch.tensor(0.0).to(device=det_tlbr.device, non_blocking=True)
                )
                bottom = (
                    torch.max(det_tlbr[:, 3])
                    if len(det_tlbr)
                    else torch.tensor(float(H)).to(device=det_tlbr.device, non_blocking=True)
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
                    left_f = float(box_out[0])
                    top_f = float(box_out[1])
                    right_f = float(box_out[2])
                    bottom_f = float(box_out[3])
                    slow_tlwh = np.asarray(
                        [
                            float(np.clip(left_f / w_denom, 0.0, 1.0)),
                            float(np.clip(top_f / h_denom, 0.0, 1.0)),
                            float(np.clip((right_f - left_f) / w_denom, 0.0, 1.0)),
                            float(np.clip((bottom_f - top_f) / h_denom, 0.0, 1.0)),
                        ],
                        dtype=np.float32,
                    )
                    d_out = int(self._gpt_cfg.d_out)
                    if d_out == 3:
                        self._prev_y = np.asarray(
                            [
                                float(slow_tlwh[0] + slow_tlwh[2] * 0.5),
                                float(slow_tlwh[1] + slow_tlwh[3] * 0.5),
                                float(slow_tlwh[3]),
                            ],
                            dtype=np.float32,
                        )
                    elif d_out == 4:
                        self._prev_y = slow_tlwh
                    elif d_out == 8:
                        fast_box = box_fast_out if box_fast_out is not None else box_out
                        lf = float(fast_box[0])
                        tf = float(fast_box[1])
                        rf = float(fast_box[2])
                        bf = float(fast_box[3])
                        fast_tlwh = np.asarray(
                            [
                                float(np.clip(lf / w_denom, 0.0, 1.0)),
                                float(np.clip(tf / h_denom, 0.0, 1.0)),
                                float(np.clip((rf - lf) / w_denom, 0.0, 1.0)),
                                float(np.clip((bf - tf) / h_denom, 0.0, 1.0)),
                            ],
                            dtype=np.float32,
                        )
                        self._prev_y = np.concatenate([slow_tlwh, fast_tlwh], axis=0).astype(
                            np.float32, copy=False
                        )
                    else:
                        self._prev_y = self._default_prev_y(d_out)
                except Exception:
                    self._prev_y = self._default_prev_y(int(self._gpt_cfg.d_out))

            cam_boxes.append(box_out)
            if box_fast_out is not None:
                cam_fast_boxes.append(box_fast_out)
            setattr(img_data_sample, "pred_cam_box", wrap_tensor(box_out))
            if box_fast_out is not None:
                setattr(img_data_sample, "pred_cam_fast_box", wrap_tensor(box_fast_out))

        # Attach into the shared data dict so downstream postprocess can access
        data["camera_boxes"] = wrap_tensor(torch.stack(cam_boxes, dim=0))
        if cam_fast_boxes and len(cam_fast_boxes) == len(cam_boxes):
            data["camera_fast_boxes"] = wrap_tensor(torch.stack(cam_fast_boxes, dim=0))
        return {"data": data}

    def input_keys(self):
        return {"data"}

    def output_keys(self):
        return {"data"}

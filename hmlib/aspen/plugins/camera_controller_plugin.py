"""Aspen trunk that computes per-frame camera boxes (pan/zoom)."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.structures import InstanceData

from hmlib.bbox.box_functions import center, clamp_box, make_box_at_center
from hmlib.builder import HM
from hmlib.camera.camera_transformer import (
    CameraNorm,
    CameraPanZoomTransformer,
    build_frame_features,
    unpack_checkpoint,
)
from hmlib.camera.clusters import ClusterMan
from hmlib.utils.gpu import wrap_tensor

from .base import Plugin


@HM.register_module()
class CameraControllerPlugin(Plugin):
    """
    Camera controller trunk that computes per-frame camera boxes (pan/zoom).

    Modes:
      - controller="rule": cluster-based heuristic similar to PlayTracker.
      - controller="transformer": use trained transformer checkpoint.

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
        self._norm: Optional[CameraNorm] = None
        self._window = int(window)
        self._feat_buf: deque = deque(maxlen=self._window)
        self._prev_center: Optional[Tuple[float, float]] = None
        self._prev_h: Optional[float] = None
        self._cluster_man: Optional[ClusterMan] = None
        self._ar = float(aspect_ratio)

        if controller == "transformer" and model_path:
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

    def _ensure_cluster_man(self, sizes: List[int] = [3, 2]):
        if self._cluster_man is None:
            self._cluster_man = ClusterMan(sizes=sizes, device="cpu")

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        track_samples = data.get("data_samples")
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        cam_boxes: List[torch.Tensor] = []
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
                continue

            det_tlbr = inst.bboxes
            if not isinstance(det_tlbr, torch.Tensor):
                det_tlbr = torch.as_tensor(det_tlbr)
            # Convert to TLWH for features
            tlwh = det_tlbr.clone()
            tlwh[:, 2] = tlwh[:, 2] - tlwh[:, 0]
            tlwh[:, 3] = tlwh[:, 3] - tlwh[:, 1]

            box_out: Optional[torch.Tensor] = None
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

            cam_boxes.append(box_out)
            setattr(img_data_sample, "pred_cam_box", box_out)

        # Attach into the shared data dict so downstream postprocess can access
        data["camera_boxes"] = wrap_tensor(torch.cat(cam_boxes))
        return {"data": data}

    def input_keys(self):
        return {"data"}

    def output_keys(self):
        return {"data"}

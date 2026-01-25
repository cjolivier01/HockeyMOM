from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from hmlib.builder import HM
from hmlib.jersey.jersey_tracker import JerseyTracker
from hmlib.log import logger
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor
from hmlib.utils.image import is_channels_last, make_channels_last

from .base import Plugin


def _batch_tlbrs_to_tlwhs(tlbrs: torch.Tensor) -> torch.Tensor:
    tlwhs = tlbrs.clone()
    tlwhs[:, 2] = tlwhs[:, 2] - tlwhs[:, 0]
    tlwhs[:, 3] = tlwhs[:, 3] - tlwhs[:, 1]
    return tlwhs


def _summarize_value(v: Any) -> str:
    try:
        if isinstance(v, StreamTensorBase):
            return f"StreamTensor({tuple(v.shape)})"
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
        if isinstance(v, np.ndarray):
            return f"ndarray(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, dict):
            return f"dict(keys={len(v)})"
        if isinstance(v, (list, tuple)):
            return f"{type(v).__name__}(len={len(v)})"
    except Exception:
        pass
    return type(v).__name__


@HM.register_module()
class OverlayPlugin(Plugin):
    """Render late-stage overlays using data produced by other trunks.

    This is meant to run *after* compute-heavy trunks (pose, jersey, etc.) so
    those trunks can execute in parallel with play tracking, and overlays can
    be composed right before the camera crop/output stage.

    Expects in context:
      - img: batched image tensor [T,H,W,C] or [T,C,H,W] (from PlayTrackerPlugin)
      - data: dict that may contain pose/jersey results
      - pose_inferencer: optional (used only for its visualizer)

    Produces in context:
      - img: updated with overlays
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        plot_pose: bool = False,
        pose_kpt_thr: Optional[float] = None,
        plot_jersey_numbers: bool = False,
        plot_tracking: bool = False,
        plot_all_detections: Optional[float] = None,
        plot_trajectories: bool = False,
        draw_tracking_circles: bool = True,
        trajectory_history: int = 75,
        print_available: bool = True,
        print_every: int = 0,
    ) -> None:
        super().__init__(enabled=enabled)
        self._plot_pose = bool(plot_pose)
        self._pose_kpt_thr = pose_kpt_thr
        self._plot_jersey_numbers = bool(plot_jersey_numbers)
        self._plot_tracking = bool(plot_tracking)
        self._plot_all_detections = plot_all_detections
        self._plot_trajectories = bool(plot_trajectories)
        self._draw_tracking_circles = bool(draw_tracking_circles)
        self._trajectory_history = max(1, int(trajectory_history))
        self._print_available = bool(print_available)
        self._print_every = max(0, int(print_every))
        self._printed_once = False
        self._jersey_tracker = JerseyTracker(show=bool(plot_jersey_numbers))
        self._traj: Dict[int, list[tuple[int, int]]] = {}

    def _maybe_print_available(self, context: Dict[str, Any]) -> None:
        if not self._print_available:
            return
        if self._print_every <= 0:
            if self._printed_once:
                return
        else:
            if (self._iter_num % self._print_every) != 0:
                return

        keys = sorted(context.keys())
        data = context.get("data") or {}
        data_keys = sorted(data.keys()) if isinstance(data, dict) else []
        msg = (
            f"OverlayPlugin available keys: {keys} | data keys: {data_keys} | "
            f"img={_summarize_value(context.get('img'))} | "
            f"frame_ids={_summarize_value(context.get('frame_ids'))} | "
            f"pose_results={_summarize_value(data.get('pose_results')) if isinstance(data, dict) else 'n/a'} | "
            f"jersey_results={_summarize_value(data.get('jersey_results')) if isinstance(data, dict) else 'n/a'}"
        )
        logger.info(msg)
        self._printed_once = True

    @staticmethod
    def _to_batched_image(img_any: Any) -> Optional[torch.Tensor]:
        if img_any is None:
            return None
        if isinstance(img_any, StreamTensorBase):
            img_any = img_any.wait()
        if isinstance(img_any, torch.Tensor):
            img = img_any
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim != 4:
                return None
            if not is_channels_last(img):
                img = make_channels_last(img)
            return img
        if isinstance(img_any, np.ndarray):
            img = torch.from_numpy(img_any)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim != 4:
                return None
            if not is_channels_last(img):
                img = make_channels_last(img)
            return img
        return None

    @staticmethod
    def _get_track_data_sample(data: Dict[str, Any]) -> Any:
        tds = data.get("data_samples")
        if isinstance(tds, list):
            if not tds:
                return None
            return tds[0]
        return tds

    @staticmethod
    def _assign_frame(img_batch: torch.Tensor, index: int, new_img: Any) -> None:
        if isinstance(new_img, StreamTensorBase):
            new_img = new_img.wait()
        if isinstance(new_img, np.ndarray):
            new_img = torch.from_numpy(new_img)
        if not isinstance(new_img, torch.Tensor):
            return
        if new_img.ndim == 4 and new_img.shape[0] == 1:
            new_img = new_img.squeeze(0)
        if new_img.ndim != 3:
            return
        if not is_channels_last(new_img):
            new_img = make_channels_last(new_img)
        if new_img.device != img_batch.device:
            new_img = new_img.to(device=img_batch.device, non_blocking=True)
        img_batch[index] = new_img

    def _update_and_draw_trajectories(
        self, img: torch.Tensor, tracking_ids: torch.Tensor, bboxes_tlwh: torch.Tensor
    ) -> torch.Tensor:
        if (not self._plot_trajectories) or tracking_ids.numel() == 0:
            return img

        for (x1, y1, w, h), tid in zip(bboxes_tlwh, tracking_ids):
            tid_i = int(tid)
            cx = int(float(x1 + 0.5 * w))
            cy = int(float(y1 + h))
            pts = self._traj.get(tid_i)
            if pts is None:
                pts = []
                self._traj[tid_i] = pts
            pts.append((cx, cy))
            if len(pts) > self._trajectory_history:
                del pts[: len(pts) - self._trajectory_history]

        for tid_i, pts in list(self._traj.items()):
            if len(pts) < 2:
                continue
            color = vis.get_color(abs(int(tid_i)))
            for (x0, y0), (x1p, y1p) in zip(pts[:-1], pts[1:]):
                img = vis.plot_line(img, (x0, y0), (x1p, y1p), color=color, thickness=2)
        return img

    def _draw_tracking_and_dets(self, img: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        if (
            (not self._plot_tracking)
            and (self._plot_all_detections is None)
            and (not self._plot_trajectories)
        ):
            return img

        track_data_sample = self._get_track_data_sample(data)
        vds = getattr(track_data_sample, "video_data_samples", None) if track_data_sample else None
        if vds is None:
            return img

        frame_count = min(img.shape[0], len(vds))
        for frame_index in range(frame_count):
            vs = vds[frame_index]

            if self._plot_all_detections is not None:
                try:
                    det_inst = getattr(vs, "pred_instances", None)
                    dets = getattr(det_inst, "bboxes", None) if det_inst is not None else None
                    scores = getattr(det_inst, "scores", None) if det_inst is not None else None
                    if isinstance(dets, torch.Tensor) and isinstance(scores, torch.Tensor):
                        img_f = img[frame_index]
                        for det, score in zip(dets, scores):
                            if float(score) >= float(self._plot_all_detections):
                                img_f = vis.plot_rectangle(
                                    img=img_f, box=det, color=(64, 64, 64), thickness=1
                                )
                                if float(score) < 0.7:
                                    img_f = vis.plot_text(
                                        img_f,
                                        format(float(score), ".2f"),
                                        (int(det[0]), int(det[1])),
                                        1,
                                        1.0,
                                        (255, 255, 255),
                                        thickness=1,
                                    )
                        self._assign_frame(img, frame_index, img_f)
                except Exception:
                    pass

            track_inst = getattr(vs, "pred_track_instances", None)
            if track_inst is None:
                continue
            try:
                ids = unwrap_tensor(track_inst.instances_id)
                bboxes_tlbr = unwrap_tensor(track_inst.bboxes)
            except Exception:
                continue
            if (not isinstance(ids, torch.Tensor)) or (not isinstance(bboxes_tlbr, torch.Tensor)):
                continue
            if bboxes_tlbr.numel() == 0:
                continue

            track_mask = get_track_mask(track_inst)
            if isinstance(track_mask, torch.Tensor):
                ids = ids[track_mask]
                bboxes_tlbr = bboxes_tlbr[track_mask]
            if ids.numel() == 0:
                continue

            bboxes_tlwh = _batch_tlbrs_to_tlwhs(bboxes_tlbr)

            if self._plot_trajectories:
                img_f = img[frame_index]
                img_f = self._update_and_draw_trajectories(img_f, ids, bboxes_tlwh)
                self._assign_frame(img, frame_index, img_f)

            if self._plot_tracking:
                try:
                    frame_id_val = int(getattr(vs, "frame_id", frame_index))
                except Exception:
                    frame_id_val = frame_index
                img_f = img[frame_index]
                img_f = vis.plot_tracking(
                    img_f,
                    bboxes_tlwh,
                    ids,
                    frame_id=torch.tensor(frame_id_val),
                    speeds=[],
                    line_thickness=2,
                    ignore_frame_id=True,
                    ignore_tracking_ids=None,
                    ignored_color=(0, 0, 0),
                    draw_tracking_circles=self._draw_tracking_circles,
                )
                self._assign_frame(img, frame_index, img_f)

        return img

    def _draw_pose(
        self, img: torch.Tensor, pose_results: Any, pose_inferencer: Any
    ) -> torch.Tensor:
        if not self._plot_pose:
            return img
        if not pose_results or pose_inferencer is None:
            return img

        pose_impl = getattr(pose_inferencer, "inferencer", None)
        vis = getattr(pose_impl, "visualizer", None)
        if vis is None:
            return img

        kpt_thr = self._pose_kpt_thr
        if kpt_thr is None:
            kpt_thr = getattr(pose_inferencer, "filter_args", {}).get("kpt_thr", None)
        if kpt_thr is None:
            kpt_thr = 0.3

        frame_count = min(img.shape[0], len(pose_results))
        for i in range(frame_count):
            frame_res = pose_results[i] if i < len(pose_results) else None
            if not isinstance(frame_res, dict):
                continue
            preds = frame_res.get("predictions")
            if not preds:
                continue
            try:
                out_img = vis.add_datasample(
                    name="pose results",
                    image=img[i],
                    data_sample=preds[0],
                    clone_image=False,
                    draw_gt=False,
                    draw_bbox=False,
                    show_kpt_idx=False,
                    kpt_thr=float(kpt_thr),
                )
                self._assign_frame(img, i, out_img)
            except Exception:
                continue
        return img

    def _draw_jerseys(self, img: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        if not self._plot_jersey_numbers:
            return img
        jersey_results = data.get("jersey_results")
        if not jersey_results:
            return img
        track_samples = data.get("data_samples")
        if track_samples is None:
            return img
        if isinstance(track_samples, list):
            if not track_samples:
                return img
            track_samples = track_samples[0]

        vds = getattr(track_samples, "video_data_samples", None)
        if vds is None:
            return img

        frame_count = min(img.shape[0], len(vds), len(jersey_results))
        frame_ids_any = data.get("frame_ids", None)
        if frame_ids_any is None:
            frame_ids_any = data.get("frame_id", None)

        for frame_index in range(frame_count):
            vs = vds[frame_index]
            track_inst = getattr(vs, "pred_track_instances", None)
            if track_inst is None:
                continue
            try:
                ids = unwrap_tensor(track_inst.instances_id)
                bboxes_tlbr = unwrap_tensor(track_inst.bboxes)
            except Exception:
                continue

            if not isinstance(ids, torch.Tensor) or not isinstance(bboxes_tlbr, torch.Tensor):
                continue
            if bboxes_tlbr.numel() == 0:
                continue

            track_mask = get_track_mask(track_inst)
            if isinstance(track_mask, torch.Tensor):
                ids = ids[track_mask]
                bboxes_tlbr = bboxes_tlbr[track_mask]

            tlwh = _batch_tlbrs_to_tlwhs(bboxes_tlbr)

            frame_id_val = frame_index
            try:
                frame_id_val = int(getattr(vs, "frame_id", frame_index))
            except Exception:
                frame_id_val = frame_index

            per_frame = jersey_results[frame_index]
            if per_frame:
                for info in per_frame:
                    try:
                        self._jersey_tracker.observe_tracking_id_number_info(
                            frame_id=frame_id_val, info=info
                        )
                    except Exception:
                        continue
            try:
                img_frame = self._jersey_tracker.draw(
                    image=img[frame_index], tracking_ids=ids, bboxes=tlwh
                )
                self._assign_frame(img, frame_index, img_frame)
            except Exception:
                continue
        return img

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        self._maybe_print_available(context)

        data: Dict[str, Any] = context.get("data", {}) or {}
        img_any = context.get("img")
        if img_any is None:
            img_any = data.get("img")
        if img_any is None:
            img_any = data.get("original_images")
        if img_any is None:
            img_any = context.get("original_images")
        img = self._to_batched_image(img_any)
        if img is None:
            return {}

        should_render = (
            self._plot_pose
            or self._plot_jersey_numbers
            or self._plot_tracking
            or (self._plot_all_detections is not None)
            or self._plot_trajectories
        )
        if not should_render and not self._print_available:
            return {}

        # Copy so in-place writes are safe even if upstream shares storage.
        if should_render:
            img = img.clone()

        pose_inferencer = context.get("pose_inferencer") or context.get("shared", {}).get(
            "pose_inferencer"
        )
        pose_results = data.get("pose_results")
        if pose_results is not None:
            img = self._draw_pose(
                img=img, pose_results=pose_results, pose_inferencer=pose_inferencer
            )

        img = self._draw_tracking_and_dets(img=img, data=data)
        img = self._draw_jerseys(img=img, data=data)

        # Preserve channels-last tensor for downstream ApplyCamera/VideoOutput.
        img = make_channels_last(img)
        return {"img": wrap_tensor(img)}

    def input_keys(self):
        return {"img", "data", "pose_inferencer", "shared"}

    def output_keys(self):
        return {"img"}

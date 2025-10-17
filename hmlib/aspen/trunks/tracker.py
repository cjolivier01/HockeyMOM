from typing import Any, Dict, List, Optional

import torch
from mmengine.structures import InstanceData

from hockeymom.core import HmByteTrackConfig, HmTracker, HmTrackerPredictionMode

from .base import Trunk


class TrackerTrunk(Trunk):
    """
    Tracker trunk that consumes per-frame detections and produces tracks.

    It wraps the C++ `HmTracker` with configurable thresholds.

    Expects in context:
      - data: dict with 'data_samples' list[TrackDataSample], possibly 'dataset_results'
      - frame_id: int for first frame in the current batch
      - tracking_dataframe, detection_dataframe: optional sinks
      - using_precalculated_tracking, using_precalculated_detection: bools
      - detect_timer: optional timer (already handled by detector trunk)

    Produces in context:
      - data: unchanged reference (with `pred_track_instances` filled)
      - data: pruned copy without heavy tensors
      - nr_tracks: int (active track count)
      - max_tracking_id: int
    """

    def __init__(self, enabled: bool = True, cpp_tracker: bool = True):
        super().__init__(enabled=enabled)
        self._cpp_tracker = bool(cpp_tracker)
        self._hm_tracker: Optional[HmTracker] = None

    def _ensure_tracker(self):
        if self._hm_tracker is not None:
            return
        config = HmByteTrackConfig()
        config.init_track_thr = 0.7
        config.obj_score_thrs_low = 0.1
        config.obj_score_thrs_high = 0.3
        config.match_iou_thrs_high = 0.1
        config.match_iou_thrs_low = 0.5
        config.match_iou_thrs_tentative = 0.3
        config.track_buffer_size = 60
        config.return_user_ids = False
        config.return_track_age = False
        config.prediction_mode = HmTrackerPredictionMode.BoundingBox
        self._hm_tracker = HmTracker(config)

    # post-detection pipeline deprecated; pruning handled by a dedicated trunk

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        # preserved_original_images = data.get("original_images")
        dataset_results = data.get("dataset_results") or context.get("dataset_results")
        frame_id0: int = int(context.get("frame_id", -1))

        using_precalc_track: bool = bool(context.get("using_precalculated_tracking", False))
        using_precalc_det: bool = bool(context.get("using_precalculated_detection", False))

        self._ensure_tracker()

        # Access TrackDataSample list
        track_samples = data.get("data_samples")
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        max_tracking_id = 0
        active_track_count = 0
        all_frame_jersey_info: List[List[Any]] = []

        def _to_tensor_1d(x):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.ndim == 0:
                x = x.unsqueeze(0)
            return x

        def _to_bboxes_2d(x):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.ndim == 1:
                # If empty, reshape to (0, 4); if size==4, make (1,4)
                if x.numel() == 0:
                    return x.reshape(0, 4)
                x = x.unsqueeze(0)
            return x

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]

            # If precomputed tracking is used, skip tracker and only log data
            if using_precalc_track:
                # Keep pred_track_instances unset; logging handled below if needed
                continue
            # Detections should have been attached by DetectorInferenceTrunk
            det_instances = getattr(img_data_sample, "pred_instances", None)
            if det_instances is None:
                # No detections, skip tracking; leave pred_track_instances unset
                continue

            det_bboxes = det_instances.bboxes
            det_labels = det_instances.labels
            det_scores = det_instances.scores
            det_src_pose_idx = getattr(det_instances, "source_pose_index", None)

            # Post-detection pruning is handled by a dedicated trunk upstream

            # Provide frame id for tracker aging
            frame_id = img_data_sample.metainfo.get("img_id")
            if isinstance(frame_id, torch.Tensor):
                frame_id = frame_id.reshape([1])[0].item()
            if frame_id is None:
                frame_id = frame_id0 + frame_index

            # Use C++ tracker
            assert self._hm_tracker is not None
            # Ensure tensors with correct dimensionality
            det_bboxes = _to_bboxes_2d(det_bboxes)
            det_labels = _to_tensor_1d(det_labels).to(dtype=torch.long)
            det_scores = _to_tensor_1d(det_scores).to(dtype=torch.float32)
            # Align lengths defensively
            N = int(det_bboxes.shape[0])
            if len(det_labels) != N:
                if len(det_labels) == 1 and N > 1:
                    det_labels = det_labels.expand(N).clone()
                else:
                    det_labels = torch.full((N,), int(det_labels[0].item()) if len(det_labels) else 0,
                                            dtype=torch.long, device=det_bboxes.device)
            if len(det_scores) != N:
                if len(det_scores) == 1 and N > 1:
                    det_scores = det_scores.expand(N).clone()
                else:
                    det_scores = torch.ones((N,), dtype=torch.float32, device=det_bboxes.device)

            ll1 = len(det_bboxes)
            assert len(det_labels) == ll1 and len(det_scores) == ll1
            # Ensure tracker receives torch tensors
            # (already tensors above)
            results = self._hm_tracker.track(
                data=dict(
                    frame_id=torch.tensor([frame_id], dtype=torch.int64),
                    bboxes=det_bboxes,
                    labels=det_labels,
                    scores=det_scores,
                )
            )
            ids = results.get("user_ids", results.get("ids"))
            ll2 = len(ids)
            assert len(results["bboxes"]) == ll2 and len(results["scores"]) == ll2 and len(results["labels"]) == ll2

            pred_track_instances = InstanceData(
                instances_id=ids.cpu(),
                bboxes=results["bboxes"].cpu(),
                scores=results["scores"].cpu(),
                labels=results["labels"].cpu(),
            )
            # Propagate source pose indices from detections to per-frame tracks
            try:
                src_idx = getattr(img_data_sample.pred_instances, "source_pose_index", None)
                if src_idx is not None:
                    tb = pred_track_instances.bboxes
                    db = img_data_sample.pred_instances.bboxes
                    if not isinstance(tb, torch.Tensor):
                        tb = torch.as_tensor(tb)
                    if not isinstance(db, torch.Tensor):
                        db = torch.as_tensor(db)
                    if tb.ndim == 1:
                        tb = tb.reshape(-1, 4)
                    if db.ndim == 1:
                        db = db.reshape(-1, 4)
                    mapped = torch.full((len(tb),), -1, dtype=torch.int64)
                    # Try exact match first
                    for j in range(len(tb)):
                        eq = torch.isclose(tb[j], db).all(dim=1) if len(db) else torch.zeros((0,), dtype=torch.bool)
                        match_idx = torch.nonzero(eq).reshape(-1)
                        if len(match_idx) == 1:
                            k = int(match_idx[0].item())
                            try:
                                mapped[j] = int(src_idx[k])
                            except Exception:
                                pass
                    if (mapped < 0).any() and len(tb) and len(db):
                        try:
                            from hmlib.tracking_utils.utils import bbox_iou as _bbox_iou
                        except Exception:
                            from hmlib.utils.utils import bbox_iou as _bbox_iou
                        iou = _bbox_iou(tb.to(dtype=torch.float32), db.to(dtype=torch.float32), x1y1x2y2=True)
                        best_iou, best_idx = torch.max(iou, dim=1)
                        for j in range(len(tb)):
                            if mapped[j] < 0 and best_iou[j] > 0:
                                try:
                                    mapped[j] = int(src_idx[int(best_idx[j].item())])
                                except Exception:
                                    pass
                    pred_track_instances.source_pose_index = mapped
            except Exception:
                pass
            active_track_count = max(active_track_count, len(pred_track_instances.instances_id))
            img_data_sample.pred_track_instances = pred_track_instances
            # Provide a simple attribute for downstream postprocessors that expect it
            try:
                setattr(img_data_sample, "frame_id", int(frame_id))
            except Exception:
                pass

            # Saving to dataframes is now handled by dedicated Save* trunks.

            # For performance: record current active tracks
            img_data_sample.set_metainfo({"nr_tracks": active_track_count})

            if len(pred_track_instances.instances_id):
                max_id = int(torch.max(pred_track_instances.instances_id))
                if max_id > max_tracking_id:
                    max_tracking_id = max_id

        # if preserved_original_images is not None:
        #     data["original_images"] = preserved_original_images
        # elif "original_images" in data:
        #     data["original_images"] = data["original_images"]
        # if "img" in data:
        #     del data["img"]

        return {
            "data": data,
            "nr_tracks": active_track_count,
            "max_tracking_id": max_tracking_id,
        }

    def input_keys(self):
        return {
            "data",
            "frame_id",
            "using_precalculated_tracking",
            "using_precalculated_detection",
            # no longer depends on model's post-detection pipeline
        }

    def output_keys(self):
        return {"data", "nr_tracks", "max_tracking_id"}

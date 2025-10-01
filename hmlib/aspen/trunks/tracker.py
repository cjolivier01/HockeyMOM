from typing import Any, Dict, List, Optional

import torch
from mmengine.structures import InstanceData
from mmcv.transforms import Compose

from .base import Trunk

from hockeymom.core import HmByteTrackConfig, HmTracker, HmTrackerPredictionMode


class TrackerTrunk(Trunk):
    """
    Tracker trunk that consumes per-frame detections and produces tracks.

    It wraps the C++ `HmTracker` with configurable thresholds. Optionally
    applies a post-detection pipeline before tracking (read from `model`
    if present in context).

    Expects in context:
      - data: dict with 'data_samples' list[TrackDataSample], possibly 'dataset_results'
      - frame_id: int for first frame in the current batch
      - tracking_dataframe, detection_dataframe: optional sinks
      - using_precalculated_tracking, using_precalculated_detection: bools
      - model: optional object with attribute `post_detection_pipeline` (list of transforms)
      - detect_timer: optional timer (already handled by detector trunk)

    Produces in context:
      - data: unchanged reference (with `pred_track_instances` filled)
      - data_to_send: pruned copy without heavy tensors
      - nr_tracks: int (active track count)
      - max_tracking_id: int
    """

    def __init__(self, enabled: bool = True, cpp_tracker: bool = True):
        super().__init__(enabled=enabled)
        self._cpp_tracker = bool(cpp_tracker)
        self._hm_tracker: Optional[HmTracker] = None
        self._post_det_compose: Optional[Compose] = None

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

    def _ensure_post_det_pipeline(self, model_like: Optional[object]):
        if self._post_det_compose is not None:
            return
        pipeline = getattr(model_like, "post_detection_pipeline", None)
        if pipeline:
            self._post_det_compose = Compose(pipeline)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        preserved_original_images = data.get("original_images")
        dataset_results = data.get("dataset_results") or context.get("dataset_results")
        frame_id0: int = int(context.get("frame_id", -1))

        using_precalc_track: bool = bool(context.get("using_precalculated_tracking", False))
        using_precalc_det: bool = bool(context.get("using_precalculated_detection", False))
        tracking_dataframe = context.get("tracking_dataframe")
        detection_dataframe = context.get("detection_dataframe")

        self._ensure_tracker()
        self._ensure_post_det_pipeline(context.get("model"))

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

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]

            # If precomputed tracking is used, skip tracker and only log data_to_send
            if using_precalc_track:
                # Keep pred_track_instances unset; logging handled below if needed
                continue
            # Detections should have been attached by DetectorTrunk
            det_instances = getattr(img_data_sample, "pred_instances", None)
            if det_instances is None:
                # No detections, skip tracking; leave pred_track_instances unset
                continue

            det_bboxes = det_instances.bboxes
            det_labels = det_instances.labels
            det_scores = det_instances.scores

            # Prepare optional post-detection pipeline input
            if self._post_det_compose is not None:
                pd_input: Dict[str, Any] = {
                    "det_bboxes": det_bboxes,
                    "labels": det_labels,
                    "scores": det_scores,
                    "prune_list": ["det_bboxes", "labels", "scores"],
                    "ori_shape": img_data_sample.metainfo.get("ori_shape"),
                    "data_samples": data.get("data_samples"),
                }
                if dataset_results:
                    pd_input["dataset_results"] = dataset_results
                pd_output = self._post_det_compose(pd_input)
                det_bboxes = pd_output["det_bboxes"]
                det_labels = pd_output["labels"]
                det_scores = pd_output["scores"]

                # Update pred_instances with pruned results
                new_inst = InstanceData()
                new_inst.scores = det_scores
                new_inst.labels = det_labels
                new_inst.bboxes = det_bboxes
                img_data_sample.pred_instances = new_inst

            # Provide frame id for tracker aging
            frame_id = img_data_sample.metainfo.get("img_id")
            if isinstance(frame_id, torch.Tensor):
                frame_id = frame_id.reshape([1])[0].item()
            if frame_id is None:
                frame_id = frame_id0 + frame_index

            # Use C++ tracker
            assert self._hm_tracker is not None
            ll1 = len(det_bboxes)
            assert len(det_labels) == ll1 and len(det_scores) == ll1
            results = self._hm_tracker.track(
                data=dict(frame_id=torch.tensor([frame_id], dtype=torch.int64), bboxes=det_bboxes, labels=det_labels, scores=det_scores)
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
            active_track_count = max(active_track_count, len(pred_track_instances.instances_id))
            img_data_sample.pred_track_instances = pred_track_instances

            # Logging
            if not using_precalc_track and tracking_dataframe is not None:
                jersey_results = None
                tracking_dataframe.add_frame_records(
                    frame_id=frame_id0 + frame_index,
                    tracking_ids=pred_track_instances.instances_id,
                    tlbr=pred_track_instances.bboxes,
                    scores=pred_track_instances.scores,
                    labels=pred_track_instances.labels,
                    jersey_info=(jersey_results),
                )
            if not using_precalc_det and detection_dataframe is not None:
                detection_dataframe.add_frame_records(
                    frame_id=frame_id0,
                    scores=img_data_sample.pred_instances.scores,
                    labels=img_data_sample.pred_instances.labels,
                    bboxes=img_data_sample.pred_instances.bboxes,
                )

            # For performance: record current active tracks
            img_data_sample.set_metainfo({"nr_tracks": active_track_count})

            if len(pred_track_instances.instances_id):
                max_id = int(torch.max(pred_track_instances.instances_id))
                if max_id > max_tracking_id:
                    max_tracking_id = max_id

        data_to_send = data.copy()
        if preserved_original_images is not None:
            data_to_send["original_images"] = preserved_original_images
        elif "original_images" in data:
            data_to_send["original_images"] = data["original_images"]
        if "img" in data_to_send:
            del data_to_send["img"]

        return {
            "data": data,
            "data_to_send": data_to_send,
            "nr_tracks": active_track_count,
            "max_tracking_id": max_tracking_id,
        }

    def input_keys(self):
        return {
            "data",
            "frame_id",
            "tracking_dataframe",
            "detection_dataframe",
            "using_precalculated_tracking",
            "using_precalculated_detection",
            "model",
        }

    def output_keys(self):
        return {"data", "data_to_send", "nr_tracks", "max_tracking_id"}

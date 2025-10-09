from typing import Any, Dict, Optional

import torch
from torch.cuda.amp import autocast

from .base import Trunk


class MMTrackingTrunk(Trunk):
    """
    Runs an MMTracking-style model that returns tracking results.

    Expects in context:
      - data: dict prepared by ImagePrepTrunk (has 'img' and 'original_images')
      - model: callable like mmdet model (already eval mode)
      - fp16: bool
      - tracking_dataframe, detection_dataframe: optional for logging
      - frame_id: int
      - using_precalculated_tracking, using_precalculated_detection: bool

    Produces in context:
      - data: model outputs merged
      - data_to_send: pruned dict for downstream post-processing
      - nr_tracks, max_tracking_id
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        # Preserve original_images from input context before model overwrites data
        preserved_original_images = data.get("original_images")
        model = context["model"]
        fp16: bool = context.get("fp16", False)

        using_precalculated_tracking: bool = context.get("using_precalculated_tracking", False)
        using_precalculated_detection: bool = context.get("using_precalculated_detection", False)
        tracking_dataframe = context.get("tracking_dataframe")
        detection_dataframe = context.get("detection_dataframe")
        frame_id: int = int(context["frame_id"]) if "frame_id" in context else -1

        detect_timer = context.get("detect_timer")
        if detect_timer is not None:
            detect_timer.tic()

        # If the model is an Aspen Trunk, delegate end-to-end to it
        if isinstance(model, Trunk):
            # Let the model's trunk implementation handle timing and context updates
            return model(context)

        with torch.no_grad():
            with autocast() if fp16 else torch.cuda.amp.autocast(enabled=False):
                data = model(return_loss=False, rescale=True, **data)

        if detect_timer is not None:
            detect_timer.toc()

        track_data_sample = data["data_samples"]
        nr_tracks = int(track_data_sample.video_data_samples[0].metainfo["nr_tracks"])
        tracking_ids = track_data_sample.video_data_samples[-1].pred_track_instances.instances_id
        max_tracking_id = int(torch.max(tracking_ids)) if len(tracking_ids) else 0

        jersey_results = data.get("jersey_results")
        for frame_index, video_data_sample in enumerate(track_data_sample.video_data_samples):
            pred_track_instances = getattr(video_data_sample, "pred_track_instances", None)
            if pred_track_instances is None:
                continue
            if not using_precalculated_tracking and tracking_dataframe is not None:
                tracking_dataframe.add_frame_records(
                    frame_id=frame_id + frame_index,
                    tracking_ids=pred_track_instances.instances_id,
                    tlbr=pred_track_instances.bboxes,
                    scores=pred_track_instances.scores,
                    labels=pred_track_instances.labels,
                    jersey_info=(jersey_results[frame_index] if jersey_results is not None else None),
                )
            if not using_precalculated_detection and detection_dataframe is not None:
                detection_dataframe.add_frame_records(
                    frame_id=frame_id,
                    scores=video_data_sample.pred_instances.scores,
                    labels=video_data_sample.pred_instances.labels,
                    bboxes=video_data_sample.pred_instances.bboxes,
                )

        # data_to_send = data.copy()
        # # Avoid passing big tensors downstream unnecessarily
        # # Re-attach original images from pre-model context if missing
        # if preserved_original_images is not None:
        #     data_to_send["original_images"] = preserved_original_images
        # elif "original_images" in data:
        #     data_to_send["original_images"] = data["original_images"]
        # if "img" in data_to_send:
        #     del data_to_send["img"]

        return {
            "data": data,
            # "data": data_to_send,
            "nr_tracks": nr_tracks,
            "max_tracking_id": max_tracking_id,
        }

    def input_keys(self):
        return {
            "data",
            "model",
            "fp16",
            "using_precalculated_tracking",
            "using_precalculated_detection",
            "tracking_dataframe",
            "detection_dataframe",
            "frame_id",
            "detect_timer",
        }

    def output_keys(self):
        return {"data", "data", "nr_tracks", "max_tracking_id"}

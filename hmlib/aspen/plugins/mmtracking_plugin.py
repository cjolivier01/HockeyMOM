from typing import Any, Dict

import numpy as np
import torch

from hmlib.tracking_utils.utils import get_track_mask

from .base import Plugin


class MMTrackingPlugin(Plugin):
    """
    Runs an MMTracking-style model that returns tracking results.

    Expects in context:
      - inputs: model input tensor
      - data_samples: TrackDataSample (or list)
      - model: callable like mmdet model (already eval mode)
      - fp16: bool
      - tracking_dataframe, detection_dataframe: optional for logging
      - frame_id: int
      - using_precalculated_tracking, using_precalculated_detection: bool

    Produces in context:
      - data_samples: updated tracking samples from the model
      - nr_tracks, max_tracking_id
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

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

        # If the model is an Aspen Plugin, delegate end-to-end to it
        if isinstance(model, Plugin):
            # Let the model's trunk implementation handle timing and context updates
            return model(context)

        inputs_any = context.get("inputs")
        data_samples_any = context.get("data_samples")
        if inputs_any is None or data_samples_any is None:
            return {}
        from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor

        inputs = (
            unwrap_tensor(inputs_any) if isinstance(inputs_any, StreamTensorBase) else inputs_any
        )

        data: Dict[str, Any] = {"inputs": inputs, "data_samples": data_samples_any}

        with torch.no_grad():
            # Enable AMP only if requested and model runs on CUDA
            try:
                use_cuda = any(p.is_cuda for p in model.parameters())  # type: ignore[attr-defined]
            except Exception:
                use_cuda = torch.cuda.is_available()
            enabled = bool(fp16 and use_cuda)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=enabled):
                data = model(return_loss=False, rescale=True, **data)

        if detect_timer is not None:
            detect_timer.toc()

        track_data_sample = data["data_samples"]
        nr_tracks_meta = track_data_sample.video_data_samples[0].metainfo.get("nr_tracks", 0)
        nr_tracks = nr_tracks_meta
        if not isinstance(nr_tracks, torch.Tensor):
            nr_tracks = int(nr_tracks)
        pred_tracks = track_data_sample.video_data_samples[-1].pred_track_instances
        tracking_ids = pred_tracks.instances_id
        track_mask = get_track_mask(pred_tracks)
        if isinstance(track_mask, torch.Tensor) and isinstance(tracking_ids, torch.Tensor):
            tracking_ids = tracking_ids[track_mask]
        if isinstance(tracking_ids, torch.Tensor):
            max_tracking_id = (
                torch.max(tracking_ids) if tracking_ids.numel() else tracking_ids.new_zeros(())
            )
        else:
            max_tracking_id = int(np.max(np.asarray(tracking_ids))) if len(tracking_ids) else 0

        jersey_results = context.get("jersey_results")
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
                    jersey_info=(
                        jersey_results[frame_index] if jersey_results is not None else None
                    ),
                )
            if not using_precalculated_detection and detection_dataframe is not None:
                detection_dataframe.add_frame_records(
                    frame_id=frame_id,
                    scores=video_data_sample.pred_instances.scores,
                    labels=video_data_sample.pred_instances.labels,
                    bboxes=video_data_sample.pred_instances.bboxes,
                )

        return {
            "data_samples": track_data_sample,
            "nr_tracks": nr_tracks,
            "max_tracking_id": max_tracking_id,
        }

    def input_keys(self):
        return {
            "inputs",
            "data_samples",
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
        return {"data_samples", "nr_tracks", "max_tracking_id"}

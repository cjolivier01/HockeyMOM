import contextlib
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import torch
from mmengine.structures import InstanceData

from hmlib.constants import WIDTH_NORMALIZATION_SIZE
from hmlib.log import get_logger
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hockeymom.core import HmByteTrackConfig, HmTrackerPredictionMode

from .base import Plugin

logger = get_logger(__name__)


class TrackerPlugin(Plugin):
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

    def __init__(
        self,
        enabled: bool = True,
        cpp_tracker: bool = True,
        tracker_class: Optional[str] = None,
        tracker_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(enabled=enabled)
        self._cpp_tracker = bool(cpp_tracker)
        if tracker_class is None:
            default_class = (
                "hockeymom.core.HmTracker" if cpp_tracker else "hockeymom.core.HmByteTrackerCuda"
            )
            self._tracker_class_path = default_class
        else:
            self._tracker_class_path = tracker_class
        self._tracker_kwargs = dict(tracker_kwargs or {})
        self._hm_tracker: Optional[Any] = None
        self._static_tracker_max_detections: Optional[int] = None
        self._static_tracker_max_tracks: Optional[int] = None
        self._static_tracker_overflow_warned = False
        self._iter_num: int = 0

    @property
    def tracker_class(self) -> str:
        """Fully-qualified tracker class path used for instantiation."""
        return self._tracker_class_path

    def _resolve_tracker_class(self):
        module_name, _, attr = self._tracker_class_path.rpartition(".")
        if not module_name:
            raise ValueError(
                f"tracker_class must be a module dotted path, got '{self._tracker_class_path}'"
            )
        module = import_module(module_name)
        tracker_cls = getattr(module, attr)
        return tracker_cls

    def _ensure_tracker(self, image_size: torch.Size):
        if self._hm_tracker is not None:
            return

        # make sure it's channels first so that we can pull the width
        image_width = image_size[-1]
        assert image_width > 4
        image_width_ratio: float = image_width / WIDTH_NORMALIZATION_SIZE

        if image_width_ratio >= 1.3:
            config = HmByteTrackConfig()
            config.init_track_thr = 0.7
            config.obj_score_thrs_low = 0.1
            config.obj_score_thrs_high = 0.3

            config.match_iou_thrs_high = 0.1
            config.match_iou_thrs_low = 0.5
            config.match_iou_thrs_tentative = 0.3
        else:
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
        tracker_cls = self._resolve_tracker_class()
        init_kwargs = dict(self._tracker_kwargs)
        try:
            self._hm_tracker = tracker_cls(config, **init_kwargs)
        except TypeError:
            if init_kwargs:
                raise
            self._hm_tracker = tracker_cls(config)
        self._update_static_tracker_limits()

    # post-detection pipeline deprecated; pruning handled by a dedicated trunk

    def __call__(self, *args, **kwargs):
        self._iter_num += 1
        # do_trace = self._iter_num == 4
        # if do_trace:
        #     pass
        # from cuda_stacktrace import CudaStackTracer

        # with CudaStackTracer(functions=["cudaStreamSynchronize"], enabled=do_trace):
        with contextlib.nullcontext():
            results = super().__call__(*args, **kwargs)
        # if do_trace:
        #     pass
        return results

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        frame_id0: int = int(context.get("frame_id", -1))

        # using_precalc_track: bool = bool(context.get("using_precalculated_tracking", False))
        # using_precalc_det: bool = bool(context.get("using_precalculated_detection", False))

        self._ensure_tracker(image_size=context["original_images"].shape)

        # Access TrackDataSample list
        track_samples = data.get("data_samples")
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        max_tracking_id: Optional[torch.Tensor] = None
        active_track_count: Optional[torch.Tensor] = None
        # all_frame_jersey_info: List[List[Any]] = []

        def _to_tensor_1d(x):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            if x.ndim == 0:
                x = x.unsqueeze(0)
            return x

        def _to_bboxes_2d(x):
            if x.ndim == 1:
                # If empty, reshape to (0, 4); if size==4, make (1,4)
                if x.numel() == 0:
                    return x.reshape(0, 4)
                x = x.unsqueeze(0)
            return x

        for frame_index in range(video_len):
            img_data_sample = track_data_sample[frame_index]

            # If precomputed tracking is used, skip tracker and only log data
            # if using_precalc_track:
            #     # Keep pred_track_instances unset; logging handled below if needed
            #     continue
            # Detections should have been attached by DetectorInferencePlugin
            det_instances = getattr(img_data_sample, "pred_instances", None)
            if det_instances is None:
                # No detections, skip tracking; leave pred_track_instances unset
                continue

            det_bboxes = unwrap_tensor(det_instances.bboxes)
            det_labels = unwrap_tensor(det_instances.labels)
            det_scores = unwrap_tensor(det_instances.scores)

            # Provide frame id for tracker aging
            frame_id = img_data_sample.metainfo.get("img_id")
            if isinstance(frame_id, torch.Tensor):
                if frame_id.device.type == "cuda":
                    frame_id = torch.tensor(
                        [frame_id0 + frame_index], dtype=torch.int64, device=frame_id.device
                    )
                else:
                    frame_id = frame_id.reshape(-1).to(dtype=torch.int64)
            elif frame_id is None:
                frame_id = torch.tensor([frame_id0 + frame_index], dtype=torch.int64)
            else:
                frame_id = torch.tensor([frame_id], dtype=torch.int64)
            frame_id_scalar = int(frame_id.reshape(-1)[:1][0].item())

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
                    if det_labels.numel() > 0:
                        fill_val = det_labels.reshape(-1)[:1]
                    else:
                        fill_val = det_labels.new_zeros((1,))
                    det_labels = fill_val.expand(N).clone()
            if len(det_scores) != N:
                if len(det_scores) == 1 and N > 1:
                    det_scores = det_scores.expand(N).clone()
                else:
                    det_scores = torch.ones((N,), dtype=torch.float32, device=det_bboxes.device)

            det_reid = getattr(det_instances, "reid_features", None)
            if det_reid is not None:
                det_reid = unwrap_tensor(det_reid)
                if not isinstance(det_reid, torch.Tensor):
                    det_reid = torch.as_tensor(det_reid)
                if det_reid.ndim == 1:
                    det_reid = det_reid.unsqueeze(0)
                if det_reid.shape[0] != N:
                    if det_reid.shape[0] == 1 and N > 1:
                        det_reid = det_reid.expand(N, -1).clone()
                    else:
                        det_reid = None

            ll1 = len(det_bboxes)
            assert len(det_labels) == ll1 and len(det_scores) == ll1
            # Ensure tracker receives torch tensors
            # (already tensors above)
            num_detections = getattr(det_instances, "num_detections", None)
            if num_detections is None:
                meta = getattr(det_instances, "metainfo", None)
                if isinstance(meta, dict):
                    num_detections = meta.get("num_detections")
            tracker_payload = self._prepare_tracker_inputs(
                frame_id=frame_id,
                det_bboxes=det_bboxes,
                det_labels=det_labels,
                det_scores=det_scores,
                det_reid=det_reid,
                num_detections=num_detections,
            )
            try:
                results = self._hm_tracker.track(tracker_payload)
            except Exception as e:
                logger.error("Tracker error at frame %d: %s", frame_id_scalar, str(e))
                raise
            results, frame_track_count = self._trim_tracker_outputs(results)
            ids = results.get("user_ids", results.get("ids"))
            num_tracks = frame_track_count
            if not isinstance(num_tracks, torch.Tensor):
                if isinstance(ids, torch.Tensor):
                    num_tracks = ids.new_tensor([int(num_tracks)])
                else:
                    num_tracks = torch.tensor([int(num_tracks)], dtype=torch.long)
            num_tracks = num_tracks.reshape(-1)[:1]
            track_mask = None
            if isinstance(ids, torch.Tensor):
                track_mask = torch.arange(ids.shape[0], device=ids.device) < num_tracks[0]

            pred_track_instances = InstanceData(
                instances_id=wrap_tensor(ids),
                bboxes=wrap_tensor(results["bboxes"]),
                scores=wrap_tensor(results["scores"]),
                labels=wrap_tensor(results["labels"]),
            )
            meta_info = {}
            if isinstance(num_tracks, torch.Tensor):
                meta_info["num_tracks"] = num_tracks
            if track_mask is not None:
                meta_info["track_mask"] = track_mask
            if "reid_features" in results:
                meta_info["reid_features"] = results["reid_features"]

            img_data_sample.pred_track_instances = pred_track_instances

            # For performance: record current active tracks
            if isinstance(num_tracks, torch.Tensor):
                if active_track_count is None:
                    active_track_count = num_tracks
                else:
                    active_track_count = torch.maximum(active_track_count, num_tracks)
            else:
                active_track_count = max(active_track_count or 0, int(num_tracks))
            meta_info["nr_tracks"] = active_track_count if active_track_count is not None else 0

            img_data_sample.set_metainfo(meta_info)

            max_id_tensor = results.get("max_id")
            if max_id_tensor is None and isinstance(ids, torch.Tensor):
                if track_mask is None:
                    max_id_tensor = ids.new_zeros((1,))
                else:
                    masked_ids = torch.where(track_mask, ids, ids.new_full(ids.shape, -1))
                    if masked_ids.numel() == 0:
                        max_id_tensor = torch.zeros((1,), dtype=ids.dtype, device=ids.device)
                    else:
                        max_id_tensor = torch.max(masked_ids)
                        max_id_tensor = torch.where(
                            num_tracks[0] > 0, max_id_tensor, max_id_tensor.new_zeros(())
                        )
                        max_id_tensor = max_id_tensor.reshape(1)
            if isinstance(max_id_tensor, torch.Tensor):
                if max_tracking_id is None:
                    max_tracking_id = max_id_tensor.reshape(-1)[:1]
                else:
                    max_tracking_id = torch.maximum(max_tracking_id, max_id_tensor.reshape(-1)[:1])

        result: Dict[str, Any] = {
            "data": data,
            "nr_tracks": active_track_count if active_track_count is not None else 0,
            "max_tracking_id": max_tracking_id if max_tracking_id is not None else 0,
        }
        return result

    def input_keys(self):
        return {
            "data",
            "frame_id",
            "original_images",
        }

    def output_keys(self):
        return {"data", "nr_tracks", "max_tracking_id"}

    @staticmethod
    def _coerce_frame_id_tensor(
        frame_id: Any, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if isinstance(frame_id, torch.Tensor):
            if frame_id.numel() == 0:
                out = torch.zeros((1,), dtype=torch.int64, device=frame_id.device)
            else:
                out = frame_id.reshape(-1).to(dtype=torch.int64)
        else:
            out = torch.tensor([frame_id], dtype=torch.int64)
        if device is not None:
            out = out.to(device=device, non_blocking=True)
        return out

    def _prepare_tracker_inputs(
        self,
        frame_id: int,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        det_reid: Optional[torch.Tensor] = None,
        num_detections: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        frame_id_t = self._coerce_frame_id_tensor(frame_id)
        if not self._using_static_tracker():
            payload = dict(
                frame_id=frame_id_t,
                bboxes=det_bboxes,
                labels=det_labels,
                scores=det_scores,
            )
            if det_reid is not None:
                payload["reid_features"] = det_reid.to(dtype=torch.float32)
            return payload
        return self._build_static_tracker_inputs(
            frame_id, det_bboxes, det_labels, det_scores, det_reid, num_detections
        )

    def _build_static_tracker_inputs(
        self,
        frame_id: int,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        det_reid: Optional[torch.Tensor] = None,
        num_detections: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        frame_id_t = self._coerce_frame_id_tensor(frame_id, device=bboxes.device)
        assert self._static_tracker_max_detections is not None
        max_det = self._static_tracker_max_detections
        bboxes = det_bboxes.to(dtype=torch.float32)
        labels = det_labels.to(dtype=torch.long)
        scores = det_scores.to(dtype=torch.float32)
        reid = None
        if det_reid is not None:
            reid = det_reid.to(dtype=torch.float32, device=bboxes.device)
        total = int(bboxes.shape[0])
        kept = min(total, max_det)
        if total > max_det:
            keep_idx = torch.topk(scores, k=max_det).indices
            keep_idx, _ = torch.sort(keep_idx)
            bboxes = bboxes.index_select(0, keep_idx)
            labels = labels.index_select(0, keep_idx)
            scores = scores.index_select(0, keep_idx)
            if reid is not None:
                reid = reid.index_select(0, keep_idx)
            if not self._static_tracker_overflow_warned:
                logger.warning(
                    "Tracker detections (%d) exceed max_detections (%d); discarding lowest scores.",
                    total,
                    max_det,
                )
                self._static_tracker_overflow_warned = True

        padded_bboxes = bboxes.new_zeros((max_det, 4))
        padded_labels = labels.new_zeros((max_det,))
        padded_scores = scores.new_zeros((max_det,))
        padded_reid = None
        if reid is not None:
            if reid.ndim == 1:
                reid = reid.unsqueeze(0)
            if reid.ndim != 2 or reid.shape[0] != kept:
                reid = None
            else:
                padded_reid = reid.new_zeros((max_det, reid.shape[1]))
        if kept:
            padded_bboxes[:kept].copy_(bboxes[:kept])
            padded_labels[:kept].copy_(labels[:kept])
            padded_scores[:kept].copy_(scores[:kept])
            if padded_reid is not None:
                padded_reid[:kept].copy_(reid[:kept])

        kept_t = torch.tensor([kept], dtype=torch.long).to(device=bboxes.device, non_blocking=True)

        payload = {
            "frame_id": frame_id_t,
            "bboxes": padded_bboxes,
            "labels": padded_labels,
            "scores": padded_scores,
            "num_detections": (
                torch.min(num_detections, kept_t) if num_detections is not None else kept_t
            ),
        }
        if padded_reid is not None:
            payload["reid_features"] = padded_reid
        return payload

    def _trim_tracker_outputs(
        self, results: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        num_tracks_tensor = results.get("num_tracks")
        if num_tracks_tensor is None:
            ids = results.get("user_ids", results.get("ids"))
            if isinstance(ids, torch.Tensor):
                count = ids.new_tensor([ids.shape[0]])
            else:
                count = torch.tensor([len(ids) if ids is not None else 0], dtype=torch.long)
            return results, count

        return results, num_tracks_tensor.reshape(-1)[:1]

    def _update_static_tracker_limits(self) -> None:
        self._static_tracker_max_detections = None
        self._static_tracker_max_tracks = None
        tracker = self._hm_tracker
        if tracker is None:
            return
        try:
            max_det = getattr(tracker, "max_detections")
            max_tracks = getattr(tracker, "max_tracks")
        except AttributeError:
            return
        try:
            self._static_tracker_max_detections = int(max_det)
            self._static_tracker_max_tracks = int(max_tracks)
            self._static_tracker_overflow_warned = False
        except (TypeError, ValueError):
            self._static_tracker_max_detections = None
            self._static_tracker_max_tracks = None

    def _using_static_tracker(self) -> bool:
        return self._static_tracker_max_detections is not None

import contextlib
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import torch
from mmengine.structures import InstanceData

from hmlib.constants import WIDTH_NORMALIZATION_SIZE
from hmlib.log import get_logger
from hmlib.utils.gpu import StreamCheckpoint, unwrap_tensor
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

    def __call__(self, *args, **kwds):
        # self._iter_num += 1
        # do_trace = self._iter_num == 4
        # if do_trace:
        #     pass
        # from cuda_stacktrace import CudaStackTracer

        # with CudaStackTracer(functions=["cudaStreamSynchronize"], enabled=do_trace):
        with contextlib.nullcontext():
            results = super().__call__(*args, **kwds)
        # if do_trace:
        #     pass
        return results

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        data: Dict[str, Any] = context["data"]
        # preserved_original_images = data.get("original_images")
        # dataset_results = data.get("dataset_results") or context.get("dataset_results")
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

        max_tracking_id = 0
        active_track_count = 0
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

            det_instances.bboxes = unwrap_tensor(det_instances.bboxes)
            det_instances.labels = unwrap_tensor(det_instances.labels)
            det_instances.scores = unwrap_tensor(det_instances.scores)

            det_bboxes = det_instances.bboxes
            det_labels = det_instances.labels
            det_scores = det_instances.scores
            # det_src_pose_idx = getattr(det_instances, "source_pose_index", None)

            # if len(det_bboxes) == 0:
            #     print("WARNING: No detections for frame", frame_index + frame_id0)
            # else:
            #     print(f"Frame {frame_index + frame_id0}: {len(det_bboxes)} detections")

            # Post-detection pruning is handled by a dedicated trunk upstream

            # Provide frame id for tracker aging
            frame_id = img_data_sample.metainfo.get("img_id")
            if isinstance(frame_id, torch.Tensor):
                frame_id = frame_id.reshape([1])[0]
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
                    det_labels = torch.full(
                        (N,),
                        int(det_labels[0].item()) if len(det_labels) else 0,
                        dtype=torch.long,
                        device=det_bboxes.device,
                    )
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
            tracker_payload = self._prepare_tracker_inputs(
                frame_id=frame_id,
                det_bboxes=det_bboxes,
                det_labels=det_labels,
                det_scores=det_scores,
                det_reid=det_reid,
            )
            results = self._hm_tracker.track(tracker_payload)
            results, frame_track_count = self._trim_tracker_outputs(results)
            ids = results.get("user_ids", results.get("ids"))
            ll2 = frame_track_count
            assert (
                len(results["bboxes"]) == ll2
                and len(results["scores"]) == ll2
                and len(results["labels"]) == ll2
            )

            pred_track_instances = InstanceData(
                instances_id=ids,
                bboxes=results["bboxes"],
                scores=results["scores"],
                labels=results["labels"],
            )
            if "reid_features" in results:
                try:
                    pred_track_instances.reid_features = results["reid_features"]
                except Exception:
                    pass
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
                        eq = (
                            torch.isclose(tb[j], db).all(dim=1)
                            if len(db)
                            else torch.zeros((0,), dtype=torch.bool)
                        )
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
                        iou = _bbox_iou(
                            tb.to(dtype=torch.float32), db.to(dtype=torch.float32), x1y1x2y2=True
                        )
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

            # Saving to dataframes is now handled by dedicated Save* plugins.

            # For performance: record current active tracks
            img_data_sample.set_metainfo({"nr_tracks": active_track_count})

            if len(pred_track_instances.instances_id):
                max_id = int(torch.max(pred_track_instances.instances_id))
                if max_id > max_tracking_id:
                    max_tracking_id = max_id

        result: Dict[str, Any] = {
            "data": data,
            "nr_tracks": active_track_count,
            "max_tracking_id": max_tracking_id,
        }
        # Record a lightweight stream token on the current CUDA stream so
        # downstream trunks (e.g., PlayTrackerPlugin) can establish proper
        # stream ordering without forcing a full synchronize here.
        try:
            original_images = context.get("original_images")
            device = None
            if isinstance(original_images, torch.Tensor):
                device = original_images.device
            else:
                device = getattr(original_images, "device", None)
            if isinstance(device, torch.device) and device.type == "cuda":
                token_tensor = torch.empty(0, device=device)
                result["tracker_stream_token"] = StreamCheckpoint(token_tensor)
        except Exception:
            # Best-effort only; fall back silently if anything goes wrong.
            pass
        return result

    def input_keys(self):
        return {
            "data",
            "frame_id",
            "original_images",
            # "using_precalculated_tracking",
            # "using_precalculated_detection",
            # no longer depends on model's post-detection pipeline
        }

    def output_keys(self):
        return {"data", "nr_tracks", "max_tracking_id", "tracker_stream_token"}

    def _prepare_tracker_inputs(
        self,
        frame_id: int,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        det_reid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._using_static_tracker():
            payload = dict(
                frame_id=torch.tensor([frame_id], dtype=torch.int64),
                bboxes=det_bboxes,
                labels=det_labels,
                scores=det_scores,
            )
            if det_reid is not None:
                payload["reid_features"] = det_reid.to(dtype=torch.float32)
            return payload
        return self._build_static_tracker_inputs(frame_id, det_bboxes, det_labels, det_scores, det_reid)

    def _build_static_tracker_inputs(
        self,
        frame_id: int,
        det_bboxes: torch.Tensor,
        det_labels: torch.Tensor,
        det_scores: torch.Tensor,
        det_reid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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

        payload = {
            "frame_id": torch.tensor([frame_id], dtype=torch.int64).to(
                device=bboxes.device, non_blocking=True
            ),
            "bboxes": padded_bboxes,
            "labels": padded_labels,
            "scores": padded_scores,
            "num_detections": torch.tensor([kept], dtype=torch.long).to(
                device=bboxes.device, non_blocking=True
            ),
        }
        if padded_reid is not None:
            payload["reid_features"] = padded_reid
        return payload

    def _trim_tracker_outputs(
        self, results: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        num_tracks_tensor = results.get("num_tracks")
        if num_tracks_tensor is None:
            ids = results.get("user_ids", results.get("ids"))
            count = len(ids) if ids is not None else 0
            return results, count

        try:
            num_tracks = int(num_tracks_tensor.reshape(-1)[0].item())
        except Exception:
            num_tracks = 0
        trimmed = dict(results)
        for key in ("user_ids", "ids", "bboxes", "labels", "scores", "reid_features"):
            tensor = trimmed.get(key)
            if tensor is not None:
                trimmed[key] = tensor[:num_tracks]
        return trimmed, num_tracks

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

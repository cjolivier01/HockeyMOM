from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from mmengine.structures import InstanceData


@dataclass
class TrtNmsConfig:
    num_classes: int
    max_num_boxes: int  # static input/output size per image
    top_k: int          # plugin topK (per class * image)
    keep_top_k: int     # plugin keepTopK (per image, before our own max_per_img)
    score_threshold: float
    iou_threshold: float
    max_per_img: int
    share_location: bool = True
    background_label_id: int = -1
    is_normalized: bool = False
    clip_boxes: bool = False
    score_bits: int = 16
    caffe_semantics: bool = True


class TrtBatchedNMS:
    """Thin TensorRT batched NMS wrapper.

    Builds a tiny TensorRT engine that only contains the BatchedNMSDynamic_TRT
    plugin and exposes a callable that takes an InstanceData and returns a
    new InstanceData with NMS applied. The engine is static-shape
    (no dynamic dims) to avoid runtime shape updates and stream syncs.
    """

    def __init__(self, cfg: TrtNmsConfig):
        self.cfg = cfg
        self._engine = None
        self._context = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._trt = None
        self._np = None

    @staticmethod
    def from_bbox_head(
        bbox_head: torch.nn.Module,
        max_num_boxes: int,
    ) -> "TrtBatchedNMS":
        """Construct a config from an mmdet/mmyolo bbox head."""
        num_classes = int(getattr(bbox_head, "num_classes", 1))
        test_cfg = getattr(bbox_head, "test_cfg", None)

        score_thr = 0.0
        iou_thr = 0.5
        max_per_img = 100
        nms_pre = max_num_boxes

        try:
            if test_cfg is not None:
                # mmengine ConfigDict behaves like a dict
                score_thr = float(test_cfg.get("score_thr", score_thr))
                max_per_img = int(test_cfg.get("max_per_img", max_per_img))
                nms_cfg = test_cfg.get("nms", {}) or {}
                if "iou_threshold" in nms_cfg:
                    iou_thr = float(nms_cfg["iou_threshold"])
                elif "iou_thr" in nms_cfg:
                    iou_thr = float(nms_cfg["iou_thr"])
                nms_pre = int(test_cfg.get("nms_pre", nms_pre))
        except Exception:
            pass

        # Prefer static max_detections when available (YOLOX static path).
        static_max = 0
        try:
            if bool(getattr(bbox_head, "static_det_enabled", False)):
                static_max = int(getattr(bbox_head, "static_det_max", 0) or 0)
        except Exception:
            static_max = 0

        max_top_k = 512 * 8
        max_num_boxes = int(max_num_boxes)
        nms_pre = int(nms_pre)
        # Bound max_num_boxes by pre-NMS clamp and plugin limits.
        if static_max > 0:
            max_num_boxes = static_max
        max_num_boxes = max(1, min(max_num_boxes, nms_pre, max_top_k))

        # Use full max_num_boxes inside the plugin, and apply max_per_img
        # explicitly after TensorRT NMS for closer parity with mmcv.batched_nms.
        top_k = max_num_boxes
        keep_top_k = max_num_boxes

        cfg = TrtNmsConfig(
            num_classes=num_classes,
            max_num_boxes=max_num_boxes,
            top_k=top_k,
            keep_top_k=keep_top_k,
            score_threshold=float(score_thr),
            iou_threshold=float(iou_thr),
            max_per_img=int(max_per_img),
        )
        return TrtBatchedNMS(cfg)

    def _lazy_imports(self):
        if self._trt is not None:
            return
        import importlib

        try:
            trt = importlib.import_module("tensorrt")
        except Exception as ex:
            raise RuntimeError("TensorRT Python package is required for TrtBatchedNMS") from ex
        try:
            import numpy as np  # type: ignore
        except Exception as ex:
            raise RuntimeError("NumPy is required for TrtBatchedNMS") from ex

        self._trt = trt
        self._np = np

    def _build_engine(self, device: torch.device) -> None:
        self._lazy_imports()
        trt = self._trt
        np = self._np

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")

        builder = trt.Builder(logger)
        flags = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        config = builder.create_builder_config()
        # NMS network is tiny; 32MB is plenty.
        try:
            # TensorRT >= 8.0 style API.
            workspace_type = trt.MemoryPoolType.WORKSPACE
            config.set_memory_pool_limit(workspace_type, 1 << 25)
        except Exception:
            # Fallback for older TensorRT versions, if ever encountered.
            if hasattr(config, "max_workspace_size"):
                config.max_workspace_size = 1 << 25

        B = 1
        N = int(self.cfg.max_num_boxes)
        C = int(self.cfg.num_classes)

        boxes = network.add_input(
            name="boxes",
            dtype=trt.DataType.FLOAT,
            shape=(B, N, 1, 4),
        )
        scores = network.add_input(
            name="scores",
            dtype=trt.DataType.FLOAT,
            shape=(B, N, C),
        )

        registry = trt.get_plugin_registry()
        creator = registry.get_plugin_creator("BatchedNMSDynamic_TRT", "1", "")
        if creator is None:
            raise RuntimeError("BatchedNMSDynamic_TRT plugin not found in TensorRT registry")

        fields = []

        def add_field(name: str, value, ftype):
            arr = np.array([value], dtype=np.int32 if "INT" in ftype.name else np.float32)
            fields.append(trt.PluginField(name, arr, ftype))

        add_field("shareLocation", int(bool(self.cfg.share_location)), trt.PluginFieldType.INT32)
        add_field("backgroundLabelId", int(self.cfg.background_label_id), trt.PluginFieldType.INT32)
        add_field("numClasses", int(self.cfg.num_classes), trt.PluginFieldType.INT32)
        add_field("topK", int(self.cfg.top_k), trt.PluginFieldType.INT32)
        add_field("keepTopK", int(self.cfg.keep_top_k), trt.PluginFieldType.INT32)
        add_field("scoreThreshold", float(self.cfg.score_threshold), trt.PluginFieldType.FLOAT32)
        add_field("iouThreshold", float(self.cfg.iou_threshold), trt.PluginFieldType.FLOAT32)
        add_field("isNormalized", int(bool(self.cfg.is_normalized)), trt.PluginFieldType.INT32)
        add_field("clipBoxes", int(bool(self.cfg.clip_boxes)), trt.PluginFieldType.INT32)
        add_field("scoreBits", int(self.cfg.score_bits), trt.PluginFieldType.INT32)
        add_field("caffeSemantics", int(bool(self.cfg.caffe_semantics)), trt.PluginFieldType.INT32)

        plugin = creator.create_plugin("batched_nms", trt.PluginFieldCollection(fields))
        layer = network.add_plugin_v2([boxes, scores], plugin)

        num_det = layer.get_output(0)
        nmsed_boxes = layer.get_output(1)
        nmsed_scores = layer.get_output(2)
        nmsed_classes = layer.get_output(3)

        num_det.name = "num_detections"
        nmsed_boxes.name = "nmsed_boxes"
        nmsed_scores.name = "nmsed_scores"
        nmsed_classes.name = "nmsed_classes"

        network.mark_output(num_det)
        network.mark_output(nmsed_boxes)
        network.mark_output(nmsed_scores)
        network.mark_output(nmsed_classes)

        # Static shapes; single profile with fixed min/opt/max.
        profile = builder.create_optimization_profile()
        profile.set_shape("boxes", (B, N, 1, 4), (B, N, 1, 4), (B, N, 1, 4))
        profile.set_shape("scores", (B, N, C), (B, N, C), (B, N, C))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT NMS network")

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT NMS engine")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT NMS execution context")

        self._engine = engine
        self._context = context
        self._stream = torch.cuda.Stream(device=device)

    def _ensure_engine(self, device: torch.device) -> None:
        if self._engine is None or self._context is None or self._stream is None:
            self._build_engine(device)
        elif self._stream.device != device:
            self._stream = torch.cuda.Stream(device=device)

    def _infer(self, boxes: torch.Tensor, scores: torch.Tensor):
        assert boxes.is_cuda and scores.is_cuda
        assert boxes.dtype == torch.float32
        assert scores.dtype == torch.float32

        device = boxes.device
        self._ensure_engine(device)

        engine = self._engine
        context = self._context
        assert engine is not None and context is not None

        B = 1
        N = self.cfg.max_num_boxes
        C = self.cfg.num_classes
        K = self.cfg.keep_top_k

        num_boxes = int(boxes.shape[0])
        if num_boxes > N:
            # Truncate to the maximum supported by the engine.
            boxes = boxes[:N]

        # Pad inputs up to static engine shapes.
        boxes_pad = torch.zeros(
            (B, N, 1, 4), device=device, dtype=torch.float32
        )
        scores_pad = torch.zeros(
            (B, N, C), device=device, dtype=torch.float32
        )
        boxes_pad[0, : boxes.shape[0], 0, :] = boxes
        scores_pad[0, : scores.shape[0], :] = scores

        num_det = torch.empty((B, 1), device=device, dtype=torch.int32)
        out_boxes = torch.empty((B, K, 4), device=device, dtype=torch.float32)
        out_scores = torch.empty((B, K), device=device, dtype=torch.float32)
        out_classes = torch.empty((B, K), device=device, dtype=torch.float32)

        # Bind buffers by tensor name.
        context.set_tensor_address("boxes", boxes_pad.data_ptr())
        context.set_tensor_address("scores", scores_pad.data_ptr())
        context.set_tensor_address("num_detections", num_det.data_ptr())
        context.set_tensor_address("nmsed_boxes", out_boxes.data_ptr())
        context.set_tensor_address("nmsed_scores", out_scores.data_ptr())
        context.set_tensor_address("nmsed_classes", out_classes.data_ptr())

        assert self._stream is not None
        trt_stream = self._stream

        ok = context.execute_async_v3(trt_stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT NMS execution failed")

        # Ensure results are visible on the caller's current stream without
        # forcing a device-wide synchronize.
        torch.cuda.current_stream(device).wait_stream(trt_stream)

        return num_det, out_boxes, out_scores, out_classes

    def __call__(self, instances: InstanceData) -> InstanceData:
        bboxes = getattr(instances, "bboxes", None)
        scores = getattr(instances, "scores", None)
        labels = getattr(instances, "labels", None)
        if bboxes is None or scores is None or labels is None:
            return instances
        if not torch.is_tensor(bboxes) or not torch.is_tensor(scores) or not torch.is_tensor(labels):
            return instances
        if bboxes.numel() == 0:
            return instances

        device = bboxes.device
        num_boxes = bboxes.shape[0]
        num_classes = self.cfg.num_classes

        # Build per-class score matrix from per-box scores + labels.
        scores_full = torch.zeros(
            (num_boxes, num_classes),
            device=device,
            dtype=torch.float32,
        )
        idx = torch.arange(num_boxes, device=device, dtype=torch.long)
        # Avoid in-place ops on tensors that may have been created under
        # torch.inference_mode() by using out-of-place clamp.
        labels_clamped = labels.to(torch.long).clamp(0, num_classes - 1)
        scores_full[idx, labels_clamped] = scores.to(torch.float32)

        num_det, out_boxes, out_scores, out_classes = self._infer(
            bboxes.to(torch.float32), scores_full
        )

        # num_det has shape [B, 1]; clamp to valid range.
        try:
            raw_valid = int(num_det[0, 0])
        except Exception:
            raw_valid = out_scores.shape[1]
        raw_valid = max(0, min(raw_valid, out_scores.shape[1]))

        # Slice to the number of valid detections reported by the plugin.
        boxes = out_boxes[0][:raw_valid]
        scores_vec = out_scores[0][:raw_valid]
        labels_vec = out_classes[0][:raw_valid].to(torch.long)

        # Apply the same max_per_img cap as the original head.
        max_per = int(self.cfg.max_per_img or 0)
        if max_per > 0 and boxes.shape[0] > max_per:
            top_scores, top_idx = scores_vec.topk(max_per)
            boxes = boxes[top_idx]
            labels_vec = labels_vec[top_idx]
            scores_vec = top_scores

        num_valid = boxes.shape[0]

        # Optionally pad back to a static shape for downstream consumers,
        # mirroring the head's static_detections behavior.
        if self.cfg.max_num_boxes > 0:
            pad_to = int(self.cfg.max_num_boxes)
            padded_bboxes = boxes.new_zeros((pad_to, 4))
            padded_scores = scores_vec.new_full((pad_to,), float("-inf"))
            padded_labels = labels_vec.new_full((pad_to,), -1, dtype=labels_vec.dtype)
            if num_valid > 0:
                padded_bboxes[:num_valid] = boxes
                padded_scores[:num_valid] = scores_vec
                padded_labels[:num_valid] = labels_vec
            boxes_out = padded_bboxes
            scores_out = padded_scores
            labels_out = padded_labels
        else:
            boxes_out = boxes
            scores_out = scores_vec
            labels_out = labels_vec

        new_inst = InstanceData()
        new_inst.bboxes = boxes_out
        new_inst.scores = scores_out
        new_inst.labels = labels_out

        # Propagate static padding metadata so _strip_static_padding can
        # trim back to num_valid on the GPU.
        try:
            new_inst.set_metainfo(
                dict(
                    num_valid_after_nms=torch.as_tensor(num_valid, device=device, dtype=torch.int32),
                    max_detections=int(self.cfg.max_num_boxes),
                    num_valid_before_nms=int(num_boxes),
                )
            )
        except Exception:
            pass
        return new_inst


__all__ = ["TrtNmsConfig", "TrtBatchedNMS"]

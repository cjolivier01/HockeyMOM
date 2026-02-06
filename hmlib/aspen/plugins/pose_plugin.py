from typing import Any, Dict, List, Optional, Set

import torch

from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.cuda_graph import CudaGraphCallable
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_last

from .base import Plugin


def _to_cuda_tensor(value: Any, *, device: torch.device) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        t = value
    else:
        t = None
        try:
            import numpy as np  # type: ignore

            if isinstance(value, np.ndarray):
                t = torch.from_numpy(value)
        except Exception:
            t = None
        if t is None:
            if (
                isinstance(value, (list, tuple))
                and value
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            ):
                t = torch.as_tensor(value)
            else:
                return None

    if t.device != device:
        t = t.to(device=device, non_blocking=True)
    if t.dtype == torch.float64:
        t = t.to(dtype=torch.float32)
    return t.contiguous()


class PosePlugin(Plugin):
    """
    Runs multi-pose inference using an MMPose inferencer.

    Expects in context:
      - original_images: stitched panorama frames (Tensor or StreamTensorBase)
      - data_samples: TrackDataSample (or list) with tracking results
      - inputs: optional detection-model input tensor (used for scaling/visualization alignment)
      - pose_inferencer: initialized MMPoseInferencer
      - plot_pose: bool (optional)

    Produces in context:
      - pose_results: list of per-frame pose inference outputs
      - original_images: optionally updated when visualization is enabled
    """

    def __init__(
        self,
        enabled: bool = True,
        plot_pose: bool = False,
        cuda_graph: bool = False,
        cuda_graph_max_instances: int = 32,
    ):
        # Need to import in order to register
        super().__init__(enabled=enabled)
        self._default_plot_pose: bool = bool(plot_pose)
        self._cuda_graph_enabled: bool = bool(cuda_graph)
        self._cuda_graph_max_instances: int = int(max(1, cuda_graph_max_instances))
        self._pose_cg: Optional[CudaGraphCallable] = None
        self._pose_cg_device: Optional[torch.device] = None

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        pose_inferencer = context.get("pose_inferencer")
        if pose_inferencer is None:
            return {}

        original_images_any = context.get("original_images")
        if original_images_any is None:
            return {}
        original_images = unwrap_tensor(original_images_any)
        drawn_original_images: list[torch.Tensor] = []

        track_data_sample = self._get_track_data_sample(context.get("data_samples"))
        frame_count = None
        if isinstance(original_images, torch.Tensor):
            frame_count = original_images.shape[0]
        elif hasattr(original_images, "__len__"):
            frame_count = len(original_images)

        for vs in track_data_sample.video_data_samples:
            pred_track_instances = getattr(vs, "pred_track_instances", None)
            if pred_track_instances is not None:
                pred_track_instances.bboxes = unwrap_tensor(pred_track_instances.bboxes)
                pred_track_instances.instances_id = unwrap_tensor(pred_track_instances.instances_id)
                pred_track_instances.labels = unwrap_tensor(pred_track_instances.labels)
                pred_track_instances.scores = unwrap_tensor(pred_track_instances.scores)
            # If track instances are absent, fall back to MMPose's internal detector path.

        per_frame_bboxes, frame_metas = self._collect_bboxes(track_data_sample, frame_count)
        if per_frame_bboxes is not None:
            try:
                if all((not torch.is_tensor(b)) or b.numel() == 0 for b in per_frame_bboxes):
                    per_frame_bboxes = None
            except Exception:
                pass

        det_imgs_tensor = context.get("inputs")
        if isinstance(det_imgs_tensor, StreamTensorBase):
            det_imgs_tensor = det_imgs_tensor.wait()
        det_inputs_tensor = self._normalize_det_images(det_imgs_tensor, frame_count)

        use_det_imgs = False
        scale_factors: List[torch.Tensor] = []
        inv_scale_factors: List[torch.Tensor] = []
        if (
            det_inputs_tensor is not None
            and frame_metas is not None
            and frame_count is not None
            and len(frame_metas) == frame_count
            and per_frame_bboxes is not None
        ):
            use_det_imgs = True
            for idx in range(frame_count):
                meta = frame_metas[idx]
                scale_tensor = self._extract_scale_tensor(meta, per_frame_bboxes[0].device)
                if (
                    scale_tensor is None
                    or not torch.isfinite(scale_tensor).all()
                    or (scale_tensor <= 0).any()
                ):
                    use_det_imgs = False
                    break
                scale_gpu = scale_tensor.detach()
                scale_factors.append(scale_gpu)
                inv_scale_factors.append(torch.reciprocal(scale_gpu))
            if not use_det_imgs:
                scale_factors.clear()
                inv_scale_factors.clear()
                det_inputs_tensor = None

        if use_det_imgs and per_frame_bboxes is not None:
            per_frame_bboxes = self._scale_bboxes_for_detection(per_frame_bboxes, scale_factors)

        if not use_det_imgs or det_inputs_tensor is None:
            inputs_tensor = original_images
            scale_factors = []
            inv_scale_factors = []
        else:
            inputs_tensor = det_inputs_tensor

        inputs: List[torch.Tensor] = []
        for img in inputs_tensor:
            inputs.append(img)
        inputs = make_channels_last(torch.stack(inputs))

        inputs = inputs.contiguous()

        all_pose_results = []
        show = bool(
            context.get("plot_pose")
            or context.get("shared", {}).get("plot_pose", self._default_plot_pose)
        )
        # If we have per-frame bboxes and a Pose2D inferencer available, run a
        # custom forward that bypasses MMPose's internal detector.
        pose_impl = getattr(pose_inferencer, "inferencer", None)
        can_bypass = (
            per_frame_bboxes is not None
            and len(per_frame_bboxes) == len(inputs)
            and pose_impl is not None
            and getattr(getattr(pose_impl, "cfg", object()), "data_mode", None) == "topdown"
        )

        if can_bypass:
            # Use the underlying Pose2DInferencer building blocks directly.
            # Respect key filter args where sensible.
            filter_args = getattr(pose_inferencer, "filter_args", {}) or {}
            pose_based_nms = bool(filter_args.get("pose_based_nms", False))
            bbox_thr = float(filter_args.get("bbox_thr", -1))
            # For visualization, read kpt_thr if present
            kpt_thr = filter_args.get("kpt_thr", None)

            model = pose_impl.model
            pipeline = pose_impl.pipeline
            collate_fn = pose_impl.collate_fn
            dataset_meta = getattr(model, "dataset_meta", {}) or {}

            def _build_empty_predictions() -> List[Any]:
                try:
                    from mmengine.structures import InstanceData
                    from mmpose.structures import PoseDataSample

                    empty_ds = PoseDataSample()
                    empty_ds.pred_instances = InstanceData()
                    return [empty_ds]
                except Exception:
                    return []

            use_pose_cudagraph = (
                self._cuda_graph_enabled
                and torch.cuda.is_available()
                and isinstance(inputs, torch.Tensor)
                and inputs.is_cuda
            )

            from mmengine.structures import InstanceData
            from mmpose.structures import PoseDataSample

            if use_pose_cudagraph:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if model_device.type != "cuda":
                    raise RuntimeError("Pose CUDA graph requires the pose model to be on CUDA")

                max_k = int(self._cuda_graph_max_instances)
                dp = getattr(model, "data_preprocessor", None)
                mean_t = getattr(dp, "mean", None) if dp is not None else None
                std_t = getattr(dp, "std", None) if dp is not None else None
                swap_rgb = bool(
                    (getattr(dp, "bgr_to_rgb", False) if dp is not None else False)
                    or (getattr(dp, "rgb_to_bgr", False) if dp is not None else False)
                )
                if isinstance(mean_t, torch.Tensor):
                    if mean_t.ndim == 1:
                        mean_t = mean_t.view(1, -1, 1, 1)
                    elif mean_t.ndim == 3:
                        mean_t = mean_t.unsqueeze(0)
                if isinstance(std_t, torch.Tensor):
                    if std_t.ndim == 1:
                        std_t = std_t.view(1, -1, 1, 1)
                    elif std_t.ndim == 3:
                        std_t = std_t.unsqueeze(0)

                mean_graph = (
                    mean_t.detach().to(device=model_device, dtype=torch.float32)
                    if isinstance(mean_t, torch.Tensor)
                    else None
                )
                std_graph = (
                    std_t.detach().to(device=model_device, dtype=torch.float32)
                    if isinstance(std_t, torch.Tensor)
                    else None
                )

                head = getattr(model, "head", None)
                neck = getattr(model, "neck", None)
                decoder = getattr(head, "decoder", None) if head is not None else None
                if head is None or decoder is None or not hasattr(decoder, "simcc_split_ratio"):
                    raise RuntimeError("Pose CUDA graph requires head.decoder.simcc_split_ratio")
                split_ratio = getattr(decoder, "simcc_split_ratio", 1.0)
                if isinstance(split_ratio, (list, tuple)) and len(split_ratio) >= 2:
                    split_ratio_t = torch.tensor(
                        [float(split_ratio[0]), float(split_ratio[1])],
                        device=model_device,
                        dtype=torch.float32,
                    ).view(1, 1, 2)
                else:
                    sr = float(split_ratio)
                    split_ratio_t = torch.tensor(
                        [sr, sr], device=model_device, dtype=torch.float32
                    ).view(1, 1, 2)

                def _pose_fn(
                    x: torch.Tensor,
                    input_center: torch.Tensor,
                    input_scale: torch.Tensor,
                    input_size: torch.Tensor,
                ):
                    if x.dtype != torch.float32:
                        x = x.to(dtype=torch.float32)
                    if swap_rgb and x.ndim == 4 and x.size(1) == 3:
                        x = x[:, [2, 1, 0], :, :]
                    if isinstance(mean_graph, torch.Tensor) and isinstance(std_graph, torch.Tensor):
                        x = (x - mean_graph) / std_graph
                    feats = model.backbone(x)
                    if neck is not None:
                        feats = neck(feats)
                    if not isinstance(feats, (tuple, list)):
                        feats = (feats,)
                    pred_x, pred_y = head.forward(feats)  # type: ignore[operator]
                    if pred_x.ndim != 3 or pred_y.ndim != 3:
                        raise RuntimeError("Pose CUDA graph expected (N,K,W) SimCC tensors")
                    n, k, _ = pred_x.shape
                    flat_x = pred_x.reshape(n * k, -1)
                    flat_y = pred_y.reshape(n * k, -1)
                    max_val_x, x_idx = flat_x.max(dim=1)
                    max_val_y, y_idx = flat_y.max(dim=1)
                    locs = torch.stack((x_idx, y_idx), dim=-1).to(dtype=torch.float32)
                    better_mask = max_val_x > max_val_y
                    vals = torch.where(better_mask, max_val_y, max_val_x)
                    invalid = vals <= 0
                    locs = torch.where(invalid.unsqueeze(-1), locs.new_full(locs.shape, -1.0), locs)
                    locs = locs.reshape(n, k, 2)
                    scores = vals.reshape(n, k)
                    keypoints = locs / split_ratio_t

                    center = input_center.to(dtype=torch.float32)
                    scale = input_scale.to(dtype=torch.float32)
                    size = input_size.to(dtype=torch.float32)
                    center = center.view(center.shape[0], 1, 2)
                    scale = scale.view(scale.shape[0], 1, 2)
                    size = size.view(size.shape[0], 1, 2)
                    offset = scale * 0.5
                    transformed = keypoints[..., :2] / size * scale + center - offset
                    keypoints = keypoints.clone()
                    keypoints[..., :2] = transformed
                    return keypoints, scores

                for frame_index, (img, bxs_full) in enumerate(zip(inputs, per_frame_bboxes)):
                    if isinstance(img, torch.Tensor) and img.device != model_device:
                        img = img.to(device=model_device, non_blocking=True)
                    bxs = bxs_full
                    if bbox_thr > 0 and bxs.numel() > 0:
                        bxs = bxs[bxs[:, 4] >= bbox_thr]
                    if bxs.numel() == 0:
                        all_pose_results.append({"predictions": _build_empty_predictions()})
                        continue

                    scores = bxs[:, 4].to(dtype=torch.float32)
                    k = min(int(bxs.shape[0]), max_k)
                    if k > 0:
                        order = torch.topk(scores, k=k).indices
                        bxs = bxs.index_select(0, order)
                    valid_count = int(bxs.shape[0])
                    if valid_count < max_k:
                        pad = bxs.new_zeros((max_k - valid_count, bxs.shape[1]))
                        pad[:, 2] = 1.0
                        pad[:, 3] = 1.0
                        bxs = torch.cat([bxs, pad], dim=0)

                    batched_data_infos: List[Any] = []
                    for j in range(bxs.shape[0]):
                        bbox = bxs[j]
                        inst: Dict[str, Any] = {
                            "img": img,
                            "img_path": None,
                            "hm_frame_index": frame_index,
                        }
                        inst.update(dataset_meta)
                        inst["bbox"] = bbox[None, :4].to(dtype=torch.float32)
                        inst["bbox_score"] = bbox[4:5].to(dtype=torch.float32)
                        batched_data_infos.append(pipeline(inst))

                    proc_inputs = collate_fn(batched_data_infos)
                    inputs_t = proc_inputs.get("inputs")
                    if not isinstance(inputs_t, torch.Tensor):
                        if isinstance(inputs_t, (list, tuple)):
                            if len(inputs_t) == 0:
                                raise RuntimeError(
                                    "Pose CUDA graph got empty proc_inputs['inputs']"
                                )
                            tensors: List[torch.Tensor] = []
                            for item in inputs_t:
                                if not isinstance(item, torch.Tensor):
                                    raise RuntimeError(
                                        "Pose CUDA graph requires tensor items in proc_inputs['inputs']"
                                    )
                                t = item
                                if t.ndim == 4 and t.size(0) == 1:
                                    t = t.squeeze(0)
                                tensors.append(t)
                            if len(tensors) == 1:
                                inputs_t = tensors[0]
                                if inputs_t.ndim == 3:
                                    inputs_t = inputs_t.unsqueeze(0)
                            else:
                                inputs_t = torch.stack(tensors, dim=0)
                            proc_inputs["inputs"] = inputs_t
                        else:
                            raise RuntimeError(
                                "Pose CUDA graph requires tensor proc_inputs['inputs']"
                            )
                    if inputs_t.device != model_device:
                        inputs_t = inputs_t.to(device=model_device, non_blocking=True)
                        proc_inputs["inputs"] = inputs_t
                    if not inputs_t.is_cuda:
                        raise RuntimeError("Pose CUDA graph requires CUDA proc_inputs['inputs']")

                    data_samples = proc_inputs.get("data_samples")
                    if not isinstance(data_samples, list) or len(data_samples) != int(
                        inputs_t.shape[0]
                    ):
                        raise RuntimeError(
                            "Pose CUDA graph requires proc_inputs['data_samples'] list"
                        )
                    centers_l: List[torch.Tensor] = []
                    scales_l: List[torch.Tensor] = []
                    sizes_l: List[torch.Tensor] = []
                    for ds in data_samples:
                        meta = getattr(ds, "metainfo", {}) or {}
                        if not isinstance(meta, dict):
                            try:
                                meta = dict(meta)
                            except Exception:
                                meta = {}
                        for k in ("input_center", "input_scale", "input_size"):
                            if k not in meta:
                                raise RuntimeError(f"Pose CUDA graph requires metainfo[{k}]")
                        c = _to_cuda_tensor(meta["input_center"], device=model_device)
                        s = _to_cuda_tensor(meta["input_scale"], device=model_device)
                        z = _to_cuda_tensor(meta["input_size"], device=model_device)
                        if c is None or s is None or z is None:
                            raise RuntimeError(
                                "Pose CUDA graph requires tensorizable input_center/scale/size"
                            )
                        centers_l.append(c.reshape(-1)[:2])
                        scales_l.append(s.reshape(-1)[:2])
                        sizes_l.append(z.reshape(-1)[:2])
                    centers = torch.stack(centers_l, dim=0).contiguous()
                    scales = torch.stack(scales_l, dim=0).contiguous()
                    sizes = torch.stack(sizes_l, dim=0).contiguous()

                    if self._pose_cg is None or self._pose_cg_device != inputs_t.device:
                        self._pose_cg = CudaGraphCallable(
                            _pose_fn,
                            (inputs_t, centers, scales, sizes),
                            name="pose",
                        )
                        self._pose_cg_device = inputs_t.device

                    keypoints, kp_scores = self._pose_cg(inputs_t, centers, scales, sizes)
                    keypoints = keypoints[:valid_count]
                    kp_scores = kp_scores[:valid_count]

                    gt_boxes: List[torch.Tensor] = []
                    gt_scores: List[torch.Tensor] = []
                    for ds in data_samples[:valid_count]:
                        gt = getattr(ds, "gt_instances", None)
                        if (
                            gt is None
                            or not hasattr(gt, "bboxes")
                            or not hasattr(gt, "bbox_scores")
                        ):
                            raise RuntimeError(
                                "Pose CUDA graph requires gt_instances.bboxes/bbox_scores"
                            )
                        b = gt.bboxes
                        sc = gt.bbox_scores
                        if not isinstance(b, torch.Tensor):
                            b = torch.as_tensor(b)
                        if not isinstance(sc, torch.Tensor):
                            sc = torch.as_tensor(sc)
                        if b.ndim == 2 and b.shape[0] == 1:
                            b = b.squeeze(0)
                        if sc.ndim == 1 and sc.shape[0] == 1:
                            sc = sc.squeeze(0)
                        gt_boxes.append(b.to(device=model_device, dtype=torch.float32))
                        gt_scores.append(sc.to(device=model_device, dtype=torch.float32))
                    bboxes = torch.stack(gt_boxes, dim=0)
                    bbox_scores = torch.stack(gt_scores, dim=0)

                    merged = PoseDataSample(metainfo=getattr(data_samples[0], "metainfo", {}))
                    merged.pred_instances = InstanceData(
                        keypoints=keypoints,
                        keypoint_scores=kp_scores,
                        bboxes=bboxes,
                        bbox_scores=bbox_scores,
                    )
                    all_pose_results.append({"predictions": [merged]})
            else:
                batched_data_infos: List[Any] = []
                frame_batch_indices: List[int] = []
                empty_frame_indices: Set[int] = set()

                for i, (img, bxs) in enumerate(zip(inputs, per_frame_bboxes)):
                    if bxs.numel() == 0:
                        empty_frame_indices.add(i)
                        continue

                    for j in range(bxs.shape[0]):
                        bbox = bxs[j]
                        inst: Dict[str, Any] = {
                            "img": img,
                            "img_path": None,
                            "hm_frame_index": i,
                        }
                        inst.update(dataset_meta)
                        inst["bbox"] = bbox[None, :4].to(dtype=torch.float32)
                        inst["bbox_score"] = bbox[4:5].to(dtype=torch.float32)
                        batched_data_infos.append(pipeline(inst))
                        frame_batch_indices.append(i)

                frame_predictions: List[Optional[List[Any]]] = [None] * len(inputs)
                preds: Optional[List[Any]] = None
                if batched_data_infos:
                    proc_inputs = collate_fn(batched_data_infos)
                    tried_runner = None
                    pose_runner = getattr(context.get("pose_inferencer"), "_hm_pose_runner", None)
                    pose_forward = getattr(pose_runner, "forward", None)
                    if callable(pose_forward):
                        tried_runner = pose_runner
                        pr = pose_forward(proc_inputs)
                        if isinstance(pr, (list, tuple)):
                            preds = list(pr)
                    if preds is None:
                        trt_runner = getattr(context.get("pose_inferencer"), "_hm_trt_runner", None)
                        trt_forward = getattr(trt_runner, "forward", None)
                        if callable(trt_forward) and trt_runner is not tried_runner:
                            pr = trt_forward(proc_inputs)
                            if isinstance(pr, (list, tuple)):
                                preds = list(pr)
                    if preds is None:
                        onnx_runner = getattr(
                            context.get("pose_inferencer"), "_hm_onnx_runner", None
                        )
                        onnx_forward = getattr(onnx_runner, "forward", None)
                        if callable(onnx_forward) and onnx_runner is not tried_runner:
                            pr = onnx_forward(proc_inputs)
                            if isinstance(pr, (list, tuple)):
                                preds = list(pr)
                    if preds is None:
                        forward_outputs = pose_impl.forward(
                            proc_inputs,
                            merge_results=False,
                            bbox_thr=bbox_thr,
                            pose_based_nms=pose_based_nms,
                        )
                        if isinstance(forward_outputs, (list, tuple)):
                            preds = list(forward_outputs)
                        elif forward_outputs is not None:
                            preds = [forward_outputs]

                if preds:
                    batch_index_iter = iter(frame_batch_indices)
                    for data_sample in preds:
                        frame_idx: Optional[int] = None
                        meta = getattr(data_sample, "metainfo", {}) or {}
                        if isinstance(meta, dict):
                            frame_idx = meta.get("hm_frame_index")
                        if not isinstance(frame_idx, int):
                            frame_idx = next(batch_index_iter, None)
                        if frame_idx is None or frame_idx < 0 or frame_idx >= len(inputs):
                            continue
                        if frame_predictions[frame_idx] is None:
                            frame_predictions[frame_idx] = []
                        frame_predictions[frame_idx].append(data_sample)

                for idx in range(len(inputs)):
                    if idx in empty_frame_indices:
                        all_pose_results.append({"predictions": _build_empty_predictions()})
                        continue
                    predictions = frame_predictions[idx] or []
                    if predictions:
                        first = predictions[0]
                        merged = PoseDataSample(metainfo=getattr(first, "metainfo", {}))
                        if hasattr(first, "gt_instances"):
                            merged.gt_instances = InstanceData.cat(
                                [p.gt_instances for p in predictions if hasattr(p, "gt_instances")]
                            )
                        if hasattr(first, "pred_instances"):
                            merged.pred_instances = InstanceData.cat(
                                [
                                    p.pred_instances
                                    for p in predictions
                                    if hasattr(p, "pred_instances")
                                ]
                            )
                        predictions = [merged]
                    all_pose_results.append({"predictions": predictions})

            if use_det_imgs and inv_scale_factors:
                self._restore_pose_outputs(all_pose_results, inv_scale_factors)

            # Manual visualization to match original behavior
            if show and getattr(pose_inferencer, "inferencer", None) is not None:
                vis = pose_inferencer.inferencer.visualizer
                if vis is not None:
                    for i, (img, pose_result) in enumerate(zip(original_images, all_pose_results)):
                        data_sample = pose_result.get("predictions", [])
                        if not data_sample:
                            continue
                        img = vis.add_datasample(
                            name="pose results",
                            image=img,
                            data_sample=data_sample[0],
                            clone_image=False,
                            draw_gt=False,
                            draw_bbox=False,
                            show_kpt_idx=False,
                            kpt_thr=kpt_thr if kpt_thr is not None else 0.3,
                        )
                        original_images[i] = img
                        drawn_original_images.append(img)
        else:
            # Fallback: let MMPose handle detection internally
            for pose_results in pose_inferencer(
                inputs=inputs,
                return_datasamples=True,
                visualize=show,
                **pose_inferencer.filter_args,
            ):
                all_pose_results.append(pose_results)

            if use_det_imgs and inv_scale_factors:
                self._restore_pose_outputs(all_pose_results, inv_scale_factors)

            if show and getattr(pose_inferencer, "inferencer", None) is not None:
                vis = pose_inferencer.inferencer.visualizer
                if vis is not None:
                    for img, pose_result in zip(original_images, all_pose_results):
                        data_sample = pose_result["predictions"]
                        assert len(data_sample) == 1
                        vis.add_datasample(
                            name="pose results",
                            image=img,
                            data_sample=data_sample[0],
                            clone_image=False,
                            draw_gt=False,
                            draw_bbox=False,
                            show_kpt_idx=False,
                        )
                        # show_image("pose", img, wait=False)

        pose_results = all_pose_results
        out: Dict[str, Any] = {"pose_results": pose_results}
        if drawn_original_images and isinstance(original_images, torch.Tensor):
            # `original_images` is mutated in-place (original_images[i] = img) above.
            out["original_images"] = wrap_tensor(original_images)
        return out

    @staticmethod
    def _get_track_data_sample(track_samples):
        if track_samples is None:
            return None
        if isinstance(track_samples, list):
            if len(track_samples) == 0:
                return None
            if len(track_samples) == 1:
                return track_samples[0]
            return track_samples[0]
        return track_samples

    @staticmethod
    def _collect_bboxes(track_data_sample, expected_len: Optional[int]):
        if track_data_sample is None:
            return None, None
        try:
            sample_len = len(track_data_sample)
        except Exception:
            return None, None
        bboxes: List[torch.Tensor] = []
        metas: List[Dict[str, Any]] = []
        for frame_index in range(sample_len):
            ds = track_data_sample[frame_index]
            meta = getattr(ds, "metainfo", {}) or {}
            if not isinstance(meta, dict):
                try:
                    meta = dict(meta)
                except Exception:
                    meta = {}
            metas.append(meta)
            inst = getattr(ds, "pred_track_instances", None)
            if inst is None or not hasattr(inst, "bboxes"):
                inst = getattr(ds, "pred_instances", None)
            if inst is None or not hasattr(inst, "bboxes"):
                bboxes.append(torch.empty((0, 5), dtype=torch.float32))
                continue
            box_tensor = inst.bboxes
            if not isinstance(box_tensor, torch.Tensor):
                box_tensor = torch.as_tensor(box_tensor)
            box_tensor = box_tensor.to(dtype=torch.float32)
            if box_tensor.ndim == 1:
                box_tensor = box_tensor.reshape(-1, 4)
            score_tensor = getattr(inst, "scores", None)
            if score_tensor is None:
                score_tensor = torch.ones(
                    (box_tensor.shape[0],), dtype=torch.float32, device=box_tensor.device
                )
            elif not isinstance(score_tensor, torch.Tensor):
                score_tensor = torch.as_tensor(
                    score_tensor, dtype=torch.float32, device=box_tensor.device
                )
            track_mask = get_track_mask(inst)
            if isinstance(track_mask, torch.Tensor):
                box_tensor = box_tensor[track_mask]
                score_tensor = score_tensor[track_mask]
            if score_tensor.ndim == 0:
                score_tensor = score_tensor.unsqueeze(0)
            if score_tensor.shape[0] != box_tensor.shape[0]:
                if score_tensor.shape[0] == 1 and box_tensor.shape[0] > 1:
                    score_tensor = score_tensor.expand(box_tensor.shape[0]).clone()
                else:
                    score_tensor = torch.ones(
                        (box_tensor.shape[0],), dtype=torch.float32, device=box_tensor.device
                    )
            combined = torch.cat([box_tensor, score_tensor.reshape(-1, 1)], dim=1)
            bboxes.append(combined.detach())
        if expected_len is not None and sample_len != expected_len:
            return None, metas
        return bboxes, metas

    @staticmethod
    def _normalize_det_images(det_imgs, expected_len: Optional[int]):
        if det_imgs is None:
            return None
        if not isinstance(det_imgs, torch.Tensor):
            return None
        if det_imgs.ndim == 5:
            if det_imgs.size(0) != 1:
                return None
            det_imgs = det_imgs.squeeze(0)
        if det_imgs.ndim != 4:
            return None
        if expected_len is not None and det_imgs.size(0) != expected_len:
            return None
        return det_imgs

    @staticmethod
    def _extract_scale_tensor(meta: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
        if not meta:
            return None
        scale_val = None
        for key in ("scale_factor", "scale_factors", "scale", "scale_factor_tensor"):
            if key in meta:
                scale_val = meta[key]
                break
        if scale_val is None:
            return None
        assert isinstance(scale_val, torch.Tensor)
        scale_tensor = scale_val.to(device=device, dtype=torch.float32)
        if scale_tensor.numel() == 1:
            scale_tensor = scale_tensor.repeat(4)
        elif scale_tensor.numel() == 2:
            scale_tensor = scale_tensor.repeat(2)
        elif scale_tensor.numel() != 4:
            return None
        return scale_tensor

    @staticmethod
    def _extract_img_hw(meta: Dict[str, Any]) -> Optional[tuple]:
        if not meta:
            return None
        for key in ("img_shape", "batch_input_shape", "pad_shape"):
            if key in meta:
                shape_val = meta[key]
                if hasattr(shape_val, "tolist"):
                    shape_val = shape_val.tolist()
                if isinstance(shape_val, (list, tuple)) and len(shape_val) >= 2:
                    return int(shape_val[0]), int(shape_val[1])
        return None

    @staticmethod
    def _scale_bboxes_for_detection(
        bboxes: List[torch.Tensor], scale_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        scaled: List[torch.Tensor] = []
        for boxes, scale in zip(bboxes, scale_list):
            if boxes.numel() == 0:
                scaled.append(boxes)
                continue
            s = scale.to(device=boxes.device, dtype=boxes.dtype)
            new_boxes = boxes.clone()
            new_boxes[:, 0] *= s[0]
            new_boxes[:, 1] *= s[1]
            new_boxes[:, 2] *= s[2]
            new_boxes[:, 3] *= s[3]
            scaled.append(new_boxes)
        return scaled

    def _restore_pose_outputs(
        self, pose_results: List[Dict[str, Any]], inv_scale_list: List[torch.Tensor]
    ) -> None:
        if not pose_results or not inv_scale_list:
            return
        count = min(len(pose_results), len(inv_scale_list))
        for idx in range(count):
            pose_result = pose_results[idx]
            inv_scale = inv_scale_list[idx]
            predictions = pose_result.get("predictions")
            if not predictions:
                continue
            for data_sample in predictions:
                inst = getattr(data_sample, "pred_instances", None)
                if inst is None:
                    continue
                if hasattr(inst, "keypoints") and inst.keypoints is not None:
                    inst.keypoints = self._apply_scale(
                        inst.keypoints, inv_scale[:2], keypoints=True
                    )
                if hasattr(inst, "bboxes") and inst.bboxes is not None:
                    inst.bboxes = self._apply_scale(inst.bboxes, inv_scale, keypoints=False)

    @staticmethod
    def _apply_scale(value, scale, *, keypoints: bool) -> torch.Tensor | None:
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            tensor = value.to(dtype=torch.float32)
            target_device = value.device
            return_tensor = True
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
            target_device = tensor.device
            return_tensor = False
        if tensor.numel() == 0:
            return value
        if keypoints:
            scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=target_device)
            if scale_tensor.numel() >= 4:
                scale_tensor = scale_tensor[:2]
            scale_tensor = scale_tensor.view(1, 1, 2)
            tensor = tensor * scale_tensor
        else:
            scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=target_device)
            if scale_tensor.numel() == 2:
                scale_tensor = torch.tensor(
                    [scale_tensor[0], scale_tensor[1], scale_tensor[0], scale_tensor[1]],
                    dtype=torch.float32,
                    device=target_device,
                )
            scale_tensor = scale_tensor.view(1, 4)
            tensor = tensor * scale_tensor
        if return_tensor:
            return tensor.to(dtype=value.dtype, device=value.device)
        return tensor

    def input_keys(self):
        return {
            "inputs",
            "data_samples",
            "original_images",
            "pose_inferencer",
            "plot_pose",
        }

    def output_keys(self):
        return {"pose_results", "original_images"}

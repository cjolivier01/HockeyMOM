from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import make_channels_last

from .base import Plugin


class PosePlugin(Plugin):
    """
    Runs multi-pose inference using an MMPose inferencer.

    Expects in context:
      - data: dict containing 'original_images'
      - pose_inferencer: initialized MMPoseInferencer
      - plot_pose: bool (optional)

    Produces in context:
      - data: updated with 'pose_results'
    """

    def __init__(self, enabled: bool = True, plot_pose: bool = False):
        # Need to import in order to register
        super().__init__(enabled=enabled)
        self._default_plot_pose: bool = bool(plot_pose)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        pose_inferencer = context.get("pose_inferencer")
        if pose_inferencer is None:
            return {}

        data: Dict[str, Any] = context["data"]
        original_images = data.get("original_images")
        if isinstance(original_images, StreamTensorBase):
            original_images = original_images.wait()
            data["original_images"] = original_images
        if original_images is None:
            return {}
        if not isinstance(original_images, torch.Tensor):
            original_images = torch.as_tensor(original_images)
            data["original_images"] = original_images

        track_data_sample = self._get_track_data_sample(data.get("data_samples"))
        frame_count = None
        if isinstance(original_images, torch.Tensor):
            frame_count = original_images.shape[0]
        elif hasattr(original_images, "__len__"):
            frame_count = len(original_images)

        per_frame_bboxes, frame_metas = self._collect_bboxes(track_data_sample, frame_count)

        det_imgs_tensor = data.get("img")
        if isinstance(det_imgs_tensor, StreamTensorBase):
            det_imgs_tensor = det_imgs_tensor.wait()
            data["img"] = det_imgs_tensor
        det_inputs_tensor = self._normalize_det_images(det_imgs_tensor, frame_count)

        use_det_imgs = False
        scale_factors: List[torch.Tensor] = []
        inv_scale_factors: List[torch.Tensor] = []
        if (
            det_inputs_tensor is not None
            and frame_metas is not None
            and frame_count is not None
            and len(frame_metas) == frame_count
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
            try:
                from mmpose.structures import merge_data_samples
            except Exception:
                merge_data_samples = None

            def _build_empty_predictions() -> List[Any]:
                try:
                    from mmengine.structures import InstanceData
                    from mmpose.structures import PoseDataSample

                    empty_ds = PoseDataSample()
                    empty_ds.pred_instances = InstanceData()
                    return [empty_ds]
                except Exception:
                    return []

            batched_data_infos: List[Any] = []
            frame_batch_indices: List[int] = []
            empty_frame_indices: Set[int] = set()
            for i, (img, bxs) in enumerate(zip(inputs, per_frame_bboxes)):
                if bxs.numel() == 0:
                    empty_frame_indices.add(i)
                    continue

                b_cpu = bxs.detach().cpu()
                b_np = np.ascontiguousarray(b_cpu.numpy())

                for bbox in b_np:
                    inst: Dict[str, Any] = {
                        "img": img,
                        # "img_path": str(i).rjust(10, "0") + ".jpg",
                        "img_path": None,
                        "hm_frame_index": i,
                    }
                    inst.update(dataset_meta)
                    inst["bbox"] = bbox[None, :4].astype(np.float32, copy=True)
                    inst["bbox_score"] = bbox[4:5].astype(np.float32, copy=True)
                    # inst["bbox"] = bbox[None, :4].to(torch.float32)
                    # inst["bbox_score"] = bbox[4:5].to(torch.float32)
                    batched_data_infos.append(pipeline(inst))
                    frame_batch_indices.append(i)

                # img_cpu = img.detach().cpu()
                # img_np = np.ascontiguousarray(img_cpu.numpy())
                # b_cpu = bxs.detach().cpu()
                # b_np = np.ascontiguousarray(b_cpu.numpy())

                # for bbox in b_np:
                #     inst: Dict[str, Any] = {
                #         "img": img_np,
                #         "img_path": str(i).rjust(10, "0") + ".jpg",
                #         "hm_frame_index": i,
                #     }
                #     inst.update(dataset_meta)
                #     inst["bbox"] = bbox[None, :4].astype(np.float32, copy=True)
                #     inst["bbox_score"] = bbox[4:5].astype(np.float32, copy=True)
                #     batched_data_infos.append(pipeline(inst))
                #     frame_batch_indices.append(i)

            frame_predictions: List[Optional[List[Any]]] = [None] * len(inputs)

            if batched_data_infos:
                proc_inputs = collate_fn(batched_data_infos)
                preds: Optional[List[Any]] = None
                # Prefer TensorRT runner if available, else ONNX
                trt_runner = getattr(
                    getattr(context.get("pose_inferencer"), "_hm_trt_runner", None),
                    "forward",
                    None,
                )
                if callable(trt_runner):
                    try:
                        pr = trt_runner(proc_inputs)
                        if isinstance(pr, (list, tuple)):
                            preds = list(pr)
                    except Exception:
                        preds = None
                if preds is None:
                    onnx_runner = getattr(
                        getattr(context.get("pose_inferencer"), "_hm_onnx_runner", None),
                        "forward",
                        None,
                    )
                    if callable(onnx_runner):
                        try:
                            pr = onnx_runner(proc_inputs)
                            if isinstance(pr, (list, tuple)):
                                preds = list(pr)
                        except Exception:
                            preds = None
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
                        else:
                            try:
                                frame_idx = meta.get("hm_frame_index")  # type: ignore[attr-defined]
                            except Exception:
                                frame_idx = None
                        if isinstance(frame_idx, np.ndarray):
                            if frame_idx.size == 1:
                                frame_idx = int(frame_idx.reshape(-1)[0])
                            else:
                                frame_idx = None
                        elif frame_idx is not None:
                            frame_idx = int(frame_idx)
                        if frame_idx is None:
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
                predictions = frame_predictions[idx]
                if predictions is None:
                    predictions = []
                elif predictions and merge_data_samples is not None:
                    try:
                        predictions = [merge_data_samples(predictions)]
                    except Exception:
                        pass
                all_pose_results.append({"predictions": predictions})

            if use_det_imgs and inv_scale_factors:
                self._restore_pose_outputs(all_pose_results, inv_scale_factors)

            # Manual visualization to match original behavior
            if show and getattr(pose_inferencer, "inferencer", None) is not None:
                vis = pose_inferencer.inferencer.visualizer
                if vis is not None:
                    for img, pose_result in zip(original_images, all_pose_results):
                        data_sample = pose_result.get("predictions", [])
                        if not data_sample:
                            continue
                        try:
                            vis.add_datasample(
                                name="pose results",
                                image=img,
                                data_sample=data_sample[0],
                                clone_image=False,
                                draw_gt=False,
                                draw_bbox=False,
                                show_kpt_idx=True,
                                kpt_thr=kpt_thr if kpt_thr is not None else 0.3,
                            )
                            # show_image("pose", img, wait=False)
                        except Exception:
                            pass
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
                        )
                        # show_image("pose", img, wait=False)

        pose_results = all_pose_results
        data["pose_results"] = pose_results
        return {"data": data}

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
        scale_tensor = scale_val
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
            s = scale.to(dtype=boxes.dtype)
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
        return {"data", "pose_inferencer", "plot_pose"}

    def output_keys(self):
        return {"data"}

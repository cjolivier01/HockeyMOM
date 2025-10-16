from typing import Any, Dict, List, Optional

import torch
import numpy as np

from hmlib.utils.gpu import StreamTensor
from hmlib.utils.image import make_channels_last

from .base import Trunk


class PoseTrunk(Trunk):
    """
    Runs multi-pose inference using an MMPose inferencer.

    Expects in context:
      - data: dict containing 'original_images'
      - pose_inferencer: initialized MMPoseInferencer
      - plot_pose: bool (optional)

    Produces in context:
      - data: updated with 'pose_results'
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        pose_inferencer = context.get("pose_inferencer")
        if pose_inferencer is None:
            return {}

        data: Dict[str, Any] = context["data"]
        cur_frame = data.get("original_images")
        if isinstance(cur_frame, StreamTensor):
            cur_frame = cur_frame.wait()
            data["original_images"] = cur_frame

        # Prepare inputs: iterate per-frame with channels-last layout
        inputs: List[torch.Tensor] = []
        for img in make_channels_last(cur_frame):
            inputs.append(img)

        # If upstream detection already produced bounding boxes per-frame,
        # reuse them for pose inference to avoid running MMPose's detector.
        # We look for `pred_instances` on each frame's TrackDataSample.
        def _collect_frame_bboxes() -> Optional[List[List[List[float]]]]:
            track_samples = data.get("data_samples")
            if track_samples is None:
                return None
            if isinstance(track_samples, list):
                if len(track_samples) != 1:
                    return None
                track_data_sample = track_samples[0]
            else:
                track_data_sample = track_samples
            try:
                video_len = len(track_data_sample)
            except Exception:
                return None

            # Build per-frame list of [x1,y1,x2,y2,score]
            out: List[List[List[float]]] = []
            any_found = False
            for frame_index in range(video_len):
                vds = track_data_sample[frame_index]
                inst = getattr(vds, "pred_instances", None)
                if inst is None:
                    # No detections for this frame; record an explicit empty list
                    out.append([])
                    continue
                b = getattr(inst, "bboxes", None)
                s = getattr(inst, "scores", None)
                if b is None:
                    out.append([])
                    continue
                # Normalize to tensors then to CPU numpy
                if not isinstance(b, torch.Tensor):
                    b = torch.as_tensor(b)
                if b.ndim == 1:
                    b = b.reshape(-1, 4)
                N = int(b.shape[0])
                if s is None:
                    s = torch.ones((N,), dtype=torch.float32, device=b.device)
                elif not isinstance(s, torch.Tensor):
                    s = torch.as_tensor(s, dtype=torch.float32, device=b.device)
                if s.ndim == 0:
                    s = s.unsqueeze(0)
                if len(s) != N:
                    if len(s) == 1 and N > 1:
                        s = s.expand(N).clone()
                    else:
                        s = torch.ones((N,), dtype=torch.float32, device=b.device)
                # Compose Nx5 arrays and convert to python lists of lists
                if N == 0:
                    out.append([])
                else:
                    any_found = True
                    bb = torch.cat([b.to(dtype=torch.float32), s.to(dtype=torch.float32).reshape(-1, 1)], dim=1)
                    out.append(bb.detach().cpu().tolist())
            # We only return a list if its length matches inputs
            if len(out) != len(inputs):
                return None
            # Even if no frames had dets, we return the empty lists to prevent
            # re-running a detector; downstream will handle empty cases.
            return out if (any_found or True) else None

        per_frame_bboxes = _collect_frame_bboxes()

        all_pose_results = []
        show = bool(context.get("plot_pose", False))
        # If we have per-frame bboxes and a Pose2D inferencer available, run a
        # custom forward that bypasses MMPose's internal detector.
        pose_impl = getattr(pose_inferencer, "inferencer", None)
        can_bypass = (
            per_frame_bboxes is not None
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

            for i, (img, bxs) in enumerate(zip(inputs, per_frame_bboxes)):
                # Build per-bbox data_infos similar to preprocess_single
                data_infos = []
                if len(bxs) > 0:
                    for bbox in bxs:
                        # bbox is [x1,y1,x2,y2,score]
                        # Create an instance dict expected by pipeline
                        inst = dict(img=img, img_path=str(i).rjust(10, "0") + ".jpg")
                        inst.update(dataset_meta)
                        arr = np.asarray(bbox, dtype=np.float32)
                        inst["bbox"] = arr[None, :4]
                        inst["bbox_score"] = arr[4:5]
                        data_infos.append(pipeline(inst))
                else:
                    # No bboxes provided; skip inference and produce an empty result later
                    data_infos = []

                if data_infos:
                    proc_inputs = collate_fn(data_infos)
                    preds = pose_impl.forward(
                        proc_inputs,
                        merge_results=True,
                        bbox_thr=bbox_thr,
                        pose_based_nms=pose_based_nms,
                    )
                    # `preds` is a list of PoseDataSample; wrap to match inferencer output
                    results = {"predictions": preds}
                else:
                    # Create an empty PoseDataSample to keep downstream visualization logic intact
                    try:
                        from mmpose.structures import PoseDataSample
                        from mmengine.structures import InstanceData

                        empty_ds = PoseDataSample()
                        empty_ds.pred_instances = InstanceData()
                        results = {"predictions": [empty_ds]}
                    except Exception:
                        # Fallback: no structured result
                        results = {"predictions": []}

                all_pose_results.append(results)

            # Manual visualization to match original behavior
            if show and getattr(pose_inferencer, "inferencer", None) is not None:
                vis = pose_inferencer.inferencer.visualizer
                if vis is not None:
                    for img, pose_result in zip(inputs, all_pose_results):
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
                                kpt_thr=kpt_thr if kpt_thr is not None else 0.3,
                            )
                        except Exception:
                            pass
        else:
            # Fallback: let MMPose handle detection internally
            for pose_results in pose_inferencer(
                inputs=inputs, return_datasamples=True, visualize=show, **pose_inferencer.filter_args
            ):
                all_pose_results.append(pose_results)

            if show and getattr(pose_inferencer, "inferencer", None) is not None:
                vis = pose_inferencer.inferencer.visualizer
                if vis is not None:
                    for img, pose_result in zip(inputs, all_pose_results):
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

        pose_results = all_pose_results
        data["pose_results"] = pose_results
        return {"data": data}

    def input_keys(self):
        return {"data", "pose_inferencer", "plot_pose"}

    def output_keys(self):
        return {"data"}

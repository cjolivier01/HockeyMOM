from typing import Any, Callable, Dict, List, Optional, Union

import mmcv
import mmengine
import numpy as np
import torch
from mmcv.transforms import Compose
from mmdet.models.mot.bytetrack import ByteTrack
from mmdet.registry import MODELS
from mmdet.structures import OptTrackSampleList
from mmdet.structures.bbox import bbox2result
from mmengine.structures import InstanceData


@MODELS.register_module()
class HmEndToEnd(ByteTrack):

    def __init__(
        self,
        *args,
        neck: Optional[Callable] = None,
        post_detection_pipeline: List[Dict[str, Any]] = None,
        post_tracking_pipeline: List[Dict[str, Any]] = None,
        enabled: bool = True,
        num_classes_override: Optional[int] = None,
        dataset_meta: Dict[str, Any] = None,
        **kwargs,
    ):
        # BaseModel tries to build it from the mmengine
        # registry, which can't find shit
        data_preprocessor = kwargs.get("data_preprocessor")
        if data_preprocessor and isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
            kwargs["data_preprocessor"] = data_preprocessor

        super().__init__(*args, **kwargs)
        self._enabled = enabled
        self.post_detection_pipeline = post_detection_pipeline
        self.post_detection_composed_pipeline = None
        self.post_tracking_pipeline = post_tracking_pipeline
        self.post_tracking_composed_pipeline = None
        self.neck = None
        self._num_classes_override = num_classes_override
        self.dataset_meta = dataset_meta

        if neck is not None:
            assert False
            # self.neck = build_neck(neck)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        if self.post_detection_pipeline and self.post_detection_composed_pipeline is None:
            self.post_detection_composed_pipeline = Compose(self.post_detection_pipeline)
        if self.post_tracking_pipeline and self.post_tracking_composed_pipeline is None:
            self.post_tracking_composed_pipeline = Compose(self.post_tracking_pipeline)
        results = super().forward(img, return_loss=return_loss, **kwargs)
        # if self.post_detection_composed_pipeline is not None:
        #     results = self.post_detection_composed_pipeline(results)
        return results

    def predict(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        data_samples: OptTrackSampleList = None,
        **kwargs: Dict[str, Any],
    ):
        return self.simple_test(inputs=inputs, data_samples=data_samples, **kwargs)

    def simple_test(
        self,
        inputs,
        data_samples: OptTrackSampleList = None,
        track: bool = True,
        **kwargs,
    ):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        if isinstance(data_samples, list):
            assert len(data_samples) == 1
            track_data_sample = data_samples[0]
        else:
            track_data_sample = data_samples
        video_len = len(track_data_sample)

        detect_all = True
        if detect_all:
            assert inputs.ndim == 5
            # only one video, but can be multiple frames
            assert inputs.shape[0] == 1
            det_inputs = inputs.squeeze(0)
            if True:
                # makes a difference?
                det_inputs = det_inputs.contiguous()

            all_det_results = self.detector.predict(det_inputs, track_data_sample)

        for frame_index in range(video_len):
            if detect_all:
                det_data_sample = all_det_results[frame_index]
                img_data_sample = track_data_sample[frame_index]
            else:
                img_data_sample = track_data_sample[frame_index]
                single_img = inputs[:, frame_index].contiguous()
                det_results = self.detector.predict(single_img, [img_data_sample])
                assert len(det_results) == 1, "Batch inference is not supported."
                det_data_sample = det_results[0]
            det_bboxes = det_data_sample.pred_instances.bboxes
            det_labels = det_data_sample.pred_instances.labels
            det_scores = det_data_sample.pred_instances.scores
            if self.post_detection_composed_pipeline is not None:
                # We may prune the detections to relevent items
                data: Dict[str, Any] = {
                    "det_bboxes": det_bboxes,
                    "labels": det_labels,
                    "scores": det_scores,
                    "prune_list": ["det_bboxes", "labels", "scores"],
                    "ori_shape": det_data_sample.metainfo["ori_shape"],
                    "data_samples": data_samples,
                }
                data.update(**kwargs)
                data = self.post_detection_composed_pipeline(data)
                det_bboxes = data["det_bboxes"]
                det_labels = data["labels"]
                det_scores = data["scores"]

                instance_data = InstanceData()
                instance_data["scores"] = det_scores
                instance_data["labels"] = det_labels
                instance_data["bboxes"] = det_bboxes
                det_data_sample.pred_instances = instance_data
                assert len(det_bboxes) == len(det_labels)
                assert len(det_scores) == len(det_labels)

            # Tracker will want to know the frame id so that it can expire lost tracks
            frame_id = det_data_sample.metainfo["img_id"].reshape([1])
            det_data_sample.set_metainfo({"frame_id": frame_id.item()})

            # track = False

            if track:
                pred_track_instances = self.tracker.track(data_sample=det_data_sample, **kwargs)
                img_data_sample.pred_track_instances = pred_track_instances

            # For performance purposes, add in the number of tracks we're tracking (both active and inactive)
            det_data_sample.set_metainfo({"nr_tracks": len(self.tracker)})

        if track and self.post_tracking_composed_pipeline is not None:
            data: Dict[str, Any] = {
                "data_samples": track_data_sample,
            }
            data = self.post_tracking_composed_pipeline(data)
            track_data_sample = data["data_samples"]

        return [track_data_sample]

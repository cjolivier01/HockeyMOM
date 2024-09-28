from typing import Any, Callable, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmdet.datasets.pipelines import Compose
from mmdet.models.builder import build_neck
from mmtrack.core import outs2results, results2outs
from mmtrack.models.mot.byte_track import ByteTrack

from ..builder import MODELS


@MODELS.register_module()
class HmEndToEnd(ByteTrack):

    def __init__(
        self,
        *args,
        neck: Optional[Callable] = None,
        post_detection_pipeline: List[Any] = None,
        enabled: bool = True,
        num_classes_override: Optional[int] = None,
        **kwargs,
    ):
        super(HmEndToEnd, self).__init__(*args, **kwargs)
        self._enabled = enabled
        self.post_detection_pipeline = post_detection_pipeline
        self.post_detection_composed_pipeline = None
        self.neck = None
        self._num_classes_override = num_classes_override

        if neck is not None:
            self.neck = build_neck(neck)

    def __call__(self, *args, **kwargs):
        return super(HmEndToEnd, self).__call__(*args, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        if self.post_detection_pipeline and self.post_detection_composed_pipeline is None:
            self.post_detection_composed_pipeline = Compose(self.post_detection_pipeline)
        results = super(HmEndToEnd, self).forward(img, return_loss=return_loss, **kwargs)
        # if self.post_detection_composed_pipeline is not None:
        #     results = self.post_detection_composed_pipeline(results)
        return results

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id == 0:
            self.tracker.reset()

        det_results = self.detector.simple_test(img, img_metas, rescale=rescale)
        assert len(det_results) == 1, "Batch inference is not supported."
        bbox_results = det_results[0]
        num_classes = len(bbox_results)

        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det["bboxes"]).to(img)
        det_labels = torch.from_numpy(outs_det["labels"]).to(img).long()

        if self.post_detection_composed_pipeline is not None:
            data = {
                "det_bboxes": det_bboxes,
                "labels": det_labels,
                "prune_list": ["det_bboxes", "labels"],
            }
            data.update(**kwargs)
            data = self.post_detection_composed_pipeline(data)
            det_bboxes = data["det_bboxes"]
            det_labels = data["labels"]
            assert len(det_bboxes) == len(det_labels)

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs,
        )
        assert len(track_bboxes) == len(track_ids)
        # print(f"track id {int(track_ids[0])} -> bbox = {[int(i) for i in track_bboxes[0]]}")
        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes,
        )
        det_results = outs2results(
            bboxes=det_bboxes,
            labels=det_labels,
            num_classes=(
                num_classes if self._num_classes_override is None else self._num_classes_override
            ),
        )

        results = dict(
            det_bboxes=det_results["bbox_results"],
            track_bboxes=track_results["bbox_results"],
            data=data,
        )
        assert results["data"]["original_images"].ndim == 4
        if self.neck is not None:
            jersey_results = self.neck(
                data=dict(
                    img=data["original_images"],
                    category_bboxes=track_results["bbox_results"],
                ),
            )
        else:
            jersey_results = None
        assert results["data"]["original_images"].ndim == 4
        results["jersey_results"] = jersey_results

        return results

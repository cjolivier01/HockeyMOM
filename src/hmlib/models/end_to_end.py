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
from mmengine.structures import BaseDataElement, InstanceData, PixelData


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
        self.neck = None
        self._num_classes_override = num_classes_override

        if neck is not None:
            assert False
            # self.neck = build_neck(neck)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        if self.post_detection_pipeline and self.post_detection_composed_pipeline is None:
            self.post_detection_composed_pipeline = Compose(self.post_detection_pipeline)
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
        rescale=False,
        data_samples: OptTrackSampleList = None,
        **kwargs,
    ):
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
        if True:
            if isinstance(data_samples, list):
                assert len(data_samples) == 1
                track_data_sample = data_samples[0]
            else:
                track_data_sample = data_samples
            video_len = len(track_data_sample)

            for frame_id in range(video_len):
                img_data_sample = track_data_sample[frame_id]
                single_img = inputs[:, frame_id].contiguous()
                # det_results List[DetDataSample]
                det_results = self.detector.predict(single_img, [img_data_sample])
                assert len(det_results) == 1, "Batch inference is not supported."
                det_data_sample = det_results[0]
                det_bboxes = det_data_sample.pred_instances.bboxes
                det_labels = det_data_sample.pred_instances.labels
                det_scores = det_data_sample.pred_instances.scores
                if self.post_detection_composed_pipeline is not None:
                    # We may prune the detections to relevent items
                    data = {
                        "det_bboxes": det_bboxes,
                        "labels": det_labels,
                        "scores": det_scores,
                        "prune_list": ["det_bboxes", "labels", "scores"],
                    }
                    data.update(**kwargs)
                    data = self.post_detection_composed_pipeline(data)
                    det_bboxes = data["det_bboxes"]
                    det_labels = data["labels"]
                    det_scores = data["scores"]

                    instance_data = InstanceData()
                    # instance_data["img_shape"] = det_data_sample.pred_instances.img_shape
                    # instance_data["pad_shape"] = det_data_sample.pred_instances.pad_shape
                    instance_data["scores"] = det_scores
                    instance_data["labels"] = det_labels
                    instance_data["bboxes"] = det_bboxes
                    det_data_sample.pred_instances = instance_data

                    # det_data_sample.pred_instances["bboxes"] = det_bboxes
                    # det_data_sample.pred_instances.labels = det_labels
                    # det_data_sample.pred_instances.scores = det_scores
                    assert len(det_bboxes) == len(det_labels)
                    assert len(det_scores) == len(det_labels)

                pred_track_instances = self.tracker.track(data_sample=det_results[0], **kwargs)
                img_data_sample.pred_track_instances = pred_track_instances

            return [track_data_sample]
        else:

            if not isinstance(data_samples, list):
                data_samples = [data_samples]
            track_data_samples = super().predict(
                inputs=inputs, data_samples=data_samples, redcale=rescale, **kwargs
            )
            assert len(track_data_samples) == 1

            frame_id = track_data_samples.video_data_samples[0].metainfo["img_id"]
            if frame_id == 0:
                self.tracker.reset()

            if img.ndim == 5:
                assert img.size(0) == 1
                img = img.squeeze(0)

            # img_meta_object = DictToObject(dict(metainfo=img_metas))
            det_results = self.detector.predict(
                img, data_samples.video_data_samples, rescale=rescale
            )
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
                    num_classes
                    if self._num_classes_override is None
                    else self._num_classes_override
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
                        img=results["data"]["original_images"],
                        category_bboxes=track_results["bbox_results"],
                    ),
                )
                results["jersey_results"] = (
                    jersey_results["jersey_results"] if "jersey_results" in jersey_results else None
                )
            assert results["data"]["original_images"].ndim == 4

            return results


def imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
    """Re-normalize the image.

    Args:
        img (Tensor | ndarray): Input image. If the input is a Tensor, the
            shape is (1, C, H, W). If the input is a ndarray, the shape
            is (H, W, C).
        img_norm_cfg (dict): Original configuration for the normalization.
        new_img_norm_cfg (dict): New configuration for the normalization.

    Returns:
        Tensor | ndarray: Output image with the same type and shape of
        the input.
    """
    if isinstance(img, torch.Tensor):
        assert img.ndim == 4 and img.shape[0] == 1
        new_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        new_img = _imrenormalize(new_img, img_norm_cfg, new_img_norm_cfg)
        new_img = new_img.transpose(2, 0, 1)[None]
        return torch.from_numpy(new_img).to(img)
    else:
        return _imrenormalize(img, img_norm_cfg, new_img_norm_cfg)


def _imrenormalize(img, img_norm_cfg, new_img_norm_cfg):
    """Re-normalize the image."""
    img_norm_cfg = img_norm_cfg.copy()
    new_img_norm_cfg = new_img_norm_cfg.copy()
    for k, v in img_norm_cfg.items():
        if (k == "mean" or k == "std") and not isinstance(v, np.ndarray):
            img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    # reverse cfg
    if "to_rgb" in img_norm_cfg:
        img_norm_cfg["to_bgr"] = img_norm_cfg["to_rgb"]
        img_norm_cfg.pop("to_rgb")
    for k, v in new_img_norm_cfg.items():
        if (k == "mean" or k == "std") and not isinstance(v, np.ndarray):
            new_img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    img = mmcv.imdenormalize(img, **img_norm_cfg)
    img = mmcv.imnormalize(img, **new_img_norm_cfg)
    return img


def outs2results(bboxes=None, labels=None, masks=None, ids=None, num_classes=None, **kwargs):
    """Convert tracking/detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        masks (torch.Tensor | np.ndarray): shape (n, h, w)
        ids (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, not including background class

    Returns:
        dict[str : list(ndarray) | list[list[np.ndarray]]]: tracking/detection
        results of each class. It may contain keys as belows:

        - bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.
        - mask_results (list[list[np.ndarray]]): Each outer list denotes masks
            of one category. Each inner list denotes one mask belonging to
            the category. Each mask has shape (h, w).
    """
    assert labels is not None
    assert num_classes is not None

    results = dict()

    if ids is not None:
        valid_inds = ids > -1
        ids = ids[valid_inds]
        labels = labels[valid_inds]

    if bboxes is not None:
        if ids is not None:
            bboxes = bboxes[valid_inds]
            if bboxes.shape[0] == 0:
                bbox_results = [np.zeros((0, 6), dtype=np.float32) for i in range(num_classes)]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate((ids[labels == i, None], bboxes[labels == i, :]), axis=1)
                    for i in range(num_classes)
                ]
        else:
            bbox_results = bbox2result(bboxes, labels, num_classes)
        results["bbox_results"] = bbox_results

    if masks is not None:
        if ids is not None:
            masks = masks[valid_inds]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks_results = [[] for _ in range(num_classes)]
        for i in range(bboxes.shape[0]):
            masks_results[labels[i]].append(masks[i])
        results["mask_results"] = masks_results

    return results


def results2outs(bbox_results=None, mask_results=None, mask_shape=None, **kwargs):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.
        mask_results (list[list[np.ndarray]]): Each outer list denotes masks of
            one category. Each inner list denotes one mask belonging to
            the category. Each mask has shape (h, w).
        mask_shape (tuple[int]): The shape (h, w) of mask.

    Returns:
        tuple: tracking results of each class. It may contain keys as belows:

        - bboxes (np.ndarray): shape (n, 5)
        - labels (np.ndarray): shape (n, )
        - masks (np.ndarray): shape (n, h, w)
        - ids (np.ndarray): shape (n, )
    """
    outputs = dict()

    if bbox_results is not None:
        labels = []
        for i, bbox in enumerate(bbox_results):
            labels.extend([i] * bbox.shape[0])
        labels = np.array(labels, dtype=np.int64)
        outputs["labels"] = labels

        bboxes = np.concatenate(bbox_results, axis=0).astype(np.float32)
        if bboxes.shape[1] == 5:
            outputs["bboxes"] = bboxes
        elif bboxes.shape[1] == 6:
            ids = bboxes[:, 0].astype(np.int64)
            bboxes = bboxes[:, 1:]
            outputs["bboxes"] = bboxes
            outputs["ids"] = ids
        else:
            raise NotImplementedError(f"Not supported bbox shape: (N, {bboxes.shape[1]})")

    if mask_results is not None:
        assert mask_shape is not None
        mask_height, mask_width = mask_shape
        mask_results = mmengine.concat_list(mask_results)
        if len(mask_results) == 0:
            masks = np.zeros((0, mask_height, mask_width)).astype(bool)
        else:
            masks = np.stack(mask_results, axis=0)
        outputs["masks"] = masks

    return outputs


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

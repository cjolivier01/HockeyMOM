import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox

from hmlib.bbox.box_functions import center, width
from hmlib.bbox.tiling import (
    clamp_boxes_to_image,
    get_non_overlapping_bbox_indices,
    get_original_bbox_index_from_tiled_image,
    pack_bounding_boxes_as_tiles,
)
from hmlib.builder import PIPELINES as TRANSFORMS
from hmlib.tracking_utils.timer import Timer
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.ui import show_image
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import image_height, image_width, make_channels_first, make_channels_last

# TV_10_1_ROSTER: Set[int] = {19, 9, 87, 7, 98, 78, 43, 10, 11, 39, 66, 92, 15}
TV_10_1_ROSTER: Set[int] = {}
SHARES_12_1_ROSTER: Set[int] = {29, 37, 40, 98, 73, 89, 54, 24, 79, 16, 27, 90, 57, 8, 96, 74}


@dataclass
class TrackJerseyInfo:
    tracking_id: int = -1
    number: int = -1
    score: float = 0.0


@TRANSFORMS.register_module()
class HmNumberClassifier:
    def __init__(
        self,
        *args,
        roster: Set[int] = set([*TV_10_1_ROSTER, *SHARES_12_1_ROSTER]),
        init_cfg: Optional[dict] = None,
        category: int = 0,
        enabled: bool = True,
        image_label: str = "img",
        **kwargs,
    ):
        # super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self._roster = roster
        self._category = category
        self._enabled = enabled
        self._image_label = image_label
        self._mean = [0.5, 0.5, 0.5]
        self._std = [0.5, 0.5, 0.5]
        self._inferencer = self.create_inferencer() if enabled else None
        self._timer = Timer()
        self._timer_interval = 50
        self._timer_count = 0
        if enabled:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in double_scalars",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Mean of empty slice.",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Support for mismatched key_padding_mask",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Use same attn_mask",
            )

    def __call__(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.forward(data, **kwargs)

    @staticmethod
    def create_inferencer():
        config = {
            "det": "FCENet",
            "det_weights": None,
            # "rec": "openmm/mmocr/configs/textrecog/svtr/svtr-small_20e_st_mj.py",
            # "rec_weights": "https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-small_20e_st_mj/svtr-small_20e_st_mj-35d800d6.pth",
            "rec": "ABINet",
            "kie": None,
            "kie_weights": None,
            "device": "cuda",
        }
        return MMOCRInferencer(**config)

    @staticmethod
    def _get_number_strings(strings: List[str]) -> List[str]:
        new_list: List[str] = []
        for s in strings:
            if s.isdigit():
                new_list.append(s)
        return new_list

    # @auto_fp16(apply_to=("img",))
    def forward(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # typing: none
        if not self._enabled or self._inferencer is None:
            return data
        self._timer.tic()
        img = data[self._image_label]
        if isinstance(img, StreamTensorBase):
            img._verbose = True
            img = img.get()
            data[self._image_label] = img
        all_jersey_results: List[List[TrackJerseyInfo]] = []
        track_data_sample = data["data_samples"]
        img = make_channels_first(img)
        w, h = int(image_width(img)), int(image_height(img))
        for image_item, data_sample in zip(img, track_data_sample):
            jersey_tracking_ids: Set[int] = set()
            jersey_results: List[TrackJerseyInfo] = []
            assert image_item.ndim == 3
            bboxes_xyxy = data_sample.pred_track_instances.bboxes
            bboxes_xyxy_clamped = clamp_boxes_to_image(boxes=bboxes_xyxy, image_size=(w, h))
            tracking_ids = data_sample.pred_track_instances.instances_id
            track_mask = get_track_mask(data_sample.pred_track_instances)
            if isinstance(track_mask, torch.Tensor):
                bboxes_xyxy_clamped = bboxes_xyxy_clamped[track_mask]
                tracking_ids = tracking_ids[track_mask]
            non_overlapping_bbox_indices = get_non_overlapping_bbox_indices(bboxes_xyxy_clamped)
            non_obvelapping_bboxes_xyxy = bboxes_xyxy_clamped[non_overlapping_bbox_indices]
            tracking_ids = tracking_ids[non_overlapping_bbox_indices]
            if not len(non_obvelapping_bboxes_xyxy):
                all_jersey_results.append(jersey_results)
                continue
            packed_image, index_map = pack_bounding_boxes_as_tiles(
                source_image=image_item, bounding_boxes=non_obvelapping_bboxes_xyxy.to(torch.int64)
            )
            use_img = make_channels_last(packed_image).contiguous().cpu().numpy()
            ocr_results = self._inferencer(
                inputs=use_img,
                return_vis=False,
                out_dir=None,
                progress_bar=False,
            )
            for vis in ocr_results["visualization"]:
                if vis is not None:
                    show_image("packed_image", vis, wait=False)
            text_and_centers = self.process_results(ocr_results)
            for text, x, y, w, score in text_and_centers:
                # if self._roster and int(text) not in self._roster:
                #     continue
                # show_image("packed_image", packed_image, wait=False)
                batch_index = get_original_bbox_index_from_tiled_image(index_map, y=y, x=x)
                bbox_width = width(bboxes_xyxy[batch_index])
                if batch_index >= 0:
                    tracking_id = tracking_ids[batch_index]
                    if tracking_id in jersey_tracking_ids:
                        print("DUPLICATE TRACKING ID FOR FRAME")
                    jersey_tracking_ids.add(tracking_id)
                    # We make the score:
                    # (% width of the bounding box that was the text) * (the recognition confidence score)
                    jersey_results.append(
                        TrackJerseyInfo(
                            tracking_id=int(tracking_id),
                            number=int(text),
                            score=float(score * (w / bbox_width)),
                        )
                    )
                else:
                    print("WTF")
            if jersey_results:
                print(f"{jersey_results=}")
            all_jersey_results.append(jersey_results)

        data["jersey_results"] = all_jersey_results
        # results["batch_numbers"] = batch_numbers
        self._timer.toc()
        self._timer_count += 1
        if self._timer_count % self._timer_interval:
            # logger.info(f"timer stuff")
            self._timer = Timer()
            self._timer_count = 0
        return data

    def process_results(
        self, ocr_results: Dict[str, Any], det_thresh: float = 0.5, rec_thresh: float = 0.8
    ) -> Dict[str, Any]:
        predictions = ocr_results["predictions"]
        assert len(predictions) == 1
        predictions = predictions[0]
        rec_texts = predictions["rec_texts"]
        rec_scores = predictions["rec_scores"]
        det_scores = predictions["det_scores"]
        det_polygons = predictions["det_polygons"]
        nr_items = len(rec_texts)
        centers: List[Tuple[str, float, float]] = []
        for index in range(nr_items):
            rec_text = rec_texts[index]
            if not rec_text or not rec_text.isdigit():
                continue
            if not det_scores[index] >= det_thresh:
                continue
            if not rec_scores[index] >= rec_thresh:
                continue
            number = int(rec_text)
            if number >= 100:
                continue
            # print(f"Good number: {rec_text}")
            bbox = poly2bbox(det_polygons[index])
            # Need # width relative to entire box width
            box_width = width(bbox)
            cc = center(bbox)
            centers.append(
                (rec_text, int(cc[0]), int(cc[1]), int(box_width), float(rec_scores[index]))
            )
        return centers


def get_polygon_center(polygon: List[float]) -> Tuple[float, float]:
    num_coords = len(polygon)
    assert num_coords % 2 == 0
    xs = polygon[1::2]
    ys = polygon[0::2]
    count = max(1, len(xs))
    return (sum(xs) / count, sum(ys) / count)


# def extract_and_resize_jerseys(
#     image,
#     bboxes,
#     out_width,
#     out_height,
#     down_from_box_top_ratio: float = 0.2,
#     number_height_from_box_size_ratio: float = 0.25,
#     # number_height_from_box_size_ratio: float = 0.2,
# ):
#     """
#     Extract and resize sub-images containing likely jersey number areas from given bounding boxes.

#     Args:
#     - image (torch.Tensor): The image tensor (C, H, W).
#     - bboxes (torch.Tensor): Tensor of bounding boxes (N, 4) where each box is (x, y, width, height).
#     - out_width (int): The desired output width of the cropped images.
#     - out_height (int): The desired output height of the cropped images.

#     Returns:
#     - torch.Tensor: A batch of cropped and resized images (N, C, out_height, out_width).
#     """
#     crops = []
#     image = make_channels_first(image)
#     iw, ih = image_width(image), image_height(image)
#     for bbox in bboxes:
#         x, y, width, height = bbox

#         # Calculate new coordinates for the jersey number area
#         new_y = int(y + down_from_box_top_ratio * height)
#         new_height = int(number_height_from_box_size_ratio * height)
#         # new_y = y
#         # new_height = int(height)

#         new_width = width
#         # new_width = int(width * 0.5)
#         # new_width = min(width, new_height)
#         # new_x = int(x + new_width / 2)
#         new_x = int(x + (width - new_width) / 2)

#         # Ensure the new box does not exceed image dimensions
#         new_y = max(new_y, 0)
#         new_height = min(new_height, ih - new_y)

#         # Crop the image
#         cropped = image[:, new_y : new_y + new_height, new_x : new_x + int(new_width)]
#         # cropped = image[:, y : y + height, x : x + width]

#         # show_image("crop", cropped, wait=False)
#         # show_image("crop", cropped, wait=True)

#         # Resize the cropped image
#         resized = F.interpolate(
#             cropped.unsqueeze(0), size=(out_height, out_width), mode="bilinear", align_corners=False
#         ).squeeze(0)
#         show_image("number?", resized, wait=True)
#         # outimg = make_channels_last(resized).cpu().numpy()
#         # did = cv2.imwrite("test-19.png", outimg)
#         crops.append(resized)

#     batch_crops = torch.stack(crops, dim=0)

#     return batch_crops


def sample():
    # Example usage
    source_image = torch.randn((3, 256, 256))  # Random source image of shape (3, 256, 256)
    N = 10  # 10 bounding boxes
    bounding_boxes = torch.randint(
        0, 200, (N, 4)
    )  # Random bounding boxes in (x, y, width, height) format

    packed_image, index_map = pack_bounding_boxes_as_tiles(source_image, bounding_boxes)


# Example usage:
# Assuming 'image_tensor' is a CxHxW tensor and 'bounding_boxes' is a Nx4 tensor
# 'desired_width' and 'desired_height' are the target dimensions for each crop
# cropped_images = extract_and_resize_jerseys(image_tensor, bounding_boxes, desired_width, desired_height)

# ARGS: List[str] = [
#     "/olivier-pool/Videos/ev-tv-10-1-2/test_numbers.png",
#     "--out-dir=/home/colivier/src/openmm/results",
#     "--rec=mmocr/configs/textrecog/abinet/abinet-vision_20e_st-an_mj.py",
#     "--rec-weights=https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth",
#     "--det=mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py",
#     "--det-weights=https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015/dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015_20220829_230108-f289bd20.pth",
#     "--kie=mmocr/configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py",
#     "--kie-weights=https://download.openmmlab.com/mmocr/kie/sdmgr/sdmgr_unet16_60e_wildreceipt/sdmgr_unet16_60e_wildreceipt_20220825_151648-22419f37.pth",
#     "--show",
# ]


def get_inferencer() -> MMOCRInferencer:

    config = {
        "det": "FCENet",
        "det_weights": None,
        "rec": "/home/colivier/src/hm/openmm/mmocr/configs/textrecog/svtr/svtr-small_20e_st_mj.py",
        "rec_weights": "https://download.openmmlab.com/mmocr/textrecog/svtr/svtr-small_20e_st_mj/svtr-small_20e_st_mj-35d800d6.pth",
        "kie": None,
        "kie_weights": None,
        "device": "cuda",
    }
    return MMOCRInferencer(**config)


def main():
    get_inferencer()


if __name__ == "__main__":
    # main()
    print("Done.")

"""
Ice Rink segmentation stuff
"""
import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from matplotlib.patches import Polygon
from mmdet.apis import inference_detector, init_detector
from mmdet.core.mask.structures import bitmap_to_polygon
from mmdet.models.detectors.base import BaseDetector
from PIL import Image

from hmlib.config import (
    get_game_config_private,
    get_game_dir,
    get_nested_value,
    save_private_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.models.loader import get_model_config
from hmlib.segm.utils import calculate_centroid, polygon_to_mask, scale_polygon
from hmlib.stitching.laplacian_blend import show_image as do_show_image
from hmlib.utils.gpu import GpuAllocator
from hmlib.utils.image import image_height, image_width, make_channels_last

DEFAULT_SCORE_THRESH = 0.3


def parse_args():
    parser = argparse.ArgumentParser(description="MMDetection video demo")
    # parser.add_argument("video", help="Video file")
    # parser.add_argument("config", help="Config file")
    # parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score-thr", type=float, default=DEFAULT_SCORE_THRESH, help="Bbox score threshold"
    )
    parser.add_argument("--out", type=str, help="Output video file")
    parser.add_argument("--perf", action="store_true", help="Performance run")
    parser.add_argument("--show", action="store_true", help="Show video")
    parser.add_argument(
        "--wait-time",
        type=float,
        default=1,
        help="The interval of show (s), 0 is block",
    )
    args = parser.parse_args()
    return args


def video_demo_main():
    args = parse_args()
    assert args.out or args.show or args.perf, (
        "Please specify at least one operation (save/show the "
        'video) with the argument "--out" or "--show"'
    )

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_reader = mmcv.VideoReader(args.video)
    if not args.perf:
        video_writer = None
        if args.out:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                args.out,
                fourcc,
                video_reader.fps,
                (video_reader.width, video_reader.height),
            )

    if args.perf:
        perf_run = True
        perf_counter = 0
        start_time = None

    for frame in mmcv.track_iter_progress(video_reader):

        # frame = torch.from_numpy(frame).to(args.device)

        if args.perf:
            # Timing
            if perf_counter == 0:
                start_time = time.time()
            perf_counter += 1

        result = inference_detector(model, frame)

        if not args.perf:
            frame = model.show_result(
                frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame,
                result,
                score_thr=args.score_thr,
            )
            if args.show:
                cv2.namedWindow("video", 0)
                mmcv.imshow(frame, "video", args.wait_time)
            if args.out:
                video_writer.write(frame)
        elif perf_counter == 20:
            stop_time = time.time()
            # fps = perf_counter / (stop_time - start_time)
            # print(f"\nfps={fps}")
            perf_counter = 0

    if not args.perf:
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def _get_first_frame(video_path: str) -> Optional[torch.Tensor]:
    video_reader = mmcv.VideoReader(str(video_path))
    frame = video_reader.read()
    if frame is None:
        return None
    return torch.from_numpy(frame).unsqueeze(0)


def enclosing_bbox(bboxes: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)

    bboxes = bboxes[:, :4]

    # Calculate widths and heights
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]

    # Filter out bounding boxes where width or height is zero
    valid_indices = (widths > 0) & (heights > 0)
    valid_bboxes = bboxes[valid_indices]

    # If no valid bboxes, return None or an indicative value
    if valid_bboxes.shape[0] == 0:
        return None

    # Calculate min and max coordinates for the remaining valid bounding boxes
    min_xy, _ = torch.min(valid_bboxes[:, :2], dim=0)
    max_xy, _ = torch.max(valid_bboxes[:, 2:], dim=0)

    # Concatenate min and max values to form the enclosing bounding box
    return torch.cat([min_xy, max_xy])


def result_to_polygons(
    inference_result: np.ndarray,
    category_id: int = 1,
    score_thr: float = 0,
    show: bool = False,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    """
    Theoretically, could return more than one polygon, especially if there's an obstruction
    """
    if isinstance(inference_result, tuple):
        bbox_result, segm_result = inference_result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = inference_result, None
    bboxes = np.vstack(bbox_result)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    category_mask = labels == category_id
    bboxes = bboxes[category_mask, :]
    labels = labels[category_mask]
    if segms is not None:
        segms = segms[category_mask, ...]

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    masks = segms

    contours_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    combined_mask = None

    combined_bbox = enclosing_bbox(bboxes)

    for _, mask in enumerate(masks):
        contours, _ = bitmap_to_polygon(mask)
        # split_points_by_x_trend_efficient(contours)
        contours_list += contours
        mask = mask.astype(bool)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask | mask
        mask_list.append(mask)

        if show:
            mask_image = mask.astype(np.uint8) * 255
            # cv2.namedWindow("Ice-rink", 0)
            # mmcv.imshow(mask_image, "Ice-rink Mask", wait_time=90)
            do_show_image("Ice-rink", mask_image)

    results: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]] = {}
    results["contours"] = contours_list
    results["masks"] = mask_list
    results["combined_mask"] = combined_mask
    results["centroid"] = calculate_centroid(contours_list)
    results["bboxes"] = bboxes
    results["combined_bbox"] = combined_bbox

    return results


def contours_to_polygons(contours: List[np.ndarray]) -> List[Polygon]:
    return [Polygon(c) for c in contours]


def detect_ice_rink_mask(
    image: Union[torch.Tensor, np.ndarray],
    model: BaseDetector,
    show: bool = False,
    scale: float = None,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            assert image.shape[0] == 1
            image = image.squeeze(0)
        image = image.cpu().numpy()
    image = make_channels_last(image)
    result = inference_detector(model, image)

    if show:
        show_image = image.cpu().unsqueeze(0).numpy() if isinstance(image, torch.Tensor) else image
        show_image = model.show_result(
            show_image, result, score_thr=DEFAULT_SCORE_THRESH, show=False
        )
        do_show_image("Ice-rink", show_image, wait=True)

    rink_results = result_to_polygons(
        inference_result=result, score_thr=DEFAULT_SCORE_THRESH, show=False
    )

    if scale and scale != 1.0:
        for contour, mask in zip(rink_results["contours"], rink_results["masks"]):
            scaled_contour = scale_polygon(contour, scale)
            generated_mask = polygon_to_mask(
                scaled_contour, height=image_height(image), width=image_width(image)
            )
            do_show_image(
                "Ice-rink Mask", generated_mask.cpu().numpy().astype(np.uint8) * 255, wait=True
            )

    return rink_results


def find_ice_rink_masks(
    image: Union[torch.Tensor, List[torch.Tensor]],
    config_file: str,
    checkpoint: str,
    device: Optional[torch.device] = None,
    show: bool = False,
    scale: float = None,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    if device is None:
        device = torch.device("cuda:0")

    was_list = isinstance(image, list)
    if not was_list:
        image = [image]

    model = init_detector(config_file, checkpoint, device=device)

    results: List[
        Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]
    ] = []
    for img in image:
        results.append(detect_ice_rink_mask(image=img, model=model, show=show, scale=scale))
    if not was_list:
        assert len(results) == 1
        return results[0]
    return results


def load_png_as_boolean_tensor(filename: str) -> torch.Tensor:
    """
    Loads a PNG image and converts it to a 2D boolean tensor.

    Args:
        filename (str): The path to the PNG image file.

    Returns:
        torch.Tensor: A 2D boolean tensor where each value represents the pixel's thresholded result.
    """
    # Load the image
    image = Image.open(filename).convert("L")  # Convert to grayscale

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Define a threshold (this example uses 128 as a threshold for mid-gray)
    threshold = 128

    # Apply threshold to create a boolean array
    boolean_array = image_array > threshold

    # Convert to a PyTorch boolean tensor
    boolean_tensor = torch.from_numpy(boolean_array).to(torch.bool)

    return boolean_tensor


def save_boolean_tensor_as_png(tensor: Union[torch.Tensor, np.ndarray], filename: str):
    """
    Saves a PyTorch boolean tensor as a PNG image.

    Args:
        tensor (torch.Tensor): A 2D boolean tensor.
        filename (str): The filename where the image will be saved.
    """
    # Ensure the tensor is on CPU and convert to uint8
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    tensor = tensor.cpu().to(torch.uint8) * 255  # Convert boolean to 0 and 255

    # Convert to a PIL image
    image = Image.fromarray(tensor.numpy())

    # Save the image
    image.save(filename)


def save_rink_profile_config(
    game_id: str,
    rink_profile: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]],
) -> Dict[str, Any]:
    game_config = get_game_config_private(game_id=game_id)
    masks = rink_profile.get("masks")
    mask_count = len(masks) if masks is not None else 0
    set_nested_value(game_config, "rink.ice_contours_mask_count", mask_count)
    centroid = rink_profile["centroid"]
    centroid = [float(centroid[0]), float(centroid[1])]
    set_nested_value(game_config, "rink.ice_contours_mask_centroid", centroid)

    combined_bbox = None
    if rink_profile["combined_bbox"] is not None:
        combined_bbox = [float(i) for i in rink_profile["combined_bbox"]]
    set_nested_value(game_config, "rink.ice_contours_combined_bbox", combined_bbox)
    mask_image_file_base = f'{os.environ["HOME"]}/Videos/{game_id}/rink_mask_'
    for i in range(mask_count):
        mask = masks[i]
        image_file = mask_image_file_base + str(i) + ".png"
        save_boolean_tensor_as_png(mask, image_file)
    save_private_config(game_id=game_id, data=game_config, verbose=True)


def load_rink_combined_mask(
    game_id: str,
) -> Optional[Dict[str, Optional[torch.Tensor]]]:
    game_config = get_game_config_private(game_id=game_id)
    if not game_config:
        return None
    mask_count = get_nested_value(game_config, "rink.ice_contours_mask_count", None)
    if mask_count is None:
        return None
    combined_mask = None
    mask_image_file_base = f'{os.environ["HOME"]}/Videos/{game_id}/rink_mask_'
    for i in range(mask_count):
        image_file = mask_image_file_base + str(i) + ".png"
        if not os.path.exists(image_file):
            # Missing the actual mask file, so return as if nothing was found
            return None
        mask = load_png_as_boolean_tensor(image_file)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask | mask
    mask_count = get_nested_value(game_config, "rink.ice_contours_mask_count", None)
    centroid = get_nested_value(game_config, "rink.ice_contours_mask_centroid", None)
    if centroid is not None:
        centroid = torch.tensor(centroid, dtype=torch.float)
    combined_bbox = get_nested_value(game_config, "rink.ice_contours_combined_bbox", None)
    results: Dict[str, Optional[torch.Tensor]] = {
        "combined_mask": combined_mask,
        "centroid": centroid,
        "combined_bbox": combined_bbox,
    }
    return results


# def load_rink_detector(game_id: str, device: torch.device) -> BaseDetector:
#     pass


def confgure_ice_rink_mask(
    game_id: str,
    device: Optional[torch.device] = None,
    force: bool = False,
    show: bool = False,
) -> Optional[torch.Tensor]:
    if not force:
        combined_mask = load_rink_combined_mask(game_id=game_id)
        if combined_mask is not None:
            return combined_mask

    model_config_file, model_checkpoint = get_model_config(game_id=game_id, model_name="ice_rink_segm")

    # game_config = get_config(game_id=game_id)
    # config_file = get_nested_value(game_config, "model.ice_rink_segm.config")
    assert model_config_file
    # checkpoint = get_nested_value(game_config, "model.ice_rink_segm.checkpoint")
    assert model_checkpoint

    image_file = Path(get_game_dir(game_id=game_id)) / "s.png"
    if not image_file.exists():
        raise AttributeError(f"Could not find stitched frame image: {image_file}")

    rink_results = find_ice_rink_masks(
        image=_get_first_frame(image_file),
        config_file=model_config_file,
        checkpoint=model_checkpoint,
        show=show,
        scale=None,
        device=device,
    )
    if rink_results:
        save_rink_profile_config(game_id=game_id, rink_profile=rink_results)
    return load_rink_combined_mask(game_id=game_id)


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / ".." / ".." / "..")

    # args.game_id = "jrmocks"
    args.game_id = "pdp"

    gpu_allocator = GpuAllocator(gpus=args.gpus)
    device: torch.device = (
        torch.device("cuda", gpu_allocator.allocate_fast())
        if not gpu_allocator.is_single_lowmem_gpu(low_threshold_mb=1024 * 10)
        else torch.device("cpu")
    )

    assert args.game_id

    results = confgure_ice_rink_mask(
        game_id=args.game_id,
        device=device,
        show=False,
        force=True,
    )
    print(f"centroid={results['centroid']}")

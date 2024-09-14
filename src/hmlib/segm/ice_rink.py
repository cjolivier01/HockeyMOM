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
from PIL import Image

from hmlib.config import (
    get_game_config,
    get_nested_value,
    save_game_config,
    set_nested_value,
)
from hmlib.hm_opts import hm_opts
from hmlib.segm.utils import calculate_centroid, polygon_to_mask, scale_polygon
from hmlib.utils.image import image_height, image_width

DEFAULT_SCORE_THRESH = 0.3


def rle_encode(tensor: Union[torch.Tensor, np.ndarray], validate: bool = True) -> torch.Tensor:
    """
    Encodes a 1D boolean tensor using run-length encoding (RLE).

    Args:
        tensor (torch.Tensor): A 1D boolean tensor.

    Returns:
        torch.Tensor: A 2xN tensor where the first row contains values and the second row contains lengths.
    """
    # Ensure tensor is flattened
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    tensor = tensor.flatten()
    # Find boundaries (where changes occur)
    diffs = torch.cat([torch.tensor([True]), tensor[1:] != tensor[:-1]])
    # Get indices of changes
    indices = torch.where(diffs)[0]
    # Calculate run lengths
    lengths = torch.diff(torch.cat([indices, torch.tensor([len(tensor)])]))
    # Get values corresponding to the runs
    values = tensor[indices]

    # Return as 2xN tensor
    rle_encoded = torch.stack((values.to(torch.int32), lengths))

    if validate:
        test_decode = rle_decode(rle_encoded)
        assert torch.sum(test_decode) == torch.sum(tensor)
        assert torch.sum(test_decode) == torch.sum(test_decode | tensor)
        assert torch.sum(test_decode) == torch.sum(test_decode & tensor)

    return rle_encoded


def rle_decode(encoded_tensor):
    """
    Decodes a run-length encoded tensor back into the original tensor.

    Args:
        encoded_tensor (torch.Tensor): A 2xN tensor from rle_encode, where the first row contains values and the second row contains lengths.

    Returns:
        torch.Tensor: The decoded 1D boolean tensor.
    """
    # Extract values and lengths
    values, lengths = encoded_tensor[0], encoded_tensor[1]
    # Convert values back to boolean
    bool_values = values.to(torch.bool)
    # Repeat each value according to its run length
    decoded = torch.repeat_interleave(bool_values, lengths)

    return decoded


# def rle_encode(data: np.ndarray, validate: bool = True) -> List[Tuple[bool, int]]:
#     orig = make_channels_last(data[np.newaxis, :])
#     assert data.dtype == np.bool
#     data = data.flatten()
#     ends = np.where(data[1:] != data[:-1])[0] + 1
#     lengths = np.diff(np.append(-1, ends))
#     encoded_pixel_count = np.sum(lengths)
#     values = data[ends]
#     encoded_vals_lengths = np.stack([values.astype(np.int32), lengths])
#     # encoded_list = list(zip(values, lengths))
#     # return encoded_list

#     if validate:
#         test_decode = rle_decode(
#             encoded_vals_lengths, image_width=image_width(orig), image_height=image_height(orig)
#         )
#         assert make_channels_last(test_decode).numpy() == orig.squeeze(axis=0)

#     return encoded_vals_lengths


# def rle_decode(rle_tensor: np.ndarray, image_width: int, image_height: int) -> torch.Tensor:
#     """
#     Decodes an RLE-compressed boolean tensor.

#     Args:
#         rle_tensor (torch.Tensor): A 2xN tensor where the first row contains values (0 or 1),
#                                    and the second row contains the corresponding lengths.

#     Returns:
#         torch.Tensor: A decompressed boolean tensor.
#     """
#     if isinstance(rle_tensor, np.ndarray):
#         rle_tensor = torch.from_numpy(rle_tensor)

#     values = rle_tensor[0, :]  # First row, values
#     lengths = rle_tensor[1, :]  # Second row, run lengths

#     # Convert values to boolean
#     bool_values = values.to(torch.bool)

#     # Repeat each boolean value according to the run length
#     decompressed = torch.repeat_interleave(bool_values, lengths)

#     expected_numel = image_width * image_height
#     actual_numel = decompressed.numel()

#     assert expected_numel == actual_numel

#     decompressed = decompressed.reshape((image_height, image_width)).unsqueeze(0)

#     return decompressed


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
            fps = perf_counter / (stop_time - start_time)
            # print(f"\nfps={fps}")
            perf_counter = 0

    if not args.perf:
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def _get_first_frame(video_path: str) -> Optional[torch.Tensor]:
    video_reader = mmcv.VideoReader(video_path)
    frame = video_reader.read()
    if frame is None:
        return None
    return torch.from_numpy(frame).unsqueeze(0)


def result_to_polygons(
    inference_result: np.ndarray,
    category_id: int = 1,
    score_thr: float = 0,
    show: bool = False,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    """
    Theoretically, could return more than one polygon, especially if there's an obstruction
    """
    polygons: List[List[Tuple[float, float]]] = []

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
            cv2.namedWindow("Ice-rink", 0)
            mmcv.imshow(mask_image, "Ice-rink Mask", wait_time=90)

    results: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]] = {}
    results["contours"] = contours_list
    results["masks"] = mask_list
    results["combined_mask"] = combined_mask
    results["centroid"] = calculate_centroid(contours_list)

    return results


def contours_to_polygons(contours: List[np.ndarray]) -> List[Polygon]:
    return [Polygon(c) for c in contours]


def find_ice_rink_mask(
    image: torch.Tensor,
    config_file: str,
    checkpoint: str,
    device: Optional[torch.device] = None,
    show: bool = False,
    scale: float = None,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]]:
    if device is None:
        device = torch.device("cuda:0")
    model = init_detector(config_file, checkpoint, device=device)
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze(0)
    result = inference_detector(model, image)

    if show:
        show_image = image.cpu().unsqueeze(0).numpy() if isinstance(image, torch.Tensor) else image
        show_image = model.show_result(
            show_image, result, score_thr=DEFAULT_SCORE_THRESH, show=False
        )
        for _ in range(10):
            cv2.namedWindow("Ice-rink", 0)
            mmcv.imshow(show_image, "Ice-rink", wait_time=1)
            time.sleep(1)

    rink_results = result_to_polygons(
        inference_result=result, score_thr=DEFAULT_SCORE_THRESH, show=False
    )

    if scale and scale != 1.0:
        for contour, mask in zip(rink_results["contours"], rink_results["masks"]):
            scaled_contour = scale_polygon(contour, scale)
            generated_mask = polygon_to_mask(
                scaled_contour, height=image_height(image), width=image_width(image)
            )
            cv2.namedWindow("Ice-rink", 0)
            for _ in range(30):
                mmcv.imshow(
                    generated_mask.cpu().numpy().astype(np.uint8) * 255,
                    "Ice-rink",
                    wait_time=10,
                )
                time.sleep(1)

    return rink_results


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
    root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    masks = rink_profile.get("masks")
    mask_count = len(masks) if masks is not None else 0
    set_nested_value(game_config, "rink.ice_contours_mask_count", mask_count)
    centroid = rink_profile["centroid"]
    centroid = [float(centroid[0]), float(centroid[1])]
    set_nested_value(game_config, "rink.ice_contours_mask_centroid", centroid)
    mask_image_file_base = f'{os.environ["HOME"]}/Videos/{args.game_id}/rink_mask_'
    for i in range(mask_count):
        mask = masks[i]
        image_file = mask_image_file_base + str(i) + ".png"
        save_boolean_tensor_as_png(mask, image_file)
    save_game_config(game_id=game_id, data=game_config, root_dir=root_dir)


def load_rink_combined_mask(
    game_id: str,
    root_dir: Optional[str] = None,
) -> Optional[Dict[str, Optional[torch.Tensor]]]:
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if not game_config:
        return None
    mask_count = get_nested_value(game_config, "rink.ice_contours_mask_count", None)
    if mask_count is None:
        return None
    combined_mask = None
    mask_image_file_base = f'{os.environ["HOME"]}/Videos/{game_id}/rink_mask_'
    for i in range(mask_count):
        image_file = mask_image_file_base + str(i) + ".png"
        mask = load_png_as_boolean_tensor(image_file)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = combined_mask | mask
    mask_count = get_nested_value(game_config, "rink.ice_contours_mask_count", None)
    centroid = get_nested_value(game_config, "rink.ice_contours_mask_centroid", None)
    if centroid is not None:
        centroid = torch.tensor(centroid, dtype=torch.float)
    results: Dict[str, Optional[torch.Tensor]] = {
        "combined_mask": combined_mask,
        "centroid": centroid,
    }
    return results


def confgure_ice_rink_mask(
    game_id: str,
    root_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    force: bool = False,
    show: bool = False,
) -> Optional[torch.Tensor]:
    if not force:
        combined_mask = load_rink_combined_mask(game_id=game_id, root_dir=root_dir)
        if combined_mask is not None:
            return combined_mask

    config_file = (
        f"{root_dir}/config/models/ice_rink/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    )
    checkpoint = f"{root_dir}/pretrained/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/ice_rink_iter_19500.pth"

    image_file = f'{os.environ["HOME"]}/Videos/{game_id}/s.png'

    rink_results = find_ice_rink_mask(
        image=_get_first_frame(image_file),
        config_file=config_file,
        checkpoint=checkpoint,
        show=show,
        scale=None,
        device=device,
    )
    if rink_results:
        save_rink_profile_config(game_id=game_id, rink_profile=rink_results, root_dir=root_dir)
    return load_rink_combined_mask(game_id=game_id, root_dir=root_dir)


if __name__ == "__main__":
    opts = hm_opts()
    args = opts.parse()

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / ".." / ".." / "..")

    # args.game_id = "jrmocks"
    args.game_id = "sharks-bb1-2"

    assert args.game_id

    results = confgure_ice_rink_mask(
        game_id=args.game_id,
        root_dir=root_dir,
        # device="cpu",
        device="cuda:0",
        show=False,
        force=True,
    )
    print(results)

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from matplotlib.patches import Polygon
from mmdet.apis import inference_detector, init_detector
from mmdet.core.mask.structures import bitmap_to_polygon

from hmlib.segm.utils import polygon_to_mask

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

    # num_bboxes = bboxes.shape[0]

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

    # draw_masks(ax, img, segms, colors, with_edge=True)
    masks = segms

    contours_list: List[List[Tuple[int, int]]] = []
    polygons_list: List[Polygon] = []
    mask_list: List[np.ndarray] = []

    for _, mask in enumerate(masks):
        contours, _ = bitmap_to_polygon(mask)
        contours_list += contours
        polygons_list += [Polygon(c) for c in contours]
        mask = mask.astype(bool)
        mask_list.append(mask)

        if show:
            mask_image = mask.astype(np.uint8) * 255
            cv2.namedWindow("Ice-rink", 0)
            mmcv.imshow(mask_image, "Ice-rink Mask", wait_time=90)

    results: Dict[str, Union[List[List[Tuple[int, int]]], List[Polygon], List[np.ndarray]]] = {}
    results["contours"] = contours_list
    results["polygons"] = polygons_list
    results["masks"] = mask_list

    return results


def find_ice_rink_mask(
    image: torch.Tensor,
    config_file: str,
    checkpoint: str,
    device: torch.device = torch.device("cuda:0"),
    show: bool = False,
) -> torch.Tensor:
    model = init_detector(config_file, checkpoint, device=device)
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze(0)
    result = inference_detector(model, image)

    if show:
        show_image = image.cpu().unsqueeze(0).numpy() if isinstance(image, torch.Tensor) else image
        show_image = model.show_result(
            show_image, result, score_thr=DEFAULT_SCORE_THRESH, show=False
        )

        # cv2.namedWindow("Ice-rink", 0)
        # mmcv.imshow(show_image, "Ice-rink", wait_time=10)

    polygons = result_to_polygons(
        inference_result=result, score_thr=DEFAULT_SCORE_THRESH, show=True
    )

    print("Done.")


if __name__ == "__main__":
    video_file = "/home/colivier/Videos/tvbb2/stitched-short.mkv"
    config_file = "/home/colivier/src/hm/config/models/ice_rink/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    checkpoint = (
        "/mnt/data/pretrained/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/ice_rink_iter_19500.pth"
    )

    mask = find_ice_rink_mask(
        image=_get_first_frame(video_file),
        config_file=config_file,
        checkpoint=checkpoint,
        show=True,
    )

    # video_demo_main()

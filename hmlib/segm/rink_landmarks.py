"""
Rink landmark segmentation: the painted features on the ice -- blue lines, the
center line, faceoff circles and dots, creases, goals, trapezoids -- as opposed
to :mod:`hmlib.segm.ice_rink`, which segments the playing surface itself.

The model is the 11-class Mask2Former trained by ``openmm/train_rink_landmarks.sh``
on the Roboflow "hockey-rink-landmarks" v2 dataset. Class names and colors come
from the checkpoint's ``dataset_meta``, so this module never hardcodes them.

Everything here is panorama-oriented: ``detect_rink_landmarks`` runs once on a
stitched frame, and ``build_overlay_layer`` bakes the result into a single
(color, alpha) pair so that per-frame compositing is one blend on the GPU.
"""

import argparse
import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from hmlib.config import prepend_root_dir
from hmlib.log import logger
from hmlib.models.loader import get_model_config
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import (
    image_height,
    image_width,
    is_channels_first,
    make_channels_first,
    make_channels_last,
)

if TYPE_CHECKING:
    from mmdet.models.detectors.base import BaseDetector
    from mmdet.structures import DetDataSample

# The landmark model emits up to 50 instances per image and is far less
# confident than the single-class rink model, so it needs its own floor.
DEFAULT_SCORE_THRESH = 0.5

# "Field" is the whole playing surface -- the ice rink model already covers that
# far better, and as a full-frame mask it would bury every other class under a
# flat tint. Drop it from overlays unless it is asked for by name.
DEFAULT_EXCLUDED_CLASSES = ("Field",)

# Longest edge handed to the detector when no explicit scale is given.
#
# This is a memory cap, not a quality knob. The model's own test pipeline
# resizes the long edge to 1008 regardless, so the network sees the same
# pixels either way -- but Mask2Former's instance_postprocess first upsamples
# all 50 query masks back to the *input* resolution, which on a 12407x4710
# panorama is 50 * 58 Mpx * 4 bytes = 11.7 GiB and OOMs a 32 GB card. Masks are
# resampled back to full resolution afterwards, so capping here costs nothing
# the 1008-pixel network output had to begin with.
DEFAULT_MAX_INFERENCE_EDGE = 2048


def _auto_inference_scale(width: int, height: int) -> Optional[float]:
    """Scale that brings the longest edge down to ``DEFAULT_MAX_INFERENCE_EDGE``."""
    longest = max(int(width), int(height))
    if longest <= DEFAULT_MAX_INFERENCE_EDGE:
        return None
    return DEFAULT_MAX_INFERENCE_EDGE / float(longest)


def landmark_metadata(model: "BaseDetector") -> Tuple[Tuple[str, ...], List[Tuple[int, int, int]]]:
    """Return ``(classes, palette)`` from a detector's ``dataset_meta``."""
    meta = getattr(model, "dataset_meta", None) or {}
    classes = tuple(meta.get("classes") or ())
    palette = [tuple(int(c) for c in color) for color in (meta.get("palette") or [])]
    if not palette and classes:
        # Deterministic fallback so an untagged checkpoint still renders.
        palette = [_hsv_color(i, len(classes)) for i in range(len(classes))]
    return classes, palette


def _hsv_color(index: int, total: int) -> Tuple[int, int, int]:
    hue = int(179 * index / max(1, total))
    pixel = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _as_numpy_image(image: Union[torch.Tensor, np.ndarray, StreamTensorBase]) -> np.ndarray:
    if isinstance(image, StreamTensorBase):
        image = image.get()
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            # Calibration only ever needs one frame; a larger stitch batch is
            # still a single panorama repeated through time.
            image = image[0]
        image = make_channels_last(image)
        if torch.is_floating_point(image):
            image = image.clamp(0, 255)
        image = image.to(torch.uint8).cpu().numpy()
    else:
        image = make_channels_last(image)
    return np.ascontiguousarray(image)


def detect_rink_landmarks(
    image: Union[torch.Tensor, np.ndarray],
    model: "BaseDetector",
    score_thr: float = DEFAULT_SCORE_THRESH,
) -> Optional[Dict[str, Any]]:
    """Run the landmark detector on one frame and return its instance masks."""
    from mmdet.apis import inference_detector

    frame = _as_numpy_image(image)
    result: "DetDataSample" = inference_detector(model, frame)

    instances = result.pred_instances
    scores = instances.scores
    keep = scores >= float(score_thr)
    if not bool(keep.any()):
        return None

    masks = instances.masks
    if masks is None:
        logger.warning("Rink landmark model returned no masks; is it a bbox-only checkpoint?")
        return None

    classes, palette = landmark_metadata(model)
    return {
        "labels": instances.labels[keep].cpu(),
        "scores": scores[keep].cpu(),
        "bboxes": instances.bboxes[keep].cpu(),
        "masks": masks[keep].to(torch.bool).cpu(),
        "classes": classes,
        "palette": palette,
    }


def _rescale_landmarks(
    landmarks: Dict[str, Any], inference_scale: float, target_hw: Tuple[int, int]
) -> Dict[str, Any]:
    """Map masks and boxes produced at ``inference_scale`` back to full resolution."""
    height, width = target_hw
    masks = landmarks["masks"]
    if masks.numel():
        resized = F.interpolate(
            masks.to(torch.float32).unsqueeze(1), size=(height, width), mode="nearest"
        )
        landmarks["masks"] = resized.squeeze(1).to(torch.bool)
    inv_scale = 1.0 / inference_scale
    landmarks["bboxes"] = landmarks["bboxes"] * inv_scale
    return landmarks


def find_rink_landmarks(
    image: Union[torch.Tensor, np.ndarray],
    config_file: str,
    checkpoint: str,
    device: Optional[torch.device] = None,
    score_thr: float = DEFAULT_SCORE_THRESH,
    inference_scale: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Load the landmark detector, run it on one frame, and free it again."""
    from mmdet.apis import init_detector

    if device is None:
        device = torch.device("cpu")

    orig_height = image_height(image)
    orig_width = image_width(image)
    infer_image = _as_numpy_image(image)

    if inference_scale is None:
        inference_scale = _auto_inference_scale(orig_width, orig_height)
        if inference_scale is not None:
            logger.info(
                "Running rink landmark inference at scale %.4f (%dx%d -> %dx%d) to stay "
                "within DEFAULT_MAX_INFERENCE_EDGE=%d.",
                inference_scale,
                orig_width,
                orig_height,
                round(orig_width * inference_scale),
                round(orig_height * inference_scale),
                DEFAULT_MAX_INFERENCE_EDGE,
            )

    if inference_scale and inference_scale != 1.0:
        new_width = max(1, int(round(orig_width * inference_scale)))
        new_height = max(1, int(round(orig_height * inference_scale)))
        interpolation = cv2.INTER_AREA if inference_scale < 1.0 else cv2.INTER_LINEAR
        infer_image = cv2.resize(infer_image, (new_width, new_height), interpolation=interpolation)

    if device.type == "cpu":
        logger.info("Looking for the painted rink landmarks, this may take awhile...")
    model = init_detector(config_file, checkpoint, device=device)
    try:
        landmarks = detect_rink_landmarks(infer_image, model=model, score_thr=score_thr)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if landmarks is None:
        return None
    if inference_scale and inference_scale != 1.0:
        landmarks = _rescale_landmarks(landmarks, inference_scale, (orig_height, orig_width))
    return landmarks


def configure_rink_landmarks(
    game_id: str,
    image: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
    score_thr: float = DEFAULT_SCORE_THRESH,
    inference_scale: Optional[float] = None,
    checkpoint: Optional[str] = None,
    model_config: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve the landmark model from the game config and run it on ``image``."""
    model_config_file, model_checkpoint = get_model_config(
        game_id=game_id, model_name="rink_landmarks_segm"
    )
    if model_config:
        model_config_file = model_config
    if checkpoint:
        model_checkpoint = checkpoint
    if not model_config_file or not model_checkpoint:
        logger.warning(
            "No rink landmark model configured (model.rink_landmarks_segm); skipping landmarks."
        )
        return None

    config_path = prepend_root_dir(model_config_file)
    checkpoint_path = prepend_root_dir(model_checkpoint)
    if "://" not in checkpoint_path and not os.path.exists(checkpoint_path):
        logger.warning(
            "Rink landmark checkpoint not found: %s. Train it with "
            "openmm/train_rink_landmarks.sh or pass --rink-landmarks-checkpoint.",
            checkpoint_path,
        )
        return None

    return find_rink_landmarks(
        image=image,
        config_file=config_path,
        checkpoint=checkpoint_path,
        device=device,
        score_thr=score_thr,
        inference_scale=inference_scale,
    )


def filter_landmarks(
    landmarks: Dict[str, Any],
    include: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDED_CLASSES,
) -> Optional[Dict[str, Any]]:
    """Keep only the named classes (case-insensitive). ``include`` wins over ``exclude``."""
    classes = landmarks["classes"]
    labels = landmarks["labels"]
    if not len(labels):
        return None

    if include:
        wanted = {str(name).strip().lower() for name in include}
        keep_ids = {i for i, name in enumerate(classes) if str(name).lower() in wanted}
        unknown = wanted - {str(classes[i]).lower() for i in keep_ids}
        if unknown:
            logger.warning(
                "Unknown rink landmark class(es) %s; known classes are %s",
                sorted(unknown),
                list(classes),
            )
    else:
        dropped = {str(name).strip().lower() for name in (exclude or ())}
        keep_ids = {i for i, name in enumerate(classes) if str(name).lower() not in dropped}

    keep = torch.tensor([int(label) in keep_ids for label in labels], dtype=torch.bool)
    if not bool(keep.any()):
        return None

    filtered = dict(landmarks)
    filtered["labels"] = labels[keep]
    filtered["scores"] = landmarks["scores"][keep]
    filtered["bboxes"] = landmarks["bboxes"][keep]
    filtered["masks"] = landmarks["masks"][keep]
    return filtered


def _mask_outline(mask_np: np.ndarray, thickness: int) -> np.ndarray:
    """Return a uint8 edge mask for one instance, drawn ``thickness`` pixels wide."""
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = np.zeros_like(mask_np)
    if contours:
        cv2.drawContours(edges, contours, -1, color=1, thickness=max(1, int(thickness)))
    return edges


def build_overlay_layer(
    height: int,
    width: int,
    landmarks: Optional[Dict[str, Any]] = None,
    rink_mask: Optional[torch.Tensor] = None,
    rink_mask_color: Tuple[int, int, int] = (0, 255, 0),
    rink_mask_alpha: float = 0.10,
    fill_alpha: float = 0.35,
    outline_alpha: float = 0.95,
    outline_thickness: int = 3,
    label_text: bool = False,
    bgr: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Bake masks into one ``(color[3,H,W] uint8, alpha[1,H,W] float32)`` pair.

    Later marks win: the rink tint goes down first, then landmark fills, then
    landmark outlines on top, so the thin classes stay legible over the fills.

    The checkpoint's palette is RGB while frames here are BGR end to end (the
    mmdet preprocessor does the ``bgr_to_rgb`` flip itself), so ``bgr`` swaps
    the palette rather than the much larger image.
    """
    color = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.float32)
    drew_anything = False

    def channel_order(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (rgb[2], rgb[1], rgb[0]) if bgr else rgb

    if rink_mask is not None and rink_mask_alpha > 0:
        mask_np = rink_mask.detach().to(torch.bool).cpu().numpy()
        if mask_np.shape != (height, width):
            logger.warning(
                "Rink mask is %s but the overlay plane is %s; skipping the rink tint.",
                mask_np.shape,
                (height, width),
            )
        else:
            color[mask_np] = channel_order(rink_mask_color)
            alpha[mask_np] = float(rink_mask_alpha)
            drew_anything = True

    if landmarks is not None and len(landmarks["labels"]):
        classes = landmarks["classes"]
        palette = landmarks["palette"]
        masks = landmarks["masks"]
        labels = landmarks["labels"]
        scores = landmarks["scores"]
        outlines: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

        for mask, label in zip(masks, labels):
            mask_np = mask.numpy().astype(np.uint8)
            if mask_np.shape != (height, width):
                logger.warning(
                    "Landmark mask is %s but the overlay plane is %s; skipping it.",
                    mask_np.shape,
                    (height, width),
                )
                continue
            index = int(label)
            rgb = channel_order(palette[index] if index < len(palette) else (255, 255, 255))
            selected = mask_np.astype(bool)
            if fill_alpha > 0:
                color[selected] = rgb
                alpha[selected] = float(fill_alpha)
            if outline_thickness > 0:
                outlines.append((_mask_outline(mask_np, outline_thickness), rgb))
            drew_anything = True

        for edges, rgb in outlines:
            selected = edges.astype(bool)
            color[selected] = rgb
            alpha[selected] = float(outline_alpha)

        if label_text:
            _draw_labels(color, alpha, landmarks, classes, palette, labels, scores, channel_order)

    if not drew_anything:
        return None

    color_t = torch.from_numpy(color).permute(2, 0, 1).contiguous()
    alpha_t = torch.from_numpy(alpha).unsqueeze(0).contiguous()
    return color_t, alpha_t


def _draw_labels(
    color: np.ndarray,
    alpha: np.ndarray,
    landmarks: Dict[str, Any],
    classes: Sequence[str],
    palette: Sequence[Tuple[int, int, int]],
    labels: torch.Tensor,
    scores: torch.Tensor,
    channel_order,
) -> None:
    # Scale text with the panorama so it stays readable on a 5k-wide frame.
    font_scale = max(0.5, color.shape[1] / 2400.0)
    thickness = max(1, int(round(font_scale * 2)))
    for bbox, label, score in zip(landmarks["bboxes"], labels, scores):
        index = int(label)
        name = classes[index] if index < len(classes) else str(index)
        rgb = channel_order(palette[index] if index < len(palette) else (255, 255, 255))
        x = int(bbox[0])
        y = max(int(bbox[1]) - 4, int(round(16 * font_scale)))
        text = f"{name} {float(score):.2f}"
        text_layer = np.zeros(color.shape[:2], dtype=np.uint8)
        cv2.putText(
            text_layer,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color=1,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        selected = text_layer.astype(bool)
        color[selected] = rgb
        alpha[selected] = 1.0


def composite_overlay(
    img: torch.Tensor, color_layer: torch.Tensor, alpha_layer: torch.Tensor
) -> torch.Tensor:
    """Alpha-blend a prebuilt overlay onto ``img``, preserving its layout and dtype."""
    was_channels_first = is_channels_first(img)
    work = make_channels_first(img)
    original_dtype = work.dtype

    color = color_layer.to(device=work.device)
    alpha = alpha_layer.to(device=work.device)
    if work.ndim == 4:
        color = color.unsqueeze(0)
        alpha = alpha.unsqueeze(0)

    blended = work.to(torch.float32) * (1.0 - alpha) + color.to(torch.float32) * alpha
    if original_dtype == torch.uint8:
        blended = blended.round_().clamp_(0, 255)
    elif not torch.is_floating_point(torch.empty(0, dtype=original_dtype)):
        blended = blended.round_()
    blended = blended.to(original_dtype)

    return blended if was_channels_first else make_channels_last(blended)


def summarize_landmarks(landmarks: Optional[Dict[str, Any]]) -> str:
    """One-line human-readable census of what was detected."""
    if landmarks is None or not len(landmarks["labels"]):
        return "no rink landmarks detected"
    classes = landmarks["classes"]
    counts: Dict[str, int] = {}
    for label in landmarks["labels"]:
        index = int(label)
        name = classes[index] if index < len(classes) else str(index)
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{name} x{count}" for name, count in sorted(counts.items())]
    return f"{len(landmarks['labels'])} landmarks: " + ", ".join(parts)


def main(args: argparse.Namespace = None) -> None:
    """Render the rink mask and landmarks onto one stitched frame and save it."""
    from hmlib.config import get_game_dir
    from hmlib.segm.ice_rink import configure_ice_rink_mask

    if args is None:
        parser = argparse.ArgumentParser(
            description="Overlay the ice rink mask and painted rink landmarks on one frame"
        )
        parser.add_argument("--game-id", "-g", type=str, required=True, help="Game ID to process")
        parser.add_argument(
            "--image",
            type=str,
            default=None,
            help="Frame to annotate; defaults to <game_dir>/s.png (written by stitch --configure-only)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Where to write the annotated frame; defaults to <game_dir>/rink_landmarks.png",
        )
        parser.add_argument("--show", action="store_true", help="Display the annotated frame")
        parser.add_argument(
            "--device", "-d", type=str, default=None, help="Device used for inference"
        )
        parser.add_argument(
            "--score-thr",
            type=float,
            default=DEFAULT_SCORE_THRESH,
            help="Minimum landmark instance score",
        )
        parser.add_argument(
            "--scale",
            "-s",
            type=float,
            default=None,
            help="Downscale factor applied before landmark inference",
        )
        parser.add_argument(
            "--classes",
            type=str,
            default=None,
            help=f"Comma-separated class allow-list; default excludes {', '.join(DEFAULT_EXCLUDED_CLASSES)}",
        )
        parser.add_argument(
            "--checkpoint", "-c", type=str, default=None, help="Override the landmark checkpoint"
        )
        parser.add_argument(
            "--model-config",
            "-m",
            type=str,
            default=None,
            help="Override the landmark mmdet config",
        )
        parser.add_argument(
            "--no-rink-mask", action="store_true", help="Skip the ice rink mask tint"
        )
        parser.add_argument("--no-labels", action="store_true", help="Skip per-instance class text")
        parser.add_argument(
            "--fill-alpha", type=float, default=0.35, help="Opacity of the landmark fills"
        )
        parser.add_argument(
            "--outline-thickness", type=int, default=3, help="Landmark outline width in pixels"
        )
        args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cpu")
    game_dir = get_game_dir(game_id=args.game_id)
    if not game_dir:
        raise AttributeError(f"Could not determine game dir for game_id={args.game_id}")

    image_path = args.image or str(Path(game_dir) / "s.png")
    if not os.path.exists(image_path):
        print(f"Could not find stitched frame image: {image_path}")
        print("Generate one with: ./stitch.sh --configure-only")
        return 1

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read frame: {image_path}")
    height, width = frame.shape[:2]
    print(f"Annotating {image_path} ({width} x {height})")

    rink_mask = None
    if not args.no_rink_mask:
        rink_profile = configure_ice_rink_mask(
            game_id=args.game_id,
            device=device,
            expected_shape=torch.Size((height, width)),
            image=frame,
        )
        rink_mask = (rink_profile or {}).get("combined_mask")
        print("rink mask: " + ("found" if rink_mask is not None else "unavailable"))

    landmarks = configure_rink_landmarks(
        game_id=args.game_id,
        image=frame,
        device=device,
        score_thr=args.score_thr,
        inference_scale=args.scale,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
    )
    print(summarize_landmarks(landmarks))
    if landmarks is not None:
        include = [name for name in (args.classes or "").split(",") if name.strip()]
        landmarks = filter_landmarks(landmarks, include=include or None)
        if include:
            print("after class filter: " + summarize_landmarks(landmarks))

    layer = build_overlay_layer(
        height=height,
        width=width,
        landmarks=landmarks,
        rink_mask=rink_mask,
        fill_alpha=args.fill_alpha,
        outline_thickness=args.outline_thickness,
        label_text=not args.no_labels,
    )
    if layer is None:
        print("Nothing to draw.")
        return 1

    color_layer, alpha_layer = layer
    annotated = composite_overlay(torch.from_numpy(frame), color_layer, alpha_layer)

    output_path = args.output or str(Path(game_dir) / "rink_landmarks.png")
    cv2.imwrite(output_path, annotated.numpy())
    print(f"Wrote {output_path}")

    if args.show:
        from hmlib.ui import show_image as do_show_image

        do_show_image("Rink landmarks", annotated.numpy(), wait=True)
    return 0


if __name__ == "__main__":
    main()

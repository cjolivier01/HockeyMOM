import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import time_to_frame


@dataclass
class Interval:
    start_frame: int
    end_frame: int


def _parse_time_or_frames(
    info: BasicVideoInfo, start: Optional[str], duration: Optional[str], start_frame: Optional[int], duration_frames: Optional[int]
) -> Interval:
    fps = float(info.fps)
    total_frames = int(info.frame_count)
    sf = 0
    ef = total_frames - 1
    if start is not None:
        sf = int(max(0, time_to_frame(start, fps)))
    if start_frame is not None:
        sf = int(max(0, start_frame))
    if duration is not None:
        df = int(max(0, time_to_frame(duration, fps)))
        ef = int(min(total_frames - 1, sf + df - 1))
    if duration_frames is not None:
        ef = int(min(total_frames - 1, sf + int(duration_frames) - 1))
    return Interval(start_frame=sf, end_frame=ef)


def _ensure_cam_df(
    info: BasicVideoInfo,
    input_camera_csv: Optional[str],
    default_h_ratio: float,
    default_aspect: Optional[float],
) -> pd.DataFrame:
    """Return a camera DataFrame with columns [Frame,BBox_X,BBox_Y,BBox_W,BBox_H].

    If input_camera_csv is provided (and exists), copy and use it as a baseline.
    Otherwise, create a baseline covering the whole video with -1 entries and default W/H.
    """
    width = float(info.width)
    height = float(info.height)
    aspect = default_aspect if default_aspect and default_aspect > 0 else width / height
    frames = np.arange(1, int(info.frame_count) + 1, dtype=np.int64)
    if input_camera_csv and os.path.exists(input_camera_csv):
        cams = pd.read_csv(input_camera_csv, header=None)
        # Normalize to our expected schema
        cams.columns = ["Frame", "BBox_X", "BBox_Y", "BBox_W", "BBox_H"]
        # Ensure frames are ints
        cams["Frame"] = cams["Frame"].astype(np.int64)
        # If CSV is shorter/longer than video, don't force adjust; we will use CSV as canonical.
        return cams.copy()
    # Create default baseline for entire video
    # If no baseline, prefer a 16:9 box height equal to full video height by default
    # (with later dynamic shrink/clamp applied during annotation)
    h = float(height)
    w = float(round(h * (16.0 / 9.0)))
    # Clamp to video bounds
    w = max(1.0, min(w, width))
    h = max(1.0, min(h, height))
    x0 = -1.0
    y0 = -1.0
    data = {
        "Frame": frames,
        "BBox_X": np.full_like(frames, fill_value=x0, dtype=np.float64),
        "BBox_Y": np.full_like(frames, fill_value=y0, dtype=np.float64),
        "BBox_W": np.full_like(frames, fill_value=w, dtype=np.float64),
        "BBox_H": np.full_like(frames, fill_value=h, dtype=np.float64),
    }
    return pd.DataFrame(data)


def _fit_box_within_frame(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int, min_scale: float = 0.25
) -> Tuple[float, float, float, float]:
    """Return a TLWH box centered at (cx,cy) scaled down as needed so that it fits the frame.

    - Attempts to fit by uniformly shrinking up to min_scale.
    - If still outside, returns a clamped TLWH at min scale bounded to the frame edges.
    """
    # desired half-sizes
    hx = w / 2.0
    hy = h / 2.0
    # how much room do we have from center to each edge
    room_left = cx
    room_right = img_w - cx
    room_top = cy
    room_bottom = img_h - cy
    if room_left < 0 or room_right < 0 or room_top < 0 or room_bottom < 0:
        # if center is outside, clamp it before proceeding
        cx = float(np.clip(cx, 0.0, img_w - 1.0))
        cy = float(np.clip(cy, 0.0, img_h - 1.0))
        room_left = cx
        room_right = img_w - cx
        room_top = cy
        room_bottom = img_h - cy
    sx = min(1.0, room_left / max(1e-6, hx), room_right / max(1e-6, hx))
    sy = min(1.0, room_top / max(1e-6, hy), room_bottom / max(1e-6, hy))
    s = max(min(sx, sy), min_scale)
    w2 = w * s
    h2 = h * s
    x = cx - w2 / 2.0
    y = cy - h2 / 2.0
    # If still outside (due to min_scale), clamp
    x = float(np.clip(x, 0.0, max(0.0, img_w - w2)))
    y = float(np.clip(y, 0.0, max(0.0, img_h - h2)))
    return x, y, w2, h2


def _draw_center_box(img: np.ndarray, cx: int, cy: int, w: int, h: int, color=(0, 255, 0)) -> None:
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.circle(img, (int(cx), int(cy)), 4, color, -1)


def _put_text(img: np.ndarray, text: str, org=(10, 24)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 240), 2, cv2.LINE_AA)


def annotate(
    video: str,
    input_camera_csv: Optional[str],
    output_camera_csv: str,
    start: Optional[str],
    duration: Optional[str],
    start_frame: Optional[int],
    duration_frames: Optional[int],
    save_every: int,
    default_h_ratio: float,
    default_aspect: Optional[float],
    playback_speed: float,
    show_overlay: bool,
    lock_baseline_height: bool,
) -> None:
    info = BasicVideoInfo(video, use_ffprobe=True)
    interval = _parse_time_or_frames(info, start, duration, start_frame, duration_frames)
    cams = _ensure_cam_df(info, input_camera_csv, default_h_ratio, default_aspect)
    # Build a mapping from frame->index (first occurrence per frame)
    if cams.empty:
        raise RuntimeError("Camera CSV baseline is empty; cannot proceed.")
    frame_to_idx: Dict[int, int] = {}
    baseline_idx_map: Dict[int, int] = {}
    for idx, row in cams.reset_index().iterrows():
        f = int(row["Frame"]) if "Frame" in row else int(row[1])
        # only record first index for frame if duplicates exist
        frame_to_idx.setdefault(f, idx)
        baseline_idx_map.setdefault(f, idx)

    # Prepare output baseline as a copy we will update in-place
    out_df = cams.copy()
    # If baseline doesn't have a row for a frame in the interval, create it so we cover the interval.
    # Use defaults for W/H and -1 center until overridden.
    width = int(info.width)
    height = int(info.height)
    # Defaults when no baseline: 16:9 with video height by default
    # With baseline, use height ratio but enforce 16:9 aspect
    if input_camera_csv and os.path.exists(input_camera_csv):
        default_h = int(height * default_h_ratio)
        default_w = int(round(default_h * (16.0 / 9.0)))
    else:
        # Use 16:9 box with full video height
        default_h = int(height)
        default_w = int(round(default_h * (16.0 / 9.0)))

    # Mouse state
    mouse_down = False
    mouse_xy: Tuple[int, int] = (width // 2, height // 2)

    # dynamic zoom (affects current frame while annotating). Always enforce 16:9.
    cur_h = float(default_h)

    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_down, mouse_xy
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            mouse_xy = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_down:
                mouse_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, interval.start_frame)
    cv2.namedWindow("Camera Annotator", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback("Camera Annotator", _on_mouse)

    processed_since_save = 0
    current_frame_abs = interval.start_frame  # 0-based for cap
    # Convert to CSV's expected 1-based frame numbers
    first_csv_frame = current_frame_abs + 1
    last_csv_frame = interval.end_frame + 1

    # Ensure all frames in [first_csv_frame, last_csv_frame] exist in out_df
    for f in range(first_csv_frame, last_csv_frame + 1):
        if f not in frame_to_idx:
            # Append a new row
            out_df.loc[len(out_df)] = [
                int(f),
                float(-1.0),
                float(-1.0),
                float(default_w),
                float(default_h),
            ]
            frame_to_idx[f] = int(len(out_df) - 1)

    # compute per-frame delay based on playback_speed
    fps = float(info.fps)
    delay_ms = max(1, int(round((1000.0 / max(1e-6, fps)) / max(1e-6, playback_speed))))
    # show overlay toggle
    overlay_enabled = bool(show_overlay)

    try:
        while current_frame_abs <= interval.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            csv_frame = current_frame_abs + 1
            # Determine w/h for this frame from baseline or defaults
            idx = frame_to_idx.get(csv_frame)
            h = default_h
            w = int(round(h * (16.0 / 9.0)))
            if idx is not None:
                try:
                    h = int(max(1, float(out_df.iloc[idx]["BBox_H"])))
                    w = int(round(h * (16.0 / 9.0)))
                except Exception:
                    pass
            # Initialize current adjustable size from baseline/default each frame
            cur_h = float(h)

            # Draw current box preview at mouse position
            # Always draw baseline overlay (original baseline CSV values) if available and enabled
            if overlay_enabled and (csv_frame in baseline_idx_map):
                try:
                    bidx = baseline_idx_map[csv_frame]
                    bx = float(cams.iloc[bidx]["BBox_X"])  # tlwh x (original baseline)
                    by = float(cams.iloc[bidx]["BBox_Y"])  # tlwh y
                    bw = float(cams.iloc[bidx]["BBox_W"])  # tlwh w
                    bh = float(cams.iloc[bidx]["BBox_H"])  # tlwh h
                    _draw_center_box(
                        disp,
                        int(bx + bw / 2.0),
                        int(by + bh / 2.0),
                        int(bw),
                        int(bh),
                        color=(0, 180, 255),
                    )
                except Exception:
                    pass

            if mouse_down:
                cx, cy = mouse_xy
                cur_w = float(round(cur_h * (16.0 / 9.0)))
                fx, fy, fw, fh = _fit_box_within_frame(cx, cy, cur_w, cur_h, width, height, min_scale=0.25)
                _draw_center_box(disp, int(fx + fw / 2.0), int(fy + fh / 2.0), int(fw), int(fh), color=(0, 255, 0))

            locked_text = " [locked]" if lock_baseline_height else ""
            _put_text(
                disp,
                f"Frame {csv_frame}  LMB=set center  +/-=zoom{locked_text}  b=back  s=save  o=overlay  ESC=exit (16:9)",
                org=(10, 24),
            )
            _put_text(
                disp,
                "Green=override, Orange=baseline; Space=pause; Playback speed set via --playback-speed",
                org=(10, 48),
            )

            cv2.imshow("Camera Annotator", disp)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == 27 or key == ord("q"):
                # ESC or q: exit
                break
            if key == ord("s"):
                out_df.sort_values("Frame", inplace=True)
                out_df.to_csv(output_camera_csv, header=False, index=False)
                processed_since_save = 0
            if key == ord("o"):
                overlay_enabled = not overlay_enabled
            if key == ord(" "):
                # pause until space pressed again
                while True:
                    key2 = cv2.waitKey(10) & 0xFF
                    if key2 == ord(" ") or key2 == 27 or key2 == ord("q"):
                        key = key2
                        break
                if key == 27 or key == ord("q"):
                    break
            # Step back one frame
            if key == ord("b"):
                current_frame_abs = max(interval.start_frame, current_frame_abs - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_abs)
                continue

            # Zoom adjustments (enforce 16:9 via width=height*16/9). Disabled when locked.
            if not lock_baseline_height:
                if key in (ord("+"), ord("=")):
                    cur_h *= 1.05
                elif key == ord("-"):
                    cur_h *= 0.95
                elif key == ord("r"):
                    cur_h = float(h)

            # Update output for this frame if mouse was down
            if mouse_down:
                cx, cy = mouse_xy
                # choose save size
                save_h = cur_h
                if lock_baseline_height:
                    base_h = None
                    if csv_frame in baseline_idx_map:
                        try:
                            base_h = float(cams.iloc[baseline_idx_map[csv_frame]]["BBox_H"])  # original baseline
                        except Exception:
                            base_h = None
                    if base_h is None or base_h <= 0:
                        # fallback to existing out_df row or default
                        try:
                            base_h = float(out_df.iloc[idx]["BBox_H"]) if idx is not None else float(h)
                        except Exception:
                            base_h = float(h)
                    save_h = float(base_h)
                cur_w = float(round(save_h * (16.0 / 9.0)))
                x, y, w_saved, h_saved = _fit_box_within_frame(
                    float(cx), float(cy), float(cur_w), float(save_h), width, height, min_scale=0.25
                )
                # Write into out_df
                idx = frame_to_idx[csv_frame]
                out_df.at[idx, "BBox_X"] = float(x)
                out_df.at[idx, "BBox_Y"] = float(y)
                out_df.at[idx, "BBox_W"] = float(w_saved)
                out_df.at[idx, "BBox_H"] = float(h_saved)
            else:
                # If no baseline was provided, ensure we mark center as -1
                if not (input_camera_csv and os.path.exists(input_camera_csv)):
                    idx = frame_to_idx[csv_frame]
                    out_df.at[idx, "BBox_X"] = -1.0
                    out_df.at[idx, "BBox_Y"] = -1.0

            processed_since_save += 1
            current_frame_abs += 1

            # Periodic save
            if processed_since_save >= max(1, save_every):
                out_df.sort_values("Frame", inplace=True)
                out_df.to_csv(output_camera_csv, header=False, index=False)
                processed_since_save = 0
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Finalize: ensure rows cover entire baseline and are sorted
    out_df.sort_values("Frame", inplace=True)
    out_df.to_csv(output_camera_csv, header=False, index=False)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Annotate camera centers over a video interval")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--input-camera-csv", default=None, help="Existing camera.csv to use as baseline")
    ap.add_argument(
        "--output-camera-csv",
        default=None,
        help="Output CSV (defaults to baseline dir with _annotated suffix or alongside video)",
    )
    ap.add_argument("--start", default=None, help="Start time (e.g. 90 or 00:01:30.0)")
    ap.add_argument("--duration", default=None, help="Duration (e.g. 30 or 00:00:30.0)")
    ap.add_argument("--start-frame", type=int, default=None, help="Start at this 0-based frame index")
    ap.add_argument("--duration-frames", type=int, default=None, help="Duration in frames")
    ap.add_argument("--save-every", type=int, default=120, help="Periodic save frequency in frames")
    ap.add_argument(
        "--default-h-ratio",
        type=float,
        default=0.6,
        help="If no baseline CSV, default camera box height as ratio of video height",
    )
    ap.add_argument(
        "--default-aspect",
        type=float,
        default=None,
        help="Default aspect ratio (width/height). If no baseline, a 16:9 box with full video height is used regardless",
    )
    ap.add_argument("--playback-speed", type=float, default=1.0, help="Playback speed factor (e.g., 0.5 for half-speed)")
    ap.add_argument("--overlay", dest="overlay", action="store_true", help="Show baseline overlay while playing")
    ap.add_argument(
        "--lock-baseline-height",
        dest="lock_baseline_height",
        action="store_true",
        help="When set, saves overrides using the original baseline height per frame (still 16:9). Zoom keys are disabled",
    )
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)

    video_path = args.video
    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    out_csv = args.output_camera_csv
    if out_csv is None:
        # Default: if baseline provided, write alongside it; otherwise alongside video
        if args.input_camera_csv:
            base = args.input_camera_csv
            root, ext = os.path.splitext(base)
            out_csv = f"{root}_annotated{ext or '.csv'}"
        else:
            vroot, _ = os.path.splitext(video_path)
            out_csv = f"{vroot}_camera_annotated.csv"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    annotate(
        video=video_path,
        input_camera_csv=args.input_camera_csv,
        output_camera_csv=out_csv,
        start=args.start,
        duration=args.duration,
        start_frame=args.start_frame,
        duration_frames=args.duration_frames,
        save_every=int(args.save_every),
        default_h_ratio=float(args.default_h_ratio),
        default_aspect=args.default_aspect if args.default_aspect else None,
        playback_speed=float(args.playback_speed),
        show_overlay=bool(args.overlay),
        lock_baseline_height=bool(args.lock_baseline_height),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

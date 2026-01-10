from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from hmlib.datasets.dataframe import find_latest_dataframe_file


@dataclass(frozen=True)
class _CameraSeries:
    tlwh_by_frame: dict[int, np.ndarray]


def _read_camera_csv(path: str) -> _CameraSeries:
    df = pd.read_csv(path, header=None)
    df.columns = ["Frame", "BBox_X", "BBox_Y", "BBox_W", "BBox_H"]
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce").astype(np.int64)
    tlwh_by_frame: dict[int, np.ndarray] = {}
    for row in df.itertuples(index=False):
        fid = int(getattr(row, "Frame"))
        tlwh = np.asarray(
            [
                float(getattr(row, "BBox_X")),
                float(getattr(row, "BBox_Y")),
                float(getattr(row, "BBox_W")),
                float(getattr(row, "BBox_H")),
            ],
            dtype=np.float32,
        )
        tlwh_by_frame[fid] = tlwh
    return _CameraSeries(tlwh_by_frame=tlwh_by_frame)


def _tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = (float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3]))
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def _iou_tlbr(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a.tolist())
    bx1, by1, bx2, by2 = map(float, b.tolist())
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center_err_px(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = map(float, a.tolist())
    bx, by, bw, bh = map(float, b.tolist())
    acx = ax + aw * 0.5
    acy = ay + ah * 0.5
    bcx = bx + bw * 0.5
    bcy = by + bh * 0.5
    dx = acx - bcx
    dy = acy - bcy
    return float(np.sqrt(dx * dx + dy * dy))


def _resolve_camera_csv(dir_path: Optional[str], stem: str, explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        return explicit_path
    if not dir_path:
        return None
    return find_latest_dataframe_file(dir_path, stem)


def _compute_metrics(a: _CameraSeries, b: _CameraSeries) -> Tuple[int, float, float]:
    frames = sorted(set(a.tlwh_by_frame.keys()).intersection(b.tlwh_by_frame.keys()))
    if not frames:
        return 0, float("nan"), float("nan")
    ious = []
    centers = []
    for fid in frames:
        ta = a.tlwh_by_frame[fid]
        tb = b.tlwh_by_frame[fid]
        ious.append(_iou_tlbr(_tlwh_to_tlbr(ta), _tlwh_to_tlbr(tb)))
        centers.append(_center_err_px(ta, tb))
    return len(frames), float(np.mean(ious)), float(np.mean(centers))


def _draw_box(img: np.ndarray, tlwh: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
    x, y, w, h = map(float, tlwh.tolist())
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, p1, p2, color, thickness)


def main() -> None:
    ap = argparse.ArgumentParser("Compare camera.csv trajectories (rule vs model)")
    ap.add_argument("--a-dir", type=str, default=None, help="Directory containing camera.csv (run A)")
    ap.add_argument("--b-dir", type=str, default=None, help="Directory containing camera.csv (run B)")
    ap.add_argument("--a-camera", type=str, default=None, help="Explicit camera.csv path for run A")
    ap.add_argument("--b-camera", type=str, default=None, help="Explicit camera.csv path for run B")
    ap.add_argument("--a-camera-fast", type=str, default=None, help="Explicit camera_fast.csv path for run A")
    ap.add_argument("--b-camera-fast", type=str, default=None, help="Explicit camera_fast.csv path for run B")
    ap.add_argument("--video", type=str, default=None, help="Optional video file to overlay both cameras on")
    ap.add_argument("--out-video", type=str, default=None, help="Optional output video path for overlay")
    ap.add_argument("--start-frame", type=int, default=1, help="1-based start frame for overlay")
    ap.add_argument("--num-frames", type=int, default=300, help="Number of frames to render for overlay (0=all)")
    ap.add_argument("--draw-fast", action="store_true", help="Also overlay camera_fast.csv when present")
    ap.add_argument(
        "--max-out-width",
        type=int,
        default=1920,
        help="Maximum overlay video width (auto-downscales large inputs).",
    )
    ap.add_argument(
        "--max-out-height",
        type=int,
        default=1080,
        help="Maximum overlay video height (auto-downscales large inputs).",
    )
    args = ap.parse_args()

    a_cam = _resolve_camera_csv(args.a_dir, "camera", args.a_camera)
    b_cam = _resolve_camera_csv(args.b_dir, "camera", args.b_camera)
    if not a_cam or not b_cam:
        raise SystemExit("Missing camera.csv; provide --a-dir/--b-dir or --a-camera/--b-camera")

    a = _read_camera_csv(a_cam)
    b = _read_camera_csv(b_cam)
    n, mean_iou, mean_center = _compute_metrics(a, b)
    print(f"slow: frames={n} mean_iou={mean_iou:.4f} mean_center_err_px={mean_center:.2f}")

    a_fast = _resolve_camera_csv(args.a_dir, "camera_fast", args.a_camera_fast)
    b_fast = _resolve_camera_csv(args.b_dir, "camera_fast", args.b_camera_fast)
    a_fast_s = _read_camera_csv(a_fast) if a_fast else None
    b_fast_s = _read_camera_csv(b_fast) if b_fast else None
    if a_fast_s is not None and b_fast_s is not None:
        n2, mean_iou2, mean_center2 = _compute_metrics(a_fast_s, b_fast_s)
        print(f"fast: frames={n2} mean_iou={mean_iou2:.4f} mean_center_err_px={mean_center2:.2f}")

    if args.video and args.out_video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Unable to open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_w = int(args.max_out_width)
        max_h = int(args.max_out_height)
        if max_w <= 0:
            max_w = width
        if max_h <= 0:
            max_h = height
        scale_x = min(1.0, float(max_w) / float(max(1, width)))
        scale_y = min(1.0, float(max_h) / float(max(1, height)))
        scale = float(min(scale_x, scale_y))
        out_w = int(round(width * scale))
        out_h = int(round(height * scale))
        # Many encoders require even dimensions.
        out_w = max(2, out_w - (out_w % 2))
        out_h = max(2, out_h - (out_h % 2))
        scale_x = float(out_w) / float(max(1, width))
        scale_y = float(out_h) / float(max(1, height))

        os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, float(fps), (out_w, out_h))
        if not writer.isOpened():
            raise SystemExit(f"Unable to open VideoWriter: {args.out_video}")

        start0 = max(0, int(args.start_frame) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start0)
        limit = int(args.num_frames)
        count = 0
        while True:
            if limit > 0 and count >= limit:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if scale_x != 1.0 or scale_y != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            fid = start0 + count + 1  # 1-based
            if fid in a.tlwh_by_frame:
                tlwh = a.tlwh_by_frame[fid].copy()
                tlwh[0] *= scale_x
                tlwh[1] *= scale_y
                tlwh[2] *= scale_x
                tlwh[3] *= scale_y
                _draw_box(frame, tlwh, (255, 0, 0), 2)
            if fid in b.tlwh_by_frame:
                tlwh = b.tlwh_by_frame[fid].copy()
                tlwh[0] *= scale_x
                tlwh[1] *= scale_y
                tlwh[2] *= scale_x
                tlwh[3] *= scale_y
                _draw_box(frame, tlwh, (0, 255, 0), 2)
            if args.draw_fast and a_fast_s is not None and fid in a_fast_s.tlwh_by_frame:
                tlwh = a_fast_s.tlwh_by_frame[fid].copy()
                tlwh[0] *= scale_x
                tlwh[1] *= scale_y
                tlwh[2] *= scale_x
                tlwh[3] *= scale_y
                _draw_box(frame, tlwh, (255, 128, 64), 1)
            if args.draw_fast and b_fast_s is not None and fid in b_fast_s.tlwh_by_frame:
                tlwh = b_fast_s.tlwh_by_frame[fid].copy()
                tlwh[0] *= scale_x
                tlwh[1] *= scale_y
                tlwh[2] *= scale_x
                tlwh[3] *= scale_y
                _draw_box(frame, tlwh, (0, 255, 255), 1)
            cv2.putText(
                frame,
                f"frame {fid}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            count += 1
        writer.release()
        cap.release()
        print(f"Wrote overlay video: {args.out_video}")


if __name__ == "__main__":
    main()

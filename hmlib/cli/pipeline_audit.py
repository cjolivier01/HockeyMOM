from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [tok.strip() for tok in str(value).split(",") if tok.strip()]


def _run(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def _compare_video_to_dump(video_path: Path, dump_dir: Path, *, max_mae: float) -> None:
    """Compare decoded video frames to dumped PNGs (lossy-aware).

    This helps detect encoder races where the written bitstream diverges from
    the frames passed into the encoder.
    """
    import cv2
    import numpy as np

    png_paths = sorted(dump_dir.glob("frame_*.png"))
    if not png_paths:
        raise RuntimeError(f"No PNG frames found under {dump_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    worst_mae = 0.0
    worst_path: Path | None = None
    for idx, png_path in enumerate(png_paths):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"Video ended early at decoded frame {idx} (expected {len(png_paths)} frames)"
            )
        ref = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if ref is None:
            raise RuntimeError(f"Failed to read PNG: {png_path}")
        if ref.shape != frame.shape:
            raise RuntimeError(
                f"Shape mismatch at {png_path.name}: png={ref.shape} video={frame.shape}"
            )
        mae = float(np.mean(np.abs(frame.astype(np.int16) - ref.astype(np.int16))))
        if mae > worst_mae:
            worst_mae = mae
            worst_path = png_path
        if mae > float(max_mae):
            raise RuntimeError(
                f"Encoded frame diverged from input beyond threshold: mae={mae:.3f} > {max_mae} "
                f"at {png_path}"
            )

    # Ensure the video doesn't contain extra frames beyond the dump.
    ok, _frame = cap.read()
    if ok:
        raise RuntimeError(
            f"Video has more frames than dump (dump={len(png_paths)}); check PTS/order in {video_path}"
        )

    if worst_path is not None:
        print(f"Roundtrip check OK. Worst MAE={worst_mae:.3f} at {worst_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "hm pipeline-audit",
        description=(
            "Run hmtrack twice (reference + stressed) while capturing per-plugin frame hashes "
            "to pinpoint CUDA stream / determinism issues."
        ),
    )
    parser.add_argument("--game-id", required=True)
    parser.add_argument("-s", "--start-time", dest="start_time", default=None)
    parser.add_argument("-t", "--max-time", dest="max_time", default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    parser.add_argument(
        "--audit-root",
        type=str,
        default=None,
        help="Root directory for audit outputs (default: ./audit_workdirs/<game-id>).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Number of stressed runs to execute (default: 3).",
    )
    parser.add_argument(
        "--plugins",
        type=str,
        default=None,
        help="Comma-separated Aspen plugin names to audit (default: all plugins).",
    )
    parser.add_argument(
        "--dump-images",
        action="store_true",
        help="Dump PNG images for all captured tensors (large).",
    )
    parser.add_argument(
        "--check-encoded",
        action="store_true",
        help=(
            "After each run, decode the output video and compare to dumped "
            "video_out input PNGs (requires --keep-video and --dump-images)."
        ),
    )
    parser.add_argument(
        "--max-encoded-mae",
        type=float,
        default=8.0,
        help="Max mean-absolute-error allowed in --check-encoded mode (default: 8.0).",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Do not pass --skip-final-video-save to hmtrack.",
    )
    parser.add_argument(
        "--stress-max-concurrent",
        type=int,
        default=6,
        help="aspen.pipeline.max_concurrent for stressed runs (default: 6).",
    )
    parser.add_argument(
        "--stress-queue-size",
        type=int,
        default=1,
        help="aspen.pipeline.queue_size for stressed runs (default: 1).",
    )
    parser.add_argument(
        "hmtrack_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to hmtrack (prefix with --).",
    )
    return parser


def _hmtrack_cmd(base_args: List[str], extra_args: List[str]) -> List[str]:
    cmd = [sys.executable, "-m", "hmlib.cli.hmtrack"]
    cmd.extend(base_args)
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args = make_parser().parse_args()
    game_id = args.game_id

    audit_root = Path(args.audit_root) if args.audit_root else Path("audit_workdirs") / game_id
    audit_root.mkdir(parents=True, exist_ok=True)
    ref_dir = audit_root / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    base_hmtrack_args: List[str] = ["--game-id", game_id]
    if args.start_time:
        base_hmtrack_args += ["-s", args.start_time]
    if args.max_time:
        base_hmtrack_args += ["-t", args.max_time]
    if args.max_frames is not None:
        base_hmtrack_args += ["--max-frames", str(args.max_frames)]

    # Avoid audio muxing during audits (keeps runs fast and deterministic).
    base_hmtrack_args += ["--no-audio"]

    # Avoid writing large output videos unless explicitly requested.
    if not args.keep_video:
        base_hmtrack_args += ["--skip-final-video-save"]

    plugin_csv = args.plugins
    if plugin_csv:
        base_hmtrack_args += ["--audit-plugins", plugin_csv]

    # Forward any extra args after "--" to hmtrack.
    passthrough = list(args.hmtrack_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # Reference run: force synchronous / serial Aspen execution.
    ref_video = ref_dir / "tracking_output.mkv"
    ref_args = base_hmtrack_args + [
        "-o",
        str(ref_video),
        "--audit-dir",
        str(ref_dir),
        "--audit-dump-images" if args.dump_images else "",
        "--audit-fail-fast",
        "1",
        "--no-aspen-threaded",
        "--no-aspen-thread-cuda-streams",
        "--aspen-max-concurrent",
        "1",
        "--aspen-thread-queue-size",
        "1",
        "--no-cuda-streams",
        "--no-async-dataset",
        "--deterministic",
        "1",
    ]
    ref_args = [a for a in ref_args if a]
    _run(_hmtrack_cmd(ref_args, passthrough))
    if args.check_encoded and args.keep_video and args.dump_images:
        dump_dir = ref_dir / "images" / "before" / "video_out" / "img"
        _compare_video_to_dump(ref_video, dump_dir, max_mae=float(args.max_encoded_mae))

    # Stressed runs: keep threaded mode and bump concurrency.
    stress_base = base_hmtrack_args + [
        "--audit-reference-dir",
        str(ref_dir),
        "--audit-fail-fast",
        "1",
        "--aspen-threaded",
        "--aspen-thread-graph",
        "--aspen-thread-cuda-streams",
        "--aspen-max-concurrent",
        str(int(args.stress_max_concurrent)),
        "--aspen-thread-queue-size",
        str(int(args.stress_queue_size)),
    ]
    if args.dump_images:
        stress_base.append("--audit-dump-images")

    for i in range(int(args.iters)):
        run_dir = audit_root / f"run_{i+1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_video = run_dir / "tracking_output.mkv"
        run_args = stress_base + ["-o", str(run_video), "--audit-dir", str(run_dir)]
        _run(_hmtrack_cmd(run_args, passthrough))
        if args.check_encoded and args.keep_video and args.dump_images:
            dump_dir = run_dir / "images" / "before" / "video_out" / "img"
            _compare_video_to_dump(run_video, dump_dir, max_mae=float(args.max_encoded_mae))

    print(f"Audit complete. Outputs in {audit_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

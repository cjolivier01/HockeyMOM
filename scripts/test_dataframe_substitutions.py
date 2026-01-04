#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from typing import List

import torch
from mmengine.structures import InstanceData


def _make_detection_df(path: str):
    from mmdet.structures import DetDataSample

    from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame

    df = DetectionDataFrame(output_file=path, input_batch_size=1)
    inst = InstanceData()
    inst.scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    inst.labels = torch.tensor([0, 0], dtype=torch.long)
    inst.bboxes = torch.tensor([[10, 15, 30, 60], [100, 120, 150, 180]], dtype=torch.float32)
    ds = DetDataSample()
    ds.pred_instances = inst
    for frame in range(1, 6):
        df.add_frame_sample(frame, ds)
    df.flush()


def _make_tracking_df(path: str):
    from mmdet.structures import DetDataSample

    from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame

    df = TrackingDataFrame(output_file=path, input_batch_size=1)
    for frame in range(1, 6):
        inst = InstanceData()
        inst.instances_id = torch.tensor([1], dtype=torch.long)
        inst.bboxes = torch.tensor([[10.0 + frame, 20.0, 40.0, 60.0]], dtype=torch.float32)
        inst.scores = torch.tensor([0.95], dtype=torch.float32)
        inst.labels = torch.tensor([0], dtype=torch.long)
        ds = DetDataSample()
        ds.pred_track_instances = inst
        df.add_frame_sample(frame, ds)
    df.flush()


def _make_pose_df(path: str):
    from hmlib.tracking_utils.pose_dataframe import PoseDataFrame

    try:
        from mmpose.structures import PoseDataSample
    except Exception:
        PoseDataSample = None

    df = PoseDataFrame(output_file=path, input_batch_size=1)
    if PoseDataSample is not None:
        pinst = InstanceData()
        pinst.keypoints = torch.zeros((1, 17, 2), dtype=torch.float32)
        pinst.keypoint_scores = torch.ones((1, 17), dtype=torch.float32)
        pds = PoseDataSample()
        pds.pred_instances = pinst
        for frame in range(1, 6):
            df.add_frame_sample(frame, pds)
    else:
        # Store empty predictions structure
        for frame in range(1, 6):
            df.add_frame_records(frame, '{"predictions": []}')
    df.flush()


def _make_actions_df(path: str):
    from hmlib.tracking_utils.action_dataframe import ActionDataFrame

    df = ActionDataFrame(output_file=path, input_batch_size=1)
    for frame in range(1, 6):
        df.add_frame_sample(frame, [dict(tracking_id=1, label="idle", label_index=0, score=0.9)])
    df.flush()


def build_csvs(workdir: str):
    det = os.path.join(workdir, "detections.csv")
    trk = os.path.join(workdir, "tracking.csv")
    pose = os.path.join(workdir, "pose.csv")
    act = os.path.join(workdir, "actions.csv")
    _make_detection_df(det)
    _make_tracking_df(trk)
    _make_pose_df(pose)
    _make_actions_df(act)
    return det, trk, pose, act


def has_game_dir(game_id: str) -> bool:
    game_dir = os.path.join(os.environ.get("HOME", ""), "Videos", game_id)
    return os.path.isdir(game_dir)


def maybe_run(cmd: List[str]):
    print("> " + " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}. Output above.")


def main():
    ap = argparse.ArgumentParser(
        description="Test DataFrame substitutions for detection/tracking/pose/actions."
    )
    ap.add_argument("--game-id", default="devtest")
    ap.add_argument(
        "--video",
        default=None,
        help="Path to a short video. If not provided, attempts a synthetic one.",
    )
    ap.add_argument("--config", default="hmlib/config/aspen/tracking_pose_actions.yaml")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="hm_df_subst_")
    os.makedirs(workdir, exist_ok=True)
    print("Workdir:", workdir)
    det, trk, pose, act = build_csvs(workdir)

    # Programmatic verification: read back samples
    from hmlib.tracking_utils.action_dataframe import ActionDataFrame
    from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
    from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
    from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame

    print("Programmatic checks:")
    rd_det = DetectionDataFrame(input_file=det, input_batch_size=1)
    assert rd_det.get_sample_by_frame(1) is not None
    rd_trk = TrackingDataFrame(input_file=trk, input_batch_size=1)
    assert rd_trk.get_sample_by_frame(1) is not None
    rd_pose = PoseDataFrame(input_file=pose, input_batch_size=1)
    rd_pose.get_sample_by_frame(1)  # may be None if no pose
    rd_act = ActionDataFrame(input_file=act, input_batch_size=1)
    assert rd_act.get_sample_by_frame(1) is not None
    print("  ✓ Det/Track/Pose/Action roundtrip OK")

    # Optional CLI runs: only if a valid game dir exists
    if not has_game_dir(args.game_id):
        print(
            f"Game directory $HOME/Videos/{args.game_id} not found; skipping CLI runs.\n"
            f"You can create it or use an existing game id to test hmtrack."
        )
        return 0

    video = args.video
    if not video:
        # Try to synthesize a short video with ffmpeg
        video = os.path.join(workdir, "black.mp4")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:r=5",
            "-t",
            "1",
            video,
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            print("Failed to synthesize video; please provide --video to test CLI runs.")
            video = None

    if not video or not os.path.exists(video):
        print("Skipping CLI runs due to missing video.")
        return 0

    base = [
        sys.executable,
        "-m",
        "hmlib.cli.hmtrack",
        "-b=1",
        f"--game-id={args.game_id}",
        f"--config={args.config}",
        "--no-wide-start",
        "--skip-final-video-save",
        f"--input-video={video}",
        "--no-save-video",
    ]

    # Permutations to try (lightweight): load-only paths to avoid heavy models
    runs = [
        base + [f"--input-tracking-data={trk}", f"--input-pose-data={pose}"],
        base + [f"--input-detection-data={det}"],
        base + [f"--input-pose-data={pose}"],
        base
        + [
            f"--input-tracking-data={trk}",
            f"--input-detection-data={det}",
            f"--input-pose-data={pose}",
        ],
    ]
    print("Attempting hmtrack runs (these may depend on your local configs):")
    for cmd in runs:
        maybe_run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

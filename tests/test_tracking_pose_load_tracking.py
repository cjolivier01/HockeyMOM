import os
import subprocess
import sys
import tempfile


def should_cli_load_tracking_smoke_runs():
    import torch

    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame

    with tempfile.TemporaryDirectory(prefix="hm_cli_ci_") as td:
        # Prepare $HOME/Videos/<game_id>
        game_id = "devtest"
        home = os.path.join(td, "home")
        os.makedirs(os.path.join(home, "Videos", game_id), exist_ok=True)
        env = os.environ.copy()
        env["HOME"] = home
        # Ensure repo root is importable when we chdir to td
        env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH','')}"

        # Create a tiny tracking.csv (5 frames)
        tracking_csv = os.path.join(td, "tracking.csv")
        df = TrackingDataFrame(output_file=tracking_csv, input_batch_size=1)
        for frame in range(1, 6):
            inst = InstanceData()
            inst.instances_id = torch.tensor([1], dtype=torch.long)
            inst.bboxes = torch.tensor([[10.0 + frame, 20.0, 40.0, 60.0]], dtype=torch.float32)
            inst.scores = torch.tensor([0.9], dtype=torch.float32)
            inst.labels = torch.tensor([0], dtype=torch.long)
            ds = DetDataSample()
            ds.pred_track_instances = inst
            df.add_frame_sample(frame, ds)
        df.flush()

        # Create a 1-second black video at 5 fps
        video = os.path.join(td, "black.mp4")
        subprocess.check_call(
            [
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
        )

        # Run hmtrack with the load-tracking config; disable postprocess to avoid rink requirements
        cmd = [
            sys.executable,
            "-m",
            "hmlib.cli.hmtrack",
            "-b=1",
            f"--input-video={video}",
            f"--game-id={game_id}",
            "--config=hmlib/config/aspen/tracking_pose_load_tracking.yaml",
            "--no-wide-start",
            "--no-crop",
            "--skip-final-video-save",
            "--no-play-tracking",
            f"--input-tracking-data={tracking_csv}",
            "-t",
            "0.5",
            "--no-save-video",
            "--plot-pose",
            "--save-pose-data",
        ]
        subprocess.check_call(cmd, env=env)

        # Verify pose.csv was saved under repo output_workdirs/<game_id>
        out_pose_dir = os.path.join(REPO_ROOT, "output_workdirs", game_id)
        out_pose = os.path.join(out_pose_dir, "pose.csv")
        assert os.path.isfile(out_pose)
        from hmlib.tracking_utils.pose_dataframe import PoseDataFrame

        pose_df = PoseDataFrame(input_file=out_pose, input_batch_size=1)
        # Either empty or with entries; ensure call path works
        _ = pose_df.get_samples()
        # Cleanup outputs to keep repo tree tidy
        try:
            import shutil

            shutil.rmtree(out_pose_dir, ignore_errors=True)
        except Exception:
            pass


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

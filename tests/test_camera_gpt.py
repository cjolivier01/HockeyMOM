import tempfile
from pathlib import Path

import torch


def should_sample_gpt_dataset_and_run_model_forward():
    from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT
    from hmlib.camera.camera_gpt_dataset import CameraPanZoomGPTIterableDataset, GameCsvPaths
    from hmlib.camera.camera_transformer import CameraNorm

    with tempfile.TemporaryDirectory(prefix="hm_camgpt_") as td:
        td_path = Path(td)
        tracking_csv = td_path / "tracking.csv"
        camera_csv = td_path / "camera.csv"
        camera_fast_csv = td_path / "camera_fast.csv"
        pose_csv = td_path / "pose.csv"

        # tracking.csv (no header): Frame,ID,BBox_X,BBox_Y,BBox_W,BBox_H,Scores,Labels,Visibility,JerseyInfo
        tracking_lines = []
        for frame in range(1, 21):
            for tid in (1, 2):
                x = 10.0 * tid + frame
                y = 5.0 * tid
                w = 10.0
                h = 20.0
                tracking_lines.append(f"{frame},{tid},{x},{y},{w},{h},0.9,0,1,\n")
        tracking_csv.write_text("".join(tracking_lines))

        # camera.csv (no header): Frame,BBox_X,BBox_Y,BBox_W,BBox_H
        cam_lines = []
        for frame in range(1, 21):
            cam_lines.append(f"{frame},0,0,100,50\n")
        camera_csv.write_text("".join(cam_lines))

        # camera_fast.csv (no header): Frame,BBox_X,BBox_Y,BBox_W,BBox_H
        cam_fast_lines = []
        for frame in range(1, 21):
            cam_fast_lines.append(f"{frame},10,5,80,40\n")
        camera_fast_csv.write_text("".join(cam_fast_lines))

        # pose.csv (no header): Frame,PoseJSON
        pose_lines = []
        pose_json = '{"predictions":[{"bboxes":[[0,0,10,10]],"keypoint_scores":[[0.9,0.8]]}]}'
        pose_json_csv = pose_json.replace('"', '""')
        for frame in range(1, 21):
            pose_lines.append(f'{frame},"{pose_json_csv}"\n')
        pose_csv.write_text("".join(pose_lines))

        paths = GameCsvPaths(
            game_id="test",
            tracking_csv=str(tracking_csv),
            camera_csv=str(camera_csv),
            camera_fast_csv=str(camera_fast_csv),
            pose_csv=str(pose_csv),
        )
        norm = CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22)
        ds = CameraPanZoomGPTIterableDataset(
            games=[paths],
            norm=norm,
            seq_len=8,
            target_mode="slow_fast_tlwh",
            feature_mode="base_prev_y",
            include_pose=True,
            seed=123,
        )
        sample = next(iter(ds))
        base = sample["base"]
        prev0 = sample["prev0"]
        y = sample["y"]
        assert base.shape == (8, 16)
        assert prev0.shape == (8,)
        assert y.shape == (8, 8)

        # Teacher forcing: prev_y[t] = prev0 for t=0 else y[t-1]
        prev_y = torch.cat([prev0.unsqueeze(0), y[:-1]], dim=0)
        x = torch.cat([base, prev_y], dim=-1)
        assert x.shape == (8, 24)

        cfg = CameraGPTConfig(
            d_in=24,
            d_out=8,
            feature_mode="base_prev_y",
            include_pose=True,
            d_model=32,
            nhead=4,
            nlayers=2,
            dropout=0.1,
        )
        model = CameraPanZoomGPT(cfg)
        out = model(x.unsqueeze(0))
        assert out.shape == (1, 8, 8)
        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

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

        paths = GameCsvPaths(game_id="test", tracking_csv=str(tracking_csv), camera_csv=str(camera_csv))
        norm = CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22)
        ds = CameraPanZoomGPTIterableDataset(games=[paths], norm=norm, seq_len=8, seed=123)
        sample = next(iter(ds))
        x = sample["x"]
        y = sample["y"]
        assert x.shape == (8, 11)
        assert y.shape == (8, 3)

        cfg = CameraGPTConfig(d_in=11, d_model=32, nhead=4, nlayers=2, dropout=0.1)
        model = CameraPanZoomGPT(cfg)
        out = model(x.unsqueeze(0))
        assert out.shape == (1, 8, 3)
        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


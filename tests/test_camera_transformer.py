# ruff: noqa: E402

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from hmlib.camera.camera_model_dataset import CameraPanZoomDataset
from hmlib.camera.camera_transformer import CameraPanZoomTransformer, pack_checkpoint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_synth_csvs(tmpdir: str, n_frames: int = 50):
    t_rows = []
    c_rows = []
    W, H = 1920, 1080
    cx, cy = W / 2, H / 2
    h = H * 0.6
    for f in range(1, n_frames + 1):
        # simulate a sinusoidal pan
        cx = W / 2 + (W / 4) * np.sin(f / 10.0)
        cy = H / 2
        w = h * 16.0 / 9.0
        c_rows.append([f, cx - w / 2, cy - h / 2, w, h])
        # players distributed around center
        rng = np.random.RandomState(42 + f)
        for pid in range(8):
            px = cx + rng.randn() * (W / 10)
            py = cy + rng.randn() * (H / 10)
            pw, ph = 30 + rng.rand() * 20, 60 + rng.rand() * 20
            t_rows.append(
                [f, pid, px - pw / 2, py - ph / 2, pw, ph, 0.9, 0, 1, "{}", -1, "", 0.0, -1]
            )
    tracking_csv = os.path.join(tmpdir, "tracking.csv")
    camera_csv = os.path.join(tmpdir, "camera.csv")
    pd.DataFrame(t_rows).to_csv(tracking_csv, header=False, index=False)
    pd.DataFrame(c_rows).to_csv(camera_csv, header=False, index=False)
    return tracking_csv, camera_csv


def test_dataset_and_model_train_smoke():
    with tempfile.TemporaryDirectory() as td:
        t_csv, c_csv = _make_synth_csvs(td, n_frames=80)
        ds = CameraPanZoomDataset(tracking_csv=t_csv, camera_csv=c_csv, window=8)
        assert len(ds) > 0
        sample = ds[0]
        model = CameraPanZoomTransformer(d_in=sample.x.shape[-1])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = sample.x.unsqueeze(0)
        y = sample.y.unsqueeze(0)
        pred = model(x)
        loss = torch.nn.functional.l1_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ckpt = pack_checkpoint(model, ds.norm, 8)
        assert "state_dict" in ckpt and "norm" in ckpt

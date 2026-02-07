from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from mmengine.structures import InstanceData


class _DummyTrackDataSample:
    def __init__(self, frames: list[object]):
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> object:
        return self._frames[idx]


def should_camera_controller_plugin_load_gpt_checkpoint_and_emit_boxes():
    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin
    from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, pack_gpt_checkpoint
    from hmlib.camera.camera_transformer import CameraNorm

    torch.manual_seed(0)

    cfg = CameraGPTConfig(
        d_in=11,
        d_out=3,
        feature_mode="legacy_prev_slow",
        include_pose=False,
        include_rink=False,
        d_model=32,
        nhead=4,
        nlayers=2,
        dropout=0.1,
    )
    model = CameraPanZoomGPT(cfg)
    norm = CameraNorm(scale_x=1920.0, scale_y=1080.0, max_players=22)
    ckpt = pack_gpt_checkpoint(model=model, norm=norm, window=8, cfg=cfg)

    with tempfile.TemporaryDirectory(prefix="hm_camctrl_gpt_") as td:
        ckpt_path = Path(td) / "camera_gpt.pt"
        torch.save(ckpt, ckpt_path)

        frames: list[object] = []
        for _ in range(3):
            inst = InstanceData(
                bboxes=torch.tensor(
                    [[100.0, 200.0, 160.0, 280.0], [500.0, 400.0, 560.0, 520.0]],
                    dtype=torch.float32,
                )
            )
            frames.append(
                SimpleNamespace(
                    pred_track_instances=inst,
                    metainfo={"ori_shape": (1080, 1920)},
                )
            )
        track_data_sample = _DummyTrackDataSample(frames)

        plugin = CameraControllerPlugin(
            controller="gpt",
            model_path=str(ckpt_path),
            window=8,
        )
        out = plugin.forward({"data_samples": track_data_sample, "inputs": torch.zeros(1)})

        cam_boxes = out.get("camera_boxes")
        assert isinstance(cam_boxes, torch.Tensor)
        assert cam_boxes.shape == (3, 4)
        assert torch.isfinite(cam_boxes).all()

        # Boxes should be clamped to frame bounds.
        assert torch.all(cam_boxes[:, 0] >= 0.0)
        assert torch.all(cam_boxes[:, 1] >= 0.0)
        assert torch.all(cam_boxes[:, 2] <= 1920.0)
        assert torch.all(cam_boxes[:, 3] <= 1080.0)

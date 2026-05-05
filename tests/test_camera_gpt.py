import tempfile
from pathlib import Path

import pytest


def _torch():
    return pytest.importorskip("torch")


def should_sample_gpt_dataset_and_run_model_forward():
    torch = _torch()

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

        ds_players = CameraPanZoomGPTIterableDataset(
            games=[paths],
            norm=norm,
            seq_len=8,
            target_mode="slow_fast_tlwh",
            feature_mode="players_prev_y",
            include_pose=True,
            seed=123,
        )
        sample_players = next(iter(ds_players))
        assert sample_players["base"].shape == (8, 104)
        assert sample_players["prev0"].shape == (8,)
        assert sample_players["y"].shape == (8, 8)

        ds_cold_start = CameraPanZoomGPTIterableDataset(
            games=[paths],
            norm=norm,
            seq_len=20,
            target_mode="slow_fast_tlwh",
            feature_mode="base_prev_y",
            include_pose=False,
            seed=123,
        )
        sample_cold_start = next(iter(ds_cold_start))
        assert torch.equal(
            sample_cold_start["prev0"],
            torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        )


def should_pack_drivegpt_metadata_and_init_from_opendrive_planning():
    torch = _torch()

    from hmlib.camera.camera_gpt import (
        CameraGPTConfig,
        CameraPanZoomGPT,
        init_from_opendrive_uniad_planning,
        pack_gpt_checkpoint,
        unpack_gpt_checkpoint,
    )
    from hmlib.camera.camera_transformer import CameraNorm

    cfg = CameraGPTConfig(
        d_in=4,
        d_out=8,
        model_kind="drivegpt",
        feature_mode="base_prev_y",
        include_pose=False,
        include_rink=True,
        source_model_id="OpenDriveLab/UniAD2.0_R101_nuScenes",
        source_checkpoint="/tmp/fake_uniad.pth",
        source_init="opendrive-uniad-planning",
        residual_prev_y=True,
        residual_scale=0.05,
        d_model=256,
        nhead=8,
        nlayers=3,
        dim_feedforward=512,
        dropout=0.0,
    )
    model = CameraPanZoomGPT(cfg)
    target_sd = model.state_dict()
    source_sd = {}
    mapping = {
        "self_attn.in_proj_weight": "self_attn.in_proj_weight",
        "self_attn.in_proj_bias": "self_attn.in_proj_bias",
        "self_attn.out_proj.weight": "self_attn.out_proj.weight",
        "self_attn.out_proj.bias": "self_attn.out_proj.bias",
        "multihead_attn.in_proj_weight": "multihead_attn.in_proj_weight",
        "multihead_attn.in_proj_bias": "multihead_attn.in_proj_bias",
        "multihead_attn.out_proj.weight": "multihead_attn.out_proj.weight",
        "multihead_attn.out_proj.bias": "multihead_attn.out_proj.bias",
        "linear1.weight": "linear1.weight",
        "linear1.bias": "linear1.bias",
        "linear2.weight": "linear2.weight",
        "linear2.bias": "linear2.bias",
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "norm3.weight": "norm3.weight",
        "norm3.bias": "norm3.bias",
    }
    for layer_idx in range(3):
        for idx, (source_suffix, target_suffix) in enumerate(mapping.items(), start=1):
            target_key = f"decoder.layers.{layer_idx}.{target_suffix}"
            source_key = f"planning_head.attn_module.layers.{layer_idx}.{source_suffix}"
            source_sd[source_key] = torch.full_like(
                target_sd[target_key], float((layer_idx * len(mapping)) + idx) / 100.0
            )

    with tempfile.TemporaryDirectory(prefix="hm_drivegpt_") as td:
        ckpt_path = Path(td) / "uniad_fake.pth"
        torch.save({"state_dict": source_sd}, ckpt_path)
        before = model.state_dict()["decoder.layers.0.linear1.bias"].clone()
        report = init_from_opendrive_uniad_planning(model, str(ckpt_path))

    after = model.state_dict()["decoder.layers.0.linear1.bias"]
    assert report["source"] == "opendrive-uniad-planning"
    assert report["copied_tensors"] == len(mapping) * 3
    assert report["initialized_layers"] == 3
    assert not torch.equal(before, after)

    packed = pack_gpt_checkpoint(
        model, norm=CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22), window=8, cfg=cfg
    )
    _, norm, window, unpacked = unpack_gpt_checkpoint(packed)
    assert norm.scale_x == 100.0
    assert window == 8
    assert unpacked.model_kind == "drivegpt"
    assert unpacked.source_model_id == "OpenDriveLab/UniAD2.0_R101_nuScenes"
    assert unpacked.source_init == "opendrive-uniad-planning"
    assert unpacked.residual_prev_y is True
    assert unpacked.residual_scale == 0.05
    assert unpacked.dim_feedforward == 512

    bad_cfg = CameraGPTConfig(
        d_in=4,
        d_out=8,
        model_kind="drivegpt",
        d_model=256,
        nhead=4,
        nlayers=3,
        dim_feedforward=512,
    )
    with tempfile.TemporaryDirectory(prefix="hm_drivegpt_bad_") as td:
        ckpt_path = Path(td) / "uniad_fake.pth"
        torch.save({"state_dict": source_sd}, ckpt_path)
        with pytest.raises(ValueError, match="requires d_model=256, nhead=8"):
            init_from_opendrive_uniad_planning(CameraPanZoomGPT(bad_cfg), str(ckpt_path))

    partial_cfg = CameraGPTConfig(
        d_in=4,
        d_out=8,
        model_kind="drivegpt",
        d_model=256,
        nhead=8,
        nlayers=1,
        dim_feedforward=512,
    )
    with tempfile.TemporaryDirectory(prefix="hm_drivegpt_partial_") as td:
        ckpt_path = Path(td) / "uniad_fake.pth"
        torch.save({"state_dict": source_sd}, ckpt_path)
        with pytest.raises(ValueError, match="requires nlayers=3"):
            init_from_opendrive_uniad_planning(CameraPanZoomGPT(partial_cfg), str(ckpt_path))


def should_load_drivegpt_controller_as_gpt_backend(monkeypatch):
    torch = _torch()

    import sys
    import types

    fake_clusters = types.ModuleType("hmlib.camera.clusters")
    fake_clusters.ClusterMan = object
    monkeypatch.setitem(sys.modules, "hmlib.camera.clusters", fake_clusters)

    from hmlib.aspen.plugins.camera_controller_plugin import CameraControllerPlugin
    from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, pack_gpt_checkpoint
    from hmlib.camera.camera_transformer import CameraNorm

    cfg = CameraGPTConfig(d_in=4, d_out=4, model_kind="drivegpt", d_model=32, nhead=4, nlayers=1)
    model = CameraPanZoomGPT(cfg)
    ckpt = pack_gpt_checkpoint(
        model, norm=CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22), window=8, cfg=cfg
    )

    monkeypatch.setattr(torch, "load", lambda path, map_location=None: ckpt)

    plugin = CameraControllerPlugin(controller="drivegpt", model_path="/tmp/fake_drivegpt.pt")

    assert plugin._controller == "gpt"
    assert plugin._gpt_model is not None
    assert plugin._gpt_cfg is not None
    assert plugin._gpt_cfg.model_kind == "drivegpt"
    pose_feat = plugin._pose_features(
        [
            {
                "predictions": [
                    {
                        "bboxes": [[0.0, 0.0, 10.0, 10.0]],
                        "keypoint_scores": [[0.9, 0.8]],
                    }
                ]
            }
        ],
        0,
        torch.device("cpu"),
    )
    assert pose_feat.shape == (8,)
    assert torch.count_nonzero(pose_feat).item() > 0

    legacy_cfg = CameraGPTConfig(d_in=4, d_out=4, model_kind="gpt", d_model=32, nhead=4, nlayers=1)
    legacy_model = CameraPanZoomGPT(legacy_cfg)
    legacy_ckpt = pack_gpt_checkpoint(
        legacy_model,
        norm=CameraNorm(scale_x=100.0, scale_y=50.0, max_players=22),
        window=8,
        cfg=legacy_cfg,
    )
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: legacy_ckpt)
    with pytest.raises(RuntimeError, match="model_kind='drivegpt'"):
        CameraControllerPlugin(controller="drivegpt", model_path="/tmp/fake_legacy_gpt.pt")

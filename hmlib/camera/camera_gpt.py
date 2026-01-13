"""GPT-style causal transformer for camera pan/zoom prediction.

This extends the existing camera transformer approach to a decoder-only,
autoregressive-friendly model that emits a camera state per frame.

The model consumes per-frame feature vectors (tracking/pose aggregates plus
previous camera state) and predicts normalized camera center/height:
  - cx, cy in [0, 1]
  - h_ratio in [0, 1] (relative to the global frame height)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

from hmlib.camera.camera_transformer import CameraNorm, PositionalEncoding


@dataclass(frozen=True)
class CameraGPTConfig:
    d_in: int = 11
    d_out: int = 3
    feature_mode: str = "legacy_prev_slow"
    include_pose: bool = False
    d_model: int = 128
    nhead: int = 4
    nlayers: int = 4
    dropout: float = 0.1


class CameraPanZoomGPT(nn.Module):
    """Causal transformer that predicts a camera state per timestep."""

    def __init__(self, cfg: CameraGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(int(cfg.d_in), int(cfg.d_model))
        self.pe = PositionalEncoding(int(cfg.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.nhead),
            batch_first=True,
            dropout=float(cfg.dropout),
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.nlayers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(cfg.d_model)),
            nn.Linear(int(cfg.d_model), int(cfg.d_model)),
            nn.ReLU(inplace=True),
            nn.Linear(int(cfg.d_model), int(cfg.d_out)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if x.ndim != 3:
            raise ValueError(f"Expected x to be [B,T,D], got shape {tuple(x.shape)}")
        t = int(x.shape[1])
        h = self.input(x)
        h = self.pe(h)
        # Causal (GPT-style) mask: prevent attending to future timesteps.
        mask = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.head(h)


def pack_gpt_checkpoint(
    model: nn.Module, norm: CameraNorm, window: int, cfg: CameraGPTConfig
) -> Dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "norm": {
            "scale_x": float(norm.scale_x),
            "scale_y": float(norm.scale_y),
            "max_players": int(norm.max_players),
        },
        "window": int(window),
        "model": {
            "d_in": int(cfg.d_in),
            "d_out": int(cfg.d_out),
            "feature_mode": str(cfg.feature_mode),
            "include_pose": bool(cfg.include_pose),
            "d_model": int(cfg.d_model),
            "nhead": int(cfg.nhead),
            "nlayers": int(cfg.nlayers),
            "dropout": float(cfg.dropout),
        },
    }


def unpack_gpt_checkpoint(
    ckpt: Dict[str, Any]
) -> Tuple[Dict[str, Any], CameraNorm, int, CameraGPTConfig]:
    sd = ckpt["state_dict"]
    n = ckpt["norm"]
    window = int(ckpt.get("window", 16))
    model_cfg = ckpt.get("model") or {}
    include_pose = model_cfg.get("include_pose", ckpt.get("include_pose", False))
    cfg = CameraGPTConfig(
        d_in=int(model_cfg.get("d_in", 11)),
        d_out=int(model_cfg.get("d_out", 3)),
        feature_mode=str(model_cfg.get("feature_mode", "legacy_prev_slow")),
        include_pose=bool(include_pose),
        d_model=int(model_cfg.get("d_model", 128)),
        nhead=int(model_cfg.get("nhead", 4)),
        nlayers=int(model_cfg.get("nlayers", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    norm = CameraNorm(
        scale_x=float(n["scale_x"]),
        scale_y=float(n["scale_y"]),
        max_players=int(n["max_players"]),
    )
    return sd, norm, window, cfg

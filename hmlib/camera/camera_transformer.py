"""Transformer-based camera pan/zoom controller utilities.

Contains helpers to build per-frame feature vectors from player TLWH boxes
and a small transformer model for camera movement prediction.

@see @ref hmlib.camera.camera_model_dataset.CameraPanZoomDataset "CameraPanZoomDataset"
     for the dataset used to train these models.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class CameraNorm:
    scale_x: float
    scale_y: float
    max_players: int


def build_frame_features(
    tlwh: np.ndarray,
    norm: CameraNorm,
    prev_cam_center: Optional[Tuple[float, float]] = None,
    prev_cam_h: Optional[float] = None,
) -> np.ndarray:
    """Build a simple aggregated feature vector for one frame.

    This is the legacy feature builder used by the original transformer camera model:
    base player statistics (10 dims) + previous camera state (3 dims).
    """
    base = build_frame_base_features(tlwh=tlwh, norm=norm)
    prev_cx, prev_cy = (0.0, 0.0) if prev_cam_center is None else prev_cam_center
    prev_h = 0.0 if prev_cam_h is None else prev_cam_h
    prev = np.asarray([prev_cx, prev_cy, prev_h], dtype=np.float32)
    return np.concatenate([base, prev], axis=0).astype(np.float32, copy=False)


def build_frame_base_features(tlwh: np.ndarray, norm: CameraNorm) -> np.ndarray:
    """Build aggregated per-frame features that *exclude* previous camera state.

    Features (all normalized):
      - num_players (0..1)
      - mean cx, mean cy
      - std cx, std cy
      - min cx, max cx, min cy, max cy
      - group width ratio (max x_right - min x_left) / scale_x
    """
    if tlwh is None or len(tlwh) == 0:
        n = 0
        cxs = np.array([0.0])
        cys = np.array([0.0])
    else:
        n = len(tlwh)
        lefts = tlwh[:, 0]
        widths = tlwh[:, 2]
        cxs = lefts + widths * 0.5
        cys = tlwh[:, 1] + tlwh[:, 3] * 0.5
    # normalize
    num_players = min(n / max(1, norm.max_players), 1.0)
    cxn = np.clip(cxs / max(1e-6, norm.scale_x), 0.0, 1.0)
    cyn = np.clip(cys / max(1e-6, norm.scale_y), 0.0, 1.0)
    min_cx = float(np.min(cxn))
    max_cx = float(np.max(cxn))
    min_cy = float(np.min(cyn))
    max_cy = float(np.max(cyn))
    mean_cx = float(np.mean(cxn))
    mean_cy = float(np.mean(cyn))
    std_cx = float(np.std(cxn))
    std_cy = float(np.std(cyn))
    if len(tlwh) == 0:
        group_width_ratio = 0.0
    else:
        group_width_ratio = float(
            (float(np.max(lefts + tlwh[:, 2])) - float(np.min(lefts))) / max(1e-6, norm.scale_x)
        )
        group_width_ratio = np.clip(group_width_ratio, 0.0, 1.0)
    feat = np.array(
        [
            num_players,
            mean_cx,
            mean_cy,
            std_cx,
            std_cy,
            min_cx,
            max_cx,
            min_cy,
            max_cy,
            group_width_ratio,
        ],
        dtype=np.float32,
    )
    return feat


def build_frame_features_torch(
    tlwh: torch.Tensor,
    norm: CameraNorm,
    prev_cam_center: Optional[Tuple[float, float]] = None,
    prev_cam_h: Optional[float] = None,
) -> torch.Tensor:
    """Torch variant of build_frame_features (stays on the input device)."""
    base = build_frame_base_features_torch(tlwh=tlwh, norm=norm)
    device = base.device
    dtype = base.dtype
    if prev_cam_center is None:
        prev_cx = torch.zeros((), device=device, dtype=dtype)
        prev_cy = torch.zeros((), device=device, dtype=dtype)
    elif torch.is_tensor(prev_cam_center):
        prev_cx = prev_cam_center[0].to(device=device, dtype=dtype)
        prev_cy = prev_cam_center[1].to(device=device, dtype=dtype)
    else:
        prev_cx = torch.tensor(float(prev_cam_center[0]), device=device, dtype=dtype)
        prev_cy = torch.tensor(float(prev_cam_center[1]), device=device, dtype=dtype)
    if prev_cam_h is None:
        prev_h = torch.zeros((), device=device, dtype=dtype)
    elif torch.is_tensor(prev_cam_h):
        prev_h = prev_cam_h.to(device=device, dtype=dtype)
    else:
        prev_h = torch.tensor(float(prev_cam_h), device=device, dtype=dtype)
    prev = torch.stack([prev_cx, prev_cy, prev_h], dim=0)
    return torch.cat([base, prev], dim=0)


def build_frame_base_features_torch(tlwh: torch.Tensor, norm: CameraNorm) -> torch.Tensor:
    """Torch variant of build_frame_base_features (stays on the input device)."""
    if tlwh is None or tlwh.numel() == 0:
        device = tlwh.device if isinstance(tlwh, torch.Tensor) else torch.device("cpu")
        cxs = torch.zeros(1, device=device, dtype=torch.float32)
        cys = torch.zeros(1, device=device, dtype=torch.float32)
        n = 0
        lefts = cxs
        widths = cxs
    else:
        tlwh = tlwh.to(dtype=torch.float32)
        n = int(tlwh.shape[0])
        lefts = tlwh[:, 0]
        tops = tlwh[:, 1]
        widths = tlwh[:, 2]
        heights = tlwh[:, 3]
        cxs = lefts + widths * 0.5
        cys = tops + heights * 0.5
    num_players = min(float(n) / max(1, norm.max_players), 1.0)
    scale_x = max(1e-6, float(norm.scale_x))
    scale_y = max(1e-6, float(norm.scale_y))
    cxn = torch.clamp(cxs / scale_x, 0.0, 1.0)
    cyn = torch.clamp(cys / scale_y, 0.0, 1.0)
    min_cx = torch.min(cxn)
    max_cx = torch.max(cxn)
    min_cy = torch.min(cyn)
    max_cy = torch.max(cyn)
    mean_cx = torch.mean(cxn)
    mean_cy = torch.mean(cyn)
    std_cx = torch.std(cxn, unbiased=False)
    std_cy = torch.std(cyn, unbiased=False)
    if n == 0:
        group_width_ratio = torch.zeros((), device=cxn.device, dtype=cxn.dtype)
    else:
        group_width_ratio = (torch.max(lefts + widths) - torch.min(lefts)) / scale_x
        group_width_ratio = torch.clamp(group_width_ratio, 0.0, 1.0)
    feat = torch.stack(
        [
            torch.tensor(num_players, device=cxn.device, dtype=cxn.dtype),
            mean_cx,
            mean_cy,
            std_cx,
            std_cy,
            min_cx,
            max_cx,
            min_cy,
            max_cy,
            group_width_ratio,
        ],
        dim=0,
    )
    return feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.size(1)
        return x + self.pe[:, :t, :]


class CameraPanZoomTransformer(nn.Module):
    def __init__(
        self, d_in: int, d_model: int = 128, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.input = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pe = PositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),  # cx, cy, h
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.input(x)
        h = self.pe(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        out = self.head(h)
        return out


def make_box_from_center_h(
    cx: float,
    cy: float,
    h_ratio: float,
    aspect_ratio: float,
    scale_x: float,
    scale_y: float,
) -> Tuple[float, float, float, float]:
    h = float(np.clip(h_ratio, 0.02, 1.0) * scale_y)
    w = float(h * aspect_ratio)
    cx_px = float(np.clip(cx, 0.0, 1.0) * scale_x)
    cy_px = float(np.clip(cy, 0.0, 1.0) * scale_y)
    left = cx_px - w / 2.0
    top = cy_px - h / 2.0
    right = left + w
    bottom = top + h
    return left, top, right, bottom


def pack_checkpoint(model: nn.Module, norm: CameraNorm, window: int) -> Dict[str, torch.Tensor]:
    return {
        "state_dict": model.state_dict(),
        "norm": {
            "scale_x": norm.scale_x,
            "scale_y": norm.scale_y,
            "max_players": norm.max_players,
        },
        "window": int(window),
    }


def unpack_checkpoint(
    ckpt: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], CameraNorm, int]:
    sd = ckpt["state_dict"]
    n = ckpt["norm"]
    window = int(ckpt.get("window", 8))
    norm = CameraNorm(
        scale_x=float(n["scale_x"]), scale_y=float(n["scale_y"]), max_players=int(n["max_players"])
    )
    return sd, norm, window

"""Datasets for training the GPT-style camera controller from saved CSVs."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from hmlib.camera.camera_transformer import CameraNorm, build_frame_features


@dataclass(frozen=True)
class GameCsvPaths:
    game_id: str
    tracking_csv: str
    camera_csv: str
    pose_csv: Optional[str] = None


def _read_tracking_dataframe(tracking_csv: str) -> pd.DataFrame:
    tracks = pd.read_csv(tracking_csv, header=None)
    # Support legacy tracking CSVs with a PoseIndex column (11/14) and current ones (10/13).
    base_cols = [
        "Frame",
        "ID",
        "BBox_X",
        "BBox_Y",
        "BBox_W",
        "BBox_H",
        "Scores",
        "Labels",
        "Visibility",
        "JerseyInfo",
    ]
    extra_cols = ["ActionLabel", "ActionScore", "ActionIndex"]
    legacy_cols = base_cols + ["PoseIndex"]
    if tracks.shape[1] >= len(legacy_cols) + len(extra_cols):
        tracks = tracks.iloc[:, : len(legacy_cols) + len(extra_cols)]
        tracks.columns = legacy_cols + extra_cols
        tracks = tracks.drop(columns=["PoseIndex"])
    elif tracks.shape[1] == len(legacy_cols):
        tracks.columns = legacy_cols
        tracks = tracks.drop(columns=["PoseIndex"])
    elif tracks.shape[1] >= len(base_cols) + len(extra_cols):
        tracks = tracks.iloc[:, : len(base_cols) + len(extra_cols)]
        tracks.columns = base_cols + extra_cols
    elif tracks.shape[1] == len(base_cols):
        tracks.columns = base_cols
    else:
        need = len(base_cols) - tracks.shape[1]
        for _ in range(need):
            tracks[tracks.shape[1]] = None
        tracks.columns = base_cols
    return tracks


def _read_camera_dataframe(camera_csv: str) -> pd.DataFrame:
    cams = pd.read_csv(camera_csv, header=None)
    cams.columns = ["Frame", "BBox_X", "BBox_Y", "BBox_W", "BBox_H"]
    return cams


def scan_game_max_xy(tracking_csv: str, camera_csv: str) -> Tuple[float, float]:
    """Return (max_x, max_y) across tracking + camera CSVs for a game."""
    max_x = float("nan")
    max_y = float("nan")
    try:
        tracks = _read_tracking_dataframe(tracking_csv)
        x2 = pd.to_numeric(tracks["BBox_X"], errors="coerce") + pd.to_numeric(
            tracks["BBox_W"], errors="coerce"
        )
        y2 = pd.to_numeric(tracks["BBox_Y"], errors="coerce") + pd.to_numeric(
            tracks["BBox_H"], errors="coerce"
        )
        max_x = float(np.nanmax(x2.to_numpy(dtype=np.float64)))
        max_y = float(np.nanmax(y2.to_numpy(dtype=np.float64)))
    except Exception:
        pass
    try:
        cams = _read_camera_dataframe(camera_csv)
        x2 = pd.to_numeric(cams["BBox_X"], errors="coerce") + pd.to_numeric(
            cams["BBox_W"], errors="coerce"
        )
        y2 = pd.to_numeric(cams["BBox_Y"], errors="coerce") + pd.to_numeric(
            cams["BBox_H"], errors="coerce"
        )
        max_x = float(np.nanmax([max_x, float(np.nanmax(x2.to_numpy(dtype=np.float64)))]))
        max_y = float(np.nanmax([max_y, float(np.nanmax(y2.to_numpy(dtype=np.float64)))]))
    except Exception:
        pass
    if not np.isfinite(max_x) or max_x <= 0:
        max_x = 1920.0
    if not np.isfinite(max_y) or max_y <= 0:
        max_y = 1080.0
    return max_x, max_y


@dataclass
class _LoadedGame:
    game_id: str
    frames: list[int]
    tracks_by_frame: Dict[int, np.ndarray]
    cam_by_frame: Dict[int, Tuple[float, float, float]]  # cx, cy, h_ratio (normalized)


def _load_game(paths: GameCsvPaths, norm: CameraNorm) -> _LoadedGame:
    tracks = _read_tracking_dataframe(paths.tracking_csv)
    cams = _read_camera_dataframe(paths.camera_csv)

    # Frames present in both tracking and camera.
    frames = sorted(set(tracks["Frame"].unique()).intersection(set(cams["Frame"].unique())))
    frames_int = [int(f) for f in frames]

    tracks_by_frame: Dict[int, np.ndarray] = {}
    if not tracks.empty:
        for frame_id, group in tracks.groupby("Frame"):
            fid = int(frame_id)
            arr = group[["BBox_X", "BBox_Y", "BBox_W", "BBox_H"]].to_numpy(dtype=np.float32)
            tracks_by_frame[fid] = arr

    cam_by_frame: Dict[int, Tuple[float, float, float]] = {}
    if not cams.empty:
        for row in cams.itertuples(index=False):
            fid = int(getattr(row, "Frame"))
            x = float(getattr(row, "BBox_X"))
            y = float(getattr(row, "BBox_Y"))
            w = float(getattr(row, "BBox_W"))
            h = float(getattr(row, "BBox_H"))
            cx = (x + w / 2.0) / max(1e-6, float(norm.scale_x))
            cy = (y + h / 2.0) / max(1e-6, float(norm.scale_y))
            hr = h / max(1e-6, float(norm.scale_y))
            cam_by_frame[fid] = (float(cx), float(cy), float(hr))

    return _LoadedGame(
        game_id=paths.game_id,
        frames=frames_int,
        tracks_by_frame=tracks_by_frame,
        cam_by_frame=cam_by_frame,
    )


class CameraPanZoomGPTIterableDataset(IterableDataset):
    """Randomly samples fixed-length sequences from multiple games."""

    def __init__(
        self,
        games: list[GameCsvPaths],
        norm: CameraNorm,
        seq_len: int = 32,
        max_players_for_norm: int = 22,
        seed: int = 0,
        max_cached_games: int = 8,
    ) -> None:
        super().__init__()
        self._games = games
        self._norm = CameraNorm(
            scale_x=float(norm.scale_x),
            scale_y=float(norm.scale_y),
            max_players=int(max_players_for_norm),
        )
        self._seq_len = int(seq_len)
        self._seed = int(seed)
        self._max_cached = int(max_cached_games)
        self._cache: Dict[str, _LoadedGame] = {}
        self._cache_order: deque[str] = deque()

    @property
    def norm(self) -> CameraNorm:
        return self._norm

    def _get_game(self, paths: GameCsvPaths) -> Optional[_LoadedGame]:
        game_id = paths.game_id
        cached = self._cache.get(game_id)
        if cached is not None:
            return cached
        try:
            loaded = _load_game(paths, norm=self._norm)
        except Exception:
            return None
        if len(loaded.frames) < self._seq_len:
            return None
        self._cache[game_id] = loaded
        self._cache_order.append(game_id)
        while len(self._cache_order) > self._max_cached:
            old = self._cache_order.popleft()
            self._cache.pop(old, None)
        return loaded

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        rng = random.Random(self._seed + worker_id)
        while True:
            paths = rng.choice(self._games)
            game = self._get_game(paths)
            if game is None:
                continue

            frames = game.frames
            start = rng.randint(0, len(frames) - self._seq_len)
            seq_frames = frames[start : start + self._seq_len]

            feats = []
            targets = []

            prev = None
            if start > 0:
                prev = game.cam_by_frame.get(frames[start - 1])
            if prev is None:
                prev = game.cam_by_frame.get(seq_frames[0], (0.5, 0.5, 1.0))
            prev_cx, prev_cy, prev_h = prev

            for f in seq_frames:
                tlwh = game.tracks_by_frame.get(f)
                if tlwh is None:
                    tlwh = np.zeros((0, 4), dtype=np.float32)
                feat = build_frame_features(
                    tlwh=tlwh,
                    norm=self._norm,
                    prev_cam_center=(prev_cx, prev_cy),
                    prev_cam_h=prev_h,
                )
                feats.append(feat)
                cam = game.cam_by_frame.get(f, (prev_cx, prev_cy, prev_h))
                targets.append(np.asarray(cam, dtype=np.float32))
                prev_cx, prev_cy, prev_h = cam

            x = torch.from_numpy(np.stack(feats, axis=0))  # [T, D]
            y = torch.from_numpy(np.stack(targets, axis=0))  # [T, 3]
            yield {"x": x, "y": y}


def resolve_csv_paths(game_id: str, game_dir: str) -> Optional[GameCsvPaths]:
    """Resolve required CSVs inside a game directory (latest suffix wins)."""
    from hmlib.datasets.dataframe import find_latest_dataframe_file

    tracking = find_latest_dataframe_file(game_dir, "tracking")
    camera = find_latest_dataframe_file(game_dir, "camera")
    if not tracking or not camera:
        return None
    pose = find_latest_dataframe_file(game_dir, "pose")
    return GameCsvPaths(game_id=game_id, tracking_csv=tracking, camera_csv=camera, pose_csv=pose)


def validate_csv_paths(paths: GameCsvPaths) -> None:
    for p in (paths.tracking_csv, paths.camera_csv):
        if not Path(p).is_file():
            raise FileNotFoundError(p)


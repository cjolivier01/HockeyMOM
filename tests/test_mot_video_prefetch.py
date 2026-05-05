from __future__ import annotations

import sys
import types
from typing import Iterator

import pytest
import torch

sys.modules.setdefault(
    "hmlib.video.video_stream",
    types.SimpleNamespace(VideoStreamReader=object),
)

from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig


class _EmbeddedFrames:
    def __init__(self, frame_count: int) -> None:
        self._frame_count = frame_count
        self._index = 0
        self.batch_size = 1
        self.fps = 30.0
        self.bit_rate = 1

    def __iter__(self) -> Iterator[torch.Tensor]:
        self._index = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self._index >= self._frame_count:
            raise StopIteration
        value = self._index
        self._index += 1
        return torch.full((8, 12, 3), value, dtype=torch.uint8)

    def __len__(self) -> int:
        return self._frame_count

    def close(self) -> None:
        return


def should_refill_async_prefetch_window_for_embedded_loader():
    loader = MOTLoadVideoWithOrig(
        path=None,
        game_id="prefetch-test",
        embedded_data_loader=_EmbeddedFrames(frame_count=5),
        original_image_only=True,
        async_mode=True,
        prefetch_batches=3,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
    )

    try:
        it = iter(loader)
        assert loader._pending_worker_requests == 3

        frame_ids = []
        frame_values = []
        for _ in range(5):
            batch = next(it)
            frame_ids.append(int(batch["frame_ids"][0]))
            frame_values.append(int(batch["img"][0, 0, 0, 0]))
            assert 1 <= loader._pending_worker_requests <= 3

        assert frame_ids == [1, 2, 3, 4, 5]
        assert frame_values == [0, 1, 2, 3, 4]

        with pytest.raises(StopIteration):
            next(it)
        assert loader._pending_worker_requests == 0
    finally:
        loader.close()

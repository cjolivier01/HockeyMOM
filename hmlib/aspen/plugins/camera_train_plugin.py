from __future__ import annotations

import os
from typing import Any, Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from hmlib.builder import HM
from hmlib.camera.camera_model_dataset import CameraPanZoomDataset
from hmlib.camera.camera_transformer import CameraPanZoomTransformer, pack_checkpoint
from hmlib.log import logger

from .base import Plugin


@HM.register_module()
class CameraTrainPlugin(Plugin):
    """
    Plugin to train the camera transformer from saved CSV dataframes.

    Expects no particular runtime inputs. This trunk runs a blocking training
    session when invoked in a pipeline and writes a checkpoint to disk.

    Args:
      tracking_csv: path to tracking.csv
      camera_csv: path to camera.csv
      out: output .pt path
      epochs: number of epochs
      batch_size: batch size
      lr: learning rate
      window: temporal window length
      val_split: fraction for validation split
    """

    def __init__(
        self,
        enabled: bool = True,
        tracking_csv: str = None,
        camera_csv: str = None,
        out: str = "camera_transformer.pt",
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        window: int = 8,
        val_split: float = 0.1,
    ) -> None:
        super().__init__(enabled=enabled)
        self._tracking_csv = tracking_csv
        self._camera_csv = camera_csv
        self._out = out
        self._epochs = int(epochs)
        self._bs = int(batch_size)
        self._lr = float(lr)
        self._window = int(window)
        self._val_split = float(val_split)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        assert self._tracking_csv and self._camera_csv, "tracking_csv and camera_csv are required"
        ds = CameraPanZoomDataset(
            tracking_csv=self._tracking_csv, camera_csv=self._camera_csv, window=self._window
        )
        n = len(ds)
        n_val = max(1, int(n * self._val_split))
        n_train = max(1, n - n_val)
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=self._bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self._bs, shuffle=False, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CameraPanZoomTransformer(d_in=ds[0].x.shape[-1]).to(device)
        opt = optim.AdamW(model.parameters(), lr=self._lr)
        best_val = 1e9

        def _epoch(loader, train: bool) -> float:
            total = 0
            loss_sum = 0.0
            if train:
                model.train()
            else:
                model.eval()
            for batch in loader:
                x = batch.x.to(device)
                y = batch.y.to(device)
                with torch.set_grad_enabled(train):
                    pred = model(x)
                    loss = nn.functional.l1_loss(pred, y)
                    if train:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                loss_sum += float(loss.item()) * x.size(0)
                total += x.size(0)
            return loss_sum / max(1, total)

        for epoch in range(self._epochs):
            tr = _epoch(train_loader, True)
            va = _epoch(val_loader, False)
            logger.info(
                "CameraTrainPlugin epoch %d: train %.4f val %.4f",
                epoch + 1,
                tr,
                va,
            )
            if va < best_val:
                best_val = va
                ckpt = pack_checkpoint(model, norm=ds.norm, window=self._window)
                os.makedirs(os.path.dirname(self._out) or ".", exist_ok=True)
                torch.save(ckpt, self._out)
                logger.info("Saved camera transformer checkpoint to %s", self._out)
        return {"camera_model_path": self._out}

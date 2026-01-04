import argparse
import os
from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from hmlib.camera.camera_model_dataset import CameraPanZoomDataset
from hmlib.camera.camera_transformer import CameraPanZoomTransformer, pack_checkpoint
from hmlib.log import logger


def train_one_epoch(
    model, loader, opt, device, limit_steps: Optional[int] = None
) -> Tuple[float, float]:
    model.train()
    total = 0
    loss_sum = 0.0
    steps = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = nn.functional.l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)
        steps += 1
        if limit_steps is not None and steps >= limit_steps:
            break
    return loss_sum / max(1, total), total


@torch.no_grad()
def eval_loss(model, loader, device, limit_steps: Optional[int] = None) -> float:
    model.eval()
    total = 0
    loss_sum = 0.0
    steps = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = nn.functional.l1_loss(pred, y)
        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)
        steps += 1
        if limit_steps is not None and steps >= limit_steps:
            break
    return loss_sum / max(1, total)


def main():
    ap = argparse.ArgumentParser("Train transformer camera model from saved dataframes")
    ap.add_argument("--tracking-csv", required=True, help="Path to tracking.csv")
    ap.add_argument("--camera-csv", required=True, help="Path to camera.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="camera_transformer.pt")
    args = ap.parse_args()
    # Optional cap to keep quick training
    limit_steps = int(os.environ.get("CAMTRAIN_LIMIT_STEPS", "0")) or None

    ds = CameraPanZoomDataset(
        tracking_csv=args.tracking_csv, camera_csv=args.camera_csv, window=args.window
    )
    n = len(ds)
    n_val = max(1, int(n * args.val_split))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample0 = ds[0]
    model = CameraPanZoomTransformer(d_in=sample0["x"].shape[-1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 1e9
    for epoch in range(args.epochs):
        tr_loss, seen = train_one_epoch(model, train_loader, opt, device, limit_steps=limit_steps)
        va = eval_loss(model, val_loader, device, limit_steps=limit_steps)
        logger.info(
            "epoch %d: train %.4f val %.4f seen=%d",
            epoch + 1,
            tr_loss,
            va,
            seen,
        )
        if va < best_val:
            best_val = va
            ckpt = pack_checkpoint(model, norm=ds.norm, window=args.window)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(ckpt, args.out)
            logger.info("Saved checkpoint to %s", args.out)


if __name__ == "__main__":
    main()

import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, pack_gpt_checkpoint
from hmlib.camera.camera_gpt_dataset import (
    CameraPanZoomGPTIterableDataset,
    GameCsvPaths,
    resolve_csv_paths,
    scan_game_max_xy,
)
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.log import logger


def _split_and_strip(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _load_game_ids(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    for item in args.game_id or []:
        out.extend(_split_and_strip(item))
    out.extend(_split_and_strip(args.game_ids))
    if args.game_ids_file:
        p = Path(args.game_ids_file)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    # de-dupe, keep order
    seen = set()
    dedup = []
    for gid in out:
        if gid in seen:
            continue
        seen.add(gid)
        dedup.append(gid)
    return dedup


@torch.no_grad()
def _eval_loss(
    model: nn.Module, loader: DataLoader, device: torch.device, steps: int
) -> float:
    model.eval()
    total = 0
    loss_sum = 0.0
    it = iter(loader)
    for _ in range(int(steps)):
        batch = next(it)
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = nn.functional.l1_loss(pred, y)
        loss_sum += float(loss.item()) * x.size(0)
        total += x.size(0)
    return loss_sum / max(1, total)


def main():
    ap = argparse.ArgumentParser("Train GPT camera model from saved tracking/camera CSVs")
    ap.add_argument("--game-id", action="append", default=[], help="Game id (repeatable)")
    ap.add_argument("--game-ids", type=str, default=None, help="Comma-separated game ids")
    ap.add_argument("--game-ids-file", type=str, default=None, help="Text file with game ids")
    ap.add_argument(
        "--videos-root",
        type=str,
        default=str(Path(os.environ.get("HOME", "")) / "Videos"),
        help="Root directory containing <game-id>/ directories (default: $HOME/Videos)",
    )
    ap.add_argument("--seq-len", type=int, default=32, help="Sequence length in frames")
    ap.add_argument("--steps", type=int, default=2000, help="Training optimizer steps")
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Optional training budget in frames (overrides --steps when >0)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-players", type=int, default=22)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--val-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", type=str, default="camera_gpt.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-cached-games", type=int, default=8)
    args = ap.parse_args()

    game_ids = _load_game_ids(args)
    if not game_ids:
        raise SystemExit("No game ids provided (use --game-id/--game-ids/--game-ids-file)")

    videos_root = Path(args.videos_root)
    if not videos_root.is_dir():
        raise SystemExit(f"videos_root does not exist: {videos_root}")

    game_csvs: List[GameCsvPaths] = []
    for gid in game_ids:
        gdir = videos_root / gid
        if not gdir.is_dir():
            logger.warning("Skipping %s (missing dir %s)", gid, gdir)
            continue
        paths = resolve_csv_paths(game_id=gid, game_dir=str(gdir))
        if paths is None:
            logger.warning("Skipping %s (missing tracking.csv/camera.csv)", gid)
            continue
        game_csvs.append(paths)

    if not game_csvs:
        raise SystemExit("No usable games found (need tracking.csv + camera.csv).")

    # Use one normalization scale across all games for consistent training.
    max_x = 0.0
    max_y = 0.0
    for p in game_csvs:
        mx, my = scan_game_max_xy(p.tracking_csv, p.camera_csv)
        max_x = max(max_x, float(mx))
        max_y = max(max_y, float(my))
    norm = CameraNorm(scale_x=max_x, scale_y=max_y, max_players=int(args.max_players))

    rng = random.Random(int(args.seed))
    rng.shuffle(game_csvs)
    n_val = int(round(len(game_csvs) * float(args.val_split)))
    val_games = game_csvs[:n_val] if n_val > 0 else []
    train_games = game_csvs[n_val:] if n_val > 0 else game_csvs
    if not train_games:
        train_games = game_csvs
        val_games = []

    train_ds = CameraPanZoomGPTIterableDataset(
        games=train_games,
        norm=norm,
        seq_len=int(args.seq_len),
        max_players_for_norm=int(args.max_players),
        seed=int(args.seed),
        max_cached_games=int(args.max_cached_games),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), num_workers=0)

    val_loader = None
    if val_games and int(args.val_steps) > 0:
        val_ds = CameraPanZoomGPTIterableDataset(
            games=val_games,
            norm=norm,
            seq_len=int(args.seq_len),
            max_players_for_norm=int(args.max_players),
            seed=int(args.seed) + 999,
            max_cached_games=max(1, int(args.max_cached_games // 2)),
        )
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), num_workers=0)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = CameraGPTConfig(
        d_in=11,
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        nlayers=int(args.nlayers),
        dropout=float(args.dropout),
    )
    model = CameraPanZoomGPT(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))

    steps = int(args.steps)
    if int(args.frames) > 0:
        steps = int(math.ceil(float(args.frames) / float(args.batch_size * args.seq_len)))
    logger.info(
        "Training camGPT: games=%d train=%d val=%d seq_len=%d steps=%d bs=%d device=%s",
        len(game_csvs),
        len(train_games),
        len(val_games),
        int(args.seq_len),
        steps,
        int(args.batch_size),
        device,
    )

    it = iter(train_loader)
    best_val = float("inf")
    for step in range(1, steps + 1):
        model.train()
        batch = next(it)
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        pred = model(x)
        loss = nn.functional.l1_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % int(args.log_every) == 0 or step == 1:
            logger.info("step %d/%d: train_l1=%.5f", step, steps, float(loss.item()))

        do_eval = val_loader is not None and int(args.eval_every) > 0 and (
            step % int(args.eval_every) == 0 or step == steps
        )
        if do_eval:
            val = _eval_loss(model, val_loader, device, steps=int(args.val_steps))
            logger.info("step %d/%d: val_l1=%.5f", step, steps, float(val))
            if val < best_val:
                best_val = float(val)
                ckpt = pack_gpt_checkpoint(model, norm=train_ds.norm, window=int(args.seq_len), cfg=cfg)
                ckpt["step"] = int(step)
                ckpt["games"] = [p.game_id for p in game_csvs]
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                torch.save(ckpt, args.out)
                logger.info("Saved best checkpoint to %s", args.out)

    if val_loader is None:
        ckpt = pack_gpt_checkpoint(model, norm=train_ds.norm, window=int(args.seq_len), cfg=cfg)
        ckpt["step"] = int(steps)
        ckpt["games"] = [p.game_id for p in game_csvs]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(ckpt, args.out)
        logger.info("Saved checkpoint to %s", args.out)


if __name__ == "__main__":
    main()


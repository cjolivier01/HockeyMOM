import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from hmlib.camera.camera_gpt import CameraGPTConfig, CameraPanZoomGPT, pack_gpt_checkpoint
from hmlib.camera.camera_gpt_dataset import (
    CameraPanZoomGPTIterableDataset,
    GameCsvPaths,
    resolve_csv_paths,
    scan_game_max_xy,
    validate_csv_paths,
)
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.log import logger


def _tlwh_to_tlbr(tlwh: torch.Tensor) -> torch.Tensor:
    x1 = tlwh[..., 0]
    y1 = tlwh[..., 1]
    w = torch.clamp(tlwh[..., 2], min=1e-6)
    h = torch.clamp(tlwh[..., 3], min=1e-6)
    x2 = x1 + w
    y2 = y1 + h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _mean_iou_tlwh(a_tlwh: torch.Tensor, b_tlwh: torch.Tensor) -> torch.Tensor:
    a = _tlwh_to_tlbr(a_tlwh)
    b = _tlwh_to_tlbr(b_tlwh)
    ix1 = torch.maximum(a[..., 0], b[..., 0])
    iy1 = torch.maximum(a[..., 1], b[..., 1])
    ix2 = torch.minimum(a[..., 2], b[..., 2])
    iy2 = torch.minimum(a[..., 3], b[..., 3])
    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih
    a_area = torch.clamp(a[..., 2] - a[..., 0], min=0.0) * torch.clamp(
        a[..., 3] - a[..., 1], min=0.0
    )
    b_area = torch.clamp(b[..., 2] - b[..., 0], min=0.0) * torch.clamp(
        b[..., 3] - b[..., 1], min=0.0
    )
    union = a_area + b_area - inter
    iou = inter / torch.clamp(union, min=1e-6)
    return iou.mean()


def _diff1(x: torch.Tensor) -> torch.Tensor:
    # [B,T,D] -> [B,T-1,D]
    return x[:, 1:, :] - x[:, :-1, :]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    # [B,T,D] -> [B,T-2,D]
    return x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]


def _compute_losses(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    w_l1: float,
    w_iou: float,
    w_vel: float,
    w_acc: float,
    fast_mult: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    d_out = int(y.shape[-1])

    if d_out == 8:
        pred_slow = pred[..., :4]
        y_slow = y[..., :4]
        pred_fast = pred[..., 4:8]
        y_fast = y[..., 4:8]

        l1_slow = nn.functional.l1_loss(pred_slow, y_slow)
        l1_fast = nn.functional.l1_loss(pred_fast, y_fast)
        l1 = l1_slow + float(fast_mult) * l1_fast
        metrics["l1_slow"] = float(l1_slow.detach().item())
        metrics["l1_fast"] = float(l1_fast.detach().item())

        iou_slow = _mean_iou_tlwh(pred_slow, y_slow)
        iou_fast = _mean_iou_tlwh(pred_fast, y_fast)
        iou_loss = (1.0 - iou_slow) + float(fast_mult) * (1.0 - iou_fast)
        metrics["iou_slow"] = float(iou_slow.detach().item())
        metrics["iou_fast"] = float(iou_fast.detach().item())

        vel_slow = (
            nn.functional.l1_loss(_diff1(pred_slow), _diff1(y_slow))
            if pred.size(1) > 1
            else pred.new_zeros(())
        )
        vel_fast = (
            nn.functional.l1_loss(_diff1(pred_fast), _diff1(y_fast))
            if pred.size(1) > 1
            else pred.new_zeros(())
        )
        vel = vel_slow + float(fast_mult) * vel_fast
        metrics["vel_slow"] = float(vel_slow.detach().item()) if pred.size(1) > 1 else 0.0
        metrics["vel_fast"] = float(vel_fast.detach().item()) if pred.size(1) > 1 else 0.0

        acc_slow = (
            nn.functional.l1_loss(_diff2(pred_slow), _diff2(y_slow))
            if pred.size(1) > 2
            else pred.new_zeros(())
        )
        acc_fast = (
            nn.functional.l1_loss(_diff2(pred_fast), _diff2(y_fast))
            if pred.size(1) > 2
            else pred.new_zeros(())
        )
        acc = acc_slow + float(fast_mult) * acc_fast
        metrics["acc_slow"] = float(acc_slow.detach().item()) if pred.size(1) > 2 else 0.0
        metrics["acc_fast"] = float(acc_fast.detach().item()) if pred.size(1) > 2 else 0.0

    else:
        l1 = nn.functional.l1_loss(pred, y)
        metrics["l1"] = float(l1.detach().item())
        iou_loss = pred.new_zeros(())
        if d_out == 4:
            iou = _mean_iou_tlwh(pred, y)
            iou_loss = 1.0 - iou
            metrics["iou"] = float(iou.detach().item())
        vel = (
            nn.functional.l1_loss(_diff1(pred), _diff1(y))
            if pred.size(1) > 1
            else pred.new_zeros(())
        )
        acc = (
            nn.functional.l1_loss(_diff2(pred), _diff2(y))
            if pred.size(1) > 2
            else pred.new_zeros(())
        )
        metrics["vel"] = float(vel.detach().item()) if pred.size(1) > 1 else 0.0
        metrics["acc"] = float(acc.detach().item()) if pred.size(1) > 2 else 0.0

    loss = float(w_l1) * l1 + float(w_iou) * iou_loss + float(w_vel) * vel + float(w_acc) * acc
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics


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


def _arg_in_argv(flag: str) -> bool:
    try:
        return flag in sys.argv
    except Exception:
        return False


def _load_game_dirs_from_list(list_path: str) -> List[Path]:
    p = Path(list_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"file list does not exist: {p}")
    root = p.parent
    out: List[Path] = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        d = Path(line).expanduser()
        if not d.is_absolute():
            d = root / d
        out.append(d)
    # de-dupe, keep order
    seen: set[str] = set()
    dedup: List[Path] = []
    for d in out:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(d)
    return dedup


def _parse_checkpoint_naming(out: str) -> Tuple[Path, str, str]:
    """Return (best_path, prefix, ext)."""
    best_path = Path(out)
    ext = "".join(best_path.suffixes) or ".pt"
    name = best_path.name
    stem = name[: -len(ext)] if name.endswith(ext) else best_path.stem
    prefix = stem
    if prefix.endswith("_best"):
        prefix = prefix[: -len("_best")]
    if not prefix:
        prefix = "camera_gpt"
    if not stem.endswith("_best"):
        best_path = best_path.with_name(f"{prefix}_best{ext}")
    return best_path, prefix, ext


def _checkpoint_path(best_path: Path, prefix: str, ext: str, step_num: int) -> Path:
    return best_path.with_name(f"{prefix}_{int(step_num)}{ext}")


def _list_numbered_checkpoints(best_path: Path, prefix: str, ext: str) -> List[tuple[int, Path]]:
    out: List[tuple[int, Path]] = []
    pat = f"{prefix}_"
    for p in best_path.parent.glob(f"{prefix}_*{ext}"):
        if p.name == best_path.name:
            continue
        name = p.name
        if not name.startswith(pat) or not name.endswith(ext):
            continue
        mid = name[len(pat) : -len(ext)]
        if not mid.isdigit():
            continue
        out.append((int(mid), p))
    out.sort(key=lambda t: t[0])
    return out


def _checkpoint_compatible(ckpt: dict, cfg: CameraGPTConfig) -> bool:
    try:
        m = ckpt.get("model") or {}
        return (
            int(m.get("d_in")) == int(cfg.d_in)
            and int(m.get("d_out")) == int(cfg.d_out)
            and str(m.get("feature_mode")) == str(cfg.feature_mode)
            and bool(m.get("include_pose")) == bool(cfg.include_pose)
            and int(m.get("d_model")) == int(cfg.d_model)
            and int(m.get("nhead")) == int(cfg.nhead)
            and int(m.get("nlayers")) == int(cfg.nlayers)
        )
    except Exception:
        return False


def _maybe_resume(
    *,
    best_path: Path,
    prefix: str,
    ext: str,
    resume_mode: str,
    device: torch.device,
    model: nn.Module,
    opt: optim.Optimizer,
    cfg: CameraGPTConfig,
) -> int:
    if resume_mode == "none":
        return 1

    numbered = _list_numbered_checkpoints(best_path, prefix, ext)
    latest = numbered[-1][1] if numbered else (best_path if best_path.is_file() else None)

    if latest is None:
        if resume_mode == "force":
            raise SystemExit(f"--resume specified but no checkpoint found for prefix={prefix}")
        return 1

    ckpt = torch.load(str(latest), map_location="cpu")
    if not _checkpoint_compatible(ckpt, cfg):
        msg = f"Checkpoint incompatible with current model cfg: {latest}"
        if resume_mode == "force":
            raise SystemExit(msg)
        logger.warning("%s (skipping resume)", msg)
        return 1

    model.load_state_dict(ckpt["state_dict"])
    opt_sd = ckpt.get("optimizer")
    if opt_sd:
        try:
            opt.load_state_dict(opt_sd)
        except Exception as ex:
            logger.warning("Failed to load optimizer state from %s: %s", latest, ex)
    model.to(device)
    step0 = int(ckpt.get("step", 0) or 0)
    logger.info("Resumed from %s (step=%d)", latest, step0)
    return max(1, step0 + 1)


def _save_training_checkpoint(
    *,
    path: Path,
    step_num: int,
    model: nn.Module,
    opt: optim.Optimizer,
    train_ds: CameraPanZoomGPTIterableDataset,
    cfg: CameraGPTConfig,
    game_csvs: List[GameCsvPaths],
    target_mode: str,
    include_pose: bool,
) -> None:
    ckpt = pack_gpt_checkpoint(model, norm=train_ds.norm, window=int(train_ds._seq_len), cfg=cfg)
    ckpt["step"] = int(step_num)
    ckpt["games"] = [p.game_id for p in game_csvs]
    ckpt["target_mode"] = str(target_mode)
    ckpt["include_pose"] = bool(include_pose)
    ckpt["optimizer"] = opt.state_dict()
    os.makedirs(str(path.parent), exist_ok=True)
    torch.save(ckpt, str(path))


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    *,
    w_l1: float,
    w_iou: float,
    w_vel: float,
    w_acc: float,
    fast_mult: float,
) -> float:
    model.eval()
    total = 0
    loss_sum = 0.0
    it = iter(loader)
    for _ in range(int(steps)):
        batch = next(it)
        y = batch["y"].to(device)
        if "x" in batch:
            x = batch["x"].to(device)
        else:
            base = batch["base"].to(device)
            prev0 = batch["prev0"].to(device)
            prev_y = torch.cat([prev0.unsqueeze(1), y[:, :-1, :]], dim=1)
            x = torch.cat([base, prev_y], dim=-1)
        pred = model(x)
        loss, _ = _compute_losses(
            pred,
            y,
            w_l1=w_l1,
            w_iou=w_iou,
            w_vel=w_vel,
            w_acc=w_acc,
            fast_mult=fast_mult,
        )
        loss_sum += float(loss.item()) * int(x.size(0))
        total += int(x.size(0))
    return loss_sum / max(1, total)


def main():
    ap = argparse.ArgumentParser("Train GPT camera model from saved tracking/camera CSVs")
    ap.add_argument("--game-id", action="append", default=[], help="Game id (repeatable)")
    ap.add_argument("--game-ids", type=str, default=None, help="Comma-separated game ids")
    ap.add_argument("--game-ids-file", type=str, default=None, help="Text file with game ids")
    ap.add_argument(
        "--file-list",
        type=str,
        default=None,
        help=(
            "Path to a .lst text file containing game directories. Each line should be either an "
            "absolute path or a path relative to the .lst file's directory. Each game directory "
            "must contain tracking.csv and camera.csv (pose.csv optional). When set, this overrides "
            "--game-id/--game-ids/--game-ids-file."
        ),
    )
    ap.add_argument(
        "--videos-root",
        type=str,
        default=str(Path(os.environ.get("HOME", "")) / "Videos"),
        help="Root directory containing <game-id>/ directories (default: $HOME/Videos)",
    )
    ap.add_argument("--seq-len", type=int, default=32, help="Sequence length in frames")
    ap.add_argument(
        "--sample-seconds",
        type=float,
        default=3.0,
        help="Window length to sample (seconds). Used when --file-list is set and --seq-len isn't provided.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS used to convert --sample-seconds into frames (used only for --file-list mode).",
    )
    ap.add_argument(
        "--target-mode",
        type=str,
        default="slow_fast_tlwh",
        choices=["slow_center_h", "slow_tlwh", "slow_fast_tlwh"],
        help="Training target format (slow camera only or slow+fast boxes).",
    )
    ap.add_argument(
        "--feature-mode",
        type=str,
        default="base_prev_y",
        choices=["legacy_prev_slow", "base_prev_y"],
        help=(
            "Input feature schema. 'base_prev_y' feeds previous camera target/prediction as input "
            "(enables scheduled sampling and improves fast-box dynamics)."
        ),
    )
    ap.add_argument(
        "--no-pose",
        dest="include_pose",
        action="store_false",
        default=True,
        help="Disable pose.csv feature aggregation even if pose.csv exists.",
    )
    ap.add_argument(
        "--include-rink",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Include rink features derived from rink_mask_0.png / rink_profile (requires retraining). "
            "Use --no-include-rink to disable."
        ),
    )
    ap.add_argument(
        "--scheduled-sampling",
        action="store_true",
        help="Enable scheduled sampling (only when --feature-mode=base_prev_y).",
    )
    ap.add_argument(
        "--ss-prob-start", type=float, default=0.0, help="Scheduled-sampling prob at step 0."
    )
    ap.add_argument(
        "--ss-prob-end", type=float, default=0.3, help="Scheduled-sampling prob after warmup."
    )
    ap.add_argument(
        "--ss-warmup-steps",
        type=int,
        default=1000,
        help="Linearly ramp scheduled-sampling prob over this many steps (0=immediate).",
    )
    ap.add_argument("--steps", type=int, default=2000, help="Training optimizer steps")
    ap.add_argument(
        "--max-iters",
        dest="max_iters",
        type=int,
        default=None,
        help="Alias for --steps (max training iterations).",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Optional training budget in frames (overrides --steps when >0)",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--data-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 runs data pipeline in the main process).",
    )
    ap.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory (useful for GPU training).",
    )
    ap.add_argument(
        "--preload-csv",
        type=str,
        default="none",
        choices=["none", "shard", "all"],
        help=(
            "Preload CSVs into each dataset worker process. "
            "'shard' loads only the worker's slice of games; 'all' loads all games in every worker."
        ),
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--nlayers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss-l1-weight", type=float, default=1.0, help="Weight for L1 loss.")
    ap.add_argument(
        "--loss-iou-weight",
        type=float,
        default=0.5,
        help="Weight for (1-IoU) box loss (TLWH modes only).",
    )
    ap.add_argument(
        "--loss-vel-weight",
        type=float,
        default=0.2,
        help="Weight for velocity matching loss (L1 on first differences).",
    )
    ap.add_argument(
        "--loss-acc-weight",
        type=float,
        default=0.1,
        help="Weight for acceleration matching loss (L1 on second differences).",
    )
    ap.add_argument(
        "--fast-loss-mult",
        type=float,
        default=2.0,
        help="Multiplier for fast-box losses when training slow_fast_tlwh.",
    )
    ap.add_argument("--max-players", type=int, default=22)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--val-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help=(
            "Save a training checkpoint every N steps (0=disabled). "
            "Checkpoints are saved alongside --out as <prefix>_<step>.pt (best is <prefix>_best.pt)."
        ),
    )
    ap.add_argument(
        "--max-checkpoints",
        type=int,
        default=10,
        help="Keep only the last N numbered checkpoints (0=keep all). Best checkpoint is kept.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if found (or error if none found).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume even if checkpoints are present.",
    )
    ap.add_argument(
        "--camera-csv-name",
        type=str,
        default=None,
        help="Optional fixed camera CSV filename inside each <videos-root>/<game-id>/ (e.g. camera_annotated.csv).",
    )
    ap.add_argument(
        "--camera-fast-csv-name",
        type=str,
        default=None,
        help="Optional fixed fast camera CSV filename inside each <videos-root>/<game-id>/ (e.g. camera_fast_annotated.csv).",
    )
    ap.add_argument(
        "--pose-csv-name",
        type=str,
        default=None,
        help="Optional fixed pose CSV filename inside each <videos-root>/<game-id>/ (e.g. pose.csv).",
    )
    ap.add_argument("--out", type=str, default="camera_gpt_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-cached-games", type=int, default=8)
    ap.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Optional cap on number of games loaded from --file-list or ids (0=no cap).",
    )
    args = ap.parse_args()

    # --max-iters is an alias for --steps. If both are provided explicitly, require they match.
    if args.max_iters is not None:
        steps_in_argv = _arg_in_argv("--steps")
        max_iters_in_argv = _arg_in_argv("--max-iters")
        if steps_in_argv and max_iters_in_argv and int(args.steps) != int(args.max_iters):
            raise SystemExit("--steps and --max-iters both provided but differ; please use one.")
        if (not steps_in_argv) and max_iters_in_argv:
            args.steps = int(args.max_iters)

    resume_mode = "auto"
    if bool(args.no_resume):
        resume_mode = "none"
    elif bool(args.resume):
        resume_mode = "force"

    game_csvs: List[GameCsvPaths] = []

    if args.file_list:
        # Default to seconds-based sampling unless --seq-len was explicitly provided.
        if not _arg_in_argv("--seq-len"):
            fps = max(1.0, float(args.fps))
            secs = max(0.1, float(args.sample_seconds))
            args.seq_len = int(round(secs * fps))

        game_dirs = _load_game_dirs_from_list(args.file_list)
        if int(args.max_games) > 0:
            game_dirs = game_dirs[: int(args.max_games)]
        if not game_dirs:
            raise SystemExit(f"No game directories found in {args.file_list}")

        for game_dir in game_dirs:
            if not game_dir.is_dir():
                logger.warning("Skipping missing game dir: %s", game_dir)
                continue
            paths = resolve_csv_paths(
                game_id=game_dir.name,
                game_dir=str(game_dir),
                camera_csv_name=args.camera_csv_name,
                camera_fast_csv_name=args.camera_fast_csv_name,
                pose_csv_name=args.pose_csv_name,
            )
            if paths is None:
                logger.warning("Skipping %s (missing tracking.csv/camera.csv)", game_dir)
                continue
            if args.target_mode == "slow_fast_tlwh" and not paths.camera_fast_csv:
                logger.warning(
                    "Skipping %s (missing camera_fast.csv for --target-mode=slow_fast_tlwh)",
                    game_dir,
                )
                continue
            try:
                validate_csv_paths(paths)
            except Exception as ex:
                logger.warning("Skipping %s (bad CSV paths): %s", game_dir, ex)
                continue
            game_csvs.append(paths)
    else:
        game_ids = _load_game_ids(args)
        if not game_ids:
            raise SystemExit("No game ids provided (use --game-id/--game-ids/--game-ids-file)")

        videos_root = Path(args.videos_root)
        if not videos_root.is_dir():
            raise SystemExit(f"videos_root does not exist: {videos_root}")

        if int(args.max_games) > 0:
            game_ids = game_ids[: int(args.max_games)]

        for gid in game_ids:
            gdir = videos_root / gid
            if not gdir.is_dir():
                logger.warning("Skipping %s (missing dir %s)", gid, gdir)
                continue
            paths = resolve_csv_paths(
                game_id=gid,
                game_dir=str(gdir),
                camera_csv_name=args.camera_csv_name,
                camera_fast_csv_name=args.camera_fast_csv_name,
                pose_csv_name=args.pose_csv_name,
            )
            if paths is None:
                logger.warning("Skipping %s (missing tracking.csv/camera.csv)", gid)
                continue
            if args.target_mode == "slow_fast_tlwh" and not paths.camera_fast_csv:
                logger.warning(
                    "Skipping %s (missing camera_fast.csv for --target-mode=slow_fast_tlwh)", gid
                )
                continue
            try:
                validate_csv_paths(paths)
            except Exception as ex:
                logger.warning("Skipping %s (bad CSV paths): %s", gid, ex)
                continue
            game_csvs.append(paths)

    if not game_csvs:
        raise SystemExit(
            "No usable games found (need tracking.csv + camera.csv [+ camera_fast.csv])."
        )

    # Use one normalization scale across all games for consistent training.
    max_x = 0.0
    max_y = 0.0
    for p in game_csvs:
        mx, my = scan_game_max_xy(p.tracking_csv, p.camera_csv, p.camera_fast_csv)
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
        target_mode=str(args.target_mode),
        feature_mode=str(args.feature_mode),
        include_pose=bool(args.include_pose),
        include_rink=bool(args.include_rink),
        max_players_for_norm=int(args.max_players),
        seed=int(args.seed),
        max_cached_games=int(args.max_cached_games),
        preload_csv=str(args.preload_csv),
        shard_games_by_worker=(int(args.data_workers) > 1),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.data_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=int(args.data_workers) > 0,
    )

    val_loader = None
    if val_games and int(args.val_steps) > 0:
        val_ds = CameraPanZoomGPTIterableDataset(
            games=val_games,
            norm=norm,
            seq_len=int(args.seq_len),
            target_mode=str(args.target_mode),
            feature_mode=str(args.feature_mode),
            include_pose=bool(args.include_pose),
            include_rink=bool(args.include_rink),
            max_players_for_norm=int(args.max_players),
            seed=int(args.seed) + 999,
            max_cached_games=max(1, int(args.max_cached_games // 2)),
            preload_csv=str(args.preload_csv),
            shard_games_by_worker=(int(args.data_workers) > 1),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            num_workers=int(args.data_workers),
            pin_memory=bool(args.pin_memory),
            persistent_workers=int(args.data_workers) > 0,
        )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = CameraGPTConfig(
        d_in=int(train_ds.feature_dim),
        d_out=int(train_ds.target_dim),
        feature_mode=str(args.feature_mode),
        include_pose=bool(args.include_pose),
        include_rink=bool(args.include_rink),
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        nlayers=int(args.nlayers),
        dropout=float(args.dropout),
    )
    model = CameraPanZoomGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %.2fM", float(n_params) / 1e6)
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
    best_path, prefix, ext = _parse_checkpoint_naming(args.out)
    start_step = _maybe_resume(
        best_path=best_path,
        prefix=prefix,
        ext=ext,
        resume_mode=resume_mode,
        device=device,
        model=model,
        opt=opt,
        cfg=cfg,
    )

    for step in range(start_step, steps + 1):
        model.train()
        batch = next(it)
        y = batch["y"].to(device)
        if "x" in batch:
            x = batch["x"].to(device)
            pred = model(x)
        else:
            base = batch["base"].to(device)
            prev0 = batch["prev0"].to(device)
            if str(args.feature_mode) != "base_prev_y":
                raise RuntimeError(
                    "Dataset emitted base/prev0 but feature-mode is not base_prev_y; this is a bug."
                )

            # Scheduled sampling: sometimes feed the model's previous prediction as the next-step input.
            if args.scheduled_sampling:
                p0 = float(args.ss_prob_start)
                p1 = float(args.ss_prob_end)
                warm = int(args.ss_warmup_steps)
                if warm <= 0:
                    p = p1
                else:
                    frac = float(min(max(step, 0), warm)) / float(warm)
                    p = p0 + (p1 - p0) * frac
                bsz, tlen, _ = base.shape
                prev = prev0
                preds = []
                x_prefix = None
                for t in range(int(tlen)):
                    x_t = torch.cat([base[:, t, :], prev], dim=-1).unsqueeze(1)  # [B,1,D]
                    x_prefix = x_t if x_prefix is None else torch.cat([x_prefix, x_t], dim=1)
                    pred_t = model(x_prefix)[:, -1, :]
                    preds.append(pred_t)
                    if t + 1 < int(tlen):
                        use_pred = torch.rand((bsz,), device=device) < float(p)
                        prev = torch.where(use_pred[:, None], pred_t.detach(), y[:, t, :])
                pred = torch.stack(preds, dim=1)
                x = x_prefix
            else:
                prev_y = torch.cat([prev0.unsqueeze(1), y[:, :-1, :]], dim=1)
                x = torch.cat([base, prev_y], dim=-1)
                pred = model(x)
        loss, metrics = _compute_losses(
            pred,
            y,
            w_l1=float(args.loss_l1_weight),
            w_iou=float(args.loss_iou_weight),
            w_vel=float(args.loss_vel_weight),
            w_acc=float(args.loss_acc_weight),
            fast_mult=float(args.fast_loss_mult),
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if int(args.checkpoint_every) > 0 and (step % int(args.checkpoint_every) == 0):
            ckpt_path = _checkpoint_path(best_path, prefix, ext, step)
            _save_training_checkpoint(
                path=ckpt_path,
                step_num=step,
                model=model,
                opt=opt,
                train_ds=train_ds,
                cfg=cfg,
                game_csvs=game_csvs,
                target_mode=str(args.target_mode),
                include_pose=bool(args.include_pose),
            )
            logger.info("Saved checkpoint to %s", ckpt_path)
            max_keep = int(args.max_checkpoints)
            if max_keep > 0:
                numbered = _list_numbered_checkpoints(best_path, prefix, ext)
                if len(numbered) > max_keep:
                    for _, p in numbered[: len(numbered) - max_keep]:
                        try:
                            p.unlink()
                        except Exception:
                            pass

        if step % int(args.log_every) == 0 or step == 1:
            msg = f"step {step}/{steps}: train_loss={metrics['loss']:.5f}"
            if "l1_slow" in metrics:
                msg += f" l1_slow={metrics['l1_slow']:.5f} l1_fast={metrics['l1_fast']:.5f}"
                msg += f" iou_slow={metrics['iou_slow']:.4f} iou_fast={metrics['iou_fast']:.4f}"
            else:
                msg += f" l1={metrics.get('l1', float('nan')):.5f}"
                if "iou" in metrics:
                    msg += f" iou={metrics['iou']:.4f}"
            logger.info(msg)

        do_eval = (
            val_loader is not None
            and int(args.eval_every) > 0
            and (step % int(args.eval_every) == 0 or step == steps)
        )
        if do_eval:
            val = _eval_loss(
                model,
                val_loader,
                device,
                steps=int(args.val_steps),
                w_l1=float(args.loss_l1_weight),
                w_iou=float(args.loss_iou_weight),
                w_vel=float(args.loss_vel_weight),
                w_acc=float(args.loss_acc_weight),
                fast_mult=float(args.fast_loss_mult),
            )
            logger.info("step %d/%d: val_loss=%.5f", step, steps, float(val))
            if val < best_val:
                best_val = float(val)
                _save_training_checkpoint(
                    path=best_path,
                    step_num=step,
                    model=model,
                    opt=opt,
                    train_ds=train_ds,
                    cfg=cfg,
                    game_csvs=game_csvs,
                    target_mode=str(args.target_mode),
                    include_pose=bool(args.include_pose),
                )
                logger.info("Saved best checkpoint to %s", best_path)

    if val_loader is None:
        _save_training_checkpoint(
            path=best_path,
            step_num=steps,
            model=model,
            opt=opt,
            train_ds=train_ds,
            cfg=cfg,
            game_csvs=game_csvs,
            target_mode=str(args.target_mode),
            include_pose=bool(args.include_pose),
        )
        logger.info("Saved checkpoint to %s", best_path)


if __name__ == "__main__":
    main()

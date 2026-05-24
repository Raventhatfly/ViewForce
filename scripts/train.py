"""
scripts/train.py  --  Train the ViewForce force prediction network.

Usage:
    python scripts/train.py --data-dir data/test_data_4_21_1

Example with options:
    python scripts/train.py --data-dir data/test_data_4_21_1 \\
                            --trim-seconds 3.0 \\
                            --force-keys Fy Fx \\
                            --epochs 100 --output checkpoints/
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from src.dataset import make_datasets
from src.model.unet import build_unet


INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_edge": 4,
    "edge": 1,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=None,
        help="Dataset folder containing EP* episode subdirectories "
             "(e.g. data/test_data_4_21_1). Mutually exclusive with --episodes."
    )
    parser.add_argument(
        "--episodes", nargs="+", default=None,
        help="Explicit list of episode directories (overrides --data-dir)."
    )
    parser.add_argument(
        "--val-episode", default=None,
        help="Single episode directory used for validation. Overrides --val-count."
    )
    parser.add_argument(
        "--val-count", type=int, default=5,
        help="Number of final sorted episodes to hold out for validation "
             "when --val-episode is not set (default: 5)."
    )
    parser.add_argument(
        "--val-multiple", type=int, default=None,
        help="Hold out episodes whose EP number is a multiple of this value "
             "(e.g. 10 holds out EP000010, EP000020, ...). Overrides --val-count."
    )
    parser.add_argument(
        "--trim-seconds", type=float, default=2.0,
        help="Seconds to drop from the start and end of each episode (default: 2.0)."
    )
    parser.add_argument("--force-keys", nargs="+", default=["Fz"],
                        help="Force columns to predict (e.g. Fy or Fy Fx).")
    parser.add_argument("--input-mode", choices=INPUT_CHANNELS.keys(), default="rgb",
                        help="Model input channels: rgb, rgb_edge for RGB plus "
                             "a mask-interior Sobel edge channel, or edge for "
                             "Sobel edge only.")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--force-pooling", choices=["avg", "spatial"], default="spatial",
                        help="Force head pooling. 'spatial' keeps a coarse spatial grid "
                             "instead of globally averaging the bottleneck.")
    parser.add_argument("--force-spatial-size", type=int, default=4,
                        help="Grid size for --force-pooling spatial.")
    parser.add_argument("--output",       default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--save-every",   type=int,   default=5,
                        help="Save a periodic checkpoint every N epochs.")
    parser.add_argument("--keep-last",    type=int,   default=3,
                        help="Number of periodic checkpoints to keep.")
    parser.add_argument("--wandb-project", default="viewforce",
                        help="W&B project name.")
    parser.add_argument("--wandb-run",     default=None,
                        help="W&B run name (defaults to auto).")
    parser.add_argument("--no-wandb",      action="store_true",
                        help="Disable W&B logging.")
    return parser.parse_args()


def find_episodes(data_dir: str) -> list[str]:
    eps = sorted(
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if d.startswith("EP") and os.path.isdir(os.path.join(data_dir, d))
        and os.path.isfile(os.path.join(data_dir, d, "video.mp4"))
        and os.path.isfile(os.path.join(data_dir, d, "mask.mp4"))
    )
    return eps


def main():
    args = parse_args()

    # ---- Discover episodes -------------------------------------------------
    if args.episodes is not None:
        episodes = args.episodes
    elif args.data_dir is not None:
        episodes = find_episodes(args.data_dir)
        if not episodes:
            raise RuntimeError(
                f"No EP* folders with video.mp4 + mask.mp4 found under {args.data_dir}"
            )
    else:
        # Legacy default
        episodes = sorted(glob.glob("data/episodes/EP*"))
    if not episodes:
        raise RuntimeError("No episodes found. Pass --data-dir or --episodes.")
    print(f"Episodes ({len(episodes)}): {[os.path.basename(e) for e in episodes]}")

    # ---- Datasets ----------------------------------------------------------
    train_ds, val_ds = make_datasets(
        episodes,
        val_episode=args.val_episode,
        val_count=args.val_count,
        val_multiple=args.val_multiple,
        force_keys=tuple(args.force_keys),
        trim_seconds=args.trim_seconds,
        input_mode=args.input_mode,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ---- Model (encoder-only UNet) ----------------------------------------
    force_dim = len(args.force_keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_unet(
        in_channels=INPUT_CHANNELS[args.input_mode],
        encoder_channels=(32, 64, 128, 256),
        force_dim=force_dim,
        force_hidden_dim=256,
        force_dropout=0.3,
        force_pooling=args.force_pooling,
        force_spatial_size=args.force_spatial_size,
        encoder_only=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- W&B ------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.wandb_run}_{ts}" if args.wandb_run else f"{args.wandb_project}_{ts}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        wandb.run.summary["n_params"] = n_params

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Training loop ----------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    best_val_mae = float("inf")
    periodic_ckpts = []  # tracks saved periodic checkpoint paths (oldest first)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {epoch}", leave=False, unit="batch"):
            frame = batch["frame"].to(device)   # (B, 3, H, W)
            force = batch["force"].to(device)   # (B, force_dim)

            optimizer.zero_grad()
            pred = model(frame)
            loss = F.mse_loss(pred, force)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(frame)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frame = batch["frame"].to(device)
                force = batch["force"].to(device)
                pred  = model(frame)
                val_mae += (pred - force).abs().sum().item()
        val_mae /= len(val_ds)

        lr_now = scheduler.get_last_lr()[0]
        epoch_bar.set_postfix(train_mse=f"{train_loss:.4f}", val_mae=f"{val_mae:.4f} N", lr=f"{lr_now:.2e}")
        tqdm.write(f"Epoch {epoch:4d}/{args.epochs}  train_MSE={train_loss:.4f}  val_MAE={val_mae:.4f} N  lr={lr_now:.2e}")
        if use_wandb:
            wandb.log({
                "metrics/train_mse": train_loss,
                "metrics/val_mae":   val_mae,
                "lr":                lr_now,
            }, step=epoch)

        ckpt_payload = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "train_mse":    train_loss,
            "val_mae":      val_mae,
            "force_keys":   args.force_keys,
            "trim_seconds": args.trim_seconds,
            "val_episode":   args.val_episode,
            "val_count":     args.val_count,
            "args":         vars(args),
        }

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(ckpt_payload, os.path.join(args.output, "best.pt"))
            print(f"  -> saved best checkpoint (val_MAE={val_mae:.4f} N)")
            if use_wandb:
                wandb.run.summary["best_val_mae"] = val_mae
                wandb.run.summary["best_epoch"]   = epoch

        if args.save_every > 0 and epoch % args.save_every == 0:
            fname = f"ep{epoch:04d}_mse{train_loss:.4f}_mae{val_mae:.4f}.pt"
            ckpt_path = os.path.join(args.output, fname)
            torch.save(ckpt_payload, ckpt_path)
            print(f"  -> periodic checkpoint: {fname}")
            periodic_ckpts.append(ckpt_path)
            while len(periodic_ckpts) > args.keep_last:
                old = periodic_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)
                    print(f"  -> deleted old checkpoint: {os.path.basename(old)}")

    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f} N")
    print(f"Checkpoint: {os.path.join(args.output, 'best.pt')}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

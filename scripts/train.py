"""
scripts/train.py  --  Train the ViewForce force prediction network.

Usage:
    conda run -n view_force python scripts/train.py [options]

Example:
    python scripts/train.py --episodes data/episodes/EP000001 data/episodes/EP000002 \\
                            --val-episode data/episodes/EP000002 \\
                            --epochs 100 --output checkpoints/
"""

import argparse
import os
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import make_datasets
from src.model.unet import build_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", nargs="+", default=None,
        help="Episode directories. Defaults to all EP* under data/episodes/."
    )
    parser.add_argument(
        "--val-episode", default=None,
        help="Episode directory to use for validation (leave-one-out). "
             "Defaults to the last episode."
    )
    parser.add_argument("--force-keys", nargs="+", default=["Fy"],
                        help="Force columns to predict (e.g. Fy or Fy Fx).")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta",  type=float, default=0.5,
                        help="Delta for Huber loss (Newtons).")
    parser.add_argument("--output", default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Discover episodes -------------------------------------------------
    if args.episodes is None:
        args.episodes = sorted(glob.glob("data/episodes/EP*"))
    if not args.episodes:
        raise RuntimeError("No episode directories found. Pass --episodes explicitly.")
    print(f"Episodes: {args.episodes}")

    # ---- Datasets ----------------------------------------------------------
    train_ds, val_ds = make_datasets(
        args.episodes,
        val_episode=args.val_episode,
        force_keys=tuple(args.force_keys),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ---- Model -------------------------------------------------------------
    force_dim = len(args.force_keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_unet(
        in_channels=3,
        stress_out_channels=1,
        encoder_channels=(32, 64, 128, 256),
        force_dim=force_dim,
        force_hidden_dim=256,
        force_dropout=0.3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Training loop ----------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            diff  = batch["diff"].to(device)        # (B, 3, H, W)
            force = batch["force"].to(device)       # (B, force_dim)

            optimizer.zero_grad()
            _, pred = model(diff)
            loss = F.huber_loss(pred, force, delta=args.huber_delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(diff)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                diff  = batch["diff"].to(device)
                force = batch["force"].to(device)
                _, pred = model(diff)
                val_mae += (pred - force).abs().sum().item()
        val_mae /= len(val_ds)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_MAE={val_mae:.4f} N  lr={lr_now:.2e}")

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            ckpt_path = os.path.join(args.output, "best.pt")
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_mae":    val_mae,
                "force_keys": args.force_keys,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✓ saved best checkpoint (val_MAE={val_mae:.4f} N)")

    print(f"\nTraining complete. Best val MAE: {best_val_mae:.4f} N")
    print(f"Checkpoint: {os.path.join(args.output, 'best.pt')}")


if __name__ == "__main__":
    main()

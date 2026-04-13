"""
scripts/evaluate.py  --  Evaluate a trained ViewForce checkpoint on one episode.

Outputs:
  - Console: MAE, RMSE per force axis
  - Plot: predicted vs ground-truth force over time  →  <output_dir>/force_curve.png
  - (Optional) stress map overlays on sampled frames → <output_dir>/stress_frames/

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
                               --episode data/episodes/EP000002
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import ForceDataset
from src.model.unet import build_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode",    required=True,
                        help="Episode directory to evaluate on.")
    parser.add_argument("--output", default="eval_output",
                        help="Directory for output plots.")
    parser.add_argument("--stress-frames", type=int, default=8,
                        help="Number of frames to save with stress map overlay (0 = skip).")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint --------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    force_keys = ckpt["force_keys"]
    force_dim  = len(force_keys)
    print(f"Checkpoint from epoch {ckpt['epoch']}  val_MAE={ckpt['val_mae']:.4f} N")
    print(f"Force keys: {force_keys}")

    model = build_unet(
        in_channels=3,
        stress_out_channels=1,
        encoder_channels=(32, 64, 128, 256),
        force_dim=force_dim,
        force_hidden_dim=256,
        force_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- Dataset ----------------------------------------------------------
    print(f"Loading episode: {args.episode}")
    ds = ForceDataset(args.episode, force_keys=force_keys, augment=False)
    if len(ds) == 0:
        print("No valid frames found.")
        return

    # ---- Run inference ----------------------------------------------------
    all_pred   = []
    all_gt     = []
    all_fidx   = []
    all_stress = []

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            diff  = sample["diff"].unsqueeze(0).to(device)
            gt    = sample["force"].numpy()
            fidx  = sample["frame_idx"]

            stress_map, pred = model(diff)

            all_pred.append(pred.cpu().numpy()[0])
            all_gt.append(gt)
            all_fidx.append(fidx)
            all_stress.append(stress_map.cpu().numpy()[0])

    all_pred  = np.array(all_pred)
    all_gt    = np.array(all_gt)
    all_fidx  = np.array(all_fidx)

    # ---- Metrics ----------------------------------------------------------
    mae  = np.abs(all_pred - all_gt).mean(axis=0)
    rmse = np.sqrt(((all_pred - all_gt) ** 2).mean(axis=0))
    print("\n--- Results ---")
    for i, k in enumerate(force_keys):
        print(f"  {k}:  MAE={mae[i]:.4f} N   RMSE={rmse[i]:.4f} N")

    # ---- Time-series plot -------------------------------------------------
    n_axes = force_dim
    fig, axes = plt.subplots(n_axes, 1, figsize=(12, 3 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]

    for i, (ax, k) in enumerate(zip(axes, force_keys)):
        ax.plot(all_fidx, all_gt[:, i],   label="Ground truth", color="steelblue", linewidth=1.5)
        ax.plot(all_fidx, all_pred[:, i], label="Predicted",    color="tomato",    linewidth=1.5, linestyle="--")
        ax.set_ylabel(f"{k} (N)")
        ax.set_title(f"{k}  MAE={mae[i]:.4f} N  RMSE={rmse[i]:.4f} N")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"ViewForce — {os.path.basename(args.episode)}", fontsize=12)
    fig.tight_layout()
    curve_path = os.path.join(args.output, "force_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Force curve saved: {curve_path}")

    # ---- Stress map overlays ----------------------------------------------
    if args.stress_frames > 0:
        import av
        from PIL import Image as PILImage

        stress_dir = os.path.join(args.output, "stress_frames")
        os.makedirs(stress_dir, exist_ok=True)

        video_path = os.path.join(args.episode, "video.mp4")
        sample_positions = np.linspace(0, len(all_fidx) - 1, args.stress_frames, dtype=int)
        sample_fidx = set(all_fidx[sample_positions].tolist())

        raw_frames = {}
        container = av.open(video_path)
        for fi, frame in enumerate(container.decode(video=0)):
            if fi in sample_fidx:
                raw_frames[fi] = np.array(frame.to_image())
            if len(raw_frames) == len(sample_fidx):
                break
        container.close()

        for fi in sorted(raw_frames.keys()):
            pos = np.where(all_fidx == fi)[0]
            if len(pos) == 0:
                continue
            pos = pos[0]

            img_np    = raw_frames[fi]
            stress_np = all_stress[pos][0]
            stress_np = (stress_np - stress_np.min()) / (np.ptp(stress_np) + 1e-8)
            stress_up = np.array(
                PILImage.fromarray((stress_np * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR
                )
            ) / 255.0

            fig, axes2 = plt.subplots(1, 2, figsize=(10, 4))
            axes2[0].imshow(img_np)
            axes2[0].set_title(f"Frame {fi}")
            axes2[0].axis("off")

            axes2[1].imshow(img_np)
            axes2[1].imshow(stress_up, cmap="hot", alpha=0.5)
            gt_str   = "  ".join(f"{k}={all_gt[pos, j]:.3f}" for j, k in enumerate(force_keys))
            pred_str = "  ".join(f"{k}={all_pred[pos, j]:.3f}" for j, k in enumerate(force_keys))
            axes2[1].set_title(f"GT:[{gt_str}]  Pred:[{pred_str}]", fontsize=8)
            axes2[1].axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(stress_dir, f"frame_{fi:06d}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)

        print(f"Stress frames saved: {stress_dir}/")


if __name__ == "__main__":
    main()

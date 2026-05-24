"""
Preview ViewForce training inputs with occlusion augmentation.

Writes side-by-side videos:
  left:  clean masked input
  right: augmented training input
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2 as cv
import numpy as np
import torch

from src.dataset import ForceDataset


def tensor_to_rgb(frame_t: torch.Tensor) -> np.ndarray:
    arr = frame_t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def h264_encode(in_path: Path, out_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", default="diagnostics/occlusion_aug_preview")
    parser.add_argument("--force-keys", nargs="+", default=["Fz"])
    parser.add_argument("--trim-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--occluder-shapes",
        nargs="+",
        choices=("rect", "ellipse"),
        default=["rect", "ellipse"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    episode = Path(args.episode)
    out_dir = Path(args.output) / episode.name
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_ds = ForceDataset(
        str(episode),
        force_keys=args.force_keys,
        augment=False,
        trim_seconds=args.trim_seconds,
    )
    aug_ds = ForceDataset(
        str(episode),
        force_keys=args.force_keys,
        augment=True,
        trim_seconds=args.trim_seconds,
        occluder_shapes=tuple(args.occluder_shapes),
    )

    n = min(len(clean_ds), len(aug_ds))
    indices = list(range(0, n, args.stride))[: args.max_frames]
    if not indices:
        raise RuntimeError(f"No frames available for {episode}")

    sample = tensor_to_rgb(clean_ds[indices[0]]["frame"])
    h, w = sample.shape[:2]
    video_path = out_dir / "clean_vs_occluded.mp4"
    writer = cv.VideoWriter(
        str(video_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w * 2, h),
    )

    for idx in indices:
        clean = tensor_to_rgb(clean_ds[idx]["frame"])
        aug = tensor_to_rgb(aug_ds[idx]["frame"])
        side = np.concatenate([clean, aug], axis=1)
        writer.write(cv.cvtColor(side, cv.COLOR_RGB2BGR))

    writer.release()

    try:
        h264_encode(video_path, out_dir / "clean_vs_occluded_h264.mp4")
    except Exception as exc:
        print(f"[warn] ffmpeg H.264 encode failed: {exc}")

    print(f"Preview saved: {video_path}")
    print(f"H.264 preview: {out_dir / 'clean_vs_occluded_h264.mp4'}")


if __name__ == "__main__":
    main()

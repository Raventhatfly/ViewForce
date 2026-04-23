"""
src/dataset.py  --  ForceDataset

Loads an episode directory and yields (masked_frame, force_label) pairs.

Pipeline per sample:
  1. Decode frame from video.mp4 via PyAV
  2. Load corresponding binary mask from mask.mp4
  3. Apply mask: zero-out pixels outside the gripper region
  4. Normalize to [0, 1] and resize to OUTPUT_SIZE
  5. Augment: random mild affine (translation + scale)

Force label is linearly interpolated from the ~20 Hz force CSV to the
frame wall-clock timestamp.  The first and last `trim_seconds` are dropped.
"""

import os
import csv
from typing import Optional

import av
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

OUTPUT_SIZE = (256, 256)

# Mild augmentation bounds
AUG_TRANSLATE = 0.08   # ±8% of image size
AUG_SCALE_LO  = 0.92
AUG_SCALE_HI  = 1.08


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _decode_video(video_path: str) -> list[np.ndarray]:
    """Decode every frame into a list of uint8 RGB arrays."""
    frames = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        frames.append(np.array(frame.to_image().convert("RGB")))
    container.close()
    return frames


def _decode_mask(mask_path: str) -> list[np.ndarray]:
    """Decode mask video into a list of bool arrays (H, W)."""
    masks = []
    container = av.open(mask_path)
    for frame in container.decode(video=0):
        gray = np.array(frame.to_image().convert("L"))
        masks.append(gray > 127)
    container.close()
    return masks


class ForceDataset(Dataset):
    """
    Args:
        episode_dir:    path to episode directory containing video.mp4, mask.mp4,
                        frame_timestamps.csv, force_timestamps.csv
        force_keys:     which force columns to predict
        augment:        apply random affine augmentation
        trim_seconds:   seconds to drop from the start and end of the episode
    """

    def __init__(
        self,
        episode_dir: str,
        force_keys: list[str] = ("Fy",),
        augment: bool = False,
        trim_seconds: float = 2.0,
    ):
        self.episode_dir  = episode_dir
        self.force_keys   = list(force_keys)
        self.augment      = augment

        video_path    = os.path.join(episode_dir, "video.mp4")
        mask_path     = os.path.join(episode_dir, "mask.mp4")
        frame_ts_path = os.path.join(episode_dir, "frame_timestamps.csv")
        force_ts_path = os.path.join(episode_dir, "force_timestamps.csv")

        print(f"  Decoding {episode_dir} ...")
        video_frames = _decode_video(video_path)
        mask_frames  = _decode_mask(mask_path)

        frame_rows = _load_csv(frame_ts_path)
        frame_t    = np.array([float(r["t_rel_s"]) for r in frame_rows])

        force_rows = _load_csv(force_ts_path)
        force_t    = np.array([float(r["t_rel_s"]) for r in force_rows])
        force_wall = np.array([float(r["t_wall_s"]) for r in force_rows])
        frame_wall = np.array([float(r["t_wall_s"]) for r in frame_rows])
        force_vals = {
            k: np.array([float(r[k]) for r in force_rows])
            for k in self.force_keys
        }

        t_max = frame_t.max() if len(frame_t) > 0 else 0.0
        t_lo  = trim_seconds
        t_hi  = t_max - trim_seconds

        self.samples = []
        n = min(len(video_frames), len(mask_frames), len(frame_t))
        for i in range(n):
            t = frame_t[i]
            if t < t_lo or t > t_hi:
                continue

            # Interpolate force to this frame's wall-clock time
            t_wall = frame_wall[i]
            label = np.array(
                [np.interp(t_wall, force_wall, force_vals[k]) for k in self.force_keys],
                dtype=np.float32,
            )
            self.samples.append((video_frames[i], mask_frames[i], label))

        print(f"  {len(self.samples)} frames kept  (trimmed {n - len(self.samples)}, total {n})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        frame_np, mask_np, label = self.samples[idx]

        # Apply gripper mask: zero outside
        masked = frame_np.astype(np.float32) * mask_np[:, :, None]   # (H, W, 3)

        # To tensor (C, H, W) in [0, 1]
        frame_t = torch.from_numpy(masked / 255.0).permute(2, 0, 1).float()

        # Resize to network input size
        frame_t = TF.resize(frame_t, list(OUTPUT_SIZE), antialias=True)

        # Augmentation: random mild affine
        if self.augment:
            h, w = OUTPUT_SIZE
            max_tx = int(w * AUG_TRANSLATE)
            max_ty = int(h * AUG_TRANSLATE)
            tx = torch.randint(-max_tx, max_tx + 1, (1,)).item()
            ty = torch.randint(-max_ty, max_ty + 1, (1,)).item()
            scale = (AUG_SCALE_LO + torch.rand(1).item() * (AUG_SCALE_HI - AUG_SCALE_LO))
            frame_t = TF.affine(
                frame_t,
                angle=0,
                translate=[tx, ty],
                scale=scale,
                shear=0,
                fill=0,
            )

        return {
            "frame": frame_t,                       # (3, H, W)
            "force": torch.from_numpy(label),       # (force_dim,)
        }


def make_datasets(
    episode_dirs: list[str],
    val_episode: Optional[str] = None,
    force_keys: tuple = ("Fy",),
    trim_seconds: float = 2.0,
) -> tuple[Dataset, Dataset]:
    """
    Build train and validation datasets using leave-one-episode-out split.
    """
    from torch.utils.data import ConcatDataset

    if val_episode is None:
        val_episode = episode_dirs[-1]

    train_dirs = [d for d in episode_dirs if d != val_episode]

    print("Building training datasets:")
    train_ds = ConcatDataset([
        ForceDataset(d, force_keys=force_keys, augment=True, trim_seconds=trim_seconds)
        for d in train_dirs
    ])
    print("Building validation dataset:")
    val_ds = ForceDataset(val_episode, force_keys=force_keys, augment=False, trim_seconds=trim_seconds)

    return train_ds, val_ds

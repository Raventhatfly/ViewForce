"""
src/dataset.py  --  ForceDataset

Loads an episode directory and yields (diff_masked_image, force_label) pairs.

Pipeline per sample:
  1. Decode frame from video via PyAV
  2. Apply orange HSV mask (gripper region only)
  3. Subtract masked reference frame (average of first N_REF_FRAMES)
  4. Normalize diff to [-1, 1]
  5. Resize to OUTPUT_SIZE

Force label is linearly interpolated from the ~20 Hz force CSV to the frame
wall-clock timestamp.
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

# ---------------------------------------------------------------------------
# Masking parameters (match color_mask.py)
# ---------------------------------------------------------------------------
H_MIN, H_MAX = 8, 38       # orange hue range (degrees)
S_MIN        = 0.40
V_MIN        = 0.25

# Number of initial frames averaged to build the reference (no-contact) frame
N_REF_FRAMES = 10

# Frames whose Laplacian variance is below this are skipped (motion blur)
MIN_SHARPNESS = 2000.0

# Output spatial resolution fed to the network
OUTPUT_SIZE = (256, 256)


# ---------------------------------------------------------------------------

def _orange_mask(img_np: np.ndarray) -> np.ndarray:
    """Return boolean mask (H, W) of orange gripper pixels."""
    pil_hsv = Image.fromarray(img_np).convert("HSV")
    hsv = np.array(pil_hsv, dtype=np.float32)
    H = hsv[:, :, 0] / 255.0 * 360.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    return (H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN) & (V >= V_MIN)


def _laplacian_var(gray: np.ndarray) -> float:
    """Laplacian variance as a sharpness proxy (higher = sharper)."""
    from numpy.lib.stride_tricks import sliding_window_view
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    pad = np.pad(gray.astype(np.float32), 1, mode="reflect")
    windows = sliding_window_view(pad, (3, 3))
    lap = (windows * k).sum(axis=(-1, -2))
    return float(lap.var())


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _decode_all_frames(video_path: str) -> list[np.ndarray]:
    """Decode every frame of a video into a list of uint8 RGB arrays."""
    frames = []
    container = av.open(video_path)
    for frame in container.decode(video=0):
        frames.append(np.array(frame.to_image()))
    container.close()
    return frames


class ForceDataset(Dataset):
    """
    Args:
        episode_dir:   path to an episode directory (e.g. data/episodes/EP000001)
        force_keys:    which force columns to use as labels (default: ["Fy"])
        augment:       apply random flip + brightness jitter
        min_sharpness: skip blurry frames below this Laplacian variance
    """

    def __init__(
        self,
        episode_dir: str,
        force_keys: list[str] = ("Fy",),
        augment: bool = False,
        min_sharpness: float = MIN_SHARPNESS,
    ):
        self.episode_dir   = episode_dir
        self.force_keys    = list(force_keys)
        self.augment       = augment
        self.min_sharpness = min_sharpness

        video_path       = os.path.join(episode_dir, "video.mp4")
        frame_ts_path    = os.path.join(episode_dir, "frame_timestamps.csv")
        force_ts_path    = os.path.join(episode_dir, "force_timestamps.csv")

        # ---- Load all frames once (fast since episodes are short) ----------
        print(f"  Decoding {video_path} ...")
        all_frames = _decode_all_frames(video_path)

        # ---- Frame timestamps ----------------------------------------------
        frame_rows = _load_csv(frame_ts_path)
        frame_t    = np.array([float(r["t_wall_s"]) for r in frame_rows])

        # ---- Force timestamps + values ------------------------------------
        force_rows = _load_csv(force_ts_path)
        force_t    = np.array([float(r["t_wall_s"]) for r in force_rows])
        force_vals = {
            k: np.array([float(r[k]) for r in force_rows])
            for k in self.force_keys
        }

        # ---- Reference frame (mean of first N_REF_FRAMES) -----------------
        ref_frames = all_frames[:N_REF_FRAMES]
        ref_mean   = np.mean(ref_frames, axis=0).astype(np.float32)  # (H, W, 3)
        ref_mask   = _orange_mask(ref_mean.astype(np.uint8))
        self.ref_masked = (ref_mean * ref_mask[:, :, None]).astype(np.float32)

        # ---- Build valid sample index -------------------------------------
        self.samples = []   # list of (frame_np, force_label)

        for i, frame_np in enumerate(all_frames):
            if i >= len(frame_t):
                break

            # Skip blurry frames
            gray = np.array(Image.fromarray(frame_np).convert("L"), dtype=np.float32)
            if _laplacian_var(gray) < self.min_sharpness:
                continue

            # Interpolate force to this frame's timestamp
            t = frame_t[i]
            label = np.array(
                [np.interp(t, force_t, force_vals[k]) for k in self.force_keys],
                dtype=np.float32,
            )

            self.samples.append((frame_np, label, i))

        print(f"  {len(self.samples)} valid frames (skipped {len(all_frames) - len(self.samples)} blurry)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        frame_np, label, frame_idx = self.samples[idx]

        # Apply orange mask to current frame
        mask = _orange_mask(frame_np)
        cur_masked = (frame_np.astype(np.float32) * mask[:, :, None])

        # Deformation map: current − reference (both masked)
        diff = cur_masked - self.ref_masked   # float32, range ~[-255, 255]

        # Augmentation (apply identically to diff to preserve meaning)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                diff = diff[:, ::-1, :].copy()   # horizontal flip (H, W, 3)

            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # ±10%
            diff = np.clip(diff * brightness, -255, 255)

        # Normalize to [-1, 1] and convert to tensor (C, H, W)
        diff_t = torch.from_numpy(diff / 255.0).permute(2, 0, 1).float()

        # Resize to OUTPUT_SIZE
        diff_t = TF.resize(diff_t, list(OUTPUT_SIZE), antialias=True)

        return {
            "diff":      diff_t,                                  # (3, H, W)
            "force":     torch.from_numpy(label),                 # (force_dim,)
            "frame_idx": frame_idx,
        }


def make_datasets(
    episode_dirs: list[str],
    val_episode: Optional[str] = None,
    force_keys: tuple = ("Fy",),
) -> tuple[Dataset, Dataset]:
    """
    Build train and validation datasets using leave-one-episode-out split.

    Args:
        episode_dirs: list of all episode directories
        val_episode:  directory of the validation episode (left out of train)
                      if None, uses the last episode in the list
        force_keys:   force columns to predict

    Returns:
        (train_dataset, val_dataset)
    """
    from torch.utils.data import ConcatDataset

    if val_episode is None:
        val_episode = episode_dirs[-1]

    train_dirs = [d for d in episode_dirs if d != val_episode]

    print("Building training datasets:")
    train_ds = ConcatDataset([
        ForceDataset(d, force_keys=force_keys, augment=True) for d in train_dirs
    ])
    print("Building validation dataset:")
    val_ds = ForceDataset(val_episode, force_keys=force_keys, augment=False)

    return train_ds, val_ds

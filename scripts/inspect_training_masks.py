"""
Inspect training masks through the deployment image path.

For each episode, this script reads video.mp4 and mask.mp4, resizes both to the
policy-server image size (320x240), and writes debug overlays.  This answers:
"what did the training mask look like if viewed through the DP deployment
resolution?"
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np


IMAGE_H = 240
IMAGE_W = 320
MASK_COLOR = np.array([255, 50, 50], dtype=np.uint8)
MASK_ALPHA = 0.5


def find_episodes(data_dir):
    data_dir = Path(data_dir)
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("EP")
        and (p / "video.mp4").is_file()
        and (p / "mask.mp4").is_file()
    )


def make_overlay(frame_rgb, mask_bool):
    out = frame_rgb.copy()
    out[mask_bool] = (
        (1.0 - MASK_ALPHA) * out[mask_bool].astype(np.float32)
        + MASK_ALPHA * MASK_COLOR.astype(np.float32)
    ).astype(np.uint8)
    return out


def h264_encode(in_path, out_path):
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


def inspect_episode(ep_dir, out_root, max_frames=None, stride=1, fps=10.0):
    ep_dir = Path(ep_dir)
    out_dir = Path(out_root) / ep_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    video_cap = cv.VideoCapture(str(ep_dir / "video.mp4"))
    mask_cap = cv.VideoCapture(str(ep_dir / "mask.mp4"))
    if not video_cap.isOpened():
        raise RuntimeError(f"Could not open video: {ep_dir / 'video.mp4'}")
    if not mask_cap.isOpened():
        raise RuntimeError(f"Could not open mask: {ep_dir / 'mask.mp4'}")

    src_w = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    src_h = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    src_fps = video_cap.get(cv.CAP_PROP_FPS) or fps
    write_fps = fps or src_fps

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    overlay_path = out_dir / "overlay_dp.mp4"
    mask_path = out_dir / "mask_dp.mp4"
    overlay_writer = cv.VideoWriter(
        str(overlay_path), fourcc, write_fps, (IMAGE_W, IMAGE_H)
    )
    mask_writer = cv.VideoWriter(
        str(mask_path), fourcc, write_fps, (IMAGE_W, IMAGE_H)
    )

    rows = []
    frame_idx = 0
    kept = 0
    while True:
        ok_v, frame_bgr = video_cap.read()
        ok_m, mask_bgr = mask_cap.read()
        if not ok_v or not ok_m:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        if max_frames is not None and kept >= max_frames:
            break

        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mask_gray = cv.cvtColor(mask_bgr, cv.COLOR_BGR2GRAY)
        frame_dp = cv.resize(frame_rgb, (IMAGE_W, IMAGE_H), interpolation=cv.INTER_AREA)
        mask_dp = cv.resize(mask_gray, (IMAGE_W, IMAGE_H), interpolation=cv.INTER_NEAREST)
        mask_bool = mask_dp > 127
        overlay_rgb = make_overlay(frame_dp, mask_bool)
        mask_rgb = np.repeat(mask_bool[:, :, None].astype(np.uint8) * 255, 3, axis=2)

        if kept == 0:
            cv.imwrite(str(out_dir / "first_frame_dp.png"), cv.cvtColor(frame_dp, cv.COLOR_RGB2BGR))
            cv.imwrite(str(out_dir / "first_overlay_dp.png"), cv.cvtColor(overlay_rgb, cv.COLOR_RGB2BGR))
            cv.imwrite(str(out_dir / "first_mask_dp.png"), cv.cvtColor(mask_rgb, cv.COLOR_RGB2BGR))

        ys, xs = np.where(mask_bool)
        bbox = "" if len(xs) == 0 else f"{xs.min()},{ys.min()},{xs.max()},{ys.max()}"
        rows.append({
            "frame_idx": frame_idx,
            "mask_px": int(mask_bool.sum()),
            "mask_frac": float(mask_bool.mean()),
            "bbox_xyxy": bbox,
        })

        overlay_writer.write(cv.cvtColor(overlay_rgb, cv.COLOR_RGB2BGR))
        mask_writer.write(cv.cvtColor(mask_rgb, cv.COLOR_RGB2BGR))
        kept += 1
        frame_idx += 1

    video_cap.release()
    mask_cap.release()
    overlay_writer.release()
    mask_writer.release()

    with open(out_dir / "mask_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "mask_px", "mask_frac", "bbox_xyxy"])
        writer.writeheader()
        writer.writerows(rows)

    try:
        h264_encode(overlay_path, out_dir / "overlay_dp_h264.mp4")
        h264_encode(mask_path, out_dir / "mask_dp_h264.mp4")
    except Exception as exc:
        print(f"[warn] ffmpeg H.264 encode failed for {ep_dir.name}: {exc}")

    print(
        f"{ep_dir.name}: source={src_w}x{src_h}, dp={IMAGE_W}x{IMAGE_H}, "
        f"frames={kept}, out={out_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/data_ball_260422")
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--out-dir", default="diagnostics/training_masks_dp")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes:
        episodes = [Path(p) for p in args.episodes]
    else:
        episodes = find_episodes(args.data_dir)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        raise RuntimeError("No episodes found")
    for ep in episodes:
        inspect_episode(
            ep,
            args.out_dir,
            max_frames=args.max_frames,
            stride=args.stride,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()

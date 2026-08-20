"""Build the compact JSON manifest consumed by the static berry viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str, default: str = "unknown") -> str:
    if key not in data:
        return default
    return str(np.asarray(data[key]).reshape(()).item())


def main() -> None:
    args = parse_args()
    root = args.dataset.expanduser().resolve()
    episodes = []
    for force_path in sorted(root.rglob("viewforce_pseudo_force_fz.npz")):
        directory = force_path.parent
        with np.load(force_path, allow_pickle=False) as data:
            force = np.asarray(data["force"], dtype=np.float32)
            if force.ndim == 1:
                force = force[:, None]
            if force.ndim != 2 or force.shape[0] == 0:
                raise ValueError(f"Expected a non-empty 1D or 2D force array in {force_path}")
            keys = [str(value) for value in data["force_keys"].tolist()]
            if len(keys) != force.shape[1]:
                raise ValueError(
                    f"Force key count does not match force columns in {force_path}"
                )
            mask = (
                np.asarray(data["mask_frac"], dtype=np.float32).tolist()
                if "mask_frac" in data
                else []
            )
            metadata = {
                "checkpoint": Path(scalar(data, "viewforce_ckpt")).name,
                "input_mode": scalar(data, "input_mode"),
                "mask_mode": scalar(data, "mask_mode"),
            }
        relative = directory.relative_to(root).as_posix()
        magnitude = np.abs(force[:, 0])
        relative_path = directory.relative_to(root)
        name_parts = relative_path.parts
        if name_parts and name_parts[0] == "episodes":
            name_parts = name_parts[1:]
        name = " / ".join(name_parts) or directory.name
        lowered_parts = [part.lower() for part in name_parts]
        episodes.append(
            {
                "name": name,
                "status": (
                    "failure"
                    if any("_fail" in part for part in lowered_parts)
                    else "unverified success"
                ),
                "difficulty": next(
                    (
                        difficulty
                        for part in lowered_parts
                        for difficulty in ("light", "hard", "predict")
                        if part.startswith(difficulty)
                    ),
                    "unknown",
                ),
                "stage": "stage2" if directory.name.lower() == "stage2" else "stage1",
                "force_keys": keys,
                "force": np.round(force, 5).tolist(),
                "mask_frac": np.round(mask, 5).tolist() if mask else [],
                "base_video": (
                    f"{relative}/base_image.mp4"
                    if (directory / "base_image.mp4").is_file()
                    else None
                ),
                "wrist_video": (
                    f"{relative}/wrist_image.mp4"
                    if (directory / "wrist_image.mp4").is_file()
                    else None
                ),
                "stats": {
                    "median": round(float(np.median(magnitude)), 4),
                    "p90": round(float(np.quantile(magnitude, 0.9)), 4),
                    "max": round(float(np.max(magnitude)), 4),
                },
                **metadata,
            }
        )
    payload = {
        "schema_version": 1,
        "episode_count": len(episodes),
        "target_band": {"low": 10.8204, "center": 12.6758, "high": 13.5166},
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Wrote {len(episodes)} episode/stages to {args.output}")


if __name__ == "__main__":
    main()

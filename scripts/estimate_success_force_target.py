"""Estimate a robust ViewForce target or force band from successful episodes.

This script consumes the cached pseudo-force files produced by
``precompute_flip_pseudo_force.py``.  It deliberately does not rerun ViewForce,
so target extraction is deterministic and uses the exact labels attached to the
training demonstrations.

Example:
    python scripts/estimate_success_force_target.py \
        --dataset-path third_party/forcelens_dp/data/pick/berry_force_prediction_all \
        --episode-glob '*stage2' \
        --exclude-substring _fail \
        --output configs/berry_success_force_target.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def discover_episodes(dataset_path: Path, episode_glob: str) -> list[Path]:
    episodes_root = dataset_path / "episodes"
    root = episodes_root if episodes_root.is_dir() else dataset_path
    return sorted(
        path
        for path in root.glob(episode_glob)
        if path.is_dir()
    )


def load_episode_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    names = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError(f"Episode allowlist is empty: {path}")
    return names


def scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a demonstrated ViewForce target from successful episodes."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--pseudo-label-name",
        default="viewforce_pseudo_force_fz.npz",
    )
    parser.add_argument("--force-key", default="Fz")
    parser.add_argument(
        "--force-mode",
        choices=("magnitude", "signed"),
        default="magnitude",
    )
    parser.add_argument(
        "--episode-glob",
        default="*stage2",
        help="Episode directory glob. The default focuses on the grasp/lift stage.",
    )
    parser.add_argument(
        "--episodes-file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited allowlist of successful episode directory "
            "names. Prefer this over inferring success from filenames."
        ),
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help="Exclude episode names containing this text; may be passed repeatedly.",
    )
    parser.add_argument(
        "--start-fraction",
        type=float,
        default=0.0,
        help="Start of the per-episode force window as a fraction of its duration.",
    )
    parser.add_argument(
        "--end-fraction",
        type=float,
        default=1.0,
        help="End of the per-episode force window as a fraction of its duration.",
    )
    parser.add_argument(
        "--episode-quantile",
        type=float,
        default=0.90,
        help="Force quantile summarized within each successful episode.",
    )
    parser.add_argument(
        "--band-low-quantile",
        type=float,
        default=0.25,
        help="Across-episode quantile used as the lower demonstrated force bound.",
    )
    parser.add_argument(
        "--band-high-quantile",
        type=float,
        default=0.75,
        help="Across-episode quantile used as the upper demonstrated force bound.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_quantile(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def main() -> None:
    args = parse_args()
    validate_quantile("episode-quantile", args.episode_quantile)
    validate_quantile("band-low-quantile", args.band_low_quantile)
    validate_quantile("band-high-quantile", args.band_high_quantile)
    if args.band_low_quantile > args.band_high_quantile:
        raise ValueError("band-low-quantile must be <= band-high-quantile")
    if not 0.0 <= args.start_fraction < args.end_fraction <= 1.0:
        raise ValueError("Require 0 <= start-fraction < end-fraction <= 1")

    allowlist = load_episode_allowlist(args.episodes_file)
    candidates = discover_episodes(args.dataset_path, args.episode_glob)
    selected = []
    summaries = []
    checkpoint_values = set()
    input_modes = set()
    mask_modes = set()

    for episode in candidates:
        if allowlist is not None and episode.name not in allowlist:
            continue
        if any(text in episode.name for text in args.exclude_substring):
            continue
        label_path = episode / args.pseudo_label_name
        if not label_path.is_file():
            continue

        with np.load(label_path) as labels:
            force_keys = labels["force_keys"].tolist()
            if args.force_key not in force_keys:
                raise ValueError(
                    f"{label_path} does not contain force key {args.force_key!r}"
                )
            force_idx = force_keys.index(args.force_key)
            force = labels["force"][:, force_idx].astype(np.float64)
            if args.force_mode == "magnitude":
                force = np.abs(force)
            if "viewforce_ckpt" in labels:
                checkpoint_values.add(scalar_string(labels["viewforce_ckpt"]))
            if "input_mode" in labels:
                input_modes.add(scalar_string(labels["input_mode"]))
            if "mask_mode" in labels:
                mask_modes.add(scalar_string(labels["mask_mode"]))

        start = int(np.floor(len(force) * args.start_fraction))
        end = int(np.ceil(len(force) * args.end_fraction))
        window = force[start:end]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        summaries.append(float(np.quantile(window, args.episode_quantile)))
        selected.append(episode.name)

    if not summaries:
        raise RuntimeError(
            "No usable pseudo-force episodes matched the selection. "
            "Check --episode-glob, --episodes-file, and exclusions."
        )
    if allowlist is None:
        print(
            "Warning: no --episodes-file was supplied; successful episodes are "
            "being inferred only from the glob/exclusion rules."
        )

    values = np.asarray(summaries, dtype=np.float64)
    payload = {
        "schema_version": 1,
        "dataset_path": str(args.dataset_path.resolve()),
        "pseudo_label_name": args.pseudo_label_name,
        "force_key": args.force_key,
        "force_mode": args.force_mode,
        "target_force": float(np.median(values)),
        "safe_min": float(np.quantile(values, args.band_low_quantile)),
        "safe_max": float(np.quantile(values, args.band_high_quantile)),
        "episode_quantile": args.episode_quantile,
        "band_low_quantile": args.band_low_quantile,
        "band_high_quantile": args.band_high_quantile,
        "start_fraction": args.start_fraction,
        "end_fraction": args.end_fraction,
        "episode_count": len(selected),
        "episodes": selected,
        "viewforce_checkpoints": sorted(checkpoint_values),
        "input_modes": sorted(input_modes),
        "mask_modes": sorted(mask_modes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote demonstrated force target: {args.output}")


if __name__ == "__main__":
    main()

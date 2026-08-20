"""Package every unique demonstration for the public static dataset explorer."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


RETIRED_DATASETS = {
    "pick_berries",
    "pick_berries_all",
    "pick_berries_brute",
    "pick_berries_brute_kelvin",
    "pick_berries_kelvin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-retired",
        action="store_true",
        help="Include legacy pick_berries* collections in the manifest.",
    )
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str, default: str = "unknown") -> str:
    if key not in data:
        return default
    return str(np.asarray(data[key]).reshape(()).item())


def copy_file(source: Path, destination: Path, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(), destination)
    return True


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> None:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == root:
        raise ValueError("--output must not be the same directory as --data-root")

    # Derived training datasets link back to canonical videos. Keying by the
    # resolved base video removes those duplicates while retaining any force
    # estimate discovered in a derived directory.
    aliases: dict[Path, list[Path]] = defaultdict(list)
    for base in root.rglob("base_image.mp4"):
        if base.exists() and not is_within(base, output):
            aliases[base.resolve()].append(base)

    force_by_video: dict[Path, list[Path]] = defaultdict(list)
    for force_file in root.rglob("viewforce_pseudo_force_fz.npz"):
        if is_within(force_file, output):
            continue
        base = force_file.parent / "base_image.mp4"
        if base.exists():
            force_by_video[base.resolve()].append(force_file)

    records = []
    copied = 0
    for base_resolved, paths in sorted(aliases.items(), key=lambda item: str(item[0])):
        try:
            relative_dir = base_resolved.parent.relative_to(root)
        except ValueError:
            relative_dir = min(paths, key=lambda path: len(path.parts)).parent.relative_to(root)
        parts = relative_dir.parts
        task = parts[0] if parts else "unknown"
        dataset = parts[1] if len(parts) > 1 else "unknown"
        if "retired" in parts and not args.include_retired:
            continue
        if dataset in RETIRED_DATASETS and not args.include_retired:
            continue
        record_id = relative_dir.as_posix()
        destination = output / "records" / relative_dir

        wrist = base_resolved.parent / "wrist_image.mp4"
        if copy_file(base_resolved, destination / "base_image.mp4", args.overwrite):
            copied += 1
        if wrist.exists() and copy_file(wrist, destination / "wrist_image.mp4", args.overwrite):
            copied += 1

        force_files = sorted(force_by_video.get(base_resolved, []), key=str)
        force_payload = None
        if force_files:
            # All currently duplicated labels for the same video are bytewise
            # identical. Keeping one avoids coupling the explorer to a derived
            # model-training dataset name.
            force_file = force_files[0]
            if copy_file(force_file, destination / "viewforce_pseudo_force_fz.npz", args.overwrite):
                copied += 1
            with np.load(force_file, allow_pickle=False) as data:
                force = np.asarray(data["force"], dtype=np.float32)
                if force.ndim == 1:
                    force = force[:, None]
                keys = [str(value) for value in data["force_keys"].tolist()]
                magnitude = np.abs(force[:, 0])
                force_payload = {
                    "keys": keys,
                    "values": np.round(force, 5).tolist(),
                    "checkpoint": Path(scalar(data, "viewforce_ckpt")).name,
                    "input_mode": scalar(data, "input_mode"),
                    "mask_mode": scalar(data, "mask_mode"),
                    "stats": {
                        "median": round(float(np.median(magnitude)), 4),
                        "p90": round(float(np.quantile(magnitude, 0.9)), 4),
                        "max": round(float(np.max(magnitude)), 4),
                    },
                }

        episode_parts = [part for part in parts[2:] if part != "episodes"]
        display_name = " / ".join(episode_parts) if episode_parts else relative_dir.name
        lowered = record_id.lower()
        if "_fail" in lowered:
            status = "failure"
            status_source = "episode name"
        elif task == "pick" and dataset == "berry_staged":
            status = "success"
            status_source = "berry light collection"
        elif task == "pick" and dataset == "berry_staged_hard":
            status = "failure"
            status_source = "berry hard collection"
        else:
            status = "unverified"
            status_source = "no outcome label"
        records.append(
            {
                "id": record_id,
                "name": display_name,
                "task": task,
                "dataset": dataset,
                "status": status,
                "status_source": status_source,
                "stage": "stage2" if relative_dir.name == "stage2" else "stage1" if relative_dir.name == "stage1" else "single",
                "frames": len(force_payload["values"]) if force_payload else None,
                "base_video": f"records/{record_id}/base_image.mp4",
                "wrist_video": f"records/{record_id}/wrist_image.mp4" if wrist.exists() else None,
                "force": force_payload,
            }
        )

    manifest = {
        "schema_version": 2,
        "record_count": len(records),
        "force_record_count": sum(record["force"] is not None for record in records),
        "target_bands": {
            "berry_staged": {"low": 10.8204, "center": 12.6758, "high": 13.5166},
            "berry_staged_hard": {"low": 10.8204, "center": 12.6758, "high": 13.5166},
            "berry_staged_hard_predict": {"low": 10.8204, "center": 12.6758, "high": 13.5166},
        },
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(
        f"Packaged {len(records)} unique records, "
        f"{manifest['force_record_count']} with force ({copied} files copied) in {output}"
    )


if __name__ == "__main__":
    main()

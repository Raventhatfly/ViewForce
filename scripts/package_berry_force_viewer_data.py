"""Package berry videos and force estimates for the dataset viewer.

The training dataset links its videos from the original demonstration tree.
This utility dereferences only the files needed by the web viewer into a
portable directory suitable for upload to a Hugging Face dataset repository.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES = ("base_image.mp4", "wrist_image.mp4", "viewforce_pseudo_force_fz.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files already present in the output directory.",
    )
    return parser.parse_args()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == source:
        raise ValueError("--output must not be the same directory as --source")
    force_files = sorted(
        path
        for path in source.rglob("viewforce_pseudo_force_fz.npz")
        if not is_within(path, output)
    )
    if not force_files:
        raise RuntimeError(f"No force estimate files found under {source}")

    copied_files = 0
    copied_episodes = 0
    for force_file in force_files:
        episode = force_file.parent
        relative = episode.relative_to(source)
        if relative.parts and relative.parts[0] == "episodes":
            relative = Path(*relative.parts[1:])
        destination = output / "episodes" / relative
        available = [filename for filename in FILES if (episode / filename).exists()]
        missing = sorted(set(FILES) - set(available))
        if missing:
            raise FileNotFoundError(
                f"Episode {episode} is missing viewer files: {', '.join(missing)}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        for filename in available:
            target = destination / filename
            if target.exists() and not args.overwrite:
                continue
            shutil.copy2((episode / filename).resolve(), target)
            copied_files += 1
        copied_episodes += 1

    print(f"Packaged {copied_episodes} episode/stages ({copied_files} files) in {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create appendix rollout figures for the Coke force-steering experiments.

This script is intentionally offline: run it manually after generating
reports/coke_force_steering/trial_peaks.csv with plot_coke_force_results.py.
It selects representative rollouts per group, extracts frames from rollout
videos, annotates predicted force, and writes a LaTeX snippet that can be
pasted after \\appendix.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "Plain DP": {
        "rollout_dir": "grab_coke_empty_dp",
        "caption": "Plain diffusion policy",
    },
    "TTS 0.2 N": {
        "rollout_dir": "grab_coke_empty_0_2N",
        "caption": r"ForceLens TTS, $F^\star=0.2\,\mathrm{N}$",
    },
    "TTS 15 N": {
        "rollout_dir": "grab_coke_empty_15_0N",
        "caption": r"ForceLens TTS, $F^\star=15\,\mathrm{N}$",
    },
}

VIDEO_COLUMNS = (
    ("original_h264.mp4", "RGB"),
    ("masked_original_h264.mp4", "Masked RGB"),
    ("edge_h264.mp4", "Edge input"),
)


@dataclass
class TrialPeak:
    group: str
    trial: str
    peak_force_n: float


def read_trial_peaks(path: Path) -> list[TrialPeak]:
    trials = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                trials.append(
                    TrialPeak(
                        group=row["group"],
                        trial=row["trial"],
                        peak_force_n=float(row["peak_force_n"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return trials


def choose_representatives(
    trials: list[TrialPeak],
    trials_per_group: int,
) -> dict[str, list[TrialPeak]]:
    reps = {}
    quantiles = np.linspace(0.25, 0.75, trials_per_group)
    for group in GROUPS:
        group_trials = [trial for trial in trials if trial.group == group]
        if not group_trials:
            raise RuntimeError(f"No trials found for group: {group}")
        peaks = np.asarray([trial.peak_force_n for trial in group_trials])
        selected = []
        used = set()
        for quantile in quantiles:
            target = float(np.quantile(peaks, quantile))
            candidates = [
                trial for trial in group_trials if trial.trial not in used
            ]
            trial = min(candidates, key=lambda item: abs(item.peak_force_n - target))
            selected.append(trial)
            used.add(trial.trial)
        reps[group] = selected
    return reps


def read_frame(video_path: Path, frac: float, swap_rb: bool = False) -> np.ndarray:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    frame_idx = int(round(frac * (total - 1)))
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if swap_rb:
        frame = frame[..., [2, 1, 0]]
    return frame


def read_force_trace(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = []
    forces = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                times.append(float(row["time_s"]))
                forces.append(float(row["control_force_n"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not forces:
        raise RuntimeError(f"No force samples found in {csv_path}")
    return np.asarray(times), np.asarray(forces)


def force_at_fraction(csv_path: Path, frac: float) -> float:
    times, forces = read_force_trace(csv_path)
    if len(forces) == 1:
        return float(forces[0])
    query_t = float(times[0] + frac * (times[-1] - times[0]))
    return float(np.interp(query_t, times, forces))


def make_group_figure(
    rollouts_dir: Path,
    output_dir: Path,
    group: str,
    trials: list[TrialPeak],
    frame_fracs: list[float],
    swap_rb: bool,
) -> Path:
    info = GROUPS[group]

    rows = len(VIDEO_COLUMNS) * len(trials)
    cols = len(frame_fracs)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.05 * cols, 1.35 * rows),
        constrained_layout=True,
    )
    if rows == 1:
        axes = axes[None, :]

    for trial_idx, trial in enumerate(trials):
        trial_dir = rollouts_dir / info["rollout_dir"] / trial.trial
        if not trial_dir.exists():
            raise RuntimeError(f"Missing rollout directory: {trial_dir}")
        csv_path = trial_dir / "force_log.csv"
        force_labels = [force_at_fraction(csv_path, frac) for frac in frame_fracs]

        for video_idx, (video_name, row_label) in enumerate(VIDEO_COLUMNS):
            row_idx = trial_idx * len(VIDEO_COLUMNS) + video_idx
            video_path = trial_dir / video_name
            for col_idx, frac in enumerate(frame_fracs):
                ax = axes[row_idx, col_idx]
                ax.imshow(read_frame(video_path, frac, swap_rb=swap_rb))
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f"{frac:.0%}", fontsize=8)
                if video_idx == 0:
                    ax.text(
                        0.03,
                        0.08,
                        f"F={force_labels[col_idx]:.1f} N",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=8,
                        bbox={
                            "facecolor": "black",
                            "alpha": 0.55,
                            "edgecolor": "none",
                            "pad": 1.5,
                        },
                    )
                if col_idx == 0:
                    ax.set_ylabel(
                        f"R{trial_idx + 1}\n{row_label}",
                        fontsize=8,
                    )

    fig.suptitle(
        f"{info['caption']} | three representative rollouts",
        fontsize=11,
        fontweight="semibold",
    )
    output_path = output_dir / f"appendix_{info['rollout_dir']}_three_rollouts.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def latex_path(path: Path, project_root: Path) -> str:
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def write_latex_snippet(
    output_dir: Path,
    project_root: Path,
    figures: list[tuple[str, list[TrialPeak], Path]],
) -> Path:
    snippet_path = output_dir / "coke_appendix_figures.tex"
    lines = [
        r"\appendix",
        "",
        r"\section{Additional Coke-Can Rollout Visualizations}",
        "",
        (
            "Figure~\\ref{fig:appendix_coke_rollouts} shows representative "
            "rollouts from the Coke-can force-steering experiments. For each "
            "condition, we visualize the RGB observation, the masked gripper "
            "input used by ForceLens, and the corresponding edge-only input "
            "at multiple points in the rollout. Each panel overlays the "
            "ForceLens predicted force for the corresponding time step."
        ),
        "",
        r"\begin{figure*}[p]",
        r"    \centering",
    ]
    for group, trials, fig_path in figures:
        caption = GROUPS[group]["caption"]
        peaks = ", ".join(f"{trial.peak_force_n:.2f}" for trial in trials)
        lines += [
            (
                "    \\includegraphics[width=\\linewidth]{"
                + latex_path(fig_path, project_root)
                + "}"
            ),
            (
                f"    \\\\[-0.25em]{{\\small {caption}; "
                f"peak estimated forces {peaks}\\,N.}}"
            ),
            r"    \vspace{0.5em}",
        ]
    lines += [
        (
            r"    \caption{Representative Coke-can rollouts used in the "
            r"force-steering evaluation. Each block shows one condition, "
            r"with three representative rollouts. Columns denote rollout "
            r"progress and rows denote RGB, masked RGB, and edge-only "
            r"ForceLens inputs. White labels indicate predicted force.}"
        ),
        r"    \label{fig:appendix_coke_rollouts}",
        r"\end{figure*}",
        "",
    ]
    snippet_path.write_text("\n".join(lines))
    return snippet_path


def parse_frame_fracs(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one frame fraction is required")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("Frame fractions must be in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--rollouts-dir", type=Path, default=Path("rollouts"))
    parser.add_argument(
        "--trial-peaks",
        type=Path,
        default=Path("reports/coke_force_steering/trial_peaks.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/coke_force_steering/appendix"),
    )
    parser.add_argument(
        "--frame-fracs",
        type=parse_frame_fracs,
        default=parse_frame_fracs("0.15,0.45,0.75"),
        help="Comma-separated rollout fractions to extract, e.g. 0.1,0.5,0.9",
    )
    parser.add_argument(
        "--trials-per-group",
        type=int,
        default=3,
        help="Number of representative rollouts per condition.",
    )
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        help="Swap red and blue channels after reading rollout videos.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    rollouts_dir = (project_root / args.rollouts_dir).resolve()
    trial_peaks = (project_root / args.trial_peaks).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reps = choose_representatives(
        read_trial_peaks(trial_peaks),
        trials_per_group=args.trials_per_group,
    )
    figures = [
        (
            group,
            group_trials,
            make_group_figure(
                rollouts_dir,
                output_dir,
                group,
                group_trials,
                args.frame_fracs,
                swap_rb=args.swap_rb,
            ),
        )
        for group, group_trials in reps.items()
    ]
    snippet_path = write_latex_snippet(output_dir, project_root, figures)

    for _, _, fig_path in figures:
        print(f"Saved figure: {fig_path}")
    print(f"Saved LaTeX snippet: {snippet_path}")


if __name__ == "__main__":
    main()

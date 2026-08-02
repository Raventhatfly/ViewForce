#!/usr/bin/env python3
"""Plot force-steering results for the empty-Coke experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Trial:
    name: str
    time_s: np.ndarray
    force_n: np.ndarray

    @property
    def peak_n(self) -> float:
        return float(self.force_n.max())


GROUPS = (
    ("Plain DP", "grab_coke_empty_dp", "#687078"),
    ("TTS 0.2 N", "grab_coke_empty_0_2N", "#2A9D8F"),
    ("TTS 15 N", "grab_coke_empty_15_0N", "#E76F51"),
)


def load_trials(root: Path, directory: str) -> list[Trial]:
    trials = []
    group_dir = root / directory
    for csv_path in sorted(group_dir.glob("*/force_log.csv")):
        if "_test" in str(csv_path):
            continue
        times = []
        forces = []
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    times.append(float(row["time_s"]))
                    forces.append(float(row["control_force_n"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if forces:
            trials.append(
                Trial(
                    name=csv_path.parent.name,
                    time_s=np.asarray(times),
                    force_n=np.asarray(forces),
                )
            )
    return trials


def representative_trial(trials: list[Trial]) -> Trial:
    median_peak = float(np.median([trial.peak_n for trial in trials]))
    return min(trials, key=lambda trial: abs(trial.peak_n - median_peak))


def summarize(trials: list[Trial], light_threshold: float, heavy_threshold: float):
    peaks = np.asarray([trial.peak_n for trial in trials])
    return {
        "trials": len(peaks),
        "light_successes": int((peaks <= light_threshold).sum()),
        "light_success_rate": float((peaks <= light_threshold).mean()),
        "heavy_successes": int((peaks >= heavy_threshold).sum()),
        "heavy_success_rate": float((peaks >= heavy_threshold).mean()),
        "middle_count": int(
            ((peaks > light_threshold) & (peaks < heavy_threshold)).sum()
        ),
        "mean_peak_n": float(peaks.mean()),
        "median_peak_n": float(np.median(peaks)),
        "q1_peak_n": float(np.quantile(peaks, 0.25)),
        "q3_peak_n": float(np.quantile(peaks, 0.75)),
        "min_peak_n": float(peaks.min()),
        "max_peak_n": float(peaks.max()),
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DDE1", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_success_rates(ax, data, light_threshold, heavy_threshold):
    dp = summarize(data["Plain DP"], light_threshold, heavy_threshold)
    low = summarize(data["TTS 0.2 N"], light_threshold, heavy_threshold)
    high = summarize(data["TTS 15 N"], light_threshold, heavy_threshold)

    labels = ["Low-force\ncriterion", "High-force\ncriterion"]
    x = np.arange(2)
    width = 0.34
    dp_rates = [dp["light_success_rate"], dp["heavy_success_rate"]]
    tts_rates = [low["light_success_rate"], high["heavy_success_rate"]]

    bars_dp = ax.bar(
        x - width / 2, dp_rates, width, label="Plain DP", color="#687078"
    )
    bars_tts = ax.bar(
        x + width / 2,
        tts_rates,
        width,
        label="Targeted TTS",
        color=["#2A9D8F", "#E76F51"],
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success rate")
    ax.set_title("Target-regime success")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)

    for bars in (bars_dp, bars_tts):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.035,
                f"{100 * bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def plot_peak_distribution(ax, data, light_threshold, heavy_threshold):
    labels = [group[0] for group in GROUPS]
    colors = [group[2] for group in GROUPS]
    values = [[trial.peak_n for trial in data[label]] for label in labels]
    positions = np.arange(1, len(labels) + 1)

    violin = ax.violinplot(
        values,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202428", "linewidth": 1.8},
        whiskerprops={"color": "#596068"},
        capprops={"color": "#596068"},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(7)
    for pos, label, color in zip(positions, labels, colors):
        peaks = np.asarray([trial.peak_n for trial in data[label]])
        jitter = rng.uniform(-0.11, 0.11, size=len(peaks))
        ax.scatter(
            pos + jitter,
            peaks,
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    ax.axhline(
        light_threshold,
        color="#2A9D8F",
        linestyle="--",
        linewidth=1.2,
        label=f"Low threshold ({light_threshold:g} N)",
    )
    ax.axhline(
        heavy_threshold,
        color="#E76F51",
        linestyle="--",
        linewidth=1.2,
        label=f"High threshold ({heavy_threshold:g} N)",
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Peak estimated force (N)")
    ax.set_title("Peak-force distribution")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)


def plot_representative_curves(ax, data):
    for label, _, color in GROUPS:
        trial = representative_trial(data[label])
        ax.plot(
            trial.time_s,
            trial.force_n,
            color=color,
            linewidth=1.8,
            label=f"{label} (peak {trial.peak_n:.1f} N)",
        )
    ax.axhline(4.0, color="#2A9D8F", linestyle=":", linewidth=1.1)
    ax.axhline(10.0, color="#E76F51", linestyle=":", linewidth=1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated force (N)")
    ax.set_title("Median-peak representative trials")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axis(ax)


def save_summary(out_dir: Path, data, light_threshold, heavy_threshold):
    summary_path = out_dir / "summary.csv"
    trial_path = out_dir / "trial_peaks.csv"
    fields = ["group"] + list(
        summarize(next(iter(data.values())), light_threshold, heavy_threshold).keys()
    )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, _, _ in GROUPS:
            writer.writerow(
                {
                    "group": label,
                    **summarize(data[label], light_threshold, heavy_threshold),
                }
            )
    with trial_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "trial",
                "peak_force_n",
                "light_success",
                "heavy_success",
            ],
        )
        writer.writeheader()
        for label, _, _ in GROUPS:
            for trial in data[label]:
                writer.writerow(
                    {
                        "group": label,
                        "trial": trial.name,
                        "peak_force_n": trial.peak_n,
                        "light_success": trial.peak_n <= light_threshold,
                        "heavy_success": trial.peak_n >= heavy_threshold,
                    }
                )
    return summary_path, trial_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-dir", type=Path, default=Path("rollouts"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/coke_force_steering")
    )
    parser.add_argument("--light-threshold", type=float, default=4.0)
    parser.add_argument("--heavy-threshold", type=float, default=10.0)
    args = parser.parse_args()

    data = {
        label: load_trials(args.rollouts_dir, directory)
        for label, directory, _ in GROUPS
    }
    missing = [label for label, trials in data.items() if not trials]
    if missing:
        raise RuntimeError(f"No valid trials found for: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#252A2E",
            "text.color": "#252A2E",
            "xtick.color": "#4D555B",
            "ytick.color": "#4D555B",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    plot_success_rates(
        axes[0], data, args.light_threshold, args.heavy_threshold
    )
    plot_peak_distribution(
        axes[1], data, args.light_threshold, args.heavy_threshold
    )
    plot_representative_curves(axes[2], data)
    fig.suptitle(
        "Visual Force Estimation Steers a Frozen Diffusion Policy",
        fontsize=15,
        fontweight="semibold",
    )
    combined_png = args.output_dir / "force_steering_summary.png"
    combined_pdf = args.output_dir / "force_steering_summary.pdf"
    fig.savefig(combined_png, dpi=220, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)

    summary_path, trial_path = save_summary(
        args.output_dir,
        data,
        args.light_threshold,
        args.heavy_threshold,
    )
    print(f"Saved: {combined_png}")
    print(f"Saved: {combined_pdf}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {trial_path}")


if __name__ == "__main__":
    main()

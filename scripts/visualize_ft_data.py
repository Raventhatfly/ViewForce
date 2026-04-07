import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = ["timestamp", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def moving_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def load_csv(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        rows = list(reader)
        if not rows:
            raise ValueError("CSV file has no data rows")

    return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in REQUIRED_COLUMNS}


def main():
    parser = argparse.ArgumentParser(description="Visualize MMS101 force/torque CSV data")
    parser.add_argument("csv", nargs="?", default="ft_data.csv", help="Path to CSV file")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window (default: 1)")
    parser.add_argument("--downsample", type=int, default=1, help="Keep every N points (default: 1)")
    parser.add_argument("--save", default=None, help="Optional image output path, e.g., ft_plot.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = load_csv(csv_path)

    t = data["timestamp"] - data["timestamp"][0]
    fx = moving_average(data["Fx"], args.smooth)
    fy = moving_average(data["Fy"], args.smooth)
    fz = moving_average(data["Fz"], args.smooth)
    mx = moving_average(data["Mx"], args.smooth)
    my = moving_average(data["My"], args.smooth)
    mz = moving_average(data["Mz"], args.smooth)

    step = max(1, args.downsample)
    t = t[::step]
    fx = fx[::step]
    fy = fy[::step]
    fz = fz[::step]
    mx = mx[::step]
    my = my[::step]
    mz = mz[::step]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, fx, label="Fx")
    axes[0].plot(t, fy, label="Fy")
    axes[0].plot(t, fz, label="Fz")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Force Channels")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, mx, label="Mx")
    axes[1].plot(t, my, label="My")
    axes[1].plot(t, mz, label="Mz")
    axes[1].set_ylabel("Torque (N*m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Torque Channels")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"MMS101 Data: {csv_path.name}")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f"Saved plot to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()

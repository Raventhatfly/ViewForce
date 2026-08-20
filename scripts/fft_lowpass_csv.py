import argparse
import csv
from pathlib import Path

import numpy as np


def estimate_fs(timestamp: np.ndarray) -> float:
    timestamp = np.asarray(timestamp, dtype=float)
    if timestamp.ndim != 1 or len(timestamp) < 2:
        raise ValueError("Need at least 2 timestamp samples to estimate sampling rate")
    if not np.isfinite(timestamp).all():
        raise ValueError("Timestamp data contains non-finite values")

    dt = np.diff(timestamp)
    if np.any(dt <= 0):
        raise ValueError("Invalid timestamp data: non-increasing or non-finite")

    dt_med = float(np.median(dt))
    return 1.0 / dt_med


def fft_lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("Input must be a 1D signal or a 2D samples-by-signals array")
    if not np.isfinite(x).all():
        raise ValueError("Signal data contains non-finite values")
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 signal samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < fs / 2.0:
        raise ValueError("Cutoff must be finite, positive, and below Nyquist")
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(x, axis=0)
    spectrum[freqs > cutoff_hz, ...] = 0.0
    y = np.fft.irfft(spectrum, n=n, axis=0)
    return y


def load_csv_numeric(input_path: Path, timestamp_col: str):
    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        columns = reader.fieldnames
        if timestamp_col not in columns:
            raise ValueError(f"Timestamp column not found: {timestamp_col}")

        rows = list(reader)
        if not rows:
            raise ValueError("CSV has no data rows")

    numeric_data = {}
    numeric_cols = []

    for col in columns:
        values = []
        is_numeric = True
        for r in rows:
            v = (r.get(col, "") or "").strip()
            try:
                values.append(float(v))
            except ValueError:
                is_numeric = False
                break
        if is_numeric:
            numeric_data[col] = np.array(values, dtype=float)
            numeric_cols.append(col)

    return columns, rows, numeric_data, numeric_cols


def write_csv(columns, rows, updated_data, output_path: Path):
    n = len(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(n):
            out_row = dict(rows[i])
            for col, arr in updated_data.items():
                out_row[col] = f"{arr[i]:.9g}"
            writer.writerow(out_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="FFT low-pass filter for CSV columns")
    parser.add_argument("csv", nargs="?", default="ft_data.csv", help="Input CSV path (default: ft_data.csv)")
    parser.add_argument("--cutoff", type=float, default=100, help="Low-pass cutoff frequency in Hz")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input>_lowpass.csv)",
    )
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name (default: timestamp)")
    args = parser.parse_args()

    input_path = Path(args.csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    columns, rows, numeric_data, numeric_cols = load_csv_numeric(input_path, args.timestamp_col)

    if args.timestamp_col not in numeric_data:
        raise ValueError(f"Timestamp column is not numeric: {args.timestamp_col}")

    timestamp = numeric_data[args.timestamp_col]
    fs = estimate_fs(timestamp)

    if args.cutoff <= 0:
        raise ValueError("--cutoff must be > 0")
    nyquist = fs / 2.0
    if args.cutoff >= nyquist:
        raise ValueError(f"--cutoff must be < Nyquist ({nyquist:.3f} Hz)")

    signal_cols = [c for c in numeric_cols if c != args.timestamp_col]

    if not signal_cols:
        raise ValueError("No numeric signal columns found to filter")

    signals = np.column_stack([numeric_data[col] for col in signal_cols])
    filtered = fft_lowpass(signals, fs=fs, cutoff_hz=args.cutoff)
    filtered_data = {
        col: filtered[:, index]
        for index, col in enumerate(signal_cols)
    }

    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_lowpass.csv")
    else:
        output_path = Path(args.output)

    write_csv(columns, rows, filtered_data, output_path)

    print(f"Input: {input_path}")
    print(f"Estimated fs: {fs:.3f} Hz")
    print(f"Cutoff: {args.cutoff:.3f} Hz")
    print(f"Filtered columns: {', '.join(signal_cols)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

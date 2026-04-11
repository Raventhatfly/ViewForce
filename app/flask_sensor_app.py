import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ft_sensor.mms101.mms101 import MMS101


app = Flask(__name__, template_folder="templates")
DATA_DIR = ROOT_DIR / "data"
FORCE_RAW_DIR = DATA_DIR / "force_raw"
FORCE_FFT_DIR = DATA_DIR / "force_fft"
EPISODE_DIR = DATA_DIR / "episodes"
EPISODE_STAGING_DIR = EPISODE_DIR / "_staging"
EPISODE_INDEX_FILE = EPISODE_DIR / "index.json"


def default_port() -> str:
    is_windows = os.environ.get("OS", "").lower().startswith("windows")
    return "COM3" if is_windows else "/dev/ttyUSB0"


def list_csv_files_in(base_dir: Path, include_freq_files: bool = True) -> List[str]:
    if not base_dir.exists():
        return []
    files = []
    for p in base_dir.rglob("*.csv"):
        if not p.is_file():
            continue
        if (not include_freq_files) and p.name.endswith("_freq.csv"):
            continue
        files.append(p.relative_to(base_dir).as_posix())
    files.sort()
    return files


def resolve_csv_under(base_dir: Path, rel_path: str) -> Path:
    candidate = (base_dir / rel_path).resolve()
    root = base_dir.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Invalid file path")
    if candidate.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("CSV file not found")
    return candidate


def load_csv_waveform(csv_path: Path) -> dict:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise ValueError("CSV has no header")
    if not rows:
        raise ValueError("CSV has no data rows")

    numeric_cols = {}
    for col in fieldnames:
        vals = []
        is_numeric = True
        for row in rows:
            raw = (row.get(col, "") or "").strip()
            try:
                vals.append(float(raw))
            except ValueError:
                is_numeric = False
                break
        if is_numeric:
            numeric_cols[col] = vals

    if not numeric_cols:
        raise ValueError("No numeric columns found")

    x_col = "timestamp" if "timestamp" in numeric_cols else next(iter(numeric_cols.keys()))
    traces = {k: v for k, v in numeric_cols.items() if k != x_col}
    if not traces:
        traces = {x_col: numeric_cols[x_col]}

    return {
        "x_col": x_col,
        "x": numeric_cols[x_col],
        "traces": traces,
        "rows": len(rows),
        "file": csv_path.name,
    }


def estimate_fs_from_timestamp(timestamp: np.ndarray) -> float:
    if len(timestamp) < 2:
        raise ValueError("Need at least 2 timestamp samples to estimate sample rate")

    dt = np.diff(timestamp)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Invalid timestamp: expected increasing numeric values")

    return 1.0 / float(np.median(dt))


def _sanitize_output_name(name: str) -> str:
    clean = name.replace("\\", "_").replace("/", "_")
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", clean)
    clean = clean.strip("._")
    return clean


def run_fft_lowpass(csv_path: Path, cutoff_hz: float, output_name: str = "") -> dict:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise ValueError("CSV has no header")
    if not rows:
        raise ValueError("CSV has no data rows")
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be > 0")

    numeric_cols = {}
    for col in fieldnames:
        vals = []
        is_numeric = True
        for row in rows:
            raw = (row.get(col, "") or "").strip()
            try:
                vals.append(float(raw))
            except ValueError:
                is_numeric = False
                break
        if is_numeric:
            numeric_cols[col] = np.array(vals, dtype=float)

    if "timestamp" not in numeric_cols:
        raise ValueError("CSV must contain numeric 'timestamp' for FFT low-pass")

    fs = estimate_fs_from_timestamp(numeric_cols["timestamp"])
    nyquist = fs / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be < Nyquist ({nyquist:.4f} Hz)")

    signal_cols = [c for c in numeric_cols.keys() if c != "timestamp"]
    if not signal_cols:
        raise ValueError("No numeric signal columns available to filter")

    filtered = {}
    fft_freqs = None
    fft_magnitude = {}

    for col in signal_cols:
        x = numeric_cols[col]
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.fft.rfft(x)
        filtered_spectrum = spectrum.copy()
        filtered_spectrum[freqs > cutoff_hz] = 0.0
        y = np.fft.irfft(filtered_spectrum, n=n)

        filtered[col] = y
        fft_freqs = freqs
        fft_magnitude[col] = np.abs(filtered_spectrum)

    FORCE_FFT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name.strip():
        clean = _sanitize_output_name(output_name.strip())
        if not clean:
            raise ValueError("Invalid output file name")
        if not clean.lower().endswith(".csv"):
            clean = f"{clean}.csv"
        out_path = FORCE_FFT_DIR / clean
    else:
        cutoff_tag = str(cutoff_hz).replace(".", "p")
        out_path = FORCE_FFT_DIR / f"{csv_path.stem}_lp_{cutoff_tag}hz.csv"

    out_fieldnames = list(fieldnames)
    if "cutoff_hz" not in out_fieldnames:
        out_fieldnames.append("cutoff_hz")
    if "source_raw_csv" not in out_fieldnames:
        out_fieldnames.append("source_raw_csv")
    if "freq_csv" not in out_fieldnames:
        out_fieldnames.append("freq_csv")

    source_raw_rel = csv_path.relative_to(FORCE_RAW_DIR).as_posix() if FORCE_RAW_DIR.resolve() in csv_path.resolve().parents else csv_path.name
    freq_path = out_path.with_name(f"{out_path.stem}_freq.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out_row = dict(row)
            for col in signal_cols:
                out_row[col] = f"{filtered[col][i]:.9g}"
            out_row["cutoff_hz"] = f"{cutoff_hz:.9g}"
            out_row["source_raw_csv"] = source_raw_rel
            out_row["freq_csv"] = freq_path.name
            writer.writerow(out_row)

    freq_fieldnames = ["freq_hz"] + signal_cols + ["cutoff_hz", "source_raw_csv", "filtered_csv"]
    with open(freq_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=freq_fieldnames)
        writer.writeheader()
        for i, freq in enumerate(fft_freqs if fft_freqs is not None else []):
            row = {"freq_hz": f"{freq:.9g}", "cutoff_hz": f"{cutoff_hz:.9g}", "source_raw_csv": source_raw_rel, "filtered_csv": out_path.name}
            for col in signal_cols:
                row[col] = f"{fft_magnitude[col][i]:.9g}"
            writer.writerow(row)

    return {
        "output_path": out_path,
        "output_rel_path": out_path.relative_to(FORCE_FFT_DIR).as_posix(),
        "freq_path": freq_path,
        "freq_rel_path": freq_path.relative_to(FORCE_FFT_DIR).as_posix(),
        "cutoff_hz": cutoff_hz,
        "sample_rate_hz": fs,
        "nyquist_hz": nyquist,
        "fft_freqs": fft_freqs.tolist() if fft_freqs is not None else [],
        "fft_magnitude": {k: v.tolist() for k, v in fft_magnitude.items()},
        "source_raw_csv": source_raw_rel,
    }


def load_fft_frequency_data(freq_csv_path: Path) -> dict:
    with open(freq_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        raise ValueError("FFT frequency CSV is empty")

    mag_cols = [c for c in fieldnames if c not in {"freq_hz", "cutoff_hz", "source_raw_csv", "filtered_csv"}]
    freqs = []
    magnitude = {c: [] for c in mag_cols}

    for row in rows:
        freqs.append(float((row.get("freq_hz") or "0").strip()))
        for c in mag_cols:
            magnitude[c].append(float((row.get(c) or "0").strip()))

    cutoff_hz = float((rows[0].get("cutoff_hz") or "0").strip())
    source_raw_csv = rows[0].get("source_raw_csv") or ""
    filtered_csv = rows[0].get("filtered_csv") or ""

    return {
        "freqs": freqs,
        "magnitude": magnitude,
        "cutoff_hz": cutoff_hz,
        "source_raw_csv": source_raw_csv,
        "filtered_csv": filtered_csv,
    }


class SensorController:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sensor: Optional[MMS101] = None
        self.sensor_open = False

        self.warmup_running = False
        self.warmup_end_ts = 0.0
        self.warmup_thread: Optional[threading.Thread] = None

        self.recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_record_event = threading.Event()
        self.last_file: Optional[str] = None

        self.params = {
            "port": default_port(),
            "medfilt_num": 5,
            "warmup_seconds": 300,
            "tare_samples": 50,
            "sample_interval": 0.05,
            "verbose": True,
        }

    def _warmup_worker(self, tare_samples: int) -> None:
        while True:
            with self.lock:
                if not self.warmup_running or not self.sensor_open:
                    return
                remaining = self.warmup_end_ts - time.time()
            if remaining <= 0:
                break
            time.sleep(0.2)

        try:
            sensor = None
            with self.lock:
                if self.sensor_open:
                    sensor = self.sensor
            if sensor is not None:
                sensor.tare(n_samples=tare_samples)
        finally:
            with self.lock:
                self.warmup_running = False

    def open_sensor(self, *, port: str, medfilt_num: int, warmup_seconds: int, tare_samples: int, verbose: bool) -> None:
        with self.lock:
            if self.sensor_open:
                raise RuntimeError("Sensor already opened")

        sensor = MMS101(port=port, medfilt_num=medfilt_num, verbose=verbose)
        sensor.start()

        # Debug convenience: allow warmup_seconds=0 to skip countdown lock.
        if warmup_seconds <= 0:
            sensor.tare(n_samples=tare_samples)

        with self.lock:
            self.sensor = sensor
            self.sensor_open = True
            self.recording = False
            self.stop_record_event.clear()
            self.last_file = None

            self.params.update(
                {
                    "port": port,
                    "medfilt_num": medfilt_num,
                    "warmup_seconds": warmup_seconds,
                    "tare_samples": tare_samples,
                    "verbose": verbose,
                }
            )

            if warmup_seconds > 0:
                self.warmup_running = True
                self.warmup_end_ts = time.time() + float(warmup_seconds)
                self.warmup_thread = threading.Thread(target=self._warmup_worker, args=(tare_samples,), daemon=True)
                self.warmup_thread.start()
            else:
                self.warmup_running = False
                self.warmup_end_ts = 0.0
                self.warmup_thread = None

    def _record_worker(self, csv_path: Path, interval_s: float) -> None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "Fx", "Fy", "Fz", "Mx", "My", "Mz"])

            while not self.stop_record_event.is_set():
                with self.lock:
                    sensor = self.sensor if self.sensor_open else None
                if sensor is None:
                    break

                ft: List[float] = sensor.get_ft_tared()
                ts = time.time()
                writer.writerow([f"{ts:.6f}"] + [f"{v:.6f}" for v in ft])
                f.flush()
                time.sleep(interval_s)

        with self.lock:
            self.recording = False
            self.recording_thread = None
            self.stop_record_event.clear()

    def start_recording(self, csv_name: str, sample_interval: float) -> str:
        with self.lock:
            if not self.sensor_open:
                raise RuntimeError("Sensor is not open")
            if self.warmup_running:
                raise RuntimeError("Warm-up is still running")
            if self.recording:
                raise RuntimeError("Recording already in progress")

            csv_path = self._resolve_csv_path(csv_name)
            self.params["sample_interval"] = sample_interval

            self.stop_record_event.clear()
            self.recording = True
            self.last_file = str(csv_path)
            self.recording_thread = threading.Thread(
                target=self._record_worker,
                args=(csv_path, sample_interval),
                daemon=True,
            )
            self.recording_thread.start()
            return str(csv_path)

    def stop_recording(self) -> Optional[str]:
        with self.lock:
            if self.warmup_running:
                raise RuntimeError("Cannot stop recording during warm-up")
            if not self.recording:
                raise RuntimeError("Recording is not in progress")

            record_thread = self.recording_thread
            self.stop_record_event.set()

        if record_thread is not None:
            record_thread.join(timeout=5)

        with self.lock:
            return self.last_file

    def close_sensor(self) -> None:
        with self.lock:
            sensor = self.sensor
            recording = self.recording

        if recording:
            self.stop_record_event.set()
            with self.lock:
                record_thread = self.recording_thread
            if record_thread is not None:
                record_thread.join(timeout=5)

        if sensor is not None:
            sensor.stop()

        with self.lock:
            self.sensor = None
            self.sensor_open = False
            self.warmup_running = False
            self.warmup_end_ts = 0.0
            self.recording = False
            self.recording_thread = None

    def get_status(self) -> dict:
        with self.lock:
            warmup_remaining = max(0, int(self.warmup_end_ts - time.time())) if self.warmup_running else 0
            sensor_open = self.sensor_open
            recording = self.recording
            warmup_running = self.warmup_running
            params = dict(self.params)
            last_file = self.last_file
            sensor = self.sensor

        latest_ft = None
        if sensor_open and sensor is not None:
            try:
                latest_ft = [float(v) for v in sensor.get_ft_tared()]
            except Exception:
                latest_ft = None

        return {
            "sensor_open": sensor_open,
            "warmup_running": warmup_running,
            "warmup_remaining": warmup_remaining,
            "recording": recording,
            "can_start_recording": sensor_open and (not warmup_running) and (not recording),
            "can_stop_recording": sensor_open and (not warmup_running) and recording,
            "latest_ft": latest_ft,
            "last_file": last_file,
            "params": params,
        }

    @staticmethod
    def _resolve_csv_path(csv_name: str) -> Path:
        FORCE_RAW_DIR.mkdir(parents=True, exist_ok=True)

        if csv_name.strip():
            clean_name = SensorController._sanitize_filename(csv_name.strip())
            if not clean_name.lower().endswith(".csv"):
                clean_name = f"{clean_name}.csv"
            return FORCE_RAW_DIR / clean_name

        ts = int(time.time())
        return FORCE_RAW_DIR / f"raw_{ts}.csv"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.replace("\\", "_").replace("/", "_")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        name = name.strip("._")
        return name or f"raw_{int(time.time())}"


controller = SensorController()


class RealSenseController:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pipeline = None
        self.running = False
        self.last_error = ""
        self.width = 640
        self.height = 480
        self.fps = 30

    def _require_rs(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self._require_rs()
        width = max(160, int(width))
        height = max(120, int(height))
        fps = max(1, int(fps))

        with self.lock:
            if self.running:
                return

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                pipeline.start(config)
            except Exception as e:
                self.last_error = str(e)
                raise RuntimeError(f"Failed to start RealSense: {e}")

            self.pipeline = pipeline
            self.running = True
            self.last_error = ""
            self.width = width
            self.height = height
            self.fps = fps

    def stop(self) -> None:
        with self.lock:
            pipeline = self.pipeline
            self.pipeline = None
            self.running = False
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def status(self) -> dict:
        with self.lock:
            return {
                "available": rs is not None,
                "running": self.running,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "last_error": self.last_error,
                "encoder": "opencv" if cv2 is not None else ("pillow" if Image is not None else "none"),
            }

    def _encode_jpeg(self, frame_bgr: np.ndarray) -> bytes:
        if cv2 is not None:
            ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("OpenCV JPEG encoding failed")
            return enc.tobytes()

        if Image is not None:
            rgb = frame_bgr[:, :, ::-1]
            im = Image.fromarray(rgb)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return buf.getvalue()

        raise RuntimeError("No JPEG encoder available. Install opencv-python or pillow")

    def get_frame_jpeg(self) -> bytes:
        bgr, _ = self.get_frame_bgr()
        return self._encode_jpeg(bgr)

    def get_frame_bgr(self) -> tuple[np.ndarray, float]:
        with self.lock:
            if not self.running or self.pipeline is None:
                raise RuntimeError("RealSense is not running")
            pipeline = self.pipeline

        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError("No color frame from RealSense")
            bgr = np.asanyarray(color.get_data())
            cam_ts_ms = float(color.get_timestamp())
            return bgr, cam_ts_ms
        except Exception as e:
            with self.lock:
                self.last_error = str(e)
            raise


camera = RealSenseController()


class EpisodeController:
    def __init__(self, sensor_controller: SensorController, camera_controller: RealSenseController) -> None:
        self.sensor_controller = sensor_controller
        self.camera_controller = camera_controller
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.active = False
        self.pending: Optional[Dict[str, Any]] = None
        self.episode_thread_force: Optional[threading.Thread] = None
        self.episode_thread_camera: Optional[threading.Thread] = None
        self.current: Optional[Dict[str, Any]] = None
        self.last_error = ""

        EPISODE_DIR.mkdir(parents=True, exist_ok=True)
        EPISODE_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> Dict[str, Any]:
        if not EPISODE_INDEX_FILE.exists():
            return {"next_episode_id": 1, "episodes": []}
        try:
            with open(EPISODE_INDEX_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            next_id = int(data.get("next_episode_id", 1))
            episodes = data.get("episodes", [])
            if not isinstance(episodes, list):
                episodes = []
            return {"next_episode_id": max(1, next_id), "episodes": episodes}
        except Exception:
            return {"next_episode_id": 1, "episodes": []}

    def _save_index(self, data: Dict[str, Any]) -> None:
        EPISODE_DIR.mkdir(parents=True, exist_ok=True)
        with open(EPISODE_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: EpisodeController._jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [EpisodeController._jsonable(v) for v in value]
        return value

    def _current_next_id(self) -> int:
        return int(self._load_index().get("next_episode_id", 1))

    def _force_worker(self, csv_path: Path, t0_perf: float, sample_interval: float) -> None:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "t_rel_s", "t_wall_s", "Fx", "Fy", "Fz", "Mx", "My", "Mz"])
            idx = 0
            while not self.stop_event.is_set():
                status = self.sensor_controller.get_status()
                if not status.get("sensor_open", False):
                    self.last_error = "Sensor closed during episode capture"
                    self.stop_event.set()
                    break

                ft = status.get("latest_ft")
                if ft is None:
                    try:
                        with self.sensor_controller.lock:
                            s = self.sensor_controller.sensor
                        ft = [float(v) for v in s.get_ft_tared()] if s is not None else None
                    except Exception:
                        ft = None

                if ft is not None:
                    t_rel = time.perf_counter() - t0_perf
                    t_wall = time.time()
                    writer.writerow([idx, f"{t_rel:.6f}", f"{t_wall:.6f}"] + [f"{float(v):.6f}" for v in ft])
                    if idx % 10 == 0:
                        f.flush()
                    idx += 1
                    with self.lock:
                        if self.current is not None:
                            self.current["force_samples"] = idx
                time.sleep(sample_interval)

    def _camera_worker(self, video_path: Path, frame_csv_path: Path, t0_perf: float, fps: int, width: int, height: int) -> None:
        if cv2 is None:
            self.last_error = "opencv-python is required for episode video recording"
            self.stop_event.set()
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(max(1, fps)), (int(width), int(height)))
        if not writer.isOpened():
            self.last_error = "Failed to open VideoWriter for episode video"
            self.stop_event.set()
            return

        try:
            with open(frame_csv_path, "w", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["frame_idx", "t_rel_s", "t_wall_s", "camera_ts_ms"])
                frame_idx = 0
                while not self.stop_event.is_set():
                    try:
                        frame_bgr, cam_ts_ms = self.camera_controller.get_frame_bgr()
                    except Exception as e:
                        self.last_error = str(e)
                        self.stop_event.set()
                        break

                    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                        frame_bgr = cv2.resize(frame_bgr, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)

                    writer.write(frame_bgr)
                    t_rel = time.perf_counter() - t0_perf
                    t_wall = time.time()
                    csv_writer.writerow([frame_idx, f"{t_rel:.6f}", f"{t_wall:.6f}", f"{cam_ts_ms:.3f}"])
                    if frame_idx % 10 == 0:
                        f.flush()
                    frame_idx += 1
                    with self.lock:
                        if self.current is not None:
                            self.current["frame_count"] = frame_idx
        finally:
            writer.release()

    def start_episode(self, label: str = "", notes: str = "") -> Dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("Episode is already running")
            if self.pending is not None:
                raise RuntimeError("Resolve pending episode first (keep/discard)")

        sensor_status = self.sensor_controller.get_status()
        camera_status = self.camera_controller.status()

        if not sensor_status.get("sensor_open", False):
            raise RuntimeError("Sensor is not open")
        if sensor_status.get("warmup_running", False):
            raise RuntimeError("Sensor warm-up is still running")
        if sensor_status.get("recording", False):
            raise RuntimeError("Raw recording is running; stop it before episode capture")
        if not camera_status.get("running", False):
            raise RuntimeError("RealSense camera is not running")

        episode_id = self._current_next_id()
        start_wall = time.time()
        t0_perf = time.perf_counter()
        staging_dir = EPISODE_STAGING_DIR / f"ep_{episode_id:06d}_{int(start_wall)}"
        staging_dir.mkdir(parents=True, exist_ok=False)

        fps = int(camera_status.get("fps", 30))
        width = int(camera_status.get("width", 640))
        height = int(camera_status.get("height", 480))
        sample_interval = float(sensor_status.get("params", {}).get("sample_interval", 0.05))

        meta = {
            "episode_id": episode_id,
            "status": "recording",
            "keep": None,
            "start_wall_time": start_wall,
            "end_wall_time": None,
            "duration_s": None,
            "label": label,
            "notes": notes,
            "sensor": {
                "port": sensor_status.get("params", {}).get("port"),
                "medfilt_num": sensor_status.get("params", {}).get("medfilt_num"),
                "sample_interval": sample_interval,
            },
            "camera": {"width": width, "height": height, "fps": fps},
            "files": {
                "video": "video.mp4",
                "frame_timestamps": "frame_timestamps.csv",
                "force": "force_timestamps.csv",
            },
        }

        with open(staging_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

        self.stop_event.clear()
        self.last_error = ""
        with self.lock:
            self.active = True
            self.current = {
                "episode_id": episode_id,
                "staging_dir": staging_dir,
                "start_wall": start_wall,
                "t0_perf": t0_perf,
                "force_samples": 0,
                "frame_count": 0,
                "sample_interval": sample_interval,
                "camera_fps": fps,
                "camera_width": width,
                "camera_height": height,
                "label": label,
                "notes": notes,
            }

        self.episode_thread_force = threading.Thread(
            target=self._force_worker,
            args=(staging_dir / "force_timestamps.csv", t0_perf, sample_interval),
            daemon=True,
        )
        self.episode_thread_camera = threading.Thread(
            target=self._camera_worker,
            args=(staging_dir / "video.mp4", staging_dir / "frame_timestamps.csv", t0_perf, fps, width, height),
            daemon=True,
        )
        self.episode_thread_force.start()
        self.episode_thread_camera.start()

        return {"episode_id": episode_id, "staging_dir": staging_dir.name}

    def stop_episode(self) -> Dict[str, Any]:
        with self.lock:
            if not self.active or self.current is None:
                raise RuntimeError("No active episode")
            cur = dict(self.current)

        self.stop_event.set()

        if self.episode_thread_force is not None:
            self.episode_thread_force.join(timeout=8)
        if self.episode_thread_camera is not None:
            self.episode_thread_camera.join(timeout=8)

        end_wall = time.time()
        duration_s = max(0.0, end_wall - float(cur["start_wall"]))
        staging_dir = Path(cur["staging_dir"])

        meta_path = staging_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        meta.update(
            {
                "episode_id": int(cur["episode_id"]),
                "status": "pending_decision",
                "keep": None,
                "end_wall_time": end_wall,
                "duration_s": duration_s,
                "force_samples": int(cur.get("force_samples", 0)),
                "frame_count": int(cur.get("frame_count", 0)),
                "last_error": self.last_error,
            }
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

        pending = {
            "episode_id": int(cur["episode_id"]),
            "staging_dir": staging_dir.name,
            "duration_s": duration_s,
            "force_samples": int(cur.get("force_samples", 0)),
            "frame_count": int(cur.get("frame_count", 0)),
            "last_error": self.last_error,
            "label": cur.get("label", ""),
            "notes": cur.get("notes", ""),
        }

        with self.lock:
            self.active = False
            self.current = None
            self.pending = pending
            self.episode_thread_force = None
            self.episode_thread_camera = None

        return pending

    def commit_episode(self, keep: bool) -> Dict[str, Any]:
        with self.lock:
            if self.active:
                raise RuntimeError("Cannot commit while an episode is still running")
            if self.pending is None:
                raise RuntimeError("No pending episode to commit")
            pending = dict(self.pending)

        episode_id = int(pending["episode_id"])
        staging_dir = EPISODE_STAGING_DIR / str(pending["staging_dir"])

        if keep:
            final_dir = EPISODE_DIR / f"EP{episode_id:06d}"
            if final_dir.exists():
                suffix = int(time.time())
                final_dir = EPISODE_DIR / f"EP{episode_id:06d}_{suffix}"

            shutil.move(str(staging_dir), str(final_dir))

            meta_path = final_dir / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            meta.update({"keep": True, "status": "saved", "saved_dir": final_dir.name})
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=True, indent=2)

            index = self._load_index()
            episodes = index.get("episodes", [])
            episodes.append(
                {
                    "episode_id": episode_id,
                    "dir": final_dir.name,
                    "created_at": time.time(),
                    "duration_s": pending.get("duration_s"),
                    "frame_count": pending.get("frame_count"),
                    "force_samples": pending.get("force_samples"),
                    "label": pending.get("label", ""),
                }
            )
            index["episodes"] = episodes
            index["next_episode_id"] = max(int(index.get("next_episode_id", 1)), episode_id + 1)
            self._save_index(index)

            result = {"kept": True, "episode_id": episode_id, "saved_dir": final_dir.name}
        else:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            result = {"kept": False, "episode_id": episode_id}

        with self.lock:
            self.pending = None

        return result

    def status(self) -> Dict[str, Any]:
        with self.lock:
            active = self.active
            current = dict(self.current) if self.current is not None else None
            pending = dict(self.pending) if self.pending is not None else None
            last_error = self.last_error

        index = self._load_index()
        episodes = index.get("episodes", [])
        recent = episodes[-5:]

        return {
            "active": active,
            "current": self._jsonable(current),
            "pending": self._jsonable(pending),
            "next_episode_id": int(index.get("next_episode_id", 1)),
            "episode_count": len(episodes),
            "recent_episodes": self._jsonable(recent),
            "last_error": last_error,
        }


episodes = EpisodeController(sensor_controller=controller, camera_controller=camera)


@app.get("/")
def index():
    return render_template("sensor_ui.html", default_port=default_port())


@app.get("/api/status")
def api_status():
    return jsonify(controller.get_status())


@app.get("/api/csv_files")
def api_csv_files():
    return jsonify({"ok": True, "files": list_csv_files_in(DATA_DIR, include_freq_files=True)})


@app.get("/api/csv_data")
def api_csv_data():
    rel_path = (request.args.get("file") or "").strip()
    if not rel_path:
        return jsonify({"ok": False, "message": "Missing file query parameter"}), 400

    try:
        csv_path = resolve_csv_under(DATA_DIR, rel_path)
        payload = load_csv_waveform(csv_path)
        payload["ok"] = True
        payload["path"] = rel_path
        return jsonify(payload)
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/raw_files")
def api_raw_files():
    return jsonify({"ok": True, "files": list_csv_files_in(FORCE_RAW_DIR, include_freq_files=False)})


@app.get("/api/fft_files")
def api_fft_files():
    return jsonify({"ok": True, "files": list_csv_files_in(FORCE_FFT_DIR, include_freq_files=False)})


@app.get("/api/raw_data")
def api_raw_data():
    rel_path = (request.args.get("file") or "").strip()
    if not rel_path:
        return jsonify({"ok": False, "message": "Missing file query parameter"}), 400

    try:
        csv_path = resolve_csv_under(FORCE_RAW_DIR, rel_path)
        payload = load_csv_waveform(csv_path)
        payload["ok"] = True
        payload["path"] = rel_path
        return jsonify(payload)
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/fft_data")
def api_fft_data():
    rel_path = (request.args.get("file") or "").strip()
    if not rel_path:
        return jsonify({"ok": False, "message": "Missing file query parameter"}), 400

    try:
        fft_csv_path = resolve_csv_under(FORCE_FFT_DIR, rel_path)
        wave = load_csv_waveform(fft_csv_path)

        wave["traces"].pop("cutoff_hz", None)
        wave["traces"].pop("timestamp", None)

        freq_csv_path = fft_csv_path.with_name(f"{fft_csv_path.stem}_freq.csv")
        freq_payload = load_fft_frequency_data(freq_csv_path)

        wave["ok"] = True
        wave["path"] = rel_path
        wave["fft"] = freq_payload
        return jsonify(wave)
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/fft_lowpass")
def api_fft_lowpass():
    payload = request.get_json(silent=True) or {}

    try:
        rel_path = str(payload.get("file", "")).strip()
        cutoff_hz = float(payload.get("cutoff_hz", 0))
        output_name = str(payload.get("output_name", "")).strip()
        if not rel_path:
            raise ValueError("Missing CSV file path")

        csv_path = resolve_csv_under(FORCE_RAW_DIR, rel_path)
        result = run_fft_lowpass(csv_path=csv_path, cutoff_hz=cutoff_hz, output_name=output_name)

        return jsonify(
            {
                "ok": True,
                "message": "FFT low-pass completed.",
                "output_file": str(result["output_path"]),
                "output_rel_path": result["output_rel_path"],
                "freq_rel_path": result["freq_rel_path"],
                "cutoff_hz": result["cutoff_hz"],
                "sample_rate_hz": result["sample_rate_hz"],
                "nyquist_hz": result["nyquist_hz"],
                "source_raw_csv": result["source_raw_csv"],
                "fft": {
                    "freqs": result["fft_freqs"],
                    "magnitude": result["fft_magnitude"],
                },
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/open_sensor")
def api_open_sensor():
    payload = request.get_json(silent=True) or {}

    try:
        port = str(payload.get("port") or default_port())
        medfilt_num = int(payload.get("medfilt_num", 5))
        warmup_seconds = int(payload.get("warmup_seconds", 300))
        tare_samples = int(payload.get("tare_samples", 50))
        verbose = bool(payload.get("verbose", True))

        if medfilt_num < 1:
            raise ValueError("medfilt_num must be >= 1")
        if warmup_seconds < 0:
            raise ValueError("warmup_seconds must be >= 0")
        if tare_samples < 1:
            raise ValueError("tare_samples must be >= 1")

        controller.open_sensor(
            port=port,
            medfilt_num=medfilt_num,
            warmup_seconds=warmup_seconds,
            tare_samples=tare_samples,
            verbose=verbose,
        )
        return jsonify({"ok": True, "message": "Sensor opened. Warm-up started."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/start_recording")
def api_start_recording():
    payload = request.get_json(silent=True) or {}

    try:
        csv_name = str(payload.get("csv_name", ""))
        sample_interval = float(payload.get("sample_interval", 0.05))
        if sample_interval <= 0:
            raise ValueError("sample_interval must be > 0")

        file_path = controller.start_recording(csv_name=csv_name, sample_interval=sample_interval)
        return jsonify({"ok": True, "message": "Recording started.", "file": file_path})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/stop_recording")
def api_stop_recording():
    try:
        file_path = controller.stop_recording()
        return jsonify({"ok": True, "message": "Recording stopped.", "file": file_path})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/close_sensor")
def api_close_sensor():
    try:
        if episodes.status().get("active", False):
            return jsonify({"ok": False, "message": "Stop episode capture before closing sensor"}), 400
        controller.close_sensor()
        return jsonify({"ok": True, "message": "Sensor closed."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/episode_status")
def api_episode_status():
    st = episodes.status()
    st["ok"] = True
    return jsonify(st)


@app.post("/api/episode_start")
def api_episode_start():
    payload = request.get_json(silent=True) or {}
    try:
        label = str(payload.get("label", "")).strip()
        notes = str(payload.get("notes", "")).strip()
        ret = episodes.start_episode(label=label, notes=notes)
        return jsonify({"ok": True, "message": "Episode started.", **ret})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/episode_stop")
def api_episode_stop():
    try:
        ret = episodes.stop_episode()
        return jsonify({"ok": True, "message": "Episode stopped. Please decide keep/discard.", **ret})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/episode_commit")
def api_episode_commit():
    payload = request.get_json(silent=True) or {}
    try:
        keep = bool(payload.get("keep", False))
        ret = episodes.commit_episode(keep=keep)
        msg = "Episode kept." if keep else "Episode discarded."
        return jsonify({"ok": True, "message": msg, **ret})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/realsense_status")
def api_realsense_status():
    st = camera.status()
    st["ok"] = True
    return jsonify(st)


@app.post("/api/realsense_start")
def api_realsense_start():
    payload = request.get_json(silent=True) or {}
    try:
        width = int(payload.get("width", 640))
        height = int(payload.get("height", 480))
        fps = int(payload.get("fps", 30))
        camera.start(width=width, height=height, fps=fps)
        return jsonify({"ok": True, "message": "RealSense started."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.post("/api/realsense_stop")
def api_realsense_stop():
    try:
        camera.stop()
        return jsonify({"ok": True, "message": "RealSense stopped."})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.get("/api/realsense_frame")
def api_realsense_frame():
    try:
        if not camera.status().get("running", False):
            return jsonify({"ok": False, "message": "RealSense is not running"}), 409
        jpeg = camera.get_frame_jpeg()
        return Response(
            jpeg,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 400


@app.errorhandler(404)
def handle_404(err):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "message": f"API not found: {request.path}"}), 404
    return err


@app.errorhandler(Exception)
def handle_exception(err):
    if request.path.startswith("/api/"):
        if isinstance(err, HTTPException):
            return jsonify({"ok": False, "message": err.description}), err.code
        return jsonify({"ok": False, "message": str(err)}), 500
    raise err


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

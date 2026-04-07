import threading
import time
import csv
import os
from collections import deque
import numpy as np
import serial

DEV_SERIAL_PATH_LINUX = '/dev/ttyUSB0'
DEV_SERIAL_PATH_WINDOWS = 'COM5'
SERIAL_BAUDRATE = 1_000_000
INTERVAL_MEASURE_TIME = 1000       # microseconds → ~1000 Hz
INTERVAL_RESTART_TIME = 1_000_000  # microseconds → temperature update every ~1 min


class MMS101:
    def __init__(self, port=None, medfilt_num=5, log_csv=None, verbose=True):
        """
        MMS101 6-axis force/torque sensor driver.

        Args:
            port (str): Serial port. Defaults to '/dev/ttyUSB0' (Linux) or 'COM5' (Windows).
            medfilt_num (int): Median filter window size. Set to 1 to disable filtering.
            log_csv (str): Path to CSV log file. If None, logging is disabled.
            verbose (bool): Print status messages.
        """
        if port is None:
            port = DEV_SERIAL_PATH_WINDOWS if os.name == 'nt' else DEV_SERIAL_PATH_LINUX

        self.port = port
        self.baudrate = SERIAL_BAUDRATE
        self.medfilt_num = medfilt_num
        self.log_csv = log_csv
        self.verbose = verbose

        self._serial = None
        self._serial_open = False
        self._board_selected = False

        self._data = [0.0] * 6        # latest raw reading [Fx,Fy,Fz,Mx,My,Mz]
        self._data_que = deque(maxlen=medfilt_num)
        self._zeros = [0.0] * 6       # tare offsets

        self._thread = None
        self._running = False

        self._csv_file = None
        self._csv_writer = None

        self._elaps_time = 0.0
        self._interval_measure = INTERVAL_MEASURE_TIME
        self._interval_restart = INTERVAL_RESTART_TIME

    def _serial_port_open(self):
        self._serial = serial.Serial(self.port, self.baudrate, timeout=2)
        self._serial_open = True
        if self.verbose:
            print(f"[MMS101] Serial port opened: {self.port}")

    def _serial_port_close(self):
        if self._serial_open:
            try:
                self._stop_measure()
            except Exception:
                pass
            self._serial.close()
            self._serial_open = False
            if self.verbose:
                print("[MMS101] Serial port closed.")

    def _read_all(self):
        j = self._serial.read(2)
        if j[0] == 0 and j[1] > 0:
            data = self._serial.read(j[1])
            j = j + data
        return j

    def _board_select(self):
        self._serial.write([0x54, 0x02, 0x10, 0x00])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Board Select failed")
        self._board_selected = True

    def _power_switch(self):
        self._serial.write([0x54, 0x03, 0x36, 0x00, 0xFF])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0 and rsp[0] != 0x10:
            raise RuntimeError("MMS101: Power Switch 1-2 failed")
        self._serial.write([0x54, 0x03, 0x36, 0x05, 0xFF])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0 and rsp[0] != 0x10:
            raise RuntimeError("MMS101: Power Switch 4-5 failed")

    def _axis_select_and_idle(self):
        for axis in range(6):
            self._serial.write([0x54, 0x02, 0x1C, axis])
            self._serial.flush()
            rsp = self._read_all()
            if rsp[0] != 0 and rsp[0] != 0x10:
                raise RuntimeError(f"MMS101: Axis Select failed for axis {axis}")
            self._serial.write([0x53, 0x02, 0x57, 0x94])
            self._serial.flush()
            rsp = self._read_all()
            if rsp[0] != 0 and rsp[0] != 0x10:
                raise RuntimeError(f"MMS101: IDLE failed for axis {axis}")
        time.sleep(0.01)

    def _bootload(self):
        self._serial.write([0x54, 0x01, 0xB0])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Bootload failed")

    def _set_interval_measure(self, interval_us):
        self._interval_measure = interval_us
        b1 = (interval_us >> 16) & 0xFF
        b2 = (interval_us >> 8) & 0xFF
        b3 = interval_us & 0xFF
        self._serial.write([0x54, 0x04, 0x43, b1, b2, b3])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Set Interval Measure failed")

    def _set_interval_restart(self, interval_us):
        self._interval_restart = interval_us
        b1 = (interval_us >> 16) & 0xFF
        b2 = (interval_us >> 8) & 0xFF
        b3 = interval_us & 0xFF
        self._serial.write([0x54, 0x04, 0x44, b1, b2, b3])
        self._serial.flush()
        rsp = self._read_all()
        if rsp[0] != 0:
            raise RuntimeError("MMS101: Set Interval Restart failed")

    def _start_measure(self):
        self._serial.write([0x54, 0x02, 0x23, 0x00])
        j = self._serial.read(2)
        if len(j) != 2 or j[0] != 0 or j[1] != 0:
            raise RuntimeError("MMS101: Start Measure failed")
        time.sleep(0.01)

    def _stop_measure(self):
        if self._board_selected:
            self._serial.write([0x54, 0x01, 0x33])
            self._board_selected = False

    def _read_data_raw(self):
        """Sync to packet header and read one 23-byte data packet."""
        prev = b'\xff'
        while True:
            curr = self._serial.read(1)
            if not curr:
                raise RuntimeError("MMS101: Serial timeout while syncing to packet header")
            if prev == b'\x00' and curr == b'\x17':
                break
            prev = curr
        return self._serial.read(23)

    def _parse_data(self, rdata):
        """Parse raw 23-byte packet into [Fx, Fy, Fz, Mx, My, Mz] in N / N·m."""
        if len(rdata) != 23 or rdata[0] != 0x80 or rdata[1] != 0x00:
            return None
        data = [0.0] * 6
        self._elaps_time += ((rdata[20] << 16) + (rdata[21] << 8) + rdata[22]) / 1_000_000
        for axis in range(6):
            val = (rdata[axis*3+2] << 16) + (rdata[axis*3+3] << 8) + rdata[axis*3+4]
            if val >= 0x00800000:
                val -= 0x1000000
            data[axis] = val
        # Scale: forces in N (divide by 1000), moments in N·m (divide by 100000)
        data[0] /= 1000
        data[1] /= 1000
        data[2] /= 1000
        data[3] /= 100000
        data[4] /= 100000
        data[5] /= 100000
        return data

    def _thread_loop(self):
        while self._running:
            try:
                rdata = self._read_data_raw()
                parsed = self._parse_data(rdata)
                if parsed is not None:
                    self._data = parsed
                    self._data_que.append(parsed)
                    if self._csv_writer is not None:
                        ts = time.time()
                        self._csv_writer.writerow([f"{ts:.6f}"] + [f"{v:.6f}" for v in parsed])
            except Exception as e:
                if self._running:
                    print(f"[MMS101] Error in read thread: {e}")
                break

    def start(self):
        """Initialize sensor and start background reading thread."""
        self._serial_port_open()
        self._board_select()
        self._power_switch()
        self._axis_select_and_idle()
        self._bootload()
        self._set_interval_measure(INTERVAL_MEASURE_TIME)
        self._set_interval_restart(INTERVAL_RESTART_TIME)
        self._start_measure()

        if self.log_csv is not None:
            self._csv_file = open(self.log_csv, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['timestamp', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])

        self._running = True
        self._thread = threading.Thread(target=self._thread_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print("[MMS101] Started. NOTE: Allow 3-5 minutes warm-up before recording data.")

    def stop(self):
        """Stop background thread and release resources."""
        self._running = False
        if self._serial is not None and self._serial_open:
            try:
                self._serial.cancel_read()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._csv_file is not None:
            self._csv_writer = None
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        self._serial_port_close()
        if self.verbose:
            print("[MMS101] Stopped.")

    def get_ft(self):
        """
        Return latest median-filtered [Fx, Fy, Fz, Mx, My, Mz] in N / N·m.
        Filter window = medfilt_num (set to 1 in constructor to disable).
        """
        if len(self._data_que) == 0:
            return self._data
        return list(np.median(np.array(self._data_que), axis=0))

    def tare(self, n_samples=50):
        """
        Zero the sensor. Collects n_samples readings and sets them as the offset.
        Call this after warm-up (3-5 min) with no load applied.
        """
        if self.verbose:
            print(f"[MMS101] Taring over {n_samples} samples...")
        samples = []
        for _ in range(n_samples):
            time.sleep(0.002)
            samples.append(self.get_ft())
        self._zeros = list(np.mean(np.array(samples), axis=0))
        if self.verbose:
            print(f"[MMS101] Tare complete. Offsets: {[f'{v:.4f}' for v in self._zeros]}")

    def get_ft_tared(self):
        """Return get_ft() minus tare offsets."""
        ft = self.get_ft()
        return [ft[i] - self._zeros[i] for i in range(6)]

    def __del__(self):
        if self._running:
            self.stop()

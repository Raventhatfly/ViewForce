# MMS101 6-Axis Force/Torque Sensor — Driver

Standalone Python driver for the [NMB MMS101](https://cdn.nmbtc.com/uploads/2023/02/mms101_datasheet_en_rev5.0.pdf) miniature 6-axis force/torque sensor.

---

## ⚠ FPC Connector Warning

**The FPC (Flexible Printed Circuit) connector is fragile. Read this before handling.**

From the MMS101 datasheet:

> "The FPC must NOT be strongly pulled in a lateral or the upper direction while the sensor body is fixed with screws. Otherwise, load is applied to the base of the FPC, and the wiring on the FPC might be snapped."

> "In the FPC termination part, a level difference exists between the FPC and the reinforcing plate. Bending the FPC at this level difference part could cut the wiring on the FPC."

> "Do not bend the FPC at a sharp angle or pull it hard so that the load is concentrated. Otherwise, the wiring on the FPC may be broken, resulting in operation failure."

> "Do not mount the product on moving parts that are bent repeatedly."

**Rules:**
- Never pull the FPC laterally or upward while the sensor is screwed down
- Never bend the FPC at the junction between the FPC and the reinforcing plate
- Fix the PCB connected to the FPC with a screw so the FPC is not repeatedly flexed
- Insert and remove cables only when the FPC is secured to the attachment

---

## ⏱ Warm-Up Notice

**Wait 3–5 minutes after starting before recording data.**

From the datasheet:

> "Immediately after AD converter starts, the built-in AFE heats up and deforms the structure. This causes the output to drift. Therefore, it is recommended to wait for stabilization about 5 min before acquiring data."

Expect **1–2 N of drift** during the initial warm-up period. Always call `tare()` after warm-up and before recording.

---

## Installation

```bash
pip install pyserial numpy
```

Copy `mms101.py` into your project directory.

---

## Hardware Setup

1. Connect the sensor to the controller board via the FPC (see warning above)
2. Connect the controller board to your PC via USB
3. Identify the serial port:
   - **Linux:** Run `ls /dev/ttyUSB*` before and after plugging in. New entry is your port (e.g. `/dev/ttyUSB0`)
   - **Windows:** Check Device Manager → Ports (COM & LPT). Note the `COMx` number.
4. On Linux, you may need to add yourself to the `dialout` group:
   ```bash
   sudo usermod -aG dialout $USER
   # Log out and back in for this to take effect
   ```

---

## Quick Start

```python
from mms101 import MMS101
import time

sensor = MMS101(port='/dev/ttyUSB0')  # 'COM5' on Windows
sensor.start()

time.sleep(300)   # Wait 3-5 minutes for warm-up
sensor.tare()     # Zero with no load applied

ft = sensor.get_ft_tared()
print(f"Fx={ft[0]:.3f}N, Fy={ft[1]:.3f}N, Fz={ft[2]:.3f}N")

sensor.stop()
```

---

## Constructor Options

| Parameter     | Default          | Description                                                  |
|---------------|------------------|--------------------------------------------------------------|
| `port`        | `/dev/ttyUSB0`   | Serial port. Use `'COM5'` on Windows.                        |
| `medfilt_num` | `5`              | Median filter window size. Set to `1` to disable.            |
| `log_csv`     | `None`           | Path to CSV file. If set, all readings are logged.           |
| `verbose`     | `True`           | Print status messages.                                       |

---

## Sensor Frequency

The sensor samples at approximately **1000 Hz** (1000 µs interval, configurable in code).

The datasheet specifies a minimum ADC conversion time of ~781 µs (~1280 Hz theoretical max).

The median filter (`medfilt_num`) does **not** reduce the sampling rate — it smooths values over the last N samples. A window of 5 means each call to `get_ft()` returns the median of the 5 most recent readings.

Output format: `[Fx, Fy, Fz, Mx, My, Mz]`
- Forces in **N** (Newtons), range ±40 N (rated), ±200 N (absolute max)
- Moments in **N·m**, range ±0.4 N·m (rated), ±1.8 N·m (absolute max)

---

## Tare / Zeroing

```python
sensor.tare()              # Average 50 samples as zero offset
ft = sensor.get_ft_tared() # Returns reading minus tare offset
ft_raw = sensor.get_ft()   # Returns raw (un-tared) reading
```

Always tare **after warm-up** and with **no load applied** to the sensor.

---

## CSV Logging

```python
sensor = MMS101(port='/dev/ttyUSB0', log_csv='ft_data.csv')
sensor.start()
# ... run your experiment ...
sensor.stop()  # Flushes and closes the CSV file
```

CSV format:
```
timestamp,Fx,Fy,Fz,Mx,My,Mz
1743480000.123456,0.001234,-0.000512,9.810000,0.000010,-0.000020,0.000003
...
```

`timestamp` is Unix time in seconds (UTC).

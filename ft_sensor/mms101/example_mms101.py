"""
MMS101 Force/Torque Sensor — Example Usage

IMPORTANT: Allow 3-5 minutes warm-up before recording data.
           Expect 1-2 N drift during warm-up.
"""
import time
import os
import argparse
from mms101 import MMS101

# Detect OS from environment variable and select default serial port.
is_windows = os.environ.get("OS", "").lower().startswith("windows")
default_port = "COM3" if is_windows else "/dev/ttyUSB0"

parser = argparse.ArgumentParser(description="MMS101 example runner")
parser.add_argument("--csv", default="ft_data.csv", help="CSV output file name for logging mode")
args = parser.parse_args()

# --- Basic usage (no logging) ---
# sensor = MMS101(
#     port=default_port,
#     medfilt_num=5,         # Median filter over 5 samples (set to 1 to disable)
#     verbose=True,
# )
sensor = MMS101(
    port=default_port,
    medfilt_num=5,         # Median filter over 5 samples (set to 1 to disable)
    verbose=True,
)
sensor.start()

print("Warming up... wait 3-5 minutes before recording.")
time.sleep(5)  # In real use, wait 3-5 minutes

# Zero the sensor with no load applied
sensor.tare()

# Read loop
record_time = 5
print("Reading for 5 seconds...")
t_end = time.time() + record_time
while time.time() < t_end:
    ft = sensor.get_ft_tared()
    print(f"Fx={ft[0]:.3f}N  Fy={ft[1]:.3f}N  Fz={ft[2]:.3f}N  "
          f"Mx={ft[3]:.5f}Nm  My={ft[4]:.5f}Nm  Mz={ft[5]:.5f}Nm")
    time.sleep(0.05)

sensor.stop()

# --- With CSV logging ---
sensor = MMS101(port=default_port, log_csv=args.csv, verbose=True)
sensor.start()
time.sleep(300)   # warm-up
sensor.tare()
time.sleep(60)    # record 60 seconds of data
sensor.stop()
print(f"Data saved to {args.csv}")

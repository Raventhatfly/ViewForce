import threading
import time
import numpy as np
import serial
from ..misc.utils import logging
import os

class BaseFTStream(object):
    def __init__(self, verbose=True):
        self.verbose = verbose

        if not verbose:
            logging("FT sensor-related messages will be turned off", True, "warning")


        self.deviceName = None
        self.baudrate = None
        self.intervalMeasureTime = None
        self.intervalRestartTime = None

        self.ft = None    # [time(us), Fx, Fy, Fz, Mx, My, Mz]
        self.elapsTime = 0.0   

        self.thread = None  # Thread for reading data from serial port
    

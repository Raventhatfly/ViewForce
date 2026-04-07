import threading
import time
import numpy as np
import serial
from ..misc.utils import logging
import os
from collections import deque

from .base_ft_stream import BaseFTStream
DEV_SERIAL_PATH = '/dev/ttyUSB1'        # for Controller
SERIAL_BAUDRATE = 1000000               # for Contoller
INTERVAL_MEASURE_TIME = 1000
INTERVAL_RESTART_TIME = 1000000        # time interval for updating temperature (every 1 min), takes 7500 usec. 

class MMS101FTStream(BaseFTStream):
    def __init__(self, verbose=True, medfilt_num=5):
        if os.name == 'nt':
            if DEV_SERIAL_PATH is None:
                DEV_SERIAL_PATH = 'COM5'        # for Controller
        elif os.name == 'posix':
            if DEV_SERIAL_PATH is None:
                DEV_SERIAL_PATH = '/dev/ttyUSB0'       # for MultiFingerEvaBoardVer.3.0
        else:
            print("Error: Unknown OS {}".format(os.name))

        super(MMS101FTStream, self).__init__(verbose=verbose)
        self.verbose = verbose

        # Device Name
        self.deviceName = DEV_SERIAL_PATH
        # Baudrate
        self.baudrate = SERIAL_BAUDRATE
        # Interval Measure Time [us]
        self.intervalMeasureTime = INTERVAL_MEASURE_TIME
        # Interval Restart Time [us]
        self.intervalRestartTime = INTERVAL_RESTART_TIME
        # Coefficient
        self.coefficient = [
            [0, 0, 0, 0, 0, 0],     #Fx
            [0, 0, 0, 0, 0, 0],     #Fy
            [0, 0, 0, 0, 0, 0],     #Fz
            [0, 0, 0, 0, 0, 0],     #Mx
            [0, 0, 0, 0, 0, 0],     #My
            [0, 0, 0, 0, 0, 0]      #Mz
        ]

        self.data = None            # raw ft 
        self.data_que = deque(maxlen=medfilt_num)

        self.fmax = 40.0
        # serial port open flag
        self.serialOpenFlag = 0
        # board select flag
        self.boardSelectFlag = 0
        # Serial port Open
        self.serialPortOpen()

    
    def __del__(self):
        self.serialPortClose()
    
    def serialPortOpen(self):
        self.serialPort = serial.Serial(self.deviceName, self.baudrate)
        self.serialOpenFlag = 1

    def serialPortClose(self):
        if self.serialOpenFlag == 1:
            
            self.stopMeasure()      # for forced termination
            self.serialPort.close()
            self.serialOpenFlag = 0

    def read_all(self):
        j = self.serialPort.read(2)
        if self.debugMode == 1:
            print(j.hex())

        if j[0] == 0 and j[1] > 0:
            data = self.serialPort.read(j[1])
            j = j + data
            if self.debugMode == 1:
                print(data.hex())
        return j

    def printFirmwareVersion(self):
        self.serialPort.write([0x54, 0x01, 0x15])
        self.serialPort.flush()
        firmversion = self.read_all()
        if firmversion[0] == 0 and firmversion[1] == 4:
            print("Firmware Version", firmversion[2:])
        else:
            print("Error: Firmware Version")
            exit()
    
    def boardSelect(self):
        if self.debugMode == 1:
            print("Board Select")
        self.serialPort.write(bytes([0x54, 0x02, 0x10, 0x00]))
        self.serialPort.flush()
        bds = self.read_all()
        if bds[0] != 0:
            print("Error: Board Select")
            # exit()
            raise Exception("Error: Board Select")
        self.boardSelectFlag = 1

    def powerSwitch(self):
        if self.debugMode == 1:
            print("Power Switch12")
        self.serialPort.write([0x54, 0x03, 0x36, 0x00, 0xFF])
        self.serialPort.flush()
        psw0 = self.read_all()
        if psw0[0] != 0 and psw0[0] != 0x10:
            print("Error: Power Switch12")
            exit()

        if self.debugMode == 1:
            print("Power Switch45")
        self.serialPort.write([0x54, 0x03, 0x36, 0x05, 0xFF])
        self.serialPort.flush()
        psw1 = self.read_all()
        if psw1[0] != 0 and psw1[0] != 0x10:
            print("Error: Power Switch45")
            exit()
    
    def axisSelectAndIdle(self):
        for axis in range(6):
            if self.debugMode == 1:
                print("Axis Select", axis)
            self.serialPort.write([0x54, 0x02, 0x1C, axis])
            self.serialPort.flush()
            axSel = self.read_all()
            if axSel[0] != 0 and axSel[0] != 0x10:
                print("Error: Axis Select", axis)
                exit()

            #Idle
            self.serialPort.write([0x53, 0x02, 0x57, 0x94])
            self.serialPort.flush()
            idl = self.read_all()
            if idl[0] != 0 and idl[0] != 0x10:
                print("Error: IDLE", axis)
                exit()

        time.sleep(0.01)
    
    def bootload(self):
        if self.debugMode == 1:
            print("Bootload")
        self.serialPort.write([0x54, 0x01, 0xB0])
        self.serialPort.flush()
        bl = self.read_all()
        if bl[0] != 0:
            print("Error: Bootload")
            exit()
    
    def readCoefficient(self):
        for axis in range(6):
            for coeff in range(6):
                if self.debugMode == 1:
                    print("Coefficient", axis, coeff)
                self.serialPort.write([0x54, 0x03, 0x27, axis, coeff])
                self.serialPort.flush()
                coeffData = self.read_all()
                if coeffData[0] == 0 and coeffData[1] == 4:
                    self.coefficient[axis][coeff] = (coeffData[2] << 24) + (coeffData[3] << 16) + (coeffData[4] << 8) + coeffData[5]
                    if self.debugMode == 1:
                        print(hex(self.coefficient[axis][coeff]))
                else:
                    print("Error: Coefficient", axis, coeff)
                    exit()
    
    def setIntervalMeasure(self, intervalTime):
        if self.debugMode == 1:
            print("Interval Measure")
        self.intervalMeasureTime = intervalTime
        bData1 = (self.intervalMeasureTime >> 16) & 0xFF
        bData2 = (self.intervalMeasureTime >> 8) & 0xFF
        bData3 = self.intervalMeasureTime & 0xFF
        self.serialPort.write([0x54, 0x04, 0x43, bData1, bData2, bData3])
        self.serialPort.flush()
        simt = self.read_all()
        if simt[0] != 0:
            print("Error: Interval Measure")
            exit()

    def setIntervalRestart(self, intervalTime):
        if self.debugMode == 1:
            print("Interval Restart")
        self.intervalRestartTime = intervalTime
        bData1 = (self.intervalRestartTime >> 16) & 0xFF
        bData2 = (self.intervalRestartTime >> 8) & 0xFF
        bData3 = self.intervalRestartTime & 0xFF
        self.serialPort.write([0x54, 0x04, 0x44, bData1, bData2, bData3])
        self.serialPort.flush()
        sirt = self.read_all()
        if sirt[0] != 0:
            print("Error: Interval Restart")
            exit()

    def startMeasure(self):
        self.serialPort.write([0x54, 0x02, 0x23, 0x00])
        j = self.serialPort.read(2)
        if self.debugMode == 1:
            print("Start")
            print(j)

        if len(j) != 2 or j[0] != 0 or j[1] != 0:
            print("Error: Start")
            exit()

        time.sleep(0.01)

    def stopMeasure(self):
        if self.debugMode == 1:
            print("Stop")
        if self.boardSelectFlag == 1:
            self.serialPort.write([0x54, 0x01, 0x33])
            # j = self.read_all()
            self.boardSelectFlag = 0

    def readData(self):
        prev = bin(0xff)
        while True:
            curr = self.serialPort.read()
            if prev == b'\x00' and curr == b'\x17':
                break
            prev = curr

        data = self.serialPort.read(23)

        return data
    
    def initialize(self):
        self.boardSelect()
        self.powerSwitch()
        self.axisSelectAndIdle()
        self.bootload()
        self.setIntervalMeasure(INTERVAL_MEASURE_TIME)
        self.setIntervalRestart(INTERVAL_RESTART_TIME)
    
    def start(self):
        self.initialize()   

        def _update_ft(sensor):
            sensor.get_ft()
            sensor.medfilt_ft()

        self.thread = threading.Timer(0.001, target=_update_ft, args=[self])
        self.thread.start()
        
    def get_ft(self):
        rdata = self.readData()
        
        # create a new data list
        self.data = [0, 0, 0, 0, 0, 0]
        
        if len(rdata) == 23 and rdata[0] == 0x80 and rdata[1] == 0x00:
            self.elapsTime += ((rdata[20] << 16) + (rdata[21] << 8) + rdata[22]) / 1000000
            for axis in range(6):
                self.data[axis] = (rdata[axis*3+2] << 16) + (rdata[axis*3+3] << 8) + rdata[axis*3+4]
                if self.data[axis] >= 0x00800000:
                    self.data[axis] -= 0x1000000
            
            self.data[0] = self.data[0] / 1000
            self.data[1] = self.data[1] / 1000
            self.data[2] = self.data[2] / 1000
            self.data[3] = self.data[3] / 100000
            self.data[4] = self.data[4] / 100000
            self.data[5] = self.data[5] / 100000

            
            # print(f'{self.elapsTime:.6f},{self.data[0]:.3f},{self.data[1]:.3f},{self.data[2]:.3f},{self.data[3]:.5f},{self.data[4]:.5f},{self.data[5]:.5f}')
        # else:
            # print('Error: Result data length', len(rdata))
        
        return self.data
    
    def medfilt_ft(self, n=5):
        self.data_que.append(self.data)
        if len(self.data_que) < n:
            return self.data
        else:
            return np.median(np.array(self.data_que), axis=0)
        

    def tare(self):
        time.sleep(0.5)
        self.zeros = self.get_ft_med()
        
        print("Tared at: ", self.zeros)

    def get_ft_med(self, n=10):

        fts = []
        for i in range(n):
            time.sleep(0.001)
            fts.append(self.data)
            
        return np.median(np.array(fts), axis=0)
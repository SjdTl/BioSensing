import time
from pySerialTransfer import pySerialTransfer as txfer
import csv
from scipy import signal
from scipy.fft import fftshift
import numpy as np
import pandas as pd
import sys
import json
import time
from telnetlib import Telnet

baud = 57600

def read_socket(tn): # read the data from the socket connection
    line = tn.read_until(str.encode('\r'))
    return json.loads(line)
    
def import_all(subject, test_type, COMport = '/dev/ttyACM0', IP = 'localhost', port = 13854):
    print("Importing data \n")
    try:        
        #tn=Telnet('localhost',13854) # create a telnet object that establishes a socket connection with the TGC program
        #tn.write(str.encode('{"enableRawOutput": true, "format": "Json"}')) # enable the output of the headset data from the TGC program
        
        link = txfer.SerialTransfer(COMport,baud) # create a SerialTransfer object to read data of the COM port
        link.open()
        time.sleep(3) # allow some time for the Arduino to completely reset        
        print(f"link status: {link.status}") 
        #print(f"socket status: {read_socket(tn)['status']}")
        print('Stop data collection by keyboard interrupt (ctrl+c)')
        
        header = ['TimeStamp', 'ECG Data', 'EMG Data', 'EDA Data', 'lowGamma', 'highGamma', 'highAlpha', 'delta', 'highBeta', 'lowAlpha', 'lowBeta', 'theta', 'attention', 'meditation']
        data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        output_file = open('test' + test_type +'data' + subject + '.csv', 'w')
        writer = csv.writer(output_file)
        writer.writerow(header)
    
        
        while True:
            
            ###################################################################
            # Wait for a response and report any errors while receiving packets
            ###################################################################
            while not link.available():
                # A negative value for status indicates an error
                if link.status < 0:
                    if link.status == txfer.Status.CRC_ERROR:
                        print('ERROR: CRC_ERROR')
                    elif link.status == txfer.Status.PAYLOAD_ERROR:
                        print('ERROR: PAYLOAD_ERROR')
                    elif link.status == txfer.Status.STOP_BYTE_ERROR:
                        print('ERROR: STOP_BYTE_ERROR')
                    else:
                        print('ERROR: {}'.format(link.status.name))
            
            
            ###################################################################
            # Parse response list
            ###################################################################
            recSize = 0

            # Import time stamp of the data sample
            data[0] = link.rx_obj(obj_type='L', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['L']

            # Import ECG data from serial connection
            data[1] = link.rx_obj(obj_type='H', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

            # Import EMG data from serial connection
            data[2] = link.rx_obj(obj_type='H', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
            
            # Import GSR data from serial connection
            data[3] = link.rx_obj(obj_type='f', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['f']
            
            # Import label data from serial connection            
            data[4] = link.rx_obj(obj_type='L', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['L']
        
            data[5]
            ###################################################################
            # Write the received data to the csv file
            ###################################################################
            writer.writerow(data)
    
    except KeyboardInterrupt:
        try:
            link.close()
            output_file.close()
            #tn.close()
            print("Serial connection & output_file closed. \n Don't forget to rename ECGdata to prevent overwriting!")

        except:
            pass
    
    except:
        import traceback
        traceback.print_exc()
        
        try:
            link.close()
            output_file.close()
            #tn.close()
        except:
            pass


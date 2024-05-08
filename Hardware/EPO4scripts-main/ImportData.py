import time
from pySerialTransfer import pySerialTransfer as txfer
import csv

baud = 57600
#new code
def import_all(COMport = '/dev/ttyACM0'):
    print("Importing time, ECG, GSR data \n")
    try:
        link = txfer.SerialTransfer(COMport,baud)
        
        link.open()
        time.sleep(3) # allow some time for the Arduino to completely reset
        print(f"link status: {link.status}")
        print('Stop data collection by keyboard interrupt (ctrl+c)')
        
        header = ['TimeStamp', 'ECG Data', 'GSR Data', 'EMG Data', 'Label']
        data = [0,0,0,0,0]

        output_file = open('Alldata.csv', 'w')
        writer = csv.writer(output_file)
        writer.writerow(header)
    
        ###Defining the arrays to store data
        ECGList = []
        GSRList = []
        EMGList = []
        LabelList = []
        ###
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

            # Import GSR data from serial connection
            data[2] = link.rx_obj(obj_type='H', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

            # Import GSR data from serial connection
            data[3] = link.rx_obj(obj_type='H', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['H']

            # Import GSR data from serial connection
            data[4] = link.rx_obj(obj_type='H', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
        
            
            ###################################################################
            # Write the received data to the csv file
            ###################################################################
            writer.writerow(data)

            ###Add data to lists
            ECGList.append(data[1])
            GSRList.append(data[2])
            EMGList.append(data[3])
            LabelList.append(data[4])
            ###
    
    except KeyboardInterrupt:
        try:
            link.close()
            output_file.close()
            print("Serial connection & output_file closed. \n Don't forget to rename ECGdata to prevent overwriting!")

        except:
            pass
    
    except:
        import traceback
        traceback.print_exc()
        
        try:
            link.close()
            output_file.close()
        except:
            pass
    #Put data in dictionary
    subject_data = {
        "ECG" : ECGList,
        "EMG" : EMGList,
        "EDA" : GSRList,
        "Labels" : LabelList
    }
    return subject_data
#old code


'''
def import_all(COMport = '/dev/ttyACM0'):
    print("Importing time, ECG, GSR data \n")
    try:
        link = txfer.SerialTransfer(COMport,baud)
        
        link.open()
        time.sleep(3) # allow some time for the Arduino to completely reset
        print(f"link status: {link.status}")
        print('Stop data collection by keyboard interrupt (ctrl+c)')
        
        header = ['TimeStamp', 'ECG Data', 'GSR Data']
        data = [0,0,0]

        output_file = open('Alldata.csv', 'w')
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

            # Import GSR data from serial connection
            data[2] = link.rx_obj(obj_type='H', start_pos=recSize)
            recSize += txfer.STRUCT_FORMAT_LENGTHS['H']
        
            
            ###################################################################
            # Write the received data to the csv file
            ###################################################################
            writer.writerow(data)
    
    except KeyboardInterrupt:
        try:
            link.close()
            output_file.close()
            print("Serial connection & output_file closed. \n Don't forget to rename ECGdata to prevent overwriting!")

        except:
            pass
    
    except:
        import traceback
        traceback.print_exc()
        
        try:
            link.close()
            output_file.close()
        except:
            pass
'''
from os.path import isfile, join, isdir
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

import os
import pickle
import numpy as np
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle #for performance

# Class to read out the data of one subject
class read_data_of_one_subject:
    """
    Description
    -----------
    Read data of one of the subjects in the WESAD dataset.

    Parameters
    ----------
    path : string
        path to the WESAD dataset
    subject : int
        number assigned to the subject which has to be read

    Returns
    -------
    labels : dictionary containing np.arrays 
        labels (0-7) indicating 0: invalid, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation. 5-7 are not of interest.
    wrist_data : dictionary containing np.arrays
        data captured by the wrist device ('ACC', 'BVP', 'EDA', 'TEMP'). Access desired data using wrist_data["ACG"]
    chest_data : dictionary containg np.arrays
        data captured by the chest device ('ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp')
    
    Notes
    -----
    Code provided by EPO4 manual
    """
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        #os.chdir(path)
        #os.chdir(subject)
        with open(os.path.join(path,"S"+str(2),"S"+str(2)+".pkl"), "rb") as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        """"""
        #label = self.data[self.keys[0]]
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        #wrist_ACC = wrist_data[self.wrist_sensor_keys[0]]
        #wrist_ECG = wrist_data[self.wrist_sensor_keys[1]]
        return wrist_data

    def get_chest_data(self):
        """"""
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data

# Path to the WESAD dataset (make sure you upload this file in when executing the code)
WESAD_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "WESAD")

def create_pickle(WESAD_path):
    """
    Description
    -----------
    Create a pickle file containing a dictionary with all relevant signals. These are EMG, ACG and EDA of the chest.
    It also creates a second file containing data of all three signals with subject 2, label 1 and 60s of measurement.

    Parameters
    ----------
    WESAD_path : string
        path to WESAD dataset
    
    Returns
    -------
    Raw_data/raw_data.pkl : pickled dictionary
        dictionary containing the wesad features with the form: \n
        data = {
            "2" :   "EMG" : 1D np array with EMG chest data
                    "ECG" : 1D np array with ECG chest data
                    "EDA" : 1D np array with EDA chest data
                    "Labels" : 1D np array labels (0 and 5-7 are already removed)
            "3" : 
            ...
        }\n
        where the first key is the subject label

    Raw_data/raw_small_test_data.pkl : pickled dictionary
        data = {
            "EMG" : 1D np array with 60s of EMG chest data
            "ECG" : 1D np array with 60s of ECG chest data
            "EDA" : 1D np array with 60s of EDA chest data
        }
    """

    # Object instantiation
    data = {}
    # All used subjects (could've been detected automatically)
    subjects = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

    # Loop through subjects and extract used data:
    # Labels: 1-4
    # Data: EMG, ECG, EDA of the chest
    # Other data (for now) not used
    for subject in subjects:
        cdata = read_data_of_one_subject(WESAD_path, subject)
        labels = cdata.get_labels()

        # Only use labels 1-4
        used_labels = np.asarray([idx for idx,val in enumerate(labels) if (val == 1 or val==2 or val==3 or val==4)])
        EMG = cdata.get_chest_data()['EMG'][used_labels,0]
        ECG = cdata.get_chest_data()['ECG'][used_labels,0]
        EDA = cdata.get_chest_data()['EDA'][used_labels,0]
        data[subject] = {"EMG" : EMG, "ECG" : ECG, "EDA" : EDA, "labels" : labels[used_labels]}

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Raw_data","raw_data.pkl"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Test data
    fs = 700
    t = 60
    start = 100000
    label = np.asarray([idx for idx,val in enumerate(labels) if (val == 1)])
    cdata = read_data_of_one_subject(WESAD_path, 2)
    test_data = {"EMG" : cdata.get_chest_data()['EMG'][label,0][start:start + fs * t],
               "ECG" : cdata.get_chest_data()['ECG'][label,0][start:start + fs * t],
               "EDA" : cdata.get_chest_data()['EDA'][label,0][start:start + fs * t],}
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Raw_data","raw_small_test_data.pkl"), 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

create_pickle(WESAD_path)
from os.path import isfile, join, isdir
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import Interface
import pandas as pd


def create_pickle_arduino():
    """
    Description
    -----------
    Create a pickle file containing a dictionary with all relevant signals. These are EMG, ACG and EDA of the chest.

    Parameters
    ----------
    
    Returns
    -------
    Raw_data/raw_data.pkl : pickled dictionary
        dictionary containing the subject features with the form: \n
        data = {
            "Name1" :   "EMG" : 1D np array with EMG chest data
                    "ECG" : 1D np array with ECG chest data
                    "EDA" : 1D np array with EDA chest data
                    "Labels" : 1D np array labels (0 and 5-7 are already removed)
            "Name2" : 
            ...
        }\n
        where the first key is the subject label
    """
    
    # Object instantiation
    data = {}
    # Subject instantiation, will be done automatically 
    subjects = []
    num_subjects = 10
    for i in range(num_subjects):
        subjects += ["S"+str(i)]
    # Loop through subjects and extract used data:
    # Labels: 1-4
    # Data: EMG, ECG, EDA of the chest
    # Other data (for now) not used
    
    for subject in subjects:
        subject_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data" + subject + ".csv")
        df = pd.read_csv(subject_path)
        data[subject] = {"EMG" : df["ECG Data"], "ECG" : df["ECG Data"], "EDA" : df["GSR Data"], "label" : df["label"]}
        print(data[subjects[0]])
    # Turn dictionary into pickle    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Pickled_data", "data" + subject + ".pkl"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data
data = create_pickle_arduino()


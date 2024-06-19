#%%
from os.path import isfile, join, isdir
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import Interface
import pandas as pd

def picklabel(test):
    if test == 'baseline':
        return 1
    elif test == 'meditation':
        return 2
    elif test == 'amusement':
        return 3
    elif test == 'anticipation':
        return 4
    elif test == 'presentation':
        return 4
    elif test == 'arithmetic':
        return 4

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
    num_subjects = 1
    for i in range(num_subjects,num_subjects+1):
        subjects += ["subject"+str(i)]
        
    test_order = {'subject1' : ['baseline','meditation','amusement','anticipation','presentation','arithmetic'],
                  'subject2' : ['baseline','meditation','amusement','anticipation','presentation','arithmetic'],
                  'subject3' : ['baseline','meditation','amusement','anticipation','presentation','arithmetic']}
    # Loop through subjects and extract used data:
    # Labels: 1-4
    # Data: EMG, ECG, EDA of the chest
    # Other data (for now) not used

    
    for subject in subjects:
        for test in test_order[subject]:
            subject_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), subject, test + '.csv')
            df = pd.read_csv(subject_path)
            data[subject] = {"EMG" : df["EMG Data"], "ECG" : df["ECG Data"], "EDA" : df["EDA Data"], "label" : picklabel(test)}
            print(data[subjects[0]])
    # Turn dictionary into pickle    
    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Pickled_data", "data" + subject + ".pkl"), 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data
data = create_pickle_arduino()


# %%

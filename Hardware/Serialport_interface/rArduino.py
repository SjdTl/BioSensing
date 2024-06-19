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
        return 4
    elif test == 'amusement':
        return 3
    elif test == 'anticipation':
        return 2
    elif test == 'presentation':
        return 2
    elif test == 'arithmetic':
        return 2

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
    x = pd.Series()
    data = {}
    # Subject instantiation, will be done automatically 
    subjects = []
    num_subjects = 3
    for i in range(1,num_subjects+1):
        subjects += ["subject"+str(i)]
        
    test_order = {'subject1' : ['baseline','meditation','amusement','anticipation','presentation','arithmetic'],
                  'subject2' : ['baseline','meditation','anticipation','presentation','arithmetic','amusement'],
                  'subject3' : ['baseline','anticipation','presentation','arithmetic','amusement','meditation']}
    
    test_rating = {'subject1' : {'baseline' : 10,'meditation' : 10,'amusement' : 10,'anticipation' : 4,'presentation' : 8,'arithmetic' : 8},
                  'subject2' : {'baseline' : 10,'meditation' : 10,'amusement' : 10,'anticipation' : 0,'presentation' : 10,'arithmetic' : 0},
                  'subject3' : {'baseline' : 10,'meditation' : 10,'amusement' : 10,'anticipation' : 4,'presentation' : 5,'arithmetic' : 8}}
    
    # Loop through subjects and extract used data:
    # Labels: 1-4
    # Data: EMG, ECG, EDA of the chest
    # Other data (for now) not used

    
    for subject in subjects:
        data[subject] = {"EMG" : x, "ECG" : x, "EDA" : x, "label" : x}
        for test in test_order[subject]:
            if test_rating[subject][test] > 7:
                subject_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'measurement_data', subject, test + '.csv')
                df = pd.read_csv(subject_path)
                data[subject] = {"EMG" : pd.concat([data[subject]['EMG'],df["EMG Data"]], ignore_index=True), 
                                 "ECG" : pd.concat([data[subject]['ECG'],df["ECG Data"]], ignore_index=True), 
                                 "EDA" : pd.concat([data[subject]['EDA'],df["EDA Data"]], ignore_index=True), 
                                 "label" : pd.concat([data[subject]['label'],pd.Series(picklabel(test))], ignore_index=True)}
                print(data[subjects[0]])
    # Turn dictionary into pickle    
    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Pickled_data", "data" + subject + ".pkl"), 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data
data = create_pickle_arduino()


# %%

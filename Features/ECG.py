import numpy as np
import pandas as pd
import os

import all_signals

def ECG(ecg, fs):
    """
    Description
    -----------
    Obtains the features for an ecg window of x seconds.

    Parameters
    ----------
    unprocessed_ecg : np.array
        ect signal as provided directly by the sensors
    fs : int or float
        sampling frequency of the sensors

    Returns
    -------
    features : pd.DataFrame
        Dataframe (1 row) containing the features:
            - ?
            and the general features:
            - Mean (no meaning in the case of emg)
            - Median
            - Std
            - ...
     
    Raises
    ------
    ValueError
        Raises error if there is a NaN value in the features
    
    Notes
    -----
    
    Examples
    --------
    >>>
    """

    # Just for now without processing
    features = pd.DataFrame([ecg.mean()], columns = ["Mean_ecg"])


    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of ECG contains a NaN value")
    return features

def test():
    """
    Description
    -----------
    Function to test the signal, without having to call the entire database. Please use this function when looking for data to plot for the report.
    
    Notes
    -----
    Returns and takes nothing
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
    ecg = all_signals.load_test_data("ECG", filename)

    all_signals.quick_plot(ecg)
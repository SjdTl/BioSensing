import numpy as np
import pandas as pd
import os

import Features.feat_gen as feat_gen

def EDA(eda, fs):
    """
    Description
    -----------
    Obtains the features for an eda window of x seconds.

    Parameters
    ----------
    unprocessed_eda : np.array
        eda signal as provided directly by the sensors
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
    features = pd.DataFrame([eda.mean()], columns = ["mean_EDA"])


    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EDA contains a NaN value")
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
    eda = feat_gen.load_test_data("EDA", filename)

    feat_gen.quick_plot(eda)
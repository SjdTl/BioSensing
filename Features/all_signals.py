# Contains functions that are used by ECG, EMG and EDA
# Basic features, loading test data and quick plotting

import pandas as pd
import numpy as np
from scipy.stats import mode
import pickle
import matplotlib.pyplot as plt

def basic_features(signal, name):
    """
    Description
    -----------
    Used in the ECG.ECG, EDA.EDA and EMG.EMG functions. Returns all standard (rms, std, ...) features that are the same for all arrays in a pandas DataFrame.

    Parameters
    ----------
    signal : array
        signal (EDA, ECG or EMG after preprocessing) to do the standard functions on
    name: string
        name added to the features to prevent having features with the same name when merging arrays of different signals

    Returns
    -------
    out : pd.DataFrame
         Pandas array containing the features:
        | index | Mean_name | Median_name | STD_name | mode |
        |   -   |     -     |       -     |     -    |   -  |
        |   0   |     x     |       y     |     z    |      |
        with only one row.
    Notes
    -----
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scipy.stats import mode
    >>> a = [2, 4, 5, 6, 7, 3, 1, 4, 2]
    >>> df = basic_features(a, "a")
    >>> print(df)
         Mean_a  Median_a     STD_a  Mode_a
    0  3.777778       4.0  1.872478       2
    """
    most_common, _ = mode(signal)

    features = {"Mean_" + str(name) : [np.mean(signal)], 
                "Median_" + str(name) : [np.median(signal)],
                "STD_" + str(name) : [np.std(signal)],
                "Mode_" + str(name) : [most_common]}
    return pd.DataFrame(features)

def load_test_data(signal, filename):
    """
    Description
    -----------
    Load the smaller dataset (subject S2, label=1, 60s frame) for testing the feature extractions.
    
    Parameters
    ----------
    signal : string
         Return ECG, EDA or EMG based on this string
    filename : string
        Path to the smaller dataset. If this is provided by the read_WESAD.py script, the file is in the Features/Raw_data/raw_small_test_data.pkl
         
    Returns
    -------
    out : np.array
         np.array containing a signal ECG, EDA or EMG of 60s
    """
    
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out[signal]

# To see the signal (do not use for the report figures)
def quick_plot(signal, t=60, fs=700):
    t = np.arange(0, signal.size * (1/fs), 1/fs)

    plt.figure()
    plt.plot(t, signal)
    plt.xlabel("Time ($s$)")
    plt.ylabel("Signal")
    plt.show()

# Use to test basic_features
def example():
    a = [2, 4, 5, 6, 7, 3, 1, 4, 2]
    df = basic_features(a, "a")
    print(df)
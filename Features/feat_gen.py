# Contains functions that are used by ECG, EMG and EDA
# Basic features, loading test data, rms and quick plotting

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

def rms(x):
  """
  Description
  -----------
  Calculate the root mean squared value of a signal

  Parameters
  ----------
  x : array
     array to calculate the RMS value of
  
  Returns
  -------
  out : float
     RMS value
  
  Examples
  --------
  >>> print(rms([2,3,4,5,6]))
  4.242640687119285
  """
  return np.sqrt(np.mean(np.square(x)))

def quick_plot(*signals, fs=700):
    """
    Description
    -----------
    Plot several sampled signals without any fancy formatting
    
    Parameters
    ----------
    signals : type
        any amount of signals to plot
    fs : int or float
        sampling frequencies of ALL signals
    
    Returns
    -------
        plt.show()

    Notes
    -----
    Do not use this to plot figures in the report

    Examples
    --------
    >>> quickplot(np.arange(0,100), np.arange(100,0), fs=700)
    """

    fig, ax = plt.subplots()
    for signal in signals:
        t = np.arange(0, signal.size * (1/fs), 1/fs)
        ax.plot(t, signal)
    ax.set_xlabel("Time ($s$)")
    ax.set_ylabel("Signal")
    plt.show()

# Use to test basic_features
def example():
    a = [2, 4, 5, 6, 7, 3, 1, 4, 2]
    df = basic_features(a, "a")
    print(df)
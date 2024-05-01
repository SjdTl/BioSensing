import numpy as np
import pandas as pd
import os
from scipy.signal import butter, iirnotch, lfilter
from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import neurokit2 as nk

import feat_gen as feat_gen

def ECG(unprocessed_ecg, fs= 700):
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

    ecg = preProcessing(unprocessed_ecg, fs)

    df_specific = ECG_specific_features(ecg, fs)
    # General features contain mean emg, but this has no meaning in the case of emg
    df_general = feat_gen.basic_features(ecg, "emg")

    features = pd.concat([df_specific, df_general], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EMG contains a NaN value")
    return features

def preProcessing(unprocessed_ecg, fs):
    """
    Description
    -----------
    Preprocessing the EDA signal using bunch of filters

    Parameters
    ----------
    unprocessed_ecg : np.array
        the ECG data as received directly by the sensors
    fs : int or float
        sampling frequency of sensor
    
    Returns
    -------
    ecg : np.array
        the ECG data processed
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    
    Examples
    --------
    >>>
    """
    nyq = 0.5*fs
    order=5

    # highpass filter
    high=0.5
    high= high/nyq
    b, a = butter(5, high, btype = 'highpass')
    ecg_h = lfilter(b,a,unprocessed_ecg)

    # lowpass filter
    low=70
    low= low/nyq
    b, a = butter(5, low)
    ecg_hl = lfilter(b,a,ecg_h)

    # notch filter
    notch=50
    notch = notch/nyq
    b, a = iirnotch(notch, 30)
    ecg = lfilter(b,a,ecg_hl)

    return ecg

def ECG_specific_features(ecg, fs):
    """
    Description
    -----------
    Calculate features specific to ecg signal
        - pNN50: The proportion of RR intervals greater than 50ms, out of the total number of
          RR intervals.
        - pNN20: The proportion of RR intervals greater than 20ms, out of the total number of
          RR intervals.
        - RMSSD: The square root of the mean of the squared successive differences between
          adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
          therefore it is redundant to report correlations with both (Ciccone, 2017)
    Parameters
    ----------
    ecg : np.array
        processed ecg data as provided by the sensors

    Returns
    -------
    out : pd.DataFrame
        dataframe containing the features mentioned in the docstring of ECG()
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    Code copied from https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_time.html#hrv_time
    """ 

    r_peaks_pan = rpeak_detector(ecg, fs = 700)

    out_dict = {}
    
    rri, rri_time, rri_missing = nk.hrv.hrv_utils._hrv_format_input(r_peaks_pan, sampling_rate=fs)
    diff_rri = np.diff(rri)
    out_dict["HRV_RMSSD"] = np.sqrt(np.nanmean(diff_rri**2))
    nn50 = np.sum(np.abs(diff_rri) > 50)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out_dict["pNN50"] = nn50 / (len(diff_rri) + 1) * 100
    out_dict["pNN20"] = nn20 / (len(diff_rri) + 1) * 100
    
    # Drop values that don't return anything for short signals 
    return pd.DataFrame(out_dict)

def rpeak_detector(ecg, fs, plot= False):

    """
    Description
    -----------
    Detect the R-peaks in the ECG signal

    Parameters
    ----------
    ecg : np.array
        processed ecg signals
    fs : int or float
        sampling rate of sensor
    plot : boolean
        If the results have to be plotted
        
    Returns
    -------
    r_peaks_pan : np.array
        array of indices of the peaks
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    Check different algorithms

    Examples
    --------
    >>>
    """

    detectors = Detectors(fs)

    r_peaks_pan = detectors.pan_tompkins_detector(ecg)
    r_peaks_pan = np.asarray(r_peaks_pan)

    if plot == True:
        plot_rPeaks(ecg, r_peaks_pan)

    return r_peaks_pan 

def plot_rPeaks(ecg, r_peaks_pan):
    """
    Description
    -----------
    Plot the ecg signal with the peaks

    Parameters
    ----------
    ecg : np.array
        processed ecg signal
    r_peaks_pan: np.array
        array of indices of the peaks
        
    Returns
    -------
    plt.show()
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    Update when using this for plots in manual

    Examples
    --------
    >>>
    """
    plt.figure(figsize=(12,4))
    plt.plot(ecg)
    plt.plot(r_peaks_pan,ecg[r_peaks_pan], 'ro')


def test(filepath):
    """
    Description
    -----------
    Function to test the signal, without having to call the entire database. Please use this function when looking for data to plot for the report.
    
    Parameters
    ----------
    Filepath: string
        Filepath to the test signal. This should be a pickled dictionary with the following format:
            dict = {EDA: [..]
                    EMG: [..]
                    ECG: [..]}
        Each signal is of one person, one label and includes only a small timeframe
    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the features 
        
    """
    ecg = feat_gen.load_test_data("ECG", filepath)

    # feat_gen.quick_plot(ecg, preProcessing(ecg, 700))
    # rpeak_detector(preProcessing(ecg), 700)

    df = ECG(ecg, 700)
    return df

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
print(test(filepath))
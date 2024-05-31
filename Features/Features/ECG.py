import numpy as np
import pandas as pd
import os
from scipy.signal import butter, iirnotch, lfilter, sosfilt
from scipy.stats import iqr
from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.stats import mode

from . import feat_gen

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
            - pNN50
            - pNN20
            - RMSSD
            - ...
            and the general features:
            - Mean (no meaning in the case of emg)
            - Median
            - Std
            - ...
     
    Raises
    ------
    ValueError
        Raises error if there is a NaN value in the features
    """

    ecg = preProcessing(unprocessed_ecg, fs)

    df_specific = ECG_specific_features(ecg, fs)
    # General features contain mean emg, but this has no meaning in the case of emg
    df_general = feat_gen.basic_features(ecg, "ECG")

    features = pd.concat([df_specific, df_general], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EMG contains a NaN value")
    return features

def preProcessing(unprocessed_ecg, fs):
    """
    Description
    -----------
    Preprocessing the EDA signal using filters:
        - Lowpass at 90 Hz
        - Highpass at 0.5 Hz
        - Notch at 50
    So keep signal 0.5-90Hz minus powerline interference

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
    """

    low, _, _ = lowpassecg(unprocessed_ecg, fs)
    high, _ = highpassecg(low, fs)
    ecg, _, _ = notchecg(high, fs)
    return ecg

def lowpassecg(ecg, fs):
    N = 5
    cut=90
    if fs < 180:
        cut = fs/2 * 0.9

    b, a = butter(N, cut, fs=fs)
    filtered = lfilter(b,a,ecg)
    return filtered, b, a
def highpassecg(ecg, fs, N = 8):
    cut=0.5
    sos = butter(N, cut, btype = 'highpass', fs=fs, output = 'sos')
    filtered = sosfilt(sos, ecg)
    return filtered, sos
def notchecg(ecg, fs):
    cut=50
    b, a = iirnotch(cut, 30, fs)
    filtered = lfilter(b,a,ecg)
    return filtered, b, a



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
    This code is a simplification of the neurokit2 library
    It is used instead of the neurokit2 library to save computing time
    Since the neurokit2 libary requires you to calculate all features
    But not all features return values at time windows
    https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/hrv/hrv_time.html#hrv_time
    """ 

    r_peaks_pan = rpeak_detector(ecg, fs = 700)
    out_dict = {}
    
    # Array of RR peak times in milliseconds
    rri = np.diff(r_peaks_pan) / fs * 1000
    # Differences between between time length of two RR peaks following eachother
    diff_rri = np.diff(rri)

    # RR peak based
    out_dict["Heart_rate1"] = r_peaks_pan.size / (ecg.size / fs) * 60
    out_dict["Heart_rate2"] = 60 / np.mean(rri) * 1000
    out_dict["MeanNN"] = np.mean(rri)
    out_dict["SDNN"] = np.std(rri)


    # Difference based
    out_dict["RMSSD"] = np.sqrt(np.nanmean(diff_rri**2))
    out_dict["SDSD"] = np.std(diff_rri)
    # Normalized
    out_dict["CVNN"] = out_dict["RMSSD"] / out_dict["MeanNN"]
    out_dict["CVSD"] = out_dict["SDSD"] / out_dict["MeanNN"]

    # Robust
    out_dict["MedianNN"] = np.nanmedian(rri)
    out_dict["IQRNN"] = iqr(rri)
    out_dict["SDRMSSD"] = out_dict["SDNN"] / out_dict["RMSSD"]  # Sollers (2007)
    out_dict["Prc20NN"] = np.nanpercentile(rri, q=20)
    out_dict["Prc80NN"] = np.nanpercentile(rri, q=80)

    nn50 = np.sum(np.abs(diff_rri) > 50)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out_dict["pNN50"] = nn50 / (len(diff_rri) + 1) * 100
    out_dict["pNN20"] = nn20 / (len(diff_rri) + 1) * 100
    out_dict["MinNN"] = np.nanmin(rri)
    out_dict["MaxNN"] = np.nanmax(rri)

    # Other statistical analysis
    most_common_NN, _ = mode(rri)
    most_common_dNN, _ = mode(diff_rri)
    out_dict["dNNmode"] = most_common_dNN
    out_dict["NNmode"] = most_common_NN
    out_dict["MaxdNN"] = np.nanmax(diff_rri)
    out_dict["MindNN"] = np.nanmin(diff_rri)
    out_dict["Prc20dNN"] = np.nanpercentile(diff_rri, q=20)
    out_dict["Prc80dNN"] = np.nanpercentile(diff_rri, q=80)

    # Drop values that don't return anything for short signals 
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("HRV_")

def rpeak_detector(ecg, fs):

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
        
    Returns
    -------
    r_peaks_pan : np.array
        array of indices of the peaks
    
    Notes
    -----
    Check different algorithms
    """

    detectors = Detectors(fs)

    r_peaks_pan = detectors.pan_tompkins_detector(ecg)
    r_peaks_pan = np.asarray(r_peaks_pan)

    return r_peaks_pan 
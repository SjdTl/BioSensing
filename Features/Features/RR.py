import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt
from scipy.ndimage import uniform_filter1d
import scipy
import matplotlib.pyplot as plt
import tqdm
import neurokit2 as nk
from sklearn.preprocessing import minmax_scale as normalize

from . import feat_gen
from .ECG import preProcessing

def RR(unprocessed_ecg, fs=700):
    """
    Description
    -----------
    Obtains the features for an respitory rate window of x seconds, derived from the ecg data.

    Parameters
    ----------
    unprocessed_ecg : np.array
        ecg signal as provided directly by the sensors
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
    rr = normalize(ECG_to_RR(ecg, fs=fs))

    df_general = feat_gen.basic_features(rr, "RR")
    df_specific = rr_peak_features(rr, fs)

    features = pd.concat([df_specific, df_general], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EDA contains a NaN value")
    return features

def ECG_to_RR(ecg, fs=100, method = "vangent2019"):
    if method == "vangent2019" or method == "soni2019" or method == "charlton2016" or method == "sarkar2015":
        # Extract peaks
        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        # Compute rate
        ecg_rate = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg))

        edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs, method = method)

        return edr
    
def rr_peak_features(rr, fs=700,method="Neurokit"):
    """
    Description
    -----------
    Calculate peak features of RR signal
    
    Parameters
    ----------
    rr : np.array
        processed rr signal
    fs : float or int
        sampling frequency of rr signal
    
    Returns
    -------
    out : pd.DataFrame
        dataframe containing the features mentioned in the docstring of RR()
    """

    peak_index, through_index = peak_detection_RR(rr, fs=fs, method=method)

    # Find features
    out_dict = {}
    T = np.size(rr) / fs

    out_dict["Respitory rate"] = np.size(peak_index) / T * 60

    # Turn dictionary into pd.DataFrame and return
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("RR_")

def peak_detection_RR(rr, fs=700, method = "Neurokit"):
    
    if method == "Neurokit":
        info, data = nk.rsp_peaks(rr, sampling_rate = fs)
        peak_index = (data["RSP_Peaks"])
        through_index = (data["RSP_Troughs"])

    return peak_index, through_index

# dir_path = os.path.dirname(os.path.realpath(__file__))
# filepath = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
# print(test(filepath))
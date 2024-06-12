import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt, decimate
from scipy.ndimage import uniform_filter1d
import scipy
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.stats import entropy


from . import feat_gen

def EEG(unprocessed_eeg, fs):
    """
    Description
    -----------
    Obtains the features for an eeg window of x seconds.

    Parameters
    ----------
    unprocessed_eeg : np.array
        eeg signal as provided directly by the sensors
    fs : int or float
        sampling frequency of the sensors

    Returns
    -------
    features : pd.DataFrame
        Dataframe (1 row) containing the features:
            - ...
        
     
    Raises
    ------
    ValueError
        Raises error if there is a NaN value in the features
    
    Notes
    -----
    """
    
    eeg = preProcessing(unprocessed_eeg, fs)
    
    df_eeg = eeg_features(eeg)

    features = pd.concat([df_eeg], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EEG contains a NaN value")
    return features

def preProcessing(unprocessed_eeg, fs=700):
    """
    Description
    -----------
    Preprocessing the EEG signal using ...

    Parameters
    ----------
    unprocessed_eeg : np.array
        the EEG data as received directly by the sensors
    
    Returns
    -------
    eeg : np.array
        the eeg data processed
    """

    eeg = unprocessed_eeg

    #normalising
    eeg_range = np.max[eeg]-np.min[eeg]
    eeg = (eeg - np.min[eeg])/eeg_range


    return eeg

def eeg_features(eeg, fs):
    """
    Description
    -----------
    Calculate general features of the eeg signal
    
    Parameters
    ----------
    eeg : np.array
        processed eeg signal
    fs : float or int
        sampling frequency of eeg signal
    
    Returns
    -------
    out : pd.DataFrame
        dataframe containing the features mentioned in the docstring of EEG()
    
    Notes
    -----
    
    """ 

    out_dict = {}

    ###Frequency bands

    
    out_dict["Mean"] = np.median(eeg)
    out_dict["STD"] = np.std(eeg)

    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EEG_")

def AverageBandPower(lower,upper,signal, fs):
    '''
    Bandpass filters a signal in frequency range lower->upper and returns the power
    '''
    butter_bandpass_filter(signal,lower,upper,fs)
    



def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Description
    -----------
    Provided polynomials of a butterworth IIR filter

    Parameters
    ----------
    lowcut, highcut: float
        bandpass frequencies
    fs : float
        sampling frequency
    order : int
        order of butterworth filter
    
    Returns
    -------
    b,a : type
         Numerator (b) and denominator (a) polynomials of the IIR filter
    
    
    Notes
    -----
    Not yet clear if butterworth is the best filter for this
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a butterworth bandpass filter. See butter_bandpass for description
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b,a, data)
    return y

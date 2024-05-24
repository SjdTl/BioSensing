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
            - Onset
            - Recovery
            - RR
            - RM
            - RT
            - ...
            See manual for more information
        and the general features:
            - Mean (no meaning in the case of emg)T
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
    downsampling_factor = 10
    eda, phasic, tonic = preProcessing(eda, fs, Q= downsampling_factor)
    
    df_eda = tot_eda_features(eda, fs/downsampling_factor)
    df_phasic = phasic_features(phasic, fs/downsampling_factor)

    features = pd.concat([df_eda, df_phasic], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EDA contains a NaN value")
    return features

def preProcessing(unprocessed_eda, fs=700, Q=10):
    """
    Description
    -----------
    Preprocessing the EDA signal using a lowpass filter and split up using the split_phasic_tonic() function

    Parameters
    ----------
    unprocessed_eda : np.array
        the EDA data as received directly by the sensors
    
    Returns
    -------
    eda : np.array
        the EDA data processed
    phasic : np.array
        eda after highpass filtering
    tonic : np.array
        eda after lowpass filtering
    """

    # Lowpass
    order = 4
    cutoff = 5
    # Downsampling factor
    fs = fs/Q
    lowpass_eda = butter_EDA(unprocessed_eda, N=order, cutoff=cutoff, fs=fs, Q=Q)

    # Smoothing
    phasic, tonic = split_phasic_tonic(lowpass_eda, fs=fs, method = "cvxEDA")

    return phasic + tonic, phasic, tonic


def butter_EDA(eda, N=4, cutoff=5, fs=700, Q=10):
    """Butterworth filter used by EDA preprocessing and downsample because most information is unnecessary"""
    b,a = butter(N = N, Wn = cutoff, fs= fs)
    eda = filtfilt(b, a, eda)
    return decimate(eda, Q)

def split_phasic_tonic(eda, fs = 700, method = "cvxEDA"):
    """
    Description
    -----------
    The electrodermal activity is made of two components:
    phasic and tonic. A very simple approach for decomposing the signal into this 
    two components is by using high and low pass filtering respectively. 
    Use 5th order Butterworth filter with output="sos". and cutoff frequency of 0.05 Hz.

    Another approach is the cvxEDA algorithm:
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).

    Parameters
    ----------
    eda : np.array
        Processed eda data
    fs : int or float
        Sampling rate of EDA sensor
    order : int 
        Order of butterworth filter
    
    Returns
    -------
    phasic : np.array
        eda after highpass filtering
    tonic : np.array
        eda after lowpass filtering
    """
    
    df = nk.eda_phasic(eda, sampling_rate = fs, method = method)
    return (df["EDA_Phasic"]).to_numpy(), (df["EDA_Tonic"]).to_numpy()

def tot_eda_features(eda, fs):
    """
    Description
    -----------
    Calculate general features of the eda signal
    
    Parameters
    ----------
    eda : np.array
        processed eda signal
        I recommend this to be: eda = phasic + tonic, such that the errors are minimized using cvxEDA
    fs : float or int
        (DOWNSAMPLED) sampling frequency of eda signal
    
    Returns
    -------
    out : pd.DataFrame
        dataframe containing the features mentioned in the docstring of EDA()
    
    Notes
    -----
    Most features are from:
    Automatic motion artifact detection in electrodermal activity data using machine learning
    Md-Billal Hossain, Hugo F. Posada-Quintero, Youngsun Kong, Riley McNaboe, Ki H. Chon 
    """ 

    out_dict = {}
    
    out_dict["Median"] = np.median(eda)
    out_dict["Range"] = np.max(eda) - np.min(eda)
    out_dict["Entropy"] = entropy(eda)

    # Derivatives
    der1 = np.gradient(eda, np.linspace(0, eda.size / fs, eda.size))
    der2 = np.gradient(der1, np.linspace(0, eda.size / fs, eda.size))
    der = [eda, der1, der2]
    for i in [0,1,2]:
        out_dict["Mean_der" + str(i)] = np.mean(der[i])
        out_dict["Std_der" + str(i)] = np.std(der[i])
        out_dict["Min_der" + str(i)] = np.min(der[i])
        out_dict["Max_der" + str(i)] = np.max(der[i])
        

    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EDA_")

def phasic_features(phasic, fs):
    """
    Description
    -----------
    Calculate features specific to phasic part of the eda signal
    
    Parameters
    ----------
    phasic : np.array
        processed phasic part of the eda signal
    fs : float or int
        (DOWNSAMPLED) sampling frequency of eda signal
    
    Returns
    -------
    out : pd.DataFrame
        dataframe containing the features mentioned in the docstring of EDA()
    """
    out_dict = {}
    # General phasic features
    out_dict["Mean"] = np.mean(phasic)


    # Peak data
    peaks, onset, offset50, offset63, magnitude = peak_detection(phasic, method="Neurokit", fs=fs)

    # Find features
    out_dict["RT"] = np.mean(-(onset-offset63)/fs)
    out_dict["Recovery63"] = np.mean(-(peaks-offset63)/fs)
    out_dict["Recovery50"] = np.mean(-(peaks-offset50)/fs)
    out_dict["Rise"] = np.mean(-(onset-peaks)/fs)
    out_dict["RM"] = np.mean(magnitude)
    out_dict["RR"] = peaks.size / (phasic.size/fs)

    # Turn dictionary into pd.DataFrame and return
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EDA_phasic_")

def peak_detection(phasic, method = "Neurokit", fs=700):
    """
    Description
    -----------
    Detect the peaks of the phasic component using different options
    The Neurokit options does not give the offset, so that is calculated by finding the the first index that is lower than the 50% or 63% value

    Parameters
    ----------
    phasic : np.array
        The phasic component of the signal
    method : string
        Method used, options:
            - manual
            - neurokit
            - gamboa2008 (neurokit)
            - kim2004 (neurokit)
            - vanhalem2020 (neurokit)
            - nabian2018 (neurokit)
    fs : int or float
        (DOWNSAMPLED) sampling frequency of the eda signal
    
    Returns
    -------
    peaks : np.array
        Indexes of the peaks
    onset : np.array
        Indexes of the start of a response
    offset50 : np.array
        Index where the response has gone back to 50% of its original value
    offset63 : np.array
        Index where the response has gone back to 67% of its original value
    magnitude : np.array
        Magnitude of a peak (peak-onset usually)
    """
    offset50 = []
    offset63 = []

    # if method == "manual":
    #     rel_height=0.63

    #     peaks, _ = scipy.signal.find_peaks(phasic)#your code here]
    #     magnitude, _, __ = scipy.signal.peak_prominences(phasic,peaks)#your code here]

    #     widths = np.asarray(scipy.signal.peak_widths(phasic,peaks,rel_height)) #your code here]
    #     rel_height=0.5
    #     widths2 = np.asarray(scipy.signal.peak_widths(phasic,peaks,rel_height)) #your code here]
    #     # find the indices with an amplitude larger that 0.1
    #     keep = np.full(len(peaks), True)
    #     amplitude_min=0.1*np.max(phasic)
    #     keep[np.where(magnitude<amplitude_min)] = False
    #     # only keep those
    #     peaks=peaks[keep]
    #     magnitude=magnitude[keep]

    #     widths=widths[:,keep]
    #     widths2 = widths2[:,keep]

        
    if method != "manual": 
        t =np.linspace(0, phasic.size/fs, phasic.size)
        df = nk.eda_findpeaks(phasic, sampling_rate=fs, method = method)
        onset = np.array(df["SCR_Onsets"])
        peaks = np.array(df["SCR_Peaks"])
        magnitude = np.array(df["SCR_Height"])


        # Sometimes returns nan values
        valid_indices = ~np.isnan(onset) & ~np.isnan(peaks) & ~np.isnan(magnitude)
        onset = onset[valid_indices]
        peaks = peaks[valid_indices]
        magnitude = magnitude[valid_indices]


        for i in range(onset.size):
            height50 = 0.5 * magnitude[i]
            height63 = (1-0.63) * magnitude[i]
            
            # Find the index where phasic drops below height50
            index50 = np.argwhere(phasic[peaks[i]:] < height50)[:,0]
            if index50.size == 0:
                offset50.append(-1)
            else:
                offset50.append((index50[0] + peaks[i]))
            
            # Find the index where phasic drops below height67
            index63 = np.argwhere(phasic[peaks[i]:] < height63)[:,0]
            if index63.size == 0:
                offset63.append(-1)
            else:
                offset63.append((index63[0] + peaks[i]))

        offset50 = np.array(offset50)
        offset63 = np.array(offset63)


        valid_indices =( np.argwhere(offset50 != -1) & np.argwhere(offset50 != -1))[:,0]
        onset = onset[valid_indices].astype(int)
        peaks = peaks[valid_indices].astype(int)
        magnitude = magnitude[valid_indices]
        offset50 = offset50[valid_indices].astype(int)
        offset63 = offset63[valid_indices].astype(int)

        # print(peaks, onset, offset50, offset63, magnitude)
        # print(peaks.shape, onset.shape, offset50.shape, offset63.shape, magnitude.shape)
        
        if onset.size == 0:
            onset = np.array([0])
            peaks = np.array([0])
            magnitude = np.array([0])
            offset50 = np.array([0])
            offset63 = np.array([0])


    return peaks, onset, offset50, offset63, magnitude
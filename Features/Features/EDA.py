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
    Preprocessing the EDA signal using a lowpass filter and split up (and smoothed) using the split_phasic_tonic() function

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
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    Most features are from:
    Automatic motion artifact detection in electrodermal activity data using machine learning
    Md-Billal Hossain, Hugo F. Posada-Quintero, Youngsun Kong, Riley McNaboe, Ki H. Chon 

    Examples
    --------
    >>>
    """ 

    out_dict = {}
    
    out_dict["Median"] = np.median(eda)
    out_dict["Range"] = np.max(eda) - np.min(eda)
    out_dict["Entropy"] = entropy(eda)

    print(eda.size/fs)
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
    out_dict = {}
    # General phasic features
    out_dict["Mean"] = np.mean(phasic)


    # Peak data
    widths, widths2, peaks = peak_detection(phasic, method="manual", fs=fs)

    # Find features
    out_dict["onset"] = np.mean(peaks - widths[2])/fs
    out_dict["recovery"] = np.mean(widths2[3] - peaks)/fs
    out_dict["RR"] = len(peaks)/len(phasic)
    out_dict["RM"] = np.mean(phasic[peaks] - widths[1])
    out_dict["RT"] = np.mean(widths[0])/fs
    
    # Turn dictionary into pd.DataFrame and return
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EDA_phasic_")

def peak_detection(phasic, method = "manual", fs=700):
    """
    Description
    -----------
    Detect the peaks of the phasic component

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
    widths : np.array
        ?
    peaks : np.array
        peaks of 
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    
    """
    if method == "manual":
        rel_height=0.63

        peaks, _ = scipy.signal.find_peaks(phasic)#your code here]
        heights, _, __ = scipy.signal.peak_prominences(phasic,peaks)#your code here]
        widths = np.asarray(scipy.signal.peak_widths(phasic,peaks,rel_height)) #your code here]
        rel_height=0.5
        widths2 = np.asarray(scipy.signal.peak_widths(phasic,peaks,rel_height)) #your code here]
        # find the indices with an amplitude larger that 0.1
        keep = np.full(len(peaks), True)
        amplitude_min=0.1*np.max(phasic)
        keep[np.where(heights<amplitude_min)] = False
        # only keep those
        peaks=peaks[keep]
        heights=heights[keep]
        widths=widths[:,keep]
        widths2 = widths2[:,keep]
    else: 
        t =np.linspace(0, phasic.size/fs, phasic.size)
        df = nk.eda_findpeaks(phasic, sampling_rate=fs)
        print(df)
        peaks = df["SCR_Onsets"]
        widths = df["SCR_Peaks"]
        widths2 = df["SCR_Height"]

        fig, ax = plt.subplots()
        ax.plot(t, phasic)
        ax.plot(t[peaks], phasic[peaks], 'o')
        ax.plot(t[widths], phasic[widths], 'o')
        ax.plot(t[widths], phasic[widths]-widths2, 'o')
        plt.show()

    return widths, widths2, peaks
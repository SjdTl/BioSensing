import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt
from scipy.ndimage import uniform_filter1d
import scipy
import matplotlib.pyplot as plt

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

    eda = preProcessing(eda, fs)
    
    df_general = feat_gen.basic_features(eda, "EDA")
    df_specific = EDA_specific_features(eda, fs)

    features = pd.concat([df_specific, df_general], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EDA contains a NaN value")
    return features

def preProcessing(unprocessed_eda, fs=700):
    """
    Description
    -----------
    Preprocessing the EDA signal using a lowpass filter an dsmooth

    Parameters
    ----------
    unprocessed_eda : np.array
        the EDA data as received directly by the sensors
    
    Returns
    -------
    eda : np.array
        the EDA data processed
    
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
    order = 4
    cutoff = 5
    b, a = butter(N = order, Wn = cutoff, fs = fs)

    lowpass_eda = filtfilt(b, a, unprocessed_eda)

    # Using a one dimentional uniform filter scipy.ndimage.uniform_filter1d() with mode='nearest' and for size (length of the uniform filter) you can use 75% of the sampling rate.
    size = int(0.75 * fs)
    eda_sm0 = uniform_filter1d(lowpass_eda, size, mode="nearest")

    # Computing the moving average using np.convolve(). To do that you need to first make window using scipy.signal.get_window with parzan kernel, with the same size as previous step. Then concatenate your signal to avoid boundry effect using np.concatenate()
    kernel = "parzen"
    window = get_window(kernel, size)
    w = window/window.sum()

    # Extend signal edges to avoid boundary effects.
    eda_sm0 = np.concatenate((eda_sm0[0] * np.ones(size), eda_sm0, eda_sm0[-1] * np.ones(size)))

    # Compute moving average
    eda_sm = np.convolve(w, eda_sm0, mode = "same")
    eda = eda_sm[size:-size]

    return eda

def split_phasic_tonic(eda, fs = 700, order = 5):
    """
    Description
    -----------
    The electrodermal activity is made of two components:
    phasic and tonic. A very simple approach for decomposing the signal into this 
    two components is by using high and low pass filtering respectively. 
    Use 5th order Butterworth filter with output="sos". and cutoff frequency of 0.05 Hz.

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
    tonic : np.array
        eda after lowpass filtering
    phasic : np.array
        eda after highpass filtering

    Raises
    ------
    error
         description
    
    Notes
    -----
    CHANGE THIS FUNCTION. VALUES ARE STILL SELECTED WITHOUT ANY REASONING

    Examples
    --------
    >>>
    """
    order=5
    freqs=0.05 #Hz
    sos = butter(N = order, Wn = freqs, fs = fs, btype = 'highpass', output="sos")
    phasic= sosfiltfilt(sos, eda)

    highcut=0.05 #Hz
    freqs=highcut
    sos = butter(N = order, Wn = freqs, fs = fs, output="sos")
    tonic= sosfiltfilt(sos, eda )

    return phasic, tonic

def EDA_specific_features(eda, fs):
    """
    Description
    -----------
    Calculate features specific to eda signal
    
    Parameters
    ----------
    eda : np.array
        processed eda siganl
    fs : float or int
        sampling frequency of eda signal
    
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
    # Find relevant data
    phasic, _ = split_phasic_tonic(eda)
    widths, widths2, peaks = peak_detection(phasic)

    # Find features
    out_dict = {}
    out_dict["onset"] = np.mean(peaks - widths[2])/fs
    out_dict["recovery"] = np.mean(widths2[3] - peaks)/fs
    out_dict["RR"] = len(peaks)/len(phasic)
    out_dict["RM"] = np.mean(phasic[peaks] - widths[1])
    out_dict["RT"] = np.mean(widths[0])/fs
    
    # Turn dictionary into pd.DataFrame and return
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EDA_")

def peak_detection(phasic, fs=700, plot = False):
    """
    Description
    -----------
    Detect the peaks of the phasic component

    Parameters
    ----------
    phasic : np.array
        The phasic component of the signal
    fs : int or float
        Sampling frequency of the eda signal
    plot : boolean
        If the results have to be plotted
    
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
    
    if plot == True:
        plot_peaks(phasic, widths, peaks)

    return widths, widths2, peaks

def plot_peaks(phasic, widths, peaks, fs=700):
    """
    Description
    -----------
    Function to plot the results of peak_detection()
    from manual REMOVE THIS:
    peaks of the phasic component (we don't need the tonic component anymore) as an indicator of the orienting responses. 
    after finding all the peaks we only keep those with an amplitude larger than 0.1 and remove the rest

    Parameters
    ----------
    phasic : np.array
         eda after highpass filtering
    widths : np.array
        ?
    peaks : np.array
        peaks of the phasic component
    fs : int or float
        sampling frequency of the eda sensor
        
    Returns
    -------
    plt.show()
    
    Notes
    -----
    CHANGE WHEN USING THE OUTPUT IN THE MANUAL and return svg

    Examples
    --------
    >>>
    """
    plt.figure(figsize=(12,4))
    t =np.arange(0,phasic.size*(1/fs),(1/fs))
    plt.plot(t,phasic,label='phasic')
    plt.plot(t[peaks],phasic[peaks],'o',label='peaks')
    plt.hlines(widths[1], *widths[2:]/np.max(widths[3])*t[-1], color="C2")
    # labels and titles
    plt.xlabel('$Time (s)$')
    plt.ylabel('$EDA$')
    plt.legend()
    plt.show()


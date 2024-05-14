import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
from scipy.signal import butter, iirnotch, lfilter

from . import feat_gen 

def EMG(unprocessed_emg, fs = 700):
    """
    Description
    -----------
    Obtains the features for an emg window of x seconds.

    Parameters
    ----------
    unprocessed_emg : np.array
        emg signal as provided directly by the sensors
    fs : int or float
        sampling frequency of the sensors

    Returns
    -------
    features : pd.DataFrame
        Dataframe (1 row) containing the features:\n
            - WL: waveform length \n
            - MAL: overall muscle activity level (RMS)\n
            - MCI: muscle contraction intensity (average absolute amplitude)\n
            - SSC: slope sign change (number of time slope of the EMG signal changes sign)\n
            and the general features:\n
            - Mean (no meaning in the case of emg)\n
            - Median\n
            - Std\n
            - ...\n
     
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

    emg = preProcessing(unprocessed_emg, fs)

    df_specific = EMG_specific_features(emg, fs)
    # General features contain mean emg, but this has no meaning in the case of emg
    df_general = feat_gen.basic_features(emg, "EMG")

    features = pd.concat([df_specific, df_general], axis=1)

    # Error messages
    if features.isnull().values.any():
        raise ValueError("The feature array of EMG contains a NaN value")
    return features

def EMG_specific_features(emg, fs=700):
    """
    Description
    -----------
    Calculate features specific to emg signal
    Features obtained from:
        Feature reduction and selection for EMG signal classification
        Angkoon Phinyomark, 
        Pornchai Phukpattaranont, 
        Chusak Limsakul
    https://www.sciencedirect.com/science/article/pii/S0957417412001200?via%3Dihub

    The features are added in the same order as mentioned in the article. Some features are ignored and 
    this is mentioned in the comments

    Parameters
    ----------
    emg : np.array
        processed emg data as provided by the sensors
    
    Returns
    -------
    out : pd.DataFrame
        dataframe containing the features mentioned in the docstring of EMG()
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    """

    df_time = timeDomain_features(emg, fs)
    df_freq = freqDomain_features(emg, fs)
    
    features = pd.concat([df_time, df_freq], axis=1)
    
    # Turn dictionary into pd.DataFrame and return
    return features

def timeDomain_features(emg, fs):
    """See EMG_specific_features()"""
    N = emg.size
    threshold = 100 * 10**-6  # Recommendation is between 50 muV and 100 mV (2.1.18 in Feature reduction and selection for EMG signal classification Angkoon Phinyomark ⇑, Pornchai Phukpattaranont, Chusak Limsakul)
    dif_emg = np.diff(emg) # often used

    out_dict = {}
    # Integrated EMG
    out_dict["IEMG"] = np.sum(np.abs(emg))

    # Mean absolute value
    out_dict["MAV"] = np.sum(np.abs(emg)) / N

    # Modified mean absolute value type 1
    # Add a weighted window function w1 to improve the robustness of the MAV feature
    w2 = 0.5 * np.ones(int(N/4))
    # Done like this instead of int(N/2) to prevent rounding errors
    w1 = 1 * np.ones(N-2*w2.size)
    w = np.concatenate([w2, w1, w2])
    out_dict["MAV1"] = np.sum(w * np.abs(emg)) / N

    # Modified mean absolute value type 2
    w1 = np.arange(0, 1, 4/N)
    w2 = np.ones(N - 2*w1.size)
    w = np.concatenate([w1, w2, np.flip(w1)])
    
    out_dict["MAV2"] = np.sum(w * np.abs(emg)) / N

    # Simple square integral is the same as variance
    # Variance is already in basic_features()

    # And the absolute value of the moments
    out_dict["TM3"] = np.abs(np.sum(emg**3)/N)
    out_dict["TM4"] = np.abs(np.sum(emg**4)/N)
    out_dict["TM5"] = np.abs(np.sum(emg**5)/N)
    
    # RMS is already in the basic_features()

    # v-order is practically the same as RMS in the optimal case (v=2)

    # Log detector
    out_dict["LOG"] = np.exp(  (np.sum(np.log(np.abs(emg))))/N  )
 
    # Waveform length
    out_dict["WL"] = np.sum(np.abs(dif_emg))

    # Average amplitude change is the same as the waveform length

    # Difference absolute standard deviation value
    out_dict["DASDV"] = np.sqrt(  np.sum(dif_emg**2) / (N-1)  )

    # Myopulse rate
    out_dict["MYOP"] = np.count_nonzero(np.abs(emg) > threshold) / N

    # Willison amplitude
    out_dict["WAMP"] = np.count_nonzero(np.abs(dif_emg)> threshold)

    # SLope sign change
    out_dict["SSC"] = np.count_nonzero((emg[1:-1]-emg[:-2])*(emg[1:-1]-emg[2:]) > threshold)

    # Overall muscle activity level: RMS is a measure of the amplitude of the EMG signal and reflects the overall muscle activity level
    out_dict["RMS"] = feat_gen.rms(emg)

    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EMG_")

def freqDomain_features(emg, fs):
    """See EMG_specific_features()"""
    out_dict = {}
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("EMG_")

def preProcessing(emg, fs=700):
    """
    Description
    -----------
    Preprocessing of EMG signal by bandpass filtering and removing DC value
        - Lowpass is for removing the DC value. This DC value is present due in the sensors (see report) and because muscle activity is between these values (see citation)
        - Highpass is because we can

    Parameters
    ----------
    emg : np.array
        emg signal as provided directly by the sensors
    fs : int or string
        Sampling rate of the sensors
    
    Returns
    -------
    out : np.array
        processed EMG signal
    
    Raises
    ------
    error
         description
    
    Notes
    -----
    
    """

    # Apply bandpass filter (10-300 Hz):
    filtered_emg = butter_bandpass_filter(emg, 10, 300, fs, 5)
    # Baseline correction is useless with a bandpass or highpass filter
    # No powerline interference removal, since there are valuable signals there 
    return filtered_emg 

def envolope_emg(emg, fs=700):
    """
    Description
    -----------
    Here we perform signal rectification to focus on the magnitude of muscle activity.

    To investigate muscle force and muscle activity, scientist often use a low pass filter to capture the shape or “envelope” of the EMG signal as this is thought to better reflect force generated by a muscle.  

    Parameters
    ----------
    emg : np.array
        processed EMG signal
    fs : int or float
        sampling frequency
    
    Returns
    -------
    emg_envelope : np.array
        Rectified and low-pass filtered emg signal
    
    Notes
    -----
    Not yet used 
    """

    emg_rec = np.abs(emg)
    low_pass= 3 /(fs/2)
    b2, a2 = butter(4, low_pass, btype='low')
    emg_envelope = filtfilt(b2, a2, emg_rec)
    return emg_envelope

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


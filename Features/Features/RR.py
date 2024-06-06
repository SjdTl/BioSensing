import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
import neurokit2 as nk
from sklearn.preprocessing import minmax_scale as normalize
from scipy.signal import butter, sosfiltfilt

from . import feat_gen
from .ECG import rpeak_detector

def RR(unprocessed_rr, fs=700, peak_prominence = 0.15):
    """
    Description
    -----------
    Obtains the features for an respitory rate window of x seconds, derived from the ecg data.

    Parameters
    ----------
    unprocessed_rr : np.array
        unprocessed rr data, the data coming from the ecg-to-rr extracter is considered unprocessed
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
    unprocessed_rr = normalize(unprocessed_rr) * 2 - 1
    rr = preProcessRR(unprocessed_rr)

    df_specific = rr_peak_features(rr, fs, peak_prominence=peak_prominence)
    df_general = general_rr_features(rr, fs)

    features = pd.concat([df_specific, df_general], axis=1)

    # Error messages
    if features.isnull().values.any():
        print(features.to_string())
        feat_gen.quick_plot(rr)
        raise ValueError("The feature array of RR contains a NaN value")
    return features

def ECG_to_RR(ecg, fs=100, method = "vangent2019"):
    if method == "vangent2019" or method == "soni2019" or method == "charlton2016" or method == "sarkar2015":
        # Extract peaks
        
        if len(ecg) == 0:
            raise ValueError("The input ECG signal is empty.")


        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        rpeaks = rpeak_detector(ecg = ecg, fs = fs)
        # Compute rate
        ecg_rate = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg))

        edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs, method = method)

        return normalize(edr) * 2 - 1
    
def preProcessRR(rr, fs=100):
    """
    Description
    -----------
    Filter the breathing signal using a highpass and a lowpass filter
    Most breathing will always happen between 3-25 rreaths per minute.
    See also Peter H Charlton et al 2017 Physiol. Meas. 38 669, Chapter 3.6
    This corresponds to 4/60 and 60/60 breath/s or Hz, since rreathing follows a sinusoidal pattern

    Parameters
    ----------
    rr : np.array
        A respitory rate signal (unit does not really matter)
    fs : float or int
        Sampling frequency of the device
    
    Returns
    -------
    RR_hl : np.array
         high- and lowpassed respitory rate signal. 
    """


    # highpass filter
    rr_hp, _ = highpassrr(rr, fs)
    # lowpass filter
    rr_lp, _ = lowpassrr(rr_hp, fs)

    rr_cut = cut_extreme_peaks(rr_lp)
    return rr_cut
def highpassrr(rr, fs, N=8):
    lowcut= 3 #breaths/min
    lowcut = lowcut/(60) #Hz
    sos = butter(N, lowcut, btype = 'highpass', fs=fs, output = 'sos')
    filtered = sosfiltfilt(sos,rr)
    return filtered, sos
def lowpassrr(rr, fs, N = 5):
    highcut=25 # breaths/min
    highcut= highcut/(60) # Hz 
    sos = butter(N, highcut, btype = 'lowpass', fs=fs, output = 'sos')
    filtered = sosfiltfilt(sos,rr)
    return filtered, sos
    
def general_rr_features(rr, fs=700):
    """
    Description
    -----------
    Calculate the general features (mean, std, etc) of RR signal
    
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

    # Find features
    out_dict = {}

    out_dict["Mean"] = np.nanmean(rr)
    out_dict["STD"] = np.nanstd(rr)
    out_dict["Median"] = np.nanmedian(rr)
    out_dict["RMS"] = np.sqrt(np.nanmean(np.square(rr)))

    # Turn dictionary into pd.DataFrame and return
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("RR_")
    
def rr_peak_features(rr, fs=700, peak_prominence = 0.15):
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

    peak_index, through_index = peak_detection_RR(rr, fs=fs, peak_prominence=peak_prominence)
    # Find features
    diff_peaks = np.diff(peak_index) / fs
    diff_diff_peaks = np.diff(diff_peaks)
    out_dict = {}
    T = np.size(rr) / fs

    # Direct breathing rate
    out_dict["Breathing_rate1"] = np.size(peak_index) / T * 60
    out_dict["Breathing_rate2"] = 60 / np.nanmean(diff_peaks)
    out_dict["Max_breath"] = np.max(diff_peaks) / T * 60
    out_dict["Min_breath"] = np.min(diff_peaks) / T * 60

    # Deviation
    out_dict["RMSSD"] = np.sqrt(np.nanmean(diff_diff_peaks ** 2))
    out_dict["SDBB"] = np.nanstd(diff_peaks)
    out_dict["SDSD"] = np.nanstd(diff_diff_peaks)
    meanBB = np.nanmean(diff_peaks)
    out_dict["CVBB"] = out_dict["SDBB"] / meanBB
    out_dict["CVSD"] = out_dict["RMSSD"] / meanBB

    # Robust
    out_dict["MedianBB"] = np.nanmedian(diff_peaks)
    # out_dict["MadBB"] = 

    nn50 = np.sum(np.abs(diff_diff_peaks) > 0.5)
    nn20 = np.sum(np.abs(diff_diff_peaks) > 0.2)
    out_dict["pNN50"] = nn50 / len(rr)
    out_dict["pNN20"] = nn20 / len(rr)

    # Turn dictionary into pd.DataFrame and return
    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("RR_")

def cut_extreme_peaks(rr, fs=700):
    """
    Description
    -----------
    Take in normalized (-1,1) rr and cut of small extreme peaks
    """

    T = rr.size / fs

    mean = np.mean(rr)

    limit = 0.5
    a =  10
    upper_range = rr > (limit+mean)
    # Only if there are extreme peaks, which are of course not common
    if np.count_nonzero(upper_range) < T / 15 * fs:
        peaks = rr[upper_range]
        relative_peaks = peaks - limit - mean
        rr[upper_range] = limit + mean + np.log(1+a * relative_peaks)/a

    lower_range = (rr < -limit + mean)
    # Only if there are extreme peaks, which are of course not common
    if np.count_nonzero(lower_range) < T / 15 * fs:
        peaks = rr[lower_range]
        relative_peaks = np.abs(peaks + limit - mean)
        rr[lower_range] = -limit + mean - np.log(1+a * relative_peaks) /a

    rr_normalized = normalize(rr) * 2 -1
    return rr_normalized - np.mean(rr_normalized)

def peak_detection_RR(rr, fs=700, peak_prominence = 0.15, peak_distance = 1, method = "scipy"):
    """
    Code copied from Neurokit: https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/rsp/rsp_findpeaks.html#rsp_findpeaks
    Neurokit isn't used since this gives an error if there are no peaks (e.g. straight line) or a 
    trough is higher than a peak (even if they are completely unrelated)
    """


    if method == "scipy":

        peak_distance = fs * peak_distance
        peaks, _ = find_peaks(
            rr, distance=peak_distance, prominence=peak_prominence
        )
        troughs, _ = find_peaks(
            -rr, distance=peak_distance, prominence=peak_prominence
        )

        # Combine peaks and troughs and sort them.
        extrema = np.sort(np.concatenate((peaks, troughs)))
        # Sanitize.
        extrema, amplitudes = _rsp_findpeaks_outliers(rr, extrema, amplitude_min=0)
        if extrema.size != 0:
            # peaks, troughs = _rsp_findpeaks_sanitize(extrema, amplitudes)
            1==1
        else:
            peaks = [0]
            troughs = [0]
    return peaks, troughs

def _rsp_findpeaks_outliers(rsp_cleaned, extrema, amplitude_min=0.3):
    """From Neurokit"""
    # Only consider those extrema that have a minimum vertical distance to
    # their direct neighbor, i.e., define outliers in absolute amplitude
    # difference between neighboring extrema.
    vertical_diff = np.abs(np.diff(rsp_cleaned[extrema]))
    median_diff = np.median(vertical_diff)
    min_diff = np.where(vertical_diff > (median_diff * amplitude_min))[0]
    extrema = extrema[min_diff]

    # Make sure that the alternation of peaks and troughs is unbroken. If
    # alternation of sign in extdiffs is broken, remove the extrema that
    # cause the breaks.
    amplitudes = rsp_cleaned[extrema]
    extdiffs = np.sign(np.diff(amplitudes))
    extdiffs = np.add(extdiffs[0:-1], extdiffs[1:])
    removeext = np.where(extdiffs != 0)[0] + 1
    extrema = np.delete(extrema, removeext)
    amplitudes = np.delete(amplitudes, removeext)

    return extrema, amplitudes


# filepath = os.path.join(dir_path, "Raw_data", "raw_small_test_data.pkl")
# print(test(filepath))
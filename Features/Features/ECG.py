import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter, sosfiltfilt, welch, find_peaks
from scipy.stats import iqr
from ecgdetectors import Detectors
from scipy.stats import mode
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.preprocessing import minmax_scale as normalize
import pywt
from statsmodels.tsa.ar_model import AutoReg


from . import feat_gen

def ECG(unprocessed_ecg, fs= 700, wavelet_AR=False):
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

    features = []

    # Commented features not used in final analysis

    features.append(ECG_specific_features(ecg, fs))
    features.append(feat_gen.basic_features(ecg, "ECG_time"))
    if wavelet_AR == True:
        features.append(ecg_wavelet_features(ecg))
        features.append(ecg_AR_features(ecg))

    features = pd.concat(features, axis=1)

    # Error messages
    if features.isnull().values.any():
        print(features.to_string())    
        raise ValueError("The feature array of ECG contains a NaN value")
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
    filtered = sosfiltfilt(sos, ecg)
    return filtered, sos
def notchecg(ecg, fs):
    cut=50
    b, a = iirnotch(cut, 30, fs)
    filtered = lfilter(b,a,ecg)
    return filtered, b, a

def ecg_wavelet_features(ecg):
    """Wavelet features of the ECG"""
    out_dict = {}

    (cA3, cD3, cD2, cD1) = pywt.wavedec(ecg, 'haar', level=3)
    coefficients = {"cA3" :cA3, "cD3": cD3, "cD2" : cD2, "cD1" : cD1}

    for coeff in coefficients:
        out_dict['mean_' + str(coeff)] = np.mean(coefficients[coeff])
        out_dict['median_' + str(coeff)] = np.median(coefficients[coeff])
        out_dict['std_' + str(coeff)] = np.std(coefficients[coeff])
        out_dict['range_' + str(coeff)] = np.ptp(coefficients[coeff])

    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("ECG_wavelet_")

def ecg_AR_features(ecg):
    """Autoregression features of the ECG"""
    out_dict = {}
    # Fit the AR(2) model
    series = pd.Series(ecg)
    model = AutoReg(series, lags=2)
    model_fit = model.fit()
    out_dict["intercept"] = model_fit.params.iloc[0]
    out_dict["lag1"] = model_fit.params.iloc[1]
    out_dict["lag2"] = model_fit.params.iloc[2]

    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("ECG_AR_")

def ECG_specific_features(ecg, fs):
    """
    Description
    -----------
    Calculate features specific to ecg signal
        - pNN50: The proportion of RR intervals greater than 50ms, out of the total number of
          RR intervals
        - pNN20: The proportion of RR intervals greater than 20ms, out of the total number of
          RR intervals
        - RMSSD: The square root of the mean of the squared successive differences between
          adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
          therefore it is redundant to report correlations with both (Ciccone, 2017)
        - ...

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
    out_dict["Heart_rate"] = 60 / np.mean(rri) * 1000
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

    nn40 = np.sum(np.abs(diff_rri) > 40)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out_dict["pNN50"] = nn40 / (len(diff_rri) + 1) * 100
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

    # if freq_domain == True:
    #     frequency, power = welch(
    #     ecg,
    #     fs=fs,
    #     return_onesided = True,
    #     nperseg = np.size(ecg)
    #     )
    #     hr_freq = 1000/np.mean(rri)
    #     closest_idx = (np.abs(frequency - hr_freq)).argmin()

    #     out_dict["HR_power"] = power[closest_idx]
    #     out_dict["Peak_frequency"] = frequency[np.argmax(power)]
    #     out_dict["Spectral_energy"] = np.mean(power)
    #     out_dict["Norm_peak_power"] = np.max(power)/out_dict["Spectral_energy"]

    return pd.DataFrame.from_dict(out_dict, orient="index").T.add_prefix("ECG_HRV_")

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

    # Find peaks
    rpeaks_dict = nk.ecg_findpeaks(ecg, sampling_rate=fs, method="Neurokit")

    # detectors = Detectors(fs)

    # r_peaks_pan = detectors.pan_tompkins_detector(ecg)
    # r_peaks_pan = np.asarray(r_peaks_pan)

    return rpeaks_dict["ECG_R_Peaks"] 
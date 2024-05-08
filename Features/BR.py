import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt
from scipy.ndimage import uniform_filter1d
import feat_gen as feat_gen
import scipy
import matplotlib.pyplot as plt
import tqdm
import neurokit2 as nk

# def RR

def ECG_to_RR(ecg, fs=100, method = "vangent2019"):
    if method == "vangent2019" or method == "soni2019" or method == "charlton2016" or method == "sarkar2015":
        # Extract peaks
        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        # Compute rate
        ecg_rate = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg))

        edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs, method = method)

        return edr
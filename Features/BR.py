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

def ECG_to_RR(ecg, fs=100, method = "Neurokit"):
    if method == "Neurokit":
        # Extract peaks
        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        # Compute rate
        ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg))

        edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs)

        return edr
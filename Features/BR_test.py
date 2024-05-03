import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt
from scipy.ndimage import uniform_filter1d
import feat_gen as feat_gen
import scipy
import matplotlib.pyplot as plt
import tqdm

import ECG as ECG
import BR as BR

dir_path = os.path.dirname(os.path.realpath(__file__))

def spider_data(folderpath):
    """
    Description
    -----------
    Open the data of the spider-fearful dataset for testing the breathing extraction. The subject data is dropped since this is not of interest

    Parameters
    ----------
    folderpath : string
        path to the spiderfearful dataset. Include the folder as downloaded from the paper. The (relevant) data is of the form:
            spiderfearful/VPOx/BitalinoBR.txt: 
                - Column 1: Respiration in % (indicates the deflection of the piezo sensor in the chest strap), 
                value range from -50% to 50%; 
                - Column 2: timestamp with format hhmmss.milliseconds, used for the mapping of the video clip time windows (can be ignored)
                - Column 3: Label for RAW data (can be ignored)
            spiderfearful/VPOx/BitalinoECG.txt:
                - Column 1: ecg in mV (value range from -1.5mV to 1.5mV)
                - Column 2: timestamp with format hhmmss.milliseconds, used for the mapping of the video clip time windows (can be ignored)
                - Column 3: can be ignored

    
    Returns
    -------
    data : dictionary of np.arrays
        dictionary of the form:
            data = {BR = [the breathing data] 
                    ECG = [the ecg data]}
    
    Notes
    -----

    Examples
    --------
    >>>
    """


    br = np.array([])
    ecg = np.array([])

    for root, dirs, files in os.walk(folderpath, topdown=False):
        for name in tqdm.tqdm(dirs):
            cur_br = np.genfromtxt(os.path.join(folderpath, name, "BitalinoBR.txt"), delimiter="")[:,0]
            cur_ecg = np.genfromtxt(os.path.join(folderpath, name, "BitalinoECG.txt"), delimiter="")[:,0]
            br = np.concatenate((br, cur_br))
            ecg = np.concatenate((ecg, cur_ecg))

    data = {"BR": br,
            "ECG": ecg}
    
    return data

def plot_spider(ecg, br, br_ex=None, fs=100):

    t = np.arange(0, ecg.size * (1/fs), 1/fs)

    if br_ex.any() != None:
        fig, ax = plt.subplots(3,1)
        ax[2].plot(t, br_ex)
        ax[2].set_xlabel("Time ($s$)")
        ax[2].set_ylabel("Repiration extracted (%)")
    else:
        fig,ax = plt.subplots(2,1)

    ax[0].plot(t, ecg)
    ax[1].plot(t, br)

    ax[0].set_xlabel("Time ($s$)")
    ax[0].set_ylabel("ECG ($mV$)")

    ax[1].set_xlabel("Time ($s$)")
    ax[1].set_ylabel("Repiration (%)")

    plt.show()

d = spider_data(os.path.join(dir_path, "spiderfearful"))
pre= ECG.preProcessing(d["ECG"][0:1500], 100)
print(pre, pre.shape)
edr = BR.test(d["ECG"][0:1500])

plot_spider(ecg = (d["ECG"][0:1500]), br = d["BR"][0:1500], br_ex=edr)


import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, iirnotch, lfilter
import feat_gen as feat_gen
import scipy
import matplotlib.pyplot as plt
import tqdm

import ECG as ECG
import BR as BR
from feat_head import split_time

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

def preProcess(RR, fs=100):
    """
    Description
    -----------
    Filter the breathing signal using a highpass and a lowpass filter
    Most breathing will always happen between 4-60 breaths per minute.
    See also Peter H Charlton et al 2017 Physiol. Meas. 38 669, Chapter 3.6
    This corresponds to 4/60 and 60/60 breath/s or Hz, since breathing follows a sinusoidal pattern

    Parameters
    ----------
    RR : np.array
        A respitory rate signal (unit does not really matter)
    fs : float or int
        Sampling frequency of the device
    
    Returns
    -------
    RR_hl : np.array
         high- and lowpassed respitory rate signal. 
    
    Notes
    -----
    
    """
    order=5

    # highpass filter
    lowcut= 4 #breaths/min
    lowcut = lowcut/(60) #Hz (does not have to be normalized since fs is specified in butter())
    b, a = butter(5, lowcut, btype = 'highpass', fs=fs)
    RR_h = lfilter(b,a,RR)

    # lowpass filter
    highcut=60 # breaths/min
    highcut= highcut/(60) # Hz (does not have to be normalized since fs is specified in butter())
    b, a = butter(5, highcut, btype="lowpass", fs=fs)
    RR_hl = lfilter(b,a,RR_h)
    return RR_hl

from scipy.fft import fft, fftfreq
def fft_RR(RR, fs=100):
    yf = fft(RR)
    xf = fftfreq(RR.size, 1/fs)

    plt.plot(xf, np.abs(yf))
    plt.show()


def plot_spider(ecg, br, br_extracted=None, br_unprocessed=None, fs=100, shift = None):

    t = np.arange(0, ecg.size * (1/fs), 1/fs)

    if np.any(br_extracted) and np.any(br_unprocessed):
        fig, ax = plt.subplots(4,1)
        ax[2].plot(t, br_extracted)
        ax[2].set_xlabel("Time ($s$)")
        ax[2].set_ylabel("Respiration extracted (%)")
        ax[3].plot(t, br_unprocessed)
        ax[3].set_xlabel("Time ($s$)")
        ax[3].set_ylabel("Unprocessed respiration rate")
    elif np.any(br_extracted) and not np.any(br_unprocessed):
        fig, ax = plt.subplots(3,1)
        ax[2].plot(t, br_extracted)
        ax[2].set_xlabel("Time ($s$)")
        ax[2].set_ylabel("Respiration extracted (%)")
    elif not np.any(br_extracted) and np.any(br_unprocessed):
        fig, ax = plt.subplots(3,1)
        ax[2].plot(t, br_unprocessed)
        ax[2].set_xlabel("Time ($s$)")
        ax[2].set_ylabel("Unprocessed respiration rate")
    else:
        fig,ax = plt.subplots(2,1)

    ax[0].plot(t, ecg)
    ax[1].plot(t, br)

    ax[0].set_xlabel("Time ($s$)")
    ax[0].set_ylabel("ECG ($mV$)")

    ax[1].set_xlabel("Time ($s$)")
    ax[1].set_ylabel("Repiration (%)")

    plt.show()

def compare_extracted_vs_real(RR_extracted, RR_real, step = 0.05, max_shift = 1, fs=100):
    """
    Description
    -----------
    Determine the accuracy of the extraction method using the Pearson Correlation Coefficient (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
    Also used in Peter H Charlton et al 2017 Physiol. Meas. 38 669, Chapter 3.6

    - Normalize both signals
    - Start by removing the phase shift between the signals (since this is irrelevant for accuracy)
    - Determine CC using numpy, 
        as defined $\rho_{X,Y}=\frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y}$ or in ASCII: Px,y = cov(X,Y)/(Sx * Sy)

    Parameters
    ----------
    RR_extracted : np.array
        Array consisting of the respitory rate extracted from the ECG data
    RR_real : np.array
        Array consisting of the respitory rate as measured (and preprocessed of course)
    step : float
        Resolution (in seconds) for calculating the shift
    max_shift : float
        Maximum seconds of phase shift
    fs : int
        Sampling rate 
    
    Intermediate 
    ------------
    CC_matrix : np.array
        Covariance matrix https://en.wikipedia.org/wiki/Covariance_matrix of the form:
        
        cov(*X*) = [1, Pxy 
                  Pxy, 1]
        
    Returns
    -------
    CC : float
        Pearson's correlation coefficent (-1<p<1)
    shift : float
        Phase shift in seconds between the two signals
    """

    # Normalize 
    RR_extracted = RR_extracted / RR_extracted.max()
    RR_real = RR_real / RR_real.max()

    CC = []
    shift = []
    # Calculate CC for all phase shifts -2s till 2s with steps of 0.05s
    # Move a standard smaller window of RR_real (called RR_window) alongside RR_extracted (RR_cur)
    RR_window = RR_real[int(fs * max_shift):int(-fs * max_shift)]

    timeshifts = np.arange(0,2 * max_shift,step) #s
    for current_shift in timeshifts:
        RR_cur = RR_extracted[int(current_shift*fs):-(int(2 * max_shift*fs)-int(current_shift*fs))] # written like this to prevent rounding errors
        CC_matrix =np.corrcoef(RR_window, RR_cur)
        CC.append(CC_matrix[0][1])
        shift.append(current_shift)

    max_index = np.argmax(CC)
    return CC[max_index], shift[max_index]


def determine_RR_accuracy(dataset, method="Neurokit", T=60, example = True):
    """
    Description
    -----------
    Determines the accuracy of an RR extraction method (RR from ECG) by comparing the extracted RR data of the
    spider dataset ecg data and the respitory rate data from this same dataset
    The comparison is done using the Pearson Correlation Coefficient
    
    The coefficient is determined for several windows of T seconds. 
    Returns the box-plot for all coefficient and optionally a plot of one of these windows with all relevant signals if specified

    Parameters
    ----------
    dataset : dictionary with np.arrays
        Dictionary extracted from the spiderdata of the form:
            dataset = {ECG : [np.array],
                        RR : [np.array]}
        The ECG and RR data are 1 dimensional arrays. The spiderfearful data is just lumped together without taking subjects into account
    method : string
        Method used to determine the RR from ECG data, options:
            - neurokit
    T : int
        Size of the windows in seconds
    
    Returns
    -------
    out : type
         description
    
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
    fs = 100 # for spider fearful dataset

    # Split up data
    RR_split = split_time(np.array([dataset["BR"]]), fs, T)[0]
    ECG_split = split_time(np.array([dataset["ECG"]]), fs, T)[0]

    CC = []
    shift = []
    for RR, ECG in (zip(RR_split, ECG_split)):
        processed_RR = preProcess(RR, fs)
        extracted_RR = BR.ECG_to_RR(ECG, fs=fs, method = method)

        cur_CC, cur_shift = compare_extracted_vs_real(extracted_RR, processed_RR)
        CC.append(cur_CC)
        shift.append(cur_shift)

    plt.boxplot([CC, shift])
    plt.show()

    # if example == True:



data = spider_data(os.path.join(dir_path, "spiderfearful"))
determine_RR_accuracy(data)





# RR = d["BR"][0:2500]
# RR_pre = preProcess(RR)
# # fft_RR(RR)
# # fft_RR(RR_pre)

# edr = BR.ECG_to_RR(d["ECG"][0:2500])

# CC, shift = compare_extracted_vs_real(RR_pre, edr)
# print(CC, shift)

# plot_spider(ecg = (d["ECG"][0:2500]), br = RR_pre/RR_pre.max(), br_extracted=edr/edr.max(), br_unprocessed=RR/RR.max(), fs=100, shift = shift)


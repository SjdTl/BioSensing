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
from sklearn.preprocessing import minmax_scale as normalize
import neurokit2 as nk

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


def plot_spider(ecg, br, br_extracted, br_unprocessed, shift, CC, fs=100):
    # normalize
    br_extracted = normalize(br_extracted)
    br_unprocessed = normalize(br_unprocessed)
    br = normalize(br)

    t = np.arange(0, ecg.size * (1/fs), 1/fs)

    fig,ax = plt.subplots(4,1)

    ax[0].plot(t, ecg)
    ax[0].set_xlabel("Time ($s$)")
    ax[0].set_ylabel("ECG ($mV$)")

    ax[1].plot(t, br_unprocessed)
    ax[1].set_xlabel("Time ($s$)")
    ax[1].set_ylabel("Repiration (unprocessed)")

    ax[2].plot(t, br)
    ax[2].set_xlabel("Time ($s$)")
    ax[2].set_ylabel("Respiration (processed)")

    br_ex_shift = np.concatenate((np.zeros(int(shift*fs)), br_extracted[int(shift * fs):]))
    ax[3].plot(t, br_ex_shift)
    ax[3].set_xlabel("Time ($s$)")
    ax[3].set_ylabel("Extracted respiration (shifted)")

    ax[0].set_title(f"Respiration rate extracted from ecg with an accuracy of {round(CC,2)}")

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
    RR_extracted = normalize(RR_extracted)
    RR_real = normalize(RR_real)
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
    return CC[max_index], shift[max_index] - max_shift


def determine_RR_accuracy(dataset, method="vangent2019", T=60, example = True):
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
    example: boolean
        Plot the ecg, RR (unprocessed and processed) and extracted RR of one of the windows (selected randomly)
    
    Returns
    -------
    CC : np.array
        array of the correlation coefficients of all the windows
    shift : np.array
        array of the timeshifts of all the windows
    
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

    # For selecting a window to plot
    randomvalue = np.random.randint(low=0, high=RR_split.shape[0], size=1)
    i = 0

    max_shift = 1
    CC = []
    shift = []
    for RR, ECG in (zip(RR_split, ECG_split)):
        processed_RR = preProcess(RR, fs)
        ECG = nk.ecg_clean(ECG, sampling_rate=fs)
        if method != "control":
            extracted_RR = BR.ECG_to_RR(ECG, fs=fs, method = method)
        if method == "control":
            extracted_RR = processed_RR + np.sin(2*np.pi * np.linspace(0, T/2, processed_RR.size))

        cur_CC, cur_shift = compare_extracted_vs_real(extracted_RR, processed_RR, max_shift=max_shift)
        CC.append(cur_CC)
        shift.append(cur_shift)

        if example == True and i==randomvalue:
            plot_spider(ECG, processed_RR, extracted_RR, RR, cur_shift+max_shift, cur_CC)
        i += 1

    return CC, shift
    

def compare_methods(dataset, methods=["control", "vangent2019", "soni2019", "charlton2016", "sarkar2015"], T=60, example = True):
    all_CC = []
    all_shift = []

    for method in methods:
        cur_CC, cur_shift = determine_RR_accuracy(dataset, method, T, example)
        all_CC.append(cur_CC)
        all_shift.append(cur_shift)

    fig, ax = plt.subplots(1,2)
    ax[0].boxplot(all_CC, labels=methods)
    ax[0].set_ylabel("Correlation coefficient")
    ax[1].boxplot(all_shift, labels=methods)
    ax[1].set_ylabel("Time (s)")
    ax[0].set_title("Correlation coefficient of breathing extraction")
    ax[1].set_title("Resulted timeshift from extraction")
    plt.show()



data = spider_data(os.path.join(dir_path, "spiderfearful"))
compare_methods(data, example=False, T=60)
# determine_RR_accuracy(data, method = "vangent2019")


# RR = d["BR"][0:2500]
# RR_pre = preProcess(RR)
# # fft_RR(RR)
# # fft_RR(RR_pre)

# edr = BR.ECG_to_RR(d["ECG"][0:2500])

# CC, shift = compare_extracted_vs_real(RR_pre, edr)
# print(CC, shift)

# plot_spider(ecg = (d["ECG"][0:2500]), br = RR_pre/RR_pre.max(), br_extracted=edr/edr.max(), br_unprocessed=RR/RR.max(), fs=100, shift = shift)


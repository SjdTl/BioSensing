import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, get_window, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, iirnotch, lfilter
import scipy
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import minmax_scale as normalize
import neurokit2 as nk
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import freqz

from Features import ECG
from Features import RR
from Features.feat_head import split_time
from Features import feat_gen

dirpath = (os.path.dirname(os.path.realpath(__file__)))

def spider_data(folderpath, testing = False, T=60, fs= 100):
    """
    Description
    -----------
    Open the data of the spider-fearful dataset for testing the rreathing extraction. The subject data is dropped since this is not of interest

    Parameters
    ----------
    folderpath : string
        path to the spiderfearful dataset. Include the folder as downloaded from the paper. The (relevant) data is of the form:
            spiderfearful/VPOx/Bitalinorr.txt: 
                - Column 1: Respiration in % (indicates the deflection of the piezo sensor in the chest strap), 
                value range from -50% to 50%; 
                - Column 2: timestamp with format hhmmss.milliseconds, used for the mapping of the video clip time windows (can be ignored)
                - Column 3: Label for RAW data (can be ignored)
            spiderfearful/VPOx/BitalinoECG.txt:
                - Column 1: ecg in mV (value range from -1.5mV to 1.5mV)
                - Column 2: timestamp with format hhmmss.milliseconds, used for the mapping of the video clip time windows (can be ignored)
                - Column 3: can be ignored
    testing : boolean
        Only return a small part of the data, don't return everything
    T : int
        Length of the small array
        

    
    Returns
    -------
    ecg : np.array
        ecg data of length T
    rr : np.array
        rr data of length T 
    """


    rr = np.array([])
    ecg = np.array([])

    i=0
    for root, dirs, files in os.walk(folderpath, topdown=False):
        for name in tqdm.tqdm(dirs):
            cur_rr = np.genfromtxt(os.path.join(folderpath, name, "Bitalinobr.txt"), delimiter="")[:,0]
            cur_ecg = np.genfromtxt(os.path.join(folderpath, name, "BitalinoECG.txt"), delimiter="")[:,0]
            rr = np.concatenate((rr, cur_rr))
            ecg = np.concatenate((ecg, cur_ecg))
            i = i+1
            if testing == True and i==3:
                break

    if testing == True:
        ecg_splitted = split_time(np.array([ecg]), fs, T)[0]
        rr_splitted = split_time(np.array([rr]), fs, T)[0]
        random_index = np.random.randint(low=0, high = ecg_splitted.shape[0]-1, size=1)
        return ecg_splitted[random_index][0], rr_splitted[random_index][0]
    else: 
        return ecg, rr

def preProcessRR(rr, fs=100):
    """
    Description
    -----------
    Filter the breathing signal using a highpass and a lowpass filter
    Most rreathing will always happen between 4-60 rreaths per minute.
    See also Peter H Charlton et al 2017 Physiol. Meas. 38 669, Chapter 3.6
    This corresponds to 4/60 and 60/60 rreath/s or Hz, since rreathing follows a sinusoidal pattern

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

    order=5

    # highpass filter
    lowcut= 4 #breaths/min
    lowcut = lowcut/(60) #Hz
    b, a = butter(5, lowcut, btype = 'highpass', fs=fs)
    rr_h = lfilter(b,a,rr)

    # lowpass filter
    highcut=60 # breaths/min
    highcut= highcut/(60) # Hz 
    b, a = butter(5, highcut, btype="lowpass", fs=fs)
    rr_hl = lfilter(b,a,rr_h)
    return rr_hl


def plot_spider(ecg, rr, rr_extracted, rr_unprocessed, method, fs=100):
    """
    Description
    -----------
    One plot of subplots of:
        - ECG data
        - Respitory rate (processed and preprocessed)
        - Extracted respitory rate using method method
    """
 
    # normalize
    rr_extracted = normalize(rr_extracted)
    rr_unprocessed = normalize(rr_unprocessed)
    rr = normalize(rr)
    ecg = normalize(ecg)

    t = np.arange(0, ecg.size * (1/fs), 1/fs)

    fig,ax = plt.subplots(4,1, figsize=(10, 15))

    ax[0].plot(t, ecg)
    ax[0].set_xlabel("Time ($s$)")
    ax[0].set_title("EMG")
    ax[0].set_ylabel("EMG")

    ax[1].plot(t, rr_unprocessed)
    ax[1].set_xlabel("Time ($s$)")
    ax[1].set_title("Unprocessed RR")
    ax[1].set_ylabel("RR")


    ax[2].plot(t, rr)
    peak_index, through_index = RR.peak_detection_RR(rr, fs=fs)
    ax[2].plot(t[peak_index], rr[peak_index], 'o')
    ax[2].plot(t[through_index], rr[through_index], 'o')
    ax[2].set_xlabel("Time ($s$)")
    ax[2].set_title("Processed RR")
    ax[2].set_ylabel("RR")

    ax[3].plot(t, rr_extracted)
    peak_index, through_index = RR.peak_detection_RR(rr_extracted, fs=fs)
    ax[3].plot(t[peak_index], rr_extracted[peak_index], 'o')
    ax[3].plot(t[through_index], rr_extracted[through_index], 'o')
    ax[3].set_xlabel("Time ($s$)")
    ax[3].set_title("Extracted RR")
    ax[3].set_ylabel("RR")

    plt.suptitle(f"Respiration rate extracted from ecg")

    plt.tight_layout()
    plt.savefig(os.path.join(dirpath, "Plots", "RR_plots", "RR_"+str(method) + ".svg"))


def find_all_features(rr, ecg, method="vangent2019", fs=100, T=60):
    """
    Description
    -----------

    """

    rr = split_time(np.array([rr]), fs, T)[0]
    ecg = split_time(np.array([ecg]), fs, T)[0]
    # For selecting a window to plot
    randomvalue = np.random.randint(low=0, high=rr.shape[0], size=1)
    i = 0

    features = pd.DataFrame()


    if method == 'Original':
        for rr in rr:
            current_feature = RR.RR(preProcessRR(rr), fs=fs)
            # Add to dataframe
            features = pd.concat([features, current_feature], ignore_index=True)
        return features
    else:
        for rr, ecg in (zip(rr, ecg)):
            processed_RR = preProcessRR(rr, fs)
            ecg = nk.ecg_clean(ecg, sampling_rate=fs)
            extracted_RR = RR.ECG_to_RR(ecg, fs=fs, method=method)

            current_feature=RR.RR(extracted_RR, fs=fs)
            features = pd.concat([features, current_feature], ignore_index=True)
            
            if i==randomvalue:
                plot_spider(ecg, processed_RR, extracted_RR, rr, method, fs=fs)
            i += 1
        return features
    

def compare_methods(methods=["Original", "soni2019", "vangent2019", "charlton2016", "sarkar2015"], T=60, fs=100):
    ecg, rr = spider_data(os.path.join(dirpath, "spiderfearful"), testing=False, fs=100)

    features_methods = {}

    for method in methods:
        features_methods[method] = find_all_features(rr, ecg, method, fs=fs, T=T)

    original = features_methods.pop("Original")

    feature_labels = original.columns
    x = np.arange(len(feature_labels))
    width = 0.1  # Width of bar

    fig, ax = plt.subplots()
    
    for i, (method, features) in enumerate(features_methods.items()):
        # % change = (new - old) / old
        percentage_change = (features - original) / original * 100
        mean_percentage_change = percentage_change.mean()

        ax.bar(x + i * width, mean_percentage_change, width, label=method)

    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Percentage Change per Feature')
    ax.set_xticks(x + width * (len(features_methods) - 1) / 2)
    ax.set_xticklabels(feature_labels, rotation=90)
    ax.set_ylim(-200,200)
    ax.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(dirpath, "Plots", "RR_plots", "RR_method_comparison.svg"))
    print(original)
    print(features_methods["soni2019"])
    print(features_methods["sarkar2015"])


def testRR(fs=100):
    """
    Description
    -----------
    Function to test the feature extraction
    
    Parameters
    ----------

    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the features 
        
    """
    ecg, rr = spider_data(os.path.join(dirpath, "spiderfearful"), testing = True)

    rr = preProcessRR(rr)

    df = RR.RR(rr, fs=fs)
    return df

def testECG(ecg):
    """
    Description
    -----------
    Same as testRR, except that the RR extraction from ECG is also tested
    """

def RR_figures(T =40, fs = 100):
    """
    Plots used for the processing flowchart in chapter four
    Returns x plots:
        - Depends on how the preprocessing is done
    """
    ecg, rr = spider_data(os.path.join(dir_path, "spiderfearful"), testing = True, T=T)

    # -----------------------------------------------
    # UNPROCESSED
    # -----------------------------------------------
    # RR unprocessed timedomain
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), rr)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("?")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "rr_unprocessed_td.svg"))


    # RR unprocessed frequency domain
    yf = fftshift(fft(rr))
    xf = fftshift(fftfreq(rr.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "rr_unprocessed_fd.svg"))

    # -----------------------------------------------
    # PROCESSED
    # -----------------------------------------------
    # RR processed timedomain
    rr = preProcessRR(rr)

    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), rr)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("?")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "rr_processed_td.svg"))


    # RR processed frequency domain
    yf = fftshift(fft(rr))
    xf = fftshift(fftfreq(rr.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "rr_processed_fd.svg"))

    # -----------------------------------------------
    # Peak detection
    # -----------------------------------------------
    # RR peak detection timedomain
    fig, ax = plt.subplots()

    peak_index, through_index = RR.peak_detection_RR(rr, fs, method = "Neurokit")

    t = np.linspace(0,T, fs*T)
    ax.plot(t, rr, label = "RR signal")
    ax.plot(t[peak_index], rr[peak_index], 'o', label="Peaks")
    ax.plot(t[through_index], rr[through_index], 'o', label="Throughs")
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("?")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "rr_peak_detection_td.svg"))



dir_path = os.path.dirname(os.path.realpath(__file__))
# print(testRR())
# RR_figures()


# filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
# print(test(filepath))

compare_methods()


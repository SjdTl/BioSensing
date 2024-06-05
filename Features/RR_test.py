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
from scipy.signal import freqz, sosfreqz
from scipy.signal import decimate

from Features import ECG
from Features import RR
from Features.feat_head import split_time
from Features import feat_gen
from Features import feat_head

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
    print("Extracting spider dataset...")

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



def plot_spider(ecg, rr_unprocessed, method, fs=100):
    """
    Description
    -----------
    One plot of subplots of:
        - ECG data
        - Respitory rate (processed and preprocessed)
        - Extracted respitory rate using method method
    """
    T = rr_unprocessed.size / fs
    # normalize
    rr_extracted = RR.ECG_to_RR(ecg, fs=fs, method = method)
    rr_extracted = normalize(rr_extracted) * 2 - 1
    rr_extracted = RR.preProcessRR(rr_extracted)

    rr_unprocessed = normalize(rr_unprocessed) * 2 - 1
    rr = RR.preProcessRR(rr_unprocessed)
    ecg = normalize(ecg) * 2 -1

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
    peak_index, trough_index = RR.peak_detection_RR(rr, fs=fs)
    br = round(np.size(peak_index) / T * 60, 2)
    ax[2].plot(t[peak_index], rr[peak_index], 'o', color='green')
    ax[2].plot(t[trough_index], rr[trough_index], 'o', color='red')
    ax[2].set_xlabel("Time ($s$)")
    ax[2].set_title(f"Processed RR with {br} breaths/min detected")
    ax[2].set_ylabel("RR")

    ax[3].plot(t, rr_extracted)
    peak_index, trough_index = RR.peak_detection_RR(rr_extracted, fs=fs)
    br = round(np.size(peak_index) / T * 60, 2)
    ax[3].plot(t[peak_index], rr_extracted[peak_index], 'o', color='green')
    ax[3].plot(t[trough_index], rr_extracted[trough_index], 'o', color='red')
    ax[3].set_xlabel("Time ($s$)")
    ax[3].set_title(f"Extracted RR with {br} breaths/min detected")
    ax[3].set_ylabel("RR")

    plt.suptitle(f"Respiration rate extracted from ecg")

    plt.tight_layout()
    name = os.path.join(dirpath, "Plots", "RR_plots", "Method_plots", "RR_" + str(method))
    name = feat_head.filename_exists(name, 'svg')
    plt.savefig(name)
    plt.close('all')


def find_all_features(rr, ecg, random_value, method="vangent2019", fs=100, T=60, peak_prominence = 0.15):
    """
    Description
    -----------

    """

    rr = split_time(np.array([rr]), fs, T)[0]
    ecg = split_time(np.array([ecg]), fs, T)[0]
    # For selecting a window to plot
    
    i = 0

    features = pd.DataFrame()


    if method == 'Original':
        for rr in rr:
            current_feature = RR.RR(rr, fs=fs, peak_prominence = peak_prominence)
            # Add to dataframe
            features = pd.concat([features, current_feature], ignore_index=True)
        return features
    else:
        for rr, ecg in (zip(rr, ecg)):
            ecg = ECG.preProcessing(ecg, fs=fs)
            extracted_RR = (RR.ECG_to_RR(ecg, fs=fs, method=method))
            current_feature=RR.RR(extracted_RR, fs=fs, peak_prominence = peak_prominence)
            features = pd.concat([features, current_feature], ignore_index=True)
            
            if i in random_value:
                plot_spider(ecg, rr, method, fs=fs)
            i += 1
        return features
    

def compare_methods(methods=["Original", "soni2019", "vangent2019", "charlton2016", "sarkar2015"], T=60, fs=100, examples = 1, peak_prominence = 0.15, name = "RR_methods_comparison"):
    ecg, rr = spider_data(os.path.join(dirpath, "spiderfearful"), testing=False, fs=100)
    print("Calculating features per method...")

    features_methods = {}

    randomvalue = np.random.randint(low=0, high=int(rr.size / (fs * T) * 0.9), size=examples)
    for method in tqdm.tqdm(methods):
        features_methods[method] = find_all_features(rr = rr, ecg = ecg, random_value =  randomvalue, method = method, fs=fs, T=T, peak_prominence = peak_prominence)

    print("Comparing methods...")

    original = features_methods.pop("Original")

    feature_labels = original.columns
    x = np.arange(len(feature_labels))
    width = 0.1  # Width of bar


    fig, ax = plt.subplots()
    
    for i, (method, features) in enumerate(features_methods.items()):
        # % change = (new - old) / old
        percentage_change = ((features - original) / (original + 10**(-4)) * 100).abs()
        mean_percentage_change = percentage_change.mean(skipna = True, numeric_only = True)

        ax.bar(x + i * width, mean_percentage_change, width, label=method)

    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Relative difference between extracted and measured feature')
    ax.set_xticks(x + width * (len(features_methods) - 1) / 2)
    ax.set_xticklabels(feature_labels, rotation=90)
    ax.set_ylim(0,75)
    ax.legend()
    plt.tight_layout()

    name = os.path.join(dirpath, "Plots", "RR_plots", str(name))
    name = feat_head.filename_exists(name, 'svg')
    plt.savefig(name)

    original.to_excel(os.path.join(dir_path, "Plots", "RR_plots", "Original.xlsx"))
    features_methods["sarkar2015"].to_excel(os.path.join(dir_path, "Plots", "RR_plots", "sarkar2015.xlsx"))


def testECG(filepath, fs=700):
    """
    Description
    -----------
    Same as testRR, except that the RR extraction from ECG is also tested
    """
    ecg = feat_gen.load_test_data("ECG", filepath)
    Q = 7
    ecg = decimate(ecg, Q)
    fs = int(fs/Q)
    processed_ecg = ECG.preProcessing(ecg, fs=fs)
    rr = RR.ECG_to_RR(processed_ecg, fs=fs)

    feat_gen.quick_plot(rr, fs = fs)
    feat_gen.quick_plot(RR.preProcessRR(rr), rr, fs=fs)
    df = RR.RR(ecg, fs)
    return df

def RR_figures(filepath, T = 60, fs = 700):
    """
    Plots used for the processing flowchart in chapter four
    Returns x plots:
        - Depends on how the preprocessing is done
    """
    ecg = feat_gen.load_test_data("ECG", filepath, T=T)
    Q = 7
    fs = int(fs/Q)
    ecg = decimate(ecg, Q)
    processed_ecg = ECG.preProcessing(ecg, fs=fs)
    rr = RR.ECG_to_RR(processed_ecg, fs=fs)

    # -----------------------------------------------
    # PROCESSED ECG
    # -----------------------------------------------
    # ECG processed timedomain
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,T, fs*T), processed_ecg)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Normalized amplitude")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_processed_ecg_td.svg"))


    # ECG processed frequency domain
    yf = fftshift(fft(processed_ecg))
    xf = fftshift(fftfreq(processed_ecg.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlim(xmin=-fs/2, xmax=fs/2)
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart","rr_processed_ecg_fd.svg"))


    # -----------------------------------------------
    # UNPROCESSED
    # -----------------------------------------------
    # RR unprocessed timedomain
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), rr)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Normalized amplitude")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots","Flowchart", "rr_unprocessed_td.svg"))


    # RR unprocessed frequency domain
    yf = fftshift(fft(rr))
    xf = fftshift(fftfreq(rr.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlim(xmin=-10, xmax=10)
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_unprocessed_fd.svg"))

    # -----------------------------------------------
    # PROCESSED
    # -----------------------------------------------
    # RR processed timedomain
    # Bandpass processing
    rrlow, sos = RR.lowpassrr(rr, fs) 
    wlow, hlow = sosfreqz(sos, fs=fs)
    bprr, sos = RR.highpassrr(rrlow, fs) 
    whigh, hhigh = sosfreqz(sos, fs=fs, worN = 512 * 4)


    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), bprr)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Normalized amplitude")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_bp_td.svg"))


    # RR processed frequency domain
    yf = fftshift(fft(bprr))
    xf = fftshift(fftfreq(bprr.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlim(xmin=-10, xmax=10)
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots","Flowchart","rr_bp_fd.svg"))

    # -----------------------------------------------
    # Extreme peaks
    # -----------------------------------------------
    rr_processed = RR.preProcessRR(rr, fs)
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), rr_processed)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Normalized amplitude")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_processed_td.svg"))


    # RR processed frequency domain
    yf = fftshift(fft(rr_processed))
    xf = fftshift(fftfreq(bprr.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_xlim(xmin=-10, xmax=10)
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_processed_fd.svg"))

    # -----------------------------------------------
    # Peak detection
    # -----------------------------------------------
    # RR peak detection timedomain
    fig, ax = plt.subplots()

    peak_index, trough_index = RR.peak_detection_RR(rr_processed, fs)

    t = np.linspace(0,T, fs*T)
    ax.plot(t, rr_processed, label = "RR signal")
    ax.plot(t[peak_index], rr_processed[peak_index], 'o', label="Peaks")
    ax.plot(t[trough_index], rr_processed[trough_index], 'o', label="Throughs")
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Normalized amplitude")

    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_peak_detection_td.svg"))

    # Butterworth bandpass

    fig, ax = plt.subplots()
    ax.semilogx(wlow, 20 * np.log10(abs(hlow)), label="Lowpass filter")
    ax.semilogx(whigh, 20 * np.log10(abs(hhigh)), label="Highpass filter")
    ax.set_xlabel('Frequency [radians / second]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlim(xmax = fs/2, xmin = 0.005)
    ax.set_ylim(ymin = -20, ymax = 5)
    ax.grid(which='both', axis='both')
    ax.legend()
    fig.savefig(os.path.join(dir_path, "plots", "RR_plots", "Flowchart", "rr_bpbutterworth.svg"))


dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
# RR_figures(filepath)
print(testECG(filepath).to_string())

# compare_methods(T=60, examples = 5)

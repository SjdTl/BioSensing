import os
from Features import feat_gen 
from Features.ECG import *
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import freqz, butter, freqs, sosfreqz, welch, find_peaks
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale as normalize
from Features.feat_head import load_dict, split_time, filename_exists

def test(filepath):
    """
    Description
    -----------
    Function to test the signal, without having to call the entire database. Please use this function when looking for data to plot for the report.
    
    Parameters
    ----------
    Filepath: string
        Filepath to the test signal. This should be a pickled dictionary with the following format:
            dict = {EDA: [..]
                    EMG: [..]
                    ECG: [..]}
        Each signal is of one person, one label and includes only a small timeframe
    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the features 
        
    """
    ecg = feat_gen.load_test_data("ECG", filepath, label=3)

    processed_ecg = preProcessing(ecg, fs=700)
    rpeaks_pan = rpeak_detector(processed_ecg, 700)
    fig, ax = plt.subplots()
    t=np.linspace(0,int(np.size(ecg)/700), np.size(ecg))
    ax.plot(t, normalize(processed_ecg), color= 'green')
    ax.plot(t[rpeaks_pan], normalize(processed_ecg)[rpeaks_pan], 'o', color= 'red')
    plt.show()

    df = ECG(ecg, 700)
    return df

def test_peak_detector(filepath, T = 45, fs=700):
    all_data = load_dict(filepath)
    for subject in all_data:
        splitted_data = split_time(np.array([all_data[subject]["ECG"]]), Fs=fs, t=T)[0]
        for frame in splitted_data:
            processed_ecg = preProcessing(frame, fs=fs)
            rpeaks_pan = rpeak_detector(processed_ecg, 700)

            fig, ax = plt.subplots()
            t=np.linspace(0,int(np.size(processed_ecg)/700), np.size(processed_ecg))
            ax.plot(t, normalize(processed_ecg), color= 'green')
            ax.plot(t[rpeaks_pan], normalize(processed_ecg)[rpeaks_pan], 'o', color= 'red')
            ax.set_xlabel("Time [$s$]")
            ax.set_ylabel("Voltage [$mV$]")

            fig.savefig(filename_exists(os.path.join(dir_path, "plots", "ECG_plots", "Peak detector", "ecg_peaks"), extension="svg"))   
            plt.close()

def ECG_figures(filepath, T =10):
    """
    Plots used for the processing flowchart in chapter four
    Returns five plots:
        - Unprocessed and processed (including different plot after notch filter) ECG signal in the timedomain
        - Unprocessed and processed (including different plot after notch filter) ECG signal in the frequency domain
        - Plot of the filters used to process
    The code is connected to the ECG such that changing some parameters their would simply 
    require this to run again to obtain the updated plots (with different cutoff frequencies e.g.)
    """
    fs=700
    ecg = feat_gen.load_test_data("ECG", filepath, T=T, label=2)
    t = np.linspace(0,T, fs*T)

    # ECG unprocessed timedomain
    fig, ax = plt.subplots()

    ax.plot(t, ecg*1000)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Voltage [$mV$]")

    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_unprocessed_td.svg"))

    # Bandpass processing
    lp_ecg, blow, alow = lowpassecg(ecg, fs) 
    wlow, hlow = freqz(blow, alow, fs=fs)

    hp_lp_ecg, sos = highpassecg(lp_ecg, fs) 
    # Increase number of samples since otherwise there are no samples under 0.5 Hz
    whigh, hhigh = sosfreqz(sos, fs=fs, worN = 512 * 4)

    fig, ax = plt.subplots()

    ax.plot(t, hp_lp_ecg*1000, color= 'green')
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Voltage [$mV$]")

    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_bpfiltered_td.svg"))

    # Butterworth bandpass

    fig, ax = plt.subplots()
    ax.semilogx(wlow, 20 * np.log10(abs(hlow)), label="Lowpass filter")
    ax.semilogx(whigh, 20 * np.log10(abs(hhigh)), label="Highpass filter")
    ax.set_xlabel('Frequency [radians / second]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlim(xmax = fs/2, xmin = 0)
    ax.set_ylim(ymin = -40)
    ax.grid(which='both', axis='both')
    ax.legend()
    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_bpbutterworth.svg"))

    # Notch
    processed_ecg, b, a = notchecg(hp_lp_ecg, fs) 
    w, h = freqz(b, a, fs=fs)

    fig, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(abs(h)))
    ax.set_xlabel('Frequency [radians / second]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlim(xmax = fs/2, xmin = 1)
    ax.set_ylim(ymin = -20)
    ax.grid(which='both', axis='both')
    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "notch.svg"))

    # Processed ecg
    fig, ax = plt.subplots()

    ax.plot(t, processed_ecg*1000)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Voltage [$mV$]")

    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_processed_td.svg"))

    # ECG unprocessed frequency domain
    yf = fftshift(fft(ecg))
    xf = fftshift(fftfreq(ecg.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -15)

    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_unprocessed_fd.svg"))

    # ECG bandpass frequency domain
    yf = fftshift(fft(hp_lp_ecg))
    xf = fftshift(fftfreq(hp_lp_ecg.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)), color='green')
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -15)

    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_bpfiltered_fd.svg"))



    # ECG notch frequency domain
    yf = fftshift(fft(processed_ecg))
    xf = fftshift(fftfreq(processed_ecg.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -15)

    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_processed_fd.svg"))

    # Peak detection
    rpeaks_pan = rpeak_detector(processed_ecg, fs)
    fig, ax = plt.subplots()


    ax.plot(t, processed_ecg*1000, color= 'green')
    ax.plot(t[rpeaks_pan], processed_ecg[rpeaks_pan]*1000, 'o', color= 'red')
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Voltage [$mV$]")
    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_peaks.svg"))    

    # HRV PSD
    fig, ax = plt.subplots()
    
    frequency, power = welch(
        processed_ecg,
        fs=fs,
        return_onesided = True,
        nperseg = np.size(processed_ecg)
    )


    ax.plot(frequency, power, color= 'green')
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("PSD [$V^2/Hz$]")
    ax.set_xlim((0, 50))
    fig.savefig(os.path.join(dir_path, "plots", "ECG_plots", "ecg_psd.svg"))    
    
dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
# print(test(filepath))
ECG_figures(filepath, T=10)
# test_peak_detector(filepath)
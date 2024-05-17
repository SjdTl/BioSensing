import os
from Features.EMG import *
from scipy.fft import fft, fftfreq
from scipy.signal import freqz, butter, freqs
import matplotlib.pyplot as plt

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
    emg = feat_gen.load_test_data("EMG", filepath)
    feat_gen.quick_plot(emg, preProcessing(emg), envolope_emg(preProcessing(emg)))
    df = EMG(emg, 700)
    return df

def EMG_figures(filepath, T =4):
    """
    Plots used for the processing flowchart in chapter four
    Returns five plots:
        - Unprocessed and processed EMG signal in the timedomain
        - Unprocessed and processed EMG signal in the frequency domain
        - Plot of the filter used to process
    The code is connected to the EMG such that changing some parameters their would simply 
    require this to run again to obtain the updated plots (with different cutoff frequencies e.g.)
    """
    fs=700
    emg = feat_gen.load_test_data("EMG", filepath, T=T)

    # EMG unprocessed timedomain
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), emg*1000)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Voltage [$mV$]")

    fig.savefig(os.path.join(dir_path, "plots", "EMG_plots", "emg_unprocessed_td.svg"))

    # EMG processed timedomain
    processed_emg, b, a = preProcessing(emg, return_filter = True)
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), processed_emg*1000, color= 'green')
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Voltage [$mV$]")

    fig.savefig(os.path.join(dir_path, "plots", "EMG_plots", "emg_processed_td.svg"))

    # Butterworth
    w, h = freqz(b, a, fs=fs)

    fig, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(abs(h)))
    ax.set_xlabel('Frequency [radians / second]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlim(xmax = fs/2)
    ax.set_ylim(ymin = -100)
    ax.grid(which='both', axis='both')
    fig.savefig(os.path.join(dir_path, "plots", "EMG_plots", "emg_butterworth.svg"))

    # EMG unprocessed frequency domain
    yf = fft(emg)
    xf = fftfreq(emg.size, d= 1/fs)

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -15)

    # EMG processed frequency domain
    yf = fft(processed_emg)

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)), color= 'green')
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -15)

    fig.savefig(os.path.join(dir_path, "plots", "EMG_plots", "emg_processed_fd.svg"))



dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
print(test(filepath))
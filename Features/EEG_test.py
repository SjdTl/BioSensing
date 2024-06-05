import os
from Features.EEG import *
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import freqz

def test(filepath):
    """
    Description
    -----------
    Function to test the signal, without having to call the entire database. Please use this function when looking for data to plot for the report.
    
    Parameters
    ----------
    Filepath: string
        Filepath to the test signal.

    Returns
    -------
    df: pd.DataFrame
        Dataframe containing the features 
        
    """

    eeg = 1

    df = EEG(eeg, 700)
    return df

def EEG_figures(filepath, T =40, fs = 700):
    """
    Plots used for the processing flowchart in chapter four
    Returns x plots:
        - Depends on how the preprocessing is done
    """
    eeg = 1

    # EEG unprocessed timedomain
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), eeg)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Conductance [$\mu S$]")

    fig.savefig(os.path.join(dir_path, "plots", "EEG_plots", "eeg_unprocessed_td.svg"))


    # EEG unprocessed frequency domain
    yf = fftshift(fft(eeg))
    xf = fftshift(fftfreq(eeg.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "EEG_plots", "eeg_unprocessed_fd.svg"))



dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "?") # path to data
print(test(filepath))
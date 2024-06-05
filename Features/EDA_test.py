import os
from Features.EDA import *
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

    eda = feat_gen.load_test_data("EDA", filepath, T=300, label=2)

    df = EDA(eda, 700)
    return df

def EDA_figures(filepath, T =40):
    """
    Plots used for the processing flowchart in chapter four
    Returns five plots:
        - Unprocessed and processed EMG signal in the timedomain
        - Unprocessed and processed EMG signal in the frequency domain
        - Plot of the filter used to process
        - Plot of the tonic and phasic split
        - Plot of the peak detection
    The code is connected to the EDA such that changing some parameters their would simply 
    require this to run again to obtain the updated plots (with different cutoff frequencies e.g.)
    EXCEPT THE BUTTERWORTH FILTER PLOT
    """
    fs=700
    eda = feat_gen.load_test_data("EDA", filepath, T=T)

    # EDA unprocessed timedomain
    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, fs*T), eda)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Conductance [$\mu S$]")

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_unprocessed_td.svg"))

    # EDA lowpass filter and downsampling
    Q = 10
    filtered_eda = butter_EDA(eda, Q=Q)

    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, int(fs*T/Q)), filtered_eda)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Conductance [$\mu S$]")
    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_filtered_td.svg"))

    # Butterworth
    b, a = butter(N = 4, Wn = 5, fs= fs)
    w, h = freqz(b, a, fs=fs)

    fig, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(abs(h)))
    ax.set_xlabel('Frequency [radians / second]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_xlim(xmax = fs/2)
    ax.set_ylim(ymin = -100)
    ax.grid(which='both', axis='both')
    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_butterworth.svg"))

    # EDA phasic and tonic
    phasic, tonic = split_phasic_tonic(filtered_eda, fs/Q)
    # Raise with tonic for displaying
    phasic_raised = tonic + phasic
    
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,T, int(fs*T/Q)), phasic_raised, label="phasic", color="blue")
    ax.plot(np.linspace(0,T, int(fs*T/Q)), tonic, label="tonic", color="aqua")
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Conductance [$\mu S$]")
    ax.legend()
    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "phasic_tonic_td.svg"))


    # EDA unprocessed frequency domain
    yf = fftshift(fft(eda))
    xf = fftshift(fftfreq(eda.size, d= 1/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)))
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_unprocessed_fd.svg"))

    # EDA processed frequency domain
    yf = fftshift(fft(filtered_eda))
    xf = fftshift(fftfreq(filtered_eda.size, d=Q/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)), color= 'green')
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_processed_fd.svg"))

    # EDA phasic tonic frequency domain
    yfphasic = fftshift(fft(phasic))
    yftonic = fftshift(fft(tonic))
    xf = fftshift(fftfreq(yfphasic.size, d=Q/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yftonic)), color= 'lime', label="tonic")
    ax.plot(xf, 10 * np.log10(np.abs(yfphasic)), color= 'green', label="phasic")
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)
    ax.legend()

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "phasic_tonic_fd.svg"))

    # PEAK detection
    peaks, onset, offset50, offset63, magnitude = peak_detection(phasic,fs= fs/Q, method = "Neurokit")

    fig, ax = plt.subplots()

    t =np.linspace(0,T, int(fs*T/Q))

    ax.plot(t,phasic,label='phasic')
    ax.plot(t[peaks],phasic[peaks],'o',label='peaks')
    ax.plot(t[onset],phasic[onset],'o',label='onset')
    ax.plot(t[offset50],phasic[offset50],'o',label=f"50% offset")
    ax.plot(t[offset63],phasic[offset63],'o',label=f'63% offset')

    ax.vlines(t[peaks], ymax = phasic[peaks], ymin = phasic[peaks]- magnitude, linestyle = '--')
    
    # labels and titles
    ax.set_xlabel('Time [$s$]')
    ax.set_ylabel('Conductivity $[\mu S]$')
    ax.legend()

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "peak_detection.svg"))


def compare_phasic_tonic_methods(filepath, T= 100):
    """
    Plots used to compare the phasic and tonic split algorithms
    """
    fs=700
    eda = feat_gen.load_test_data("EDA", filepath, T=T)
    # EDA lowpass filter and downsampling
    Q = 10
    filtered_eda = butter_EDA(eda, Q=Q)

    methods= ["cvxeda", "smoothmedian", "highpass"]

    fig, ax = plt.subplots(2,2, figsize=(10, 10))

    ax[0][0].plot(np.linspace(0,T, int(fs*T/Q)), filtered_eda)
    ax[0][0].set_title("Preprocessed EDA signal")
    ax[0][0].set_xlabel("Time [$s$]")
    ax[0][0].set_ylabel("Conductance [$\mu S$]")
    
    for i in range(0,3):
        # EDA phasic and tonic
        phasic, tonic = split_phasic_tonic(filtered_eda, fs/Q, method=methods[i])
        # Raise with tonic for displaying
        phasic_raised = tonic + phasic
        
        if i < 1:
            ax[i+1][0].plot(np.linspace(0,T, int(fs*T/Q)), phasic_raised, label="phasic", color="blue")
            ax[i+1][0].plot(np.linspace(0,T, int(fs*T/Q)), tonic, label="tonic", color="aqua")
            ax[i+1][0].set_xlabel("Time [$s$]")
            ax[i+1][0].set_ylabel("Conductance [$\mu S$]")
            ax[i+1][0].set_title(str(methods[i]))
            ax[i+1][0].legend()
        else:
            ax[i-2][1].plot(np.linspace(0,T, int(fs*T/Q)), phasic_raised, label="phasic", color="blue")
            ax[i-2][1].plot(np.linspace(0,T, int(fs*T/Q)), tonic, label="tonic", color="aqua")
            ax[i-2][1].set_title(str(methods[i]))
            ax[i-2][1].set_xlabel("Time [$s$]")
            ax[i-2][1].set_ylabel("Conductance [$\mu S$]")
            ax[i-2][1].legend()
    fig.suptitle("Comparison between different eda decomposition methods")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "phasic_tonic_comparison.svg"))

def compare_peak_detection(filepath, T=150):
    """
    Plots used to compare the peak detection algorithms
    """
    fs=700
    eda = feat_gen.load_test_data("EDA", filepath, T=T)
    # EDA lowpass filter and downsampling
    Q = 10
    t =np.linspace(0,T, int(fs*T/Q)) 

    filtered_eda = butter_EDA(eda, Q=Q)
    phasic, tonic = split_phasic_tonic(filtered_eda, fs/Q)

    methods = ["Neurokit", "gamboa2008", "kim2004", "vanhalem2020", "nabian2018"]
    
    fig, ax = plt.subplots(3,2, figsize=(13,9))

    for i in range(len(methods)):

        peaks, onset, offset50, offset63, magnitude = peak_detection(phasic, method = methods[i], fs=fs/Q)

        if i == 0 or i == 1:
            k = i
            j = 0
        if i == 2 or i == 3:
            k = i-2
            j = 1
        if i == 4 or i==5:
            k = i-4
            j=2

        ax[j][k].plot(t,phasic,label='phasic')
        ax[j][k].plot(t[peaks],phasic[peaks],'o',label='peaks')
        ax[j][k].plot(t[onset],phasic[onset],'o',label='onset')
        ax[j][k].plot(t[offset50],phasic[offset50],'o',label=f"50% offset")
        ax[j][k].plot(t[offset63],phasic[offset63],'o',label=f'63% offset')

        ax[j][k].vlines(t[peaks], ymax = phasic[peaks], ymin = phasic[peaks]- magnitude, linestyle = '--')
        
        # labels and titles
        ax[j][k].set_xlabel('Time [$s$]')
        ax[j][k].set_ylabel('Conductivity $[\mu S]$')
        ax[j][k].set_title(str(methods[i]))

    fig.suptitle("Comparison of different peak detection methods")
    # Create the legend in ax[1][2] (which is the same as ax[2][3] in a 2x3 grid)
    handles, labels = ax[0][0].get_legend_handles_labels()
    ax[2][1].legend(handles, labels, loc='upper left')
    ax[2][1].axis('off')  # Turn off the axis
    fig.tight_layout()

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "peak_detection_comparison.svg"))


dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
# EDA_figures(filepath)
# compare_peak_detection(filepath)
print(test(filepath))
# compare_phasic_tonic_methods(filepath)
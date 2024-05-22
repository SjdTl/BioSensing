import os
from Features.EDA import *
from scipy.fft import fft, fftfreq, fftshift

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

    # Comparing raw and preprocessed
    preprocessed_eda = preProcessing(eda)
    feat_gen.quick_plot(eda, preprocessed_eda)

    # # Comparing tonic, phasic and processed
    phasic, tonic = split_phasic_tonic(preprocessed_eda)
    feat_gen.quick_plot(preprocessed_eda, phasic, tonic)

    # # Test peak detection
    # peak_detection(phasic, plot=True)

    # df = EDA(eda, 700)
    # return df

def EDA_figures(filepath, T =20):
    """
    Plots used for the processing flowchart in chapter four
    Returns five plots:
        - Unprocessed and processed EMG signal in the timedomain
        - Unprocessed and processed EMG signal in the frequency domain
        - Plot of the filter used to process
    The code is connected to the EDA such that changing some parameters their would simply 
    require this to run again to obtain the updated plots (with different cutoff frequencies e.g.)
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

    # EDA smoothing
    smooth_eda = smooth_EDA(filtered_eda)

    fig, ax = plt.subplots()

    ax.plot(np.linspace(0,T, int(fs*T/Q)), smooth_eda)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Conductance [$\mu S$]")
    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_smooth_td.svg"))

    # EDA phasic and tonic
    phasic, tonic = split_phasic_tonic(smooth_eda, fs)
    # Raise with DC for displaying
    phasic_raised = np.mean(tonic) + phasic

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

    # Smooth EDA frequency domain
    yf = fftshift(fft(smooth_eda))
    xf = fftshift(fftfreq(smooth_eda.size, d=Q/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yf)), color= 'green')
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "eda_smooth_fd.svg"))

    # EDA phasic tonic frequency domain
    yfphasic = fftshift(fft(phasic))
    yftonic = fftshift(fft(tonic))
    xf = fftshift(fftfreq(smooth_eda.size, d=Q/fs))

    fig, ax = plt.subplots()

    ax.plot(xf, 10 * np.log10(np.abs(yfphasic)), color= 'green', label="phasic")
    ax.plot(xf, 10 * np.log10(np.abs(yftonic)), color= 'lime', label="tonic")
    ax.set_xlabel("Frequency [$Hz$]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_ylim(ymin = -25)
    ax.legend()

    fig.savefig(os.path.join(dir_path, "plots", "EDA_plots", "phasic_tonic_fd.svg"))

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "Raw_data", "raw_data.pkl")
EDA_figures(filepath)
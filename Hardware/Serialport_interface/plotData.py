import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fftshift

# Get rid of jump in TimeStamps
def ChronologicalTimeStamps(data):
    smallestTimeStamp = data['TimeStamp'].idxmin()
    CorrectedData = data[smallestTimeStamp:-1]
    return CorrectedData

def plot_data(inputfile = 'ECGdata.csv'):

    importData = pd.read_csv(inputfile)

    # Get rid of jump in TimeStamps
    CorrectedData = ChronologicalTimeStamps(importData)

    ecg_plot = CorrectedData.plot(x = 'TimeStamp', subplots = True)
    # Frequency = ecg_plot.add_subplot()
    # f, t, Sxx = signal.spectrogram(CorrectedData['EDA Data'], 100)
    # Frequency.pcolormesh(t, f, Sxx, shading='gouraud')
    # Frequency.ylabel('Frequency [Hz]')
    plt.show()


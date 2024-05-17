import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

    plt.show()

